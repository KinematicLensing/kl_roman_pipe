"""Tests for closed-form (analytic) grism dispersal.

The analytic path is the exact n_lambda -> infinity limit of the slice
method under identical conventions (erf bin integration in wavelength +
bilinear interpolation of the per-slice shift), so a dense slice render
must converge toward it and any residual is the slice reference's own
O(ds^2) discretization error.
"""

import dataclasses

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import galsim
from astropy.wcs import WCS
from scipy.integrate import quad

from kl_pipe.dispersion import (
    build_grism_pars_for_line,
    continuum_trace_kernel,
    disperse_continuum_analytic,
    gaussian_tent_profile,
)
from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.observation import build_grism_obs
from kl_pipe.parameters import ImagePars
from kl_pipe.render import RenderConfig
from kl_pipe.source import SourceModel, EmissionLine
from kl_pipe.velocity import CenteredVelocityModel


TRUE_PARS = {
    'cosi': 0.6,
    'theta_int': np.pi / 4,
    'g1': 0.02,
    'g2': -0.01,
    'vel.v0': 10.0,
    'vel.vcirc': 200.0,
    'vel.rscale': 0.3,
    'Halpha.flux': 100.0,
    'Halpha.rscale': 0.25,
    'Halpha.h_over_r': 0.1,
    'Halpha.x0': 0.0,
    'Halpha.y0': 0.0,
    'Halpha.dispersion': 50.0,
    'Halpha.cont.flux_per_nm': 25.0,
    'Halpha.cont.rscale': 0.25,
    'Halpha.cont.h_over_r': 0.1,
    'Halpha.cont.x0': 0.0,
    'Halpha.cont.y0': 0.0,
    'z': 1.0,
}

# equivalence bounds vs a dense n=501 slice reference (16x16, os=3).
# Measured 2026-07-06 (measure-then-freeze, ~4x headroom): full scene
# max 2.1e-5, line-only 2.7e-5, ramp-throughput 1.9e-4 of peak; the
# residual is the n=501 reference's own convergence error, so growth
# past these bounds means one of the paths changed behavior.
MAX_TOL_FLAT = 1e-4
MAX_TOL_RAMP = 7e-4
FLUX_TOL = 1e-4


@pytest.fixture(scope='module')
def scene():
    ip = ImagePars(shape=(16, 16), pixel_scale=0.11, indexing='ij')
    gp = build_grism_pars_for_line(656.28, redshift=1.0, image_pars=ip, dispersion=1.1)
    psf = galsim.Gaussian(fwhm=0.18)
    source = SourceModel(
        velocity_model=CenteredVelocityModel(),
        emission_lines={
            'Halpha': EmissionLine(
                intensity=InclinedExponentialModel(),
                continuum=InclinedExponentialModel(),
            )
        },
    )
    return ip, gp, psf, source


class TestGaussianTentProfile:
    def test_matches_quadrature(self):
        # closed form vs adaptive quadrature of Gaussian x triangle kernel
        for u, sig in [(0.0, 0.5), (1.3, 0.2), (-2.1, 1.5), (0.7, 3.0)]:

            def integrand(x):
                g = np.exp(-0.5 * (x / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
                return g * max(1.0 - abs(u - x), 0.0)

            num, _ = quad(integrand, u - 1.0, u + 1.0, limit=200, epsabs=1e-14)
            ana = float(gaussian_tent_profile(jnp.asarray(u), jnp.asarray(sig)))
            assert abs(num - ana) < 1e-12

    def test_narrow_limit_is_tent(self):
        # sigma -> 0 reduces to the bare triangle kernel of bilinear
        # interpolation (a zero-width line lands on one sub-pixel position)
        u = jnp.linspace(-2.0, 2.0, 41)
        sigma = 1e-7
        prof = gaussian_tent_profile(u, jnp.asarray(sigma))
        tent = jnp.clip(1.0 - jnp.abs(u), 0.0, None)
        # deviation is O(sigma) at the tent kinks (Gaussian rounding)
        np.testing.assert_allclose(np.asarray(prof), np.asarray(tent), atol=sigma)

    def test_normalization(self):
        # unit flux: samples along the dispersion axis sum to 1 at any
        # sub-pixel center offset (translates of the triangle kernel sum to 1)
        u = jnp.arange(-40, 41, dtype=float)
        for frac in (0.0, 0.3, 0.77):
            total = float(gaussian_tent_profile(u - frac, jnp.asarray(1.8)).sum())
            assert abs(total - 1.0) < 1e-12


class TestContinuumTraceKernel:
    def test_flat_kernel_total(self, scene):
        # kernel sums to the wavelength window span in nm (box integral)
        _, gp, _, _ = scene
        lam = jnp.linspace(1290.0, 1335.0, 41)
        kern, _ = continuum_trace_kernel(gp, lam, oversample=3)
        span = float(lam[-1] - lam[0])
        assert abs(kern.sum() - span) < 1e-10 * span

    def test_ramp_kernel_total(self, scene):
        # T-weighted kernel integrates to integral of T over the window
        _, gp, _, _ = scene
        lam = np.linspace(1290.0, 1335.0, 41)
        T = 1.0 + 2.0 * (lam - lam[0]) / (lam[-1] - lam[0])
        gp_T = dataclasses.replace(gp, throughput=jnp.asarray(T))
        kern, _ = continuum_trace_kernel(gp_T, jnp.asarray(lam), oversample=3)
        expected = np.trapz(T, lam)
        assert abs(kern.sum() - expected) < 1e-8 * expected

    def test_correlation_matches_bruteforce(self, scene):
        _, gp, _, _ = scene
        lam = jnp.linspace(1300.0, 1325.0, 21)
        kern, m_lo = continuum_trace_kernel(gp, lam, oversample=1)
        rng = np.random.default_rng(3)
        img = rng.uniform(size=(4, 30))
        out = np.asarray(disperse_continuum_analytic(jnp.asarray(img), kern, m_lo))
        brute = np.zeros_like(img)
        for q in range(img.shape[1]):
            for j in range(img.shape[1]):
                m = q - j - m_lo
                if 0 <= m < len(kern):
                    brute[:, q] += img[:, j] * kern[m]
        np.testing.assert_allclose(out, brute, atol=1e-12)


class TestRenderEquivalence:
    """Analytic render vs dense (n=501) slice render on the same obs."""

    def _obs(self, gp, psf, method, **kw):
        rc = RenderConfig(oversample=3, dispersal_method=method)
        return build_grism_obs(gp, z=1.0, psf=psf, render_config=rc, **kw)

    def test_full_scene(self, scene):
        _, gp, psf, source = scene
        ref = np.asarray(
            source.render_grism(TRUE_PARS, self._obs(gp, psf, 'slice', n_lambda=501))
        )
        ana = np.asarray(source.render_grism(TRUE_PARS, self._obs(gp, psf, 'analytic')))
        peak = np.abs(ref).max()
        assert np.abs(ana - ref).max() / peak < MAX_TOL_FLAT
        assert abs(ana.sum() / ref.sum() - 1.0) < FLUX_TOL

    def test_line_only(self, scene):
        _, gp, psf, source = scene
        pars = dict(TRUE_PARS)
        pars['Halpha.cont.flux_per_nm'] = 0.0
        ref = np.asarray(
            source.render_grism(pars, self._obs(gp, psf, 'slice', n_lambda=501))
        )
        ana = np.asarray(source.render_grism(pars, self._obs(gp, psf, 'analytic')))
        peak = np.abs(ref).max()
        assert np.abs(ana - ref).max() / peak < MAX_TOL_FLAT

    def test_ramp_throughput(self, scene):
        _, gp, psf, source = scene

        def with_ramp(method, **kw):
            base = self._obs(gp, psf, method, **kw)
            lam = np.asarray(base.cube_pars.lambda_grid)
            T = 1.0 + 2.0 * (lam - lam[0]) / (lam[-1] - lam[0])
            gp_T = dataclasses.replace(gp, throughput=jnp.asarray(T))
            return build_grism_obs(
                gp_T,
                z=1.0,
                psf=psf,
                render_config=base.render_config,
                n_lambda=len(lam),
            )

        ref = np.asarray(
            source.render_grism(TRUE_PARS, with_ramp('slice', n_lambda=501))
        )
        ana = np.asarray(source.render_grism(TRUE_PARS, with_ramp('analytic')))
        peak = np.abs(ref).max()
        assert np.abs(ana - ref).max() / peak < MAX_TOL_RAMP
        assert abs(ana.sum() / ref.sum() - 1.0) < FLUX_TOL

    def test_rolled_obs_matches_dense_slice(self, scene):
        # nonzero WCS roll: both paths rotate model parameters into the
        # detector frame and keep the dispersal along detector +x, so the
        # rolled equivalence must hold at the unrolled floor
        _, _, psf, source = scene
        shape = (16, 16)
        rot = 0.35
        c, s = float(np.cos(rot)), float(np.sin(rot))
        wcs = WCS(naxis=2)
        wcs.wcs.pc = np.array([[c, -s], [s, c]])
        wcs.wcs.cdelt = np.array([0.11, 0.11])
        wcs.wcs.crpix = np.array([shape[1] / 2, shape[0] / 2])
        wcs.wcs.crval = np.array([0.0, 0.0])
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        wcs.wcs.cunit = ['arcsec', 'arcsec']
        wcs.pixel_shape = (shape[1], shape[0])
        wcs.wcs.set()
        ip_rot = ImagePars(shape=shape, wcs=wcs)
        gp_rot = build_grism_pars_for_line(
            656.28, redshift=1.0, image_pars=ip_rot, dispersion=1.1
        )
        ref = np.asarray(
            source.render_grism(
                TRUE_PARS, self._obs(gp_rot, psf, 'slice', n_lambda=501)
            )
        )
        ana = np.asarray(
            source.render_grism(TRUE_PARS, self._obs(gp_rot, psf, 'analytic'))
        )
        peak = np.abs(ref).max()
        assert np.abs(ana - ref).max() / peak < MAX_TOL_FLAT
        assert abs(ana.sum() / ref.sum() - 1.0) < FLUX_TOL

    def test_production_slice_rule_matches_analytic(self, scene):
        """Slice pathway at its production settings (slice_width_kms=40,
        oversample=5) vs the analytic default.

        This bound is what a user of the slice pathway at production
        settings accepts relative to the exact model, and it is the
        equivalence guarantee that lets the likelihood-recovery tests run
        on the analytic path alone. Measured 2026-07-08: max 2.3e-4 of
        peak, flux agreement 1.2e-5; frozen at ~3x.
        """
        _, gp, psf, source = scene
        obs_s = build_grism_obs(
            gp,
            z=1.0,
            psf=psf,
            render_config=RenderConfig(oversample=5, dispersal_method='slice'),
            slice_width_kms=40.0,
        )
        obs_a = build_grism_obs(
            gp,
            z=1.0,
            psf=psf,
            render_config=RenderConfig(oversample=5, dispersal_method='analytic'),
        )
        ref = np.asarray(source.render_grism(TRUE_PARS, obs_s))
        ana = np.asarray(source.render_grism(TRUE_PARS, obs_a))
        peak = np.abs(ana).max()
        assert np.abs(ana - ref).max() / peak < 7e-4
        assert abs(ref.sum() / ana.sum() - 1.0) < 5e-5

    def test_jit_matches_eager_and_grad_finite(self, scene):
        _, gp, psf, source = scene
        rc = RenderConfig(
            oversample=3, dispersal_method='analytic', line_window_halfwidth=16
        )
        obs = build_grism_obs(gp, z=1.0, psf=psf, render_config=rc)
        eager = np.asarray(source.render_grism(TRUE_PARS, obs))
        jitted = np.asarray(jax.jit(lambda p: source.render_grism(p, obs))(TRUE_PARS))
        np.testing.assert_allclose(jitted, eager, rtol=0, atol=1e-12)

        pars_j = {k: jnp.asarray(v) for k, v in TRUE_PARS.items()}
        grads = jax.grad(lambda p: jnp.sum(source.render_grism(p, obs) ** 2))(pars_j)
        assert all(np.isfinite(np.asarray(v)).all() for v in grads.values())


class TestGuards:
    def test_render_config_validation(self):
        with pytest.raises(ValueError, match='dispersal_method'):
            RenderConfig(dispersal_method='magic')
        with pytest.raises(ValueError, match='line_window_halfwidth'):
            RenderConfig(line_window_halfwidth=0)
        with pytest.raises(ValueError, match='line_window_halfwidth'):
            RenderConfig(line_window_halfwidth=2.5)

    def test_per_slice_psf_rejected(self, scene):
        _, gp, psf, source = scene
        rc = RenderConfig(oversample=3, dispersal_method='analytic')
        obs = build_grism_obs(gp, z=1.0, psf=psf, render_config=rc)
        with pytest.raises(ValueError, match='post_dispersion'):
            source.render_grism(TRUE_PARS, obs, psf_mode='per_slice')

    def test_jit_without_halfwidth_rejected(self, scene):
        _, gp, psf, source = scene
        rc = RenderConfig(oversample=3, dispersal_method='analytic')
        obs = build_grism_obs(gp, z=1.0, psf=psf, render_config=rc)
        with pytest.raises(ValueError, match='line_window_halfwidth'):
            jax.jit(lambda p: source.render_grism(p, obs))(TRUE_PARS)

    def test_rotated_dispersion_rejected(self, scene):
        _, gp, psf, source = scene
        gp_rot = dataclasses.replace(
            gp, dispersion_angle_detector=0.3, dispersion_angle=None
        )
        rc = RenderConfig(oversample=3, dispersal_method='analytic')
        obs = build_grism_obs(gp_rot, z=1.0, psf=psf, render_config=rc)
        with pytest.raises(NotImplementedError, match='axis-aligned'):
            source.render_grism(TRUE_PARS, obs)

    def test_group_path_rejected(self, scene):
        _, gp, psf, source = scene
        rc = RenderConfig(oversample=3, dispersal_method='analytic')
        obs = build_grism_obs(gp, z=1.0, psf=psf, render_config=rc)
        with pytest.raises(NotImplementedError, match='shared-cube group'):
            source.render_grism_group(TRUE_PARS, {'roll0': obs})


class TestHalfwidthAutoSizing:
    """line_window_halfwidth sized from prior extremes in from_obs."""

    def _priors(self):
        from kl_pipe.priors import Gaussian, PriorDict, TruncatedNormal

        sampled = {
            'vel.vcirc': TruncatedNormal(200.0, 50.0, 80.0, 400.0),
            'vel.v0': Gaussian(10.0, 10.0),
            'Halpha.dispersion': TruncatedNormal(50.0, 20.0, 5.0, 150.0),
            'Halpha.flux': TruncatedNormal(100.0, 20.0, 30.0, 250.0),
        }
        fixed = {k: v for k, v in TRUE_PARS.items() if k not in sampled}
        return PriorDict({**fixed, **sampled})

    def test_formula(self, scene):
        from kl_pipe.constants import C_KMS
        from kl_pipe.render import line_window_halfwidth_for_priors

        _, gp, _, source = scene
        oversample = 3
        hw = line_window_halfwidth_for_priors(
            source, self._priors(), gp, oversample=oversample
        )
        # independent arithmetic: v_max = (|mu| + 6 sd for the Gaussian v0)
        # + vcirc upper bound; sigma_v_max = dispersion upper bound
        lam_line = 656.28 * 2.0
        scale = oversample / gp.dispersion
        v_max = (10.0 + 6.0 * 10.0) + 400.0
        expected = (
            int(
                np.ceil(
                    abs(lam_line - gp.lambda_ref) * scale
                    + lam_line * v_max / C_KMS * scale
                    + 4.0 * lam_line * 150.0 / C_KMS * scale
                )
            )
            + 2
        )
        assert hw == expected

    def test_from_obs_fills_halfwidth(self, scene):
        # analytic rc without halfwidth: from_obs sizes it from priors, so
        # the jitted likelihood traces (raises ValueError without the fill)
        from kl_pipe.sampling import InferenceTask

        _, gp, psf, source = scene
        rc = RenderConfig(oversample=3, dispersal_method='analytic')
        clean = build_grism_obs(gp, z=1.0, psf=psf, render_config=rc)
        data = source.render_grism(TRUE_PARS, clean)
        obs = build_grism_obs(
            gp, z=1.0, psf=psf, render_config=rc, data=data, variance=1.0
        )
        priors = self._priors()
        task = InferenceTask.from_obs(source, priors, grism_obs={'r0': obs})
        theta = jnp.asarray([TRUE_PARS[n] for n in priors.sampled_names])
        assert np.isfinite(float(task.log_likelihood(theta)))

    def test_checked_obs_has_expected_halfwidth(self, scene):
        from kl_pipe.render import line_window_halfwidth_for_priors
        from kl_pipe.sampling.task import _check_source_priors_fit_obs

        _, gp, psf, source = scene
        priors = self._priors()
        rc = RenderConfig(oversample=3, dispersal_method='analytic')
        obs = build_grism_obs(gp, z=1.0, psf=psf, render_config=rc)
        checked = _check_source_priors_fit_obs(source, priors, obs)
        expected = line_window_halfwidth_for_priors(source, priors, gp, oversample=3)
        assert checked.line_window_halfwidth == expected

    def test_explicit_halfwidth_respected(self, scene):
        from kl_pipe.sampling.task import _check_source_priors_fit_obs

        _, gp, psf, source = scene
        rc = RenderConfig(
            oversample=3, dispersal_method='analytic', line_window_halfwidth=16
        )
        obs = build_grism_obs(gp, z=1.0, psf=psf, render_config=rc)
        checked = _check_source_priors_fit_obs(source, self._priors(), obs)
        assert checked.line_window_halfwidth == 16

    def test_missing_dispersion_prior_raises(self, scene):
        from kl_pipe.priors import PriorDict
        from kl_pipe.render import line_window_halfwidth_for_priors

        _, gp, _, source = scene
        pruned = {
            k: v
            for k, v in self._priors()._param_spec.items()
            if k != 'Halpha.dispersion'
        }
        with pytest.raises(KeyError, match='Halpha.dispersion'):
            line_window_halfwidth_for_priors(
                source, PriorDict(pruned), gp, oversample=3
            )


@pytest.mark.slow
@pytest.mark.diagnostic_plots
def test_diagnostic_analytic_slice_figure(scene):
    """Render the composite analytic-vs-slice diagnostic figure.

    Writes tests/out/analytic_dispersal/analytic_vs_slice.png: slice
    convergence toward the analytic image, error-map morphology, the
    line-dispersion error bar vs slice count, and gradient cost vs
    accuracy. Assertions here are light sanity checks on the figure's
    inputs; the strict accuracy gates live in TestRenderEquivalence and
    the likelihood-slice tier.
    """
    import time

    from kl_pipe.diagnostics.grism import plot_analytic_slice_comparison
    from kl_pipe.utils import get_test_dir

    _, gp, psf, source = scene
    snr = 1000.0

    def make_obs(method, n_lambda=None):
        rc = RenderConfig(
            oversample=5,
            dispersal_method=method,
            line_window_halfwidth=25 if method == 'analytic' else None,
        )
        return build_grism_obs(gp, z=1.0, psf=psf, render_config=rc, n_lambda=n_lambda)

    obs_ana = make_obs('analytic')
    img_ana = np.asarray(source.render_grism(TRUE_PARS, obs_ana))
    peak = np.abs(img_ana).max()

    convergence = {}
    diff_maps = {}
    for n in (25, 51, 101, 151, 201, 401, 801):
        img = np.asarray(source.render_grism(TRUE_PARS, make_obs('slice', n)))
        convergence[n] = float(np.abs(img - img_ana).max() / peak)
        if n in (25, 151, 401):
            diff_maps[n] = img - img_ana

    # dispersion error bar from the whitened model derivative (all other
    # parameters at truth), at the matched-filter noise convention
    def sigma_dispersion(obs):
        clean = np.asarray(source.render_grism(TRUE_PARS, obs))
        var = float((clean**2).sum()) / snr**2

        def f(d):
            pars = dict(TRUE_PARS)
            pars['Halpha.dispersion'] = d
            return source.render_grism(pars, obs)

        _, J = jax.jvp(f, (jnp.asarray(50.0),), (jnp.asarray(1.0),))
        fisher = float((np.asarray(J) ** 2).sum()) / var
        return 1.0 / np.sqrt(fisher)

    sigma_slice = {n: sigma_dispersion(make_obs('slice', n)) for n in (25, 151, 401)}
    sigma_ana = sigma_dispersion(obs_ana)

    # indicative gradient timings (this machine, min of 3); ratios are the
    # meaningful quantity
    def grad_ms(obs):
        fn = jax.jit(
            jax.grad(
                lambda d: jnp.sum(
                    source.render_grism({**TRUE_PARS, 'Halpha.dispersion': d}, obs) ** 2
                )
            )
        )
        jax.block_until_ready(fn(jnp.asarray(50.0)))  # compile
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(jnp.asarray(50.0)))
            times.append(time.perf_counter() - t0)
        return min(times) * 1e3

    timings = {
        'slice $N_\\lambda$=25': (grad_ms(make_obs('slice', 25)), convergence[25]),
        'slice $N_\\lambda$=151': (grad_ms(make_obs('slice', 151)), convergence[151]),
        'analytic': (grad_ms(obs_ana), min(convergence.values())),
    }

    out = plot_analytic_slice_comparison(
        get_test_dir() / 'out' / 'analytic_dispersal' / 'analytic_vs_slice.png',
        convergence,
        img_ana,
        diff_maps,
        sigma_slice,
        sigma_ana,
        timings_ms=timings,
        scene_note=(
            'Scene: 16x16 @ 0.11"/px, oversample 5,\n'
            'Halpha + flat continuum at z=1,\n'
            'PSF FWHM 0.18", dispersion 1.1 nm/px\n\n'
            'Error bars: grism-only, matched-filter\n'
            'SNR=1000 convention\n\n'
            'Timings: this machine, min of 3;\n'
            'ratios are the meaningful quantity'
        ),
    )
    assert out.exists()

    # sanity: convergence is monotone with no plateau at the dense end,
    # and the dense-slice error bar agrees with the analytic one
    ns = sorted(convergence)
    errs = [convergence[n] for n in ns]
    assert all(a > b for a, b in zip(errs, errs[1:]))
    assert errs[-1] < 1e-4
    assert abs(sigma_slice[401] / sigma_ana - 1.0) < 0.15
