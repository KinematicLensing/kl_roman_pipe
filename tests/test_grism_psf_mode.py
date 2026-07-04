"""
Tests for the grism PSF pathway selector ``psf_mode``:
'post_dispersion' (default; disperse the raw cube, then one exact padded
convolution of the 2D dispersed image) vs 'per_slice' (convolve every
wavelength slice before dispersion; general reference path, required for
future wavelength-dependent PSFs).

For a shared wavelength-independent PSF the two orderings are
mathematically identical on the infinite plane; on the finite stamp they
differ only by boundary-truncation terms of the same flux class budgeted
by ``folding_threshold``. The equivalence tests therefore assert
L1(A - B) / F_tot <= C * folding_threshold with a frozen order-unity C
measured across the validation grid (canonical, high-dispersion-shift,
high-velocity-gradient, dispersion angles, PSF sizes, with/without
continuum). Gradient-level tests assert the fast path is
adjoint-consistent, not just primal-consistent.

Diagnostic plots saved to tests/out/grism_psf_mode/.
"""

import os
import matplotlib

matplotlib.use('Agg')
import pytest
import numpy as np
import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import galsim

from kl_pipe.parameters import ImagePars
from kl_pipe.velocity import CenteredVelocityModel
from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.lines import EmissionLine, LINE_LAMBDAS
from kl_pipe.source import SourceModel
from kl_pipe.render import RenderConfig
from kl_pipe.dispersion import GrismPars
from kl_pipe.observation import build_grism_obs
from kl_pipe.priors import PriorDict, Uniform
from kl_pipe.sampling import InferenceTask

# output directory for diagnostic plots
OUT_DIR = os.path.join(os.path.dirname(__file__), 'out', 'grism_psf_mode')
os.makedirs(OUT_DIR, exist_ok=True)

_IMAGE_PARS = ImagePars(shape=(24, 24), pixel_scale=0.11, indexing='ij')

_Z = 1.0
_LAM_CENTER = LINE_LAMBDAS['Halpha'] * (1 + _Z)
_DISPERSION_NM = 1.1
_FOLDING_THRESHOLD = 5e-3  # RenderConfig default; the aliasing budget

# frozen order-unity constant for the A/B equivalence bound
# L1(A-B)/F_tot <= _C_EQUIV * folding_threshold. Measured across the
# full validation grid (angles x FWHMs x stress cases x continuum),
# 2026-07-04: canonical configs C ~ 0.08; worst case C = 1.34 at the
# high-shift stress (line dispersed to ~7 coarse px from stamp center
# on a 24-px stamp -- the dispersed line approaches the stamp boundary,
# so BOTH paths truncate heavily and their difference peaks). Frozen at
# 2.0 = ~1.5x margin over the measured worst case; still ties the
# fast-path discrepancy to the user-selected aliasing budget.
_C_EQUIV = 2.0

# gradient-level equivalence: max relative difference of dchi2/dtheta
# between paths, measured (2026-07-04, off-truth evaluation) across the
# parameters below at the canonical config: 1.4e-3 max (vel.vcirc),
# same boundary-truncation class as the primal errors. Frozen at 5e-3
# (~3.5x margin).
_GRAD_RTOL = 5e-3

_BASE_PARS = {
    'cosi': 0.5,
    'theta_int': 0.7,
    'g1': 0.02,
    'g2': -0.03,
    'z': _Z,
    'vel.v0': 10.0,
    'vel.vcirc': 200.0,
    'vel.rscale': 0.5,
    'Halpha.flux': 100.0,
    'Halpha.rscale': 0.3,
    'Halpha.h_over_r': 0.1,
    'Halpha.x0': 0.0,
    'Halpha.y0': 0.0,
    'Halpha.dispersion': 50.0,
}

_CONTINUUM_EXTRA = {
    'Halpha.cont.flux_per_nm': 2.0,
    'Halpha.cont.rscale': 0.3,
    'Halpha.cont.h_over_r': 0.1,
    'Halpha.cont.x0': 0.0,
    'Halpha.cont.y0': 0.0,
}


@pytest.fixture(scope='module')
def source_ha():
    return SourceModel(
        velocity_model=CenteredVelocityModel(),
        emission_lines={'Halpha': EmissionLine(intensity=InclinedExponentialModel())},
    )


@pytest.fixture(scope='module')
def source_ha_cont():
    return SourceModel(
        velocity_model=CenteredVelocityModel(),
        emission_lines={
            'Halpha': EmissionLine(
                intensity=InclinedExponentialModel(),
                continuum=InclinedExponentialModel(),
            )
        },
    )


def _make_obs(
    fwhm: float = 0.11,
    angle_deg: float = 0.0,
    lambda_ref_offset_nm: float = 0.0,
    **build_kwargs,
):
    """Grism obs on the module grid.

    ``lambda_ref_offset_nm`` displaces the dispersion reference from the
    line center so every line-carrying slice picks up a large shift --
    the high-shift stress case of the equivalence protocol.
    """
    gp = GrismPars(
        image_pars=_IMAGE_PARS,
        dispersion=_DISPERSION_NM,
        lambda_ref=_LAM_CENTER + lambda_ref_offset_nm,
        dispersion_angle_detector=np.deg2rad(angle_deg),
    )
    return build_grism_obs(
        gp,
        z=_Z,
        psf=galsim.Gaussian(fwhm=fwhm),
        render_config=RenderConfig(oversample=3),
        **build_kwargs,
    )


def _ab_error(source, pars, obs) -> float:
    """L1(per_slice - post_dispersion) / F_tot for one configuration."""
    img_a = source.render_grism(pars, obs, psf_mode='per_slice')
    img_b = source.render_grism(pars, obs, psf_mode='post_dispersion')
    ftot = float(jnp.sum(jnp.abs(img_a)))
    return float(jnp.sum(jnp.abs(img_a - img_b))) / ftot


# =============================================================================
# Protocol item 1: image-level A/B equivalence grid
# =============================================================================


class TestImageEquivalence:
    @pytest.mark.parametrize('angle_deg', [0.0, 30.0, 90.0])
    @pytest.mark.parametrize('fwhm', [0.11, 0.18, 0.30])
    def test_canonical_grid(self, source_ha, angle_deg, fwhm):
        """Canonical parameters over dispersion angles x PSF sizes."""
        err = _ab_error(source_ha, _BASE_PARS, _make_obs(fwhm, angle_deg))
        bound = _C_EQUIV * _FOLDING_THRESHOLD
        assert err < bound, (
            f"psf_mode A/B error {err:.3e} exceeds C*f_t = {bound:.1e} "
            f"(angle={angle_deg}, fwhm={fwhm})"
        )

    def test_high_shift_stress(self, source_ha):
        """Line displaced 8 nm from lambda_ref: every line slice carries a
        large dispersion shift, maximizing the boundary-truncation terms
        that distinguish the two orderings."""
        err = _ab_error(source_ha, _BASE_PARS, _make_obs(lambda_ref_offset_nm=8.0))
        assert err < _C_EQUIV * _FOLDING_THRESHOLD, f"high-shift stress: {err:.3e}"

    def test_high_velocity_gradient(self, source_ha):
        """Large vcirc + near-edge-on: maximal Doppler spread across the
        wavelength window."""
        pars = dict(_BASE_PARS, cosi=0.15, **{'vel.vcirc': 350.0})
        err = _ab_error(source_ha, pars, _make_obs())
        assert err < _C_EQUIV * _FOLDING_THRESHOLD, f"high-velocity: {err:.3e}"

    def test_with_continuum(self, source_ha_cont):
        """Continuum flux spans the full wavelength window, populating the
        largest-shift slices (the bound's dominant contributors)."""
        pars = dict(_BASE_PARS, **_CONTINUUM_EXTRA)
        err = _ab_error(source_ha_cont, pars, _make_obs())
        assert err < _C_EQUIV * _FOLDING_THRESHOLD, f"continuum: {err:.3e}"

    def test_no_psf_paths_identical(self, source_ha):
        """Without a PSF both modes are the same no-op: bitwise equality.
        oversample pinned to 1: render_grism only builds fine cubes
        alongside a PSF, so a no-PSF obs with derived oversample > 1 is
        an unsupported (pre-existing) combination."""
        obs = _make_obs()
        obs_nopsf = build_grism_obs(
            obs.grism_pars, z=_Z, render_config=RenderConfig(oversample=1)
        )
        img_a = source_ha.render_grism(_BASE_PARS, obs_nopsf, psf_mode='per_slice')
        img_b = source_ha.render_grism(
            _BASE_PARS, obs_nopsf, psf_mode='post_dispersion'
        )
        np.testing.assert_array_equal(np.asarray(img_a), np.asarray(img_b))


# =============================================================================
# Protocol item 2: gradient-level (adjoint) equivalence
# =============================================================================


class TestGradientEquivalence:
    def test_chi2_gradient_between_paths(self, source_ha):
        """dchi2/dtheta through both paths agrees to _GRAD_RTOL: the fast
        path must be adjoint-consistent for NUTS, not just
        primal-consistent. Data is generated at truth via the reference
        path; gradients are evaluated off-truth so they are O(1)."""
        obs = _make_obs()
        data = source_ha.render_grism(_BASE_PARS, obs, psf_mode='per_slice')

        pars_off = dict(_BASE_PARS)
        pars_off['vel.vcirc'] = 220.0
        pars_off['cosi'] = 0.45
        pars_off['Halpha.flux'] = 110.0
        pars_jax = {k: jnp.asarray(v) for k, v in pars_off.items()}

        def chi2(pars, mode):
            model = source_ha.render_grism(pars, obs, psf_mode=mode)
            return jnp.sum((model - data) ** 2)

        g_a = jax.grad(lambda p: chi2(p, 'per_slice'))(pars_jax)
        g_b = jax.grad(lambda p: chi2(p, 'post_dispersion'))(pars_jax)

        for key in ('vel.vcirc', 'cosi', 'g1', 'g2', 'Halpha.flux', 'z'):
            ga, gb = float(g_a[key]), float(g_b[key])
            rel = abs(gb - ga) / abs(ga)
            assert rel < _GRAD_RTOL, (
                f"grad[{key}]: per_slice={ga:.8e} post_dispersion={gb:.8e} "
                f"(rel diff {rel:.2e} > {_GRAD_RTOL:.0e})"
            )

    def test_likelihood_factory_end_to_end(self, source_ha):
        """psf_mode threads through InferenceTask.from_obs: likelihood
        values and gradients from the two modes agree within the same
        truncation class (scaled by chi2's quadratic amplification)."""
        obs = _make_obs(
            data=np.asarray(
                source_ha.render_grism(_BASE_PARS, _make_obs(), psf_mode='per_slice')
            ),
            variance=0.25,
        )
        priors = PriorDict(
            {
                'cosi': Uniform(0.2, 0.8),
                'vel.vcirc': Uniform(100.0, 300.0),
                **{
                    k: v
                    for k, v in _BASE_PARS.items()
                    if k not in ('cosi', 'vel.vcirc')
                },
            }
        )
        src = source_ha
        tasks = {
            mode: InferenceTask.from_obs(
                src, priors, grism_obs={'roll0': obs}, psf_mode=mode
            )
            for mode in ('per_slice', 'post_dispersion')
        }
        # evaluate OFF-truth so both gradients are O(1): comparing at the
        # reference path's truth point makes its own gradient ~0 and any
        # ratio meaningless (the flawed metric behind a previously
        # retracted "wrong gradients" claim -- see
        # tests/test_spectral_methods.py history)
        theta = jnp.array([0.45, 230.0])  # (cosi, vel.vcirc) alphabetical
        vals = {m: float(t.log_likelihood(theta)) for m, t in tasks.items()}
        grads = {
            m: np.asarray(jax.grad(t.log_likelihood)(theta)) for m, t in tasks.items()
        }
        # both dominated by the true off-truth residual; A/B part is tiny
        assert vals['per_slice'] != vals['post_dispersion']
        rel_val = abs((vals['post_dispersion'] - vals['per_slice']) / vals['per_slice'])
        assert rel_val < 1e-2, f"log-likelihood rel diff {rel_val:.2e}"
        # vector-norm metric: per-component relative comparison breaks
        # down on near-zero components (small-denominator noise), while
        # the truncation-class error scales with the overall gradient
        # magnitude
        diff = np.linalg.norm(grads['post_dispersion'] - grads['per_slice'])
        scale = np.linalg.norm(grads['per_slice'])
        assert diff / scale < _GRAD_RTOL, (
            f"||grad_B - grad_A|| / ||grad_A|| = {diff / scale:.2e} "
            f"(A={grads['per_slice']}, B={grads['post_dispersion']})"
        )


# =============================================================================
# Protocol item 4: pathway preservation + plumbing
# =============================================================================


class TestPathwayPreservation:
    def test_per_slice_matches_manual_chain(self, source_ha):
        """psf_mode='per_slice' is bit-identical to the pre-refactor
        algorithm spelled out manually (build_cube -> vmap per-slice
        convolve -> disperse -> pixel response)."""
        from kl_pipe.dispersion import disperse_cube
        from kl_pipe.grism import _apply_post_dispersion_pixel_response
        from kl_pipe.psf import convolve_fft
        from kl_pipe.source import image_rotation_from_wcs
        from kl_pipe.spectral import CubePars

        obs = _make_obs()
        img = source_ha.render_grism(_BASE_PARS, obs, psf_mode='per_slice')

        build_cube_pars = CubePars(
            image_pars=obs.fine_image_pars,
            lambda_grid=obs.cube_pars.lambda_grid,
        )
        rotation = image_rotation_from_wcs(obs.grism_pars.image_pars.wcs)
        cube = source_ha.build_cube(
            _BASE_PARS,
            build_cube_pars,
            spectral_oversample=obs.spectral_oversample,
            image_rotation=rotation,
            spectral_method=obs.spectral_method,
        )
        cube_t = jnp.moveaxis(cube, -1, 0)
        cube_t = jax.vmap(lambda s: convolve_fft(s, obs.psf_data, bin=False))(cube_t)
        cube = jnp.moveaxis(cube_t, 0, -1)
        dispersed = disperse_cube(
            cube, obs.grism_pars, obs.cube_pars.lambda_grid, oversample=obs.oversample
        )
        expected = _apply_post_dispersion_pixel_response(
            dispersed,
            obs.pixel_response_fft,
            (obs.grism_pars.image_pars.Nrow, obs.grism_pars.image_pars.Ncol),
            obs.oversample,
            obs.grism_pars.image_pars.pixel_scale,
        )
        np.testing.assert_array_equal(np.asarray(img), np.asarray(expected))

    def test_default_is_post_dispersion(self, source_ha):
        assert RenderConfig().psf_mode == 'post_dispersion'
        obs = _make_obs()
        assert obs.psf_mode == 'post_dispersion'
        img_default = source_ha.render_grism(_BASE_PARS, obs)
        img_explicit = source_ha.render_grism(
            _BASE_PARS, obs, psf_mode='post_dispersion'
        )
        np.testing.assert_array_equal(np.asarray(img_default), np.asarray(img_explicit))

    def test_selectable_via_render_config(self, source_ha):
        """An obs carrying psf_mode='per_slice' in its RenderConfig renders
        via the reference path without any explicit kwarg."""
        gp = GrismPars(
            image_pars=_IMAGE_PARS,
            dispersion=_DISPERSION_NM,
            lambda_ref=_LAM_CENTER,
            dispersion_angle_detector=0.0,
        )
        obs_slice = build_grism_obs(
            gp,
            z=_Z,
            psf=galsim.Gaussian(fwhm=0.11),
            render_config=RenderConfig(oversample=3, psf_mode='per_slice'),
        )
        assert obs_slice.psf_mode == 'per_slice'
        img_from_rc = source_ha.render_grism(_BASE_PARS, obs_slice)
        img_explicit = source_ha.render_grism(
            _BASE_PARS, _make_obs(), psf_mode='per_slice'
        )
        np.testing.assert_array_equal(np.asarray(img_from_rc), np.asarray(img_explicit))

    def test_render_config_validates_psf_mode(self):
        with pytest.raises(ValueError, match="psf_mode"):
            RenderConfig(psf_mode='bogus')

    def test_render_grism_rejects_unknown_psf_mode(self, source_ha):
        with pytest.raises(ValueError, match="psf_mode"):
            source_ha.render_grism(_BASE_PARS, _make_obs(), psf_mode='fused')

    def test_from_obs_mismatched_psf_mode_raises(self, source_ha):
        priors = PriorDict(
            {
                'cosi': Uniform(0.2, 0.8),
                **{k: v for k, v in _BASE_PARS.items() if k != 'cosi'},
            }
        )
        gp = GrismPars(
            image_pars=_IMAGE_PARS,
            dispersion=_DISPERSION_NM,
            lambda_ref=_LAM_CENTER,
            dispersion_angle_detector=0.0,
        )

        rng = np.random.default_rng(7)

        def obs_with(mode):
            return build_grism_obs(
                gp,
                z=_Z,
                psf=galsim.Gaussian(fwhm=0.11),
                render_config=RenderConfig(oversample=3, psf_mode=mode),
                data=rng.normal(size=_IMAGE_PARS.shape),
                variance=0.25,
            )

        mixed = {'a': obs_with('per_slice'), 'b': obs_with('post_dispersion')}
        with pytest.raises(ValueError, match="mismatched psf_mode"):
            InferenceTask.from_obs(source_ha, priors, grism_obs=mixed)
        # explicit override resolves the mismatch
        task = InferenceTask.from_obs(
            source_ha, priors, grism_obs=mixed, psf_mode='post_dispersion'
        )
        assert jnp.isfinite(task.log_likelihood(jnp.array([0.5])))


# =============================================================================
# Visual diagnostics
# =============================================================================


class TestDiagnostics:
    def test_psf_mode_diagnostics(self, source_ha, source_ha_cont):
        """Two-row diagnostic: (top) per_slice / post_dispersion / signed
        difference images for the canonical config; (bottom) A/B error
        across the validation grid vs the C*f_t bound, and forward/grad
        timing of the two paths."""
        import csv
        import time

        import matplotlib.pyplot as plt

        obs = _make_obs()
        img_a = np.asarray(
            source_ha.render_grism(_BASE_PARS, obs, psf_mode='per_slice')
        )
        img_b = np.asarray(
            source_ha.render_grism(_BASE_PARS, obs, psf_mode='post_dispersion')
        )

        configs = [
            ('canonical', source_ha, _BASE_PARS, _make_obs()),
            ('angle=30', source_ha, _BASE_PARS, _make_obs(angle_deg=30.0)),
            ('angle=90', source_ha, _BASE_PARS, _make_obs(angle_deg=90.0)),
            ('fwhm=0.30', source_ha, _BASE_PARS, _make_obs(fwhm=0.30)),
            (
                'high shift',
                source_ha,
                _BASE_PARS,
                _make_obs(lambda_ref_offset_nm=8.0),
            ),
            (
                'high vel',
                source_ha,
                dict(_BASE_PARS, cosi=0.15, **{'vel.vcirc': 350.0}),
                _make_obs(),
            ),
            (
                'continuum',
                source_ha_cont,
                dict(_BASE_PARS, **_CONTINUUM_EXTRA),
                _make_obs(),
            ),
        ]
        labels = [c[0] for c in configs]
        errors = [_ab_error(src, pars, o) for _, src, pars, o in configs]

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        for ax, img, title in [
            (axes[0, 0], img_a, "per_slice (reference path)"),
            (axes[0, 1], img_b, "post_dispersion (default)"),
        ]:
            im = ax.imshow(img, origin='lower')
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046)
        diff = img_b - img_a
        dmax = float(np.abs(diff).max())
        im = axes[0, 2].imshow(
            diff, origin='lower', cmap='RdBu_r', vmin=-dmax, vmax=dmax
        )
        axes[0, 2].set_title(
            f"difference (peak {np.abs(diff).max():.2e} vs "
            f"image peak {img_a.max():.2e})"
        )
        fig.colorbar(im, ax=axes[0, 2], fraction=0.046)

        ax = axes[1, 0]
        ax.bar(range(len(errors)), errors, color='steelblue')
        ax.axhline(
            _C_EQUIV * _FOLDING_THRESHOLD,
            color='crimson',
            ls='--',
            label=f'test bound C*f_t = {_C_EQUIV * _FOLDING_THRESHOLD:.1e}',
        )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('L1(A-B) / F_tot')
        ax.set_yscale('log')
        ax.set_title('A/B error across the validation grid')
        ax.legend(fontsize=8)

        # timing: forward + gradient through a chi2, both paths
        pars_jax = {k: jnp.asarray(v) for k, v in _BASE_PARS.items()}
        data = jnp.asarray(img_a)
        rows = []
        t_fwd_ms = {}
        for mode in ('per_slice', 'post_dispersion'):

            def chi2(pars, m=mode):
                model = source_ha.render_grism(pars, obs, psf_mode=m)
                return jnp.sum((model - data) ** 2)

            fwd = jax.jit(chi2)
            grd = jax.jit(jax.grad(chi2))
            jax.block_until_ready(fwd(pars_jax))
            jax.block_until_ready(grd(pars_jax))
            t0 = time.perf_counter()
            for _ in range(20):
                jax.block_until_ready(fwd(pars_jax))
            t_fwd = (time.perf_counter() - t0) / 20 * 1e3
            t0 = time.perf_counter()
            for _ in range(20):
                jax.block_until_ready(grd(pars_jax))
            t_grd = (time.perf_counter() - t0) / 20 * 1e3
            rows.append((mode, f'{t_fwd:.3f}', f'{t_grd:.3f}'))
            t_fwd_ms[mode] = t_fwd

        ax = axes[1, 1]
        x = np.arange(2)
        ax.bar(x - 0.2, [float(r[1]) for r in rows], 0.4, label='forward')
        ax.bar(x + 0.2, [float(r[2]) for r in rows], 0.4, label='gradient')
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in rows])
        ax.set_ylabel('ms per call (jitted)')
        ax.set_title(
            'chi2 timing: post_dispersion replaces 2*Nlambda per-slice\n'
            'FFTs with one padded convolution of the dispersed image'
        )
        ax.legend(fontsize=8)
        axes[1, 2].axis('off')
        axes[1, 2].text(
            0.05,
            0.5,
            "psf_mode='post_dispersion' (default):\n"
            "disperse raw cube, then ONE exact padded\n"
            "PSF convolution of the 2D dispersed image.\n\n"
            "psf_mode='per_slice' (reference):\n"
            "convolve all wavelength slices, then disperse.\n\n"
            "Identical for a shared PSF up to stamp-boundary\n"
            "truncation terms (folding_threshold flux class).",
            fontsize=9,
            family='monospace',
            va='center',
        )

        fig.suptitle(
            'Grism PSF pathway: per_slice vs post_dispersion '
            f'(24x24 grid, oversample=3, {len(obs.cube_pars.lambda_grid)} channels)',
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out_png = os.path.join(OUT_DIR, 'psf_mode_comparison.png')
        fig.savefig(out_png, dpi=130)
        plt.close(fig)

        out_csv = os.path.join(OUT_DIR, 'psf_mode_timing.csv')
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['psf_mode', 'forward_ms', 'grad_ms', 'grid=24x24 os=3'])
            w.writerows(rows)

        assert os.path.exists(out_png)
        assert os.path.exists(out_csv)
        # the speedup must actually materialize on the forward pass
        assert t_fwd_ms['post_dispersion'] < t_fwd_ms['per_slice'], (
            f"post_dispersion ({t_fwd_ms['post_dispersion']:.2f} ms) not faster "
            f"than per_slice ({t_fwd_ms['per_slice']:.2f} ms)"
        )
        print(
            f"A/B errors: {dict(zip(labels, [f'{e:.2e}' for e in errors]))}; "
            f"implied C values: "
            f"{[f'{e / _FOLDING_THRESHOLD:.2f}' for e in errors]}; "
            f"timing rows: {rows}; plots: {out_png}"
        )


# =============================================================================
# Protocol item 3: posterior-level A/B (short seeded run)
# =============================================================================


@pytest.mark.slow
class TestPosteriorEquivalence:
    def test_seeded_posterior_ab(self, source_ha):
        """Seeded NUTS with identical data through both PSF pathways:
        posterior means shift < 0.1 sigma (shear params especially),
        widths within 20%, divergences not degraded. Small config
        (24x24, 4 sampled params, 1 chain) -- a screening A/B, not the
        flagship-depth run (deferred to GPU-era hardware)."""
        import csv

        from kl_pipe.sampling import build_sampler
        from kl_pipe.sampling.configs import NumpyroSamplerConfig

        obs_clean = _make_obs()
        truth = source_ha.render_grism(_BASE_PARS, obs_clean, psf_mode='per_slice')
        peak = float(jnp.max(truth))
        sigma_noise = peak / 50.0  # SNR ~ 50 at the peak
        rng = np.random.default_rng(20260704)
        data = np.asarray(truth) + rng.normal(0.0, sigma_noise, truth.shape)

        obs = _make_obs(data=data, variance=sigma_noise**2)
        sampled = {
            'cosi': Uniform(0.2, 0.8),
            'g1': Uniform(-0.1, 0.1),
            'g2': Uniform(-0.1, 0.1),
            'vel.vcirc': Uniform(100.0, 300.0),
        }
        priors = PriorDict(
            {
                **sampled,
                **{k: v for k, v in _BASE_PARS.items() if k not in sampled},
            }
        )
        config = NumpyroSamplerConfig(
            n_samples=300, n_warmup=300, n_chains=1, seed=42, progress=False
        )

        results = {}
        for mode in ('per_slice', 'post_dispersion'):
            task = InferenceTask.from_obs(
                source_ha, priors, grism_obs={'roll0': obs}, psf_mode=mode
            )
            results[mode] = build_sampler('numpyro', task, config).run()

        res_a, res_b = results['per_slice'], results['post_dispersion']
        assert res_a.param_names == res_b.param_names

        rows = []
        for i, name in enumerate(res_a.param_names):
            mean_a = float(res_a.samples[:, i].mean())
            mean_b = float(res_b.samples[:, i].mean())
            std_a = float(res_a.samples[:, i].std())
            std_b = float(res_b.samples[:, i].std())
            shift_sigma = abs(mean_b - mean_a) / std_a
            width_ratio = std_b / std_a
            rows.append(
                (
                    name,
                    f'{mean_a:.5g}',
                    f'{mean_b:.5g}',
                    f'{std_a:.3g}',
                    f'{shift_sigma:.3f}',
                    f'{width_ratio:.3f}',
                )
            )
            assert shift_sigma < 0.1, (
                f"{name}: posterior mean shift {shift_sigma:.3f} sigma "
                f"(A={mean_a:.5g}, B={mean_b:.5g}, sigma_A={std_a:.3g})"
            )
            assert (
                0.8 < width_ratio < 1.2
            ), f"{name}: posterior width ratio {width_ratio:.3f}"

        div_a = res_a.diagnostics.get('n_divergences', 0)
        div_b = res_b.diagnostics.get('n_divergences', 0)
        assert div_b <= div_a + 5, f"divergences degraded: {div_a} -> {div_b}"

        out_csv = os.path.join(OUT_DIR, 'posterior_ab.csv')
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(
                [
                    'param',
                    'mean_per_slice',
                    'mean_post_dispersion',
                    'sigma_per_slice',
                    'shift_sigma',
                    'width_ratio',
                ]
            )
            w.writerows(rows)
        print(
            f"posterior A/B (SNR~50, 300+300 NUTS): {rows}; "
            f"divergences A={div_a} B={div_b}; csv: {out_csv}"
        )
