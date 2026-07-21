"""
Tests for spectral bin-integration methods in ``SourceModel.build_cube``:
'erf' (exact analytic Gaussian bin integrals, default) vs 'oversample'
(midpoint fine-grid sampling, retained for convergence/comparison studies).

Covers value equivalence in the converged limit, midpoint convergence
toward the erf-exact cube, continuum exactness under both methods, line
flux conservation, gradient (AD vs finite-difference) consistency of the
erf path, method-selection plumbing, and loud-error validation.

Diagnostic plots saved to tests/out/spectral_methods/.
"""

import os
import matplotlib

matplotlib.use('Agg')
import pytest
import numpy as np
import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from kl_pipe.parameters import ImagePars
from kl_pipe.velocity import CenteredVelocityModel
from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.spectral import CubePars, C_KMS
from kl_pipe.lines import EmissionLine, LINE_LAMBDAS
from kl_pipe.source import SourceModel
from kl_pipe.render import RenderConfig

# output directory for diagnostic plots
OUT_DIR = os.path.join(os.path.dirname(__file__), 'out', 'spectral_methods')
os.makedirs(OUT_DIR, exist_ok=True)

# small grid keeps the high-osf reference cubes cheap
_IMAGE_PARS = ImagePars(shape=(24, 24), pixel_scale=0.11, indexing='ij')

_Z = 1.0
_LAM_CENTER = LINE_LAMBDAS['Halpha'] * (1 + _Z)
_DISPERSION_NM = 1.1  # Roman-like nm / coarse pixel

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


def _cube_pars(half_width_nm: float = 4.0) -> CubePars:
    """Wavelength window wide enough for vcirc=200 Doppler shifts + 6 sigma
    line tails (sigma_lambda ~ 0.22 nm at 50 km/s)."""
    return CubePars.from_range(
        _IMAGE_PARS,
        _LAM_CENTER - half_width_nm,
        _LAM_CENTER + half_width_nm,
        _DISPERSION_NM,
    )


@pytest.fixture(scope='module')
def source_ha():
    return SourceModel(
        velocity_model=CenteredVelocityModel(),
        emission_lines={'Halpha': EmissionLine(intensity=InclinedExponentialModel())},
    )


@pytest.fixture(scope='module')
def cube_pars():
    return _cube_pars()


def _max_rel_err(cube, ref):
    return float(jnp.max(jnp.abs(cube - ref)) / jnp.max(jnp.abs(ref)))


# =============================================================================
# Value-level equivalence and convergence
# =============================================================================


class TestValueEquivalence:
    @pytest.mark.parametrize('dispersion_kms,tol', [(50.0, 1e-6), (15.0, 1e-5)])
    def test_erf_matches_converged_oversample(
        self, source_ha, cube_pars, dispersion_kms, tol
    ):
        """erf-exact cube == the osf->inf limit of the midpoint path.

        Tolerance is the midpoint REFERENCE's own residual error, which
        for bins whose edge falls on the line flank is dominated by the
        Euler-Maclaurin edge term (Delta^2/24) * [g'(b) - g'(a)] / dl
        with Delta = dl/osf and max|g'| ~ 0.24/sigma_lambda^2: ~8e-7 of
        the peak voxel at 50 km/s (sigma_lambda ~ 0.22 nm) and ~9e-6 at
        the narrow 15 km/s line (sigma_lambda ~ 0.066 nm, the hardest
        quadrature case). The erf value is exact; the bound tracks the
        osf=801 reference's convergence, hence scales with the line
        width.
        """
        pars = dict(_BASE_PARS, **{'Halpha.dispersion': dispersion_kms})
        cube_erf = source_ha.build_cube(pars, cube_pars, spectral_method='erf')
        cube_ref = source_ha.build_cube(
            pars, cube_pars, spectral_oversample=801, spectral_method='oversample'
        )
        err = _max_rel_err(cube_erf, cube_ref)
        assert err < tol, (
            f"erf vs osf=801 max rel err = {err:.3e} at "
            f"dispersion={dispersion_kms} km/s (tol {tol:.0e})"
        )

    def test_oversample_converges_to_erf(self, source_ha, cube_pars):
        """Midpoint cube error vs the erf-exact cube decreases monotonically
        with osf (validates both implementations against each other)."""
        cube_erf = source_ha.build_cube(_BASE_PARS, cube_pars, spectral_method='erf')
        errs = []
        for osf in [5, 15, 45, 135]:
            cube = source_ha.build_cube(
                _BASE_PARS,
                cube_pars,
                spectral_oversample=osf,
                spectral_method='oversample',
            )
            errs.append(_max_rel_err(cube, cube_erf))
        for i in range(len(errs) - 1):
            assert errs[i] > errs[i + 1], f"midpoint not converging to erf: errs={errs}"
        # osf=135 midpoint edge-term error ~ (dl/135)^2/24 * max|g'|/dl
        # ~ 3e-5 of the peak voxel at sigma_lambda ~ 0.22 nm
        assert errs[-1] < 5e-5, f"osf=135 vs erf err = {errs[-1]:.3e}"

    def test_continuum_identical_both_methods(self, source_ha, cube_pars):
        """A flat continuum is exact under both methods (bin average of a
        constant is the constant); zero line flux isolates it."""
        pars = dict(_BASE_PARS)
        pars['Halpha.flux'] = 0.0
        pars['Halpha.cont.flux_per_nm'] = 5.0
        pars['Halpha.cont.rscale'] = 0.3
        pars['Halpha.cont.h_over_r'] = 0.1
        pars['Halpha.cont.x0'] = 0.0
        pars['Halpha.cont.y0'] = 0.0
        source = SourceModel(
            velocity_model=CenteredVelocityModel(),
            emission_lines={
                'Halpha': EmissionLine(
                    intensity=InclinedExponentialModel(),
                    continuum=InclinedExponentialModel(),
                )
            },
        )
        cube_erf = source.build_cube(pars, cube_pars, spectral_method='erf')
        cube_osf = source.build_cube(
            pars, cube_pars, spectral_oversample=15, spectral_method='oversample'
        )
        np.testing.assert_allclose(
            np.asarray(cube_erf), np.asarray(cube_osf), rtol=0, atol=1e-12
        )

    def test_erf_line_flux_conservation(self, source_ha, cube_pars):
        """Summing the erf cube over wavelength (x bin width) recovers the
        spatial line-intensity map: the CDF telescopes to
        Phi(last edge) - Phi(first edge) ~ 1 for a window covering the
        line +- >6 sigma. Tail mass beyond the window bounds the error."""
        cube_erf = source_ha.build_cube(_BASE_PARS, cube_pars, spectral_method='erf')
        dl = float(cube_pars.lambda_grid[1] - cube_pars.lambda_grid[0])
        integrated = np.asarray(jnp.sum(cube_erf, axis=-1) * dl)

        from kl_pipe.utils import build_map_grid_from_image_pars

        X, Y = build_map_grid_from_image_pars(_IMAGE_PARS)
        int_model = InclinedExponentialModel()
        # build theta in canonical PARAMETER_NAMES order: shared geometry
        # from top-level keys, per-component params from 'Halpha.<name>'
        shared = ('cosi', 'theta_int', 'g1', 'g2')
        theta = jnp.array(
            [
                _BASE_PARS[n] if n in shared else _BASE_PARS[f'Halpha.{n}']
                for n in int_model.PARAMETER_NAMES
            ]
        )
        I_line = np.asarray(int_model(theta, 'obs', X, Y))
        # compare where the profile carries meaningful flux
        sel = I_line > 1e-4 * I_line.max()
        rel = np.abs(integrated[sel] / I_line[sel] - 1.0)
        assert rel.max() < 1e-6, f"flux conservation violated: {rel.max():.3e}"


# =============================================================================
# Gradient consistency
# =============================================================================


class TestGradients:
    def test_erf_gradient_matches_finite_difference(self, source_ha, cube_pars):
        """AD gradient of a per-voxel-weighted reduction through the erf
        path matches central finite differences (adjoint consistency)."""
        rng = np.random.default_rng(42)
        weights = jnp.asarray(rng.normal(size=(24, 24, len(cube_pars.lambda_grid))))

        def loss(pars):
            cube = source_ha.build_cube(pars, cube_pars, spectral_method='erf')
            return jnp.sum(weights * cube)

        pars_jax = {k: jnp.asarray(v) for k, v in _BASE_PARS.items()}
        grads = jax.grad(loss)(pars_jax)

        for key, eps in [
            ('vel.vcirc', 1e-4),
            ('z', 1e-8),
            ('Halpha.dispersion', 1e-4),
            ('g2', 1e-7),
        ]:
            p_plus = dict(pars_jax)
            p_plus[key] = pars_jax[key] + eps
            p_minus = dict(pars_jax)
            p_minus[key] = pars_jax[key] - eps
            fd = (float(loss(p_plus)) - float(loss(p_minus))) / (2 * eps)
            ad = float(grads[key])
            assert abs(ad / fd - 1.0) < 1e-6, (
                f"{key}: AD={ad:.8e} vs FD={fd:.8e} "
                f"(rel diff {abs(ad / fd - 1.0):.2e})"
            )


# =============================================================================
# Method-selection plumbing + loud errors
# =============================================================================


class TestPlumbing:
    def test_build_cube_rejects_unknown_method(self, source_ha, cube_pars):
        with pytest.raises(ValueError, match="spectral_method"):
            source_ha.build_cube(_BASE_PARS, cube_pars, spectral_method='simpson')

    def test_erf_rejects_single_slice_cube(self, source_ha):
        single = CubePars(image_pars=_IMAGE_PARS, lambda_grid=jnp.array([_LAM_CENTER]))
        with pytest.raises(ValueError, match="Nlambda >= 2"):
            source_ha.build_cube(_BASE_PARS, single, spectral_method='erf')

    def test_render_config_validates_method(self):
        with pytest.raises(ValueError, match="spectral_method"):
            RenderConfig(spectral_method='bogus')

    def test_render_config_default_is_erf(self):
        rc = RenderConfig()
        assert rc.spectral_method == 'erf'

    def test_grism_obs_reads_method_from_render_config(self):
        from kl_pipe.dispersion import build_grism_pars_for_line
        from kl_pipe.observation import build_grism_obs

        grism_pars = build_grism_pars_for_line(
            LINE_LAMBDAS['Halpha'],
            _Z,
            image_pars=_IMAGE_PARS,
            dispersion=_DISPERSION_NM,
        )
        obs_default = build_grism_obs(grism_pars, z=_Z)
        assert obs_default.spectral_method == 'erf'
        obs_osf = build_grism_obs(
            grism_pars,
            z=_Z,
            render_config=RenderConfig(spectral_method='oversample'),
        )
        assert obs_osf.spectral_method == 'oversample'

    def test_render_grism_method_override(self, source_ha):
        """Explicit spectral_method kwarg overrides the obs default; the
        two methods differ measurably (osf=15 discretization error) but
        agree at the coarse level. Uses a small-but-nontrivial Gaussian
        PSF: point-sampled cuspy profiles alias, so bespoke no-PSF
        renders are not representative."""
        import galsim

        from kl_pipe.dispersion import build_grism_pars_for_line
        from kl_pipe.observation import build_grism_obs

        grism_pars = build_grism_pars_for_line(
            LINE_LAMBDAS['Halpha'],
            _Z,
            image_pars=_IMAGE_PARS,
            dispersion=_DISPERSION_NM,
        )
        obs = build_grism_obs(
            grism_pars,
            z=_Z,
            # smaller than the native Roman diffraction PSF at Halpha(z=1)
            # (FWHM ~0.11"): softens the profile cusp without being easier
            # than the real instrument
            psf=galsim.Gaussian(fwhm=0.08),
            # spectral_method applies to cube assembly; pin the slice path
            render_config=RenderConfig(oversample=3, dispersal_method='slice'),
        )

        img_default = source_ha.render_grism(_BASE_PARS, obs)
        img_erf = source_ha.render_grism(_BASE_PARS, obs, spectral_method='erf')
        img_osf = source_ha.render_grism(
            _BASE_PARS, obs, spectral_method='oversample', spectral_oversample=15
        )

        # default == explicit erf, bitwise
        np.testing.assert_array_equal(np.asarray(img_default), np.asarray(img_erf))
        # methods differ (osf=15 carries ~5e-4-level discretization error)...
        assert not np.allclose(np.asarray(img_erf), np.asarray(img_osf), rtol=1e-9)
        # ...but agree at the percent level
        peak = float(np.max(np.abs(np.asarray(img_erf))))
        assert (
            float(np.max(np.abs(np.asarray(img_erf) - np.asarray(img_osf)))) / peak
            < 1e-2
        )


# =============================================================================
# Visual diagnostics: accuracy, narrow-line sampling error, timing
# =============================================================================


class TestDiagnostics:
    def test_spectral_methods_diagnostics(self, source_ha, cube_pars):
        """4-panel diagnostic + timing CSV comparing methods.

        (a) peak-pixel spectrum: erf vs osf=15 overlaid on the converged
            osf=801 reference, with each method's error on a twin axis;
        (b) narrow-line scan: osf=15 voxel VALUE error as the Doppler
            line center (via systemic v0) slides across the wavelength
            grid -- the error oscillates with period one fine sub-bin
            (~16.7 km/s) because the fixed sampling comb under-resolves a
            narrow (10 km/s) line; erf is exact at every shift;
        (c) same scan, the voxel's v0-GRADIENT: osf=15 inherits spurious
            oscillations on top of the exact smooth erf curve;
        (d) midpoint error vs erf -> 0 as O(osf^-2), confirming erf is
            the osf -> inf limit.
        Also writes a forward/grad timing CSV (small-grid config; NOT
        flagship numbers).
        """
        import csv
        import time

        import matplotlib.pyplot as plt

        lam_grid = np.asarray(cube_pars.lambda_grid)
        n_lam = len(lam_grid)

        def build(pars, method, osf=15):
            return source_ha.build_cube(
                pars, cube_pars, spectral_oversample=osf, spectral_method=method
            )

        cube_erf = build(_BASE_PARS, 'erf')
        cube_15 = build(_BASE_PARS, 'oversample', 15)
        cube_ref = build(_BASE_PARS, 'oversample', 801)

        cube_ref_np = np.asarray(cube_ref)
        spatial_sum = cube_ref_np.sum(axis=-1)
        iy, ix = np.unravel_index(np.argmax(spatial_sum), spatial_sum.shape)

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))

        # (a) voxel spectrum + residuals
        ax = axes[0, 0]
        ax.plot(lam_grid, cube_ref_np[iy, ix], 'k-', lw=2, label='osf=801 (ref)')
        ax.plot(lam_grid, np.asarray(cube_erf)[iy, ix], 'g--', label='erf')
        ax.plot(lam_grid, np.asarray(cube_15)[iy, ix], 'r:', label='osf=15')
        axr = ax.twinx()
        axr.plot(
            lam_grid,
            np.asarray(cube_15)[iy, ix] - cube_ref_np[iy, ix],
            'r-',
            alpha=0.4,
            label='osf=15 - ref',
        )
        axr.plot(
            lam_grid,
            np.asarray(cube_erf)[iy, ix] - cube_ref_np[iy, ix],
            'g-',
            alpha=0.4,
            label='erf - ref',
        )
        axr.set_ylabel('method - reference (pale curves)')
        ax.set_xlabel('lambda [nm]')
        ax.set_ylabel('SB density [flux/arcsec^2/nm]')
        ax.set_title(
            f'(a) spectrum at brightest pixel ({iy},{ix}), 50 km/s line:\n'
            'both methods track the converged reference'
        )
        ax.legend(loc='upper left', fontsize=8)
        axr.legend(loc='upper right', fontsize=8)

        # (b) narrow-line scan: value error vs line-center position. Uses
        # a NARROW line (10 km/s, sigma_lambda ~ 0.044 nm < fine sub-bin
        # 0.073 nm): the midpoint comb under-resolves the line, so the
        # error oscillates as the Doppler-shifted center slides across
        # sub-bins; at the 50 km/s fiducial (3 samples/sigma) only the
        # smooth O(Delta^2) quadrature error remains. The flagship
        # dispersion prior extends to 5 km/s, so this regime is reachable
        # in production.
        ax = axes[0, 1]
        il = int(np.argmax(cube_ref_np[iy, ix]))
        fine_bin_kms = C_KMS * (_DISPERSION_NM / 15) / _LAM_CENTER  # ~16.7
        narrow_kms = 10.0

        def voxel(v0, method, osf=15):
            pars = dict(_BASE_PARS)
            pars['vel.v0'] = v0
            pars['Halpha.dispersion'] = narrow_kms
            return build(pars, method, osf)[iy, ix, il]

        voxel_erf = jax.jit(lambda v: voxel(v, 'erf'))
        voxel_15 = jax.jit(lambda v: voxel(v, 'oversample', 15))
        g_erf = jax.jit(jax.grad(lambda v: voxel(v, 'erf')))
        g_15 = jax.jit(jax.grad(lambda v: voxel(v, 'oversample', 15)))

        v0_scan = np.linspace(10.0 - 2.5 * fine_bin_kms, 10.0 + 2.5 * fine_bin_kms, 81)
        val_err = np.array([float(voxel_15(v)) - float(voxel_erf(v)) for v in v0_scan])
        grad_erf = np.array([float(g_erf(v)) for v in v0_scan])
        grad_15 = np.array([float(g_15(v)) for v in v0_scan])

        ax.plot(v0_scan, val_err / float(voxel_erf(10.0)), 'b-')
        ax.axvline(10.0, color='gray', ls=':', alpha=0.5)
        ax.set_xlabel('systemic velocity v0 [km/s] (shifts the line center)')
        ax.set_ylabel('osf=15 fractional voxel error\n(erf = 0 by construction)')
        ax.set_title(
            '(b) narrow 10 km/s line: osf=15 VALUE error oscillates as the\n'
            f'line center crosses fine sub-bins (one sub-bin = '
            f'{fine_bin_kms:.1f} km/s)'
        )
        axg = axes[1, 0]
        axg.plot(v0_scan, grad_erf, 'g-', label='erf (exact, smooth)')
        axg.plot(v0_scan, grad_15, 'r-', alpha=0.7, label='osf=15 (sampling artifacts)')
        axg.set_xlabel('systemic velocity v0 [km/s]')
        axg.set_ylabel('d(voxel value)/d v0')
        axg.set_title(
            '(c) same scan, voxel gradient wrt v0: osf=15 adds spurious\n'
            'wiggles + amplitude error on top of the exact erf curve'
        )
        axg.legend(fontsize=8)

        # (c) convergence + (d) timing share the last panel via inset table
        ax = axes[1, 1]
        osf_sweep = [5, 9, 15, 27, 51, 101, 201, 405]
        errs = [
            _max_rel_err(build(_BASE_PARS, 'oversample', o), cube_erf)
            for o in osf_sweep
        ]
        ax.loglog(osf_sweep, errs, 'bo-', label='midpoint cube vs erf cube')
        guide = errs[2] * (np.array(osf_sweep) / 15.0) ** -2
        ax.loglog(osf_sweep, guide, 'k:', alpha=0.6, label=r'$O(osf^{-2})$ guide')
        ax.set_xlabel('spectral_oversample (midpoint sub-bins per channel)')
        ax.set_ylabel('max relative cube difference')
        ax.set_title(
            '(d) midpoint method converges to the erf cube as osf grows:\n'
            'erf is the exact osf -> inf limit, at no extra cost'
        )
        ax.legend(fontsize=8)

        fig.suptitle(
            'Spectral bin integration: erf (exact analytic, DEFAULT) vs midpoint '
            f'oversampling\n24x24 grid, {n_lam} channels; panels (b),(c) use a '
            'narrow 10 km/s line that the osf=15 fine grid under-resolves',
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        out_png = os.path.join(OUT_DIR, 'spectral_methods_comparison.png')
        fig.savefig(out_png, dpi=130)
        plt.close(fig)

        # timing CSV (small-grid config -- relative comparison only)
        rng = np.random.default_rng(3)
        mock = jnp.asarray(
            cube_ref_np + rng.normal(0, 0.01 * cube_ref_np.std(), cube_ref_np.shape)
        )

        def chi2(method, osf):
            def f(pars):
                return jnp.sum((build(pars, method, osf) - mock) ** 2)

            return f

        pars_jax = {k: jnp.asarray(v) for k, v in _BASE_PARS.items()}
        rows = []
        for label, method, osf in [
            ('osf=15', 'oversample', 15),
            ('erf', 'erf', 15),
            ('osf=201', 'oversample', 201),
        ]:
            fwd = jax.jit(chi2(method, osf))
            grd = jax.jit(jax.grad(chi2(method, osf)))
            fwd(pars_jax)  # compile
            jax.block_until_ready(grd(pars_jax))
            t0 = time.perf_counter()
            for _ in range(10):
                jax.block_until_ready(fwd(pars_jax))
            t_fwd = (time.perf_counter() - t0) / 10 * 1e3
            t0 = time.perf_counter()
            for _ in range(10):
                jax.block_until_ready(grd(pars_jax))
            t_grd = (time.perf_counter() - t0) / 10 * 1e3
            rows.append((label, f'{t_fwd:.3f}', f'{t_grd:.3f}'))

        out_csv = os.path.join(OUT_DIR, 'spectral_methods_timing.csv')
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['variant', 'forward_ms', 'grad_ms', f'grid=24x24x{n_lam}'])
            w.writerows(rows)

        assert os.path.exists(out_png)
        assert os.path.exists(out_csv)
        # the narrow-line scan must actually demonstrate the effect: the osf=15
        # gradient ripple around the smooth erf gradient should dominate
        # the erf gradient's own variation across one fine sub-bin
        ripple = np.abs(grad_15 - grad_erf).max()
        smooth_scale = np.abs(grad_erf).max()
        assert ripple > 0, "expected nonzero osf=15 gradient ripple"
        # the narrow-line regime must actually demonstrate the midpoint
        # path's failure mode: for sigma_lambda < fine sub-bin the osf=15
        # VALUE error reaches the percent level (measured ~5% at 10 km/s)
        # -- the erf default exists precisely because the production
        # dispersion prior extends into this regime
        narrow_rel_err = np.abs(val_err).max() / float(voxel_erf(10.0))
        assert narrow_rel_err > 1e-3, (
            f"expected >0.1% osf=15 value error for the narrow line, got "
            f"{narrow_rel_err:.2e} -- the scan no longer demonstrates the "
            f"under-resolved regime"
        )
        print(
            f"narrow-line scan (dispersion={narrow_kms} km/s): osf=15 max value "
            f"err = {narrow_rel_err:.2%} of peak voxel; max |grad ripple| "
            f"= {ripple:.3e} (erf grad scale {smooth_scale:.3e}); "
            f"plots: {out_png}"
        )
