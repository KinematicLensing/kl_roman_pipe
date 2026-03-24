"""
Tests for datacube construction: CubePars, SpectralModel.build_cube, correctness.

Diagnostic plots saved to tests/out/datacube/.
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
from kl_pipe.spectral import (
    SpectralConfig,
    SpectralModel,
    CubePars,
    halpha_line,
    halpha_nii_lines,
    make_spectral_config,
    roman_grism_R,
    C_KMS,
    HALPHA,
    NII_6583,
)
from kl_pipe.dispersion import GrismPars, build_grism_pars_for_line
from kl_pipe.utils import build_map_grid_from_image_pars
from kl_pipe.diagnostics.datacube import plot_datacube_overview

# output directory for diagnostic plots
OUT_DIR = os.path.join(os.path.dirname(__file__), 'out', 'datacube')
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# Shared fixtures
# =============================================================================

# common test parameters
_IMAGE_PARS = ImagePars(shape=(32, 32), pixel_scale=0.11, indexing='ij')

_VEL_PARS = {
    'cosi': 0.5,
    'theta_int': 0.7,
    'g1': 0.0,
    'g2': 0.0,
    'v0': 10.0,
    'vcirc': 200.0,
    'vel_rscale': 0.5,
}

_INT_PARS = {
    'cosi': 0.5,
    'theta_int': 0.7,
    'g1': 0.0,
    'g2': 0.0,
    'flux': 100.0,
    'int_rscale': 0.3,
    'int_h_over_r': 0.1,
    'int_x0': 0.0,
    'int_y0': 0.0,
}

_SHARED_PARS = {'cosi', 'theta_int', 'g1', 'g2'}


@pytest.fixture(scope='module')
def vel_model():
    return CenteredVelocityModel()


@pytest.fixture(scope='module')
def int_model():
    return InclinedExponentialModel()


@pytest.fixture(scope='module')
def ha_config():
    return SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)


@pytest.fixture(scope='module')
def ha_nii_config():
    return make_spectral_config()


@pytest.fixture(scope='module')
def spec_model_ha(vel_model, int_model, ha_config):
    return SpectralModel(ha_config, int_model, vel_model)


@pytest.fixture(scope='module')
def spec_model_ha_nii(vel_model, int_model, ha_nii_config):
    return SpectralModel(ha_nii_config, int_model, vel_model)


@pytest.fixture(scope='module')
def cube_pars():
    z = 1.0
    lam_center = HALPHA.lambda_rest * (1 + z)
    dlam = lam_center * 2000.0 / C_KMS
    return CubePars.from_range(_IMAGE_PARS, lam_center - dlam, lam_center + dlam, 1.1)


# =============================================================================
# CubePars tests
# =============================================================================


class TestCubePars:
    def test_from_range(self):
        """Correct lambda_grid spacing."""
        cp = CubePars.from_range(_IMAGE_PARS, 1300.0, 1320.0, 1.0)
        assert cp.n_lambda == 21
        assert float(cp.lambda_grid[0]) == pytest.approx(1300.0)
        assert float(cp.lambda_grid[-1]) == pytest.approx(1320.0)
        assert float(cp.delta_lambda) == pytest.approx(1.0, abs=0.01)

    def test_from_R(self):
        """Correct grid for given R value."""
        cp = CubePars.from_R(_IMAGE_PARS, 1300.0, 1320.0, R=1000)
        assert cp.n_lambda >= 2
        lam_c = 0.5 * (1300.0 + 1320.0)
        expected_dl = lam_c / 1000
        assert float(cp.delta_lambda) == pytest.approx(expected_dl, rel=0.1)

    def test_to_cube_pars(self):
        """GrismPars.to_cube_pars(z=1.0) covers Ha at z=1."""
        gp = build_grism_pars_for_line(
            HALPHA.lambda_rest,
            redshift=1.0,
            image_pars=_IMAGE_PARS,
            dispersion=1.1,
        )
        cp = gp.to_cube_pars(z=1.0)
        lam_obs = HALPHA.lambda_rest * 2.0  # z=1

        # lambda grid must bracket Ha observed wavelength
        assert float(cp.lambda_grid[0]) < lam_obs
        assert float(cp.lambda_grid[-1]) > lam_obs
        # grid should be >10 pixels (sufficient spectral sampling)
        assert cp.n_lambda > 10

    def test_from_R_vs_from_range_consistency(self):
        """CubePars.from_R(R=1000) ≈ from_range with delta_lambda=lam_c/1000."""
        lam_min, lam_max = 1300.0, 1320.0
        lam_c = 0.5 * (lam_min + lam_max)
        dl = lam_c / 1000

        cp_R = CubePars.from_R(_IMAGE_PARS, lam_min, lam_max, R=1000)
        cp_range = CubePars.from_range(_IMAGE_PARS, lam_min, lam_max, dl)

        assert float(cp_R.delta_lambda) == pytest.approx(
            float(cp_range.delta_lambda), rel=0.05
        )
        assert cp_R.n_lambda == pytest.approx(cp_range.n_lambda, abs=1)


# =============================================================================
# SpectralModel cube tests
# =============================================================================


class TestBuildCube:
    def test_build_cube_shape(self, spec_model_ha, cube_pars, vel_model, int_model):
        """Cube shape matches (Nrow, Ncol, Nlambda)."""
        theta_vel = vel_model.pars2theta(_VEL_PARS)
        theta_int = int_model.pars2theta(_INT_PARS)
        theta_spec = jnp.array([1.0, 50.0, 100.0, 0.01])

        cube = spec_model_ha.build_cube(theta_spec, theta_vel, theta_int, cube_pars)

        assert cube.shape == (32, 32, cube_pars.n_lambda)

    def test_cube_flux_conservation(self, spec_model_ha, vel_model, int_model):
        """Integral over lambda ~ line_flux (normalized Gaussian)."""
        z = 1.0
        lam_center = HALPHA.lambda_rest * (1 + z)
        # wide enough wavelength range for >99% of Gaussian
        dlam = lam_center * 5000.0 / C_KMS
        cube_pars = CubePars.from_range(
            _IMAGE_PARS, lam_center - dlam, lam_center + dlam, 0.5
        )

        line_flux = 100.0
        theta_vel = vel_model.pars2theta({**_VEL_PARS, 'vcirc': 0.0})  # no rotation
        theta_int = int_model.pars2theta(_INT_PARS)
        theta_spec = jnp.array([z, 50.0, line_flux, 0.0])  # no continuum

        cube = spec_model_ha.build_cube(theta_spec, theta_vel, theta_int, cube_pars)

        # spatial integral at each wavelength, then spectral integral
        dl = float(cube_pars.lambda_grid[1] - cube_pars.lambda_grid[0])
        ps = _IMAGE_PARS.pixel_scale
        total_flux = float(jnp.sum(cube) * ps**2 * dl)

        # measured 0.34%; 0.5% gives ~1.5x headroom
        assert total_flux == pytest.approx(
            line_flux, rel=0.005
        ), f"Flux conservation: total={total_flux:.3f}, expected={line_flux}"

    def test_cube_zero_velocity(self, vel_model, int_model):
        """Zero velocity -> symmetric peak at (1+z)*lambda_rest."""
        z = 1.0
        lam_center = HALPHA.lambda_rest * (1 + z)
        dlam = lam_center * 3000.0 / C_KMS
        cube_pars = CubePars.from_range(
            _IMAGE_PARS, lam_center - dlam, lam_center + dlam, 0.5
        )

        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)

        theta_vel = vel_model.pars2theta({**_VEL_PARS, 'vcirc': 0.0, 'v0': 0.0})
        theta_int = int_model.pars2theta(_INT_PARS)
        theta_spec = jnp.array([z, 50.0, 100.0, 0.0])

        cube = sm.build_cube(theta_spec, theta_vel, theta_int, cube_pars)

        # at center pixel, find peak wavelength
        center_r = _IMAGE_PARS.Nrow // 2
        center_c = _IMAGE_PARS.Ncol // 2
        spectrum = cube[center_r, center_c, :]
        peak_idx = int(jnp.argmax(spectrum))
        peak_lam = float(cube_pars.lambda_grid[peak_idx])

        assert peak_lam == pytest.approx(
            lam_center, abs=1.0
        ), f"Peak at {peak_lam:.2f} nm, expected {lam_center:.2f} nm"

    def test_cube_velocity_shift(self, vel_model, int_model):
        """Positive v_rotation -> peak shifts redward."""
        z = 1.0
        lam_center = HALPHA.lambda_rest * (1 + z)
        dlam = lam_center * 3000.0 / C_KMS
        # finer grid (0.2 nm) so argmax resolves velocity shifts
        cube_pars = CubePars.from_range(
            _IMAGE_PARS, lam_center - dlam, lam_center + dlam, 0.2
        )

        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)

        # galaxy with rotation — approaching and receding sides
        theta_vel = vel_model.pars2theta(_VEL_PARS)
        theta_int = int_model.pars2theta(_INT_PARS)
        theta_spec = jnp.array([z, 50.0, 100.0, 0.0])

        cube = sm.build_cube(theta_spec, theta_vel, theta_int, cube_pars)

        # check two pixels on opposite sides
        cr, cc = _IMAGE_PARS.Nrow // 2, _IMAGE_PARS.Ncol // 2
        # left and right of center along kinematic axis
        spec_left = cube[cr, max(cc - 8, 0), :]
        spec_right = cube[cr, min(cc + 8, _IMAGE_PARS.Ncol - 1), :]

        lam = cube_pars.lambda_grid

        def _parabolic_peak(spectrum):
            """Sub-grid peak via parabolic interpolation around argmax."""
            idx = int(jnp.argmax(spectrum))
            if idx == 0 or idx == len(spectrum) - 1:
                return float(lam[idx])
            y0, y1, y2 = (
                float(spectrum[idx - 1]),
                float(spectrum[idx]),
                float(spectrum[idx + 1]),
            )
            denom = y0 - 2 * y1 + y2
            if abs(denom) < 1e-30:
                return float(lam[idx])
            shift = 0.5 * (y0 - y2) / denom
            return float(lam[idx]) + shift * float(lam[1] - lam[0])

        peak_left = _parabolic_peak(spec_left)
        peak_right = _parabolic_peak(spec_right)

        # peaks should be at different wavelengths (velocity gradient)
        assert abs(peak_left - peak_right) > 0.5, (
            f"Expected velocity shift between left ({peak_left:.2f}) "
            f"and right ({peak_right:.2f}) pixels"
        )

    def test_cube_multi_line_peaks(self, spec_model_ha_nii, vel_model, int_model):
        """Ha + NII produce separate peaks at correct wavelengths."""
        z = 1.0
        # wide range to cover all 3 lines
        lam_min = 654.0 * (1 + z) - 10
        lam_max = 659.0 * (1 + z) + 10
        cube_pars = CubePars.from_range(_IMAGE_PARS, lam_min, lam_max, 0.3)

        theta_vel = vel_model.pars2theta({**_VEL_PARS, 'vcirc': 0.0, 'v0': 0.0})
        theta_int = int_model.pars2theta(_INT_PARS)
        # z, vel_disp, Ha_flux, Ha_cont, NII6548_flux, NII6548_cont, NII6583_flux, NII6583_cont
        theta_spec = jnp.array([z, 50.0, 100.0, 0.0, 30.0, 0.0, 90.0, 0.0])

        cube = spec_model_ha_nii.build_cube(theta_spec, theta_vel, theta_int, cube_pars)

        # spatially summed spectrum
        total_spec = jnp.sum(cube, axis=(0, 1))
        lam = cube_pars.lambda_grid

        # find peaks (local maxima)
        total_np = np.array(total_spec)
        peaks = []
        for i in range(1, len(total_np) - 1):
            if total_np[i] > total_np[i - 1] and total_np[i] > total_np[i + 1]:
                peaks.append(float(lam[i]))

        # should find 3 peaks near the expected observed wavelengths
        expected = sorted(
            [
                HALPHA.lambda_rest * (1 + z),
                NII_6583.lambda_rest * (1 + z),
                654.80 * (1 + z),  # NII_6548
            ]
        )
        assert len(peaks) >= 3, f"Expected 3 peaks, found {len(peaks)}: {peaks}"
        for exp, found in zip(expected, sorted(peaks)[:3]):
            assert found == pytest.approx(
                exp, abs=2.0
            ), f"Peak at {found:.1f}, expected near {exp:.1f}"

    def test_sigma_eff_wavelength_dependent(self):
        """sigma_eff differs between Ha and NII due to R(lambda)."""
        z = 1.0
        lam_ha = HALPHA.lambda_rest * (1 + z)
        lam_nii = NII_6583.lambda_rest * (1 + z)

        vel_disp = 50.0

        R_ha = roman_grism_R(lam_ha)
        R_nii = roman_grism_R(lam_nii)

        sigma_inst_ha = C_KMS / (2.355 * R_ha)
        sigma_inst_nii = C_KMS / (2.355 * R_nii)

        sigma_eff_ha = np.sqrt(vel_disp**2 + sigma_inst_ha**2)
        sigma_eff_nii = np.sqrt(vel_disp**2 + sigma_inst_nii**2)

        # sigma_eff should differ due to different R
        assert sigma_eff_ha != pytest.approx(sigma_eff_nii, rel=1e-3)
        # both should be dominated by instrumental broadening for typical vel_disp
        assert sigma_inst_ha > vel_disp


# =============================================================================
# Correctness tests
# =============================================================================


class TestCorrectness:
    def test_cube_collapses_to_broadband(self, vel_model, int_model):
        """Spectrally collapsed cube ~ render_unconvolved at cube's grid (rtol=0.02)."""
        z = 1.0
        lam_center = HALPHA.lambda_rest * (1 + z)
        dlam = lam_center * 5000.0 / C_KMS
        cube_pars = CubePars.from_range(
            _IMAGE_PARS, lam_center - dlam, lam_center + dlam, 0.5
        )

        # single line, flux=100, zero continuum, zero velocity
        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)

        theta_vel = vel_model.pars2theta({**_VEL_PARS, 'vcirc': 0.0, 'v0': 0.0})
        theta_int = int_model.pars2theta(_INT_PARS)
        theta_spec = jnp.array([z, 50.0, _INT_PARS['flux'], 0.0])

        cube = sm.build_cube(theta_spec, theta_vel, theta_int, cube_pars)

        # collapse: integral over wavelength
        dl = float(cube_pars.lambda_grid[1] - cube_pars.lambda_grid[0])
        collapsed = jnp.sum(cube, axis=2) * dl

        # reference: render_unconvolved (same flux)
        broadband = int_model.render_unconvolved(theta_int, _IMAGE_PARS)

        # normalize both to compare morphology
        collapsed_norm = collapsed / jnp.max(collapsed)
        broadband_norm = broadband / jnp.max(broadband)

        diff = jnp.max(jnp.abs(collapsed_norm - broadband_norm))
        # measured 0.0% (exact); 1e-4 gives 100x headroom
        assert (
            float(diff) < 1e-4
        ), f"Cube collapse vs broadband max diff = {float(diff):.6f}"

    def test_cube_vs_numpy_reference(self, vel_model, int_model):
        """JAX datacube matches independent numpy implementation."""
        from kl_pipe.synthetic import generate_datacube_3d

        z = 1.0
        lam_center = HALPHA.lambda_rest * (1 + z)
        dlam = lam_center * 2000.0 / C_KMS
        cube_pars = CubePars.from_range(
            _IMAGE_PARS, lam_center - dlam, lam_center + dlam, 1.0
        )

        # JAX path
        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)

        theta_vel = vel_model.pars2theta(_VEL_PARS)
        theta_int = int_model.pars2theta(_INT_PARS)
        line_flux = 100.0
        theta_spec = jnp.array([z, 50.0, line_flux, 0.0])

        cube_jax = sm.build_cube(theta_spec, theta_vel, theta_int, cube_pars)

        # numpy path
        np_spectral_pars = {
            'z': z,
            'vel_dispersion': 50.0,
            'lines': [
                {'lambda_rest': HALPHA.lambda_rest, 'flux': line_flux, 'cont': 0.0}
            ],
        }
        np_int_pars = {k: v for k, v in _INT_PARS.items()}
        np_int_pars['n_sersic'] = 1.0

        cube_np = generate_datacube_3d(
            _IMAGE_PARS,
            _VEL_PARS,
            np_int_pars,
            np_spectral_pars,
            np.array(cube_pars.lambda_grid),
            spatial_oversample=5,
            spectral_oversample=5,
        )

        # total flux
        jax_total = float(jnp.sum(cube_jax))
        np_total = float(np.sum(cube_np))

        assert jax_total == pytest.approx(
            np_total, rel=0.001
        ), f"JAX total={jax_total:.3f}, numpy total={np_total:.3f}"

        # peak pixel: both paths now use 5x spatial oversampling
        jax_peak = float(jnp.max(cube_jax))
        np_peak = float(np.max(cube_np))
        assert jax_peak == pytest.approx(
            np_peak, rel=0.005
        ), f"JAX peak={jax_peak:.3f}, numpy peak={np_peak:.3f}"

    def test_v0_z_consistency(self, vel_model, int_model):
        """Galaxy at (v0=100, z=1.0) produces similar peak as (v0=0, z=1.0+100/c)."""
        lam_center1 = HALPHA.lambda_rest * (1 + 1.0)
        dlam = lam_center1 * 3000.0 / C_KMS
        cube_pars = CubePars.from_range(
            _IMAGE_PARS, lam_center1 - dlam, lam_center1 + dlam, 0.5
        )

        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)

        # case 1: v0=100, z=1.0
        theta_vel_1 = vel_model.pars2theta({**_VEL_PARS, 'vcirc': 0.0, 'v0': 100.0})
        theta_int = int_model.pars2theta(_INT_PARS)
        theta_spec_1 = jnp.array([1.0, 50.0, 100.0, 0.0])

        # v0 is subtracted before Doppler, so v0 does NOT shift the line.
        # the line center should be at lambda_rest * (1 + z)
        cube1 = sm.build_cube(theta_spec_1, theta_vel_1, theta_int, cube_pars)

        # case 2: v0=0, z=1.0
        theta_vel_2 = vel_model.pars2theta({**_VEL_PARS, 'vcirc': 0.0, 'v0': 0.0})
        theta_spec_2 = jnp.array([1.0, 50.0, 100.0, 0.0])
        cube2 = sm.build_cube(theta_spec_2, theta_vel_2, theta_int, cube_pars)

        # both should have the same line center (v0 subtracted before Doppler)
        cr, cc = 16, 16
        peak1 = float(cube_pars.lambda_grid[jnp.argmax(cube1[cr, cc, :])])
        peak2 = float(cube_pars.lambda_grid[jnp.argmax(cube2[cr, cc, :])])

        assert peak1 == pytest.approx(
            peak2, abs=1.0
        ), f"v0=100 peak={peak1:.2f}, v0=0 peak={peak2:.2f} should match"

    def test_spectral_oversample_convergence(self, vel_model, int_model):
        """Sweep oversample factors, verify monotonic convergence; 5x < 0.5% error."""
        z = 1.0
        lam_center = HALPHA.lambda_rest * (1 + z)
        dlam = lam_center * 2000.0 / C_KMS
        cube_pars = CubePars.from_range(
            _IMAGE_PARS, lam_center - dlam, lam_center + dlam, 1.0
        )

        theta_vel = vel_model.pars2theta(_VEL_PARS)
        theta_int = int_model.pars2theta(_INT_PARS)
        theta_spec = jnp.array([z, 50.0, 100.0, 0.0])

        # truth at oversample=25
        config_truth = SpectralConfig(lines=(halpha_line(),), spectral_oversample=25)
        sm_truth = SpectralModel(config_truth, int_model, vel_model)
        cube_truth = sm_truth.build_cube(theta_spec, theta_vel, theta_int, cube_pars)

        errors = {}
        for osf in [1, 3, 5, 7, 9]:
            config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=osf)
            sm = SpectralModel(config, int_model, vel_model)
            cube_test = sm.build_cube(theta_spec, theta_vel, theta_int, cube_pars)

            max_err = float(
                jnp.max(jnp.abs(cube_test - cube_truth)) / jnp.max(jnp.abs(cube_truth))
            )
            errors[osf] = max_err

        # monotonic convergence
        err_list = [errors[k] for k in [1, 3, 5, 7, 9]]
        for i in range(len(err_list) - 1):
            assert err_list[i] >= err_list[i + 1] * 0.9, (
                f"Not monotonically converging: osf={[1,3,5,7,9][i]} "
                f"error={err_list[i]:.4f} vs osf={[1,3,5,7,9][i+1]} "
                f"error={err_list[i+1]:.4f}"
            )

        # 5x achieves <0.5% (or at least <2% — some discretization allowed)
        assert (
            errors[5] < 0.02
        ), f"oversample=5 error = {errors[5]:.4f}, expected < 0.02"

        # save diagnostic plot
        try:
            import matplotlib

            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.semilogy(list(errors.keys()), list(errors.values()), 'bo-')
            ax.set_xlabel('Spectral oversample factor')
            ax.set_ylabel('Max relative error vs truth (osf=25)')
            ax.set_title('Spectral oversample convergence')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(
                os.path.join(OUT_DIR, 'spectral_oversample_convergence.png'), dpi=150
            )
            plt.close(fig)
        except Exception:
            pass


# =============================================================================
# Diagnostic plots
# =============================================================================


class TestDiagnosticPlots:
    """Diagnostic plots saved to tests/out/datacube/. Not pass/fail tests."""

    def test_plot_datacube_slices(self, vel_model, int_model):
        """Wavelength slices showing spatial morphology evolution."""
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        z = 1.0
        lam_center = HALPHA.lambda_rest * (1 + z)
        dlam = lam_center * 2000.0 / C_KMS
        cube_pars = CubePars.from_range(
            _IMAGE_PARS, lam_center - dlam, lam_center + dlam, 1.0
        )

        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)

        theta_vel = vel_model.pars2theta(_VEL_PARS)
        theta_int = int_model.pars2theta(_INT_PARS)
        theta_spec = jnp.array([z, 50.0, 100.0, 0.01])

        cube = sm.build_cube(theta_spec, theta_vel, theta_int, cube_pars)

        n_slices = min(6, cube_pars.n_lambda)
        indices = np.linspace(0, cube_pars.n_lambda - 1, n_slices, dtype=int)

        fig, axes = plt.subplots(1, n_slices, figsize=(3 * n_slices, 3))
        for i, idx in enumerate(indices):
            ax = axes[i] if n_slices > 1 else axes
            im = ax.imshow(np.array(cube[:, :, idx]), origin='lower')
            ax.set_title(f'{float(cube_pars.lambda_grid[idx]):.1f} nm')
            plt.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle('Datacube wavelength slices')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, 'datacube_slices.png'), dpi=150)
        plt.close(fig)

    def test_plot_spaxel_spectra(self, vel_model, int_model):
        """Spectrum at 5 spatial pixels: center, approaching, receding, minor axis, edge."""
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        z = 1.0
        lam_center = HALPHA.lambda_rest * (1 + z)
        dlam = lam_center * 3000.0 / C_KMS
        cube_pars = CubePars.from_range(
            _IMAGE_PARS, lam_center - dlam, lam_center + dlam, 0.3
        )

        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)

        theta_vel = vel_model.pars2theta(_VEL_PARS)
        theta_int = int_model.pars2theta(_INT_PARS)
        theta_spec = jnp.array([z, 50.0, 100.0, 0.0])

        cube = sm.build_cube(theta_spec, theta_vel, theta_int, cube_pars)

        cr, cc = 16, 16
        pixels = {
            'center': (cr, cc),
            'approaching': (cr, max(cc - 6, 0)),
            'receding': (cr, min(cc + 6, 31)),
            'minor axis': (min(cr + 6, 31), cc),
            'edge': (cr, min(cc + 12, 31)),
        }

        fig, ax = plt.subplots(figsize=(8, 5))
        lam = np.array(cube_pars.lambda_grid)
        for label, (r, c) in pixels.items():
            spec = np.array(cube[r, c, :])
            ax.plot(lam, spec, label=f'{label} ({r},{c})')
        ax.axvline(
            lam_center,
            color='gray',
            ls='--',
            alpha=0.5,
            label=f'Ha obs ({lam_center:.1f} nm)',
        )
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Flux density')
        ax.set_title('Spaxel spectra — velocity shifts')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, 'spaxel_spectra.png'), dpi=150)
        plt.close(fig)

    def test_plot_datacube_overview(self, vel_model, int_model):
        """Multi-panel datacube overview: intensity, velocity, stacked, channel slices."""
        z = 1.0
        lam_center = HALPHA.lambda_rest * (1 + z)
        dlam = lam_center * 2000.0 / C_KMS
        cube_pars = CubePars.from_range(
            _IMAGE_PARS, lam_center - dlam, lam_center + dlam, 1.0
        )

        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)

        theta_vel = vel_model.pars2theta(_VEL_PARS)
        theta_int = int_model.pars2theta(_INT_PARS)
        theta_spec = jnp.array([z, 50.0, 100.0, 0.01])

        cube = sm.build_cube(theta_spec, theta_vel, theta_int, cube_pars)

        # render imap and vmap
        imap = int_model.render_unconvolved(theta_int, _IMAGE_PARS)
        X, Y = build_map_grid_from_image_pars(_IMAGE_PARS)
        vmap = vel_model(theta_vel, 'obs', X, Y)

        fig = plot_datacube_overview(
            cube,
            np.array(cube_pars.lambda_grid),
            imap=np.array(imap),
            vmap=np.array(vmap),
            lam_center=lam_center,
            v0=float(_VEL_PARS['v0']),
            title='Datacube overview',
            save_path=os.path.join(OUT_DIR, 'datacube_overview.png'),
        )
        assert fig is not None
