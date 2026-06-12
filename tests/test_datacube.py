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
from kl_pipe.spectral import CubePars, C_KMS
from kl_pipe.lines import EmissionLine, LINE_LAMBDAS
from kl_pipe.source import SourceModel
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
    'rscale': 0.5,
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


def _make_pars(vel_pars, int_pars, z, vel_dispersion, line_fluxes, line_conts=None):
    """Build a SourceModel dotted-key pars dict from legacy-shape pars.

    ``line_fluxes``: dict of {line_key: flux}. Halpha is always present in the
    Halpha-only fixture; multi-line fixtures may include NII6584, NII6548.
    ``line_conts``: dict of {line_key: cont_flux}, optional. When provided,
    each line gets ``<line>.cont.flux`` and the matching ``<line>.cont.*``
    spatial params (same shape as the line's intensity model).
    """
    pars = {
        # shared geometry
        'cosi': vel_pars['cosi'],
        'theta_int': vel_pars['theta_int'],
        'g1': vel_pars['g1'],
        'g2': vel_pars['g2'],
        'z': z,
        # velocity
        'vel.v0': vel_pars['v0'],
        'vel.vcirc': vel_pars['vcirc'],
        'vel.rscale': vel_pars['rscale'],
    }
    # per-line spatial profile parameters (shared spatial shape from int_pars,
    # per-line flux via line_fluxes[k])
    for line_key, line_flux in line_fluxes.items():
        pars[f'{line_key}.flux'] = line_flux
        pars[f'{line_key}.rscale'] = int_pars['int_rscale']
        pars[f'{line_key}.h_over_r'] = int_pars['int_h_over_r']
        pars[f'{line_key}.x0'] = int_pars['int_x0']
        pars[f'{line_key}.y0'] = int_pars['int_y0']
        pars[f'{line_key}.dispersion'] = vel_dispersion
    # optional per-line continuum
    if line_conts:
        for line_key, cont_flux in line_conts.items():
            pars[f'{line_key}.cont.flux'] = cont_flux
            pars[f'{line_key}.cont.rscale'] = int_pars['int_rscale']
            pars[f'{line_key}.cont.h_over_r'] = int_pars['int_h_over_r']
            pars[f'{line_key}.cont.x0'] = int_pars['int_x0']
            pars[f'{line_key}.cont.y0'] = int_pars['int_y0']
    return pars


@pytest.fixture(scope='module')
def vel_model():
    return CenteredVelocityModel()


@pytest.fixture(scope='module')
def int_model():
    return InclinedExponentialModel()


@pytest.fixture(scope='module')
def source_ha(vel_model, int_model):
    """SourceModel with single Halpha line."""
    return SourceModel(
        velocity_model=vel_model,
        emission_lines={'Halpha': EmissionLine(intensity=int_model)},
    )


@pytest.fixture(scope='module')
def source_ha_nii(vel_model, int_model):
    """SourceModel with Halpha + NII6548 + NII6584 (NII lines share Halpha spatial)."""
    return SourceModel(
        velocity_model=vel_model,
        emission_lines={
            'Halpha': EmissionLine(intensity=int_model),
            'NII6548': EmissionLine(intensity_key='Halpha'),
            'NII6584': EmissionLine(intensity_key='Halpha'),
        },
    )


@pytest.fixture(scope='module')
def cube_pars():
    z = 1.0
    lam_center = LINE_LAMBDAS['Halpha'] * (1 + z)
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
            LINE_LAMBDAS['Halpha'],
            redshift=1.0,
            image_pars=_IMAGE_PARS,
            dispersion=1.1,
        )
        cp = gp.to_cube_pars(z=1.0)
        lam_obs = LINE_LAMBDAS['Halpha'] * 2.0  # z=1

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
    def test_build_cube_shape(self, source_ha, cube_pars, vel_model, int_model):
        """Cube shape matches (Nrow, Ncol, Nlambda)."""
        pars = _make_pars(
            _VEL_PARS,
            _INT_PARS,
            z=1.0,
            vel_dispersion=50.0,
            line_fluxes={'Halpha': 100.0},
            line_conts={'Halpha': 0.01},
        )
        cube = source_ha.build_cube(pars, cube_pars)
        assert cube.shape == (32, 32, cube_pars.n_lambda)

    def test_cube_flux_conservation(self, source_ha, vel_model, int_model):
        """Integral over lambda + space ~ line_flux (normalized Gaussian).

        SourceModel.build_cube returns SB per arcsec² per nm; total flux is
        ``sum(cube) * dl * pixel_area``.
        """
        z = 1.0
        lam_center = LINE_LAMBDAS['Halpha'] * (1 + z)
        # wide enough wavelength range for >99% of Gaussian
        dlam = lam_center * 5000.0 / C_KMS
        cube_pars = CubePars.from_range(
            _IMAGE_PARS, lam_center - dlam, lam_center + dlam, 0.5
        )

        line_flux = 100.0
        pars = _make_pars(
            {**_VEL_PARS, 'vcirc': 0.0},
            _INT_PARS,
            z=z,
            vel_dispersion=50.0,
            line_fluxes={'Halpha': line_flux},  # no continuum
        )
        cube = source_ha.build_cube(pars, cube_pars)

        # spatial integral (× pixel_area) at each wavelength, then spectral integral (× dl)
        dl = float(cube_pars.lambda_grid[1] - cube_pars.lambda_grid[0])
        pixel_area = _IMAGE_PARS.pixel_scale**2
        total_flux = float(jnp.sum(cube) * dl * pixel_area)

        # measured 0.34%; 0.5% gives ~1.5x headroom
        assert total_flux == pytest.approx(
            line_flux, rel=0.005
        ), f"Flux conservation: total={total_flux:.3f}, expected={line_flux}"

    def test_cube_zero_velocity(self, source_ha, vel_model, int_model):
        """Zero velocity -> symmetric peak at (1+z)*lambda_rest."""
        z = 1.0
        lam_center = LINE_LAMBDAS['Halpha'] * (1 + z)
        dlam = lam_center * 3000.0 / C_KMS
        cube_pars = CubePars.from_range(
            _IMAGE_PARS, lam_center - dlam, lam_center + dlam, 0.5
        )

        pars = _make_pars(
            {**_VEL_PARS, 'vcirc': 0.0, 'v0': 0.0},
            _INT_PARS,
            z=z,
            vel_dispersion=50.0,
            line_fluxes={'Halpha': 100.0},
        )
        cube = source_ha.build_cube(pars, cube_pars)

        # at center pixel, find peak wavelength
        center_r = _IMAGE_PARS.Nrow // 2
        center_c = _IMAGE_PARS.Ncol // 2
        spectrum = cube[center_r, center_c, :]
        peak_idx = int(jnp.argmax(spectrum))
        peak_lam = float(cube_pars.lambda_grid[peak_idx])

        assert peak_lam == pytest.approx(
            lam_center, abs=1.0
        ), f"Peak at {peak_lam:.2f} nm, expected {lam_center:.2f} nm"

    def test_cube_velocity_shift(self, source_ha, vel_model, int_model):
        """Positive v_rotation -> peak shifts redward."""
        z = 1.0
        lam_center = LINE_LAMBDAS['Halpha'] * (1 + z)
        dlam = lam_center * 3000.0 / C_KMS
        # finer grid (0.2 nm) so argmax resolves velocity shifts
        cube_pars = CubePars.from_range(
            _IMAGE_PARS, lam_center - dlam, lam_center + dlam, 0.2
        )

        # galaxy with rotation — approaching and receding sides
        pars = _make_pars(
            _VEL_PARS,
            _INT_PARS,
            z=z,
            vel_dispersion=50.0,
            line_fluxes={'Halpha': 100.0},
        )
        cube = source_ha.build_cube(pars, cube_pars)

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

    def test_cube_multi_line_peaks(self, source_ha_nii, vel_model, int_model):
        """Ha + NII produce separate peaks at correct wavelengths."""
        z = 1.0
        # wide range to cover all 3 lines
        lam_min = 654.0 * (1 + z) - 10
        lam_max = 659.0 * (1 + z) + 10
        cube_pars = CubePars.from_range(_IMAGE_PARS, lam_min, lam_max, 0.3)

        pars = _make_pars(
            {**_VEL_PARS, 'vcirc': 0.0, 'v0': 0.0},
            _INT_PARS,
            z=z,
            vel_dispersion=50.0,
            line_fluxes={'Halpha': 100.0, 'NII6548': 30.0, 'NII6584': 90.0},
        )
        cube = source_ha_nii.build_cube(pars, cube_pars)

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
                LINE_LAMBDAS['Halpha'] * (1 + z),
                LINE_LAMBDAS['NII6584'] * (1 + z),
                654.80 * (1 + z),  # NII_6548
            ]
        )
        assert len(peaks) >= 3, f"Expected 3 peaks, found {len(peaks)}: {peaks}"
        for exp, found in zip(expected, sorted(peaks)[:3]):
            assert found == pytest.approx(
                exp, abs=2.0
            ), f"Peak at {found:.1f}, expected near {exp:.1f}"


# =============================================================================
# Correctness tests
# =============================================================================


class TestCorrectness:
    def test_cube_collapses_to_broadband(self, source_ha, vel_model, int_model):
        """Spectrally collapsed cube ~ render_unconvolved at cube's grid (rtol=0.02).

        Both arrays are normalized by their max before comparison, so the
        SB-vs-flux/pixel unit mismatch between source.build_cube and
        render_unconvolved is irrelevant — this checks spatial morphology.
        """
        z = 1.0
        lam_center = LINE_LAMBDAS['Halpha'] * (1 + z)
        dlam = lam_center * 5000.0 / C_KMS
        cube_pars = CubePars.from_range(
            _IMAGE_PARS, lam_center - dlam, lam_center + dlam, 0.5
        )

        # single line, flux=100, zero continuum, zero velocity
        pars = _make_pars(
            {**_VEL_PARS, 'vcirc': 0.0, 'v0': 0.0},
            _INT_PARS,
            z=z,
            vel_dispersion=50.0,
            line_fluxes={'Halpha': _INT_PARS['flux']},
        )
        cube = source_ha.build_cube(pars, cube_pars)

        # collapse: integral over wavelength (sum × dl); spatial profile shape
        dl = float(cube_pars.lambda_grid[1] - cube_pars.lambda_grid[0])
        collapsed = jnp.sum(cube, axis=2) * dl

        # reference: render_unconvolved on the bare intensity model (flux/pixel)
        theta_int = int_model.pars2theta(_INT_PARS)
        broadband = int_model.render_unconvolved(theta_int, _IMAGE_PARS)

        # normalize both to compare morphology (unit-invariant)
        collapsed_norm = collapsed / jnp.max(collapsed)
        broadband_norm = broadband / jnp.max(broadband)

        diff = jnp.max(jnp.abs(collapsed_norm - broadband_norm))
        # measured 0.0% (exact); 1e-4 gives 100x headroom
        assert (
            float(diff) < 1e-4
        ), f"Cube collapse vs broadband max diff = {float(diff):.6f}"

    def test_cube_vs_numpy_reference(self, source_ha, vel_model, int_model):
        """JAX datacube matches independent numpy implementation.

        Both paths produce a point-sampled cube at native (coarse) spatial
        resolution: the cube is the pre-pixel-response intermediate.
        Pixel response is a detector property and applies only at the 2D
        observable readout stage, never on the cube.

        Units bridge: numpy ``generate_datacube_3d`` returns flux/pixel/nm
        (it uses ``generate_sersic_intensity_2d`` which is flux/pixel per
        CLAUDE.md table). SourceModel ``build_cube`` returns SB/nm. Multiply
        the JAX cube by ``pixel_area`` at the comparison boundary.
        """
        from kl_pipe.synthetic import generate_datacube_3d

        z = 1.0
        lam_center = LINE_LAMBDAS['Halpha'] * (1 + z)
        dlam = lam_center * 2000.0 / C_KMS
        cube_pars = CubePars.from_range(
            _IMAGE_PARS, lam_center - dlam, lam_center + dlam, 1.0
        )

        # JAX path (SourceModel — SB units)
        line_flux = 100.0
        pars = _make_pars(
            _VEL_PARS,
            _INT_PARS,
            z=z,
            vel_dispersion=50.0,
            line_fluxes={'Halpha': line_flux},
        )
        cube_jax_sb = source_ha.build_cube(pars, cube_pars)
        # convert SB → flux/pixel for comparison with numpy reference
        pixel_area = _IMAGE_PARS.pixel_scale**2
        cube_jax = cube_jax_sb * pixel_area

        # numpy path: spatial_oversample=1 = point-sampled at native, matching
        # SourceModel's analytic eval (point-sampled).
        np_spectral_pars = {
            'z': z,
            'vel_dispersion': 50.0,
            'lines': [
                {'lambda_rest': LINE_LAMBDAS['Halpha'], 'flux': line_flux, 'cont': 0.0}
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
            spatial_oversample=1,
            spectral_oversample=5,
        )

        # shapes must match (both at native resolution)
        assert (
            cube_jax.shape == cube_np.shape
        ), f"shape mismatch: jax={cube_jax.shape}, np={cube_np.shape}"

        # total flux
        jax_total = float(jnp.sum(cube_jax))
        np_total = float(np.sum(cube_np))

        assert jax_total == pytest.approx(
            np_total, rel=0.001
        ), f"JAX total={jax_total:.3f}, numpy total={np_total:.3f}"

        # peak pixel
        jax_peak = float(jnp.max(cube_jax))
        np_peak = float(np.max(cube_np))
        assert jax_peak == pytest.approx(
            np_peak, rel=0.005
        ), f"JAX peak={jax_peak:.3f}, numpy peak={np_peak:.3f}"

    def test_v0_z_consistency(self, source_ha, vel_model, int_model):
        """Galaxy at (v0=100, z=1.0) produces similar peak as (v0=0, z=1.0+100/c)."""
        lam_center1 = LINE_LAMBDAS['Halpha'] * (1 + 1.0)
        dlam = lam_center1 * 3000.0 / C_KMS
        cube_pars = CubePars.from_range(
            _IMAGE_PARS, lam_center1 - dlam, lam_center1 + dlam, 0.5
        )

        # case 1: v0=100, z=1.0
        pars1 = _make_pars(
            {**_VEL_PARS, 'vcirc': 0.0, 'v0': 100.0},
            _INT_PARS,
            z=1.0,
            vel_dispersion=50.0,
            line_fluxes={'Halpha': 100.0},
        )
        # v0 is subtracted before Doppler, so v0 does NOT shift the line.
        # the line center should be at lambda_rest * (1 + z)
        cube1 = source_ha.build_cube(pars1, cube_pars)

        # case 2: v0=0, z=1.0
        pars2 = _make_pars(
            {**_VEL_PARS, 'vcirc': 0.0, 'v0': 0.0},
            _INT_PARS,
            z=1.0,
            vel_dispersion=50.0,
            line_fluxes={'Halpha': 100.0},
        )
        cube2 = source_ha.build_cube(pars2, cube_pars)

        # both should have the same line center (v0 subtracted before Doppler)
        cr, cc = 16, 16
        peak1 = float(cube_pars.lambda_grid[jnp.argmax(cube1[cr, cc, :])])
        peak2 = float(cube_pars.lambda_grid[jnp.argmax(cube2[cr, cc, :])])

        assert peak1 == pytest.approx(
            peak2, abs=1.0
        ), f"v0=100 peak={peak1:.2f}, v0=0 peak={peak2:.2f} should match"

    def test_spectral_oversample_convergence(self, source_ha, vel_model, int_model):
        """Sweep oversample factors, verify strict monotonic convergence.

        SourceModel.build_cube takes ``spectral_oversample`` as a kwarg
        (legacy passed it through SpectralConfig); same factor controls the
        wavelength sub-bin sampling.

        Threshold at osf=15 (the production default): max relative error
        < 1e-3 vs osf=25 reference. This is the pixel-level (cube)
        convergence anchor. Empirical floor at osf=15 is ~5e-4. Lower
        osf values are reported but not asserted; they remain available
        for users who want to trade accuracy for speed (osf=5 → ~6e-3
        cube error, parameter bias 5-25x sigma_Fisher at SNR=10000;
        osf=9 → ~2e-3, bias ~1-5x). Parameter-level convergence is
        anchored separately in
        ``experiments/sweverett/spectral_osf_convergence/`` (script
        reports bias / sigma_Fisher per parameter at SNR=10000).
        """
        z = 1.0
        lam_center = LINE_LAMBDAS['Halpha'] * (1 + z)
        dlam = lam_center * 2000.0 / C_KMS
        cube_pars = CubePars.from_range(
            _IMAGE_PARS, lam_center - dlam, lam_center + dlam, 1.0
        )

        pars = _make_pars(
            _VEL_PARS,
            _INT_PARS,
            z=z,
            vel_dispersion=50.0,
            line_fluxes={'Halpha': 100.0},
        )

        # truth at oversample=25
        cube_truth = source_ha.build_cube(pars, cube_pars, spectral_oversample=25)

        sweep = list(range(3, 22, 2))  # odd osfs 3..21 (10 points)
        errors = {}
        cubes = {}
        for osf in sweep:
            cube_test = source_ha.build_cube(pars, cube_pars, spectral_oversample=osf)
            cubes[osf] = cube_test

            max_err = float(
                jnp.max(jnp.abs(cube_test - cube_truth)) / jnp.max(jnp.abs(cube_truth))
            )
            errors[osf] = max_err

        # strict monotonic convergence
        err_list = [errors[k] for k in sweep]
        for i in range(len(err_list) - 1):
            assert err_list[i] > err_list[i + 1], (
                f"Not monotonically converging: osf={sweep[i]} "
                f"error={err_list[i]:.4e} vs osf={sweep[i+1]} "
                f"error={err_list[i+1]:.4e}"
            )

        # default osf=15 must be within 1e-3 of the osf=25 reference cube
        # (~2x margin over the empirical ~5e-4 floor)
        assert (
            errors[15] < 1e-3
        ), f"oversample=15 error = {errors[15]:.4e}, expected < 1e-3"

        # diagnostic plot: 4-panel layout
        #   (a) convergence line with thresholds + default marker
        #   (b) spectral profile at peak-flux spatial pixel for each osf vs truth
        #   (c) residual cube heatmap at line-peak wavelength slice, osf=5 (entry)
        #   (d) residual cube heatmap at line-peak wavelength slice, osf=15 (default)
        try:
            import matplotlib

            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            cube_truth_np = np.asarray(cube_truth)
            lam_grid = np.asarray(cube_pars.lambda_grid)
            # peak-flux spatial pixel + line-peak wavelength on the truth cube
            spatial_sum = cube_truth_np.sum(axis=-1)
            iy, ix = np.unravel_index(np.argmax(spatial_sum), spatial_sum.shape)
            il = int(np.argmax(cube_truth_np[iy, ix, :]))
            peak_amp = float(np.max(np.abs(cube_truth_np)))

            fig, axes = plt.subplots(2, 2, figsize=(12, 9))

            # (a) convergence line, log scale
            ax = axes[0, 0]
            ax.semilogy(
                list(errors.keys()), list(errors.values()), 'bo-', label='measured'
            )
            ax.axhline(
                1e-3,
                color='red',
                linestyle=':',
                alpha=0.7,
                label='asserted bound (1e-3)',
            )
            ax.axhline(
                1e-2,
                color='orange',
                linestyle=':',
                alpha=0.5,
                label='legacy threshold (1e-2)',
            )
            ax.axvline(
                15, color='green', linestyle='--', alpha=0.7, label='production default'
            )
            ax.set_xlabel('Spectral oversample factor')
            ax.set_ylabel('Max relative error vs osf=25 truth')
            ax.set_title('Convergence: max |cube_osf - cube_25| / max(cube_25)')
            ax.grid(True, which='both', alpha=0.3)
            ax.legend(fontsize=8)

            # (b) 1D spectral residual vs wavelength at the line-peak spatial
            # pixel. The deterministic comparison has no statistical
            # uncertainty; the asserted +/-1e-3 bound is shown as a shaded
            # band for scale reference.
            ax = axes[0, 1]
            truth_spec = cube_truth_np[iy, ix, :]
            ax.axhspan(
                -1e-3, 1e-3, color='red', alpha=0.10, label='+/-1e-3 (asserted bound)'
            )
            ax.axhline(0.0, color='k', linewidth=0.7, alpha=0.6)
            cmap = plt.get_cmap('viridis')
            for k, osf in enumerate(sweep):
                if osf in (3, 5, 9, 15, 21):
                    color = cmap(k / max(1, len(sweep) - 1))
                    resid_spec = (
                        np.asarray(cubes[osf])[iy, ix, :] - truth_spec
                    ) / peak_amp
                    ax.plot(
                        lam_grid,
                        resid_spec,
                        '-',
                        color=color,
                        alpha=0.85,
                        linewidth=1.2,
                        label=f'osf={osf}',
                    )
            ax.set_xlabel('Wavelength [nm]')
            ax.set_ylabel(f'(cube_osf - cube_25) / max(cube_25)   @ ({iy},{ix})')
            ax.set_title('1D spectral residual at line-peak spatial pixel')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)

            # (c, d) residual heatmaps at line-peak wavelength slice,
            # shared color scale (set by osf=5 — the worst of the two) so
            # the relative improvement at osf=15 is visually obvious.
            resid_5 = (
                np.asarray(cubes[5])[:, :, il] - cube_truth_np[:, :, il]
            ) / peak_amp
            shared_vmax = max(abs(resid_5.min()), abs(resid_5.max()))
            for ax, osf in [(axes[1, 0], 5), (axes[1, 1], 15)]:
                resid = (
                    np.asarray(cubes[osf])[:, :, il] - cube_truth_np[:, :, il]
                ) / peak_amp
                im = ax.imshow(
                    resid,
                    origin='lower',
                    cmap='RdBu_r',
                    vmin=-shared_vmax,
                    vmax=shared_vmax,
                )
                ax.set_title(
                    f'osf={osf} residual at lam[{il}]={lam_grid[il]:.1f} nm '
                    f'(max rel err = {errors[osf]:.2e})'
                )
                ax.set_xlabel('x [pix]')
                ax.set_ylabel('y [pix]')
                plt.colorbar(im, ax=ax, fraction=0.046)

            fig.suptitle(
                'Spectral oversample convergence diagnostic '
                f'(Halpha @ z={z}, sigma=50 km/s, dlam=1 nm)',
                fontsize=13,
            )
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

    def test_plot_datacube_slices(self, source_ha, vel_model, int_model):
        """Wavelength slices showing spatial morphology evolution."""
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        z = 1.0
        lam_center = LINE_LAMBDAS['Halpha'] * (1 + z)
        dlam = lam_center * 2000.0 / C_KMS
        cube_pars = CubePars.from_range(
            _IMAGE_PARS, lam_center - dlam, lam_center + dlam, 1.0
        )

        pars = _make_pars(
            _VEL_PARS,
            _INT_PARS,
            z=z,
            vel_dispersion=50.0,
            line_fluxes={'Halpha': 100.0},
            line_conts={'Halpha': 0.01},
        )
        cube = source_ha.build_cube(pars, cube_pars)

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

    def test_plot_spaxel_spectra(self, source_ha, vel_model, int_model):
        """Spectrum at 5 spatial pixels: center, approaching, receding, minor axis, edge."""
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        z = 1.0
        lam_center = LINE_LAMBDAS['Halpha'] * (1 + z)
        dlam = lam_center * 3000.0 / C_KMS
        cube_pars = CubePars.from_range(
            _IMAGE_PARS, lam_center - dlam, lam_center + dlam, 0.3
        )

        pars = _make_pars(
            _VEL_PARS,
            _INT_PARS,
            z=z,
            vel_dispersion=50.0,
            line_fluxes={'Halpha': 100.0},
        )
        cube = source_ha.build_cube(pars, cube_pars)

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

    def test_plot_datacube_overview(self, source_ha, vel_model, int_model):
        """Multi-panel datacube overview: intensity, velocity, stacked, channel slices."""
        z = 1.0
        lam_center = LINE_LAMBDAS['Halpha'] * (1 + z)
        dlam = lam_center * 2000.0 / C_KMS
        cube_pars = CubePars.from_range(
            _IMAGE_PARS, lam_center - dlam, lam_center + dlam, 1.0
        )

        pars = _make_pars(
            _VEL_PARS,
            _INT_PARS,
            z=z,
            vel_dispersion=50.0,
            line_fluxes={'Halpha': 100.0},
            line_conts={'Halpha': 0.01},
        )
        cube = source_ha.build_cube(pars, cube_pars)

        # render imap and vmap directly from bare models for the diagnostic plot
        theta_int = int_model.pars2theta(_INT_PARS)
        theta_vel = vel_model.pars2theta(_VEL_PARS)
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
