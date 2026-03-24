"""
Tests for grism forward model: SpectralModel params, dispersion, KLModel integration.

Datacube-specific tests live in test_datacube.py.
Diagnostic plots saved to tests/out/grism/.
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
from kl_pipe.model import KLModel
from kl_pipe.spectral import (
    LineSpec,
    EmissionLine,
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
from kl_pipe.dispersion import (
    GrismPars,
    disperse_cube,
    build_grism_pars_for_line,
)
from kl_pipe.utils import build_map_grid_from_image_pars
from kl_pipe.diagnostics.grism import (
    plot_grism_overview,
    plot_dispersion_angles,
    plot_dispersion_angle_study,
)

# output directory for diagnostic plots
OUT_DIR = os.path.join(os.path.dirname(__file__), 'out', 'grism')
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


def _build_composite_theta(vel_pars, int_pars, spec_pars_dict, kl_model):
    """Build composite theta array from dicts + KLModel."""
    merged = {}
    merged.update(vel_pars)
    merged.update(int_pars)
    merged.update(spec_pars_dict)
    return kl_model.pars2theta(merged)


# =============================================================================
# SpectralModel parameter tests
# =============================================================================


class TestSpectralModelParameters:
    def test_parameter_names_single_line(self, vel_model, int_model):
        """Ha, flux only -> z, vel_dispersion, Ha_flux, Ha_cont."""
        config = SpectralConfig(lines=(halpha_line(),))
        sm = SpectralModel(config, int_model, vel_model)
        assert 'z' in sm.PARAMETER_NAMES
        assert 'vel_dispersion' in sm.PARAMETER_NAMES
        assert 'Ha_flux' in sm.PARAMETER_NAMES
        assert 'Ha_cont' in sm.PARAMETER_NAMES
        assert len(sm.PARAMETER_NAMES) == 4

    def test_parameter_names_multi_param(self, vel_model, int_model):
        """Ha with {flux, int_rscale} -> z, vel_disp, Ha_flux, Ha_int_rscale, Ha_cont."""
        line = EmissionLine(
            line_spec=HALPHA, own_params=frozenset({'flux', 'int_rscale'})
        )
        config = SpectralConfig(lines=(line,))
        sm = SpectralModel(config, int_model, vel_model)
        assert 'Ha_flux' in sm.PARAMETER_NAMES
        assert 'Ha_int_rscale' in sm.PARAMETER_NAMES
        assert 'Ha_cont' in sm.PARAMETER_NAMES
        # z, vel_dispersion, Ha_flux, Ha_int_rscale, Ha_cont = 5
        assert len(sm.PARAMETER_NAMES) == 5

    def test_parameter_names_multi_line(self, spec_model_ha_nii):
        """Ha + NII_6548 + NII_6583 -> 8 params."""
        names = spec_model_ha_nii.PARAMETER_NAMES
        assert 'Ha_flux' in names
        assert 'Ha_cont' in names
        assert 'NII_6548_flux' in names
        assert 'NII_6548_cont' in names
        assert 'NII_6583_flux' in names
        assert 'NII_6583_cont' in names
        # z, vel_dispersion + 3*(flux + cont) = 2 + 6 = 8
        assert len(names) == 8

    def test_per_line_cont_always_present(self, vel_model, int_model):
        """Every line gets {prefix}_cont even if not in own_params."""
        line = EmissionLine(line_spec=HALPHA, own_params=frozenset())
        config = SpectralConfig(lines=(line,))
        sm = SpectralModel(config, int_model, vel_model)
        assert 'Ha_cont' in sm.PARAMETER_NAMES

    def test_build_line_theta_int(self, vel_model, int_model):
        """Per-line theta override swaps flux correctly."""
        line = EmissionLine(line_spec=HALPHA, own_params=frozenset({'flux'}))
        config = SpectralConfig(lines=(line,))
        sm = SpectralModel(config, int_model, vel_model)

        # broadband theta_int
        theta_int = int_model.pars2theta(_INT_PARS)

        # spectral theta: Ha_flux=50, Ha_cont=0.01
        theta_spec = jnp.array([1.0, 50.0, 50.0, 0.01])  # z, vel_disp, Ha_flux, Ha_cont

        theta_line = sm._build_line_theta_int(theta_spec, theta_int, 0)

        # flux should be overridden to 50.0
        flux_idx = list(int_model.PARAMETER_NAMES).index('flux')
        assert float(theta_line[flux_idx]) == pytest.approx(50.0)

        # other params unchanged
        cosi_idx = list(int_model.PARAMETER_NAMES).index('cosi')
        assert float(theta_line[cosi_idx]) == pytest.approx(_INT_PARS['cosi'])

    def test_invalid_own_params_raises(self, vel_model, int_model):
        """own_params with non-existent IntensityModel param raises ValueError."""
        line = EmissionLine(
            line_spec=HALPHA, own_params=frozenset({'nonexistent_param'})
        )
        config = SpectralConfig(lines=(line,))
        with pytest.raises(ValueError, match="not in IntensityModel.PARAMETER_NAMES"):
            SpectralModel(config, int_model, vel_model)


# =============================================================================
# Dispersion tests
# =============================================================================


class TestDispersion:
    def test_disperse_shape(self):
        """Output shape matches spatial grid."""
        Nrow, Ncol, Nlam = 16, 16, 20
        cube = jnp.ones((Nrow, Ncol, Nlam))
        ip = ImagePars(shape=(Nrow, Ncol), pixel_scale=0.11, indexing='ij')
        gp = GrismPars(
            image_pars=ip,
            dispersion=1.1,
            lambda_ref=1312.0,
            dispersion_angle=0.0,
        )
        lam = jnp.linspace(1300, 1320, Nlam)
        result = disperse_cube(cube, gp, lam)
        assert result.shape == (Nrow, Ncol)

    def test_disperse_flux_conservation(self):
        """Total flux conserved through dispersion."""
        Nrow, Ncol, Nlam = 32, 32, 20
        # point source at center, one wavelength slice
        cube = jnp.zeros((Nrow, Ncol, Nlam))
        cube = cube.at[16, 16, 10].set(1.0)

        ip = ImagePars(shape=(Nrow, Ncol), pixel_scale=0.11, indexing='ij')
        lam = jnp.linspace(1300, 1320, Nlam)
        gp = GrismPars(
            image_pars=ip,
            dispersion=1.1,
            lambda_ref=float(lam[10]),
            dispersion_angle=0.0,
        )
        dlam = float(lam[1] - lam[0])

        result = disperse_cube(cube, gp, lam)
        total_input = float(jnp.sum(cube)) * dlam
        total_output = float(jnp.sum(result))

        # measured 0.0% (exact to float64)
        assert total_output == pytest.approx(total_input, rel=1e-6)

    def test_disperse_angle_0(self):
        """Dispersion along x (angle=0) shifts horizontally."""
        Nrow, Ncol, Nlam = 32, 32, 5
        cube = jnp.zeros((Nrow, Ncol, Nlam))
        # point source at center
        cube = cube.at[16, 16, :].set(jnp.array([0.0, 0.5, 1.0, 0.5, 0.0]))

        ip = ImagePars(shape=(Nrow, Ncol), pixel_scale=0.11, indexing='ij')
        lam = jnp.linspace(1308, 1316, Nlam)
        gp = GrismPars(
            image_pars=ip,
            dispersion=1.1,
            lambda_ref=float(lam[2]),
            dispersion_angle=0.0,
        )

        result = disperse_cube(cube, gp, lam)

        # peak should be at center row, but spread along cols
        assert float(result[16, 16]) > 0
        # signal should spread along x (cols), not y (rows)
        col_profile = np.array(result[16, :])
        row_profile = np.array(result[:, 16])
        assert np.std(col_profile) > np.std(row_profile)

    def test_disperse_angle_90(self):
        """Dispersion along y (angle=pi/2) shifts vertically."""
        Nrow, Ncol, Nlam = 32, 32, 5
        cube = jnp.zeros((Nrow, Ncol, Nlam))
        cube = cube.at[16, 16, :].set(jnp.array([0.0, 0.5, 1.0, 0.5, 0.0]))

        ip = ImagePars(shape=(Nrow, Ncol), pixel_scale=0.11, indexing='ij')
        lam = jnp.linspace(1308, 1316, Nlam)
        gp = GrismPars(
            image_pars=ip,
            dispersion=1.1,
            lambda_ref=float(lam[2]),
            dispersion_angle=jnp.pi / 2,
        )

        result = disperse_cube(cube, gp, lam)

        # signal should spread along y (rows), not x (cols)
        col_profile = np.array(result[16, :])
        row_profile = np.array(result[:, 16])
        assert np.std(row_profile) > np.std(col_profile)


# =============================================================================
# KLModel integration tests
# =============================================================================


class TestKLModelIntegration:
    def test_backwards_compat(self, vel_model, int_model):
        """spectral_model=None works, existing behavior unchanged."""
        kl = KLModel(vel_model, int_model, shared_pars=_SHARED_PARS)
        assert kl.spectral_model is None
        assert kl._spectral_indices is None
        assert 'v0' in kl.PARAMETER_NAMES
        assert 'flux' in kl.PARAMETER_NAMES

    def test_3way_parameter_merge(self, vel_model, int_model, spec_model_ha):
        """Correct PARAMETER_NAMES with 3 components."""
        kl = KLModel(
            vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=spec_model_ha
        )

        # vel params come first, then unique int params, then unique spec params
        names = kl.PARAMETER_NAMES
        assert names.index('v0') < names.index('flux')
        assert names.index('flux') < names.index('z')
        assert 'z' in names
        assert 'vel_dispersion' in names
        assert 'Ha_flux' in names
        assert 'Ha_cont' in names

        # shared params appear once
        assert names.count('cosi') == 1
        assert names.count('theta_int') == 1

    def test_param_slicing_roundtrip(self, vel_model, int_model, spec_model_ha):
        """get_vel/int/spec_pars all extract correct values."""
        kl = KLModel(
            vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=spec_model_ha
        )

        spec_pars_dict = {
            'z': 1.0,
            'vel_dispersion': 50.0,
            'Ha_flux': 100.0,
            'Ha_cont': 0.01,
        }
        merged = {}
        merged.update(_VEL_PARS)
        merged.update(_INT_PARS)
        merged.update(spec_pars_dict)
        theta = kl.pars2theta(merged)

        theta_vel = kl.get_velocity_pars(theta)
        theta_int = kl.get_intensity_pars(theta)
        theta_spec = kl.get_spectral_pars(theta)

        # check roundtrip
        for name, val in _VEL_PARS.items():
            idx = list(vel_model.PARAMETER_NAMES).index(name)
            assert float(theta_vel[idx]) == pytest.approx(val, abs=1e-10)

        for name, val in _INT_PARS.items():
            idx = list(int_model.PARAMETER_NAMES).index(name)
            assert float(theta_int[idx]) == pytest.approx(val, abs=1e-10)

        for name, val in spec_pars_dict.items():
            idx = list(spec_model_ha.PARAMETER_NAMES).index(name)
            assert float(theta_spec[idx]) == pytest.approx(val, abs=1e-10)

    def test_render_cube_via_klmodel(
        self, vel_model, int_model, spec_model_ha, cube_pars
    ):
        """render_cube returns correct shape."""
        kl = KLModel(
            vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=spec_model_ha
        )

        spec_dict = {
            'z': 1.0,
            'vel_dispersion': 50.0,
            'Ha_flux': 100.0,
            'Ha_cont': 0.01,
        }
        merged = {**_VEL_PARS, **_INT_PARS, **spec_dict}
        theta = kl.pars2theta(merged)

        cube = kl.render_cube(theta, cube_pars)
        assert cube.shape == (32, 32, cube_pars.n_lambda)

    def test_render_grism_via_klmodel(self, vel_model, int_model, spec_model_ha):
        """render_grism returns correct shape."""
        kl = KLModel(
            vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=spec_model_ha
        )

        gp = build_grism_pars_for_line(
            HALPHA.lambda_rest,
            redshift=1.0,
            image_pars=_IMAGE_PARS,
            dispersion=1.1,
        )

        spec_dict = {
            'z': 1.0,
            'vel_dispersion': 50.0,
            'Ha_flux': 100.0,
            'Ha_cont': 0.01,
        }
        merged = {**_VEL_PARS, **_INT_PARS, **spec_dict}
        theta = kl.pars2theta(merged)

        grism = kl.render_grism(theta, gp)
        assert grism.shape == (32, 32)
        assert float(jnp.sum(grism)) > 0

    def test_render_cube_jit(self, vel_model, int_model, spec_model_ha, cube_pars):
        """jax.jit compiles and runs for render_cube."""
        kl = KLModel(
            vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=spec_model_ha
        )

        spec_dict = {
            'z': 1.0,
            'vel_dispersion': 50.0,
            'Ha_flux': 100.0,
            'Ha_cont': 0.01,
        }
        merged = {**_VEL_PARS, **_INT_PARS, **spec_dict}
        theta = kl.pars2theta(merged)

        from functools import partial

        render_jit = jax.jit(partial(kl.render_cube, cube_pars=cube_pars))
        cube = render_jit(theta)
        assert cube.shape == (32, 32, cube_pars.n_lambda)

    def test_render_grism_jit(self, vel_model, int_model, spec_model_ha):
        """jax.jit compiles and runs for render_grism."""
        kl = KLModel(
            vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=spec_model_ha
        )

        gp = build_grism_pars_for_line(
            HALPHA.lambda_rest,
            redshift=1.0,
            image_pars=_IMAGE_PARS,
            dispersion=1.1,
        )
        # pre-compute cube_pars with concrete z for JIT compatibility
        cp = gp.to_cube_pars(z=1.0)

        spec_dict = {
            'z': 1.0,
            'vel_dispersion': 50.0,
            'Ha_flux': 100.0,
            'Ha_cont': 0.01,
        }
        merged = {**_VEL_PARS, **_INT_PARS, **spec_dict}
        theta = kl.pars2theta(merged)

        from functools import partial

        render_jit = jax.jit(partial(kl.render_grism, grism_pars=gp, cube_pars=cp))
        grism = render_jit(theta)
        assert grism.shape == (32, 32)

    def test_render_grism_differentiable(self, vel_model, int_model, spec_model_ha):
        """jax.grad w.r.t. theta succeeds for grism rendering."""
        kl = KLModel(
            vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=spec_model_ha
        )

        gp = build_grism_pars_for_line(
            HALPHA.lambda_rest,
            redshift=1.0,
            image_pars=_IMAGE_PARS,
            dispersion=1.1,
        )
        # pre-compute cube_pars with concrete z for grad compatibility
        cp = gp.to_cube_pars(z=1.0)

        spec_dict = {
            'z': 1.0,
            'vel_dispersion': 50.0,
            'Ha_flux': 100.0,
            'Ha_cont': 0.01,
        }
        merged = {**_VEL_PARS, **_INT_PARS, **spec_dict}
        theta = kl.pars2theta(merged)

        def loss(th):
            grism = kl.render_grism(th, gp, cube_pars=cp)
            return jnp.sum(grism**2)

        grad_fn = jax.grad(loss)
        g = grad_fn(theta)
        assert g.shape == theta.shape
        assert jnp.isfinite(g).all()

    def test_render_dispatch(self, vel_model, int_model, spec_model_ha, cube_pars):
        """render(data_type='cube') and render(data_type='grism') dispatch correctly."""
        kl = KLModel(
            vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=spec_model_ha
        )

        spec_dict = {
            'z': 1.0,
            'vel_dispersion': 50.0,
            'Ha_flux': 100.0,
            'Ha_cont': 0.01,
        }
        merged = {**_VEL_PARS, **_INT_PARS, **spec_dict}
        theta = kl.pars2theta(merged)

        cube = kl.render(theta, 'cube', cube_pars)
        assert cube.ndim == 3

        gp = build_grism_pars_for_line(
            HALPHA.lambda_rest,
            redshift=1.0,
            image_pars=_IMAGE_PARS,
            dispersion=1.1,
        )
        grism = kl.render(theta, 'grism', gp)
        assert grism.ndim == 2


# =============================================================================
# Correctness tests
# =============================================================================


class TestCorrectness:
    def test_dispersion_sign(self):
        """Single-pixel source at known lambda: verify shift direction."""
        Nrow, Ncol, Nlam = 32, 32, 11
        cube = jnp.zeros((Nrow, Ncol, Nlam))
        # source at center, wavelength at idx 8 (redder than ref at idx 5)
        cube = cube.at[16, 16, 8].set(1.0)

        ip = ImagePars(shape=(Nrow, Ncol), pixel_scale=0.11, indexing='ij')
        lam = jnp.linspace(1300, 1310, Nlam)
        gp = GrismPars(
            image_pars=ip,
            dispersion=1.0,
            lambda_ref=float(lam[5]),
            dispersion_angle=0.0,
        )

        result = disperse_cube(cube, gp, lam)

        # redder wavelength -> positive pixel offset along x
        # so the peak should appear at col > 16
        peak_col = int(jnp.argmax(result[16, :]))
        assert peak_col > 16, f"Expected peak at col > 16, got {peak_col}"

    def test_grism_with_shear(self, vel_model, int_model):
        """Non-zero g1=0.05: transforms + dispersion compose correctly."""
        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)

        kl = KLModel(vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=sm)

        gp = build_grism_pars_for_line(
            HALPHA.lambda_rest,
            redshift=1.0,
            image_pars=_IMAGE_PARS,
            dispersion=1.1,
        )

        shear_pars = {**_VEL_PARS, **_INT_PARS, 'g1': 0.05, 'g2': 0.0}
        spec_dict = {
            'z': 1.0,
            'vel_dispersion': 50.0,
            'Ha_flux': 100.0,
            'Ha_cont': 0.01,
        }
        merged = {**shear_pars, **spec_dict}
        theta = kl.pars2theta(merged)

        grism = kl.render_grism(theta, gp)
        assert grism.shape == (32, 32)
        assert float(jnp.sum(grism)) > 0
        assert jnp.isfinite(grism).all()

    def test_edge_on_galaxy(self, vel_model, int_model):
        """cosi=0.1: large velocity gradient + spectral line broadening."""
        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)

        z = 1.0
        lam_center = HALPHA.lambda_rest * (1 + z)
        dlam = lam_center * 3000.0 / C_KMS
        cube_pars = CubePars.from_range(
            _IMAGE_PARS, lam_center - dlam, lam_center + dlam, 0.5
        )

        edge_on_pars = {**_VEL_PARS, 'cosi': 0.1}
        theta_vel = vel_model.pars2theta(edge_on_pars)
        theta_int = int_model.pars2theta({**_INT_PARS, 'cosi': 0.1})
        theta_spec = jnp.array([z, 50.0, 100.0, 0.0])

        cube = sm.build_cube(theta_spec, theta_vel, theta_int, cube_pars)

        # spatially integrated spectrum should be broader than face-on
        total_spec = jnp.sum(cube, axis=(0, 1))
        # verify it has significant width (velocity ~200 km/s * sin(84 deg))
        above_half = total_spec > 0.5 * jnp.max(total_spec)
        fwhm_pixels = int(jnp.sum(above_half))
        assert fwhm_pixels > 3, f"FWHM = {fwhm_pixels} pixels, expected > 3 for edge-on"

    def test_velocity_signature_antisymmetry(self, vel_model, int_model):
        """grism(rotating) - grism(vcirc=0) is antisymmetric along kinematic axis."""
        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)
        kl = KLModel(vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=sm)

        # theta_int=0 so kinematic axis is along x (cols)
        pars = {**_VEL_PARS, **_INT_PARS, 'theta_int': 0.0}
        spec_dict = {
            'z': 1.0,
            'vel_dispersion': 50.0,
            'Ha_flux': 100.0,
            'Ha_cont': 0.0,
        }
        merged = {**pars, **spec_dict}

        # dispersion along x to align with kinematic axis
        gp = GrismPars(
            image_pars=_IMAGE_PARS,
            dispersion=1.1,
            lambda_ref=HALPHA.lambda_rest * 2.0,
            dispersion_angle=0.0,
        )

        theta_rot = kl.pars2theta(merged)
        grism_rot = kl.render_grism(theta_rot, gp)

        merged_norot = {**merged, 'vcirc': 0.0}
        theta_norot = kl.pars2theta(merged_norot)
        grism_norot = kl.render_grism(theta_norot, gp)

        diff = np.array(grism_rot - grism_norot)

        # approaching vs receding halves: sum full 2D left/right of center col
        cc = _IMAGE_PARS.Ncol // 2
        left_sum = float(np.sum(diff[:, :cc]))
        right_sum = float(np.sum(diff[:, cc:]))

        # opposite signs = antisymmetric velocity signature
        assert left_sum * right_sum < 0, (
            f"Expected antisymmetric velocity signature: "
            f"left_sum={left_sum:.4f}, right_sum={right_sum:.4f}"
        )


# =============================================================================
# Diagnostic plots
# =============================================================================


class TestDiagnosticPlots:
    """Diagnostic plots saved to tests/out/grism/. Not pass/fail tests."""

    def test_plot_dispersed_image(self, vel_model, int_model):
        """2D grism image at dispersion angles 0, 45, 90 deg."""
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)

        kl = KLModel(vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=sm)

        spec_dict = {
            'z': 1.0,
            'vel_dispersion': 50.0,
            'Ha_flux': 100.0,
            'Ha_cont': 0.01,
        }
        merged = {**_VEL_PARS, **_INT_PARS, **spec_dict}
        theta = kl.pars2theta(merged)

        angles = [0, np.pi / 4, np.pi / 2]
        labels = ['0 deg', '45 deg', '90 deg']

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i, (angle, label) in enumerate(zip(angles, labels)):
            gp = GrismPars(
                image_pars=_IMAGE_PARS,
                dispersion=1.1,
                lambda_ref=HALPHA.lambda_rest * 2.0,
                dispersion_angle=angle,
            )
            grism = kl.render_grism(theta, gp)
            im = axes[i].imshow(np.array(grism), origin='lower')
            axes[i].set_title(f'Dispersion angle = {label}')
            plt.colorbar(im, ax=axes[i], fraction=0.046)
        fig.suptitle('Grism dispersion angle comparison')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, 'dispersed_image.png'), dpi=150)
        plt.close(fig)

    def test_plot_velocity_signature(self, vel_model, int_model):
        """Grism(rotating) - grism(vcirc=0) isolates pure velocity signature."""
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)
        kl = KLModel(vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=sm)

        gp = build_grism_pars_for_line(
            HALPHA.lambda_rest,
            redshift=1.0,
            image_pars=_IMAGE_PARS,
            dispersion=1.1,
        )

        spec_dict = {
            'z': 1.0,
            'vel_dispersion': 50.0,
            'Ha_flux': 100.0,
            'Ha_cont': 0.01,
        }

        # rotating galaxy
        merged = {**_VEL_PARS, **_INT_PARS, **spec_dict}
        theta_rot = kl.pars2theta(merged)
        grism_rot = kl.render_grism(theta_rot, gp)

        # non-rotating galaxy
        merged_norot = {**_VEL_PARS, **_INT_PARS, **spec_dict, 'vcirc': 0.0}
        theta_norot = kl.pars2theta(merged_norot)
        grism_norot = kl.render_grism(theta_norot, gp)

        diff = grism_rot - grism_norot

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        im0 = axes[0].imshow(np.array(grism_rot), origin='lower')
        axes[0].set_title('Rotating galaxy')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(np.array(grism_norot), origin='lower')
        axes[1].set_title('Non-rotating (vcirc=0)')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        vmax = float(jnp.max(jnp.abs(diff)))
        im2 = axes[2].imshow(
            np.array(diff), origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax
        )
        axes[2].set_title('Velocity signature (difference)')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        fig.suptitle('Velocity signature in grism')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, 'velocity_signature.png'), dpi=150)
        plt.close(fig)

    def test_plot_grism_overview(self, vel_model, int_model):
        """Master 3-row grism diagnostic."""
        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)
        kl = KLModel(vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=sm)

        gp = build_grism_pars_for_line(
            HALPHA.lambda_rest,
            redshift=1.0,
            image_pars=_IMAGE_PARS,
            dispersion=1.1,
        )
        cp = gp.to_cube_pars(z=1.0)

        spec_dict = {
            'z': 1.0,
            'vel_dispersion': 50.0,
            'Ha_flux': 100.0,
            'Ha_cont': 0.01,
        }
        merged = {**_VEL_PARS, **_INT_PARS, **spec_dict}
        theta = kl.pars2theta(merged)

        # rotating grism
        cube = kl.render_cube(theta, cp)
        grism_rot = kl.render_grism(theta, gp, cube_pars=cp)

        # non-rotating grism
        merged_norot = {**merged, 'vcirc': 0.0}
        theta_norot = kl.pars2theta(merged_norot)
        grism_norot = kl.render_grism(theta_norot, gp, cube_pars=cp)

        # imap and vmap
        theta_int = int_model.pars2theta(_INT_PARS)
        theta_vel = vel_model.pars2theta(_VEL_PARS)
        imap = int_model.render_unconvolved(theta_int, _IMAGE_PARS)
        X, Y = build_map_grid_from_image_pars(_IMAGE_PARS)
        vmap = vel_model(theta_vel, 'obs', X, Y)

        fig = plot_grism_overview(
            np.array(cube),
            np.array(grism_rot),
            np.array(cp.lambda_grid),
            gp,
            imap=np.array(imap),
            vmap=np.array(vmap),
            grism_norot=np.array(grism_norot),
            v0=float(_VEL_PARS['v0']),
            title='Grism overview',
            save_path=os.path.join(OUT_DIR, 'grism_overview.png'),
        )
        assert fig is not None

    def test_plot_dispersion_angles(self, vel_model, int_model):
        """Cardinal-angle grism comparison (0, 90, 180, 270 deg)."""
        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)
        kl = KLModel(vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=sm)

        spec_dict = {
            'z': 1.0,
            'vel_dispersion': 50.0,
            'Ha_flux': 100.0,
            'Ha_cont': 0.01,
        }
        merged = {**_VEL_PARS, **_INT_PARS, **spec_dict}
        theta = kl.pars2theta(merged)

        def build_grism_fn(angle):
            gp = GrismPars(
                image_pars=_IMAGE_PARS,
                dispersion=1.1,
                lambda_ref=HALPHA.lambda_rest * 2.0,
                dispersion_angle=angle,
            )
            return np.array(kl.render_grism(theta, gp))

        fig = plot_dispersion_angles(
            build_grism_fn,
            title='Dispersion angle comparison',
            save_path=os.path.join(OUT_DIR, 'dispersion_angles.png'),
        )
        assert fig is not None

    def test_plot_dispersion_angle_study(self, vel_model, int_model):
        """Deep-dive: theta_int=0, 4 cardinal angles, grism vs broadband."""
        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)
        kl = KLModel(vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=sm)

        # theta_int=0 so kinematic axis along x
        pars_study = {**_VEL_PARS, **_INT_PARS, 'theta_int': 0.0}
        spec_dict = {
            'z': 1.0,
            'vel_dispersion': 50.0,
            'Ha_flux': 100.0,
            'Ha_cont': 0.01,
        }
        merged = {**pars_study, **spec_dict}
        theta = kl.pars2theta(merged)

        # build broadband stacked (collapsed cube, no dispersion)
        gp_ref = build_grism_pars_for_line(
            HALPHA.lambda_rest,
            redshift=1.0,
            image_pars=_IMAGE_PARS,
            dispersion=1.1,
        )
        cp = gp_ref.to_cube_pars(z=1.0)
        cube = kl.render_cube(theta, cp)
        dlam = float(cp.lambda_grid[1] - cp.lambda_grid[0])
        broadband_stacked = np.array(jnp.sum(cube, axis=2) * dlam)

        def build_grism_fn(angle):
            gp = GrismPars(
                image_pars=_IMAGE_PARS,
                dispersion=1.1,
                lambda_ref=HALPHA.lambda_rest * 2.0,
                dispersion_angle=angle,
            )
            return np.array(kl.render_grism(theta, gp, cube_pars=cp))

        fig = plot_dispersion_angle_study(
            build_grism_fn,
            broadband_stacked,
            title='Dispersion angle study (theta_int=0)',
            save_path=os.path.join(OUT_DIR, 'dispersion_angle_study.png'),
        )
        assert fig is not None
