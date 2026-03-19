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
from scipy.ndimage import convolve1d

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
        import matplotlib.pyplot as plt

        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)
        kl = KLModel(vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=sm)

        # theta_int=0 so kinematic axis is along x (cols); v0=0 for clean antisymmetry
        pars = {**_VEL_PARS, **_INT_PARS, 'theta_int': 0.0, 'v0': 0.0}
        spec_dict = {
            'z': 1.0,
            'vel_dispersion': 50.0,
            'Ha_flux': 100.0,
            'Ha_cont': 0.0,
        }
        merged = {**pars, **spec_dict}

        # use larger odd grid to avoid edge flux loss
        gp = GrismPars(
            image_pars=_ANALYTICAL_IMAGE_PARS,
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

        # symmetric broadening overwhelms raw sums; extract the velocity dipole
        Ncol = _ANALYTICAL_IMAGE_PARS.Ncol
        diff_flip = diff[:, ::-1]
        asym = (diff - diff_flip) / 2.0
        sym = (diff + diff_flip) / 2.0
        cc = Ncol // 2
        left_asym = np.sum(asym[:, :cc])
        right_asym = np.sum(asym[:, cc + 1 :])

        # diagnostic
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        vmax_diff = max(np.abs(diff).max(), 1e-16)
        im0 = axes[0, 0].imshow(
            diff, origin='lower', cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff
        )
        axes[0, 0].set_title('Raw diff (rot − static)')
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

        vmax_asym = max(np.abs(asym).max(), 1e-16)
        im1 = axes[0, 1].imshow(
            asym, origin='lower', cmap='RdBu_r', vmin=-vmax_asym, vmax=vmax_asym
        )
        axes[0, 1].set_title('Antisymmetric (velocity dipole)')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

        vmax_sym = max(np.abs(sym).max(), 1e-16)
        im2 = axes[1, 0].imshow(
            sym, origin='lower', cmap='RdBu_r', vmin=-vmax_sym, vmax=vmax_sym
        )
        axes[1, 0].set_title('Symmetric (Doppler broadening)')
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

        col_sum_asym = asym.sum(axis=0)
        axes[1, 1].plot(np.arange(Ncol), col_sum_asym, 'k-', lw=1)
        axes[1, 1].axvline(cc, color='grey', ls='--', alpha=0.5)
        axes[1, 1].axhline(0, color='grey', ls='-', alpha=0.3)
        axes[1, 1].set_xlabel('Column')
        axes[1, 1].set_ylabel('Column-summed antisymmetric flux')
        axes[1, 1].set_title('Velocity dipole profile')

        ok = left_asym * right_asym < 0
        color = 'green' if ok else 'red'
        fig.suptitle(
            f'Velocity antisymmetry — {"PASS" if ok else "FAIL"} '
            f'(left={left_asym:.4f}, right={right_asym:.4f})'
            f'\nDecomposing (rotating − static) grism into symmetric + '
            f'antisymmetric parts isolates the approaching/receding velocity dipole',
            color=color,
            fontsize=11,
        )
        fig.tight_layout()
        _save_diagnostic(fig, 'velocity_antisymmetry')

        # opposite signs = antisymmetric velocity signature
        assert ok, (
            f"Expected antisymmetric velocity dipole: "
            f"left_asym={left_asym:.4f}, right_asym={right_asym:.4f}"
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


# =============================================================================
# Analytical pixel-level tests
# =============================================================================

# output subdirectory for analytical diagnostic plots
_ANALYTICAL_OUT = os.path.join(OUT_DIR, 'analytical')
os.makedirs(_ANALYTICAL_OUT, exist_ok=True)

# odd non-square grid: catches row/col transposition bugs AND places grid
# center at an integer pixel (avoids test-only half-pixel normalization artifact)
_ANALYTICAL_IMAGE_PARS = ImagePars(shape=(47, 63), pixel_scale=0.11, indexing='ij')
_ANALYTICAL_NROW, _ANALYTICAL_NCOL = 47, 63


def _save_diagnostic(fig, name):
    """Save figure to analytical output directory."""
    fig.savefig(
        os.path.join(_ANALYTICAL_OUT, f'{name}.png'), dpi=150, bbox_inches='tight'
    )
    import matplotlib.pyplot as plt

    plt.close(fig)


class TestAnalytical:
    """Pixel-level analytical tests for grism forward model.

    Every test constructs expected values from pure math (numpy/scipy),
    then compares against kl_pipe's build_cube / disperse_cube / render_grism.
    """

    # ------------------------------------------------------------------
    # 1. Gaussian source + Gaussian line = wider Gaussian
    # ------------------------------------------------------------------
    def test_gaussian_convolution(self):
        """Gaussian spatial * Gaussian spectral = wider Gaussian along dispersion."""
        import matplotlib.pyplot as plt

        Nrow, Ncol = _ANALYTICAL_NROW, _ANALYTICAL_NCOL
        pixel_scale = 0.11

        # spatial Gaussian (in pixel coords)
        sigma_s_pix = 3.0  # pixels
        row_center = (Nrow - 1) / 2.0
        col_center = (Ncol - 1) / 2.0
        rows = np.arange(Nrow, dtype=np.float64)
        cols = np.arange(Ncol, dtype=np.float64)
        R, C = np.meshgrid(rows, cols, indexing='ij')
        I_spatial = np.exp(
            -((R - row_center) ** 2 + (C - col_center) ** 2) / (2 * sigma_s_pix**2)
        )
        # normalize to unit peak
        I_spatial = I_spatial / I_spatial.max()

        # spectral Gaussian parameters
        sigma_lambda = 2.0  # nm
        # set dispersion = dlam so offsets are integers -> bilinear exact
        dlam = 1.0  # nm/pix AND nm spacing
        dispersion = dlam
        Nlam = 41
        lam_center = 1312.0  # nm
        lam_grid = np.linspace(
            lam_center - (Nlam - 1) / 2 * dlam,
            lam_center + (Nlam - 1) / 2 * dlam,
            Nlam,
        )

        # build synthetic datacube: I(x,y) * G(lam)
        G_lam = np.exp(-0.5 * ((lam_grid - lam_center) / sigma_lambda) ** 2)
        G_lam_norm = G_lam / (sigma_lambda * np.sqrt(2 * np.pi))
        cube = I_spatial[:, :, None] * G_lam_norm[None, None, :]

        # disperse with angle=0 (along cols), lambda_ref = lam_center
        ip = ImagePars(shape=(Nrow, Ncol), pixel_scale=pixel_scale, indexing='ij')
        gp = GrismPars(
            image_pars=ip,
            dispersion=dispersion,
            lambda_ref=lam_center,
            dispersion_angle=0.0,
        )
        grism = np.array(disperse_cube(jnp.array(cube), gp, jnp.array(lam_grid)))

        # expected: separable Gaussian with sigma_total along cols
        sigma_spectral_pix = sigma_lambda / dispersion
        sigma_total = np.sqrt(sigma_s_pix**2 + sigma_spectral_pix**2)

        # convolution of unit-peak Gaussian (sigma_s) with unit-area Gaussian
        # (sigma_spectral) gives Gaussian with peak = sigma_s / sigma_total
        expected_grism = (
            (sigma_s_pix / sigma_total)
            * np.exp(-((R - row_center) ** 2) / (2 * sigma_s_pix**2))
            * np.exp(-((C - col_center) ** 2) / (2 * sigma_total**2))
        )

        # with integer offsets and exact Gaussian, tolerance is very tight
        residual = np.abs(grism - expected_grism)
        max_resid = residual.max()
        peak = expected_grism.max()
        rel_max = max_resid / peak

        # diagnostic plot
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        im0 = axes[0, 0].imshow(expected_grism, origin='lower')
        axes[0, 0].set_title('Expected grism')
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

        im1 = axes[0, 1].imshow(grism, origin='lower')
        axes[0, 1].set_title('Actual grism')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

        vmax_r = max(residual.max(), 1e-16)
        im2 = axes[0, 2].imshow(
            residual, origin='lower', cmap='RdBu_r', vmin=-vmax_r, vmax=vmax_r
        )
        axes[0, 2].set_title(f'|Residual| (max={max_resid:.2e})')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

        # cross-dispersion 1D profile at center col
        cr = int(col_center)
        axes[1, 0].plot(rows, expected_grism[:, cr], 'k--', label='expected')
        axes[1, 0].plot(rows, grism[:, cr], 'g.', label='actual', ms=3)
        axes[1, 0].set_title('Cross-dispersion (center col)')
        axes[1, 0].legend(fontsize=8)

        # dispersion 1D profile at center row
        rr = int(row_center)
        axes[1, 1].plot(cols, expected_grism[rr, :], 'k--', label='expected')
        axes[1, 1].plot(cols, grism[rr, :], 'g.', label='actual', ms=3)
        axes[1, 1].set_title(f'Dispersion (center row), σ_total={sigma_total:.2f} pix')
        axes[1, 1].legend(fontsize=8)

        # residual histogram
        axes[1, 2].hist(residual.ravel(), bins=50, color='grey', edgecolor='black')
        axes[1, 2].axvline(
            max_resid, color='red', ls='--', label=f'max={max_resid:.2e}'
        )
        axes[1, 2].set_title('Residual histogram')
        axes[1, 2].legend(fontsize=8)

        ok = rel_max < 1e-6
        color = 'green' if ok else 'red'
        fig.suptitle(
            f'Gaussian⊛Gaussian — {"PASS" if ok else "FAIL"} '
            f'(rel_max={rel_max:.2e}, tol=1e-6)'
            f'\nDispersing a Gaussian source with a Gaussian line profile '
            f'yields a wider Gaussian (σ²_total = σ²_spatial + σ²_spectral)',
            color=color,
            fontsize=11,
        )
        fig.tight_layout()
        _save_diagnostic(fig, 'gaussian_convolution')

        # integer-pixel shifts -> bilinear exact -> 1e-6 tolerance
        assert (
            rel_max < 1e-6
        ), f"Gaussian convolution rel error {rel_max:.2e} exceeds 1e-6"

    # ------------------------------------------------------------------
    # 2. Static datacube factorization (vcirc=0, v0=0)
    # ------------------------------------------------------------------
    def test_static_datacube_factorization(self):
        """When vcirc=0 and v0=0, each cube slice = I(x,y) * G(lam_k)."""
        import matplotlib.pyplot as plt

        vel_model = CenteredVelocityModel()
        int_model = InclinedExponentialModel()
        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)

        z = 1.0
        vel_disp = 50.0
        flux = 100.0
        lam_obs = HALPHA.lambda_rest * (1 + z)

        # CubePars
        R_at_line = roman_grism_R(lam_obs)
        sigma_inst_kms = C_KMS / (2.355 * R_at_line)
        sigma_eff_kms = np.sqrt(vel_disp**2 + sigma_inst_kms**2)
        sigma_lambda = lam_obs * sigma_eff_kms / C_KMS

        dlam = 1.0
        half_width = 5 * sigma_lambda
        Nlam = int(2 * half_width / dlam) + 1
        Nlam = max(Nlam, 11)
        cp = CubePars.from_range(
            _ANALYTICAL_IMAGE_PARS,
            lam_obs - half_width,
            lam_obs + half_width,
            dlam,
        )

        # static params: vcirc=0, v0=0
        vel_pars = {
            'cosi': 0.5,
            'theta_int': 0.7,
            'g1': 0.0,
            'g2': 0.0,
            'v0': 0.0,
            'vcirc': 0.0,
            'vel_rscale': 0.5,
        }
        int_pars = {
            'cosi': 0.5,
            'theta_int': 0.7,
            'g1': 0.0,
            'g2': 0.0,
            'flux': flux,
            'int_rscale': 0.3,
            'int_h_over_r': 0.1,
            'int_x0': 0.0,
            'int_y0': 0.0,
        }
        theta_vel = vel_model.pars2theta(vel_pars)
        theta_int = int_model.pars2theta(int_pars)
        theta_spec = jnp.array(
            [z, vel_disp, flux, 0.0]
        )  # z, vel_disp, Ha_flux, Ha_cont

        cube = np.array(sm.build_cube(theta_spec, theta_vel, theta_int, cp))

        # independent intensity map
        I_map = np.array(
            int_model.render_unconvolved(theta_int, _ANALYTICAL_IMAGE_PARS)
        )

        # expected Gaussian per slice (lam_obs is uniform since v=0)
        lam_grid = np.array(cp.lambda_grid)

        # for each slice, ratio C[:,:,k] / I(x,y) should be constant = G_k
        # use pixels with significant flux to avoid 0/0
        flux_mask = I_map > 0.01 * I_map.max()

        max_ratio_err = 0.0
        n_slices_to_plot = min(5, len(lam_grid))
        slice_indices = np.linspace(0, len(lam_grid) - 1, n_slices_to_plot, dtype=int)

        fig, axes = plt.subplots(
            n_slices_to_plot, 3, figsize=(14, 3 * n_slices_to_plot)
        )
        if n_slices_to_plot == 1:
            axes = axes[None, :]

        for plot_idx, k in enumerate(slice_indices):
            ratio = cube[:, :, k] / np.where(flux_mask, I_map, 1.0)
            ratio_masked = ratio[flux_mask]
            G_k = np.median(ratio_masked)
            ratio_err = np.abs(ratio_masked / G_k - 1.0)
            max_err_k = ratio_err.max()
            max_ratio_err = max(max_ratio_err, max_err_k)

            # expected slice
            expected = I_map * G_k

            axes[plot_idx, 0].imshow(expected, origin='lower')
            axes[plot_idx, 0].set_title(f'Expected I*G_k (k={k})')
            axes[plot_idx, 1].imshow(cube[:, :, k], origin='lower')
            axes[plot_idx, 1].set_title(f'Actual C[:,:,{k}]')
            resid = np.abs(cube[:, :, k] - expected)
            im = axes[plot_idx, 2].imshow(resid, origin='lower', cmap='RdBu_r')
            axes[plot_idx, 2].set_title(f'|Residual| (max={resid.max():.2e})')
            plt.colorbar(im, ax=axes[plot_idx, 2], fraction=0.046)

        ok = max_ratio_err < 1e-4
        color = 'green' if ok else 'red'
        fig.suptitle(
            f'Static factorization — {"PASS" if ok else "FAIL"} '
            f'(max_ratio_err={max_ratio_err:.2e}, tol=1e-4)'
            f'\nWith no velocity field, each cube slice factorizes as '
            f'C(x,y,λ_k) = I(x,y) × G(λ_k)',
            color=color,
            fontsize=11,
        )
        fig.tight_layout()
        _save_diagnostic(fig, 'static_factorization')

        # spectral oversampling discretization: O(dlam^2 / (12*osf^2*sigma^2))
        assert (
            max_ratio_err < 1e-4
        ), f"Static factorization ratio error {max_ratio_err:.2e} exceeds 1e-4"

    # ------------------------------------------------------------------
    # 3. Point source grism — exact 1D Gaussian
    # ------------------------------------------------------------------
    def test_point_source_grism_angle0(self):
        """Delta source at grid center, dispersion along x: center row = 1D Gaussian."""
        self._run_point_source_test(angle=0.0, label='angle0')

    def test_point_source_grism_angle90(self):
        """Delta source at grid center, dispersion along y: center col = 1D Gaussian."""
        self._run_point_source_test(angle=np.pi / 2, label='angle90')

    def _run_point_source_test(self, angle, label):
        import matplotlib.pyplot as plt

        vel_model = CenteredVelocityModel()
        int_model = InclinedExponentialModel()
        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)
        kl = KLModel(vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=sm)

        z = 1.0
        vel_disp = 50.0
        flux = 100.0
        lam_obs = HALPHA.lambda_rest * (1 + z)

        # resolved source to get correct spectral shape, then delta-cube
        pars = {
            'cosi': 1.0,  # face-on to avoid velocity projection
            'theta_int': 0.0,
            'g1': 0.0,
            'g2': 0.0,
            'v0': 0.0,
            'vcirc': 0.0,
            'vel_rscale': 0.5,
            'flux': flux,
            'int_rscale': 0.3,  # resolved to avoid k-space aliasing
            'int_h_over_r': 0.1,
            'int_x0': 0.0,
            'int_y0': 0.0,
            'z': z,
            'vel_dispersion': vel_disp,
            'Ha_flux': flux,
            'Ha_cont': 0.0,
        }
        theta = kl.pars2theta(pars)

        # construct lambda_grid manually so dlam = dispersion exactly
        dlam = 1.0  # nm — also the dispersion (nm/pix)
        Nlam_half = 10
        lam_grid = lam_obs + np.arange(-Nlam_half, Nlam_half + 1) * dlam
        cp = CubePars(
            image_pars=_ANALYTICAL_IMAGE_PARS, lambda_grid=jnp.array(lam_grid)
        )

        gp = GrismPars(
            image_pars=_ANALYTICAL_IMAGE_PARS,
            dispersion=dlam,
            lambda_ref=lam_obs,
            dispersion_angle=angle,
        )

        # build resolved cube, extract center pixel spectrum, construct delta cube
        cube_full = np.array(kl.render_cube(theta, cp))
        cube = np.zeros_like(cube_full)
        cube[_ANALYTICAL_NROW // 2, _ANALYTICAL_NCOL // 2, :] = cube_full[
            _ANALYTICAL_NROW // 2, _ANALYTICAL_NCOL // 2, :
        ]
        grism = np.array(disperse_cube(jnp.array(cube), gp, jnp.array(lam_grid)))

        r0 = _ANALYTICAL_NROW // 2
        c0 = _ANALYTICAL_NCOL // 2

        # center pixel spectrum from actual cube — tests disperse_cube only
        center_spec = cube[r0, c0, :]
        actual_dlam = float(cp.delta_lambda)

        if np.abs(angle) < 0.01:
            # dispersion along cols
            actual_1d = grism[r0, :]
            expected_1d = np.zeros(_ANALYTICAL_NCOL)
            for k in range(len(lam_grid)):
                offset = int(round((lam_grid[k] - lam_obs) / dlam))
                c_shifted = c0 + offset
                if 0 <= c_shifted < _ANALYTICAL_NCOL:
                    expected_1d[c_shifted] += center_spec[k] * actual_dlam
            axis_label = 'col'
            axis_vals = np.arange(_ANALYTICAL_NCOL)
        else:
            # dispersion along rows
            actual_1d = grism[:, c0]
            expected_1d = np.zeros(_ANALYTICAL_NROW)
            for k in range(len(lam_grid)):
                offset = int(round((lam_grid[k] - lam_obs) / dlam))
                r_shifted = r0 + offset
                if 0 <= r_shifted < _ANALYTICAL_NROW:
                    expected_1d[r_shifted] += center_spec[k] * actual_dlam
            axis_label = 'row'
            axis_vals = np.arange(_ANALYTICAL_NROW)

        # compare where expected is significant
        sig_mask = expected_1d > 0.01 * expected_1d.max()
        residual = np.abs(actual_1d[sig_mask] - expected_1d[sig_mask])
        peak = expected_1d.max()
        rel_max = residual.max() / peak

        # diagnostic plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(axis_vals, expected_1d, 'k--', lw=2, label='expected')
        axes[0].plot(axis_vals, actual_1d, 'g.', ms=4, label='actual')
        axes[0].set_xlabel(axis_label)
        axes[0].set_title(f'Point source grism ({label})')
        axes[0].legend()

        axes[1].imshow(grism, origin='lower', aspect='auto')
        axes[1].set_title('Grism image')

        axes[2].plot(axis_vals[sig_mask], residual / peak, 'r.-')
        axes[2].set_xlabel(axis_label)
        axes[2].set_title(f'Relative residual (max={rel_max:.2e})')

        ok = rel_max < 1e-6
        color = 'green' if ok else 'red'
        fig.suptitle(
            f'Point source ({label}) — {"PASS" if ok else "FAIL"} '
            f'(rel_max={rel_max:.2e}, tol=1e-6)'
            f'\nDelta-function source dispersed along one axis '
            f'must reproduce the exact 1D spectral profile',
            color=color,
            fontsize=11,
        )
        fig.tight_layout()
        _save_diagnostic(fig, f'point_source_{label}')

        # integer-pixel offsets + actual cube spectrum -> bilinear exact -> 1e-6
        assert (
            rel_max < 1e-6
        ), f"Point source {label} rel error {rel_max:.2e} exceeds 1e-6"

    # ------------------------------------------------------------------
    # 4. Static separability — independent 1D convolution
    # ------------------------------------------------------------------
    def test_static_separability(self):
        """For vcirc=0, grism = 1D convolution of I(x,y) with spectral kernel."""
        import matplotlib.pyplot as plt

        vel_model = CenteredVelocityModel()
        int_model = InclinedExponentialModel()
        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)
        kl = KLModel(vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=sm)

        z = 1.0
        vel_disp = 50.0
        flux = 100.0
        lam_obs = HALPHA.lambda_rest * (1 + z)

        pars = {
            'cosi': 0.5,
            'theta_int': 0.7,
            'g1': 0.0,
            'g2': 0.0,
            'v0': 0.0,
            'vcirc': 0.0,
            'vel_rscale': 0.5,
            'flux': flux,
            'int_rscale': 0.3,
            'int_h_over_r': 0.1,
            'int_x0': 0.0,
            'int_y0': 0.0,
            'z': z,
            'vel_dispersion': vel_disp,
            'Ha_flux': flux,
            'Ha_cont': 0.0,
        }
        theta = kl.pars2theta(pars)

        # construct lambda_grid manually so dlam = dispersion exactly
        dlam = 1.0
        Nlam_half = 10
        lam_grid = lam_obs + np.arange(-Nlam_half, Nlam_half + 1) * dlam
        cp = CubePars(
            image_pars=_ANALYTICAL_IMAGE_PARS, lambda_grid=jnp.array(lam_grid)
        )
        gp = GrismPars(
            image_pars=_ANALYTICAL_IMAGE_PARS,
            dispersion=dlam,
            lambda_ref=lam_obs,
            dispersion_angle=0.0,
        )

        # build cube first, then disperse — tests disperse_cube + build_cube consistency
        cube = np.array(kl.render_cube(theta, cp))
        grism = np.array(disperse_cube(jnp.array(cube), gp, jnp.array(lam_grid)))

        # independent convolution using actual cube spectrum as kernel
        # for static case (vcirc=0), all pixels share same spectral profile
        # so cube[:,:,k] = I(x,y) * G_k; the 1D kernel is G_k * dlam
        # extract kernel from center pixel spectrum (highest SNR)
        r0 = _ANALYTICAL_NROW // 2
        c0 = _ANALYTICAL_NCOL // 2
        center_spec = cube[r0, c0, :]
        theta_int = kl.get_intensity_pars(theta)
        I_map = np.array(
            int_model.render_unconvolved(theta_int, _ANALYTICAL_IMAGE_PARS)
        )
        I_center = I_map[r0, c0]

        # kernel in pixel space: G_k = cube_center_spec / I_center
        kernel_1d = (center_spec / I_center) * dlam
        # kernel is centered (offsets symmetric around 0)
        # convolve each row of I_map (angle=0 -> dispersion along cols)
        expected_grism = convolve1d(I_map, kernel_1d, axis=1, mode='constant', origin=0)

        # compare
        residual = np.abs(grism - expected_grism)
        peak = expected_grism.max()
        rel_max = residual.max() / peak

        # diagnostic
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes[0, 0].imshow(expected_grism, origin='lower')
        axes[0, 0].set_title('Expected (scipy convolve)')
        axes[0, 1].imshow(grism, origin='lower')
        axes[0, 1].set_title('Actual grism')
        vmax = max(residual.max(), 1e-16)
        im = axes[0, 2].imshow(residual, origin='lower', cmap='hot', vmin=0, vmax=vmax)
        axes[0, 2].set_title(f'|Residual| (max={residual.max():.2e})')
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

        # 1D cross-sections
        rr = _ANALYTICAL_NROW // 2
        cc = _ANALYTICAL_NCOL // 2
        axes[1, 0].plot(expected_grism[rr, :], 'k--', label='expected')
        axes[1, 0].plot(grism[rr, :], 'g.', ms=3, label='actual')
        axes[1, 0].set_title('Center row')
        axes[1, 0].legend(fontsize=8)
        axes[1, 1].plot(expected_grism[:, cc], 'k--', label='expected')
        axes[1, 1].plot(grism[:, cc], 'g.', ms=3, label='actual')
        axes[1, 1].set_title('Center col')
        axes[1, 1].legend(fontsize=8)
        axes[1, 2].hist(residual.ravel(), bins=50, color='grey')
        axes[1, 2].set_title('Residual histogram')

        ok = rel_max < 1e-5
        color = 'green' if ok else 'red'
        fig.suptitle(
            f'Static separability — {"PASS" if ok else "FAIL"} '
            f'(rel_max={rel_max:.2e}, tol=1e-5)'
            f'\nWith no velocity, render_grism must equal independent '
            f'scipy 1D convolution of I(x,y) with spectral kernel',
            color=color,
            fontsize=11,
        )
        fig.tight_layout()
        _save_diagnostic(fig, 'static_separability')

        # integer-pixel shifts -> 1e-5 tolerance (tighter than 1e-3 general case)
        assert (
            rel_max < 1e-5
        ), f"Static separability rel error {rel_max:.2e} exceeds 1e-5"

    # ------------------------------------------------------------------
    # 5a. Face-on kills kinematic signal
    # ------------------------------------------------------------------
    def test_face_on_kills_signal(self):
        """cosi=1.0 -> sin(i)=0 -> no Doppler -> grism independent of vcirc."""
        vel_model = CenteredVelocityModel()
        int_model = InclinedExponentialModel()
        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)
        kl = KLModel(vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=sm)

        base_pars = {
            'cosi': 1.0,
            'theta_int': 0.7,
            'g1': 0.0,
            'g2': 0.0,
            'v0': 0.0,
            'vcirc': 200.0,
            'vel_rscale': 0.5,
            'flux': 100.0,
            'int_rscale': 0.3,
            'int_h_over_r': 0.1,
            'int_x0': 0.0,
            'int_y0': 0.0,
            'z': 1.0,
            'vel_dispersion': 50.0,
            'Ha_flux': 100.0,
            'Ha_cont': 0.0,
        }

        gp = build_grism_pars_for_line(
            HALPHA.lambda_rest,
            redshift=1.0,
            image_pars=_ANALYTICAL_IMAGE_PARS,
            dispersion=1.1,
        )
        cp = gp.to_cube_pars(z=1.0)

        theta_rot = kl.pars2theta(base_pars)
        grism_rot = np.array(kl.render_grism(theta_rot, gp, cube_pars=cp))

        norot_pars = {**base_pars, 'vcirc': 0.0}
        theta_norot = kl.pars2theta(norot_pars)
        grism_norot = np.array(kl.render_grism(theta_norot, gp, cube_pars=cp))

        max_diff = np.max(np.abs(grism_rot - grism_norot))
        assert (
            max_diff < 1e-12
        ), f"Face-on galaxy shows kinematic signal: max_diff={max_diff:.2e}"

    # ------------------------------------------------------------------
    # 5b. Velocity reversal = dispersion reversal
    # ------------------------------------------------------------------
    def test_velocity_reversal_dispersion_reversal(self):
        """grism(theta_int=pi, angle=a) == grism(theta_int=0, angle=a+pi)."""
        import matplotlib.pyplot as plt

        vel_model = CenteredVelocityModel()
        int_model = InclinedExponentialModel()
        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)
        kl = KLModel(vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=sm)

        base_pars = {
            'cosi': 0.5,
            'theta_int': 0.0,
            'g1': 0.0,
            'g2': 0.0,
            'v0': 0.0,
            'vcirc': 200.0,
            'vel_rscale': 0.5,
            'flux': 100.0,
            'int_rscale': 0.3,
            'int_h_over_r': 0.1,
            'int_x0': 0.0,
            'int_y0': 0.0,
            'z': 1.0,
            'vel_dispersion': 50.0,
            'Ha_flux': 100.0,
            'Ha_cont': 0.0,
        }

        disp_angle_a = 0.3  # arbitrary

        # case A: theta_int=pi, dispersion_angle=a
        pars_A = {**base_pars, 'theta_int': np.pi}
        theta_A = kl.pars2theta(pars_A)
        gp_A = GrismPars(
            image_pars=_ANALYTICAL_IMAGE_PARS,
            dispersion=1.1,
            lambda_ref=HALPHA.lambda_rest * 2.0,
            dispersion_angle=disp_angle_a,
        )
        cp = gp_A.to_cube_pars(z=1.0)
        grism_A = np.array(kl.render_grism(theta_A, gp_A, cube_pars=cp))

        # case B: theta_int=0, dispersion_angle=a+pi
        pars_B = {**base_pars, 'theta_int': 0.0}
        theta_B = kl.pars2theta(pars_B)
        gp_B = GrismPars(
            image_pars=_ANALYTICAL_IMAGE_PARS,
            dispersion=1.1,
            lambda_ref=HALPHA.lambda_rest * 2.0,
            dispersion_angle=disp_angle_a + np.pi,
        )
        grism_B = np.array(kl.render_grism(theta_B, gp_B, cube_pars=cp))

        # rotating both PA and dispersion angle by pi is a 180° rotation of the
        # output image, so compare grism_A with grism_B rotated 180°
        diff = np.abs(grism_A - grism_B[::-1, ::-1])
        max_diff = diff.max()
        peak = max(grism_A.max(), grism_B.max())
        rel_max = max_diff / peak

        # diagnostic
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].imshow(grism_A, origin='lower')
        axes[0].set_title('theta_int=π, angle=α')
        axes[1].imshow(grism_B[::-1, ::-1], origin='lower')
        axes[1].set_title('rot180(theta_int=0, angle=α+π)')
        vmax = max(max_diff, 1e-16)
        im = axes[2].imshow(diff, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[2].set_title(f'|Difference| (max={max_diff:.2e})')
        plt.colorbar(im, ax=axes[2], fraction=0.046)

        ok = rel_max < 1e-10
        color = 'green' if ok else 'red'
        fig.suptitle(
            f'Velocity reversal — {"PASS" if ok else "FAIL"} '
            f'(rel_max={rel_max:.2e}, tol=1e-10)'
            f'\nRotating PA and dispersion angle by π is a 180° coordinate '
            f'rotation: grism(θ+π, α) = rot180[grism(θ, α+π)]',
            color=color,
            fontsize=11,
        )
        fig.tight_layout()
        _save_diagnostic(fig, 'velocity_reversal')

        assert (
            rel_max < 1e-10
        ), f"Velocity reversal identity failed: rel_max={rel_max:.2e}"

    # ------------------------------------------------------------------
    # 5c. Static spatial symmetry (vcirc=0, centered source)
    # ------------------------------------------------------------------
    def test_static_spatial_symmetry(self):
        """For vcirc=0 centered source, grism is 180-deg symmetric."""
        import matplotlib.pyplot as plt

        vel_model = CenteredVelocityModel()
        int_model = InclinedExponentialModel()
        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)
        kl = KLModel(vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=sm)

        pars = {
            'cosi': 0.5,
            'theta_int': 0.7,
            'g1': 0.0,
            'g2': 0.0,
            'v0': 0.0,
            'vcirc': 0.0,
            'vel_rscale': 0.5,
            'flux': 100.0,
            'int_rscale': 0.3,
            'int_h_over_r': 0.1,
            'int_x0': 0.0,
            'int_y0': 0.0,
            'z': 1.0,
            'vel_dispersion': 50.0,
            'Ha_flux': 100.0,
            'Ha_cont': 0.0,
        }
        theta = kl.pars2theta(pars)

        gp = GrismPars(
            image_pars=_ANALYTICAL_IMAGE_PARS,
            dispersion=1.1,
            lambda_ref=HALPHA.lambda_rest * 2.0,
            dispersion_angle=0.3,
        )
        cp = gp.to_cube_pars(z=1.0)
        grism = np.array(kl.render_grism(theta, gp, cube_pars=cp))

        # 180-deg rotation = flip both axes
        grism_rot = grism[::-1, ::-1]
        diff = np.abs(grism - grism_rot)
        max_diff = diff.max()

        # diagnostic
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        im0 = axes[0].imshow(grism, origin='lower')
        axes[0].set_title('Grism (vcirc=0)')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(grism_rot, origin='lower')
        axes[1].set_title('Rot 180°')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        vmax = max(max_diff, 1e-16)
        im2 = axes[2].imshow(diff, origin='lower', cmap='hot', vmin=0, vmax=vmax)
        axes[2].set_title(f'|Difference| (max={max_diff:.2e})')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        ok = max_diff < 1e-12
        color = 'green' if ok else 'red'
        fig.suptitle(
            f'Spatial symmetry — {"PASS" if ok else "FAIL"} '
            f'(max_diff={max_diff:.2e}, tol=1e-12)'
            f'\nA static centered source must produce a grism image '
            f'with exact 180° rotational symmetry',
            color=color,
            fontsize=11,
        )
        fig.tight_layout()
        _save_diagnostic(fig, 'spatial_symmetry')

        assert (
            max_diff < 1e-12
        ), f"Static spatial symmetry violated: max_diff={max_diff:.2e}"

    # ------------------------------------------------------------------
    # 6. Flux conservation
    # ------------------------------------------------------------------
    def test_flux_conservation(self):
        """Total grism flux = total intensity map flux (for compact source, cont=0)."""
        import matplotlib.pyplot as plt

        vel_model = CenteredVelocityModel()
        int_model = InclinedExponentialModel()
        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)
        kl = KLModel(vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=sm)

        # compact source (small rscale) on large grid, cont=0
        pars = {
            'cosi': 0.5,
            'theta_int': 0.7,
            'g1': 0.0,
            'g2': 0.0,
            'v0': 0.0,
            'vcirc': 150.0,
            'vel_rscale': 0.5,
            'flux': 100.0,
            'int_rscale': 0.15,  # compact
            'int_h_over_r': 0.1,
            'int_x0': 0.0,
            'int_y0': 0.0,
            'z': 1.0,
            'vel_dispersion': 50.0,
            'Ha_flux': 100.0,
            'Ha_cont': 0.0,
        }
        theta = kl.pars2theta(pars)

        gp = GrismPars(
            image_pars=_ANALYTICAL_IMAGE_PARS,
            dispersion=1.1,
            lambda_ref=HALPHA.lambda_rest * 2.0,
            dispersion_angle=0.0,
        )
        cp = gp.to_cube_pars(z=1.0)
        grism = np.array(kl.render_grism(theta, gp, cube_pars=cp))

        # independent intensity map
        theta_int = kl.get_intensity_pars(theta)
        I_map = np.array(
            int_model.render_unconvolved(theta_int, _ANALYTICAL_IMAGE_PARS)
        )

        total_grism = grism.sum()
        total_imap = I_map.sum()
        rel_err = abs(total_grism - total_imap) / total_imap

        # also per-row check (for angle=0, each row should conserve)
        row_grism = grism.sum(axis=1)
        row_imap = I_map.sum(axis=1)
        # only check interior rows (edges lose shifted flux)
        margin = 10
        interior = slice(margin, _ANALYTICAL_NROW - margin)
        row_ratio = row_grism[interior] / np.where(
            row_imap[interior] > 0.01 * row_imap.max(),
            row_imap[interior],
            1.0,
        )
        sig_rows = row_imap[interior] > 0.01 * row_imap.max()
        max_row_err = np.max(np.abs(row_ratio[sig_rows] - 1.0))

        # diagnostic
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        rows = np.arange(_ANALYTICAL_NROW)
        ax.bar(
            rows[interior][sig_rows], row_ratio[sig_rows], color='steelblue', alpha=0.7
        )
        ax.axhline(1.0, color='black', ls='--')
        ax.axhspan(0.995, 1.005, alpha=0.15, color='grey', label='0.5% band')
        ax.set_xlabel('Row')
        ax.set_ylabel('Grism row sum / I row sum')
        ax.set_title(f'Per-row flux ratio (total rel err={rel_err:.4e})')
        ax.legend()

        ok_total = rel_err < 0.005
        ok_row = max_row_err < 0.005
        color = 'green' if (ok_total and ok_row) else 'red'
        fig.suptitle(
            f'Flux conservation — '
            f'{"PASS" if ok_total and ok_row else "FAIL"} '
            f'(total={rel_err:.2e}, row_max={max_row_err:.2e}, tol=0.5%)'
            f'\nTotal dispersed flux must equal total intensity map flux; '
            f'per-row sums must also be conserved',
            color=color,
            fontsize=11,
        )
        fig.tight_layout()
        _save_diagnostic(fig, 'flux_conservation')

        assert rel_err < 0.005, f"Total flux conservation failed: rel_err={rel_err:.4e}"
        assert (
            max_row_err < 0.005
        ), f"Per-row flux conservation failed: max_row_err={max_row_err:.4e}"

    # ------------------------------------------------------------------
    # 7. First spectral moment = Doppler-shifted wavelength
    # ------------------------------------------------------------------
    def test_first_spectral_moment(self):
        """Flux-weighted mean wavelength = Doppler-shifted line center per pixel."""
        import matplotlib.pyplot as plt

        vel_model = CenteredVelocityModel()
        int_model = InclinedExponentialModel()
        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)

        z = 1.0
        vel_disp = 50.0
        flux = 100.0
        lam_rest = HALPHA.lambda_rest
        lam_obs_center = lam_rest * (1 + z)

        R_at_line = roman_grism_R(lam_obs_center)
        sigma_inst_kms = C_KMS / (2.355 * R_at_line)
        sigma_eff_kms = np.sqrt(vel_disp**2 + sigma_inst_kms**2)
        sigma_lambda = lam_obs_center * sigma_eff_kms / C_KMS

        dlam = 1.0
        half_width = 5 * sigma_lambda
        cp = CubePars.from_range(
            _ANALYTICAL_IMAGE_PARS,
            lam_obs_center - half_width,
            lam_obs_center + half_width,
            dlam,
        )

        vel_pars = {
            'cosi': 0.5,
            'theta_int': 0.7,
            'g1': 0.0,
            'g2': 0.0,
            'v0': 10.0,
            'vcirc': 200.0,
            'vel_rscale': 0.5,
        }
        int_pars = {
            'cosi': 0.5,
            'theta_int': 0.7,
            'g1': 0.0,
            'g2': 0.0,
            'flux': flux,
            'int_rscale': 0.3,
            'int_h_over_r': 0.1,
            'int_x0': 0.0,
            'int_y0': 0.0,
        }
        theta_vel = vel_model.pars2theta(vel_pars)
        theta_int = int_model.pars2theta(int_pars)
        theta_spec = jnp.array([z, vel_disp, flux, 0.0])

        cube = np.array(sm.build_cube(theta_spec, theta_vel, theta_int, cp))
        lam_grid = np.array(cp.lambda_grid)

        # flux-weighted mean wavelength per pixel
        total_flux = cube.sum(axis=2)
        weighted_lam = (cube * lam_grid[None, None, :]).sum(axis=2)
        # mask low-flux pixels
        flux_mask = total_flux > 0.01 * total_flux.max()
        measured_lam = np.where(flux_mask, weighted_lam / total_flux, 0.0)

        # expected: lam_obs(x,y) = lam_rest * (1+z) * (1 + V_rot/c)
        X, Y = build_map_grid_from_image_pars(_ANALYTICAL_IMAGE_PARS)
        v_map = np.array(vel_model(theta_vel, 'obs', X, Y))
        v0 = vel_pars['v0']
        v_rotation = v_map - v0
        expected_lam = lam_rest * (1 + z) * (1.0 + v_rotation / C_KMS)

        # compare where both have signal
        diff = np.abs(measured_lam - expected_lam) * flux_mask
        max_diff_nm = diff[flux_mask].max()

        # diagnostic
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        im0 = axes[0, 0].imshow(expected_lam, origin='lower')
        axes[0, 0].set_title('Expected λ_obs (nm)')
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

        im1 = axes[0, 1].imshow(
            np.where(flux_mask, measured_lam, np.nan), origin='lower'
        )
        axes[0, 1].set_title('Measured first moment (nm)')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

        im2 = axes[1, 0].imshow(
            np.where(flux_mask, diff, np.nan), origin='lower', cmap='hot'
        )
        axes[1, 0].set_title(f'|Difference| (max={max_diff_nm:.3f} nm)')
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

        exp_flat = expected_lam[flux_mask]
        meas_flat = measured_lam[flux_mask]
        axes[1, 1].scatter(exp_flat, meas_flat, s=1, alpha=0.3)
        lam_range = [exp_flat.min(), exp_flat.max()]
        axes[1, 1].plot(lam_range, lam_range, 'r--')
        axes[1, 1].set_xlabel('Expected λ (nm)')
        axes[1, 1].set_ylabel('Measured λ (nm)')
        axes[1, 1].set_title('Scatter')

        ok = max_diff_nm < 0.1
        color = 'green' if ok else 'red'
        fig.suptitle(
            f'Spectral first moment — {"PASS" if ok else "FAIL"} '
            f'(max_diff={max_diff_nm:.3f} nm, tol=0.1 nm)'
            f'\nFlux-weighted mean wavelength per pixel must equal '
            f'the Doppler-shifted line center from the velocity field',
            color=color,
            fontsize=11,
        )
        fig.tight_layout()
        _save_diagnostic(fig, 'spectral_first_moment')

        # discretization: O(dlam^2/sigma^2) ~ 0.1 nm
        assert (
            max_diff_nm < 0.1
        ), f"First moment error {max_diff_nm:.3f} nm exceeds 0.1 nm"

    # ------------------------------------------------------------------
    # 8. Dispersion centroid linearity
    # ------------------------------------------------------------------
    def test_dispersion_centroid_linearity(self):
        """Point source centroid follows linear relation with lambda."""
        import matplotlib.pyplot as plt

        vel_model = CenteredVelocityModel()
        int_model = InclinedExponentialModel()
        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)
        kl = KLModel(vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=sm)

        base_lam_rest = HALPHA.lambda_rest
        # test several redshifts that shift lam_obs
        delta_lams = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])  # nm offsets from nominal
        angle = 0.0
        dispersion = 1.1
        lam_ref = base_lam_rest * 2.0  # z=1 nominal

        measured_centroids = []
        expected_centroids = []

        for dlam_offset in delta_lams:
            # shift the line's observed wavelength by adjusting z
            lam_obs_target = lam_ref + dlam_offset
            z_eff = lam_obs_target / base_lam_rest - 1.0

            pars = {
                'cosi': 1.0,  # face-on, no velocity
                'theta_int': 0.0,
                'g1': 0.0,
                'g2': 0.0,
                'v0': 0.0,
                'vcirc': 0.0,
                'vel_rscale': 0.5,
                'flux': 100.0,
                'int_rscale': 0.3,  # resolved to avoid k-space aliasing
                'int_h_over_r': 0.1,
                'int_x0': 0.0,
                'int_y0': 0.0,
                'z': z_eff,
                'vel_dispersion': 50.0,
                'Ha_flux': 100.0,
                'Ha_cont': 0.0,
            }
            theta = kl.pars2theta(pars)

            gp = GrismPars(
                image_pars=_ANALYTICAL_IMAGE_PARS,
                dispersion=dispersion,
                lambda_ref=lam_ref,
                dispersion_angle=angle,
            )
            cp = gp.to_cube_pars(z=z_eff)

            # build resolved cube, extract center pixel, construct delta cube
            r0 = _ANALYTICAL_NROW // 2
            c0 = _ANALYTICAL_NCOL // 2
            cube_full = np.array(kl.render_cube(theta, cp))
            cube_delta = np.zeros_like(cube_full)
            cube_delta[r0, c0, :] = cube_full[r0, c0, :]
            grism = np.array(
                disperse_cube(
                    jnp.array(cube_delta), gp, jnp.array(np.array(cp.lambda_grid))
                )
            )

            # measured centroid (flux-weighted)
            cols = np.arange(_ANALYTICAL_NCOL, dtype=np.float64)
            total = grism.sum()
            centroid_c = (grism.sum(axis=0) * cols).sum() / total
            measured_centroids.append(centroid_c)

            # expected centroid
            c0 = (_ANALYTICAL_NCOL - 1) / 2.0
            expected_c = c0 + (lam_obs_target - lam_ref) / dispersion * np.cos(angle)
            expected_centroids.append(expected_c)

        measured_centroids = np.array(measured_centroids)
        expected_centroids = np.array(expected_centroids)
        errors = np.abs(measured_centroids - expected_centroids)
        max_err = errors.max()

        # diagnostic
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.plot(delta_lams, expected_centroids, 'k--o', label='expected')
        ax.plot(delta_lams, measured_centroids, 'gs', label='measured', ms=8)
        ax.set_xlabel('Δλ from ref (nm)')
        ax.set_ylabel('Centroid column (pix)')
        ax.legend()

        ok = max_err < 0.1
        color = 'green' if ok else 'red'
        fig.suptitle(
            f'Centroid linearity — {"PASS" if ok else "FAIL"} '
            f'(max_err={max_err:.3f} pix, tol=0.1)'
            f'\nPoint source centroid must shift linearly with wavelength: '
            f'Δpix = Δλ / dispersion',
            color=color,
            fontsize=11,
        )
        fig.tight_layout()
        _save_diagnostic(fig, 'centroid_linearity')

        assert max_err < 0.1, f"Centroid linearity error {max_err:.3f} pix exceeds 0.1"

    # ------------------------------------------------------------------
    # 10. PA–dispersion angle coupling
    # ------------------------------------------------------------------
    def test_pa_dispersion_coupling(self):
        """Antisymmetric grism signal scales approximately as
        |cos(dispersion_angle - theta_int)|. Zero when perpendicular.

        Column-flip decomposition: flip diff image left-right and subtract to
        extract the antisymmetric (velocity dipole) component along the
        dispersion axis. Exact on the pixel grid — no interpolation artifacts.
        Scaling is ~cos^1.5 (profile-dependent); cos is used as a reference
        for correlation with a loose threshold.
        """
        import matplotlib.pyplot as plt

        vel_model = CenteredVelocityModel()
        int_model = InclinedExponentialModel()
        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)
        kl = KLModel(vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=sm)

        pars = {
            'cosi': 0.5,
            'theta_int': 0.0,
            'g1': 0.0,
            'g2': 0.0,
            'v0': 0.0,
            'vcirc': 200.0,
            'vel_rscale': 0.5,
            'flux': 100.0,
            'int_rscale': 0.3,
            'int_h_over_r': 0.1,
            'int_x0': 0.0,
            'int_y0': 0.0,
            'z': 1.0,
            'vel_dispersion': 50.0,
            'Ha_flux': 100.0,
            'Ha_cont': 0.0,
        }
        theta_rot = kl.pars2theta(pars)

        pars_norot = {**pars, 'vcirc': 0.0}
        theta_norot = kl.pars2theta(pars_norot)

        Ncol = _ANALYTICAL_NCOL

        angles = np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])
        amplitudes = []

        for angle in angles:
            gp = GrismPars(
                image_pars=_ANALYTICAL_IMAGE_PARS,
                dispersion=1.1,
                lambda_ref=HALPHA.lambda_rest * 2.0,
                dispersion_angle=angle,
            )
            cp = gp.to_cube_pars(z=1.0)

            grism_rot = np.array(kl.render_grism(theta_rot, gp, cube_pars=cp))
            grism_norot = np.array(kl.render_grism(theta_norot, gp, cube_pars=cp))

            diff = grism_rot - grism_norot

            # column-flip antisymmetric decomposition (exact on pixel grid)
            asym = (diff - diff[:, ::-1]) / 2.0
            cc = Ncol // 2
            A = np.abs(np.sum(asym[:, :cc])) + np.abs(np.sum(asym[:, cc + 1 :]))
            amplitudes.append(A)

        amplitudes = np.array(amplitudes)

        # expected: |cos(angle - theta_int)| with theta_int=0
        expected_scaling = np.abs(np.cos(angles - pars['theta_int']))
        # normalize both to [0, 1] for correlation
        A_norm = amplitudes / amplitudes.max()
        E_norm = expected_scaling / expected_scaling.max()
        corr = np.corrcoef(A_norm, E_norm)[0, 1]

        # perpendicular angle (π/2 for theta_int=0) should have near-zero signal
        perp_idx = np.argmin(expected_scaling)
        perp_ratio = amplitudes[perp_idx] / amplitudes.max()

        # diagnostic
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        cos2_scaling = np.cos(angles - pars['theta_int']) ** 2
        C2_norm = cos2_scaling / cos2_scaling.max()

        angles_deg = np.degrees(angles)
        axes[0].plot(angles_deg, A_norm, 'bo-', ms=8, label='measured A/A_max')
        axes[0].plot(angles_deg, E_norm, 'r--s', ms=6, label='|cos(α − θ_int)|')
        axes[0].plot(angles_deg, C2_norm, 'g:^', ms=5, label='cos²(α − θ_int)')
        axes[0].set_xlabel('Dispersion angle (deg)')
        axes[0].set_ylabel('Normalized amplitude')
        axes[0].legend()
        axes[0].set_title(f'corr={corr:.4f}, perp_ratio={perp_ratio:.4f}')

        axes[1].bar(angles_deg, amplitudes, width=8, color='steelblue')
        axes[1].set_xlabel('Dispersion angle (deg)')
        axes[1].set_ylabel('Absolute amplitude A')
        axes[1].set_title('Raw kinematic amplitude')

        # monotonic decrease from aligned to perpendicular
        monotonic = all(A_norm[i] > A_norm[i + 1] for i in range(len(A_norm) - 1))

        ok = corr > 0.95 and perp_ratio < 0.01 and monotonic
        color = 'green' if ok else 'red'
        fig.suptitle(
            f'PA–dispersion coupling — {"PASS" if ok else "FAIL"} '
            f'(corr={corr:.4f}, perp_ratio={perp_ratio:.4f})'
            f'\nAntisymmetric signal ~cos(dispersion_angle − theta_int)',
            color=color,
            fontsize=11,
        )
        fig.tight_layout()
        _save_diagnostic(fig, 'pa_dispersion_coupling')

        assert monotonic, f"A_norm not monotonically decreasing: {A_norm}"
        assert perp_ratio < 0.01, f"perpendicular ratio {perp_ratio:.4f} >= 0.01"
        assert corr > 0.95, f"cos-scaling correlation {corr:.4f} < 0.95"

    # ------------------------------------------------------------------
    # 11. Kinematic amplitude scales with vcirc * sin(i)
    # ------------------------------------------------------------------
    def test_kinematic_amplitude_scaling(self):
        """In the small-shift regime (vsini << sigma_eff), grism antisymmetric
        amplitude is linear in vcirc * sin(i) with near-zero intercept.

        Uses low vsini values (max 130 km/s vs sigma_eff ~ 280 km/s) to stay
        in the linear regime where Doppler shift << line width.
        """
        import matplotlib.pyplot as plt

        vel_model = CenteredVelocityModel()
        int_model = InclinedExponentialModel()
        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)
        kl = KLModel(vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=sm)

        base_pars = {
            'theta_int': 0.0,
            'g1': 0.0,
            'g2': 0.0,
            'v0': 0.0,
            'vel_rscale': 0.5,
            'flux': 100.0,
            'int_rscale': 0.3,
            'int_h_over_r': 0.1,
            'int_x0': 0.0,
            'int_y0': 0.0,
            'z': 1.0,
            'vel_dispersion': 50.0,
            'Ha_flux': 100.0,
            'Ha_cont': 0.0,
        }

        gp = GrismPars(
            image_pars=_ANALYTICAL_IMAGE_PARS,
            dispersion=1.1,
            lambda_ref=HALPHA.lambda_rest * 2.0,
            dispersion_angle=0.0,
        )
        cp = gp.to_cube_pars(z=1.0)

        # (vcirc, cosi) combos — max vsini=130, shift/sigma ~ 0.46 (linear)
        combos = [
            (50.0, 0.5),  # vsini = 43.3
            (100.0, 0.5),  # vsini = 86.6
            (100.0, 0.866),  # vsini = 50.0
            (150.0, 0.5),  # vsini = 129.9
        ]

        vsini_vals = []
        amplitudes = []

        for vcirc, cosi in combos:
            sini = np.sqrt(1 - cosi**2)
            vsini_vals.append(vcirc * sini)

            # per-combo static reference matching cosi
            norot_pars = {**base_pars, 'vcirc': 0.0, 'cosi': cosi}
            theta_norot = kl.pars2theta(norot_pars)
            grism_norot = np.array(kl.render_grism(theta_norot, gp, cube_pars=cp))

            p = {**base_pars, 'vcirc': vcirc, 'cosi': cosi}
            theta = kl.pars2theta(p)
            grism = np.array(kl.render_grism(theta, gp, cube_pars=cp))

            diff = grism - grism_norot
            Ncol = _ANALYTICAL_NCOL
            diff_flip = diff[:, ::-1]
            asym = (diff - diff_flip) / 2.0
            cc = Ncol // 2
            A = np.abs(np.sum(asym[:, :cc])) + np.abs(np.sum(asym[:, cc + 1 :]))
            amplitudes.append(A)

        vsini_vals = np.array(vsini_vals)
        amplitudes = np.array(amplitudes)

        # linear fit: A = m * vsini + b
        m, b = np.polyfit(vsini_vals, amplitudes, 1)
        A_max = amplitudes.max()
        A_fit = m * vsini_vals + b
        ss_res = np.sum((amplitudes - A_fit) ** 2)
        ss_tot = np.sum((amplitudes - amplitudes.mean()) ** 2)
        R2 = 1 - ss_res / ss_tot

        # diagnostic
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.plot(vsini_vals, amplitudes, 'bo', ms=8, label='measured')
        vsini_line = np.linspace(0, vsini_vals.max() * 1.1, 50)
        ax.plot(
            vsini_line, m * vsini_line + b, 'k--', label=f'fit: m={m:.2f}, b={b:.2f}'
        )
        ax.set_xlabel('v_circ × sin(i)  [km/s]')
        ax.set_ylabel('Antisymmetric amplitude A')
        ax.legend()

        ok = R2 > 0.99 and abs(b) / A_max < 0.05
        color = 'green' if ok else 'red'
        fig.suptitle(
            f'Kinematic amplitude — {"PASS" if ok else "FAIL"} '
            f'(R²={R2:.4f}, |b|/A_max={abs(b)/A_max:.4f})'
            f'\nGrism antisymmetric signal must be linear in v_circ × sin(i)',
            color=color,
            fontsize=11,
        )
        fig.tight_layout()
        _save_diagnostic(fig, 'kinematic_amplitude_scaling')

        assert R2 > 0.99, f"R² = {R2:.4f} < 0.99"
        assert abs(b) / A_max < 0.05, f"|b|/A_max = {abs(b)/A_max:.4f} >= 0.05"

    # ------------------------------------------------------------------
    # 12. Velocity dispersion broadening
    # ------------------------------------------------------------------
    def test_dispersion_broadening(self):
        """Delta-cube FWHM matches spectral prediction
        FWHM = 2.355 * lambda_obs * sigma_eff / (c * dispersion),
        isolating spectral broadening from spatial extent.

        Sub-pixel FWHM via linear interpolation at half-max crossings
        avoids discretization artifacts for narrow profiles.
        """
        import matplotlib.pyplot as plt

        vel_model = CenteredVelocityModel()
        int_model = InclinedExponentialModel()
        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)
        kl = KLModel(vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=sm)

        z = 1.0
        lam_obs = HALPHA.lambda_rest * (1 + z)
        dispersion = 1.0  # nm/pix

        base_pars = {
            'cosi': 1.0,  # face-on
            'theta_int': 0.0,
            'g1': 0.0,
            'g2': 0.0,
            'v0': 0.0,
            'vcirc': 0.0,  # no rotation
            'vel_rscale': 0.5,
            'flux': 100.0,
            'int_rscale': 0.3,
            'int_h_over_r': 0.1,
            'int_x0': 0.0,
            'int_y0': 0.0,
            'z': z,
            'Ha_flux': 100.0,
            'Ha_cont': 0.0,
        }

        # use integer dlam = dispersion so bilinear is exact
        dlam = dispersion
        Nlam_half = 15
        lam_grid = lam_obs + np.arange(-Nlam_half, Nlam_half + 1) * dlam
        cp = CubePars(
            image_pars=_ANALYTICAL_IMAGE_PARS, lambda_grid=jnp.array(lam_grid)
        )
        gp = GrismPars(
            image_pars=_ANALYTICAL_IMAGE_PARS,
            dispersion=dispersion,
            lambda_ref=lam_obs,
            dispersion_angle=0.0,
        )

        r0 = _ANALYTICAL_NROW // 2
        c0 = _ANALYTICAL_NCOL // 2

        vel_disps = [30.0, 50.0, 100.0, 200.0]
        measured_fwhm = []
        expected_fwhm = []
        profiles = []

        for sigma_v in vel_disps:
            p = {**base_pars, 'vel_dispersion': sigma_v}
            theta = kl.pars2theta(p)

            # delta-cube: build resolved cube, extract center pixel spectrum
            cube_full = np.array(kl.render_cube(theta, cp))
            cube_delta = np.zeros_like(cube_full)
            cube_delta[r0, c0, :] = cube_full[r0, c0, :]

            grism = np.array(
                disperse_cube(jnp.array(cube_delta), gp, jnp.array(lam_grid))
            )

            # center row profile (purely spectral, no spatial convolution)
            profile = grism[r0, :]
            profiles.append(profile / profile.max())

            # sub-pixel FWHM via linear interpolation at half-max crossings
            half_max = profile.max() / 2.0
            above = profile >= half_max
            indices = np.where(above)[0]
            if len(indices) < 2:
                measured_fwhm.append(1.0)
                continue
            i_left, i_right = indices[0], indices[-1]

            # interpolate left crossing
            if i_left > 0:
                x_left = (i_left - 1) + (half_max - profile[i_left - 1]) / (
                    profile[i_left] - profile[i_left - 1]
                )
            else:
                x_left = float(i_left)

            # interpolate right crossing
            if i_right < len(profile) - 1:
                x_right = i_right + (half_max - profile[i_right]) / (
                    profile[i_right + 1] - profile[i_right]
                )
            else:
                x_right = float(i_right)

            fwhm_meas = x_right - x_left
            measured_fwhm.append(fwhm_meas)

            # expected FWHM (spectral only)
            R_at_line = roman_grism_R(lam_obs)
            sigma_inst = C_KMS / (2.355 * R_at_line)
            sigma_eff = np.sqrt(sigma_v**2 + sigma_inst**2)
            fwhm_expected = 2.355 * lam_obs * sigma_eff / (C_KMS * dispersion)
            expected_fwhm.append(fwhm_expected)

        measured_fwhm = np.array(measured_fwhm)
        expected_fwhm = np.array(expected_fwhm)
        errors = np.abs(measured_fwhm - expected_fwhm)
        max_err = errors.max()

        # diagnostic
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        cols = np.arange(_ANALYTICAL_NCOL)
        for i, sigma_v in enumerate(vel_disps):
            axes[0].plot(cols, profiles[i], label=f'σ_v={sigma_v} km/s')
        axes[0].set_xlabel('Column (pix)')
        axes[0].set_ylabel('Normalized profile')
        axes[0].legend(fontsize=8)
        axes[0].set_title('Center row profiles (delta-cube)')

        axes[1].plot(expected_fwhm, measured_fwhm, 'bo', ms=8)
        fwhm_range = [min(expected_fwhm) * 0.9, max(expected_fwhm) * 1.1]
        axes[1].plot(fwhm_range, fwhm_range, 'k--', alpha=0.5)
        axes[1].set_xlabel('Expected FWHM (pix)')
        axes[1].set_ylabel('Measured FWHM (pix)')
        axes[1].set_title(f'max |err| = {max_err:.2f} pix')

        ok = max_err < 0.5
        color = 'green' if ok else 'red'
        fig.suptitle(
            f'Dispersion broadening — {"PASS" if ok else "FAIL"} '
            f'(max_err={max_err:.2f} pix, tol=0.5)'
            f'\nDelta-cube isolates spectral broadening; '
            f'FWHM_pix = 2.355 × λ_obs × σ_eff / (c × dispersion)',
            color=color,
            fontsize=11,
        )
        fig.tight_layout()
        _save_diagnostic(fig, 'dispersion_broadening')

        assert max_err < 0.5, f"FWHM error {max_err:.2f} pix exceeds 0.5 pix tolerance"

    # ------------------------------------------------------------------
    # 13. Multi-line wavelength separation
    # ------------------------------------------------------------------
    def test_multiline_separation(self):
        """Hα and [NII] 6583 peaks at correct pixel separation."""
        import matplotlib.pyplot as plt

        vel_model = CenteredVelocityModel()
        int_model = InclinedExponentialModel()

        # two lines: Ha + NII_6583 (no NII_6548 to avoid overlap)
        ha_line = EmissionLine(line_spec=HALPHA, own_params=frozenset({'flux'}))
        nii_line = EmissionLine(line_spec=NII_6583, own_params=frozenset({'flux'}))
        config = SpectralConfig(lines=(ha_line, nii_line), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)
        kl = KLModel(vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=sm)

        z = 1.0
        dispersion = 1.0  # nm/pix for integer offsets
        ha_flux = 100.0
        nii_flux = 50.0

        pars = {
            'cosi': 1.0,  # face-on
            'theta_int': 0.0,
            'g1': 0.0,
            'g2': 0.0,
            'v0': 0.0,
            'vcirc': 0.0,
            'vel_rscale': 0.5,
            'flux': 100.0,
            'int_rscale': 0.3,
            'int_h_over_r': 0.1,
            'int_x0': 0.0,
            'int_y0': 0.0,
            'z': z,
            'vel_dispersion': 30.0,  # narrow lines for clean peaks
            'Ha_flux': ha_flux,
            'Ha_cont': 0.0,
            'NII_6583_flux': nii_flux,
            'NII_6583_cont': 0.0,
        }
        theta = kl.pars2theta(pars)

        # build lambda grid covering both lines
        lam_ha = HALPHA.lambda_rest * (1 + z)
        lam_nii = NII_6583.lambda_rest * (1 + z)
        lam_center = (lam_ha + lam_nii) / 2.0

        # wide enough to capture both lines with margin
        R_at_line = roman_grism_R(lam_center)
        sigma_inst = C_KMS / (2.355 * R_at_line)
        sigma_eff = np.sqrt(30.0**2 + sigma_inst**2)
        sigma_lam = lam_center * sigma_eff / C_KMS
        half_width = max(5 * sigma_lam, (lam_nii - lam_ha) + 5 * sigma_lam)
        cp = CubePars.from_range(
            _ANALYTICAL_IMAGE_PARS,
            lam_center - half_width,
            lam_center + half_width,
            dispersion,
        )

        gp = GrismPars(
            image_pars=_ANALYTICAL_IMAGE_PARS,
            dispersion=dispersion,
            lambda_ref=lam_center,
            dispersion_angle=0.0,
        )

        # delta-cube approach: extract center pixel
        r0 = _ANALYTICAL_NROW // 2
        c0 = _ANALYTICAL_NCOL // 2
        cube_full = np.array(kl.render_cube(theta, cp))
        cube_delta = np.zeros_like(cube_full)
        cube_delta[r0, c0, :] = cube_full[r0, c0, :]
        grism = np.array(
            disperse_cube(
                jnp.array(cube_delta), gp, jnp.array(np.array(cp.lambda_grid))
            )
        )

        # center row profile
        profile = grism[r0, :]

        # find peaks (local maxima above 10% of max)
        from scipy.signal import find_peaks

        threshold = 0.1 * profile.max()
        peaks, props = find_peaks(profile, height=threshold)

        assert len(peaks) >= 2, f"Expected >= 2 peaks, found {len(peaks)}"

        # sort by height descending, take top 2
        peak_heights = profile[peaks]
        top2 = np.argsort(peak_heights)[::-1][:2]
        p1, p2 = sorted(peaks[top2])
        sep_measured = p2 - p1

        # expected separation
        sep_expected = (
            (NII_6583.lambda_rest - HALPHA.lambda_rest) * (1 + z) / dispersion
        )

        sep_err = abs(sep_measured - sep_expected)

        # flux ratio
        flux_ratio = profile[p1] / profile[p2]
        # Ha is bluer -> p1 should be Ha (higher flux)
        expected_ratio = ha_flux / nii_flux
        ratio_err = abs(flux_ratio - expected_ratio) / expected_ratio

        # diagnostic
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        cols = np.arange(_ANALYTICAL_NCOL)
        ax.plot(cols, profile, 'b-', lw=1.5)
        ax.axvline(p1, color='red', ls='--', alpha=0.7, label=f'peak1 (col={p1})')
        ax.axvline(p2, color='orange', ls='--', alpha=0.7, label=f'peak2 (col={p2})')

        # expected peak positions
        exp_col_ha = c0 + (lam_ha - lam_center) / dispersion
        exp_col_nii = c0 + (lam_nii - lam_center) / dispersion
        ax.axvline(exp_col_ha, color='red', ls=':', alpha=0.4, label='expected Hα')
        ax.axvline(
            exp_col_nii, color='orange', ls=':', alpha=0.4, label='expected [NII]'
        )
        ax.set_xlabel('Column (pix)')
        ax.set_ylabel('Flux')
        ax.legend(fontsize=8)

        ok = sep_err < 1.0 and ratio_err < 0.10
        color = 'green' if ok else 'red'
        fig.suptitle(
            f'Multi-line separation — {"PASS" if ok else "FAIL"} '
            f'(sep_err={sep_err:.2f} pix, ratio_err={ratio_err:.2%})'
            f'\nΔpix = (λ_NII − λ_Hα)(1+z)/dispersion = {sep_expected:.1f}; '
            f'flux ratio = {expected_ratio:.1f}',
            color=color,
            fontsize=11,
        )
        fig.tight_layout()
        _save_diagnostic(fig, 'multiline_separation')

        assert sep_err < 1.0, f"Peak separation error {sep_err:.2f} pix >= 1.0"
        assert ratio_err < 0.10, f"Flux ratio error {ratio_err:.2%} >= 10%"

    # ------------------------------------------------------------------
    # 14. Continuum pedestal
    # ------------------------------------------------------------------
    def test_continuum_pedestal(self):
        """Continuum and emission line are linearly separable in the grism."""
        import matplotlib.pyplot as plt

        vel_model = CenteredVelocityModel()
        int_model = InclinedExponentialModel()
        config = SpectralConfig(lines=(halpha_line(),), spectral_oversample=5)
        sm = SpectralModel(config, int_model, vel_model)
        kl = KLModel(vel_model, int_model, shared_pars=_SHARED_PARS, spectral_model=sm)

        z = 1.0
        cont_val = 1.0

        base_pars = {
            'cosi': 0.5,
            'theta_int': 0.0,
            'g1': 0.0,
            'g2': 0.0,
            'v0': 0.0,
            'vcirc': 0.0,  # static to avoid velocity complications
            'vel_rscale': 0.5,
            'flux': 100.0,
            'int_rscale': 0.15,  # compact
            'int_h_over_r': 0.1,
            'int_x0': 0.0,
            'int_y0': 0.0,
            'z': z,
            'vel_dispersion': 50.0,
            'Ha_flux': 100.0,
        }

        gp = GrismPars(
            image_pars=_ANALYTICAL_IMAGE_PARS,
            dispersion=1.1,
            lambda_ref=HALPHA.lambda_rest * (1 + z),
            dispersion_angle=0.0,
        )
        cp = gp.to_cube_pars(z=z)

        # with continuum
        pars_cont = {**base_pars, 'Ha_cont': cont_val}
        theta_cont = kl.pars2theta(pars_cont)
        grism_cont = np.array(kl.render_grism(theta_cont, gp, cube_pars=cp))

        # without continuum
        pars_nocont = {**base_pars, 'Ha_cont': 0.0}
        theta_nocont = kl.pars2theta(pars_nocont)
        grism_nocont = np.array(kl.render_grism(theta_nocont, gp, cube_pars=cp))

        diff = grism_cont - grism_nocont

        # build expected by explicitly dispersing a uniform-spectrum cube
        # each slice = I_broadband * cont_val
        theta_int = kl.get_intensity_pars(theta_cont)
        I_broadband = np.array(
            int_model.render_unconvolved(theta_int, _ANALYTICAL_IMAGE_PARS)
        )
        cont_cube = (
            np.broadcast_to(
                I_broadband[:, :, None], (*I_broadband.shape, cp.n_lambda)
            ).copy()
            * cont_val
        )
        expected = np.array(disperse_cube(jnp.array(cont_cube), gp, cp.lambda_grid))

        peak = max(expected.max(), 1e-16)
        residual = np.abs(diff - expected)
        rel_max = residual.max() / peak

        # diagnostic
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        im0 = axes[0].imshow(diff, origin='lower')
        axes[0].set_title('Continuum diff (with − without)')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(expected, origin='lower')
        axes[1].set_title('Expected (dispersed uniform cube)')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        vmax_r = max(residual.max(), 1e-16)
        im2 = axes[2].imshow(residual, origin='lower', cmap='hot', vmax=vmax_r)
        axes[2].set_title(f'|Residual| (max={residual.max():.2e})')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        ok = rel_max < 1e-4
        color = 'green' if ok else 'red'
        fig.suptitle(
            f'Continuum pedestal — {"PASS" if ok else "FAIL"} '
            f'(rel_max={rel_max:.2e}, tol=1e-4)'
            f'\nContinuum and emission line are linearly separable '
            f'in the grism',
            color=color,
            fontsize=11,
        )
        fig.tight_layout()
        _save_diagnostic(fig, 'continuum_pedestal')

        assert (
            rel_max < 1e-4
        ), f"Continuum pedestal rel error {rel_max:.2e} exceeds 1e-4"
