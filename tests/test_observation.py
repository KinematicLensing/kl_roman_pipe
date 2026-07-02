"""
Tests for observation types and factory functions.

Covers:
- Factory construction (with/without PSF, with/without data, oversample=1/5)
- JAX pytree round-trip (flatten -> unflatten -> verify fields match)
- Shape validation
- VelocityObs flux source modes
- GrismObs construction
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from kl_pipe.parameters import ImagePars
from kl_pipe.observation import (
    ImageObs,
    VelocityObs,
    GrismObs,
    build_image_obs,
    build_velocity_obs,
    build_grism_obs,
)
from kl_pipe.utils import build_map_grid_from_image_pars
from kl_pipe.render import RenderConfig


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def image_pars():
    return ImagePars(shape=(16, 16), pixel_scale=0.11, indexing='ij')


@pytest.fixture
def data_16(image_pars):
    return jnp.ones((image_pars.Nrow, image_pars.Ncol))


@pytest.fixture
def variance_16(image_pars):
    return jnp.ones((image_pars.Nrow, image_pars.Ncol)) * 0.01


@pytest.fixture
def gaussian_psf():
    import galsim

    return galsim.Gaussian(fwhm=0.2)


# ==============================================================================
# Array validation (shared across all three builders)
# ==============================================================================


class TestObsArrayValidation:
    """Loud validation of data/variance/mask in the obs builders."""

    def test_negative_variance_raises(self, image_pars, data_16):
        bad = jnp.ones((16, 16)) * 0.01
        bad = bad.at[0, 0].set(-1.0)
        with pytest.raises(ValueError, match='variance must be strictly positive'):
            build_image_obs(image_pars, data=data_16, variance=bad)

    def test_zero_variance_raises(self, image_pars, data_16):
        bad = jnp.zeros((16, 16))
        with pytest.raises(ValueError, match='variance must be strictly positive'):
            build_image_obs(image_pars, data=data_16, variance=bad)

    def test_scalar_variance_ok(self, image_pars, data_16):
        # scalar variance is allowed and broadcasts; only positivity checked
        obs = build_image_obs(image_pars, data=data_16, variance=0.01)
        assert obs.variance is not None
        with pytest.raises(ValueError, match='variance must be strictly positive'):
            build_image_obs(image_pars, data=data_16, variance=0.0)

    def test_variance_shape_mismatch_raises(self, image_pars, data_16):
        with pytest.raises(ValueError, match='variance shape'):
            build_image_obs(image_pars, data=data_16, variance=jnp.ones((8, 8)) * 0.01)

    def test_mask_shape_mismatch_raises(self, image_pars, data_16, variance_16):
        with pytest.raises(ValueError, match='mask shape'):
            build_image_obs(
                image_pars,
                data=data_16,
                variance=variance_16,
                mask=jnp.ones((8, 8), dtype=bool),
            )

    def test_velocity_builder_validates(self, image_pars, data_16):
        with pytest.raises(ValueError, match='variance must be strictly positive'):
            build_velocity_obs(image_pars, data=data_16, variance=jnp.zeros((16, 16)))


# ==============================================================================
# ImageObs factory tests
# ==============================================================================


class TestBuildImageObs:
    """Tests for build_image_obs factory."""

    def test_no_psf_no_data(self, image_pars):
        obs = build_image_obs(image_pars)
        assert obs.image_pars is image_pars
        assert obs.X.shape == (16, 16)
        assert obs.Y.shape == (16, 16)
        assert obs.psf_data is None
        assert obs.oversample == 5
        assert obs.data is None
        assert obs.variance is None
        assert obs.mask is None

    def test_with_data(self, image_pars, data_16, variance_16):
        obs = build_image_obs(image_pars, data=data_16, variance=variance_16)
        assert obs.data is not None
        assert obs.variance is not None
        assert obs.data.shape == (16, 16)

    def test_with_mask(self, image_pars, data_16, variance_16):
        mask = jnp.ones((16, 16), dtype=bool)
        obs = build_image_obs(image_pars, data=data_16, variance=variance_16, mask=mask)
        assert obs.mask is not None
        assert obs.mask.shape == (16, 16)

    def test_with_psf_oversample_1(self, image_pars, gaussian_psf):
        obs = build_image_obs(
            image_pars, psf=gaussian_psf, render_config=RenderConfig(oversample=1)
        )
        assert obs.psf_data is not None
        assert obs.oversample == 1
        assert obs.fine_X is None
        assert obs.fine_Y is None

    def test_with_psf_oversample_5(self, image_pars, gaussian_psf):
        obs = build_image_obs(
            image_pars, psf=gaussian_psf, render_config=RenderConfig(oversample=5)
        )
        assert obs.psf_data is not None
        assert obs.oversample == 5
        assert obs.fine_X is not None
        assert obs.fine_Y is not None
        assert obs.fine_X.shape == (80, 80)
        assert obs.fine_Y.shape == (80, 80)

    def test_with_psf_kspace(self, image_pars, gaussian_psf):
        from kl_pipe.intensity import InclinedExponentialModel

        model = InclinedExponentialModel()
        obs = build_image_obs(
            image_pars,
            psf=gaussian_psf,
            render_config=RenderConfig(oversample=5),
            int_model=model,
        )
        assert obs.kspace_psf_fft is not None


# ==============================================================================
# VelocityObs factory tests
# ==============================================================================


class TestBuildVelocityObs:
    """Tests for build_velocity_obs factory."""

    def test_no_psf(self, image_pars, data_16, variance_16):
        obs = build_velocity_obs(image_pars, data=data_16, variance=variance_16)
        assert isinstance(obs, VelocityObs)
        assert obs.psf_data is None
        assert obs.flux_model is None

    def test_with_psf_flux_model(self, image_pars, gaussian_psf, data_16, variance_16):
        from kl_pipe.intensity import InclinedExponentialModel

        int_model = InclinedExponentialModel()
        int_theta = int_model.pars2theta(
            {
                'flux': 100.0,
                'rscale': 0.5,
                'h_over_r': 0.1,
                'x0': 0.0,
                'y0': 0.0,
                'cosi': 0.5,
                'theta_int': 0.0,
                'g1': 0.0,
                'g2': 0.0,
            }
        )
        obs = build_velocity_obs(
            image_pars,
            psf=gaussian_psf,
            data=data_16,
            variance=variance_16,
            flux_model=int_model,
            flux_theta=int_theta,
        )
        assert isinstance(obs, VelocityObs)
        assert obs.flux_model is int_model
        assert obs.flux_theta is not None


# ==============================================================================
# JAX pytree round-trip tests
# ==============================================================================


class TestPytreeRoundTrip:
    """JAX pytree flatten -> unflatten preserves all fields."""

    def test_image_obs_roundtrip(self, image_pars, data_16, variance_16):
        obs = build_image_obs(image_pars, data=data_16, variance=variance_16)
        leaves, treedef = jax.tree_util.tree_flatten(obs)
        obs2 = treedef.unflatten(leaves)

        assert obs2.image_pars == obs.image_pars
        assert obs2.oversample == obs.oversample
        np.testing.assert_array_equal(np.array(obs2.X), np.array(obs.X))
        np.testing.assert_array_equal(np.array(obs2.data), np.array(obs.data))

    def test_velocity_obs_roundtrip(self, image_pars, data_16, variance_16):
        obs = build_velocity_obs(image_pars, data=data_16, variance=variance_16)
        leaves, treedef = jax.tree_util.tree_flatten(obs)
        obs2 = treedef.unflatten(leaves)

        assert isinstance(obs2, VelocityObs)
        assert obs2.image_pars == obs.image_pars
        np.testing.assert_array_equal(np.array(obs2.X), np.array(obs.X))

    def test_grism_obs_roundtrip(self):
        from kl_pipe.dispersion import GrismPars

        ip = ImagePars(shape=(8, 8), pixel_scale=0.11, indexing='ij')
        gp = GrismPars(
            image_pars=ip,
            dispersion=1.0,
            lambda_ref=1.5,
            dispersion_angle=0.0,
        )
        cp = gp.to_cube_pars(z=1.0)
        obs = GrismObs(grism_pars=gp, cube_pars=cp)

        leaves, treedef = jax.tree_util.tree_flatten(obs)
        obs2 = treedef.unflatten(leaves)

        assert obs2.grism_pars is obs.grism_pars
        assert obs2.cube_pars is obs.cube_pars
        assert obs2.oversample == obs.oversample

    def test_image_obs_with_psf_roundtrip(self, image_pars, gaussian_psf):
        obs = build_image_obs(
            image_pars, psf=gaussian_psf, render_config=RenderConfig(oversample=3)
        )
        leaves, treedef = jax.tree_util.tree_flatten(obs)
        obs2 = treedef.unflatten(leaves)

        assert obs2.oversample == 3
        assert obs2.psf_data is not None
        assert obs2.fine_X is not None
        np.testing.assert_array_equal(np.array(obs2.fine_X), np.array(obs.fine_X))


# ==============================================================================
# GrismObs factory tests
# ==============================================================================


class TestBuildGrismObs:
    """Tests for build_grism_obs factory."""

    def test_basic_construction(self):
        from kl_pipe.dispersion import GrismPars

        ip = ImagePars(shape=(8, 8), pixel_scale=0.11, indexing='ij')
        gp = GrismPars(
            image_pars=ip,
            dispersion=1.0,
            lambda_ref=1.5,
            dispersion_angle=0.0,
        )
        obs = build_grism_obs(gp, z=1.0)

        assert isinstance(obs, GrismObs)
        assert obs.grism_pars is gp
        assert obs.cube_pars is not None
        assert obs.psf_data is None

    def test_with_psf(self):
        import galsim
        from kl_pipe.dispersion import GrismPars

        ip = ImagePars(shape=(8, 8), pixel_scale=0.11, indexing='ij')
        gp = GrismPars(
            image_pars=ip,
            dispersion=1.0,
            lambda_ref=1.5,
            dispersion_angle=0.0,
        )
        psf = galsim.Gaussian(fwhm=0.2)
        obs = build_grism_obs(
            gp, z=1.0, psf=psf, render_config=RenderConfig(oversample=3)
        )

        assert obs.psf_data is not None
        assert obs.oversample == 3
        assert obs.fine_image_pars is not None
