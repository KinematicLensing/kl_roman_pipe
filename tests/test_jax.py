# test_jax.py
"""
Tests for JAX compatibility of kl_pipe models.

Verifies that models work correctly with JAX transformations:
- jax.jit (compilation)
- jax.grad (automatic differentiation)
- jax.vmap (vectorization)
"""

import pytest
import jax
import jax.numpy as jnp
from kl_pipe.velocity import OffsetVelocityModel, CenteredVelocityModel
from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.model import KLModel
from kl_pipe.parameters import ImagePars
from kl_pipe.utils import build_map_grid_from_image_pars


# ----------------------------------------------------------------------
# Fixtures


@pytest.fixture
def simple_velocity_model():
    """Simple centered velocity model for testing."""
    return CenteredVelocityModel()


@pytest.fixture
def offset_velocity_model():
    """Offset velocity model for testing."""
    return OffsetVelocityModel()


@pytest.fixture
def simple_theta():
    """Parameter array for CenteredVelocityModel."""
    return jnp.array([0.6, 0.785, 0.05, -0.03, 10.0, 200.0, 5.0])


@pytest.fixture
def offset_theta():
    """Parameter array for OffsetVelocityModel."""
    return jnp.array([0.6, 0.785, 0.05, -0.03, 10.0, 200.0, 5.0, 1.0, -0.5])


@pytest.fixture
def test_grid():
    """Test coordinate grids."""
    x = jnp.linspace(-10, 10, 20)
    y = jnp.linspace(-10, 10, 20)
    return jnp.meshgrid(x, y, indexing='ij')


@pytest.fixture
def test_image_pars():
    """ImagePars for testing."""
    return ImagePars(shape=(32, 32), pixel_scale=0.5, indexing='ij')


@pytest.fixture
def kl_model_setup():
    """Setup a basic KLModel for testing."""
    vel_model = OffsetVelocityModel()
    int_model = InclinedExponentialModel()
    kl_model = KLModel(
        vel_model, int_model, shared_pars={'g1', 'g2', 'theta_int', 'cosi'}
    )
    return kl_model


@pytest.fixture
def kl_theta():
    """Composite theta for KLModel with shared transformation params."""
    # Order based on KLModel.PARAMETER_NAMES after construction
    # This will depend on your specific KLModel setup
    return jnp.array(
        [
            0.6,  # cosi (shared)
            0.785,  # theta_int (shared)
            0.05,  # g1 (shared)
            -0.03,  # g2 (shared
            10.0,  # v0
            200.0,  # vcirc
            5.0,  # vel_rscale
            1.0,  # vel_x0
            -0.5,  # vel_y0
            1.0,  # I0
            3.0,  # int_rscale
            1.0,  # int_x0
            -0.5,  # int_y0
        ]
    )


# ----------------------------------------------------------------------
# Velocity Model JIT compilation tests


def test_velocity_model_jit_compilation(simple_velocity_model, simple_theta, test_grid):
    """Test that velocity model __call__ can be JIT compiled."""
    X, Y = test_grid

    # Create JIT-compiled version
    jitted_call = jax.jit(lambda theta: simple_velocity_model(theta, 'obs', X, Y))

    # Should compile and run without error
    result = jitted_call(simple_theta)

    assert result.shape == X.shape
    assert jnp.isfinite(result).all()


def test_render_image_jit_compilation(
    simple_velocity_model, simple_theta, test_image_pars
):
    """Test that render_image can be JIT compiled."""
    # JIT compile render_image
    jitted_render = jax.jit(
        lambda theta: simple_velocity_model.render_image(
            theta, test_image_pars, plane='obs'
        )
    )

    result = jitted_render(simple_theta)

    assert result.shape == test_image_pars.shape
    assert jnp.isfinite(result).all()


def test_jit_with_return_speed(simple_velocity_model, simple_theta, test_grid):
    """Test JIT compilation with return_speed parameter."""
    X, Y = test_grid

    # JIT compile with static return_speed
    jitted_velocity = jax.jit(
        lambda theta: simple_velocity_model(theta, 'obs', X, Y, return_speed=False)
    )
    jitted_speed = jax.jit(
        lambda theta: simple_velocity_model(theta, 'obs', X, Y, return_speed=True)
    )

    v_map = jitted_velocity(simple_theta)
    s_map = jitted_speed(simple_theta)

    assert not jnp.allclose(v_map, s_map)
    assert jnp.all(s_map >= 0)


def test_offset_model_jit(offset_velocity_model, offset_theta, test_grid):
    """Test JIT compilation for offset velocity model."""
    X, Y = test_grid

    jitted_call = jax.jit(lambda theta: offset_velocity_model(theta, 'obs', X, Y))
    result = jitted_call(offset_theta)

    assert result.shape == X.shape
    assert jnp.isfinite(result).all()


# ----------------------------------------------------------------------
# Velocity Model gradient tests


def test_velocity_model_gradient(simple_velocity_model, simple_theta, test_grid):
    """Test that gradients can be computed through velocity model."""
    X, Y = test_grid

    def loss_fn(theta):
        v_map = simple_velocity_model(theta, 'obs', X, Y)
        return jnp.sum(v_map**2)

    # Compute gradient
    grad_fn = jax.grad(loss_fn)
    gradient = grad_fn(simple_theta)

    assert gradient.shape == simple_theta.shape
    assert jnp.isfinite(gradient).all()
    assert not jnp.all(gradient == 0)  # Should have non-zero gradients


def test_render_image_gradient(simple_velocity_model, simple_theta, test_image_pars):
    """Test gradients through render_image."""

    def loss_fn(theta):
        image = simple_velocity_model.render_image(theta, test_image_pars, plane='obs')
        return jnp.sum(image**2)

    gradient = jax.grad(loss_fn)(simple_theta)

    assert gradient.shape == simple_theta.shape
    assert jnp.isfinite(gradient).all()


def test_gradient_of_specific_parameters(simple_velocity_model, test_grid):
    """Test gradients with respect to individual parameters."""
    X, Y = test_grid
    theta = jnp.array([0.6, 0.785, 0.05, -0.03, 10.0, 200.0, 5.0])

    def v_wrt_vcirc(vcirc):
        theta_local = theta.at[1].set(vcirc)
        v_map = simple_velocity_model(theta_local, 'obs', X, Y)
        return jnp.mean(jnp.abs(v_map))

    # Gradient w.r.t. vcirc
    grad_vcirc = jax.grad(v_wrt_vcirc)(200.0)

    assert jnp.isfinite(grad_vcirc)
    assert grad_vcirc != 0


def test_value_and_grad(simple_velocity_model, simple_theta, test_image_pars):
    """Test value_and_grad for efficiency."""

    def objective(theta):
        image = simple_velocity_model.render_image(theta, test_image_pars)
        return jnp.sum(image**2)

    value_and_grad_fn = jax.value_and_grad(objective)
    value, grad = value_and_grad_fn(simple_theta)

    assert jnp.isfinite(value)
    assert grad.shape == simple_theta.shape
    assert jnp.isfinite(grad).all()


def test_offset_model_gradient(offset_velocity_model, offset_theta, test_grid):
    """Test gradients for offset model (includes position parameters)."""
    X, Y = test_grid

    def loss_fn(theta):
        v_map = offset_velocity_model(theta, 'obs', X, Y)
        return jnp.sum(v_map**2)

    gradient = jax.grad(loss_fn)(offset_theta)

    assert gradient.shape == offset_theta.shape
    assert jnp.isfinite(gradient).all()
    # Position parameters should have gradients
    assert gradient[7] != 0  # vel_x0
    assert gradient[8] != 0  # vel_y0


# ----------------------------------------------------------------------
# Vmap tests


def test_vmap_over_theta_samples(simple_velocity_model, test_grid):
    """Test vectorization over multiple parameter samples."""
    X, Y = test_grid

    # Multiple theta samples
    theta_samples = jnp.array(
        [
            [0.6, 0.785, 0.05, -0.03, 10.0, 180.0, 5.0],
            [0.6, 0.785, 0.05, -0.03, 10.0, 200.0, 5.0],
            [0.6, 0.785, 0.05, -0.03, 10.0, 220.0, 5.0],
        ]
    )

    # Vmap over first axis (different theta samples)
    vmapped_eval = jax.vmap(lambda theta: simple_velocity_model(theta, 'obs', X, Y))

    results = vmapped_eval(theta_samples)

    assert results.shape == (3,) + X.shape
    assert jnp.isfinite(results).all()


def test_vmap_render_image(simple_velocity_model, test_image_pars):
    """Test vmap with render_image."""
    theta_samples = jnp.array(
        [
            [0.6, 0.785, 0.05, -0.03, 10.0, 200.0, 5.0],
            [0.6, 0.785, 0.05, -0.03, 10.0, 150.0, 5.0],
        ]
    )

    vmapped_render = jax.vmap(
        lambda theta: simple_velocity_model.render_image(theta, test_image_pars)
    )

    images = vmapped_render(theta_samples)

    assert images.shape == (2,) + test_image_pars.shape
    assert jnp.isfinite(images).all()


# ----------------------------------------------------------------------
# Combined transformations


def test_jit_and_grad_composition(simple_velocity_model, simple_theta, test_image_pars):
    """Test that JIT and grad can be composed."""

    def loss_fn(theta):
        image = simple_velocity_model.render_image(theta, test_image_pars)
        return jnp.sum(image**2)

    # Compose JIT and grad
    jitted_grad = jax.jit(jax.grad(loss_fn))

    gradient = jitted_grad(simple_theta)

    assert gradient.shape == simple_theta.shape
    assert jnp.isfinite(gradient).all()


def test_jit_vmap_grad_composition(simple_velocity_model):
    """Test complex composition of JAX transformations."""
    image_pars = ImagePars(shape=(16, 16), pixel_scale=0.5, indexing='ij')

    theta_samples = jnp.array(
        [
            [0.6, 0.785, 0.05, -0.03, 10.0, 200.0, 5.0],
            [0.6, 0.785, 0.05, -0.03, 10.0, 180.0, 5.0],
        ]
    )

    def loss_fn(theta):
        image = simple_velocity_model.render_image(theta, image_pars)
        return jnp.mean(image**2)

    # JIT(vmap(grad))
    jitted_vmapped_grad = jax.jit(jax.vmap(jax.grad(loss_fn)))

    gradients = jitted_vmapped_grad(theta_samples)

    assert gradients.shape == theta_samples.shape
    assert jnp.isfinite(gradients).all()


# ----------------------------------------------------------------------
# Performance/compilation tests


def test_recompilation_with_same_shapes(
    simple_velocity_model, simple_theta, test_image_pars
):
    """Test that repeated calls don't cause recompilation."""
    jitted_fn = jax.jit(
        lambda theta: simple_velocity_model.render_image(theta, test_image_pars)
    )

    # First call (compilation)
    result1 = jitted_fn(simple_theta)

    # Subsequent calls (should use cached compilation)
    result2 = jitted_fn(simple_theta * 1.1)
    result3 = jitted_fn(simple_theta * 0.9)

    # Results should be different but shapes same
    assert result1.shape == result2.shape == result3.shape
    assert not jnp.allclose(result1, result2)


# ----------------------------------------------------------------------
# KLModel JIT compilation tests


def test_kl_model_jit_compilation(kl_model_setup, kl_theta, test_grid):
    """Test that KLModel evaluation can be JIT compiled."""
    kl_model = kl_model_setup
    X, Y = test_grid

    # JIT compile the full model evaluation
    jitted_eval = jax.jit(lambda theta: kl_model(theta, 'obs', X, Y))

    vel_map, int_map = jitted_eval(kl_theta)

    assert vel_map.shape == X.shape
    assert int_map.shape == X.shape
    assert jnp.isfinite(vel_map).all()
    assert jnp.isfinite(int_map).all()


def test_kl_model_render_image_jit(kl_model_setup, kl_theta, test_image_pars):
    """Test that KLModel render methods can be JIT compiled."""
    kl_model = kl_model_setup

    # JIT compile velocity render
    jitted_vel_render = jax.jit(
        lambda theta: kl_model.velocity_model.render_image(theta, test_image_pars)
    )

    # JIT compile intensity render
    jitted_int_render = jax.jit(
        lambda theta: kl_model.intensity_model.render_image(theta, test_image_pars)
    )

    # Extract sub-thetas
    theta_vel = kl_model.get_velocity_pars(kl_theta)
    theta_int = kl_model.get_intensity_pars(kl_theta)

    vel_image = jitted_vel_render(theta_vel)
    int_image = jitted_int_render(theta_int)

    assert vel_image.shape == test_image_pars.shape
    assert int_image.shape == test_image_pars.shape


def test_kl_model_parameter_extraction_jit(kl_model_setup, kl_theta):
    """Test that parameter extraction is JIT compatible."""
    kl_model = kl_model_setup

    # These should work in JIT context
    @jax.jit
    def extract_params(theta):
        theta_vel = kl_model.get_velocity_pars(theta)
        theta_int = kl_model.get_intensity_pars(theta)
        return theta_vel, theta_int

    theta_vel, theta_int = extract_params(kl_theta)

    assert theta_vel.shape[0] == len(kl_model.velocity_model.PARAMETER_NAMES)
    assert theta_int.shape[0] == len(kl_model.intensity_model.PARAMETER_NAMES)


# ----------------------------------------------------------------------
# KLModel gradient tests


def test_kl_model_velocity_gradient(kl_model_setup, kl_theta, test_grid):
    """Test gradients through velocity component of KLModel."""
    kl_model = kl_model_setup
    X, Y = test_grid

    def velocity_loss(theta):
        theta_vel = kl_model.get_velocity_pars(theta)
        vel_map = kl_model.velocity_model(theta_vel, 'obs', X, Y)
        return jnp.sum(vel_map**2)

    gradient = jax.grad(velocity_loss)(kl_theta)

    assert gradient.shape == kl_theta.shape
    assert jnp.isfinite(gradient).all()
    # Check that velocity parameters have non-zero gradients
    assert not jnp.all(gradient[:9] == 0)


def test_kl_model_intensity_gradient(kl_model_setup, kl_theta, test_grid):
    """Test gradients through intensity component of KLModel."""
    kl_model = kl_model_setup
    X, Y = test_grid

    def intensity_loss(theta):
        theta_int = kl_model.get_intensity_pars(theta)
        int_map = kl_model.intensity_model(theta_int, 'obs', X, Y)
        return jnp.sum(int_map**2)

    gradient = jax.grad(intensity_loss)(kl_theta)

    assert gradient.shape == kl_theta.shape
    assert jnp.isfinite(gradient).all()
    # Check that intensity parameters have non-zero gradients
    assert not jnp.all(gradient[9:] == 0)


def test_kl_model_combined_gradient(kl_model_setup, kl_theta, test_grid):
    """Test gradients through combined velocity * intensity."""
    kl_model = kl_model_setup
    X, Y = test_grid

    def combined_loss(theta):
        vel_map, int_map = kl_model(theta, 'obs', X, Y)
        combined = vel_map * int_map
        return jnp.sum(combined**2)

    gradient = jax.grad(combined_loss)(kl_theta)

    assert gradient.shape == kl_theta.shape
    assert jnp.isfinite(gradient).all()
    # Both velocity and intensity params should have gradients
    assert not jnp.all(gradient[:9] == 0)  # velocity params
    assert not jnp.all(gradient[9:] == 0)  # intensity params


def test_kl_model_shared_parameter_gradient(kl_model_setup, kl_theta, test_grid):
    """Test that shared parameters get gradients from both models."""
    kl_model = kl_model_setup
    X, Y = test_grid

    # Find index of a shared parameter (e.g., 'cosi')
    cosi_idx = list(kl_model.PARAMETER_NAMES).index('cosi')

    def loss_wrt_cosi(cosi_val):
        theta_local = kl_theta.at[cosi_idx].set(cosi_val)
        vel_map, int_map = kl_model(theta_local, 'obs', X, Y)
        # Loss depends on both velocity and intensity
        return jnp.sum(vel_map**2) + jnp.sum(int_map**2)

    grad_cosi = jax.grad(loss_wrt_cosi)(0.6)

    assert jnp.isfinite(grad_cosi)
    assert grad_cosi != 0  # Shared param affects both models


def test_kl_model_render_gradient(kl_model_setup, kl_theta, test_image_pars):
    """Test gradients through render_image for composite model."""
    kl_model = kl_model_setup

    def render_loss(theta):
        # Get component parameters
        theta_vel = kl_model.get_velocity_pars(theta)
        theta_int = kl_model.get_intensity_pars(theta)

        # Render both
        vel_img = kl_model.velocity_model.render_image(theta_vel, test_image_pars)
        int_img = kl_model.intensity_model.render_image(theta_int, test_image_pars)

        # Combined loss
        combined = vel_img * int_img
        return jnp.mean(combined**2)

    value, gradient = jax.value_and_grad(render_loss)(kl_theta)

    assert jnp.isfinite(value)
    assert gradient.shape == kl_theta.shape
    assert jnp.isfinite(gradient).all()


# ----------------------------------------------------------------------
# KLModel vmap tests


def test_kl_model_vmap_over_samples(kl_model_setup, test_grid):
    """Test vmapping over multiple parameter samples."""
    kl_model = kl_model_setup
    X, Y = test_grid

    # Multiple theta samples (vary vcirc)
    base_theta = jnp.array(
        [0.6, 0.785, 0.05, -0.03, 10.0, 200.0, 5.0, 1.0, -0.5, 1.0, 3.0, 1.0, -0.5]
    )
    theta_samples = jnp.stack(
        [
            base_theta.at[5].set(180.0),
            base_theta.at[5].set(200.0),
            base_theta.at[5].set(220.0),
        ]
    )

    # Vmap over theta samples
    vmapped_eval = jax.vmap(lambda theta: kl_model(theta, 'obs', X, Y))

    vel_maps, int_maps = vmapped_eval(theta_samples)

    assert vel_maps.shape == (3,) + X.shape
    assert int_maps.shape == (3,) + X.shape
    assert jnp.isfinite(vel_maps).all()
    assert jnp.isfinite(int_maps).all()


def test_kl_model_vmap_render(kl_model_setup):
    """Test vmapping render_image over parameter samples."""
    kl_model = kl_model_setup
    image_pars = ImagePars(shape=(16, 16), pixel_scale=0.5, indexing='ij')

    base_theta = jnp.array(
        [10.0, 200.0, 5.0, 0.6, 0.785, 0.0, 0.0, 1.0, -0.5, 1.0, 3.0, 1.0, -0.5]
    )
    theta_samples = jnp.stack(
        [
            base_theta.at[1].set(180.0),
            base_theta.at[1].set(220.0),
        ]
    )

    def render_both(theta):
        theta_vel = kl_model.get_velocity_pars(theta)
        theta_int = kl_model.get_intensity_pars(theta)
        vel_img = kl_model.velocity_model.render_image(theta_vel, image_pars)
        int_img = kl_model.intensity_model.render_image(theta_int, image_pars)
        return vel_img, int_img

    vmapped_render = jax.vmap(render_both)
    vel_images, int_images = vmapped_render(theta_samples)

    assert vel_images.shape == (2, 16, 16)
    assert int_images.shape == (2, 16, 16)


# ----------------------------------------------------------------------
# Combined JAX transformation tests for KLModel


def test_kl_model_jit_grad_composition(kl_model_setup, kl_theta, test_grid):
    """Test JIT(grad) composition for KLModel."""
    kl_model = kl_model_setup
    X, Y = test_grid

    def loss_fn(theta):
        vel_map, int_map = kl_model(theta, 'obs', X, Y)
        return jnp.sum((vel_map * int_map) ** 2)

    jitted_grad = jax.jit(jax.grad(loss_fn))
    gradient = jitted_grad(kl_theta)

    assert gradient.shape == kl_theta.shape
    assert jnp.isfinite(gradient).all()


def test_kl_model_jit_vmap_grad(kl_model_setup):
    """Test JIT(vmap(grad)) composition for KLModel."""
    kl_model = kl_model_setup
    image_pars = ImagePars(shape=(16, 16), pixel_scale=0.5, indexing='ij')

    base_theta = jnp.array(
        [0.6, 0.785, 0.05, -0.03, 10.0, 200.0, 5.0, 1.0, -0.5, 1.0, 3.0, 1.0, -0.5]
    )
    theta_samples = jnp.stack(
        [
            base_theta,
            base_theta.at[1].set(180.0),
        ]
    )

    def loss_fn(theta):
        theta_vel = kl_model.get_velocity_pars(theta)
        theta_int = kl_model.get_intensity_pars(theta)
        vel_img = kl_model.velocity_model.render_image(theta_vel, image_pars)
        int_img = kl_model.intensity_model.render_image(theta_int, image_pars)
        return jnp.mean((vel_img * int_img) ** 2)

    # Compose transformations
    jitted_vmapped_grad = jax.jit(jax.vmap(jax.grad(loss_fn)))
    gradients = jitted_vmapped_grad(theta_samples)

    assert gradients.shape == theta_samples.shape
    assert jnp.isfinite(gradients).all()


# ----------------------------------------------------------------------
# Likelihood-style tests (realistic MCMC use case)


def test_kl_model_likelihood_gradient(kl_model_setup, kl_theta):
    """Test gradient computation for a likelihood-style objective (MCMC use case)."""
    kl_model = kl_model_setup
    image_pars = ImagePars(shape=(32, 32), pixel_scale=0.5, indexing='ij')

    # Simulate observed data
    X, Y = build_map_grid_from_image_pars(image_pars, unit='arcsec', centered=True)

    # Get "true" data
    vel_true, int_true = kl_model(kl_theta, 'obs', X, Y)
    data = vel_true * int_true

    # Define log-likelihood
    def log_likelihood(theta):
        vel_map, int_map = kl_model(theta, 'obs', X, Y)
        model_pred = vel_map * int_map
        residuals = data - model_pred
        chi2 = jnp.sum(residuals**2)
        return -0.5 * chi2

    # Compute gradient (as MCMC would)
    log_prob, gradient = jax.value_and_grad(log_likelihood)(kl_theta)

    assert jnp.isfinite(log_prob)
    assert gradient.shape == kl_theta.shape
    assert jnp.isfinite(gradient).all()
    # Gradient at true parameters should be small (we're at maximum)
    assert jnp.linalg.norm(gradient) < 1e3


def test_kl_model_jitted_likelihood(kl_model_setup, kl_theta):
    """Test JIT-compiled likelihood for MCMC."""
    kl_model = kl_model_setup
    image_pars = ImagePars(shape=(32, 32), pixel_scale=0.5, indexing='ij')

    X, Y = build_map_grid_from_image_pars(image_pars, unit='arcsec', centered=True)

    vel_true, int_true = kl_model(kl_theta, 'obs', X, Y)
    data = vel_true * int_true

    # JIT-compiled likelihood
    @jax.jit
    def log_likelihood(theta):
        vel_map, int_map = kl_model(theta, 'obs', X, Y)
        model_pred = vel_map * int_map
        residuals = data - model_pred
        return -0.5 * jnp.sum(residuals**2)

    # Should compile once and run fast
    log_prob1 = log_likelihood(kl_theta)
    log_prob2 = log_likelihood(kl_theta * 1.01)

    assert jnp.isfinite(log_prob1)
    assert jnp.isfinite(log_prob2)
    assert log_prob1 != log_prob2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
