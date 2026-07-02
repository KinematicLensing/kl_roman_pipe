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
from kl_pipe.source import SourceModel, _build_component_theta
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
    return jnp.meshgrid(x, y, indexing='xy')


@pytest.fixture
def test_image_pars():
    """ImagePars for testing."""
    return ImagePars(shape=(32, 32), pixel_scale=0.5, indexing='ij')


@pytest.fixture
def source_setup():
    """SourceModel with OffsetVelocity + InclinedExponential broadband (F087)."""
    vel_model = OffsetVelocityModel()
    int_model = InclinedExponentialModel()
    source = SourceModel(velocity_model=vel_model, broadband_models={'F087': int_model})
    return source


@pytest.fixture
def kl_pars():
    """Dotted-key pars dict for the joint vel + broadband model.

    The 13 sampled-equivalents mirror the legacy 13-entry kl_theta:
    4 shared geometry (cosi, theta_int, g1, g2) + 5 vel + 4 broadband-only
    (broadband and vel share cosi/theta_int/g1/g2 at the top level).
    """
    return {
        # shared geometry (top-level, unprefixed)
        'cosi': 0.6,
        'theta_int': 0.785,
        'g1': 0.05,
        'g2': -0.03,
        # velocity
        'vel.v0': 10.0,
        'vel.vcirc': 200.0,
        'vel.rscale': 5.0,
        'vel.x0': 1.0,
        'vel.y0': -0.5,
        # F087 broadband intensity
        'F087.flux': 1.0,
        'F087.rscale': 3.0,
        'F087.h_over_r': 0.1,
        'F087.x0': 1.0,
        'F087.y0': -0.5,
    }


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
    assert gradient[7] != 0  # x0
    assert gradient[8] != 0  # y0


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


def test_source_model_jit_compilation(source_setup, kl_pars, test_grid):
    """Test that SourceModel velocity + broadband evaluation can be JIT compiled."""
    source = source_setup
    X, Y = test_grid
    vel_m = source.velocity_model
    int_m = source.broadband_models['F087']

    def eval_both(pars):
        theta_vel = _build_component_theta(pars, 'vel', vel_m.PARAMETER_NAMES)
        theta_int = _build_component_theta(pars, 'F087', int_m.PARAMETER_NAMES)
        vel_map = vel_m(theta_vel, 'obs', X, Y)
        int_map = int_m(theta_int, 'obs', X, Y)
        return vel_map, int_map

    jitted_eval = jax.jit(eval_both)
    vel_map, int_map = jitted_eval(kl_pars)

    assert vel_map.shape == X.shape
    assert int_map.shape == X.shape
    assert jnp.isfinite(vel_map).all()
    assert jnp.isfinite(int_map).all()


def test_source_model_render_image_jit(source_setup, kl_pars, test_image_pars):
    """Test that bare velocity_model / broadband_model render methods JIT compile."""
    source = source_setup
    vel_m = source.velocity_model
    int_m = source.broadband_models['F087']

    jitted_vel_render = jax.jit(
        lambda theta: vel_m.render_image(theta, test_image_pars)
    )
    jitted_int_render = jax.jit(
        lambda theta: int_m.render_image(theta, test_image_pars)
    )

    theta_vel = _build_component_theta(kl_pars, 'vel', vel_m.PARAMETER_NAMES)
    theta_int = _build_component_theta(kl_pars, 'F087', int_m.PARAMETER_NAMES)

    vel_image = jitted_vel_render(theta_vel)
    int_image = jitted_int_render(theta_int)

    assert vel_image.shape == test_image_pars.shape
    assert int_image.shape == test_image_pars.shape


def test_source_model_parameter_extraction_jit(source_setup, kl_pars):
    """Test that per-component theta extraction is JIT compatible."""
    source = source_setup
    vel_m = source.velocity_model
    int_m = source.broadband_models['F087']

    @jax.jit
    def extract_params(pars):
        theta_vel = _build_component_theta(pars, 'vel', vel_m.PARAMETER_NAMES)
        theta_int = _build_component_theta(pars, 'F087', int_m.PARAMETER_NAMES)
        return theta_vel, theta_int

    theta_vel, theta_int = extract_params(kl_pars)

    assert theta_vel.shape[0] == len(vel_m.PARAMETER_NAMES)
    assert theta_int.shape[0] == len(int_m.PARAMETER_NAMES)


# ----------------------------------------------------------------------
# KLModel gradient tests


def test_source_model_velocity_gradient(source_setup, kl_pars, test_grid):
    """Test gradients through velocity component of SourceModel."""
    source = source_setup
    X, Y = test_grid
    vel_m = source.velocity_model

    def velocity_loss(pars):
        theta_vel = _build_component_theta(pars, 'vel', vel_m.PARAMETER_NAMES)
        vel_map = vel_m(theta_vel, 'obs', X, Y)
        return jnp.sum(vel_map**2)

    gradient = jax.grad(velocity_loss)(kl_pars)

    assert set(gradient.keys()) == set(kl_pars.keys())
    assert all(jnp.isfinite(jnp.asarray(v)).all() for v in gradient.values())
    # velocity-named keys should have non-zero gradients
    assert any(
        float(jnp.asarray(gradient[k])) != 0 for k in gradient if k.startswith('vel.')
    )


def test_source_model_intensity_gradient(source_setup, kl_pars, test_grid):
    """Test gradients through broadband intensity component of SourceModel."""
    source = source_setup
    X, Y = test_grid
    int_m = source.broadband_models['F087']

    def intensity_loss(pars):
        theta_int = _build_component_theta(pars, 'F087', int_m.PARAMETER_NAMES)
        int_map = int_m(theta_int, 'obs', X, Y)
        return jnp.sum(int_map**2)

    gradient = jax.grad(intensity_loss)(kl_pars)

    assert set(gradient.keys()) == set(kl_pars.keys())
    assert all(jnp.isfinite(jnp.asarray(v)).all() for v in gradient.values())
    assert any(
        float(jnp.asarray(gradient[k])) != 0 for k in gradient if k.startswith('F087.')
    )


def test_source_model_combined_gradient(source_setup, kl_pars, test_grid):
    """Test gradients through combined velocity * intensity."""
    source = source_setup
    X, Y = test_grid
    vel_m = source.velocity_model
    int_m = source.broadband_models['F087']

    def combined_loss(pars):
        theta_vel = _build_component_theta(pars, 'vel', vel_m.PARAMETER_NAMES)
        theta_int = _build_component_theta(pars, 'F087', int_m.PARAMETER_NAMES)
        vel_map = vel_m(theta_vel, 'obs', X, Y)
        int_map = int_m(theta_int, 'obs', X, Y)
        combined = vel_map * int_map
        return jnp.sum(combined**2)

    gradient = jax.grad(combined_loss)(kl_pars)

    assert set(gradient.keys()) == set(kl_pars.keys())
    assert all(jnp.isfinite(jnp.asarray(v)).all() for v in gradient.values())
    assert any(
        float(jnp.asarray(gradient[k])) != 0 for k in gradient if k.startswith('vel.')
    )
    assert any(
        float(jnp.asarray(gradient[k])) != 0 for k in gradient if k.startswith('F087.')
    )


def test_source_model_shared_parameter_gradient(source_setup, kl_pars, test_grid):
    """Test that shared parameters (cosi) get gradients from both models."""
    source = source_setup
    X, Y = test_grid
    vel_m = source.velocity_model
    int_m = source.broadband_models['F087']

    def loss_wrt_cosi(cosi_val):
        pars_local = {**kl_pars, 'cosi': cosi_val}
        theta_vel = _build_component_theta(pars_local, 'vel', vel_m.PARAMETER_NAMES)
        theta_int = _build_component_theta(pars_local, 'F087', int_m.PARAMETER_NAMES)
        vel_map = vel_m(theta_vel, 'obs', X, Y)
        int_map = int_m(theta_int, 'obs', X, Y)
        return jnp.sum(vel_map**2) + jnp.sum(int_map**2)

    grad_cosi = jax.grad(loss_wrt_cosi)(0.6)

    assert jnp.isfinite(grad_cosi)
    assert grad_cosi != 0  # shared param affects both models


def test_source_model_render_gradient(source_setup, kl_pars, test_image_pars):
    """Test gradients through render_image for SourceModel components."""
    source = source_setup
    vel_m = source.velocity_model
    int_m = source.broadband_models['F087']

    def render_loss(pars):
        theta_vel = _build_component_theta(pars, 'vel', vel_m.PARAMETER_NAMES)
        theta_int = _build_component_theta(pars, 'F087', int_m.PARAMETER_NAMES)
        vel_img = vel_m.render_image(theta_vel, test_image_pars)
        int_img = int_m.render_image(theta_int, test_image_pars)
        combined = vel_img * int_img
        return jnp.mean(combined**2)

    value, gradient = jax.value_and_grad(render_loss)(kl_pars)

    assert jnp.isfinite(value)
    assert set(gradient.keys()) == set(kl_pars.keys())
    assert all(jnp.isfinite(jnp.asarray(v)).all() for v in gradient.values())


# ----------------------------------------------------------------------
# KLModel vmap tests


def test_source_model_vmap_over_samples(source_setup, kl_pars, test_grid):
    """Test vmapping over multiple parameter samples (varying vel.vcirc)."""
    source = source_setup
    X, Y = test_grid
    vel_m = source.velocity_model
    int_m = source.broadband_models['F087']

    # build a stacked array of vcirc values + broadcast the rest of pars
    vcircs = jnp.array([180.0, 200.0, 220.0])

    def eval_for_vcirc(vcirc):
        pars_local = {**kl_pars, 'vel.vcirc': vcirc}
        theta_vel = _build_component_theta(pars_local, 'vel', vel_m.PARAMETER_NAMES)
        theta_int = _build_component_theta(pars_local, 'F087', int_m.PARAMETER_NAMES)
        vel_map = vel_m(theta_vel, 'obs', X, Y)
        int_map = int_m(theta_int, 'obs', X, Y)
        return vel_map, int_map

    vmapped_eval = jax.vmap(eval_for_vcirc)
    vel_maps, int_maps = vmapped_eval(vcircs)

    assert vel_maps.shape == (3,) + X.shape
    assert int_maps.shape == (3,) + X.shape
    assert jnp.isfinite(vel_maps).all()
    assert jnp.isfinite(int_maps).all()


def test_source_model_vmap_render(source_setup, kl_pars):
    """Test vmapping render_image over parameter samples (varying vel.vcirc)."""
    source = source_setup
    image_pars = ImagePars(shape=(16, 16), pixel_scale=0.5, indexing='ij')
    vel_m = source.velocity_model
    int_m = source.broadband_models['F087']

    vcircs = jnp.array([180.0, 220.0])

    def render_both(vcirc):
        pars_local = {**kl_pars, 'vel.vcirc': vcirc}
        theta_vel = _build_component_theta(pars_local, 'vel', vel_m.PARAMETER_NAMES)
        theta_int = _build_component_theta(pars_local, 'F087', int_m.PARAMETER_NAMES)
        vel_img = vel_m.render_image(theta_vel, image_pars)
        int_img = int_m.render_image(theta_int, image_pars)
        return vel_img, int_img

    vmapped_render = jax.vmap(render_both)
    vel_images, int_images = vmapped_render(vcircs)

    assert vel_images.shape == (2, 16, 16)
    assert int_images.shape == (2, 16, 16)


# ----------------------------------------------------------------------
# Combined JAX transformation tests for KLModel


def test_source_model_jit_grad_composition(source_setup, kl_pars, test_grid):
    """Test JIT(grad) composition for SourceModel."""
    source = source_setup
    X, Y = test_grid
    vel_m = source.velocity_model
    int_m = source.broadband_models['F087']

    def loss_fn(pars):
        theta_vel = _build_component_theta(pars, 'vel', vel_m.PARAMETER_NAMES)
        theta_int = _build_component_theta(pars, 'F087', int_m.PARAMETER_NAMES)
        vel_map = vel_m(theta_vel, 'obs', X, Y)
        int_map = int_m(theta_int, 'obs', X, Y)
        return jnp.sum((vel_map * int_map) ** 2)

    jitted_grad = jax.jit(jax.grad(loss_fn))
    gradient = jitted_grad(kl_pars)

    assert set(gradient.keys()) == set(kl_pars.keys())
    assert all(jnp.isfinite(jnp.asarray(v)).all() for v in gradient.values())


def test_source_model_jit_vmap_grad(source_setup, kl_pars):
    """Test JIT(vmap(grad)) composition for SourceModel (vary vel.vcirc)."""
    source = source_setup
    image_pars = ImagePars(shape=(16, 16), pixel_scale=0.5, indexing='ij')
    vel_m = source.velocity_model
    int_m = source.broadband_models['F087']

    vcircs = jnp.array([200.0, 180.0])

    def loss_fn(vcirc):
        pars_local = {**kl_pars, 'vel.vcirc': vcirc}
        theta_vel = _build_component_theta(pars_local, 'vel', vel_m.PARAMETER_NAMES)
        theta_int = _build_component_theta(pars_local, 'F087', int_m.PARAMETER_NAMES)
        vel_img = vel_m.render_image(theta_vel, image_pars)
        int_img = int_m.render_image(theta_int, image_pars)
        return jnp.mean((vel_img * int_img) ** 2)

    jitted_vmapped_grad = jax.jit(jax.vmap(jax.grad(loss_fn)))
    gradients = jitted_vmapped_grad(vcircs)

    assert gradients.shape == vcircs.shape
    assert jnp.isfinite(gradients).all()


# ----------------------------------------------------------------------
# Likelihood-style tests (realistic MCMC use case)


def test_source_model_likelihood_gradient(source_setup, kl_pars):
    """Test gradient computation for a likelihood-style objective (MCMC use case)."""
    source = source_setup
    image_pars = ImagePars(shape=(32, 32), pixel_scale=0.5, indexing='ij')
    vel_m = source.velocity_model
    int_m = source.broadband_models['F087']

    X, Y = build_map_grid_from_image_pars(image_pars, unit='arcsec', centered=True)

    def eval_combined(pars):
        theta_vel = _build_component_theta(pars, 'vel', vel_m.PARAMETER_NAMES)
        theta_int = _build_component_theta(pars, 'F087', int_m.PARAMETER_NAMES)
        return vel_m(theta_vel, 'obs', X, Y) * int_m(theta_int, 'obs', X, Y)

    # "true" data
    data = eval_combined(kl_pars)

    def log_likelihood(pars):
        model_pred = eval_combined(pars)
        residuals = data - model_pred
        return -0.5 * jnp.sum(residuals**2)

    log_prob, gradient = jax.value_and_grad(log_likelihood)(kl_pars)

    assert jnp.isfinite(log_prob)
    assert set(gradient.keys()) == set(kl_pars.keys())
    assert all(jnp.isfinite(jnp.asarray(v)).all() for v in gradient.values())
    # gradient at true parameters should be small (we're at maximum)
    grad_norm = float(
        jnp.sqrt(sum(jnp.sum(jnp.asarray(v) ** 2) for v in gradient.values()))
    )
    assert grad_norm < 1e3


def test_source_model_jitted_likelihood(source_setup, kl_pars):
    """Test JIT-compiled likelihood for MCMC."""
    source = source_setup
    image_pars = ImagePars(shape=(32, 32), pixel_scale=0.5, indexing='ij')
    vel_m = source.velocity_model
    int_m = source.broadband_models['F087']

    X, Y = build_map_grid_from_image_pars(image_pars, unit='arcsec', centered=True)

    def eval_combined(pars):
        theta_vel = _build_component_theta(pars, 'vel', vel_m.PARAMETER_NAMES)
        theta_int = _build_component_theta(pars, 'F087', int_m.PARAMETER_NAMES)
        return vel_m(theta_vel, 'obs', X, Y) * int_m(theta_int, 'obs', X, Y)

    data = eval_combined(kl_pars)

    @jax.jit
    def log_likelihood(pars):
        model_pred = eval_combined(pars)
        residuals = data - model_pred
        return -0.5 * jnp.sum(residuals**2)

    perturbed_pars = {k: v * 1.01 for k, v in kl_pars.items()}
    log_prob1 = log_likelihood(kl_pars)
    log_prob2 = log_likelihood(perturbed_pars)

    assert jnp.isfinite(log_prob1)
    assert jnp.isfinite(log_prob2)
    assert log_prob1 != log_prob2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
