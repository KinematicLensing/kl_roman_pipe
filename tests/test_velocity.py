'''
Unit tests for velocity models.

Tests include:
- Model instantiation
- Parameter conversion (theta <-> pars)
- Model evaluation in different planes
- Coordinate transformations
- Plotting utilities
'''

import pytest
import jax.numpy as jnp
import matplotlib.pyplot as plt

from kl_pipe.velocity import (
    OffsetVelocityModel,
    CenteredVelocityModel,
    build_velocity_model,
    VELOCITY_MODEL_TYPES,
)
from kl_pipe.parameters import ImagePars
from kl_pipe.utils import get_base_dir, get_test_dir
from kl_pipe import plotting


# ----------------------------------------------------------------------
# pytest fixtures


@pytest.fixture
def output_dir():
    """Create and return output directory for test plots."""
    base_dir = get_base_dir()
    out_dir = base_dir / "tests/out/velocity"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture
def basic_meta_pars():
    """Basic metadata for model instantiation."""
    return {'test_param': 'test_value'}


@pytest.fixture
def centered_theta():
    """Standard theta array for CenteredVelocityModel."""
    # (cosi, theta_int, g1, g2, v0, vcirc, vel_rscale)
    return jnp.array([0.6, 0.785, 0.05, -0.03, 10.0, 200.0, 5.0])


@pytest.fixture
def offset_theta():
    """Standard theta array for OffsetVelocityModel."""
    # (cosi, theta_int, g1, g2, v0, vcirc, vel_rscale, vel_x0, vel_y0)
    return jnp.array([0.6, 0.785, 0.05, -0.03, 10.0, 200.0, 5.0, 5.0, -3.0])


@pytest.fixture
def test_image_pars():
    """ImagePars for testing."""
    return ImagePars(shape=(64, 64), pixel_scale=0.3, indexing='ij')


# ----------------------------------------------------------------------
# basic instantiation tests


def test_centered_model_instantiation(basic_meta_pars):
    """Test CenteredVelocityModel can be instantiated."""
    model = CenteredVelocityModel(meta_pars=basic_meta_pars)
    assert model is not None
    assert model.name == 'centered'
    assert len(model.PARAMETER_NAMES) == 7


def test_offset_model_instantiation(basic_meta_pars):
    """Test OffsetVelocityModel can be instantiated."""
    model = OffsetVelocityModel(meta_pars=basic_meta_pars)
    assert model is not None
    assert model.name == 'offset'
    assert len(model.PARAMETER_NAMES) == 9


def test_model_parameter_names():
    """Test that parameter names are correctly defined."""
    centered = CenteredVelocityModel()
    offset = OffsetVelocityModel()

    assert 'v0' in centered.PARAMETER_NAMES
    assert 'vcirc' in centered.PARAMETER_NAMES
    assert 'vel_x0' not in centered.PARAMETER_NAMES

    assert 'vel_x0' in offset.PARAMETER_NAMES
    assert 'vel_y0' in offset.PARAMETER_NAMES


# Parameter conversion tests
def test_centered_theta2pars(centered_theta):
    """Test theta to pars conversion for centered model."""
    pars = CenteredVelocityModel.theta2pars(centered_theta)

    assert isinstance(pars, dict)
    assert len(pars) == 7
    assert 'v0' in pars
    assert 'vcirc' in pars
    assert pars['v0'] == 10.0
    assert pars['vcirc'] == 200.0


def test_offset_theta2pars(offset_theta):
    """Test theta to pars conversion for offset model."""
    pars = OffsetVelocityModel.theta2pars(offset_theta)

    assert isinstance(pars, dict)
    assert len(pars) == 9
    print('\nPARS:', pars, '\n')
    assert 'vel_x0' in pars
    assert 'vel_y0' in pars
    assert pars['vel_x0'] == 5.0
    assert pars['vel_y0'] == -3


def test_centered_pars2theta():
    """Test pars to theta conversion for centered model."""
    pars = {
        'cosi': 0.6,
        'theta_int': 0.785,
        'g1': 0.05,
        'g2': -0.03,
        'v0': 10.0,
        'vcirc': 200.0,
        'vel_rscale': 5.0,
    }
    theta = CenteredVelocityModel.pars2theta(pars)

    assert isinstance(theta, jnp.ndarray)
    assert len(theta) == 7
    assert theta[4] == 10.0
    assert theta[5] == 200.0


def test_offset_pars2theta():
    """Test pars to theta conversion for offset model."""
    pars = {
        'cosi': 0.6,
        'theta_int': 0.785,
        'g1': 0.05,
        'g2': -0.03,
        'v0': 10.0,
        'vcirc': 200.0,
        'vel_rscale': 5.0,
        'vel_x0': 2.0,
        'vel_y0': -1.5,
    }
    theta = OffsetVelocityModel.pars2theta(pars)

    assert isinstance(theta, jnp.ndarray)
    assert len(theta) == 9
    assert theta[7] == 2.0
    assert theta[8] == -1.5


def test_roundtrip_conversion_centered(centered_theta):
    """Test theta -> pars -> theta roundtrip for centered model."""
    pars = CenteredVelocityModel.theta2pars(centered_theta)
    theta_reconstructed = CenteredVelocityModel.pars2theta(pars)

    assert jnp.allclose(centered_theta, theta_reconstructed)


def test_roundtrip_conversion_offset(offset_theta):
    """Test theta -> pars -> theta roundtrip for offset model."""
    pars = OffsetVelocityModel.theta2pars(offset_theta)
    theta_reconstructed = OffsetVelocityModel.pars2theta(pars)

    assert jnp.allclose(offset_theta, theta_reconstructed)


# Parameter extraction tests
def test_get_param_centered(centered_theta):
    """Test get_param method for centered model."""
    model = CenteredVelocityModel()

    v0 = model.get_param('v0', centered_theta)
    vcirc = model.get_param('vcirc', centered_theta)

    assert float(v0) == 10.0
    assert float(vcirc) == 200.0


def test_get_param_offset(offset_theta):
    """Test get_param method for offset model."""
    model = OffsetVelocityModel()

    x0 = model.get_param('vel_x0', offset_theta)
    y0 = model.get_param('vel_y0', offset_theta)

    assert float(x0) == 5.0
    assert float(y0) == -3


# Model evaluation tests
def test_centered_circular_velocity_evaluation(centered_theta):
    """Test circular velocity evaluation for centered model."""
    model = CenteredVelocityModel()

    X = jnp.linspace(-10, 10, 20)
    Y = jnp.linspace(-10, 10, 20)
    X_grid, Y_grid = jnp.meshgrid(X, Y, indexing='ij')

    v_circ = model.evaluate_circular_velocity(centered_theta, X_grid, Y_grid)

    assert v_circ.shape == X_grid.shape
    assert jnp.all(v_circ >= 0)  # Circular velocity should be non-negative
    assert jnp.isfinite(v_circ).all()


def test_offset_circular_velocity_evaluation(offset_theta):
    """Test circular velocity evaluation for offset model."""
    model = OffsetVelocityModel()

    X = jnp.linspace(-10, 10, 20)
    Y = jnp.linspace(-10, 10, 20)
    X_grid, Y_grid = jnp.meshgrid(X, Y, indexing='ij')

    v_circ = model.evaluate_circular_velocity(offset_theta, X_grid, Y_grid)

    assert v_circ.shape == X_grid.shape
    assert jnp.all(v_circ >= 0)
    assert jnp.isfinite(v_circ).all()


def test_centered_velocity_map_evaluation(centered_theta):
    """Test full velocity map evaluation in obs plane."""
    model = CenteredVelocityModel()

    X = jnp.linspace(-10, 10, 20)
    Y = jnp.linspace(-10, 10, 20)
    X_grid, Y_grid = jnp.meshgrid(X, Y, indexing='ij')

    v_map = model(centered_theta, 'obs', X_grid, Y_grid)

    assert v_map.shape == X_grid.shape
    assert jnp.isfinite(v_map).all()
    # check that systemic velocity is included
    assert jnp.abs(jnp.mean(v_map) - 10.0) < 0.5  # should be near v0


def test_offset_velocity_map_evaluation(offset_theta):
    """Test full velocity map evaluation in obs plane."""
    model = OffsetVelocityModel()

    X = jnp.linspace(-5, 5, 20)
    Y = jnp.linspace(-5, 5, 20)
    X_grid, Y_grid = jnp.meshgrid(X, Y, indexing='ij')

    v_map = model(offset_theta, 'obs', X_grid, Y_grid)

    assert v_map.shape == X_grid.shape
    assert jnp.isfinite(v_map).all()


def test_velocity_map_different_planes(centered_theta):
    """Test evaluation in different coordinate planes."""
    model = CenteredVelocityModel()

    X = jnp.linspace(-10, 10, 15)
    Y = jnp.linspace(-10, 10, 15)
    X_grid, Y_grid = jnp.meshgrid(X, Y, indexing='ij')

    planes = ['disk', 'gal', 'source', 'obs']

    for plane in planes:
        v_map = model(centered_theta, plane, X_grid, Y_grid)
        assert v_map.shape == X_grid.shape
        assert jnp.isfinite(v_map).all()


# Factory function tests
def test_build_velocity_model_centered():
    """Test factory function for centered model."""
    model = build_velocity_model('centered')

    assert isinstance(model, CenteredVelocityModel)
    assert model.name == 'centered'


def test_build_velocity_model_offset():
    """Test factory function for offset model."""
    model = build_velocity_model('offset')

    assert isinstance(model, OffsetVelocityModel)
    assert model.name == 'offset'


def test_build_velocity_model_default():
    """Test factory function with default name."""
    model = build_velocity_model('default')

    # Default should be OffsetVelocityModel
    assert isinstance(model, OffsetVelocityModel)


def test_build_velocity_model_invalid():
    """Test factory function with invalid name."""
    with pytest.raises(ValueError, match="not a registered velocity model"):
        build_velocity_model('nonexistent')


def test_velocity_model_types_registry():
    """Test that model registry contains expected models."""
    assert 'centered' in VELOCITY_MODEL_TYPES
    assert 'offset' in VELOCITY_MODEL_TYPES
    assert 'default' in VELOCITY_MODEL_TYPES


# Plotting tests
def test_plot_centered_velocity_map(centered_theta, output_dir):
    """Test plotting centered velocity map."""
    model = CenteredVelocityModel()

    fig, ax = plotting.plot_velocity_map(
        model,
        centered_theta,
        plane='obs',
        rmax=20.0,
        Ngrid=50,
        show=False,
        outfile=output_dir / "centered_velocity_map_obs.png",
    )

    assert fig is not None
    assert ax is not None


def test_plot_offset_velocity_map(offset_theta, output_dir):
    """Test plotting offset velocity map."""
    model = OffsetVelocityModel()

    fig, ax = plotting.plot_velocity_map(
        model,
        offset_theta,
        plane='obs',
        rmax=20.0,
        Ngrid=50,
        show=False,
        outfile=output_dir / "offset_velocity_map_obs.png",
    )

    assert fig is not None
    assert ax is not None


def test_plot_speed_map(centered_theta, output_dir):
    """Test plotting speed map instead of velocity."""
    model = CenteredVelocityModel()

    fig, ax = plotting.plot_velocity_map(
        model,
        centered_theta,
        plane='disk',
        rmax=20.0,
        Ngrid=50,
        speed=True,
        show=False,
        outfile=output_dir / "centered_speed_map_disk.png",
    )

    assert fig is not None
    assert ax is not None


def test_plot_all_planes_centered(centered_theta, output_dir):
    """Test plotting velocity in all planes for centered model."""
    model = CenteredVelocityModel()

    fig, axes = plotting.plot_all_planes(
        model,
        centered_theta,
        rmax=25.0,
        Ngrid=40,
        show=False,
        outfile=output_dir / "centered_all_planes.png",
    )

    assert fig is not None
    assert axes is not None


def test_plot_all_planes_offset(offset_theta, output_dir):
    """Test plotting velocity in all planes for offset model."""
    model = OffsetVelocityModel()

    fig, axes = plotting.plot_all_planes(
        model,
        offset_theta,
        rmax=25.0,
        Ngrid=40,
        show=False,
        outfile=output_dir / "offset_all_planes.png",
    )

    assert fig is not None
    assert axes is not None


def test_plot_with_image_pars(offset_theta, test_image_pars, output_dir):
    """Test plotting with explicit ImagePars."""
    model = OffsetVelocityModel()

    fig, ax = plotting.plot_velocity_map(
        model,
        offset_theta,
        image_pars=test_image_pars,
        plane='source',
        show=False,
        outfile=output_dir / "offset_with_image_pars.png",
    )

    assert fig is not None
    assert ax is not None


def test_plot_rotation_curve(offset_theta, output_dir):
    """Test rotation curve extraction and plotting."""
    model = OffsetVelocityModel()

    # Create ImagePars for rotation curve
    image_pars = ImagePars(shape=(100, 100), pixel_scale=0.3, indexing='ij')

    fig, axes, r, vrot, verr = plotting.plot_rotation_curve(
        model,
        offset_theta,
        image_pars=image_pars,
        plane='obs',
        threshold_dist=3.0,
        Nrbins=15,
        show=False,
        outfile=output_dir / "offset_rotation_curve.png",
    )

    assert fig is not None
    assert len(axes) == 2
    assert len(r) == 15
    assert len(vrot) == 15
    assert len(verr) == 15


def test_plot_multiple_planes(centered_theta, output_dir):
    """Test plotting in multiple individual planes."""
    model = CenteredVelocityModel()
    planes = ['disk', 'gal', 'source', 'obs']

    for plane in planes:
        fig, ax = plotting.plot_velocity_map(
            model,
            centered_theta,
            plane=plane,
            rmax=15.0,
            Ngrid=40,
            show=False,
            outfile=output_dir / f"centered_{plane}_plane.png",
        )
        assert fig is not None


def test_plot_high_inclination(output_dir):
    """Test plotting with high inclination (edge-on)."""
    model = OffsetVelocityModel()

    # High inclination: cosi ~ 1 (nearly edge-on)
    theta = jnp.array([10.0, 200.0, 5.0, 0.95, 0.785, 0.0, 0.0, 0.0, 0.0])

    fig, ax = plotting.plot_velocity_map(
        model,
        theta,
        plane='obs',
        rmax=20.0,
        Ngrid=50,
        show=False,
        outfile=output_dir / "offset_high_inclination.png",
    )

    assert fig is not None


def test_plot_low_inclination(output_dir):
    """Test plotting with low inclination (face-on)."""
    model = CenteredVelocityModel()

    # Low inclination: cosi ~ 0 (nearly face-on)
    theta = jnp.array([10.0, 200.0, 5.0, 0.1, 0.785, 0.0, 0.0])

    fig, ax = plotting.plot_velocity_map(
        model,
        theta,
        plane='obs',
        rmax=20.0,
        Ngrid=50,
        show=False,
        outfile=output_dir / "centered_low_inclination.png",
    )

    assert fig is not None


def test_plot_with_shear(output_dir):
    """Test plotting with lensing shear."""
    model = OffsetVelocityModel()

    # Add significant shear
    theta = jnp.array([10.0, 200.0, 5.0, 0.6, 0.785, 0.15, 0.10, 1.0, -1.0])

    fig, ax = plotting.plot_velocity_map(
        model,
        theta,
        plane='obs',
        rmax=20.0,
        Ngrid=50,
        show=False,
        outfile=output_dir / "offset_with_shear.png",
    )

    assert fig is not None


# Edge case tests
def test_zero_circular_velocity():
    """Test model with zero circular velocity."""
    model = CenteredVelocityModel()
    theta = jnp.array([0.6, 0.785, 0.05, -0.03, 10.0, 0.0, 5.0])

    X = jnp.linspace(-5, 5, 10)
    Y = jnp.linspace(-5, 5, 10)
    X_grid, Y_grid = jnp.meshgrid(X, Y, indexing='ij')

    v_map = model(theta, 'obs', X_grid, Y_grid)

    # Should just be systemic velocity everywhere
    assert jnp.allclose(v_map, 10.0)


def test_single_point_evaluation(offset_theta):
    """Test evaluation at a single point."""
    model = OffsetVelocityModel()

    X = jnp.array([[5.0]])
    Y = jnp.array([[3.0]])

    v_map = model(offset_theta, 'obs', X, Y)

    assert v_map.shape == (1, 1)
    assert jnp.isfinite(v_map[0, 0])


# ----------------------------------------------------------------------
# Render method tests


def test_render_image_centered(centered_theta, test_image_pars):
    """Test render_image method for centered model."""
    model = CenteredVelocityModel()

    image = model.render_image(centered_theta, test_image_pars, plane='obs')

    assert image.shape == test_image_pars.shape
    assert jnp.isfinite(image).all()
    assert jnp.abs(jnp.mean(image) - 10.0) < 50.0  # Near v0


def test_render_image_offset(offset_theta, test_image_pars):
    """Test render_image method for offset model."""
    model = OffsetVelocityModel()

    image = model.render_image(offset_theta, test_image_pars, plane='obs')

    assert image.shape == test_image_pars.shape
    assert jnp.isfinite(image).all()


def test_render_image_speed(centered_theta, test_image_pars):
    """Test render_image with return_speed=True."""
    model = CenteredVelocityModel()

    speed_map = model.render_image(
        centered_theta, test_image_pars, plane='disk', return_speed=True
    )
    velocity_map = model.render_image(
        centered_theta, test_image_pars, plane='disk', return_speed=False
    )

    assert speed_map.shape == test_image_pars.shape
    assert velocity_map.shape == test_image_pars.shape

    # Speed should be non-negative
    assert jnp.all(speed_map >= 0)

    # In disk plane with return_speed=False, should be mostly v0
    assert jnp.abs(jnp.mean(velocity_map) - 10.0) < 5.0


def test_render_dispatcher_image(offset_theta, test_image_pars):
    """Test render() dispatcher with data_type='image'."""
    model = OffsetVelocityModel()

    image = model.render(offset_theta, 'image', test_image_pars, plane='obs')

    assert image.shape == test_image_pars.shape
    assert jnp.isfinite(image).all()


def test_render_dispatcher_cube(offset_theta, test_image_pars):
    """Test render() dispatcher with data_type='cube' (should raise NotImplementedError)."""
    model = OffsetVelocityModel()

    with pytest.raises(NotImplementedError, match="Cube rendering"):
        model.render(offset_theta, 'cube', test_image_pars)


def test_render_dispatcher_slit(offset_theta, test_image_pars):
    """Test render() dispatcher with data_type='slit' (should raise NotImplementedError)."""
    model = OffsetVelocityModel()

    with pytest.raises(NotImplementedError, match="Slit rendering"):
        model.render(offset_theta, 'slit', test_image_pars)


def test_render_dispatcher_grism(offset_theta, test_image_pars):
    """Test render() dispatcher with data_type='grism' (should raise NotImplementedError)."""
    model = OffsetVelocityModel()

    with pytest.raises(NotImplementedError, match="Grism rendering"):
        model.render(offset_theta, 'grism', test_image_pars)


def test_render_dispatcher_invalid_type(offset_theta, test_image_pars):
    """Test render() dispatcher with invalid data_type."""
    model = OffsetVelocityModel()

    with pytest.raises(ValueError, match="Unknown data_type"):
        model.render(offset_theta, 'invalid', test_image_pars)


def test_render_dispatcher_wrong_pars_type(offset_theta):
    """Test render() dispatcher with wrong data_pars type for 'image'."""
    model = OffsetVelocityModel()

    with pytest.raises(TypeError, match="must be ImagePars"):
        model.render(offset_theta, 'image', {'not': 'ImagePars'})


def test_render_image_different_planes(centered_theta, test_image_pars):
    """Test render_image in different planes."""
    model = CenteredVelocityModel()
    planes = ['disk', 'gal', 'source', 'obs']

    for plane in planes:
        image = model.render_image(centered_theta, test_image_pars, plane=plane)
        assert image.shape == test_image_pars.shape
        assert jnp.isfinite(image).all()


def test_render_image_vs_call_consistency(offset_theta, test_image_pars):
    """Test that render_image produces same result as manual __call__."""
    from kl_pipe.utils import build_map_grid_from_image_pars

    model = OffsetVelocityModel()

    # Method 1: render_image
    image1 = model.render_image(offset_theta, test_image_pars, plane='obs')

    # Method 2: manual __call__
    X, Y = build_map_grid_from_image_pars(test_image_pars, unit='arcsec', centered=True)
    image2 = model(offset_theta, 'obs', X, Y)

    assert jnp.allclose(image1, image2)


def test_return_speed_parameter(centered_theta):
    """Test return_speed parameter in __call__ method."""
    model = CenteredVelocityModel()

    X = jnp.linspace(-10, 10, 20)
    Y = jnp.linspace(-10, 10, 20)
    X_grid, Y_grid = jnp.meshgrid(X, Y, indexing='ij')

    # Get velocity (default)
    v_map = model(centered_theta, 'obs', X_grid, Y_grid, return_speed=False)

    # Get speed
    speed_map = model(centered_theta, 'obs', X_grid, Y_grid, return_speed=True)

    # Speed should be non-negative
    assert jnp.all(speed_map >= 0)

    # Velocity can be negative (redshifted/blueshifted)
    assert not jnp.all(v_map >= 0)

    # They should be different
    assert not jnp.allclose(v_map, speed_map)


def test_disk_plane_velocity_vs_speed(centered_theta):
    """Test that disk plane returns ~v0 for velocity but non-zero speed."""
    model = CenteredVelocityModel()

    X = jnp.linspace(-10, 10, 20)
    Y = jnp.linspace(-10, 10, 20)
    X_grid, Y_grid = jnp.meshgrid(X, Y, indexing='ij')

    # Velocity in disk plane should be ~v0 (no line-of-sight component)
    v_map = model(centered_theta, 'disk', X_grid, Y_grid, return_speed=False)

    # Speed should be non-zero at non-zero radii
    speed_map = model(centered_theta, 'disk', X_grid, Y_grid, return_speed=True)

    # Velocity should be mostly v0
    assert jnp.abs(jnp.mean(v_map) - 10.0) < 2.0

    # Speed should have significant variation
    assert speed_map.max() > 50.0  # Should have high speeds at large radii
    assert jnp.all(speed_map >= 0)


# Plotting tests with new render methods
def test_plot_rendered_image(offset_theta, test_image_pars, output_dir):
    """Test plotting a rendered image."""

    model = OffsetVelocityModel()

    # Render image
    image = model.render_image(offset_theta, test_image_pars, plane='obs')

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(image.T, origin='lower', cmap='RdBu_r')
    plt.colorbar(im, ax=ax, label='km/s')
    ax.set_title('Rendered Velocity Image')
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')

    plt.savefig(
        output_dir / "rendered_velocity_image.png", dpi=150, bbox_inches='tight'
    )
    plt.close()

    assert fig is not None


def test_plot_speed_vs_velocity_comparison(centered_theta, test_image_pars, output_dir):
    """Test plotting speed vs velocity side-by-side."""

    model = CenteredVelocityModel()

    # Render both
    velocity_map = model.render_image(
        centered_theta, test_image_pars, plane='obs', return_speed=False
    )
    speed_map = model.render_image(
        centered_theta, test_image_pars, plane='obs', return_speed=True
    )

    # Plot side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(velocity_map.T, origin='lower', cmap='RdBu_r')
    plt.colorbar(im1, ax=ax1, label='km/s')
    ax1.set_title('Line-of-sight Velocity')

    im2 = ax2.imshow(speed_map.T, origin='lower', cmap='viridis')
    plt.colorbar(im2, ax=ax2, label='km/s')
    ax2.set_title('Circular Speed')

    plt.tight_layout()
    plt.savefig(
        output_dir / "speed_vs_velocity_comparison.png", dpi=150, bbox_inches='tight'
    )
    plt.close()

    assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
