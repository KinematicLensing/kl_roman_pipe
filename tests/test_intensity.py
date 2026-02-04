"""
Unit tests for intensity models.

Tests include:
- Model instantiation
- Parameter conversion (theta <-> pars)
- Model evaluation in different planes
- Coordinate transformations
- Sersic profile properties
"""

import pytest
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from kl_pipe.intensity import InclinedExponentialModel, build_intensity_model
from kl_pipe.parameters import ImagePars
from kl_pipe.utils import get_test_dir, build_map_grid_from_image_pars
from kl_pipe import plotting


# ==============================================================================
# Pytest Fixtures
# ==============================================================================


@pytest.fixture
def output_dir():
    """Create and return output directory for test plots."""
    out_dir = get_test_dir() / "out" / "intensity"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture
def basic_meta_pars():
    """Basic metadata for model instantiation."""
    return {'test_param': 'test_value'}


@pytest.fixture
def exponential_theta():
    """Standard theta array for InclinedExponentialModel."""
    # (cosi, theta_int, g1, g2, flux, int_rscale, int_x0, int_y0)
    return jnp.array([0.7, 0.785, 0.02, -0.01, 1.0, 3.0, 0.0, 0.0])


@pytest.fixture
def exponential_theta_offset():
    """Theta with non-zero centroid offset."""
    # (cosi, theta_int, g1, g2, flux, int_rscale, int_x0, int_y0)
    return jnp.array([0.7, 0.785, 0.02, -0.01, 1.0, 3.0, 2.0, -1.5])


@pytest.fixture
def test_image_pars():
    """ImagePars for testing."""
    return ImagePars(shape=(51, 51), pixel_scale=0.1, indexing='xy')


# ==============================================================================
# Basic Instantiation Tests
# ==============================================================================


def test_exponential_model_instantiation(basic_meta_pars):
    """Test InclinedExponentialModel can be instantiated."""
    model = InclinedExponentialModel(meta_pars=basic_meta_pars)
    assert model is not None
    assert model.name == 'inclined_exp'
    assert len(model.PARAMETER_NAMES) == 8


def test_model_parameter_names():
    """Test that parameter names are correctly defined."""
    model = InclinedExponentialModel()

    assert 'flux' in model.PARAMETER_NAMES
    assert 'int_rscale' in model.PARAMETER_NAMES
    assert 'cosi' in model.PARAMETER_NAMES
    assert 'theta_int' in model.PARAMETER_NAMES
    assert 'int_x0' in model.PARAMETER_NAMES
    assert 'int_y0' in model.PARAMETER_NAMES


# ==============================================================================
# Parameter Conversion Tests
# ==============================================================================


def test_exponential_theta2pars(exponential_theta):
    """Test theta to pars conversion."""
    pars = InclinedExponentialModel.theta2pars(exponential_theta)

    assert isinstance(pars, dict)
    assert len(pars) == 8
    assert 'flux' in pars
    assert 'int_rscale' in pars
    assert pars['flux'] == 1.0
    assert pars['int_rscale'] == 3.0


def test_exponential_pars2theta():
    """Test pars to theta conversion."""
    pars = {
        'cosi': 0.7,
        'theta_int': 0.785,
        'g1': 0.02,
        'g2': -0.01,
        'flux': 1.0,
        'int_rscale': 3.0,
        'int_x0': 0.0,
        'int_y0': 0.0,
    }
    theta = InclinedExponentialModel.pars2theta(pars)

    assert isinstance(theta, jnp.ndarray)
    assert len(theta) == 8
    assert float(theta[4]) == 1.0  # flux
    assert float(theta[5]) == 3.0  # int_rscale


def test_roundtrip_conversion(exponential_theta):
    """Test theta -> pars -> theta roundtrip."""
    pars = InclinedExponentialModel.theta2pars(exponential_theta)
    theta_reconstructed = InclinedExponentialModel.pars2theta(pars)

    assert jnp.allclose(exponential_theta, theta_reconstructed)


# ==============================================================================
# Parameter Extraction Tests
# ==============================================================================


def test_get_param(exponential_theta):
    """Test get_param method."""
    model = InclinedExponentialModel()

    flux = model.get_param('flux', exponential_theta)
    rscale = model.get_param('int_rscale', exponential_theta)

    assert float(flux) == 1.0
    assert float(rscale) == 3.0


def test_get_param_offset(exponential_theta_offset):
    """Test get_param for centroid offsets."""
    model = InclinedExponentialModel()

    x0 = model.get_param('int_x0', exponential_theta_offset)
    y0 = model.get_param('int_y0', exponential_theta_offset)

    assert float(x0) == 2.0
    assert float(y0) == -1.5


# ==============================================================================
# Model Evaluation Tests
# ==============================================================================


def test_evaluate_in_disk_plane(exponential_theta, test_image_pars):
    """Test disk plane evaluation returns proper Sersic profile."""
    model = InclinedExponentialModel()

    X_grid, Y_grid = build_map_grid_from_image_pars(
        test_image_pars, unit='arcsec', centered=True
    )

    # Evaluate in disk plane (face-on, no projection effects)
    intensity_disk = model.evaluate_in_disk_plane(exponential_theta, X_grid, Y_grid)

    assert intensity_disk.shape == X_grid.shape
    assert jnp.all(intensity_disk >= 0)  # Surface brightness non-negative
    assert jnp.isfinite(intensity_disk).all()

    # Check that intensity decreases with radius
    r = jnp.sqrt(X_grid**2 + Y_grid**2)
    center_intensity = intensity_disk[25, 25]  # Center pixel
    edge_intensity = intensity_disk[0, 0]  # Corner pixel
    assert center_intensity > edge_intensity


def test_intensity_map_evaluation(exponential_theta, test_image_pars):
    """Test full intensity map evaluation in obs plane."""
    model = InclinedExponentialModel()

    X_grid, Y_grid = build_map_grid_from_image_pars(
        test_image_pars, unit='arcsec', centered=True
    )

    # Evaluate in observer frame
    intensity_obs = model(exponential_theta, 'obs', X_grid, Y_grid)

    assert intensity_obs.shape == X_grid.shape
    assert jnp.all(intensity_obs >= 0)
    assert jnp.isfinite(intensity_obs).all()


def test_intensity_with_offset(exponential_theta_offset, test_image_pars):
    """Test that centroid offset shifts the profile correctly."""
    model = InclinedExponentialModel()

    X_grid, Y_grid = build_map_grid_from_image_pars(
        test_image_pars, unit='arcsec', centered=True
    )

    intensity = model(exponential_theta_offset, 'obs', X_grid, Y_grid)

    # Peak should be shifted from grid center
    # Grid center is at index (25, 25)
    # With x0=2.0, y0=-1.5, peak should be shifted
    peak_idx = jnp.unravel_index(jnp.argmax(intensity), intensity.shape)

    # Peak should not be at grid center
    assert peak_idx != (25, 25)

    # Check approximate shift direction (x0=2.0 means shift right)
    import ipdb; ipdb.set_trace()
    assert peak_idx[0] > 25  # Shifted in +x direction


# ==============================================================================
# Physical Property Tests
# ==============================================================================


def test_inclination_effect(test_image_pars):
    """Test that inclination affects surface brightness correctly.

    **What this test validates:**
    - In the 'disk' plane (face-on view), flux is conserved regardless of cosi parameter
    - When projecting to 'obs' plane, peak surface brightness increases by 1/cos(i)

    **What this test does NOT validate:**
    - Flux conservation on the observed grid (requires pixel convolution - see issue #5)
    - The current implementation evaluates models at discrete grid points (point sampling)
      rather than integrating over pixels. Proper flux conservation in the observed plane
      requires PSF/pixel response convolution, which is planned future work.

    **Physics:**
    - An inclined disk has the same intrinsic flux but compressed projected area
    - Surface brightness I_obs = I_disk / cos(i) to conserve flux over smaller area
    - Without pixel convolution, discrete sampling doesn't capture this properly
    """
    model = InclinedExponentialModel()

    X_grid, Y_grid = build_map_grid_from_image_pars(
        test_image_pars, unit='arcsec', centered=True
    )

    # Face-on (cosi=1.0, i=0)
    theta_faceon = jnp.array([1.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0])
    # Inclined (cosi=0.6, i~53 deg)
    theta_inclined = jnp.array([0.6, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0])

    # Test 1: Flux conservation in disk plane (intrinsic, before projection)
    # Both models should have identical flux in the disk plane since cosi
    # only affects the projection to obs plane
    intensity_faceon_disk = model(theta_faceon, 'disk', X_grid, Y_grid)
    intensity_inclined_disk = model(theta_inclined, 'disk', X_grid, Y_grid)

    flux_faceon_disk = jnp.sum(intensity_faceon_disk)
    flux_inclined_disk = jnp.sum(intensity_inclined_disk)

    rel_diff_disk = jnp.abs(flux_faceon_disk - flux_inclined_disk) / flux_faceon_disk
    assert (
        rel_diff_disk < 0.01
    ), f"Flux not conserved in disk plane: {rel_diff_disk:.1%} difference"

    # Test 2: Peak surface brightness increases by 1/cos(i) when projecting to obs plane
    # This tests the I_obs = I_disk / cos(i) scaling
    intensity_faceon_obs = model(theta_faceon, 'obs', X_grid, Y_grid)
    intensity_inclined_obs = model(theta_inclined, 'obs', X_grid, Y_grid)

    peak_faceon_obs = jnp.max(intensity_faceon_obs)
    peak_inclined_obs = jnp.max(intensity_inclined_obs)

    expected_ratio = 1.0 / 0.6  # 1/cosi
    actual_ratio = peak_inclined_obs / peak_faceon_obs

    assert jnp.isclose(
        actual_ratio, expected_ratio, rtol=0.05
    ), f"Peak SB ratio incorrect: expected {expected_ratio:.3f}, got {actual_ratio:.3f}"


def test_shear_effect(test_image_pars):
    """Test that shear distorts the profile."""
    model = InclinedExponentialModel()

    X_grid, Y_grid = build_map_grid_from_image_pars(
        test_image_pars, unit='arcsec', centered=True
    )

    # No shear
    theta_no_shear = jnp.array([0.7, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0])
    intensity_no_shear = model(theta_no_shear, 'obs', X_grid, Y_grid)

    # With shear
    theta_shear = jnp.array([0.7, 0.0, 0.05, 0.0, 1.0, 3.0, 0.0, 0.0])
    intensity_shear = model(theta_shear, 'obs', X_grid, Y_grid)

    # Images should be different
    assert not jnp.allclose(intensity_no_shear, intensity_shear)

    # But total flux should still be approximately conserved
    flux_no_shear = jnp.sum(intensity_no_shear)
    flux_shear = jnp.sum(intensity_shear)
    assert jnp.abs(flux_no_shear - flux_shear) / flux_no_shear < 0.05


# ==============================================================================
# Plotting Tests
# ==============================================================================


@pytest.mark.parametrize("inclination", [0.0, 30.0, 60.0])
def test_plot_exponential_at_inclinations(inclination, output_dir, test_image_pars):
    """Generate diagnostic plots at different inclinations."""
    model = InclinedExponentialModel()

    # Create test grid
    X, Y = build_map_grid_from_image_pars(test_image_pars, unit='arcsec', centered=True)

    # Set parameters with varying inclination
    cosi = np.cos(np.deg2rad(inclination))
    theta = jnp.array([cosi, 0.785, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0])

    # Evaluate
    intensity = model(theta, 'obs', X, Y)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        np.array(intensity).T,
        origin='lower',
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        cmap='viridis',
    )
    ax.set_xlabel('X (arcsec)')
    ax.set_ylabel('Y (arcsec)')
    ax.set_title(f'Exponential Disk, i={inclination:.0f}°')
    plt.colorbar(im, ax=ax, label='Surface Brightness')

    outfile = output_dir / f"exponential_i{inclination:.0f}.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()


def test_plot_high_inclination(output_dir):
    """Test visualization at high inclination (edge-on)."""
    model = InclinedExponentialModel()

    image_pars = ImagePars(shape=(80, 80), pixel_scale=0.1, indexing='xy')
    X, Y = build_map_grid_from_image_pars(image_pars, unit='arcsec', centered=True)

    # Nearly edge-on (i=85 deg)
    cosi = np.cos(np.deg2rad(85))
    theta = jnp.array([cosi, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0])

    intensity = model(theta, 'obs', X, Y)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        np.array(intensity).T,
        origin='lower',
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        cmap='viridis',
        aspect='auto',
    )
    ax.set_xlabel('X (arcsec)')
    ax.set_ylabel('Y (arcsec)')
    ax.set_title('Nearly Edge-On Disk (i=85°)')
    plt.colorbar(im, ax=ax, label='Surface Brightness')

    outfile = output_dir / "exponential_edge_on.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
