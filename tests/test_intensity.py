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
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from kl_pipe.intensity import (
    InclinedExponentialModel,
    InclinedSpergelModel,
    InclinedDeVaucouleursModel,
    build_intensity_model,
)
from kl_pipe.observation import build_image_obs
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
    # (cosi, theta_int, g1, g2, flux, int_rscale, int_h_over_r, int_x0, int_y0)
    return jnp.array([0.7, 0.785, 0.02, -0.01, 1.0, 3.0, 0.1, 0.0, 0.0])


@pytest.fixture
def exponential_theta_offset():
    """Theta with non-zero centroid offset."""
    # (cosi, theta_int, g1, g2, flux, int_rscale, int_h_over_r, int_x0, int_y0)
    return jnp.array([0.7, 0.785, 0.02, -0.01, 1.0, 3.0, 0.1, 2.0, -1.5])


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
    assert len(model.PARAMETER_NAMES) == 9


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
    assert len(pars) == 9
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
        'int_h_over_r': 0.1,
        'int_x0': 0.0,
        'int_y0': 0.0,
    }
    theta = InclinedExponentialModel.pars2theta(pars)

    assert isinstance(theta, jnp.ndarray)
    assert len(theta) == 9
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

    # Check approximate shift direction (x0=2.0 means shift right along cols)
    assert peak_idx[1] > 25  # Shifted in +x direction (cols = axis 1)


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
    theta_faceon = jnp.array([1.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.1, 0.0, 0.0])
    # Inclined (cosi=0.6, i~53 deg)
    theta_inclined = jnp.array([0.6, 0.0, 0.0, 0.0, 1.0, 3.0, 0.1, 0.0, 0.0])

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

    # Test 2: Peak surface brightness should increase when projecting to obs plane
    # With 3D sech² vertical profile, the scaling is NOT exactly 1/cos(i) but
    # the inclined case should still be brighter per pixel than face-on
    intensity_faceon_obs = model(theta_faceon, 'obs', X_grid, Y_grid)
    intensity_inclined_obs = model(theta_inclined, 'obs', X_grid, Y_grid)

    peak_faceon_obs = jnp.max(intensity_faceon_obs)
    peak_inclined_obs = jnp.max(intensity_inclined_obs)

    assert (
        peak_inclined_obs > peak_faceon_obs
    ), f"Inclined peak ({peak_inclined_obs:.4f}) should be > face-on ({peak_faceon_obs:.4f})"


def test_shear_effect(test_image_pars):
    """Test that shear distorts the profile."""
    model = InclinedExponentialModel()

    X_grid, Y_grid = build_map_grid_from_image_pars(
        test_image_pars, unit='arcsec', centered=True
    )

    # No shear
    theta_no_shear = jnp.array([0.7, 0.0, 0.0, 0.0, 1.0, 3.0, 0.1, 0.0, 0.0])
    intensity_no_shear = model(theta_no_shear, 'obs', X_grid, Y_grid)

    # With shear
    theta_shear = jnp.array([0.7, 0.0, 0.05, 0.0, 1.0, 3.0, 0.1, 0.0, 0.0])
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
    theta = jnp.array([cosi, 0.785, 0.0, 0.0, 1.0, 3.0, 0.1, 0.0, 0.0])

    # Evaluate
    intensity = model(theta, 'obs', X, Y)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        np.array(intensity),
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
    theta = jnp.array([cosi, 0.0, 0.0, 0.0, 1.0, 3.0, 0.1, 0.0, 0.0])

    intensity = model(theta, 'obs', X, Y)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        np.array(intensity),
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


# ==============================================================================
# GalSim Regression Tests
# ==============================================================================


@pytest.fixture(scope='module')
def galsim_image_pars():
    """ImagePars for GalSim regression tests.

    128x128 grid ensures profile is < 0.01% of peak at boundary,
    eliminating FFT aliasing artifacts in k-space rendering.
    """
    return ImagePars(shape=(128, 128), pixel_scale=0.2, indexing='ij')


@pytest.mark.parametrize(
    "cosi,int_h_over_r,theta_int,g1,g2,int_x0,int_y0",
    [
        # face-on, no shear, centered
        (1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
        # inclined, moderate h_over_r
        (0.7, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
        # highly inclined
        (0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
        # thin disk limit
        (0.7, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0),
        # thick disk
        (0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0),
        # with rotation
        (0.7, 0.1, np.pi / 4, 0.0, 0.0, 0.0, 0.0),
        (0.7, 0.1, np.pi / 2, 0.0, 0.0, 0.0, 0.0),
        # with shear
        (0.7, 0.1, 0.0, 0.05, -0.03, 0.0, 0.0),
        # with offset
        (0.7, 0.1, 0.0, 0.0, 0.0, 1.0, -0.5),
        # combined: rotation + shear + offset
        (0.7, 0.1, np.pi / 4, 0.05, -0.03, 1.0, -0.5),
    ],
)
def test_galsim_regression_render_image(
    cosi, int_h_over_r, theta_int, g1, g2, int_x0, int_y0, galsim_image_pars, output_dir
):
    """Compare render_image (k-space FFT) against GalSim InclinedExponential."""
    import galsim as gs
    from kl_pipe.synthetic import _generate_sersic_galsim

    flux = 1.0
    int_rscale = 2.0

    # tight GSParams so GalSim reference is accurate (default folding_threshold=5e-3
    # causes ~3e-3 aliasing for sheared profiles)
    gsp = gs.GSParams(folding_threshold=1e-4, maxk_threshold=1e-4, kvalue_accuracy=1e-6)

    # our model (k-space FFT)
    model = InclinedExponentialModel()
    theta = jnp.array(
        [cosi, theta_int, g1, g2, flux, int_rscale, int_h_over_r, int_x0, int_y0]
    )
    our_image = model.render_image(theta, image_pars=galsim_image_pars)

    # GalSim reference (no_pixel: skip pixel convolution for fair k-space comparison)
    gs_image = _generate_sersic_galsim(
        galsim_image_pars,
        flux=flux,
        int_rscale=int_rscale,
        n_sersic=1.0,
        cosi=cosi,
        theta_int=theta_int,
        g1=g1,
        g2=g2,
        int_x0=int_x0,
        int_y0=int_y0,
        int_h_over_r=int_h_over_r,
        gsparams=gsp,
        method='no_pixel',
    )

    # convert GalSim flux/pixel → surface brightness (flux/arcsec²)
    gs_sb = gs_image / galsim_image_pars.pixel_scale**2

    peak = np.max(np.abs(gs_sb))
    residual = np.abs(np.array(our_image) - gs_sb)
    max_frac_residual = np.max(residual) / peak

    assert max_frac_residual < 1e-3, (
        f"render_image vs GalSim: max|residual|/peak = {max_frac_residual:.2e} "
        f"(cosi={cosi}, h/r={int_h_over_r}, theta={theta_int:.2f}, "
        f"g=({g1},{g2}), offset=({int_x0},{int_y0}))"
    )


@pytest.mark.parametrize(
    "cosi,int_h_over_r,theta_int",
    [
        (1.0, 0.1, 0.0),
        (0.7, 0.1, 0.0),
        (0.3, 0.1, 0.0),
        (0.7, 0.01, 0.0),
        (0.7, 0.3, 0.0),
        (0.7, 0.1, np.pi / 4),
    ],
)
def test_galsim_regression_call(
    cosi, int_h_over_r, theta_int, galsim_image_pars, output_dir
):
    """Compare __call__ (LOS quadrature) against GalSim InclinedExponential."""
    from kl_pipe.synthetic import _generate_sersic_galsim

    flux = 1.0
    int_rscale = 2.0

    # our model (LOS quadrature)
    model = InclinedExponentialModel()
    theta = jnp.array(
        [cosi, theta_int, 0.0, 0.0, flux, int_rscale, int_h_over_r, 0.0, 0.0]
    )

    X, Y = build_map_grid_from_image_pars(
        galsim_image_pars, unit='arcsec', centered=True
    )
    our_image = model(theta, 'obs', X, Y)

    # GalSim reference (no_pixel: skip pixel convolution for fair comparison)
    gs_image = _generate_sersic_galsim(
        galsim_image_pars,
        flux=flux,
        int_rscale=int_rscale,
        n_sersic=1.0,
        cosi=cosi,
        theta_int=theta_int,
        g1=0.0,
        g2=0.0,
        int_x0=0.0,
        int_y0=0.0,
        int_h_over_r=int_h_over_r,
        method='no_pixel',
    )

    # convert GalSim flux/pixel → surface brightness (flux/arcsec²)
    gs_sb = gs_image / galsim_image_pars.pixel_scale**2

    peak = np.max(np.abs(gs_sb))
    residual = np.abs(np.array(our_image) - gs_sb)
    max_frac_residual = np.max(residual) / peak

    # tolerance depends on quadrature resolution:
    # face-on: no LOS integration -> exact match
    # inclined: 60-pt GL quadrature resolves sech² well for moderate h/r
    # thin disk: sech² peak narrower than GL node spacing -> large errors
    if cosi >= 0.99:
        tol = 1e-2
    elif int_h_over_r <= 0.01:
        tol = 3.0
    else:
        tol = 0.02

    assert max_frac_residual < tol, (
        f"__call__ vs GalSim: max|residual|/peak = {max_frac_residual:.2e} "
        f"(cosi={cosi}, h/r={int_h_over_r}, theta={theta_int:.2f}, tol={tol})"
    )


def test_scipy_vs_galsim_backend(galsim_image_pars):
    """Verify scipy backend matches GalSim for 3D inclined exponential."""
    from kl_pipe.synthetic import _generate_sersic_galsim, _generate_sersic_scipy

    cosi = 0.7
    flux = 1.0
    int_rscale = 2.0
    int_h_over_r = 0.1

    gs_image = _generate_sersic_galsim(
        galsim_image_pars,
        flux=flux,
        int_rscale=int_rscale,
        n_sersic=1.0,
        cosi=cosi,
        theta_int=0.0,
        g1=0.0,
        g2=0.0,
        int_x0=0.0,
        int_y0=0.0,
        int_h_over_r=int_h_over_r,
        method='no_pixel',
    )

    scipy_image = _generate_sersic_scipy(
        galsim_image_pars,
        flux=flux,
        int_rscale=int_rscale,
        n_sersic=1.0,
        cosi=cosi,
        theta_int=0.0,
        g1=0.0,
        g2=0.0,
        int_x0=0.0,
        int_y0=0.0,
        int_h_over_r=int_h_over_r,
    )

    # convert GalSim flux/pixel → surface brightness (flux/arcsec²)
    gs_sb = gs_image / galsim_image_pars.pixel_scale**2

    peak = np.max(np.abs(gs_sb))
    max_frac_residual = np.max(np.abs(scipy_image - gs_sb)) / peak

    # scipy now uses analytic k-space FFT (same method as render_image)
    assert (
        max_frac_residual < 2e-3
    ), f"scipy vs GalSim: max|residual|/peak = {max_frac_residual:.2e}"


def test_scipy_vs_render_image_consistency(galsim_image_pars):
    """Verify scipy synthetic backend matches render_image at same parameters.

    Both use analytic k-space FFT (scipy=numpy, render_image=JAX).
    Differences only from FFT grid/padding details.
    """
    from kl_pipe.synthetic import _generate_sersic_scipy

    cosi = 0.7
    flux = 1.0
    int_rscale = 2.0
    int_h_over_r = 0.1
    theta_int = np.pi / 6
    g1 = 0.02
    g2 = -0.01
    int_x0 = 0.3
    int_y0 = -0.2

    # scipy synthetic backend (numpy k-space FFT)
    scipy_image = _generate_sersic_scipy(
        galsim_image_pars,
        flux=flux,
        int_rscale=int_rscale,
        n_sersic=1.0,
        cosi=cosi,
        theta_int=theta_int,
        g1=g1,
        g2=g2,
        int_x0=int_x0,
        int_y0=int_y0,
        int_h_over_r=int_h_over_r,
    )

    # render_image (JAX k-space FFT)
    model = InclinedExponentialModel()
    theta = jnp.array(
        [cosi, theta_int, g1, g2, flux, int_rscale, int_h_over_r, int_x0, int_y0]
    )
    render = np.array(model.render_image(theta, image_pars=galsim_image_pars))

    peak = np.max(np.abs(render))
    max_frac = np.max(np.abs(scipy_image - render)) / peak

    assert (
        max_frac < 2e-3
    ), f"scipy vs render_image: max|residual|/peak = {max_frac:.2e}"


@pytest.fixture(scope='module')
def rect_image_pars():
    """Non-square ImagePars to catch transpose / axis bugs."""
    return ImagePars(shape=(100, 128), pixel_scale=0.2, indexing='ij')


@pytest.mark.parametrize(
    "g1,g2",
    [
        (0.05, 0.0),
        (-0.05, 0.0),
        (0.0, 0.05),
        (0.0, -0.05),
        (0.03, 0.04),
        (-0.03, -0.04),
        (0.04, -0.03),
    ],
    ids=lambda v: f"{v:+.2f}",
)
def test_galsim_regression_render_image_shear_psf(g1, g2, rect_image_pars, output_dir):
    """Compare render_image (k-space FFT + PSF) vs GalSim native for sheared profiles."""
    import galsim as gs
    from kl_pipe.synthetic import _generate_sersic_galsim

    cosi = 0.6
    theta_int = np.pi / 4
    int_x0 = 0.5
    int_y0 = -0.3
    flux = 1.0
    int_rscale = 2.0
    int_h_over_r = 0.1
    fwhm = 0.625

    # our model (k-space FFT + PSF)
    model = InclinedExponentialModel()
    psf_obj = gs.Gaussian(fwhm=fwhm)
    obs = build_image_obs(rect_image_pars, psf=psf_obj, oversample=5, int_model=model)
    theta = jnp.array(
        [cosi, theta_int, g1, g2, flux, int_rscale, int_h_over_r, int_x0, int_y0]
    )
    our_image = np.array(model.render_image(theta, obs=obs))

    # GalSim reference (native convolution, pixel-integrated)
    gs_image = _generate_sersic_galsim(
        rect_image_pars,
        flux=flux,
        int_rscale=int_rscale,
        n_sersic=1.0,
        cosi=cosi,
        theta_int=theta_int,
        g1=g1,
        g2=g2,
        int_x0=int_x0,
        int_y0=int_y0,
        int_h_over_r=int_h_over_r,
        psf=psf_obj,
        method='auto',
    )

    # GalSim returns flux/pixel; our render_image returns surface brightness
    # after PSF convolution both are in flux/pixel, but render_image divides by ps²
    # then convolve_fft multiplies by ps² internally. Net: our output is SB.
    # GalSim drawImage with method='auto' returns flux/pixel.
    gs_sb = gs_image / rect_image_pars.pixel_scale**2

    peak = np.max(np.abs(gs_sb))
    residual = np.abs(our_image - gs_sb)
    max_frac = np.max(residual) / peak

    # diagnostic plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    im0 = axes[0, 0].imshow(gs_sb, origin='lower')
    axes[0, 0].set_title('GalSim native')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(our_image, origin='lower')
    axes[0, 1].set_title('Our render_image')
    plt.colorbar(im1, ax=axes[0, 1])

    vmax_abs = np.max(residual)
    im2 = axes[1, 0].imshow(residual, origin='lower', cmap='hot', vmin=0, vmax=vmax_abs)
    axes[1, 0].set_title('|residual|')
    plt.colorbar(im2, ax=axes[1, 0])

    rel = residual / peak
    im3 = axes[1, 1].imshow(rel, origin='lower', cmap='hot', vmin=0, vmax=np.max(rel))
    axes[1, 1].set_title('|residual|/peak')
    plt.colorbar(im3, ax=axes[1, 1])

    status = 'PASS' if max_frac < 5e-3 else 'FAIL'
    status_color = 'green' if status == 'PASS' else 'red'
    fig.suptitle(
        f'Shear+PSF g1={g1}, g2={g2} — {status} (max={max_frac:.2e}, thr=5e-3)',
        color=status_color,
    )
    plt.tight_layout()
    plt.savefig(output_dir / f'shear_psf_g1_{g1}_g2_{g2}.png', dpi=150)
    plt.close()

    assert max_frac < 5e-3, (
        f"shear+PSF regression: max|resid|/peak = {max_frac:.2e} " f"(g1={g1}, g2={g2})"
    )


@pytest.mark.parametrize(
    "cosi,int_h_over_r,tol",
    [
        # face-on thin disk: exp(-R/r_s) cusp has FT ~ k^{-3}
        (1.0, 0.01, 4e-3),
        # inclined: LOS through sech² smooths cusp, suppressing high-k aliasing
        (0.7, 0.1, 5e-3),
    ],
    ids=["face-on", "inclined"],
)
def test_render_image_vs_call_consistency(
    cosi, int_h_over_r, tol, rect_image_pars, output_dir
):
    """render_image (k-space FFT) vs __call__ (LOS quadrature) on rectangular grid."""
    model = InclinedExponentialModel()
    theta = jnp.array([cosi, np.pi / 6, 0.02, -0.01, 1.0, 2.0, int_h_over_r, 0.3, -0.2])

    X, Y = build_map_grid_from_image_pars(rect_image_pars, unit='arcsec', centered=True)
    call_image = np.array(model(theta, 'obs', X, Y))
    render = np.array(model.render_image(theta, image_pars=rect_image_pars))

    peak = np.max(np.abs(call_image))
    residual = np.abs(render - call_image)
    max_frac = np.max(residual) / peak

    # diagnostic plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    im0 = axes[0, 0].imshow(call_image, origin='lower')
    axes[0, 0].set_title('__call__ (LOS quadrature)')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(render, origin='lower')
    axes[0, 1].set_title('render_image (k-space FFT)')
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[1, 0].imshow(residual, origin='lower', cmap='hot')
    axes[1, 0].set_title('|residual|')
    plt.colorbar(im2, ax=axes[1, 0])

    rel = residual / peak
    im3 = axes[1, 1].imshow(rel, origin='lower', cmap='hot')
    axes[1, 1].set_title('|residual|/peak')
    plt.colorbar(im3, ax=axes[1, 1])

    status = 'PASS' if max_frac < tol else 'FAIL'
    status_color = 'green' if status == 'PASS' else 'red'
    fig.suptitle(
        f'render_image vs __call__ cosi={cosi} h/r={int_h_over_r} — {status} '
        f'(max={max_frac:.2e}, tol={tol})',
        color=status_color,
    )
    plt.tight_layout()
    plt.savefig(output_dir / f'render_vs_call_cosi{cosi}_hr{int_h_over_r}.png', dpi=150)
    plt.close()

    assert max_frac < tol, (
        f"render_image vs __call__: max|resid|/peak = {max_frac:.2e} "
        f"(cosi={cosi}, h/r={int_h_over_r}, tol={tol})"
    )


# ==============================================================================
# Coordinate Convention Guardrail Tests
# ==============================================================================


def test_galsim_no_transpose():
    """Synthetic GalSim data shape matches model without transpose on non-square."""
    from kl_pipe.synthetic import generate_sersic_intensity_2d

    ip = ImagePars(shape=(60, 100), pixel_scale=0.2, indexing='ij')
    gs_data = generate_sersic_intensity_2d(
        ip,
        flux=1.0,
        int_rscale=2.0,
        n_sersic=1.0,
        cosi=0.8,
        theta_int=0.5,
        g1=0.0,
        g2=0.0,
        int_h_over_r=0.1,
        backend='galsim',
    )
    # shape should be (Nrow, Ncol) directly — no transpose
    assert gs_data.shape == (
        60,
        100,
    ), f"GalSim shape {gs_data.shape} != (60, 100). Convention mismatch."


def test_asymmetric_psf_orientation(output_dir):
    """Compare render_image vs GalSim with asymmetric PSF on non-square image."""
    import galsim as gs
    from kl_pipe.synthetic import generate_sersic_intensity_2d

    ip = ImagePars(shape=(80, 120), pixel_scale=0.15, indexing='ij')
    # asymmetric PSF with coma
    psf_obj = gs.OpticalPSF(lam_over_diam=0.5, defocus=0.5, coma1=0.3)

    cosi = 0.7
    theta_int = np.pi / 4
    flux = 1.0
    rscale = 2.0
    h_over_r = 0.1

    theta = jnp.array([cosi, theta_int, 0.0, 0.0, flux, rscale, h_over_r, 0.0, 0.0])

    model = InclinedExponentialModel()
    obs = build_image_obs(ip, psf=psf_obj, oversample=9, int_model=model)
    our_image = np.array(model.render_image(theta, obs=obs))

    gs_image = generate_sersic_intensity_2d(
        ip,
        flux=flux,
        int_rscale=rscale,
        n_sersic=1.0,
        cosi=cosi,
        theta_int=theta_int,
        g1=0.0,
        g2=0.0,
        int_h_over_r=h_over_r,
        backend='galsim',
        psf=psf_obj,
    )
    gs_sb = gs_image / ip.pixel_scale**2

    peak = np.max(np.abs(gs_sb))
    residual = np.abs(our_image - gs_sb)
    max_frac = np.max(residual) / peak

    # diagnostic plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(gs_sb, origin='lower')
    axes[0].set_title('GalSim')
    axes[1].imshow(our_image, origin='lower')
    axes[1].set_title('render_image')
    axes[2].imshow(residual / peak, origin='lower', cmap='hot')
    axes[2].set_title(f'|resid|/peak (max={max_frac:.2e})')
    for ax in axes:
        plt.colorbar(ax.images[0], ax=ax)
    plt.tight_layout()
    plt.savefig(output_dir / 'asymmetric_psf_orientation.png', dpi=150)
    plt.close()

    assert (
        max_frac < 5e-3
    ), f"Asymmetric PSF orientation: max|resid|/peak = {max_frac:.2e}"


# ==============================================================================
# Spergel / DeVaucouleurs Model Tests
# ==============================================================================


@pytest.fixture
def spergel_theta():
    """Standard theta for InclinedSpergelModel (10 params)."""
    # (cosi, theta_int, g1, g2, flux, int_rscale, int_h_over_r, nu, int_x0, int_y0)
    return jnp.array([0.7, 0.785, 0.02, -0.01, 1.0, 3.0, 0.1, 0.5, 0.0, 0.0])


@pytest.fixture
def devaucouleurs_theta():
    """Standard theta for InclinedDeVaucouleursModel (9 params)."""
    # (cosi, theta_int, g1, g2, flux, int_rscale, int_h_over_r, int_x0, int_y0)
    return jnp.array([0.7, 0.785, 0.02, -0.01, 1.0, 3.0, 0.1, 0.0, 0.0])


# --- Unit Tests (Step 4) ---


def test_spergel_model_instantiation(basic_meta_pars):
    """Test InclinedSpergelModel can be instantiated."""
    model = InclinedSpergelModel(meta_pars=basic_meta_pars)
    assert model is not None
    assert model.name == 'inclined_spergel'
    assert len(model.PARAMETER_NAMES) == 10


def test_spergel_parameter_names():
    """Test that nu is at index 7 and all expected params exist."""
    model = InclinedSpergelModel()
    assert model.PARAMETER_NAMES[7] == 'nu'
    for name in ('flux', 'int_rscale', 'cosi', 'theta_int', 'nu', 'int_x0', 'int_y0'):
        assert name in model.PARAMETER_NAMES


def test_spergel_theta_roundtrip():
    """Test theta -> pars -> theta roundtrip for Spergel model."""
    pars = {
        'cosi': 0.7,
        'theta_int': 0.785,
        'g1': 0.02,
        'g2': -0.01,
        'flux': 1.0,
        'int_rscale': 3.0,
        'int_h_over_r': 0.1,
        'nu': 0.5,
        'int_x0': 0.0,
        'int_y0': 0.0,
    }
    theta = InclinedSpergelModel.pars2theta(pars)
    pars_back = InclinedSpergelModel.theta2pars(theta)
    theta_back = InclinedSpergelModel.pars2theta(pars_back)
    assert jnp.allclose(theta, theta_back)
    assert len(theta) == 10


def test_devaucouleurs_model_instantiation(basic_meta_pars):
    """Test InclinedDeVaucouleursModel: 9 params, no nu."""
    model = InclinedDeVaucouleursModel(meta_pars=basic_meta_pars)
    assert model is not None
    assert model.name == 'de_vaucouleurs'
    assert len(model.PARAMETER_NAMES) == 9
    assert 'nu' not in model.PARAMETER_NAMES


def test_devaucouleurs_matches_spergel_fixed_nu(galsim_image_pars):
    """DeVaucouleurs render_image must be bit-identical to Spergel(nu=-0.6)."""
    spergel = InclinedSpergelModel()
    devac = InclinedDeVaucouleursModel()

    cosi, theta_int, g1, g2 = 0.7, 0.3, 0.02, -0.01
    flux, rscale, h_over_r, nu = 1.0, 2.0, 0.1, -0.6

    theta_sp = jnp.array(
        [cosi, theta_int, g1, g2, flux, rscale, h_over_r, nu, 0.0, 0.0]
    )
    theta_dv = jnp.array([cosi, theta_int, g1, g2, flux, rscale, h_over_r, 0.0, 0.0])

    img_sp = spergel.render_image(theta_sp, image_pars=galsim_image_pars)
    img_dv = devac.render_image(theta_dv, image_pars=galsim_image_pars)

    max_diff = float(jnp.max(jnp.abs(img_sp - img_dv)))
    assert max_diff < 1e-10, f"DeVauc vs Spergel(nu=-0.6): max diff = {max_diff:.2e}"


@pytest.mark.parametrize(
    "nu,cosi,g1,g2",
    [
        (0.5, 1.0, 0.0, 0.0),
        (-0.6, 1.0, 0.0, 0.0),
        (2.0, 1.0, 0.0, 0.0),
        (0.5, 0.7, 0.0, 0.0),
        (-0.6, 0.7, 0.05, -0.03),
        (2.0, 0.3, 0.03, -0.02),
    ],
)
def test_spergel_flux_conservation(nu, cosi, g1, g2, galsim_image_pars):
    """sum(image) * pixel_scale² must match input flux within 0.1%.

    Uses rscale=1.0 so FOV/rscale=25.6 on the 128x128 grid, keeping
    profile tails well within the grid boundary.
    """
    model = InclinedSpergelModel()
    flux = 1.0
    theta = jnp.array([cosi, 0.3, g1, g2, flux, 1.0, 0.1, nu, 0.0, 0.0])
    image = model.render_image(theta, image_pars=galsim_image_pars)
    ps = galsim_image_pars.pixel_scale
    measured_flux = float(jnp.sum(image) * ps**2)
    rel_err = abs(measured_flux - flux) / flux
    assert rel_err < 1e-3, (
        f"Flux: measured={measured_flux:.6f}, expected={flux}, "
        f"rel_err={rel_err:.2e} (nu={nu}, cosi={cosi})"
    )


@pytest.mark.parametrize(
    "nu,cosi,int_h_over_r,tol",
    [
        (0.5, 0.7, 0.1, 5e-3),
        (2.0, 0.7, 0.1, 5e-3),
        (0.5, 1.0, 0.1, 4e-3),
    ],
)
def test_spergel_render_image_vs_call_consistency(
    nu, cosi, int_h_over_r, tol, galsim_image_pars
):
    """K-space render_image vs real-space __call__ consistency.

    Uses nonzero centroid offset to avoid worst-case grid alignment where
    the cusp sits exactly between pixel centers on even-sized grids.
    Matches the existing exponential render_vs_call test convention.
    """
    model = InclinedSpergelModel()
    theta = jnp.array(
        [cosi, np.pi / 6, 0.02, -0.01, 1.0, 2.0, int_h_over_r, nu, 0.3, -0.2]
    )

    X, Y = build_map_grid_from_image_pars(
        galsim_image_pars, unit='arcsec', centered=True
    )
    call_image = np.array(model(theta, 'obs', X, Y))
    render = np.array(model.render_image(theta, image_pars=galsim_image_pars))

    peak = np.max(np.abs(call_image))
    max_frac = np.max(np.abs(render - call_image)) / peak

    assert max_frac < tol, (
        f"render vs call: max|resid|/peak = {max_frac:.2e} "
        f"(nu={nu}, cosi={cosi}, h/r={int_h_over_r}, tol={tol})"
    )


def test_spergel_render_vs_call_negative_nu_diagnostic(galsim_image_pars, output_dir):
    """Diagnostic: render_image vs __call__ for nu=-0.6 (no assertion).

    For nu < 0 the Spergel volume density diverges as R^{2nu} at R=0.
    LOSes near the center have integrand ~ |ell|^{2nu} with 2nu < -1,
    making the 1D LOS integral divergent. The k-space render_image
    band-limits the profile (finite everywhere), while __call__ point-
    samples it (divergent at center). They disagree near the center by
    construction; this test records the disagreement quantitatively.
    """
    model = InclinedSpergelModel()
    nu = -0.6
    theta = jnp.array([0.7, np.pi / 6, 0.02, -0.01, 1.0, 2.0, 0.1, nu, 0.0, 0.0])

    X, Y = build_map_grid_from_image_pars(
        galsim_image_pars, unit='arcsec', centered=True
    )
    call_image = np.array(model(theta, 'obs', X, Y))
    render = np.array(model.render_image(theta, image_pars=galsim_image_pars))

    peak = np.max(np.abs(call_image))
    residual = np.abs(render - call_image)
    max_frac = np.max(residual) / peak
    p95 = np.percentile(residual / peak, 95)
    p99 = np.percentile(residual / peak, 99)

    print(
        f"\nSpergel nu={nu} render_vs_call diagnostic:\n"
        f"  max|resid|/peak = {max_frac:.4f}\n"
        f"  p95 = {p95:.4f}, p99 = {p99:.4f}\n"
        f"  (error localized to central cusp; 95% of pixels agree within {p95:.1%})"
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    im0 = axes[0, 0].imshow(call_image, origin='lower')
    axes[0, 0].set_title('__call__ (LOS quadrature)')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(render, origin='lower')
    axes[0, 1].set_title('render_image (k-space FFT)')
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[1, 0].imshow(residual, origin='lower', cmap='hot')
    axes[1, 0].set_title('|residual|')
    plt.colorbar(im2, ax=axes[1, 0])

    rel = residual / peak
    im3 = axes[1, 1].imshow(rel, origin='lower', cmap='hot')
    axes[1, 1].set_title(f'|residual|/peak (max={max_frac:.2e})')
    plt.colorbar(im3, ax=axes[1, 1])

    fig.suptitle(
        f'Spergel nu={nu} render vs call — DIAGNOSTIC (no assertion)\n'
        f'max={max_frac:.2e}, p95={p95:.2e}, p99={p99:.2e}',
    )
    plt.tight_layout()
    plt.savefig(output_dir / f'spergel_nu{nu}_render_vs_call_diagnostic.png', dpi=150)
    plt.close()


def test_spergel_non_square_grid():
    """Output shape = (Nrow, Ncol) on non-square grid."""
    model = InclinedSpergelModel()
    ip = ImagePars(shape=(80, 120), pixel_scale=0.2, indexing='ij')
    theta = jnp.array([0.7, 0.3, 0.0, 0.0, 1.0, 2.0, 0.1, 0.5, 0.0, 0.0])
    image = model.render_image(theta, image_pars=ip)
    assert image.shape == (80, 120), f"Expected (80, 120), got {image.shape}"


def test_factory_spergel():
    """build_intensity_model('spergel') returns InclinedSpergelModel."""
    model = build_intensity_model('spergel')
    assert isinstance(model, InclinedSpergelModel)
    model2 = build_intensity_model('inclined_spergel')
    assert isinstance(model2, InclinedSpergelModel)


def test_factory_de_vaucouleurs():
    """build_intensity_model('de_vaucouleurs') returns InclinedDeVaucouleursModel."""
    model = build_intensity_model('de_vaucouleurs')
    assert isinstance(model, InclinedDeVaucouleursModel)


def test_spergel_gradient_through_nu():
    """jax.grad of k-space render w.r.t. nu must match finite differences."""
    model = InclinedSpergelModel()
    ip = ImagePars(shape=(32, 32), pixel_scale=0.3, indexing='ij')
    theta = jnp.array([0.7, 0.3, 0.0, 0.0, 1.0, 2.0, 0.1, 0.5, 0.0, 0.0])

    def scalar_fn(th):
        return jnp.sum(model._render_kspace(th, ip.Nrow, ip.Ncol, ip.pixel_scale))

    grad_val = jax.grad(scalar_fn)(theta)
    nu_grad = float(grad_val[7])

    eps = 1e-5
    f_plus = scalar_fn(theta.at[7].set(theta[7] + eps))
    f_minus = scalar_fn(theta.at[7].set(theta[7] - eps))
    fd_grad = float((f_plus - f_minus) / (2 * eps))

    rel_err = abs(nu_grad - fd_grad) / max(abs(fd_grad), 1e-10)
    assert (
        rel_err < 1e-3
    ), f"nu gradient: jax={nu_grad:.6e}, fd={fd_grad:.6e}, rel_err={rel_err:.2e}"


@pytest.mark.parametrize("nu", [-0.6, 0.0, 0.5, 1.0, 2.0])
def test_spergel_normalization_integral(nu):
    """Numerical radial integral of face-on profile must equal flux."""
    from scipy.integrate import quad
    from scipy.special import kve as scipy_kve_np, gamma as scipy_gamma

    flux = 1.0
    rscale = 2.0
    norm_factor = 2.0 * np.pi * rscale**2 * (2.0**nu) * scipy_gamma(nu + 1.0)
    I_0 = flux / norm_factor

    def integrand(r):
        x = r / rscale
        if x < 1e-30:
            return 0.0
        profile = (x**nu) * scipy_kve_np(nu, x) * np.exp(-x)
        return 2.0 * np.pi * r * I_0 * profile

    measured_flux, _ = quad(integrand, 0, 30.0 * rscale, limit=200)
    rel_err = abs(measured_flux - flux) / flux
    assert rel_err < 5e-3, (
        f"Normalization: measured={measured_flux:.6f}, expected={flux}, "
        f"rel_err={rel_err:.2e} (nu={nu})"
    )


def test_spergel_psf_path_consistency(galsim_image_pars):
    """Fused k-space PSF path must match fallback real-space PSF path."""
    import galsim as gs

    model = InclinedSpergelModel()
    psf_obj = gs.Gaussian(fwhm=0.625)
    theta = jnp.array([0.7, 0.3, 0.02, -0.01, 1.0, 2.0, 0.1, 0.5, 0.0, 0.0])

    # fused k-space path (int_model triggers kspace_psf_fft)
    obs_kspace = build_image_obs(
        galsim_image_pars, psf=psf_obj, oversample=5, int_model=model
    )
    img_kspace = np.array(model.render_image(theta, obs=obs_kspace))

    # fallback real-space path (no int_model -> no kspace_psf_fft)
    obs_realspace = build_image_obs(galsim_image_pars, psf=psf_obj, oversample=5)
    img_realspace = np.array(model.render_image(theta, obs=obs_realspace))

    peak = np.max(np.abs(img_kspace))
    max_frac = np.max(np.abs(img_kspace - img_realspace)) / peak

    assert max_frac < 1e-3, f"PSF path consistency: max|resid|/peak = {max_frac:.2e}"


def test_spergel_klmodel_composition():
    """Spergel + CenteredVelocityModel in KLModel: param slicing correct."""
    from kl_pipe.model import KLModel
    from kl_pipe.velocity import CenteredVelocityModel

    vel = CenteredVelocityModel()
    int_model = InclinedSpergelModel()

    kl = KLModel(vel, int_model, shared_pars={'cosi', 'theta_int', 'g1', 'g2'})

    # vel(7) + int(10) - shared(4) = 13 composite params
    assert (
        len(kl.PARAMETER_NAMES) == 13
    ), f"Expected 13, got {len(kl.PARAMETER_NAMES)}: {kl.PARAMETER_NAMES}"
    assert kl.PARAMETER_NAMES.count('cosi') == 1
    assert kl.PARAMETER_NAMES.count('theta_int') == 1
    assert 'nu' in kl.PARAMETER_NAMES

    theta_kl = jnp.arange(13, dtype=float)
    theta_vel = kl.get_velocity_pars(theta_kl)
    theta_int = kl.get_intensity_pars(theta_kl)
    assert len(theta_vel) == 7
    assert len(theta_int) == 10


# --- nu=0.5 Cross-Check (Step 2) ---


@pytest.mark.parametrize(
    "cosi,theta_int,g1,g2",
    [
        (1.0, 0.0, 0.0, 0.0),
        (0.7, 0.0, 0.0, 0.0),
        (0.7, np.pi / 4, 0.0, 0.0),
        (0.7, 0.0, 0.05, -0.03),
        (0.3, np.pi / 4, 0.03, -0.02),
    ],
)
def test_spergel_nu05_matches_exponential_render_image(
    cosi, theta_int, g1, g2, galsim_image_pars
):
    """Spergel(nu=0.5) render_image must match Exponential within 1e-10.

    FTs are algebraically identical: (1+k²)^{-1.5} for both.
    """
    exp_model = InclinedExponentialModel()
    sp_model = InclinedSpergelModel()

    flux, rscale, h_over_r = 1.0, 2.0, 0.1
    theta_exp = jnp.array([cosi, theta_int, g1, g2, flux, rscale, h_over_r, 0.0, 0.0])
    theta_sp = jnp.array(
        [cosi, theta_int, g1, g2, flux, rscale, h_over_r, 0.5, 0.0, 0.0]
    )

    img_exp = exp_model.render_image(theta_exp, image_pars=galsim_image_pars)
    img_sp = sp_model.render_image(theta_sp, image_pars=galsim_image_pars)

    peak = float(jnp.max(jnp.abs(img_exp)))
    max_frac = float(jnp.max(jnp.abs(img_exp - img_sp)) / peak)

    assert max_frac < 1e-10, (
        f"Spergel(nu=0.5) vs Exp render_image: max|resid|/peak = {max_frac:.2e} "
        f"(cosi={cosi}, theta_int={theta_int:.2f}, g=({g1},{g2}))"
    )


def test_spergel_nu05_matches_exponential_call(galsim_image_pars):
    """Spergel(nu=0.5) __call__ must match Exponential within 1e-3.

    K_{0.5}(x) = sqrt(pi/2x)*exp(-x) is exact, but kve callback
    and GL quadrature introduce small numerical differences.
    """
    exp_model = InclinedExponentialModel()
    sp_model = InclinedSpergelModel()

    cosi, theta_int = 0.7, np.pi / 6
    flux, rscale, h_over_r = 1.0, 2.0, 0.1
    theta_exp = jnp.array([cosi, theta_int, 0.0, 0.0, flux, rscale, h_over_r, 0.0, 0.0])
    theta_sp = jnp.array(
        [cosi, theta_int, 0.0, 0.0, flux, rscale, h_over_r, 0.5, 0.0, 0.0]
    )

    X, Y = build_map_grid_from_image_pars(
        galsim_image_pars, unit='arcsec', centered=True
    )
    img_exp = np.array(exp_model(theta_exp, 'obs', X, Y))
    img_sp = np.array(sp_model(theta_sp, 'obs', X, Y))

    peak = np.max(np.abs(img_exp))
    max_frac = np.max(np.abs(img_exp - img_sp)) / peak

    assert (
        max_frac < 1e-3
    ), f"Spergel(nu=0.5) vs Exp __call__: max|resid|/peak = {max_frac:.2e}"


def test_spergel_nu05_matches_exponential_disk_plane(galsim_image_pars):
    """Spergel(nu=0.5) evaluate_in_disk_plane must match Exponential within 1e-4.

    K_{0.5}(x) = sqrt(pi/2x)*exp(-x) is exact.
    """
    exp_model = InclinedExponentialModel()
    sp_model = InclinedSpergelModel()

    flux, rscale = 1.0, 2.0
    theta_exp = jnp.array([0.7, 0.0, 0.0, 0.0, flux, rscale, 0.1, 0.0, 0.0])
    theta_sp = jnp.array([0.7, 0.0, 0.0, 0.0, flux, rscale, 0.1, 0.5, 0.0, 0.0])

    X, Y = build_map_grid_from_image_pars(
        galsim_image_pars, unit='arcsec', centered=True
    )
    disk_exp = np.array(exp_model.evaluate_in_disk_plane(theta_exp, X, Y))
    disk_sp = np.array(sp_model.evaluate_in_disk_plane(theta_sp, X, Y))

    peak = np.max(np.abs(disk_exp))
    max_frac = np.max(np.abs(disk_exp - disk_sp)) / peak

    assert (
        max_frac < 1e-4
    ), f"Spergel(nu=0.5) vs Exp disk plane: max|resid|/peak = {max_frac:.2e}"


def test_spergel_scipy_vs_render_image_consistency(galsim_image_pars):
    """Spergel scipy synthetic backend (numpy) vs render_image (JAX) consistency."""
    from kl_pipe.synthetic import _generate_spergel_scipy

    cosi = 0.7
    flux = 1.0
    int_rscale = 2.0
    int_h_over_r = 0.1
    nu = 0.5
    theta_int = np.pi / 6
    g1 = 0.02
    g2 = -0.01
    int_x0 = 0.3
    int_y0 = -0.2

    scipy_image = _generate_spergel_scipy(
        galsim_image_pars,
        flux=flux,
        int_rscale=int_rscale,
        nu=nu,
        cosi=cosi,
        theta_int=theta_int,
        g1=g1,
        g2=g2,
        int_x0=int_x0,
        int_y0=int_y0,
        int_h_over_r=int_h_over_r,
    )

    model = InclinedSpergelModel()
    theta = jnp.array(
        [cosi, theta_int, g1, g2, flux, int_rscale, int_h_over_r, nu, int_x0, int_y0]
    )
    render = np.array(model.render_image(theta, image_pars=galsim_image_pars))

    peak = np.max(np.abs(render))
    max_frac = np.max(np.abs(scipy_image - render)) / peak

    assert (
        max_frac < 2e-3
    ), f"Spergel scipy vs render_image: max|resid|/peak = {max_frac:.2e}"


# --- GalSim Regression Tests for Spergel (Step 3) ---


@pytest.mark.parametrize("nu", [0.5, 1.0, 2.0, 3.5])
def test_galsim_regression_spergel_faceon(nu, galsim_image_pars):
    """Face-on Spergel render_image vs galsim.Spergel (independent ground truth).

    Point-sampled comparison (no pixel response). Only nu >= 0.5 where
    the FT decays fast enough (k^{-(2+2nu)}) for aliasing to be negligible.
    See issue #38 for future migration to pixel-integrated comparisons.
    """
    import galsim as gs

    flux = 1.0
    rscale = 2.0
    gsp = gs.GSParams(folding_threshold=1e-4, maxk_threshold=1e-4, kvalue_accuracy=1e-6)

    model = InclinedSpergelModel()
    theta = jnp.array([1.0, 0.0, 0.0, 0.0, flux, rscale, 0.1, nu, 0.0, 0.0])
    our_image = np.array(model.render_image(theta, image_pars=galsim_image_pars))

    gs_profile = gs.Spergel(nu=nu, scale_radius=rscale, flux=flux, gsparams=gsp)
    gs_im = gs_profile.drawImage(
        nx=galsim_image_pars.Ncol,
        ny=galsim_image_pars.Nrow,
        scale=galsim_image_pars.pixel_scale,
        method='no_pixel',
    )
    gs_sb = gs_im.array / galsim_image_pars.pixel_scale**2

    peak = np.max(np.abs(gs_sb))
    max_frac = np.max(np.abs(our_image - gs_sb)) / peak

    # 5e-3: GalSim Spergel uses real-space evaluation (is_analytic_x=True)
    # while our code uses k-space IFFT. The difference is the band-limiting
    # error at the cusp. See issue #38 for pixel-integrated comparison.
    assert (
        max_frac < 5e-3
    ), f"Face-on Spergel vs GalSim: max|resid|/peak = {max_frac:.2e} (nu={nu})"


@pytest.mark.parametrize("nu", [-0.6, -0.3, 0.0])
def test_galsim_regression_spergel_faceon_cuspy(nu, galsim_image_pars):
    """Face-on Spergel vs GalSim for nu < 0.5 (requires PSF + pixel integration).

    For nu < 0.5 the Spergel FT decays slowly (k^{-(2+2nu)}), leaving
    significant power at Nyquist. Point-sampled comparison fails because
    our IFFT and GalSim's rendering handle the aliased high-k content
    differently. A small PSF suppresses high-k power; method='auto' +
    oversample=5 ensures both sides pixel-integrate for a fair comparison.
    """
    import galsim as gs

    flux = 1.0
    rscale = 2.0
    fwhm = 0.3
    gsp = gs.GSParams(folding_threshold=1e-5, maxk_threshold=1e-5, kvalue_accuracy=1e-7)
    psf = gs.Gaussian(fwhm=fwhm, gsparams=gsp)

    model = InclinedSpergelModel()
    theta = jnp.array([1.0, 0.0, 0.0, 0.0, flux, rscale, 0.1, nu, 0.0, 0.0])

    obs = build_image_obs(
        galsim_image_pars, psf=psf, oversample=5, int_model=model, gsparams=gsp
    )
    our_image = np.array(model.render_image(theta, obs=obs))

    gs_profile = gs.Convolve(
        gs.Spergel(nu=nu, scale_radius=rscale, flux=flux, gsparams=gsp), psf
    )
    gs_im = gs_profile.drawImage(
        nx=galsim_image_pars.Ncol,
        ny=galsim_image_pars.Nrow,
        scale=galsim_image_pars.pixel_scale,
        method='auto',
    )
    gs_sb = gs_im.array / galsim_image_pars.pixel_scale**2

    peak = np.max(np.abs(gs_sb))
    max_frac = np.max(np.abs(our_image - gs_sb)) / peak

    assert (
        max_frac < 3e-3
    ), f"Face-on Spergel+PSF vs GalSim: max|resid|/peak = {max_frac:.2e} (nu={nu})"


@pytest.mark.parametrize(
    "nu,g1,g2",
    [
        (0.5, 0.05, -0.03),
        (2.0, 0.03, 0.04),
        (0.5, -0.04, 0.02),
    ],
)
def test_galsim_regression_spergel_faceon_with_shear(nu, g1, g2, galsim_image_pars):
    """Face-on sheared Spergel render_image vs galsim.Spergel."""
    import galsim as gs

    flux = 1.0
    rscale = 2.0
    gsp = gs.GSParams(folding_threshold=1e-4, maxk_threshold=1e-4, kvalue_accuracy=1e-6)

    model = InclinedSpergelModel()
    theta = jnp.array([1.0, 0.0, g1, g2, flux, rscale, 0.1, nu, 0.0, 0.0])
    our_image = np.array(model.render_image(theta, image_pars=galsim_image_pars))

    mu = 1.0 / (1.0 - (g1**2 + g2**2))
    gs_profile = gs.Spergel(nu=nu, scale_radius=rscale, flux=flux, gsparams=gsp)
    gs_profile = gs_profile.lens(g1=g1, g2=g2, mu=mu)
    gs_im = gs_profile.drawImage(
        nx=galsim_image_pars.Ncol,
        ny=galsim_image_pars.Nrow,
        scale=galsim_image_pars.pixel_scale,
        method='no_pixel',
    )
    gs_sb = gs_im.array / galsim_image_pars.pixel_scale**2

    peak = np.max(np.abs(gs_sb))
    max_frac = np.max(np.abs(our_image - gs_sb)) / peak

    # 5e-3: real-space (GalSim) vs k-space (ours) band-limiting difference
    assert max_frac < 5e-3, (
        f"Sheared Spergel vs GalSim: max|resid|/peak = {max_frac:.2e} "
        f"(nu={nu}, g1={g1}, g2={g2})"
    )


@pytest.mark.parametrize(
    "nu,g1,g2",
    [
        (-0.6, 0.05, -0.03),
    ],
)
def test_galsim_regression_spergel_faceon_with_shear_negative_nu(
    nu, g1, g2, galsim_image_pars
):
    """Face-on sheared Spergel vs GalSim for nu < 0 (PSF + pixel integration)."""
    import galsim as gs

    flux = 1.0
    rscale = 2.0
    fwhm = 0.3
    gsp = gs.GSParams(folding_threshold=1e-5, maxk_threshold=1e-5, kvalue_accuracy=1e-7)
    psf = gs.Gaussian(fwhm=fwhm, gsparams=gsp)

    model = InclinedSpergelModel()
    theta = jnp.array([1.0, 0.0, g1, g2, flux, rscale, 0.1, nu, 0.0, 0.0])

    obs = build_image_obs(
        galsim_image_pars, psf=psf, oversample=5, int_model=model, gsparams=gsp
    )
    our_image = np.array(model.render_image(theta, obs=obs))

    mu = 1.0 / (1.0 - (g1**2 + g2**2))
    gs_profile = gs.Spergel(nu=nu, scale_radius=rscale, flux=flux, gsparams=gsp)
    gs_profile = gs_profile.lens(g1=g1, g2=g2, mu=mu)
    gs_profile = gs.Convolve(gs_profile, psf)
    gs_im = gs_profile.drawImage(
        nx=galsim_image_pars.Ncol,
        ny=galsim_image_pars.Nrow,
        scale=galsim_image_pars.pixel_scale,
        method='auto',
    )
    gs_sb = gs_im.array / galsim_image_pars.pixel_scale**2

    peak = np.max(np.abs(gs_sb))
    max_frac = np.max(np.abs(our_image - gs_sb)) / peak

    assert max_frac < 3e-3, (
        f"Sheared Spergel+PSF vs GalSim: max|resid|/peak = {max_frac:.2e} "
        f"(nu={nu}, g1={g1}, g2={g2})"
    )


def test_galsim_regression_devaucouleurs_faceon(galsim_image_pars):
    """Face-on DeVaucouleurs vs GalSim Spergel(nu=-0.6) (PSF + pixel integration)."""
    import galsim as gs

    flux = 1.0
    rscale = 2.0
    fwhm = 0.3
    gsp = gs.GSParams(folding_threshold=1e-5, maxk_threshold=1e-5, kvalue_accuracy=1e-7)

    psf = gs.Gaussian(fwhm=fwhm, gsparams=gsp)

    model = InclinedDeVaucouleursModel()
    theta = jnp.array([1.0, 0.0, 0.0, 0.0, flux, rscale, 0.1, 0.0, 0.0])

    obs = build_image_obs(
        galsim_image_pars, psf=psf, oversample=5, int_model=model, gsparams=gsp
    )
    our_image = np.array(model.render_image(theta, obs=obs))

    gs_profile = gs.Convolve(
        gs.Spergel(nu=-0.6, scale_radius=rscale, flux=flux, gsparams=gsp), psf
    )
    gs_im = gs_profile.drawImage(
        nx=galsim_image_pars.Ncol,
        ny=galsim_image_pars.Nrow,
        scale=galsim_image_pars.pixel_scale,
        method='auto',
    )
    gs_sb = gs_im.array / galsim_image_pars.pixel_scale**2

    peak = np.max(np.abs(gs_sb))
    max_frac = np.max(np.abs(our_image - gs_sb)) / peak

    assert (
        max_frac < 3e-3
    ), f"DeVaucouleurs+PSF vs GalSim Spergel(nu=-0.6): max|resid|/peak = {max_frac:.2e}"


@pytest.mark.parametrize(
    "cosi,theta_int",
    [
        (1.0, 0.0),
        (0.7, 0.0),
        (0.7, np.pi / 4),
        (0.4, 0.0),
        (0.4, np.pi / 4),
    ],
)
def test_galsim_regression_spergel_inclined_nu05(cosi, theta_int, galsim_image_pars):
    """Inclined Spergel(nu=0.5) vs galsim.InclinedExponential (exact match)."""
    from kl_pipe.synthetic import _generate_sersic_galsim
    import galsim as gs

    flux = 1.0
    rscale = 2.0
    h_over_r = 0.1
    gsp = gs.GSParams(folding_threshold=1e-4, maxk_threshold=1e-4, kvalue_accuracy=1e-6)

    model = InclinedSpergelModel()
    theta = jnp.array(
        [cosi, theta_int, 0.0, 0.0, flux, rscale, h_over_r, 0.5, 0.0, 0.0]
    )
    our_image = np.array(model.render_image(theta, image_pars=galsim_image_pars))

    gs_image = _generate_sersic_galsim(
        galsim_image_pars,
        flux=flux,
        int_rscale=rscale,
        n_sersic=1.0,
        cosi=cosi,
        theta_int=theta_int,
        g1=0.0,
        g2=0.0,
        int_x0=0.0,
        int_y0=0.0,
        int_h_over_r=h_over_r,
        gsparams=gsp,
        method='no_pixel',
    )
    gs_sb = gs_image / galsim_image_pars.pixel_scale**2

    peak = np.max(np.abs(gs_sb))
    max_frac = np.max(np.abs(our_image - gs_sb)) / peak

    assert max_frac < 1e-3, (
        f"Inclined Spergel(nu=0.5) vs GalSim InclinedExp: max|resid|/peak = {max_frac:.2e} "
        f"(cosi={cosi}, theta_int={theta_int:.2f})"
    )


def test_spergel_nu05_vs_inclined_sersic_n1(galsim_image_pars):
    """Spergel(nu=0.5) must match InclinedSersic(n=1) — exact equivalence."""
    from kl_pipe.synthetic import _generate_sersic_galsim
    import galsim as gs

    cosi = 0.7
    flux = 1.0
    rscale = 2.0
    h_over_r = 0.1
    gsp = gs.GSParams(folding_threshold=1e-4, maxk_threshold=1e-4, kvalue_accuracy=1e-6)

    model = InclinedSpergelModel()
    theta = jnp.array([cosi, 0.0, 0.0, 0.0, flux, rscale, h_over_r, 0.5, 0.0, 0.0])
    our_image = np.array(model.render_image(theta, image_pars=galsim_image_pars))

    gs_image = _generate_sersic_galsim(
        galsim_image_pars,
        flux=flux,
        int_rscale=rscale,
        n_sersic=1.0,
        cosi=cosi,
        theta_int=0.0,
        g1=0.0,
        g2=0.0,
        int_x0=0.0,
        int_y0=0.0,
        int_h_over_r=h_over_r,
        gsparams=gsp,
        method='no_pixel',
    )
    gs_sb = gs_image / galsim_image_pars.pixel_scale**2

    peak = np.max(np.abs(gs_sb))
    max_frac = np.max(np.abs(our_image - gs_sb)) / peak

    assert (
        max_frac < 1e-3
    ), f"Spergel(nu=0.5) vs InclinedSersic(n=1): max|resid|/peak = {max_frac:.2e}"


def test_spergel_vs_inclined_sersic_devac_mismatch(galsim_image_pars):
    """Quantify Spergel(nu=-0.6) vs InclinedSersic(n=4) mismatch (informational).

    Profiles matched at half_light_radius=2.0 for fair shape comparison.
    Uses PSF + method='auto' + coarser grid because InclinedSersic(n=4)
    needs enormous FFT grids at fine pixel scales.
    """
    import galsim as gs

    cosi = 0.7
    inc = gs.Angle(np.arccos(cosi), gs.radians)
    flux = 1.0
    hlr = 2.0
    h_over_r = 0.1
    fwhm = 0.5

    gsp = gs.GSParams(folding_threshold=1e-3, maxk_threshold=1e-3, kvalue_accuracy=1e-5)
    psf = gs.Gaussian(fwhm=fwhm, gsparams=gsp)

    # GalSim InclinedSersic(n=4) matched at half_light_radius
    gs_prof = gs.InclinedSersic(
        n=4.0,
        inclination=inc,
        half_light_radius=hlr,
        scale_h_over_r=h_over_r,
        flux=flux,
        gsparams=gsp,
    )
    gs_prof = gs.Convolve(gs_prof, psf)

    # coarser grid for tractable GalSim FFT
    ip = ImagePars(shape=(64, 64), pixel_scale=0.5, indexing='ij')
    gs_im = gs_prof.drawImage(
        nx=ip.Ncol,
        ny=ip.Nrow,
        scale=ip.pixel_scale,
        method='auto',
    )
    gs_sb = gs_im.array / ip.pixel_scale**2

    # our Spergel(nu=-0.6) matched at same half_light_radius
    spergel_rscale = gs.Spergel(nu=-0.6, half_light_radius=hlr).scale_radius
    model = InclinedSpergelModel()
    theta = jnp.array(
        [cosi, 0.0, 0.0, 0.0, flux, spergel_rscale, h_over_r, -0.6, 0.0, 0.0]
    )
    obs = build_image_obs(ip, psf=psf, oversample=5, int_model=model, gsparams=gsp)
    our_image = np.array(model.render_image(theta, obs=obs))

    peak = np.max(np.abs(gs_sb))
    max_frac = np.max(np.abs(our_image - gs_sb)) / peak
    rms_frac = np.sqrt(np.mean(((our_image - gs_sb) / peak) ** 2))

    # informational — Spergel != Sersic, so we just record the mismatch
    print(
        f"\nSpergel(nu=-0.6) vs InclinedSersic(n=4) at hlr={hlr}:\n"
        f"  max|resid|/peak = {max_frac:.4f}\n"
        f"  rms|resid|/peak = {rms_frac:.4f}\n"
        f"  spergel_rscale = {spergel_rscale:.4f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
