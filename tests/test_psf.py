"""
PSF convolution module unit tests.

Tests include:
A. GalSim regression (JAX FFT vs GalSim native Convolve)
B. JAX compatibility (JIT, grad)
C. Physical correctness (delta, normalization, flux conservation, constant velocity,
   flux-weighted shift)
D. Additional (symmetry, linearity, monotonicity, Parseval, wrap-around)
E. Render layer integration
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import galsim as gs
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from kl_pipe.psf import (
    PSFData,
    gsobj_to_kernel,
    precompute_psf_fft,
    convolve_fft,
    convolve_flux_weighted,
    convolve_fft_numpy,
    convolve_flux_weighted_numpy,
)
from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.velocity import CenteredVelocityModel
from kl_pipe.parameters import ImagePars
from kl_pipe.utils import build_map_grid_from_image_pars, get_test_dir


# ==============================================================================
# Fixtures
# ==============================================================================

OUTPUT_DIR = get_test_dir() / "out" / "psf"


@pytest.fixture(scope="module")
def output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


@pytest.fixture(scope="module")
def image_pars():
    return ImagePars(shape=(60, 80), pixel_scale=0.3, indexing='xy')


@pytest.fixture(scope="module")
def gaussian_psf():
    return gs.Gaussian(fwhm=0.625)


@pytest.fixture(scope="module")
def psf_data(gaussian_psf, image_pars):
    return precompute_psf_fft(gaussian_psf, image_pars.shape, image_pars.pixel_scale)


@pytest.fixture(scope="module")
def test_image(image_pars):
    """Simple exponential disk for testing."""
    X, Y = build_map_grid_from_image_pars(image_pars, unit='arcsec', centered=True)
    r = np.sqrt(X**2 + Y**2)
    return np.exp(-r / 3.0)


# ==============================================================================
# A. GalSim Regression Tests
# ==============================================================================

PSF_TYPES = [
    gs.Gaussian(fwhm=0.625),
    gs.Gaussian(fwhm=1.25),
    gs.Moffat(beta=3.5, fwhm=0.625),
    gs.Moffat(beta=2.5, fwhm=1.0),
    gs.Airy(lam_over_diam=0.5),
    gs.OpticalPSF(lam_over_diam=0.5, defocus=0.5, coma1=0.3),
]

PSF_NAMES = [
    'Gaussian_0.625',
    'Gaussian_1.25',
    'Moffat_3.5_0.625',
    'Moffat_2.5_1.0',
    'Airy_0.5',
    'OpticalPSF_0.5',
]


@pytest.mark.parametrize(
    "psf_obj,psf_name", zip(PSF_TYPES, PSF_NAMES), ids=PSF_NAMES
)
def test_galsim_regression(psf_obj, psf_name, image_pars, output_dir):
    """
    Convolve Sersic(n=1, hlr=3, flux=1) via GalSim native vs JAX FFT.
    Assert max|residual|/peak < 1e-4.
    """
    pixel_scale = image_pars.pixel_scale
    nx, ny = image_pars.Nx, image_pars.Ny

    # source profile
    sersic = gs.Exponential(half_light_radius=3.0, flux=1.0)

    # GalSim native: Convolve(sersic, psf).drawImage (pixel-integrated)
    conv_gs = gs.Convolve(sersic, psf_obj)
    img_gs = conv_gs.drawImage(nx=nx, ny=ny, scale=pixel_scale).array

    # JAX FFT path:
    # 1. point-sampled source (method='no_pixel' matches model.__call__)
    img_source = sersic.drawImage(
        nx=nx, ny=ny, scale=pixel_scale, method='no_pixel'
    ).array

    # 2. convolve with PSF kernel (default method = pixel-integrated)
    pdata = precompute_psf_fft(psf_obj, (ny, nx), pixel_scale)
    img_jax = np.array(convolve_fft(jnp.array(img_source), pdata))

    # compute residual
    residual = img_gs - img_jax
    peak = np.max(np.abs(img_gs))
    max_rel_resid = np.max(np.abs(residual)) / peak

    # diagnostic plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    im0 = axes[0, 0].imshow(img_gs, origin='lower')
    axes[0, 0].set_title('GalSim native')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(img_jax, origin='lower')
    axes[0, 1].set_title('JAX FFT')
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[1, 0].imshow(residual, origin='lower', cmap='RdBu_r')
    axes[1, 0].set_title('Absolute residual')
    plt.colorbar(im2, ax=axes[1, 0])

    rel_err = np.abs(residual) / peak
    im3 = axes[1, 1].imshow(rel_err, origin='lower')
    axes[1, 1].set_title('Relative error |resid|/peak')
    plt.colorbar(im3, ax=axes[1, 1])

    status = 'PASS' if max_rel_resid < 1e-4 else 'FAIL'
    fig.suptitle(
        f'Sersic(n=1,hlr=3) x {psf_name} -- {status} (max_resid={max_rel_resid:.2e})',
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(output_dir / f'galsim_regression_{psf_name}.png', dpi=150)
    plt.close()

    assert max_rel_resid < 1e-4, (
        f"GalSim regression failed for {psf_name}: max_rel_resid={max_rel_resid:.2e}"
    )


# ==============================================================================
# B. JAX Compatibility Tests
# ==============================================================================


def test_jit_convolve_fft(test_image, psf_data):
    """convolve_fft runs under jax.jit."""
    jitted = jax.jit(convolve_fft)
    result = jitted(jnp.array(test_image), psf_data)
    assert result.shape == test_image.shape
    assert jnp.all(jnp.isfinite(result))


def test_jit_convolve_flux_weighted(test_image, psf_data):
    """convolve_flux_weighted runs under jax.jit."""
    vel = jnp.ones_like(jnp.array(test_image)) * 100.0
    jitted = jax.jit(convolve_flux_weighted)
    result = jitted(vel, jnp.array(test_image), psf_data)
    assert result.shape == test_image.shape
    assert jnp.all(jnp.isfinite(result))


def test_grad_convolve_fft(test_image, psf_data):
    """Gradient through convolve_fft w.r.t. image is all finite."""
    img = jnp.array(test_image)
    grad_fn = jax.grad(lambda x: convolve_fft(x, psf_data).sum())
    g = grad_fn(img)
    assert jnp.all(jnp.isfinite(g))


def test_grad_flux_weighted_wrt_velocity(test_image, psf_data):
    """Gradient through convolve_flux_weighted w.r.t. velocity."""
    vel = jnp.ones_like(jnp.array(test_image)) * 100.0
    intensity = jnp.array(test_image)
    grad_fn = jax.grad(lambda v: convolve_flux_weighted(v, intensity, psf_data).sum())
    g = grad_fn(vel)
    assert jnp.all(jnp.isfinite(g))


def test_grad_flux_weighted_wrt_intensity(test_image, psf_data):
    """Gradient through convolve_flux_weighted w.r.t. intensity."""
    vel = jnp.ones_like(jnp.array(test_image)) * 100.0
    intensity = jnp.array(test_image)
    grad_fn = jax.grad(lambda i: convolve_flux_weighted(vel, i, psf_data).sum())
    g = grad_fn(intensity)
    assert jnp.all(jnp.isfinite(g))


# ==============================================================================
# C. Physical Correctness Tests
# ==============================================================================


def test_delta_function_psf_is_identity(image_pars, test_image):
    """Convolving with a delta-function PSF returns the input."""
    # tiny Gaussian approximating a delta function (sub-pixel FWHM)
    delta_psf = gs.Gaussian(fwhm=1e-4)
    pdata = precompute_psf_fft(delta_psf, image_pars.shape, image_pars.pixel_scale)
    result = np.array(convolve_fft(jnp.array(test_image), pdata))

    np.testing.assert_allclose(result, test_image, atol=1e-6)


def test_kernel_normalization(image_pars):
    """All PSF kernels sum to 1."""
    for psf_obj in PSF_TYPES:
        kernel_shifted, _ = gsobj_to_kernel(psf_obj, image_pars.shape, image_pars.pixel_scale)
        total = np.sum(kernel_shifted)
        np.testing.assert_allclose(total, 1.0, atol=1e-10, err_msg=str(psf_obj))


def test_flux_conservation(image_pars, test_image, psf_data):
    """sum(convolved) ~ sum(original)."""
    result = np.array(convolve_fft(jnp.array(test_image), psf_data))
    np.testing.assert_allclose(
        np.sum(result), np.sum(test_image), rtol=1e-6
    )


def test_constant_velocity_invariance(image_pars, test_image, psf_data):
    """If v(x,y)=c everywhere, flux-weighted PSF returns c."""
    c = 42.0
    vel = jnp.full_like(jnp.array(test_image), c)
    intensity = jnp.array(test_image)
    result = convolve_flux_weighted(vel, intensity, psf_data)

    # only check where intensity is nonnegligible
    mask = np.array(intensity) > 1e-8
    np.testing.assert_allclose(np.array(result)[mask], c, atol=1e-6)


def test_flux_weighted_velocity_shift(image_pars, output_dir):
    """
    Left-bright intensity + left-to-right velocity gradient:
    flux-weighted result should shift velocity toward brighter (left) side.
    """
    ny, nx = image_pars.shape
    X, Y = build_map_grid_from_image_pars(image_pars, unit='arcsec', centered=True)

    # left-bright intensity (exponential falloff from left)
    intensity = np.exp(-((X - X.min()) / 3.0))

    # linear velocity gradient left-to-right
    velocity = X * 10.0  # negative on left, positive on right

    psf_obj = gs.Gaussian(fwhm=1.5)
    pdata = precompute_psf_fft(psf_obj, image_pars.shape, image_pars.pixel_scale)

    # no PSF
    v_no_psf = velocity.copy()

    # with PSF
    v_psf = np.array(
        convolve_flux_weighted(
            jnp.array(velocity), jnp.array(intensity), pdata
        )
    )

    # mean velocity should shift toward the bright (left/negative) side
    mask = intensity > 1e-6
    mean_no_psf = np.mean(v_no_psf[mask])
    mean_psf = np.mean(v_psf[mask])

    assert mean_psf < mean_no_psf, (
        f"Flux-weighted PSF should shift velocity toward bright side: "
        f"mean_no_psf={mean_no_psf:.3f}, mean_psf={mean_psf:.3f}"
    )


# ==============================================================================
# D. Additional Unique Tests
# ==============================================================================


def test_symmetry_preservation(image_pars):
    """
    Face-on circular galaxy convolved with circular PSF -> circularly symmetric.
    """
    X, Y = build_map_grid_from_image_pars(image_pars, unit='arcsec', centered=True)
    r = np.sqrt(X**2 + Y**2)
    image = np.exp(-r / 3.0)

    psf_obj = gs.Gaussian(fwhm=0.9)
    pdata = precompute_psf_fft(psf_obj, image_pars.shape, image_pars.pixel_scale)
    result = np.array(convolve_fft(jnp.array(image), pdata))

    # check result is symmetric under 180 deg rotation
    rotated = np.flip(result)
    # not exact due to even grid size, so use moderate tolerance
    np.testing.assert_allclose(result, rotated, atol=1e-6)


def test_linearity(image_pars, psf_data):
    """Conv(a*I1 + b*I2, PSF) == a*Conv(I1, PSF) + b*Conv(I2, PSF)."""
    X, Y = build_map_grid_from_image_pars(image_pars, unit='arcsec', centered=True)
    r = np.sqrt(X**2 + Y**2)
    I1 = jnp.array(np.exp(-r / 2.0))
    I2 = jnp.array(np.exp(-r / 5.0))
    a, b = 2.5, -0.7

    lhs = convolve_fft(a * I1 + b * I2, psf_data)
    rhs = a * convolve_fft(I1, psf_data) + b * convolve_fft(I2, psf_data)

    np.testing.assert_allclose(np.array(lhs), np.array(rhs), atol=1e-10)


def test_psf_size_monotonicity(image_pars, test_image):
    """Larger FWHM -> lower peak value (monotonically decreasing)."""
    fwhms = [0.3, 0.9, 2.0]
    peaks = []
    for fwhm in fwhms:
        psf_obj = gs.Gaussian(fwhm=fwhm)
        pdata = precompute_psf_fft(psf_obj, image_pars.shape, image_pars.pixel_scale)
        result = np.array(convolve_fft(jnp.array(test_image), pdata))
        peaks.append(np.max(result))

    for i in range(len(peaks) - 1):
        assert peaks[i] > peaks[i + 1], (
            f"Peak not monotonically decreasing: FWHM={fwhms[i]}->{fwhms[i+1]}, "
            f"peaks={peaks[i]:.6f}->{peaks[i+1]:.6f}"
        )


def test_parseval_bound(image_pars, test_image, psf_data):
    """sum(Conv(I,PSF)^2) <= sum(I^2)."""
    result = np.array(convolve_fft(jnp.array(test_image), psf_data))
    power_original = np.sum(test_image**2)
    power_convolved = np.sum(result**2)

    assert power_convolved <= power_original * (1 + 1e-10), (
        f"Parseval bound violated: {power_convolved:.6f} > {power_original:.6f}"
    )


def test_wrap_around_artifact(image_pars):
    """
    Bright source at corner should not wrap to opposite corner.
    """
    ny, nx = image_pars.shape
    image = np.zeros((ny, nx))
    # bright source in top-right corner
    image[ny - 3 : ny, nx - 3 : nx] = 1000.0

    psf_obj = gs.Gaussian(fwhm=0.9)
    pdata = precompute_psf_fft(psf_obj, (ny, nx), image_pars.pixel_scale)
    result = np.array(convolve_fft(jnp.array(image), pdata))

    # opposite corner (bottom-left) should be ~zero
    corner_value = np.max(np.abs(result[:3, :3]))
    assert corner_value < 1e-6, (
        f"Wrap-around artifact: opposite corner = {corner_value:.2e}"
    )


# ==============================================================================
# E. Render Layer Integration Tests
# ==============================================================================


def test_no_psf_regression(image_pars):
    """
    render_image with no PSF == model(theta, 'obs', X, Y).
    """
    model = InclinedExponentialModel()
    theta = jnp.array([0.7, 0.785, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0])
    X, Y = build_map_grid_from_image_pars(image_pars)

    # direct call
    direct = model(theta, 'obs', X, Y)

    # render_image (no PSF configured)
    rendered = model.render_image(theta, image_pars)

    np.testing.assert_array_equal(np.array(direct), np.array(rendered))


def test_psf_render_image_consistency(image_pars, gaussian_psf):
    """
    Configure PSF -> render_image == manual convolve_fft(model(...), psf_data).
    """
    model = InclinedExponentialModel()
    theta = jnp.array([0.7, 0.785, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0])
    X, Y = build_map_grid_from_image_pars(image_pars)

    # manual convolution
    raw = model(theta, 'obs', X, Y)
    pdata = precompute_psf_fft(gaussian_psf, image_pars.shape, image_pars.pixel_scale)
    manual = convolve_fft(raw, pdata)

    # render_image with PSF
    model.configure_psf(gaussian_psf, image_pars.shape, image_pars.pixel_scale)
    rendered = model.render_image(theta, image_pars)
    model.clear_psf()

    np.testing.assert_allclose(np.array(rendered), np.array(manual), atol=1e-12)


def test_velocity_render_image_flux_weighted(image_pars, gaussian_psf):
    """
    Velocity render_image with flux_model uses flux-weighted convolution.
    """
    vel_model = CenteredVelocityModel()
    int_model = InclinedExponentialModel()

    theta_vel = jnp.array([0.6, 0.785, 0.0, 0.0, 10.0, 200.0, 5.0])
    theta_int = jnp.array([0.6, 0.785, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0])

    X, Y = build_map_grid_from_image_pars(image_pars)

    # manual flux-weighted convolution
    raw_vel = vel_model(theta_vel, 'obs', X, Y)
    raw_int = int_model(theta_int, 'obs', X, Y)
    pdata = precompute_psf_fft(gaussian_psf, image_pars.shape, image_pars.pixel_scale)
    manual = convolve_flux_weighted(raw_vel, raw_int, pdata)

    # render_image with PSF + flux_model
    vel_model.configure_velocity_psf(
        gaussian_psf,
        image_pars.shape,
        image_pars.pixel_scale,
        flux_model=int_model,
        flux_theta=theta_int,
    )
    rendered = vel_model.render_image(theta_vel, image_pars)
    vel_model.clear_psf()

    np.testing.assert_allclose(np.array(rendered), np.array(manual), atol=1e-10)


def test_psf_frozen_raises():
    """configure_psf raises if frozen."""
    model = InclinedExponentialModel()
    psf = gs.Gaussian(fwhm=0.5)
    model.configure_psf(psf, (32, 32), 0.3, freeze=True)

    with pytest.raises(ValueError, match="frozen"):
        model.configure_psf(psf, (32, 32), 0.3)

    model.clear_psf()
    assert not model.has_psf


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
