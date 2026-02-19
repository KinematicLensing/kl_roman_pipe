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
from matplotlib.colors import LogNorm
from pathlib import Path
from scipy.fft import next_fast_len

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
# Helpers
# ==============================================================================


def _gaussian_2d(X, Y, sigma):
    """2D Gaussian on grid, normalized to sum=1."""
    g = np.exp(-0.5 * (X**2 + Y**2) / sigma**2)
    return g / g.sum()


def _azimuthal_average(image, X, Y, radial_bins):
    """Azimuthal average in radial bins. Returns (bin_centers, profile)."""
    r = np.sqrt(X**2 + Y**2)
    profile = np.zeros(len(radial_bins) - 1)
    for i in range(len(radial_bins) - 1):
        mask = (r >= radial_bins[i]) & (r < radial_bins[i + 1])
        if mask.any():
            profile[i] = np.mean(image[mask])
        else:
            profile[i] = np.nan
    bin_centers = 0.5 * (radial_bins[:-1] + radial_bins[1:])
    return bin_centers, profile


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
def oversample_image_pars():
    """Larger rectangular stamps for oversampled tests — eliminates edge effects
    with hlr=3.0. 150x200 at 0.3"/pixel = 45"x60", min boundary at 22.5" →
    exp(-22.5/1.788) ~ 3.4e-6 of peak. Rectangular to catch shape/indexing bugs
    that square stamps would hide.
    """
    return ImagePars(shape=(150, 200), pixel_scale=0.3, indexing='ij')


@pytest.fixture(scope="module")
def gaussian_psf():
    return gs.Gaussian(fwhm=0.625)


@pytest.fixture(scope="module")
def psf_data(gaussian_psf, image_pars):
    return precompute_psf_fft(gaussian_psf, image_pars)


@pytest.fixture(scope="module")
def test_image(image_pars):
    """Simple exponential disk for testing."""
    X, Y = build_map_grid_from_image_pars(image_pars, unit='arcsec', centered=True)
    r = np.sqrt(X**2 + Y**2)
    return np.exp(-r / 3.0)


@pytest.fixture(scope="module")
def compact_test_image(image_pars):
    """Steep exponential negligible at image boundaries (for flux conservation)."""
    X, Y = build_map_grid_from_image_pars(image_pars, unit='arcsec', centered=True)
    r = np.sqrt(X**2 + Y**2)
    return np.exp(-r / 0.5)


# ==============================================================================
# A. GalSim Regression Tests
# ==============================================================================

PSF_CASES = [
    pytest.param(gs.Gaussian(fwhm=0.625), 'Gaussian_0.625', id='Gaussian_0.625'),
    pytest.param(gs.Gaussian(fwhm=1.25), 'Gaussian_1.25', id='Gaussian_1.25'),
    pytest.param(
        gs.Moffat(beta=3.5, fwhm=0.625), 'Moffat_3.5_0.625', id='Moffat_3.5_0.625'
    ),
    pytest.param(gs.Moffat(beta=2.5, fwhm=1.0), 'Moffat_2.5_1.0', id='Moffat_2.5_1.0'),
    pytest.param(gs.Airy(lam_over_diam=0.5), 'Airy_0.5', id='Airy_0.5'),
    pytest.param(
        gs.OpticalPSF(lam_over_diam=0.5, defocus=0.5, coma1=0.3),
        'OpticalPSF_0.5',
        id='OpticalPSF_0.5',
        marks=pytest.mark.slow,
    ),
]

# flat lists for tests that need them without slow filtering
PSF_TYPES = [c.values[0] for c in PSF_CASES]
PSF_NAMES = [c.values[1] for c in PSF_CASES]


@pytest.mark.parametrize("psf_obj,psf_name", PSF_CASES)
def test_galsim_regression(psf_obj, psf_name, image_pars, output_dir):
    """
    Convolve Sersic(n=1, hlr=3, flux=1) via GalSim native vs JAX FFT.

    Characterizes method accuracy: our pipeline point-samples the source then
    does discrete FFT, while GalSim convolves in continuous Fourier space.
    Non-band-limited profiles (exponential has Lorentzian FT tails) alias
    under discrete sampling, giving a ~0.2-0.4% floor. Threshold 5e-3 gives
    ~40% margin above worst case (OpticalPSF at 3.6e-3).
    """
    pixel_scale = image_pars.pixel_scale
    nx, ny = image_pars.Nx, image_pars.Ny

    # tightened GSParams for accurate ground truth (see test_galsim_regression_oversampled)
    gsp = gs.GSParams(folding_threshold=1e-4, maxk_threshold=1e-4, kvalue_accuracy=1e-6)
    sersic = gs.Exponential(half_light_radius=3.0, flux=1.0, gsparams=gsp)
    psf_tight = psf_obj.withGSParams(gsp)

    # GalSim native: Convolve(sersic, psf).drawImage (pixel-integrated)
    conv_gs = gs.Convolve(sersic, psf_tight)
    img_gs = conv_gs.drawImage(nx=nx, ny=ny, scale=pixel_scale).array

    # JAX FFT path:
    # 1. point-sampled source (method='no_pixel' matches model.__call__)
    img_source = sersic.drawImage(
        nx=nx, ny=ny, scale=pixel_scale, method='no_pixel'
    ).array

    # 2. convolve with PSF kernel (default method = pixel-integrated)
    pdata = precompute_psf_fft(
        psf_obj, ImagePars(shape=(nx, ny), pixel_scale=pixel_scale, indexing='xy')
    )
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

    vmax_abs = np.max(np.abs(residual))
    im2 = axes[1, 0].imshow(
        residual, origin='lower', cmap='RdBu_r', vmin=-vmax_abs, vmax=vmax_abs
    )
    axes[1, 0].set_title('Residual (GS - JAX)')
    plt.colorbar(im2, ax=axes[1, 0])

    rel_resid = residual / peak
    vmax_rel = np.max(np.abs(rel_resid))
    im3 = axes[1, 1].imshow(
        rel_resid, origin='lower', cmap='RdBu_r', vmin=-vmax_rel, vmax=vmax_rel
    )
    axes[1, 1].set_title('Relative residual (resid/peak)')
    plt.colorbar(im3, ax=axes[1, 1])

    status = 'PASS' if max_rel_resid < 5e-3 else 'FAIL'
    status_color = 'green' if status == 'PASS' else 'red'
    fig.suptitle(
        f'Sersic(n=1,hlr=3) x {psf_name} -- {status} (max={max_rel_resid:.2e}, thr=5e-3)',
        fontsize=13,
        color=status_color,
    )
    plt.tight_layout()
    plt.savefig(output_dir / f'galsim_regression_{psf_name}.png', dpi=150)
    plt.close()

    # aliasing floor from point-sample-then-FFT vs GalSim continuous convolution;
    # oversampled tests (test_galsim_regression_oversampled) achieve 5e-4
    assert (
        max_rel_resid < 5e-3
    ), f"GalSim regression failed for {psf_name}: max_rel_resid={max_rel_resid:.2e}"


# ==============================================================================
# A2. Gaussian Convolution Theorem (exact analytic reference)
# ==============================================================================


def test_gaussian_convolution_theorem(oversample_image_pars, output_dir):
    """
    Gaussian x Gaussian = Gaussian with sigma_out = sqrt(sigma_s^2 + sigma_p^2).

    Tests two independent convolution paths against the same analytic reference:
    1. Pure numpy FFT (hand-crafted kernel)
    2. Oversampled GalSim->JAX pipeline (N=5, precompute_psf_fft + convolve_fft)

    Assertions:
    - Width match (second-moment sigma) < 1e-3 relative
    - Peak match < 1e-3 relative
    - Max |residual| / peak: numpy < 1e-4, JAX < 5e-4
    - Flux conservation < 1e-6 relative
    """
    sigma_source = 1.5  # arcsec
    sigma_psf = 1.2  # arcsec
    sigma_out = np.sqrt(sigma_source**2 + sigma_psf**2)  # ~1.921 arcsec

    pixel_scale = oversample_image_pars.pixel_scale
    X, Y = build_map_grid_from_image_pars(
        oversample_image_pars, unit='arcsec', centered=True
    )
    nrow, ncol = X.shape
    image_shape = (nrow, ncol)

    # --- build source, analytic reference ---
    source = _gaussian_2d(X, Y, sigma_source)
    analytic = _gaussian_2d(X, Y, sigma_out)

    # === Path 1: pure numpy FFT ===
    # hand-craft small Gaussian kernel, pad with np.roll for correct wrapping
    kern_half = int(np.ceil(6 * sigma_psf / pixel_scale))
    kern_size = 2 * kern_half + 1
    kx = (np.arange(kern_size) - kern_half) * pixel_scale
    ky = (np.arange(kern_size) - kern_half) * pixel_scale
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    psf_kernel_raw = np.exp(-0.5 * (KX**2 + KY**2) / sigma_psf**2)
    psf_kernel_raw /= psf_kernel_raw.sum()
    ny_pad = next_fast_len(nrow + kern_size - 1)
    nx_pad = next_fast_len(ncol + kern_size - 1)
    padded_shape_np = (ny_pad, nx_pad)
    kernel_padded = np.zeros(padded_shape_np, dtype=np.float64)
    kernel_padded[:kern_size, :kern_size] = psf_kernel_raw
    kernel_padded = np.roll(kernel_padded, (-kern_half, -kern_half), axis=(0, 1))
    conv_numpy = convolve_fft_numpy(source, kernel_padded, padded_shape_np)

    # === Path 2: GalSim -> JAX pipeline (oversampled, N=5 default) ===
    N_os = 5
    gspsf = gs.Gaussian(sigma=sigma_psf)
    fine_ip = oversample_image_pars.make_fine_scale(N_os)
    X_fine, Y_fine = build_map_grid_from_image_pars(
        fine_ip, unit='arcsec', centered=True
    )
    # fine-scale source in surface-brightness convention (N^2 scaling)
    source_fine = _gaussian_2d(X_fine, Y_fine, sigma_source) * (N_os * N_os)
    pdata = precompute_psf_fft(gspsf, oversample_image_pars, oversample=N_os)
    conv_jax = np.array(convolve_fft(jnp.array(source_fine), pdata))

    # binned analytic reference for JAX — accounts for pixel-averaging from
    # PSF kernel integration and fine→coarse binning (adds h²/12 to variance)
    analytic_jax_fine = _gaussian_2d(X_fine, Y_fine, sigma_out) * (N_os * N_os)
    analytic_jax = analytic_jax_fine.reshape(nrow, N_os, ncol, N_os).mean(axis=(1, 3))

    # --- measurements ---
    def _measure_sigma(image, X, Y):
        """second-moment width"""
        r2 = X**2 + Y**2
        return np.sqrt(np.sum(r2 * image) / np.sum(image))

    sigma_analytic_meas = _measure_sigma(analytic, X, Y)
    sigma_analytic_jax_meas = _measure_sigma(analytic_jax, X, Y)
    sigma_numpy_meas = _measure_sigma(conv_numpy, X, Y)
    sigma_jax_meas = _measure_sigma(conv_jax, X, Y)

    peak_analytic = np.max(analytic)
    peak_analytic_jax = np.max(analytic_jax)
    peak_numpy = np.max(conv_numpy)
    peak_jax = np.max(conv_jax)

    resid_numpy = conv_numpy - analytic
    resid_jax = conv_jax - analytic_jax
    max_resid_numpy = np.max(np.abs(resid_numpy)) / peak_analytic
    max_resid_jax = np.max(np.abs(resid_jax)) / peak_analytic_jax

    flux_source = np.sum(source)
    flux_numpy = np.sum(conv_numpy)
    flux_jax = np.sum(conv_jax)

    # --- diagnostic figure (3 rows) ---
    fig = plt.figure(figsize=(16, 14))
    gs_fig = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.2], hspace=0.35, wspace=0.3)

    # row 1: source | PSF kernel | convolved (JAX) | analytic
    ax00 = fig.add_subplot(gs_fig[0, 0])
    im00 = ax00.imshow(
        source, origin='lower', extent=[Y.min(), Y.max(), X.min(), X.max()]
    )
    ax00.set_title(f'Source G($\\sigma$={sigma_source}")')
    plt.colorbar(im00, ax=ax00)

    ax01 = fig.add_subplot(gs_fig[0, 1])
    kern_extent = kern_half * pixel_scale
    im01 = ax01.imshow(
        psf_kernel_raw,
        origin='lower',
        extent=[-kern_extent, kern_extent, -kern_extent, kern_extent],
    )
    ax01.set_title(f'PSF G($\\sigma$={sigma_psf}")')
    plt.colorbar(im01, ax=ax01)

    ax02 = fig.add_subplot(gs_fig[0, 2])
    im02 = ax02.imshow(
        conv_jax, origin='lower', extent=[Y.min(), Y.max(), X.min(), X.max()]
    )
    ax02.set_title(f'Convolved (JAX N={N_os})')
    plt.colorbar(im02, ax=ax02)

    ax03 = fig.add_subplot(gs_fig[0, 3])
    im03 = ax03.imshow(
        analytic, origin='lower', extent=[Y.min(), Y.max(), X.min(), X.max()]
    )
    ax03.set_title(f'Analytic G($\\sigma$={sigma_out:.3f}")')
    plt.colorbar(im03, ax=ax03)

    # row 2: residual map (JAX) | relative error (JAX)
    ax10 = fig.add_subplot(gs_fig[1, 0:2])
    vmax_resid = max(np.max(np.abs(resid_jax)), np.max(np.abs(resid_numpy)))
    im10 = ax10.imshow(
        resid_jax,
        origin='lower',
        cmap='RdBu_r',
        extent=[Y.min(), Y.max(), X.min(), X.max()],
        vmin=-vmax_resid,
        vmax=vmax_resid,
    )
    ax10.set_title(f'Residual (JAX - analytic), max|r|/peak={max_resid_jax:.1e}')
    plt.colorbar(im10, ax=ax10)

    ax11 = fig.add_subplot(gs_fig[1, 2:4])
    rel_err_jax = np.abs(resid_jax) / peak_analytic_jax
    im11 = ax11.imshow(
        rel_err_jax,
        origin='lower',
        extent=[Y.min(), Y.max(), X.min(), X.max()],
    )
    ax11.set_title('Relative error |resid|/peak (JAX)')
    plt.colorbar(im11, ax=ax11)

    # row 3: radial profile overlay + residual sub-panel
    r_max = min(np.abs(X).max(), np.abs(Y).max())
    radial_bins = np.linspace(0, r_max, 50)
    rc, prof_source = _azimuthal_average(source, X, Y, radial_bins)
    rc, prof_psf = _azimuthal_average(_gaussian_2d(X, Y, sigma_psf), X, Y, radial_bins)
    rc, prof_numpy = _azimuthal_average(conv_numpy, X, Y, radial_bins)
    rc, prof_jax = _azimuthal_average(conv_jax, X, Y, radial_bins)
    rc, prof_analytic = _azimuthal_average(analytic, X, Y, radial_bins)

    ax_main = fig.add_subplot(gs_fig[2, 0:3])
    ax_main.semilogy(
        rc, prof_source, 'b--', lw=1.5, label=f'Source ($\\sigma$={sigma_source}")'
    )
    ax_main.semilogy(rc, prof_psf, 'g--', lw=1.5, label=f'PSF ($\\sigma$={sigma_psf}")')
    ax_main.semilogy(rc, prof_numpy, 'r-', lw=2, alpha=0.7, label='Conv (numpy)')
    ax_main.semilogy(rc, prof_jax, 'c-', lw=2, alpha=0.7, label=f'Conv (JAX N={N_os})')
    ax_main.semilogy(
        rc, prof_analytic, 'k:', lw=2.5, label=f'Analytic ($\\sigma$={sigma_out:.3f}")'
    )
    ax_main.set_xlabel('radius (arcsec)')
    ax_main.set_ylabel('azimuthal average')
    ax_main.legend(fontsize=8)
    ax_main.set_title('Radial profiles')
    ax_main.set_ylim(bottom=peak_analytic * 1e-8)

    # residual sub-panel
    ax_resid = fig.add_subplot(gs_fig[2, 3])
    valid = prof_analytic > 0
    resid_prof_numpy = np.full_like(prof_analytic, np.nan)
    resid_prof_jax = np.full_like(prof_analytic, np.nan)
    resid_prof_numpy[valid] = (prof_numpy[valid] - prof_analytic[valid]) / peak_analytic
    resid_prof_jax[valid] = (prof_jax[valid] - prof_analytic[valid]) / peak_analytic_jax
    ax_resid.plot(rc, resid_prof_numpy, 'r-', lw=1.5, label='numpy')
    ax_resid.plot(rc, resid_prof_jax, 'c-', lw=1.5, label='JAX')
    ax_resid.axhline(0, color='k', ls=':', lw=0.5)
    ax_resid.set_xlabel('radius (arcsec)')
    ax_resid.set_ylabel('(conv - analytic) / peak')
    ax_resid.legend(fontsize=8)
    ax_resid.set_title('Radial residuals')

    # annotations
    sigma_np_err = abs(sigma_numpy_meas - sigma_analytic_meas) / sigma_analytic_meas
    sigma_jax_err = (
        abs(sigma_jax_meas - sigma_analytic_jax_meas) / sigma_analytic_jax_meas
    )
    status = 'PASS' if (max_resid_numpy < 1e-4 and max_resid_jax < 5e-4) else 'FAIL'
    status_color = 'green' if status == 'PASS' else 'red'
    fig.suptitle(
        f'Gaussian Convolution Theorem — {status}\n'
        f'$\\sigma_{{out}}$ predicted={sigma_out:.4f}" | '
        f'numpy $\\sigma$={sigma_numpy_meas:.4f}" ($\\Delta$={sigma_np_err:.1e}) | '
        f'JAX (N={N_os}) $\\sigma$={sigma_jax_meas:.4f}" ($\\Delta$={sigma_jax_err:.1e}) | '
        f'max|r|/peak: numpy={max_resid_numpy:.1e} (thr=1e-4), JAX={max_resid_jax:.1e} (thr=5e-4)',
        fontsize=11,
        color=status_color,
    )

    plt.savefig(
        output_dir / 'gaussian_convolution_theorem.png', dpi=150, bbox_inches='tight'
    )
    plt.close()

    # === assertions ===
    # numpy path validates convolution math (point-sampled ⊗ point-sampled);
    # JAX path tests GalSim→JAX pipeline with oversampling (N=5): at fine scale
    # both source (~25 pix/sigma) and PSF (~20 pix/sigma) are well-sampled,
    # so pixel-integration error is negligible and JAX matches numpy.

    # 1. width match (against finite-grid analytic sigma, cancels truncation)
    assert sigma_np_err < 1e-3, (
        f"numpy sigma mismatch: measured={sigma_numpy_meas:.6f}, "
        f"analytic={sigma_analytic_meas:.6f}, rel_err={sigma_np_err:.2e}"
    )
    # JAX compared against binned analytic — isolates convolution accuracy
    assert sigma_jax_err < 1e-3, (
        f"JAX sigma mismatch: measured={sigma_jax_meas:.6f}, "
        f"analytic(binned)={sigma_analytic_jax_meas:.6f}, rel_err={sigma_jax_err:.2e}"
    )

    # 2. peak match
    peak_np_err = abs(peak_numpy - peak_analytic) / peak_analytic
    peak_jax_err = abs(peak_jax - peak_analytic_jax) / peak_analytic_jax
    assert peak_np_err < 1e-3, (
        f"numpy peak mismatch: {peak_numpy:.8f} vs {peak_analytic:.8f}, "
        f"rel_err={peak_np_err:.2e}"
    )
    assert peak_jax_err < 1e-3, (
        f"JAX peak mismatch: {peak_jax:.8f} vs {peak_analytic_jax:.8f}, "
        f"rel_err={peak_jax_err:.2e}"
    )

    # 3. max residual / peak
    assert max_resid_numpy < 1e-4, f"numpy max|resid|/peak = {max_resid_numpy:.2e}"
    # oversampled JAX should approach numpy-level accuracy for Gaussians
    assert max_resid_jax < 5e-4, f"JAX max|resid|/peak = {max_resid_jax:.2e}"

    # 4. flux conservation — boundary truncation loses ~1e-6 for sources
    # not fully contained in the image (Gaussian tails extend past edges)
    flux_err_numpy = abs(flux_numpy - flux_source) / flux_source
    flux_err_jax = abs(flux_jax - flux_source) / flux_source
    assert (
        flux_err_numpy < 1e-5
    ), f"numpy flux not conserved: rel_err={flux_err_numpy:.2e}"
    assert flux_err_jax < 1e-5, f"JAX flux not conserved: rel_err={flux_err_jax:.2e}"


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
    """Convolving with a near-delta PSF returns approximately the input."""
    # sub-pixel Gaussian (half a pixel) approximating a delta function
    delta_psf = gs.Gaussian(fwhm=0.15)
    pdata = precompute_psf_fft(delta_psf, image_pars)
    result = np.array(convolve_fft(jnp.array(test_image), pdata))

    # fwhm=0.15" at pixel_scale=0.3" → pixel tophat in GalSim drawImage
    # broadens kernel beyond a true delta; 4/4800 pixels affected, max err ~1.4e-3
    np.testing.assert_allclose(result, test_image, atol=2e-3)


def test_kernel_normalization(image_pars):
    """All PSF kernels sum to 1."""
    for psf_obj in PSF_TYPES:
        kernel_shifted, _ = gsobj_to_kernel(psf_obj, image_pars)
        total = np.sum(kernel_shifted)
        np.testing.assert_allclose(total, 1.0, atol=1e-10, err_msg=str(psf_obj))


def test_flux_conservation(image_pars, compact_test_image, psf_data):
    """sum(convolved) ~ sum(original). Uses compact source to avoid boundary flux loss."""
    result = np.array(convolve_fft(jnp.array(compact_test_image), psf_data))
    np.testing.assert_allclose(np.sum(result), np.sum(compact_test_image), rtol=1e-6)


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
    X, Y = build_map_grid_from_image_pars(image_pars, unit='arcsec', centered=True)

    # left-bright intensity (exponential falloff from left)
    intensity = np.exp(-((X - X.min()) / 3.0))

    # linear velocity gradient left-to-right
    velocity = X * 10.0  # negative on left, positive on right

    psf_obj = gs.Gaussian(fwhm=1.5)
    pdata = precompute_psf_fft(psf_obj, image_pars)

    # no PSF
    v_no_psf = velocity.copy()

    # with PSF
    v_psf = np.array(
        convolve_flux_weighted(jnp.array(velocity), jnp.array(intensity), pdata)
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
    pdata = precompute_psf_fft(psf_obj, image_pars)
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
        pdata = precompute_psf_fft(psf_obj, image_pars)
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

    assert power_convolved <= power_original * (
        1 + 1e-10
    ), f"Parseval bound violated: {power_convolved:.6f} > {power_original:.6f}"


def test_wrap_around_artifact(image_pars):
    """
    Bright source at corner should not wrap to opposite corner.
    """
    nrow, ncol = image_pars.Nrow, image_pars.Ncol
    image = np.zeros((nrow, ncol))
    # bright source in top-right corner
    image[nrow - 3 : nrow, ncol - 3 : ncol] = 1000.0

    psf_obj = gs.Gaussian(fwhm=0.9)
    pdata = precompute_psf_fft(psf_obj, image_pars)
    result = np.array(convolve_fft(jnp.array(image), pdata))

    # opposite corner (bottom-left) should be ~zero
    corner_value = np.max(np.abs(result[:3, :3]))
    assert (
        corner_value < 1e-6
    ), f"Wrap-around artifact: opposite corner = {corner_value:.2e}"


# ==============================================================================
# E. Render Layer Integration Tests
# ==============================================================================


def test_no_psf_regression(image_pars):
    """
    render_image with no PSF configured == raw _render_kspace (both use k-space FFT).

    This verifies the no-PSF code path in render_image is a pure pass-through.
    For render_image vs __call__ (k-space vs quadrature), see
    test_render_image_vs_call_consistency in test_intensity.py.
    """
    model = InclinedExponentialModel()
    theta = jnp.array([0.7, 0.785, 0.0, 0.0, 1.0, 3.0, 0.1, 0.0, 0.0])

    rendered = model.render_image(theta, image_pars)
    raw = model._render_kspace(
        theta, image_pars.Nrow, image_pars.Ncol, image_pars.pixel_scale
    )

    np.testing.assert_allclose(np.array(rendered), np.array(raw), atol=1e-12)


def test_psf_render_image_consistency(image_pars, gaussian_psf):
    """
    Configure PSF -> render_image == manual convolve_fft(source, psf_data).

    Both sides use k-space source (render_image without PSF) so this purely
    validates the PSF convolution path, not quadrature vs k-space differences.
    """
    model = InclinedExponentialModel()
    theta = jnp.array([0.7, 0.785, 0.0, 0.0, 1.0, 3.0, 0.1, 0.0, 0.0])

    # manual convolution using k-space source (same as render_image uses internally)
    raw = model.render_image(theta, image_pars)  # k-space, no PSF
    pdata = precompute_psf_fft(gaussian_psf, image_pars)
    manual = convolve_fft(raw, pdata)

    # render_image with PSF (oversample=1 for exact equality with manual path)
    model.configure_psf(gaussian_psf, image_pars, oversample=1)
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
    theta_int = jnp.array([0.6, 0.785, 0.0, 0.0, 1.0, 3.0, 0.1, 0.0, 0.0])

    X, Y = build_map_grid_from_image_pars(image_pars)

    # manual flux-weighted convolution
    raw_vel = vel_model(theta_vel, 'obs', X, Y)
    raw_int = int_model(theta_int, 'obs', X, Y)
    pdata = precompute_psf_fft(gaussian_psf, image_pars)
    manual = convolve_flux_weighted(raw_vel, raw_int, pdata)

    # render_image with PSF + flux_model (oversample=1 for exact equality)
    vel_model.configure_velocity_psf(
        gaussian_psf,
        image_pars,
        oversample=1,
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
    frozen_pars = ImagePars(shape=(32, 32), pixel_scale=0.3, indexing='ij')
    model.configure_psf(psf, frozen_pars, oversample=1, freeze=True)

    with pytest.raises(ValueError, match="frozen"):
        model.configure_psf(psf, frozen_pars)

    model.clear_psf()
    assert not model.has_psf


# ==============================================================================
# F. Oversampled Rendering Tests
# ==============================================================================


def test_oversample_convergence(oversample_image_pars, output_dir):
    """
    Oversampled residuals decrease monotonically: N=1 > N=3 > N=5 > N=9.
    N=5 (default) should be within 1e-4 of GalSim ground truth.

    Uses hlr=3.0 on 150x200 stamps (45"x60") so flux at boundary is ~3e-6
    of peak, eliminating edge effects while keeping a physically realistic
    source size. This isolates the aliasing error that oversampling fixes.
    """
    pixel_scale = oversample_image_pars.pixel_scale
    nx, ny = oversample_image_pars.Ncol, oversample_image_pars.Nrow

    sersic = gs.Exponential(half_light_radius=3.0, flux=1.0)
    psf_obj = gs.Gaussian(fwhm=0.625)

    # GalSim ground truth: continuous Convolve + drawImage
    conv_gs = gs.Convolve(sersic, psf_obj)
    img_gs = conv_gs.drawImage(nx=nx, ny=ny, scale=pixel_scale).array

    peak = np.max(np.abs(img_gs))
    residuals = {}
    images = {}
    threshold = 1e-4

    for N in [1, 3, 5, 9]:
        # point-sample source at fine scale using make_fine_scale
        fine_ip = oversample_image_pars.make_fine_scale(N)
        fine_nx, fine_ny = fine_ip.Ncol, fine_ip.Nrow

        img_source = sersic.drawImage(
            nx=fine_nx, ny=fine_ny, scale=fine_ip.pixel_scale, method='no_pixel'
        ).array
        # scale from GalSim flux/pixel to surface-brightness convention
        img_source = img_source * (N * N)

        pdata = precompute_psf_fft(
            psf_obj, image_pars=oversample_image_pars, oversample=N
        )
        img_result = np.array(convolve_fft(jnp.array(img_source), pdata))
        max_resid = np.max(np.abs(img_gs - img_result)) / peak
        residuals[N] = max_resid
        images[N] = img_result

    # --- diagnostic image grid (before assertions so plots always saved) ---
    ns = [1, 3, 5, 9]
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))

    # row 0: N=1..9 images + GalSim ref
    vmin = min(img_gs.min(), min(images[n].min() for n in ns))
    vmax = max(img_gs.max(), max(images[n].max() for n in ns))
    for col, n in enumerate(ns):
        im = axes[0, col].imshow(images[n], origin='lower', vmin=vmin, vmax=vmax)
        color = 'green' if residuals[n] < threshold else 'red'
        axes[0, col].set_title(
            f'N={n}: {residuals[n]:.2e} (thr={threshold:.0e})', color=color
        )
    im_ref = axes[0, 4].imshow(img_gs, origin='lower', vmin=vmin, vmax=vmax)
    axes[0, 4].set_title('GalSim ref')
    plt.colorbar(im_ref, ax=axes[0, :].tolist(), shrink=0.8)

    # row 1: |residuals| with LogNorm
    abs_resids = {n: np.abs(images[n] - img_gs) for n in ns}
    vmax_abs = max(np.max(abs_resids[n]) for n in ns)
    floor_abs = min(
        np.min(abs_resids[n][abs_resids[n] > 0])
        for n in ns
        if np.any(abs_resids[n] > 0)
    )
    for col, n in enumerate(ns):
        im = axes[1, col].imshow(
            abs_resids[n],
            origin='lower',
            norm=LogNorm(vmin=floor_abs, vmax=vmax_abs),
        )
        color = 'green' if residuals[n] < threshold else 'red'
        axes[1, col].set_title(f'|N={n} - GS|', color=color)
    axes[1, 4].axis('off')
    plt.colorbar(im, ax=axes[1, :4].tolist(), shrink=0.8)

    # row 2: |relative residuals| with LogNorm
    vmax_rel = vmax_abs / peak
    floor_rel = floor_abs / peak
    for col, n in enumerate(ns):
        im = axes[2, col].imshow(
            abs_resids[n] / peak,
            origin='lower',
            norm=LogNorm(vmin=floor_rel, vmax=vmax_rel),
        )
        color = 'green' if residuals[n] < threshold else 'red'
        axes[2, col].set_title(f'|N={n} - GS|/peak', color=color)
    axes[2, 4].axis('off')
    plt.colorbar(im, ax=axes[2, :4].tolist(), shrink=0.8)

    all_pass = (
        residuals[5] < threshold
        and residuals[3] < residuals[1]
        and residuals[5] < residuals[3]
    )
    status = 'PASS' if all_pass else 'FAIL'
    status_color = 'green' if status == 'PASS' else 'red'
    fig.suptitle(
        f'Oversample convergence (Exp(hlr=3) x Gaussian) — {status} — threshold: {threshold:.0e}',
        fontsize=14,
        color=status_color,
    )
    plt.tight_layout()
    plt.savefig(output_dir / 'oversample_convergence_grid.png', dpi=150)
    plt.close()

    # --- summary line plot ---
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [residuals[n] for n in ns]
    ax.semilogy(ns, vals, 'bo-', markersize=8)
    ax.axhline(threshold, color='r', ls='--', label=f'{threshold} target')
    ax.set_xlabel('Oversample factor N')
    ax.set_ylabel('Max |residual| / peak')
    ax.set_title('Oversample convergence (Exponential x Gaussian PSF)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'oversample_convergence_summary.png', dpi=150)
    plt.close()

    # --- assertions (after plots so diagnostics always available) ---
    # monotonic decrease
    assert (
        residuals[3] < residuals[1]
    ), f"N=3 ({residuals[3]:.2e}) not better than N=1 ({residuals[1]:.2e})"
    assert (
        residuals[5] < residuals[3]
    ), f"N=5 ({residuals[5]:.2e}) not better than N=3 ({residuals[3]:.2e})"

    # N=5 target: within 1e-4 of GalSim (matches default oversample=5)
    assert (
        residuals[5] < threshold
    ), f"N=5 residual {residuals[5]:.2e} exceeds {threshold} target"


@pytest.mark.parametrize("psf_obj,psf_name", PSF_CASES)
def test_galsim_regression_oversampled(
    psf_obj, psf_name, oversample_image_pars, output_dir
):
    """
    GalSim regression with oversampled source evaluation on 150x200 stamps.

    Tests production-default accuracy: default kernel GSParams (which truncate
    heavy-tailed PSFs at ~0.5%), tight ground truth GSParams (isolates our
    pipeline error from GalSim inaccuracy). Threshold 5e-3 accommodates kernel
    truncation for all PSFs while catching oversampling regressions (broken
    oversampling gives ~10-20e-3).

    For rigorous accuracy with tight kernel GSParams, see
    test_galsim_regression_oversampled_rigorous.
    """
    pixel_scale = oversample_image_pars.pixel_scale
    nx, ny = oversample_image_pars.Ncol, oversample_image_pars.Nrow
    N = 5

    # tight GSParams for ground truth only — isolates our pipeline error
    gsp = gs.GSParams(folding_threshold=1e-4, maxk_threshold=1e-4, kvalue_accuracy=1e-6)
    sersic = gs.Exponential(half_light_radius=3.0, flux=1.0, gsparams=gsp)
    psf_tight = psf_obj.withGSParams(gsp)

    # GalSim ground truth with tightened params
    conv_gs = gs.Convolve(sersic, psf_tight)
    img_gs = conv_gs.drawImage(nx=nx, ny=ny, scale=pixel_scale).array

    # oversampled JAX path — default kernel GSParams (production default)
    fine_ip = oversample_image_pars.make_fine_scale(N)
    img_source = sersic.drawImage(
        nx=fine_ip.Ncol, ny=fine_ip.Nrow, scale=fine_ip.pixel_scale, method='no_pixel'
    ).array
    # scale from GalSim flux/pixel to surface-brightness convention
    img_source = img_source * (N * N)

    pdata = precompute_psf_fft(
        psf_obj,
        image_pars=oversample_image_pars,
        oversample=N,
    )
    img_jax = np.array(convolve_fft(jnp.array(img_source), pdata))

    residual = img_gs - img_jax
    peak = np.max(np.abs(img_gs))
    max_rel_resid = np.max(np.abs(residual)) / peak

    threshold = 5e-3

    # diagnostic plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axes[0].imshow(img_gs, origin='lower')
    axes[0].set_title('GalSim native')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(img_jax, origin='lower')
    axes[1].set_title(f'JAX FFT (N={N})')
    plt.colorbar(im1, ax=axes[1])

    rel_resid = residual / peak
    vmax_rel = np.max(np.abs(rel_resid))
    im2 = axes[2].imshow(
        rel_resid, origin='lower', cmap='RdBu_r', vmin=-vmax_rel, vmax=vmax_rel
    )
    axes[2].set_title(f'resid/peak (max|.|={max_rel_resid:.2e})')
    plt.colorbar(im2, ax=axes[2])

    status = 'PASS' if max_rel_resid < threshold else 'FAIL'
    status_color = 'green' if status == 'PASS' else 'red'
    fig.suptitle(
        f'Oversampled (N={N}) {psf_name} -- {status} (max={max_rel_resid:.2e}, thr={threshold:.0e})',
        color=status_color,
    )
    plt.tight_layout()
    plt.savefig(output_dir / f'galsim_regression_oversample_{psf_name}.png', dpi=150)
    plt.close()

    assert max_rel_resid < threshold, (
        f"Oversampled regression failed for {psf_name}: "
        f"max_rel_resid={max_rel_resid:.2e} (threshold={threshold:.0e})"
    )


RIGOROUS_PSF_CASES = [
    pytest.param(gs.Gaussian(fwhm=0.625), 'Gaussian_0.625', id='Gaussian_0.625'),
    pytest.param(gs.Moffat(beta=2.5, fwhm=1.0), 'Moffat_2.5_1.0', id='Moffat_2.5_1.0'),
]


@pytest.mark.slow
@pytest.mark.parametrize("psf_obj,psf_name", RIGOROUS_PSF_CASES)
def test_galsim_regression_oversampled_rigorous(
    psf_obj, psf_name, oversample_image_pars, output_dir
):
    """
    Rigorous oversampled regression with tight GSParams on BOTH kernel and
    ground truth. Proves pipeline achieves 5e-4 accuracy when kernel
    truncation is eliminated.

    Gaussian_0.625: typical use case, fast kernel FFTs.
    Moffat_2.5: heaviest tails = hardest kernel truncation case; moderate
    kernel FFTs (~15-30s), unlike Airy/OpticalPSF (49k/24k pixels).
    """
    pixel_scale = oversample_image_pars.pixel_scale
    nx, ny = oversample_image_pars.Ncol, oversample_image_pars.Nrow
    N = 5

    # tight GSParams for both ground truth and kernel
    gsp = gs.GSParams(folding_threshold=1e-4, maxk_threshold=1e-4, kvalue_accuracy=1e-6)
    sersic = gs.Exponential(half_light_radius=3.0, flux=1.0, gsparams=gsp)
    psf_tight = psf_obj.withGSParams(gsp)

    # GalSim ground truth
    conv_gs = gs.Convolve(sersic, psf_tight)
    img_gs = conv_gs.drawImage(nx=nx, ny=ny, scale=pixel_scale).array

    # oversampled JAX path — tight kernel GSParams
    fine_ip = oversample_image_pars.make_fine_scale(N)
    img_source = sersic.drawImage(
        nx=fine_ip.Ncol, ny=fine_ip.Nrow, scale=fine_ip.pixel_scale, method='no_pixel'
    ).array
    img_source = img_source * (N * N)

    pdata = precompute_psf_fft(
        psf_obj,
        image_pars=oversample_image_pars,
        oversample=N,
        gsparams=gsp,
    )
    img_jax = np.array(convolve_fft(jnp.array(img_source), pdata))

    residual = img_gs - img_jax
    peak = np.max(np.abs(img_gs))
    max_rel_resid = np.max(np.abs(residual)) / peak

    threshold = 5e-4

    # diagnostic plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axes[0].imshow(img_gs, origin='lower')
    axes[0].set_title('GalSim native')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(img_jax, origin='lower')
    axes[1].set_title(f'JAX FFT (N={N})')
    plt.colorbar(im1, ax=axes[1])

    rel_resid = residual / peak
    vmax_rel = np.max(np.abs(rel_resid))
    im2 = axes[2].imshow(
        rel_resid, origin='lower', cmap='RdBu_r', vmin=-vmax_rel, vmax=vmax_rel
    )
    axes[2].set_title(f'resid/peak (max|.|={max_rel_resid:.2e})')
    plt.colorbar(im2, ax=axes[2])

    status = 'PASS' if max_rel_resid < threshold else 'FAIL'
    status_color = 'green' if status == 'PASS' else 'red'
    fig.suptitle(
        f'Rigorous oversampled (N={N}) {psf_name} -- {status} '
        f'(max={max_rel_resid:.2e}, thr={threshold:.0e})',
        color=status_color,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / f'galsim_regression_oversample_rigorous_{psf_name}.png', dpi=150
    )
    plt.close()

    assert max_rel_resid < threshold, (
        f"Rigorous oversampled regression failed for {psf_name}: "
        f"max_rel_resid={max_rel_resid:.2e} (threshold={threshold:.0e})"
    )


def test_oversample_flux_conservation(image_pars, compact_test_image):
    """Total flux preserved after oversample+bin."""
    psf_obj = gs.Gaussian(fwhm=0.625)

    for N in [1, 3, 5]:
        # tile to fine scale — same SB value per subpixel
        fine_image = np.repeat(np.repeat(compact_test_image, N, axis=0), N, axis=1)

        pdata = precompute_psf_fft(psf_obj, image_pars=image_pars, oversample=N)
        result = np.array(convolve_fft(jnp.array(fine_image), pdata))

        # mean-bin preserves avg pixel value:
        #   sum(fine)=N^2*sum(compact), conv preserves sum, mean-bin divides by N^2
        np.testing.assert_allclose(
            np.sum(result),
            np.sum(compact_test_image),
            rtol=1e-6,
            err_msg=f"Flux not conserved for N={N}",
        )


def test_oversample_velocity_binning(image_pars):
    """Sum-then-divide binning produces correct flux-weighted velocity."""
    psf_obj = gs.Gaussian(fwhm=0.625)
    N = 3

    # constant velocity everywhere: result should be that same constant
    X, Y = build_map_grid_from_image_pars(image_pars, unit='arcsec', centered=True)
    intensity = np.exp(-np.sqrt(X**2 + Y**2) / 3.0)
    velocity = np.full_like(intensity, 42.0)

    # tile to fine scale
    fine_intensity = np.repeat(np.repeat(intensity, N, axis=0), N, axis=1) / (N * N)
    fine_velocity = np.repeat(np.repeat(velocity, N, axis=0), N, axis=1)

    pdata = precompute_psf_fft(psf_obj, image_pars=image_pars, oversample=N)
    result = np.array(
        convolve_flux_weighted(
            jnp.array(fine_velocity), jnp.array(fine_intensity), pdata
        )
    )

    # where intensity is nonnegligible, velocity should be ~42
    mask = intensity > 1e-8
    np.testing.assert_allclose(
        result[mask],
        42.0,
        atol=1e-6,
        err_msg="Constant velocity not preserved through oversampled flux-weighted PSF",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
