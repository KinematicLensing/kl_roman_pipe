"""
PSF convolution for kinematic lensing models.

Provides JAX-compatible FFT convolution for applying point-spread functions
to model-rendered images. PSF kernels are pre-computed from GalSim GSObjects
and stored as FFT-ready arrays for efficient repeated convolution during
likelihood evaluation.

Key functions:
- precompute_psf_fft: GSObject -> PSFData (one-time setup)
- convolve_fft: standard image convolution (JAX JIT + autodiff compatible)
- convolve_flux_weighted: v_obs = Conv(I*v, PSF) / Conv(I, PSF)

Numpy variants are provided for synthetic data generation (convolve_fft_numpy,
convolve_flux_weighted_numpy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import jax.numpy as jnp
import numpy as np
from scipy.fft import next_fast_len

if TYPE_CHECKING:
    import galsim


@dataclass(frozen=True)
class PSFData:
    """Pre-computed PSF arrays for JAX FFT convolution."""

    kernel_fft: jnp.ndarray  # pre-FFT'd kernel (padded)
    padded_shape: tuple  # (Ny_pad, Nx_pad)
    original_shape: tuple  # (Ny, Nx)


# ==============================================================================
# Kernel preparation (runs once, before JIT)
# ==============================================================================


def gsobj_to_kernel(
    gsobj: 'galsim.GSObject',
    image_shape: Tuple[int, int],
    pixel_scale: float,
) -> Tuple[np.ndarray, tuple]:
    """
    Convert galsim.GSObject to a normalized, FFT-ready numpy kernel.

    Parameters
    ----------
    gsobj : galsim.GSObject
        PSF profile.
    image_shape : tuple
        (Ny, Nx) of the data images this kernel will convolve.
    pixel_scale : float
        arcsec/pixel.

    Returns
    -------
    kernel_shifted : np.ndarray
        ifftshift'd, zero-padded kernel ready for FFT.
    padded_shape : tuple
        (Ny_pad, Nx_pad) after padding for linear (non-circular) convolution.
    """
    import galsim as gs

    # determine kernel rendering size from GalSim
    kern_size = gsobj.getGoodImageSize(pixel_scale)
    # ensure odd so center pixel is well-defined
    if kern_size % 2 == 0:
        kern_size += 1

    # render PSF kernel (pixel-integrated via default method)
    kern_img = gsobj.drawImage(nx=kern_size, ny=kern_size, scale=pixel_scale)
    kernel = kern_img.array.astype(np.float64)

    # normalize to unit sum
    kernel /= kernel.sum()

    # compute padded shape for linear convolution (avoid wrap-around)
    ny_pad = next_fast_len(image_shape[0] + kernel.shape[0] - 1)
    nx_pad = next_fast_len(image_shape[1] + kernel.shape[1] - 1)
    padded_shape = (ny_pad, nx_pad)

    # zero-pad kernel to padded_shape
    padded_kernel = np.zeros(padded_shape, dtype=np.float64)
    padded_kernel[: kernel.shape[0], : kernel.shape[1]] = kernel

    # ifftshift so kernel center is at (0,0) for FFT convention
    kernel_shifted = np.fft.ifftshift(padded_kernel)

    return kernel_shifted, padded_shape


def precompute_psf_fft(
    gsobj: 'galsim.GSObject',
    image_shape: Tuple[int, int],
    pixel_scale: float,
) -> PSFData:
    """
    Full PSF setup: GSObject -> JAX-ready PSFData.

    Calls gsobj_to_kernel, converts to jnp.array, pre-computes FFT.

    Parameters
    ----------
    gsobj : galsim.GSObject
        PSF profile.
    image_shape : tuple
        (Ny, Nx) of the data images.
    pixel_scale : float
        arcsec/pixel.

    Returns
    -------
    PSFData
        Pre-computed PSF data for use with convolve_fft.
    """
    kernel_shifted, padded_shape = gsobj_to_kernel(gsobj, image_shape, pixel_scale)
    kernel_fft = jnp.fft.fft2(jnp.array(kernel_shifted))

    return PSFData(
        kernel_fft=kernel_fft,
        padded_shape=padded_shape,
        original_shape=image_shape,
    )


# ==============================================================================
# JAX convolution (JIT + autodiff compatible)
# ==============================================================================


def convolve_fft(image: jnp.ndarray, psf_data: PSFData) -> jnp.ndarray:
    """
    2D FFT convolution. Fully JAX JIT and autodiff compatible.

    Parameters
    ----------
    image : jnp.ndarray
        2D image to convolve, shape == psf_data.original_shape.
    psf_data : PSFData
        Pre-computed PSF from precompute_psf_fft.

    Returns
    -------
    jnp.ndarray
        Convolved image, same shape as input.
    """
    ny, nx = psf_data.original_shape
    py, px = psf_data.padded_shape

    # zero-pad image
    padded = jnp.zeros((py, px), dtype=image.dtype)
    padded = padded.at[:ny, :nx].set(image)

    # FFT multiply IFFT
    result = jnp.fft.ifft2(jnp.fft.fft2(padded) * psf_data.kernel_fft)

    # crop to original shape and take real part
    return result[:ny, :nx].real


def convolve_flux_weighted(
    velocity: jnp.ndarray,
    intensity: jnp.ndarray,
    psf_data: PSFData,
    epsilon: float = 1e-10,
) -> jnp.ndarray:
    """
    Flux-weighted velocity PSF convolution.

    v_obs = Conv(I * v, PSF) / max(Conv(I, PSF), epsilon)

    Parameters
    ----------
    velocity : jnp.ndarray
        2D velocity map.
    intensity : jnp.ndarray
        2D intensity map (flux weighting source).
    psf_data : PSFData
        Pre-computed PSF.
    epsilon : float
        Floor to prevent division by zero / NaN gradients.

    Returns
    -------
    jnp.ndarray
        Flux-weighted, PSF-convolved velocity map.
    """
    conv_iv = convolve_fft(intensity * velocity, psf_data)
    conv_i = convolve_fft(intensity, psf_data)

    return conv_iv / jnp.maximum(conv_i, epsilon)


# ==============================================================================
# Numpy variants (for synthetic data generation)
# ==============================================================================


def convolve_fft_numpy(
    image: np.ndarray,
    kernel: np.ndarray,
    padded_shape: tuple,
) -> np.ndarray:
    """
    Numpy version of convolve_fft for synthetic data generation.

    Parameters
    ----------
    image : np.ndarray
        2D image to convolve.
    kernel : np.ndarray
        ifftshift'd, zero-padded kernel from gsobj_to_kernel.
    padded_shape : tuple
        (Ny_pad, Nx_pad).

    Returns
    -------
    np.ndarray
        Convolved image, same shape as input.
    """
    ny, nx = image.shape
    py, px = padded_shape

    # zero-pad image
    padded = np.zeros((py, px), dtype=np.float64)
    padded[:ny, :nx] = image

    # FFT multiply IFFT
    result = np.fft.ifft2(np.fft.fft2(padded) * np.fft.fft2(kernel))

    return result[:ny, :nx].real


def convolve_flux_weighted_numpy(
    velocity: np.ndarray,
    intensity: np.ndarray,
    kernel: np.ndarray,
    padded_shape: tuple,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Numpy version of convolve_flux_weighted for synthetic data generation.

    Parameters
    ----------
    velocity : np.ndarray
        2D velocity map.
    intensity : np.ndarray
        2D intensity map (flux weighting source).
    kernel : np.ndarray
        ifftshift'd, zero-padded kernel from gsobj_to_kernel.
    padded_shape : tuple
        (Ny_pad, Nx_pad).
    epsilon : float
        Floor to prevent division by zero.

    Returns
    -------
    np.ndarray
        Flux-weighted, PSF-convolved velocity map.
    """
    conv_iv = convolve_fft_numpy(intensity * velocity, kernel, padded_shape)
    conv_i = convolve_fft_numpy(intensity, kernel, padded_shape)

    return conv_iv / np.maximum(conv_i, epsilon)


# ==============================================================================
# Image resampling helper
# ==============================================================================


def _resample_to_grid(
    image: np.ndarray,
    source_image_pars,
    target_shape: Tuple[int, int],
    target_pixel_scale: float,
) -> np.ndarray:
    """
    Resample image from source grid to target grid using GalSim InterpolatedImage.

    Called at configure time (before JIT), so GalSim is fine here.

    Parameters
    ----------
    image : np.ndarray
        Source image.
    source_image_pars : ImagePars
        Image parameters describing the source grid.
    target_shape : tuple
        (Ny, Nx) of target grid.
    target_pixel_scale : float
        arcsec/pixel of target grid.

    Returns
    -------
    np.ndarray
        Resampled image on target grid.
    """
    import galsim

    gs_image = galsim.Image(np.asarray(image, dtype=np.float64), scale=source_image_pars.pixel_scale)
    interp = galsim.InterpolatedImage(gs_image)
    # method='no_pixel' because source data already has pixel integration
    target = interp.drawImage(
        nx=target_shape[1], ny=target_shape[0], scale=target_pixel_scale, method='no_pixel'
    )

    return target.array
