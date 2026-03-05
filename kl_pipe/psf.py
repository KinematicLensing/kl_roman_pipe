"""
PSF convolution for kinematic lensing models.

Provides JAX-compatible FFT convolution for applying point-spread functions
to model-rendered images. PSF kernels are pre-computed from GalSim GSObjects
and stored as FFT-ready arrays for efficient repeated convolution during
likelihood evaluation.

Requires JAX float64 mode: ``jax.config.update("jax_enable_x64", True)``
must be called before any PSF operations. This is enforced at PSFData
creation time.

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

import jax
import jax.numpy as jnp
import numpy as np
from scipy.fft import next_fast_len

if TYPE_CHECKING:
    import galsim
    from kl_pipe.parameters import ImagePars


@dataclass(frozen=True)
class PSFData:
    """Pre-computed PSF arrays for JAX FFT convolution."""

    kernel_fft: jnp.ndarray  # pre-FFT'd kernel (padded)
    padded_shape: tuple  # (Ny_pad, Nx_pad) — fine-scale when oversampled
    original_shape: tuple  # (Ny, Nx) — fine-scale when oversampled
    oversample: int  # oversampling factor (1 = no oversampling)
    coarse_shape: tuple  # output shape (Ny, Nx) — same as original_shape when N=1


def _psf_flatten(p):
    return (p.kernel_fft,), (
        p.padded_shape,
        p.original_shape,
        p.oversample,
        p.coarse_shape,
    )


def _psf_unflatten(aux, children):
    return PSFData(
        kernel_fft=children[0],
        padded_shape=aux[0],
        original_shape=aux[1],
        oversample=aux[2],
        coarse_shape=aux[3],
    )


jax.tree_util.register_pytree_node(PSFData, _psf_flatten, _psf_unflatten)


# ==============================================================================
# Kernel preparation (runs once, before JIT)
# ==============================================================================


def gsobj_to_kernel(
    gsobj: 'galsim.GSObject',
    image_pars: 'ImagePars' = None,
    *,
    image_shape: Tuple[int, int] = None,
    pixel_scale: float = None,
    gsparams: 'galsim.GSParams' = None,
) -> Tuple[np.ndarray, tuple]:
    """
    Convert galsim.GSObject to a normalized, FFT-ready numpy kernel.

    Two calling conventions (image_pars is preferred):
    - gsobj_to_kernel(gsobj, image_pars=image_pars)
    - gsobj_to_kernel(gsobj, image_shape=(Ny, Nx), pixel_scale=scale)

    Parameters
    ----------
    gsobj : galsim.GSObject
        PSF profile.
    image_pars : ImagePars, optional
        Image parameters. Extracts (Nrow, Ncol) and pixel_scale internally.
    image_shape : tuple, optional
        (Ny, Nx) of the data images this kernel will convolve.
    pixel_scale : float, optional
        arcsec/pixel.
    gsparams : galsim.GSParams, optional
        Override GSParams for kernel rendering. Controls truncation radius
        (folding_threshold) and accuracy. Default None uses the GSObject's
        own GSParams.

    Returns
    -------
    kernel_shifted : np.ndarray
        ifftshift'd, zero-padded kernel ready for FFT.
    padded_shape : tuple
        (Ny_pad, Nx_pad) after padding for linear (non-circular) convolution.
    """
    import galsim as gs

    if gsparams is not None:
        gsobj = gsobj.withGSParams(gsparams)

    # extract from image_pars or use explicit values
    if image_pars is not None:
        image_shape = (image_pars.Nrow, image_pars.Ncol)
        pixel_scale = image_pars.pixel_scale
    elif image_shape is None or pixel_scale is None:
        raise ValueError("Provide image_pars OR both image_shape and pixel_scale")

    # determine kernel rendering size from GalSim
    kern_size = gsobj.getGoodImageSize(pixel_scale)
    if kern_size < 3:
        raise ValueError(
            f"PSF too small relative to pixel_scale={pixel_scale}: "
            f"GalSim computed kern_size={kern_size}. "
            f"Increase PSF size or decrease pixel_scale."
        )
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

    # zero-pad kernel then roll center to (0,0) for FFT convention.
    # np.roll correctly wraps negative-offset values to the end of
    # the padded array, unlike ifftshift+place which mispositions them.
    padded_kernel = np.zeros(padded_shape, dtype=np.float64)
    padded_kernel[: kernel.shape[0], : kernel.shape[1]] = kernel
    ky_half = kernel.shape[0] // 2
    kx_half = kernel.shape[1] // 2
    padded_kernel = np.roll(padded_kernel, (-ky_half, -kx_half), axis=(0, 1))

    return padded_kernel, padded_shape


def precompute_psf_fft(
    gsobj: 'galsim.GSObject',
    image_pars: 'ImagePars' = None,
    *,
    image_shape: Tuple[int, int] = None,
    pixel_scale: float = None,
    oversample: int = 1,
    gsparams: 'galsim.GSParams' = None,
) -> PSFData:
    """
    Full PSF setup: GSObject -> JAX-ready PSFData.

    Calls gsobj_to_kernel, converts to jnp.array, pre-computes FFT.

    Two calling conventions (image_pars is preferred):
    - precompute_psf_fft(gsobj, image_pars=image_pars)
    - precompute_psf_fft(gsobj, image_shape=(Ny, Nx), pixel_scale=scale)

    Parameters
    ----------
    gsobj : galsim.GSObject
        PSF profile.
    image_pars : ImagePars, optional
        Image parameters. Extracts (Nrow, Ncol) and pixel_scale internally.
    image_shape : tuple, optional
        (Ny, Nx) of the data images.
    pixel_scale : float, optional
        arcsec/pixel.
    oversample : int, optional
        Oversampling factor for source evaluation. When > 1, the PSF kernel
        is rendered at finer pixel scale (pixel_scale / oversample) on a
        larger grid (Ny*oversample, Nx*oversample). Must be a positive odd
        integer to avoid centroid-shift artifacts. Default is 1 (no oversampling).
    gsparams : galsim.GSParams, optional
        Override GSParams for kernel rendering. Controls truncation radius
        (folding_threshold) and accuracy. Passed through to gsobj_to_kernel.

    Returns
    -------
    PSFData
        Pre-computed PSF data for use with convolve_fft.
    """
    if not jax.config.jax_enable_x64:
        raise ValueError(
            "JAX float64 mode required for PSF convolution. "
            "Call jax.config.update('jax_enable_x64', True) before using PSF functions."
        )

    if oversample < 1 or oversample % 2 == 0:
        raise ValueError(f"oversample must be a positive odd integer, got {oversample}")

    # extract from image_pars or use explicit values
    if image_pars is not None:
        image_shape = (image_pars.Nrow, image_pars.Ncol)
        pixel_scale = image_pars.pixel_scale
    elif image_shape is None or pixel_scale is None:
        raise ValueError("Provide image_pars OR both image_shape and pixel_scale")

    coarse_shape = image_shape

    if oversample > 1:
        # fine-scale parameters for oversampled rendering
        fine_shape = (image_shape[0] * oversample, image_shape[1] * oversample)
        fine_pixel_scale = pixel_scale / oversample
    else:
        fine_shape = image_shape
        fine_pixel_scale = pixel_scale

    kernel_shifted, padded_shape = gsobj_to_kernel(
        gsobj,
        image_shape=fine_shape,
        pixel_scale=fine_pixel_scale,
        gsparams=gsparams,
    )
    kernel_fft = jnp.fft.fft2(jnp.array(kernel_shifted))

    return PSFData(
        kernel_fft=kernel_fft,
        padded_shape=padded_shape,
        original_shape=fine_shape,
        oversample=oversample,
        coarse_shape=coarse_shape,
    )


# ==============================================================================
# JAX convolution (JIT + autodiff compatible)
# ==============================================================================


def _convolve_fft_raw(image: jnp.ndarray, psf_data: PSFData) -> jnp.ndarray:
    """
    Inner FFT convolution — returns fine-scale result without binning.

    Parameters
    ----------
    image : jnp.ndarray
        2D image, shape == psf_data.original_shape (fine-scale when oversampled).
    psf_data : PSFData
        Pre-computed PSF.

    Returns
    -------
    jnp.ndarray
        Convolved image at fine-scale, shape == psf_data.original_shape.
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


def convolve_fft(image: jnp.ndarray, psf_data: PSFData) -> jnp.ndarray:
    """
    2D FFT convolution with optional oversampled binning.

    When psf_data.oversample > 1, the input image must be at fine-scale
    (shape == original_shape == coarse_shape * oversample). The result
    is binned down to coarse_shape by averaging N×N blocks.

    Fully JAX JIT and autodiff compatible.

    Parameters
    ----------
    image : jnp.ndarray
        2D image to convolve, shape == psf_data.original_shape.
    psf_data : PSFData
        Pre-computed PSF from precompute_psf_fft.

    Returns
    -------
    jnp.ndarray
        Convolved image, shape == psf_data.coarse_shape.
    """
    if image.shape != psf_data.original_shape:
        raise ValueError(
            f"Image shape {image.shape} != PSFData.original_shape {psf_data.original_shape}. "
            f"Recompute PSFData with matching image_shape."
        )

    result = _convolve_fft_raw(image, psf_data)

    if psf_data.oversample > 1:
        N = psf_data.oversample
        Ny_c, Nx_c = psf_data.coarse_shape
        result = result.reshape(Ny_c, N, Nx_c, N).mean(axis=(1, 3))

    return result


def convolve_flux_weighted(
    velocity: jnp.ndarray,
    intensity: jnp.ndarray,
    psf_data: PSFData,
    epsilon: float = 1e-10,
) -> jnp.ndarray:
    """
    Flux-weighted velocity PSF convolution.

    v_obs = Conv(I * v, PSF) / max(Conv(I, PSF), epsilon)

    When oversampled, numerator and denominator are convolved at fine scale,
    then binned by sum (not mean) before division. This correctly approximates
    the pixel-integrated flux-weighted velocity.

    Parameters
    ----------
    velocity : jnp.ndarray
        2D velocity map (fine-scale when oversampled).
    intensity : jnp.ndarray
        2D intensity map (fine-scale when oversampled).
    psf_data : PSFData
        Pre-computed PSF.
    epsilon : float
        Floor to prevent division by zero / NaN gradients.

    Returns
    -------
    jnp.ndarray
        Flux-weighted, PSF-convolved velocity map, shape == psf_data.coarse_shape.
    """
    conv_iv = _convolve_fft_raw(intensity * velocity, psf_data)
    conv_i = _convolve_fft_raw(intensity, psf_data)

    if psf_data.oversample > 1:
        N = psf_data.oversample
        Ny_c, Nx_c = psf_data.coarse_shape
        num = conv_iv.reshape(Ny_c, N, Nx_c, N).sum(axis=(1, 3))
        den = conv_i.reshape(Ny_c, N, Nx_c, N).sum(axis=(1, 3))
        return num / jnp.maximum(den, epsilon)
    else:
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
    source_image_pars: 'ImagePars',
    target_image_pars: 'ImagePars' = None,
    *,
    target_shape: Tuple[int, int] = None,
    target_pixel_scale: float = None,
) -> np.ndarray:
    """
    Resample image from source grid to target grid using GalSim InterpolatedImage.

    Called at configure time (before JIT), so GalSim is fine here.

    Two calling conventions (target_image_pars is preferred):
    - _resample_to_grid(image, source_pars, target_image_pars=target_pars)
    - _resample_to_grid(image, source_pars, target_shape=(Ny, Nx), target_pixel_scale=scale)

    Parameters
    ----------
    image : np.ndarray
        Source image.
    source_image_pars : ImagePars
        Image parameters describing the source grid.
    target_image_pars : ImagePars, optional
        Image parameters for target grid. Extracts (Nrow, Ncol) and pixel_scale.
    target_shape : tuple, optional
        (Ny, Nx) of target grid.
    target_pixel_scale : float, optional
        arcsec/pixel of target grid.

    Returns
    -------
    np.ndarray
        Resampled image on target grid.
    """
    import galsim

    # extract from target_image_pars or use explicit values
    if target_image_pars is not None:
        target_shape = (target_image_pars.Nrow, target_image_pars.Ncol)
        target_pixel_scale = target_image_pars.pixel_scale
    elif target_shape is None or target_pixel_scale is None:
        raise ValueError(
            "Provide target_image_pars OR both target_shape and target_pixel_scale"
        )

    gs_image = galsim.Image(
        np.asarray(image, dtype=np.float64), scale=source_image_pars.pixel_scale
    )
    interp = galsim.InterpolatedImage(gs_image)
    # method='no_pixel' because source data already has pixel integration
    target = interp.drawImage(
        nx=target_shape[1],
        ny=target_shape[0],
        scale=target_pixel_scale,
        method='no_pixel',
    )

    return target.array
