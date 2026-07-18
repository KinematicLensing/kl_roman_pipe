"""
Grism-specific rendering primitives.

Helpers that operate on the post-dispersion 2D image stage of the grism
pipeline. The cube-assembly + dispersion steps live in
``kl_pipe.spectral`` + ``kl_pipe.dispersion``; this module hosts the
detector-pixel-stage operations (pixel response, SB→flux/pixel conversion)
that finalize a dispersed cube into an observable.
"""

from __future__ import annotations

import jax.numpy as jnp


def _apply_post_dispersion_pixel_response(
    dispersed: jnp.ndarray,
    pixel_response_fft: jnp.ndarray,
    coarse_shape: tuple,
    oversample: int,
    coarse_pixel_scale: float,
) -> jnp.ndarray:
    """Apply BoxPixel sinc + convert SB to flux per coarse pixel.

    Pixel response (square detector top-hat) is applied once at the 2D
    observable readout stage rather than per-channel on the cube. After
    grism dispersion produces a fine-resolution 2D image in SB units
    (flux / arcsec²), this function:

    1. Multiplies by the precomputed BoxPixel sinc on the fine k-grid
       (``GrismObs.pixel_response_fft``) -- only when ``oversample > 1``;
       at ``oversample == 1`` the input is already at coarse detector
       resolution and no fine-grid sinc is required.
    2. Bins fine cells to coarse pixels (mean over each N×N block).
    3. Multiplies by ``coarse_pixel_scale**2`` to convert SB (per arcsec²)
       to flux per coarse pixel. See ``docs/units_and_conventions.md``.

    Parameters
    ----------
    dispersed : jnp.ndarray
        2D dispersed image in SB units, shape ``(Nrow*N, Ncol*N)`` for
        ``oversample=N``, or ``(Nrow, Ncol)`` for ``oversample=1``.
    pixel_response_fft : jnp.ndarray
        Precomputed BoxPixel sinc on the fine k-grid (from
        ``GrismObs.pixel_response_fft``). Unused at ``oversample == 1``.
    coarse_shape : tuple
        ``(Nrow_c, Ncol_c)`` of the coarse detector grid.
    oversample : int
        Spatial oversampling factor of the input.
    coarse_pixel_scale : float
        Coarse detector pixel scale (arcsec); used for the SB→flux/pixel
        conversion.

    Returns
    -------
    jnp.ndarray
        Coarse-pixel 2D image in flux per coarse pixel, shape ``coarse_shape``.
    """
    coarse_area = coarse_pixel_scale * coarse_pixel_scale

    if oversample <= 1:
        # input already at coarse resolution; just convert SB -> flux/pixel
        return dispersed * coarse_area

    Nrow_c, Ncol_c = coarse_shape
    N = oversample

    # FFT -> sinc multiply -> IFFT at fine grid (still in SB units)
    img_fft = jnp.fft.fft2(dispersed)
    pixel_integrated_fine = jnp.fft.ifft2(img_fft * pixel_response_fft).real

    # mean-bin SB to coarse, then multiply by coarse area to get flux/pixel.
    # Equivalent to sum-bin × fine_pixel_area = sum × (coarse/N)² , i.e.
    # mean × coarse_area.
    sb_coarse = pixel_integrated_fine.reshape(Nrow_c, N, Ncol_c, N).mean(axis=(1, 3))
    return sb_coarse * coarse_area
