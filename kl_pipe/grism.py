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
       resolution and no fine-grid sinc is required. The sinc has the
       COARSE pixel side, so after the IFFT each fine cell holds the
       coarse-box-AVERAGED SB centered on that cell -- the sinc alone
       performs the full coarse-pixel integration.
    2. Reads out by SAMPLING the box-averaged field at each coarse pixel
       center: the center fine cell of every N×N block, which for odd N
       lies exactly on the coarse pixel center. The readout is a sample,
       not an average -- mean-binning the block would convolve the already
       box-averaged field with a second coarse-pixel-wide box, biasing
       peaks of compact sources low by several percent while conserving
       flux.
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
        Spatial oversampling factor of the input. Must be odd when > 1
        (even factors have no fine cell centered on a coarse pixel).
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

    if oversample % 2 == 0:
        raise ValueError(
            f"oversample must be odd for the post-dispersion pixel-response "
            f"readout, got {oversample}. The coarse-pixel sinc already "
            f"performs the pixel integration, so the readout samples the "
            f"box-averaged field at each coarse pixel center; with an even "
            f"oversample no fine cell is centered on a coarse pixel."
        )

    Nrow_c, Ncol_c = coarse_shape
    N = oversample
    if dispersed.shape != (Nrow_c * N, Ncol_c * N):
        raise ValueError(
            f"dispersed shape {dispersed.shape} does not match "
            f"coarse_shape × oversample = ({Nrow_c * N}, {Ncol_c * N})."
        )

    # FFT -> sinc multiply -> IFFT at fine grid (still in SB units); each
    # fine cell now holds the coarse-box-averaged SB centered on that cell
    img_fft = jnp.fft.fft2(dispersed)
    pixel_integrated_fine = jnp.fft.ifft2(img_fft * pixel_response_fft).real

    # sample the box-averaged SB at coarse pixel centers (center fine cell
    # of each N×N block; exact for odd N), then convert to flux/pixel
    c = N // 2
    sb_coarse = pixel_integrated_fine[c::N, c::N]
    return sb_coarse * coarse_area
