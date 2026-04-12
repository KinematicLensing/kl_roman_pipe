"""
Pixel response functions for k-space rendering.

The pixel response models the detector's integration of flux over each
pixel area. In Fourier space, a square top-hat pixel of width ``w`` has
FT = sinc(kx·w/(2π)) × sinc(ky·w/(2π)), which suppresses high-frequency
power and acts as an anti-aliasing filter.

For k-space intensity rendering, the pixel response is applied as a
multiplicative factor alongside the profile FT and PSF FT:

    I_measured = IFFT( profile_FT × pixel_FT × PSF_FT )

This is mathematically exact (for a box pixel), replacing the O(N²)
spatial oversampling approximation with an O(1) multiplication.

Classes
-------
PixelResponse
    Abstract base class for pixel response functions.
BoxPixel
    Square top-hat pixel (default for all detectors). FT is a 2D sinc.

Future subclasses (not yet implemented):
- ``RomanPixel``: box pixel + interpixel capacitance (IPC) correction
  for Roman WFI H4RG-10 detectors (Kannawadi et al. 2016).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    pass


class PixelResponse(ABC):
    """Abstract pixel response function applied in k-space.

    Subclasses must implement ``ft`` (the Fourier transform of the pixel
    response, normalized to 1 at DC) and ``maxk`` (the wavenumber where
    the FT amplitude drops below a given threshold).

    The ``maxk`` value is used by the adaptive grid sizing machinery to
    determine whether the pixel response has suppressed the profile's
    high-frequency power sufficiently at the DFT Nyquist frequency.
    """

    @abstractmethod
    def ft(self, KX: jnp.ndarray, KY: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the pixel response FT on the given k-space grids.

        Parameters
        ----------
        KX, KY : jnp.ndarray
            k-space coordinate grids (rad/arcsec), from
            ``2π × fftfreq(N, d=pixel_scale)``.

        Returns
        -------
        jnp.ndarray
            Real-valued FT array, normalized so ft(0, 0) = 1.
        """
        ...

    @abstractmethod
    def maxk(self, threshold: float = 1e-3) -> float:
        """Wavenumber where the FT amplitude drops below threshold.

        Used for adaptive grid sizing: the effective maxk of a rendering
        chain is ``min(profile_maxk, pixel_maxk, psf_maxk)``.

        Parameters
        ----------
        threshold : float
            Maximum acceptable FT amplitude. Default 1e-3.

        Returns
        -------
        float
            Wavenumber in rad/arcsec.
        """
        ...


class BoxPixel(PixelResponse):
    """Square top-hat pixel response.

    Models the detector pixel as a square aperture of width
    ``pixel_scale`` in both x and y. The Fourier transform is a
    separable 2D sinc:

        ft(kx, ky) = sinc(kx · pixel_scale / (2π))
                    × sinc(ky · pixel_scale / (2π))

    where sinc is the normalized sinc: sin(πx) / (πx).

    Parameters
    ----------
    pixel_scale : float
        Pixel width in arcsec (or consistent units). Must be > 0.

    Raises
    ------
    ValueError
        If pixel_scale <= 0.
    """

    def __init__(self, pixel_scale: float):
        if pixel_scale <= 0:
            raise ValueError(f"pixel_scale must be > 0, got {pixel_scale}")
        self.pixel_scale = float(pixel_scale)

    def ft(self, KX: jnp.ndarray, KY: jnp.ndarray) -> jnp.ndarray:
        """Evaluate sinc × sinc pixel FT.

        Parameters
        ----------
        KX, KY : jnp.ndarray
            k-space grids in rad/arcsec.

        Returns
        -------
        jnp.ndarray
            sinc(kx·w/(2π)) × sinc(ky·w/(2π)), where w = pixel_scale.
        """
        w = self.pixel_scale
        return jnp.sinc(KX * w / (2.0 * jnp.pi)) * jnp.sinc(KY * w / (2.0 * jnp.pi))

    def maxk(self, threshold: float = 1e-3) -> float:
        """Wavenumber where sinc envelope drops below threshold.

        Uses the sinc envelope bound: |sinc(x)| <= 1/(π|x|) for |x| > 0.
        Solving 1/(π·x) = threshold gives x = 1/(π·threshold).
        Converting back: x = k·w/(2π), so k = 2/(threshold·w).

        Parameters
        ----------
        threshold : float
            Maximum FT amplitude. Default 1e-3.

        Returns
        -------
        float
            Wavenumber in rad/arcsec.
        """
        if threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {threshold}")
        return 2.0 / (threshold * self.pixel_scale)

    def __repr__(self) -> str:
        return f"BoxPixel(pixel_scale={self.pixel_scale})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, BoxPixel):
            return NotImplemented
        return self.pixel_scale == other.pixel_scale


# ============================================================================
# JAX pytree registration
# ============================================================================
# BoxPixel holds a single float (pixel_scale) as static aux data.
# No JAX-traced children — pixel_scale is fixed at configure time.


def _box_pixel_flatten(bp):
    return (), (bp.pixel_scale,)


def _box_pixel_unflatten(aux, children):
    return BoxPixel(pixel_scale=aux[0])


jax.tree_util.register_pytree_node(BoxPixel, _box_pixel_flatten, _box_pixel_unflatten)


# Sentinel for distinguishing "not provided" from explicit None
_PIXEL_RESPONSE_UNSET = object()
