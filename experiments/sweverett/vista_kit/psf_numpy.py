"""GalSim-free Gaussian PSF construction for the Vista benchmark kit.

The kl_pipe obs builders normally take a galsim.GSObject and derive PSFData
via kl_pipe.psf.precompute_psf_fft (which imports galsim and hard-requires
x64). NGC JAX containers on Vista do not ship galsim (C++ build, painful on
aarch64), so this module provides:

1. GaussianPSFShim -- a duck-typed stand-in for galsim.Gaussian(fwhm=...)
   exposing exactly the methods/attrs kl_pipe touches:
   getGoodImageSize, drawImage, drawKImage, kValue, maxk, stepk.
   Formulas verified numerically against galsim 2.x locally (see
   check_parity.py): getGoodImageSize matches exactly (10/26/42 for the
   flagship scales), drawKImage matches to 1e-16, drawImage matches to
   ~1e-5 absolute (galsim's own FFT accuracy; the analytic pixel-integrated
   kernel used here is exact).

2. install_galsim_stub() -- if `import galsim` fails, registers a minimal
   stub module exposing PositionD only, so kl_pipe.render's grid-adequacy
   validation (`galsim.PositionD` + psf.kValue) runs against the shim.
   LOUD: prints a notice. If any code path needs more of galsim, it fails
   with AttributeError instead of silently doing something wrong.

3. install_psf_patch() -- monkeypatches kl_pipe.psf.precompute_psf_fft /
   precompute_psf_kspace_fft so that when the psf argument is a
   GaussianPSFShim, an analytic numpy construction is used instead
   (identical numerics minus galsim import and minus the hard x64 guard --
   the guard is what blocks the fp32 pass). Non-shim PSFs fall through to
   the original functions untouched.

This file only touches sys.modules / kl_pipe module attributes at runtime
inside this experiment kit; the repo source is never modified.
"""

from __future__ import annotations

import math
import sys
import types

import numpy as np
from scipy.special import erf as _erf

# galsim GSParams defaults (verified against galsim 2.x)
FOLDING_THRESHOLD = 5e-3
MAXK_THRESHOLD = 1e-3
STEPK_MINIMUM_HLR = 5.0
FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


class _ArrayImage:
    """Minimal stand-in for the galsim.Image return of draw*Image."""

    def __init__(self, array):
        self.array = array


class GaussianPSFShim:
    """Duck-typed circular Gaussian PSF (flux=1), galsim.Gaussian-compatible.

    Only the surface kl_pipe actually uses is implemented.
    """

    def __init__(self, fwhm: float):
        self.fwhm = float(fwhm)
        self.sigma = self.fwhm * FWHM_TO_SIGMA
        hlr = self.sigma * math.sqrt(2.0 * math.log(2.0))
        # R enclosing (1 - folding_threshold) of flux, floored at 5*hlr
        R = max(
            self.sigma * math.sqrt(-2.0 * math.log(FOLDING_THRESHOLD)),
            STEPK_MINIMUM_HLR * hlr,
        )
        self.stepk = math.pi / R
        self.maxk = math.sqrt(-2.0 * math.log(MAXK_THRESHOLD)) / self.sigma

    # -- grid sizing (exact galsim GSObject.getGoodImageSize replica) -------
    def getGoodImageSize(self, pixel_scale: float) -> int:
        Nd = 2.0 * math.pi / (self.stepk * pixel_scale)
        N = int(math.ceil(Nd * (1.0 - 1.0e-12)))
        return 2 * ((N + 1) // 2)

    # -- real-space pixel-integrated draw (exact analytic) ------------------
    def drawImage(self, nx: int, ny: int, scale: float) -> _ArrayImage:
        sq2s = self.sigma * math.sqrt(2.0)
        x = (np.arange(nx) - (nx - 1) / 2.0) * scale
        y = (np.arange(ny) - (ny - 1) / 2.0) * scale
        ex = 0.5 * (_erf((x + scale / 2) / sq2s) - _erf((x - scale / 2) / sq2s))
        ey = 0.5 * (_erf((y + scale / 2) / sq2s) - _erf((y - scale / 2) / sq2s))
        return _ArrayImage(np.outer(ey, ex))  # rows = y, cols = x

    # -- k-space draw (analytic FT, galsim centered layout: DC at n//2) -----
    def drawKImage(self, nx: int, ny: int, scale: float) -> _ArrayImage:
        kx = (np.arange(nx) - nx // 2) * scale
        ky = (np.arange(ny) - ny // 2) * scale
        KX, KY = np.meshgrid(kx, ky)  # rows = ky, cols = kx
        arr = np.exp(-0.5 * self.sigma**2 * (KX**2 + KY**2))
        return _ArrayImage(arr.astype(np.complex128))

    # -- point k-space eval (used by render.py grid-adequacy scans) ---------
    def kValue(self, pos) -> float:
        kx, ky = float(pos.x), float(pos.y)
        return math.exp(-0.5 * self.sigma**2 * (kx * kx + ky * ky))

    def __repr__(self):
        return f'GaussianPSFShim(fwhm={self.fwhm})'


# ---------------------------------------------------------------------------
# galsim stub (installed only when the real galsim is absent)
# ---------------------------------------------------------------------------


class _StubPositionD:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)


def install_galsim_stub() -> bool:
    """Register a minimal 'galsim' stub if the real package is missing.

    Returns True if the stub was installed (real galsim absent), False if
    the real galsim is available (nothing done).
    """
    if 'galsim' in sys.modules and getattr(
        sys.modules['galsim'], '__kl_vista_stub__', False
    ):
        return True
    try:
        import galsim  # noqa: F401

        return False
    except ImportError:
        pass
    stub = types.ModuleType('galsim')
    stub.PositionD = _StubPositionD
    stub.__kl_vista_stub__ = True
    sys.modules['galsim'] = stub
    print(
        '[vista_kit] real galsim not importable -- installed minimal stub '
        '(PositionD only). Any code path needing more of galsim will fail '
        'loudly with AttributeError.'
    )
    return True


def block_galsim() -> None:
    """Force galsim-absent behavior even where galsim is installed.

    Local Vista-readiness simulation: any `import galsim` returns the stub.
    """
    if 'galsim' in sys.modules and not getattr(
        sys.modules['galsim'], '__kl_vista_stub__', False
    ):
        raise RuntimeError('real galsim already imported; block too late')
    stub = types.ModuleType('galsim')
    stub.PositionD = _StubPositionD
    stub.__kl_vista_stub__ = True
    sys.modules['galsim'] = stub
    print('[vista_kit] galsim BLOCKED (stub forced) -- simulating Vista env')


# ---------------------------------------------------------------------------
# analytic PSFData construction (mirrors kl_pipe.psf.precompute_psf_fft
# minus the galsim import and minus the hard x64 guard)
# ---------------------------------------------------------------------------


def build_gaussian_psf_data(
    psf: GaussianPSFShim,
    image_shape,
    pixel_scale: float,
    oversample: int = 1,
):
    """Numpy/analytic replica of kl_pipe.psf.precompute_psf_fft for the shim.

    Same kernel sizing (getGoodImageSize, oddified), same padding
    (next_fast_len), same roll-to-origin, same normalization. Works in both
    fp64 and fp32 JAX modes (no x64 guard).
    """
    import jax.numpy as jnp
    from scipy.fft import next_fast_len

    from kl_pipe.psf import PSFData

    if oversample < 1 or oversample % 2 == 0:
        raise ValueError(f'oversample must be positive odd, got {oversample}')

    coarse_shape = tuple(image_shape)
    if oversample > 1:
        fine_shape = (image_shape[0] * oversample, image_shape[1] * oversample)
        fine_ps = pixel_scale / oversample
    else:
        fine_shape = coarse_shape
        fine_ps = pixel_scale

    kern_size = psf.getGoodImageSize(fine_ps)
    if kern_size < 3:
        raise ValueError(f'PSF too small vs pixel_scale={fine_ps}')
    if kern_size % 2 == 0:
        kern_size += 1

    kernel = psf.drawImage(nx=kern_size, ny=kern_size, scale=fine_ps).array
    kernel = kernel.astype(np.float64)
    kernel /= kernel.sum()

    nrow_pad = next_fast_len(fine_shape[0] + kernel.shape[0] - 1)
    ncol_pad = next_fast_len(fine_shape[1] + kernel.shape[1] - 1)
    padded = np.zeros((nrow_pad, ncol_pad), dtype=np.float64)
    padded[: kernel.shape[0], : kernel.shape[1]] = kernel
    padded = np.roll(
        padded, (-(kernel.shape[0] // 2), -(kernel.shape[1] // 2)), axis=(0, 1)
    )
    kernel_fft = jnp.fft.fft2(jnp.asarray(padded))

    return PSFData(
        kernel_fft=kernel_fft,
        padded_shape=(nrow_pad, ncol_pad),
        original_shape=fine_shape,
        oversample=oversample,
        coarse_shape=coarse_shape,
    )


def build_gaussian_kspace_psf_fft(psf: GaussianPSFShim, padded_shape, fine_ps):
    """Numpy/analytic replica of kl_pipe.psf.precompute_psf_kspace_fft."""
    import jax.numpy as jnp

    pad_sq = padded_shape[0]
    if padded_shape[0] != padded_shape[1]:
        raise ValueError(f'square grid required, got {padded_shape}')
    dk = 2.0 * np.pi / (pad_sq * fine_ps)
    arr = psf.drawKImage(nx=pad_sq, ny=pad_sq, scale=dk).array
    fft_ordered = np.fft.ifftshift(arr)
    fft_ordered = fft_ordered / fft_ordered[0, 0]
    return jnp.asarray(fft_ordered)


_PATCH_INSTALLED = False


def install_psf_patch() -> None:
    """Route kl_pipe.psf.precompute_psf_* through the analytic builders when
    the psf is a GaussianPSFShim; original behavior otherwise.

    build_image_obs / build_grism_obs resolve these via function-local
    `from kl_pipe.psf import ...` at call time, so patching the module
    attribute is sufficient. Idempotent.
    """
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return
    import kl_pipe.psf as klpsf

    _orig_fft = klpsf.precompute_psf_fft
    _orig_kspace = klpsf.precompute_psf_kspace_fft

    def precompute_psf_fft(
        gsobj,
        image_pars=None,
        *,
        image_shape=None,
        pixel_scale=None,
        oversample=1,
        gsparams=None,
    ):
        if isinstance(gsobj, GaussianPSFShim):
            if gsparams is not None:
                raise ValueError('GaussianPSFShim does not support gsparams')
            if image_pars is not None:
                image_shape = (image_pars.Nrow, image_pars.Ncol)
                pixel_scale = image_pars.pixel_scale
            return build_gaussian_psf_data(
                gsobj, image_shape, pixel_scale, oversample=oversample
            )
        return _orig_fft(
            gsobj,
            image_pars,
            image_shape=image_shape,
            pixel_scale=pixel_scale,
            oversample=oversample,
            gsparams=gsparams,
        )

    def precompute_psf_kspace_fft(gsobj, padded_shape, pixel_scale, gsparams=None):
        if isinstance(gsobj, GaussianPSFShim):
            if gsparams is not None:
                raise ValueError('GaussianPSFShim does not support gsparams')
            return build_gaussian_kspace_psf_fft(gsobj, padded_shape, pixel_scale)
        return _orig_kspace(gsobj, padded_shape, pixel_scale, gsparams=gsparams)

    klpsf.precompute_psf_fft = precompute_psf_fft
    klpsf.precompute_psf_kspace_fft = precompute_psf_kspace_fft
    _PATCH_INSTALLED = True
    print(
        '[vista_kit] kl_pipe.psf.precompute_psf_fft / precompute_psf_kspace_fft '
        'patched: GaussianPSFShim -> analytic numpy path (no galsim, no x64 '
        'guard); other PSF types untouched.'
    )
