"""Tests for compute_effective_maxk product scan (Branch D from PR #41).

Pins the unified product-scan implementation against future drift toward
the old min-based hybrid or the BoxPixel-only restriction.
"""

import galsim
import numpy as np
import pytest

from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.pixel import BoxPixel, PixelResponse
from kl_pipe.render import compute_effective_maxk


@pytest.fixture
def model():
    return InclinedExponentialModel()


@pytest.fixture
def params():
    return {
        'cosi': 0.7,
        'theta_int': 0.5,
        'g1': 0.0,
        'g2': 0.0,
        'flux': 1.0,
        'int_rscale': 0.3,
        'int_h_over_r': 0.1,
        'int_x0': 0.0,
        'int_y0': 0.0,
    }


def test_product_scan_is_smaller_than_min_of_individual_maxks(model, params):
    """Pin product-scan vs min-of-individual: product crosses threshold first.

    Pre-PR-41 code used min(profile_maxk, pixel_maxk) for the pixel case;
    docstring even claimed min(profile, pixel, psf). Product scan returns
    a smaller k because at the min-k, both factors are individually at
    threshold so the product is at most threshold² — i.e. the product
    crossed below threshold well before either single factor.
    """
    threshold = 1e-3
    pixel_response = BoxPixel(0.11)

    profile_only = compute_effective_maxk(
        model, params, pixel_response=None, psf=None, threshold=threshold
    )
    profile_pixel = compute_effective_maxk(
        model, params, pixel_response=pixel_response, psf=None, threshold=threshold
    )
    # individual pixel maxk for comparison
    pixel_maxk = pixel_response.maxk(threshold=threshold)
    min_individual = min(profile_only, pixel_maxk)

    # product result must be no larger than min individual; for typical
    # smooth profiles+pixel it is strictly smaller
    assert profile_pixel <= min_individual + 1e-9
    # for this profile, expect product is meaningfully below min
    assert profile_pixel < min_individual * 0.95


def test_psf_in_product_scan_tightens_grid(model, params):
    """PSF is folded into the product scan; tighter PSF → smaller maxk."""
    threshold = 1e-3
    pixel_response = BoxPixel(0.11)

    # naming follows real-space FWHM (smaller FWHM = narrower PSF in real space)
    psf_narrow = galsim.Gaussian(fwhm=0.1)
    psf_med = galsim.Gaussian(fwhm=0.3)
    psf_wide = galsim.Gaussian(fwhm=0.6)

    maxk_narrow = compute_effective_maxk(
        model,
        params,
        pixel_response=pixel_response,
        psf=psf_narrow,
        threshold=threshold,
    )
    maxk_med = compute_effective_maxk(
        model, params, pixel_response=pixel_response, psf=psf_med, threshold=threshold
    )
    maxk_wide = compute_effective_maxk(
        model, params, pixel_response=pixel_response, psf=psf_wide, threshold=threshold
    )
    # wider FWHM in real → narrower FT → tighter maxk.
    # Verify monotonic decrease as PSF FWHM grows (FT decays faster).
    assert maxk_wide < maxk_med < maxk_narrow


def test_generic_pixel_response_no_boxpixel_assumption(model, params):
    """Non-BoxPixel PixelResponse subclasses are exercised in the product scan."""

    class GaussianPixel(PixelResponse):
        """Mock Gaussian-pixel response (FT = exp(-k²σ²/2))."""

        def __init__(self, sigma):
            self.sigma = float(sigma)

        def ft(self, KX, KY):
            import jax.numpy as jnp

            return jnp.exp(-0.5 * (KX**2 + KY**2) * self.sigma**2)

        def maxk(self, threshold=1e-3):
            return float(np.sqrt(-2.0 * np.log(threshold)) / self.sigma)

        def ft_radial(self, k):
            return np.exp(-0.5 * np.asarray(k) ** 2 * self.sigma**2)

    threshold = 1e-3
    # narrow Gaussian pixel: very tight bandlimit, dominates the product
    narrow = GaussianPixel(sigma=0.5)
    wide = GaussianPixel(sigma=0.05)

    maxk_narrow = compute_effective_maxk(
        model, params, pixel_response=narrow, threshold=threshold
    )
    maxk_wide = compute_effective_maxk(
        model, params, pixel_response=wide, threshold=threshold
    )
    # narrower-σ Gaussian pixel = wider FT = larger maxk?
    # actually σ_pixel=0.5 means real-space pixel is wider → FT decays faster
    # → tighter maxk. σ_pixel=0.05 = real-space narrow → FT decays slowly → maxk≈profile_maxk.
    assert maxk_narrow < maxk_wide

    # most importantly: under the old BoxPixel-only check, both calls would
    # silently fall back to bare profile_maxk and produce the SAME answer.
    # Verify they differ.
    assert abs(maxk_narrow - maxk_wide) > 0.1


def test_compute_effective_maxk_handles_small_profile_maxk(model, params):
    """profile_maxk < 0.1 doesn't return a value that exceeds it.

    Pre-PR-41 the scan started at np.linspace(0.1, profile_maxk, ...). For
    profile_maxk < 0.1, the linspace was reversed/degenerate and the
    fallback returned 0.1 — exceeding the true bandlimit.
    """

    class TinyMaxkModel:
        """Mock model with maxk < 0.1, just enough to test the edge case."""

        def maxk(self, params, threshold=1e-3):
            return 0.05

        def _ft_envelope(self, k, params):
            # smooth envelope so the scan has something to work with
            return float(np.exp(-((k / 0.02) ** 2)))

    pixel_response = BoxPixel(0.11)
    res = compute_effective_maxk(
        TinyMaxkModel(), params, pixel_response=pixel_response, threshold=1e-3
    )
    # must not exceed the model's own profile_maxk
    assert res <= 0.05 + 1e-9
