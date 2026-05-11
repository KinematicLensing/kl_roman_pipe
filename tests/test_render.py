"""Tests for compute_effective_maxk product scan (Branch D from PR #41).

Pins the unified product-scan implementation against future drift toward
the old min-based hybrid or the BoxPixel-only restriction.
"""

import galsim
import numpy as np
import pytest

from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.pixel import BoxPixel, PixelResponse
from kl_pipe.priors import LogUniform, PriorDict, Uniform
from kl_pipe.render import RenderConfig, compute_effective_maxk


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


# =============================================================================
# RenderConfig.for_priors with PSF (regression for commit eb9b8b5)
# =============================================================================
#
# Isolated unit test of the worst-case sizing arithmetic. Complementary to
# tests/test_render_config.py::TestPSFEffectiveMaxk (which uses cusp profiles
# at face-on inclination). Here we use the smooth exponential profile so any
# PSF-vs-no-PSF divergence comes from the PSF factor, not from a cusp.


@pytest.fixture
def exp_priors():
    """Smooth, non-cusp exponential priors with moderate inclination."""
    return PriorDict(
        {
            'cosi': Uniform(0.3, 0.99),
            'theta_int': Uniform(0, np.pi),
            'flux': LogUniform(0.01, 1000.0),
            'int_rscale': Uniform(0.5, 2.0),
            'int_h_over_r': 0.2,
            'g1': 0.0,
            'g2': 0.0,
            'int_x0': 0.0,
            'int_y0': 0.0,
        }
    )


def test_for_priors_psf_caps_effective_maxk(model, exp_priors):
    """RenderConfig.for_priors with PSF returns strictly tighter maxk than without."""
    pixel_scale = 0.1
    pixel_response = BoxPixel(pixel_scale)

    rc_bare = RenderConfig.for_priors(
        model,
        exp_priors,
        pixel_scale,
        pixel_response=pixel_response,
        psf=None,
    )
    rc_psf = RenderConfig.for_priors(
        model,
        exp_priors,
        pixel_scale,
        pixel_response=pixel_response,
        psf=galsim.Gaussian(fwhm=0.2),
    )

    assert rc_psf.effective_maxk < rc_bare.effective_maxk
    assert rc_psf.oversample <= rc_bare.oversample


def test_for_priors_oversample_monotone_in_psf_width(model, exp_priors):
    """Sweep PSF width: wider real-space PSF → smaller worst-case oversample.

    Detects regressions in the PSF kValue handling inside compute_effective_maxk
    that would flatten or invert this trend (e.g., dropping abs(), missing the
    galsim.PositionD wrapping, computing kValue at fixed k).
    """
    pixel_scale = 0.1
    pixel_response = BoxPixel(pixel_scale)
    fwhms = [0.05, 0.10, 0.20, 0.40, 0.80]

    oversamples = []
    eff_maxks = []
    for fwhm in fwhms:
        rc = RenderConfig.for_priors(
            model,
            exp_priors,
            pixel_scale,
            pixel_response=pixel_response,
            psf=galsim.Gaussian(fwhm=fwhm),
        )
        oversamples.append(rc.oversample)
        eff_maxks.append(rc.effective_maxk)

    # non-increasing in PSF width (allow plateaus from oversample rounding)
    for a, b in zip(oversamples[:-1], oversamples[1:]):
        assert (
            b <= a
        ), f"oversample(fwhm={fwhms}) = {oversamples}; expected non-increasing"
    # effective_maxk must monotonically decrease (no rounding to hide drift)
    for a, b in zip(eff_maxks[:-1], eff_maxks[1:]):
        assert b < a, (
            f"effective_maxk(fwhm={fwhms}) = "
            f"{[f'{m:.2f}' for m in eff_maxks]}; expected strictly decreasing"
        )
    # widest fwhm must yield strictly smaller oversample than narrowest
    assert oversamples[-1] < oversamples[0], (
        f"oversample({fwhms[0]}) == oversample({fwhms[-1]}); "
        f"PSF factor not contributing to product scan"
    )


def test_for_priors_tiny_psf_approaches_no_psf(model, exp_priors):
    """In the narrow-PSF limit, with-PSF effective_maxk approaches no-PSF."""
    pixel_scale = 0.1
    pixel_response = BoxPixel(pixel_scale)

    rc_bare = RenderConfig.for_priors(
        model,
        exp_priors,
        pixel_scale,
        pixel_response=pixel_response,
        psf=None,
    )
    rc_tiny = RenderConfig.for_priors(
        model,
        exp_priors,
        pixel_scale,
        pixel_response=pixel_response,
        psf=galsim.Gaussian(fwhm=1e-3),  # essentially a delta
    )
    # PSF FT of a near-delta is ~1 over the scan range → product unchanged
    # by the PSF factor → effective_maxk matches the no-PSF case to within
    # the scan resolution (n_scan=500 points).
    rel_diff = (
        abs(rc_tiny.effective_maxk - rc_bare.effective_maxk) / rc_bare.effective_maxk
    )
    assert rel_diff < 0.02, (
        f"Near-delta PSF should not change effective_maxk significantly; "
        f"got bare={rc_bare.effective_maxk:.2f}, tiny={rc_tiny.effective_maxk:.2f}, "
        f"rel_diff={rel_diff:.3f}"
    )
