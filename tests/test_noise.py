"""Unit tests for kl_pipe.noise module."""

import numpy as np
import pytest

from kl_pipe.noise import (
    add_intensity_noise,
    add_map_noise,
    add_velocity_noise,
    matched_filter_snr,
    physical_variance_map,
)


@pytest.fixture
def intensity_map():
    """Simple 8x8 intensity map with spatial structure."""
    rng = np.random.default_rng(42)
    # exponential-ish profile so pixels have different values
    y, x = np.mgrid[-4:4, -4:4].astype(float)
    r = np.sqrt(x**2 + y**2)
    return 1000.0 * np.exp(-r / 2.0)


class TestPoissonVariancePerPixel:
    """Verify Poisson variance is per-pixel, not scalar.

    All tests below pass ``include_poisson=True`` explicitly (default
    flipped to ``False`` in issue #24's noise-consolidation work). SNRs
    are chosen so Poisson sub-dominates the matched-filter target;
    otherwise ``add_intensity_noise`` raises (see
    ``TestPoissonOverdominanceRaises``).
    """

    def test_variance_is_2d_array(self, intensity_map):
        _, variance = add_intensity_noise(
            intensity_map, target_snr=50, include_poisson=True, seed=0
        )
        assert variance.shape == intensity_map.shape

    def test_variance_not_uniform(self, intensity_map):
        # at target_snr=50 on this fixture, Gaussian dominates but Poisson
        # still contributes per-pixel structure proportional to intensity.
        _, variance = add_intensity_noise(
            intensity_map, target_snr=50, include_poisson=True, seed=0
        )
        assert variance.max() > variance.min()

    def test_variance_matches_expected_low_snr(self, intensity_map):
        # at target_snr=50 the Gaussian addend is uniform; the per-pixel
        # variation in ``variance`` comes from the Poisson term
        # ``intensity / gain``. Reconstruct it.
        _, variance = add_intensity_noise(
            intensity_map, target_snr=50, include_poisson=True, seed=0
        )
        # gauss_var = (||I||_2 / SNR)^2 - mean(I/gain)
        norm_l2 = np.sqrt(np.sum(intensity_map**2))
        target_pixel_var = (norm_l2 / 50.0) ** 2
        gauss_var = target_pixel_var - (intensity_map / 1.0).mean()
        expected = intensity_map / 1.0 + gauss_var
        np.testing.assert_allclose(variance, expected, rtol=1e-10)


class TestPoissonOverdominanceRaises:
    """Pin the new loud-failure contract: Poisson > target raises.

    Previously a silent ``max(0.0, ...)`` clamp let labeled SNRs run at
    much lower effective SNR than requested. See issue #24.
    """

    def test_high_snr_with_poisson_on_raises(self, intensity_map):
        with pytest.raises(ValueError, match="include_poisson=True is inconsistent"):
            add_intensity_noise(
                intensity_map, target_snr=1000, include_poisson=True, seed=0
            )

    def test_extreme_snr_with_poisson_on_raises(self, intensity_map):
        with pytest.raises(ValueError, match="Effective SNR with Poisson alone"):
            add_intensity_noise(
                intensity_map, target_snr=1e6, include_poisson=True, seed=0
            )

    def test_high_snr_with_poisson_off_succeeds(self, intensity_map):
        # default is now include_poisson=False; high target_snr is honored.
        noisy, variance = add_intensity_noise(intensity_map, target_snr=1e6, seed=0)
        assert noisy.shape == intensity_map.shape
        # Gaussian-only path → uniform per-pixel variance
        np.testing.assert_allclose(variance, variance.flat[0])

    def test_default_is_poisson_off(self, intensity_map):
        # regression guard against re-flipping the default to True
        import inspect

        sig = inspect.signature(add_intensity_noise)
        assert sig.parameters['include_poisson'].default is False


class TestGainParameter:
    """Verify gain param scales Poisson noise correctly."""

    def test_gain_scales_variance(self, intensity_map):
        # higher gain → lower Poisson variance (more photons per data unit).
        # use moderate SNR so Poisson contributes without overshooting target.
        _, var1 = add_intensity_noise(
            intensity_map, target_snr=50, include_poisson=True, gain=1.0, seed=0
        )
        _, var2 = add_intensity_noise(
            intensity_map, target_snr=50, include_poisson=True, gain=2.0, seed=0
        )
        # Poisson term scales 1/gain; Gaussian term is the same uniform
        # value across both calls (target matched-filter variance is
        # gain-independent). Check the Poisson contribution alone.
        poisson1 = intensity_map / 1.0
        poisson2 = intensity_map / 2.0
        np.testing.assert_allclose(
            var1 - var1.min(), poisson1 - poisson1.min(), rtol=1e-10
        )
        np.testing.assert_allclose(
            var2 - var2.min(), poisson2 - poisson2.min(), rtol=1e-10
        )

    def test_gain_affects_noise_realization(self, intensity_map):
        noisy1, _ = add_intensity_noise(
            intensity_map, target_snr=50, include_poisson=True, gain=1.0, seed=0
        )
        noisy2, _ = add_intensity_noise(
            intensity_map, target_snr=50, include_poisson=True, gain=5.0, seed=0
        )
        # different gain → different noise (Poisson lambda differs)
        assert not np.allclose(noisy1, noisy2)

    def test_gain_zero_raises(self, intensity_map):
        with pytest.raises(ValueError, match="gain must be positive"):
            add_intensity_noise(intensity_map, target_snr=50, gain=0.0)

    def test_gain_negative_raises(self, intensity_map):
        with pytest.raises(ValueError, match="gain must be positive"):
            add_intensity_noise(intensity_map, target_snr=50, gain=-1.0)


class TestVelocityNoiseReturnTypes:
    """Verify add_velocity_noise return type matches annotation.

    Regression for Copilot review: previously annotated
    Tuple[ndarray, float] but actually returns per-pixel variance array.
    """

    def test_returns_array_variance(self):
        rng = np.random.default_rng(0)
        velocity = rng.normal(0, 100, size=(8, 8))
        _, variance = add_velocity_noise(velocity, target_snr=50, seed=0)
        assert variance.shape == velocity.shape
        assert variance.dtype.kind == 'f'

    def test_variance_is_uniform(self):
        # Gaussian-only noise → uniform per-pixel variance
        rng = np.random.default_rng(0)
        velocity = rng.normal(0, 100, size=(8, 8))
        _, variance = add_velocity_noise(velocity, target_snr=50, seed=0)
        np.testing.assert_allclose(variance, variance.flat[0])


class TestPhysicalVarianceMap:
    """physical_variance_map: flat background + source shot noise."""

    def test_formula_elementwise(self, intensity_map):
        var = physical_variance_map(
            intensity_map, sigma_bg=3.0, electrons_per_flux=10.0
        )
        np.testing.assert_allclose(var, 9.0 + intensity_map / 10.0, rtol=1e-14)

    def test_small_negative_ringing_clipped(self, intensity_map):
        # FFT-rendering ringing at the sub-percent level contributes zero
        # shot noise rather than negative variance
        image = intensity_map.copy()
        image[0, 0] = -1e-4 * image.max()
        var = physical_variance_map(image, sigma_bg=3.0, electrons_per_flux=10.0)
        assert var[0, 0] == pytest.approx(9.0, rel=1e-14)

    def test_strongly_negative_raises(self, intensity_map):
        image = intensity_map.copy()
        image[0, 0] = -0.5 * image.max()
        with pytest.raises(ValueError, match='wrong image'):
            physical_variance_map(image, sigma_bg=3.0, electrons_per_flux=10.0)

    def test_invalid_inputs_raise(self, intensity_map):
        with pytest.raises(ValueError, match='sigma_bg'):
            physical_variance_map(intensity_map, sigma_bg=0.0, electrons_per_flux=10.0)
        with pytest.raises(ValueError, match='electrons_per_flux'):
            physical_variance_map(intensity_map, sigma_bg=3.0, electrons_per_flux=-1.0)
        with pytest.raises(ValueError, match='no positive flux'):
            physical_variance_map(
                np.zeros((4, 4)), sigma_bg=3.0, electrons_per_flux=10.0
            )


class TestAddMapNoise:
    """add_map_noise: heteroscedastic Gaussian draw from a variance map."""

    def test_seeded_determinism(self, intensity_map):
        var = physical_variance_map(intensity_map, 3.0, 10.0)
        a = add_map_noise(intensity_map, var, seed=11)
        b = add_map_noise(intensity_map, var, seed=11)
        c = add_map_noise(intensity_map, var, seed=12)
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_empirical_variance_matches_map(self, intensity_map):
        # per-pixel sample variance over many reps must match the map:
        # sample/true ~ chi2_(N-1)/(N-1), sd = sqrt(2/N); the bound is
        # 5 sigma of that spread per pixel (N = 4000, sd ~ 0.0224 -> 0.112),
        # deterministic under the pinned seed
        n_reps = 4000
        var = physical_variance_map(intensity_map, 3.0, 10.0)
        rng_seeds = range(n_reps)
        draws = np.stack(
            [add_map_noise(intensity_map, var, seed=1_000_000 + s) for s in rng_seeds]
        )
        sample_var = draws.var(axis=0, ddof=1)
        ratio = sample_var / var
        bound = 5.0 * np.sqrt(2.0 / n_reps)
        assert np.abs(ratio - 1.0).max() < bound, (
            f'per-pixel variance off by up to {np.abs(ratio - 1.0).max():.3f} '
            f'(bound {bound:.3f})'
        )

    def test_shape_mismatch_raises(self, intensity_map):
        with pytest.raises(ValueError, match='shape'):
            add_map_noise(intensity_map, np.ones((2, 2)), seed=0)

    def test_nonpositive_variance_raises(self, intensity_map):
        var = np.ones_like(intensity_map)
        var[3, 3] = 0.0
        with pytest.raises(ValueError, match='positive'):
            add_map_noise(intensity_map, var, seed=0)


class TestMatchedFilterSNR:
    """matched_filter_snr: sqrt(sum(T^2/var)) for scalar or map variance."""

    def test_uniform_reduces_to_l2_over_sigma(self, intensity_map):
        sigma = 7.0
        snr = matched_filter_snr(intensity_map, sigma**2)
        expected = np.sqrt(np.sum(intensity_map**2)) / sigma
        assert snr == pytest.approx(expected, rel=1e-14)

    def test_scalar_and_uniform_map_agree(self, intensity_map):
        snr_scalar = matched_filter_snr(intensity_map, 49.0)
        snr_map = matched_filter_snr(intensity_map, np.full_like(intensity_map, 49.0))
        assert snr_scalar == pytest.approx(snr_map, rel=1e-14)

    def test_extra_variance_lowers_snr(self, intensity_map):
        base = np.full_like(intensity_map, 49.0)
        shot = base + intensity_map / 10.0
        assert matched_filter_snr(intensity_map, shot) < matched_filter_snr(
            intensity_map, base
        )

    def test_nonpositive_variance_raises(self, intensity_map):
        with pytest.raises(ValueError, match='positive'):
            matched_filter_snr(intensity_map, 0.0)
