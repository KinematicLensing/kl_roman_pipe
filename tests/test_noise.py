"""Unit tests for kl_pipe.noise module."""

import numpy as np
import pytest

from kl_pipe.noise import add_intensity_noise, add_velocity_noise


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
