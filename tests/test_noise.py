"""Unit tests for kl_pipe.noise module."""

import numpy as np
import pytest

from kl_pipe.noise import add_intensity_noise


@pytest.fixture
def intensity_map():
    """Simple 8x8 intensity map with spatial structure."""
    rng = np.random.default_rng(42)
    # exponential-ish profile so pixels have different values
    y, x = np.mgrid[-4:4, -4:4].astype(float)
    r = np.sqrt(x**2 + y**2)
    return 1000.0 * np.exp(-r / 2.0)


class TestPoissonVariancePerPixel:
    """Verify Poisson variance is per-pixel, not scalar."""

    def test_variance_is_2d_array(self, intensity_map):
        _, variance = add_intensity_noise(intensity_map, target_snr=50, seed=0)
        assert variance.shape == intensity_map.shape

    def test_variance_not_uniform(self, intensity_map):
        _, variance = add_intensity_noise(intensity_map, target_snr=1000, seed=0)
        # poisson dominates at high SNR; variance should track intensity
        assert variance.max() > variance.min()

    def test_variance_matches_expected(self, intensity_map):
        # at high SNR, gaussian contribution ~0 so variance ≈ intensity/gain
        _, variance = add_intensity_noise(intensity_map, target_snr=1e6, seed=0)
        np.testing.assert_allclose(
            variance,
            intensity_map / 1.0,
            rtol=0.01,
        )


class TestGainParameter:
    """Verify gain param scales Poisson noise correctly."""

    def test_gain_scales_variance(self, intensity_map):
        # higher gain → lower variance (more photons per data unit)
        _, var1 = add_intensity_noise(intensity_map, target_snr=1e6, gain=1.0, seed=0)
        _, var2 = add_intensity_noise(intensity_map, target_snr=1e6, gain=2.0, seed=0)
        # var ∝ 1/gain, so var2 ≈ var1 / 2
        np.testing.assert_allclose(var2 / var1, 0.5, rtol=0.01)

    def test_gain_affects_noise_realization(self, intensity_map):
        noisy1, _ = add_intensity_noise(intensity_map, target_snr=50, gain=1.0, seed=0)
        noisy2, _ = add_intensity_noise(intensity_map, target_snr=50, gain=5.0, seed=0)
        # different gain → different noise (even with same seed, Poisson lambda differs)
        assert not np.allclose(noisy1, noisy2)

    def test_gain_zero_raises(self, intensity_map):
        with pytest.raises(ValueError, match="gain must be positive"):
            add_intensity_noise(intensity_map, target_snr=50, gain=0.0)

    def test_gain_negative_raises(self, intensity_map):
        with pytest.raises(ValueError, match="gain must be positive"):
            add_intensity_noise(intensity_map, target_snr=50, gain=-1.0)
