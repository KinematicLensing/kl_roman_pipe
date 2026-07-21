"""
Unit tests for kl_pipe/calibration.py.

Analytic and synthetic checks for shear frame rotation, linear shear-bias
fits, and effective shape-noise aggregation.
"""

import numpy as np
import pytest

from kl_pipe.calibration import (
    rotate_to_galaxy_frame,
    measure_shear_bias,
    per_galaxy_sigma_eps,
    compute_shape_noise,
)


# ==============================================================================
# rotate_to_galaxy_frame
# ==============================================================================


class TestRotateToGalaxyFrame:
    def test_zero_angle_identity(self):
        """theta_int = 0 leaves components unchanged."""
        gp, gx = rotate_to_galaxy_frame(0.03, -0.01, 0.0)
        assert np.isclose(gp, 0.03)
        assert np.isclose(gx, -0.01)

    def test_ninety_degrees_flips_sign(self):
        """Spin-2: theta_int = pi/2 flips the sign of both components."""
        gp, gx = rotate_to_galaxy_frame(0.03, -0.01, np.pi / 2)
        assert np.isclose(gp, -0.03)
        assert np.isclose(gx, 0.01)

    def test_forty_five_degrees_swaps(self):
        """theta_int = pi/4: g+ <- g2, gx <- -g1."""
        gp, gx = rotate_to_galaxy_frame(0.03, -0.01, np.pi / 4)
        assert np.isclose(gp, -0.01)
        assert np.isclose(gx, -0.03)

    def test_magnitude_preserved(self):
        """|g| is invariant under frame rotation."""
        rng = np.random.default_rng(0)
        g1 = rng.normal(0, 0.03, 100)
        g2 = rng.normal(0, 0.03, 100)
        theta = rng.uniform(0, np.pi, 100)
        gp, gx = rotate_to_galaxy_frame(g1, g2, theta)
        assert np.allclose(gp**2 + gx**2, g1**2 + g2**2)

    def test_vectorized(self):
        """Accepts arrays and broadcasts elementwise."""
        gp, gx = rotate_to_galaxy_frame(
            np.array([0.05, 0.0]), np.array([0.0, 0.05]), np.array([0.0, 0.0])
        )
        assert np.allclose(gp, [0.05, 0.0])
        assert np.allclose(gx, [0.0, 0.05])


# ==============================================================================
# measure_shear_bias
# ==============================================================================


class TestMeasureShearBias:
    def test_exact_recovery_noiseless(self):
        """Recovers injected (m, c) exactly from noiseless points."""
        g_true = np.array([-0.05, -0.03, -0.01, 0.01, 0.03, 0.05])
        m_in, c_in = 0.02, -0.001
        g_meas = (1 + m_in) * g_true + c_in
        res = measure_shear_bias(g_true, g_meas)
        assert np.isclose(res.m, m_in)
        assert np.isclose(res.c, c_in)

    def test_weighted_errors_analytic(self):
        """WLS parameter errors match the analytic normal-equation covariance."""
        g_true = np.array([-0.05, -0.03, -0.01, 0.01, 0.03, 0.05])
        g_meas = 1.01 * g_true + 0.002
        sigma = np.full_like(g_true, 0.004)
        res = measure_shear_bias(g_true, g_meas, sigma_meas=sigma)

        w = 1.0 / sigma**2
        delta = np.sum(w) * np.sum(w * g_true**2) - np.sum(w * g_true) ** 2
        assert np.isclose(res.sigma_m, np.sqrt(np.sum(w) / delta))
        assert np.isclose(res.sigma_c, np.sqrt(np.sum(w * g_true**2) / delta))

    def test_noisy_recovery_within_errors(self):
        """Noisy fit recovers truth within 5 sigma (seeded)."""
        rng = np.random.default_rng(42)
        g_true = np.tile(np.array([-0.05, -0.03, -0.01, 0.01, 0.03, 0.05]), 50)
        sigma = np.full_like(g_true, 0.01)
        m_in, c_in = -0.03, 0.0005
        g_meas = (1 + m_in) * g_true + c_in + rng.normal(0, 0.01, g_true.size)
        res = measure_shear_bias(g_true, g_meas, sigma_meas=sigma)
        assert abs(res.m - m_in) < 5 * res.sigma_m
        assert abs(res.c - c_in) < 5 * res.sigma_c

    def test_too_few_points(self):
        with pytest.raises(ValueError, match="need >= 3 points"):
            measure_shear_bias([0.01, 0.02], [0.01, 0.02])

    def test_constant_g_true(self):
        with pytest.raises(ValueError, match="identical"):
            measure_shear_bias([0.01, 0.01, 0.01], [0.01, 0.02, 0.03])

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape"):
            measure_shear_bias([0.01, 0.02, 0.03], [0.01, 0.02])

    def test_nonpositive_sigma(self):
        with pytest.raises(ValueError, match="strictly positive"):
            measure_shear_bias(
                [0.01, 0.02, 0.03], [0.01, 0.02, 0.03], sigma_meas=[0.1, 0.0, 0.1]
            )


# ==============================================================================
# shape noise
# ==============================================================================


class TestShapeNoise:
    def test_per_galaxy_formula(self):
        """sigma_eps_j = sqrt[(s+^2 + sx^2)/2] (Pranjal Eq. 20)."""
        out = per_galaxy_sigma_eps(np.array([0.03]), np.array([0.04]))
        assert np.isclose(out[0], np.sqrt((0.03**2 + 0.04**2) / 2))

    def test_homogeneous_ensemble(self):
        """Identical galaxies: ensemble value equals the per-galaxy value."""
        n = 50
        sp = np.full(n, 0.03)
        sx = np.full(n, 0.03)
        sigma_eps, err = compute_shape_noise(sp, sx)
        assert np.isclose(sigma_eps, 0.03)
        assert err == 0.0

    def test_inverse_variance_weighting(self):
        """Heterogeneous ensemble: harmonic (inverse-variance) combination."""
        sp = np.array([0.02, 0.06])
        sx = np.array([0.02, 0.06])
        sigma_eps, _ = compute_shape_noise(sp, sx)
        sigma_j = np.array([0.02, 0.06])
        expected = np.sqrt(2 / np.sum(1.0 / sigma_j**2))
        assert np.isclose(sigma_eps, expected)
        # dominated by the better-measured galaxy: below the arithmetic mean
        assert sigma_eps < np.mean(sigma_j)

    def test_invalid_widths(self):
        with pytest.raises(ValueError, match="strictly positive"):
            per_galaxy_sigma_eps(np.array([0.03, -0.01]), np.array([0.03, 0.01]))

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            per_galaxy_sigma_eps(np.array([0.03, 0.01]), np.array([0.03]))
