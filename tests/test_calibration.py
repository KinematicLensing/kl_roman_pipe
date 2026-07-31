"""
Unit tests for kl_pipe/ensemble/calibration.py and the galaxy-frame shear
rotation helpers in kl_pipe/coordinates.py.

Analytic and synthetic checks for shear frame rotation, linear shear-bias
fits, and effective shape-noise aggregation.
"""

import numpy as np
import pytest

from kl_pipe.coordinates import rotate_to_galaxy_frame
from kl_pipe.ensemble.calibration import (
    measure_shear_bias,
    measure_shear_bias_shrinkage_corrected,
    per_galaxy_sigma_eps,
    compute_shape_noise,
    shrinkage_factor,
)

pytestmark = pytest.mark.roman_ensemble


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
# prior-shrinkage correction
# ==============================================================================


SIGMA_PRIOR = 0.2  # production shear prior: isotropic Gaussian(0, 0.2)


def _shrunk_ensemble(rng, n, m, c, sigma_pop=0.03, like_lo=0.10, like_hi=0.18):
    """Ensemble of Gaussian shear posteriors with a known (m, c).

    Mirrors the conjugate update the sampler performs: a likelihood of width
    sigma_like centred on the biased truth, times the Gaussian prior. The
    sigma_like range is chosen so the mean shrinkage lands near the ~0.67 the
    census posteriors show (widths ~0.10-0.18 against a 0.2 prior).
    """
    g = sigma_pop * rng.standard_normal(n)
    sigma_like = rng.uniform(like_lo, like_hi, size=n)
    d = (1.0 + m) * g + c + sigma_like * rng.standard_normal(n)
    s = SIGMA_PRIOR**2 / (SIGMA_PRIOR**2 + sigma_like**2)
    return g, s * d, np.sqrt(s) * sigma_like


class TestShrinkageFactor:
    def test_inverts_the_conjugate_update(self):
        # s and sigma_like are recovered exactly from the posterior width
        sigma_like = np.array([0.05, 0.10, 0.20, 0.50])
        s_true = SIGMA_PRIOR**2 / (SIGMA_PRIOR**2 + sigma_like**2)
        sigma_post = np.sqrt(s_true) * sigma_like
        out = shrinkage_factor(sigma_post, SIGMA_PRIOR)
        np.testing.assert_allclose(out.s, s_true, rtol=1e-12)
        np.testing.assert_allclose(out.sigma_like, sigma_like, rtol=1e-12)
        assert out.n_clipped == 0

    def test_tight_posterior_is_data_dominated(self):
        out = shrinkage_factor(np.array([1e-4]), SIGMA_PRIOR)
        assert out.s[0] == pytest.approx(1.0, abs=1e-6)

    def test_posterior_wider_than_prior_is_counted(self):
        # impossible under a Gaussian conjugate update, so it flags a
        # non-Gaussian posterior rather than silently producing a huge ghat
        out = shrinkage_factor(np.array([0.05, 0.2, 0.3]), SIGMA_PRIOR)
        assert out.n_clipped == 2
        assert np.all(out.s > 0)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError, match='strictly positive'):
            shrinkage_factor(np.array([0.05, -0.01]), SIGMA_PRIOR)
        with pytest.raises(ValueError, match='sigma_prior'):
            shrinkage_factor(np.array([0.05]), 0.0)
        with pytest.raises(ValueError, match='s_floor'):
            shrinkage_factor(np.array([0.05]), SIGMA_PRIOR, s_floor=1.5)


class TestShrinkageCorrectedBias:
    def test_noiseless_recovery_is_exact(self):
        # with no measurement noise the correction is an identity on the
        # maximum-likelihood points, so (m, c) come back exactly
        rng = np.random.default_rng(20260725)
        g = np.linspace(-0.06, 0.06, 50)
        sigma_like = rng.uniform(0.10, 0.18, size=g.size)
        s = SIGMA_PRIOR**2 / (SIGMA_PRIOR**2 + sigma_like**2)
        m_true, c_true = 0.05, -0.002
        mu = s * ((1.0 + m_true) * g + c_true)
        res = measure_shear_bias_shrinkage_corrected(
            g, mu, np.sqrt(s) * sigma_like, SIGMA_PRIOR
        )
        assert res.m == pytest.approx(m_true, abs=1e-12)
        assert res.c == pytest.approx(c_true, abs=1e-12)
        assert res.n_used == res.n_total == 50

    def test_removes_the_shrinkage_bias_the_naive_fit_shows(self):
        # the headline property: with NO instrumental bias present, the naive
        # posterior-mean regression reports m ~ -<s> + 1 purely from prior
        # shrinkage, while the corrected estimator returns zero. Bounds are
        # k * the estimator's own reported sigma, k = 4 (two-sided false
        # alarm 6e-5 per check), not tuned numbers.
        rng = np.random.default_rng(20260725)
        g, mu, sigma_post = _shrunk_ensemble(rng, 20000, m=0.0, c=0.0)

        naive = measure_shear_bias(g, mu)
        corrected = measure_shear_bias_shrinkage_corrected(
            g, mu, sigma_post, SIGMA_PRIOR
        )

        # naive is biased low by (1 - <s>). Its own sampling error is large
        # here -- the sigma_pop = 0.03 signal is weak against sigma_like ~
        # 0.14 -- so bound it by the fit's reported sigma, not a fixed number
        assert abs(naive.m - (corrected.mean_shrinkage - 1.0)) < 4.0 * naive.sigma_m
        assert naive.m < -0.2
        # corrected is consistent with zero
        assert abs(corrected.m) < 4.0 * corrected.sigma_m
        assert abs(corrected.c) < 4.0 * corrected.sigma_c
        # and the naive bias is not a fluke of this seed: measured against
        # its OWN error bar (the correction divides by s ~ 0.68 and so
        # inflates the corrected sigma by 1/s), it is ~13 sigma
        assert abs(naive.m) > 10.0 * naive.sigma_m

    def test_recovers_an_injected_multiplicative_bias(self):
        rng = np.random.default_rng(11)
        m_true = 0.10
        g, mu, sigma_post = _shrunk_ensemble(rng, 20000, m=m_true, c=0.0)
        res = measure_shear_bias_shrinkage_corrected(g, mu, sigma_post, SIGMA_PRIOR)
        assert abs(res.m - m_true) < 4.0 * res.sigma_m

    def test_robust_survives_catastrophic_outliers(self):
        # wrong-mode fits report a CONFIDENT posterior at a wrong shear, so
        # they arrive with both a large residual and a large data-only
        # weight (48x a clean fit here). Least squares follows them; a
        # redescending biweight rejects them outright. A Huber loss does not
        # suffice -- its linear downweighting leaves them at ~4x a clean
        # fit's weight, which moves m by ~0.8 at this contamination.
        rng = np.random.default_rng(7)
        g, mu, sigma_post = _shrunk_ensemble(rng, 4000, m=0.0, c=0.0)
        n_bad = 80  # 2%, the census catastrophic-rate scale
        bad = rng.choice(g.size, size=n_bad, replace=False)
        mu[bad] = 0.3 * np.sign(g[bad] + 1e-12)  # railed to a wrong mode
        sigma_post[bad] = 0.02  # and confident about it

        wls = measure_shear_bias_shrinkage_corrected(g, mu, sigma_post, SIGMA_PRIOR)
        robust = measure_shear_bias_shrinkage_corrected(
            g, mu, sigma_post, SIGMA_PRIOR, estimator='robust'
        )
        assert abs(robust.m) < abs(wls.m) / 10.0
        assert abs(robust.m) < 4.0 * robust.sigma_m
        assert wls.n_downweighted == 0
        # n_outliers is the contamination measure; n_downweighted also counts
        # the biweight's soft transition, which touches a large fraction of a
        # perfectly clean sample and so is not a contamination rate
        assert robust.n_outliers >= n_bad // 2
        assert robust.n_downweighted > robust.n_outliers

    def test_s_min_drops_prior_dominated_fits(self):
        rng = np.random.default_rng(3)
        g, mu, sigma_post = _shrunk_ensemble(
            rng, 2000, m=0.0, c=0.0, like_lo=0.05, like_hi=0.40
        )
        res = measure_shear_bias_shrinkage_corrected(
            g, mu, sigma_post, SIGMA_PRIOR, s_min=0.5
        )
        assert res.n_used < res.n_total
        assert res.mean_shrinkage >= 0.5

    def test_invalid_inputs(self):
        g = np.linspace(-0.05, 0.05, 10)
        sp = np.full(10, 0.12)
        with pytest.raises(ValueError, match='estimator'):
            measure_shear_bias_shrinkage_corrected(
                g, g, sp, SIGMA_PRIOR, estimator='ols'
            )
        with pytest.raises(ValueError, match='shape mismatch'):
            measure_shear_bias_shrinkage_corrected(g, g[:5], sp, SIGMA_PRIOR)
        with pytest.raises(ValueError, match='s_min'):
            measure_shear_bias_shrinkage_corrected(g, g, sp, SIGMA_PRIOR, s_min=1.5)
        with pytest.raises(ValueError, match='>= 3'):
            measure_shear_bias_shrinkage_corrected(g[:2], g[:2], sp[:2], SIGMA_PRIOR)


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
