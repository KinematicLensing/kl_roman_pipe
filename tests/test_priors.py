"""
Unit tests for prior distributions.

Tests the prior classes in kl_pipe/priors.py:
- log_prob correctness
- sampling distributions
- PriorDict separation of sampled/fixed
- JAX compatibility
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random

from kl_pipe.priors import (
    Prior,
    Uniform,
    Gaussian,
    Normal,
    LogUniform,
    TruncatedNormal,
    PriorDict,
)


# ==============================================================================
# Uniform Prior Tests
# ==============================================================================


class TestUniform:
    """Tests for Uniform prior."""

    def test_log_prob_inside(self):
        """Log prob is constant inside bounds."""
        prior = Uniform(0, 10)

        # Inside bounds
        lp = prior.log_prob(5.0)
        expected = -np.log(10)
        assert np.isclose(lp, expected)

        # At boundaries
        assert np.isclose(prior.log_prob(0.0), expected)
        assert np.isclose(prior.log_prob(10.0), expected)

    def test_log_prob_outside(self):
        """Log prob is -inf outside bounds."""
        prior = Uniform(0, 10)

        assert prior.log_prob(-1.0) == -np.inf
        assert prior.log_prob(11.0) == -np.inf

    def test_sample_in_bounds(self):
        """Samples are within bounds."""
        prior = Uniform(5, 15)
        key = random.PRNGKey(42)
        samples = prior.sample(key, (1000,))

        assert jnp.all(samples >= 5)
        assert jnp.all(samples <= 15)

    def test_sample_distribution(self):
        """Samples are roughly uniform."""
        prior = Uniform(0, 10)
        key = random.PRNGKey(42)
        samples = prior.sample(key, (10000,))

        # Mean should be ~5, std should be ~10/sqrt(12) = 2.89
        assert np.isclose(np.mean(samples), 5.0, atol=0.2)
        assert np.isclose(np.std(samples), 10 / np.sqrt(12), atol=0.2)

    def test_bounds_property(self):
        """Bounds property returns correct values."""
        prior = Uniform(3, 7)
        assert prior.bounds == (3, 7)

    def test_invalid_bounds(self):
        """Raises error for invalid bounds."""
        with pytest.raises(ValueError, match="high.*must be > low"):
            Uniform(10, 5)

    def test_repr(self):
        """Repr is informative."""
        prior = Uniform(0, 10)
        assert "Uniform(0, 10)" in repr(prior)


# ==============================================================================
# Gaussian Prior Tests
# ==============================================================================


class TestGaussian:
    """Tests for Gaussian prior."""

    def test_log_prob_at_mean(self):
        """Log prob is maximum at mean."""
        prior = Gaussian(5, 2)
        lp_at_mean = prior.log_prob(5.0)
        lp_away = prior.log_prob(7.0)

        assert lp_at_mean > lp_away

    def test_log_prob_formula(self):
        """Log prob matches Gaussian formula."""
        prior = Gaussian(0, 1)
        # log p(0) = -0.5 * log(2 * pi)
        expected = -0.5 * np.log(2 * np.pi)
        assert np.isclose(prior.log_prob(0.0), expected)

        # log p(1) = -0.5 - 0.5 * log(2 * pi)
        expected = -0.5 - 0.5 * np.log(2 * np.pi)
        assert np.isclose(prior.log_prob(1.0), expected)

    def test_sample_distribution(self):
        """Samples match Gaussian distribution."""
        prior = Gaussian(10, 3)
        key = random.PRNGKey(42)
        samples = prior.sample(key, (10000,))

        assert np.isclose(np.mean(samples), 10.0, atol=0.1)
        assert np.isclose(np.std(samples), 3.0, atol=0.1)

    def test_bounds_unbounded(self):
        """Gaussian has no bounds."""
        prior = Gaussian(0, 1)
        assert prior.bounds == (None, None)

    def test_invalid_sigma(self):
        """Raises error for invalid sigma."""
        with pytest.raises(ValueError, match="sigma.*must be positive"):
            Gaussian(0, -1)
        with pytest.raises(ValueError, match="sigma.*must be positive"):
            Gaussian(0, 0)

    def test_normal_alias(self):
        """Normal is an alias for Gaussian."""
        assert Normal is Gaussian


# ==============================================================================
# LogUniform Prior Tests
# ==============================================================================


class TestLogUniform:
    """Tests for LogUniform prior."""

    def test_log_prob_formula(self):
        """Log prob matches log-uniform formula."""
        prior = LogUniform(1, 100)

        # p(x) = 1 / (x * log(high/low))
        # log p(x) = -log(x) - log(log(high/low))
        x = 10.0
        expected = -np.log(x) - np.log(np.log(100 / 1))
        assert np.isclose(prior.log_prob(x), expected)

    def test_log_prob_outside(self):
        """Log prob is -inf outside bounds."""
        prior = LogUniform(1, 100)

        assert prior.log_prob(0.5) == -np.inf
        assert prior.log_prob(101) == -np.inf

    def test_sample_log_uniform(self):
        """Samples are uniform in log space."""
        prior = LogUniform(1, 100)
        key = random.PRNGKey(42)
        samples = prior.sample(key, (10000,))

        log_samples = np.log(samples)

        # In log space, should be uniform on [0, log(100)]
        expected_mean = (np.log(1) + np.log(100)) / 2
        assert np.isclose(np.mean(log_samples), expected_mean, atol=0.1)

    def test_invalid_bounds(self):
        """Raises error for invalid bounds."""
        with pytest.raises(ValueError, match="must be positive"):
            LogUniform(-1, 10)
        with pytest.raises(ValueError, match="must be positive"):
            LogUniform(0, 10)
        with pytest.raises(ValueError, match="must be > low"):
            LogUniform(10, 5)


# ==============================================================================
# TruncatedNormal Prior Tests
# ==============================================================================


class TestTruncatedNormal:
    """Tests for TruncatedNormal prior."""

    def test_log_prob_in_bounds(self):
        """Log prob is finite inside bounds."""
        prior = TruncatedNormal(0.5, 0.2, 0.1, 0.9)

        lp = prior.log_prob(0.5)
        assert np.isfinite(lp)

        lp_edge = prior.log_prob(0.1)
        assert np.isfinite(lp_edge)

    def test_log_prob_outside(self):
        """Log prob is -inf outside bounds."""
        prior = TruncatedNormal(0.5, 0.2, 0.1, 0.9)

        assert prior.log_prob(0.05) == -np.inf
        assert prior.log_prob(0.95) == -np.inf

    def test_sample_in_bounds(self):
        """Samples are within bounds."""
        prior = TruncatedNormal(0.5, 0.3, 0.1, 0.9)
        key = random.PRNGKey(42)
        samples = prior.sample(key, (1000,))

        assert jnp.all(samples >= 0.1)
        assert jnp.all(samples <= 0.9)

    def test_bounds_property(self):
        """Bounds property returns truncation bounds."""
        prior = TruncatedNormal(0.5, 0.2, 0.1, 0.9)
        assert prior.bounds == (0.1, 0.9)

    def test_invalid_params(self):
        """Raises errors for invalid parameters."""
        with pytest.raises(ValueError, match="sigma.*must be positive"):
            TruncatedNormal(0.5, 0, 0.1, 0.9)
        with pytest.raises(ValueError, match="high.*must be > low"):
            TruncatedNormal(0.5, 0.2, 0.9, 0.1)


# ==============================================================================
# PriorDict Tests
# ==============================================================================


class TestPriorDict:
    """Tests for PriorDict class."""

    def test_separation_sampled_fixed(self):
        """Correctly separates sampled and fixed parameters."""
        priors = PriorDict(
            {
                'a': Uniform(0, 1),
                'b': Gaussian(0, 1),
                'c': 5.0,
                'd': 10,
            }
        )

        assert set(priors.sampled_names) == {'a', 'b'}
        assert set(priors.fixed_names) == {'c', 'd'}
        assert priors.fixed_values == {'c': 5.0, 'd': 10.0}

    def test_n_params(self):
        """Counts sampled and fixed correctly."""
        priors = PriorDict(
            {
                'a': Uniform(0, 1),
                'b': Gaussian(0, 1),
                'c': 5.0,
            }
        )

        assert priors.n_sampled == 2
        assert priors.n_fixed == 1
        assert len(priors) == 3

    def test_log_prior(self):
        """Computes correct joint log prior."""
        priors = PriorDict(
            {
                'a': Uniform(0, 1),
                'b': Uniform(0, 1),
            }
        )

        # Both in bounds: sum of two uniform(0,1) log probs
        theta = jnp.array([0.5, 0.5])
        expected = 0.0  # -log(1) + -log(1) = 0
        assert np.isclose(priors.log_prior(theta), expected)

        # One out of bounds
        theta = jnp.array([0.5, 1.5])
        assert priors.log_prior(theta) == -np.inf

    def test_sample(self):
        """Samples from all priors."""
        priors = PriorDict(
            {
                'a': Uniform(0, 1),
                'b': Gaussian(0, 1),
            }
        )

        key = random.PRNGKey(42)
        samples = priors.sample(key, 100)

        assert samples.shape == (100, 2)

        # Check samples are reasonable
        # 'a' should be in [0, 1]
        a_idx = priors.sampled_names.index('a')
        assert jnp.all(samples[:, a_idx] >= 0)
        assert jnp.all(samples[:, a_idx] <= 1)

    def test_get_bounds(self):
        """Returns bounds for all sampled parameters."""
        priors = PriorDict(
            {
                'a': Uniform(0, 10),
                'b': Gaussian(0, 1),
                'c': 5.0,
            }
        )

        bounds = priors.get_bounds()
        # Sorted order: ['a', 'b']
        assert bounds[0] == (0, 10)  # a: Uniform
        assert bounds[1] == (None, None)  # b: Gaussian

    def test_theta_to_full_pars(self):
        """Converts theta to full parameter dict."""
        priors = PriorDict(
            {
                'vcirc': Uniform(100, 300),
                'cosi': Uniform(0.1, 0.9),
                'v0': 10.0,
            }
        )

        theta = jnp.array([0.5, 200.0])  # [cosi, vcirc] in sorted order
        full_pars = priors.theta_to_full_pars(theta, ('vcirc', 'cosi', 'v0'))

        assert full_pars['cosi'] == 0.5
        assert full_pars['vcirc'] == 200.0
        assert full_pars['v0'] == 10.0

    def test_full_pars_to_theta(self):
        """Extracts theta from full parameter dict."""
        priors = PriorDict(
            {
                'vcirc': Uniform(100, 300),
                'cosi': Uniform(0.1, 0.9),
                'v0': 10.0,
            }
        )

        full_pars = {'vcirc': 200.0, 'cosi': 0.5, 'v0': 10.0}
        theta = priors.full_pars_to_theta(full_pars)

        # Sorted order: ['cosi', 'vcirc']
        assert np.isclose(theta[0], 0.5)  # cosi
        assert np.isclose(theta[1], 200.0)  # vcirc

    def test_get_prior(self):
        """Can retrieve individual priors."""
        priors = PriorDict(
            {
                'a': Uniform(0, 10),
                'b': 5.0,
            }
        )

        assert isinstance(priors.get_prior('a'), Uniform)

        with pytest.raises(KeyError):
            priors.get_prior('b')  # Fixed, not a prior

    def test_invalid_type(self):
        """Raises error for invalid parameter types."""
        with pytest.raises(TypeError, match="must be Prior or numeric"):
            PriorDict({'a': 'string'})


# ==============================================================================
# JAX Compatibility Tests
# ==============================================================================


class TestJAXCompatibility:
    """Tests that priors work with JAX transformations."""

    def test_log_prob_jit(self):
        """log_prob can be JIT compiled."""
        prior = Gaussian(0, 1)
        jit_log_prob = jax.jit(prior.log_prob)

        result = jit_log_prob(0.0)
        assert np.isfinite(result)

    def test_log_prob_grad(self):
        """log_prob gradient can be computed."""
        prior = Gaussian(0, 1)
        grad_fn = jax.grad(lambda x: prior.log_prob(x))

        # Gradient at x=0 should be 0 (peak of Gaussian)
        grad_at_0 = grad_fn(0.0)
        assert np.isclose(grad_at_0, 0.0, atol=1e-6)

        # Gradient at x=1 should be negative (moving away from peak)
        grad_at_1 = grad_fn(1.0)
        assert grad_at_1 < 0

    def test_log_prob_vmap(self):
        """log_prob can be vmapped."""
        prior = Uniform(0, 10)
        vmap_log_prob = jax.vmap(prior.log_prob)

        values = jnp.array([5.0, -1.0, 15.0])
        results = vmap_log_prob(values)

        assert np.isfinite(results[0])  # Inside
        assert results[1] == -np.inf  # Below
        assert results[2] == -np.inf  # Above

    def test_prior_dict_log_prior_jit(self):
        """PriorDict.log_prior can be JIT compiled."""
        priors = PriorDict(
            {
                'a': Uniform(0, 1),
                'b': Gaussian(0, 1),
            }
        )

        jit_log_prior = jax.jit(priors.log_prior)
        theta = jnp.array([0.5, 0.0])

        result = jit_log_prior(theta)
        assert np.isfinite(result)
