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

import math

from kl_pipe.priors import (
    Prior,
    ConditionalLogNormal,
    Uniform,
    Gaussian,
    Normal,
    LogUniform,
    LogNormal,
    TruncatedNormal,
    TruncatedNormalMixture,
    TruncatedLogNormal,
    PriorDict,
    make_tf_prior,
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
# LogNormal Prior Tests
# ==============================================================================


class TestLogNormal:
    """Tests for LogNormal prior."""

    def test_log_prob_formula(self):
        """Log prob matches the analytic log-normal density."""
        mu, sigma = np.log(200.0), 0.2
        prior = LogNormal(mu, sigma)

        x = 250.0
        z = (np.log(x) - mu) / sigma
        expected = -np.log(x) - np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * z**2
        assert np.isclose(prior.log_prob(x), expected)

    def test_log_prob_nonpositive(self):
        """Log prob is -inf at and below zero, with no NaNs."""
        prior = LogNormal(0.0, 1.0)
        assert prior.log_prob(0.0) == -np.inf
        assert prior.log_prob(-5.0) == -np.inf

    def test_log_prob_gradient_finite_at_negative(self):
        """Gradient has no NaN poisoning from the masked log branch."""
        prior = LogNormal(0.0, 1.0)
        g = jax.grad(prior.log_prob)(jnp.array(-1.0))
        assert np.isfinite(g)

    def test_sample_distribution(self):
        """Sample moments match analytic log-normal moments."""
        mu, sigma = np.log(200.0), 0.184
        prior = LogNormal(mu, sigma)
        key = random.PRNGKey(42)
        samples = prior.sample(key, (100000,))

        assert jnp.all(samples > 0)
        mean = np.exp(mu + 0.5 * sigma**2)
        std = mean * np.sqrt(np.expm1(sigma**2))
        assert np.isclose(np.mean(samples), mean, rtol=0.01)
        assert np.isclose(np.std(samples), std, rtol=0.05)
        assert np.isclose(np.median(samples), np.exp(mu), rtol=0.01)

    def test_moment_properties(self):
        """mean/std/median properties match analytic values."""
        mu, sigma = np.log(150.0), 0.3
        prior = LogNormal(mu, sigma)
        assert np.isclose(prior.mean, np.exp(mu + 0.5 * sigma**2))
        assert np.isclose(prior.std, prior.mean * np.sqrt(np.expm1(sigma**2)))
        assert np.isclose(prior.median, 150.0)

    def test_bounds_property(self):
        """Support is x > 0, unbounded above."""
        prior = LogNormal(0.0, 1.0)
        assert prior.bounds == (0.0, None)

    def test_invalid_sigma(self):
        """Raises error for non-positive sigma."""
        with pytest.raises(ValueError, match="sigma.*must be positive"):
            LogNormal(0.0, 0.0)
        with pytest.raises(ValueError, match="sigma.*must be positive"):
            LogNormal(0.0, -1.0)

    def test_normalization(self):
        """Density integrates to ~1 over the support."""
        prior = LogNormal(np.log(200.0), 0.184)
        x = np.linspace(1e-3, 2000.0, 200001)
        pdf = np.exp(np.asarray(jax.vmap(prior.log_prob)(jnp.asarray(x))))
        integral = np.trapz(pdf, x)
        assert np.isclose(integral, 1.0, atol=1e-3)


class TestMakeTFPrior:
    """Tests for the Tully-Fisher prior factory."""

    def test_dex_encoding(self):
        """sigma_tf_dex is converted to natural log: sigma = dex * ln(10)."""
        prior = make_tf_prior(200.0, 0.08)
        assert isinstance(prior, LogNormal)
        assert np.isclose(prior.mu, np.log(200.0))
        assert np.isclose(prior.sigma, 0.08 * np.log(10.0))
        # 0.08 dex -> sigma_ln ~ 0.184
        assert np.isclose(prior.sigma, 0.184, atol=1e-3)

    def test_median_is_center(self):
        """Median of the prior equals the TF center velocity."""
        prior = make_tf_prior(200.0, 0.08)
        assert np.isclose(prior.median, 200.0)

    def test_invalid_args(self):
        """Raises for non-positive center or scatter."""
        with pytest.raises(ValueError, match="v_center_kms.*must be positive"):
            make_tf_prior(-100.0, 0.08)
        with pytest.raises(ValueError, match="sigma_tf_dex.*must be positive"):
            make_tf_prior(200.0, 0.0)


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

    def test_get_param_bounds_tri_state(self):
        """Sampled -> prior bounds; fixed -> (v, v); absent -> (None, None)."""
        priors = PriorDict(
            {
                'a': Uniform(0.1, 9.9),
                'b': Gaussian(0, 1),
                'c': 5.0,
            }
        )
        assert priors.get_param_bounds('a') == (0.1, 9.9)
        assert priors.get_param_bounds('b') == (None, None)  # unbounded Gaussian
        assert priors.get_param_bounds('c') == (5.0, 5.0)
        assert priors.get_param_bounds('missing') == (None, None)

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
        full_pars = priors.theta_to_full_pars(theta)

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


# ==============================================================================
# Serialization (to_dict / describe) -- provenance
# ==============================================================================


class TestPriorSerialization:
    """to_dict() is lossless onto the flat {dist,loc,scale,low,high} schema."""

    def test_uniform_to_dict(self):
        assert Uniform(0.05, 0.9).to_dict() == {
            'dist': 'uniform',
            'loc': None,
            'scale': None,
            'low': 0.05,
            'high': 0.9,
        }

    def test_gaussian_to_dict(self):
        assert Gaussian(0.0, 0.2).to_dict() == {
            'dist': 'gaussian',
            'loc': 0.0,
            'scale': 0.2,
            'low': None,
            'high': None,
        }

    def test_loguniform_to_dict(self):
        d = LogUniform(0.1, 100.0).to_dict()
        assert d['dist'] == 'loguniform'
        assert (d['low'], d['high']) == (0.1, 100.0)
        assert d['loc'] is None and d['scale'] is None

    def test_truncated_normal_to_dict(self):
        assert TruncatedNormal(0.6, 0.15, 0.05, 0.99).to_dict() == {
            'dist': 'truncated_normal',
            'loc': 0.6,
            'scale': 0.15,
            'low': 0.05,
            'high': 0.99,
        }

    def test_lognormal_to_dict_natural_log_params(self):
        prior = make_tf_prior(200.0, 0.08)
        d = prior.to_dict()
        assert d['dist'] == 'lognormal'
        # mu, sigma are in natural-log space
        assert d['loc'] == pytest.approx(np.log(200.0))
        assert d['scale'] == pytest.approx(0.08 * np.log(10.0))
        assert d['low'] is None and d['high'] is None

    def test_base_to_dict_raises(self):
        """An unserializable prior must fail loudly, not silently drop."""

        class _CustomPrior(Prior):
            def log_prob(self, value):
                return jnp.array(0.0)

            def sample(self, rng_key, shape=()):
                return jnp.zeros(shape)

            @property
            def bounds(self):
                return (None, None)

        with pytest.raises(NotImplementedError, match='to_dict'):
            _CustomPrior().to_dict()

    def test_describe_covers_sampled_and_fixed(self):
        priors = PriorDict(
            {
                'g1': Gaussian(0.0, 0.2),
                'cosi': Uniform(0.05, 0.99),
                'z': 1.3,  # fixed
            }
        )
        desc = priors.describe()
        assert set(desc) == {'g1', 'cosi', 'z'}
        assert desc['g1']['dist'] == 'gaussian' and desc['g1']['scale'] == 0.2
        assert desc['cosi']['dist'] == 'uniform'
        # fixed param recorded with dist='fixed' and its value in loc
        assert desc['z'] == {
            'dist': 'fixed',
            'loc': 1.3,
            'scale': None,
            'low': None,
            'high': None,
        }


# ==============================================================================
# TruncatedNormalMixture
# ==============================================================================


def _bulge_mixture() -> TruncatedNormalMixture:
    """The Gadotti 2009 pseudobulge/classical split used by the population."""
    return TruncatedNormalMixture(
        (TruncatedNormal(1.5, 0.9, 0.5, 2.0), TruncatedNormal(3.4, 1.3, 2.0, 6.0)),
        (0.7, 0.3),
    )


class TestTruncatedNormalMixture:
    def test_pdf_integrates_to_one(self):
        m = _bulge_mixture()
        x = jnp.linspace(0.5, 6.0, 200001)
        integral = float(jnp.trapezoid(jnp.exp(m.log_prob(x)), x))
        assert integral == pytest.approx(1.0, abs=1e-4)

    def test_outside_every_component_is_neg_inf(self):
        m = _bulge_mixture()
        assert float(m.log_prob(jnp.array(0.2))) == -np.inf
        assert float(m.log_prob(jnp.array(6.5))) == -np.inf

    def test_matches_weighted_component_density(self):
        # inside a single component's support the mixture is exactly that
        # component's density times its weight
        m = _bulge_mixture()
        v = jnp.array(1.2)  # pseudobulge-only region
        expected = np.log(0.7) + float(TruncatedNormal(1.5, 0.9, 0.5, 2.0).log_prob(v))
        assert float(m.log_prob(v)) == pytest.approx(expected, rel=1e-12)

    def test_bounds_span_all_components(self):
        assert _bulge_mixture().bounds == (0.5, 6.0)

    def test_sampling_reproduces_the_density(self):
        m = _bulge_mixture()
        s = np.asarray(m.sample(random.PRNGKey(0), (200000,)))
        x = jnp.linspace(0.5, 6.0, 200001)
        pdf = jnp.exp(m.log_prob(x))
        analytic_mean = float(jnp.trapezoid(x * pdf, x))
        # standard error of the mean over 2e5 draws is std/sqrt(N) ~ 0.003;
        # 0.02 is ~6 sigma, so this catches a wrong mixture, not noise
        assert s.mean() == pytest.approx(analytic_mean, abs=0.02)
        # the component weights must show up as the mass below the split
        assert np.mean(s <= 2.0) == pytest.approx(0.7, abs=0.01)
        assert s.min() >= 0.5 and s.max() <= 6.0

    def test_jit_and_grad(self):
        m = _bulge_mixture()
        assert np.isfinite(float(jax.jit(m.log_prob)(jnp.array(3.0))))
        g = float(jax.grad(m.log_prob)(jnp.array(3.0)))
        assert np.isfinite(g)

    def test_validation(self):
        tn = TruncatedNormal(1.0, 1.0, 0.0, 2.0)
        with pytest.raises(ValueError, match='>= 2 components'):
            TruncatedNormalMixture((tn,), (1.0,))
        with pytest.raises(ValueError, match='weights for'):
            TruncatedNormalMixture((tn, tn), (0.5, 0.3, 0.2))
        with pytest.raises(ValueError, match='must be positive'):
            TruncatedNormalMixture((tn, tn), (1.2, -0.2))
        with pytest.raises(ValueError, match='sum to 1'):
            TruncatedNormalMixture((tn, tn), (0.5, 0.4))

    def test_to_dict_roundtrips_components(self):
        d = _bulge_mixture().to_dict()
        assert d['dist'] == 'truncated_normal_mixture'
        assert d['weights'] == [0.7, 0.3]
        assert [c['dist'] for c in d['components']] == ['truncated_normal'] * 2


# ==============================================================================
# Truncated Log-Normal
# ==============================================================================


class TestTruncatedLogNormal:
    def test_validation(self):
        with pytest.raises(ValueError, match='sigma'):
            TruncatedLogNormal(0.0, 0.0, 0.1, 1.0)
        with pytest.raises(ValueError, match='low'):
            TruncatedLogNormal(0.0, 1.0, 0.0, 1.0)
        with pytest.raises(ValueError, match='high'):
            TruncatedLogNormal(0.0, 1.0, 1.0, 0.5)

    def test_bounds_and_support(self):
        p = TruncatedLogNormal(math.log(0.2), 0.5, 0.05, 1.0)
        assert p.bounds == (0.05, 1.0)
        assert float(p.log_prob(jnp.array(0.04))) == -np.inf
        assert float(p.log_prob(jnp.array(1.5))) == -np.inf
        assert np.isfinite(float(p.log_prob(jnp.array(0.2))))

    def test_normalized_over_the_truncation_interval(self):
        p = TruncatedLogNormal(math.log(0.2), 0.5, 0.05, 1.0)
        x = np.linspace(0.05, 1.0, 200001)
        pdf = np.exp(np.asarray(p.log_prob(jnp.array(x))))
        assert np.trapz(pdf, x) == pytest.approx(1.0, abs=1e-4)

    def test_samples_stay_in_support(self):
        p = TruncatedLogNormal(math.log(0.2), 0.8, 0.05, 1.0)
        draws = np.asarray(p.sample(random.PRNGKey(0), (20000,)))
        assert draws.min() >= 0.05 and draws.max() <= 1.0


# ==============================================================================
# Conditional Log-Normal
# ==============================================================================


def _ratio_prior(parent='disk_rscale', median=0.4, dex=0.25):
    return ConditionalLogNormal(
        parent, math.log(median), dex * math.log(10.0), 0.01, 5.0
    )


class TestConditionalLogNormal:
    def test_validation(self):
        with pytest.raises(ValueError, match='parent'):
            ConditionalLogNormal('', 0.0, 1.0, 0.1, 1.0)
        with pytest.raises(ValueError, match='sigma_ratio'):
            ConditionalLogNormal('p', 0.0, 0.0, 0.1, 1.0)
        with pytest.raises(ValueError, match='low'):
            ConditionalLogNormal('p', 0.0, 1.0, -0.1, 1.0)
        with pytest.raises(ValueError, match='high'):
            ConditionalLogNormal('p', 0.0, 1.0, 1.0, 0.5)

    def test_unconditional_calls_raise(self):
        p = _ratio_prior()
        with pytest.raises(TypeError, match='without its parent'):
            p.log_prob(jnp.array(0.1))
        with pytest.raises(TypeError, match='without its parent'):
            p.sample(random.PRNGKey(0))

    def test_bounds_do_not_move_with_the_parent(self):
        # grid sizing and the unconstraining transform read bounds, so the
        # support has to be static even though the density shifts
        p = _ratio_prior()
        assert p.bounds == (0.01, 5.0)

    def test_matches_the_collapsed_prior_at_a_concrete_parent(self):
        p = _ratio_prior()
        collapsed = p.at_parent(0.3)
        for v in (0.05, 0.12, 0.4, 2.0):
            given = float(p.log_prob_given(jnp.array(v), jnp.array(0.3)))
            assert given == pytest.approx(float(collapsed.log_prob(jnp.array(v))))

    def test_normalized_for_a_given_parent(self):
        p = _ratio_prior()
        x = np.linspace(0.01, 5.0, 400001)
        pdf = np.exp(np.asarray(p.log_prob_given(jnp.array(x), jnp.array(0.3))))
        assert np.trapz(pdf, x) == pytest.approx(1.0, abs=1e-4)

    def test_draws_recover_the_ratio_distribution(self):
        p = _ratio_prior(median=0.4, dex=0.25)
        parent = 0.3
        draws = np.asarray(
            p.sample_given(random.PRNGKey(0), jnp.array(parent), (200000,))
        )
        ratio = draws / parent
        assert np.median(ratio) == pytest.approx(0.4, rel=0.02)
        assert np.std(np.log10(ratio)) == pytest.approx(0.25, rel=0.02)

    def test_nonpositive_parent_is_out_of_support(self):
        p = _ratio_prior()
        assert float(p.log_prob_given(jnp.array(0.1), jnp.array(0.0))) == -np.inf
        assert float(p.log_prob_given(jnp.array(0.1), jnp.array(-1.0))) == -np.inf

    def test_to_dict_records_the_parent(self):
        d = _ratio_prior().to_dict()
        assert d['dist'] == 'conditional_lognormal'
        assert d['parent'] == 'disk_rscale'
        assert d['loc'] == pytest.approx(math.log(0.4))


class TestConditionalPriorsInPriorDict:
    def _pd(self):
        return PriorDict(
            {
                'cosi': Uniform(0.05, 0.95),
                'disk_rscale': LogNormal(math.log(0.21), 0.237 * math.log(10.0)),
                'vel.rscale': _ratio_prior(),
            }
        )

    def test_dependency_bookkeeping(self):
        pd = self._pd()
        assert pd.conditional_parents == {'vel.rscale': 'disk_rscale'}
        order = pd.topological_order
        assert order.index('disk_rscale') < order.index('vel.rscale')
        assert sorted(order) == pd.sampled_names

    def test_log_prior_uses_the_sampled_parent(self):
        pd = self._pd()
        i_parent = pd.sampled_names.index('disk_rscale')
        i_child = pd.sampled_names.index('vel.rscale')
        theta = jnp.array([0.5, 0.21, 0.08][: len(pd.sampled_names)])

        # moving the parent must change the child's contribution
        expected = pd.get_prior('vel.rscale').log_prob_given(
            theta[i_child], theta[i_parent]
        )
        others = sum(
            float(pd.get_prior(n).log_prob(theta[i]))
            for i, n in enumerate(pd.sampled_names)
            if n != 'vel.rscale'
        )
        assert float(pd.log_prior(theta)) == pytest.approx(float(expected) + others)

    def test_gradient_with_respect_to_the_parent_is_correct(self):
        # the truncation normalization depends on the parent; dropping it
        # leaves the density right and the parent gradient wrong
        pd = self._pd()
        theta = jnp.array([0.5, 0.21, 0.08])
        grad = np.asarray(jax.grad(pd.log_prior)(theta))
        eps = 1e-6
        fd = np.array(
            [
                float(
                    (
                        pd.log_prior(theta.at[i].add(eps))
                        - pd.log_prior(theta.at[i].add(-eps))
                    )
                    / (2 * eps)
                )
                for i in range(theta.size)
            ]
        )
        np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-6)

    def test_log_prior_is_jittable(self):
        pd = self._pd()
        theta = jnp.array([0.5, 0.21, 0.08])
        assert float(jax.jit(pd.log_prior)(theta)) == pytest.approx(
            float(pd.log_prior(theta))
        )

    def test_sampling_respects_the_dependency(self):
        pd = self._pd()
        draws = np.asarray(pd.sample(random.PRNGKey(3), 20000))
        parent = draws[:, pd.sampled_names.index('disk_rscale')]
        child = draws[:, pd.sampled_names.index('vel.rscale')]
        ratio = child / parent
        assert np.median(ratio) == pytest.approx(0.4, rel=0.05)

    def test_fixed_parent_collapses_to_an_unconditional_prior(self):
        pd = PriorDict({'vel.rscale': _ratio_prior('p'), 'p': 0.3})
        prior = pd.get_prior('vel.rscale')
        assert isinstance(prior, TruncatedLogNormal)
        assert pd.conditional_parents == {}
        # the collapsed prior is the conditional evaluated at the pinned value
        ref = _ratio_prior('p').at_parent(0.3)
        assert prior == ref
        assert pd.describe()['vel.rscale']['dist'] == 'truncated_lognormal'

    def test_unknown_parent_raises(self):
        with pytest.raises(KeyError, match='not a parameter'):
            PriorDict({'a': _ratio_prior('nope')})

    def test_dependency_cycles_raise(self):
        with pytest.raises(ValueError, match='cycle'):
            PriorDict({'a': _ratio_prior('a')})
        with pytest.raises(ValueError, match='cycle'):
            PriorDict({'a': _ratio_prior('b'), 'b': _ratio_prior('a')})

    def test_describe_records_the_parent(self):
        assert self._pd().describe()['vel.rscale']['parent'] == 'disk_rscale'


class TestMixtureFlatSchema:
    def test_to_dict_carries_the_flat_keys(self):
        # every describe() consumer reads dist/loc/scale/low/high; a mixture
        # that omitted them raised KeyError when writing the summary row
        d = _bulge_mixture().to_dict()
        for key in ('dist', 'loc', 'scale', 'low', 'high'):
            assert key in d
        assert d['loc'] == pytest.approx(0.7 * 1.5 + 0.3 * 3.4)
        assert d['low'] == 0.5 and d['high'] == 6.0

    def test_summary_row_schema_is_complete(self):
        rec = PriorDict({'n': _bulge_mixture()}).describe()['n']
        assert all(rec[k] is not None for k in ('dist', 'loc', 'scale', 'low', 'high'))
