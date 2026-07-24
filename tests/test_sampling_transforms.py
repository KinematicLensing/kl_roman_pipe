"""Unit tests for the unconstrained sampling-coordinate bijections.

These validate the pure-math contract of
kl_pipe.sampling.transforms.UnconstrainingTransform: kind selection from
prior bounds, forward/inverse round-trips, exact Jacobians (checked against
autodiff), analytic pushforward densities for Uniform and LogNormal priors,
boundary guards, and mass-matrix transformation. The end-to-end sampler
behavior is covered separately (test_numpyro.py + experiment confirmations).
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from kl_pipe.priors import (
    Gaussian,
    LogNormal,
    LogUniform,
    PriorDict,
    TruncatedNormal,
    Uniform,
)
from kl_pipe.sampling.configs import NumpyroSamplerConfig
from kl_pipe.sampling.transforms import UnconstrainingTransform


@pytest.fixture(scope='module')
def priors():
    # one prior of every supported kind; names chosen so the alphabetical
    # sampled ordering is easy to reason about
    return PriorDict(
        {
            'a_uniform': Uniform(-2.0, 3.0),
            'b_truncnorm': TruncatedNormal(0.1, 0.1, 0.0, 0.3),
            'c_loguniform': LogUniform(0.005, 2.0),
            'd_lognormal': LogNormal(np.log(0.05), 0.3),
            'e_gaussian': Gaussian(0.0, 0.2),
            'f_fixed': 1.5,
        }
    )


@pytest.fixture(scope='module')
def transform(priors):
    return UnconstrainingTransform.from_priors(priors)


@pytest.fixture(scope='module')
def theta_draws(priors):
    return np.asarray(priors.sample(jax.random.PRNGKey(0), n_samples=200))


class TestKindSelection:
    def test_kinds_from_bounds(self, transform):
        # sampled_names sorted alphabetically: a_uniform, b_truncnorm,
        # c_loguniform, d_lognormal, e_gaussian (f_fixed excluded)
        assert transform.names == (
            'a_uniform',
            'b_truncnorm',
            'c_loguniform',
            'd_lognormal',
            'e_gaussian',
        )
        assert transform.kind_names == ('logit', 'logit', 'logit', 'log', 'identity')

    def test_identity_property(self, transform):
        assert not transform.is_identity
        only_gauss = PriorDict({'x': Gaussian(0.0, 1.0)})
        assert UnconstrainingTransform.from_priors(only_gauss).is_identity


class TestRoundTrip:
    def test_forward_inverse_roundtrip(self, transform, theta_draws):
        eta = transform.forward(theta_draws)
        theta_back = np.asarray(transform.inverse(jnp.asarray(eta)))
        np.testing.assert_allclose(theta_back, theta_draws, rtol=1e-12, atol=1e-14)

    def test_inverse_maps_into_support(self, transform, priors):
        # even extreme unconstrained values must land strictly inside bounds
        rng = np.random.default_rng(1)
        eta = rng.uniform(-40.0, 40.0, size=(500, priors.n_sampled))
        theta = np.asarray(transform.inverse(jnp.asarray(eta)))
        for i, (low, high) in enumerate(priors.get_bounds()):
            if low is not None:
                assert (theta[:, i] >= low).all()
            if high is not None:
                assert (theta[:, i] <= high).all()


class TestJacobian:
    def test_jacobian_diag_matches_autodiff(self, transform, theta_draws):
        eta = transform.forward(theta_draws[:20])
        jac_fn = jax.vmap(jax.jacfwd(transform.inverse))
        jac_full = np.asarray(jac_fn(jnp.asarray(eta)))
        # bijection is per-dimension: off-diagonals exactly zero
        off_diag = jac_full - np.einsum('nij,ij->nij', jac_full, np.eye(5))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-15)
        diag_ad = np.einsum('nii->ni', jac_full)
        np.testing.assert_allclose(
            transform.jacobian_diag(eta), np.abs(diag_ad), rtol=1e-10
        )

    def test_log_jacobian_matches_diag(self, transform, theta_draws):
        eta = transform.forward(theta_draws[:50])
        expected = np.log(transform.jacobian_diag(eta)).sum(axis=-1)
        actual = np.asarray(transform.log_jacobian(jnp.asarray(eta)))
        np.testing.assert_allclose(actual, expected, rtol=1e-10)

    def test_gradients_finite_at_extreme_eta(self, transform):
        # a potential built on the transform must stay differentiable far
        # into the tails (NUTS explores there); NaN gradients would silently
        # corrupt trajectories
        def fake_potential(eta):
            theta = transform.inverse(eta)
            return jnp.sum(theta**2) - transform.log_jacobian(eta)

        grad = jax.grad(fake_potential)
        for scale in (0.0, 5.0, 30.0):
            eta = jnp.full((5,), scale)
            assert np.isfinite(np.asarray(grad(eta))).all()
            assert np.isfinite(np.asarray(grad(-eta))).all()


class TestPushforwardDensities:
    """The transformed potential must equal the analytic pushforward."""

    def test_uniform_maps_to_standard_logistic(self):
        priors = PriorDict({'x': Uniform(-2.0, 3.0)})
        tr = UnconstrainingTransform.from_priors(priors)

        def log_density_eta(eta):
            theta = tr.inverse(eta)
            return priors.log_prior(theta) + tr.log_jacobian(eta)

        eta = jnp.linspace(-8.0, 8.0, 41).reshape(-1, 1)
        actual = jax.vmap(log_density_eta)(eta)
        expected = jax.nn.log_sigmoid(eta[:, 0]) + jax.nn.log_sigmoid(-eta[:, 0])
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-8)

    def test_lognormal_maps_to_gaussian(self):
        mu, sigma = np.log(0.05), 0.3
        priors = PriorDict({'x': LogNormal(mu, sigma)})
        tr = UnconstrainingTransform.from_priors(priors)

        def log_density_eta(eta):
            theta = tr.inverse(eta)
            return priors.log_prior(theta) + tr.log_jacobian(eta)

        eta = jnp.linspace(mu - 5 * sigma, mu + 5 * sigma, 31).reshape(-1, 1)
        actual = jax.vmap(log_density_eta)(eta)
        z = (eta[:, 0] - mu) / sigma
        expected = -0.5 * z**2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-8)


class TestBoundaryGuards:
    def test_forward_raises_on_bound(self, transform, priors):
        theta = np.array(priors.sample(jax.random.PRNGKey(2), n_samples=1))[0]
        theta[1] = 0.0  # b_truncnorm exactly on its lower bound
        with pytest.raises(ValueError, match='b_truncnorm'):
            transform.forward(theta)

    def test_forward_raises_outside_support(self, transform, priors):
        theta = np.array(priors.sample(jax.random.PRNGKey(3), n_samples=1))[0]
        theta[3] = -0.01  # d_lognormal below zero
        with pytest.raises(ValueError, match='d_lognormal'):
            transform.forward(theta)

    def test_forward_clipped_reports_and_recovers(self, transform, priors):
        theta = np.array(priors.sample(jax.random.PRNGKey(4), n_samples=1))[0]
        theta[1] = 0.0
        eta, clipped = transform.forward_clipped(theta, u_margin=1e-6)
        assert clipped[1] and clipped.sum() == 1
        assert np.isfinite(eta).all()
        theta_back = np.asarray(transform.inverse(jnp.asarray(eta)))
        low, high = priors.get_param_bounds('b_truncnorm')
        assert low < theta_back[1] < high
        # clip margin is u_margin of the interval width
        np.testing.assert_allclose(theta_back[1], low + 1e-6 * (high - low), rtol=1e-6)

    def test_forward_clipped_no_clip_matches_forward(self, transform, theta_draws):
        eta_c, clipped = transform.forward_clipped(theta_draws)
        assert not clipped.any()
        np.testing.assert_allclose(eta_c, transform.forward(theta_draws), rtol=1e-12)


class TestMassMatrixTransform:
    def test_transform_preserves_spd(self, transform, theta_draws):
        rng = np.random.default_rng(5)
        a = rng.normal(size=(5, 5))
        cov_theta = a @ a.T + 5.0 * np.eye(5)
        eta = transform.forward(theta_draws[0])
        cov_eta = transform.transform_inverse_mass(cov_theta, eta)
        np.testing.assert_allclose(cov_eta, cov_eta.T, rtol=1e-12)
        assert (np.linalg.eigvalsh(cov_eta) > 0).all()

    def test_transform_is_correct_similarity(self, transform, theta_draws):
        eta = transform.forward(theta_draws[0])
        d = transform.jacobian_diag(eta)
        cov_theta = np.diag(d**2)  # posterior std per dim == local jacobian
        cov_eta = transform.transform_inverse_mass(cov_theta, eta)
        np.testing.assert_allclose(cov_eta, np.eye(5), atol=1e-12)


class TestConfigValidation:
    def test_unconstrained_requires_laplace(self):
        with pytest.raises(ValueError, match='precondition_unconstrained'):
            NumpyroSamplerConfig(precondition_unconstrained=True)

    def test_unconstrained_with_laplace_ok(self):
        cfg = NumpyroSamplerConfig(
            precondition='laplace', precondition_unconstrained=True
        )
        assert cfg.precondition_unconstrained
