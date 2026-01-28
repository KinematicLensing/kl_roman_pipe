"""
nautilus neural network nested sampler backend.

nautilus uses neural networks to learn the likelihood contours,
providing both posterior samples and Bayesian evidence estimates.

References
----------
Lange (2023): https://arxiv.org/abs/2306.16923
"""

from __future__ import annotations

import time
from typing import Optional, TYPE_CHECKING

import numpy as np

from kl_pipe.sampling.base import Sampler, SamplerResult
from kl_pipe.sampling.configs import NestedSamplerConfig

if TYPE_CHECKING:
    from kl_pipe.sampling.task import InferenceTask


class NautilusSampler(Sampler):
    """
    nautilus nested sampler backend.

    nautilus uses neural network emulation of the likelihood to
    efficiently explore the parameter space. It provides:
    - Posterior samples
    - Bayesian evidence estimates (for model comparison)
    - Effective sample size diagnostics

    Particularly good for:
    - Multi-modal posteriors
    - Complex likelihood surfaces
    - When evidence estimates are needed

    Parameters
    ----------
    task : InferenceTask
        The inference task to solve.
    config : NestedSamplerConfig
        Sampler configuration options.

    Attributes
    ----------
    requires_gradients : bool
        False - nautilus does not use gradients.
    provides_evidence : bool
        True - nautilus computes evidence estimates.
    config_class : type
        NestedSamplerConfig

    Examples
    --------
    >>> from kl_pipe.sampling import InferenceTask, NestedSamplerConfig
    >>> from kl_pipe.sampling.nautilus import NautilusSampler
    >>>
    >>> config = NestedSamplerConfig(
    ...     n_live=500,
    ...     seed=42,
    ... )
    >>> sampler = NautilusSampler(task, config)
    >>> result = sampler.run()
    >>> print(f"Log evidence: {result.evidence:.2f}")
    """

    requires_gradients = False
    provides_evidence = True
    config_class = NestedSamplerConfig

    def __init__(self, task: 'InferenceTask', config: NestedSamplerConfig):
        super().__init__(task, config)

        # Extract config options
        self._n_live = config.n_live
        self._n_networks = config.n_networks
        self._verbose = config.verbose

    def _build_prior_transform(self):
        """
        Build prior transform function for nautilus.

        nautilus requires a function that transforms samples from [0, 1]^n
        to the prior support. We use inverse CDF sampling.

        Returns
        -------
        callable
            Function mapping uniform [0,1] samples to prior samples.
        """
        from scipy import stats

        bounds = self.task.get_bounds()
        priors = self.task.priors

        def prior_transform(u):
            """Transform uniform [0,1] samples to prior."""
            theta = np.zeros(len(u))
            for i, name in enumerate(priors.sampled_names):
                prior = priors.get_prior(name)
                low, high = bounds[i]

                # Handle different prior types
                prior_class = prior.__class__.__name__

                if prior_class == 'Uniform':
                    # Simple linear transform for uniform
                    theta[i] = low + u[i] * (high - low)

                elif prior_class == 'LogUniform':
                    # Log-uniform: sample uniformly in log space
                    log_low = np.log(low)
                    log_high = np.log(high)
                    theta[i] = np.exp(log_low + u[i] * (log_high - log_low))

                elif prior_class == 'Gaussian' or prior_class == 'Normal':
                    # Gaussian: use inverse CDF (ppf)
                    theta[i] = stats.norm.ppf(u[i], loc=prior.mu, scale=prior.sigma)

                elif prior_class == 'TruncatedNormal':
                    # Truncated normal: use scipy's truncnorm
                    a = (prior.low - prior.mu) / prior.sigma
                    b = (prior.high - prior.mu) / prior.sigma
                    theta[i] = stats.truncnorm.ppf(
                        u[i], a, b, loc=prior.mu, scale=prior.sigma
                    )

                else:
                    # Fallback: assume bounded and use linear transform
                    if low is not None and high is not None:
                        theta[i] = low + u[i] * (high - low)
                    else:
                        # Unbounded - use normal approximation
                        theta[i] = stats.norm.ppf(u[i])

            return theta

        return prior_transform

    def run(self) -> SamplerResult:
        """
        Run nautilus nested sampler.

        Returns
        -------
        SamplerResult
            Posterior samples with evidence estimates.
        """
        import nautilus
        import jax.numpy as jnp

        start_time = time.time()

        n_params = self.task.n_params

        # Get log likelihood function
        jax_log_prob = self.task.get_log_posterior_fn()

        def log_likelihood(theta):
            """Wrapper for nautilus compatibility."""
            return float(jax_log_prob(jnp.array(theta)))

        # Build prior transform
        prior_transform = self._build_prior_transform()

        # Create nautilus sampler
        # nautilus uses prior as the transform function
        sampler = nautilus.Sampler(
            prior_transform,
            log_likelihood,
            n_dim=n_params,
            n_live=self._n_live,
            n_networks=self._n_networks,
            seed=self.config.seed,
        )

        # Run sampler
        sampler.run(verbose=self._verbose)

        # Extract results
        samples, log_weights, log_likelihood_vals = sampler.posterior()

        # Get evidence estimate (use log_z property, evidence() is deprecated)
        log_evidence = sampler.log_z

        elapsed = time.time() - start_time

        # Compute effective sample size from weights
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= np.sum(weights)
        n_eff = 1.0 / np.sum(weights**2)

        # Resample to produce unweighted samples for downstream tools
        # This ensures corner plots, KDE, etc. work correctly without needing
        # to handle weighted samples specially
        n_resample = max(int(n_eff), 1000)  # At least 1000 samples
        n_resample = min(n_resample, len(samples) * 2)  # Don't oversample too much
        rng = np.random.default_rng(self.config.seed)
        resample_idx = rng.choice(len(samples), size=n_resample, p=weights)
        samples_resampled = samples[resample_idx]
        log_prob_resampled = log_likelihood_vals[resample_idx]

        # Compute log_prob (log_likelihood since we used posterior transform)
        # Note: nautilus returns samples from the posterior already

        return SamplerResult(
            samples=samples_resampled,
            log_prob=log_prob_resampled,
            param_names=self.task.sampled_names,
            fixed_params=self.task.fixed_params,
            evidence=float(log_evidence),
            evidence_error=None,  # nautilus doesn't provide error estimate directly
            converged=True,
            diagnostics={
                'n_effective': float(n_eff),
                'n_resampled': n_resample,
                'n_raw_samples': len(samples),
                'n_live': self._n_live,
            },
            metadata={
                'backend': 'nautilus',
                'elapsed_seconds': elapsed,
            },
        )
