"""
emcee ensemble sampler backend.

emcee is a gradient-free affine-invariant ensemble sampler,
well-suited for multi-modal posteriors and parameter degeneracies.

References
----------
Foreman-Mackey et al. (2013): https://arxiv.org/abs/1202.3665
"""

from __future__ import annotations

import time
from typing import Optional, TYPE_CHECKING

import numpy as np
import jax.random as random

from kl_pipe.sampling.base import Sampler, SamplerResult
from kl_pipe.sampling.configs import EnsembleSamplerConfig

if TYPE_CHECKING:
    from kl_pipe.sampling.task import InferenceTask


class EmceeSampler(Sampler):
    """
    emcee ensemble sampler backend.

    emcee uses an ensemble of walkers that communicate to explore
    the posterior. It is gradient-free and works well for:
    - Multi-modal distributions
    - Parameters with strong degeneracies
    - Problems where gradients are expensive or unavailable

    Parameters
    ----------
    task : InferenceTask
        The inference task to solve.
    config : EnsembleSamplerConfig
        Sampler configuration options.

    Attributes
    ----------
    requires_gradients : bool
        False - emcee does not use gradients.
    provides_evidence : bool
        False - emcee does not compute evidence.
    config_class : type
        EnsembleSamplerConfig

    Examples
    --------
    >>> from kl_pipe.sampling import InferenceTask, EnsembleSamplerConfig
    >>> from kl_pipe.sampling.emcee import EmceeSampler
    >>>
    >>> config = EnsembleSamplerConfig(
    ...     n_walkers=64,
    ...     n_iterations=5000,
    ...     burn_in=1000,
    ...     seed=42,
    ... )
    >>> sampler = EmceeSampler(task, config)
    >>> result = sampler.run()
    """

    requires_gradients = False
    provides_evidence = False
    config_class = EnsembleSamplerConfig

    def __init__(self, task: 'InferenceTask', config: EnsembleSamplerConfig):
        super().__init__(task, config)

        # Extract config options
        self._moves = config.moves
        self._vectorize = config.vectorize

    def _initialize_walkers(self) -> np.ndarray:
        """
        Initialize walker positions from prior.

        Draws samples from the prior, ensuring all have finite log_prob.

        Returns
        -------
        np.ndarray
            Initial positions with shape (n_walkers, n_params).
        """
        if self.config.seed is not None:
            key = random.PRNGKey(self.config.seed)
        else:
            key = random.PRNGKey(int(time.time() * 1000) % 2**32)

        n_walkers = self.config.n_walkers
        n_params = self.task.n_params

        # Draw initial positions from prior
        positions = np.array(self.task.sample_prior(key, n_walkers))

        # Verify all positions have finite log_prob
        log_prob_fn = self.task.get_log_posterior_fn()
        for i in range(n_walkers):
            lp = float(log_prob_fn(positions[i]))
            if not np.isfinite(lp):
                # Re-sample this walker until valid
                for attempt in range(100):
                    key, subkey = random.split(key)
                    positions[i] = np.array(self.task.sample_prior(subkey, 1))[0]
                    lp = float(log_prob_fn(positions[i]))
                    if np.isfinite(lp):
                        break
                else:
                    raise ValueError(
                        f"Could not initialize walker {i} with finite log_prob "
                        f"after 100 attempts. Check your priors and likelihood."
                    )

        return positions

    def run(self) -> SamplerResult:
        """
        Run emcee sampler.

        Returns
        -------
        SamplerResult
            Posterior samples with burn-in removed and thinned.
        """
        import emcee

        start_time = time.time()

        n_walkers = self.config.n_walkers
        n_params = self.task.n_params
        n_iterations = self.config.n_iterations

        # Check walker count
        if n_walkers < 2 * n_params:
            import warnings

            warnings.warn(
                f"n_walkers ({n_walkers}) is less than 2 * n_params ({2 * n_params}). "
                f"emcee recommends at least 2 * n_params walkers for good performance."
            )

        # Get log probability function (numpy-compatible wrapper)
        jax_log_prob = self.task.get_log_posterior_fn()

        def log_prob(theta):
            """Wrapper for emcee compatibility."""
            import jax.numpy as jnp

            return float(jax_log_prob(jnp.array(theta)))

        # Initialize sampler
        sampler = emcee.EnsembleSampler(
            n_walkers,
            n_params,
            log_prob,
            moves=self._moves,
            vectorize=self._vectorize,
        )

        # Initialize walkers
        initial_state = self._initialize_walkers()

        # Run sampler
        sampler.run_mcmc(
            initial_state,
            n_iterations,
            progress=self.config.progress,
        )

        # Store full chains before flattening
        full_chains = sampler.get_chain()  # Shape: (n_iterations, n_walkers, n_params)
        full_log_prob = sampler.get_log_prob()  # Shape: (n_iterations, n_walkers)

        # Extract chains after burn-in and thinning
        burn_in = self.config.burn_in
        thin = self.config.thin

        samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
        log_prob_samples = sampler.get_log_prob(discard=burn_in, thin=thin, flat=True)

        # Get blobs if available
        blobs = None
        try:
            blobs_raw = sampler.get_blobs(discard=burn_in, thin=thin, flat=True)
            if blobs_raw is not None and len(blobs_raw) > 0:
                blobs = np.array(blobs_raw)
        except Exception:
            pass  # No blobs available

        # Compute diagnostics
        autocorr_time = None
        try:
            autocorr_time = sampler.get_autocorr_time(quiet=True)
        except emcee.autocorr.AutocorrError:
            pass  # Chain too short for autocorrelation estimate

        acceptance = np.mean(sampler.acceptance_fraction)

        elapsed = time.time() - start_time

        # Convergence check: acceptance rate should be reasonable
        converged = 0.1 < acceptance < 0.9

        return SamplerResult(
            samples=samples,
            log_prob=log_prob_samples,
            param_names=self.task.sampled_names,
            fixed_params=self.task.fixed_params,
            chains=full_chains,
            blobs=blobs,
            acceptance_fraction=float(acceptance),
            autocorr_time=autocorr_time,
            converged=converged,
            diagnostics={
                'acceptance_per_walker': sampler.acceptance_fraction.tolist(),
                'n_iterations': n_iterations,
                'n_walkers': n_walkers,
                'burn_in': burn_in,
                'thin': thin,
            },
            metadata={
                'backend': 'emcee',
                'elapsed_seconds': elapsed,
            },
        )
