"""
BlackJAX gradient-based MCMC backend.

BlackJAX provides JAX-native implementations of HMC, NUTS, and other
gradient-based samplers that leverage JAX's automatic differentiation.

.. warning::
   **Known Issue with Joint Models**

   BlackJAX may experience numerical collapse when sampling joint
   velocity+intensity models where parameter gradients span multiple
   orders of magnitude (~10^4 difference between intensity and velocity
   gradients). Symptoms include:

   - Adapted step_size collapsing to ~1e-8
   - Zero variance in posterior chains
   - NaN/Inf in corner plots

   For joint models, consider using ``NumpyroSampler`` instead, which
   has more robust mass matrix adaptation and built-in Z-score
   reparameterization to handle multi-scale parameters.

   See: tests/test_blackjax.py::TestBlackJAXJointModel for diagnostics.

References
----------
BlackJAX documentation: https://blackjax-devs.github.io/blackjax/
"""

from __future__ import annotations

import time
from typing import Optional, TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random

from kl_pipe.sampling.base import Sampler, SamplerResult
from kl_pipe.sampling.configs import GradientSamplerConfig

if TYPE_CHECKING:
    from kl_pipe.sampling.task import InferenceTask


class BlackJAXSampler(Sampler):
    """
    BlackJAX gradient-based sampler backend.

    BlackJAX provides JAX-native HMC and NUTS implementations that
    leverage automatic differentiation for efficient sampling.

    Advantages over gradient-free methods:
    - Much faster convergence for smooth, unimodal posteriors
    - Better scaling with dimensionality
    - Native JAX integration (GPU/TPU compatible)

    Requires:
    - Differentiable log probability function
    - Smooth posterior (no hard boundaries inside support)

    Parameters
    ----------
    task : InferenceTask
        The inference task to solve.
    config : GradientSamplerConfig
        Sampler configuration options.

    Attributes
    ----------
    requires_gradients : bool
        True - BlackJAX uses gradients.
    provides_evidence : bool
        False - BlackJAX does not compute evidence.
    config_class : type
        GradientSamplerConfig

    Examples
    --------
    >>> from kl_pipe.sampling import InferenceTask, GradientSamplerConfig
    >>> from kl_pipe.sampling.blackjax import BlackJAXSampler
    >>>
    >>> config = GradientSamplerConfig(
    ...     n_samples=2000,
    ...     n_warmup=500,
    ...     algorithm='nuts',
    ...     seed=42,
    ... )
    >>> sampler = BlackJAXSampler(task, config)
    >>> result = sampler.run()
    """

    requires_gradients = True
    provides_evidence = False
    config_class = GradientSamplerConfig

    def __init__(self, task: 'InferenceTask', config: GradientSamplerConfig):
        super().__init__(task, config)

        # Extract config options
        self._algorithm = config.algorithm
        self._step_size = config.step_size
        self._num_integration_steps = config.num_integration_steps
        self._max_tree_depth = config.max_tree_depth
        self._target_acceptance = config.target_acceptance
        self._n_warmup = config.n_warmup

    def _initialize_position(self, key: jax.Array) -> jnp.ndarray:
        """
        Initialize single chain starting position.

        For gradient-based methods, we typically run fewer, longer chains
        rather than many short chains.

        Parameters
        ----------
        key : jax.Array
            JAX random key.

        Returns
        -------
        jnp.ndarray
            Initial position for the chain.
        """
        # Start from a single prior sample
        initial = self.task.sample_prior(key, 1)[0]

        # Verify it has finite log_prob
        log_prob_fn = self.task.get_log_posterior_fn()
        lp = float(log_prob_fn(initial))

        if not np.isfinite(lp):
            # Try a few more samples
            for _ in range(100):
                key, subkey = random.split(key)
                initial = self.task.sample_prior(subkey, 1)[0]
                lp = float(log_prob_fn(initial))
                if np.isfinite(lp):
                    break
            else:
                raise ValueError(
                    "Could not find initial position with finite log_prob. "
                    "Check your priors and likelihood."
                )

        return initial

    def run(self) -> SamplerResult:
        """
        Run BlackJAX sampler (NUTS or HMC).

        Returns
        -------
        SamplerResult
            Posterior samples from gradient-based MCMC.
        """
        import blackjax

        start_time = time.time()

        # Get log probability function
        log_prob_fn = self.task.get_log_posterior_fn()

        # Initialize random key
        if self.config.seed is not None:
            key = random.PRNGKey(self.config.seed)
        else:
            key = random.PRNGKey(int(time.time() * 1000) % 2**32)

        # Initialize position
        key, init_key = random.split(key)
        initial_position = self._initialize_position(init_key)

        n_samples = self.config.n_samples
        n_warmup = self._n_warmup

        if self._algorithm == 'nuts':
            samples, acceptance_rate, step_size = self._run_nuts(
                key, log_prob_fn, initial_position, n_samples, n_warmup
            )
        elif self._algorithm == 'hmc':
            samples, acceptance_rate, step_size = self._run_hmc(
                key, log_prob_fn, initial_position, n_samples, n_warmup
            )
        else:
            raise ValueError(f"Unknown algorithm: {self._algorithm}")

        # Convert to numpy
        samples_np = np.array(samples)

        # Compute log_prob for final samples
        log_prob_vals = np.array([float(log_prob_fn(s)) for s in samples_np])

        elapsed = time.time() - start_time

        return SamplerResult(
            samples=samples_np,
            log_prob=log_prob_vals,
            param_names=self.task.sampled_names,
            fixed_params=self.task.fixed_params,
            acceptance_fraction=(
                float(acceptance_rate) if acceptance_rate is not None else None
            ),
            converged=True,
            diagnostics={
                'algorithm': self._algorithm,
                'n_warmup': n_warmup,
                'n_samples': n_samples,
                'step_size': step_size,
            },
            metadata={
                'backend': 'blackjax',
                'elapsed_seconds': elapsed,
            },
        )

    def _run_nuts(
        self,
        key: jax.Array,
        log_prob_fn,
        initial_position: jnp.ndarray,
        n_samples: int,
        n_warmup: int,
    ):
        """Run NUTS with window adaptation."""
        import blackjax

        # Validate gradients at initial position before warmup
        grad_fn = jax.grad(log_prob_fn)
        initial_grad = grad_fn(initial_position)
        if not jnp.all(jnp.isfinite(initial_grad)):
            nan_indices = np.where(~np.isfinite(np.array(initial_grad)))[0]
            nan_params = [self.task.sampled_names[i] for i in nan_indices]
            raise ValueError(
                f"Gradient at initial position contains NaN/Inf for parameters: {nan_params}. "
                f"Initial position: {initial_position}. "
                f"Gradient: {initial_grad}. "
                "This indicates numerical issues in your likelihood or prior functions. "
                "Check that your priors are well-defined and the likelihood doesn't produce "
                "NaN at valid parameter values."
            )

        # NUTS with step size adaptation
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            log_prob_fn,
            target_acceptance_rate=self._target_acceptance,
        )

        # Run warmup/adaptation
        key, warmup_key = random.split(key)
        (state, params), _ = warmup.run(
            warmup_key,
            initial_position,
            num_steps=n_warmup,
        )

        # Validate adapted parameters
        step_size = params.get('step_size') if isinstance(params, dict) else None
        inverse_mass_matrix = (
            params.get('inverse_mass_matrix') if isinstance(params, dict) else None
        )

        if step_size is None or not np.isfinite(step_size) or step_size <= 0:
            raise ValueError(
                f"BlackJAX warmup produced invalid step_size={step_size}. "
                f"This typically indicates gradient issues during adaptation. "
                f"Initial position was: {initial_position}. "
                "Try: (1) using Gaussian priors instead of bounded priors, "
                "(2) checking your likelihood for numerical issues, "
                "(3) increasing n_warmup."
            )

        if inverse_mass_matrix is not None:
            if not jnp.all(jnp.isfinite(inverse_mass_matrix)):
                raise ValueError(
                    f"BlackJAX warmup produced invalid inverse_mass_matrix with NaN/Inf values. "
                    f"This typically indicates gradient issues during adaptation. "
                    "Try using Gaussian priors instead of bounded priors."
                )

        # Setup NUTS with adapted parameters
        nuts = blackjax.nuts(log_prob_fn, **params)

        # Sampling loop using lax.scan for efficiency
        @jax.jit
        def one_step(carry, _):
            state, key = carry
            key, subkey = random.split(key)
            state, info = nuts.step(subkey, state)
            return (state, key), (state.position, info.acceptance_rate)

        key, sample_key = random.split(key)
        _, (samples, acceptance_rates) = jax.lax.scan(
            one_step, (state, sample_key), None, length=n_samples
        )

        mean_acceptance = float(jnp.mean(acceptance_rates))

        # Extract step size from adapted parameters
        step_size = (
            float(params.get('step_size', 0.0)) if isinstance(params, dict) else None
        )

        return samples, mean_acceptance, step_size

    def _run_hmc(
        self,
        key: jax.Array,
        log_prob_fn,
        initial_position: jnp.ndarray,
        n_samples: int,
        n_warmup: int,
    ):
        """Run standard HMC."""
        import blackjax

        n_params = self.task.n_params

        # Standard HMC with fixed parameters
        hmc = blackjax.hmc(
            log_prob_fn,
            step_size=self._step_size,
            inverse_mass_matrix=jnp.ones(n_params),
            num_integration_steps=self._num_integration_steps,
        )

        state = hmc.init(initial_position)

        # Warmup (just run without recording)
        key, warmup_key = random.split(key)
        for _ in range(n_warmup):
            warmup_key, subkey = random.split(warmup_key)
            state, _ = hmc.step(subkey, state)

        # Sampling loop
        @jax.jit
        def one_step(carry, _):
            state, key = carry
            key, subkey = random.split(key)
            state, info = hmc.step(subkey, state)
            return (state, key), (state.position, info.acceptance_rate)

        key, sample_key = random.split(key)
        _, (samples, acceptance_rates) = jax.lax.scan(
            one_step, (state, sample_key), None, length=n_samples
        )

        mean_acceptance = float(jnp.mean(acceptance_rates))

        # HMC uses fixed step size from config
        step_size = self._step_size

        return samples, mean_acceptance, step_size
