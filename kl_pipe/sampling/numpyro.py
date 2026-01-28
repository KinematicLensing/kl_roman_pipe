"""
NumPyro gradient-based MCMC backend.

NumPyro provides robust JAX-native NUTS/HMC with superior mass matrix
adaptation compared to raw BlackJAX. This is the recommended backend for
joint velocity+intensity models where parameter gradients span multiple
orders of magnitude.

Key Features
------------
- **Z-score reparameterization**: Automatic scaling to O(1) latent space
- **Dense mass matrix**: Handles parameter correlations
- **R-hat and ESS diagnostics**: Built-in convergence assessment
- **Multi-chain support**: Sequential, parallel, or vectorized execution

Architecture
------------
The sampler wraps the existing JAX log-posterior from InferenceTask using
``numpyro.factor()``. The Z-score reparameterization samples in a standardized
latent space (all parameters ~N(0,1)), then transforms to physical space
before evaluating the likelihood.

.. note::
   This sampler uses ``task.get_log_posterior_fn()`` which INCLUDES priors.
   The latent Normal(0,1) variables are purely for numerical conditioning,
   not for specifying priors. Do not add informative numpyro.sample() calls
   as this would double-count the prior.

References
----------
NumPyro documentation: https://num.pyro.ai/
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Callable, Tuple, TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random

from kl_pipe.sampling.base import Sampler, SamplerResult
from kl_pipe.sampling.configs import NumpyroSamplerConfig, ReparamStrategy
from kl_pipe.priors import Prior, Gaussian, TruncatedNormal, Uniform, LogUniform

if TYPE_CHECKING:
    from kl_pipe.sampling.task import InferenceTask


def compute_reparam_scales(prior: Prior, name: str) -> Tuple[float, float]:
    """
    Compute (loc, scale) for Z-score reparameterization.

    Maps each prior type to appropriate centering and scaling values
    so that sampling in z ~ N(0,1) explores the prior support efficiently.

    Parameters
    ----------
    prior : Prior
        The prior distribution for this parameter.
    name : str
        Parameter name (for error messages).

    Returns
    -------
    loc : float
        Center point in physical space.
    scale : float
        Characteristic scale in physical space.

    Notes
    -----
    For bounded priors (Uniform, TruncatedNormal), the bounds are still
    enforced by the prior log_prob returning -inf outside support.
    The scaling just ensures the sampler starts in the right ballpark.
    """
    if isinstance(prior, Gaussian):
        # Standard case: use prior parameters directly
        return float(prior.mu), float(prior.sigma)

    elif isinstance(prior, TruncatedNormal):
        # Use the UNDERLYING Gaussian parameters
        # The truncation is enforced by prior.log_prob() returning -inf
        return float(prior.mu), float(prior.sigma)

    elif isinstance(prior, Uniform):
        # Center at midpoint, scale so ±2σ approximately covers the range
        loc = (prior.low + prior.high) / 2
        scale = (prior.high - prior.low) / 4  # 4σ spans the range
        return float(loc), float(scale)

    elif isinstance(prior, LogUniform):
        # Work in log-space conceptually
        # Geometric mean as center
        log_mid = (jnp.log(prior.low) + jnp.log(prior.high)) / 2
        log_scale = (jnp.log(prior.high) - jnp.log(prior.low)) / 4
        return float(jnp.exp(log_mid)), float(jnp.exp(log_mid) * log_scale)

    else:
        raise TypeError(f"Unknown prior type for '{name}': {type(prior)}")


def compute_empirical_scales(
    task: 'InferenceTask',
    rng_key: jax.Array,
    n_samples: int = 100,
) -> Dict[str, Tuple[float, float]]:
    """
    Estimate parameter scales empirically from short sampling.

    Runs a quick exploration to estimate posterior means and standard
    deviations, which are then used for Z-score reparameterization.

    Parameters
    ----------
    task : InferenceTask
        The inference task.
    rng_key : jax.Array
        Random key for sampling.
    n_samples : int
        Number of samples for estimation.

    Returns
    -------
    dict
        Mapping from parameter name to (loc, scale) tuple.
    """
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    # Start with prior-based scales for initial run
    prior_scales = {}
    for name in task.sampled_names:
        prior = task.priors.get_prior(name)
        prior_scales[name] = compute_reparam_scales(prior, name)

    # Build simple model with prior scaling
    def preconditioning_model():
        theta_physical = []
        for name in task.sampled_names:
            loc, scale = prior_scales[name]
            z = numpyro.sample(f"_z_{name}", dist.Normal(0, 1))
            param = loc + scale * z
            theta_physical.append(param)

        theta = jnp.stack(theta_physical)
        log_post = task.get_log_posterior_fn()(theta)
        numpyro.factor("log_posterior", log_post)

    # Run short MCMC
    kernel = NUTS(preconditioning_model, dense_mass=False)  # Diagonal for speed
    mcmc = MCMC(kernel, num_warmup=50, num_samples=n_samples, num_chains=1, progress_bar=False)
    mcmc.run(rng_key)

    # Extract samples and compute scales
    samples = mcmc.get_samples()
    empirical_scales = {}

    for name in task.sampled_names:
        loc_prior, scale_prior = prior_scales[name]
        z_samples = samples[f"_z_{name}"]
        # Convert to physical space
        physical_samples = loc_prior + scale_prior * z_samples
        # Estimate from samples
        emp_loc = float(jnp.mean(physical_samples))
        emp_scale = float(jnp.std(physical_samples))
        # Guard against degenerate cases
        if emp_scale < 1e-10:
            emp_scale = scale_prior
        empirical_scales[name] = (emp_loc, emp_scale)

    return empirical_scales


class NumpyroSampler(Sampler):
    """
    NumPyro gradient-based sampler with Z-score reparameterization.

    This sampler wraps the existing JAX log-posterior using numpyro.factor()
    and applies automatic Z-score reparameterization to normalize parameter
    scales. This is critical for joint models where gradients can vary by
    10^4+ across parameters.

    Advantages over BlackJAX
    ------------------------
    - More robust mass matrix adaptation
    - Built-in R-hat and ESS diagnostics
    - Better handling of challenging posteriors
    - Z-score reparameterization for multi-scale problems

    Parameters
    ----------
    task : InferenceTask
        The inference task to solve.
    config : NumpyroSamplerConfig
        Sampler configuration options.

    Attributes
    ----------
    requires_gradients : bool
        True - NumPyro NUTS/HMC uses gradients.
    provides_evidence : bool
        False - NumPyro MCMC does not compute evidence.
    config_class : type
        NumpyroSamplerConfig

    Examples
    --------
    >>> from kl_pipe.sampling import InferenceTask, NumpyroSamplerConfig
    >>> from kl_pipe.sampling.numpyro import NumpyroSampler
    >>>
    >>> config = NumpyroSamplerConfig(
    ...     n_samples=2000,
    ...     n_warmup=1000,
    ...     n_chains=4,
    ...     seed=42,
    ... )
    >>> sampler = NumpyroSampler(task, config)
    >>> result = sampler.run()

    See Also
    --------
    BlackJAXSampler : Simpler but less robust gradient-based sampler.
    """

    requires_gradients = True
    provides_evidence = False
    config_class = NumpyroSamplerConfig

    def __init__(self, task: 'InferenceTask', config: NumpyroSamplerConfig):
        super().__init__(task, config)
        self._reparam_scales: Optional[Dict[str, Tuple[float, float]]] = None

    def _compute_reparam_scales(self, rng_key: jax.Array) -> Dict[str, Tuple[float, float]]:
        """
        Compute reparameterization scales based on config strategy.

        Parameters
        ----------
        rng_key : jax.Array
            Random key (needed for empirical strategy).

        Returns
        -------
        dict
            Mapping from parameter name to (loc, scale) tuple.
        """
        strategy = self.config.reparam_strategy

        if strategy == ReparamStrategy.NONE:
            # No reparameterization: identity transform
            return {name: (0.0, 1.0) for name in self.task.sampled_names}

        elif strategy == ReparamStrategy.PRIOR:
            # Use prior statistics
            scales = {}
            for name in self.task.sampled_names:
                prior = self.task.priors.get_prior(name)
                scales[name] = compute_reparam_scales(prior, name)
            return scales

        elif strategy == ReparamStrategy.EMPIRICAL:
            # Run short preconditioning phase
            n_precond = max(50, int(self.config.n_warmup * self.config.empirical_warmup_frac))
            return compute_empirical_scales(self.task, rng_key, n_samples=n_precond)

        else:
            raise ValueError(f"Unknown reparam strategy: {strategy}")

    def _build_numpyro_model(
        self,
        reparam_scales: Dict[str, Tuple[float, float]],
    ) -> Callable:
        """
        Build NumPyro model that wraps task's log_posterior.

        IMPORTANT: Uses task.get_log_posterior_fn() which INCLUDES the prior.
        We sample from uninformative Normal(0,1) in latent space purely for
        numerical conditioning - the actual prior is in the task's log_prob.

        Parameters
        ----------
        reparam_scales : dict
            Mapping from parameter name to (loc, scale) for Z-score transform.

        Returns
        -------
        callable
            NumPyro model function.
        """
        import numpyro
        import numpyro.distributions as dist

        task = self.task
        log_posterior_fn = task.get_log_posterior_fn()
        sampled_names = task.sampled_names

        def model():
            theta_physical = []

            for name in sampled_names:
                loc, scale = reparam_scales[name]

                # Sample in latent space - this is NOT the prior!
                # Using improper flat "prior" in z-space; actual prior in log_posterior
                z = numpyro.sample(f"_z_{name}", dist.Normal(0.0, 1.0))

                # Transform to physical space
                param = loc + scale * z

                # Store as deterministic for output
                numpyro.deterministic(name, param)
                theta_physical.append(param)

            # Stack into theta array (in sampled_names order, which matches task)
            theta = jnp.stack(theta_physical)

            # Evaluate log posterior (includes both likelihood AND prior)
            log_post = log_posterior_fn(theta)

            # Add to model via factor
            numpyro.factor("log_posterior", log_post)

        return model

    def _get_init_params(
        self,
        rng_key: jax.Array,
        reparam_scales: Dict[str, Tuple[float, float]],
    ) -> Optional[Dict[str, jnp.ndarray]]:
        """
        Get initial parameters for MCMC chains.

        Parameters
        ----------
        rng_key : jax.Array
            Random key.
        reparam_scales : dict
            Mapping from parameter name to (loc, scale).

        Returns
        -------
        dict or None
            Initial values for latent z parameters. Returns None to let
            NumPyro initialize from the prior (which is Normal(0,1) for z).
        """
        init_strategy = self.config.init_strategy
        sampled_names = self.task.sampled_names

        # For 'prior' strategy with our setup, z~N(0,1) IS the prior in latent space
        # So we can just let NumPyro initialize from its prior
        if init_strategy == 'prior':
            # Let NumPyro handle initialization from the model's prior
            # Since we sample z ~ Normal(0, 1), this is equivalent to prior init
            return None

        elif init_strategy == 'median':
            # Start at z=0 (which maps to prior means in physical space)
            init_params = {}
            for name in sampled_names:
                init_params[f"_z_{name}"] = jnp.array(0.0)
            return init_params

        elif init_strategy == 'jitter':
            # Small random perturbation around z=0
            keys = random.split(rng_key, len(sampled_names))
            init_params = {}
            for i, name in enumerate(sampled_names):
                z_init = 0.1 * random.normal(keys[i], ())
                init_params[f"_z_{name}"] = z_init
            return init_params

        else:
            raise ValueError(f"Unknown init_strategy: {init_strategy}")

    def _collect_diagnostics(
        self,
        mcmc,
        reparam_scales: Dict[str, Tuple[float, float]],
    ) -> Dict:
        """
        Collect all diagnostics from NumPyro MCMC run.

        Parameters
        ----------
        mcmc : numpyro.infer.MCMC
            Completed MCMC run.
        reparam_scales : dict
            The scales used for reparameterization.

        Returns
        -------
        dict
            Comprehensive diagnostics.
        """
        from numpyro.diagnostics import summary as numpyro_summary

        extra_fields = mcmc.get_extra_fields()
        samples = mcmc.get_samples(group_by_chain=True)

        # Get physical-space samples for diagnostics
        physical_samples = {}
        for name in self.task.sampled_names:
            if name in samples:
                physical_samples[name] = np.array(samples[name])

        # Compute R-hat and ESS using numpyro's built-in functions
        summary_dict = numpyro_summary(physical_samples)

        r_hat = {}
        ess = {}
        for name in self.task.sampled_names:
            if name in summary_dict:
                r_hat[name] = float(summary_dict[name]['r_hat'])
                ess[name] = float(summary_dict[name]['n_eff'])

        # Divergences
        diverging = np.array(extra_fields.get('diverging', []))
        n_divergences = int(diverging.sum()) if diverging.size > 0 else 0

        # Acceptance probabilities
        accept_prob = extra_fields.get('accept_prob', None)
        if accept_prob is not None:
            accept_prob = np.array(accept_prob)
            mean_accept = float(accept_prob.mean())
        else:
            mean_accept = None

        # Number of leapfrog steps (tree depth proxy)
        num_steps = extra_fields.get('num_steps', None)
        if num_steps is not None:
            num_steps = np.array(num_steps)
            mean_tree_depth = float(np.log2(num_steps + 1).mean())
        else:
            mean_tree_depth = None

        # Step size from last state
        try:
            step_size = float(mcmc.last_state.adapt_state.step_size)
        except (AttributeError, TypeError):
            step_size = None

        diagnostics = {
            # Divergence info
            'diverging': diverging,
            'n_divergences': n_divergences,
            'divergence_rate': n_divergences / diverging.size if diverging.size > 0 else 0.0,

            # Acceptance
            'accept_prob': accept_prob,
            'mean_accept_prob': mean_accept,

            # Tree depth / steps
            'num_steps': num_steps,
            'mean_tree_depth': mean_tree_depth,

            # Adaptation
            'step_size': step_size,

            # Convergence diagnostics
            'r_hat': r_hat,
            'ess': ess,

            # Reparameterization info
            'reparam_strategy': self.config.reparam_strategy.value,
            'reparam_scales': reparam_scales,
        }

        # Optionally save mass matrix
        if self.config.save_mass_matrix:
            try:
                mass_matrix = mcmc.last_state.adapt_state.inverse_mass_matrix
                diagnostics['inverse_mass_matrix'] = np.array(mass_matrix)
            except (AttributeError, TypeError):
                pass

        return diagnostics

    def run(self) -> SamplerResult:
        """
        Run NumPyro NUTS sampler.

        Returns
        -------
        SamplerResult
            Posterior samples and diagnostics.
        """
        import numpyro
        from numpyro.infer import MCMC, NUTS

        start_time = time.time()

        # Setup random key
        seed = self.config.seed if self.config.seed is not None else int(time.time())
        rng_key = random.PRNGKey(seed)
        rng_key, init_key, sample_key = random.split(rng_key, 3)

        # Compute reparameterization scales
        self._reparam_scales = self._compute_reparam_scales(rng_key)

        # Build model
        model = self._build_numpyro_model(self._reparam_scales)

        # Setup NUTS kernel
        kernel = NUTS(
            model,
            dense_mass=self.config.dense_mass,
            max_tree_depth=self.config.max_tree_depth,
            target_accept_prob=self.config.target_accept_prob,
        )

        # Setup MCMC
        mcmc = MCMC(
            kernel,
            num_warmup=self.config.n_warmup,
            num_samples=self.config.n_samples,
            num_chains=self.config.n_chains,
            chain_method=self.config.chain_method,
            progress_bar=self.config.progress,
        )

        # Get initial parameters
        init_params = self._get_init_params(init_key, self._reparam_scales)

        # Run MCMC
        mcmc.run(
            sample_key,
            init_params=init_params,
            extra_fields=(
                'diverging',
                'accept_prob',
                'num_steps',
                'energy',
            ),
        )

        # Extract samples (physical space via deterministic nodes)
        samples_dict = mcmc.get_samples(group_by_chain=False)

        # Build samples array in sampled_names order
        n_total_samples = self.config.n_samples * self.config.n_chains
        samples_list = []
        for name in self.task.sampled_names:
            if name in samples_dict:
                samples_list.append(np.array(samples_dict[name]).flatten())
            else:
                # Fallback: compute from z samples
                z_samples = np.array(samples_dict[f"_z_{name}"]).flatten()
                loc, scale = self._reparam_scales[name]
                samples_list.append(loc + scale * z_samples)

        samples = np.column_stack(samples_list)

        # Compute log probabilities for samples
        log_posterior_fn = self.task.get_log_posterior_fn()
        log_probs = np.array([
            float(log_posterior_fn(jnp.array(theta)))
            for theta in samples
        ])

        # Collect diagnostics
        diagnostics = self._collect_diagnostics(mcmc, self._reparam_scales)

        # Compute acceptance fraction
        acceptance_fraction = diagnostics.get('mean_accept_prob', None)

        # Check convergence
        r_hats = diagnostics.get('r_hat', {})
        max_rhat = max(r_hats.values()) if r_hats else 1.0
        converged = (
            max_rhat < 1.1 and
            diagnostics.get('divergence_rate', 0) < 0.1
        )

        # Build metadata
        elapsed = time.time() - start_time
        metadata = {
            'sampler': 'numpyro',
            'algorithm': 'nuts',
            'elapsed_seconds': elapsed,
            'n_chains': self.config.n_chains,
            'n_warmup': self.config.n_warmup,
            'n_samples_per_chain': self.config.n_samples,
            'seed': seed,
            'dense_mass': self.config.dense_mass,
            'reparam_strategy': self.config.reparam_strategy.value,
        }

        return SamplerResult(
            samples=samples,
            log_prob=log_probs,
            param_names=self.task.sampled_names,
            fixed_params=self.task.fixed_params,
            acceptance_fraction=acceptance_fraction,
            converged=converged,
            diagnostics=diagnostics,
            metadata=metadata,
        )
