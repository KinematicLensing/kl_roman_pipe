"""
InferenceTask: Complete specification of a Bayesian inference task.

This module defines the InferenceTask class which bundles together all
components needed for MCMC sampling:
- Model (velocity, intensity, or joint)
- Likelihood function
- Priors for sampled parameters
- Fixed parameter values
- Data and variance
- Optional metadata (PSF, systematics, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Union, Tuple, Any, TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from kl_pipe.model import Model, VelocityModel, IntensityModel, KLModel
    from kl_pipe.parameters import ImagePars
    from kl_pipe.priors import PriorDict


@dataclass
class InferenceTask:
    """
    Complete specification of a Bayesian inference task.

    Bundles together all components needed for sampling:
    - Model: The forward model (velocity, intensity, or joint KLModel)
    - Likelihood: JIT-compiled log-likelihood function
    - Priors: PriorDict specifying sampled vs fixed parameters
    - Data: Observed data arrays
    - Variance: Observation variance (same shape as data, or scalar)
    - Meta parameters: Optional metadata (PSF, systematics, etc.)

    Provides methods for computing the log posterior and its gradient,
    which are used by sampler backends.

    Parameters
    ----------
    model : Model or KLModel
        The model to fit.
    likelihood_fn : callable
        JIT-compiled log-likelihood function taking full theta array
        (in model's PARAMETER_NAMES order).
    priors : PriorDict
        Prior specifications for all parameters.
    data : dict
        Dictionary containing observed data arrays.
        Keys depend on model type: 'velocity', 'intensity', or both.
    variance : dict
        Dictionary containing variance arrays or scalars.
        Keys should match data dict.
    meta_pars : dict, optional
        Additional metadata (PSF parameters, systematics, etc.).

    Examples
    --------
    >>> from kl_pipe.velocity import OffsetVelocityModel
    >>> from kl_pipe.priors import Uniform, Gaussian, PriorDict
    >>> from kl_pipe.sampling import InferenceTask
    >>>
    >>> # Define priors (sampled) and fixed values
    >>> priors = PriorDict({
    ...     'vcirc': Uniform(100, 350),
    ...     'cosi': Uniform(0.1, 0.99),
    ...     'v0': 10.0,  # Fixed
    ... })
    >>>
    >>> # Create inference task
    >>> task = InferenceTask.from_velocity_model(
    ...     model=OffsetVelocityModel(),
    ...     priors=priors,
    ...     data_vel=observed_velocity,
    ...     variance_vel=25.0,
    ...     image_pars=image_pars,
    ... )
    >>>
    >>> # Get log posterior function for sampling
    >>> log_prob_fn = task.get_log_posterior_fn()
    """

    model: Union['Model', 'KLModel']
    likelihood_fn: Callable[[jnp.ndarray], float]
    priors: 'PriorDict'
    data: Dict[str, jnp.ndarray]
    variance: Dict[str, Union[jnp.ndarray, float]]
    meta_pars: Dict[str, Any] = field(default_factory=dict)

    # Cached functions (computed lazily)
    _log_posterior_fn: Optional[Callable] = field(default=None, init=False, repr=False)
    _log_posterior_grad_fn: Optional[Callable] = field(default=None, init=False, repr=False)

    # Pre-computed mapping for JIT-compatible theta building
    _sampled_to_full_indices: Optional[jnp.ndarray] = field(default=None, init=False, repr=False)
    _fixed_theta_template: Optional[jnp.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Pre-compute index mapping for JIT-compatible theta construction."""
        self._setup_theta_mapping()

    def _setup_theta_mapping(self):
        """
        Pre-compute the mapping from sampled to full parameter space.

        This allows JIT-compatible construction of full theta from sampled theta.
        """
        param_names = self.model.PARAMETER_NAMES
        sampled_names = self.priors.sampled_names
        fixed_values = self.priors.fixed_values

        # Build template with fixed values
        template = []
        sampled_indices = []

        for i, name in enumerate(param_names):
            if name in fixed_values:
                template.append(fixed_values[name])
            else:
                # Will be filled from sampled theta
                template.append(0.0)
                # Find index in sampled_names (sorted)
                sampled_idx = sampled_names.index(name)
                sampled_indices.append((i, sampled_idx))

        self._fixed_theta_template = jnp.array(template)
        # Store as (full_idx, sampled_idx) pairs
        self._sampled_to_full_indices = jnp.array(
            [[full_idx, sampled_idx] for full_idx, sampled_idx in sampled_indices]
        )

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        """Full parameter names from the model."""
        return self.model.PARAMETER_NAMES

    @property
    def sampled_names(self) -> list:
        """Names of parameters being sampled."""
        return self.priors.sampled_names

    @property
    def n_params(self) -> int:
        """Number of sampled parameters."""
        return self.priors.n_sampled

    @property
    def fixed_params(self) -> Dict[str, float]:
        """Fixed parameter values."""
        return self.priors.fixed_values

    def _build_full_theta(self, theta_sampled: jnp.ndarray) -> jnp.ndarray:
        """
        Build full theta array from sampled parameters plus fixed values.

        Maps from sampled parameter space to model parameter space.
        This method is JIT-compatible.

        Parameters
        ----------
        theta_sampled : jnp.ndarray
            Array of sampled parameter values (length = n_params).

        Returns
        -------
        jnp.ndarray
            Full theta array in model's PARAMETER_NAMES order.
        """
        # Get indices
        full_indices = self._sampled_to_full_indices[:, 0].astype(int)
        sampled_indices = self._sampled_to_full_indices[:, 1].astype(int)

        # Reorder sampled values to match full array positions
        sampled_values = theta_sampled[sampled_indices]

        # Scatter sampled values into the template
        theta_full = self._fixed_theta_template.at[full_indices].set(sampled_values)

        return theta_full

    def log_likelihood(self, theta_sampled: jnp.ndarray) -> float:
        """
        Compute log likelihood for sampled parameters.

        Parameters
        ----------
        theta_sampled : jnp.ndarray
            Array of sampled parameter values (length = n_params).

        Returns
        -------
        float
            Log likelihood value.
        """
        theta_full = self._build_full_theta(theta_sampled)
        return self.likelihood_fn(theta_full)

    def log_prior(self, theta_sampled: jnp.ndarray) -> float:
        """
        Compute log prior probability.

        Parameters
        ----------
        theta_sampled : jnp.ndarray
            Array of sampled parameter values.

        Returns
        -------
        float
            Log prior probability.
        """
        return self.priors.log_prior(theta_sampled)

    def log_posterior(self, theta_sampled: jnp.ndarray) -> float:
        """
        Compute log posterior (log_likelihood + log_prior).

        This is the target function for MCMC sampling.

        Parameters
        ----------
        theta_sampled : jnp.ndarray
            Array of sampled parameter values.

        Returns
        -------
        float
            Log posterior probability.
        """
        log_prior = self.log_prior(theta_sampled)

        # Short-circuit if prior is -inf (outside support)
        # Note: This check is not JIT-compatible, but we handle it
        # in the jitted version using jnp.where
        log_like = self.log_likelihood(theta_sampled)

        return log_prior + log_like

    def _log_posterior_jittable(self, theta_sampled: jnp.ndarray) -> float:
        """
        JIT-compatible log posterior function.

        Uses jnp.where to handle -inf prior values without branching.
        """
        log_prior = self.log_prior(theta_sampled)
        log_like = self.log_likelihood(theta_sampled)

        # Return -inf if prior is -inf, otherwise return sum
        return jnp.where(
            jnp.isfinite(log_prior),
            log_prior + log_like,
            -jnp.inf
        )

    def get_log_posterior_fn(self) -> Callable:
        """
        Get JIT-compiled log posterior function.

        Returns
        -------
        callable
            JIT-compiled function theta -> log_posterior.
        """
        if self._log_posterior_fn is None:
            self._log_posterior_fn = jax.jit(self._log_posterior_jittable)
        return self._log_posterior_fn

    def get_log_posterior_and_grad_fn(self) -> Callable:
        """
        Get JIT-compiled log posterior with gradients.

        Returns function that returns (log_prob, gradient).
        Required for gradient-based samplers like BlackJAX.

        Returns
        -------
        callable
            JIT-compiled function theta -> (log_posterior, grad_log_posterior).
        """
        if self._log_posterior_grad_fn is None:
            self._log_posterior_grad_fn = jax.jit(
                jax.value_and_grad(self._log_posterior_jittable)
            )
        return self._log_posterior_grad_fn

    def get_bounds(self) -> list:
        """
        Get parameter bounds as list of (low, high) tuples.

        Useful for bounded optimizers and some samplers.

        Returns
        -------
        list of tuple
            List of (low, high) bounds for each sampled parameter.
            None indicates unbounded in that direction.
        """
        return self.priors.get_bounds()

    def sample_prior(self, rng_key: jax.Array, n_samples: int = 1) -> jnp.ndarray:
        """
        Draw samples from the prior distribution.

        Useful for initializing walkers.

        Parameters
        ----------
        rng_key : jax.Array
            JAX random key.
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        jnp.ndarray
            Array of shape (n_samples, n_params) with prior samples.
        """
        return self.priors.sample(rng_key, n_samples)

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def from_velocity_model(
        cls,
        model: 'VelocityModel',
        priors: 'PriorDict',
        data_vel: jnp.ndarray,
        variance_vel: Union[jnp.ndarray, float],
        image_pars: 'ImagePars',
        meta_pars: Optional[Dict] = None,
    ) -> 'InferenceTask':
        """
        Create inference task for velocity-only inference.

        Parameters
        ----------
        model : VelocityModel
            Velocity model instance.
        priors : PriorDict
            Prior specifications.
        data_vel : jnp.ndarray
            Observed velocity map.
        variance_vel : jnp.ndarray or float
            Velocity variance (map or scalar).
        image_pars : ImagePars
            Image parameters for coordinate grids.
        meta_pars : dict, optional
            Additional metadata.

        Returns
        -------
        InferenceTask
            Configured task ready for sampling.
        """
        from kl_pipe.likelihood import create_jitted_likelihood_velocity

        likelihood_fn = create_jitted_likelihood_velocity(
            model, image_pars, variance_vel, data_vel
        )

        return cls(
            model=model,
            likelihood_fn=likelihood_fn,
            priors=priors,
            data={'velocity': data_vel},
            variance={'velocity': variance_vel},
            meta_pars=meta_pars or {},
        )

    @classmethod
    def from_intensity_model(
        cls,
        model: 'IntensityModel',
        priors: 'PriorDict',
        data_int: jnp.ndarray,
        variance_int: Union[jnp.ndarray, float],
        image_pars: 'ImagePars',
        meta_pars: Optional[Dict] = None,
    ) -> 'InferenceTask':
        """
        Create inference task for intensity-only inference.

        Parameters
        ----------
        model : IntensityModel
            Intensity model instance.
        priors : PriorDict
            Prior specifications.
        data_int : jnp.ndarray
            Observed intensity map.
        variance_int : jnp.ndarray or float
            Intensity variance (map or scalar).
        image_pars : ImagePars
            Image parameters for coordinate grids.
        meta_pars : dict, optional
            Additional metadata.

        Returns
        -------
        InferenceTask
            Configured task ready for sampling.
        """
        from kl_pipe.likelihood import create_jitted_likelihood_intensity

        likelihood_fn = create_jitted_likelihood_intensity(
            model, image_pars, variance_int, data_int
        )

        return cls(
            model=model,
            likelihood_fn=likelihood_fn,
            priors=priors,
            data={'intensity': data_int},
            variance={'intensity': variance_int},
            meta_pars=meta_pars or {},
        )

    @classmethod
    def from_joint_model(
        cls,
        model: 'KLModel',
        priors: 'PriorDict',
        data_vel: jnp.ndarray,
        data_int: jnp.ndarray,
        variance_vel: Union[jnp.ndarray, float],
        variance_int: Union[jnp.ndarray, float],
        image_pars_vel: 'ImagePars',
        image_pars_int: 'ImagePars',
        meta_pars: Optional[Dict] = None,
    ) -> 'InferenceTask':
        """
        Create inference task for joint velocity + intensity inference.

        Parameters
        ----------
        model : KLModel
            Combined kinematic-lensing model instance.
        priors : PriorDict
            Prior specifications.
        data_vel : jnp.ndarray
            Observed velocity map.
        data_int : jnp.ndarray
            Observed intensity map.
        variance_vel : jnp.ndarray or float
            Velocity variance (map or scalar).
        variance_int : jnp.ndarray or float
            Intensity variance (map or scalar).
        image_pars_vel : ImagePars
            Image parameters for velocity map.
        image_pars_int : ImagePars
            Image parameters for intensity map.
        meta_pars : dict, optional
            Additional metadata.

        Returns
        -------
        InferenceTask
            Configured task ready for sampling.
        """
        from kl_pipe.likelihood import create_jitted_likelihood_joint

        likelihood_fn = create_jitted_likelihood_joint(
            model,
            image_pars_vel, image_pars_int,
            variance_vel, variance_int,
            data_vel, data_int,
        )

        return cls(
            model=model,
            likelihood_fn=likelihood_fn,
            priors=priors,
            data={'velocity': data_vel, 'intensity': data_int},
            variance={'velocity': variance_vel, 'intensity': variance_int},
            meta_pars=meta_pars or {},
        )
