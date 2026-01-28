"""
Prior distributions for Bayesian inference.

All priors are designed to be JAX-compatible with jittable log_prob methods.
Fixed parameters are specified as numeric values (int/float) in PriorDict,
which automatically separates them from sampled parameters.

Examples
--------
>>> from kl_pipe.priors import Uniform, Gaussian, TruncatedNormal, PriorDict
>>>
>>> # Define priors for sampled parameters, fixed values for others
>>> priors = PriorDict({
...     'vcirc': Uniform(100, 300),
...     'cosi': TruncatedNormal(0.5, 0.2, 0.1, 0.99),
...     'g1': Gaussian(0, 0.05),
...     'v0': 10.0,  # Fixed at 10.0
... })
>>>
>>> priors.sampled_names  # ['cosi', 'g1', 'vcirc']
>>> priors.fixed_values   # {'v0': 10.0}
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple, Union, Optional, List

import jax
import jax.numpy as jnp
import jax.random as random


class Prior(ABC):
    """
    Abstract base class for prior distributions.

    All priors must implement:
    - log_prob(value): Compute log probability density (JAX-compatible)
    - sample(rng_key, shape): Draw samples from the prior
    - bounds: Property returning (lower, upper) bounds for bounded priors

    Priors should be immutable after construction.

    Methods
    -------
    log_prob(value) -> jnp.ndarray
        Compute log probability density at value.
        Returns -inf for values outside the support.
        Used in: log_posterior = log_likelihood + sum(log_prior for each param)

    sample(rng_key, shape) -> jnp.ndarray
        Draw random samples from the distribution.
        Used for: initializing walkers/chains from the prior

    bounds -> Tuple[Optional[float], Optional[float]]
        Property returning (lower, upper) bounds of support.
        None means unbounded in that direction.
    """

    @abstractmethod
    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute log probability density at value.

        Must be JAX-jittable. Returns -inf for values outside support.

        Parameters
        ----------
        value : jnp.ndarray
            Parameter value(s) to evaluate.

        Returns
        -------
        jnp.ndarray
            Log probability density at each value.
        """
        pass

    @abstractmethod
    def sample(self, rng_key: jax.Array, shape: Tuple[int, ...] = ()) -> jnp.ndarray:
        """
        Draw samples from the prior distribution.

        Parameters
        ----------
        rng_key : jax.Array
            JAX random key for reproducibility.
        shape : tuple of int, optional
            Shape of samples to draw. Default is () for single sample.

        Returns
        -------
        jnp.ndarray
            Samples from the prior.
        """
        pass

    @property
    @abstractmethod
    def bounds(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Return (lower, upper) bounds of the prior support.

        None indicates unbounded in that direction.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@dataclass(frozen=True)
class Uniform(Prior):
    """
    Uniform prior on [low, high].

    log p(x) = -log(high - low) if low <= x <= high, else -inf

    Parameters
    ----------
    low : float
        Lower bound of support.
    high : float
        Upper bound of support.

    Examples
    --------
    >>> prior = Uniform(0, 10)
    >>> prior.log_prob(5.0)  # Returns -log(10)
    >>> prior.log_prob(15.0)  # Returns -inf (outside bounds)
    """

    low: float
    high: float

    def __post_init__(self):
        if self.high <= self.low:
            raise ValueError(f"high ({self.high}) must be > low ({self.low})")

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        log_width = jnp.log(self.high - self.low)
        in_bounds = (value >= self.low) & (value <= self.high)
        return jnp.where(in_bounds, -log_width, -jnp.inf)

    def sample(self, rng_key: jax.Array, shape: Tuple[int, ...] = ()) -> jnp.ndarray:
        return random.uniform(rng_key, shape, minval=self.low, maxval=self.high)

    @property
    def bounds(self) -> Tuple[float, float]:
        return (self.low, self.high)

    def __repr__(self) -> str:
        return f"Uniform({self.low}, {self.high})"


@dataclass(frozen=True)
class Gaussian(Prior):
    """
    Gaussian (Normal) prior with mean mu and standard deviation sigma.

    log p(x) = -0.5 * ((x - mu) / sigma)^2 - log(sigma) - 0.5*log(2*pi)

    Parameters
    ----------
    mu : float
        Mean of the distribution.
    sigma : float
        Standard deviation (must be positive).

    Examples
    --------
    >>> prior = Gaussian(0, 1)
    >>> prior.log_prob(0.0)  # Maximum at mean
    >>> prior.sample(jax.random.PRNGKey(0), (100,))  # 100 samples
    """

    mu: float
    sigma: float

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"sigma ({self.sigma}) must be positive")

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        z = (value - self.mu) / self.sigma
        return -0.5 * z**2 - jnp.log(self.sigma) - 0.5 * jnp.log(2 * jnp.pi)

    def sample(self, rng_key: jax.Array, shape: Tuple[int, ...] = ()) -> jnp.ndarray:
        return self.mu + self.sigma * random.normal(rng_key, shape)

    @property
    def bounds(self) -> Tuple[None, None]:
        return (None, None)

    def __repr__(self) -> str:
        return f"Gaussian({self.mu}, {self.sigma})"


# Alias for clarity
Normal = Gaussian


@dataclass(frozen=True)
class LogUniform(Prior):
    """
    Log-uniform prior (uniform in log space) on [low, high].

    Useful for scale parameters that span orders of magnitude.

    log p(x) = -log(x) - log(log(high/low)) if low <= x <= high, else -inf

    Parameters
    ----------
    low : float
        Lower bound (must be positive).
    high : float
        Upper bound (must be > low).

    Examples
    --------
    >>> prior = LogUniform(0.1, 100)  # Scale parameter spanning 3 orders of magnitude
    >>> prior.sample(jax.random.PRNGKey(0), (1000,))
    """

    low: float
    high: float

    def __post_init__(self):
        if self.low <= 0:
            raise ValueError(f"low ({self.low}) must be positive for LogUniform")
        if self.high <= self.low:
            raise ValueError(f"high ({self.high}) must be > low ({self.low})")

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        log_range = jnp.log(self.high / self.low)
        in_bounds = (value >= self.low) & (value <= self.high)
        return jnp.where(
            in_bounds,
            -jnp.log(value) - jnp.log(log_range),
            -jnp.inf
        )

    def sample(self, rng_key: jax.Array, shape: Tuple[int, ...] = ()) -> jnp.ndarray:
        log_low = jnp.log(self.low)
        log_high = jnp.log(self.high)
        return jnp.exp(random.uniform(rng_key, shape, minval=log_low, maxval=log_high))

    @property
    def bounds(self) -> Tuple[float, float]:
        return (self.low, self.high)

    def __repr__(self) -> str:
        return f"LogUniform({self.low}, {self.high})"


@dataclass(frozen=True)
class TruncatedNormal(Prior):
    """
    Truncated normal prior with bounds [low, high].

    A Gaussian distribution truncated to lie within specified bounds.
    Useful when you have a Gaussian belief but with hard physical constraints.

    Parameters
    ----------
    mu : float
        Mean of the underlying normal distribution.
    sigma : float
        Standard deviation of the underlying normal.
    low : float
        Lower truncation bound.
    high : float
        Upper truncation bound.

    Examples
    --------
    >>> # Inclination with Gaussian prior but physical bounds
    >>> prior = TruncatedNormal(0.5, 0.2, 0.1, 0.99)
    """

    mu: float
    sigma: float
    low: float
    high: float

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"sigma ({self.sigma}) must be positive")
        if self.high <= self.low:
            raise ValueError(f"high ({self.high}) must be > low ({self.low})")

    def _norm_cdf(self, x: jnp.ndarray) -> jnp.ndarray:
        """Standard normal CDF via error function."""
        return 0.5 * (1.0 + jax.scipy.special.erf(x / jnp.sqrt(2.0)))

    def _norm_ppf(self, p: jnp.ndarray) -> jnp.ndarray:
        """Standard normal inverse CDF (quantile function)."""
        return jnp.sqrt(2.0) * jax.scipy.special.erfinv(2.0 * p - 1.0)

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        # Standardized bounds
        alpha = (self.low - self.mu) / self.sigma
        beta = (self.high - self.mu) / self.sigma

        # Normalization constant
        Z = self._norm_cdf(beta) - self._norm_cdf(alpha)

        # Standard Gaussian log prob
        z = (value - self.mu) / self.sigma
        log_gaussian = -0.5 * z**2 - jnp.log(self.sigma) - 0.5 * jnp.log(2 * jnp.pi)

        # Apply truncation
        in_bounds = (value >= self.low) & (value <= self.high)
        return jnp.where(in_bounds, log_gaussian - jnp.log(Z), -jnp.inf)

    def sample(self, rng_key: jax.Array, shape: Tuple[int, ...] = ()) -> jnp.ndarray:
        # Inverse CDF sampling for truncated normal
        alpha = (self.low - self.mu) / self.sigma
        beta = (self.high - self.mu) / self.sigma

        cdf_alpha = self._norm_cdf(alpha)
        cdf_beta = self._norm_cdf(beta)

        u = random.uniform(rng_key, shape)
        p = cdf_alpha + u * (cdf_beta - cdf_alpha)

        return self.mu + self.sigma * self._norm_ppf(p)

    @property
    def bounds(self) -> Tuple[float, float]:
        return (self.low, self.high)

    def __repr__(self) -> str:
        return f"TruncatedNormal({self.mu}, {self.sigma}, {self.low}, {self.high})"


class PriorDict:
    """
    Collection of priors for multiple parameters.

    Distinguishes between sampled parameters (have Prior) and fixed parameters
    (have scalar values). Provides methods for computing joint log probability
    and building parameter arrays.

    Parameters
    ----------
    param_spec : dict
        Dictionary mapping parameter names to either:
        - Prior instance: parameter will be sampled
        - scalar (int, float): parameter is fixed at this value

    Examples
    --------
    >>> priors = PriorDict({
    ...     'vcirc': Uniform(100, 300),
    ...     'cosi': TruncatedNormal(0.5, 0.2, 0.1, 0.99),
    ...     'g1': Gaussian(0, 0.05),
    ...     'v0': 10.0,  # Fixed value
    ... })
    >>> priors.sampled_names
    ['cosi', 'g1', 'vcirc']
    >>> priors.fixed_names
    ['v0']
    >>> priors.fixed_values
    {'v0': 10.0}
    """

    def __init__(self, param_spec: Dict[str, Union[Prior, float, int]]):
        self._param_spec = dict(param_spec)

        # Separate into priors and fixed values
        self._priors: Dict[str, Prior] = {}
        self._fixed: Dict[str, float] = {}

        for name, value in param_spec.items():
            if isinstance(value, Prior):
                self._priors[name] = value
            elif isinstance(value, (int, float)):
                self._fixed[name] = float(value)
            else:
                raise TypeError(
                    f"Parameter '{name}' must be Prior or numeric, got {type(value)}"
                )

        # Establish ordering for sampled parameters (stable sorted ordering)
        self._sampled_names = sorted(self._priors.keys())
        self._fixed_names = sorted(self._fixed.keys())
        self._all_names = self._sampled_names + self._fixed_names

    @property
    def sampled_names(self) -> List[str]:
        """List of parameter names that are sampled (have priors)."""
        return list(self._sampled_names)

    @property
    def fixed_names(self) -> List[str]:
        """List of parameter names that are fixed."""
        return list(self._fixed_names)

    @property
    def all_names(self) -> List[str]:
        """All parameter names in order: sampled first, then fixed."""
        return list(self._all_names)

    @property
    def fixed_values(self) -> Dict[str, float]:
        """Dictionary of fixed parameter values."""
        return dict(self._fixed)

    @property
    def n_sampled(self) -> int:
        """Number of sampled parameters."""
        return len(self._priors)

    @property
    def n_fixed(self) -> int:
        """Number of fixed parameters."""
        return len(self._fixed)

    def get_prior(self, name: str) -> Prior:
        """Get prior for a sampled parameter."""
        if name not in self._priors:
            raise KeyError(f"'{name}' is not a sampled parameter")
        return self._priors[name]

    def get_bounds(self) -> List[Tuple[Optional[float], Optional[float]]]:
        """
        Get bounds for sampled parameters as list of (low, high) tuples.

        Useful for bounded optimizers (scipy L-BFGS-B, etc.)
        """
        return [self._priors[name].bounds for name in self._sampled_names]

    def log_prior(self, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Compute joint log prior probability.

        Parameters
        ----------
        theta : jnp.ndarray
            Array of sampled parameter values in self.sampled_names order.

        Returns
        -------
        jnp.ndarray
            Sum of log prior probabilities.
        """
        log_prob = jnp.array(0.0)
        for i, name in enumerate(self._sampled_names):
            log_prob = log_prob + self._priors[name].log_prob(theta[i])
        return log_prob

    def sample(self, rng_key: jax.Array, n_samples: int = 1) -> jnp.ndarray:
        """
        Draw samples from all priors.

        Parameters
        ----------
        rng_key : jax.Array
            JAX random key.
        n_samples : int, optional
            Number of samples to draw. Default is 1.

        Returns
        -------
        jnp.ndarray
            Array of shape (n_samples, n_sampled) with samples.
        """
        keys = random.split(rng_key, len(self._sampled_names))
        samples = []
        for key, name in zip(keys, self._sampled_names):
            samples.append(self._priors[name].sample(key, (n_samples,)))
        return jnp.stack(samples, axis=-1)

    def theta_to_full_pars(
        self,
        theta: jnp.ndarray,
        parameter_names: Tuple[str, ...],
    ) -> Dict[str, float]:
        """
        Convert sampled theta array to full parameter dict including fixed values.

        Parameters
        ----------
        theta : jnp.ndarray
            Array of sampled parameter values.
        parameter_names : tuple of str
            The model's PARAMETER_NAMES defining expected parameters.

        Returns
        -------
        dict
            Full parameter dictionary with both sampled and fixed values.
        """
        # Start with fixed values
        pars = dict(self._fixed)

        # Add sampled values
        for i, name in enumerate(self._sampled_names):
            pars[name] = float(theta[i])

        return pars

    def full_pars_to_theta(self, pars: Dict[str, float]) -> jnp.ndarray:
        """
        Extract sampled parameters from full parameter dict to theta array.

        Parameters
        ----------
        pars : dict
            Full parameter dictionary.

        Returns
        -------
        jnp.ndarray
            Array of sampled parameter values in self.sampled_names order.
        """
        return jnp.array([pars[name] for name in self._sampled_names])

    def __repr__(self) -> str:
        lines = ["PriorDict({"]
        for name in self._sampled_names:
            lines.append(f"    '{name}': {self._priors[name]},")
        for name in self._fixed_names:
            lines.append(f"    '{name}': {self._fixed[name]},  # fixed")
        lines.append("})")
        return "\n".join(lines)

    def __len__(self) -> int:
        """Total number of parameters (sampled + fixed)."""
        return len(self._priors) + len(self._fixed)
