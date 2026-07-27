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

import math
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

    def to_dict(self) -> Dict[str, Optional[float]]:
        """
        Serialize the prior to a flat, provenance-friendly record.

        Returns a dict with the fixed schema ``{dist, loc, scale, low, high}``
        (unused fields are None), lossless for every built-in prior. Subclasses
        must override; the base raises so an unserializable prior fails loudly.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement to_dict()"
        )


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

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            'dist': 'uniform',
            'loc': None,
            'scale': None,
            'low': float(self.low),
            'high': float(self.high),
        }


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

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            'dist': 'gaussian',
            'loc': float(self.mu),
            'scale': float(self.sigma),
            'low': None,
            'high': None,
        }


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
        return jnp.where(in_bounds, -jnp.log(value) - jnp.log(log_range), -jnp.inf)

    def sample(self, rng_key: jax.Array, shape: Tuple[int, ...] = ()) -> jnp.ndarray:
        log_low = jnp.log(self.low)
        log_high = jnp.log(self.high)
        return jnp.exp(random.uniform(rng_key, shape, minval=log_low, maxval=log_high))

    @property
    def bounds(self) -> Tuple[float, float]:
        return (self.low, self.high)

    def __repr__(self) -> str:
        return f"LogUniform({self.low}, {self.high})"

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            'dist': 'loguniform',
            'loc': None,
            'scale': None,
            'low': float(self.low),
            'high': float(self.high),
        }


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

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            'dist': 'truncated_normal',
            'loc': float(self.mu),
            'scale': float(self.sigma),
            'low': float(self.low),
            'high': float(self.high),
        }


@dataclass(frozen=True)
class TruncatedNormalMixture(Prior):
    """
    Weighted mixture of truncated normals on disjoint or overlapping supports.

    Exists so a population painted from a mixture can be marginalized under
    the distribution that generated it, rather than under a unimodal
    approximation to it. Bulge Sersic index is the motivating case: the
    literature distribution is a pseudobulge/classical mixture split at
    n = 2, and moment-matching it to a single normal would misspecify the
    population being recovered.

    Parameters
    ----------
    components : tuple of TruncatedNormal
        Mixture components.
    weights : tuple of float
        Mixture weights; must be positive and sum to 1.

    Examples
    --------
    >>> # Gadotti 2009 bulge index: 70% pseudobulge, 30% classical
    >>> prior = TruncatedNormalMixture(
    ...     (TruncatedNormal(1.5, 0.9, 0.5, 2.0),
    ...      TruncatedNormal(3.4, 1.3, 2.0, 6.0)),
    ...     (0.7, 0.3),
    ... )
    """

    components: Tuple[TruncatedNormal, ...]
    weights: Tuple[float, ...]

    def __post_init__(self):
        if len(self.components) < 2:
            raise ValueError(
                f"a mixture needs >= 2 components, got {len(self.components)}"
            )
        if len(self.weights) != len(self.components):
            raise ValueError(
                f"{len(self.weights)} weights for {len(self.components)} components"
            )
        if any(w <= 0 for w in self.weights):
            raise ValueError(f"weights must be positive, got {self.weights}")
        total = float(sum(self.weights))
        if abs(total - 1.0) > 1e-12:
            raise ValueError(f"weights must sum to 1, got {total}")

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        # logsumexp over components. Out-of-support components contribute
        # -inf, which logsumexp handles; a value outside EVERY component's
        # support gives -inf overall, as it must.
        terms = jnp.stack(
            [
                jnp.log(w) + c.log_prob(value)
                for c, w in zip(self.components, self.weights)
            ]
        )
        return jax.scipy.special.logsumexp(terms, axis=0)

    def sample(self, rng_key: jax.Array, shape: Tuple[int, ...] = ()) -> jnp.ndarray:
        # pick a component per draw, then sample every component and gather;
        # sampling all of them keeps the shapes static for JIT
        pick_key, *comp_keys = random.split(rng_key, len(self.components) + 1)
        idx = random.choice(
            pick_key, len(self.components), shape=shape, p=jnp.array(self.weights)
        )
        draws = jnp.stack(
            [c.sample(k, shape) for c, k in zip(self.components, comp_keys)]
        )
        return jnp.take_along_axis(draws, idx[None, ...], axis=0)[0]

    @property
    def bounds(self) -> Tuple[float, float]:
        return (
            min(c.low for c in self.components),
            max(c.high for c in self.components),
        )

    def __repr__(self) -> str:
        parts = ', '.join(f'{w}*{c!r}' for c, w in zip(self.components, self.weights))
        return f"TruncatedNormalMixture({parts})"

    def to_dict(self) -> Dict[str, Optional[float]]:
        # flat loc/scale summarize the mixture (weighted mean and total sd);
        # components/weights carry the lossless description. The flat keys are
        # part of the schema every consumer reads, so they must be present
        # even where a single loc/scale cannot capture a bimodal prior.
        mean = sum(w * c.mu for c, w in zip(self.components, self.weights))
        var = sum(
            w * (c.sigma**2 + (c.mu - mean) ** 2)
            for c, w in zip(self.components, self.weights)
        )
        low, high = self.bounds
        return {
            'dist': 'truncated_normal_mixture',
            'loc': float(mean),
            'scale': float(math.sqrt(var)),
            'low': float(low),
            'high': float(high),
            'components': [c.to_dict() for c in self.components],
            'weights': [float(w) for w in self.weights],
        }


@dataclass(frozen=True)
class LogNormal(Prior):
    """
    Log-normal prior: ln(x) ~ Normal(mu, sigma), support x > 0.

    log p(x) = -ln(x) - ln(sigma) - 0.5*ln(2*pi) - (ln(x) - mu)^2 / (2*sigma^2)

    Parameters
    ----------
    mu : float
        Mean of ln(x) (natural log).
    sigma : float
        Standard deviation of ln(x) (must be positive).

    Examples
    --------
    >>> # Tully-Fisher scatter on circular velocity (see make_tf_prior)
    >>> import numpy as np
    >>> prior = LogNormal(np.log(200.0), 0.08 * np.log(10.0))
    """

    mu: float
    sigma: float

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"sigma ({self.sigma}) must be positive")

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        # guard log against non-positive values; masked to -inf below
        safe = jnp.where(value > 0, value, 1.0)
        log_x = jnp.log(safe)
        z = (log_x - self.mu) / self.sigma
        log_pdf = -log_x - jnp.log(self.sigma) - 0.5 * jnp.log(2 * jnp.pi) - 0.5 * z**2
        return jnp.where(value > 0, log_pdf, -jnp.inf)

    def sample(self, rng_key: jax.Array, shape: Tuple[int, ...] = ()) -> jnp.ndarray:
        return jnp.exp(self.mu + self.sigma * random.normal(rng_key, shape))

    @property
    def bounds(self) -> Tuple[float, None]:
        return (0.0, None)

    @property
    def mean(self) -> float:
        """Mean of x: exp(mu + sigma^2 / 2)."""
        return float(jnp.exp(self.mu + 0.5 * self.sigma**2))

    @property
    def std(self) -> float:
        """Standard deviation of x."""
        return float(self.mean * jnp.sqrt(jnp.expm1(self.sigma**2)))

    @property
    def median(self) -> float:
        """Median of x: exp(mu)."""
        return float(jnp.exp(self.mu))

    def __repr__(self) -> str:
        return f"LogNormal({self.mu}, {self.sigma})"

    def to_dict(self) -> Dict[str, Optional[float]]:
        # mu, sigma are in natural-log space (see make_tf_prior for the TF case)
        return {
            'dist': 'lognormal',
            'loc': float(self.mu),
            'scale': float(self.sigma),
            'low': None,
            'high': None,
        }


@dataclass(frozen=True)
class TruncatedLogNormal(Prior):
    """
    Log-normal prior truncated to [low, high], support 0 < low <= x <= high.

    Same density as :class:`LogNormal` renormalized over the truncation
    interval. Finite bounds matter beyond the density itself: grid sizing
    (``RenderConfig.for_priors``) and bounded optimizers read ``bounds``, and
    an unbounded-below size parameter sends the worst-case ``maxk`` to
    infinity.

    Parameters
    ----------
    mu : float
        Mean of ln(x) (natural log).
    sigma : float
        Standard deviation of ln(x) (must be positive).
    low : float
        Lower truncation bound (must be positive).
    high : float
        Upper truncation bound.
    """

    mu: float
    sigma: float
    low: float
    high: float

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"sigma ({self.sigma}) must be positive")
        if self.low <= 0:
            raise ValueError(f"low ({self.low}) must be positive for a log-normal")
        if self.high <= self.low:
            raise ValueError(f"high ({self.high}) must be > low ({self.low})")

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        return _trunc_lognormal_log_prob(
            value, self.mu, self.sigma, self.low, self.high
        )

    def sample(self, rng_key: jax.Array, shape: Tuple[int, ...] = ()) -> jnp.ndarray:
        return _trunc_lognormal_sample(
            rng_key, self.mu, self.sigma, self.low, self.high, shape
        )

    @property
    def bounds(self) -> Tuple[float, float]:
        return (self.low, self.high)

    def __repr__(self) -> str:
        return f"TruncatedLogNormal({self.mu}, {self.sigma}, {self.low}, {self.high})"

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            'dist': 'truncated_lognormal',
            'loc': float(self.mu),
            'scale': float(self.sigma),
            'low': float(self.low),
            'high': float(self.high),
        }


def _norm_cdf(x: jnp.ndarray) -> jnp.ndarray:
    """Standard normal CDF via error function."""
    return 0.5 * (1.0 + jax.scipy.special.erf(x / jnp.sqrt(2.0)))


def _norm_ppf(p: jnp.ndarray) -> jnp.ndarray:
    """Standard normal inverse CDF (quantile function)."""
    return jnp.sqrt(2.0) * jax.scipy.special.erfinv(2.0 * p - 1.0)


def _trunc_lognormal_log_prob(value, mu, sigma, low, high):
    """Truncated log-normal log density; ``mu`` may be a traced array."""
    safe = jnp.where(value > 0, value, 1.0)
    log_x = jnp.log(safe)
    z = (log_x - mu) / sigma
    log_pdf = -log_x - jnp.log(sigma) - 0.5 * jnp.log(2 * jnp.pi) - 0.5 * z**2

    # renormalize over the truncation interval. When mu is traced (the
    # conditional case) this factor depends on a sampled parameter, so
    # dropping it would give the wrong gradient with respect to the parent.
    alpha = (jnp.log(low) - mu) / sigma
    beta = (jnp.log(high) - mu) / sigma
    log_Z = jnp.log(_norm_cdf(beta) - _norm_cdf(alpha))

    in_bounds = (value >= low) & (value <= high)
    return jnp.where(in_bounds, log_pdf - log_Z, -jnp.inf)


def _trunc_lognormal_sample(rng_key, mu, sigma, low, high, shape):
    """Inverse-CDF draw from a truncated log-normal; ``mu`` may be traced."""
    cdf_low = _norm_cdf((jnp.log(low) - mu) / sigma)
    cdf_high = _norm_cdf((jnp.log(high) - mu) / sigma)
    u = random.uniform(rng_key, shape)
    p = cdf_low + u * (cdf_high - cdf_low)
    return jnp.exp(mu + sigma * _norm_ppf(p))


@dataclass(frozen=True)
class ConditionalLogNormal(Prior):
    """
    Log-normal prior on the ratio of this parameter to another sampled one.

    If ``r = x / x_parent`` is log-normal with median ``exp(mu_ratio)`` and
    log-scatter ``sigma_ratio``, then ``x`` given the parent is log-normal
    with location shifted by ``ln(x_parent)``. Expressing the relation this
    way keeps the model parameters intrinsic to their components -- nothing
    is reparameterized into a ratio -- while the relational structure lives
    entirely in the prior.

    The truncation bounds are absolute, not relative to the parent, so the
    support does not move with the parent and ``RenderConfig.for_priors`` and
    ``UnconstrainingTransform`` see a static interval.

    ``PriorDict`` resolves the parent value and calls ``log_prob_given``;
    calling ``log_prob`` directly raises, since the density is undefined
    without a parent.

    Parameters
    ----------
    parent : str
        Name of the sampled parameter this prior conditions on.
    mu_ratio : float
        Mean of ln(x / x_parent) (natural log).
    sigma_ratio : float
        Standard deviation of ln(x / x_parent) (must be positive).
    low, high : float
        Absolute truncation bounds on x (low must be positive).

    Examples
    --------
    >>> # kinematic turnover radius at 0.4x the disk scale length
    >>> import math
    >>> prior = ConditionalLogNormal(
    ...     'F129.disk_rscale', math.log(0.4), 0.25 * math.log(10.0), 0.01, 5.0
    ... )
    """

    parent: str
    mu_ratio: float
    sigma_ratio: float
    low: float
    high: float

    def __post_init__(self):
        if not self.parent:
            raise ValueError("parent must be a non-empty parameter name")
        if self.sigma_ratio <= 0:
            raise ValueError(f"sigma_ratio ({self.sigma_ratio}) must be positive")
        if self.low <= 0:
            raise ValueError(f"low ({self.low}) must be positive for a log-normal")
        if self.high <= self.low:
            raise ValueError(f"high ({self.high}) must be > low ({self.low})")

    def log_prob_given(
        self, value: jnp.ndarray, parent_value: jnp.ndarray
    ) -> jnp.ndarray:
        """Log density of ``value`` given the parent's value. JIT-safe."""
        safe_parent = jnp.where(parent_value > 0, parent_value, 1.0)
        mu = self.mu_ratio + jnp.log(safe_parent)
        lp = _trunc_lognormal_log_prob(value, mu, self.sigma_ratio, self.low, self.high)
        return jnp.where(parent_value > 0, lp, -jnp.inf)

    def sample_given(
        self,
        rng_key: jax.Array,
        parent_value: jnp.ndarray,
        shape: Tuple[int, ...] = (),
    ) -> jnp.ndarray:
        """Draw from the conditional given the parent's value."""
        mu = self.mu_ratio + jnp.log(parent_value)
        return _trunc_lognormal_sample(
            rng_key, mu, self.sigma_ratio, self.low, self.high, shape
        )

    def at_parent(self, parent_value: float) -> TruncatedLogNormal:
        """Collapse to an unconditional prior at a concrete parent value.

        Used when the parent is a fixed (not sampled) parameter, and by
        diagnostics that need a plain prior object.
        """
        pv = float(parent_value)
        if pv <= 0:
            raise ValueError(f"parent value ({pv}) must be positive")
        return TruncatedLogNormal(
            self.mu_ratio + math.log(pv), self.sigma_ratio, self.low, self.high
        )

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        raise TypeError(
            f"ConditionalLogNormal is undefined without its parent "
            f"'{self.parent}'; use log_prob_given(value, parent_value). "
            f"PriorDict.log_prior resolves this automatically."
        )

    def sample(self, rng_key: jax.Array, shape: Tuple[int, ...] = ()) -> jnp.ndarray:
        raise TypeError(
            f"ConditionalLogNormal is undefined without its parent "
            f"'{self.parent}'; use sample_given(rng_key, parent_value, shape). "
            f"PriorDict.sample resolves this automatically."
        )

    @property
    def bounds(self) -> Tuple[float, float]:
        return (self.low, self.high)

    @property
    def ratio_median(self) -> float:
        """Median of x / x_parent."""
        return float(math.exp(self.mu_ratio))

    def __repr__(self) -> str:
        return (
            f"ConditionalLogNormal({self.parent!r}, {self.mu_ratio}, "
            f"{self.sigma_ratio}, {self.low}, {self.high})"
        )

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            'dist': 'conditional_lognormal',
            'parent': self.parent,
            'loc': float(self.mu_ratio),
            'scale': float(self.sigma_ratio),
            'low': float(self.low),
            'high': float(self.high),
        }


def make_tf_prior(v_center_kms: float, sigma_tf_dex: float) -> LogNormal:
    """
    Build a Tully-Fisher LogNormal prior on circular velocity.

    Encodes TF scatter quoted in dex (log10) as a natural-log LogNormal:
    mu = ln(v_center), sigma = sigma_tf_dex * ln(10).

    Parameters
    ----------
    v_center_kms : float
        Central circular velocity in km/s (must be positive).
    sigma_tf_dex : float
        TF scatter in dex (log10 scatter; must be positive).

    Returns
    -------
    LogNormal
        Prior with median v_center_kms and log-scatter sigma_tf_dex dex.

    Examples
    --------
    >>> prior = make_tf_prior(200.0, 0.08)  # sigma_ln ~ 0.184
    """
    if v_center_kms <= 0:
        raise ValueError(f"v_center_kms ({v_center_kms}) must be positive")
    if sigma_tf_dex <= 0:
        raise ValueError(f"sigma_tf_dex ({sigma_tf_dex}) must be positive")
    return LogNormal(math.log(v_center_kms), sigma_tf_dex * math.log(10.0))


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

        # a conditional whose parent is fixed is not conditional on anything
        # sampled -- collapse it to the plain truncated log-normal it induces,
        # so describe() records the prior that was actually used
        for name, prior in list(self._priors.items()):
            if isinstance(prior, ConditionalLogNormal) and prior.parent in self._fixed:
                collapsed = prior.at_parent(self._fixed[prior.parent])
                self._priors[name] = collapsed
                self._param_spec[name] = collapsed

        # Establish ordering for sampled parameters (stable sorted ordering)
        self._sampled_names = sorted(self._priors.keys())
        self._fixed_names = sorted(self._fixed.keys())
        self._all_names = self._sampled_names + self._fixed_names

        # Conditional-prior dependency graph over sampled parameters
        self._conditional_parents: Dict[str, str] = {
            name: prior.parent
            for name, prior in self._priors.items()
            if isinstance(prior, ConditionalLogNormal)
        }
        for name, parent in self._conditional_parents.items():
            if parent not in self._priors:
                raise KeyError(
                    f"conditional prior '{name}' names parent '{parent}', which is "
                    f"not a parameter of this PriorDict"
                )
        self._topological_order = self._build_topological_order()
        self._parent_index = {
            name: self._sampled_names.index(parent)
            for name, parent in self._conditional_parents.items()
        }

    def _build_topological_order(self) -> List[str]:
        """Sampled names ordered so every conditional follows its parent.

        Alphabetical apart from the reordering the dependencies force, so the
        order is deterministic for a given spec. Raises on a dependency cycle.
        """
        order: List[str] = []
        done = set()
        visiting = set()

        def visit(name: str, chain: List[str]) -> None:
            if name in done:
                return
            if name in visiting:
                cycle = ' -> '.join(chain + [name])
                raise ValueError(f"conditional prior dependency cycle: {cycle}")
            visiting.add(name)
            parent = self._conditional_parents.get(name)
            if parent is not None:
                visit(parent, chain + [name])
            visiting.discard(name)
            done.add(name)
            order.append(name)

        for name in self._sampled_names:
            visit(name, [])
        return order

    @property
    def conditional_parents(self) -> Dict[str, str]:
        """Map of conditional parameter name -> the name it conditions on."""
        return dict(self._conditional_parents)

    @property
    def topological_order(self) -> List[str]:
        """Sampled names ordered so every conditional follows its parent."""
        return list(self._topological_order)

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

    def get_param_bounds(self, name: str) -> Tuple[Optional[float], Optional[float]]:
        """Return (low, high) for a parameter, sampled or fixed.

        Tri-state:
        - sampled parameter -> the prior's ``.bounds``
        - fixed parameter   -> ``(value, value)``
        - absent            -> ``(None, None)``

        Useful for callers that need to reason about parameter ranges
        without caring whether the parameter is being sampled or fixed
        (e.g., model construction-time prior validation).
        """
        if name in self._sampled_names:
            return self._priors[name].bounds
        if name in self._fixed:
            v = self._fixed[name]
            return (v, v)
        return (None, None)

    def describe(self) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Serialize every parameter's prior to a flat provenance record.

        Returns a dict mapping each parameter name (sampled and fixed) to the
        flat ``{dist, loc, scale, low, high}`` schema. Fixed parameters use
        ``dist='fixed'`` with ``loc`` holding the pinned value. Conditional
        priors add ``parent``; mixtures add ``components`` and ``weights``
        alongside their flat summary. Records exactly how each param was set
        per fit.
        """
        out: Dict[str, Dict[str, Optional[float]]] = {}
        for name in self._sampled_names:
            out[name] = self._priors[name].to_dict()
        for name in self._fixed_names:
            out[name] = {
                'dist': 'fixed',
                'loc': float(self._fixed[name]),
                'scale': None,
                'low': None,
                'high': None,
            }
        return out

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
            prior = self._priors[name]
            if name in self._parent_index:
                term = prior.log_prob_given(theta[i], theta[self._parent_index[name]])
            else:
                term = prior.log_prob(theta[i])
            log_prob = log_prob + term
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
        # keys are assigned by alphabetical index, not draw order, so a
        # parameter's draws are unchanged by whether some other parameter in
        # the dict happens to be conditional
        keys = random.split(rng_key, len(self._sampled_names))
        key_for = dict(zip(self._sampled_names, keys))

        drawn: Dict[str, jnp.ndarray] = {}
        for name in self._topological_order:
            prior = self._priors[name]
            if name in self._conditional_parents:
                parent = drawn[self._conditional_parents[name]]
                drawn[name] = prior.sample_given(key_for[name], parent, (n_samples,))
            else:
                drawn[name] = prior.sample(key_for[name], (n_samples,))

        return jnp.stack([drawn[name] for name in self._sampled_names], axis=-1)

    def theta_to_full_pars(self, theta: jnp.ndarray) -> Dict[str, float]:
        """
        Convert sampled theta array to full parameter dict including fixed values.

        ``theta`` is indexed in ``self.sampled_names`` (alphabetical) order.

        Parameters
        ----------
        theta : jnp.ndarray
            Array of sampled parameter values.

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
