"""
Configuration classes for MCMC samplers.

Each sampler category has its own config class with only relevant fields,
avoiding confusion about which options apply to which samplers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any, Dict, Union
from pathlib import Path

import yaml


@dataclass
class BaseSamplerConfig:
    """
    Minimal configuration shared by all samplers.

    Attributes
    ----------
    seed : int, optional
        Random seed for reproducibility. If None, uses system entropy.
    progress : bool
        Whether to show progress bar during sampling.
    """

    seed: Optional[int] = None
    progress: bool = True


@dataclass
class EnsembleSamplerConfig(BaseSamplerConfig):
    """
    Configuration for ensemble MCMC samplers (emcee, zeus).

    Ensemble samplers use multiple "walkers" that communicate to explore
    the posterior. They are gradient-free and work well for multi-modal
    distributions.

    Attributes
    ----------
    n_walkers : int
        Number of walkers in the ensemble. Should be at least 2 * n_params.
    n_iterations : int
        Total number of iterations to run (including burn-in).
    burn_in : int
        Number of initial iterations to discard as burn-in.
    thin : int
        Thinning factor for samples (keep every thin-th sample).
    moves : Any, optional
        Custom move proposals (emcee-specific). If None, uses default.
    vectorize : bool
        Whether the log_prob function can be vectorized over walkers.

    Examples
    --------
    >>> config = EnsembleSamplerConfig(
    ...     n_walkers=64,
    ...     n_iterations=5000,
    ...     burn_in=1000,
    ...     seed=42,
    ... )
    """

    n_walkers: int = 32
    n_iterations: int = 2000
    burn_in: int = 500
    thin: int = 1
    moves: Optional[Any] = None
    vectorize: bool = False

    def __post_init__(self):
        if self.n_walkers < 2:
            raise ValueError("n_walkers must be >= 2")
        if self.burn_in >= self.n_iterations:
            raise ValueError("burn_in must be < n_iterations")
        if self.thin < 1:
            raise ValueError("thin must be >= 1")


@dataclass
class NestedSamplerConfig(BaseSamplerConfig):
    """
    Configuration for nested sampling algorithms (nautilus, ultranest).

    Nested samplers explore the posterior by tracking iso-likelihood contours.
    They naturally provide evidence (marginal likelihood) estimates for
    model comparison.

    Attributes
    ----------
    n_live : int
        Number of live points. Higher values give more accurate evidence
        estimates but are slower.
    n_networks : int
        Number of neural networks in the ensemble (nautilus-specific).
    verbose : bool
        Whether to print detailed progress information.
    log_dir : str, optional
        Directory for saving checkpoints (ultranest-specific).
    resume : bool
        Whether to resume from a previous run (ultranest-specific).

    Examples
    --------
    >>> config = NestedSamplerConfig(
    ...     n_live=500,
    ...     seed=42,
    ... )
    """

    n_live: int = 500
    n_networks: int = 4
    verbose: bool = True
    log_dir: Optional[str] = None
    resume: bool = False


@dataclass
class GradientSamplerConfig(BaseSamplerConfig):
    """
    Configuration for gradient-based MCMC (BlackJAX HMC/NUTS).

    These samplers use the gradient of the log posterior to make
    efficient proposals. They require differentiable likelihoods
    (which JAX provides via autodiff).

    Attributes
    ----------
    n_samples : int
        Number of samples to draw (after warmup).
    n_warmup : int
        Number of warmup iterations for adaptation.
    max_tree_depth : int
        Maximum tree depth for NUTS. Higher values allow longer
        trajectories but are more expensive.
    target_acceptance : float
        Target acceptance probability for step size adaptation.
        Typically 0.6-0.9, with 0.8 being a good default.
    step_size : float
        Initial step size for HMC. Only used if not doing adaptation.
    num_integration_steps : int
        Number of leapfrog integration steps (HMC only, NUTS adapts this).
    algorithm : str
        Which algorithm to use: 'nuts' (default) or 'hmc'.

    Examples
    --------
    >>> config = GradientSamplerConfig(
    ...     n_samples=2000,
    ...     n_warmup=500,
    ...     seed=42,
    ... )
    """

    n_samples: int = 2000
    n_warmup: int = 500
    max_tree_depth: int = 10
    target_acceptance: float = 0.8
    step_size: float = 0.1
    num_integration_steps: int = 10
    algorithm: str = 'nuts'

    def __post_init__(self):
        if self.algorithm not in ('nuts', 'hmc'):
            raise ValueError(f"algorithm must be 'nuts' or 'hmc', got '{self.algorithm}'")
        if not 0 < self.target_acceptance < 1:
            raise ValueError("target_acceptance must be in (0, 1)")


# =============================================================================
# NumPyro Sampler Configuration
# =============================================================================


class ReparamStrategy(str, Enum):
    """
    Strategy for numerical conditioning in gradient-based samplers.

    The Z-score reparameterization transforms sampling to a standardized
    latent space where all parameters have O(1) scale. This is critical
    for joint models where gradients can vary by 10^4+ across parameters.

    Attributes
    ----------
    NONE : str
        Sample directly in physical space. Fast startup but may have
        poor conditioning for multi-scale problems.
    PRIOR : str
        Use prior mean/std for Z-score transform. Default choice.
        Fast, requires no pre-computation, works well when prior
        roughly matches posterior scale.
    EMPIRICAL : str
        Run short pre-conditioning phase to estimate posterior
        scales, then use those for the main sampling. Slower
        startup but better conditioning for challenging problems.
    """

    NONE = 'none'
    PRIOR = 'prior'
    EMPIRICAL = 'empirical'


@dataclass
class NumpyroSamplerConfig(BaseSamplerConfig):
    """
    Configuration for NumPyro NUTS/HMC sampler.

    NumPyro provides robust gradient-based MCMC with superior mass matrix
    adaptation compared to raw BlackJAX. Recommended for joint models with
    parameters spanning multiple scales (e.g., intensity ~10^7, velocity ~10^3).

    The key feature is Z-score reparameterization via ``reparam_strategy``,
    which normalizes parameter scales to prevent numerical collapse.

    Attributes
    ----------
    n_samples : int
        Number of posterior samples per chain (after warmup).
    n_warmup : int
        Warmup iterations for step size and mass matrix adaptation.
    n_chains : int
        Number of independent chains. Multiple chains enable R-hat diagnostics.
        Minimum 4 chains recommended for reliable convergence assessment.
    dense_mass : bool
        Use dense (True) or diagonal (False) mass matrix. Dense handles
        parameter correlations but is O(n_params²). Default True for safety.
    max_tree_depth : int
        Maximum NUTS tree depth. Higher allows longer trajectories (2^depth
        leapfrog steps max). Increase if hitting depth limit frequently.
    target_accept_prob : float
        Target acceptance probability for dual averaging. 0.8 is standard;
        increase to 0.9-0.95 for challenging posteriors.
    reparam_strategy : ReparamStrategy
        Numerical conditioning strategy:

        - 'none': Sample in physical space (no transform)
        - 'prior': Z-score using prior mean/std (default, fast)
        - 'empirical': Estimate scales from short warmup (slower, robust)
    empirical_warmup_frac : float
        Fraction of n_warmup for empirical preconditioning (if strategy='empirical').
    chain_method : str
        How to run multiple chains:

        - 'sequential': Run chains one after another (default, works everywhere)
        - 'parallel': Run chains in parallel via JAX pmap (requires multi-device)
        - 'vectorized': Vectorize across chains (single device, memory intensive)
    save_warmup : bool
        Whether to save warmup samples in result.
    save_mass_matrix : bool
        Whether to save adapted inverse mass matrix in diagnostics.
        Off by default since it can be large for dense_mass=True.
    init_strategy : str
        Initialization strategy:

        - 'prior': Sample initial positions from prior (default)
        - 'median': Start at prior medians
        - 'jitter': Prior median with small random perturbation

    Examples
    --------
    >>> # Default config (good for most joint models)
    >>> config = NumpyroSamplerConfig(
    ...     n_samples=2000,
    ...     n_warmup=1000,
    ...     n_chains=4,
    ...     seed=42,
    ... )

    >>> # For very challenging posteriors
    >>> config = NumpyroSamplerConfig(
    ...     n_samples=2000,
    ...     n_warmup=2000,
    ...     n_chains=4,
    ...     dense_mass=True,
    ...     reparam_strategy='empirical',
    ...     target_accept_prob=0.9,
    ... )

    See Also
    --------
    GradientSamplerConfig : BlackJAX configuration (simpler but less robust)
    """

    n_samples: int = 2000
    n_warmup: int = 1000
    n_chains: int = 4

    dense_mass: bool = True
    max_tree_depth: int = 10
    target_accept_prob: float = 0.8

    reparam_strategy: ReparamStrategy = ReparamStrategy.PRIOR
    empirical_warmup_frac: float = 0.1

    chain_method: str = 'sequential'
    save_warmup: bool = False
    save_mass_matrix: bool = False

    init_strategy: str = 'prior'

    def __post_init__(self):
        if not 0 < self.target_accept_prob < 1:
            raise ValueError("target_accept_prob must be in (0, 1)")
        if self.n_chains < 1:
            raise ValueError("n_chains must be >= 1")
        if self.chain_method not in ('sequential', 'parallel', 'vectorized'):
            raise ValueError(
                f"chain_method must be 'sequential', 'parallel', or 'vectorized', "
                f"got '{self.chain_method}'"
            )
        if self.init_strategy not in ('prior', 'median', 'jitter'):
            raise ValueError(
                f"init_strategy must be 'prior', 'median', or 'jitter', "
                f"got '{self.init_strategy}'"
            )
        # Convert string to enum if needed
        if isinstance(self.reparam_strategy, str):
            try:
                object.__setattr__(
                    self, 'reparam_strategy', ReparamStrategy(self.reparam_strategy)
                )
            except ValueError:
                valid = [e.value for e in ReparamStrategy]
                raise ValueError(
                    f"reparam_strategy must be one of {valid}, "
                    f"got '{self.reparam_strategy}'"
                )


# =============================================================================
# YAML Configuration Loading
# =============================================================================

# Mapping from YAML prior type names to classes
PRIOR_TYPES = {
    'uniform': 'Uniform',
    'gaussian': 'Gaussian',
    'normal': 'Gaussian',
    'loguniform': 'LogUniform',
    'log_uniform': 'LogUniform',
    'truncated_normal': 'TruncatedNormal',
    'truncatednormal': 'TruncatedNormal',
}


def parse_prior_spec(spec: Union[dict, float, int]):
    """
    Parse a prior specification from YAML.

    Parameters
    ----------
    spec : dict or numeric
        Prior specification. If numeric, returns as fixed value.
        If dict, must have 'type' key specifying prior distribution.

    Returns
    -------
    Prior or float
        Prior instance or fixed value.

    Examples
    --------
    >>> parse_prior_spec(10.0)
    10.0
    >>> parse_prior_spec({'type': 'uniform', 'low': 0, 'high': 1})
    Uniform(0.0, 1.0)
    """
    from kl_pipe.priors import Uniform, Gaussian, LogUniform, TruncatedNormal

    if isinstance(spec, (int, float)):
        return float(spec)

    if not isinstance(spec, dict):
        raise TypeError(f"Prior spec must be dict or numeric, got {type(spec)}")

    if 'type' not in spec:
        raise ValueError("Prior spec dict must have 'type' key")

    prior_type = spec['type'].lower()

    if prior_type not in PRIOR_TYPES:
        available = ', '.join(PRIOR_TYPES.keys())
        raise ValueError(f"Unknown prior type '{prior_type}'. Available: {available}")

    # Get prior class
    prior_classes = {
        'Uniform': Uniform,
        'Gaussian': Gaussian,
        'LogUniform': LogUniform,
        'TruncatedNormal': TruncatedNormal,
    }
    prior_class = prior_classes[PRIOR_TYPES[prior_type]]

    # Build kwargs, excluding 'type'
    kwargs = {k: v for k, v in spec.items() if k != 'type'}

    return prior_class(**kwargs)


@dataclass
class SamplingYAMLConfig:
    """
    Complete configuration for a sampling run loaded from YAML.

    This class is used to load and validate YAML configuration files
    that specify the full inference setup.

    Attributes
    ----------
    model_type : str
        Model type ('velocity', 'intensity', 'joint').
    velocity_model : str, optional
        Velocity model name (e.g., 'offset', 'centered').
    intensity_model : str, optional
        Intensity model name (e.g., 'inclined_exp').
    priors : dict
        Prior specifications for each parameter.
    sampler : str
        Sampler backend name ('emcee', 'nautilus', 'blackjax').
    sampler_config : dict
        Options for the sampler config.
    data : dict
        Data file paths and specifications.
    meta_pars : dict, optional
        Additional metadata (PSF, etc.).
    output : dict, optional
        Output configuration (paths, formats).
    """

    model_type: str
    priors: Dict[str, Any]
    sampler: str
    sampler_config: Dict[str, Any] = field(default_factory=dict)
    velocity_model: Optional[str] = None
    intensity_model: Optional[str] = None
    shared_params: Optional[list] = None
    data: Dict[str, Any] = field(default_factory=dict)
    meta_pars: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'SamplingYAMLConfig':
        """
        Load configuration from YAML file.

        Parameters
        ----------
        path : str or Path
            Path to YAML configuration file.

        Returns
        -------
        SamplingYAMLConfig
            Loaded configuration.
        """
        path = Path(path)

        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def get_prior_dict(self):
        """
        Convert priors specification to PriorDict.

        Returns
        -------
        PriorDict
            Prior dictionary with parsed priors.
        """
        from kl_pipe.priors import PriorDict

        parsed = {}
        for name, spec in self.priors.items():
            parsed[name] = parse_prior_spec(spec)
        return PriorDict(parsed)

    def get_sampler_config(self) -> BaseSamplerConfig:
        """
        Convert sampler_config to appropriate config class.

        Returns
        -------
        BaseSamplerConfig
            Sampler configuration of the appropriate type.
        """
        sampler_lower = self.sampler.lower()

        if sampler_lower in ('emcee', 'zeus'):
            return EnsembleSamplerConfig(**self.sampler_config)
        elif sampler_lower in ('nautilus', 'ultranest'):
            return NestedSamplerConfig(**self.sampler_config)
        elif sampler_lower in ('blackjax',):
            return GradientSamplerConfig(**self.sampler_config)
        elif sampler_lower in ('numpyro', 'nuts', 'hmc'):
            return NumpyroSamplerConfig(**self.sampler_config)
        else:
            raise ValueError(f"Unknown sampler type: {self.sampler}")
