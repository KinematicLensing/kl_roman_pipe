"""
Sampler factory and backend registry.

Provides the `build_sampler()` function for creating sampler instances
by name, following the factory pattern used elsewhere in kl_pipe.
"""

from __future__ import annotations

from typing import Dict, Type, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from kl_pipe.sampling.base import Sampler, SamplerResult
    from kl_pipe.sampling.configs import BaseSamplerConfig
    from kl_pipe.sampling.task import InferenceTask


# Backend registry
_SAMPLER_REGISTRY: Dict[str, Type['Sampler']] = {}


def register_sampler(name: str, sampler_class: Type['Sampler']) -> None:
    """
    Register a sampler backend.

    Parameters
    ----------
    name : str
        Name to register the sampler under (case-insensitive).
    sampler_class : type
        Sampler class to register.

    Examples
    --------
    >>> from kl_pipe.sampling.factory import register_sampler
    >>> from my_custom_sampler import MySampler
    >>> register_sampler('my_sampler', MySampler)
    """
    _SAMPLER_REGISTRY[name.lower()] = sampler_class


def get_available_samplers() -> List[str]:
    """
    Get list of registered sampler names.

    Returns
    -------
    list of str
        Available sampler backend names.

    Examples
    --------
    >>> from kl_pipe.sampling import get_available_samplers
    >>> print(get_available_samplers())
    ['emcee', 'nautilus', 'blackjax', 'nuts', 'hmc', 'ultranest']
    """
    return sorted(_SAMPLER_REGISTRY.keys())


def build_sampler(
    name: str,
    task: 'InferenceTask',
    config: Optional['BaseSamplerConfig'] = None,
) -> 'Sampler':
    """
    Build a sampler instance by name.

    Factory function for creating sampler instances. This is the
    primary entry point for users.

    Parameters
    ----------
    name : str
        Sampler backend name (case-insensitive).
        Available: 'emcee', 'nautilus', 'blackjax', 'nuts', 'hmc'.
        Use `get_available_samplers()` to see all registered backends.
    task : InferenceTask
        The inference task to solve.
    config : BaseSamplerConfig, optional
        Sampler configuration. If None, uses default config for that
        sampler type.

    Returns
    -------
    Sampler
        Configured sampler instance ready to run.

    Raises
    ------
    ValueError
        If the sampler name is not registered.
    TypeError
        If config type doesn't match what the sampler expects.

    Examples
    --------
    >>> from kl_pipe.sampling import build_sampler, EnsembleSamplerConfig
    >>>
    >>> # With custom config
    >>> config = EnsembleSamplerConfig(n_walkers=64, n_iterations=3000)
    >>> sampler = build_sampler('emcee', task, config)
    >>> result = sampler.run()
    >>>
    >>> # With default config
    >>> sampler = build_sampler('emcee', task)
    >>> result = sampler.run()
    >>>
    >>> # Using aliases
    >>> sampler = build_sampler('nuts', task)  # Same as 'blackjax' with NUTS
    """
    name_lower = name.lower()

    if name_lower not in _SAMPLER_REGISTRY:
        available = ', '.join(get_available_samplers())
        raise ValueError(f"Unknown sampler '{name}'. Available: {available}")

    sampler_class = _SAMPLER_REGISTRY[name_lower]

    # If no config provided, create default for this sampler type
    if config is None:
        config = sampler_class.config_class()

    return sampler_class(task, config)


def _register_builtins() -> None:
    """Register built-in sampler backends."""
    from kl_pipe.sampling.emcee import EmceeSampler
    from kl_pipe.sampling.nautilus import NautilusSampler
    from kl_pipe.sampling.blackjax import BlackJAXSampler
    from kl_pipe.sampling.ultranest import UltraNestSampler
    from kl_pipe.sampling.numpyro import NumpyroSampler

    register_sampler('emcee', EmceeSampler)
    register_sampler('nautilus', NautilusSampler)
    register_sampler('blackjax', BlackJAXSampler)
    register_sampler('ultranest', UltraNestSampler)
    register_sampler('numpyro', NumpyroSampler)

    # Aliases for common usage patterns
    # 'nuts' and 'hmc' now point to NumPyro (the recommended gradient sampler)
    register_sampler('nuts', NumpyroSampler)
    register_sampler('hmc', NumpyroSampler)


# Auto-register built-in backends on module import
_register_builtins()
