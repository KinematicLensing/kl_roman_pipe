"""
MCMC sampling infrastructure for kinematic lensing inference.

This module provides a unified interface for running MCMC samplers on
kinematic lensing models. It supports multiple backends (emcee, nautilus,
BlackJAX) through a common interface.

Quick Start
-----------
>>> from kl_pipe.velocity import OffsetVelocityModel
>>> from kl_pipe.priors import Uniform, Gaussian, TruncatedNormal, PriorDict
>>> from kl_pipe.sampling import (
...     InferenceTask, EnsembleSamplerConfig, build_sampler
... )
>>>
>>> # Define priors (sampled) and fixed values
>>> priors = PriorDict({
...     'vcirc': Uniform(100, 350),
...     'cosi': TruncatedNormal(0.5, 0.2, 0.1, 0.99),
...     'v0': 10.0,  # Fixed
... })
>>>
>>> # Create inference task
>>> task = InferenceTask.from_velocity_model(
...     model=OffsetVelocityModel(),
...     priors=priors,
...     data_vel=data,
...     variance_vel=25.0,
...     image_pars=image_pars,
... )
>>>
>>> # Configure and run sampler
>>> config = EnsembleSamplerConfig(n_walkers=64, n_iterations=5000)
>>> sampler = build_sampler('emcee', task, config)
>>> result = sampler.run()
>>>
>>> # Analyze results
>>> summary = result.get_summary()

See Also
--------
kl_pipe.sampling.README : Detailed documentation and design philosophy
"""

from kl_pipe.sampling.base import Sampler, SamplerResult
from kl_pipe.sampling.configs import (
    BaseSamplerConfig,
    EnsembleSamplerConfig,
    NestedSamplerConfig,
    GradientSamplerConfig,
    NumpyroSamplerConfig,
    ReparamStrategy,
)
from kl_pipe.sampling.task import InferenceTask
from kl_pipe.sampling.factory import (
    build_sampler,
    get_available_samplers,
    register_sampler,
)

# Import backends for registration (they auto-register on import)
from kl_pipe.sampling.emcee import EmceeSampler
from kl_pipe.sampling.nautilus import NautilusSampler
from kl_pipe.sampling.blackjax import BlackJAXSampler
from kl_pipe.sampling.ultranest import UltraNestSampler
from kl_pipe.sampling.numpyro import NumpyroSampler

__all__ = [
    # Core classes
    'Sampler',
    'SamplerResult',
    'InferenceTask',
    # Config classes
    'BaseSamplerConfig',
    'EnsembleSamplerConfig',
    'NestedSamplerConfig',
    'GradientSamplerConfig',
    'NumpyroSamplerConfig',
    'ReparamStrategy',
    # Factory
    'build_sampler',
    'get_available_samplers',
    'register_sampler',
    # Backends
    'EmceeSampler',
    'NautilusSampler',
    'BlackJAXSampler',
    'UltraNestSampler',
    'NumpyroSampler',
]
