"""
Likelihood functions for kinematic-lensing models.

Provides JAX-compatible log-likelihood functions for:
- Velocity-only observations (via VelocityObs)
- Intensity-only observations (via ImageObs)
- Combined velocity + intensity observations

All functions are designed to be JIT-compilable and support automatic differentiation.
The likelihood functions include proper normalization constants for model comparison.

Examples
--------
Basic usage via the SourceModel inference path:

>>> from kl_pipe.likelihood import create_jitted_likelihood_from_obs
>>> log_like = create_jitted_likelihood_from_obs(
...     source, sampled_names, fixed_pars, velocity_obs=obs_vel
... )
>>> log_prob = log_like(theta_sampled)
>>>
>>> # compute gradients
>>> grad_fn = jax.grad(log_like)
>>> gradient = grad_fn(theta_sampled)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from kl_pipe.observation import ImageObs, VelocityObs, GrismObs
    from kl_pipe.source import SourceModel


# ==============================================================================
# SourceModel-aware likelihoods
# ==============================================================================
#
# These primitives consume a SourceModel + a dotted-key parameter dict, not a
# flat theta in a model's PARAMETER_NAMES order. ``theta_sampled`` is the
# array indexed by the priors' sampled-name ordering; the primitives build the
# per-call ``pars`` dict by combining ``theta_sampled`` (traced) with the
# closed-over ``sampled_names`` (static aux) and ``fixed_pars`` (static aux),
# then dispatch to ``SourceModel.render_*`` for that channel.


def _build_pars_dict(
    theta_sampled: jnp.ndarray,
    sampled_names: tuple,
    fixed_pars: dict,
) -> dict:
    """Merge sampled (traced) + fixed (static) values into a dotted-key dict.

    ``sampled_names`` is the priors' sorted-sample-name tuple (static aux);
    ``fixed_pars`` is the dict of fixed-value priors (static aux). Result is
    a dict the same shape callers (SourceModel.render_*) expect.
    """
    pars = dict(fixed_pars)
    for i, name in enumerate(sampled_names):
        pars[name] = theta_sampled[i]
    return pars


def _gaussian_log_likelihood(
    data: jnp.ndarray,
    model: jnp.ndarray,
    variance,
    mask=None,
) -> float:
    """Gaussian log-likelihood with normalization constants.

    ``log L = -0.5 * [N*log(2pi) + sum log(sigma^2) + chi^2]``.

    Factored out so the SourceModel-channel likelihoods share one shape.
    The ``mask is not None`` branch resolves at JIT trace time (mask is
    static aux from the obs pytree).
    """
    residuals = data - model
    variance = jnp.broadcast_to(jnp.asarray(variance), data.shape)
    if mask is not None:
        chi2 = jnp.sum(jnp.where(mask, residuals**2 / variance, 0.0))
        n_data = jnp.sum(mask).astype(float)
        log_det = jnp.sum(jnp.where(mask, jnp.log(variance), 0.0))
    else:
        chi2 = jnp.sum(residuals**2 / variance)
        n_data = float(data.size)
        log_det = jnp.sum(jnp.log(variance))
    normalization = -0.5 * n_data * jnp.log(2.0 * jnp.pi) - 0.5 * log_det
    return normalization - 0.5 * chi2


def _log_likelihood_broadband_source(
    theta_sampled: jnp.ndarray,
    source: 'SourceModel',
    obs: 'ImageObs',
    band_key: str,
    sampled_names: tuple,
    fixed_pars: dict,
) -> float:
    """SourceModel broadband log-likelihood for one filter."""
    pars = _build_pars_dict(theta_sampled, sampled_names, fixed_pars)
    model_img = source.render_broadband(pars, obs, band_key)
    return _gaussian_log_likelihood(obs.data, model_img, obs.variance, obs.mask)


def _log_likelihood_grism_source(
    theta_sampled: jnp.ndarray,
    source: 'SourceModel',
    obs: 'GrismObs',
    sampled_names: tuple,
    fixed_pars: dict,
    spectral_oversample: int = 15,
) -> float:
    """SourceModel grism log-likelihood for one dispersed observation."""
    pars = _build_pars_dict(theta_sampled, sampled_names, fixed_pars)
    model_img = source.render_grism(pars, obs, spectral_oversample=spectral_oversample)
    return _gaussian_log_likelihood(obs.data, model_img, obs.variance, obs.mask)

def _log_likelihood_fiber_source(
    theta_sampled: jnp.ndarray,
    source: 'SourceModel',
    obs: 'FiberObs',
    sampled_names: tuple,
    fixed_pars: dict,
    spectral_oversample: int = 15,
) -> float:
    pars = _build_pars_dict(theta_sampled, sampled_names, fixed_pars)
    model_img = source.render_fiber(pars, obs, spectral_oversample=spectral_oversample)
    return _gaussian_log_likelihood(obs.data, model_img, obs.variance, obs.mask)


def _log_likelihood_velocity_source(
    theta_sampled: jnp.ndarray,
    source: 'SourceModel',
    obs: 'VelocityObs',
    sampled_names: tuple,
    fixed_pars: dict,
) -> float:
    """SourceModel velocity log-likelihood for one velocity-map observation."""
    pars = _build_pars_dict(theta_sampled, sampled_names, fixed_pars)
    model_v = source.render_velocity(pars, obs)
    return _gaussian_log_likelihood(obs.data, model_v, obs.variance, obs.mask)

#need to add fiberobs into this
def _log_likelihood_total_source(
    theta_sampled: jnp.ndarray,
    source: 'SourceModel',
    image_obs: dict,
    grism_obs: dict,
    fiber_obs: dict,
    velocity_obs,
    sampled_names: tuple,
    fixed_pars: dict,
    spectral_oversample: int = 15,
) -> float:
    """Dispatch sum over all populated channels for SourceModel inference.

    ``image_obs`` and ``grism_obs`` are dicts (or None / empty) keyed by
    component name (e.g. ``'F087'``, ``'roll0'``). ``velocity_obs`` is a
    single VelocityObs or None. The ``if ... is not None`` and dict-iter
    branches resolve at JIT trace time because the obs structure is fixed
    (closed over via partial).
    """
    log_l = 0.0
    if image_obs:
        for band_key, obs in image_obs.items():
            log_l = log_l + _log_likelihood_broadband_source(
                theta_sampled, source, obs, band_key, sampled_names, fixed_pars
            )
    if grism_obs:
        for _grism_key, obs in grism_obs.items():
            log_l = log_l + _log_likelihood_grism_source(
                theta_sampled,
                source,
                obs,
                sampled_names,
                fixed_pars,
                spectral_oversample=spectral_oversample,
            )

    if fiber_obs:
        for _fiber_key, obs in fiber_obs.items():
            log_l = log_l + _log_likelihood_fiber_source(
                theta_sampled,
                source,
                obs,
                sampled_names,
                fixed_pars,
                spectral_oversample=spectral_oversample,
            )

    if velocity_obs is not None:
        log_l = log_l + _log_likelihood_velocity_source(
            theta_sampled, source, velocity_obs, sampled_names, fixed_pars
        )
    return log_l


def create_jitted_likelihood_from_obs(
    source: 'SourceModel',
    sampled_names: tuple,
    fixed_pars: dict,
    *,
    image_obs: dict = None,
    grism_obs: dict = None,
    fiber_obs: dict = None,
    velocity_obs=None,
    spectral_oversample: int = 15,
) -> Callable[[jnp.ndarray], float]:
    """JIT-compiled total log-likelihood for SourceModel-based inference.

    The returned function takes ``theta_sampled`` (priors-ordered) and
    returns the Gaussian log-likelihood summed over all provided channels.
    All obs + the source + the static name/fixed tuples are frozen via
    ``functools.partial``.
    """
    return jax.jit(
        partial(
            _log_likelihood_total_source,
            source=source,
            image_obs=image_obs,
            grism_obs=grism_obs,
            fiber_obs=fiber_obs,
            velocity_obs=velocity_obs,
            sampled_names=sampled_names,
            fixed_pars=fixed_pars,
            spectral_oversample=spectral_oversample,
        )
    )
