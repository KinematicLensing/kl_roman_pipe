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
    from kl_pipe.observation import ImageObs, VelocityObs, GrismObs, FiberObs
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

    Precision of the chi-squared reduction
    --------------------------------------
    Under the default float64 mode the sums below accumulate in float64.
    Under ``KLPIPE_FP32`` they run in float32 -- JAX canonicalizes any
    float64 request back to float32 when x64 is off, so a mixed-precision
    sum is not available. This is safe in practice: XLA reduces in a
    blocked/pairwise order, keeping the measured relative error near 1e-8
    for our array sizes, orders of magnitude below what sampling can
    resolve. If float32 sampling ever shows likelihood-resolution
    artifacts, this reduction is the first place to look.
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
    spectral_method: str = 'erf',
    psf_mode: str = 'post_dispersion',
) -> float:
    """SourceModel grism log-likelihood for one dispersed observation."""
    pars = _build_pars_dict(theta_sampled, sampled_names, fixed_pars)
    model_img = source.render_grism(
        pars,
        obs,
        spectral_oversample=spectral_oversample,
        spectral_method=spectral_method,
        psf_mode=psf_mode,
    )
    return _gaussian_log_likelihood(obs.data, model_img, obs.variance, obs.mask)

def _log_likelihood_grism_group_source(
    theta_sampled: jnp.ndarray,
    source: 'SourceModel',
    obs_group: dict,
    sampled_names: tuple,
    fixed_pars: dict,
    spectral_oversample: int = 15,
    spectral_method: str = 'erf',
    psf_mode: str = 'post_dispersion',
    operators: dict = None,
) -> float:
    """SourceModel grism log-likelihood for a shared-cube group of
    dispersed observations (one anchor-frame cube, per-roll precomputed
    dispersion operators)."""
    pars = _build_pars_dict(theta_sampled, sampled_names, fixed_pars)
    model_imgs = source.render_grism_group(
        pars,
        obs_group,
        spectral_oversample=spectral_oversample,
        spectral_method=spectral_method,
        psf_mode=psf_mode,
        operators=operators,
    )
    log_l = 0.0
    for key, obs in obs_group.items():
        log_l = log_l + _gaussian_log_likelihood(
            obs.data, model_imgs[key], obs.variance, obs.mask
        )
    return log_l

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

def _log_likelihood_fiber_group_source(
    theta_sampled: jnp.ndarray,
    source: 'SourceModel',
    obs_group: dict, #list, #dict
    sampled_names: tuple,
    fixed_pars: dict,
    spectral_oversample: int = 15,
    #spectral_method: str = 'erf',
    #psf_mode: str = 'post_dispersion',
    #operators: dict = None,
) -> float:
    pars = _build_pars_dict(theta_sampled, sampled_names, fixed_pars)
    model_spectra = source.render_fiber_group(
        pars,
        obs_group,
        spectral_oversample=spectral_oversample,
        #spectral_method=spectral_method,
        #psf_mode=psf_mode,
        #operators=operators,
    )

    log_l = 0.0
    for key, obs in obs_group.items():
        log_l = log_l + _gaussian_log_likelihood(
            obs.data, model_spectra[key], obs.variance, obs.mask
        )
    return log_l

    #log_l = 0.0
    #for i in range(0, len(obs_group)):
        #log_l = log_l + _gaussian_log_likelihood(
            #obs_group[i], model_imgs[i], obs_group[i].variance, obs_group[i].mask
        #)

    #is this slow when jitted?
    #for obs, img in zip(obs_group, model_imgs):
        #log_l = log_l + _gaussian_log_likelihood(
            #obs.data, img, obs.variance, obs.mask
        #)
    #return log_l

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
    spectral_method: str = 'erf',
    psf_mode: str = 'post_dispersion',
    grism_groups: list = None,
    grism_group_operators: list = None,
    fiber_groups: list = None
) -> float:
    """Dispatch sum over all populated channels for SourceModel inference.

    ``image_obs`` and ``grism_obs`` are dicts (or None / empty) keyed by
    component name (e.g. ``'F087'``, ``'roll0'``). ``velocity_obs`` is a
    single VelocityObs or None. The ``if ... is not None`` and dict-iter
    branches resolve at JIT trace time because the obs structure is fixed
    (closed over via partial).

    ``grism_groups`` (cube_mode='shared'; built once by
    ``create_jitted_likelihood_from_obs``) supersedes ``grism_obs`` when
    set: each multi-obs group renders from one shared celestial cube;
    singleton groups take the identical per-obs path.
    """
    log_l = 0.0
    if image_obs:
        for band_key, obs in image_obs.items():
            log_l = log_l + _log_likelihood_broadband_source(
                theta_sampled, source, obs, band_key, sampled_names, fixed_pars
            )
    if grism_groups is not None:
        for i, group in enumerate(grism_groups):
            if len(group) == 1:
                (obs,) = group.values()
                log_l = log_l + _log_likelihood_grism_source(
                    theta_sampled,
                    source,
                    obs,
                    sampled_names,
                    fixed_pars,
                    spectral_oversample=spectral_oversample,
                    spectral_method=spectral_method,
                    psf_mode=psf_mode,
                )
            else:
                log_l = log_l + _log_likelihood_grism_group_source(
                    theta_sampled,
                    source,
                    group,
                    sampled_names,
                    fixed_pars,
                    spectral_oversample=spectral_oversample,
                    spectral_method=spectral_method,
                    psf_mode=psf_mode,
                    operators=(
                        grism_group_operators[i]
                        if grism_group_operators is not None
                        else None
                    ),
                )
    elif grism_obs:
        for _grism_key, obs in grism_obs.items():
            log_l = log_l + _log_likelihood_grism_source(
                theta_sampled,
                source,
                obs,
                sampled_names,
                fixed_pars,
                spectral_oversample=spectral_oversample,
                spectral_method=spectral_method,
                psf_mode=psf_mode,
            )
    if fiber_groups is not None:
        for i, group in enumerate(fiber_groups):
            if len(group) == 1:
                (obs,) = group.values()
                log_l = log_l + _log_likelihood_fiber_source(
                    theta_sampled,
                    source,
                    obs,
                    sampled_names,
                    fixed_pars,
                    spectral_oversample=spectral_oversample)
            else:
                log_l = log_l + _log_likelihood_fiber_group_source(
                    theta_sampled,
                    source,
                    group,
                    sampled_names,
                    fixed_pars,
                    spectral_oversample=spectral_oversample)
    elif fiber_obs:
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
    #fiber_groups: list = None, 
    velocity_obs=None,
    spectral_oversample: int = 15,
    spectral_method: str = 'erf',
    psf_mode: str = 'post_dispersion',
    cube_mode: str = 'shared',
) -> Callable[[jnp.ndarray], float]:
    """JIT-compiled total log-likelihood for SourceModel-based inference.

    The returned function takes ``theta_sampled`` (priors-ordered) and
    returns the Gaussian log-likelihood summed over all provided channels.
    All obs + the source + the static name/fixed tuples are frozen via
    ``functools.partial``.

    ``cube_mode='shared'`` (default) groups cube-compatible grism obs at
    construction time so each group renders from one shared celestial
    cube; guards (psf_mode, flipped WCS, mixed PSF presence, near-miss
    wavelength grids) raise HERE, eagerly, not at first trace.
    ``'per_roll'`` keeps the classic independent per-obs rendering.
    """
    if cube_mode not in ('shared', 'per_roll'):
        raise ValueError(f"cube_mode must be 'shared' or 'per_roll', got {cube_mode!r}")
    grism_groups = None
    grism_group_operators = None
    if cube_mode == 'shared' and grism_obs:
        from kl_pipe.observation import (
            build_group_dispersion_operators,
            group_grism_obs_by_cube_compat,
            validate_shared_cube_group,
        )

        grism_groups = group_grism_obs_by_cube_compat(grism_obs)
        # build the static dispersion operators ONCE per multi-obs group
        # (galaxy- and theta-independent; singleton groups use the per-obs
        # path and need none)
        grism_group_operators = []
        for group in grism_groups:
            validate_shared_cube_group(group, psf_mode)
            grism_group_operators.append(
                build_group_dispersion_operators(group) if len(group) > 1 else None
            )

    #whatever I'll just make the fiber group here
    from kl_pipe.observation import group_fiber_obs
    fiber_groups = group_fiber_obs(fiber_obs)

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
            spectral_method=spectral_method,
            psf_mode=psf_mode,
            grism_groups=grism_groups,
            grism_group_operators=grism_group_operators,
            fiber_groups = fiber_groups
        )
    )
