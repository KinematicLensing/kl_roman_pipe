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
Basic usage with the helper functions:

>>> from kl_pipe.likelihood import create_jitted_likelihood_velocity
>>> from kl_pipe.observation import build_velocity_obs
>>>
>>> obs_vel = build_velocity_obs(image_pars, data=data_vel, variance=variance)
>>> log_like = create_jitted_likelihood_velocity(model, obs_vel)
>>> log_prob = log_like(theta)
>>>
>>> # compute gradients
>>> grad_fn = jax.grad(log_like)
>>> gradient = grad_fn(theta)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from kl_pipe.model import VelocityModel, IntensityModel, KLModel
    from kl_pipe.observation import ImageObs, VelocityObs, GrismObs
    from kl_pipe.source import SourceModel


def _log_likelihood_velocity(
    theta: jnp.ndarray,
    obs: 'VelocityObs',
    vel_model: 'VelocityModel',
    flux_theta_override: jnp.ndarray = None,
) -> float:
    """
    Log-likelihood for velocity observations.

    Computes the Gaussian log-likelihood including normalization constants:
        log L = -0.5 * [N*log(2pi) + log(det(Sigma)) + chi2]

    Where N is the number of data points, Sigma is the covariance (diagonal),
    and chi2 is the weighted sum of squared residuals.

    Parameters
    ----------
    theta : jnp.ndarray
        Velocity model parameters.
    obs : VelocityObs
        Velocity observation with data, variance, mask, and PSF.
    vel_model : VelocityModel
        Velocity model instance.
    flux_theta_override : jnp.ndarray, optional
        Intensity params for joint mode flux weighting.

    Returns
    -------
    float
        Log-likelihood value.

    Notes
    -----
    This function is designed to be JIT-compiled. The variance can be either
    a scalar (constant noise) or an array (spatially varying noise), and the
    same formula handles both cases without conditionals.

    The ``if obs.mask is not None`` check is a Python-level branch resolved at
    JIT compile time (obs is frozen via ``partial()``), not a traced
    conditional.
    """
    model_vel = vel_model.render_image(
        theta, obs=obs, flux_theta_override=flux_theta_override
    )

    residuals = obs.data - model_vel
    variance = jnp.broadcast_to(jnp.asarray(obs.variance), obs.data.shape)

    if obs.mask is not None:
        chi2 = jnp.sum(jnp.where(obs.mask, residuals**2 / variance, 0.0))
        n_data = jnp.sum(obs.mask).astype(float)
        log_det_term = jnp.sum(jnp.where(obs.mask, jnp.log(variance), 0.0))
    else:
        chi2 = jnp.sum(residuals**2 / variance)
        n_data = obs.data.size
        log_det_term = jnp.sum(jnp.log(variance))

    normalization = -0.5 * n_data * jnp.log(2 * jnp.pi) - 0.5 * log_det_term
    return normalization - 0.5 * chi2


def _log_likelihood_intensity(
    theta: jnp.ndarray,
    obs: 'ImageObs',
    int_model: 'IntensityModel',
    render_config=None,
) -> float:
    """
    Log-likelihood for intensity observations.

    Computes the Gaussian log-likelihood including normalization constants:
        log L = -0.5 * [N*log(2pi) + log(det(Sigma)) + chi2]

    Parameters
    ----------
    theta : jnp.ndarray
        Intensity model parameters.
    obs : ImageObs
        Image observation with data, variance, mask, and PSF.
    int_model : IntensityModel
        Intensity model instance.
    render_config : RenderConfig, optional
        K-space grid parameters. When provided (inference path), frozen
        into the JIT closure for deterministic grid sizing.

    Returns
    -------
    float
        Log-likelihood value.

    Notes
    -----
    This function is designed to be JIT-compiled. The variance can be either
    a scalar (constant noise) or an array (spatially varying noise), and the
    same formula handles both cases without conditionals.

    The ``if obs.mask is not None`` check is a Python-level branch resolved at
    JIT compile time (obs is frozen via ``partial()``), not a traced
    conditional.
    """
    model_int = int_model.render_image(theta, obs=obs, render_config=render_config)

    residuals = obs.data - model_int
    variance = jnp.broadcast_to(jnp.asarray(obs.variance), obs.data.shape)

    if obs.mask is not None:
        chi2 = jnp.sum(jnp.where(obs.mask, residuals**2 / variance, 0.0))
        n_data = jnp.sum(obs.mask).astype(float)
        log_det_term = jnp.sum(jnp.where(obs.mask, jnp.log(variance), 0.0))
    else:
        chi2 = jnp.sum(residuals**2 / variance)
        n_data = obs.data.size
        log_det_term = jnp.sum(jnp.log(variance))

    normalization = -0.5 * n_data * jnp.log(2 * jnp.pi) - 0.5 * log_det_term
    return normalization - 0.5 * chi2


def _log_likelihood_joint(
    theta: jnp.ndarray,
    obs_vel: 'VelocityObs',
    obs_int: 'ImageObs',
    kl_model: 'KLModel',
    render_config_int=None,
    render_config_vel=None,
) -> float:
    """
    Log-likelihood for combined velocity + intensity observations.

    Evaluates velocity and intensity models on their respective grids
    and returns the combined log-likelihood. The two datasets are assumed
    to be independent, so the joint likelihood is the sum of individual
    log-likelihoods.

    Parameters
    ----------
    theta : jnp.ndarray
        Combined model parameters (kl_model.PARAMETER_NAMES order).
    obs_vel : VelocityObs
        Velocity observation.
    obs_int : ImageObs
        Intensity observation.
    kl_model : KLModel
        Combined kinematic-lensing model.

    Returns
    -------
    float
        Combined log-likelihood value.

    Notes
    -----
    This function calls the individual velocity and intensity likelihood
    functions internally, ensuring consistency in the likelihood calculation
    across different use cases. It is designed to be JIT-compiled.

    The velocity and intensity maps can have different shapes and pixel scales,
    as they are evaluated on their own coordinate grids.
    """
    theta_vel = kl_model.get_velocity_pars(theta)
    theta_int = kl_model.get_intensity_pars(theta)

    log_prob_vel = _log_likelihood_velocity(
        theta_vel,
        obs_vel,
        kl_model.velocity_model,
        flux_theta_override=theta_int,
    )
    log_prob_int = _log_likelihood_intensity(
        theta_int,
        obs_int,
        kl_model.intensity_model,
        render_config=render_config_int,
    )

    return log_prob_vel + log_prob_int


def _log_likelihood_grism(
    theta: jnp.ndarray,
    obs: 'GrismObs',
    kl_model: 'KLModel',
) -> float:
    """
    Log-likelihood for grism observations.

    Computes the Gaussian log-likelihood including normalization constants:
        log L = -0.5 * [N*log(2pi) + log(det(Sigma)) + chi2]

    Where N is the number of data points, Sigma is the covariance (diagonal),
    and chi2 is the weighted sum of squared residuals.

    Parameters
    ----------
    theta : jnp.ndarray
        Composite KLModel parameter array.
    obs : GrismObs
        Grism observation with data, variance, mask, dispersion params, and PSF.
    kl_model : KLModel
        Combined kinematic-lensing model. Must have spectral_model configured.

    Returns
    -------
    float
        Log-likelihood value.

    Notes
    -----
    This function is designed to be JIT-compiled. The variance can be either
    a scalar (constant noise) or an array (spatially varying noise), and the
    same formula handles both cases without conditionals.

    The ``if obs.mask is not None`` check is a Python-level branch resolved at
    JIT compile time (obs is frozen via ``partial()``), not a traced
    conditional.

    Known limitations
    -----------------
    - Spectral line broadening in ``kl_pipe/spectral.py`` adds an instrumental
      sigma (derived from the quoted resolving power R) in quadrature with
      the kinematic velocity dispersion BEFORE the PSF-per-slice + dispersion
      stages run. For slitless grism geometry the spectral resolution element
      IS the PSF projected along the dispersion axis, so this likely
      double-counts. As a consequence, ``vel_dispersion`` is poorly identified
      under this likelihood. Tracked in ``docs/plans/phase2_lsf_refactor.md``.
    - ``kl_model.intensity_model`` provides a single ``int_x0``/``int_y0``
      centroid used by both broadband image rendering and the emission cube
      spatial distribution. If the photometric and grism observations have
      independent astrometric solutions, the shared centroid is the wrong
      degree of freedom. Tracked in ``docs/plans/phase3_sourcemodel_refactor.md``.
    """
    model_grism = kl_model.render_grism(theta, obs)

    residuals = obs.data - model_grism
    variance = jnp.broadcast_to(jnp.asarray(obs.variance), obs.data.shape)

    if obs.mask is not None:
        chi2 = jnp.sum(jnp.where(obs.mask, residuals**2 / variance, 0.0))
        n_data = jnp.sum(obs.mask).astype(float)
        log_det_term = jnp.sum(jnp.where(obs.mask, jnp.log(variance), 0.0))
    else:
        chi2 = jnp.sum(residuals**2 / variance)
        n_data = obs.data.size
        log_det_term = jnp.sum(jnp.log(variance))

    normalization = -0.5 * n_data * jnp.log(2 * jnp.pi) - 0.5 * log_det_term
    return normalization - 0.5 * chi2


def _log_likelihood_joint_photometry_grism(
    theta: jnp.ndarray,
    obs_int: 'ImageObs',
    obs_grism: 'GrismObs',
    kl_model: 'KLModel',
    render_config_int=None,
) -> float:
    """
    Log-likelihood for combined photometric image + grism observations.

    Evaluates intensity and grism models on their respective grids and returns
    the combined log-likelihood. The two datasets are assumed to be
    independent, so the joint likelihood is the sum of individual
    log-likelihoods.

    Parameters
    ----------
    theta : jnp.ndarray
        Combined KLModel parameter array (kl_model.PARAMETER_NAMES order).
    obs_int : ImageObs
        Broadband photometric image observation.
    obs_grism : GrismObs
        Grism observation.
    kl_model : KLModel
        Combined kinematic-lensing model with spectral_model configured.
    render_config_int : RenderConfig, optional
        K-space grid parameters for intensity rendering.

    Returns
    -------
    float
        Combined log-likelihood value.

    Notes
    -----
    Known limitation: the photometric image and the emission cube inside
    ``render_grism`` both use ``kl_model.intensity_model``'s single centroid
    parameter pair (``int_x0``/``int_y0``). If the photometric and grism
    observations have independent astrometric solutions this shared centroid
    is the wrong degree of freedom — the two channels need independent
    centroid offsets. Tracked in ``docs/plans/phase3_sourcemodel_refactor.md``.
    """
    theta_int = kl_model.get_intensity_pars(theta)

    log_prob_int = _log_likelihood_intensity(
        theta_int,
        obs_int,
        kl_model.intensity_model,
        render_config=render_config_int,
    )
    log_prob_grism = _log_likelihood_grism(theta, obs_grism, kl_model)

    return log_prob_int + log_prob_grism


# ==============================================================================
# Helper functions for creating JIT-compiled likelihoods
# ==============================================================================


def create_jitted_likelihood_velocity(
    vel_model: 'VelocityModel',
    obs_vel: 'VelocityObs',
) -> Callable[[jnp.ndarray], float]:
    """
    Create a JIT-compiled velocity-only likelihood function.

    Creates a JIT-compiled likelihood that only requires the parameter array
    theta as input. The observation (grids, variance, data, PSF) is "frozen"
    using functools.partial.

    The resulting function is optimized for repeated evaluation (e.g., in MCMC
    or optimization), as it compiles once and reuses the compiled code for all
    subsequent calls.

    Parameters
    ----------
    vel_model : VelocityModel
        Velocity model instance.
    obs_vel : VelocityObs
        Velocity observation with data, variance, PSF, etc.

    Returns
    -------
    Callable[[jnp.ndarray], float]
        JIT-compiled function that takes theta and returns log-likelihood.

    Examples
    --------
    >>> from kl_pipe.velocity import CenteredVelocityModel
    >>> from kl_pipe.observation import build_velocity_obs
    >>> from kl_pipe.parameters import ImagePars
    >>>
    >>> model = CenteredVelocityModel()
    >>> image_pars = ImagePars(shape=(64, 64), pixel_scale=0.3)
    >>> obs_vel = build_velocity_obs(image_pars, data=data, variance=10.0)
    >>> log_like = create_jitted_likelihood_velocity(model, obs_vel)
    >>>
    >>> theta = jnp.array([10.0, 200.0, 5.0, 0.6, 0.785, 0.0, 0.0])
    >>> log_prob = log_like(theta)  # Fast evaluation
    >>> grad_fn = jax.grad(log_like)
    >>> gradient = grad_fn(theta)

    Notes
    -----
    The first call to the returned function will trigger JIT compilation, which
    may take a few seconds. Subsequent calls will be very fast (microseconds).

    The function is pure and has no side effects, making it safe for use with
    JAX transformations (grad, vmap, etc.).
    """
    return jax.jit(
        partial(
            _log_likelihood_velocity,
            obs=obs_vel,
            vel_model=vel_model,
        )
    )


def create_jitted_likelihood_intensity(
    int_model: 'IntensityModel',
    obs_int: 'ImageObs',
    render_config=None,
) -> Callable[[jnp.ndarray], float]:
    """
    Create a JIT-compiled intensity-only likelihood function.

    Creates a JIT-compiled likelihood that only requires the parameter array
    theta as input. The observation (grids, variance, data, PSF) is "frozen"
    using functools.partial.

    Parameters
    ----------
    int_model : IntensityModel
        Intensity model instance.
    obs_int : ImageObs
        Image observation with data, variance, PSF, etc.
    render_config : RenderConfig, optional
        K-space grid parameters. When provided, frozen into the JIT closure
        for deterministic grid sizing across MCMC samples.

    Returns
    -------
    Callable[[jnp.ndarray], float]
        JIT-compiled function that takes theta and returns log-likelihood.

    Notes
    -----
    See create_jitted_likelihood_velocity for additional usage notes and
    performance considerations.
    """
    return jax.jit(
        partial(
            _log_likelihood_intensity,
            obs=obs_int,
            int_model=int_model,
            render_config=render_config,
        )
    )


def create_jitted_likelihood_joint(
    kl_model: 'KLModel',
    obs_vel: 'VelocityObs',
    obs_int: 'ImageObs',
    render_config_int=None,
    render_config_vel=None,
) -> Callable[[jnp.ndarray], float]:
    """
    Create a JIT-compiled joint velocity + intensity likelihood function.

    Creates a JIT-compiled likelihood for combined kinematic-lensing
    observations. The velocity and intensity data can have different shapes,
    pixel scales, and noise properties.

    Parameters
    ----------
    kl_model : KLModel
        Combined kinematic-lensing model.
    obs_vel : VelocityObs
        Velocity observation.
    obs_int : ImageObs
        Intensity observation.

    Returns
    -------
    Callable[[jnp.ndarray], float]
        JIT-compiled function that takes composite theta and returns
        joint log-likelihood.

    Examples
    --------
    >>> from kl_pipe.model import KLModel
    >>> from kl_pipe.velocity import OffsetVelocityModel
    >>> from kl_pipe.intensity import InclinedExponentialModel
    >>> from kl_pipe.observation import build_joint_obs
    >>> from kl_pipe.parameters import ImagePars
    >>>
    >>> vel_model = OffsetVelocityModel()
    >>> int_model = InclinedExponentialModel()
    >>> kl_model = KLModel(vel_model, int_model,
    ...                    shared_pars={'g1', 'g2', 'theta_int', 'cosi'})
    >>>
    >>> vel_pars = ImagePars(shape=(32, 32), pixel_scale=0.3)
    >>> int_pars = ImagePars(shape=(64, 64), pixel_scale=0.1)
    >>>
    >>> obs_vel, obs_int = build_joint_obs(
    ...     vel_pars, int_pars, int_model,
    ...     data_vel=data_vel, variance_vel=10.0,
    ...     data_int=data_int, variance_int=0.01,
    ... )
    >>> log_like = create_jitted_likelihood_joint(kl_model, obs_vel, obs_int)
    >>> log_prob = log_like(theta)

    Notes
    -----
    The composite theta array should follow the order defined in
    kl_model.PARAMETER_NAMES. Use kl_model.get_velocity_pars(theta) and
    kl_model.get_intensity_pars(theta) to extract component parameters
    if needed for inspection.

    This function is particularly useful for joint kinematic-lensing analysis
    where velocity and intensity observations have different resolutions or
    fields of view.
    """
    return jax.jit(
        partial(
            _log_likelihood_joint,
            obs_vel=obs_vel,
            obs_int=obs_int,
            kl_model=kl_model,
            render_config_int=render_config_int,
            render_config_vel=render_config_vel,
        )
    )


def create_jitted_likelihood_grism(
    kl_model: 'KLModel',
    obs_grism: 'GrismObs',
) -> Callable[[jnp.ndarray], float]:
    """
    Create a JIT-compiled grism-only likelihood function.

    Creates a JIT-compiled likelihood that only requires the parameter array
    theta as input. The observation (grism_pars, PSF, data, variance) and
    kl_model are "frozen" using functools.partial.

    Parameters
    ----------
    kl_model : KLModel
        Combined kinematic-lensing model with spectral_model configured.
    obs_grism : GrismObs
        Grism observation.

    Returns
    -------
    Callable[[jnp.ndarray], float]
        JIT-compiled function that takes theta and returns log-likelihood.

    Examples
    --------
    >>> from kl_pipe.model import KLModel
    >>> from kl_pipe.observation import build_grism_obs
    >>>
    >>> obs_grism = build_grism_obs(grism_pars, z=1.0, data=data, variance=var,
    ...                              psf=psf)
    >>> log_like = create_jitted_likelihood_grism(kl_model, obs_grism)
    >>> log_prob = log_like(theta)

    Notes
    -----
    See create_jitted_likelihood_velocity for additional usage notes and
    performance considerations.

    Known limitation: ``kl_pipe/spectral.py`` adds an instrumental sigma
    derived from the quoted grism resolving power in quadrature with the
    kinematic velocity dispersion before the PSF + dispersion stages, which
    likely double-counts the PSF's spectral resolution contribution for
    slitless geometry. ``vel_dispersion`` is correspondingly poorly identified.
    Tracked in ``docs/plans/phase2_lsf_refactor.md``.
    """
    return jax.jit(
        partial(
            _log_likelihood_grism,
            obs=obs_grism,
            kl_model=kl_model,
        )
    )


def create_jitted_likelihood_joint_photometry_grism(
    kl_model: 'KLModel',
    obs_int: 'ImageObs',
    obs_grism: 'GrismObs',
    render_config_int=None,
) -> Callable[[jnp.ndarray], float]:
    """
    Create a JIT-compiled joint photometry + grism likelihood function.

    Creates a JIT-compiled likelihood for combined broadband image + grism
    observations. The two datasets can have different shapes, pixel scales,
    and noise properties.

    Parameters
    ----------
    kl_model : KLModel
        Combined kinematic-lensing model with spectral_model configured.
    obs_int : ImageObs
        Broadband photometric image observation.
    obs_grism : GrismObs
        Grism observation.
    render_config_int : RenderConfig, optional
        K-space grid parameters for intensity rendering. When provided,
        frozen into the JIT closure for deterministic grid sizing.

    Returns
    -------
    Callable[[jnp.ndarray], float]
        JIT-compiled function that takes composite theta and returns
        joint log-likelihood.

    Notes
    -----
    The composite theta array should follow the order defined in
    kl_model.PARAMETER_NAMES. Use kl_model.get_intensity_pars(theta) to
    extract intensity-only params if needed for inspection.

    Known limitation: the photometric image and the emission cube inside
    ``render_grism`` both use ``kl_model.intensity_model``'s single centroid
    parameter pair (``int_x0``/``int_y0``). If the two channels have
    independent astrometric solutions the shared centroid is the wrong degree
    of freedom. Tracked in ``docs/plans/phase3_sourcemodel_refactor.md``.
    """
    return jax.jit(
        partial(
            _log_likelihood_joint_photometry_grism,
            obs_int=obs_int,
            obs_grism=obs_grism,
            kl_model=kl_model,
            render_config_int=render_config_int,
        )
    )


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

    Mirrors the formula in ``_log_likelihood_velocity`` / ``_log_likelihood_intensity``;
    factored out so the SourceModel-channel likelihoods share one shape.
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
    spectral_oversample: int = 5,
) -> float:
    """SourceModel grism log-likelihood for one dispersed observation."""
    pars = _build_pars_dict(theta_sampled, sampled_names, fixed_pars)
    model_img = source.render_grism(pars, obs, spectral_oversample=spectral_oversample)
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


def _log_likelihood_total_source(
    theta_sampled: jnp.ndarray,
    source: 'SourceModel',
    image_obs: dict,
    grism_obs: dict,
    velocity_obs,
    sampled_names: tuple,
    fixed_pars: dict,
    spectral_oversample: int = 5,
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
    velocity_obs=None,
    spectral_oversample: int = 5,
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
            velocity_obs=velocity_obs,
            sampled_names=sampled_names,
            fixed_pars=fixed_pars,
            spectral_oversample=spectral_oversample,
        )
    )
