"""
Likelihood functions for kinematic-lensing models.

Provides JAX-compatible log-likelihood functions for:
- Velocity-only observations
- Intensity-only observations
- Combined velocity + intensity observations

All functions are designed to be JIT-compilable and support automatic differentiation.
The likelihood functions include proper normalization constants for model comparison.

Examples
--------
Basic usage with JIT compilation:

>>> from functools import partial
>>> import jax
>>> from kl_pipe.likelihood import _log_likelihood_velocity_only
>>> from kl_pipe.velocity import CenteredVelocityModel
>>> from kl_pipe.utils import build_map_grid_from_image_pars
>>>
>>> # setup model and grids
>>> model = CenteredVelocityModel()
>>> X, Y = build_map_grid_from_image_pars(image_pars)
>>> variance = 10.0  # km/s variance
>>>
>>> # create JIT-compiled likelihood
>>> log_like = jax.jit(
...     partial(_log_likelihood_velocity_only,
...             X_vel=X, Y_vel=Y, variance_vel=variance, vel_model=model)
... )
>>>
>>> # evaluate
>>> theta = jnp.array([10.0, 200.0, 5.0, 0.6, 0.785, 0.0, 0.0])
>>> log_prob = log_like(theta, data_vel)

Using the helper functions:

>>> from kl_pipe.likelihood import create_jitted_likelihood_velocity
>>> log_like = create_jitted_likelihood_velocity(model, image_pars, variance, data_vel)
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
    from kl_pipe.parameters import ImagePars

from kl_pipe.utils import build_map_grid_from_image_pars


def _log_likelihood_velocity_only(
    theta: jnp.ndarray,
    data_vel: jnp.ndarray,
    X_vel: jnp.ndarray,
    Y_vel: jnp.ndarray,
    variance_vel: jnp.ndarray | float,
    vel_model: VelocityModel,
    flux_theta_override: jnp.ndarray = None,
) -> float:
    """
    Log-likelihood for velocity observations only.

    Computes the Gaussian log-likelihood including normalization constants:
        log L = -0.5 * [N*log(2π) + log(det(Σ)) + χ²]

    Where N is the number of data points, Σ is the covariance (diagonal),
    and χ² is the weighted sum of squared residuals.

    Parameters
    ----------
    theta : jnp.ndarray
        Velocity model parameters.
    data_vel : jnp.ndarray
        Observed velocity map (2D array).
    X_vel, Y_vel : jnp.ndarray
        Coordinate grids for velocity map evaluation.
    variance_vel : jnp.ndarray or float
        Variance map (2D array) or scalar variance for velocity data.
        If array, must have same shape as data_vel.
    vel_model : VelocityModel
        Velocity model instance.
    flux_theta_override : jnp.ndarray, optional
        Intensity params for joint mode flux weighting (passed to render_image).

    Returns
    -------
    float
        Log-likelihood value.

    Notes
    -----
    This function is designed to be JIT-compiled. The variance can be either
    a scalar (constant noise) or an array (spatially varying noise), and the
    same formula handles both cases without conditionals.
    """

    # evaluate model via render_image (applies PSF if configured)
    model_vel = vel_model.render_image(
        theta, X=X_vel, Y=Y_vel, flux_theta_override=flux_theta_override
    )

    # compute chi-squared
    residuals = data_vel - model_vel
    chi2 = jnp.sum(residuals**2 / variance_vel)

    # compute normalization (works for both scalar and array variance)
    n_data = data_vel.size
    log_det_term = jnp.sum(jnp.log(variance_vel))
    normalization = -0.5 * n_data * jnp.log(2 * jnp.pi) - 0.5 * log_det_term

    return normalization - 0.5 * chi2


def _log_likelihood_intensity_only(
    theta: jnp.ndarray,
    data_int: jnp.ndarray,
    image_pars_int: 'ImagePars',
    variance_int: jnp.ndarray | float,
    int_model: IntensityModel,
) -> float:
    """
    Log-likelihood for intensity observations only.

    Computes the Gaussian log-likelihood including normalization constants:
        log L = -0.5 * [N*log(2π) + log(det(Σ)) + χ²]

    Parameters
    ----------
    theta : jnp.ndarray
        Intensity model parameters.
    data_int : jnp.ndarray
        Observed intensity map (2D array).
    image_pars_int : ImagePars
        Image parameters for intensity map (shape, pixel_scale).
    variance_int : jnp.ndarray or float
        Variance map (2D array) or scalar variance for intensity data.
        If array, must have same shape as data_int.
    int_model : IntensityModel
        Intensity model instance.

    Returns
    -------
    float
        Log-likelihood value.

    Notes
    -----
    This function is designed to be JIT-compiled. The variance can be either
    a scalar (constant noise) or an array (spatially varying noise), and the
    same formula handles both cases without conditionals.
    """

    # evaluate model via render_image (applies PSF if configured)
    model_int = int_model.render_image(theta, image_pars=image_pars_int)

    # compute chi-squared
    residuals = data_int - model_int
    chi2 = jnp.sum(residuals**2 / variance_int)

    # compute normalization (works for both scalar and array variance)
    n_data = data_int.size
    log_det_term = jnp.sum(jnp.log(variance_int))
    normalization = -0.5 * n_data * jnp.log(2 * jnp.pi) - 0.5 * log_det_term

    return normalization - 0.5 * chi2


def _log_likelihood_separate_images(
    theta: jnp.ndarray,
    data_vel: jnp.ndarray,
    data_int: jnp.ndarray,
    X_vel: jnp.ndarray,
    Y_vel: jnp.ndarray,
    image_pars_int: 'ImagePars',
    variance_vel: jnp.ndarray | float,
    variance_int: jnp.ndarray | float,
    kl_model: KLModel,
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
        Combined model parameters (following kl_model.PARAMETER_NAMES order).
    data_vel : jnp.ndarray
        Observed velocity map (2D array).
    data_int : jnp.ndarray
        Observed intensity map (2D array).
    X_vel, Y_vel : jnp.ndarray
        Coordinate grids for velocity map evaluation.
    image_pars_int : ImagePars
        Image parameters for intensity map (shape, pixel_scale).
    variance_vel : jnp.ndarray or float
        Variance for velocity data.
    variance_int : jnp.ndarray or float
        Variance for intensity data.
    kl_model : KLModel
        Combined kinematic-lensing model instance.

    Returns
    -------
    float
        Combined log-likelihood value (sum of velocity and intensity components).

    Notes
    -----
    This function calls the individual velocity and intensity likelihood
    functions internally, ensuring consistency in the likelihood calculation
    across different use cases. It is designed to be JIT-compiled.

    The velocity and intensity maps can have different shapes and pixel scales,
    as they are evaluated on their own coordinate grids.
    """

    # extract component parameters from composite theta
    theta_vel = kl_model.get_velocity_pars(theta)
    theta_int = kl_model.get_intensity_pars(theta)

    # compute log-likelihood for each component
    # pass theta_int to velocity for joint PSF flux weighting
    log_prob_vel = _log_likelihood_velocity_only(
        theta_vel,
        data_vel,
        X_vel,
        Y_vel,
        variance_vel,
        kl_model.velocity_model,
        flux_theta_override=theta_int,
    )
    log_prob_int = _log_likelihood_intensity_only(
        theta_int, data_int, image_pars_int, variance_int, kl_model.intensity_model
    )

    # independent observations: joint likelihood is sum of log-likelihoods
    return log_prob_vel + log_prob_int


# ==============================================================================
# Helper functions for creating JIT-compiled likelihoods
# ==============================================================================


def create_jitted_likelihood_velocity(
    vel_model: VelocityModel,
    image_pars_vel: ImagePars,
    variance_vel: jnp.ndarray | float,
    data_vel: jnp.ndarray,
) -> Callable[[jnp.ndarray], float]:
    """
    Create a JIT-compiled velocity-only likelihood function.

    This helper function creates a JIT-compiled likelihood that only requires
    the parameter array theta as input. All other arguments (grids, variance,
    data, model) are "frozen" using functools.partial.

    The coordinate grids are automatically generated from the ImagePars using
    build_map_grid_from_image_pars before JIT compilation, so they are
    pre-computed and reused for all likelihood evaluations.

    The resulting function is optimized for repeated evaluation (e.g., in MCMC
    or optimization), as it compiles once and reuses the compiled code for all
    subsequent calls.

    Parameters
    ----------
    vel_model : VelocityModel
        Velocity model instance to evaluate.
    image_pars_vel : ImagePars
        Image parameters defining the coordinate grid for velocity evaluation.
        Grids are automatically generated using build_map_grid_from_image_pars.
    variance_vel : jnp.ndarray or float
        Variance map or scalar variance for velocity data.
    data_vel : jnp.ndarray
        Observed velocity data (2D array).

    Returns
    -------
    Callable[[jnp.ndarray], float]
        JIT-compiled function that takes theta and returns log-likelihood.

    Examples
    --------
    >>> from kl_pipe.velocity import CenteredVelocityModel
    >>> from kl_pipe.parameters import ImagePars
    >>> import jax.numpy as jnp
    >>>
    >>> # Setup
    >>> model = CenteredVelocityModel()
    >>> image_pars = ImagePars(shape=(64, 64), pixel_scale=0.3)
    >>> data = jnp.array(...)  # Your observed data
    >>> variance = 10.0  # km/s²
    >>>
    >>> # Create JIT-compiled likelihood
    >>> log_like = create_jitted_likelihood_velocity(model, image_pars, variance, data)
    >>>
    >>> # Use in optimization or MCMC
    >>> theta = jnp.array([10.0, 200.0, 5.0, 0.6, 0.785, 0.0, 0.0])
    >>> log_prob = log_like(theta)  # Fast evaluation
    >>>
    >>> # Compute gradients
    >>> grad_fn = jax.grad(log_like)
    >>> gradient = grad_fn(theta)

    Notes
    -----
    The first call to the returned function will trigger JIT compilation, which
    may take a few seconds. Subsequent calls will be very fast (microseconds).

    The function is pure and has no side effects, making it safe for use with
    JAX transformations (grad, vmap, etc.).
    """

    # pre-compute coordinate grids from ImagePars
    X_vel, Y_vel = build_map_grid_from_image_pars(image_pars_vel)

    return jax.jit(
        partial(
            _log_likelihood_velocity_only,
            data_vel=data_vel,
            X_vel=X_vel,
            Y_vel=Y_vel,
            variance_vel=variance_vel,
            vel_model=vel_model,
        )
    )


def create_jitted_likelihood_intensity(
    int_model: IntensityModel,
    image_pars_int: ImagePars,
    variance_int: jnp.ndarray | float,
    data_int: jnp.ndarray,
) -> Callable[[jnp.ndarray], float]:
    """
    Create a JIT-compiled intensity-only likelihood function.

    This helper function creates a JIT-compiled likelihood that only requires
    the parameter array theta as input. All other arguments (grids, variance,
    data, model) are "frozen" using functools.partial.

    The coordinate grids are automatically generated from the ImagePars using
    build_map_grid_from_image_pars before JIT compilation.

    Parameters
    ----------
    int_model : IntensityModel
        Intensity model instance to evaluate.
    image_pars_int : ImagePars
        Image parameters defining the coordinate grid for intensity evaluation.
        Grids are automatically generated using build_map_grid_from_image_pars.
    variance_int : jnp.ndarray or float
        Variance map or scalar variance for intensity data.
    data_int : jnp.ndarray
        Observed intensity data (2D array).

    Returns
    -------
    Callable[[jnp.ndarray], float]
        JIT-compiled function that takes theta and returns log-likelihood.

    Examples
    --------
    >>> from kl_pipe.intensity import InclinedExponentialModel
    >>> from kl_pipe.parameters import ImagePars
    >>>
    >>> # Setup
    >>> model = InclinedExponentialModel()
    >>> image_pars = ImagePars(shape=(64, 64), pixel_scale=0.1)
    >>> data = jnp.array(...)  # Your observed data
    >>> variance = 0.01  # Intensity variance
    >>>
    >>> # Create JIT-compiled likelihood
    >>> log_like = create_jitted_likelihood_intensity(model, image_pars, variance, data)
    >>>
    >>> # Evaluate and optimize
    >>> theta = jnp.array([1.0, 3.0, 0.6, 0.785, 0.0, 0.0, 0.0, 0.0])
    >>> log_prob = log_like(theta)

    Notes
    -----
    See create_jitted_likelihood_velocity for additional usage notes and
    performance considerations.
    """

    return jax.jit(
        partial(
            _log_likelihood_intensity_only,
            data_int=data_int,
            image_pars_int=image_pars_int,
            variance_int=variance_int,
            int_model=int_model,
        )
    )


def create_jitted_likelihood_joint(
    kl_model: KLModel,
    image_pars_vel: ImagePars,
    image_pars_int: ImagePars,
    variance_vel: jnp.ndarray | float,
    variance_int: jnp.ndarray | float,
    data_vel: jnp.ndarray,
    data_int: jnp.ndarray,
) -> Callable[[jnp.ndarray], float]:
    """
    Create a JIT-compiled joint velocity + intensity likelihood function.

    This helper function creates a JIT-compiled likelihood for combined
    kinematic-lensing observations. The velocity and intensity data can have
    different shapes, pixel scales, and noise properties.

    The coordinate grids are automatically generated from the respective
    ImagePars objects using build_map_grid_from_image_pars before JIT
    compilation.

    Parameters
    ----------
    kl_model : KLModel
        Combined kinematic-lensing model instance.
    image_pars_vel : ImagePars
        Image parameters for velocity map. Grids are automatically generated.
    image_pars_int : ImagePars
        Image parameters for intensity map. Grids are automatically generated.
    variance_vel : jnp.ndarray or float
        Variance for velocity data.
    variance_int : jnp.ndarray or float
        Variance for intensity data.
    data_vel : jnp.ndarray
        Observed velocity data (2D array).
    data_int : jnp.ndarray
        Observed intensity data (2D array).

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
    >>> from kl_pipe.parameters import ImagePars
    >>>
    >>> # Setup models
    >>> vel_model = OffsetVelocityModel()
    >>> int_model = InclinedExponentialModel()
    >>> kl_model = KLModel(vel_model, int_model,
    ...                    shared_pars={'g1', 'g2', 'theta_int', 'sini'})
    >>>
    >>> # Setup grids (can be different sizes!)
    >>> vel_pars = ImagePars(shape=(32, 32), pixel_scale=0.3)
    >>> int_pars = ImagePars(shape=(64, 64), pixel_scale=0.1)
    >>>
    >>> # Load data
    >>> data_vel = jnp.array(...)
    >>> data_int = jnp.array(...)
    >>> variance_vel = 10.0
    >>> variance_int = 0.01
    >>>
    >>> # Create JIT-compiled joint likelihood
    >>> log_like = create_jitted_likelihood_joint(
    ...     kl_model, vel_pars, int_pars,
    ...     variance_vel, variance_int, data_vel, data_int
    ... )
    >>>
    >>> # Use in inference
    >>> theta = jnp.array([...])  # Composite parameters
    >>> log_prob = log_like(theta)
    >>> grad_fn = jax.grad(log_like)
    >>> gradient = grad_fn(theta)

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

    # pre-compute coordinate grids from ImagePars (velocity still needs X, Y)
    X_vel, Y_vel = build_map_grid_from_image_pars(image_pars_vel)

    return jax.jit(
        partial(
            _log_likelihood_separate_images,
            data_vel=data_vel,
            data_int=data_int,
            X_vel=X_vel,
            Y_vel=Y_vel,
            image_pars_int=image_pars_int,
            variance_vel=variance_vel,
            variance_int=variance_int,
            kl_model=kl_model,
        )
    )


# NOTE: OLD!!!
# TODO: Remove when ready
# This was the first implementation for simple JAX testing; now replaced by above
# should be removed when ready
# def log_likelihood(
#     theta: jnp.ndarray, kl_model: KLModel, datavector: jnp.ndarray, meta_pars: dict
# ) -> float:
#     """
#     Compute log-likelihood for kinematic lensing model.

#     Parameters
#     ----------
#     theta : jnp.ndarray
#         Composite parameter array.
#     kl_model : KLModel
#         Combined velocity and intensity model.
#     datavector : jnp.ndarray
#         Observed data vector.
#     meta_pars : dict
#         Fixed metadata including coordinate grids.

#     Returns
#     -------
#     float
#         Log-likelihood value.
#     """

#     velocity_map, intensity_map = kl_model(
#         theta, plane='obs', X=meta_pars['X'], Y=meta_pars['Y']
#     )

#     model_prediction = velocity_map * intensity_map

#     residuals = datavector - model_prediction
#     chi2 = jnp.sum(residuals**2)

#     return -0.5 * chi2
