"""
InferenceTask: Complete specification of a Bayesian inference task.

This module defines the InferenceTask class which bundles together all
components needed for MCMC sampling:
- Model (velocity, intensity, or joint)
- Likelihood function
- Priors for sampled parameters
- Observation objects (ImageObs, VelocityObs)
- Optional metadata (systematics, etc.)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Union, Tuple, Any, TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp


class NoPSFWarning(UserWarning):
    """Inference task created without PSF — model will be unconvolved."""


class GridAdequacyWarning(UserWarning):
    """FFT grid may be too small for the model + prior combination."""


if TYPE_CHECKING:
    from kl_pipe.model import Model, VelocityModel, IntensityModel, KLModel
    from kl_pipe.parameters import ImagePars
    from kl_pipe.priors import PriorDict
    from kl_pipe.observation import ImageObs, VelocityObs


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
    - Meta parameters: Optional metadata (systematics, etc.)

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
    mask : dict, optional
        Dictionary of boolean masks (True=valid). Keys match data dict.
    meta_pars : dict, optional
        Additional metadata (systematics, etc.).

    Examples
    --------
    >>> from kl_pipe.velocity import OffsetVelocityModel
    >>> from kl_pipe.priors import Uniform, PriorDict
    >>> from kl_pipe.observation import build_velocity_obs
    >>> from kl_pipe.sampling import InferenceTask
    >>>
    >>> priors = PriorDict({
    ...     'vcirc': Uniform(100, 350),
    ...     'cosi': Uniform(0.1, 0.99),
    ...     'v0': 10.0,  # Fixed
    ... })
    >>>
    >>> obs = build_velocity_obs(image_pars, data=data_vel, variance=25.0)
    >>> task = InferenceTask.from_velocity_obs(
    ...     model=OffsetVelocityModel(),
    ...     priors=priors,
    ...     obs=obs,
    ... )
    >>> log_prob_fn = task.get_log_posterior_fn()
    """

    model: Union['Model', 'KLModel']
    likelihood_fn: Callable[[jnp.ndarray], float]
    priors: 'PriorDict'
    data: Dict[str, jnp.ndarray]
    variance: Dict[str, Union[jnp.ndarray, float]]
    mask: Dict[str, Optional[jnp.ndarray]] = field(default_factory=dict)
    meta_pars: Dict[str, Any] = field(default_factory=dict)

    # Cached functions (computed lazily)
    _log_posterior_fn: Optional[Callable] = field(default=None, init=False, repr=False)
    _log_posterior_grad_fn: Optional[Callable] = field(
        default=None, init=False, repr=False
    )

    # Pre-computed mapping for JIT-compatible theta building
    _sampled_to_full_indices: Optional[jnp.ndarray] = field(
        default=None, init=False, repr=False
    )
    _fixed_theta_template: Optional[jnp.ndarray] = field(
        default=None, init=False, repr=False
    )

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
        log_like = self.log_likelihood(theta_sampled)

        return log_prior + log_like

    def _log_posterior_jittable(self, theta_sampled: jnp.ndarray) -> float:
        """
        JIT-compatible log posterior function.

        Uses jnp.where to handle -inf prior values without branching.
        """
        log_prior = self.log_prior(theta_sampled)
        log_like = self.log_likelihood(theta_sampled)

        return jnp.where(jnp.isfinite(log_prior), log_prior + log_like, -jnp.inf)

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
    # Grid adequacy validation
    # =========================================================================

    @staticmethod
    def _validate_grid_adequacy(model, priors, obs, psf=None):
        """Warn if the obs FFT grid is likely inadequate for the model + priors.

        Computes worst-case maxk from priors and compares to the obs grid's
        effective Nyquist. Issues a GridAdequacyWarning if the grid is too
        small. Does not raise — the rendering will still work, just with
        potential aliasing.
        """
        try:
            from kl_pipe.render import RenderConfig

            pixel_scale = obs.image_pars.pixel_scale
            pixel_response = getattr(obs, 'pixel_response', None)

            rc = RenderConfig.for_priors(
                model,
                priors,
                pixel_scale,
                pixel_response=pixel_response,
                psf=psf,
            )

            # current effective Nyquist including oversample
            k_nyquist = np.pi / pixel_scale
            current_nyq = obs.oversample * k_nyquist
            if rc.effective_maxk > current_nyq * 1.1:  # 10% margin
                warnings.warn(
                    f"\nFFT grid may be inadequate for worst-case priors: "
                    f"effective_maxk={rc.effective_maxk:.1f} rad/arcsec > "
                    f"grid Nyquist={current_nyq:.1f} rad/arcsec "
                    f"(oversample={obs.oversample}). "
                    f"Recommended oversample={rc.oversample}.\n",
                    GridAdequacyWarning,
                    stacklevel=3,
                )
        except (NotImplementedError, KeyError, TypeError):
            pass  # model doesn't support maxk/stepk; skip validation

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def from_velocity_obs(
        cls,
        model: 'VelocityModel',
        priors: 'PriorDict',
        obs: 'VelocityObs',
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
        obs : VelocityObs
            Velocity observation (with data, variance, PSF, flux weighting).
        meta_pars : dict, optional
            Additional metadata.
        """
        if obs.data is None:
            raise ValueError("VelocityObs has no data; cannot create inference task")

        if obs.psf_data is None:
            warnings.warn(
                "\nNo PSF configured — velocity model will be unconvolved. Intentional?\n",
                NoPSFWarning,
                stacklevel=2,
            )

        from kl_pipe.likelihood import create_jitted_likelihood_velocity

        likelihood_fn = create_jitted_likelihood_velocity(model, obs)

        return cls(
            model=model,
            likelihood_fn=likelihood_fn,
            priors=priors,
            data={'velocity': obs.data},
            variance={'velocity': obs.variance},
            mask={'velocity': obs.mask},
            meta_pars=meta_pars or {},
        )

    @classmethod
    def from_intensity_obs(
        cls,
        model: 'IntensityModel',
        priors: 'PriorDict',
        obs: 'ImageObs',
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
        obs : ImageObs
            Image observation (with data, variance, PSF).
        meta_pars : dict, optional
            Additional metadata.
        """
        if obs.data is None:
            raise ValueError("ImageObs has no data; cannot create inference task")

        if obs.psf_data is None:
            warnings.warn(
                "\nNo PSF configured — intensity model will be unconvolved. Intentional?\n",
                NoPSFWarning,
                stacklevel=2,
            )

        # compute render_config from priors for optimal grid sizing
        from kl_pipe.render import RenderConfig

        try:
            rc_int = RenderConfig.for_priors(
                model,
                priors,
                obs.image_pars.pixel_scale,
                pixel_response=obs.pixel_response,
            )
        except (KeyError, NotImplementedError):
            rc_int = RenderConfig()  # fallback to defaults

        from kl_pipe.likelihood import create_jitted_likelihood_intensity

        likelihood_fn = create_jitted_likelihood_intensity(
            model, obs, render_config=rc_int
        )

        task = cls(
            model=model,
            likelihood_fn=likelihood_fn,
            priors=priors,
            data={'intensity': obs.data},
            variance={'intensity': obs.variance},
            mask={'intensity': obs.mask},
            meta_pars=meta_pars or {},
        )
        task._render_configs = {'intensity': rc_int}
        return task

    @classmethod
    def from_joint_obs(
        cls,
        model: 'KLModel',
        priors: 'PriorDict',
        obs_vel: 'VelocityObs',
        obs_int: 'ImageObs',
        meta_pars: Optional[Dict] = None,
    ) -> 'InferenceTask':
        """
        Create inference task for joint velocity + intensity inference.

        Parameters
        ----------
        model : KLModel
            Combined kinematic-lensing model.
        priors : PriorDict
            Prior specifications.
        obs_vel : VelocityObs
            Velocity observation.
        obs_int : ImageObs
            Intensity observation.
        meta_pars : dict, optional
            Additional metadata.
        """
        if obs_vel.data is None:
            raise ValueError("VelocityObs has no data; cannot create inference task")
        if obs_int.data is None:
            raise ValueError("ImageObs has no data; cannot create inference task")

        missing = []
        if obs_vel.psf_data is None:
            missing.append('velocity')
        if obs_int.psf_data is None:
            missing.append('intensity')
        if missing:
            channels = ' and '.join(missing)
            warnings.warn(
                f"\nNo PSF configured for {channels} channel(s) — model will be unconvolved. Intentional?\n",
                NoPSFWarning,
                stacklevel=2,
            )

        # compute render_configs from priors
        from kl_pipe.render import RenderConfig

        int_model = model.intensity_model if hasattr(model, 'intensity_model') else None
        rc_int = None
        if int_model is not None:
            try:
                rc_int = RenderConfig.for_priors(
                    int_model,
                    priors,
                    obs_int.image_pars.pixel_scale,
                    pixel_response=obs_int.pixel_response,
                )
            except (KeyError, NotImplementedError):
                rc_int = RenderConfig()

        rc_vel = RenderConfig(oversample=obs_vel.oversample)

        from kl_pipe.likelihood import create_jitted_likelihood_joint

        likelihood_fn = create_jitted_likelihood_joint(
            model,
            obs_vel,
            obs_int,
            render_config_int=rc_int,
            render_config_vel=rc_vel,
        )

        task = cls(
            model=model,
            likelihood_fn=likelihood_fn,
            priors=priors,
            data={'velocity': obs_vel.data, 'intensity': obs_int.data},
            variance={'velocity': obs_vel.variance, 'intensity': obs_int.variance},
            mask={'velocity': obs_vel.mask, 'intensity': obs_int.mask},
            meta_pars=meta_pars or {},
        )
        task._render_configs = {'velocity': rc_vel, 'intensity': rc_int}
        return task

    # =========================================================================
    # Legacy Factory Methods (delegate to new ones)
    # =========================================================================

    @classmethod
    def from_velocity_model(
        cls,
        model: 'VelocityModel',
        priors: 'PriorDict',
        data_vel: jnp.ndarray,
        variance_vel,
        image_pars: 'ImagePars',
        meta_pars: Optional[Dict] = None,
        psf=None,
        flux_model=None,
        flux_theta=None,
        flux_image=None,
        flux_image_pars=None,
        psf_gsparams=None,
        mask_vel=None,
    ) -> 'InferenceTask':
        """
        Create inference task for velocity-only inference (legacy API).

        Delegates to from_velocity_obs() after constructing a VelocityObs.

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
        psf : galsim.GSObject, optional
            PSF for velocity channel. Requires flux weighting source.
        flux_model : IntensityModel, optional
            Intensity model for PSF flux weighting.
        flux_theta : jnp.ndarray, optional
            Fixed intensity params (used with flux_model).
        flux_image : ndarray, optional
            Pre-rendered intensity map for PSF flux weighting.
        flux_image_pars : ImagePars, optional
            Image parameters of flux_image (for resampling if needed).
        psf_gsparams : galsim.GSParams, optional
            GalSim rendering parameters for PSF kernel accuracy.
        mask_vel : jnp.ndarray, optional
            Boolean mask (True=valid, False=masked). Same shape as data_vel.

        Returns
        -------
        InferenceTask
            Configured task ready for sampling.
        """
        from kl_pipe.observation import build_velocity_obs

        if psf is not None:
            obs = build_velocity_obs(
                image_pars,
                psf=psf,
                gsparams=psf_gsparams,
                data=data_vel,
                variance=variance_vel,
                mask=mask_vel,
                flux_model=flux_model,
                flux_theta=flux_theta,
                flux_image=flux_image,
                flux_image_pars=flux_image_pars,
            )
        else:
            obs = build_velocity_obs(
                image_pars,
                data=data_vel,
                variance=variance_vel,
                mask=mask_vel,
            )

        return cls.from_velocity_obs(model, priors, obs, meta_pars=meta_pars)

    @classmethod
    def from_intensity_model(
        cls,
        model: 'IntensityModel',
        priors: 'PriorDict',
        data_int: jnp.ndarray,
        variance_int,
        image_pars: 'ImagePars',
        meta_pars: Optional[Dict] = None,
        psf=None,
        psf_gsparams=None,
        mask_int=None,
    ) -> 'InferenceTask':
        """
        Create inference task for intensity-only inference (legacy API).

        Delegates to from_intensity_obs() after constructing an ImageObs.

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
        psf : galsim.GSObject, optional
            PSF for intensity channel.
        psf_gsparams : galsim.GSParams, optional
            GalSim rendering parameters for PSF kernel accuracy.
        mask_int : jnp.ndarray, optional
            Boolean mask (True=valid, False=masked). Same shape as data_int.

        Returns
        -------
        InferenceTask
            Configured task ready for sampling.
        """
        from kl_pipe.observation import build_image_obs

        obs = build_image_obs(
            image_pars,
            psf=psf,
            gsparams=psf_gsparams,
            data=data_int,
            variance=variance_int,
            mask=mask_int,
            int_model=model if psf is not None else None,
        )

        return cls.from_intensity_obs(model, priors, obs, meta_pars=meta_pars)

    @classmethod
    def from_joint_model(
        cls,
        model: 'KLModel',
        priors: 'PriorDict',
        data_vel: jnp.ndarray,
        data_int: jnp.ndarray,
        variance_vel,
        variance_int,
        image_pars_vel: 'ImagePars',
        image_pars_int: 'ImagePars',
        meta_pars: Optional[Dict] = None,
        psf_vel=None,
        psf_int=None,
        psf_gsparams=None,
        mask_vel=None,
        mask_int=None,
    ) -> 'InferenceTask':
        """
        Create inference task for joint velocity + intensity inference (legacy API).

        Delegates to from_joint_obs() after constructing obs objects.

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
        psf_vel : galsim.GSObject, optional
            PSF for velocity channel.
        psf_int : galsim.GSObject, optional
            PSF for intensity channel.
        psf_gsparams : galsim.GSParams, optional
            GalSim rendering parameters for PSF kernel accuracy.
        mask_vel : jnp.ndarray, optional
            Boolean mask for velocity data (True=valid). Same shape as data_vel.
        mask_int : jnp.ndarray, optional
            Boolean mask for intensity data (True=valid). Same shape as data_int.

        Returns
        -------
        InferenceTask
            Configured task ready for sampling.
        """
        from kl_pipe.observation import build_joint_obs

        obs_vel, obs_int = build_joint_obs(
            image_pars_vel,
            image_pars_int,
            model.intensity_model,
            psf_vel=psf_vel,
            psf_int=psf_int,
            gsparams=psf_gsparams,
            data_vel=data_vel,
            variance_vel=variance_vel,
            mask_vel=mask_vel,
            data_int=data_int,
            variance_int=variance_int,
            mask_int=mask_int,
        )

        return cls.from_joint_obs(model, priors, obs_vel, obs_int, meta_pars=meta_pars)
