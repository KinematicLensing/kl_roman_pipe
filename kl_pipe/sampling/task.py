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
    from kl_pipe.observation import ImageObs, VelocityObs, GrismObs


def _check_priors_fit_obs_rc(model, priors, obs, obs_rc):
    """Raise if priors imply a more demanding grid than what obs was built for.

    Bug B / Issue #42 prevention: under (b3) architecture obs holds the
    rendering recipe (render_config), and InferenceTask must not silently
    recompute a different one. If priors imply a tighter rc than the obs
    was sized for, the PSF FFT shape will mismatch the wrap-path expectation
    -- raise loudly with the fix instructions instead of crashing in the
    middle of a JIT trace.

    Threads the obs's PSF into the worst-case scan so slow-decay profiles
    (DeVauc, Spergel cusp) don't over-estimate the required grid. Without
    the PSF, the product scan terminates only when ``profile_FT * pixel_FT``
    drops below threshold — which for steep-Sersic profiles is far beyond
    where the Gaussian-like PSF would damp it.
    """
    from kl_pipe.render import RenderConfig

    try:
        priors_rc = RenderConfig.for_priors(
            model,
            priors,
            obs.image_pars.pixel_scale,
            pixel_response=obs.pixel_response,
            psf=getattr(obs, 'psf', None),
        )
    except (KeyError, NotImplementedError, AttributeError):
        return  # priors-based sizing not applicable for this model

    if priors_rc.oversample > obs_rc.oversample:
        raise ValueError(
            f"Priors imply oversample={priors_rc.oversample} but obs was built "
            f"with oversample={obs_rc.oversample}. Rebuild obs with explicit "
            f"render_config:\n"
            f"    rc = RenderConfig.for_priors(model, priors, pixel_scale, "
            f"pixel_response=..., psf=...)\n"
            f"    obs = build_image_obs(image_pars, ..., render_config=rc)"
        )


# _component_priors_for_intensity moved to kl_pipe.source -- its dotted-key
# resolution belongs with the SourceModel namespace it serves. Re-exported
# here for backward compatibility with this module's existing callers.
from kl_pipe.source import _component_priors_for_intensity  # noqa: E402, F401


def _check_source_priors_fit_obs(
    source: 'SourceModel',
    priors: 'PriorDict',
    obs,
    *,
    component_key: Optional[str] = None,
):
    """Obs-type-aware rc handling for SourceModel inference.

    Dispatches on ``type(obs)``:

    - ``VelocityObs``: no-op. Velocity rendering uses spatial oversampling
      rather than a k-space FFT product scan, so the prior-grid adequacy
      question doesn't apply directly at this layer. Returns the obs
      unchanged.
    - ``ImageObs``: if ``obs._rc_was_default`` is True, derive
      the priors-sized rc via ``build_image_render_config`` and rebuild the
      obs with freshly-recomputed precomputed grids; return the rebuilt obs.
      Otherwise, validate the user-supplied rc against the priors (raise
      loudly on mismatch) and return the obs unchanged.
    - ``GrismObs``: analogous to ImageObs, using
      ``build_grism_render_config`` (iterates emission lines for worst-case
      cube-slice bandwidth).

    Parameters
    ----------
    source : SourceModel
        Source description with the component models being validated.
    priors : PriorDict
        Dotted-key priors as consumed by ``InferenceTask.from_obs``.
    obs : ImageObs | VelocityObs | GrismObs
    component_key : str, optional
        Band key for ``ImageObs``; ignored for other obs types.

    Returns
    -------
    ImageObs | VelocityObs | GrismObs
        The (possibly rebuilt) obs. Callers in ``InferenceTask.from_obs``
        replace each channel's input obs with the returned one before
        constructing the likelihood.
    """
    from kl_pipe.observation import GrismObs, ImageObs, VelocityObs

    if isinstance(obs, VelocityObs):
        return obs  # k-space grid sizing N/A for spatial-oversampling rendering

    if isinstance(obs, GrismObs):
        from kl_pipe.render import RenderConfig, build_grism_render_config

        rc_obs = obs.render_config if obs.render_config is not None else RenderConfig()

        # auto-derive + rebuild when the obs carries a builder-default rc;
        # the priors-sized rc bounds cube-slice bandwidth via the Minkowski
        # sum (intensity FT support + velocity-modulation Gaussian FT,
        # damped by PSF). See docs/notes/grism_cube_bandwidth.tex.
        if obs._rc_was_default:
            derived_rc = build_grism_render_config(
                source, priors, obs.grism_pars, psf=obs.psf
            )
            return obs.with_render_config(derived_rc)

        # explicit user rc: validate against priors, raise on mismatch
        priors_rc = build_grism_render_config(
            source, priors, obs.grism_pars, psf=obs.psf
        )
        if priors_rc.oversample > rc_obs.oversample:
            raise ValueError(
                f"Grism priors imply oversample={priors_rc.oversample} but "
                f"grism obs was built with oversample={rc_obs.oversample}. "
                f"Either drop the explicit render_config to let from_obs "
                f"auto-derive, or pass a properly-sized one:\n"
                f"    from kl_pipe.render import build_grism_render_config\n"
                f"    rc = build_grism_render_config(source, priors, grism_pars, "
                f"psf=psf)\n"
                f"    obs = build_grism_obs(grism_pars, z, render_config=rc, psf=psf)"
            )
        return obs

    if isinstance(obs, ImageObs):
        if component_key is None:
            raise ValueError(
                "_check_source_priors_fit_obs(ImageObs) requires component_key"
            )
        if component_key not in source.broadband_models:
            raise ValueError(
                f"component_key '{component_key}' not in source.broadband_models "
                f"(have: {sorted(source.broadband_models)})"
            )
        model = source.broadband_models[component_key]
        sub_priors = _component_priors_for_intensity(
            priors, component_key, model.PARAMETER_NAMES
        )
        if hasattr(model, 'check_priors_safe'):
            model.check_priors_safe(sub_priors)

        from kl_pipe.render import RenderConfig, build_image_render_config

        rc_obs = obs.render_config if obs.render_config is not None else RenderConfig()

        # auto-derive + rebuild when the obs carries a builder-default rc.
        if obs._rc_was_default:
            derived_rc = build_image_render_config(
                source, priors, obs.image_pars, component_key, psf=obs.psf
            )
            return obs.with_render_config(derived_rc, int_model=model)

        # explicit user rc: validate against priors, raise on mismatch
        _check_priors_fit_obs_rc(model, sub_priors, obs, rc_obs)
        return obs

    raise TypeError(
        f"_check_source_priors_fit_obs: unrecognized obs type " f"{type(obs).__name__}"
    )


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
        """Pre-compute index mapping for JIT-compatible theta construction.

        SourceModel-based tasks (via ``from_obs``) skip the mapping --
        their ``likelihood_fn`` consumes ``theta_sampled`` directly using
        the priors' sorted-sample-name ordering, with no flat
        ``model.PARAMETER_NAMES`` to align against.
        """
        from kl_pipe.source import SourceModel

        if isinstance(self.model, SourceModel):
            return
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
        """Full parameter names from the model.

        For SourceModel-based tasks there is no flat model PARAMETER_NAMES;
        the dotted-key namespace lives in the priors. The sampled-name
        ordering is returned as the canonical parameter list.
        """
        from kl_pipe.source import SourceModel

        if isinstance(self.model, SourceModel):
            return tuple(self.priors.sampled_names)
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

        For SourceModel-based tasks the likelihood_fn takes the sampled
        theta directly (it builds the dotted-key pars dict internally
        using the priors' sampled-name tuple closed over via partial),
        so this method is an identity in that path.

        Parameters
        ----------
        theta_sampled : jnp.ndarray
            Array of sampled parameter values (length = n_params).

        Returns
        -------
        jnp.ndarray
            Full theta array in model's PARAMETER_NAMES order (legacy)
            or ``theta_sampled`` unchanged (SourceModel).
        """
        from kl_pipe.source import SourceModel

        if isinstance(self.model, SourceModel):
            return theta_sampled

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
    def from_obs(
        cls,
        source: 'SourceModel',
        priors: 'PriorDict',
        *,
        image_obs: Optional[Dict[str, 'ImageObs']] = None,
        grism_obs: Optional[Dict[str, 'GrismObs']] = None,
        velocity_obs: Optional['VelocityObs'] = None,
        meta_pars: Optional[Dict] = None,
        spectral_oversample: Optional[int] = None,
    ) -> 'InferenceTask':
        """Unified SourceModel inference factory.

        Builds an InferenceTask whose likelihood sums Gaussian
        log-likelihoods over any combination of:

        - per-band broadband images (``image_obs[band_key]``),
        - per-roll grism observations (``grism_obs[grism_key]``),
        - a single velocity observation (``velocity_obs``).

        Source-to-obs validation:

        - At least one obs must be provided.
        - Each ``image_obs`` key must reference an entry in
          ``source.broadband_models``; the obs's ``broadband_key`` (if
          set) must match its dict key.
        - ``grism_obs`` (when non-empty) requires non-empty
          ``source.emission_lines`` AND ``source.velocity_model`` set.
        - ``velocity_obs`` requires ``source.velocity_model`` set. If
          ``velocity_obs.flux_weight_key`` is not None, it must
          reference a key in ``source.emission_lines``;
          ``flux_weight_key=None`` is allowed for unweighted (no PSF)
          velocity rendering.

        Parameters
        ----------
        source : SourceModel
            The source description -- velocity / broadband / emission
            components plus their cross-references.
        priors : PriorDict
            Prior specifications keyed by the dotted-key namespace.
        image_obs : dict[str, ImageObs], optional
            Per-band imaging observations.
        grism_obs : dict[str, GrismObs], optional
            Per-roll grism observations (or any user-chosen string key).
        velocity_obs : VelocityObs, optional
            Single velocity-map observation.
        meta_pars : dict, optional
            User metadata.
        spectral_oversample : int, optional
            Wavelength sub-bin count for cube assembly. When ``None``
            (default), reads from each grism obs's
            ``render_config.spectral_oversample`` (default 5); raises if
            multiple grism obs disagree. Pass an explicit value only to
            override the obs-recorded settings uniformly.
        """
        from kl_pipe.likelihood import create_jitted_likelihood_from_obs
        from kl_pipe.source import SourceModel

        # ---- validate source-to-obs binding -----------------------------

        # at least one obs
        if not image_obs and not grism_obs and velocity_obs is None:
            raise ValueError(
                "InferenceTask.from_obs requires at least one of "
                "image_obs, grism_obs, or velocity_obs"
            )

        # image_obs: every key must be in source.broadband_models
        if image_obs:
            for band_key, obs in image_obs.items():
                if band_key not in source.broadband_models:
                    raise ValueError(
                        f"image_obs key '{band_key}' has no entry in "
                        f"source.broadband_models "
                        f"(have: {sorted(source.broadband_models)})"
                    )
                if obs.broadband_key is not None and obs.broadband_key != band_key:
                    raise ValueError(
                        f"image_obs['{band_key}'].broadband_key="
                        f"'{obs.broadband_key}' disagrees with its dict key"
                    )
                if obs.data is None:
                    raise ValueError(
                        f"image_obs['{band_key}'] has no data; cannot "
                        f"build a likelihood"
                    )

        # grism_obs: non-empty requires velocity + emission lines
        if grism_obs:
            if source.velocity_model is None:
                raise ValueError(
                    "grism_obs requires source.velocity_model to be set "
                    "(Doppler shifts need it)"
                )
            if not source.emission_lines:
                raise ValueError("grism_obs requires non-empty source.emission_lines")
            for grism_key, obs in grism_obs.items():
                if obs.data is None:
                    raise ValueError(
                        f"grism_obs['{grism_key}'] has no data; cannot "
                        f"build a likelihood"
                    )

        # velocity_obs: requires velocity_model; optional flux_weight_key
        if velocity_obs is not None:
            if source.velocity_model is None:
                raise ValueError(
                    "velocity_obs requires source.velocity_model to be set"
                )
            if velocity_obs.data is None:
                raise ValueError("velocity_obs has no data; cannot build a likelihood")
            fwk = velocity_obs.flux_weight_key
            if fwk is not None and fwk not in source.emission_lines:
                raise ValueError(
                    f"velocity_obs.flux_weight_key='{fwk}' has no entry in "
                    f"source.emission_lines "
                    f"(have: {sorted(source.emission_lines)})"
                )

        # ---- per-channel rc handling (auto-derive or validate) -----------
        # _check_source_priors_fit_obs returns the (possibly rebuilt) obs:
        #   - builder-default rc → derive priors-sized rc + rebuild obs
        #   - explicit user rc   → validate against priors (raise on mismatch)
        # The (possibly rebuilt) obs replaces the input one in each channel
        # dict so the likelihood closure operates on it.

        if image_obs:
            image_obs = {
                band_key: _check_source_priors_fit_obs(
                    source, priors, obs, component_key=band_key
                )
                for band_key, obs in image_obs.items()
            }
        if grism_obs:
            grism_obs = {
                grism_key: _check_source_priors_fit_obs(source, priors, obs)
                for grism_key, obs in grism_obs.items()
            }
        if velocity_obs is not None:
            velocity_obs = _check_source_priors_fit_obs(source, priors, velocity_obs)

        # ---- build the likelihood closure --------------------------------

        sampled_names = tuple(priors.sampled_names)
        fixed_pars = dict(priors.fixed_values)

        # resolve spectral_oversample: explicit kwarg wins; else read from
        # grism obs (every roll must agree); else irrelevant (no grism)
        if spectral_oversample is None and grism_obs:
            osfs = {k: o.spectral_oversample for k, o in grism_obs.items()}
            unique = set(osfs.values())
            if len(unique) > 1:
                raise ValueError(
                    f"grism_obs have mismatched spectral_oversample {osfs}; "
                    f"pass spectral_oversample=N explicitly to override"
                )
            spectral_oversample = unique.pop()
        elif spectral_oversample is None:
            spectral_oversample = 15  # unused (no grism), but pass a concrete int

        likelihood_fn = create_jitted_likelihood_from_obs(
            source,
            sampled_names,
            fixed_pars,
            image_obs=image_obs,
            grism_obs=grism_obs,
            velocity_obs=velocity_obs,
            spectral_oversample=spectral_oversample,
        )

        return cls(
            model=source,
            likelihood_fn=likelihood_fn,
            priors=priors,
            data={},
            variance={},
            mask={},
            meta_pars=meta_pars or {},
        )

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

        # model-specific prior validation (e.g. Spergel cusp regime).
        # Runs before grid-fit checks so misconfigured priors fail loudly
        # without first triggering an unmanageable FFT grid computation.
        model.check_priors_safe(priors)

        # rc is canonical on obs (built by build_image_obs); do NOT recompute.
        # Validate that priors fit within the obs's pre-built grid so a stale
        # default obs doesn't silently mismatch with tighter priors.
        from kl_pipe.render import RenderConfig

        rc_int = obs.render_config
        if rc_int is None:
            rc_int = (
                RenderConfig()
            )  # legacy obs without rc; should not happen in new code
        _check_priors_fit_obs_rc(model, priors, obs, rc_int)

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

        # rc is canonical on each obs; do NOT recompute. Validate that priors
        # fit within the obs's pre-built grid (loud failure on drift).
        from kl_pipe.render import RenderConfig

        int_model = model.intensity_model if hasattr(model, 'intensity_model') else None
        rc_int = obs_int.render_config
        if rc_int is None:
            rc_int = RenderConfig()
        if int_model is not None:
            # model-specific prior validation BEFORE grid-fit (fail fast on
            # misconfigured priors so we don't first trigger an unmanageable
            # FFT grid computation in for_priors)
            int_model.check_priors_safe(priors)
            _check_priors_fit_obs_rc(int_model, priors, obs_int, rc_int)

        rc_vel = obs_vel.render_config
        if rc_vel is None:
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

    @classmethod
    def from_grism_obs(
        cls,
        model: 'KLModel',
        priors: 'PriorDict',
        obs: 'GrismObs',
        meta_pars: Optional[Dict] = None,
    ) -> 'InferenceTask':
        """
        Create inference task for grism-only inference.

        Parameters
        ----------
        model : KLModel
            Combined kinematic-lensing model. Must have spectral_model configured.
        priors : PriorDict
            Prior specifications.
        obs : GrismObs
            Grism observation (with grism_pars, cube_pars at concrete z, PSF,
            data, variance, mask).
        meta_pars : dict, optional
            Additional metadata.

        Returns
        -------
        InferenceTask
            Configured task ready for sampling.

        Notes
        -----
        Known limitations
        -----------------
        - The prior-grid adequacy validation that ``from_intensity_obs`` runs
          on ImageObs is NOT run here for this legacy factory. The
          SourceModel-era ``from_obs`` factory dispatches an obs-type-aware
          analogue (``_check_source_priors_fit_obs`` → fine-grid Nyquist
          comparison for grism) and is the preferred entry point.
        - ``kl_pipe/spectral.py`` adds an instrumental sigma derived from the
          quoted grism resolving power in quadrature with the kinematic
          velocity dispersion before the PSF + dispersion stages, which
          likely double-counts the PSF's spectral resolution contribution
          for slitless geometry. ``vel_dispersion`` is correspondingly poorly
          identified. Tracked in ``docs/plans/phase2_lsf_refactor.md``.
        """
        if obs.data is None:
            raise ValueError("GrismObs has no data; cannot create inference task")

        if obs.psf_data is None:
            warnings.warn(
                "\nNo PSF configured — grism model will be unconvolved. Intentional?\n",
                NoPSFWarning,
                stacklevel=2,
            )

        from kl_pipe.likelihood import create_jitted_likelihood_grism

        likelihood_fn = create_jitted_likelihood_grism(model, obs)

        return cls(
            model=model,
            likelihood_fn=likelihood_fn,
            priors=priors,
            data={'grism': obs.data},
            variance={'grism': obs.variance},
            mask={'grism': obs.mask},
            meta_pars=meta_pars or {},
        )

    @classmethod
    def from_joint_photometry_grism_obs(
        cls,
        model: 'KLModel',
        priors: 'PriorDict',
        obs_int: 'ImageObs',
        obs_grism: 'GrismObs',
        meta_pars: Optional[Dict] = None,
    ) -> 'InferenceTask':
        """
        Create inference task for joint broadband image + grism inference.

        Parameters
        ----------
        model : KLModel
            Combined kinematic-lensing model with spectral_model configured.
        priors : PriorDict
            Prior specifications.
        obs_int : ImageObs
            Broadband photometric image observation.
        obs_grism : GrismObs
            Grism observation.
        meta_pars : dict, optional
            Additional metadata.

        Returns
        -------
        InferenceTask
            Configured task ready for sampling.

        Notes
        -----
        Runs the intensity-channel prior validation (``check_priors_safe`` +
        ``_check_priors_fit_obs_rc``) on the ImageObs. The grism-channel
        analogue is not yet implemented; see ``from_grism_obs`` notes.

        Known limitation: the photometric image and the emission cube inside
        ``render_grism`` share ``kl_model.intensity_model``'s single centroid
        parameter pair (``int_x0``/``int_y0``). If the two channels have
        independent astrometric solutions this shared centroid is the wrong
        degree of freedom. Tracked in
        ``docs/plans/phase3_sourcemodel_refactor.md``.
        """
        if obs_int.data is None:
            raise ValueError("ImageObs has no data; cannot create inference task")
        if obs_grism.data is None:
            raise ValueError("GrismObs has no data; cannot create inference task")

        missing = []
        if obs_int.psf_data is None:
            missing.append('intensity')
        if obs_grism.psf_data is None:
            missing.append('grism')
        if missing:
            channels = ' and '.join(missing)
            warnings.warn(
                f"\nNo PSF configured for {channels} channel(s) — model will be unconvolved. Intentional?\n",
                NoPSFWarning,
                stacklevel=2,
            )

        # intensity-channel prior validation (mirrors from_intensity_obs /
        # from_joint_obs). Grism-channel analogue not yet implemented; see
        # from_grism_obs notes and docs/plans/phase3_sourcemodel_refactor.md.
        from kl_pipe.render import RenderConfig

        int_model = model.intensity_model if hasattr(model, 'intensity_model') else None
        rc_int = obs_int.render_config
        if rc_int is None:
            rc_int = RenderConfig()
        if int_model is not None:
            int_model.check_priors_safe(priors)
            _check_priors_fit_obs_rc(int_model, priors, obs_int, rc_int)

        from kl_pipe.likelihood import (
            create_jitted_likelihood_joint_photometry_grism,
        )

        likelihood_fn = create_jitted_likelihood_joint_photometry_grism(
            model, obs_int, obs_grism, render_config_int=rc_int
        )

        task = cls(
            model=model,
            likelihood_fn=likelihood_fn,
            priors=priors,
            data={'intensity': obs_int.data, 'grism': obs_grism.data},
            variance={'intensity': obs_int.variance, 'grism': obs_grism.variance},
            mask={'intensity': obs_int.mask, 'grism': obs_grism.mask},
            meta_pars=meta_pars or {},
        )
        task._render_configs = {'intensity': rc_int}
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
        from kl_pipe.pixel import BoxPixel
        from kl_pipe.render import RenderConfig

        # legacy convenience API builds obs internally; thread priors-derived
        # rc so default oversample=5 doesn't undersize the grid for tight
        # priors (would later trigger the loud-failure check in from_intensity_obs)
        try:
            rc = RenderConfig.for_priors(
                model,
                priors,
                image_pars.pixel_scale,
                pixel_response=BoxPixel(image_pars.pixel_scale),
                psf=psf,
            )
        except (KeyError, NotImplementedError, AttributeError):
            rc = None  # fall through to default in build_image_obs

        obs = build_image_obs(
            image_pars,
            psf=psf,
            gsparams=psf_gsparams,
            data=data_int,
            variance=variance_int,
            mask=mask_int,
            int_model=model if psf is not None else None,
            render_config=rc,
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
        from kl_pipe.pixel import BoxPixel
        from kl_pipe.render import RenderConfig

        # legacy convenience API builds obs internally; thread priors-derived
        # rc so default oversample=5 doesn't undersize the grid for tight
        # priors (would later trigger the loud-failure check in from_joint_obs)
        try:
            rc_int = RenderConfig.for_priors(
                model.intensity_model,
                priors,
                image_pars_int.pixel_scale,
                pixel_response=BoxPixel(image_pars_int.pixel_scale),
                psf=psf_int,
            )
        except (KeyError, NotImplementedError, AttributeError):
            rc_int = None  # fall through to default in build_joint_obs

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
            render_config_int=rc_int,
        )

        return cls.from_joint_obs(model, priors, obs_vel, obs_int, meta_pars=meta_pars)
