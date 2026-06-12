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


@dataclass
class LaplacePreconditioner:
    """Laplace-approximation preconditioner for gradient-based sampling.

    Bundles the MAP point and a regularized inverse-Hessian to use as a NUTS
    mass matrix, letting warmup start near-optimally conditioned instead of
    climbing from an identity metric. Produced by
    ``InferenceTask.laplace_preconditioner``; consumed by ``NumpyroSampler``
    (directly, or via ``NumpyroSamplerConfig(precondition='laplace')``).

    Attributes
    ----------
    map_point : np.ndarray
        MAP estimate in sampled-parameter (physical) space, ``sampled_names``
        order. Used as the chain init.
    inverse_mass_matrix : np.ndarray
        Regularized inverse Hessian of the negative log-posterior at the MAP
        (``(n_params, n_params)``), in the same space/order. Used as the
        fixed NUTS inverse mass matrix.
    n_starts_converged : int
        Number of multi-start optimizations that converged to the best mode.
    condition_number : float
        Condition number of the regularized Hessian (post eigenvalue floor).
    """

    map_point: np.ndarray
    inverse_mass_matrix: np.ndarray
    n_starts_converged: int
    condition_number: float


if TYPE_CHECKING:
    from kl_pipe.model import Model
    from kl_pipe.source import SourceModel
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
    - Model: The forward model (a SourceModel, or a bare velocity/intensity model)
    - Likelihood: JIT-compiled log-likelihood function
    - Priors: PriorDict specifying sampled vs fixed parameters
    - Data: Observed data arrays
    - Variance: Observation variance (same shape as data, or scalar)
    - Meta parameters: Optional metadata (systematics, etc.)

    Provides methods for computing the log posterior and its gradient,
    which are used by sampler backends.

    Parameters
    ----------
    model : Model or SourceModel
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
    >>> from kl_pipe.source import SourceModel
    >>> from kl_pipe.priors import Uniform, PriorDict
    >>> from kl_pipe.observation import build_velocity_obs
    >>> from kl_pipe.sampling import InferenceTask
    >>>
    >>> source = SourceModel(velocity_model=OffsetVelocityModel())
    >>> priors = PriorDict({
    ...     'vel.vcirc': Uniform(100, 350),
    ...     'cosi': Uniform(0.1, 0.99),
    ...     'vel.v0': 10.0,  # Fixed
    ... })
    >>>
    >>> obs = build_velocity_obs(image_pars, data=data_vel, variance=25.0)
    >>> task = InferenceTask.from_obs(source, priors, velocity_obs=obs)
    >>> log_prob_fn = task.get_log_posterior_fn()
    """

    model: Union['Model', 'SourceModel']
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

    def laplace_preconditioner(
        self,
        n_starts: int = 4,
        eig_floor: float = 1e-4,
        maxiter: int = 300,
        seed: int = 0,
    ) -> 'LaplacePreconditioner':
        """Compute a Laplace preconditioner (MAP + regularized inverse Hessian).

        Multi-start L-BFGS-B from prior draws (truth-free) locates the MAP,
        then the Hessian of the negative log-posterior there is regularized
        (eigenvalue floor) and inverted to serve as a NUTS mass matrix. Lets
        warmup begin near-optimally conditioned, skipping the expensive
        early-warmup transient. See
        ``experiments/sweverett/flagship_speedup`` for the validating study.

        The optimizer uses the prior bounds (``get_bounds``), so iterates stay
        in-support; the Hessian is taken at the interior MAP.

        Parameters
        ----------
        n_starts : int, default 4
            Number of L-BFGS-B starts from independent prior draws. The best
            (highest log-posterior) converged mode is used. Multi-start guards
            against local modes (e.g. the position-angle multimodality).
        eig_floor : float, default 1e-4
            Hessian eigenvalues below ``eig_floor * max_eigenvalue`` are floored
            to that value before inversion, capping the mass-matrix condition
            number at ``1/eig_floor`` (handles near-degenerate directions).
        maxiter : int, default 300
            Max L-BFGS-B iterations per start.
        seed : int, default 0
            PRNG seed for the prior-draw starting points.

        Returns
        -------
        LaplacePreconditioner
            MAP point + regularized inverse-Hessian mass matrix.

        Raises
        ------
        RuntimeError
            If no optimization start converges to a finite-log-posterior mode.
        """
        from scipy.optimize import minimize

        val_and_grad = self.get_log_posterior_and_grad_fn()
        hess_fn = jax.jit(jax.hessian(lambda t: -self._log_posterior_jittable(t)))

        # Per-parameter characteristic scale from prior draws. The physical
        # problem is badly scaled (e.g. vcirc~200 vs g1~0.02); optimizing in
        # scaled coords u (theta = loc + scale*u) conditions L-BFGS-B well.
        prior_batch = np.asarray(
            self.sample_prior(jax.random.PRNGKey(seed), n_samples=512)
        )
        loc = prior_batch.mean(axis=0)
        scale = prior_batch.std(axis=0)
        scale = np.where(scale > 0, scale, 1.0)  # guard degenerate/fixed dims

        def neg_u(u):
            theta = jnp.asarray(loc + scale * u)
            v, g = val_and_grad(theta)
            # chain rule: d(-logpost)/du = -grad_theta * scale
            return float(-v), np.asarray(-g, dtype=np.float64) * scale

        # Multi-start from prior draws (truth-free), in scaled coords.
        starts = np.asarray(
            self.sample_prior(jax.random.PRNGKey(seed + 1), n_samples=n_starts)
        )
        best = None
        n_converged = 0
        for s0 in starts:
            u0 = (np.asarray(s0, dtype=np.float64) - loc) / scale
            res = minimize(
                neg_u,
                u0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': maxiter},
            )
            if res.success and np.isfinite(res.fun):
                n_converged += 1
                if best is None or res.fun < best.fun:
                    best = res

        if best is None:
            raise RuntimeError(
                "laplace_preconditioner: no optimization start converged to a "
                "finite-log-posterior mode (tried "
                f"{n_starts} starts). Check priors/data."
            )

        theta_map = jnp.asarray(loc + scale * best.x)
        H = np.asarray(hess_fn(theta_map), dtype=np.float64)
        H = 0.5 * (H + H.T)
        # Scale-aware regularization: normalize the Hessian by the per-parameter
        # scale (diag(scale) @ H @ diag(scale)) BEFORE flooring eigenvalues, so
        # the eigenvalue floor caps only genuine degeneracy -- not the benign
        # scale spread (e.g. vcirc~200 vs g1~0.02, which the mass matrix must
        # keep). Floor a raw physical Hessian and you destroy that scale range.
        Hn = (scale[:, None] * H) * scale[None, :]
        Hn = 0.5 * (Hn + Hn.T)
        w, V = np.linalg.eigh(Hn)
        w_floored = np.maximum(w, w.max() * eig_floor)  # floor soft/neg dirs
        Hn_reg = (V * w_floored) @ V.T
        inv_n = np.linalg.inv(Hn_reg)
        # map back: inv_mass_theta = S @ inv(Hn_reg) @ S = inv(H_reg) with scale kept
        inv_mass = (scale[:, None] * inv_n) * scale[None, :]
        inv_mass = 0.5 * (inv_mass + inv_mass.T)
        cond = float(w_floored.max() / w_floored.min())

        return LaplacePreconditioner(
            map_point=np.asarray(theta_map, dtype=np.float64),
            inverse_mass_matrix=inv_mass,
            n_starts_converged=n_converged,
            condition_number=cond,
        )

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
