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
        import dataclasses

        from kl_pipe.render import (
            RenderConfig,
            build_grism_render_config,
            line_window_halfwidth_for_priors,
            local_line_window_halfwidth_for_priors,
        )

        rc_obs = obs.render_config if obs.render_config is not None else RenderConfig()

        def _fill_line_window_halfwidth(rc):
            # analytic dispersal needs a static deposit window under JIT;
            # size it from prior extremes when the caller left it None
            if (
                rc.dispersal_method != 'analytic'
                or rc.line_window_halfwidth is not None
            ):
                return rc
            size = (
                local_line_window_halfwidth_for_priors
                if rc.line_window_mode == 'local'
                else line_window_halfwidth_for_priors
            )
            hw = size(source, priors, obs.grism_pars, rc.oversample)
            return dataclasses.replace(rc, line_window_halfwidth=hw)

        # auto-derive + rebuild when the obs carries a builder-default rc;
        # the priors-sized rc bounds cube-slice bandwidth via the Minkowski
        # sum (intensity FT support + velocity-modulation Gaussian FT,
        # damped by PSF). See docs/notes/grism_cube_bandwidth.tex.
        if obs._rc_was_default:
            derived_rc = build_grism_render_config(
                source, priors, obs.grism_pars, psf=obs.psf
            )
            return obs.with_render_config(_fill_line_window_halfwidth(derived_rc))

        # explicit user rc: validate against the aliasing requirement only
        # (min_oversample=1); the accuracy floor applies to auto-derived
        # configs, while an explicit rc is an informed speed choice
        priors_rc = build_grism_render_config(
            source, priors, obs.grism_pars, psf=obs.psf, min_oversample=1
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
        # an explicit rc is an informed speed choice for everything it sets,
        # but a None line_window_halfwidth on the analytic path would raise
        # at trace time -- fill it from priors instead
        filled_rc = _fill_line_window_halfwidth(rc_obs)
        if filled_rc is not rc_obs:
            return obs.with_render_config(filled_rc)
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
        maxiter: int = 2000,
        seed: int = 0,
        hessian_method: str = 'fd',
        fd_rel_step: float = 1e-5,
        extra_starts: Optional[np.ndarray] = None,
    ) -> 'LaplacePreconditioner':
        """Compute a Laplace preconditioner (MAP + regularized inverse Hessian).

        Multi-start L-BFGS-B from prior draws (truth-free) locates the MAP,
        then the Hessian of the negative log-posterior there is regularized
        (eigenvalue floor) and inverted to serve as a NUTS mass matrix. Lets
        warmup begin near-optimally conditioned, skipping the expensive
        early-warmup transient (~2x faster warmup measured on correlated
        joint posteriors).

        The optimizer runs unbounded in scaled coordinates (``theta = loc +
        scale * u``); out-of-support iterates receive ``-inf`` log-posterior
        from the prior, which acts as a soft barrier keeping the converged MAP
        in-support. The Hessian is taken at that MAP.

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
        maxiter : int, default 2000
            Max L-BFGS-B iterations per start. Sharp high-SNR posteriors need
            more than a few hundred iterations to reach the mode; too low a cap
            leaves the mode-finding start unconverged and produces a garbage MAP.
        seed : int, default 0
            PRNG seed for the prior-draw starting points.
        hessian_method : {'fd', 'ad'}, default 'fd'
            How to evaluate the Hessian at the MAP. 'fd' (default) uses
            central finite differences of the already-compiled gradient
            (2 * n_params grad evaluations, no new compilation); with float64
            and prior-scaled steps the resulting mass matrix agrees with 'ad'
            to well below the ``eig_floor`` regularization. 'ad' uses exact
            second-order autodiff (``jax.hessian``); tracing and compiling
            that second-order graph is paid on every call and dominates the
            preconditioner cost on large joint tasks -- use it only for
            float32 runs or as a cross-check.
        fd_rel_step : float, default 1e-5
            Relative step for ``hessian_method='fd'``, in units of the
            per-parameter prior scale. Steps are shrunk to stay inside prior
            bounds when the MAP sits near a bound.
        extra_starts : np.ndarray, optional
            Additional explicit start points, shape (n_extra, n_sampled), in
            sampled-parameter order, appended to the prior-draw starts. Use
            when random prior draws can miss a known multimodal basin (e.g.
            position-angle-stratified starts: random draws may all land in
            the wrong PA basin, whose shape-shear-compensated mode then traps
            the sampler).

        Returns
        -------
        LaplacePreconditioner
            MAP point + regularized inverse-Hessian mass matrix.

        Raises
        ------
        RuntimeError
            If no optimization start converges to a finite-log-posterior mode,
            or if the finite-difference Hessian encounters non-finite
            gradients.
        """
        from scipy.optimize import minimize

        if hessian_method not in ('ad', 'fd'):
            raise ValueError(
                f"hessian_method must be 'ad' or 'fd', got '{hessian_method}'"
            )

        val_and_grad = self.get_log_posterior_and_grad_fn()

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
        if extra_starts is not None:
            extra_starts = np.atleast_2d(np.asarray(extra_starts, dtype=np.float64))
            if extra_starts.shape[1] != starts.shape[1]:
                raise ValueError(
                    f"extra_starts has {extra_starts.shape[1]} columns; task "
                    f"has {starts.shape[1]} sampled parameters"
                )
            starts = np.vstack([starts, extra_starts])
        # Keep the best finite objective across all starts, converged or not.
        # A sharp high-SNR posterior can need > maxiter iterations to satisfy
        # L-BFGS-B's convergence test, so the mode-finding start may return
        # success=False; discarding it (keeping only success=True starts) lets a
        # spuriously-"converged" start on a flat plateau win, yielding a garbage
        # MAP -> a mis-scaled mass matrix -> ~100% NUTS divergence. Lower neg-log
        # posterior is always closer to the mode, so best-of-finite is the right
        # pick; warn loudly if it did not formally converge.
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
            if not np.isfinite(res.fun):
                continue
            if res.success:
                n_converged += 1
            if best is None or res.fun < best.fun:
                best = res

        if best is None:
            raise RuntimeError(
                "laplace_preconditioner: no optimization start reached a "
                "finite log-posterior (tried "
                f"{len(starts)} starts). Check priors/data."
            )
        if not best.success:
            warnings.warn(
                "laplace_preconditioner: best MAP start did not satisfy "
                f"L-BFGS-B convergence (neg_logpost={best.fun:.4g}); using it "
                f"anyway as the closest of {len(starts)} starts to the mode. "
                f"Raise maxiter (currently {maxiter}) if sampling diverges.",
                RuntimeWarning,
            )

        theta_map = jnp.asarray(loc + scale * best.x)
        if hessian_method == 'ad':
            hess_fn = jax.jit(jax.hessian(lambda t: -self._log_posterior_jittable(t)))
            H = np.asarray(hess_fn(theta_map), dtype=np.float64)
        else:
            H = self._fd_hessian(val_and_grad, theta_map, scale, fd_rel_step)
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

    def _fd_hessian(
        self,
        val_and_grad: Callable,
        theta: jnp.ndarray,
        scale: np.ndarray,
        rel_step: float,
    ) -> np.ndarray:
        """Hessian of the negative log-posterior by central differences.

        Differences the already-compiled gradient (2 * n_params evaluations),
        so no second-order autodiff graph is traced or compiled. Steps are
        prior-scaled and shrunk to keep both stencil points strictly inside
        prior bounds.
        """
        if not jax.config.jax_enable_x64:
            raise RuntimeError(
                "_fd_hessian requires float64 (jax_enable_x64): float32 "
                "gradient noise is amplified by the difference quotient and "
                "silently degrades the mass matrix. Use hessian_method='ad' "
                "for float32 runs."
            )
        th = np.asarray(theta, dtype=np.float64)
        D = th.size
        bounds = self.get_bounds()
        cols = []
        for j in range(D):
            h = rel_step * scale[j]
            low, high = bounds[j]
            # keep both stencil points strictly in-support (out-of-support
            # gradients are meaningless through the -inf prior barrier)
            if low is not None and np.isfinite(low):
                h = min(h, 0.5 * (th[j] - low))
            if high is not None and np.isfinite(high):
                h = min(h, 0.5 * (high - th[j]))
            if not h > 0:
                raise RuntimeError(
                    f"_fd_hessian: MAP is at a prior bound for parameter "
                    f"index {j} (theta={th[j]:.6g}, bounds=({low}, {high})); "
                    "cannot take a finite-difference step. Use "
                    "hessian_method='ad' or check priors/data."
                )
            tp = th.copy()
            tp[j] += h
            tm = th.copy()
            tm[j] -= h
            _, gp = val_and_grad(jnp.asarray(tp))
            _, gm = val_and_grad(jnp.asarray(tm))
            gp = np.asarray(gp, dtype=np.float64)
            gm = np.asarray(gm, dtype=np.float64)
            if not (np.all(np.isfinite(gp)) and np.all(np.isfinite(gm))):
                raise RuntimeError(
                    f"_fd_hessian: non-finite gradient at stencil point for "
                    f"parameter index {j} (step {h:.3g}). Use "
                    "hessian_method='ad' or check priors/data."
                )
            # grad of +log-posterior; negate for the negative-log-posterior H
            cols.append((gm - gp) / (2.0 * h))
        H = np.stack(cols, axis=1)
        return 0.5 * (H + H.T)

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
        spectral_method: Optional[str] = None,
        psf_mode: Optional[str] = None,
        cube_mode: Optional[str] = None,
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
            Wavelength sub-bin count for cube assembly (only used when
            the resolved ``spectral_method`` is ``'oversample'``). When
            ``None`` (default), reads from each grism obs's
            ``render_config.spectral_oversample`` (default 15); raises
            if multiple grism obs disagree. Pass an explicit value only
            to override the obs-recorded settings uniformly.
        spectral_method : str, optional
            Spectral bin-integration method, ``'erf'`` (exact, default)
            or ``'oversample'``. When ``None`` (default), reads from
            each grism obs's ``render_config.spectral_method``; raises
            if multiple grism obs disagree. Pass an explicit value only
            to override uniformly.
        psf_mode : str, optional
            Grism PSF pathway, ``'post_dispersion'`` (single convolution
            of the dispersed image; default) or ``'per_slice'``
            (reference path). When ``None`` (default), reads from each
            grism obs's ``render_config.psf_mode``; raises if multiple
            grism obs disagree. Pass an explicit value only to override
            uniformly.
        cube_mode : str, optional
            Cube-sharing strategy across grism obs, ``'shared'``
            (default; cube-compatible obs render from one celestial-frame
            cube -- multi-obs groups require
            ``psf_mode='post_dispersion'`` and pure-rotation WCSs, loud
            errors otherwise) or ``'per_roll'`` (reference path; every
            obs rebuilds its own detector-frame cube). When ``None``
            (default), reads from each grism obs's
            ``render_config.cube_mode``; raises if multiple grism obs
            disagree. Pass an explicit value only to override uniformly.
        """
        from kl_pipe.likelihood import create_jitted_likelihood_from_obs
        from kl_pipe.source import SourceModel
        from kl_pipe.transformation import COSI_FLOOR

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

        # cosi floor: the inclined-disk surface-brightness brightening
        # (IntensityModel.__call__, ~1/cosi) diverges at edge-on. Reject cosi
        # priors that reach COSI_FLOOR for any task that renders an intensity
        # component (broadband image, grism emission line, or PSF flux-weighted
        # velocity). Velocity-only tasks are exempt: edge-on cosi is physical
        # and informative there (LOS projection uses sin i, not 1/cosi).
        # Unbounded priors (e.g. Gaussian -> bounds (None, None)) are not
        # statically checkable and pass through.
        renders_intensity = (
            bool(image_obs)
            or bool(grism_obs)
            or (velocity_obs is not None and velocity_obs.flux_weight_key is not None)
        )
        if renders_intensity:
            cosi_low, _ = priors.get_param_bounds('cosi')
            if cosi_low is not None and cosi_low < COSI_FLOOR:
                raise ValueError(
                    f"cosi prior lower bound ({cosi_low:g}) reaches the edge-on "
                    f"floor (COSI_FLOOR={COSI_FLOOR:g}); the 1/cosi surface-"
                    f"brightness brightening diverges there. Use a strictly "
                    f"positive lower bound (e.g. Uniform(0.05, 0.99)) for "
                    f"intensity-rendering tasks."
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

        # resolve spectral_method the same way (every roll must agree)
        if spectral_method is None and grism_obs:
            methods = {k: o.spectral_method for k, o in grism_obs.items()}
            unique_methods = set(methods.values())
            if len(unique_methods) > 1:
                raise ValueError(
                    f"grism_obs have mismatched spectral_method {methods}; "
                    f"pass spectral_method=... explicitly to override"
                )
            spectral_method = unique_methods.pop()
        elif spectral_method is None:
            spectral_method = 'erf'  # unused (no grism), but pass a concrete str

        # resolve psf_mode the same way (every roll must agree)
        if psf_mode is None and grism_obs:
            modes = {k: o.psf_mode for k, o in grism_obs.items()}
            unique_modes = set(modes.values())
            if len(unique_modes) > 1:
                raise ValueError(
                    f"grism_obs have mismatched psf_mode {modes}; "
                    f"pass psf_mode=... explicitly to override"
                )
            psf_mode = unique_modes.pop()
        elif psf_mode is None:
            psf_mode = 'post_dispersion'  # unused (no grism); concrete str

        # resolve cube_mode the same way (every roll must agree)
        if cube_mode is None and grism_obs:
            cmodes = {k: o.cube_mode for k, o in grism_obs.items()}
            unique_cmodes = set(cmodes.values())
            if len(unique_cmodes) > 1:
                raise ValueError(
                    f"grism_obs have mismatched cube_mode {cmodes}; "
                    f"pass cube_mode=... explicitly to override"
                )
            cube_mode = unique_cmodes.pop()
        elif cube_mode is None:
            cube_mode = 'shared'  # unused (no grism); concrete str

        likelihood_fn = create_jitted_likelihood_from_obs(
            source,
            sampled_names,
            fixed_pars,
            image_obs=image_obs,
            grism_obs=grism_obs,
            velocity_obs=velocity_obs,
            spectral_oversample=spectral_oversample,
            spectral_method=spectral_method,
            psf_mode=psf_mode,
            cube_mode=cube_mode,
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
