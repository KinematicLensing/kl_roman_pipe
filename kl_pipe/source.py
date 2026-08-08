"""
SourceModel: unified source description.

A ``SourceModel`` may populate any subset of three component slots:

- ``velocity_model``: a singular ``VelocityModel`` (kinematics).
- ``broadband_models``: a dict ``{filter_name: IntensityModel}`` for broadband
  imaging in one or more filters.
- ``emission_lines``: a dict ``{line_name: EmissionLine}`` for grism / IFU
  emission line cube assembly.

The SourceModel layer is the only place that knows about the dotted-key
parameter namespace (``vel.vcirc``, ``F087.flux``, ``Halpha.flux``, etc.).
The underlying VelocityModel / IntensityModel classes operate on flat
theta arrays matching their own ``PARAMETER_NAMES`` and stay unaware of
the namespace.

``EmissionLine`` and the ``LINE_LAMBDAS`` rest-wavelength registry live
in ``kl_pipe.lines`` (import from there directly).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import jax

from kl_pipe._precision import ensure_precision

ensure_precision()

import jax.numpy as jnp  # noqa: E402

from kl_pipe.coordinates import (  # noqa: E402
    image_rotation_from_wcs,
    rotate_position,
    rotate_shear,
)
from kl_pipe.lines import LINE_LAMBDAS, EmissionLine  # noqa: E402

if TYPE_CHECKING:
    from kl_pipe.model import IntensityModel, VelocityModel
    from kl_pipe.observation import GrismObs, ImageObs, VelocityObs, FiberObs
    from kl_pipe.spectral import CubePars

from kl_pipe.constants import C_KMS as _C_KMS  # noqa: E402

import matplotlib.pyplot as plt

# ===========================================================================
# Theta routing helpers (module-level so they JIT cleanly)
# ===========================================================================


def _lookup_param(pars: dict, prefix: str, param_name: str):
    """Resolve one model parameter from a dotted-key ``pars`` dict.

    Resolution order:
      1. ``<prefix>.<param_name>`` -- per-component value (e.g. ``F087.rscale``).
      2. ``<param_name>``          -- top-level shared value (e.g. ``cosi``, ``g1``).

    Raises ``KeyError`` if neither key is present.
    """
    for key in (f'{prefix}.{param_name}', param_name):
        if key in pars:
            return pars[key]
    raise KeyError(
        f"could not resolve param '{param_name}' for component '{prefix}'; "
        f"tried '{prefix}.{param_name}', '{param_name}'"
    )


def _build_component_theta(
    pars: dict, prefix: str, param_names: Tuple[str, ...]
) -> jnp.ndarray:
    """Build a flat theta array matching ``param_names`` from the pars dict."""
    return jnp.asarray([_lookup_param(pars, prefix, p) for p in param_names])


def _component_priors_for_intensity(
    priors, prefix: str, model_param_names: Tuple[str, ...]
):
    """Build a per-component ``PriorDict`` view keyed by ``model.PARAMETER_NAMES``.

    Translates dotted-key SourceModel priors into a flat-keyed PriorDict so
    that ``RenderConfig.for_priors`` and ``check_priors_safe`` (which both
    query by ``model.PARAMETER_NAMES``) see the right specs. Resolution
    mirrors ``_lookup_param``:

    1. ``<prefix>.<param_name>`` -- per-component value (e.g. ``F087.rscale``).
    2. ``<param_name>``          -- top-level shared value (e.g. ``cosi``).

    Skips params absent from both lookups; ``_extract_worst_case_params``
    tolerates missing names.
    """
    from kl_pipe.priors import PriorDict

    spec = {}
    for name in model_param_names:
        for key in (f'{prefix}.{name}', name):
            if key in priors._param_spec:
                spec[name] = priors._param_spec[key]
                break
    return PriorDict(spec)


def _apply_obs_rotation(
    theta: jnp.ndarray,
    param_names: Tuple[str, ...],
    image_rotation: float,
) -> jnp.ndarray:
    """Rotate celestial-frame ``theta_int`` + ``(g1, g2)`` + ``(x0, y0)``
    into the obs's detector frame.

    Sign convention matches ``OrientedAngle._sky2cartesian`` from kl-tools:
    ``theta_int_det = theta_int_celestial - image_rotation``. Shear rotates
    spin-2 by ``2 * image_rotation`` via ``coordinates.rotate_shear``;
    centroid offsets rotate spin-1 via ``coordinates.rotate_position`` (a
    fixed sky position appears rotated in each roll's detector frame).

    ``image_rotation`` is read from frozen obs aux (a Python float). If it is
    exactly zero (default-WCS path), the rotation step is skipped at trace
    time.
    """
    if image_rotation == 0.0:
        return theta
    if 'theta_int' in param_names:
        i = param_names.index('theta_int')
        theta = theta.at[i].set(theta[i] - image_rotation)
    if 'g1' in param_names and 'g2' in param_names:
        ig1, ig2 = param_names.index('g1'), param_names.index('g2')
        g1d, g2d = rotate_shear(theta[ig1], theta[ig2], image_rotation)
        theta = theta.at[ig1].set(g1d).at[ig2].set(g2d)
    if 'x0' in param_names and 'y0' in param_names:
        ix, iy = param_names.index('x0'), param_names.index('y0')
        x0d, y0d = rotate_position(theta[ix], theta[iy], image_rotation)
        theta = theta.at[ix].set(x0d).at[iy].set(y0d)
    return theta


@dataclass
class SourceModel:
    """Unified source description: velocity + broadband + emission line components.

    A SourceModel may populate any subset of {velocity_model,
    broadband_models, emission_lines}. ``__post_init__`` raises if all
    three are empty. Otherwise each slot is independently optional.

    Parameters
    ----------
    velocity_model : VelocityModel, optional
        Singular velocity model (rotation curve + flux-weight projection).
        Required for grism and velocity-channel inference.
    broadband_models : dict[str, IntensityModel], optional
        Per-filter broadband intensity models. Keys are user-chosen
        filter labels (e.g. ``'F087'``, ``'F184'``).
    emission_lines : dict[str, EmissionLine], optional
        Per-line emission models. Keys are line names. If a key exists
        in ``LINE_LAMBDAS``, the line's ``lambda_rest`` is auto-resolved
        from the registry; otherwise the line must set ``lambda_rest``
        explicitly.
    cube_remat : bool, default True
        Wrap ``build_cube`` in ``jax.checkpoint`` (policy
        ``dots_saveable``) so the backward pass recomputes the cube
        assembly instead of storing its intermediates. Rematerializing
        the cube assembly trades a cheap forward recompute for less
        memory traffic; measured ~1.2x speedup on posterior gradients on
        CPU, with gradients identical to the unwrapped path. Set False
        to disable (e.g. for profiling comparisons).

    Validation (in __post_init__):
    - At least one component must be non-empty.
    - ``broadband_models`` and ``emission_lines`` keys must be disjoint
      (so dotted-key priors like ``F087.flux`` are unambiguous).
    - For each emission line:
        * If ``intensity_key`` is set, it must reference an existing
          line key in ``emission_lines``.
        * If ``continuum_key`` is set, the referenced line must have
          a ``continuum`` set (otherwise there is nothing to share).
        * If ``dispersion_key`` is set, it must reference an existing
          line key, and that line must not itself reference dispersion
          via its own ``dispersion_key`` (no chained sharing).
        * If ``lambda_rest`` is None, the line's name must be in
          ``LINE_LAMBDAS``; resolved automatically.
    """

    velocity_model: Optional['VelocityModel'] = None
    broadband_models: Dict[str, 'IntensityModel'] = field(default_factory=dict)
    emission_lines: Dict[str, EmissionLine] = field(default_factory=dict)
    cube_remat: bool = True

    # ----------------------------------------------------------------- render

    def render_broadband(
        self,
        pars: dict,
        obs: 'ImageObs',
        band_key: str,
    ) -> jnp.ndarray:
        """Render the broadband intensity model for filter ``band_key``.

        ``pars`` is the dotted-key parameter dict (sampled + fixed merged
        by the caller, typically via ``PriorDict.theta_to_full_pars``).
        SourceModel routes ``pars`` into a flat theta for
        ``broadband_models[band_key]``, applies the celestial-to-detector
        rotation derived from ``obs.image_pars.wcs``, and dispatches to
        the intensity model's ``render_image``.
        """
        if band_key not in self.broadband_models:
            raise KeyError(
                f"band_key '{band_key}' not in broadband_models "
                f"(have: {sorted(self.broadband_models)})"
            )
        model = self.broadband_models[band_key]
        theta = _build_component_theta(pars, band_key, model.PARAMETER_NAMES)
        image_rotation = image_rotation_from_wcs(obs.image_pars.wcs)
        theta = _apply_obs_rotation(theta, model.PARAMETER_NAMES, image_rotation)
        return model.render_image(theta, obs=obs)

    def render_grism(
        self,
        pars: dict,
        obs: 'GrismObs',
        plane: str = 'obs',
        spectral_oversample: int | None = None,
        spectral_method: str | None = None,
        psf_mode: str | None = None,
    ) -> jnp.ndarray:
        """Render the dispersed 2D grism image.

        Pipeline:
          1. Build the intrinsic cube via ``build_cube`` (at fine spatial
             scale when ``obs.oversample > 1``), deriving the celestial-
             to-detector rotation from ``obs.grism_pars.image_pars.wcs``
             to thread celestial-frame priors into detector-frame thetas.
          2. PSF convolution, placement set by ``psf_mode``:
             ``'post_dispersion'`` (default) disperses the raw cube and
             convolves the 2D dispersed image once (exact padded linear
             convolution); ``'per_slice'`` convolves every wavelength
             slice before dispersion via ``vmap``. For a shared
             wavelength-independent PSF the two are mathematically
             identical up to stamp-boundary truncation terms bounded by
             the ``folding_threshold`` flux class. Both use
             ``bin=False`` so images stay at fine resolution.
          3. Disperse via ``disperse_cube`` (existing kl_pipe.dispersion).
          4. Apply the precomputed BoxPixel sinc (coarse-pixel integration
             in k-space) and read out the box-averaged field at coarse
             pixel centers at the 2D output.

        .. note::
           Both modes apply a **single shared** PSF (``obs.psf_data``).
           Per-line / wavelength-dependent PSFs require ``'per_slice'``
           (or per-line sub-cubes) and are tracked as an open
           architectural item — see issue #51. Deferred past Phase 3.

        Parameters
        ----------
        spectral_oversample : int, optional
            Wavelength sub-bin count for cube assembly (only used by
            ``spectral_method='oversample'``). When ``None`` (default),
            reads ``obs.spectral_oversample`` (which itself reads from
            ``obs.render_config.spectral_oversample``, default 15). Pass
            an explicit value only to override the obs-recorded setting
            (e.g., convergence tests).
        spectral_method : str, optional
            Spectral bin-integration method for ``build_cube``: ``'erf'``
            (exact, default) or ``'oversample'``. When ``None``
            (default), reads ``obs.spectral_method`` (canonical source
            ``obs.render_config.spectral_method``). Pass an explicit
            value only to override (e.g., method-comparison tests).
        psf_mode : str, optional
            PSF pathway: ``'post_dispersion'`` (single convolution of
            the dispersed image; default) or ``'per_slice'`` (reference
            path). When ``None`` (default), reads ``obs.psf_mode``
            (canonical source ``obs.render_config.psf_mode``). Pass an
            explicit value only to override (e.g., equivalence tests).
        """
        from kl_pipe.dispersion import disperse_cube
        from kl_pipe.grism import _apply_post_dispersion_pixel_response
        from kl_pipe.spectral import CubePars

        # resolve spectral_oversample / spectral_method / psf_mode:
        # explicit kwarg wins, else read from obs
        if spectral_oversample is None:
            spectral_oversample = obs.spectral_oversample
        if spectral_method is None:
            spectral_method = obs.spectral_method
        if psf_mode is None:
            psf_mode = obs.psf_mode
        if psf_mode not in ('post_dispersion', 'per_slice'):
            raise ValueError(
                f"psf_mode must be 'post_dispersion' or 'per_slice', got "
                f"{psf_mode!r}"
            )

        if obs.dispersal_method == 'analytic':
            return self._render_grism_analytic(
                pars, obs, plane=plane, psf_mode=psf_mode
            )

        # build_cube spatial grid: fine when oversampling is active
        if obs.psf_data is not None and obs.oversample > 1:
            build_cube_pars = CubePars(
                image_pars=obs.fine_image_pars,
                lambda_grid=obs.cube_pars.lambda_grid,
            )
        else:
            build_cube_pars = obs.cube_pars

        image_rotation = image_rotation_from_wcs(obs.grism_pars.image_pars.wcs)
        # realistic continuum is dispersed separately (below) so it can fill
        # the stamp; keep it out of the line cube in that mode
        cont_fills = obs.continuum_fills_stamp
        cube = self.build_cube(
            pars,
            build_cube_pars,
            spectral_oversample=spectral_oversample,
            plane=plane,
            image_rotation=image_rotation,
            spectral_method=spectral_method,
            include_continuum=not cont_fills,
        )

        # per-slice PSF convolution (vmap over wavelength), bin=False keeps
        # fine. The general/reference path; required for future
        # wavelength-dependent PSFs (issue #51).
        if obs.psf_data is not None and psf_mode == 'per_slice':
            from kl_pipe.psf import convolve_fft

            cube_transposed = jnp.moveaxis(cube, -1, 0)
            cube_transposed = jax.vmap(
                lambda s: convolve_fft(s, obs.psf_data, bin=False)
            )(cube_transposed)
            cube = jnp.moveaxis(cube_transposed, 0, -1)

        # disperse to 2D (fine if oversample > 1)
        dispersed = disperse_cube(
            cube,
            obs.grism_pars,
            obs.cube_pars.lambda_grid,
            oversample=obs.oversample,
        )

        # stamp-filling continuum dispersed by the closed-form trace kernel
        # (no interior box edges); the legacy narrow-window continuum instead
        # rides inside the cube above
        cont_disp = None
        if cont_fills:
            cont_disp = self._render_continuum_dispersed(
                pars,
                obs.grism_pars,
                plane,
                image_rotation,
                build_cube_pars.image_pars,
                obs.oversample,
                obs.cube_pars.lambda_grid,
            )

        # post-dispersion PSF convolution: one exact padded linear
        # convolution of the dispersed image replaces the 2*Nlambda
        # per-slice FFTs (a shared PSF commutes with per-slice uniform
        # shifts and their weighted sum). The unit-sum kernel preserves
        # SB units.
        if psf_mode == 'post_dispersion':
            # the continuum shares the line's single post-dispersion pass
            if cont_disp is not None:
                dispersed = dispersed + cont_disp
            if obs.psf_data is not None:
                from kl_pipe.psf import convolve_fft

                dispersed = convolve_fft(dispersed, obs.psf_data, bin=False)
        else:
            # per_slice: the line cube is already convolved; convolve the
            # separately-dispersed continuum with the same shared PSF to
            # match, then add
            if cont_disp is not None:
                if obs.psf_data is not None:
                    from kl_pipe.psf import convolve_fft

                    cont_disp = convolve_fft(cont_disp, obs.psf_data, bin=False)
                dispersed = dispersed + cont_disp

        # post-dispersion BoxPixel sinc + SB→flux/pixel conversion
        coarse_shape = (
            obs.grism_pars.image_pars.Nrow,
            obs.grism_pars.image_pars.Ncol,
        )
        return _apply_post_dispersion_pixel_response(
            dispersed,
            obs.pixel_response_fft,
            coarse_shape,
            obs.oversample,
            obs.grism_pars.image_pars.pixel_scale,
        )

    def render_fiber(
        self,
        pars: dict,
        obs: 'FiberObs',
        plane: str = 'obs',
        spectral_oversample: int | None = None,
    ) -> jnp.ndarray:
        """
        Render the fiber spectra.
        Parameters
        ----------
        spectral_oversample : int, optional
            Wavelength sub-bin count for cube assembly. When ``None``
            (default), reads ``obs.spectral_oversample`` (which itself
            reads from ``obs.render_config.spectral_oversample``,
            default 5). Pass an explicit value only to override the
            obs-recorded setting (e.g., convergence tests).
        """
        # resolve spectral_oversample: explicit kwarg wins, else read from obs
        if spectral_oversample is None:
            spectral_oversample = obs.render_config.spectral_oversample

        from kl_pipe.spectral import CubePars
        # build_cube spatial grid: fine when oversampling is active
        if obs.psf_data is not None and obs.render_config.oversample > 1:
            build_cube_pars = CubePars(
                image_pars=obs.fine_image_pars,
                lambda_grid=obs.cube_pars.lambda_grid,
            )
        else:
            build_cube_pars = obs.cube_pars

        image_rotation = image_rotation_from_wcs(obs.fiber_pars.image_pars.wcs)

        cube = self.build_cube(
            pars,
            build_cube_pars,
            spectral_oversample=spectral_oversample,
            plane=plane,
            image_rotation=image_rotation,
        )

        #plt.imshow(cube[:,:,0])

        cube_pixel_scale = build_cube_pars.image_pars.pixel_scale
        spec_1D = jnp.sum(
                (obs.ATMPSF_conv_fiber_mask[:, :, jnp.newaxis] * cube),
                axis=(0, 1))* cube_pixel_scale**2

        spec_1D = spec_1D *  obs.throughput

        # fiber PSF can result in degradation in spectral resolution
        if obs.resolution_matrix is not None:
            spec_1D = jnp.dot(obs.resolution_matrix, spec_1D)

        return spec_1D  

    #wip
    def render_fiber_group(
        self,
        pars: dict,
        obs_group: dict,
        #fiber_obs_group: list, #take a dictionary instead?
        plane: str = 'obs',
        spectral_oversample: int | None = None,
    ) -> jnp.ndarray:

        #first_obs = fiber_obs_group[0]
        first = next(iter(obs_group.values()))
        
        # resolve spectral_oversample: explicit kwarg wins, else read from obs
        # all of the FiberObs in the list share the same spectral_oversample, so just use the first one
        if spectral_oversample is None:
            spectral_oversample = first.render_config.spectral_oversample

        from kl_pipe.spectral import CubePars

        if len(obs_group) == 1:
            (key,) = obs_group
            return {
                key: self.render_fiber(
                    pars,
                    obs_group[key],
                    plane=plane,
                    spectral_oversample=spectral_oversample,
                )
            }

        image_rotation = image_rotation_from_wcs(first.fiber_pars.image_pars.wcs)
        # build_cube spatial grid: fine when oversampling is active
        if first.psf_data is not None and first.render_config.oversample > 1:
            build_cube_pars = CubePars(
                image_pars=first.fine_image_pars,
                lambda_grid=first.cube_pars.lambda_grid,
            )
        else:
            build_cube_pars = first.cube_pars

        cube = self.build_cube(
            pars,
            build_cube_pars,
            spectral_oversample=spectral_oversample,
            plane=plane,
            image_rotation=image_rotation,
        )

        #plt.imshow(cube[:,:,0])

        #coarse_pixel_scale = obs.fiber_pars.image_pars.pixel_scale
        cube_pixel_scale = build_cube_pars.image_pars.pixel_scale

        out = {}
        for key, obs in obs_group.items():
            #mask = obs.ATMPSF_conv_fiber_mask
            #throughput = obs.throughput
            #resolution_matrix = obs.resolution_matrix
            spec_1D = jnp.sum(
                            (obs.ATMPSF_conv_fiber_mask[:, :, jnp.newaxis] * cube),
                            axis=(0, 1))* cube_pixel_scale**2
            spec_1D = spec_1D *  obs.throughput
            if obs.resolution_matrix is not None:
                spec_1D = jnp.dot(obs.resolution_matrix, spec_1D)
            out[key] = spec_1D

        return out

        #each fiber observation can have a different mask with a different PSF
        #masks = jnp.stack([obs.ATMPSF_conv_fiber_mask for obs in obs_group])

        #separate, but are probably the same for every fiber in the group
        #throughputs = jnp.stack([obs.throughput for obs in obs_group])  # (n_obs, n_lambda)
        #resolution_matrices = jnp.stack([obs.resolution_matrix for obs in obs_group])

        #spectra = jnp.einsum(
            #"oij,ijk->ok",
            #masks,
            #cube,
        #) * cube_pixel_scale**2 

        #spectra = spectra * throughputs

        # Apply each observation's resolution matrix to its spectrum
        #spectra = jnp.einsum(
            #"oij,oj->oi",
            #resolution_matrices,
            #spectra,
        #)

        #return spectra  

    def _render_continuum_dispersed(
        self,
        pars: dict,
        grism_pars: 'GrismPars',
        plane: str,
        image_rotation: float,
        fine_image_pars: 'ImagePars',
        oversample: int,
        lam_grid: jnp.ndarray,
    ):
        """Disperse the flat stellar continuum to fill the stamp (closed form).

        The continuum is flat in wavelength, so its dispersed trace is a
        throughput-weighted box convolution of the continuum spatial
        profile along the dispersion axis. A real slitless-grism continuum
        trace spans the whole bandpass (hundreds of pixels) while the stamp
        is a tiny window on it, so within the stamp the continuum is a
        smooth stripe with no interior edges. This reproduces that by
        integrating the trace over a window wide enough to cover the stamp
        (``+/- (Ncol + 2)`` coarse pixels about ``lambda_ref``) rather than
        the narrow emission-line velocity window. Dispersion is axis-
        aligned (detector frame); roll is handled by ``image_rotation``
        rotating the spatial profile into the detector frame, matching the
        line term. Returns the fine 2D continuum image (pre-PSF, pre-pixel-
        response), or None when no line carries a continuum.

        See ``RenderConfig.continuum_fills_stamp``.
        """
        from kl_pipe.dispersion import (
            continuum_trace_kernel,
            disperse_continuum_analytic,
        )
        from kl_pipe.utils import build_map_grid_from_image_pars

        X, Y = build_map_grid_from_image_pars(fine_image_pars)
        I_cont_total = None
        unit_cache: dict = {}
        for line_key, line in self.emission_lines.items():
            cont_theta, cont_model = self._build_emission_continuum_theta(
                pars, line_key
            )
            if cont_model is None:
                continue
            cont_theta = _apply_obs_rotation(
                cont_theta, cont_model.PARAMETER_NAMES, image_rotation
            )
            cont_owner = (
                line.continuum_key if line.continuum_key is not None else line_key
            )
            amp_idx = cont_model.PARAMETER_NAMES.index(cont_model.amplitude_param)
            amplitude = cont_theta[amp_idx]
            cache_key = f'cont:{cont_owner}'
            if cache_key not in unit_cache:
                unit_cache[cache_key] = cont_model(
                    cont_theta.at[amp_idx].set(1.0), plane, X, Y
                )
            I_cont = amplitude * unit_cache[cache_key]
            I_cont_total = I_cont if I_cont_total is None else I_cont_total + I_cont

        if I_cont_total is None:
            return None

        # widen the trace integration window so the continuum fills the
        # stamp along the dispersion axis (no interior box edges)
        half_nm = (grism_pars.image_pars.Ncol + 2) * grism_pars.dispersion
        window = (grism_pars.lambda_ref - half_nm, grism_pars.lambda_ref + half_nm)
        kernel, m_lo = continuum_trace_kernel(
            grism_pars, lam_grid, oversample, integration_window=window
        )
        return disperse_continuum_analytic(I_cont_total, kernel, m_lo)

    def _render_grism_analytic(
        self,
        pars: dict,
        obs,
        plane: str = 'obs',
        psf_mode: str = 'post_dispersion',
    ) -> jnp.ndarray:
        """Render a grism image via closed-form per-spaxel dispersal.

        Each emission-line spaxel's dispersed footprint is evaluated analytically (a
        Gaussian convolved with the bilinear tent along the dispersion
        axis, the exact wavelength-continuum limit of the slice method);
        the flat continuum disperses through a precomputed exact trace
        kernel. No wavelength grid enters the line calculation, so
        accuracy is independent of ``n_lambda``. Shares the
        post-dispersion PSF and pixel-response stages with the slice
        path.

        Roll rotation (a nonzero WCS ``image_rotation``) is handled the
        same way the per-obs slice path handles it: model parameters are
        rotated into the detector frame before evaluation, and the
        detector-frame dispersal itself stays axis-aligned. Only the
        shared-cube group fast path (one celestial cube reused across
        rolls, rotation fused into the dispersal sampling) is not
        supported -- there each roll's dispersion direction crosses the
        shared celestial-frame pixel grid diagonally, and that closed
        form (piecewise moments of the interpolation kernel along the
        trace) is not implemented.

        Restrictions (loud): requires ``psf_mode='post_dispersion'`` and
        axis-aligned dispersion (``dispersion_angle_detector == 0``, true
        for Roman).
        """
        from kl_pipe.dispersion import (
            continuum_trace_kernel,
            disperse_continuum_analytic,
            disperse_line_analytic,
        )
        from kl_pipe.grism import _apply_post_dispersion_pixel_response
        from kl_pipe.utils import build_map_grid_from_image_pars

        gp = obs.grism_pars
        if psf_mode != 'post_dispersion':
            raise ValueError(
                "dispersal_method='analytic' requires "
                f"psf_mode='post_dispersion', got {psf_mode!r} (per-slice "
                "PSF needs a wavelength-slice cube)"
            )
        if float(gp.dispersion_angle_detector) != 0.0:
            raise NotImplementedError(
                "dispersal_method='analytic' supports only axis-aligned "
                f"dispersion; got dispersion_angle_detector="
                f"{gp.dispersion_angle_detector}"
            )
        image_rotation = image_rotation_from_wcs(gp.image_pars.wcs)
        if self.velocity_model is None:
            raise ValueError(
                "analytic grism dispersal requires velocity_model to be set"
            )
        if not self.emission_lines:
            raise ValueError(
                "analytic grism dispersal requires non-empty emission_lines"
            )

        if obs.psf_data is not None and obs.oversample > 1:
            fine_ip = obs.fine_image_pars
        else:
            fine_ip = gp.image_pars
        X, Y = build_map_grid_from_image_pars(fine_ip)

        vm = self.velocity_model
        theta_vel = _build_component_theta(pars, 'vel', vm.PARAMETER_NAMES)
        theta_vel = _apply_obs_rotation(theta_vel, vm.PARAMETER_NAMES, image_rotation)
        v_los = vm(theta_vel, plane, X, Y)

        z = pars['z']
        os_f = obs.oversample
        lam_grid = obs.cube_pars.lambda_grid
        throughput = gp.throughput

        # evaluate each spatial owner once at unit amplitude (same dedupe
        # as build_cube; spatial LOS quadrature dominates eval cost)
        unit_eval_cache: dict = {}

        def _amplitude_scaled_eval(owner_key, theta_c, model, amp_name):
            amp_idx = model.PARAMETER_NAMES.index(amp_name)
            amplitude = theta_c[amp_idx]
            if owner_key not in unit_eval_cache:
                theta_unit = theta_c.at[amp_idx].set(1.0)
                unit_eval_cache[owner_key] = model(theta_unit, plane, X, Y)
            return amplitude * unit_eval_cache[owner_key]

        dispersed = jnp.zeros(X.shape)
        cont_fills = obs.continuum_fills_stamp
        I_cont_total = None
        for line_key, line in self.emission_lines.items():
            theta_int, int_model = self._build_emission_intensity_theta(pars, line_key)
            theta_int = _apply_obs_rotation(
                theta_int, int_model.PARAMETER_NAMES, image_rotation
            )
            int_owner = (
                line.intensity_key if line.intensity_key is not None else line_key
            )
            I_line = _amplitude_scaled_eval(
                f'int:{int_owner}', theta_int, int_model, int_model.amplitude_param
            )

            lam_obs = line.lambda_rest * (1.0 + z) * (1.0 + v_los / _C_KMS)
            disp_owner = (
                line.dispersion_key if line.dispersion_key is not None else line_key
            )
            sigma_kms = pars[f'{disp_owner}.dispersion']
            sigma_s = lam_obs * sigma_kms / _C_KMS / gp.dispersion * os_f
            xi = (lam_obs - gp.lambda_ref) / gp.dispersion * os_f

            halfwidth = obs.line_window_halfwidth
            if halfwidth is None:
                # standalone sizing from the concrete parameter values;
                # jitted/inference use must freeze it in RenderConfig
                try:
                    halfwidth = (
                        int(float(jnp.max(jnp.abs(xi))) + 4.0 * float(jnp.max(sigma_s)))
                        + 3
                    )
                except jax.errors.ConcretizationTypeError as err:
                    raise ValueError(
                        "line_window_halfwidth must be set on RenderConfig for "
                        "jitted/inference use of dispersal_method='analytic' "
                        "(the standalone default sizes the flux-spreading window "
                        "from concrete parameter values, which is impossible "
                        "under tracing)"
                    ) from err

            weight = (
                jnp.interp(lam_obs, lam_grid, throughput)
                if throughput is not None
                else None
            )
            dispersed = dispersed + disperse_line_analytic(
                I_line, xi, sigma_s, halfwidth, weight=weight
            )

            # legacy line-window continuum accumulation; the realistic
            # stamp-filling continuum is dispersed after the loop
            cont_theta, cont_model = self._build_emission_continuum_theta(
                pars, line_key
            )
            if not cont_fills and cont_model is not None:
                cont_theta = _apply_obs_rotation(
                    cont_theta, cont_model.PARAMETER_NAMES, image_rotation
                )
                cont_owner = (
                    line.continuum_key if line.continuum_key is not None else line_key
                )
                I_cont = _amplitude_scaled_eval(
                    f'cont:{cont_owner}',
                    cont_theta,
                    cont_model,
                    cont_model.amplitude_param,
                )
                I_cont_total = I_cont if I_cont_total is None else I_cont_total + I_cont

        if cont_fills:
            cont_disp = self._render_continuum_dispersed(
                pars, gp, plane, image_rotation, fine_ip, os_f, lam_grid
            )
            if cont_disp is not None:
                dispersed = dispersed + cont_disp
        elif I_cont_total is not None:
            kernel, m_lo = continuum_trace_kernel(gp, lam_grid, os_f)
            dispersed = dispersed + disperse_continuum_analytic(
                I_cont_total, kernel, m_lo
            )

        if obs.psf_data is not None:
            from kl_pipe.psf import convolve_fft

            dispersed = convolve_fft(dispersed, obs.psf_data, bin=False)

        coarse_shape = (gp.image_pars.Nrow, gp.image_pars.Ncol)
        return _apply_post_dispersion_pixel_response(
            dispersed,
            obs.pixel_response_fft,
            coarse_shape,
            obs.oversample,
            gp.image_pars.pixel_scale,
        )

    def render_grism_group(
        self,
        pars: dict,
        obs_group: dict,
        plane: str = 'obs',
        spectral_oversample: int | None = None,
        spectral_method: str | None = None,
        psf_mode: str | None = None,
        operators: dict | None = None,
    ) -> dict:
        """Render a group of cube-compatible grism observations from ONE
        shared cube.

        The unconvolved cube is roll-independent physics: LOS velocity is a
        scalar field on the sky and all model parameters are celestial-
        frame. The cube is built in the FIRST obs's detector frame (the
        anchor roll -- exact parameter-level rotation, and that roll needs
        no rotational resampling); every obs is then dispersed by a
        precomputed sparse operator that fuses dispersion, the relative
        roll rotation, and Catmull-Rom cubic interpolation into a single
        matvec (``dispersion.precompute_dispersion_operator``). Bilinear
        resampling is deliberately not used here: its sub-pixel smoothing
        biases inclination-like posterior modes at the 0.35-sigma level in
        tight-posterior tests, while cubic measures at the per-roll
        accuracy floor (Fisher-projected shift 0.005 sigma). The
        detector-frame PSF + pixel response then apply per obs. PSFs MAY
        differ across the group (per-detector-position kernels are free);
        wavelength grids, spatial grids, oversample, and spectral method
        must be identical (see
        ``observation.group_grism_obs_by_cube_compat``).

        Parameters
        ----------
        pars : dict
            Celestial-frame parameter dict (dotted SourceModel keys).
        obs_group : dict[str, GrismObs]
            Cube-compatible observations keyed by roll name; the FIRST
            entry defines the anchor frame. Multi-obs groups require
            ``psf_mode='post_dispersion'`` (the shared cube is unconvolved)
            and pure-rotation WCSs; violations raise via
            ``observation.validate_shared_cube_group``.
        plane, spectral_oversample, spectral_method, psf_mode
            As in ``render_grism``; resolved from the first obs when None
            (the group is validated homogeneous in these).
        operators : dict[str, BCOO], optional
            Precomputed dispersion operators keyed like ``obs_group``
            (from ``observation.build_group_dispersion_operators``). When
            None they are built here -- fine for one-off rendering; for
            inference the likelihood factory builds them once.

        Returns
        -------
        dict[str, jnp.ndarray]
            Dispersed detector-frame images keyed like ``obs_group``.
        """
        # a rotated-frame analytic closed form exists on paper but was
        # deliberately not implemented: on CPU, independent per-obs renders
        # measured faster than any shared-evaluation variant could be.
        # Worth revisiting only if GPU profiling changes that trade-off.
        for key, obs in obs_group.items():
            if obs.dispersal_method == 'analytic':
                raise NotImplementedError(
                    f"dispersal_method='analytic' (obs {key!r}) is not "
                    "supported on the shared-cube group path (the analytic "
                    "path builds no cube to share). Inference routes analytic "
                    "obs to independent per-obs renders automatically; render "
                    "each obs with render_grism, or use "
                    "dispersal_method='slice' for group rendering."
                )
        from kl_pipe.dispersion import apply_dispersion_operator
        from kl_pipe.grism import _apply_post_dispersion_pixel_response
        from kl_pipe.observation import (
            build_group_dispersion_operators,
            group_grism_obs_by_cube_compat,
            validate_shared_cube_group,
        )
        from kl_pipe.spectral import CubePars

        if not obs_group:
            raise ValueError("render_grism_group requires a non-empty obs_group")

        groups = group_grism_obs_by_cube_compat(obs_group)
        if len(groups) > 1:
            raise ValueError(
                f"obs_group is not cube-compatible: it splits into "
                f"{[list(g) for g in groups]}. All obs must share identical "
                f"wavelength grid, spatial grid, oversample, and spectral "
                f"method; render incompatible groups separately."
            )

        first = next(iter(obs_group.values()))
        if spectral_oversample is None:
            spectral_oversample = first.spectral_oversample
        if spectral_method is None:
            spectral_method = first.spectral_method
        if psf_mode is None:
            psf_mode = first.psf_mode
        if psf_mode not in ('post_dispersion', 'per_slice'):
            raise ValueError(
                f"psf_mode must be 'post_dispersion' or 'per_slice', got "
                f"{psf_mode!r}"
            )
        validate_shared_cube_group(obs_group, psf_mode)

        # single-obs group: the per-obs path is identical and supports the
        # per_slice reference mode too
        if len(obs_group) == 1:
            (key,) = obs_group
            return {
                key: self.render_grism(
                    pars,
                    obs_group[key],
                    plane=plane,
                    spectral_oversample=spectral_oversample,
                    spectral_method=spectral_method,
                    psf_mode=psf_mode,
                )
            }

        # shared cube built ONCE in the anchor (first) obs's detector
        # frame: exact parameter-level rotation, and the anchor roll needs
        # no rotational resampling. Fine spatial grid when PSF +
        # oversampling are active (validated homogeneous across the group).
        anchor_rotation = image_rotation_from_wcs(first.grism_pars.image_pars.wcs)
        if first.psf_data is not None and first.oversample > 1:
            build_cube_pars = CubePars(
                image_pars=first.fine_image_pars,
                lambda_grid=first.cube_pars.lambda_grid,
            )
        else:
            build_cube_pars = first.cube_pars

        # realistic continuum fills the stamp and is dispersed per obs
        # (axis-aligned, rotated by that obs's roll) rather than shared
        # through the operator; keep it out of the shared line cube then.
        # The group is homogeneous in render settings (resolved from first).
        cont_fills = first.continuum_fills_stamp
        cube = self.build_cube(
            pars,
            build_cube_pars,
            spectral_oversample=spectral_oversample,
            plane=plane,
            image_rotation=anchor_rotation,
            spectral_method=spectral_method,
            include_continuum=not cont_fills,
        )

        if operators is None:
            operators = build_group_dispersion_operators(obs_group)

        out = {}
        for key, obs in obs_group.items():
            dispersed = apply_dispersion_operator(operators[key], cube)
            if cont_fills:
                # multi-obs groups are post_dispersion (validated), so the
                # continuum joins the line before the single PSF pass
                obs_rotation = image_rotation_from_wcs(obs.grism_pars.image_pars.wcs)
                cont_disp = self._render_continuum_dispersed(
                    pars,
                    obs.grism_pars,
                    plane,
                    obs_rotation,
                    build_cube_pars.image_pars,
                    obs.oversample,
                    obs.cube_pars.lambda_grid,
                )
                if cont_disp is not None:
                    dispersed = dispersed + cont_disp
            if obs.psf_data is not None:
                from kl_pipe.psf import convolve_fft

                dispersed = convolve_fft(dispersed, obs.psf_data, bin=False)
            coarse_shape = (
                obs.grism_pars.image_pars.Nrow,
                obs.grism_pars.image_pars.Ncol,
            )
            out[key] = _apply_post_dispersion_pixel_response(
                dispersed,
                obs.pixel_response_fft,
                coarse_shape,
                obs.oversample,
                obs.grism_pars.image_pars.pixel_scale,
            )
        return out

    def render_velocity(
        self,
        pars: dict,
        obs: 'VelocityObs',
    ) -> jnp.ndarray:
        """Render the line-of-sight velocity map.

        Builds the velocity model's flat theta from ``pars`` (with the
        celestial-to-detector rotation applied), evaluates the velocity
        field, and applies PSF flux-weighted convolution when
        ``obs.psf_data`` is set.

        Flux weighting source: ``obs.flux_weight_key`` references a key in
        ``self.emission_lines``. SourceModel computes the spatial flux map
        from that emission line's intensity (resolving ``intensity_key``
        sharing if needed) at the rotated theta, and threads it to the
        velocity render via a temporary obs with ``flux_image`` populated.
        ``flux_weight_key=None`` is allowed when ``obs.psf_data is None``
        (velocity-only inference with no PSF); a PSF without
        ``flux_weight_key`` raises (no flux source).
        """
        from dataclasses import replace

        if self.velocity_model is None:
            raise ValueError(
                "SourceModel.render_velocity requires velocity_model to be set"
            )

        vm = self.velocity_model
        theta_vel = _build_component_theta(pars, 'vel', vm.PARAMETER_NAMES)
        image_rotation = image_rotation_from_wcs(obs.image_pars.wcs)
        theta_vel = _apply_obs_rotation(theta_vel, vm.PARAMETER_NAMES, image_rotation)

        # Flux weighting only matters when PSF is on. If no PSF, just
        # render the velocity map (or unweighted oversample bin) directly.
        if obs.psf_data is None or obs.flux_weight_key is None:
            return vm.render_image(theta_vel, obs=obs)

        # Precompute the flux-weight image at the obs's evaluation grid.
        flux_image = self._compute_flux_weight_image(pars, obs)
        obs_with_flux = replace(
            obs, flux_image=flux_image, flux_model=None, flux_theta=None
        )
        return vm.render_image(theta_vel, obs=obs_with_flux)

    def build_cube(
        self,
        pars: dict,
        cube_pars: 'CubePars',
        spectral_oversample: int = 15,
        plane: str = 'obs',
        image_rotation: float = 0.0,
        spectral_method: str = 'erf',
        include_continuum: bool = True,
    ) -> jnp.ndarray:
        """Build the intrinsic 3D datacube ``C(x, y, lambda)`` (no PSF).

        Thin dispatch wrapper around ``_build_cube_impl``. When
        ``self.cube_remat`` is True (default), the assembly is wrapped in
        ``jax.checkpoint`` with the ``dots_saveable`` policy:
        rematerializing the cube assembly in the backward pass trades a
        cheap forward recompute for less memory traffic; measured ~1.2x
        speedup on posterior gradients on CPU, gradients identical. Only
        ``pars`` is treated as traced input; ``cube_pars`` and the
        remaining arguments are static and closed over.

        See ``_build_cube_impl`` for the full parameter and method
        documentation.
        """
        if not self.cube_remat:
            return self._build_cube_impl(
                pars,
                cube_pars,
                spectral_oversample=spectral_oversample,
                plane=plane,
                image_rotation=image_rotation,
                spectral_method=spectral_method,
                include_continuum=include_continuum,
            )

        def _cube_of_pars(traced_pars: dict) -> jnp.ndarray:
            return self._build_cube_impl(
                traced_pars,
                cube_pars,
                spectral_oversample=spectral_oversample,
                plane=plane,
                image_rotation=image_rotation,
                spectral_method=spectral_method,
                include_continuum=include_continuum,
            )

        return jax.checkpoint(
            _cube_of_pars, policy=jax.checkpoint_policies.dots_saveable
        )(pars)

    def _build_cube_impl(
        self,
        pars: dict,
        cube_pars: 'CubePars',
        spectral_oversample: int = 15,
        plane: str = 'obs',
        image_rotation: float = 0.0,
        spectral_method: str = 'erf',
        include_continuum: bool = True,
    ) -> jnp.ndarray:
        """Build the intrinsic 3D datacube ``C(x, y, lambda)`` (no PSF).

        Iterates over ``self.emission_lines``; for each line resolves
        ``intensity_key`` / ``continuum_key`` / ``dispersion_key`` to the
        owning line's spatial/continuum/dispersion data, applies the
        celestial-to-detector rotation, evaluates the per-line intensity
        and (optional) continuum spatial profiles on the cube spatial
        grid, and accumulates the Gaussian-broadened line +
        flat-near-line continuum contributions into the coarse output
        cube. The spectral integral over each coarse wavelength bin is
        computed by one of two selectable methods (``spectral_method``):

        - ``'erf'`` (default): exact analytic bin integral of the
          per-pixel Gaussian line kernel via the error function
          evaluated at the ``Nlambda + 1`` bin edges. Exact for the
          Gaussian kernel (no spectral discretization error);
          ``spectral_oversample`` is ignored.
        - ``'oversample'``: midpoint sampling on a fine grid of
          ``spectral_oversample`` sub-bins per coarse bin, mean-binned
          to coarse. Midpoint quadrature carries an
          ``O(spectral_oversample**-2)`` systematic error (~5e-4
          relative at the default 15 for Roman-like line widths);
          retained for convergence/comparison studies.

        Parameters
        ----------
        pars : dict
            Dotted-key parameter dict (sampled + fixed merged by caller).
            Must contain top-level ``z``, per-line ``<line>.dispersion``
            (or the referenced line's dispersion via ``dispersion_key``),
            and all per-component intensity / continuum params.
        cube_pars : CubePars
            Spatial grid (``image_pars``) + wavelength array.
        spectral_oversample : int, default 15
            Number of fine wavelength sub-bins per coarse lambda pixel.
            Only consulted when ``spectral_method='oversample'``; set to
            1 for center-point sampling with no oversampling.
        plane : str, default 'obs'
            Coordinate plane for velocity + intensity evaluation.
        image_rotation : float, default 0.0
            Celestial-to-detector rotation (radians) for this obs. Used to
            convert celestial-frame ``theta_int`` / ``(g1, g2)`` priors
            into the detector-frame thetas the model classes expect.
            ``render_grism`` derives this from ``obs.grism_pars.image_pars.wcs``
            via ``image_rotation_from_wcs`` before passing it through.
        spectral_method : str, default 'erf'
            Spectral bin-integration method, ``'erf'`` or
            ``'oversample'`` (see above).

        Returns
        -------
        jnp.ndarray
            Cube of shape ``(Nrow, Ncol, Nlambda)``.
        """
        from kl_pipe.utils import build_map_grid_from_image_pars

        if self.velocity_model is None:
            raise ValueError(
                "SourceModel.build_cube requires velocity_model to be set "
                "(Doppler shifts use the velocity field)"
            )
        if not self.emission_lines:
            raise ValueError("SourceModel.build_cube requires non-empty emission_lines")

        z = pars['z']

        # Full LOS velocity at the cube spatial grid. ``v_map`` already
        # includes the systemic ``v0`` (the velocity model evaluates
        # ``v_los = v0 + v_rotation``); we pass it straight into the Doppler
        # shift below so v0 has its proper physical effect on the observed
        # line wavelength. The degeneracy between v0 and z is real and is
        # handled at the prior level (typical workflow: fix z from an outside
        # estimate and either fix v0=0 or let it absorb residual systemic
        # motion within a sigma_z-aware prior).
        X, Y = build_map_grid_from_image_pars(cube_pars.image_pars)
        vm = self.velocity_model
        theta_vel = _build_component_theta(pars, 'vel', vm.PARAMETER_NAMES)
        theta_vel = _apply_obs_rotation(theta_vel, vm.PARAMETER_NAMES, image_rotation)
        v_los = vm(theta_vel, plane, X, Y)

        if spectral_method not in ('erf', 'oversample'):
            raise ValueError(
                f"spectral_method must be 'erf' or 'oversample', got "
                f"{spectral_method!r}"
            )

        # spectral grids per method
        lambda_coarse = cube_pars.lambda_grid
        n_lam = len(lambda_coarse)
        Nrow, Ncol = cube_pars.spatial_shape

        if spectral_method == 'erf':
            if n_lam < 2:
                raise ValueError(
                    "spectral_method='erf' requires Nlambda >= 2 to define "
                    f"bin edges (got Nlambda={n_lam}); use "
                    "spectral_method='oversample' for single-slice cubes "
                    "(center-point sampling)."
                )
            # coarse bin edges: centers +/- dl/2, shape (n_lam + 1,)
            dl = lambda_coarse[1] - lambda_coarse[0]
            bin_edges = jnp.concatenate(
                [lambda_coarse - dl / 2.0, lambda_coarse[-1:] + dl / 2.0]
            )
            cube_fine = jnp.zeros((Nrow, Ncol, n_lam))
        else:
            # build the fine wavelength grid (midpoint sub-bins)
            osf = spectral_oversample
            if osf > 1 and n_lam >= 2:
                dl = lambda_coarse[1] - lambda_coarse[0]
                half = dl / 2.0
                fine_offsets = jnp.linspace(-half + half / osf, half - half / osf, osf)
                lambda_fine = (lambda_coarse[:, None] + fine_offsets[None, :]).reshape(
                    -1
                )
            else:
                lambda_fine = lambda_coarse
                osf = 1
            cube_fine = jnp.zeros((Nrow, Ncol, len(lambda_fine)))

        # Spatial evals are the dominant forward-model cost (LOS quadrature
        # per pixel), and all intensity models are linear in their amplitude
        # parameter. Lines sharing a spatial owner (intensity_key /
        # continuum_key) differ from the owner only in amplitude (enforced
        # by _build_split_owner_theta), so evaluate each owner's profile
        # once at unit amplitude and scale per line -- exact, and avoids
        # re-running the quadrature per sharing line.
        unit_eval_cache: dict = {}

        def _amplitude_scaled_eval(owner_key, theta_c, model, amp_name):
            amp_idx = model.PARAMETER_NAMES.index(amp_name)
            amplitude = theta_c[amp_idx]
            if owner_key not in unit_eval_cache:
                theta_unit = theta_c.at[amp_idx].set(1.0)
                unit_eval_cache[owner_key] = model(theta_unit, plane, X, Y)
            return amplitude * unit_eval_cache[owner_key]

        # accumulate emission line contributions
        for line_key, line in self.emission_lines.items():
            # spatial intensity (resolving intensity_key)
            theta_int, int_model = self._build_emission_intensity_theta(pars, line_key)
            theta_int = _apply_obs_rotation(
                theta_int, int_model.PARAMETER_NAMES, image_rotation
            )
            int_owner = (
                line.intensity_key if line.intensity_key is not None else line_key
            )
            I_line = _amplitude_scaled_eval(
                f'int:{int_owner}', theta_int, int_model, int_model.amplitude_param
            )

            # Doppler-shifted observed wavelength per pixel (full LOS v_los
            # includes systemic v0 + rotation contribution).
            lam_obs = line.lambda_rest * (1.0 + z) * (1.0 + v_los / _C_KMS)

            # intrinsic kinematic dispersion (resolving dispersion_key)
            disp_owner = (
                line.dispersion_key if line.dispersion_key is not None else line_key
            )
            sigma_kms = pars[f'{disp_owner}.dispersion']
            sigma_lambda = lam_obs * sigma_kms / _C_KMS  # (Nrow, Ncol)

            if spectral_method == 'erf':
                # exact bin-averaged Gaussian kernel [1/nm]: the Gaussian
                # CDF differenced at the coarse bin edges, divided by the
                # bin width. Matches the oversample path's mean-of-samples
                # convention exactly in the osf -> inf limit.
                u = (bin_edges[None, None, :] - lam_obs[:, :, None]) / (
                    sigma_lambda[:, :, None] * jnp.sqrt(2.0)
                )
                cdf = 0.5 * (1.0 + jax.scipy.special.erf(u))
                kernel = (cdf[:, :, 1:] - cdf[:, :, :-1]) / dl
            else:
                # midpoint-sampled normalized Gaussian, shape
                # (Nrow, Ncol, n_fine)
                dlam = lambda_fine[None, None, :] - lam_obs[:, :, None]
                sig = sigma_lambda[:, :, None]
                kernel = (1.0 / (sig * jnp.sqrt(2.0 * jnp.pi))) * jnp.exp(
                    -0.5 * (dlam / sig) ** 2
                )
            cube_fine = cube_fine + I_line[:, :, None] * kernel

            # optional stellar continuum near this line (flat in wavelength).
            # skipped when the caller disperses the continuum separately to
            # fill the stamp (see RenderConfig.continuum_fills_stamp)
            cont_theta, cont_model = self._build_emission_continuum_theta(
                pars, line_key
            )
            if include_continuum and cont_model is not None:
                cont_theta = _apply_obs_rotation(
                    cont_theta, cont_model.PARAMETER_NAMES, image_rotation
                )
                # continuum amplitude is 'flux_per_nm', a spectral density
                # [flux/nm]. The profile is linear in its amplitude, so I_cont
                # is already a density surface brightness [flux/arcsec^2/nm] =
                # SB/nm -- matching the line term's SB/nm voxel. No extra
                # per-nm factor needed (that is carried by the parameter's
                # units, enforced by the flux_per_nm name + the guard below).
                cont_owner = (
                    line.continuum_key if line.continuum_key is not None else line_key
                )
                I_cont = _amplitude_scaled_eval(
                    f'cont:{cont_owner}',
                    cont_theta,
                    cont_model,
                    cont_model.amplitude_param,
                )
                cube_fine = cube_fine + I_cont[:, :, None]

        # bin fine to coarse (oversample path only; erf accumulates coarse)
        if spectral_method == 'oversample' and osf > 1 and n_lam >= 2:
            cube = cube_fine.reshape(Nrow, Ncol, n_lam, osf).mean(axis=-1)
        else:
            cube = cube_fine
        return cube

    # ----------------------------------------------------------------- internal

    def _build_emission_intensity_theta(
        self,
        pars: dict,
        line_key: str,
    ) -> Tuple[jnp.ndarray, 'IntensityModel']:
        """Build the flat theta + return the intensity model for one
        emission line, resolving ``intensity_key`` sharing.

        When ``intensity_key`` is set, the spatial profile is owned by the
        referenced line; the only per-line value is ``flux``. All other
        spatial params (``rscale``, ``h_over_r``, ``x0``, ``y0``, ...) are
        read from the spatial-owner namespace.
        """
        if line_key not in self.emission_lines:
            raise KeyError(
                f"emission line '{line_key}' not in emission_lines "
                f"(have: {sorted(self.emission_lines)})"
            )
        line = self.emission_lines[line_key]
        if line.intensity_key is not None:
            spatial_owner = line.intensity_key
            intensity_model = self.emission_lines[spatial_owner].intensity
            theta = self._build_split_owner_theta(
                pars, intensity_model, line_key, spatial_owner
            )
            return theta, intensity_model

        intensity_model = line.intensity
        theta = _build_component_theta(pars, line_key, intensity_model.PARAMETER_NAMES)
        return theta, intensity_model

    def _build_emission_continuum_theta(
        self,
        pars: dict,
        line_key: str,
    ) -> Tuple[Optional[jnp.ndarray], Optional['IntensityModel']]:
        """Build the flat theta + return the continuum intensity model for
        one emission line, resolving ``continuum_key`` sharing.

        Returns ``(None, None)`` if this line has no continuum.

        When ``continuum_key`` is set, the spatial profile is owned by the
        referenced line's continuum; only this line's ``cont.flux_per_nm`` is
        per-line. Other continuum spatial params (``rscale``, ``x0``, ...)
        come from the spatial-owner's ``<owner>.cont.*`` namespace.
        """
        line = self.emission_lines[line_key]
        if line.continuum is None and line.continuum_key is None:
            return None, None
        # guard: the continuum amplitude is a spectral density named
        # 'flux_per_nm' [flux/nm], not an integrated 'flux'. Reject the wrong
        # name loudly rather than silently dropping the amplitude (the
        # flux_per_nm lookup would otherwise KeyError with a less clear message,
        # or a stray '.cont.flux' key would be silently ignored).
        forbidden = f'{line_key}.cont.flux'
        if forbidden in pars:
            raise ValueError(
                f"continuum amplitude is a spectral density: use "
                f"'{line_key}.cont.flux_per_nm' [flux/nm], not '{forbidden}'. A "
                f"stellar continuum has no single integrated 'flux' (see "
                f"docs/units_and_conventions.md)."
            )
        if line.continuum_key is not None:
            spatial_owner = line.continuum_key
            cont_model = self.emission_lines[spatial_owner].continuum
            theta = self._build_split_owner_theta(
                pars,
                cont_model,
                own_prefix=f'{line_key}.cont',
                spatial_owner_prefix=f'{spatial_owner}.cont',
            )
            return theta, cont_model

        cont_model = line.continuum
        theta = _build_component_theta(
            pars, f'{line_key}.cont', cont_model.PARAMETER_NAMES
        )
        return theta, cont_model

    @staticmethod
    def _build_split_owner_theta(
        pars: dict,
        model: 'IntensityModel',
        own_prefix: str,
        spatial_owner_prefix: str,
    ) -> jnp.ndarray:
        """Build a flat theta where ``flux`` comes from ``own_prefix`` and
        all other spatial params come from ``spatial_owner_prefix``.

        Used to resolve ``intensity_key`` / ``continuum_key`` sharing:
        the spatial profile is owned by the referenced component, the
        flux is always per-line.
        """
        values = []
        for p in model.PARAMETER_NAMES:
            # amplitude is per-line ('flux', 'flux_per_nm', or a composite's
            # 'total_flux'); all other (shape) params come from the spatial
            # owner. Resolving the amplitude name from the model avoids
            # silently reading the OWNER's amplitude (a latent bug: matching a
            # hardcoded {'flux','flux_per_nm'} set missed composite continua).
            if p == model.amplitude_param:
                values.append(_lookup_param(pars, own_prefix, p))
            else:
                values.append(_lookup_param(pars, spatial_owner_prefix, p))
        return jnp.asarray(values)

    def _compute_flux_weight_image(
        self,
        pars: dict,
        obs: 'VelocityObs',
    ) -> jnp.ndarray:
        """Evaluate the flux-weight emission line's intensity profile on
        the obs's evaluation grid. Uses the fine grid when ``oversample>1``,
        matching the grid that ``VelocityModel.render_image`` evaluates on.
        """
        theta_int, intensity_model = self._build_emission_intensity_theta(
            pars, obs.flux_weight_key
        )
        image_rotation = image_rotation_from_wcs(obs.image_pars.wcs)
        theta_int = _apply_obs_rotation(
            theta_int, intensity_model.PARAMETER_NAMES, image_rotation
        )
        if obs.fine_X is not None and obs.fine_Y is not None:
            X, Y = obs.fine_X, obs.fine_Y
        else:
            X, Y = obs.X, obs.Y
        return intensity_model(theta_int, 'obs', X, Y)

    def __post_init__(self):
        # require at least one component
        if (
            self.velocity_model is None
            and not self.broadband_models
            and not self.emission_lines
        ):
            raise ValueError(
                "SourceModel requires at least one of velocity_model, "
                "broadband_models, or emission_lines to be non-empty"
            )

        # disjoint broadband / emission keys
        overlap = set(self.broadband_models) & set(self.emission_lines)
        if overlap:
            raise ValueError(
                f"broadband_models and emission_lines keys must be disjoint; "
                f"collision: {sorted(overlap)}"
            )

        # validate emission line *_key cross-references + auto-resolve lambda_rest
        line_names = set(self.emission_lines)
        for name, line in self.emission_lines.items():
            # intensity_key references another line
            if line.intensity_key is not None and line.intensity_key not in line_names:
                raise ValueError(
                    f"emission line '{name}' has intensity_key='{line.intensity_key}' "
                    f"but that line is not in emission_lines"
                )
            # continuum_key references another line that has a continuum
            if line.continuum_key is not None:
                if line.continuum_key not in line_names:
                    raise ValueError(
                        f"emission line '{name}' has continuum_key='{line.continuum_key}' "
                        f"but that line is not in emission_lines"
                    )
                referenced = self.emission_lines[line.continuum_key]
                if referenced.continuum is None and referenced.continuum_key is None:
                    raise ValueError(
                        f"emission line '{name}' has continuum_key='{line.continuum_key}' "
                        f"but that line has no continuum to share"
                    )
            # dispersion_key references another line; no chained sharing
            if line.dispersion_key is not None:
                if line.dispersion_key not in line_names:
                    raise ValueError(
                        f"emission line '{name}' has dispersion_key="
                        f"'{line.dispersion_key}' but that line is not in "
                        f"emission_lines"
                    )
                referenced = self.emission_lines[line.dispersion_key]
                if referenced.dispersion_key is not None:
                    raise ValueError(
                        f"emission line '{name}' has dispersion_key="
                        f"'{line.dispersion_key}', but that line itself "
                        f"references another line via dispersion_key; chained "
                        f"sharing is not allowed"
                    )
            # auto-resolve lambda_rest from LINE_LAMBDAS
            if line.lambda_rest is None:
                if name not in LINE_LAMBDAS:
                    raise ValueError(
                        f"emission line '{name}' has no lambda_rest and is "
                        f"not in LINE_LAMBDAS. Provide lambda_rest=... "
                        f"explicitly or use a registered line name."
                    )
                line.lambda_rest = LINE_LAMBDAS[name]


# ===========================================================================
# JAX pytree registration
# ===========================================================================
#
# SourceModel is treated as an opaque Python object from JAX's perspective:
# no traceable children, the instance itself is the aux. Users pass it via
# ``partial(jit_fn, source=source)``, so JAX never needs to flatten it at
# the leaf level. Registering it as a pytree with this trivial split lets
# ``jax.tree_util.tree_flatten`` succeed and gives users a place to extend
# later if SourceModel ever needs to carry traceable data.


def _source_model_flatten(source):
    # empty children, instance as aux: every configuration field
    # (velocity_model, broadband_models, emission_lines, cube_remat)
    # rides in aux and round-trips unflatten unchanged
    return (), source


def _source_model_unflatten(aux, children):
    return aux


jax.tree_util.register_pytree_node(
    SourceModel, _source_model_flatten, _source_model_unflatten
)
