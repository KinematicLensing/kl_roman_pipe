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

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from kl_pipe.coordinates import image_rotation_from_wcs, rotate_shear  # noqa: E402
from kl_pipe.lines import LINE_LAMBDAS, EmissionLine  # noqa: E402

if TYPE_CHECKING:
    from kl_pipe.model import IntensityModel, VelocityModel
    from kl_pipe.observation import GrismObs, ImageObs, VelocityObs, FiberObs
    from kl_pipe.spectral import CubePars

# speed of light, km/s
_C_KMS = 299792.458

import matplotlib.pyplot as plt
import numpy as np

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
    """Rotate celestial-frame ``theta_int`` + ``(g1, g2)`` into the obs's
    detector frame.

    Sign convention matches ``OrientedAngle._sky2cartesian`` from kl-tools:
    ``theta_int_det = theta_int_celestial - image_rotation``. Shear rotates
    spin-2 by ``2 * image_rotation`` via ``coordinates.rotate_shear``.

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
          line key, and that line must NOT itself reference dispersion
          via its own ``dispersion_key`` (no chained sharing).
        * If ``lambda_rest`` is None, the line's name must be in
          ``LINE_LAMBDAS``; resolved automatically.
    """

    velocity_model: Optional['VelocityModel'] = None
    broadband_models: Dict[str, 'IntensityModel'] = field(default_factory=dict)
    emission_lines: Dict[str, EmissionLine] = field(default_factory=dict)

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
    ) -> jnp.ndarray:
        """Render the dispersed 2D grism image.

        Pipeline:
          1. Build the intrinsic cube via ``build_cube`` (at fine spatial
             scale when ``obs.oversample > 1``), deriving the celestial-
             to-detector rotation from ``obs.grism_pars.image_pars.wcs``
             to thread celestial-frame priors into detector-frame thetas.
          2. Per-slice PSF convolution via ``vmap`` over wavelength, with
             ``bin=False`` so the cube stays at fine resolution.
          3. Disperse via ``disperse_cube`` (existing kl_pipe.dispersion).
          4. Apply the precomputed BoxPixel sinc + sum-bin to coarse
             detector pixels at the 2D output.

        .. note::
           Step 2 applies a **single shared** PSF (``obs.psf_data``) to
           every wavelength slice. Per-line / wavelength-dependent PSFs
           and per-line sub-cubes are tracked as an open architectural
           item — see issue #51. Deferred past Phase 3.

        Parameters
        ----------
        spectral_oversample : int, optional
            Wavelength sub-bin count for cube assembly. When ``None``
            (default), reads ``obs.spectral_oversample`` (which itself
            reads from ``obs.render_config.spectral_oversample``,
            default 5). Pass an explicit value only to override the
            obs-recorded setting (e.g., convergence tests).
        """
        from kl_pipe.dispersion import disperse_cube
        from kl_pipe.grism import _apply_post_dispersion_pixel_response
        from kl_pipe.spectral import CubePars

        # resolve spectral_oversample: explicit kwarg wins, else read from obs
        if spectral_oversample is None:
            spectral_oversample = obs.spectral_oversample

        # build_cube spatial grid: fine when oversampling is active
        if obs.psf_data is not None and obs.oversample > 1:
            build_cube_pars = CubePars(
                image_pars=obs.fine_image_pars,
                lambda_grid=obs.cube_pars.lambda_grid,
            )
        else:
            build_cube_pars = obs.cube_pars

        image_rotation = image_rotation_from_wcs(obs.grism_pars.image_pars.wcs)
        cube = self.build_cube(
            pars,
            build_cube_pars,
            spectral_oversample=spectral_oversample,
            plane=plane,
            image_rotation=image_rotation,
        )

        # per-slice PSF convolution (vmap over wavelength), bin=False keeps fine
        if obs.psf_data is not None:
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

        #needs to be updated to handle oversampling scenario?
        spec_1D = jnp.sum(
                (obs.ATMPSF_conv_fiber_mask[:, :, jnp.newaxis] * cube),
                axis=(0, 1))

        spec_1D = spec_1D *  obs.throughput

        #these things aren't in render_grism so I'm not so sure if they should be included here either.. for now ignore
        #but it's important to account for situations where one fiber is exposed for longer than another, right? well it depends on the units
        #factor = (
            #jnp.pi
            #* (obs.fiber_pars.obs_conf['DIAMETER'] / 2.0) ** 2
            #* obs.fiber_pars.obs_conf['EXPTIME']
            #/ obs.fiber_pars.obs_conf['GAIN']
        #)
        spec_1D = spec_1D #* factor

        # fiber PSF can result in degradation in spectral resolution
        if obs.resolution_matrix is not None:
            spec_1D = jnp.dot(obs.resolution_matrix, spec_1D)

        return spec_1D  

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
    ) -> jnp.ndarray:
        """Build the intrinsic 3D datacube ``C(x, y, lambda)`` (no PSF).

        Iterates over ``self.emission_lines``; for each line resolves
        ``intensity_key`` / ``continuum_key`` / ``dispersion_key`` to the
        owning line's spatial/continuum/dispersion data, applies the
        celestial-to-detector rotation, evaluates the per-line intensity
        and (optional) continuum spatial profiles on the cube spatial
        grid, and accumulates the Gaussian-broadened line + flat-near-line
        continuum contributions onto a wavelength-oversampled fine grid
        before binning to the coarse output cube.

        Parameters
        ----------
        pars : dict
            Dotted-key parameter dict (sampled + fixed merged by caller).
            Must contain top-level ``z``, per-line ``<line>.dispersion``
            (or the referenced line's dispersion via ``dispersion_key``),
            and all per-component intensity / continuum params.
        cube_pars : CubePars
            Spatial grid (``image_pars``) + wavelength array.
        spectral_oversample : int, default 5
            Number of fine wavelength sub-bins per coarse lambda pixel.
            Set to 1 for no spectral oversampling.
        plane : str, default 'obs'
            Coordinate plane for velocity + intensity evaluation.
        image_rotation : float, default 0.0
            Celestial-to-detector rotation (radians) for this obs. Used to
            convert celestial-frame ``theta_int`` / ``(g1, g2)`` priors
            into the detector-frame thetas the model classes expect.
            ``render_grism`` derives this from ``obs.grism_pars.image_pars.wcs``
            via ``image_rotation_from_wcs`` before passing it through.

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

        # build the fine wavelength grid
        lambda_coarse = cube_pars.lambda_grid
        n_lam = len(lambda_coarse)
        osf = spectral_oversample
        if osf > 1 and n_lam >= 2:
            dl = lambda_coarse[1] - lambda_coarse[0]
            half = dl / 2.0
            fine_offsets = jnp.linspace(-half + half / osf, half - half / osf, osf)
            lambda_fine = (lambda_coarse[:, None] + fine_offsets[None, :]).reshape(-1)
        else:
            lambda_fine = lambda_coarse
            osf = 1

        n_fine = len(lambda_fine)
        Nrow, Ncol = cube_pars.spatial_shape
        cube_fine = jnp.zeros((Nrow, Ncol, n_fine))

        # accumulate emission line contributions
        for line_key, line in self.emission_lines.items():
            # spatial intensity (resolving intensity_key)
            theta_int, int_model = self._build_emission_intensity_theta(pars, line_key)
            theta_int = _apply_obs_rotation(
                theta_int, int_model.PARAMETER_NAMES, image_rotation
            )
            I_line = int_model(theta_int, plane, X, Y)

            # Doppler-shifted observed wavelength per pixel (full LOS v_los
            # includes systemic v0 + rotation contribution).
            lam_obs = line.lambda_rest * (1.0 + z) * (1.0 + v_los / _C_KMS)

            # intrinsic kinematic dispersion (resolving dispersion_key)
            disp_owner = (
                line.dispersion_key if line.dispersion_key is not None else line_key
            )
            sigma_kms = pars[f'{disp_owner}.dispersion']
            sigma_lambda = lam_obs * sigma_kms / _C_KMS  # (Nrow, Ncol)

            # normalized Gaussian line kernel, shape (Nrow, Ncol, n_fine)
            dlam = lambda_fine[None, None, :] - lam_obs[:, :, None]
            sig = sigma_lambda[:, :, None]
            gauss = (1.0 / (sig * jnp.sqrt(2.0 * jnp.pi))) * jnp.exp(
                -0.5 * (dlam / sig) ** 2
            )
            cube_fine = cube_fine + I_line[:, :, None] * gauss

            # optional stellar continuum near this line (flat in wavelength)
            cont_theta, cont_model = self._build_emission_continuum_theta(
                pars, line_key
            )
            if cont_model is not None:
                cont_theta = _apply_obs_rotation(
                    cont_theta, cont_model.PARAMETER_NAMES, image_rotation
                )
                I_cont = cont_model(cont_theta, plane, X, Y)
                cube_fine = cube_fine + I_cont[:, :, None]

        # bin fine to coarse
        if osf > 1 and n_lam >= 2:
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
        referenced line's continuum; only this line's ``cont.flux`` is
        per-line. Other continuum spatial params (``rscale``, ``x0``, ...)
        come from the spatial-owner's ``<owner>.cont.*`` namespace.
        """
        line = self.emission_lines[line_key]
        if line.continuum is None and line.continuum_key is None:
            return None, None
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
            if p == 'flux':
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
    return (), source  # empty children, instance as aux


def _source_model_unflatten(aux, children):
    return aux


jax.tree_util.register_pytree_node(
    SourceModel, _source_model_flatten, _source_model_unflatten
)
