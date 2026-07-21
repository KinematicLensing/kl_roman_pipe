import inspect
import jax.numpy as jnp
import jax
import numpy as np

from abc import abstractmethod, ABC
from typing import Any

from kl_pipe.transformation import COSI_FLOOR, transform_to_disk_plane
from kl_pipe.parameters import ImagePars
from kl_pipe.utils import build_map_grid_from_image_pars


class Model(ABC):
    """
    Base class for all models (velocity, intensity, etc.)
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Only enforce PARAMETER_NAMES for concrete classes
        if not inspect.isabstract(cls):
            if not hasattr(cls, 'PARAMETER_NAMES') or cls.PARAMETER_NAMES is None:
                raise TypeError(
                    f"{cls.__name__} must define PARAMETER_NAMES class variable"
                )

        return

    def __init__(self, meta_pars=None) -> None:
        self.meta_pars = meta_pars or {}
        self._param_indices = {name: i for i, name in enumerate(self.PARAMETER_NAMES)}

        return

    def get_param(self, name: str, theta: jnp.ndarray) -> float:
        idx = self._param_indices[name]

        return theta[idx]

    @classmethod
    def theta2pars(cls, theta: jnp.ndarray) -> dict:
        return {name: float(theta[i]) for i, name in enumerate(cls.PARAMETER_NAMES)}

    @classmethod
    def pars2theta(cls, pars: dict) -> jnp.ndarray:
        return jnp.array([pars[name] for name in cls.PARAMETER_NAMES])

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def render(
        self,
        theta: jnp.ndarray,
        data_type: str,
        data_pars: Any,
        plane: str = 'obs',
        **kwargs,
    ) -> jnp.ndarray:
        """
        High-level rendering interface for different data products.

        Parameters
        ----------
        theta : jnp.ndarray
            Parameter array.
        data_type : str
            Type of data product to render. Options: 'image', 'cube', 'slit', 'grism'.
        data_pars : object
            Parameters defining the data product (e.g., ImagePars for 'image').
        plane : str
            Coordinate plane for evaluation. Default is 'obs'.
        **kwargs
            Additional arguments passed to specific render methods.

        Returns
        -------
        jnp.ndarray
            Rendered data product.
        """

        if data_type == 'image':
            if not isinstance(data_pars, ImagePars):
                raise TypeError("data_pars must be ImagePars for data_type='image'")
            from kl_pipe.observation import ImageObs

            obs = ImageObs(
                image_pars=data_pars,
                X=build_map_grid_from_image_pars(data_pars)[0],
                Y=build_map_grid_from_image_pars(data_pars)[1],
            )
            return self.render_image(theta, obs=obs, plane=plane, **kwargs)

        elif data_type == 'cube':
            raise NotImplementedError("Cube rendering not yet implemented")

        elif data_type == 'slit':
            raise NotImplementedError("Slit rendering not yet implemented")

        elif data_type == 'grism':
            raise NotImplementedError("Grism rendering not yet implemented")

        else:
            raise ValueError(
                f"Unknown data_type '{data_type}'. "
                f"Must be one of: 'image', 'cube', 'slit', 'grism'"
            )

    def render_image(
        self,
        theta: jnp.ndarray,
        image_pars: ImagePars = None,
        plane: str = 'obs',
        X: jnp.ndarray = None,
        Y: jnp.ndarray = None,
        *,
        obs: Any = None,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Render model as a 2D image, including observational effects (PSF).

        When obs has oversample > 1, the model is evaluated on a fine-scale
        grid and convolved at that resolution; convolve_fft bins the result
        back to coarse scale automatically.

        Calling conventions:
        - render_image(theta, obs=obs) -- with PSF from obs
        - render_image(theta, image_pars) -- no PSF, builds grids
        - render_image(theta, X=X, Y=Y) -- no PSF, pre-computed grids

        Parameters
        ----------
        theta : jnp.ndarray
            Parameter array.
        image_pars : ImagePars, optional
            Image parameters defining grid, pixel scale, etc.
        plane : str
            Coordinate plane for evaluation. Default is 'obs'.
        X, Y : jnp.ndarray, optional
            Pre-computed coordinate grids (coarse-scale).
        obs : ImageObs, optional
            Observation object with PSF, grids, and oversampling config.
        **kwargs
            Additional model-specific arguments.

        Returns
        -------
        jnp.ndarray
            2D image array (coarse-scale).
        """
        rc = kwargs.pop('render_config', None)

        if obs is not None:
            oversample = rc.oversample if rc is not None else obs.oversample
            if oversample > 1 and obs.fine_X is not None:
                # oversampled: evaluate on fine grid
                model_map = self(theta, plane, obs.fine_X, obs.fine_Y, **kwargs)
                if obs.psf_data is not None:
                    from kl_pipe.psf import convolve_fft

                    return convolve_fft(model_map, obs.psf_data)
                # no PSF but oversampled → mean-bin SB over fine cells, then
                # multiply by coarse pixel area to convert SB-coarse → flux/pixel
                # (sum-over-N² fine cells × fine area = mean × coarse area).
                N = oversample
                Nrow = obs.image_pars.Nrow
                Ncol = obs.image_pars.Ncol
                ps = obs.image_pars.pixel_scale
                sb_coarse = model_map.reshape(Nrow, N, Ncol, N).mean(axis=(1, 3))
                return sb_coarse * (ps * ps)

            model_map = self(theta, plane, obs.X, obs.Y, **kwargs)

            if obs.psf_data is not None:
                from kl_pipe.psf import convolve_fft

                model_map = convolve_fft(model_map, obs.psf_data)

            return model_map

        # legacy/convenience path (no obs)
        if X is None or Y is None:
            if image_pars is None:
                raise ValueError("Provide obs, image_pars, or (X, Y)")
            X, Y = build_map_grid_from_image_pars(image_pars)

        return self(theta, plane, X, Y, **kwargs)

    @abstractmethod
    def __call__(
        self,
        theta: jnp.ndarray,
        plane: str,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        Z: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Evaluate the model at specified coordinates in a given plane.
        """
        raise NotImplementedError("Subclasses must implement __call__ method.")


class VelocityModel(Model):
    """
    Base class for velocity models (vector fields projected to line-of-sight).

    Velocity models require special handling because they represent vector fields
    that must be projected along the line of sight. The projection depends on
    the viewing geometry (inclination and azimuthal angle).
    """

    def __init__(self, meta_pars=None) -> None:
        super().__init__(meta_pars)

        return

    def __call__(
        self,
        theta: jnp.ndarray,
        plane: str,
        x: jnp.ndarray,
        y: jnp.ndarray,
        z: jnp.ndarray = None,
        return_speed: bool = False,
    ) -> jnp.ndarray:
        """
        Evaluate line-of-sight velocity at coordinates in the specified plane.

        The velocity is computed as:
        1. Transform coordinates to disk plane
        2. Evaluate circular velocity (speed) in disk plane
        3. If return_speed=False: Project to line-of-sight based on viewing geometry
        4. Add systemic velocity (only if return_speed=False)

        Parameters
        ----------
        theta : jnp.ndarray
            Parameter array.
        plane : str
            Coordinate plane for input coordinates.
        x, y : jnp.ndarray
            Coordinate arrays.
        z : jnp.ndarray, optional
            Z-coordinate array for 3D evaluation.
        return_speed : bool
            If True, return circular speed (scalar). If False, return line-of-sight
            velocity (projected). Default is False.

        Returns
        -------
        jnp.ndarray
            Velocity map (line-of-sight if return_speed=False, circular speed if True).
        """

        # extract transformation parameters
        g1 = self.get_param('g1', theta)
        g2 = self.get_param('g2', theta)
        theta_int = self.get_param('theta_int', theta)
        cosi = self.get_param('cosi', theta)

        # centroid offsets are not present in all models, so check first
        x0 = self.get_param('x0', theta) if 'x0' in self._param_indices else 0.0
        y0 = self.get_param('y0', theta) if 'y0' in self._param_indices else 0.0

        # transform to disk plane
        x_disk, y_disk = transform_to_disk_plane(
            x, y, plane, x0, y0, g1, g2, theta_int, cosi
        )

        # always evaluate circular velocity (speed) in disk plane first
        v_circ = self.evaluate_circular_velocity(theta, x_disk, y_disk, z)

        # return speed or project to line-of-sight
        if return_speed:
            return v_circ
        else:
            v0 = self.get_param('v0', theta)

            # SPECIAL CASE: In disk plane, we're viewing face-on (no LOS projection)
            if plane == 'disk':
                return jnp.full_like(v_circ, v0)

            # project to line-of-sight velocity
            phi = jnp.arctan2(y_disk, x_disk)
            v_los = jnp.sqrt(1 - jnp.square(cosi)) * jnp.cos(phi) * v_circ

            return v0 + v_los

    @abstractmethod
    def evaluate_circular_velocity(
        self, theta: jnp.ndarray, X: jnp.ndarray, Y: jnp.ndarray, Z: jnp.ndarray = None
    ) -> jnp.ndarray:
        """
        Evaluate circular velocity (speed) in disk plane.

        This is the magnitude of the circular velocity at each point,
        before projection to line-of-sight.

        Parameters
        ----------
        theta : jnp.ndarray
            Parameter array.
        X, Y : jnp.ndarray
            Coordinates in disk plane.
        Z : jnp.ndarray, optional
            Z-coordinates.

        Returns
        -------
        jnp.ndarray
            Circular velocity (speed) at each position.
        """
        raise NotImplementedError(
            "Subclasses must implement evaluate_circular_velocity method."
        )

    def grad_bandwidth(self, params: dict) -> float:
        """Maximum spatial gradient |grad v_LOS| (km/s per arcsec) at ``params``.

        Used by the grism cube fine-grid sizing in
        ``RenderConfig.for_grism_priors`` to bound the spatial bandwidth
        contributed by the velocity-modulated wavelength Gaussian factor in
        ``SourceModel.build_cube``. See ``docs/notes/grism_cube_bandwidth.tex``
        for the derivation.

        Default implementation assumes the arctan rotation curve
        ``v_c(R) = (2/pi) * v_circ * arctan(R / r_v)``, for which
        ``max |dv_c/dR| = (2/pi) * v_circ / r_v`` (achieved at the center).
        The line-of-sight projection multiplies by ``sin(i)``. Subclasses
        with non-arctan rotation curves should override.
        """
        vcirc = float(params['vcirc'])
        r_v = float(params['rscale'])
        cosi = float(params['cosi'])
        sini = (1.0 - cosi**2) ** 0.5
        return (2.0 / np.pi) * vcirc * sini / r_v

    def render_image(
        self,
        theta: jnp.ndarray,
        image_pars: ImagePars = None,
        plane: str = 'obs',
        X: jnp.ndarray = None,
        Y: jnp.ndarray = None,
        return_speed: bool = False,
        *,
        obs: Any = None,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Render velocity model as a 2D image, with optional PSF convolution.

        When obs has oversample > 1, velocity and flux are evaluated on
        fine-scale grids; convolve_flux_weighted handles sum-then-divide
        binning back to coarse scale.

        Calling conventions:
        - render_image(theta, obs=obs) -- with PSF from obs
        - render_image(theta, image_pars) -- no PSF, builds grids
        - render_image(theta, X=X, Y=Y) -- no PSF, pre-computed grids

        Parameters
        ----------
        theta : jnp.ndarray
            Parameter array.
        image_pars : ImagePars, optional
            Image parameters defining the grid.
        plane : str
            Coordinate plane for evaluation. Default is 'obs'.
        X, Y : jnp.ndarray, optional
            Pre-computed coordinate grids (coarse-scale).
        return_speed : bool
            If True, return speed map. Default is False.
        obs : VelocityObs, optional
            Observation object with PSF, grids, flux source, and oversampling.
        **kwargs
            Additional model-specific arguments.

        Returns
        -------
        jnp.ndarray
            2D velocity or speed map (coarse-scale).
        """
        rc = kwargs.pop('render_config', None)

        if obs is not None:
            oversample = rc.oversample if rc is not None else obs.oversample
            if oversample > 1 and obs.fine_X is not None:
                fine_X, fine_Y = obs.fine_X, obs.fine_Y
                model_vel = self(
                    theta, plane, fine_X, fine_Y, return_speed=return_speed
                )

                if obs.psf_data is not None:
                    from kl_pipe.psf import convolve_flux_weighted

                    flux_map = _get_flux_map(obs, plane, fine_X, fine_Y)
                    return convolve_flux_weighted(model_vel, flux_map, obs.psf_data)

                # no PSF but oversampled → bin for pixel integration.
                # Unweighted mean is a documented approximation for the
                # velocity-only setup (no flux source on obs); it is
                # physically correct only when fine intensity is uniform
                # within each coarse pixel. Real-data production uses the
                # PSF branch above with convolve_flux_weighted, which
                # applies proper sum(I*V)/sum(I) flux weighting. See
                # follow-up issue for the velocity-only oversample-no-PSF
                # path discussion.
                N = oversample
                Nrow = obs.image_pars.Nrow
                Ncol = obs.image_pars.Ncol
                return model_vel.reshape(Nrow, N, Ncol, N).mean(axis=(1, 3))

            model_vel = self(theta, plane, obs.X, obs.Y, return_speed=return_speed)

            if obs.psf_data is not None:
                from kl_pipe.psf import convolve_flux_weighted

                flux_map = _get_flux_map(obs, plane, obs.X, obs.Y)
                model_vel = convolve_flux_weighted(model_vel, flux_map, obs.psf_data)

            return model_vel

        # legacy/convenience path (no obs, no PSF)
        if X is None or Y is None:
            if image_pars is None:
                raise ValueError("Provide obs, image_pars, or (X, Y)")
            X, Y = build_map_grid_from_image_pars(image_pars)

        return self(theta, plane, X, Y, return_speed=return_speed)


def _get_flux_map(obs, plane, X, Y):
    """Extract flux map from VelocityObs for PSF weighting."""
    if obs.flux_image is not None:
        return obs.flux_image
    elif obs.flux_model is not None and obs.flux_theta is not None:
        return obs.flux_model(obs.flux_theta, plane, X, Y)
    else:
        raise ValueError("No flux source for velocity PSF weighting")


class IntensityModel(Model):
    """
    Base class for intensity models (scalar fields).

    Intensity models are evaluated in the disk plane and transformed through
    coordinate systems, but the intensity value itself doesn't change with
    projection
    """

    # name of the single parameter the model is linear in (its amplitude).
    # Simple profiles use 'flux'; composites override to 'total_flux'.
    # Rendering factors this out to cache a unit-amplitude spatial eval, so
    # it must name the linear amplitude, not a shape parameter.
    amplitude_param = 'flux'

    def __call__(
        self,
        theta: jnp.ndarray,
        plane: str,
        x: jnp.ndarray,
        y: jnp.ndarray,
        z: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Evaluate intensity at coordinates in the specified plane.
        """

        # extract transformation parameters
        x0 = self.get_param('x0', theta)
        y0 = self.get_param('y0', theta)
        g1 = self.get_param('g1', theta)
        g2 = self.get_param('g2', theta)
        theta_int = self.get_param('theta_int', theta)
        cosi = self.get_param('cosi', theta)

        # transform to disk plane
        x_disk, y_disk = transform_to_disk_plane(
            x, y, plane, x0, y0, g1, g2, theta_int, cosi
        )

        I_disk = self.evaluate_in_disk_plane(theta, x_disk, y_disk, z)

        # surface brightness projection depends on whether we're in the disk plane
        # or not
        if plane == 'disk':
            return I_disk
        else:
            # apply cos(i) brightening factor for projected intensity. clamp
            # cosi to COSI_FLOOR to match the gal->disk deprojection guard in
            # transformation.py; inference rejects cosi priors that reach the
            # floor, so within the valid range this is a no-op edge guard.
            cosi_safe = jnp.maximum(cosi, COSI_FLOOR)
            return I_disk / cosi_safe

    def render_unconvolved(self, theta, image_pars, oversample=5):
        """Render intensity image WITHOUT PSF, using k-space FT.

        For use by cube assembly — fast, anti-aliased, no PSF.
        Subclasses should override with their own k-space implementation.
        """
        raise NotImplementedError(
            "Subclasses must implement render_unconvolved method."
        )

    def maxk(self, params: dict, threshold: float = 1e-3) -> float:
        """Wavenumber where the bare profile FT drops below threshold.

        Used for adaptive grid sizing. The effective maxk of a rendering
        chain is computed from the full product: profile × pixel × PSF.

        Parameters
        ----------
        params : dict
            Profile parameters (e.g., rscale, hlr, n_sersic).
        threshold : float
            Maximum acceptable FT amplitude. Default 1e-3.

        Returns
        -------
        float
            Wavenumber in rad/arcsec.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement maxk(). "
            f"Required for adaptive grid sizing."
        )

    def stepk(self, params: dict, folding_threshold: float = 5e-3) -> float:
        """Minimum k-spacing to contain (1 - folding_threshold) of flux.

        Determines the real-space image extent needed to avoid periodic
        boundary folding in the DFT. Following GalSim convention:
        stepk = π / (stepk_min_hlr × hlr), where stepk_min_hlr ≈ 5.

        Parameters
        ----------
        params : dict
            Profile parameters.
        folding_threshold : float
            Maximum fraction of flux allowed to fold. Default 5e-3.

        Returns
        -------
        float
            k-spacing in rad/arcsec.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement stepk(). "
            f"Required for adaptive grid sizing."
        )

    def _ft_image(
        self,
        theta: jnp.ndarray,
        KX: jnp.ndarray,
        KY: jnp.ndarray,
        pixel_scale: float,
        Nrow: int,
        Ncol: int,
    ) -> jnp.ndarray:
        """Evaluate complex k-space FT at grid points (KX, KY).

        Called by ``_render_kspace`` and by ``CompositeIntensityModel``
        to sum component FTs before a single IFFT pass. Subclasses with
        k-space rendering must override.

        Parameters
        ----------
        theta : jnp.ndarray
            Parameter array for this model.
        KX, KY : jnp.ndarray
            k-space grids (rad/arcsec).
        pixel_scale : float
            Coarse pixel scale (arcsec/pixel) for half-pixel phase.
        Nrow, Ncol : int
            Coarse grid dimensions for half-pixel phase.

        Returns
        -------
        jnp.ndarray
            Complex FT array, same shape as KX/KY.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement _ft_image")

    def check_priors_safe(self, priors) -> None:
        """Raise if priors permit unphysical regimes for this model.

        Default: no-op. Override in subclasses with known prior-induced
        failure modes (e.g. ``InclinedSpergelModel`` cusp at high
        inclination with concentrated profiles). Called by
        ``InferenceTask.from_*_obs`` factories at construction time so
        misconfigured priors fail loudly before the first JIT trace.

        Parameters
        ----------
        priors : PriorDict
            Prior specification. Sampled parameters expose ``.bounds``;
            fixed parameters carry their scalar value.

        Raises
        ------
        ValueError
            If priors permit a regime where the model produces unphysical
            output. Error message must explain the regime and suggest
            remediation.
        """
        return

    @abstractmethod
    def evaluate_in_disk_plane(
        self, theta: jnp.ndarray, X: jnp.ndarray, Y: jnp.ndarray, Z: jnp.ndarray = None
    ) -> jnp.ndarray:
        """
        Evaluate intensity in disk plane (face-on)
        """
        raise NotImplementedError(
            "Subclasses must implement evaluate_in_disk_plane method."
        )


class ContinuumModel:
    """Adapter wrapping an ``IntensityModel`` for use as a stellar continuum.

    A continuum's amplitude is a spectral flux *density* (flux per unit
    wavelength), unlike an emission line's integrated ``flux``. To keep that
    explicit in the dotted-key namespace, this adapter relabels the wrapped
    profile's ``flux`` parameter to ``flux_per_nm`` in ``PARAMETER_NAMES``.
    ``flux_per_nm`` is a spectral flux density [flux / nm]; because the profile
    is linear in its amplitude, feeding a density produces a density surface
    brightness [flux / arcsec^2 / nm] = SB/nm, matching the emission-line term's
    cube voxel (see ``SourceModel.build_cube`` / ``units_and_conventions.md``).

    The relabel is purely external: the wrapped profile still evaluates from a
    positionally-identical ``theta`` (only the label at the flux index differs),
    so ``__call__`` delegates unchanged. All other attributes/methods (``name``,
    ``maxk``, ``render_image``, ``get_param``, ...) delegate to the wrapped
    profile via ``__getattr__``. Not an ``IntensityModel`` subclass (its
    ``PARAMETER_NAMES`` is instance-level, incompatible with the class-level
    contract), but registered as a virtual subclass so ``isinstance`` holds.
    """

    def __init__(self, profile: 'IntensityModel'):
        if isinstance(profile, ContinuumModel):
            # idempotent: unwrap so double-wrapping is a no-op
            profile = profile._profile
        self._profile = profile
        self.PARAMETER_NAMES = tuple(
            'flux_per_nm' if p == 'flux' else p for p in profile.PARAMETER_NAMES
        )

    @property
    def amplitude_param(self) -> str:
        # mirror the PARAMETER_NAMES relabel: a simple profile's 'flux'
        # amplitude becomes 'flux_per_nm'; a composite's 'total_flux' (which
        # has no 'flux' to relabel) passes through unchanged.
        amp = self._profile.amplitude_param
        return 'flux_per_nm' if amp == 'flux' else amp

    def __call__(self, theta, plane, x, y, z=None):
        # theta is positionally identical to the wrapped profile's (only the
        # flux label changed), so delegate the evaluation unchanged.
        return self._profile(theta, plane, x, y, z)

    def theta2pars(self, theta) -> dict:
        return {name: float(theta[i]) for i, name in enumerate(self.PARAMETER_NAMES)}

    def pars2theta(self, pars: dict) -> jnp.ndarray:
        return jnp.array([pars[name] for name in self.PARAMETER_NAMES])

    @property
    def name(self) -> str:
        return f'continuum({self._profile.name})'

    def __getattr__(self, item):
        # only reached for attributes not set on the adapter; delegate to the
        # wrapped profile. Guard _profile to avoid recursion before __init__.
        if item == '_profile':
            raise AttributeError(item)
        return getattr(self._profile, item)


IntensityModel.register(ContinuumModel)
