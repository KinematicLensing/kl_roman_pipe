import inspect
import jax.numpy as jnp
import jax
import galsim

from abc import abstractmethod, ABC
from typing import Tuple, Set, Any

from kl_pipe.transformation import transform_to_disk_plane
from kl_pipe.parameters import ImagePars
from kl_pipe.spectral import FiberPars
from kl_pipe.utils import build_map_grid_from_image_pars
from kl_pipe.psf import convolve_fft

# from kl_pipe.spectral import SpectralModel
# from kl_pipe.spectral import FiberPars

import galsim


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
        self._psf_data = None
        self._psf_frozen = False
        self._psf_oversample = 1
        self._psf_fine_X = None
        self._psf_fine_Y = None
        self._psf_kspace_fft = None

        return

    def configure_psf(
        self,
        gsobj,
        image_pars: 'ImagePars' = None,
        *,
        image_shape: tuple = None,
        pixel_scale: float = None,
        oversample: int = 5,
        gsparams=None,
        freeze: bool = False,
    ):
        """
        Configure PSF for rendering. Call BEFORE creating likelihood.

        Two calling conventions (image_pars is preferred):
        - configure_psf(gsobj, image_pars=image_pars)
        - configure_psf(gsobj, image_shape=(Nrow, Ncol), pixel_scale=scale)

        Parameters
        ----------
        gsobj : galsim.GSObject
            PSF profile.
        image_pars : ImagePars, optional
            Image parameters. Extracts (Nrow, Ncol) and pixel_scale internally.
        image_shape : tuple, optional
            (Nrow, Ncol) of data images.
        pixel_scale : float, optional
            arcsec/pixel.
        oversample : int
            Oversampling factor for source evaluation. Must be a positive odd
            integer. Default is 5. Set to 1 to disable oversampling.
        gsparams : galsim.GSParams, optional
            Override GSParams for PSF kernel rendering. Controls truncation
            (folding_threshold) and accuracy.
        freeze : bool
            If True, prevent reconfiguration (set by factory methods).
        """
        if self._psf_frozen:
            raise ValueError(
                "PSF is frozen (bound to a likelihood). Call clear_psf() first."
            )
        from kl_pipe.psf import precompute_psf_fft

        # extract coarse-scale image params
        if image_pars is not None:
            coarse_shape = (image_pars.Nrow, image_pars.Ncol)
            ps = image_pars.pixel_scale
        elif image_shape is not None and pixel_scale is not None:
            coarse_shape = image_shape
            ps = pixel_scale
        else:
            raise ValueError("Provide image_pars OR both image_shape and pixel_scale")

        self._psf_data = precompute_psf_fft(
            gsobj,
            image_shape=coarse_shape,
            pixel_scale=ps,
            oversample=oversample,
            gsparams=gsparams,
        )
        self._psf_oversample = oversample
        self._psf_frozen = freeze

        # pre-build fine-scale grids for oversampled evaluation
        if oversample > 1:
            if image_pars is not None:
                fine_image_pars = image_pars.make_fine_scale(oversample)
            else:
                fine_image_pars = ImagePars(
                    shape=(coarse_shape[0] * oversample, coarse_shape[1] * oversample),
                    pixel_scale=ps / oversample,
                    indexing='ij',
                )
            self._psf_fine_X, self._psf_fine_Y = build_map_grid_from_image_pars(
                fine_image_pars
            )
        else:
            self._psf_fine_X = None
            self._psf_fine_Y = None

    def clear_psf(self):
        """Remove PSF config and unfreeze."""
        self._psf_data = None
        self._psf_frozen = False
        self._psf_oversample = 1
        self._psf_fine_X = None
        self._psf_fine_Y = None
        self._psf_kspace_fft = None

    @property
    def has_psf(self):
        return self._psf_data is not None

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
            return self.render_image(theta, data_pars, plane=plane, **kwargs)

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
        **kwargs,
    ) -> jnp.ndarray:
        """
        Render model as a 2D image, including observational effects (PSF).

        Two calling conventions:
        - render_image(theta, image_pars) -- builds grids (scripts, notebooks)
        - render_image(theta, X=X, Y=Y) -- pre-computed grids (likelihood hot path)

        When PSF is configured with oversample > 1, the model is evaluated on
        a fine-scale grid and convolved at that resolution; convolve_fft bins
        the result back to coarse scale automatically.

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
        **kwargs
            Additional model-specific arguments.

        Returns
        -------
        jnp.ndarray
            2D image array (coarse-scale).
        """
        if self._psf_data is not None and self._psf_oversample > 1:
            # oversampled path: evaluate on fine grids
            model_map = self(theta, plane, self._psf_fine_X, self._psf_fine_Y, **kwargs)
            from kl_pipe.psf import convolve_fft

            return convolve_fft(model_map, self._psf_data)

        # non-oversampled path
        if X is None or Y is None:
            if image_pars is None:
                raise ValueError("Provide image_pars or (X, Y)")
            X, Y = build_map_grid_from_image_pars(image_pars)

        model_map = self(theta, plane, X, Y, **kwargs)

        if self._psf_data is not None:
            from kl_pipe.psf import convolve_fft

            model_map = convolve_fft(model_map, self._psf_data)

        return model_map

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
        self._psf_flux_model = None
        self._psf_flux_theta = None
        self._psf_flux_image = None

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
        x0 = self.get_param('vel_x0', theta) if 'vel_x0' in self._param_indices else 0.0
        y0 = self.get_param('vel_y0', theta) if 'vel_y0' in self._param_indices else 0.0

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

    def configure_velocity_psf(
        self,
        gsobj,
        image_pars: 'ImagePars' = None,
        *,
        image_shape: tuple = None,
        pixel_scale: float = None,
        oversample: int = 5,
        gsparams=None,
        flux_model=None,
        flux_theta=None,
        flux_image=None,
        flux_image_pars=None,
        freeze: bool = False,
    ):
        """
        Configure velocity PSF with flux weighting.

        Must provide ONE of:
        - flux_model + flux_theta: IntensityModel + fixed params
        - flux_image: pre-rendered intensity map
        In joint mode (KLModel), neither needed -- uses fitted intensity.

        Two calling conventions (image_pars is preferred):
        - configure_velocity_psf(gsobj, image_pars=image_pars, ...)
        - configure_velocity_psf(gsobj, image_shape=(Nrow, Ncol), pixel_scale=scale, ...)

        Parameters
        ----------
        gsobj : galsim.GSObject
            PSF profile.
        image_pars : ImagePars, optional
            Image parameters. Extracts (Nrow, Ncol) and pixel_scale internally.
        image_shape : tuple, optional
            (Nrow, Ncol) of velocity data images.
        pixel_scale : float, optional
            arcsec/pixel.
        oversample : int
            Oversampling factor. Default is 5.
        gsparams : galsim.GSParams, optional
            Override GSParams for PSF kernel rendering.
        flux_model : IntensityModel, optional
            Intensity model for rendering flux on velocity grid.
        flux_theta : jnp.ndarray, optional
            Fixed intensity params (used with flux_model).
        flux_image : ndarray, optional
            Pre-rendered intensity map for flux weighting.
        flux_image_pars : ImagePars, optional
            Image parameters of flux_image (for resampling if shape differs).
        freeze : bool
            If True, prevent reconfiguration.
        """
        self.configure_psf(
            gsobj,
            image_pars=image_pars,
            image_shape=image_shape,
            pixel_scale=pixel_scale,
            oversample=oversample,
            gsparams=gsparams,
            freeze=freeze,
        )
        self._psf_flux_model = flux_model
        self._psf_flux_theta = flux_theta

        if flux_model is None and flux_image is None:
            raise ValueError(
                "Velocity PSF requires flux weighting. Provide flux_model + "
                "flux_theta, or flux_image. For joint inference use KLModel."
            )

        # extract target shape/scale for resampling check
        if image_pars is not None:
            target_shape = (image_pars.Nrow, image_pars.Ncol)
            target_pixel_scale = image_pars.pixel_scale
        else:
            target_shape = image_shape
            target_pixel_scale = pixel_scale

        if flux_image is not None:
            # first resample to coarse-scale if shapes differ
            if flux_image.shape != target_shape:
                if flux_image_pars is None:
                    raise ValueError(
                        f"flux_image shape {flux_image.shape} != velocity grid "
                        f"{target_shape}. Provide flux_image_pars for resampling."
                    )
                from kl_pipe.psf import _resample_to_grid

                flux_image = _resample_to_grid(
                    flux_image,
                    flux_image_pars,
                    target_shape=target_shape,
                    target_pixel_scale=target_pixel_scale,
                )

            # upsample to fine-scale if oversampled
            if oversample > 1:
                from kl_pipe.psf import _resample_to_grid

                coarse_pars = ImagePars(
                    shape=target_shape, pixel_scale=target_pixel_scale, indexing='ij'
                )
                fine_shape = (
                    target_shape[0] * oversample,
                    target_shape[1] * oversample,
                )
                fine_ps = target_pixel_scale / oversample
                flux_image = _resample_to_grid(
                    flux_image,
                    coarse_pars,
                    target_shape=fine_shape,
                    target_pixel_scale=fine_ps,
                )

            self._psf_flux_image = jnp.asarray(flux_image)
        else:
            self._psf_flux_image = None

    def render_image(
        self,
        theta: jnp.ndarray,
        image_pars: ImagePars = None,
        plane: str = 'obs',
        X: jnp.ndarray = None,
        Y: jnp.ndarray = None,
        return_speed: bool = False,
        flux_theta_override: jnp.ndarray = None,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Render velocity model as a 2D image, with optional PSF convolution.

        When PSF is configured with oversample > 1, velocity and flux are
        evaluated on fine-scale grids; convolve_flux_weighted handles
        sum-then-divide binning back to coarse scale.

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
        flux_theta_override : jnp.ndarray, optional
            Intensity params for joint mode flux weighting.

        Returns
        -------
        jnp.ndarray
            2D velocity or speed map (coarse-scale).
        """
        if self._psf_data is not None and self._psf_oversample > 1:
            # oversampled path: evaluate on fine grids
            fine_X, fine_Y = self._psf_fine_X, self._psf_fine_Y
            model_vel = self(theta, plane, fine_X, fine_Y, return_speed=return_speed)

            from kl_pipe.psf import convolve_flux_weighted

            if flux_theta_override is not None and self._psf_flux_model is not None:
                flux_map = self._psf_flux_model(
                    flux_theta_override, plane, fine_X, fine_Y
                )
            elif self._psf_flux_image is not None:
                # already upsampled at configure time
                flux_map = self._psf_flux_image
            elif self._psf_flux_model is not None and self._psf_flux_theta is not None:
                flux_map = self._psf_flux_model(
                    self._psf_flux_theta, plane, fine_X, fine_Y
                )
            else:
                raise ValueError("No flux source for velocity PSF weighting")

            return convolve_flux_weighted(model_vel, flux_map, self._psf_data)

        # non-oversampled path
        if X is None or Y is None:
            if image_pars is None:
                raise ValueError("Provide image_pars or (X, Y)")
            X, Y = build_map_grid_from_image_pars(image_pars)

        model_vel = self(theta, plane, X, Y, return_speed=return_speed)

        if self._psf_data is not None:
            from kl_pipe.psf import convolve_flux_weighted

            if flux_theta_override is not None and self._psf_flux_model is not None:
                flux_map = self._psf_flux_model(flux_theta_override, plane, X, Y)
            elif self._psf_flux_image is not None:
                flux_map = self._psf_flux_image
            elif self._psf_flux_model is not None and self._psf_flux_theta is not None:
                flux_map = self._psf_flux_model(self._psf_flux_theta, plane, X, Y)
            else:
                raise ValueError("No flux source for velocity PSF weighting")

            model_vel = convolve_flux_weighted(model_vel, flux_map, self._psf_data)

        return model_vel


class IntensityModel(Model):
    """
    Base class for intensity models (scalar fields).

    Intensity models are evaluated in the disk plane and transformed through
    coordinate systems, but the intensity value itself doesn't change with
    projection
    """

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
        x0 = self.get_param('int_x0', theta)
        y0 = self.get_param('int_y0', theta)
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
            # apply cos(i) brightening factor for projected intensity
            return I_disk / cosi

    def render_unconvolved(self, theta, image_pars, oversample=5):
        """Render intensity image WITHOUT PSF, using k-space FT.

        For use by SpectralModel.build_cube() — fast, anti-aliased, no PSF.
        Subclasses should override with their own k-space implementation.
        """
        raise NotImplementedError(
            "Subclasses must implement render_unconvolved method."
        )

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


class KLModel(object):
    """
    Kinematic lensing model combining velocity and intensity components.

    Handles parameter management for models with shared and independent parameters.
    Builds a unified parameter space and provides slicing to extract sub-arrays
    for each component model.

    Parameters
    ----------
    velocity_model : VelocityModel
        Velocity model component.
    intensity_model : IntensityModel
        Intensity model component.
    shared_pars : set of str, optional
        Parameter names that are shared between models. If a parameter appears
        in both models and is in shared_pars, it will appear only once in the
        composite parameter array. Default is None (no shared parameters).
    meta_pars : dict, optional
        Fixed metadata for both models.

    Attributes
    ----------
    PARAMETER_NAMES : tuple
        Unified parameter names in order.
    velocity_slice : slice or array
        Indices to extract velocity parameters from composite theta.
    intensity_slice : slice or array
        Indices to extract intensity parameters from composite theta.

    Examples
    --------
    >>> # Models with independent parameters
    >>> vel_model = OffsetVelocityModel(meta)  # params: v0, vcirc, vel_x0, ve_y0
    >>> int_model = ExponentialIntensity(meta)  # params: flux, scale
    >>> kl_model = KLModel(vel_model, int_model)
    >>> kl_model.PARAMETER_NAMES
    ('v0', 'vcirc', 'vel_x0', 'vel_y0', 'flux', 'scale')
    >>>
    >>> # Models with shared parameters
    >>> vel_model = OffsetVelocityModel(meta_pars)  # params: v0, vcirc, x0, y0
    >>> int_model = OffsetIntensity(meta_pars)      # params: flux, x0, y0
    >>> kl_model = KLModel(vel_model, int_model, shared_pars={'x0', 'y0'})
    >>> kl_model.PARAMETER_NAMES
    ('v0', 'vcirc', 'x0', 'y0', 'flux')
    """

    def __init__(
        self,
        velocity_model: VelocityModel,
        intensity_model: IntensityModel,
        shared_pars: Set[str] = None,
        meta_pars: dict = None,
        spectral_model=None,
    ):
        self.velocity_model = velocity_model
        self.intensity_model = intensity_model
        self.shared_pars = shared_pars or set()
        self.meta_pars = meta_pars or {}
        self.spectral_model = spectral_model
        self._grism_psf_data = None

        self._build_parameter_structure()

        return

    def configure_joint_psf(
        self,
        psf_vel=None,
        psf_int=None,
        image_pars_vel: 'ImagePars' = None,
        image_pars_int: 'ImagePars' = None,
        *,
        image_shape_vel: tuple = None,
        pixel_scale_vel: float = None,
        image_shape_int: tuple = None,
        pixel_scale_int: float = None,
        oversample: int = 5,
        freeze: bool = False,
        gsparams=None,
    ):
        """
        Configure PSF for joint model.

        Velocity PSF: sets flux_model = self.intensity_model (rendered on
        velocity grid via flux_theta_override in joint likelihood).

        Two calling conventions (image_pars is preferred):
        - configure_joint_psf(..., image_pars_vel=pars_vel, image_pars_int=pars_int)
        - configure_joint_psf(..., image_shape_vel=(Nrow,Ncol), pixel_scale_vel=..., ...)

        Parameters
        ----------
        psf_vel : galsim.GSObject, optional
            PSF for velocity channel.
        psf_int : galsim.GSObject, optional
            PSF for intensity channel.
        image_pars_vel : ImagePars, optional
            Image parameters for velocity data.
        image_pars_int : ImagePars, optional
            Image parameters for intensity data.
        image_shape_vel : tuple, optional
            (Nrow, Ncol) of velocity data.
        pixel_scale_vel : float, optional
            arcsec/pixel for velocity grid.
        image_shape_int : tuple, optional
            (Nrow, Ncol) of intensity data.
        pixel_scale_int : float, optional
            arcsec/pixel for intensity grid.
        oversample : int
            Oversampling factor for source evaluation. Default is 5.
        freeze : bool
            If True, prevent reconfiguration.
        gsparams : galsim.GSParams, optional
            GalSim rendering parameters for PSF kernel accuracy.
        """
        if psf_vel is not None:
            self.velocity_model.configure_psf(
                psf_vel,
                image_pars=image_pars_vel,
                image_shape=image_shape_vel,
                pixel_scale=pixel_scale_vel,
                oversample=oversample,
                freeze=freeze,
                gsparams=gsparams,
            )
            # in joint mode, intensity model provides flux weighting
            self.velocity_model._psf_flux_model = self.intensity_model
            self.velocity_model._psf_flux_theta = None
            self.velocity_model._psf_flux_image = None

        if psf_int is not None:
            self.intensity_model.configure_psf(
                psf_int,
                image_pars=image_pars_int,
                image_shape=image_shape_int,
                pixel_scale=pixel_scale_int,
                oversample=oversample,
                freeze=freeze,
                gsparams=gsparams,
            )

    def _build_parameter_structure(self):
        """
        Build the unified parameter space and component slicing indices.

        When spectral_model is present, uses 3-way ordered deduplication:
        iterate vel_pars, int_pars, spectral_pars in order; first occurrence
        of each param name wins position.
        """

        vel_pars = self.velocity_model.PARAMETER_NAMES
        int_pars = self.intensity_model.PARAMETER_NAMES

        vel_pars_set = set(vel_pars)
        int_pars_set = set(int_pars)

        if not self.shared_pars.issubset(vel_pars_set & int_pars_set):
            invalid = self.shared_pars - (vel_pars_set & int_pars_set)
            raise ValueError(f"Shared parameters {invalid} not present in both models")

        # 3-way ordered deduplication
        seen = set()
        param_list = []

        for name in vel_pars:
            if name not in seen:
                param_list.append(name)
                seen.add(name)

        for name in int_pars:
            if name not in seen:
                param_list.append(name)
                seen.add(name)
            elif name in self.shared_pars:
                pass  # already added from vel_pars

        if self.spectral_model is not None:
            spec_pars = self.spectral_model.PARAMETER_NAMES
            for name in spec_pars:
                if name not in seen:
                    param_list.append(name)
                    seen.add(name)

        self.PARAMETER_NAMES = tuple(param_list)

        composite_param_dict = {name: i for i, name in enumerate(self.PARAMETER_NAMES)}

        self._velocity_indices = jnp.array(
            [composite_param_dict[name] for name in vel_pars]
        )
        self._intensity_indices = jnp.array(
            [composite_param_dict[name] for name in int_pars]
        )

        if self.spectral_model is not None:
            self._spectral_indices = jnp.array(
                [composite_param_dict[name] for name in spec_pars]
            )
        else:
            self._spectral_indices = None

        self._param_indices = composite_param_dict

        return

    def get_param(self, name: str, theta: jnp.ndarray):
        """
        Extract a parameter value by name from the composite parameter array.

        Parameters
        ----------
        name : str
            Parameter name (must be in PARAMETER_NAMES).
        theta : jnp.ndarray
            Composite parameter array.

        Returns
        -------
        scalar or jnp.ndarray
            Parameter value at the corresponding index.
        """
        idx = self._param_indices[name]

        return theta[idx]

    def get_velocity_pars(self, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Get velocity model parameters from composite array.

        Parameters
        ----------
        theta : jnp.ndarray
            Composite parameter array.

        Returns
        -------
        jnp.ndarray
            Parameter array for velocity model.
        """
        return theta[self._velocity_indices]

    def get_intensity_pars(self, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Get intensity model parameters from composite array.

        Parameters
        ----------
        theta : jnp.ndarray
            Composite parameter array.

        Returns
        -------
        jnp.ndarray
            Parameter array for intensity model.
        """
        return theta[self._intensity_indices]

    def get_spectral_pars(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Extract spectral component parameters from composite theta."""
        if self._spectral_indices is None:
            raise ValueError("No spectral model configured")
        return theta[self._spectral_indices]

    def render_cube(
        self,
        theta: jnp.ndarray,
        cube_pars,
        plane: str = 'obs',
    ) -> jnp.ndarray:
        """Render PSF-convolved datacube.

        1. Calls spectral_model.build_cube() -> intrinsic cube (no PSF)
        2. Per-slice convolve_fft(cube[:,:,k], psf_data) if PSF configured

        Returns shape (Nrow, Ncol, Nlambda).
        """
        if self.spectral_model is None:
            raise ValueError("No spectral model configured — use render_image for 2D")

        theta_vel = self.get_velocity_pars(theta)
        theta_int = self.get_intensity_pars(theta)
        theta_spec = self.get_spectral_pars(theta)

        cube = self.spectral_model.build_cube(
            theta_spec, theta_vel, theta_int, cube_pars, plane=plane
        )

        # per-slice PSF convolution
        if self._grism_psf_data is not None:
            from kl_pipe.psf import convolve_fft

            Nlam = cube.shape[2]
            slices = []
            for k in range(Nlam):
                slices.append(convolve_fft(cube[:, :, k], self._grism_psf_data))
            cube = jnp.stack(slices, axis=-1)

        return cube

    def render_grism(
        self,
        theta: jnp.ndarray,
        grism_pars,
        plane: str = 'obs',
        cube_pars=None,
    ) -> jnp.ndarray:
        """Render dispersed grism image.

        1. Calls render_cube -> PSF-convolved cube
        2. Calls disperse_cube -> 2D dispersed image

        For JIT/grad, pre-compute cube_pars with a concrete z and pass it in:
            cube_pars = gp.to_cube_pars(z=1.0)
            jit(partial(kl.render_grism, grism_pars=gp, cube_pars=cube_pars))(theta)

        Returns shape (Nrow, Ncol).
        """
        if self.spectral_model is None:
            raise ValueError("No spectral model configured")

        from kl_pipe.dispersion import disperse_cube

        if cube_pars is None:
            # convenience path: auto-build from z (not JIT-compatible)
            theta_spec = self.get_spectral_pars(theta)
            z = float(self.spectral_model.get_param('z', theta_spec))
            cube_pars = grism_pars.to_cube_pars(z)

        cube = self.render_cube(theta, cube_pars, plane=plane)

        return disperse_cube(cube, grism_pars, cube_pars.lambda_grid)

    def configure_grism_psf(
        self,
        gsobj,
        cube_pars,
        oversample: int = 5,
        gsparams=None,
    ):
        """Pre-compute PSF FFT at the cube's spatial grid.

        Uses existing precompute_psf_fft() from psf.py.
        Stores _grism_psf_data for use by render_cube.
        """
        from kl_pipe.psf import precompute_psf_fft

        self._grism_psf_data = precompute_psf_fft(
            gsobj,
            image_pars=cube_pars.image_pars,
            oversample=oversample,
            gsparams=gsparams,
        )

    def render_fiber(
        self,
        theta: jnp.ndarray,
        fiber_pars,
        plane: str = 'obs',
        cube_pars=None,
        force_noise_free=True,
        run_mode='ETC',
    ) -> jnp.ndarray:

        if self.spectral_model is None:
            raise ValueError("No spectral model configured")

        # if cube_pars is None: #I don't think I have actually tried this out
        ## convenience path: auto-build from z (not JIT-compatible)
        # theta_spec = self.get_spectral_pars(theta)
        # z = float(self.spectral_model.get_param('z', theta_spec))
        # cube_pars = fiber_pars.to_cube_pars(z) #yeah this is not a thing yet

        cube = self.render_cube(theta, cube_pars, plane=plane)  # theoretical cube
        # intensity model
        intensity_model = self.intensity_model

        return self.fiber_observe_cube(cube, fiber_pars, force_noise_free, run_mode)

    def fiber_observe_cube(
        self, cube, fiber_pars, force_noise_free=True, run_mode='ETC'
    ):
        # if photometry image...wip
        if not fiber_pars.is_dispersed:
            from kl_pipe.psf import precompute_psf_fft, convolve_fft

            self.ATMPSF_conv_fiber_mask = None
            self.resolution_mat = None
            psfdata = self._fiber_psf_data
            # galsim_psf = self._build_PSF_model_fiber(fiber_pars.obs_conf, lam_mean=fiber_pars.lambda_eff)
            # self.intensity_model.configure_psf(galsim_psf, image_pars = fiber_pars.cube_pars.image_pars, image_shape= (fiber_pars.obs_conf.NAXIS1,fiber_pars.obs_conf.NAXIS2), pixel_scale=fiber_pars.obs_conf['PIXSCALE'], oversample=1)
            if run_mode == 'ETC':
                # psf_convolved_int = self.intensity_model.render_image(theta=theta_int, image_pars = fiberpars_instance.cube_pars.image_pars, plane='obs', oversample =1)
                cube_bp = cube * jnp.array(fiber_pars._bp_array)
                raw_img = jnp.sum(cube_bp, axis=2) * fiber_pars.dlambda
                print(fiber_pars.dlambda)
                # if self._fiber_psf_data.oversample == 1:
                # highres_img = raw_img
                # elif self._fiber_psf_data.oversample > 1: #might not be correct
                # highres_img = jnp.repeat(
                # jnp.repeat(raw_img, self._fiber_psf_data.oversample, axis=0),
                # self._fiber_psf_data.oversample,
                # axis=1,
                # )
                factor = (
                    jnp.pi
                    * (fiber_pars.obs_conf['DIAMETER'] / 2.0) ** 2
                    * fiber_pars.obs_conf['EXPTIME']
                    / fiber_pars.obs_conf['GAIN']
                )

                img = factor * convolve_fft(
                    raw_img, psfdata
                )  # downsampling baked into convolve_fft

            # add SNR mode too

            if force_noise_free:
                return img, None

            else:
                print('photometry noise implementation WIP')
                # noise_type = fiber_pars.obs_conf['NOISETYP']
                # if noise_type == 'ccd':
                # read_noise = fiber_pars.obs_conf['RDNOISE'] #electrons/pixel probably
                # gain = fiber_pars.obs_conf['GAIN'] #electrons/ADU
                # exp_time = fiber_pars.obs_conf['EXPTIME']
                # sky_level = fiber_pars.obs_conf['SKYLEVEL']*exp_time/gain #what are the units of this? ADU/pixel? ADU/pixel/second? if /second then I should multiply by exposure time. #divide sky level by gain to get electrons/pixel
                # noise_std = ((sky_level+img)+read_noise**2)**0.5
                # key = jax.random.key(0)
                # noise = (jax.random.normal(key, shape=(img.shape[0],img.shape[1])) * noise_std)
                # print('noise', noise)
                # print('shape', jnp.shape(noise))
                # img_withNoise = img.copy()
                # img_withNoise += noise
                # noise_img = img_withNoise - img
                # assert (img_withNoise.array is not None), "Null data"
                # assert (img.array is not None), "Null data"
                # if fiber_pars.obs_conf['ADDNOISE']:
                # return img_withNoise.array, noise_img.array
                # else:
                # return img.array, noise_img.array
                return img, None

        # if fiber 1D spectrum
        else:
            # see notes in kl-tools for why it can be done this way
            self.wave = fiber_pars.lambda_grid
            spec_1D = jnp.sum(
                (self.ATMPSF_conv_fiber_mask[:, :, jnp.newaxis] * cube),
                axis=jnp.array([0, 1]),
            )
            if (
                run_mode == 'ETC'
            ):  # "exposure time calculator" -- unit of spectrum = counts in detector.
                spec_1D = spec_1D * fiber_pars._bp_array
                factor = (
                    jnp.pi
                    * (fiber_pars.obs_conf['DIAMETER'] / 2.0) ** 2
                    * fiber_pars.obs_conf['EXPTIME']
                    / fiber_pars.obs_conf['GAIN']
                )  # units cm^2 * seconds * ADU/electron
                spec_1D = spec_1D * factor

            # fiber PSF can result in degrade in spectra resolution
            if self.resolution_mat is not None:
                spec_1D = jnp.dot(self.resolution_mat, spec_1D)
            if force_noise_free:
                return spec_1D, None
            else:
                if (
                    run_mode == 'ETC'
                ):  # noise computed from sky level; realistic sky level from kitt peak for example
                    # poisson noise is included too
                    # dark current is ignored; readout noise not ignored

<<<<<<< Updated upstream
=======
            cube = self.render_cube(theta, cube_pars, plane=plane)
            return disperse_cube(cube, gp, cube_pars.lambda_grid)
        
    def render_fiber(
        self,
        theta: jnp.ndarray,
        #fiber_pars,
        obs,
        plane: str = 'obs',
        #cube_pars=None,
        force_noise_free=True,
        run_mode='ETC',
    ) -> jnp.ndarray:

        if self.spectral_model is None:
            raise ValueError("No spectral model configured")

        # if cube_pars is None: #I don't think I have actually tried this out
        ## convenience path: auto-build from z (not JIT-compatible)
        # theta_spec = self.get_spectral_pars(theta)
        # z = float(self.spectral_model.get_param('z', theta_spec))
        # cube_pars = fiber_pars.to_cube_pars(z) #yeah this is not a thing yet

        fiber_pars=obs.fiber_pars
        cube_pars=fiber_pars.cube_pars
        cube = self.render_cube(theta, cube_pars, plane=plane)  # theoretical cube

        return self.fiber_observe_cube(cube, fiber_pars, obs, force_noise_free, run_mode)

    def fiber_observe_cube(
        self, 
        cube, 
        fiber_pars, 
        obs, 
        force_noise_free=True, 
        run_mode='ETC'
    ):
        # if photometry image...wip
        if not fiber_pars.is_dispersed:

            #self.ATMPSF_conv_fiber_mask = None
            #self.resolution_mat = None #do not set to None because the model will be reused for photometry

            psfdata = obs.psf_data #self._fiber_photo_psf_data
            #psfdata = self._fiber_psf_data
            # galsim_psf = self._build_PSF_model_fiber(fiber_pars.obs_conf, lam_mean=fiber_pars.lambda_eff)
            # self.intensity_model.configure_psf(galsim_psf, image_pars = fiber_pars.cube_pars.image_pars, image_shape= (fiber_pars.obs_conf.NAXIS1,fiber_pars.obs_conf.NAXIS2), pixel_scale=fiber_pars.obs_conf['PIXSCALE'], oversample=1)
            if run_mode == 'ETC':
                # psf_convolved_int = self.intensity_model.render_image(theta=theta_int, image_pars = fiberpars_instance.cube_pars.image_pars, plane='obs', oversample =1)
                cube_bp = cube * jnp.array(fiber_pars._bp_array)
                raw_img = jnp.sum(cube_bp, axis=2) * fiber_pars.dlambda
                #print(fiber_pars.dlambda)
                # if self._fiber_psf_data.oversample == 1:
                # highres_img = raw_img
                # elif self._fiber_psf_data.oversample > 1: #might not be correct
                # highres_img = jnp.repeat(
                # jnp.repeat(raw_img, self._fiber_psf_data.oversample, axis=0),
                # self._fiber_psf_data.oversample,
                # axis=1,
                # )
                factor = (
                    jnp.pi
                    * (fiber_pars.obs_conf['DIAMETER'] / 2.0) ** 2
                    * fiber_pars.obs_conf['EXPTIME']
                    / fiber_pars.obs_conf['GAIN']
                )

                img = factor * convolve_fft(
                    raw_img, psfdata
                )  # downsampling baked into convolve_fft

            # add SNR mode too

            if force_noise_free:
                return img, None

            else:
                print('photometry noise implementation WIP')
                # noise_type = fiber_pars.obs_conf['NOISETYP']
                # if noise_type == 'ccd':
                # read_noise = fiber_pars.obs_conf['RDNOISE'] #electrons/pixel probably
                # gain = fiber_pars.obs_conf['GAIN'] #electrons/ADU
                # exp_time = fiber_pars.obs_conf['EXPTIME']
                # sky_level = fiber_pars.obs_conf['SKYLEVEL']*exp_time/gain #what are the units of this? ADU/pixel? ADU/pixel/second? if /second then I should multiply by exposure time. #divide sky level by gain to get electrons/pixel
                # noise_std = ((sky_level+img)+read_noise**2)**0.5
                # key = jax.random.key(0)
                # noise = (jax.random.normal(key, shape=(img.shape[0],img.shape[1])) * noise_std)
                # print('noise', noise)
                # print('shape', jnp.shape(noise))
                # img_withNoise = img.copy()
                # img_withNoise += noise
                # noise_img = img_withNoise - img
                # assert (img_withNoise.array is not None), "Null data"
                # assert (img.array is not None), "Null data"
                # if fiber_pars.obs_conf['ADDNOISE']:
                # return img_withNoise.array, noise_img.array
                # else:
                # return img.array, noise_img.array
                return img, None

        # if fiber 1D spectrum
        else:
            # see notes in kl-tools for why it can be done this way
            wave = fiber_pars.lambda_grid
            spec_1D = jnp.sum(
                (obs.ATMPSF_conv_fiber_mask[:, :, jnp.newaxis] * cube),
                axis=(0, 1))
            
            if (run_mode == 'ETC'):  # "exposure time calculator" -- unit of spectrum = counts in detector.
                spec_1D = spec_1D * fiber_pars._bp_array
                factor = (
                    jnp.pi
                    * (fiber_pars.obs_conf['DIAMETER'] / 2.0) ** 2
                    * fiber_pars.obs_conf['EXPTIME']
                    / fiber_pars.obs_conf['GAIN']
                )  # units cm^2 * seconds * ADU/electron
                spec_1D = spec_1D * factor

            # fiber PSF can result in degrade in spectra resolution
            if obs.resolution_matrix is not None: #self.resolution_mat
                spec_1D = jnp.dot(obs.resolution_matrix, spec_1D)
                print(spec_1D)
            if force_noise_free:
                return spec_1D, None
            else: #work in progress
                if (
                    run_mode == 'ETC'
                ):  # noise computed from sky level; realistic sky level from kitt peak for example
                    # poisson noise is included too
                    # dark current is ignored; readout noise not ignored

>>>>>>> Stashed changes
                    # should precompute skysb, definitely don't repeat every likelihood eval
                    skysb = galsim.LookupTable.from_file(
                        fiber_pars.obs_conf["SKYMODEL"], f_log=True
                    )  # Ang v.s. 1e-17 erg s-1 cm-2 A-1 arcsec-2
                    fiber_area = jnp.pi * (fiber_pars.obs_conf["FIBERRAD"]) ** 2
<<<<<<< Updated upstream
                    _wave = self.wave * 10  # Angstrom
=======
                    _wave = wave * 10  # Angstrom
>>>>>>> Stashed changes
                    _dwave = _wave[1] - _wave[0]  # Angstrom
                    _hnu = 1986445857.148928 / _wave  # 1e-17 erg
                    skyct = skysb(_wave) * fiber_area * _dwave / _hnu  # s-1 cm-2
                    skyct *= (
                        fiber_pars._bp_array
                        * jnp.pi
                        * (fiber_pars.obs_conf['DIAMETER'] / 2.0) ** 2
                        * fiber_pars.obs_conf['EXPTIME']
                        / fiber_pars.obs_conf['GAIN']
                    )
                    ## noise std, not including dark current
                    ## eff read noise = sqrt(Npix along y) * read noise
                    noise_std = (
                        skyct + spec_1D + fiber_pars.obs_conf['RDNOISE'] ** 2
                    ) ** 0.5
                    key = jax.random.key(0)
                    noise = (
                        jax.random.normal(key, shape=(spec_1D.shape[0],)) * noise_std
                    )

                else:  # for SNR mode, you provide flux and noise parameter and "the code doesn't care what the unit is"
                    key = jax.random.key(0)
                    noise = (
                        jax.random.normal(key, shape=(spec_1D.shape[0],))
                        * fiber_pars.obs_conf['NOISESIG']
                    )
                if fiber_pars.obs_conf['ADDNOISE']:
                    # print('spec_1D, noise', spec_1D, noise)
                    return spec_1D + noise, noise
                else:
                    return spec_1D, noise

<<<<<<< Updated upstream
    def get_fiber_mask(self, fiber_pars):
        from photutils.geometry import (
            circular_overlap_grid as cog,
        )  # is it alright for me to still use this?

        mNx, mNy = fiber_pars.spatial_shape[1], fiber_pars.spatial_shape[0]
        mscale = fiber_pars.pix_scale
        if fiber_pars.is_dispersed:
            fiber_cen = [
                fiber_pars.obs_conf['FIBERDX'],
                fiber_pars.obs_conf['FIBERDY'],
            ]  # dx, dy in arcsec
            fiber_rad = fiber_pars.obs_conf['FIBERRAD']  # radius in arcsec
            xmin, xmax = -mNx / 2 * mscale, mNx / 2 * mscale
            ymin, ymax = -mNy / 2 * mscale, mNy / 2 * mscale
            mask = cog(
                xmin - fiber_cen[0],
                xmax - fiber_cen[0],
                ymin - fiber_cen[1],
                ymax - fiber_cen[1],
                mNx,
                mNy,
                fiber_rad,
                1,
                2,
            )
        else:
            mask = jnp.ones([mNy, mNx])
        return mask

    def configure_fiber_psf(
        self,
        gsobj,
        cube_pars,
        oversample: int = 1,  # oversampling not really necessary unless you don't have an analytical PSF
        gsparams=None,
    ):
        """Pre-compute PSF FFT at the cube's spatial grid.

        Uses existing precompute_psf_fft() from psf.py.
        Stores _grism_psf_data for use by render_cube.
        """
        from kl_pipe.psf import precompute_psf_fft

        self._fiber_psf_data = precompute_psf_fft(
            gsobj,
            image_pars=cube_pars.image_pars,
            oversample=oversample,
            gsparams=gsparams,
        )
        # return self._fiber_psf_data  # grism version doesn't return anything here btw

    def precompute_PSF_convolved_fiber_mask(
        self, fiber_pars
    ):  # precompute fiber mask and make it a jax array
        '''get atm-PSF convolved fiber mask'''
        mNx, mNy = fiber_pars.spatial_shape[1], fiber_pars.spatial_shape[0]
        mscale = fiber_pars.pix_scale

        galsim_psf = self._build_PSF_model_fiber(
            fiber_pars.obs_conf, lam_mean=fiber_pars.lambda_eff
        )

        mask = galsim.InterpolatedImage(
            galsim.Image(array=self.get_fiber_mask(fiber_pars)), scale=mscale
        )

        # convolve fiber mask with atmospheric PSF
        maskC = mask if galsim_psf is None else galsim.Convolve([mask, galsim_psf])
        ary = maskC.drawImage(nx=mNx, ny=mNy, scale=mscale).array

        # replace galsim convolution?
        # fiber_psf_data = self.configure_fiber_psf(galsim_psf, fiber_pars.cube_pars)
        # if self._fiber_psf_data is not None:
        # from kl_pipe.psf import convolve_fft
        # oversample = self._fiber_psf_data.oversample
        # maskC = convolve_fft(self.get_fiber_mask(fiber_pars), self._fiber_psf_data) #mask needs to be 5x bigger in size if oversampling = 5
        ##maskC = convolve_fft(self.get_fiber_mask(fiber_pars), fiber_psf_data)
        # else:
        # maskC = self.get_fiber_mask(fiber_pars)
        # print('maskC', maskC)
        # ary=maskC

        self.ATMPSF_conv_fiber_mask = jnp.array(ary)

    def _build_PSF_model_fiber(self, config, **kwargs):  # should check all this stuff
        '''Generate Galsim PSF model

        Inputs:
        =======
        kwargs: keyword arguments for building psf model
            - lam: wavelength in nm
            - scale: pixel scale

        Outputs:
        ========
        psf_model: GalSim PSF object

        '''
        _type = config.get('PSFTYPE', 'none').lower()
        if _type != 'none':
            if _type == 'airy':
                lam = kwargs.get('lam', 1000)  # nm
                scale_unit = kwargs.get('scale_unit', galsim.arcsec)
                return galsim.Airy(
                    lam=lam, diam=config['DIAMETER'] / 100, scale_unit=scale_unit
                )
            elif _type == 'airy_mean':
                scale_unit = kwargs.get('scale_unit', galsim.arcsec)
                # return galsim.Airy(config['psf_fwhm']/1.028993969962188,
                #               scale_unit=scale_unit)
                lam = kwargs.get("lam_mean", 1000)  # nm
                return galsim.Airy(
                    lam=lam, diam=config['DIAMETER'] / 100, scale_unit=scale_unit
                )
            elif _type == 'airy_fwhm':
                loverd = config['PSFFWHM'] / 1.028993969962188
                scale_unit = kwargs.get('scale_unit', galsim.arcsec)
                return galsim.Airy(lam_over_diam=loverd, scale_unit=scale_unit)
            elif _type == 'moffat':
                beta = config.get('PSFBETA', 2.5)
                fwhm = config.get('PSFFWHM', 0.5)
                return galsim.Moffat(beta=beta, fwhm=fwhm)
            else:
                raise ValueError(f'{_type} has not been implemented yet!')
        else:
            return None

    def get_resolution_matrix_fiber(self, fiber_pars):
        from scipy.sparse import dia_matrix

        if fiber_pars.is_dispersed:
            diameter_in_pixel = fiber_pars.obs_conf['FIBRBLUR']
            sigma = diameter_in_pixel / 4.0
            x_in_pixel = jnp.arange(-5, 6)
            # assume Gaussian for now
            kernel = jnp.exp(-0.5 * (x_in_pixel / sigma) ** 2) / (
                (2 * jnp.pi) ** 0.5 * sigma
            )
            # get the resolution matrix (sparse matrix)
            band = jnp.array([kernel]).repeat(fiber_pars.n_lambda, axis=0).T
            offset = jnp.arange(kernel.shape[0] // 2, -(kernel.shape[0] // 2) - 1, -1)
            Rmat = dia_matrix(
                (band, offset), shape=(fiber_pars.n_lambda, fiber_pars.n_lambda)
            )
        else:
            Rmat = None
        self.resolution_mat = jnp.array(
            Rmat.toarray()
        )  # need to figure out how to make jnp array of sparse matrix directly. but oh well, for now this
=======
>>>>>>> Stashed changes

    def evaluate_velocity(
        self,
        theta: jnp.ndarray,
        plane: str,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        Z: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Evaluate velocity model component.

        Parameters
        ----------
        theta : jnp.ndarray
            Composite parameter array.
        plane : str
            Evaluation plane.
        X, Y : jnp.ndarray
            Coordinate arrays.
        Z : jnp.ndarray, optional
            Z-coordinate array.

        Returns
        -------
        jnp.ndarray
            Velocity map.
        """
        theta_vel = self.get_velocity_pars(theta)

        return self.velocity_model(theta_vel, plane, X, Y, Z)

    def evaluate_intensity(
        self,
        theta: jnp.ndarray,
        plane: str,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        Z: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Evaluate intensity model component.

        Parameters
        ----------
        theta : jnp.ndarray
            Composite parameter array.
        plane : str
            Evaluation plane.
        X, Y : jnp.ndarray
            Coordinate arrays.
        Z : jnp.ndarray, optional
            Z-coordinate array.

        Returns
        -------
        jnp.ndarray
            Intensity map.
        """
        theta_int = self.get_intensity_pars(theta)

        return self.intensity_model(theta_int, plane, X, Y, Z)

    def __call__(
        self,
        theta: jnp.ndarray,
        plane: str,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        Z: jnp.ndarray = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Evaluate both model components.

        Parameters
        ----------
        theta : jnp.ndarray
            Composite parameter array.
        plane : str
            Evaluation plane.
        X, Y : jnp.ndarray
            Coordinate arrays.
        Z : jnp.ndarray, optional
            Z-coordinate array.

        Returns
        -------
        velocity_map : jnp.ndarray
            Velocity map.
        intensity_map : jnp.ndarray
            Intensity map.
        """

        velocity_map = self.evaluate_velocity(theta, plane, X, Y, Z)
        intensity_map = self.evaluate_intensity(theta, plane, X, Y, Z)

        return velocity_map, intensity_map

    def render(
        self,
        theta: jnp.ndarray,
        data_type: str,
        data_pars,
        plane: str = 'obs',
        **kwargs,
    ) -> jnp.ndarray:
        """High-level rendering interface for different data products."""
        if data_type == 'cube':
            return self.render_cube(theta, data_pars, plane=plane, **kwargs)
        elif data_type == 'grism':
            return self.render_grism(theta, data_pars, plane=plane, **kwargs)
        else:
            raise ValueError(
                f"Unknown data_type '{data_type}'. " f"Must be one of: 'cube', 'grism'"
            )

    def theta2pars(self, theta: jnp.ndarray) -> dict:
        """
        Convert parameter array to dictionary.

        Parameters
        ----------
        theta : jnp.ndarray
            Composite parameter array.

        Returns
        -------
        dict
            Dictionary mapping parameter names to values.
        """

        return {name: float(theta[i]) for i, name in enumerate(self.PARAMETER_NAMES)}

    def pars2theta(self, pars: dict) -> jnp.ndarray:
        """
        Convert parameter dictionary to array.

        Parameters
        ----------
        pars : dict
            Dictionary with keys matching self.PARAMETER_NAMES.

        Returns
        -------
        jnp.ndarray
            Composite parameter array ordered according to self.PARAMETER_NAMES.
        """

        return jnp.array([pars[name] for name in self.PARAMETER_NAMES])


# class FiberModel(object):
# SpectralModel returns INTRINSIC cube (no PSF). PSF applied by KLModel per-slice.
# so, should I be using KLModel first?

#'''This class wraps the theoretical cube after it is generated to produce
# simulated images.
#'''

##first make fiber parameters
#''' Store the `FiberPars` object'''
# def __init__(
# self,
# meta_pars: dict,
# obs_conf: dict,
# spectral_model: SpectralModel):

# right now I have FiberPars taking cube_pars but I should just have it take meta_pars idk
# self.Fpars = FiberPars(meta_pars, obs_conf=obs_conf)

# calculate the atmospheric PSF convolved fiber mask, if PSF model is
# not sampled.
# if self.Fpars.is_dispersed:
# self.ATMPSF_conv_fiber_mask = self.get_PSF_convolved_fiber_mask() #need to make these
# self.resolution_mat = self.get_resolution_matrix() #need to make these
# else:
# self.ATMPSF_conv_fiber_mask = None
# self.resolution_mat = None

# return

# @property
# def bp_array(self):
# return self.Fpars.bp_array
# @property
# def conf(self):
# return self.Fpars.conf
# @property
# def lambdas(self): # (Nlam, 2)
# return self.Fpars.lambdas
# @property
# def wave(self):
# return self.Fpars.wave
# @property
# def lambda_eff(self):
# return self.Fpars.lambda_eff

# def get_fiber_mask(self):
# mNx, mNy = self.Fpars['shape'][2], self.Fpars['shape'][1]
# mscale = self.Fpars['pix_scale']
# if self.Fpars.is_dispersed:
# fiber_cen = [self.conf['FIBERDX'], self.conf['FIBERDY']] # dx, dy in arcsec
# fiber_rad = self.conf['FIBERRAD'] # radius in arcsec
# xmin, xmax = -mNx/2*mscale, mNx/2*mscale
# ymin, ymax = -mNy/2*mscale, mNy/2*mscale
# mask = cog(xmin-fiber_cen[0], xmax-fiber_cen[0],
# ymin-fiber_cen[1], ymax-fiber_cen[1],
# mNx, mNy, fiber_rad, 1, 2)
# else:
# mask = np.ones([mNy, mNx])
# return mask
