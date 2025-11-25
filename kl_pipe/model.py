import inspect
import jax.numpy as jnp
import jax

from abc import abstractmethod, ABC
from typing import Tuple, Set, Any

from kl_pipe.transformation import transform_to_disk_plane
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
        self, theta: jnp.ndarray, image_pars: ImagePars, plane: str = 'obs', **kwargs
    ) -> jnp.ndarray:
        """
        Render model as a 2D image.

        NOTE: Some model subclasses may override this method due to special rendering
        needs.

        Parameters
        ----------
        theta : jnp.ndarray
            Parameter array.
        image_pars : ImagePars
            Image parameters defining grid, pixel scale, etc.
        plane : str
            Coordinate plane for evaluation. Default is 'obs'.
        **kwargs
            Additional model-specific arguments.

        Returns
        -------
        jnp.ndarray
            2D image array with shape matching image_pars.shape.
        """

        # build coordinate grids from ImagePars
        X, Y = build_map_grid_from_image_pars(image_pars)

        # evaluate model on grid
        return self(theta, plane, X, Y)

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

    def render_image(
        self,
        theta: jnp.ndarray,
        image_pars: ImagePars,
        plane: str = 'obs',
        return_speed: bool = False,
    ) -> jnp.ndarray:
        """
        Render velocity model as a 2D image.

        Parameters
        ----------
        theta : jnp.ndarray
            Parameter array.
        image_pars : ImagePars
            Image parameters defining the grid.
        plane : str
            Coordinate plane for evaluation. Default is 'obs'.
        return_speed : bool
            If True, return speed map. If False, return line-of-sight velocity map.
            Default is False.

        Returns
        -------
        jnp.ndarray
            2D velocity or speed map with shape image_pars.shape.
        """

        # build coordinate grids from ImagePars
        X, Y = build_map_grid_from_image_pars(image_pars)

        # evaluate model on grid
        return self(theta, plane, X, Y, return_speed=return_speed)


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
    ):
        self.velocity_model = velocity_model
        self.intensity_model = intensity_model
        self.shared_pars = shared_pars or set()
        self.meta_pars = meta_pars or {}

        self._build_parameter_structure()

        return

    def _build_parameter_structure(self):
        """
        Build the unified parameter space and component slicing indices.

        Creates PARAMETER_NAMES with shared parameters appearing once, and builds
        index mappings for extracting component-specific parameter arrays.
        """

        vel_pars = self.velocity_model.PARAMETER_NAMES
        int_pars = self.intensity_model.PARAMETER_NAMES

        vel_pars_set = set(vel_pars)
        int_pars_set = set(int_pars)

        if not self.shared_pars.issubset(vel_pars_set & int_pars_set):
            invalid = self.shared_pars - (vel_pars_set & int_pars_set)
            raise ValueError(f"Shared parameters {invalid} not present in both models")

        param_list = list(vel_pars)
        for param in int_pars:
            if param not in self.shared_pars:
                param_list.append(param)

        self.PARAMETER_NAMES = tuple(param_list)

        composite_param_dict = {name: i for i, name in enumerate(self.PARAMETER_NAMES)}

        self._velocity_indices = jnp.array(
            [composite_param_dict[name] for name in vel_pars]
        )
        self._intensity_indices = jnp.array(
            [composite_param_dict[name] for name in int_pars]
        )

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

        return {name: float(theta[i]) for i, name in enumerate(self.PARAM_NAMES)}

    def pars2theta(self, pars: dict) -> jnp.ndarray:
        """
        Convert parameter dictionary to array.

        Parameters
        ----------
        pars : dict
            Dictionary with keys matching self.PARAM_NAMES.

        Returns
        -------
        jnp.ndarray
            Composite parameter array ordered according to self.PARAM_NAMES.
        """

        return jnp.array([pars[name] for name in self.PARAM_NAMES])
