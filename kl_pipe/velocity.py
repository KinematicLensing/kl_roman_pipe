import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from kl_pipe.model import VelocityModel
from kl_pipe.transformation import transform_to_disk_plane


class CenteredVelocityModel(VelocityModel):
    """
    Velocity model with no spatial offset from the origin.

    Parameters
    ----------
    cosi : float
        Cosine of inclination angle.
    theta_int : float
        Intrinsic position angle.
    g1 : float
        First component of the shear.
    g2 : float
        Second component of the shear.
    v0 : float
        Systemic velocity.
    vcirc : float
        Circular velocity.
    vel_rscale : float
        Scale radius.
    """

    PARAMETER_NAMES = ('cosi', 'theta_int', 'g1', 'g2', 'v0', 'vcirc', 'vel_rscale')

    @property
    def name(self) -> str:
        return 'centered'

    def evaluate_circular_velocity(
        self,
        theta: jnp.ndarray,
        x: jnp.ndarray,
        y: jnp.ndarray,
        z: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Evaluate arctant rotation curve in disk plane.

        v_circ(r) = (2/π) * vcirc * arctan(r / rscale)
        """
        vcirc = self.get_param('vcirc', theta)
        rscale = self.get_param('vel_rscale', theta)

        # circular radius in (centered) disk plane
        r = jnp.sqrt(x**2 + y**2)

        # arctan rotation curve
        v_circ = (2.0 / jnp.pi) * vcirc * jnp.arctan(r / rscale)

        return v_circ


class OffsetVelocityModel(VelocityModel):
    """
    Velocity model with spatial offset from the origin.

    Parameters
    ----------
    cosi : float
        Cosine of inclination angle.
    theta_int : float
        Intrinsic position angle.
    g1 : float
        First component of the shear
    g2 : float
        Second component of the shear
    v0 : float
        Systemic velocity.
    vcirc : float
        Circular velocity.
    vel_rscale : float
        Scale radius.
    vel_x0 : float
        X-coordinate offset for the velocity image.
    vel_y0 : float
        Y-coordinate offset for the velocity image.
    """

    PARAMETER_NAMES = (
        'cosi',
        'theta_int',
        'g1',
        'g2',
        'v0',
        'vcirc',
        'vel_rscale',
        'vel_x0',
        'vel_y0',
    )

    @property
    def name(self) -> str:
        return 'offset'

    def evaluate_circular_velocity(
        self,
        theta: jnp.ndarray,
        x: jnp.ndarray,
        y: jnp.ndarray,
        z: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Evaluate arctan rotation curve in disk plane.

        v_circ(r) = (2/π) * vcirc * arctan(r / rscale)
        """
        vcirc = self.get_param('vcirc', theta)
        rscale = self.get_param('vel_rscale', theta)

        # circular radius in (centered) disk plane
        r = jnp.sqrt(x**2 + y**2)

        # arctan rotation curve
        v_circ = (2.0 / jnp.pi) * vcirc * jnp.arctan(r / rscale)

        return v_circ


VELOCITY_MODEL_TYPES = {
    'default': OffsetVelocityModel,
    'centered': CenteredVelocityModel,
    'offset': OffsetVelocityModel,
}


def get_velocity_model_types():
    """
    Get dictionary of registered velocity model types.

    Returns
    -------
    dict
        Mapping from model name strings to velocity model classes.
    """
    return VELOCITY_MODEL_TYPES


def build_velocity_model(
    name: str,
    meta_pars: dict = None,
) -> VelocityModel:
    """
    Factory function for constructing velocity models by name.

    Parameters
    ----------
    name : str
        Name of the model to construct (case-insensitive).
    meta_pars : dict, optional
        Fixed metadata for the model.

    Returns
    -------
    VelocityModel
        Instantiated velocity model.

    Raises
    ------
    ValueError
        If the specified model name is not registered.
    """

    name = name.lower()

    if name not in VELOCITY_MODEL_TYPES:
        raise ValueError(f'{name} is not a registered velocity model!')

    return VELOCITY_MODEL_TYPES[name](meta_pars=meta_pars)
