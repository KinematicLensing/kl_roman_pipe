import jax.numpy as jnp

from kl_pipe.model import IntensityModel


class InclinedExponentialModel(IntensityModel):
    """
    Exponential intensity profile.

    Parameters
    ----------
    I0 : float
        Central intensity.
    rscale : float
        Exponential scale length.
    int_x0 : float
        X-coordinate offset for the intensity image.
    int_y0 : float
        Y-coordinate offset for the intensity image.
    """

    PARAMETER_NAMES = (
        'I0',
        'rscale',
        'sini',
        'theta_int',
        'g1',
        'g2',
        'int_x0',
        'int_y0',
    )

    @property
    def name(self) -> str:
        return 'inclined_exp'

    def evaluate_in_disk_plane(
        self,
        theta: jnp.ndarray,
        x: jnp.ndarray,
        y: jnp.ndarray,
        z: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Evaluate exponential profile in disk plane (circularly symmetric).
        """

        I0 = self.get_param('I0', theta)
        scale = self.get_param('rscale', theta)

        r = jnp.sqrt(x**2 + y**2)
        intensity_map = I0 * jnp.exp(-r / scale)

        return intensity_map


INTENSITY_MODEL_TYPES = {
    'default': InclinedExponentialModel,
    'inclined_exp': InclinedExponentialModel,
}


def get_intensity_model_types():
    """
    Get dictionary of registered intensity model types.

    Returns
    -------
    dict
        Mapping from model name strings to intensity model classes.
    """
    return INTENSITY_MODEL_TYPES


def build_intensity_model(
    name: str,
    meta_pars: dict = None,
) -> IntensityModel:
    """
    Factory function for constructing intensity models by name.

    Parameters
    ----------
    name : str
        Name of the model to construct (case-insensitive).
    meta_pars : dict, optional
        Fixed metadata for the model.

    Returns
    -------
    IntensityModel
        Instantiated intensity model.

    Raises
    ------
    ValueError
        If the specified model name is not registered.
    """

    name = name.lower()

    if name not in INTENSITY_MODEL_TYPES:
        raise ValueError(f'{name} is not a registered intensity model!')

    return INTENSITY_MODEL_TYPES[name](meta_pars=meta_pars)
