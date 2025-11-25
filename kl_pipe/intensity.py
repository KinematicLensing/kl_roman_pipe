import jax.numpy as jnp

from kl_pipe.model import IntensityModel


class InclinedExponentialModel(IntensityModel):
    """
    Exponential disk intensity model, with possible inclination and shear. Intrinsically
    assumes circular exponential profile in disk plane:
        I(r) = I0 * exp(-r / r_scale)
    where r is the circular radius in the disk plane. This corresponds to a Sersic
    profile with n=1.

    New approach uses a flux-based parameterization where flux is conserved under
    projection. Includes cos(i) surface brightness correction.

    Parameters
    ----------
    cosi : float
        Cosine of inclination (0=edge-on, 1=face-on)
    theta_int : float
        Position angle (radians)
    g1, g2 : float
        Shear components
    flux : float
        Total integrated flux (conserved quantity)
    int_rscale : float
        Exponential scale length
    int_x0, int_y0 : float
        Centroid position
    """

    PARAMETER_NAMES = (
        'cosi',
        'theta_int',
        'g1',
        'g2',
        'flux',
        'int_rscale',
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
        Evaluate exponential profile in disk plane, then project to observer.

        Steps:
        1. Extract flux and scale length
        2. Convert flux to I0: I0 = flux / (2π * r_scale²)
        3. Evaluate exponential in disk plane
        4. Apply cos(i) projection: I_obs = I_disk / cos(i)

        Parameters
        ----------
        theta : jnp.ndarray
            Model parameters.
        x, y : jnp.ndarray
            Coordinates in disk plane.
        z : jnp.ndarray, optional
            Currently unused for this intensity model.
        """

        flux = self.get_param('flux', theta)
        rscale = self.get_param('int_rscale', theta)
        cosi = self.get_param('cosi', theta)

        # compute radius in disk plane
        r_disk = jnp.sqrt(x**2 + y**2)

        # convert flux to central surface brightness
        #   for exponential: F = 2π * I0 * r_scale²
        I0_disk = flux / (2.0 * jnp.pi * rscale**2)

        # evaluate exponential profile in disk plane
        intensity_disk = I0_disk * jnp.exp(-r_disk / rscale)

        # NOTE: no inclination correction here to flux, as this is evaluated in disk
        # plane and cosi correction is done in IntensityModel.__call__() method
        #  i.e. intensity_obs = intensity_disk / cosi

        return intensity_disk


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
