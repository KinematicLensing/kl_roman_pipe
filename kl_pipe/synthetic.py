"""
Synthetic data generation for testing and validation.

Provides simple, independent implementations of galaxy models that do NOT use
the main kl_pipe modeling code, to enable proper regression testing and
parameter recovery validation.

The generators implement transformations (inclination, position angle, shear)
directly without relying on kl_pipe.transformation, ensuring independence for
testing purposes.

Examples
--------
Generate synthetic velocity data:

>>> from kl_pipe.synthetic import SyntheticVelocity
>>> import numpy as np
>>>
>>> # Define true parameters
>>> true_params = {
...     'v0': 10.0, 'vcirc': 200.0, 'rscale': 5.0,
...     'cosi': 0.8, 'theta_int': 0.785,
...     'g1': 0.0, 'g2': 0.0, 'x0': 0.0, 'y0': 0.0
... }
>>>
>>> # Create synthetic observation
>>> synth_vel = SyntheticVelocity(true_params, model_type='arctan', seed=42)
>>>
>>> # Generate data on a grid
>>> X, Y = np.meshgrid(np.linspace(-10, 10, 64), np.linspace(-10, 10, 64))
>>> data_noisy = synth_vel.generate(X, Y, snr=50)
>>>
>>> # Access results
>>> print(synth_vel.data_true)  # Noiseless data
>>> print(synth_vel.variance)   # Noise variance used

Generate synthetic intensity data:

>>> from kl_pipe.synthetic import SyntheticIntensity
>>>
>>> true_params = {
...     'I0': 1.0, 'rscale': 3.0, 'n_sersic': 1.0,
...     'cosi': 0.8, 'theta_int': 0.785,
...     'g1': 0.0, 'g2': 0.0, 'x0': 0.0, 'y0': 0.0
... }
>>>
>>> synth_int = SyntheticIntensity(true_params, model_type='sersic', seed=42)
>>> data_noisy = synth_int.generate(X, Y, snr=100)
"""

import numpy as np
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional

# Required parameters for each model type
REQUIRED_PARAMS = {
    'arctan': {'v0', 'vcirc', 'rscale', 'cosi', 'theta_int', 'g1', 'g2', 'x0', 'y0'},
    'sersic': {'I0', 'rscale', 'n_sersic', 'cosi', 'theta_int', 'g1', 'g2', 'x0', 'y0'},
    'exponential': {'I0', 'rscale', 'cosi', 'theta_int', 'g1', 'g2', 'x0', 'y0'},
}


# ==============================================================================
# Velocity field generators
# ==============================================================================


def generate_arctan_velocity_2d(
    X: np.ndarray,
    Y: np.ndarray,
    v0: float,
    vcirc: float,
    rscale: float,
    cosi: float,
    theta_int: float,
    g1: float = 0.0,
    g2: float = 0.0,
    x0: float = 0.0,
    y0: float = 0.0,
) -> np.ndarray:
    """
    Generate arctan rotation curve velocity field.

    Implements a simple, independent calculation of the velocity field with
    coordinate transformations for inclination, position angle, and shear.
    The rotation curve follows v_circ(r) = (2/π) * vcirc * arctan(r/rscale).

    Parameters
    ----------
    X, Y : ndarray
        Coordinate grids in consistent units (e.g., arcsec, kpc, pixels).
    v0 : float
        Systemic velocity in km/s.
    vcirc : float
        Asymptotic circular velocity in km/s.
    rscale : float
        Scale radius for rotation curve, same units as X, Y.
    cosi: float
        Cosine of inclination angle (0=face-on, 1=edge-on).
    theta_int : float
        Intrinsic position angle in radians.
    g1, g2 : float, optional
        Shear components. Default is 0.0 (no shear).
    x0, y0 : float, optional
        Centroid offsets, same units as X, Y. Default is 0.0.

    Returns
    -------
    ndarray
        Line-of-sight velocity map in km/s, same shape as X and Y.

    Notes
    -----
    All spatial parameters (X, Y, rscale, x0, y0) must be in the same units.
    The function works with any consistent unit system.

    This is an independent implementation that does not use kl_pipe.transformation,
    ensuring proper regression testing.
    """

    # cosi easier to sample over, sini easier to use in calculations
    sini = np.sqrt(1.0 - cosi**2)

    # step 1: recenter
    X_c = X - x0
    Y_c = Y - y0

    # step 2: apply shear (simple implementation)
    X_shear = X_c + g1 * X_c - g2 * Y_c
    Y_shear = Y_c + g2 * X_c + g1 * Y_c

    # step 3: rotate by position angle
    cos_pa = np.cos(theta_int)
    sin_pa = np.sin(theta_int)
    X_rot = X_shear * cos_pa + Y_shear * sin_pa
    Y_rot = -X_shear * sin_pa + Y_shear * cos_pa

    # step 4: deproject inclination to get disk-plane coordinates
    X_disk = X_rot
    Y_disk = Y_rot / sini if sini > 0 else Y_rot

    # evaluate rotation curve in disk plane
    r_disk = np.sqrt(X_disk**2 + Y_disk**2)
    v_circ = (2.0 / np.pi) * vcirc * np.arctan(r_disk / rscale)

    # project to line-of-sight velocity
    phi = np.arctan2(Y_disk, X_disk)
    v_los = v0 + sini * np.cos(phi) * v_circ

    return v_los


# TODO: when we're ready to test more complex velocity models
def generate_arctan_velocity_3d():
    pass


# ==============================================================================
# Intensity profile generators
# ==============================================================================


def generate_sersic_intensity_2d(
    X: np.ndarray,
    Y: np.ndarray,
    I0: float,
    rscale: float,
    n_sersic: float,
    cosi: float,
    theta_int: float,
    g1: float = 0.0,
    g2: float = 0.0,
    x0: float = 0.0,
    y0: float = 0.0,
    backend: str = 'scipy',
) -> np.ndarray:
    """
    Generate Sersic intensity profile.

    The profile follows I(r) ~ exp(-(r/rscale)^(1/n)) after accounting for
    inclination, position angle, and shear transformations.

    Parameters
    ----------
    X, Y : ndarray
        Coordinate grids in consistent units (e.g., arcsec, kpc, pixels).
    I0 : float
        Central intensity normalization.
    rscale : float
        Scale radius in the Sersic profile I(r) ~ exp(-(r/rscale)^(1/n)).
        Same units as X, Y. This is the r_0 parameter we fit in our models.
    n_sersic : float
        Sersic index. n=1 gives exponential, n=4 gives de Vaucouleurs.
    cosi: float
        Cosine of inclination angle (0=face-on, 1=edge-on).
    theta_int : float
        Intrinsic position angle in radians.
    g1, g2 : float, optional
        Shear components. Default is 0.0 (no shear).
    x0, y0 : float, optional
        Centroid offsets, same units as X, Y. Default is 0.0.
    backend : str, optional
        Backend to use: 'scipy' (default) or 'galsim'.

    Returns
    -------
    ndarray
        Intensity map, same shape as X and Y.

    Notes
    -----
    All spatial parameters (X, Y, rscale, x0, y0) must be in the same units.

    The 'scipy' backend uses direct calculation with numpy/scipy.
    The 'galsim' backend uses GalSim for profile generation (requires galsim).

    This is an independent implementation for regression testing.
    """

    if backend == 'galsim':
        return _generate_sersic_galsim(
            X, Y, I0, rscale, n_sersic, cosi, theta_int, g1, g2, x0, y0
        )
    elif backend == 'scipy':
        return _generate_sersic_scipy(
            X, Y, I0, rscale, n_sersic, cosi, theta_int, g1, g2, x0, y0
        )
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use 'scipy' or 'galsim'.")


def _generate_sersic_scipy(
    X: np.ndarray,
    Y: np.ndarray,
    I0: float,
    rscale: float,
    n_sersic: float,
    cosi: float,
    theta_int: float,
    g1: float,
    g2: float,
    x0: float,
    y0: float,
) -> np.ndarray:
    """Generate Sersic profile using scipy/numpy."""

    sini = np.sqrt(1.0 - cosi**2)

    # step 1: recenter
    X_c = X - x0
    Y_c = Y - y0

    # step 2: apply shear
    X_shear = X_c + g1 * X_c - g2 * Y_c
    Y_shear = Y_c + g2 * X_c + g1 * Y_c

    # step 3: rotate by position angle
    cos_pa = np.cos(theta_int)
    sin_pa = np.sin(theta_int)
    X_rot = X_shear * cos_pa + Y_shear * sin_pa
    Y_rot = -X_shear * sin_pa + Y_shear * cos_pa

    # step 4: deproject inclination to get disk-plane coordinates
    X_disk = X_rot
    Y_disk = Y_rot / sini if sini > 0 else Y_rot

    # compute radius in disk plane
    r_disk = np.sqrt(X_disk**2 + Y_disk**2)

    # evaluate Sersic profile: I(r) = I0 * exp(-(r/rscale)^(1/n))
    intensity = I0 * np.exp(-np.power(r_disk / rscale, 1.0 / n_sersic))

    return intensity


def _generate_sersic_galsim(
    X: np.ndarray,
    Y: np.ndarray,
    I0: float,
    rscale: float,
    n_sersic: float,
    cosi: float,
    theta_int: float,
    g1: float,
    g2: float,
    x0: float,
    y0: float,
) -> np.ndarray:
    """Generate Sersic profile using GalSim."""

    raise NotImplementedError(
        "GalSim backend for Sersic profile not yet implemented. "
        "Use backend='scipy' instead."
    )


# ==============================================================================
# Noise generation
# ==============================================================================


def add_gaussian_noise(
    image: np.ndarray,
    target_snr: float,
    seed: Optional[int] = None,
    return_variance: bool = True,
) -> Tuple[np.ndarray, float] | np.ndarray:
    """
    Add Gaussian noise to achieve target total signal-to-noise ratio.

    The total S/N is defined as:
        S/N = sqrt(sum(signal^2)) / sqrt(sum(noise^2))

    The noise is Gaussian with constant variance across the image.

    Parameters
    ----------
    image : ndarray
        Input noiseless image.
    target_snr : float
        Target total signal-to-noise ratio.
    seed : int, optional
        Random seed for reproducibility.
    return_variance : bool, optional
        If True, return both noisy image and variance used.
        If False, return only noisy image. Default is True.

    Returns
    -------
    noisy_image : ndarray
        Image with Gaussian noise added.
    variance : float, optional
        Variance of the noise (returned only if return_variance=True).

    Examples
    --------
    >>> image = np.random.rand(64, 64)
    >>> noisy, var = add_gaussian_noise(image, target_snr=50, seed=42)
    >>> print(f"Noise variance: {var:.6f}")
    """

    rng = np.random.default_rng(seed)

    # calculate required noise level for target total S/N
    total_signal = np.sqrt(np.sum(image**2))
    target_noise_power = total_signal / target_snr

    # per-pixel noise stddev (constant variance)
    n_pixels = image.size
    sigma_per_pixel = target_noise_power / np.sqrt(n_pixels)

    # add Gaussian noise
    noise = rng.normal(0, sigma_per_pixel, image.shape)
    noisy_image = image + noise

    if return_variance:
        return noisy_image, sigma_per_pixel**2
    else:
        return noisy_image


# ==============================================================================
# Synthetic observation classes
# ==============================================================================


class SyntheticObservation(ABC):
    """
    Base class for synthetic observations.

    Provides structure for generating synthetic data with known true parameters,
    adding noise, and storing results for testing and validation.

    Parameters
    ----------
    true_params : dict
        Dictionary of true model parameters.
    seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    true_params : dict
        True parameters used to generate data.
    X, Y : ndarray or None
        Coordinate grids from the most recent call to generate().
        These are updated each time generate() is called with new grids.
    data_true : ndarray or None
        Noiseless synthetic data from most recent generation.
    data_noisy : ndarray or None
        Noisy synthetic data from most recent generation.
    variance : float or None
        Noise variance from most recent generation.
    """

    def __init__(self, true_params: Dict[str, float], seed: Optional[int] = None):
        self.true_params = true_params
        self.seed = seed

        # storage for last generated data
        self.X = None
        self.Y = None
        self.data_true = None
        self.data_noisy = None
        self.variance = None

    @abstractmethod
    def generate(
        self, X: np.ndarray, Y: np.ndarray, snr: float, seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate synthetic data on specified grid with noise.

        Parameters
        ----------
        X, Y : ndarray
            Coordinate grids.
        snr : float
            Target signal-to-noise ratio (total S/N).
        seed : int, optional
            Random seed for noise generation. If None, uses self.seed.

        Returns
        -------
        ndarray
            Noisy synthetic data.
        """
        pass


class SyntheticVelocity(SyntheticObservation):
    """
    Synthetic velocity field observations.

    Generates velocity maps using specified kinematic models with known
    true parameters, optionally adding noise.

    Parameters
    ----------
    true_params : dict
        True velocity model parameters. Required keys depend on model_type.
    model_type : str, optional
        Velocity model type. Options: 'arctan'. Default is 'arctan'.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> true_params = {
    ...     'v0': 10.0, 'vcirc': 200.0, 'rscale': 5.0,
    ...     'cosi': 0.6, 'theta_int': 0.785,
    ...     'g1': 0.0, 'g2': 0.0, 'x0': 0.0, 'y0': 0.0
    ... }
    >>> synth = SyntheticVelocity(true_params, model_type='arctan')
    >>> X, Y = np.meshgrid(np.linspace(-10, 10, 64), np.linspace(-10, 10, 64))
    >>> data = synth.generate(X, Y, snr=50, seed=42)
    """

    def __init__(
        self,
        true_params: Dict[str, float],
        model_type: str = 'arctan',
        seed: Optional[int] = None,
    ):
        super().__init__(true_params, seed)
        self.model_type = model_type

        # validate parameters
        self._validate_params()

    def _validate_params(self):
        """Validate that required parameters are present."""
        if self.model_type not in REQUIRED_PARAMS:
            raise ValueError(
                f"Unknown model_type '{self.model_type}'. "
                f"Available: {list(REQUIRED_PARAMS.keys())}"
            )

        required = REQUIRED_PARAMS[self.model_type]
        provided = set(self.true_params.keys())
        missing = required - provided

        if missing:
            raise ValueError(
                f"Missing required parameters for {self.model_type}: {missing}"
            )

    def generate(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        snr: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate synthetic velocity data.

        Parameters
        ----------
        X, Y : ndarray
            Coordinate grids in consistent units.
        snr : float
            Target signal-to-noise ratio (total S/N).
        seed : int, optional
            Random seed for noise generation. If None, uses self.seed.

        Returns
        -------
        ndarray
            Noisy velocity map in km/s.
        """

        # use instance seed if none provided
        if seed is None:
            seed = self.seed

        # dispatch to appropriate generator
        if self.model_type == 'arctan':
            self.data_true = generate_arctan_velocity_2d(X, Y, **self.true_params)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # add noise
        self.data_noisy, self.variance = add_gaussian_noise(
            self.data_true, snr, seed=seed, return_variance=True
        )

        # store grids (from most recent call)
        self.X = X
        self.Y = Y

        return self.data_noisy


class SyntheticIntensity(SyntheticObservation):
    """
    Synthetic intensity/surface brightness observations.

    Generates intensity maps using specified surface brightness profiles
    with known true parameters, optionally adding noise.

    Parameters
    ----------
    true_params : dict
        True intensity model parameters. Required keys depend on model_type.
    model_type : str, optional
        Intensity model type. Options: 'sersic', 'exponential'.
        Default is 'sersic'.
    sersic_backend : str, optional
        Backend for Sersic profile generation. Options: 'scipy', 'galsim'.
        Default is 'scipy'. Only used if model_type='sersic'.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> true_params = {
    ...     'I0': 1.0, 'rscale': 3.0, 'n_sersic': 1.0,
    ...     'cosi': 0.6, 'theta_int': 0.785,
    ...     'g1': 0.0, 'g2': 0.0, 'x0': 0.0, 'y0': 0.0
    ... }
    >>> synth = SyntheticIntensity(true_params, model_type='sersic')
    >>> X, Y = np.meshgrid(np.linspace(-10, 10, 64), np.linspace(-10, 10, 64))
    >>> data = synth.generate(X, Y, snr=100, seed=42)
    """

    def __init__(
        self,
        true_params: Dict[str, float],
        model_type: str = 'sersic',
        sersic_backend: str = 'scipy',
        seed: Optional[int] = None,
    ):
        super().__init__(true_params, seed)
        self.model_type = model_type
        self.sersic_backend = sersic_backend

        # validate parameters
        self._validate_params()

    def _validate_params(self):
        """Validate that required parameters are present."""
        if self.model_type not in REQUIRED_PARAMS:
            raise ValueError(
                f"Unknown model_type '{self.model_type}'. "
                f"Available: {list(REQUIRED_PARAMS.keys())}"
            )

        required = REQUIRED_PARAMS[self.model_type]
        provided = set(self.true_params.keys())
        missing = required - provided

        if missing:
            raise ValueError(
                f"Missing required parameters for {self.model_type}: {missing}"
            )

    def generate(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        snr: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate synthetic intensity data.

        Parameters
        ----------
        X, Y : ndarray
            Coordinate grids in consistent units.
        snr : float
            Target signal-to-noise ratio (total S/N).
        seed : int, optional
            Random seed for noise generation. If None, uses self.seed.

        Returns
        -------
        ndarray
            Noisy intensity map.
        """

        # use instance seed if none provided
        if seed is None:
            seed = self.seed

        # dispatch to appropriate generator
        if self.model_type == 'sersic':
            self.data_true = generate_sersic_intensity_2d(
                X, Y, backend=self.sersic_backend, **self.true_params
            )
        elif self.model_type == 'exponential':
            # exponential is Sersic with n=1
            params = self.true_params.copy()
            params['n_sersic'] = 1.0
            self.data_true = generate_sersic_intensity_2d(
                X, Y, backend='scipy', **params
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # add noise
        self.data_noisy, self.variance = add_gaussian_noise(
            self.data_true, snr, seed=seed, return_variance=True
        )

        # store grids (from most recent call)
        self.X = X
        self.Y = Y

        return self.data_noisy


class SyntheticKLObservation:
    """
    Combined kinematic-lensing synthetic observation.

    Bundles together synthetic velocity and intensity observations for
    joint kinematic-lensing analysis testing.

    Parameters
    ----------
    velocity_obs : SyntheticVelocity
        Synthetic velocity observation.
    intensity_obs : SyntheticIntensity
        Synthetic intensity observation.

    Attributes
    ----------
    velocity : SyntheticVelocity
        Velocity component.
    intensity : SyntheticIntensity
        Intensity component.

    Examples
    --------
    >>> vel_params = {...}
    >>> int_params = {...}
    >>> vel_obs = SyntheticVelocity(vel_params)
    >>> int_obs = SyntheticIntensity(int_params)
    >>> kl_obs = SyntheticKLObservation(vel_obs, int_obs)
    >>>
    >>> # Generate both on same or different grids
    >>> X_v, Y_v = ...
    >>> X_i, Y_i = ...
    >>> kl_obs.velocity.generate(X_v, Y_v, snr=50)
    >>> kl_obs.intensity.generate(X_i, Y_i, snr=100)
    """

    def __init__(
        self,
        velocity_obs: SyntheticVelocity,
        intensity_obs: SyntheticIntensity,
    ):
        self.velocity = velocity_obs
        self.intensity = intensity_obs
