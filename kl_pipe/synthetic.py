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
...     'v0': 10.0, 'vcirc': 200.0, 'vel_rscale': 5.0,
...     'cosi': 0.8, 'theta_int': 0.785,
...     'g1': 0.0, 'g2': 0.0, 'vel_x0': 0.0, 'vel_y0': 0.0
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
...     'I0': 1.0, 'vel_rscale': 3.0, 'n_sersic': 1.0,
...     'cosi': 0.8, 'theta_int': 0.785,
...     'g1': 0.0, 'g2': 0.0, 'int_x0': 0.0, 'int_y0': 0.0
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
    'arctan': {
        'v0',
        'vcirc',
        'vel_rscale',
        'cosi',
        'theta_int',
        'g1',
        'g2',
        'vel_x0',
        'vel_y0',
    },
    'sersic': {
        'I0',
        'int_rscale',
        'n_sersic',
        'cosi',
        'theta_int',
        'g1',
        'g2',
        'int_x0',
        'int_y0',
    },
    'exponential': {
        'I0',
        'int_rscale',
        'cosi',
        'theta_int',
        'g1',
        'g2',
        'int_x0',
        'int_y0',
    },
}


# ==============================================================================
# Velocity field generators
# ==============================================================================


def generate_arctan_velocity_2d(
    X: np.ndarray,
    Y: np.ndarray,
    v0: float,
    vcirc: float,
    vel_rscale: float,  # UPDATED: was 'rscale'
    cosi: float,
    theta_int: float,
    g1: float = 0.0,
    g2: float = 0.0,
    vel_x0: float = 0.0,  # UPDATED: was 'x0'
    vel_y0: float = 0.0,  # UPDATED: was 'y0'
) -> np.ndarray:
    """
    Generate arctan rotation curve velocity field.

    UPDATED: Parameter names now match VelocityModel classes.

    Parameters
    ----------
    X, Y : ndarray
        Coordinate grids in consistent units (e.g., arcsec, kpc, pixels).
    v0 : float
        Systemic velocity in km/s.
    vcirc : float
        Asymptotic circular velocity in km/s.
    vel_rscale : float
        Scale radius for rotation curve, same units as X, Y.
    cosi: float
        Cosine of inclination angle (0=face-on, 1=edge-on).
    theta_int : float
        Intrinsic position angle in radians.
    g1, g2 : float, optional
        Shear components. Default is 0.0 (no shear).
    vel_x0, vel_y0 : float, optional
        Centroid offsets, same units as X, Y. Default is 0.0.

    Returns
    -------
    ndarray
        Line-of-sight velocity map in km/s, same shape as X and Y.
    """

    # Implementation same as before, just using new parameter names
    sini = np.sqrt(1.0 - cosi**2)

    # Step 1: recenter
    X_c = X - vel_x0  # UPDATED: was x0
    Y_c = Y - vel_y0  # UPDATED: was y0

    # Step 2: apply shear
    X_shear = X_c + g1 * X_c - g2 * Y_c
    Y_shear = Y_c + g2 * X_c + g1 * Y_c

    # Step 3: rotate by position angle
    cos_pa = np.cos(theta_int)
    sin_pa = np.sin(theta_int)
    X_rot = X_shear * cos_pa + Y_shear * sin_pa
    Y_rot = -X_shear * sin_pa + Y_shear * cos_pa

    # Step 4: deproject inclination to get disk-plane coordinates
    X_disk = X_rot
    Y_disk = Y_rot / sini if sini > 0 else Y_rot

    # Compute radius in disk plane
    r_disk = np.sqrt(X_disk**2 + Y_disk**2)

    # Evaluate arctan rotation curve
    v_circ = (
        (2.0 / np.pi) * vcirc * np.arctan(r_disk / vel_rscale)
    )  # UPDATED: was rscale

    # Project to line-of-sight
    phi = np.arctan2(Y_disk, X_disk)
    v_los = sini * np.cos(phi) * v_circ

    return v0 + v_los


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
    int_rscale: float,  # UPDATED: was 'rscale'
    n_sersic: float,
    cosi: float,
    theta_int: float,
    g1: float = 0.0,
    g2: float = 0.0,
    int_x0: float = 0.0,  # UPDATED: was 'x0'
    int_y0: float = 0.0,  # UPDATED: was 'y0'
    backend: str = 'scipy',
) -> np.ndarray:
    """
    Generate Sersic intensity profile.

    UPDATED: Parameter names now match IntensityModel classes.

    Parameters
    ----------
    X, Y : ndarray
        Coordinate grids.
    I0 : float
        Central intensity.
    int_rscale : float
        Scale radius.
    n_sersic : float
        Sersic index (n=1 for exponential, n=4 for de Vaucouleurs).
    cosi : float
        Cosine of inclination angle.
    theta_int : float
        Intrinsic position angle in radians.
    g1, g2 : float, optional
        Shear components.
    int_x0, int_y0 : float, optional
        Centroid offsets.
    backend : str, optional
        Backend for computation ('scipy' or 'galsim'). Default is 'scipy'.

    Returns
    -------
    ndarray
        Intensity map.
    """

    if backend == 'galsim':
        return _generate_sersic_galsim(
            X, Y, I0, int_rscale, n_sersic, cosi, theta_int, g1, g2, int_x0, int_y0
        )
    else:
        return _generate_sersic_scipy(
            X, Y, I0, int_rscale, n_sersic, cosi, theta_int, g1, g2, int_x0, int_y0
        )


def _generate_sersic_scipy(
    X: np.ndarray,
    Y: np.ndarray,
    I0: float,
    int_rscale: float,  # UPDATED
    n_sersic: float,
    cosi: float,
    theta_int: float,
    g1: float,
    g2: float,
    int_x0: float,  # UPDATED
    int_y0: float,  # UPDATED
) -> np.ndarray:
    """Generate Sersic profile using scipy (simple implementation)."""

    sini = np.sqrt(1.0 - cosi**2)

    # Step 1: recenter
    X_c = X - int_x0  # UPDATED
    Y_c = Y - int_y0  # UPDATED

    # Step 2: apply shear
    X_shear = X_c + g1 * X_c - g2 * Y_c
    Y_shear = Y_c + g2 * X_c + g1 * Y_c

    # Step 3: rotate by position angle
    cos_pa = np.cos(theta_int)
    sin_pa = np.sin(theta_int)
    X_rot = X_shear * cos_pa + Y_shear * sin_pa
    Y_rot = -X_shear * sin_pa + Y_shear * cos_pa

    # Step 4: deproject inclination to get disk-plane coordinates
    X_disk = X_rot
    Y_disk = Y_rot / sini if sini > 0 else Y_rot

    # Compute radius in disk plane
    r_disk = np.sqrt(X_disk**2 + Y_disk**2)

    # Evaluate Sersic profile
    intensity = I0 * np.exp(-np.power(r_disk / int_rscale, 1.0 / n_sersic))  # UPDATED

    return intensity


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


def add_noise(
    image: np.ndarray,
    target_snr: float,
    seed: Optional[int] = None,
    include_poisson: bool = True,
    poisson_scale: float = 1.0,
    return_variance: bool = True,
) -> Tuple[np.ndarray, float] | np.ndarray:
    """
    Add realistic noise to achieve target total signal-to-noise ratio.

    Can include both Poisson (shot) noise and Gaussian (read) noise.

    The total S/N is defined as:
        S/N = sqrt(sum(signal^2)) / sqrt(sum(noise^2))

    Parameters
    ----------
    image : ndarray
        Input noiseless image.
    target_snr : float
        Target total signal-to-noise ratio.
    seed : int, optional
        Random seed for reproducibility.
    include_poisson : bool, optional
        If True, include Poisson (shot) noise. If False, only Gaussian noise.
        Default is True.
    poisson_scale : float, optional
        Scale factor to convert image values to photon counts for Poisson noise.
        Higher values = more shot noise. Default is 1.0.
    return_variance : bool, optional
        If True, return both noisy image and variance used.
        If False, return only noisy image. Default is True.

    Returns
    -------
    noisy_image : ndarray
        Image with noise added.
    variance : float, optional
        Effective variance of the noise (returned only if return_variance=True).

    Notes
    -----
    When include_poisson=True:
        - Poisson noise is added: sqrt(counts) per pixel
        - Gaussian read noise is added to reach target SNR

    When include_poisson=False:
        - Only Gaussian noise with constant variance
        - Matches the original add_gaussian_noise behavior

    Examples
    --------
    >>> image = np.random.rand(64, 64) * 100  # counts
    >>>
    >>> # With Poisson + Gaussian noise (more realistic)
    >>> noisy, var = add_noise(image, target_snr=50, include_poisson=True)
    >>>
    >>> # Only Gaussian noise (simpler, for testing)
    >>> noisy, var = add_noise(image, target_snr=50, include_poisson=False)
    """

    rng = np.random.default_rng(seed)

    if include_poisson:
        # Convert image to photon counts
        counts = np.abs(image) * poisson_scale

        # Add Poisson noise (shot noise)
        # For negative values, we'll handle them as if they're background-subtracted
        noisy_counts = np.where(
            counts > 0,
            rng.poisson(counts),
            counts + rng.normal(0, np.sqrt(np.abs(counts)), counts.shape),
        )

        # Convert back to original units
        noisy_image = noisy_counts / poisson_scale

        # Compute variance from Poisson noise
        poisson_var = np.mean(np.abs(image) / poisson_scale)

        # Calculate how much Gaussian noise to add to reach target SNR
        total_signal = np.sqrt(np.sum(image**2))
        target_noise_power = total_signal / target_snr

        # Current noise power from Poisson
        current_noise_power = np.sqrt(np.sum((noisy_image - image) ** 2))

        # Additional Gaussian noise needed
        if target_noise_power > current_noise_power:
            additional_noise_power = np.sqrt(
                target_noise_power**2 - current_noise_power**2
            )
            n_pixels = image.size
            sigma_gaussian = additional_noise_power / np.sqrt(n_pixels)

            gaussian_noise = rng.normal(0, sigma_gaussian, image.shape)
            noisy_image += gaussian_noise

            # Effective variance (combination of Poisson and Gaussian)
            variance = poisson_var + sigma_gaussian**2
        else:
            # Poisson noise alone is sufficient
            variance = poisson_var

    else:
        # Original behavior: Gaussian noise only
        total_signal = np.sqrt(np.sum(image**2))
        target_noise_power = total_signal / target_snr

        # Per-pixel noise stddev (constant variance)
        n_pixels = image.size
        sigma_per_pixel = target_noise_power / np.sqrt(n_pixels)

        # Add Gaussian noise
        noise = rng.normal(0, sigma_per_pixel, image.shape)
        noisy_image = image + noise

        variance = sigma_per_pixel**2

    if return_variance:
        return noisy_image, variance
    else:
        return noisy_image


# Backward compatibility alias
add_gaussian_noise = lambda *args, **kwargs: add_noise(
    *args, include_poisson=False, **kwargs
)


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


class SyntheticVelocity:
    """
    Synthetic velocity field observations.

    UPDATED: Now uses model-matching parameter names and supports Poisson noise.
    """

    def __init__(
        self,
        true_params: Dict[str, float],
        model_type: str = 'arctan',
        seed: Optional[int] = None,
    ):
        self.true_params = true_params
        self.model_type = model_type
        self.seed = seed

        # Storage for last generated data
        self.X = None
        self.Y = None
        self.data_true = None
        self.data_noisy = None
        self.variance = None

        # Validate parameters
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
        include_poisson: bool = True,  # UPDATED: new parameter
    ) -> np.ndarray:
        """
        Generate synthetic velocity data.

        UPDATED: Now supports optional Poisson noise.

        Parameters
        ----------
        X, Y : ndarray
            Coordinate grids in consistent units.
        snr : float
            Target signal-to-noise ratio (total S/N).
        seed : int, optional
            Random seed for noise generation. If None, uses self.seed.
        include_poisson : bool, optional
            Whether to include Poisson (shot) noise. Default is True.

        Returns
        -------
        ndarray
            Noisy velocity map in km/s.
        """

        if seed is None:
            seed = self.seed

        # Generate true velocity field
        if self.model_type == 'arctan':
            self.data_true = generate_arctan_velocity_2d(X, Y, **self.true_params)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # Add noise
        self.data_noisy, self.variance = add_noise(
            self.data_true,
            snr,
            seed=seed,
            include_poisson=include_poisson,
            return_variance=True,
        )

        # Store grids
        self.X = X
        self.Y = Y

        return self.data_noisy


class SyntheticIntensity:
    """
    Synthetic intensity/surface brightness observations.

    UPDATED: Now uses model-matching parameter names and supports Poisson noise.
    """

    def __init__(
        self,
        true_params: Dict[str, float],
        model_type: str = 'sersic',
        sersic_backend: str = 'scipy',
        seed: Optional[int] = None,
    ):
        self.true_params = true_params
        self.model_type = model_type
        self.sersic_backend = sersic_backend
        self.seed = seed

        # Storage for last generated data
        self.X = None
        self.Y = None
        self.data_true = None
        self.data_noisy = None
        self.variance = None

        # Validate parameters
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
        include_poisson: bool = True,  # UPDATED: new parameter
    ) -> np.ndarray:
        """
        Generate synthetic intensity data.

        UPDATED: Now supports optional Poisson noise.

        Parameters
        ----------
        X, Y : ndarray
            Coordinate grids in consistent units.
        snr : float
            Target signal-to-noise ratio (total S/N).
        seed : int, optional
            Random seed for noise generation. If None, uses self.seed.
        include_poisson : bool, optional
            Whether to include Poisson (shot) noise. Default is True.

        Returns
        -------
        ndarray
            Noisy intensity map.
        """

        if seed is None:
            seed = self.seed

        # Generate true intensity field
        if self.model_type == 'sersic':
            self.data_true = generate_sersic_intensity_2d(
                X, Y, backend=self.sersic_backend, **self.true_params
            )
        elif self.model_type == 'exponential':
            # Exponential is Sersic with n=1
            params = self.true_params.copy()
            params['n_sersic'] = 1.0
            self.data_true = generate_sersic_intensity_2d(
                X, Y, backend='scipy', **params
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # Add noise
        self.data_noisy, self.variance = add_noise(
            self.data_true,
            snr,
            seed=seed,
            include_poisson=include_poisson,
            return_variance=True,
        )

        # Store grids
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
