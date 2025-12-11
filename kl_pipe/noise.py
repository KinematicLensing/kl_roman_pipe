"""
Noise generation utilities for synthetic observations.

This module provides functions for adding realistic noise to simulated
data, with specialized methods for intensity (photon) and velocity data.
"""

import numpy as np
from typing import Tuple, Optional


def add_intensity_noise(
    intensity: np.ndarray,
    target_snr: float,
    include_poisson: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add noise to intensity/flux maps.

    Adds Poisson photon noise and Gaussian read noise to intensity data.
    Signal-to-noise is defined as total_flux / noise_std.

    Parameters
    ----------
    intensity : np.ndarray
        Input intensity/flux map (should be non-negative)
    target_snr : float
        Target signal-to-noise ratio (total_flux / noise_std)
    include_poisson : bool, default=True
        Whether to include Poisson photon noise. Should generally be True
        for realistic intensity maps.
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    noisy_intensity : np.ndarray
        Intensity map with added noise
    variance : np.ndarray
        Variance map (uniform across image)

    Notes
    -----
    - Poisson noise is sqrt(N) where N is the photon count
    - Gaussian noise is added to reach target SNR
    - Total variance = poisson_variance + gaussian_variance

    Examples
    --------
    >>> intensity = model.generate_intensity_map()
    >>> noisy, var = add_intensity_noise(intensity, target_snr=50)
    """
    if seed is not None:
        np.random.seed(seed)

    # Signal is total integrated flux
    total_flux = np.abs(intensity).sum()
    if total_flux == 0:
        raise ValueError("Cannot add noise to zero-flux intensity map")

    # Initialize with original data
    noisy_data = intensity.copy()

    # Add Poisson noise
    poisson_variance = 0.0
    if include_poisson:
        # Ensure non-negative
        if intensity.min() < 0:
            raise ValueError("Intensity must be non-negative for Poisson noise")

        # For very large values, use Gaussian approximation
        max_lambda = 1e9
        if intensity.max() > max_lambda:
            # Gaussian approximation: std = sqrt(mean)
            poisson_noise = np.random.normal(0, np.sqrt(intensity))
            poisson_variance = intensity.mean()
        else:
            # Standard Poisson sampling
            poisson_counts = np.random.poisson(intensity)
            poisson_noise = poisson_counts - intensity
            poisson_variance = intensity.mean()

        noisy_data += poisson_noise

    # Calculate Gaussian noise needed to reach target SNR
    target_noise_power = (total_flux / target_snr) ** 2
    gaussian_variance = max(0, target_noise_power - poisson_variance)
    gaussian_std = np.sqrt(gaussian_variance)

    # Add Gaussian read noise
    if gaussian_std > 0:
        gaussian_noise = np.random.normal(0, gaussian_std, intensity.shape)
        noisy_data += gaussian_noise

    # Total variance (uniform across map)
    total_variance = gaussian_variance + poisson_variance
    variance = np.full_like(intensity, total_variance)

    return noisy_data, variance


def add_velocity_noise(
    velocity: np.ndarray,
    target_snr: float,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add Gaussian noise to velocity maps.

    Adds only Gaussian noise (no Poisson component) to velocity data.
    Signal-to-noise is defined as velocity_range / noise_std.

    Parameters
    ----------
    velocity : np.ndarray
        Input velocity map (km/s)
    target_snr : float
        Target signal-to-noise ratio (velocity_range / noise_std)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    noisy_velocity : np.ndarray
        Velocity map with added Gaussian noise
    variance : np.ndarray
        Variance map (uniform across image)

    Notes
    -----
    - Velocity measurements have Gaussian uncertainties, not Poisson
    - Signal is defined as max(velocity) - min(velocity)
    - This is appropriate for spectroscopic velocity measurements

    Examples
    --------
    >>> velocity = model.generate_velocity_map()
    >>> noisy, var = add_velocity_noise(velocity, target_snr=100)
    """
    if seed is not None:
        np.random.seed(seed)

    # Signal is velocity range
    velocity_range = velocity.max() - velocity.min()
    if velocity_range == 0:
        # Fallback for constant velocity map
        velocity_range = np.abs(velocity).max()
        if velocity_range == 0:
            raise ValueError("Cannot add noise to constant zero velocity map")

    # Calculate Gaussian noise std to achieve target SNR
    noise_std = velocity_range / target_snr

    # Add Gaussian noise
    gaussian_noise = np.random.normal(0, noise_std, velocity.shape)
    noisy_velocity = velocity + gaussian_noise

    # Variance is constant across map
    variance = np.full_like(velocity, noise_std**2)

    return noisy_velocity, variance


def add_noise(
    data: np.ndarray,
    target_snr: float,
    include_poisson: bool = False,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add noise to data to achieve target signal-to-noise ratio.

    DEPRECATED: Use add_intensity_noise() or add_velocity_noise() instead.

    This function adds Gaussian read noise (and optionally Poisson photon noise)
    to synthetic data to simulate realistic observations. The noise level is
    calibrated to achieve a specified SNR.

    Parameters
    ----------
    data : np.ndarray
        Input data array (e.g., intensity map, velocity map)
    target_snr : float
        Target signal-to-noise ratio. SNR is defined as:
        - For intensity: total_flux / noise_std
        - For velocity: velocity_range / noise_std
    include_poisson : bool, default=False
        Whether to include Poisson noise (appropriate for photon counts).
        If True, assumes data represents photon counts and adds sqrt(N) noise
        before Gaussian noise. If False, only adds Gaussian noise.
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    noisy_data : np.ndarray
        Data with added noise, same shape as input
    variance : np.ndarray
        Variance map (uniform across image), same shape as input

    Notes
    -----
    DEPRECATED: This generic function is kept for backward compatibility.
    New code should use:
    - add_intensity_noise() for flux/intensity maps
    - add_velocity_noise() for velocity maps

    Examples
    --------
    Add noise to intensity map with Poisson:

    >>> intensity = model.generate_intensity_map()
    >>> noisy, var = add_noise(intensity, target_snr=50, include_poisson=True)

    Add noise to velocity map (Gaussian only):

    >>> velocity = model.generate_velocity_map()
    >>> noisy, var = add_noise(velocity, target_snr=100, include_poisson=False)
    """
    if include_poisson:
        return add_intensity_noise(data, target_snr, include_poisson=True, seed=seed)
    else:
        return add_velocity_noise(data, target_snr, seed=seed)


def calculate_snr(data: np.ndarray, noise_std: float, mode: str = 'range') -> float:
    """
    Calculate signal-to-noise ratio for data.

    Parameters
    ----------
    data : np.ndarray
        Data array
    noise_std : float
        Standard deviation of noise
    mode : str, default='range'
        How to define signal:
        - 'range': max - min (for velocity maps)
        - 'total': sum of absolute values (for intensity maps)
        - 'peak': maximum absolute value (for point sources)

    Returns
    -------
    snr : float
        Signal-to-noise ratio
    """
    if mode == 'range':
        signal = np.abs(data).max() - np.abs(data).min()
    elif mode == 'total':
        signal = np.abs(data).sum()
    elif mode == 'peak':
        signal = np.abs(data).max()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return signal / noise_std if noise_std > 0 else np.inf
