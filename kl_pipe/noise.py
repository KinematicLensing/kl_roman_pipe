"""Noise models for synthetic observations.

This module is the home for all noise generation in kl_pipe. It is
intentionally scoped wider than its current contents -- future additions
(correlated read noise, shot-noise-limited datacube channels, sky
background, mask-aware variance maps, etc.) belong here too.

Current contents
----------------
- ``add_intensity_noise``: Poisson + Gaussian on a non-negative intensity
  map.
- ``add_velocity_noise``: Gaussian-only on a (signed) velocity map. Poisson
  enters at the spectral-cube layer, never on the moment.

SNR convention (current baseline: matched-filter)
-------------------------------------------------
For a uniform per-pixel noise standard deviation ``sigma`` and signal
template ``T``, the matched-filter amplitude SNR is

    SNR_MF = ||T||_2 / sigma .

This module's helpers therefore set per-pixel ``sigma = ||T||_2 / target_snr``,
so the input ``target_snr`` corresponds directly to the matched-filter
amplitude SNR an observer would quote. This is stamp-shape-invariant for
compact sources and is the most physically meaningful single number to
attach to a synthetic dataset.

For the Poisson-on path the per-pixel variance is non-uniform. We use the
uniform-equivalent matched-filter target: pick Gaussian ``sigma_g`` so
that ``mean(poisson_var) + sigma_g^2 = (||T||_2 / target_snr)^2``. This
is exact when ``include_poisson=False`` and an effective approximation
when shot noise is on.

Other conventions (e.g. range-based for velocity, L2-RMS for stamp-fixed
test calibrations) can be added later as alternative entry points; do not
silently change the meaning of ``target_snr`` here.

Returns
-------
Both helpers return ``(noisy_image, variance_map)`` where the variance is
a per-pixel array of the same shape as the input. Gaussian-only paths give
a uniform variance map; Poisson contributes per-pixel structure proportional
to ``intensity / gain``. Downstream observation builders and JIT likelihoods
broadcast scalar or array variance equivalently, so callers do not need to
special-case shape.
"""

import numpy as np
from typing import Optional, Tuple


def add_intensity_noise(
    intensity: np.ndarray,
    target_snr: float,
    include_poisson: bool = True,
    gain: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add Poisson + Gaussian noise to a non-negative intensity map.

    Parameters
    ----------
    intensity : ndarray
        Noiseless intensity map. Must be non-negative when
        ``include_poisson=True``.
    target_snr : float
        Matched-filter amplitude SNR (see module docstring).
    include_poisson : bool, default True
        Add Poisson shot noise. Raises ``ValueError`` on any negative
        input pixel.
    gain : float, default 1.0
        Detector gain converting intensity units to photon counts:
        ``counts = intensity * gain``. Per-pixel Poisson variance in
        intensity units is ``intensity / gain``.
    seed : int, optional
        RNG seed.

    Returns
    -------
    noisy_image : ndarray
    variance : ndarray
        Per-pixel variance map, same shape as ``intensity``.
    """
    if gain <= 0:
        raise ValueError(f"gain must be positive, got {gain}")

    rng = np.random.default_rng(seed)
    intensity = np.asarray(intensity)

    if include_poisson:
        if np.any(intensity < 0):
            raise ValueError(
                "add_intensity_noise(include_poisson=True) requires "
                "non-negative input (photon-count semantics). Got an array "
                "with negative values; use include_poisson=False, or route "
                "signed data (e.g. velocity fields) through "
                "add_velocity_noise."
            )

        counts = intensity * gain
        noisy_counts = np.where(counts > 0, rng.poisson(counts), counts)
        noisy_image = noisy_counts / gain
        poisson_var_per_pixel = intensity / gain
    else:
        noisy_image = intensity.copy()
        poisson_var_per_pixel = np.zeros_like(intensity)

    norm_l2 = float(np.sqrt(np.sum(intensity**2)))
    if norm_l2 == 0:
        raise ValueError("Cannot add noise to a zero-norm intensity map.")

    # Matched-filter target: pick uniform Gaussian sigma so that the
    # uniform-equivalent per-pixel variance matches (||I||_2 / SNR)^2.
    target_pixel_var = (norm_l2 / target_snr) ** 2
    gauss_var = max(0.0, target_pixel_var - float(poisson_var_per_pixel.mean()))

    if gauss_var > 0:
        sigma_g = float(np.sqrt(gauss_var))
        noisy_image = noisy_image + rng.normal(0, sigma_g, intensity.shape)

    variance = poisson_var_per_pixel + gauss_var
    return noisy_image, variance


def add_velocity_noise(
    velocity: np.ndarray,
    target_snr: float,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add Gaussian noise to a velocity map.

    Velocity is a flux-weighted moment of the spectral cube, not a
    photon-count map; Poisson statistics belong at the datacube layer,
    not here. Noise is therefore Gaussian by construction with the same
    matched-filter SNR convention as ``add_intensity_noise``.

    Parameters
    ----------
    velocity : ndarray
        Noiseless velocity map (km/s, signed).
    target_snr : float
        Matched-filter amplitude SNR.
    seed : int, optional

    Returns
    -------
    noisy_velocity : ndarray
    variance : ndarray
        Per-pixel variance map (uniform), same shape as ``velocity``.
    """
    rng = np.random.default_rng(seed)
    velocity = np.asarray(velocity)

    norm_l2 = float(np.sqrt(np.sum(velocity**2)))
    if norm_l2 == 0:
        raise ValueError("Cannot add noise to a zero-norm velocity map.")

    sigma = norm_l2 / target_snr
    noisy_velocity = velocity + rng.normal(0, sigma, velocity.shape)
    variance = np.full_like(velocity, sigma**2, dtype=float)
    return noisy_velocity, variance
