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
- ``grism_line_noise``: Gaussian on a dispersed grism stamp, normalized so the
  labeled SNR is the emission-LINE matched-filter SNR (not the whole stamp).

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

Grism line-SNR convention
--------------------------
A dispersed grism stamp contains the emission line plus the galaxy continuum,
but only the line carries the kinematic (shear/velocity) signal. Normalizing
the SNR over the whole stamp lets the continuum -- which for a realistic disk
dominates the dispersed power -- absorb most of the SNR budget, so a nominal
"grism SNR" corresponds to a much lower effective LINE SNR. ``grism_line_noise``
therefore defines the labeled SNR on the LINE component alone:

    var = ||I_line||_2^2 / line_snr^2 ,

a uniform per-pixel Gaussian variance set by the line matched filter. This
assumes the sky/background-dominated regime (faint slitless-grism sources),
where the per-pixel noise floor is independent of the source; the continuum is
still rendered and marginalized as a nuisance but does not enter the SNR
normalization. This matches the emission-line SNR used in the Roman KL
literature (Xu et al., in the sky-dominated limit).

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
    include_poisson: bool = False,
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
    include_poisson : bool, default False
        Add Poisson shot noise. Raises ``ValueError`` on any negative
        input pixel. Raises ``ValueError`` if Poisson alone would
        overshoot the matched-filter target (see Notes); use
        ``include_poisson=False`` or lower ``target_snr`` in that case.
        Default flipped to ``False`` because the matched-filter SNR
        contract is only honestly satisfiable when Poisson is sub-dominant.
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

    Notes
    -----
    ``include_poisson=True`` enforces a hard consistency check: the mean
    per-pixel Poisson variance, ``mean(intensity / gain)``, must not
    exceed the matched-filter target variance, ``(||intensity||_2 /
    target_snr)**2``. If it does, Poisson noise alone delivers a lower
    effective SNR than ``target_snr`` and the labeled SNR cannot be
    honored. The function raises ``ValueError`` rather than silently
    clipping the Gaussian contribution to zero (the prior behavior). See
    issue #24 for the consolidation work tracking the broader sim/inference
    variance-model alignment.
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
    poisson_var_mean = float(poisson_var_per_pixel.mean())

    if include_poisson and poisson_var_mean > target_pixel_var:
        # Poisson alone overshoots the matched-filter target. The previous
        # silent max(0.0, ...) clamp hid this and let labeled SNRs run at
        # effective SNRs orders of magnitude lower. See issue #24.
        effective_snr = norm_l2 / np.sqrt(poisson_var_mean)
        raise ValueError(
            f"include_poisson=True is inconsistent with target_snr="
            f"{target_snr:g}: mean Poisson variance "
            f"({poisson_var_mean:.3g}) exceeds matched-filter target "
            f"({target_pixel_var:.3g}). Effective SNR with Poisson alone "
            f"would be {effective_snr:.3g}. Lower target_snr or set "
            f"include_poisson=False."
        )

    gauss_var = target_pixel_var - poisson_var_mean

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


def grism_line_noise(
    full_render: np.ndarray,
    line_render: np.ndarray,
    line_snr: float,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """Add Gaussian noise to a dispersed grism stamp at a target LINE SNR.

    The uniform per-pixel variance is set by the emission-LINE matched filter,
    ``var = sum(line_render**2) / line_snr**2``, so the labeled ``line_snr`` is
    the matched-filter amplitude SNR of the line alone (see the module docstring,
    "Grism line-SNR convention"). Noise is added to ``full_render`` (line +
    continuum); the continuum contributes to the observed data and is
    marginalized in the fit, but does NOT enter the SNR normalization. This is
    the sky/background-dominated regime, where the noise floor is independent of
    the source.

    Parameters
    ----------
    full_render : ndarray
        Noiseless dispersed stamp including line + continuum.
    line_render : ndarray
        Noiseless dispersed stamp of the LINE component only (continuum
        amplitude zeroed), same shape as ``full_render``.
    line_snr : float
        Target emission-line matched-filter amplitude SNR.
    seed : int, optional
        RNG seed.

    Returns
    -------
    noisy_image : ndarray
        ``full_render`` plus Gaussian noise of the line-normalized variance.
    variance : float
        Scalar per-pixel variance ``||line_render||^2 / line_snr**2``.
    """
    full = np.asarray(full_render)
    line = np.asarray(line_render)
    if full.shape != line.shape:
        raise ValueError(
            f"full_render shape {full.shape} != line_render shape {line.shape}"
        )
    if line_snr <= 0:
        raise ValueError(f"line_snr must be positive, got {line_snr}")
    line_power = float(np.sum(line**2))
    if line_power <= 0:
        raise ValueError(
            "line_render has zero power; cannot set a line SNR (is the line "
            "flux zero, or was the wrong component passed?)"
        )
    var = line_power / line_snr**2
    rng = np.random.default_rng(seed)
    noisy = full + rng.normal(0.0, np.sqrt(var), size=full.shape)
    return noisy, var

def add_fiberspec_noise(
    spectrum: np.ndarray,
    target_snr: float,
    include_poisson: bool = False,
    gain: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add Poisson and Gaussian noise to a one-dimensional fiber spectrum.
    
    The input is interpreted as the expected signal in each wavelength bin.
    With Poisson noise enabled, ``spectrum * gain`` must therefore be a
    non-negative expected photon count. Negative, background-subtracted
    values must not be converted with ``abs``; use Gaussian-only noise or
    model the non-negative source plus background counts before subtraction.
    
    Parameters
    ----------
    spectrum : ndarray
        Noiseless one-dimensional spectrum.
    target_snr : float
        Matched-filter amplitude SNR (see the module docstring).
    include_poisson : bool, default False
        Include photon shot noise. The Poisson contribution must not already
        exceed the requested target variance.
    gain : float, default 1.0
        Conversion from spectrum units to expected photon counts.
    seed : int, optional
        RNG seed.
        
    Returns
    -------
    noisy_spectrum : ndarray
    variance : ndarray
        Per-wavelength-bin variance, with the same shape as ``spectrum``.
    """
    
    spectrum = np.asarray(spectrum)

    if spectrum.ndim != 1:
        raise ValueError(
            f"spectrum must be one-dimensional, got shape {spectrum.shape}"
        )
    if spectrum.size == 0:
        raise ValueError("spectrum must not be empty")
    if not np.all(np.isfinite(spectrum)):
        raise ValueError("spectrum must contain only finite values")
    if not np.isfinite(target_snr) or target_snr <= 0:
        raise ValueError(f"target_snr must be positive and finite, got {target_snr}")
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError(f"gain must be positive and finite, got {gain}")
    if include_poisson and np.any(spectrum < 0):
        raise ValueError(
            "add_fiberspec_noise(include_poisson=True) requires a "
            "non-negative expected photon signal; do not use abs() on a "
            "signed or background-subtracted spectrum"
        )
    
    # A spectrum is a one-dimensional intensity array, so the detector model
    # and matched-filter SNR calibration are identical to the image case.
    return add_intensity_noise(
        spectrum,
        target_snr=target_snr,
        include_poisson=include_poisson,
        gain=gain,
        seed=seed,
    )