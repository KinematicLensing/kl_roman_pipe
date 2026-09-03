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
- ``physical_variance_map`` / ``add_map_noise``: flat background + source
  shot noise as a per-pixel variance map, and the heteroscedastic Gaussian
  draw from it (the high-count limit of Poisson statistics).
- ``matched_filter_snr``: sqrt(sum(T^2/var)) -- realized matched-filter SNR
  of a template against a scalar or per-pixel noise model.

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
assumes the background-dominated regime (faint slitless-grism sources), where
the per-pixel noise floor is independent of the source; for the Roman grism
that floor is mostly read noise (~96% of the background variance at 1.1
nm/pix), not sky. The continuum is still rendered and
marginalized as a nuisance but does not enter the SNR normalization. This
matches the emission-line SNR used in the Roman KL literature (Xu et al., in
the background-dominated limit).

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

from kl_pipe.photometry import EXP_R50_OVER_RSCALE


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
    the background-dominated regime, where the noise floor is independent of
    the source (for the Roman grism that background is mostly read noise, not
    sky -- see the module docstring).

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


def physical_variance_map(
    truth_image: np.ndarray,
    sigma_bkg: float,
    electrons_per_flux: float,
) -> np.ndarray:
    """Per-pixel variance map: flat background plus source shot noise.

    var = sigma_bkg**2 + max(I, 0) / electrons_per_flux, all in the squared
    flux units of ``truth_image``. The Poisson term is the standard
    counts -> flux propagation: counts = I * g gives var_counts = counts,
    i.e. var_flux = I / g with g = electrons_per_flux (detected electrons
    per flux unit of the image; NOT the detector e-/ADU gain).

    Small negative pixels (FFT rendering ringing) contribute zero shot
    noise; a strongly negative pixel means the wrong image was passed and
    raises.

    Parameters
    ----------
    truth_image : ndarray
        Noiseless truth render in flux/pixel units.
    sigma_bkg : float
        Flat per-pixel background standard deviation, same flux units.
    electrons_per_flux : float
        Detected electrons per unit of the image's flux.

    Returns
    -------
    ndarray
        Per-pixel variance, same shape as ``truth_image``.
    """
    if not np.isfinite(sigma_bkg) or sigma_bkg <= 0:
        raise ValueError(f"sigma_bkg must be positive and finite, got {sigma_bkg}")
    if not np.isfinite(electrons_per_flux) or electrons_per_flux <= 0:
        raise ValueError(
            f"electrons_per_flux must be positive and finite, got "
            f"{electrons_per_flux}"
        )
    truth_image = np.asarray(truth_image, dtype=np.float64)
    if not np.all(np.isfinite(truth_image)):
        raise ValueError("truth_image contains non-finite pixels")
    peak = float(truth_image.max())
    if peak <= 0:
        raise ValueError("truth_image has no positive flux; cannot set shot noise")
    if float(truth_image.min()) < -0.05 * peak:
        raise ValueError(
            f"truth_image has a pixel at {truth_image.min():.3g} against peak "
            f"{peak:.3g}; rendering ringing is at the sub-percent level, so "
            f"this is the wrong image (residual? background-subtracted data?)"
        )
    return sigma_bkg**2 + np.clip(truth_image, 0.0, None) / electrons_per_flux


def add_map_noise(
    image: np.ndarray, variance: np.ndarray, seed: Optional[int] = None
) -> np.ndarray:
    """Add zero-mean Gaussian noise with a per-pixel variance map.

    The high-count Gaussian limit of Poisson + background noise: the draw
    is exactly the heteroscedastic Gaussian the per-pixel ``variance``
    describes, so a likelihood carrying the same map is exact for this
    data by construction.
    """
    image = np.asarray(image, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    if variance.shape != image.shape:
        raise ValueError(
            f"variance shape {variance.shape} != image shape {image.shape}"
        )
    if not np.all(np.isfinite(image)):
        raise ValueError("image contains non-finite pixels")
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
        raise ValueError("variance map must be finite and strictly positive")
    rng = np.random.default_rng(seed)
    return image + rng.normal(0.0, np.sqrt(variance))


def matched_filter_snr(template: np.ndarray, variance) -> float:
    """Matched-filter amplitude SNR of a template against a noise model.

    SNR = sqrt( sum(T**2 / var) ): the optimal detection SNR for a known
    template in independent Gaussian noise. ``variance`` may be a scalar
    (uniform noise; reduces to ||T||_2 / sigma) or a per-pixel map. This is
    the definition behind the ensemble's ``snr_effective`` columns: the
    realized SNR of the noiseless truth against the actual variance,
    including any shot-noise term the map carries.
    """
    template = np.asarray(template, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    if variance.ndim != 0 and variance.shape != template.shape:
        raise ValueError(
            f"variance must be a scalar or match the template shape "
            f"{template.shape}, got {variance.shape}; broadcasting a "
            f"mismatched map would silently weight the wrong pixels"
        )
    if not np.all(np.isfinite(template)):
        raise ValueError("template contains non-finite pixels")
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
        raise ValueError("variance must be finite and strictly positive")
    return float(np.sqrt(np.sum(template**2 / variance)))


# Gaussian FWHM -> sigma divisor, 2*sqrt(2 ln 2) ~= 2.355
FWHM_TO_SIGMA = 2.355


def gaussian_matched_filter_compactness(
    r50_arcsec: np.ndarray, cosi: np.ndarray, psf_fwhm_arcsec: np.ndarray
) -> np.ndarray:
    """Matched-filter SNR of an extended source relative to a point source.

    The compactness ratio C in (0, 1]: the matched-filter SNR of a source at
    a given flux divided by that of an unresolved source at the same flux.
    Pure detection math -- it needs no noise estimate, only the source and
    PSF shapes:

        C = sigma_psf / sqrt(s1 * s2),  s_i = sqrt(sigma_psf^2 + sig_i^2)

    with the galaxy approximated as an elliptical Gaussian of
    sigma_major = r50 / 1.678 (the exponential scalelength) and
    sigma_minor = cosi * sigma_major, and the PSF as a Gaussian of the given
    FWHM divided by 2.355. This is the correct matched-filter amplitude
    ratio for Gaussians (NOT the peak-pixel square of an earlier prototype,
    which was a bug).

    Parameters
    ----------
    r50_arcsec : np.ndarray
        Source half-light radius [arcsec]; must be positive.
    cosi : np.ndarray
        Cosine of inclination (minor/major axis ratio of the thin disk).
    psf_fwhm_arcsec : np.ndarray
        PSF full width at half maximum [arcsec].

    Returns
    -------
    np.ndarray
        Compactness C in (0, 1]; C -> 1 for unresolved sources.
    """
    r50_arcsec = np.asarray(r50_arcsec, dtype=np.float64)
    cosi = np.asarray(cosi, dtype=np.float64)
    if np.any(r50_arcsec <= 0):
        raise ValueError("compactness: r50_arcsec must be positive")

    sigma_psf = np.asarray(psf_fwhm_arcsec, dtype=np.float64) / FWHM_TO_SIGMA
    sig_maj = r50_arcsec / EXP_R50_OVER_RSCALE
    sig_min = cosi * sig_maj
    s1 = np.sqrt(sigma_psf**2 + sig_maj**2)
    s2 = np.sqrt(sigma_psf**2 + sig_min**2)
    return sigma_psf / np.sqrt(s1 * s2)
