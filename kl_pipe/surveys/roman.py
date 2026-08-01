"""
Roman Space Telescope HLWAS medium-tier survey parameters.

Published depths, flux limits, and band definitions for the High Latitude
Wide Area Survey medium tier, plus the depth-referenced SNR and
measurement-error helpers built on them. Primary sources: the ROTAC Final
Report and Recommendations (arXiv:2505.10574) and Wang et al. 2022
(arXiv:2110.01829). Every number carries its provenance in a comment;
changing one means the source changed.
"""

from __future__ import annotations

import numpy as np

from kl_pipe.constants import ARCSEC_PER_RAD
from kl_pipe.noise import gaussian_matched_filter_compactness
from kl_pipe.photometry import HALPHA_REST_A, ab_mag_to_ujy

# ROTAC (Roman Observations Time Allocation Committee) Final Report and
# Recommendations, 2025-04-24 v3, Sect. 3.1: the HLWAS medium tier reaches a
# "spectroscopic depth of 1.5 x 10^-16 erg/cm2/sec (5 sigma line flux limit,
# texp ~ 1500 sec)" [erg/s/cm2]. Sect. 4.4.3 gives the exposure structure
# behind that texp: four grism passes at distinct roll angles, two dither
# positions per pass, 189.75 s per exposure (8 x 189.75 = 1518 s). This is
# therefore the COADDED limit over all four passes -- the depth the joint
# multi-roll fit sees. Sources are penalized (or rewarded) by the
# matched-filter compactness ratio C below, relative to the source the limit
# itself was derived for -- see SNR_LINE_REFERENCE.
F_LIM_COADD_CGS = 1.5e-16

# grism passes the coadded limit covers (ROTAC Sect. 4.4.3). Must equal the
# number of grism rolls in the observation config, or the total SNR the fit
# sees differs from the total the selection assumed; the expander enforces
# that when it builds catalog manifest rows.
N_GRISM_PASSES = 4

# a single pass is sqrt(N) shallower than the coadd. This is the limit the
# per-pass SNR is referenced to, and the per-pass SNR is what normalizes the
# noise of each individual grism roll's mock (see mocks.grism_line_noise).
F_LIM_PER_PASS_CGS = F_LIM_COADD_CGS * np.sqrt(N_GRISM_PASSES)

# the SNR the flux-limit anchor corresponds to (ROTAC Sect. 3.1: "5 sigma
# emission line integrated flux limit"). Wang et al. 2022 additionally
# characterize the Reference HLSS at 6.5 sigma; ROTAC does not, so no
# sigma-convention conversion applies to the number above.
F_LIM_NSIGMA = 5.0

# What source the flux limit is referenced to. Not a free choice: it must
# match the convention of the published limit, and getting it wrong
# double-counts (or omits) the extended-source penalty, an order of
# magnitude in selected number density.
#
#   'point_source'      -- C is taken relative to an unresolved source.
#   'extended_fiducial' -- C is taken relative to a round galaxy of
#                          half-light radius F_LIM_REF_R50_ARCSEC observed
#                          at F_LIM_REF_LAMBDA_A.
#
# The ROTAC limit is an extended-source limit. It does not say so in the
# sentence quoting the number, but four independent lines agree:
#
#   - The committee's own request to the HLSS PIT (Appendix B.2, item 1a)
#     asks for "Line flux limits for a realistic extended source at 1.1,
#     1.2, 1.5, 1.8, 1.9 mu". The parallel imaging request (B.1, item 1a)
#     asks separately for "point sources and rexp=0.3 arcsec extended
#     sources", so the distinction is deliberate and spectroscopy was asked
#     for extended only.
#   - Sect. 3.1 labels the imaging depth "5 sigma point source" and, in the
#     same sentence, the spectroscopic depth only "5 sigma line flux limit".
#     The point-source qualifier is used where it applies.
#   - Sect. 4.2.1 refers the reader to Wang et al. 2022 for HLWAS
#     spectroscopy. Wang Sect. 5.1 adopts "line flux limits derived for
#     galaxies with radius 0.25 arcsec at 1.5 micron", that radius being the
#     median half-light radius of Halpha emitters at z ~ 1.5 from WISP (Dore
#     et al. 2018 Fig. 17), and warns that "the point source sensitivity is
#     significantly better than for galaxies with a finite size".
#   - Arithmetic. Wang's 8.5e-17 (5 sigma, 0.25") is at 8 x 301 s; scaled to
#     ROTAC's 8 x 190 s in the background-limited regime that is ~1.1e-16,
#     within 1.4x of the 1.5e-16 ROTAC published (ROTAC Sect. 4.4.3 notes it
#     ran its own ETC calculation folding in chip gaps and ecliptic-latitude
#     variation, so a somewhat shallower characteristic depth is expected). A
#     point-source limit would be ~4e-17, i.e. 3-4x DEEPER than the number
#     published -- the wrong direction by a wide margin.
#
# End-to-end cross-check: under 'extended_fiducial' this selection returns
# 4114 Halpha/deg2 at the published coadded depth over 99.5 deg2 of
# Flagship2, against the 5917/deg2 ROTAC forecasts for the medium tier
# (14.2M Halpha redshifts over 2400 deg2, Sect. 4.4.3) -- 0.70x, the right
# ballpark for a pure line-SNR count against a secure-redshift count.
# 'point_source' returns 504/deg2, low by 12x against a forecast made with
# the same ETC that produced the flux limit we consume.
SNR_LINE_REFERENCE = 'extended_fiducial'
F_LIM_REF_R50_ARCSEC = 0.25
F_LIM_REF_LAMBDA_A = 1.5e4

# ROTAC Final Report Table 1 (arXiv:2505.10574, report p. 2; verified against
# the STScI HLWAS survey-definition page, both read 2026-07-31): HLWAS
# medium-tier imaging, 5-sigma POINT-SOURCE coadded depths [AB mag] (two
# passes x three dithers x 107.25 s = 643.5 s per filter; the table rounds
# the exposure to 107 s, the embedded Definition Committee report Sect. 4.4.3
# gives 107.25 s). The imaging depths are point-source numbers -- the table
# caption states "r1/2 = 0.3 arcsec extended source thresholds are typically
# ~1.1 mag brighter" -- so the imaging SNR is referenced to a point source
# (C_ref = 1) and extended sources are penalized through their own
# compactness; a test reproduces the ~1.1 mag statement from this machinery.
# The grism limit in the same table is an extended-source number; each
# channel anchors to its own published convention (see SNR_LINE_REFERENCE).
IMAGING_DEPTH_AB = {'F106': 26.5, 'F129': 26.4, 'F158': 26.4}

# the imaging bands every catalog adapter carries physical fluxes for; any
# observation config's bands must be a subset
ROMAN_IMAGING_BANDS = tuple(sorted(IMAGING_DEPTH_AB))

# the SNR the imaging depths correspond to (ROTAC Table 1: "5sigma point
# source detection thresholds in AB magnitudes")
IMAGING_DEPTH_NSIGMA = 5.0

# Roman WFI imaging-band effective wavelengths [Angstrom], from
# galsim.roman.getBandpasses() effective_wavelength (Roman_effarea_20210614
# throughput tables; galsim uses the legacy names Y106/J129/H158): 1059.5,
# 1293.6, 1579.1 nm. Keys are the official F-names the ensemble uses; a test
# pins these against the galsim values.
BAND_EFFECTIVE_LAMBDA_A = {'F106': 1.0595e4, 'F129': 1.2936e4, 'F158': 1.5791e4}

# Roman primary-mirror diameter [m]; diffraction PSF proxy
# FWHM = 1.22 lambda / D
ROMAN_APERTURE_M = 2.36


def psf_fwhm_arcsec(z: np.ndarray) -> np.ndarray:
    """Diffraction PSF FWHM at the observed Halpha wavelength [arcsec].

    FWHM = 1.22 * lambda_obs / D with lambda_obs = HALPHA_REST_A * (1 + z).
    Used both by the compactness ratio and by the resolvability cut, so the
    two share one PSF definition.
    """
    lambda_obs_m = HALPHA_REST_A * (1.0 + np.asarray(z, dtype=np.float64)) * 1e-10
    return 1.22 * lambda_obs_m / ROMAN_APERTURE_M * ARCSEC_PER_RAD


def _compactness_at_lambda(
    reff_arcsec: np.ndarray, cosi: np.ndarray, lambda_obs_a: np.ndarray
) -> np.ndarray:
    """Gaussian matched-filter compactness at an arbitrary wavelength.

    Shared core of ``matched_filter_compactness`` (grism, at the observed
    Halpha wavelength) and ``imaging_compactness`` (broadband, at the band's
    effective wavelength). The diffraction PSF proxy FWHM = 1.22 lambda / D
    feeds ``kl_pipe.noise.gaussian_matched_filter_compactness``, which holds
    the formula and conventions.
    """
    lambda_m = np.asarray(lambda_obs_a, dtype=np.float64) * 1e-10
    fwhm = 1.22 * lambda_m / ROMAN_APERTURE_M * ARCSEC_PER_RAD
    return gaussian_matched_filter_compactness(reff_arcsec, cosi, fwhm)


def matched_filter_compactness(
    reff_arcsec: np.ndarray, cosi: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """Matched-filter SNR compactness ratio C in (0, 1] at the Halpha line.

    C is the matched-filter SNR of a source relative to an unresolved one at
    the same line flux; ``fiducial_compactness`` supplies the reference the
    published flux limit is normalized to. Evaluated at
    lambda_obs = HALPHA_REST_A * (1 + z) with the Roman diffraction PSF
    proxy; the formula lives in
    ``kl_pipe.noise.gaussian_matched_filter_compactness``.

    Parameters
    ----------
    reff_arcsec : np.ndarray
        Disk half-light radius [arcsec] (the adapter's disk_r50 column).
    cosi : np.ndarray
        Cosine of inclination (minor/major axis ratio of the thin disk).
    z : np.ndarray
        Redshift; sets lambda_obs = 6562.8 * (1 + z) for the PSF proxy.

    Returns
    -------
    np.ndarray
        Compactness C in (0, 1]; C -> 1 for unresolved sources.
    """
    z = np.asarray(z, dtype=np.float64)
    lambda_obs_a = HALPHA_REST_A * (1.0 + z)
    return _compactness_at_lambda(reff_arcsec, cosi, lambda_obs_a)


def imaging_compactness(
    reff_arcsec: np.ndarray, cosi: np.ndarray, band: str
) -> np.ndarray:
    """Matched-filter compactness C in an imaging band.

    Same Gaussian matched-filter amplitude ratio as
    ``matched_filter_compactness``, evaluated at the band's effective
    wavelength (``BAND_EFFECTIVE_LAMBDA_A``) instead of the observed Halpha
    wavelength. C -> 1 for unresolved sources, which is the reference the
    point-source imaging depths are quoted for.
    """
    if band not in BAND_EFFECTIVE_LAMBDA_A:
        raise KeyError(
            f"band '{band}' has no effective wavelength; known bands: "
            f"{sorted(BAND_EFFECTIVE_LAMBDA_A)}"
        )
    lam = np.full_like(
        np.asarray(reff_arcsec, dtype=np.float64), BAND_EFFECTIVE_LAMBDA_A[band]
    )
    return _compactness_at_lambda(reff_arcsec, cosi, lam)


def band_flux_limit_ujy(band: str) -> float:
    """The published imaging depth as a point-source flux [uJy].

    f_lim = 10 ** ((23.9 - depth_AB) / 2.5); a point source at this flux has
    matched-filter SNR = IMAGING_DEPTH_NSIGMA in the tier coadd.
    """
    if band not in IMAGING_DEPTH_AB:
        raise KeyError(
            f"band '{band}' has no published HLWAS medium imaging depth; "
            f"known bands: {sorted(IMAGING_DEPTH_AB)}"
        )
    return float(ab_mag_to_ujy(IMAGING_DEPTH_AB[band]))


def compute_band_snr(
    f_ujy: np.ndarray, reff_arcsec: np.ndarray, cosi: np.ndarray, band: str
) -> np.ndarray:
    """Coadded matched-filter imaging SNR referenced to the published depth.

    SNR = IMAGING_DEPTH_NSIGMA * (f / f_lim) * C, with C_ref = 1 because the
    imaging depths are point-source numbers (see IMAGING_DEPTH_AB). This is
    the depth the fit sees in one band over the full tier coadd, and the
    quantity that normalizes the band mock's noise.
    """
    return (
        IMAGING_DEPTH_NSIGMA
        * np.asarray(f_ujy, dtype=np.float64)
        / band_flux_limit_ujy(band)
        * imaging_compactness(reff_arcsec, cosi, band)
    )


def band_flux_sigma_ujy(
    reff_arcsec: np.ndarray, cosi: np.ndarray, band: str
) -> np.ndarray:
    """Expected photometric flux error [uJy] for a galaxy in a band.

    sigma_f = f / SNR = f_lim / (IMAGING_DEPTH_NSIGMA * C): the flux cancels,
    so the measurement error depends only on the source's shape through its
    compactness -- the background-dominated regime the published depths
    describe. Sets the simulated-photometry prior width and the seeded
    measurement draw.
    """
    return band_flux_limit_ujy(band) / (
        IMAGING_DEPTH_NSIGMA * imaging_compactness(reff_arcsec, cosi, band)
    )


def line_flux_sigma_cgs(compactness: np.ndarray) -> np.ndarray:
    """Expected line-flux measurement error [erg/s/cm2], full-tier coadd.

    sigma_f = f / SNR_total = (F_LIM_COADD_CGS / F_LIM_NSIGMA) * C_ref / C;
    the flux cancels as in ``band_flux_sigma_ujy``. Referenced to the coadded
    limit because a line-flux measurement uses all passes jointly.
    """
    return (
        F_LIM_COADD_CGS
        / F_LIM_NSIGMA
        * fiducial_compactness()
        / np.asarray(compactness, dtype=np.float64)
    )


def fiducial_compactness() -> float:
    """Compactness C of the source the published flux limit refers to.

    A round (cos i = 1) galaxy of half-light radius ``F_LIM_REF_R50_ARCSEC``
    observed at ``F_LIM_REF_LAMBDA_A`` -- the source Wang et al. 2022 derive
    the Roman HLSS line-flux limits for. Returns 1.0 for a point-source
    reference, which is the identity normalization.
    """
    if SNR_LINE_REFERENCE == 'point_source':
        return 1.0
    if SNR_LINE_REFERENCE != 'extended_fiducial':
        raise ValueError(
            f"SNR_LINE_REFERENCE must be 'point_source' or "
            f"'extended_fiducial', got {SNR_LINE_REFERENCE!r}"
        )
    z_ref = F_LIM_REF_LAMBDA_A / HALPHA_REST_A - 1.0
    return float(
        matched_filter_compactness(
            np.array([F_LIM_REF_R50_ARCSEC]), np.array([1.0]), np.array([z_ref])
        )[0]
    )


def compute_line_snr_per_pass(
    f_line: np.ndarray, compactness: np.ndarray
) -> np.ndarray:
    """Matched-filter line SNR in a SINGLE grism pass.

    SNR = F_LIM_NSIGMA * (f_line / F_LIM_PER_PASS_CGS) * C / C_ref, where
    C_ref is the compactness of the source the flux limit is referenced to
    (``fiducial_compactness``). A source identical to that reference at
    f_line = F_LIM_PER_PASS_CGS therefore has SNR = F_LIM_NSIGMA; more
    compact sources do better and larger ones worse.

    This is the quantity that normalizes the noise of each individual grism
    roll's mock (mocks.grism_line_noise is called once per roll). It is NOT
    the quantity to apply a selection cut to -- use
    ``compute_line_snr_total``, since the fit ingests every pass.
    """
    return (
        F_LIM_NSIGMA
        * np.asarray(f_line, dtype=np.float64)
        / F_LIM_PER_PASS_CGS
        * np.asarray(compactness, dtype=np.float64)
        / fiducial_compactness()
    )


def compute_line_snr_total(f_line: np.ndarray, compactness: np.ndarray) -> np.ndarray:
    """Matched-filter line SNR coadded over all ``N_GRISM_PASSES`` passes.

    sqrt(N_GRISM_PASSES) times ``compute_line_snr_per_pass``, equivalently
    the SNR referenced directly to the published coadded limit
    ``F_LIM_COADD_CGS``. This is the depth the joint multi-roll fit sees and
    the quantity a selection cut belongs on: cutting the per-pass SNR at 10
    selects a total of 20, which is not what the number reads as.
    """
    return np.sqrt(N_GRISM_PASSES) * compute_line_snr_per_pass(f_line, compactness)
