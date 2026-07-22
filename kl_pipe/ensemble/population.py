"""
Catalog-backed galaxy population: Flagship2 rows + kinematic paint.

Builds the per-galaxy population table for ``population.type: catalog``
ensemble campaigns. Structural and flux truths (sizes, Halpha flux,
continuum, bulge properties, redshift) come from Euclid Flagship2 catalog
rows; kinematics (vcirc via an inverted Tully-Fisher relation, sigma0 via an
affine-in-z relation) are painted on with seeded scatter; orientation is an
isotropic redraw (the catalog inclination is kept for validation only);
shear is drawn per ring pair.

Determinism contract
--------------------
- Every per-galaxy draw uses a numpy SeedSequence keyed on
  ``[spec.seed, STREAM_TAG, halo_id, galaxy_id]``. Flagship2 ``galaxy_id``
  is a within-halo index (NOT globally unique); the (halo_id, galaxy_id)
  pair is the unique catalog key, so both ids enter the seed key. Draws are
  therefore independent of catalog row order and of the selection applied:
  a galaxy keeps its cosi/theta/paint/shear values under any superset
  catalog download.
- The subsample draw uses a single stream keyed ``[spec.seed, SAMPLE_TAG]``
  over the selected rows sorted by (halo_id, galaxy_id).
- Stream tags are disjoint from the expander's (1/2/3).

This module is numpy+pandas only (population building is a one-shot host
task, not traced model code).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from kl_pipe.ensemble.spec import CatalogPopulationSpec, EnsembleSpec

# =============================================================================
# Constants (each with provenance)
# =============================================================================

# Halpha rest wavelength [Angstrom] (air; standard line-list value)
HALPHA_REST_A = 6562.8

# speed of light [Angstrom / s]; converts f_nu [erg/cm2/s/Hz] to
# f_lambda = f_nu * c / lambda^2 [erg/cm2/s/A]
C_A_PER_S = 2.998e18

# ROTAC (Roman Observations Time Allocation Committee) HLWAS spectroscopy
# report: per-pass 5-sigma point-source emission-line flux limit
# [erg/s/cm2]. Extended sources are penalized by the matched-filter
# compactness ratio C below.
F_LIM_CGS = 3.1e-16

# the SNR the F_LIM anchor corresponds to (5-sigma point source)
F_LIM_NSIGMA = 5.0

# Roman primary-mirror diameter [m]; diffraction PSF proxy
# FWHM = 1.22 lambda / D
ROMAN_APERTURE_M = 2.36

# Gaussian FWHM -> sigma divisor, 2*sqrt(2 ln 2) ~= 2.355
FWHM_TO_SIGMA = 2.355

# radians -> arcsec
ARCSEC_PER_RAD = 180.0 / np.pi * 3600.0

# exponential disk: r50 = 1.678 * scalelength; the matched-filter Gaussian
# proxy takes sigma_major = the exponential scalelength = r50 / 1.678
R50_TO_SIGMA = 1.678

# Euclid NISP band edges [Angstrom] for continuum selection at the observed
# Halpha wavelength: Y covers lambda < 11900, J covers [11900, 15450),
# H covers >= 15450 (nominal NISP passband boundaries)
NISP_YJ_EDGE_A = 11900.0
NISP_JH_EDGE_A = 15450.0

# seed-stream domain tags (disjoint from expander's 1/2/3); per-galaxy
# streams key [seed, TAG, halo_id, galaxy_id], the sample stream keys
# [seed, TAG] only
_POP_SAMPLE = 10
_POP_GEOMETRY = 11
_POP_PAINT = 12
_POP_SHEAR = 13
_POP_PRIOR = 14

# full Flagship2 query-spec schema (data/cosmohub/flagship2_dev.yaml);
# downloads are validated against this exact column set
FLAGSHIP2_COLUMNS = (
    'galaxy_id',
    'halo_id',
    'ra_gal',
    'dec_gal',
    'true_redshift_gal',
    'observed_redshift_gal',
    'logf_halpha_model3',
    'logf_halpha_model3_ext',
    'logf_halpha_model1',
    'logf_halpha_model1_ext',
    'halpha_scatter',
    'logf_n2_model3_ext',
    'logf_o3_model3_ext',
    'log_stellar_mass',
    'log_sfr',
    'disk_r50',
    'disk_scalelength',
    'disk_axis_ratio',
    'disk_angle',
    'disk_ellipticity',
    'inclination_angle',
    'bulge_fraction',
    'bulge_r50',
    'bulge_nsersic',
    'bulge_axis_ratio',
    'euclid_nisp_y',
    'euclid_nisp_j',
    'euclid_nisp_h',
    'euclid_nisp_y_el_model3_ext',
    'euclid_nisp_j_el_model3_ext',
    'euclid_nisp_h_el_model3_ext',
    'euclid_vis',
    'euclid_vis_el_model3_ext',
    'kappa',
    'gamma1',
    'gamma2',
)


# =============================================================================
# Catalog loading
# =============================================================================


def load_flagship2_catalog(name: str, data_dir: Union[str, Path]) -> pd.DataFrame:
    """Load and verify a named Flagship2 catalog download.

    Reads ``<data_dir>/<name>.parquet``, verifies its sha256 against the
    mandatory ``<name>.provenance.json`` sidecar, and validates the exact
    36-column schema, non-emptiness, and (halo_id, galaxy_id) uniqueness.

    Parameters
    ----------
    name : str
        Basename of the download (e.g. 'flagship2_dev').
    data_dir : str or Path
        Directory holding the parquet + provenance sidecar.

    Returns
    -------
    pd.DataFrame
        The raw catalog rows.
    """
    data_dir = Path(data_dir)
    parquet_path = data_dir / f'{name}.parquet'
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"catalog parquet not found: {parquet_path}; run "
            f"'make download-cosmohub-dev' (or download-cosmohub-data for "
            f"the production bank)"
        )
    sidecar_path = data_dir / f'{name}.provenance.json'
    if not sidecar_path.exists():
        raise RuntimeError(
            f"provenance sidecar not found: {sidecar_path}; provenance is "
            f"mandatory (re-download via 'make download-cosmohub-dev' to "
            f"regenerate it)"
        )
    provenance = json.loads(sidecar_path.read_text())
    if 'sha256' not in provenance:
        raise RuntimeError(f"{sidecar_path}: sidecar carries no 'sha256' key")
    actual_sha = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    if actual_sha != provenance['sha256']:
        raise RuntimeError(
            f"catalog sha256 mismatch for {parquet_path}: file has "
            f"{actual_sha}, provenance sidecar records "
            f"{provenance['sha256']}; the download is corrupt or was "
            f"modified -- re-download it"
        )

    df = pd.read_parquet(parquet_path)
    missing = [c for c in FLAGSHIP2_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in FLAGSHIP2_COLUMNS]
    if missing or extra:
        raise ValueError(
            f"{parquet_path}: schema mismatch vs the Flagship2 query spec; "
            f"missing columns {missing}, unexpected columns {extra}"
        )
    if len(df) == 0:
        raise ValueError(f"{parquet_path}: catalog is empty")
    # flagship2 galaxy_id is a within-halo index; (halo_id, galaxy_id) is
    # the unique key all seed streams rely on
    if df.duplicated(subset=['halo_id', 'galaxy_id']).any():
        raise ValueError(
            f"{parquet_path}: duplicate (halo_id, galaxy_id) pairs; the "
            f"per-galaxy seed keys require a unique compound id"
        )
    return df


def catalog_provenance(name: str, data_dir: Union[str, Path]) -> dict:
    """Load the provenance sidecar for a named catalog download."""
    sidecar_path = Path(data_dir) / f'{name}.provenance.json'
    if not sidecar_path.exists():
        raise RuntimeError(f"provenance sidecar not found: {sidecar_path}")
    return json.loads(sidecar_path.read_text())


# =============================================================================
# Preprocessing
# =============================================================================


def preprocess(df: pd.DataFrame, spec: CatalogPopulationSpec) -> pd.DataFrame:
    """Derive physical population columns from raw Flagship2 rows.

    Drops one-component rows (``disk_r50 <= 0``: bulge-only galaxies with
    no disk), corrects the stellar mass for little-h, converts the Halpha
    flux variant to linear flux, and derives the observed-frame continuum
    and rest-frame equivalent width at the observed Halpha wavelength.

    Unit chain (dimensional sanity):
    - f_line = 10**logf [erg/s/cm2] (integrated line flux)
    - lambda_obs = 6562.8 * (1 + z) [A]
    - f_nu (NISP band containing lambda_obs) [erg/cm2/s/Hz]
    - f_lambda = f_nu * c / lambda_obs^2
      -> [erg/cm2/s/Hz] * [A/s] / [A^2] = [erg/cm2/s/A]
    - EW_obs = f_line / f_lambda -> [erg/s/cm2] / [erg/cm2/s/A] = [A]
    - EW_rest = EW_obs / (1 + z) [A]

    Parameters
    ----------
    df : pd.DataFrame
        Raw catalog rows (full 36-column schema).
    spec : CatalogPopulationSpec
        Preprocess settings (flux_variant, h).

    Returns
    -------
    pd.DataFrame
        One row per two-component galaxy with derived physical columns;
        ``attrs['n_dropped_one_component']`` records the dropped count.
    """
    n_raw = len(df)
    disk_mask = df['disk_r50'].to_numpy() > 0.0
    out = df.loc[disk_mask].reset_index(drop=True).copy()
    n_dropped = int(n_raw - len(out))
    if len(out) == 0:
        raise ValueError(
            f"preprocess: all {n_raw} rows are one-component (disk_r50 <= 0)"
        )

    # little-h mass correction: Flagship masses carry h^-2; physical
    # logM* = catalog column - 2*log10(h)
    logm = out['log_stellar_mass'].to_numpy(dtype=np.float64) - 2.0 * np.log10(spec.h)

    flux_col = f'logf_halpha_{spec.flux_variant}'
    f_line = 10.0 ** out[flux_col].to_numpy(dtype=np.float64)  # erg/s/cm2

    z = out['true_redshift_gal'].to_numpy(dtype=np.float64)
    lambda_obs_a = HALPHA_REST_A * (1.0 + z)  # A

    # NISP band selection at the observed Halpha wavelength
    f_nu_y = out['euclid_nisp_y'].to_numpy(dtype=np.float64)
    f_nu_j = out['euclid_nisp_j'].to_numpy(dtype=np.float64)
    f_nu_h = out['euclid_nisp_h'].to_numpy(dtype=np.float64)
    f_nu = np.where(
        lambda_obs_a < NISP_YJ_EDGE_A,
        f_nu_y,
        np.where(lambda_obs_a < NISP_JH_EDGE_A, f_nu_j, f_nu_h),
    )  # erg/cm2/s/Hz

    # continuum density and rest-frame EW (see unit chain in docstring)
    f_lambda_obs = f_nu * C_A_PER_S / lambda_obs_a**2  # erg/cm2/s/A
    ew_rest_a = f_line / f_lambda_obs / (1.0 + z)  # A

    out['logm'] = logm
    out['f_line_cgs'] = f_line
    out['lambda_obs_a'] = lambda_obs_a
    out['f_lambda_cont_cgs'] = f_lambda_obs
    out['ew_rest_a'] = ew_rest_a
    out['rscale_arcsec'] = out['disk_scalelength'].to_numpy(dtype=np.float64)

    derived = {
        'logm': logm,
        'f_line_cgs': f_line,
        'f_lambda_cont_cgs': f_lambda_obs,
        'ew_rest_a': ew_rest_a,
        'rscale_arcsec': out['rscale_arcsec'].to_numpy(),
    }
    bad_counts = {}
    for col, values in derived.items():
        finite = np.isfinite(values)
        positive = values > 0 if col != 'logm' else np.ones_like(finite, dtype=bool)
        n_bad = int((~(finite & positive)).sum())
        if n_bad:
            bad_counts[col] = n_bad
    if bad_counts:
        raise ValueError(
            f"preprocess: non-finite or non-positive derived values: "
            f"{bad_counts} (of {len(out)} rows)"
        )

    keep = [
        'galaxy_id',
        'halo_id',
        'true_redshift_gal',
        'observed_redshift_gal',
        'logm',
        'log_sfr',
        'f_line_cgs',
        'lambda_obs_a',
        'f_lambda_cont_cgs',
        'ew_rest_a',
        'rscale_arcsec',
        'disk_r50',
        'disk_axis_ratio',
        'inclination_angle',
        'bulge_fraction',
        'bulge_r50',
        'bulge_nsersic',
        'kappa',
        'gamma1',
        'gamma2',
    ]
    out = out[keep]
    out.attrs['n_dropped_one_component'] = n_dropped
    return out


# =============================================================================
# Matched-filter line SNR
# =============================================================================


def matched_filter_compactness(
    reff_arcsec: np.ndarray, cosi: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """Matched-filter SNR compactness ratio C in (0, 1].

    The line-flux limit F_LIM is a point-source (5-sigma) limit; an
    extended source's matched-filter SNR is degraded by

        C = sigma_psf / sqrt(s1 * s2),  s_i = sqrt(sigma_psf^2 + sig_i^2)

    with the galaxy approximated as an elliptical Gaussian of
    sigma_major = reff / 1.678 (the exponential scalelength) and
    sigma_minor = cosi * sigma_major, and the PSF as a Gaussian with the
    diffraction proxy FWHM = 1.22 * lambda_obs / D (D = 2.36 m), converted
    to arcsec and divided by 2.355. This is the correct matched-filter
    amplitude ratio for Gaussians (NOT the peak-pixel square of an earlier
    prototype, which was a bug).

    Parameters
    ----------
    reff_arcsec : np.ndarray
        Disk half-light radius [arcsec] (Flagship2 disk_r50).
    cosi : np.ndarray
        Cosine of inclination (minor/major axis ratio of the thin disk).
    z : np.ndarray
        Redshift; sets lambda_obs = 6562.8 * (1 + z) for the PSF proxy.

    Returns
    -------
    np.ndarray
        Compactness C in (0, 1]; C -> 1 for unresolved sources.
    """
    reff_arcsec = np.asarray(reff_arcsec, dtype=np.float64)
    cosi = np.asarray(cosi, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if np.any(reff_arcsec <= 0):
        raise ValueError("matched_filter_compactness: reff_arcsec must be positive")

    # diffraction PSF proxy: FWHM = 1.22 lambda/D [rad] -> arcsec -> sigma
    lambda_obs_m = HALPHA_REST_A * (1.0 + z) * 1e-10  # A -> m
    fwhm_arcsec = 1.22 * lambda_obs_m / ROMAN_APERTURE_M * ARCSEC_PER_RAD
    sigma_psf = fwhm_arcsec / FWHM_TO_SIGMA

    sig_maj = reff_arcsec / R50_TO_SIGMA
    sig_min = cosi * sig_maj
    s1 = np.sqrt(sigma_psf**2 + sig_maj**2)
    s2 = np.sqrt(sigma_psf**2 + sig_min**2)
    return sigma_psf / np.sqrt(s1 * s2)


def compute_line_snr(f_line: np.ndarray, compactness: np.ndarray) -> np.ndarray:
    """Per-exposure matched-filter line SNR.

    SNR = 5 * (f_line / F_LIM) * C: F_LIM is the 5-sigma point-source
    limit, so a point source at F_LIM has SNR 5; extended sources are
    degraded by the compactness ratio C.
    """
    return (
        F_LIM_NSIGMA
        * np.asarray(f_line, dtype=np.float64)
        / F_LIM_CGS
        * np.asarray(compactness, dtype=np.float64)
    )


# =============================================================================
# Seeded per-galaxy draws
# =============================================================================


def _galaxy_rng(
    seed: int, tag: int, halo_id: int, galaxy_id: int
) -> np.random.Generator:
    """Per-galaxy generator keyed on the unique (halo_id, galaxy_id) pair."""
    ss = np.random.SeedSequence([seed, tag, int(halo_id), int(galaxy_id)])
    return np.random.default_rng(ss)


def _draw_geometry(
    seed: int,
    halo_ids: np.ndarray,
    galaxy_ids: np.ndarray,
    cosi_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Isotropic orientation redraw: cosi ~ U(range), theta_int ~ U(0, pi).

    One GEOMETRY-stream generator per galaxy (draw order: cosi then theta);
    O(n_catalog) generator constructions, ~10 s at 300k rows.
    """
    n = len(halo_ids)
    cosi = np.empty(n, dtype=np.float64)
    theta = np.empty(n, dtype=np.float64)
    lo, hi = cosi_range
    for i in range(n):
        rng = _galaxy_rng(seed, _POP_GEOMETRY, halo_ids[i], galaxy_ids[i])
        cosi[i] = rng.uniform(lo, hi)
        theta[i] = rng.uniform(0.0, np.pi)
    return cosi, theta


def _paint_kinematics(
    seed: int,
    halo_ids: np.ndarray,
    galaxy_ids: np.ndarray,
    logm: np.ndarray,
    z: np.ndarray,
    spec: CatalogPopulationSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    """Paint vcirc (inverted TFR) and sigma0 (affine in z) with scatter.

    TFR (Ubler+2017, inverted to logv|logM form):
        logv = logv0 + (logM - logm0) / slope + N(0, scatter_dex)
    sigma0 (Ubler+2019 affine evolution):
        sigma0 = intercept + slope * z + N(0, scatter), resampled while
        below min_kms.

    One PAINT-stream generator per galaxy; draw order (TFR normal first,
    then sigma0 normals) is part of the determinism contract.
    """
    n = len(halo_ids)
    vcirc = np.empty(n, dtype=np.float64)
    sigma0 = np.empty(n, dtype=np.float64)
    for i in range(n):
        rng = _galaxy_rng(seed, _POP_PAINT, halo_ids[i], galaxy_ids[i])
        logv = (
            spec.tfr_logv0
            + (logm[i] - spec.tfr_logm0) / spec.tfr_slope
            + rng.normal(0.0, spec.tfr_scatter_dex)
        )
        vcirc[i] = 10.0**logv
        mu = spec.sigma0_intercept_kms + spec.sigma0_slope_kms * z[i]
        value = rng.normal(mu, spec.sigma0_scatter_kms)
        while value < spec.sigma0_min_kms:
            value = rng.normal(mu, spec.sigma0_scatter_kms)
        sigma0[i] = value
    return vcirc, sigma0


def _draw_shear(
    seed: int,
    halo_ids: np.ndarray,
    galaxy_ids: np.ndarray,
    spec: CatalogPopulationSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-galaxy shear draw shared by both ring members.

    One (g1, g2) per galaxy row (the downstream ring expansion duplicates
    the row, so the pair shares the draw by construction): iid
    N(0, sigma^2) per component, redrawn until |g| < gmax.
    """
    n = len(halo_ids)
    g1 = np.empty(n, dtype=np.float64)
    g2 = np.empty(n, dtype=np.float64)
    for i in range(n):
        rng = _galaxy_rng(seed, _POP_SHEAR, halo_ids[i], galaxy_ids[i])
        while True:
            a = rng.normal(0.0, spec.shear_sigma)
            b = rng.normal(0.0, spec.shear_sigma)
            if np.hypot(a, b) < spec.shear_gmax:
                break
        g1[i] = a
        g2[i] = b
    return g1, g2


def _draw_mass_prior(
    seed: int,
    halo_ids: np.ndarray,
    galaxy_ids: np.ndarray,
    logm: np.ndarray,
    spec: CatalogPopulationSpec,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulated photometric mass + the TFR-implied vcirc prior.

    logm_obs = logm + N(0, logm_obs_scatter_dex); the fit prior center is
    the TFR at logm_obs, with width combining the intrinsic TFR scatter and
    the propagated mass error: sigma_dex = sqrt(scatter_dex^2 +
    (logm_obs_scatter_dex / slope)^2).
    """
    n = len(halo_ids)
    logm_obs = np.empty(n, dtype=np.float64)
    for i in range(n):
        rng = _galaxy_rng(seed, _POP_PRIOR, halo_ids[i], galaxy_ids[i])
        logm_obs[i] = logm[i] + rng.normal(0.0, spec.logm_obs_scatter_dex)
    prior_mu = 10.0 ** (spec.tfr_logv0 + (logm_obs - spec.tfr_logm0) / spec.tfr_slope)
    prior_sigma_dex = np.full(
        n,
        np.sqrt(
            spec.tfr_scatter_dex**2 + (spec.logm_obs_scatter_dex / spec.tfr_slope) ** 2
        ),
    )
    return logm_obs, prior_mu, prior_sigma_dex


# =============================================================================
# Full chain
# =============================================================================


def build_population(
    spec: EnsembleSpec, data_dir: Optional[Union[str, Path]] = None
) -> Tuple[pd.DataFrame, dict]:
    """Build the catalog-backed population table for a campaign spec.

    Chain: load + verify catalog -> preprocess -> selection cuts that do
    not depend on orientation (z_range, bulge cuts) -> per-galaxy isotropic
    orientation redraw -> matched-filter line SNR with the REDRAWN cosi ->
    SNR cut -> seeded subsample of n_galaxies (without replacement) ->
    kinematic paint + shear + mass-prior draws.

    All per-galaxy draws are keyed on (halo_id, galaxy_id), so values are
    independent of the cut order and of the catalog superset.

    Parameters
    ----------
    spec : EnsembleSpec
        A catalog-mode ensemble spec (spec.catalog_population set).
    data_dir : str or Path, optional
        Override for the spec's catalog data_dir.

    Returns
    -------
    (pd.DataFrame, dict)
        One row per galaxy (ring expansion happens downstream) and a meta
        dict with per-stage counts, hashes, and physics provenance.
    """
    cp = spec.catalog_population
    if cp is None:
        raise ValueError(
            "build_population requires a catalog-mode spec "
            "(population.type: catalog)"
        )
    data_dir = Path(data_dir) if data_dir is not None else Path(cp.catalog_data_dir)

    raw = load_flagship2_catalog(cp.catalog_download, data_dir)
    n_raw = len(raw)
    pre = preprocess(raw, cp)
    n_disk = len(pre)
    kills: Dict[str, int] = {'one_component': int(pre.attrs['n_dropped_one_component'])}

    # orientation-independent cuts first (shrinks the per-galaxy geometry
    # loop; id-keyed streams make the result order-independent)
    z = pre['true_redshift_gal'].to_numpy(dtype=np.float64)
    z_mask = (z >= cp.z_range[0]) & (z <= cp.z_range[1])
    kills['z_range'] = int((~z_mask).sum())
    pre = pre.loc[z_mask].reset_index(drop=True)

    if cp.bulge_fraction_max is not None:
        bf_mask = pre['bulge_fraction'].to_numpy() <= cp.bulge_fraction_max
        kills['bulge_fraction'] = int((~bf_mask).sum())
        pre = pre.loc[bf_mask].reset_index(drop=True)
    else:
        kills['bulge_fraction'] = 0

    if cp.bulge_nsersic_range is not None:
        ns = pre['bulge_nsersic'].to_numpy()
        ns_mask = (ns >= cp.bulge_nsersic_range[0]) & (ns <= cp.bulge_nsersic_range[1])
        kills['bulge_nsersic'] = int((~ns_mask).sum())
        pre = pre.loc[ns_mask].reset_index(drop=True)
    else:
        kills['bulge_nsersic'] = 0

    # isotropic orientation redraw (GEOMETRY stream), then line SNR with
    # the redrawn cosi
    halo_ids = pre['halo_id'].to_numpy()
    galaxy_ids = pre['galaxy_id'].to_numpy()
    cosi, theta_int = _draw_geometry(spec.seed, halo_ids, galaxy_ids, cp.cosi_range)
    compactness = matched_filter_compactness(
        pre['disk_r50'].to_numpy(dtype=np.float64),
        cosi,
        pre['true_redshift_gal'].to_numpy(dtype=np.float64),
    )
    snr_line = compute_line_snr(pre['f_line_cgs'].to_numpy(), compactness)
    pre = pre.assign(
        cosi=cosi, theta_int=theta_int, compactness=compactness, snr_line=snr_line
    )

    snr_mask = pre['snr_line'].to_numpy() >= cp.snr_line_min
    kills['snr_line'] = int((~snr_mask).sum())
    selected = pre.loc[snr_mask].reset_index(drop=True)
    n_selected = len(selected)

    if n_selected < cp.n_galaxies:
        raise ValueError(
            f"selection yields {n_selected} galaxies but sample.n_galaxies "
            f"= {cp.n_galaxies}; loosen the selection or shrink the sample "
            f"(stage counts: n_raw={n_raw}, n_disk={n_disk}, kills={kills})"
        )

    # seeded subsample without replacement (SAMPLE stream); sorting by the
    # unique compound id first makes the draw independent of catalog row
    # order
    selected = selected.sort_values(['halo_id', 'galaxy_id']).reset_index(drop=True)
    sample_rng = np.random.default_rng(np.random.SeedSequence([spec.seed, _POP_SAMPLE]))
    picks = sample_rng.choice(n_selected, size=cp.n_galaxies, replace=False)
    sample = (
        selected.iloc[np.sort(picks)]
        .sort_values(['halo_id', 'galaxy_id'])
        .reset_index(drop=True)
    )

    halo_ids = sample['halo_id'].to_numpy()
    galaxy_ids = sample['galaxy_id'].to_numpy()
    logm = sample['logm'].to_numpy(dtype=np.float64)
    z_sample = sample['true_redshift_gal'].to_numpy(dtype=np.float64)
    vcirc, sigma0 = _paint_kinematics(
        spec.seed, halo_ids, galaxy_ids, logm, z_sample, cp
    )
    g1, g2 = _draw_shear(spec.seed, halo_ids, galaxy_ids, cp)
    logm_obs, prior_mu, prior_sigma_dex = _draw_mass_prior(
        spec.seed, halo_ids, galaxy_ids, logm, cp
    )

    population = pd.DataFrame(
        {
            'pop_index': np.arange(cp.n_galaxies, dtype=np.int64),
            'galaxy_id': galaxy_ids,
            'halo_id': halo_ids,
            'z': z_sample,
            'z_obs_catalog': sample['observed_redshift_gal'].to_numpy(dtype=np.float64),
            'logm': logm,
            'log_sfr': sample['log_sfr'].to_numpy(dtype=np.float64),
            'f_line_cgs': sample['f_line_cgs'].to_numpy(),
            'ew_rest_a': sample['ew_rest_a'].to_numpy(),
            'f_lambda_cont_cgs': sample['f_lambda_cont_cgs'].to_numpy(),
            'rscale_arcsec': sample['rscale_arcsec'].to_numpy(),
            'bulge_fraction': sample['bulge_fraction'].to_numpy(dtype=np.float64),
            'bulge_r50_arcsec': sample['bulge_r50'].to_numpy(dtype=np.float64),
            'bulge_nsersic': sample['bulge_nsersic'].to_numpy(dtype=np.float64),
            'cosi': sample['cosi'].to_numpy(),
            'theta_int': sample['theta_int'].to_numpy(),
            'g1': g1,
            'g2': g2,
            'vcirc_kms': vcirc,
            'sigma0_kms': sigma0,
            'snr_line': sample['snr_line'].to_numpy(),
            'compactness': sample['compactness'].to_numpy(),
            'logm_obs': logm_obs,
            'prior_vcirc_mu_kms': prior_mu,
            'prior_vcirc_sigma_dex': prior_sigma_dex,
            'catalog_inclination_deg': sample['inclination_angle'].to_numpy(
                dtype=np.float64
            ),
            'catalog_disk_axis_ratio': sample['disk_axis_ratio'].to_numpy(
                dtype=np.float64
            ),
            'kappa': sample['kappa'].to_numpy(dtype=np.float64),
            'gamma1_field': sample['gamma1'].to_numpy(dtype=np.float64),
            'gamma2_field': sample['gamma2'].to_numpy(dtype=np.float64),
        }
    )

    provenance = catalog_provenance(cp.catalog_download, data_dir)
    meta = {
        'run_name': spec.run_name,
        'seed': spec.seed,
        'catalog_name': cp.catalog_download,
        'catalog_sha256': provenance['sha256'],
        'catalog_query_id': provenance.get('query_id'),
        'n_raw': n_raw,
        'n_disk': n_disk,
        'kills': kills,
        'n_selected': n_selected,
        'n_sampled': int(cp.n_galaxies),
        'flux_variant': cp.flux_variant,
        'f_lim_cgs': F_LIM_CGS,
        'f_lim_provenance': (
            'ROTAC HLWAS spectroscopy per-pass 5-sigma point-source '
            'emission-line flux limit'
        ),
        'psf_proxy': (
            f'diffraction FWHM = 1.22 * lambda_obs / {ROMAN_APERTURE_M} m; '
            f'sigma = FWHM / {FWHM_TO_SIGMA}'
        ),
        'ew_note': 'EW_obs = EW_rest * (1 + z); table stores rest-frame EW [A]',
        'stream_tags': {
            'sample': _POP_SAMPLE,
            'geometry': _POP_GEOMETRY,
            'paint': _POP_PAINT,
            'shear': _POP_SHEAR,
            'prior': _POP_PRIOR,
        },
    }
    return population, meta


def write_population(
    out_dir: Union[str, Path], df: pd.DataFrame, meta: dict
) -> Tuple[Path, Path]:
    """Write population.parquet + population_meta.json into a directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / 'population.parquet'
    meta_path = out_dir / 'population_meta.json'
    df.to_parquet(parquet_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2) + '\n')
    return parquet_path, meta_path
