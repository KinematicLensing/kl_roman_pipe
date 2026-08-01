"""
Catalog-backed galaxy population: catalog rows + kinematic paint.

Builds the per-galaxy population table for ``population.type: catalog``
ensemble campaigns. Structural and flux truths (disk sizes, Halpha flux,
continuum, redshift, bulge fraction where the catalog has one) come from an
input catalog behind a ``CatalogAdapter`` (``kl_pipe.ensemble.catalogs``;
the spec's ``population.catalog.kind`` selects it); kinematics (vcirc via
an inverted Tully-Fisher relation, sigma0 via an affine-in-z relation) and
bulge morphology (Sersic index + size, for catalogs whose own bulge
assignments are unusable -- see BULGE_* constants) are painted on with
seeded scatter; orientation is an isotropic redraw (the catalog inclination
is kept for validation only); shear is drawn per ring pair.

Determinism
-----------
- Every per-galaxy draw uses a numpy SeedSequence keyed on
  ``[spec.seed, STREAM_TAG, *ids]`` where ``ids`` are the row's values of
  the adapter's ``id_columns``, in the adapter's declared order (for
  Flagship2: (halo_id, galaxy_id), since its galaxy_id is a within-halo
  index and only the pair is unique). Draws are therefore independent of
  catalog row order and of the selection applied: a galaxy keeps its
  cosi/theta/paint/shear values under any superset catalog download.
- The subsample draw uses a single stream keyed ``[spec.seed, SAMPLE_TAG]``
  over the selected rows sorted by the id columns.
- Stream tags are disjoint from the expander's (1/2/3).

This module is numpy+pandas only (population building is a one-shot host
task, not traced model code).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from kl_pipe.ensemble.catalogs import (
    catalog_provenance,
    get_catalog_adapter,
    load_catalog,
    validate_columns,
)
from kl_pipe.ensemble.spec import CatalogPopulationSpec, EnsembleSpec
from kl_pipe.noise import FWHM_TO_SIGMA
from kl_pipe.surveys import roman
from kl_pipe.surveys.roman import (
    F_LIM_COADD_CGS,
    F_LIM_NSIGMA,
    F_LIM_PER_PASS_CGS,
    IMAGING_DEPTH_AB,
    IMAGING_DEPTH_NSIGMA,
    N_GRISM_PASSES,
    ROMAN_APERTURE_M,
    ROMAN_IMAGING_BANDS,
    band_flux_sigma_ujy,
    compute_band_snr,
    compute_line_snr_per_pass,
    compute_line_snr_total,
    fiducial_compactness,
    line_flux_sigma_cgs,
    matched_filter_compactness,
    psf_fwhm_arcsec,
)

# =============================================================================
# Constants (each with provenance)
# =============================================================================

# seed-stream domain tags (disjoint from expander's 1/2/3); per-galaxy
# streams key [seed, TAG, halo_id, galaxy_id], the sample stream keys
# [seed, TAG] only
_POP_SAMPLE = 10
_POP_GEOMETRY = 11
_POP_PAINT = 12
_POP_SHEAR = 13
_POP_PRIOR = 14
_POP_BULGE = 15
_POP_STRUCTURE = 16
_POP_FLUXOBS = 17

# Component scales relative to the catalog disk scale length. The catalog
# gives one size per galaxy; the rotation curve and the line-emitting gas do
# not share it, so each carries a painted ratio. Fit priors use these same
# constants, so the prior is the distribution the truth was drawn from.
#
# Turnover radius of the arctan rotation curve, in disk scale lengths.
# Derived from the PROBES-I public per-galaxy rotation-curve fits (Stone et
# al. 2022, ApJS 262, 33; Zenodo 10.5281/zenodo.10456320): 753 late-type,
# clean-photometry galaxies with published tanh and Courteau-97 turnover
# radii, converted to the arctan form by least-squares refit of each
# galaxy's model curve over 0-3.2 R_d and divided by R_d = Re(r-band)/1.678.
# Measured: median 0.56 (0.53 for v_c > 120 km/s, our selected regime),
# scatter 0.28 dex; form systematic (tanh vs C97) 0.02 dex, refit-range
# systematic +/-0.05 dex over 2.2-4.5 R_d, denominator systematic (Sersic vs
# nonparametric half-light radius; n < 2 subset) <= 0.03 dex. Adopted below
# rounded to one significant figure, scatter likewise.
#
# No z ~ 1 survey publishes this distribution, so the anchor is z ~ 0 and the
# redshift extrapolation is the stated residual systematic. z ~ 1 surveys
# either pin the turnover as a resolution nuisance (Weiner et al. 2006,
# Kassin et al. 2007: fixed 0.2") or fit it freely and never tabulate it
# (Miller et al. 2011, whose Fig. 2 caption calls r_t ~ 0.4 R_d typical --
# consistent with the adopted median at 0.35 sigma; Stott et al. 2016 fit
# the same arctan form to ~600 KROSS galaxies and release no turnover
# column). Catinella et al. 2006's Polyex template scales convert to
# r_t/R_d = 0.24-0.57 with a matching luminosity trend, bracketing the
# adopted value from an independent local tradition.
VEL_RSCALE_RATIO_MEDIAN = 0.5
VEL_RSCALE_RATIO_DEX = 0.3

# Halpha extent relative to the stellar continuum. Nelson et al. 2012 measure
# a median of 1.3 and an rms scatter of 0.2 dex at z ~ 1 in 3D-HST, on the
# ratio of effective radii, which for two exponentials is the ratio of scale
# lengths. Both numbers come from that one sample rather than being mixed
# across sources. Their rms is the observed one and so includes measurement
# error, making this a slight over-dispersion of the intrinsic population
# rather than an underestimate. Matharu et al. 2022 measure 1.2 +/- 0.1 over
# 0.5 < z < 1.7 with no redshift evolution, consistent with the median here,
# but quote no population width, so they cannot supply the scatter.
HALPHA_RSCALE_RATIO_MEDIAN = 1.3
HALPHA_RSCALE_RATIO_DEX = 0.2

# Per-component astrometric centroid offset. Each component's offset is drawn
# separately rather than shared; about one Roman pixel.
CENTROID_SCATTER_ARCSEC = 0.1

# Offset between the ionized-gas and stellar-continuum centroids of the same
# galaxy. Star formation is clumpy and need not be centered on the older
# stellar disk, so the two are close but not identical. Half a Roman pixel is
# 0.42-0.48 kpc over our redshift range, about 11% of the median Halpha
# half-light radius at z ~ 1 (Nelson et al. 2012), which is the right order
# given that clumps carry only 10-15% of the star formation.
CONT_CENTROID_OFFSET_ARCSEC = 0.055

# Systemic velocity relative to the catalog redshift. One grism pixel at
# 1.1 nm is about 200 km/s at the observed Halpha wavelength, and the line
# centroid is measured to tens of km/s, so a painted offset of this size sits
# well inside what the data constrains.
V0_SCATTER_KMS = 25.0

# Literature-anchored bulge morphology paint. Flagship2 assigns the bulge
# Sersic index and bulge size as random draws uncorrelated with every other
# galaxy property (Castander et al. 2025 Sect. 5.5; Sect. 7.1: "we have not
# enforced the correlations between different morphological parameters"),
# and the low-n tail of its calibrating CANDELS decompositions is known to
# be inflated by fit artifacts (Dimauro et al. 2018 Sect. 5.1). Both are
# therefore replaced with simple empirical distributions; the magnitude-
# calibrated bulge_fraction is kept from the catalog. Catalog values are
# retained in the population table as catalog_* columns for validation.
#
# Sersic index: bimodal pseudobulge/classical mixture split at n = 2
# (Fisher & Drory 2008). Component medians and widths from Gadotti 2009
# Table 3 (i band): pseudobulge n = 1.5 +/- 0.9, classical n = 3.4 +/- 1.3.
# Mixture weight for a disk-dominated sample from Mendez-Abreu et al. 2010:
# ~70% of bulges at B/T <= 0.3 have n <= 2. Upper bound 6.0 = the Sersic
# emulator's support, and the renderer is validated over that whole range.
# Stopping the classical component short of it would misspecify 10.7% of all
# bulges at no saving: render grid and per-render cost are flat in n.
BULGE_N_MAX = 6.0
BULGE_PSEUDO_WEIGHT = 0.7
BULGE_PSEUDO_N = (1.5, 0.9, 0.5, 2.0)  # (mu, sd, low, high)
BULGE_CLASSICAL_N = (3.4, 1.3, 2.0, BULGE_N_MAX)
# Bulge-to-disk size ratio bulge_r50 / disk_r50: lognormal. Median 0.3
# brackets the direct z ~ 0.5-2.5 measurement (Lang et al. 2014: ~0.2, bulge
# n fixed) and local Gadotti 2009-derived values (0.25-0.36 by bulge type);
# ln-scatter 0.4 from Gadotti 2009 Table 3 sd/median converted to log space.
# Capped below 1: a steep bulge larger than its disk is unphysical (and the
# uncapped Flagship2 paint produced such objects by chance).
BULGE_SIZE_RATIO_MEDIAN = 0.3
BULGE_SIZE_RATIO_LN_SCATTER = 0.4
BULGE_SIZE_RATIO_MAX = 1.0

# =============================================================================
# Seeded per-galaxy draws
# =============================================================================


def _galaxy_rng(seed: int, tag: int, ids: np.ndarray) -> np.random.Generator:
    """Per-galaxy generator keyed on the row's unique id-column values.

    ``ids`` is one row of the (n, k) id array (the adapter's ``id_columns``
    values, in declared order); the key composition must never change --
    it defines every seeded draw.
    """
    ss = np.random.SeedSequence([seed, tag, *(int(v) for v in ids)])
    return np.random.default_rng(ss)


def _draw_geometry(
    seed: int,
    ids: np.ndarray,
    cosi_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Isotropic orientation redraw: cosi ~ U(range), theta_int ~ U(0, pi).

    One GEOMETRY-stream generator per galaxy (draw order: cosi then theta);
    O(n_catalog) generator constructions, ~10 s at 300k rows. ``ids`` is
    the (n, k) array of id-column values.
    """
    n = len(ids)
    cosi = np.empty(n, dtype=np.float64)
    theta = np.empty(n, dtype=np.float64)
    lo, hi = cosi_range
    for i in range(n):
        rng = _galaxy_rng(seed, _POP_GEOMETRY, ids[i])
        cosi[i] = rng.uniform(lo, hi)
        theta[i] = rng.uniform(0.0, np.pi)
    return cosi, theta


def _paint_kinematics(
    seed: int,
    ids: np.ndarray,
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

    One PAINT-stream generator per galaxy; the draw order (TFR normal
    first, then sigma0 normals) must never change.
    """
    n = len(ids)
    vcirc = np.empty(n, dtype=np.float64)
    sigma0 = np.empty(n, dtype=np.float64)
    for i in range(n):
        rng = _galaxy_rng(seed, _POP_PAINT, ids[i])
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


def _paint_structure(
    seed: int,
    ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paint the component scale ratios and the systemic velocity offset.

    Returns the kinematic and Halpha scale lengths as ratios to the catalog
    disk scale length, plus v0 in km/s. One generator per galaxy; the draw
    order must never change.
    """
    n = len(ids)
    vel_ratio = np.empty(n, dtype=np.float64)
    line_ratio = np.empty(n, dtype=np.float64)
    v0 = np.empty(n, dtype=np.float64)
    ln10 = np.log(10.0)
    for i in range(n):
        rng = _galaxy_rng(seed, _POP_STRUCTURE, ids[i])
        vel_ratio[i] = VEL_RSCALE_RATIO_MEDIAN * np.exp(
            rng.normal(0.0, VEL_RSCALE_RATIO_DEX * ln10)
        )
        line_ratio[i] = HALPHA_RSCALE_RATIO_MEDIAN * np.exp(
            rng.normal(0.0, HALPHA_RSCALE_RATIO_DEX * ln10)
        )
        v0[i] = rng.normal(0.0, V0_SCATTER_KMS)
    return vel_ratio, line_ratio, v0


def _draw_shear(
    seed: int,
    ids: np.ndarray,
    spec: CatalogPopulationSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-galaxy shear draw shared by both ring members.

    One (g1, g2) per galaxy row (the downstream ring expansion duplicates
    the row, so the pair shares the draw by construction): iid
    N(0, sigma^2) per component, redrawn until |g| < gmax.
    """
    n = len(ids)
    g1 = np.empty(n, dtype=np.float64)
    g2 = np.empty(n, dtype=np.float64)
    for i in range(n):
        rng = _galaxy_rng(seed, _POP_SHEAR, ids[i])
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
    ids: np.ndarray,
    logm: np.ndarray,
    spec: CatalogPopulationSpec,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulated photometric mass + the TFR-implied vcirc prior.

    logm_obs = logm + N(0, logm_obs_scatter_dex); the fit prior center is
    the TFR at logm_obs, with width combining the intrinsic TFR scatter and
    the propagated mass error: sigma_dex = sqrt(scatter_dex^2 +
    (logm_obs_scatter_dex / slope)^2).
    """
    n = len(ids)
    logm_obs = np.empty(n, dtype=np.float64)
    for i in range(n):
        rng = _galaxy_rng(seed, _POP_PRIOR, ids[i])
        logm_obs[i] = logm[i] + rng.normal(0.0, spec.logm_obs_scatter_dex)
    prior_mu = 10.0 ** (spec.tfr_logv0 + (logm_obs - spec.tfr_logm0) / spec.tfr_slope)
    prior_sigma_dex = np.full(
        n,
        np.sqrt(
            spec.tfr_scatter_dex**2 + (spec.logm_obs_scatter_dex / spec.tfr_slope) ** 2
        ),
    )
    return logm_obs, prior_mu, prior_sigma_dex


def _draw_flux_measurements(
    seed: int,
    ids: np.ndarray,
    f_line_cgs: np.ndarray,
    sigma_line_cgs: np.ndarray,
    band_fluxes_ujy: Dict[str, np.ndarray],
    band_sigmas_ujy: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Seeded simulated flux measurements (the flux-prior centers).

    One FLUXOBS-stream generator per galaxy; the draw order (line first,
    then the bands in sorted name order) must never change. Mirrors the logm_obs pattern: the fit prior is centered on a noisy
    simulated measurement rather than on truth. A draw can be non-positive
    at low SNR -- real photometry reports negative fluxes there -- and the
    TruncatedNormal prior support handles an out-of-bounds center as a
    proper one-sided distribution, so draws are not clipped. Production
    selections (SNR >~ 20) never produce one.
    """
    n = len(ids)
    band_names = sorted(band_fluxes_ujy)
    f_line_obs = np.empty(n, dtype=np.float64)
    band_obs = {b: np.empty(n, dtype=np.float64) for b in band_names}
    for i in range(n):
        rng = _galaxy_rng(seed, _POP_FLUXOBS, ids[i])
        f_line_obs[i] = f_line_cgs[i] + rng.normal(0.0, sigma_line_cgs[i])
        for b in band_names:
            band_obs[b][i] = band_fluxes_ujy[b][i] + rng.normal(
                0.0, band_sigmas_ujy[b][i]
            )
    return f_line_obs, band_obs


def _truncated_normal(
    rng: np.random.Generator, mu: float, sd: float, low: float, high: float
) -> float:
    """Rejection-sampled truncated normal (the draw count advances the
    per-galaxy stream, so the rejection scheme must never change)."""
    while True:
        value = rng.normal(mu, sd)
        if low <= value <= high:
            return value


def _paint_bulge(
    seed: int,
    ids: np.ndarray,
    disk_r50_arcsec: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Paint bulge Sersic index and size (see BULGE_* constants).

    One BULGE-stream generator per galaxy; the draw order (class uniform,
    then the n rejection draws, then the size-ratio rejection draws) must
    never change.
    """
    n = len(ids)
    nsersic = np.empty(n, dtype=np.float64)
    bulge_r50 = np.empty(n, dtype=np.float64)
    for i in range(n):
        rng = _galaxy_rng(seed, _POP_BULGE, ids[i])
        pars = (
            BULGE_PSEUDO_N if rng.uniform() < BULGE_PSEUDO_WEIGHT else BULGE_CLASSICAL_N
        )
        nsersic[i] = _truncated_normal(rng, *pars)
        while True:
            ratio = BULGE_SIZE_RATIO_MEDIAN * np.exp(
                rng.normal(0.0, BULGE_SIZE_RATIO_LN_SCATTER)
            )
            if ratio < BULGE_SIZE_RATIO_MAX:
                break
        bulge_r50[i] = ratio * disk_r50_arcsec[i]
    return nsersic, bulge_r50


# =============================================================================
# Full chain
# =============================================================================


def build_population(
    spec: EnsembleSpec, data_dir: Optional[Union[str, Path]] = None
) -> Tuple[pd.DataFrame, dict]:
    """Build the catalog-backed population table for a campaign spec.

    Chain: load + verify catalog (via the spec's catalog adapter) ->
    adapter preprocess to the standardized columns -> selection cuts that do not
    depend on orientation (z_range, bulge cuts) -> per-galaxy isotropic
    orientation redraw -> matched-filter line SNR with the REDRAWN cosi ->
    SNR cut -> seeded subsample of n_galaxies (without replacement) ->
    kinematic paint + shear + mass-prior draws.

    All per-galaxy draws are keyed on the adapter's id columns, so values
    are independent of the cut order and of the catalog superset.

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
    adapter = get_catalog_adapter(cp.catalog_kind)
    data_dir = Path(data_dir) if data_dir is not None else Path(cp.catalog_data_dir)

    raw = load_catalog(adapter, cp.catalog_download, data_dir)
    n_raw = len(raw)
    pre = adapter.preprocess(raw, cp)
    validate_columns(adapter, pre)
    n_disk = len(pre)
    kills: Dict[str, int] = {k: int(v) for k, v in pre.attrs['kills'].items()}

    # orientation-independent cuts first (shrinks the per-galaxy geometry
    # loop; id-keyed streams make the result order-independent)
    z = pre['z'].to_numpy(dtype=np.float64)
    z_mask = (z >= cp.z_range[0]) & (z <= cp.z_range[1])
    kills['z_range'] = int((~z_mask).sum())
    pre = pre.loc[z_mask].reset_index(drop=True)

    if cp.bulge_fraction_max is not None:
        if not adapter.has_bulge:
            raise ValueError(
                f"selection.bulge_fraction_max cuts on the catalog "
                f"bulge_fraction column, which the '{adapter.kind}' catalog "
                f"does not carry; remove the cut from the spec"
            )
        bf_mask = pre['bulge_fraction'].to_numpy() <= cp.bulge_fraction_max
        kills['bulge_fraction'] = int((~bf_mask).sum())
        pre = pre.loc[bf_mask].reset_index(drop=True)
    else:
        kills['bulge_fraction'] = 0

    if cp.bulge_nsersic_range is not None:
        raise ValueError(
            "selection.bulge_nsersic_range cuts on the catalog bulge_nsersic "
            "column, which the population paint now replaces (see BULGE_* "
            "constants); the cut is meaningless -- remove it from the spec"
        )
    kills['bulge_nsersic'] = 0

    if cp.paint_bulge and not adapter.has_bulge:
        raise ValueError(
            f"paint.bulge: true requires a catalog with bulge columns; the "
            f"'{adapter.kind}' catalog carries none -- set paint.bulge: false"
        )

    # isotropic orientation redraw (GEOMETRY stream), then line SNR with
    # the redrawn cosi
    ids = pre[list(adapter.id_columns)].to_numpy()
    cosi, theta_int = _draw_geometry(spec.seed, ids, cp.cosi_range)
    compactness = matched_filter_compactness(
        pre['disk_r50'].to_numpy(dtype=np.float64),
        cosi,
        pre['z'].to_numpy(dtype=np.float64),
    )
    f_line = pre['f_line_cgs'].to_numpy()
    snr_line_per_pass = compute_line_snr_per_pass(f_line, compactness)
    snr_line_total = compute_line_snr_total(f_line, compactness)
    pre = pre.assign(
        cosi=cosi,
        theta_int=theta_int,
        compactness=compactness,
        snr_line_per_pass=snr_line_per_pass,
        snr_line_total=snr_line_total,
    )

    # resolvability: a galaxy much smaller than the PSF has no measurable
    # velocity gradient, so there is nothing for a kinematic fit to recover.
    # The cut belongs here rather than in the model: referencing the flux
    # limit to the extended fiducial REWARDS compact sources (C -> 1), so the
    # corrected selection actively pulls in sub-PSF objects that the
    # point-source reference happened to screen out.
    if cp.min_r50_over_psf_fwhm is not None:
        r50_over_fwhm = pre['disk_r50'].to_numpy(dtype=np.float64) / psf_fwhm_arcsec(
            pre['z'].to_numpy(dtype=np.float64)
        )
        res_mask = r50_over_fwhm >= cp.min_r50_over_psf_fwhm
        kills['resolvability'] = int((~res_mask).sum())
        pre = pre.loc[res_mask].reset_index(drop=True)
    else:
        kills['resolvability'] = 0

    snr_mask = pre['snr_line_total'].to_numpy() >= cp.snr_line_total_min
    kills['snr_line_total'] = int((~snr_mask).sum())
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
    id_cols = list(adapter.id_columns)
    selected = selected.sort_values(id_cols).reset_index(drop=True)
    sample_rng = np.random.default_rng(np.random.SeedSequence([spec.seed, _POP_SAMPLE]))
    picks = sample_rng.choice(n_selected, size=cp.n_galaxies, replace=False)
    sample = selected.iloc[np.sort(picks)].sort_values(id_cols).reset_index(drop=True)

    ids = sample[id_cols].to_numpy()
    logm = sample['logm'].to_numpy(dtype=np.float64)
    z_sample = sample['z'].to_numpy(dtype=np.float64)
    vcirc, sigma0 = _paint_kinematics(spec.seed, ids, logm, z_sample, cp)
    g1, g2 = _draw_shear(spec.seed, ids, cp)
    vel_ratio, line_ratio, v0_kms = _paint_structure(spec.seed, ids)
    logm_obs, prior_mu, prior_sigma_dex = _draw_mass_prior(spec.seed, ids, logm, cp)

    # per-band imaging SNR against the published point-source depths, plus
    # the simulated flux measurements (line + bands) that center the fit's
    # flux priors; the measurement errors depend only on shape (compactness),
    # not on the flux itself
    reff = sample['disk_r50'].to_numpy(dtype=np.float64)
    cosi_sample = sample['cosi'].to_numpy(dtype=np.float64)
    f_line_sample = sample['f_line_cgs'].to_numpy(dtype=np.float64)
    sigma_line = line_flux_sigma_cgs(sample['compactness'].to_numpy(dtype=np.float64))
    band_fluxes = {
        b: sample[f'flux_{b.lower()}_ujy'].to_numpy(dtype=np.float64)
        for b in ROMAN_IMAGING_BANDS
    }
    band_sigmas = {
        b: band_flux_sigma_ujy(reff, cosi_sample, b) for b in ROMAN_IMAGING_BANDS
    }
    band_snrs = {
        b: compute_band_snr(band_fluxes[b], reff, cosi_sample, b)
        for b in ROMAN_IMAGING_BANDS
    }
    f_line_obs, band_obs = _draw_flux_measurements(
        spec.seed, ids, f_line_sample, sigma_line, band_fluxes, band_sigmas
    )
    columns: Dict[str, np.ndarray] = {
        'pop_index': np.arange(cp.n_galaxies, dtype=np.int64),
    }
    for c in id_cols:
        columns[c] = sample[c].to_numpy()
    columns['z'] = z_sample
    if adapter.has_observed_redshift:
        columns['z_obs_catalog'] = sample['z_obs'].to_numpy(dtype=np.float64)
    columns.update(
        {
            'logm': logm,
            'log_sfr': sample['log_sfr'].to_numpy(dtype=np.float64),
            'f_line_cgs': sample['f_line_cgs'].to_numpy(),
            'ew_rest_a': sample['ew_rest_a'].to_numpy(),
            'f_lambda_cont_cgs': sample['f_lambda_cont_cgs'].to_numpy(),
            'rscale_arcsec': sample['rscale_arcsec'].to_numpy(),
        }
    )
    if adapter.has_bulge:
        columns['bulge_fraction'] = sample['bulge_fraction'].to_numpy(dtype=np.float64)
    columns.update(
        {
            'vel_rscale_ratio': vel_ratio,
            'halpha_rscale_ratio': line_ratio,
            'v0_kms': v0_kms,
        }
    )
    # painted bulge morphology columns exist only when the paint is enabled;
    # a disk-only twin (paint.bulge: false) omits them so any downstream
    # bulge read fails loudly instead of using stale values
    if cp.paint_bulge:
        bulge_nsersic, bulge_r50 = _paint_bulge(
            spec.seed,
            ids,
            sample['disk_r50'].to_numpy(dtype=np.float64),
        )
        columns['bulge_r50_arcsec'] = bulge_r50
        columns['bulge_nsersic'] = bulge_nsersic
    if adapter.has_bulge:
        columns['catalog_bulge_r50_arcsec'] = sample['bulge_r50'].to_numpy(
            dtype=np.float64
        )
        columns['catalog_bulge_nsersic'] = sample['bulge_nsersic'].to_numpy(
            dtype=np.float64
        )
    columns.update(
        {
            'cosi': sample['cosi'].to_numpy(),
            'theta_int': sample['theta_int'].to_numpy(),
            'g1': g1,
            'g2': g2,
            'vcirc_kms': vcirc,
            'sigma0_kms': sigma0,
            'snr_line_per_pass': sample['snr_line_per_pass'].to_numpy(),
            'snr_line_total': sample['snr_line_total'].to_numpy(),
            'compactness': sample['compactness'].to_numpy(),
            'logm_obs': logm_obs,
            'prior_vcirc_mu_kms': prior_mu,
            'prior_vcirc_sigma_dex': prior_sigma_dex,
            'f_line_obs_cgs': f_line_obs,
            'f_line_sigma_cgs': sigma_line,
        }
    )
    for b in ROMAN_IMAGING_BANDS:
        lb = b.lower()
        columns[f'flux_{lb}_ujy'] = band_fluxes[b]
        columns[f'flux_obs_{lb}_ujy'] = band_obs[b]
        columns[f'flux_sigma_{lb}_ujy'] = band_sigmas[b]
        columns[f'snr_bb_{lb}'] = band_snrs[b]
    # catalog values carried through purely for validation/diagnostics
    for out_name, src in adapter.validation_columns.items():
        columns[out_name] = sample[src].to_numpy(dtype=np.float64)
    population = pd.DataFrame(columns)

    provenance = catalog_provenance(cp.catalog_download, data_dir)
    meta = {
        'run_name': spec.run_name,
        'seed': spec.seed,
        'catalog_kind': adapter.kind,
        'catalog_id_columns': id_cols,
        'catalog_name': cp.catalog_download,
        'catalog_sha256': provenance['sha256'],
        'catalog_query_id': provenance.get('query_id'),
        'n_raw': n_raw,
        'n_disk': n_disk,
        'kills': kills,
        'n_selected': n_selected,
        'n_sampled': int(cp.n_galaxies),
        'flux_variant': cp.flux_variant,
        'f_lim_coadd_cgs': F_LIM_COADD_CGS,
        'f_lim_per_pass_cgs': F_LIM_PER_PASS_CGS,
        'n_grism_passes': N_GRISM_PASSES,
        'snr_line_reference': roman.SNR_LINE_REFERENCE,
        'fiducial_compactness': fiducial_compactness(),
        'imaging_depth_ab': dict(IMAGING_DEPTH_AB),
        'imaging_depth_provenance': (
            f'ROTAC Final Report Table 1 (arXiv:2505.10574): HLWAS medium '
            f'{IMAGING_DEPTH_NSIGMA:g}-sigma point-source coadded depths; '
            f'band SNR = nsigma * f/f_lim * C with C_ref = 1 (point source)'
        ),
        'f_lim_provenance': (
            f'ROTAC Final Report 2025-04-24 v3 Sect. 3.1: HLWAS medium-tier '
            f'{F_LIM_COADD_CGS:.1e} erg/s/cm2, {F_LIM_NSIGMA:g}-sigma line '
            f'flux limit coadded over {N_GRISM_PASSES} grism passes '
            f'(texp ~ 1500 s; Sect. 4.4.3). Referenced to a '
            f'{roman.SNR_LINE_REFERENCE} source'
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
            'bulge': _POP_BULGE,
        },
        'bulge_paint': (
            {
                'enabled': True,
                'pseudo_weight': BULGE_PSEUDO_WEIGHT,
                'pseudo_n': BULGE_PSEUDO_N,
                'classical_n': BULGE_CLASSICAL_N,
                'size_ratio_median': BULGE_SIZE_RATIO_MEDIAN,
                'size_ratio_ln_scatter': BULGE_SIZE_RATIO_LN_SCATTER,
                'size_ratio_max': BULGE_SIZE_RATIO_MAX,
                'note': (
                    'bulge_nsersic + bulge_r50_arcsec painted (catalog values '
                    'in catalog_* columns); bulge_fraction kept from catalog'
                ),
            }
            if cp.paint_bulge
            else {
                'enabled': False,
                'note': (
                    'paint.bulge: false -- disk-only twin; no painted bulge '
                    'columns (catalog bulge_fraction + catalog_* columns '
                    'retained for diagnostics)'
                ),
            }
        ),
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
