"""
COSMOS25 catalog adapter: real COSMOS-Web sources + painted Halpha.

Structural truths (single-Sersic sizes, axis ratios) and photometry are
real SE++ measurements from the COSMOS2025 master catalog (Shuntov et al.
2025, arXiv:2506.03243); redshifts are LePhare photo-zs treated as truth;
Halpha fluxes are the painted emission-line section regenerated from
Jiachuan Xu's painting notebook (private, row-matched;
scripts/regen_cosmos25_painting.py). The joined parquet is produced by
scripts/build_cosmos25_catalog.py, which gates the row-order join.

The only flux variant is ``as_delivered``: the painted fluxes verbatim.
History: the original 2026-07-28 delivered file applied its nebular dust
term with the wrong sign (brightening instead of attenuating) and used
the Salpeter-calibrated Kennicutt 1998 Halpha constant with Chabrier-IMF
CIGALE SFRs; this adapter carried ``dust_fixed``/``dust_imf_fixed``
correction variants that inverted both. The revised notebook (received
2026-07-29) fixes both upstream (attenuation sign, Kennicutt & Evans 2012
constant, Curti FMR metallicity-based line ratios, Song et al. 2026
mass-dependent nebular E(B-V) scaling), so the correction variants were
retired when the pipeline moved to regenerating the painting from the
notebook. Median rest EW(Halpha) against the catalog's own photometry
(this adapter's EW convention) is ~305 A for the flux-limited medium tier
(F_Ha > 1.5e-16, 0.55 < z < 1.9) and ~650 A for the SNR-selected census
sample, whose matched-filter cut rides the high-EW tail (measured
2026-07-30 on the regenerated painting). Both run high against measured
COSMOS EWs: 3D-HST at a matched flux cut gives median ~90 A with a 95th
percentile of ~340 A, and an in-field cross-match shows the painted
line-bright selection is dominated by up-scattered fluxes. Treat the
painted EW tail as a known model optimism, not a literature-consistent
population.

The mass column is LePhare ``mass_med`` (log10 Msun, already physical for
the catalog's fiducial cosmology), so the spec's little-h key is not
applied here, unlike Flagship2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from kl_pipe.ensemble.catalogs.base import CatalogAdapter, CatalogPriorConstants

if TYPE_CHECKING:
    from kl_pipe.ensemble.spec import CatalogPopulationSpec

# COSMOS2025 catalog version the painted section was built on. Adjudicated
# bitwise 2026-07-28: the painted 'redshift' column equals v1 'zfinal' on
# all 784,016 rows, while v1.1 revised zfinal for 86% of rows -- the painted
# lines are only coherent with the v1 sections.
COSMOS25_SOURCE_VERSION = 'v1'

# exponential-disk half-light-to-scale-length ratio, r50 = 1.678 * rscale;
# the SE++ single-Sersic effective radius is read as the disk r50 under the
# ensemble's disk-only convention
R50_OVER_RSCALE = 1.678

# JWST NIRCam pivot wavelengths [Angstrom] and the band-assignment edges
# for the continuum at the observed Halpha wavelength: F115W covers up to
# ~1.28 um, F150W up to its red half-power edge at ~1.67 um; beyond that
# (Halpha at z > 1.54) NIRCam has no band until F277W, so the continuum is
# a power-law interpolation between the F150W and F277W pivots
F115W_PIVOT_A = 1.154e4
F150W_PIVOT_A = 1.501e4
F277W_PIVOT_A = 2.776e4
F115W_F150W_EDGE_A = 1.30e4
F150W_RED_EDGE_A = 1.668e4

# SE++ model fluxes are in microJansky; f_nu[erg/cm2/s/Hz] = 1e-29 * uJy
UJY_TO_CGS = 1e-29


def _powerlaw_fnu(
    f_blue: np.ndarray,
    lam_blue: float,
    f_red: np.ndarray,
    lam_red: float,
    lam: np.ndarray,
) -> np.ndarray:
    """f_nu at lam from a power law through two pivot measurements.

    Log-log interpolation, the same local power-law continuity assumption as
    the continuum-at-lambda_obs gap interpolation. Inputs must be positive
    (the caller cuts non-positive photometry first).
    """
    alpha = np.log(f_red / f_blue) / np.log(lam_red / lam_blue)
    return f_blue * (np.asarray(lam, dtype=np.float64) / lam_blue) ** alpha


# joined-parquet schema (scripts/build_cosmos25_catalog.py); loads are
# validated against this exact column set
COSMOS25_COLUMNS = (
    'id',
    'ra',
    'dec',
    'warn_flag',
    'mag_model_f150w',
    'snr_f150w',
    'radius_sersic',
    'radius_sersic_err',
    'sersic',
    'axratio_sersic',
    'e1_err',
    'e2_err',
    'flux_model_f115w',
    'flux_model_f150w',
    'flux_model_f277w',
    'zfinal',
    'type',
    'ebv_minchi2',
    'mass_med',
    'sfr_med',
    'F_Ha',
    'F_OII',
    'F_OIII',
    'lambda_Ha_obs',
    'lambda_OII_obs',
    'lambda_OIII_obs',
    'redshift',
    'sfr_young',
    'log_OH',
)

# Catalog-fitted scene-prior constants. Provenance:
#
# rscale / continuum-amplitude population distributions: fitted to the
# selected cosmos25 sample (census cuts: z 0.55-1.9, isotropic-cosi redraw,
# r50/PSF_FWHM >= 1, line SNR_total >= 20 on the as_delivered fluxes of the
# catalog regenerated from the revised 2026-07-29 painting notebook;
# n = 294, measured 2026-07-29), log10 median and log10 scatter. The
# selected sizes span rscale 0.077-0.56 arcsec. (Previous values, measured
# 2026-07-28 on the dust_imf_fixed correction of the original delivery,
# n = 469: mu -0.752/-0.196, sigma 0.166/0.319.) Re-verified 2026-07-30
# against the full selection of the cosmos25_ab spec (n = 301): measured
# rscale -0.772/0.154 and continuum -0.177/0.277, within a few percent of
# the pinned values.
#
# rscale support [arcsec]: floor and ceiling sized against the painted size
# products (catalog rscale x lognormal ratio for the Halpha line and the
# velocity turnover), matching the Flagship2 bound-setting rule. Under the
# fitted size population, P(line product > 3.0) = 4e-6 and P(turnover
# product > 3.0) = 4e-6 per galaxy; P(turnover product < 0.005) = 1.4e-4
# (measured 2026-07-29, 2e6 Monte Carlo draws). A rare draw outside support
# fails the truth-in-support check loudly, which is the intended behavior.
#
# continuum amplitude [1e-17 erg/s/cm2 per nm]: log10 median and log10
# scatter of f_line / EW_obs on the full cosmos25_ab selection (n = 301,
# seed 20260728, measured 2026-07-31; the selection membership is unchanged
# by the photometry-positivity kill, verified id-for-id against the previous
# chain). Support: the selected sample spans 0.087-23; bounds set well clear
# on both sides, mirroring the Flagship2 margin style. (The pre-physical-
# units constants, fitted at the scene-constant line flux of 100, were
# mu -0.183 / sigma 0.299 on the same convention -- the shift is the
# per-galaxy line flux, not a selection change.)
#
# line_flux support [1e-17 erg/s/cm2]: selection spans 34.8-407 (same n=301
# sample); band_flux support [uJy]: 0.39-131 across F106/F129/F158. Both
# bounds only guard TruncatedNormal support for measurement priors whose
# sigma is ~2-4% of the center, so the margins are deliberately lavish.
COSMOS25_PRIOR_CONSTANTS = CatalogPriorConstants(
    rscale_log10_mu=-0.774,
    rscale_log10_sigma=0.161,
    rscale_low=0.005,
    rscale_high=3.0,
    cont_flux_log10_mu=-0.344,
    cont_flux_log10_sigma=0.338,
    cont_flux_low=0.005,
    cont_flux_high=500.0,
    line_flux_low=5.0,
    line_flux_high=5000.0,
    band_flux_low=0.05,
    band_flux_high=2000.0,
)


class Cosmos25Adapter(CatalogAdapter):
    """COSMOS2025 + painted emission lines (private join) adapter."""

    kind = 'cosmos25'
    columns = COSMOS25_COLUMNS
    id_columns = ('id',)
    flux_variants = ('as_delivered',)
    download_hint = (
        "run 'make download-cosmos2025', regenerate the private painted "
        "section with 'python scripts/regen_cosmos25_painting.py', then run "
        "'python scripts/build_cosmos25_catalog.py'"
    )
    has_bulge = False
    has_observed_redshift = False
    validation_columns = {
        'catalog_axis_ratio': 'axratio_sersic',
        'catalog_sersic_n': 'sersic',
    }
    prior_constants = COSMOS25_PRIOR_CONSTANTS
    citation_bibkeys = ('Shuntov2025',)

    def preprocess(
        self, df: pd.DataFrame, spec: 'CatalogPopulationSpec'
    ) -> pd.DataFrame:
        """Map raw joined rows to the contract columns.

        Kill chain (each stage counted in ``attrs['kills']``):
        LePhare non-galaxies (``type != 0``), photometry warn flags,
        rows without a positive painted Halpha flux (stars, artifacts,
        z <= 0, unfit SEDs), non-finite structure (SE++ radius/axis
        ratio), non-finite LePhare mass, and rows whose derived continuum
        is not positive.

        Unit chain (dimensional sanity):
        - f_line = painted F_Ha [erg/s/cm2], verbatim
        - lambda_obs = painted lambda_Ha_obs [A] (== 6562.8 * (1 + z))
        - f_nu (SE++ model photometry at lambda_obs) [uJy] * 1e-29
          -> [erg/cm2/s/Hz]
        - f_lambda = f_nu * c / lambda_obs^2
          -> [erg/cm2/s/Hz] * [A/s] / [A^2] = [erg/cm2/s/A]
        - EW_rest = f_line / f_lambda / (1 + z) [A]
        - r50 = radius_sersic [deg] * 3600 [arcsec]; rscale = r50 / 1.678

        The broadband model flux contains the line itself, so the derived
        continuum is biased high by roughly EW_obs / bandwidth (~10% for
        typical selected EWs); accepted at population-prior level.
        """
        # local import: population owns the generic line-physics constants
        # (same layering as the flagship2 adapter)
        from kl_pipe.ensemble.population import C_A_PER_S, HALPHA_REST_A

        kills = {}

        def cut(mask: np.ndarray, stage: str, frame: pd.DataFrame) -> pd.DataFrame:
            kills[stage] = int((~mask).sum())
            return frame.loc[mask].reset_index(drop=True)

        out = df
        out = cut(out['type'].to_numpy() == 0, 'not_galaxy', out)
        out = cut(out['warn_flag'].to_numpy() == 0, 'warn_flag', out)

        f_ha = out['F_Ha'].to_numpy(dtype=np.float64)
        out = cut(np.isfinite(f_ha) & (f_ha > 0.0), 'no_painted_line', out)

        r50_deg = out['radius_sersic'].to_numpy(dtype=np.float64)
        axratio = out['axratio_sersic'].to_numpy(dtype=np.float64)
        structure_ok = (
            np.isfinite(r50_deg)
            & (r50_deg > 0.0)
            & np.isfinite(axratio)
            & (axratio > 0.0)
        )
        out = cut(structure_ok, 'bad_structure', out)

        logm = out['mass_med'].to_numpy(dtype=np.float64)
        out = cut(np.isfinite(logm) & (logm > 0.0), 'bad_mass', out)

        if len(out) == 0:
            raise ValueError("preprocess: no rows survive the quality cuts")

        z = out['zfinal'].to_numpy(dtype=np.float64)
        lambda_obs_a = out['lambda_Ha_obs'].to_numpy(dtype=np.float64)
        if not np.allclose(lambda_obs_a, HALPHA_REST_A * (1.0 + z), rtol=1e-6):
            raise ValueError(
                "preprocess: painted lambda_Ha_obs disagrees with "
                "6562.8 * (1 + zfinal); the join or the painting is broken"
            )

        # single variant: the regenerated painting is used verbatim (module
        # docstring records the retired correction variants)
        f_line = out['F_Ha'].to_numpy(dtype=np.float64)
        if spec.flux_variant != 'as_delivered':
            raise ValueError(
                f"unknown flux_variant '{spec.flux_variant}' (spec validation "
                f"should have caught this); the dust_fixed/dust_imf_fixed "
                f"corrections were retired when the painting was fixed "
                f"upstream -- use 'as_delivered'"
            )

        # the Roman band fluxes and the gap continuum both take logs of the
        # SE++ model photometry, so every row must carry positive, finite
        # fluxes in all three bands
        f115 = out['flux_model_f115w'].to_numpy(dtype=np.float64)
        f150 = out['flux_model_f150w'].to_numpy(dtype=np.float64)
        f277 = out['flux_model_f277w'].to_numpy(dtype=np.float64)
        phot_ok = (
            np.isfinite(f115)
            & (f115 > 0.0)
            & np.isfinite(f150)
            & (f150 > 0.0)
            & np.isfinite(f277)
            & (f277 > 0.0)
        )
        out = out.assign(_z=z, _lambda=lambda_obs_a, _f_line=f_line)
        out = cut(phot_ok, 'photometry_nonpositive', out)
        z = out['_z'].to_numpy()
        lambda_obs_a = out['_lambda'].to_numpy()
        f115 = out['flux_model_f115w'].to_numpy(dtype=np.float64)
        f150 = out['flux_model_f150w'].to_numpy(dtype=np.float64)
        f277 = out['flux_model_f277w'].to_numpy(dtype=np.float64)

        # continuum f_nu at lambda_obs from SE++ model photometry: band
        # average inside F115W/F150W coverage, power-law pivot interpolation
        # across the F150W-F277W gap (module constants)
        in_gap = lambda_obs_a >= F150W_RED_EDGE_A
        f_nu_gap = _powerlaw_fnu(f150, F150W_PIVOT_A, f277, F277W_PIVOT_A, lambda_obs_a)
        f_nu_ujy = np.where(
            lambda_obs_a < F115W_F150W_EDGE_A,
            f115,
            np.where(in_gap, f_nu_gap, f150),
        )
        cont_ok = np.isfinite(f_nu_ujy) & (f_nu_ujy > 0.0)
        out = out.assign(_f_nu=f_nu_ujy)
        out = cut(cont_ok, 'continuum_nonpositive', out)
        f115 = out['flux_model_f115w'].to_numpy(dtype=np.float64)
        f150 = out['flux_model_f150w'].to_numpy(dtype=np.float64)
        f277 = out['flux_model_f277w'].to_numpy(dtype=np.float64)

        z = out.pop('_z').to_numpy()
        lambda_obs_a = out.pop('_lambda').to_numpy()
        f_line = out.pop('_f_line').to_numpy()
        f_nu_cgs = out.pop('_f_nu').to_numpy() * UJY_TO_CGS

        # Roman imaging fluxes [uJy], line-inclusive by construction (SE++
        # model photometry contains the emission line): power-law
        # interpolation of the JWST pivots to the Roman effective
        # wavelengths. F106 and F129 use the F115W-F150W pair (F106 sits
        # slightly blueward of the F115W pivot, a mild extrapolation of the
        # local slope); F158 uses the F150W-F277W pair across the NIRCam
        # gap, the same convention as the gap continuum above.
        from kl_pipe.ensemble.population import BAND_EFFECTIVE_LAMBDA_A

        band_flux_ujy = {}
        for band, (fb, lb, fr, lr) in {
            'F106': (f115, F115W_PIVOT_A, f150, F150W_PIVOT_A),
            'F129': (f115, F115W_PIVOT_A, f150, F150W_PIVOT_A),
            'F158': (f150, F150W_PIVOT_A, f277, F277W_PIVOT_A),
        }.items():
            lam = np.full_like(f115, BAND_EFFECTIVE_LAMBDA_A[band])
            band_flux_ujy[band] = _powerlaw_fnu(fb, lb, fr, lr, lam)

        f_lambda_obs = f_nu_cgs * C_A_PER_S / lambda_obs_a**2  # erg/cm2/s/A
        ew_rest_a = f_line / f_lambda_obs / (1.0 + z)  # A

        disk_r50 = out['radius_sersic'].to_numpy(dtype=np.float64) * 3600.0
        out['z'] = z
        out['logm'] = out['mass_med'].to_numpy(dtype=np.float64)
        out['log_sfr'] = out['sfr_med'].to_numpy(dtype=np.float64)
        out['f_line_cgs'] = f_line
        out['lambda_obs_a'] = lambda_obs_a
        out['f_lambda_cont_cgs'] = f_lambda_obs
        out['ew_rest_a'] = ew_rest_a
        out['disk_r50'] = disk_r50
        out['rscale_arcsec'] = disk_r50 / R50_OVER_RSCALE
        for band, values in band_flux_ujy.items():
            out[f'flux_{band.lower()}_ujy'] = values

        derived = {
            'f_line_cgs': f_line,
            'f_lambda_cont_cgs': f_lambda_obs,
            'ew_rest_a': ew_rest_a,
            'rscale_arcsec': out['rscale_arcsec'].to_numpy(),
            **{f'flux_{b.lower()}_ujy': v for b, v in band_flux_ujy.items()},
        }
        bad_counts = {}
        for col, values in derived.items():
            n_bad = int((~(np.isfinite(values) & (values > 0.0))).sum())
            if n_bad:
                bad_counts[col] = n_bad
        if bad_counts:
            raise ValueError(
                f"preprocess: non-finite or non-positive derived values: "
                f"{bad_counts} (of {len(out)} rows)"
            )

        out = out[list(self.contract_columns())]
        out.attrs['kills'] = kills
        return out
