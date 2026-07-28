"""
COSMOS25 catalog adapter: real COSMOS-Web sources + painted Halpha.

Structural truths (single-Sersic sizes, axis ratios) and photometry are
real SE++ measurements from the COSMOS2025 master catalog (Shuntov et al.
2025, arXiv:2506.03243); redshifts are LePhare photo-zs treated as truth;
Halpha fluxes are Jiachuan Xu's painted emission-line section (private,
row-matched, delivered 2026-07-28). The joined parquet is produced by
scripts/build_cosmos25_catalog.py, which gates the row-order join.

Flux variants (``preprocess.flux_variant``):

- ``as_delivered``: the painted fluxes verbatim. The painting applies its
  nebular dust term with the wrong sign (multiplies the SFR-derived
  intrinsic luminosity by 10^(+0.4 A) instead of 10^(-0.4 A); confirmed
  against the delivered file, where every flux carries exactly
  10^(+0.4 * 2.53 * 1.3 * ebv_minchi2)), and converts SFR to L(Halpha)
  with the Kennicutt 1998 Salpeter constant 1.26e41 while the input CIGALE
  SFRs follow a Chabrier-type IMF. Kept for handshakes against the
  delivering notebook's numbers only.
- ``dust_fixed``: dust sign corrected, i.e. delivered * 10^(-0.8 A) with
  A = 2.53 * 1.3 * ebv_minchi2 (Cardelli k at Halpha, nebular-to-stellar
  scale 1.3, both from the painting recipe).
- ``dust_imf_fixed``: additionally rescales to the Chabrier-consistent
  Kennicutt & Evans 2012 constant (log C = 41.27; Table 1), a factor
  10^41.27 * 7.9e-42 = 1.4711. The production default; the delivered
  variant's median rest EW(Halpha) of ~300 A (84th pct ~1200 A) against
  the catalog's own photometry is unphysical, while the corrected
  variants sit in the literature 100-300 A range.

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

# painting dust constants (IAEstimate.ipynb): Cardelli+89 k(Halpha) and the
# nebular-to-stellar E(B-V) scale the recipe applied; the dust_* variants
# must invert exactly what the painting did, so these are pinned to the
# recipe, not to a preferred extinction law
PAINT_K_HALPHA = 2.53
PAINT_DUST_SCALE = 1.3

# Kennicutt 1998 (Salpeter, Eq. 2) -> Kennicutt & Evans 2012 (Table 1,
# log C = 41.27; Kroupa, "nearly identical" to Chabrier) constant ratio
IMF_CONSTANT_RATIO = 10**41.27 * 7.9e-42  # = 1.4711

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
    'log_U',
)

# Catalog-fitted scene-prior constants. Provenance:
#
# rscale / continuum-amplitude population distributions: fitted to the
# selected cosmos25 sample (census cuts: z 0.55-1.9, isotropic-cosi redraw,
# r50/PSF_FWHM >= 1, line SNR_total >= 20 on the dust_imf_fixed variant;
# n = 469, measured 2026-07-28), log10 median and log10 scatter. The
# selected sizes span rscale 0.077-0.90 arcsec.
#
# rscale support [arcsec]: floor and ceiling sized against the painted size
# products (catalog rscale x lognormal ratio for the Halpha line and the
# velocity turnover), matching the Flagship2 bound-setting rule. Under the
# fitted size population, P(line product > 3.0) = 9e-6 and P(turnover
# product > 3.0) = 5e-6 per galaxy; P(turnover product < 0.005) = 1.2e-4
# (measured 2026-07-28, 2e6 Monte Carlo draws). A rare draw outside support
# fails the truth-in-support check loudly, which is the intended behavior.
#
# continuum support [internal flux units per nm]: the selected sample spans
# 0.14-22.0 (0.1/99.9 percentiles 0.14/15.5); bounds set well clear on both
# sides, mirroring the Flagship2 margin style.
COSMOS25_PRIOR_CONSTANTS = CatalogPriorConstants(
    rscale_log10_mu=-0.752,
    rscale_log10_sigma=0.166,
    rscale_low=0.005,
    rscale_high=3.0,
    cont_flux_log10_mu=-0.196,
    cont_flux_log10_sigma=0.319,
    cont_flux_low=0.01,
    cont_flux_high=100.0,
)


class Cosmos25Adapter(CatalogAdapter):
    """COSMOS2025 + painted emission lines (private join) adapter."""

    kind = 'cosmos25'
    columns = COSMOS25_COLUMNS
    id_columns = ('id',)
    flux_variants = ('dust_imf_fixed', 'dust_fixed', 'as_delivered')
    download_hint = (
        "run 'make download-cosmos2025', place the private painted section "
        "under data/cosmos2025/private/, then run "
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
        - f_line = painted F_Ha [erg/s/cm2], variant-corrected
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

        # flux variant (module docstring); the painting's dust boost is
        # exactly 10^(+0.4 k s ebv), so the fix is the delivered flux times
        # 10^(-0.8 k s ebv)
        f_line = out['F_Ha'].to_numpy(dtype=np.float64)
        if spec.flux_variant in ('dust_fixed', 'dust_imf_fixed'):
            ebv = out['ebv_minchi2'].to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(ebv) & (ebv >= 0.0)):
                raise ValueError(
                    "preprocess: non-finite or negative ebv_minchi2 on rows "
                    "with painted flux; the dust correction would be wrong"
                )
            f_line = f_line * 10.0 ** (-0.8 * PAINT_K_HALPHA * PAINT_DUST_SCALE * ebv)
            if spec.flux_variant == 'dust_imf_fixed':
                f_line = f_line * IMF_CONSTANT_RATIO
        elif spec.flux_variant != 'as_delivered':
            raise ValueError(
                f"unknown flux_variant '{spec.flux_variant}' (spec validation "
                f"should have caught this)"
            )

        # continuum f_nu at lambda_obs from SE++ model photometry: band
        # average inside F115W/F150W coverage, power-law pivot interpolation
        # across the F150W-F277W gap (module constants)
        f115 = out['flux_model_f115w'].to_numpy(dtype=np.float64)
        f150 = out['flux_model_f150w'].to_numpy(dtype=np.float64)
        f277 = out['flux_model_f277w'].to_numpy(dtype=np.float64)
        in_gap = lambda_obs_a >= F150W_RED_EDGE_A
        gap_ok = ~in_gap | ((f150 > 0.0) & (f277 > 0.0))
        with np.errstate(divide='ignore', invalid='ignore'):
            alpha = np.log(f277 / f150) / np.log(F277W_PIVOT_A / F150W_PIVOT_A)
            f_nu_gap = f150 * (lambda_obs_a / F150W_PIVOT_A) ** alpha
        f_nu_ujy = np.where(
            lambda_obs_a < F115W_F150W_EDGE_A,
            f115,
            np.where(in_gap, f_nu_gap, f150),
        )
        cont_ok = gap_ok & np.isfinite(f_nu_ujy) & (f_nu_ujy > 0.0)
        out = out.assign(_z=z, _lambda=lambda_obs_a, _f_line=f_line, _f_nu=f_nu_ujy)
        out = cut(cont_ok, 'continuum_nonpositive', out)

        z = out.pop('_z').to_numpy()
        lambda_obs_a = out.pop('_lambda').to_numpy()
        f_line = out.pop('_f_line').to_numpy()
        f_nu_cgs = out.pop('_f_nu').to_numpy() * UJY_TO_CGS

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

        derived = {
            'f_line_cgs': f_line,
            'f_lambda_cont_cgs': f_lambda_obs,
            'ew_rest_a': ew_rest_a,
            'rscale_arcsec': out['rscale_arcsec'].to_numpy(),
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
