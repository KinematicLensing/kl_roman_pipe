"""
Euclid Flagship2 catalog adapter.

Structural and flux truths (disk sizes, Halpha flux, continuum, bulge
fraction, redshift) come from Flagship2 rows downloaded from CosmoHub;
``preprocess`` derives the standardized physical columns (little-h mass
correction, flux-variant selection, NISP-band continuum and rest-frame
equivalent width at the observed Halpha wavelength).

Flagship2 ``galaxy_id`` is a within-halo index (NOT globally unique); the
``(halo_id, galaxy_id)`` pair is the unique catalog key, so both ids enter
every per-galaxy seed key, in that order (the order must never change).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from kl_pipe.ensemble.catalogs.base import CatalogAdapter, CatalogPriorConstants

if TYPE_CHECKING:
    from kl_pipe.ensemble.spec import CatalogPopulationSpec

# Euclid NISP band edges [Angstrom] for continuum selection at the observed
# Halpha wavelength: Y covers lambda < 11900, J covers [11900, 15450),
# H covers >= 15450 (nominal NISP passband boundaries)
NISP_YJ_EDGE_A = 11900.0
NISP_JH_EDGE_A = 15450.0

# NISP photometric pivot wavelengths [Angstrom] (Schirmer et al. 2022, Euclid
# NISP photometric system: 1081 / 1367 / 1771 nm), the nodes of the log-log
# power-law interpolation to the Roman band effective wavelengths
NISP_Y_PIVOT_A = 1.0809e4
NISP_J_PIVOT_A = 1.3673e4
NISP_H_PIVOT_A = 1.7714e4

# NISP fluxes are f_nu in erg/cm2/s/Hz; uJy = f_nu / 1e-29
CGS_FNU_TO_UJY = 1e29

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

# Catalog-fitted scene-prior constants. Provenance:
#
# rscale support [arcsec]: the sampled-mode (0.05, 1.0) bounds do NOT contain
# the catalog truths, so bounds are set below/above the reachable extremes
# rather than silently clipping truth out of support. The ceiling must clear
# the painted products, not just the catalog sizes: the line and turnover
# truths are catalog size x lognormal ratio, and at the old ceiling of 2.0 a
# real dev draw landed at Halpha.rscale = 2.12 (1 of 400, measured
# 2026-07-27), leaving that truth with zero prior support. At 3.0 the
# exceedance under the fitted size population drops from 2.8e-3 to 4.2e-4
# per galaxy for the line size (7e-5 for the turnover), so a rare census
# draw can still land outside and will fail the truth-in-support check
# loudly, which is the intended behavior. The render grid is unchanged
# (oversample 3, pad 2, per-eval cost flat; PSF damping sets the k-space
# extent, measured 2026-07-27).
#
# rscale population distribution: fitted to the selected Flagship2 sample
# (flagship2_dev at snr_line_total_min 20, n = 400, measured 2026-07-25),
# log10 median and log10 scatter.
#
# continuum amplitude [1e-17 erg/s/cm2 per nm]: log10 median and scatter of
# f_line / EW_obs on the flagship2_shear_dev selection (n = 474, seed
# 20260721, z 0.55-1.9, B/T <= 0.3, r50 >= PSF FWHM, SNR_total >= 20;
# measured 2026-07-31), span 0.84-54.8 with bounds set well clear. line_flux
# support [1e-17 erg/s/cm2]: same selection spans 34.6-1010. band_flux
# support [uJy]: 3.8-364 across F106/F129/F158; the F129 percentiles
# (13.5/26.8/64.2, p16/50/84) reproduce the independent parquet-inversion
# magnitudes measured 2026-07-25 (13.8/27.5/64.1), validating the NISP
# pivot interpolation.
#
# bulge_frac: matched to the selected population. Flagship2 dev, z 0.55-1.9,
# two-component disks, B/T <= 0.3 (the bulge_fraction_max selection cut):
# B/T mean 0.080, std 0.080, n=206k (measured 2026-07-23; stable against a
# brightest-decile flux cut).
#
# bulge_hlr support: floor matched to the disk floor rather than set lower
# (the smallest allowed size drives the worst-case maxk, so an unnecessarily
# small floor enlarges the render grid on every evaluation); the population's
# bulge sizes reach about 0.006 arcsec at four sigma.
FLAGSHIP2_PRIOR_CONSTANTS = CatalogPriorConstants(
    rscale_log10_mu=-0.672,
    rscale_log10_sigma=0.237,
    rscale_low=0.005,
    rscale_high=3.0,
    cont_flux_log10_mu=0.646,
    cont_flux_log10_sigma=0.379,
    cont_flux_low=0.05,
    cont_flux_high=1000.0,
    line_flux_low=5.0,
    line_flux_high=10000.0,
    band_flux_low=0.5,
    band_flux_high=5000.0,
    bulge_frac_loc=0.08,
    bulge_frac_scale=0.08,
    bulge_hlr_low=0.005,
    bulge_hlr_high=2.0,
)


class Flagship2Adapter(CatalogAdapter):
    """Euclid Flagship2 (CosmoHub) adapter."""

    kind = 'flagship2'
    columns = FLAGSHIP2_COLUMNS
    id_columns = ('halo_id', 'galaxy_id')
    flux_variants = ('model3_ext', 'model3', 'model1_ext', 'model1')
    download_hint = (
        "run 'make download-cosmohub-dev' (or download-cosmohub-data for "
        "the production bank)"
    )
    has_bulge = True
    has_observed_redshift = True
    validation_columns = {
        'catalog_inclination_deg': 'inclination_angle',
        'catalog_disk_axis_ratio': 'disk_axis_ratio',
        'kappa': 'kappa',
        'gamma1_field': 'gamma1',
        'gamma2_field': 'gamma2',
    }
    prior_constants = FLAGSHIP2_PRIOR_CONSTANTS
    citation_bibkeys = ('Castander2025',)

    def preprocess(
        self, df: pd.DataFrame, spec: 'CatalogPopulationSpec'
    ) -> pd.DataFrame:
        """Derive the standardized physical columns from raw Flagship2 rows.

        Drops one-component rows (``disk_r50 <= 0``: bulge-only galaxies
        with no disk), corrects the stellar mass for little-h, converts the
        Halpha flux variant to linear flux, and derives the observed-frame
        continuum and rest-frame equivalent width at the observed Halpha
        wavelength.

        Unit chain (dimensional sanity):
        - f_line = 10**logf [erg/s/cm2] (integrated line flux)
        - lambda_obs = 6562.8 * (1 + z) [A]
        - f_nu (NISP band containing lambda_obs) [erg/cm2/s/Hz]
        - f_lambda = f_nu * c / lambda_obs^2
          -> [erg/cm2/s/Hz] * [A/s] / [A^2] = [erg/cm2/s/A]
        - EW_obs = f_line / f_lambda -> [erg/s/cm2] / [erg/cm2/s/A] = [A]
        - EW_rest = EW_obs / (1 + z) [A]
        """
        # local import: population owns the generic line-physics constants
        # and imports the spec module, so a module-level import here would
        # widen the import graph for no gain
        from kl_pipe.ensemble.population import C_A_PER_S, HALPHA_REST_A

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
        logm = out['log_stellar_mass'].to_numpy(dtype=np.float64) - 2.0 * np.log10(
            spec.h
        )

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

        out['z'] = z
        out['z_obs'] = out['observed_redshift_gal'].to_numpy(dtype=np.float64)
        out['logm'] = logm
        out['f_line_cgs'] = f_line
        out['lambda_obs_a'] = lambda_obs_a
        out['f_lambda_cont_cgs'] = f_lambda_obs
        out['ew_rest_a'] = ew_rest_a
        out['rscale_arcsec'] = out['disk_scalelength'].to_numpy(dtype=np.float64)

        # Roman imaging fluxes [uJy], line-inclusive: log-log interpolation
        # of the NISP *_el_model3_ext photometry (emission lines included,
        # matching what a broadband image contains) from the NISP pivots to
        # the Roman effective wavelengths. F106 sits slightly blueward of the
        # Y pivot (mild extrapolation of the local Y-J slope); F129 uses Y-J,
        # F158 uses J-H. Only the model3_ext line-inclusive photometry exists
        # in the schema, so other flux variants have no consistent band
        # photometry and are rejected.
        from kl_pipe.ensemble.population import BAND_EFFECTIVE_LAMBDA_A

        if spec.flux_variant != 'model3_ext':
            raise ValueError(
                f"flux_variant '{spec.flux_variant}': the catalog carries "
                f"line-inclusive band photometry (euclid_nisp_*_el_model3_ext) "
                f"only for model3_ext, so the required Roman band-flux "
                f"columns cannot be built consistently for this variant"
            )
        f_y = out['euclid_nisp_y_el_model3_ext'].to_numpy(dtype=np.float64)
        f_j = out['euclid_nisp_j_el_model3_ext'].to_numpy(dtype=np.float64)
        f_h = out['euclid_nisp_h_el_model3_ext'].to_numpy(dtype=np.float64)
        from kl_pipe.ensemble.catalogs.cosmos25 import _powerlaw_fnu

        band_flux_ujy = {}
        for band, (fb, lb, fr, lr) in {
            'F106': (f_y, NISP_Y_PIVOT_A, f_j, NISP_J_PIVOT_A),
            'F129': (f_y, NISP_Y_PIVOT_A, f_j, NISP_J_PIVOT_A),
            'F158': (f_j, NISP_J_PIVOT_A, f_h, NISP_H_PIVOT_A),
        }.items():
            lam = np.full_like(f_y, BAND_EFFECTIVE_LAMBDA_A[band])
            band_flux_ujy[band] = _powerlaw_fnu(fb, lb, fr, lr, lam) * CGS_FNU_TO_UJY
            out[f'flux_{band.lower()}_ujy'] = band_flux_ujy[band]

        derived = {
            'logm': logm,
            'f_line_cgs': f_line,
            'f_lambda_cont_cgs': f_lambda_obs,
            'ew_rest_a': ew_rest_a,
            'rscale_arcsec': out['rscale_arcsec'].to_numpy(),
            **{f'flux_{b.lower()}_ujy': v for b, v in band_flux_ujy.items()},
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

        out = out[list(self.required_columns())]
        out.attrs['kills'] = {'one_component': n_dropped}
        return out
