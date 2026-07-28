"""
Euclid Flagship2 catalog adapter.

Structural and flux truths (disk sizes, Halpha flux, continuum, bulge
fraction, redshift) come from Flagship2 rows downloaded from CosmoHub;
``preprocess`` derives the physical contract columns (little-h mass
correction, flux-variant selection, NISP-band continuum and rest-frame
equivalent width at the observed Halpha wavelength).

Flagship2 ``galaxy_id`` is a within-halo index (NOT globally unique); the
``(halo_id, galaxy_id)`` pair is the unique catalog key, so both ids enter
every per-galaxy seed key, in that order (determinism contract).
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
# rscale / continuum-amplitude population distributions: fitted to the
# selected Flagship2 sample (flagship2_dev at snr_line_total_min 20, n = 400,
# measured 2026-07-25), log10 median and log10 scatter. The continuum support
# is wide enough to contain the selected sample's continuum amplitudes,
# which span roughly 1.3-93 in the scene's internal flux units per nm.
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
    cont_flux_log10_mu=0.770,
    cont_flux_log10_sigma=0.305,
    cont_flux_low=0.05,
    cont_flux_high=400.0,
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
        """Derive physical contract columns from raw Flagship2 rows.

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

        out = out[list(self.contract_columns())]
        out.attrs['kills'] = {'one_component': n_dropped}
        return out
