"""
CI-safe unit tests for the catalog-backed population backend
(kl_pipe/ensemble/population.py + the catalog branch of the spec).

Runs entirely on a synthetic Flagship2-schema catalog written to tmp dirs;
the real-data tier lives in tests/test_population_cosmohub.py (marked
cosmohub). Statistical bounds state their derivation next to the number.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import stats

from kl_pipe.ensemble import population
from kl_pipe.ensemble.catalogs import get_catalog_adapter, load_catalog
from kl_pipe.ensemble.catalogs.flagship2 import FLAGSHIP2_COLUMNS
from kl_pipe.ensemble.population import (
    BULGE_CLASSICAL_N,
    BULGE_PSEUDO_N,
    BULGE_PSEUDO_WEIGHT,
    BULGE_SIZE_RATIO_LN_SCATTER,
    BULGE_SIZE_RATIO_MAX,
    BULGE_SIZE_RATIO_MEDIAN,
    R50_TO_SIGMA,
    build_population,
    compute_line_snr_per_pass,
    compute_line_snr_total,
    matched_filter_compactness,
    write_population,
)
from kl_pipe.ensemble.spec import CatalogPopulationSpec, EnsembleSpec

pytestmark = pytest.mark.roman_ensemble

# the flagship2 adapter under test (loader + preprocess live on it now)
FLAGSHIP2 = get_catalog_adapter('flagship2')

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_SPEC = REPO_ROOT / 'configs' / 'ensembles' / 'flagship2_shear_dev.yaml'
SAMPLED_SPEC = REPO_ROOT / 'configs' / 'ensembles' / 'sigma_eps_cosi_dev.yaml'


# ==============================================================================
# Fake Flagship2 catalog generator (full 36-column schema)
# ==============================================================================


def fake_flagship2_rows(
    n: int = 400,
    seed: int = 1234,
    z_low: float = 0.6,
    z_high: float = 1.8,
    id_offset: int = 0,
) -> pd.DataFrame:
    """Physically-sensible fake Flagship2 rows with the full 36-column schema.

    ~3% of rows get disk_r50 = 0 (one-component galaxies). galaxy_id is a
    within-halo index with duplicates across halos, mirroring the real
    catalog; (halo_id, galaxy_id) is unique. ``id_offset`` shifts halo_id so
    disjoint row blocks can be concatenated into a superset catalog.
    """
    rng = np.random.default_rng(seed)
    halo_id = 4_000_000_000_000 + id_offset + np.arange(n) // 3
    galaxy_id = np.arange(n) % 3

    z = rng.uniform(z_low, z_high, n)
    logf3ext = rng.uniform(-15.6, -14.6, n)  # bright enough for SNR tests
    disk_r50 = rng.uniform(0.1, 0.6, n)
    n_zero = max(1, int(round(0.03 * n)))
    zero_idx = rng.choice(n, size=n_zero, replace=False)
    disk_r50[zero_idx] = 0.0

    f_nu_y = 10.0 ** rng.uniform(-28.8, -27.8, n)
    f_nu_j = f_nu_y * rng.uniform(1.0, 1.3, n)
    f_nu_h = f_nu_j * rng.uniform(1.0, 1.3, n)

    df = pd.DataFrame(
        {
            'galaxy_id': galaxy_id.astype(np.int64),
            'halo_id': halo_id.astype(np.int64),
            'ra_gal': rng.uniform(200.0, 203.0, n),
            'dec_gal': rng.uniform(0.0, 2.0, n),
            'true_redshift_gal': z,
            'observed_redshift_gal': z + rng.normal(0.0, 1e-3, n),
            'logf_halpha_model3': logf3ext + rng.uniform(0.05, 0.4, n),
            'logf_halpha_model3_ext': logf3ext,
            'logf_halpha_model1': logf3ext + rng.normal(0.2, 0.1, n),
            'logf_halpha_model1_ext': logf3ext + rng.normal(0.0, 0.1, n),
            'halpha_scatter': rng.normal(0.0, 1.0, n),
            'logf_n2_model3_ext': logf3ext - 0.7,
            'logf_o3_model3_ext': logf3ext - 0.4,
            'log_stellar_mass': rng.uniform(9.0, 11.2, n),
            'log_sfr': rng.uniform(-0.5, 1.5, n),
            'disk_r50': disk_r50,
            'disk_scalelength': disk_r50 / 1.678,
            'disk_axis_ratio': rng.uniform(0.15, 0.95, n),
            'disk_angle': rng.uniform(-90.0, 90.0, n),
            'disk_ellipticity': rng.uniform(0.05, 0.8, n),
            'inclination_angle': rng.uniform(2.0, 88.0, n),
            'bulge_fraction': rng.uniform(0.0, 0.7, n),
            'bulge_r50': rng.uniform(0.05, 0.3, n),
            'bulge_nsersic': rng.uniform(0.5, 4.5, n),
            'bulge_axis_ratio': rng.uniform(0.5, 0.98, n),
            'euclid_nisp_y': f_nu_y,
            'euclid_nisp_j': f_nu_j,
            'euclid_nisp_h': f_nu_h,
            'euclid_nisp_y_el_model3_ext': f_nu_y * 1.02,
            'euclid_nisp_j_el_model3_ext': f_nu_j * 1.02,
            'euclid_nisp_h_el_model3_ext': f_nu_h * 1.02,
            'euclid_vis': f_nu_y * 0.5,
            'euclid_vis_el_model3_ext': f_nu_y * 0.51,
            'kappa': rng.normal(0.0, 0.02, n),
            'gamma1': rng.normal(0.0, 0.01, n),
            'gamma2': rng.normal(0.0, 0.01, n),
        }
    )
    # mirror the real download dtypes: int64 ids, float32 everything else
    for col in df.columns:
        if col not in ('galaxy_id', 'halo_id'):
            df[col] = df[col].astype(np.float32)
    return df[list(FLAGSHIP2_COLUMNS)]


def write_fake_catalog(
    data_dir: Path, df: pd.DataFrame, name: str = 'flagship2_fake'
) -> Path:
    """Write a fake catalog parquet + matching provenance sidecar."""
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / f'{name}.parquet'
    df.to_parquet(parquet_path, index=False)
    sidecar = {
        'name': name,
        'query_id': -1,
        'sha256': hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
    }
    (data_dir / f'{name}.provenance.json').write_text(json.dumps(sidecar))
    return parquet_path


def catalog_spec_dict(data_dir: Path, **population_overrides) -> dict:
    """A full catalog-mode ensemble spec dict pointing at a fake catalog."""
    d = {
        'run': {
            'name': 'pop_test',
            'version': 1,
            'description': 'catalog population unit test',
            'seed': 777,
            'measurement': 'shear_bias',
            'noise_reps': 1,
        },
        'population': {
            'type': 'catalog',
            'catalog': {'download': 'flagship2_fake', 'data_dir': str(data_dir)},
            'preprocess': {'flux_variant': 'model3_ext', 'h': 0.67},
            'selection': {
                'z_range': [0.55, 1.9],
                'snr_line_total_min': 0.5,
                'bulge_fraction_max': None,
                'bulge_nsersic_range': None,
                'min_r50_over_psf_fwhm': None,
            },
            'sample': {'n_galaxies': 300, 'replace': False},
            'paint': {
                'tfr': {
                    'logv0': 2.384,
                    'logm0': 10.50,
                    'slope': 3.60,
                    'scatter_dex': 0.061,
                },
                'sigma0': {
                    'intercept_kms': 21.1,
                    'slope_kms': 11.3,
                    'scatter_kms': 11.3,
                    'min_kms': 5.0,
                },
            },
            'orientation': {'cosi_range': [0.05, 0.95], 'ring': {'members': 2}},
            'shear': {'sigma': 0.03, 'gmax': 0.1},
            'priors': {'logm_obs_scatter_dex': 0.25},
        },
        'observation': {
            # no snr block: catalog mode takes both channels' per-galaxy SNRs
            # from the population table (a scalar block here is rejected);
            # hlwas_medium carries the F129+F158 pair the physical band
            # fluxes exist for
            'config': 'hlwas_medium',
        },
        'fit': {'pin_z_to_truth': True},
        'dispatch': {'backend': 'local'},
        'output': {},
    }
    for key, value in population_overrides.items():
        d['population'][key] = value
    return d


def spec_from_dict(tmp_path: Path, d: dict) -> EnsembleSpec:
    path = tmp_path / 'spec.yaml'
    path.write_text(yaml.safe_dump(d))
    return EnsembleSpec.from_yaml(path)


@pytest.fixture(scope='module')
def fake_data_dir(tmp_path_factory) -> Path:
    data_dir = tmp_path_factory.mktemp('cosmohub_fake')
    write_fake_catalog(data_dir, fake_flagship2_rows(n=400, seed=1234))
    return data_dir


@pytest.fixture(scope='module')
def built(fake_data_dir, tmp_path_factory):
    """One standard build shared by the statistical tests (n_galaxies=300)."""
    spec = spec_from_dict(
        tmp_path_factory.mktemp('spec'), catalog_spec_dict(fake_data_dir)
    )
    return build_population(spec)


# ==============================================================================
# Loader validation
# ==============================================================================


class TestLoader:
    def test_missing_parquet_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match='download-cosmohub-dev'):
            load_catalog(FLAGSHIP2, 'nope', tmp_path)

    def test_missing_sidecar_raises(self, tmp_path):
        write_fake_catalog(tmp_path, fake_flagship2_rows(n=20))
        (tmp_path / 'flagship2_fake.provenance.json').unlink()
        with pytest.raises(RuntimeError, match='provenance'):
            load_catalog(FLAGSHIP2, 'flagship2_fake', tmp_path)

    def test_sha_mismatch_raises(self, tmp_path):
        parquet_path = write_fake_catalog(tmp_path, fake_flagship2_rows(n=20))
        with parquet_path.open('ab') as f:
            f.write(b'corruption')
        with pytest.raises(RuntimeError, match='sha256 mismatch'):
            load_catalog(FLAGSHIP2, 'flagship2_fake', tmp_path)

    def test_missing_column_raises(self, tmp_path):
        df = fake_flagship2_rows(n=20).drop(columns=['bulge_nsersic'])
        write_fake_catalog(tmp_path, df)
        with pytest.raises(ValueError, match='bulge_nsersic'):
            load_catalog(FLAGSHIP2, 'flagship2_fake', tmp_path)

    def test_duplicate_compound_id_raises(self, tmp_path):
        df = fake_flagship2_rows(n=20)
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        write_fake_catalog(tmp_path, df)
        with pytest.raises(ValueError, match='duplicate'):
            load_catalog(FLAGSHIP2, 'flagship2_fake', tmp_path)

    def test_valid_catalog_loads(self, fake_data_dir):
        df = load_catalog(FLAGSHIP2, 'flagship2_fake', fake_data_dir)
        assert len(df) == 400
        assert set(df.columns) == set(FLAGSHIP2_COLUMNS)


# ==============================================================================
# Catalog adapter seam
# ==============================================================================


class TestCatalogAdapter:
    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match='unknown catalog kind'):
            get_catalog_adapter('cosmos99')

    def test_lookup_is_case_insensitive(self):
        assert get_catalog_adapter('Flagship2') is FLAGSHIP2

    def test_flagship2_id_columns_are_the_seed_key(self):
        # (halo_id, galaxy_id) in this order is the per-galaxy seed key
        # composition; changing it breaks every frozen census draw
        assert FLAGSHIP2.id_columns == ('halo_id', 'galaxy_id')

    def test_spec_kind_defaults_to_flagship2(self, fake_data_dir, tmp_path):
        spec = spec_from_dict(tmp_path, catalog_spec_dict(fake_data_dir))
        assert spec.catalog_population.catalog_kind == 'flagship2'

    def test_spec_explicit_kind_parses(self, fake_data_dir, tmp_path):
        d = catalog_spec_dict(fake_data_dir)
        d['population']['catalog']['kind'] = 'flagship2'
        spec = spec_from_dict(tmp_path, d)
        assert spec.catalog_population.catalog_kind == 'flagship2'

    def test_spec_unknown_kind_rejected(self, fake_data_dir, tmp_path):
        d = catalog_spec_dict(fake_data_dir)
        d['population']['catalog']['kind'] = 'cosmos99'
        with pytest.raises(ValueError, match='unknown catalog kind'):
            spec_from_dict(tmp_path, d)

    def test_preprocess_output_matches_columns(self):
        from kl_pipe.ensemble.catalogs import validate_columns

        pre = FLAGSHIP2.preprocess(fake_flagship2_rows(n=50, seed=11), _cp())
        validate_columns(FLAGSHIP2, pre)  # exact column-set equality
        assert set(pre.columns) == set(FLAGSHIP2.required_columns())

    def test_wrong_column_set_raises(self):
        from kl_pipe.ensemble.catalogs import validate_columns

        pre = FLAGSHIP2.preprocess(fake_flagship2_rows(n=50, seed=11), _cp())
        with pytest.raises(ValueError, match='wrong column set'):
            validate_columns(FLAGSHIP2, pre.drop(columns=['ew_rest_a']))


# ==============================================================================
# Preprocess
# ==============================================================================


def _cp(**overrides) -> CatalogPopulationSpec:
    """A CatalogPopulationSpec with test defaults."""
    kwargs = dict(
        catalog_download='flagship2_fake',
        catalog_data_dir='unused',
        flux_variant='model3_ext',
        h=0.67,
        z_range=(0.55, 1.9),
        snr_line_total_min=0.5,
        bulge_fraction_max=None,
        bulge_nsersic_range=None,
        min_r50_over_psf_fwhm=None,
        n_galaxies=10,
        tfr_logv0=2.384,
        tfr_logm0=10.50,
        tfr_slope=3.60,
        tfr_scatter_dex=0.061,
        sigma0_intercept_kms=21.1,
        sigma0_slope_kms=11.3,
        sigma0_scatter_kms=11.3,
        sigma0_min_kms=5.0,
        cosi_range=(0.05, 0.95),
        ring_members=2,
        shear_sigma=0.03,
        shear_gmax=0.1,
        logm_obs_scatter_dex=0.25,
    )
    kwargs.update(overrides)
    return CatalogPopulationSpec(**kwargs)


class TestPreprocess:
    def test_one_component_rows_dropped(self):
        df = fake_flagship2_rows(n=400, seed=42)
        n_zero = int((df['disk_r50'].to_numpy() <= 0).sum())
        assert n_zero > 0  # generator plants ~3%
        pre = FLAGSHIP2.preprocess(df, _cp())
        assert len(pre) == len(df) - n_zero
        assert pre.attrs['kills']['one_component'] == n_zero
        assert (pre['rscale_arcsec'] > 0).all()

    def test_h_mass_correction_exact(self):
        # logM*_phys = column - 2*log10(h); for h = 0.67 the shift is
        # -2*log10(0.67) = +0.34785039... dex, applied identically per row
        df = fake_flagship2_rows(n=50, seed=7)
        pre = FLAGSHIP2.preprocess(df, _cp(h=0.67))
        raw = df.loc[df['disk_r50'] > 0, 'log_stellar_mass'].to_numpy(np.float64)
        shift = -2.0 * np.log10(0.67)
        assert abs(shift - 0.3478503946) < 1e-10
        np.testing.assert_allclose(
            pre['logm'].to_numpy(), raw + shift, rtol=0, atol=1e-12
        )

    def test_ew_hand_round_trip(self):
        # hand-built row: z = 1 -> lambda_obs = 13125.6 A (J band:
        # 11900 <= lambda < 15450); fix f_nu_J and f_line and compute the
        # expected rest EW by hand:
        #   f_lambda = f_nu * c / lambda^2 [erg/cm2/s/A]
        #   EW_rest  = f_line / f_lambda / (1 + z) [A]
        df = fake_flagship2_rows(n=1, seed=3)
        df['disk_r50'] = np.float32(0.3)
        df['disk_scalelength'] = np.float32(0.3 / 1.678)
        df['true_redshift_gal'] = np.float32(1.0)
        df['logf_halpha_model3_ext'] = np.float32(np.log10(2e-16))
        df['euclid_nisp_j'] = np.float32(1e-28)
        pre = FLAGSHIP2.preprocess(df, _cp())
        # recompute expected from the float32-stored inputs
        z = float(np.float32(1.0))
        lam = 6562.8 * (1.0 + z)
        f_line = 10.0 ** float(np.float32(np.log10(2e-16)))
        f_nu = float(np.float32(1e-28))
        f_lambda = f_nu * 2.998e18 / lam**2
        expected_ew_rest = f_line / f_lambda / (1.0 + z)
        # sanity scale: EW_obs = EW_rest * (1 + z) ~ 115 A here
        assert 40.0 < expected_ew_rest < 90.0
        np.testing.assert_allclose(
            pre['ew_rest_a'].iloc[0], expected_ew_rest, rtol=1e-12
        )
        np.testing.assert_allclose(
            pre['f_lambda_cont_cgs'].iloc[0], f_lambda, rtol=1e-12
        )

    def test_nisp_band_selection_edges(self):
        # z = 0.75 -> 11485 A (Y, < 11900); z = 1.0 -> 13126 A (J);
        # z = 1.5 -> 16407 A (H, >= 15450)
        df = fake_flagship2_rows(n=3, seed=5)
        df['disk_r50'] = np.float32(0.3)
        df['disk_scalelength'] = np.float32(0.3 / 1.678)
        df['true_redshift_gal'] = np.array([0.75, 1.0, 1.5], dtype=np.float32)
        df['euclid_nisp_y'] = np.float32(1e-28)
        df['euclid_nisp_j'] = np.float32(2e-28)
        df['euclid_nisp_h'] = np.float32(4e-28)
        pre = FLAGSHIP2.preprocess(df, _cp())
        lam = pre['lambda_obs_a'].to_numpy()
        f_nu_used = pre['f_lambda_cont_cgs'].to_numpy() * lam**2 / 2.998e18
        np.testing.assert_allclose(f_nu_used, [1e-28, 2e-28, 4e-28], rtol=1e-6)

    def test_nonpositive_flux_raises(self):
        df = fake_flagship2_rows(n=10, seed=9)
        df['euclid_nisp_j'] = np.float32(0.0)
        df['euclid_nisp_y'] = np.float32(0.0)
        df['euclid_nisp_h'] = np.float32(0.0)
        with pytest.raises(ValueError, match='non-finite or non-positive'):
            FLAGSHIP2.preprocess(df, _cp())


# ==============================================================================
# Matched-filter compactness + line SNR
# ==============================================================================


class TestCompactness:
    def test_monotonic_decreasing_in_reff(self):
        reff = np.linspace(0.05, 2.0, 50)
        c = matched_filter_compactness(reff, np.full(50, 0.5), np.full(50, 1.0))
        assert (np.diff(c) < 0).all()

    def test_bounds(self):
        rng = np.random.default_rng(11)
        c = matched_filter_compactness(
            rng.uniform(0.01, 3.0, 500),
            rng.uniform(0.05, 0.95, 500),
            rng.uniform(0.55, 1.9, 500),
        )
        assert (c > 0).all() and (c <= 1.0).all()

    def test_unresolved_limit(self):
        # reff << PSF -> C -> 1
        c = matched_filter_compactness(
            np.array([1e-4]), np.array([0.5]), np.array([1.0])
        )
        assert c[0] > 0.999

    def test_snr_reference_source_anchor(self):
        # the source the limit is referenced to, at exactly the per-pass
        # limit, has SNR = F_LIM_NSIGMA -- true under either reference
        snr = compute_line_snr_per_pass(
            np.array([population.F_LIM_PER_PASS_CGS]),
            np.array([population.fiducial_compactness()]),
        )
        np.testing.assert_allclose(snr, [population.F_LIM_NSIGMA], rtol=1e-12)

    def test_nonpositive_reff_raises(self):
        with pytest.raises(ValueError, match='positive'):
            matched_filter_compactness(
                np.array([0.0]), np.array([0.5]), np.array([1.0])
            )


# ==============================================================================
# build_population
# ==============================================================================


class TestBuildPopulation:
    def test_tfr_paint_stats(self, built):
        # residual = log10(vcirc) - TFR(logm) ~ N(0, 0.061) iid over N=300:
        # |mean| < 3 * 0.061/sqrt(300) = 0.0106 and
        # |std - 0.061| < 3 * 0.061/sqrt(2*300) = 0.0075 (3-sigma stat
        # bounds from the standard errors of mean and std)
        df, _ = built
        n = len(df)
        assert n == 300
        resid = np.log10(df['vcirc_kms'].to_numpy()) - (
            2.384 + (df['logm'].to_numpy() - 10.50) / 3.60
        )
        assert abs(resid.mean()) < 3 * 0.061 / np.sqrt(n)
        assert abs(resid.std(ddof=1) - 0.061) < 3 * 0.061 / np.sqrt(2 * n)

    def test_sigma0_min_respected(self, fake_data_dir, tmp_path):
        # min_kms = 30 with mu = 21.1 + 11.3 z (z in 0.6-1.8 -> mu in
        # 27.9-41.4) and scatter 11.3 forces frequent resampling
        d = catalog_spec_dict(fake_data_dir)
        d['population']['paint']['sigma0']['min_kms'] = 30.0
        d['population']['sample']['n_galaxies'] = 100
        df, _ = build_population(spec_from_dict(tmp_path, d))
        assert (df['sigma0_kms'] >= 30.0).all()

    def test_shear_bounds_and_pair_semantics(self, fake_data_dir, tmp_path):
        # gmax = 0.03 = sigma forces redraws; one (g1, g2) PER GALAXY ROW
        # by construction (ring expansion downstream duplicates the row, so
        # both members share the draw)
        d = catalog_spec_dict(fake_data_dir)
        d['population']['shear'] = {'sigma': 0.03, 'gmax': 0.03}
        df, _ = build_population(spec_from_dict(tmp_path, d))
        g = np.hypot(df['g1'].to_numpy(), df['g2'].to_numpy())
        assert (g < 0.03).all()
        assert df.shape[0] == 300  # one row per galaxy, not per ring member

    def test_cosi_range_and_uniformity(self, built):
        # isotropic redraw: cosi ~ U(0.05, 0.95); KS test at alpha = 1e-3
        # (false-alarm rate 0.1% on this fixed seed -- deterministic)
        df, _ = built
        cosi = df['cosi'].to_numpy()
        assert (cosi >= 0.05).all() and (cosi <= 0.95).all()
        u = (cosi - 0.05) / 0.90
        assert stats.kstest(u, 'uniform').pvalue > 1e-3

    def test_theta_int_range(self, built):
        df, _ = built
        theta = df['theta_int'].to_numpy()
        assert (theta >= 0).all() and (theta < np.pi).all()

    def test_selection_monotonic_in_snr_min(self, fake_data_dir, tmp_path):
        # raising snr_line_total_min never increases the selected count
        counts = []
        for i, snr_min in enumerate([0.5, 2.0, 5.0, 8.0]):
            d = catalog_spec_dict(fake_data_dir)
            d['population']['selection']['snr_line_total_min'] = snr_min
            d['population']['sample']['n_galaxies'] = 1
            sub = tmp_path / f's{i}'
            sub.mkdir()
            _, meta = build_population(spec_from_dict(sub, d))
            counts.append(meta['n_selected'])
        assert all(a >= b for a, b in zip(counts, counts[1:]))

    def test_n_galaxies_shortfall_raises(self, fake_data_dir, tmp_path):
        d = catalog_spec_dict(fake_data_dir)
        d['population']['sample']['n_galaxies'] = 100000
        with pytest.raises(ValueError, match='n_raw=400'):
            build_population(spec_from_dict(tmp_path, d))

    def test_determinism_exact(self, fake_data_dir, tmp_path, built):
        df1, meta1 = built
        df2, meta2 = build_population(
            spec_from_dict(tmp_path, catalog_spec_dict(fake_data_dir))
        )
        pd.testing.assert_frame_equal(df1, df2, check_exact=True)
        assert meta1['kills'] == meta2['kills']

    def test_seed_changes_draws(self, fake_data_dir, tmp_path, built):
        df1, _ = built
        d = catalog_spec_dict(fake_data_dir)
        d['run']['seed'] = 778
        df2, _ = build_population(spec_from_dict(tmp_path, d))
        assert not np.allclose(df1['cosi'], df2['cosi'])
        assert not np.allclose(df1['vcirc_kms'], df2['vcirc_kms'])
        assert not np.allclose(df1['g1'], df2['g1'])

    def test_superset_catalog_stability(self, tmp_path):
        # id-keyed streams: a superset catalog (extra rows that fail the
        # z cut) must yield the IDENTICAL population when the same galaxies
        # are selected and n_galaxies = all selected
        base = fake_flagship2_rows(n=200, seed=21)
        extra = fake_flagship2_rows(
            n=60, seed=22, z_low=2.2, z_high=2.8, id_offset=10_000_000
        )
        base_dir = tmp_path / 'catalog_base'
        super_dir = tmp_path / 'catalog_super'
        write_fake_catalog(base_dir, base)
        write_fake_catalog(super_dir, pd.concat([base, extra], ignore_index=True))

        # first pass: learn n_selected on the base catalog
        d = catalog_spec_dict(base_dir)
        d['population']['sample']['n_galaxies'] = 1
        probe_dir = tmp_path / 'probe'
        probe_dir.mkdir()
        _, meta = build_population(spec_from_dict(probe_dir, d))
        n_all = meta['n_selected']
        assert n_all > 100

        results = []
        for label, data_dir in [('base', base_dir), ('super', super_dir)]:
            d = catalog_spec_dict(data_dir)
            d['population']['sample']['n_galaxies'] = n_all
            sub = tmp_path / label
            sub.mkdir()
            results.append(build_population(spec_from_dict(sub, d)))
        (df_base, meta_base), (df_super, meta_super) = results
        assert meta_super['n_raw'] == meta_base['n_raw'] + 60
        assert meta_super['n_selected'] == meta_base['n_selected']
        pd.testing.assert_frame_equal(df_base, df_super, check_exact=True)

    def test_prior_columns(self, built):
        # prior width combines TFR scatter and propagated mass error:
        # sqrt(0.061^2 + (0.25/3.6)^2) = 0.09258...
        df, _ = built
        expected = np.sqrt(0.061**2 + (0.25 / 3.6) ** 2)
        np.testing.assert_allclose(df['prior_vcirc_sigma_dex'], expected, rtol=1e-12)
        # prior center is the TFR at logm_obs
        np.testing.assert_allclose(
            df['prior_vcirc_mu_kms'],
            10.0 ** (2.384 + (df['logm_obs'] - 10.50) / 3.60),
            rtol=1e-12,
        )

    def test_meta_stage_counts(self, built):
        df, meta = built
        assert meta['n_raw'] == 400
        assert meta['n_disk'] == 400 - meta['kills']['one_component']
        assert meta['n_selected'] <= meta['n_disk']
        assert meta['n_sampled'] == len(df) == 300
        assert meta['catalog_sha256']

    def test_write_population(self, built, tmp_path):
        df, meta = built
        parquet_path, meta_path = write_population(tmp_path / 'out', df, meta)
        df2 = pd.read_parquet(parquet_path)
        pd.testing.assert_frame_equal(df, df2, check_exact=True)
        assert json.loads(meta_path.read_text())['n_sampled'] == 300


# ==============================================================================
# Bulge morphology paint
# ==============================================================================


class TestBulgePaint:
    def test_ranges_and_size_cap(self, built):
        df, _ = built
        n = df['bulge_nsersic'].to_numpy()
        lo = min(BULGE_PSEUDO_N[2], BULGE_CLASSICAL_N[2])
        hi = max(BULGE_PSEUDO_N[3], BULGE_CLASSICAL_N[3])
        assert np.all((n >= lo) & (n <= hi))
        disk_r50 = R50_TO_SIGMA * df['rscale_arcsec'].to_numpy()
        ratio = df['bulge_r50_arcsec'].to_numpy() / disk_r50
        # cap is strict in the paint; 1e-5 covers the fake catalog's float32
        # scalelength round-trip
        assert np.all(ratio < BULGE_SIZE_RATIO_MAX * (1 + 1e-5))
        assert np.all(ratio > 0)

    def test_mixture_and_ratio_stats(self, built):
        df, _ = built
        n = df['bulge_nsersic'].to_numpy()
        # pseudo branch is fully below 2, classical fully above: P(n <= 2)
        # equals the mixture weight exactly. n=300 draws -> binomial
        # sd = sqrt(0.7*0.3/300) = 0.026; bound = 4 sigma ~= 0.11
        pseudo_frac = float((n <= BULGE_PSEUDO_N[3]).mean())
        assert abs(pseudo_frac - BULGE_PSEUDO_WEIGHT) < 0.11
        # lognormal ratio median: asymptotic sd of the sample ln-median is
        # 1.2533 * 0.4 / sqrt(300) = 0.029; 4 sigma -> |ln(med/0.3)| < 0.116
        disk_r50 = R50_TO_SIGMA * df['rscale_arcsec'].to_numpy()
        ratio = df['bulge_r50_arcsec'].to_numpy() / disk_r50
        assert abs(np.log(np.median(ratio) / BULGE_SIZE_RATIO_MEDIAN)) < 0.116

    def test_catalog_columns_retained_and_replaced(self, built):
        df, meta = built
        assert 'catalog_bulge_nsersic' in df.columns
        assert 'catalog_bulge_r50_arcsec' in df.columns
        # painted values are draws, not the catalog columns
        assert not np.allclose(df['bulge_nsersic'], df['catalog_bulge_nsersic'])
        assert not np.allclose(df['bulge_r50_arcsec'], df['catalog_bulge_r50_arcsec'])
        assert meta['bulge_paint']['pseudo_weight'] == BULGE_PSEUDO_WEIGHT

    def test_bulge_nsersic_range_cut_raises(self, fake_data_dir, tmp_path):
        d = catalog_spec_dict(fake_data_dir)
        d['population']['selection']['bulge_nsersic_range'] = [0.5, 4.0]
        spec = spec_from_dict(tmp_path, d)
        with pytest.raises(ValueError, match='bulge_nsersic_range'):
            build_population(spec)

    def test_resolvability_cut_removes_sub_psf_galaxies(self, fake_data_dir, tmp_path):
        # a sub-PSF disk has no velocity gradient to fit, and the
        # extended-fiducial flux reference rewards compact sources, so the cut
        # is what keeps the selection above the resolution limit
        from kl_pipe.ensemble.population import psf_fwhm_arcsec

        # the fake catalog draws disk_r50 from U(0.1, 0.6) arcsec against a
        # PSF FWHM of ~0.15 arcsec, so r50 / FWHM spans ~0.65-4; a floor of
        # 1.0 lands inside that range and actually removes rows
        floor = 1.0
        d = catalog_spec_dict(fake_data_dir)
        d['population']['sample']['n_galaxies'] = 50
        d['population']['selection']['min_r50_over_psf_fwhm'] = floor
        df, meta = build_population(spec_from_dict(tmp_path, d))
        ratio = df['rscale_arcsec'].to_numpy() * R50_TO_SIGMA / psf_fwhm_arcsec(df['z'])
        assert (ratio >= floor).all()
        assert meta['kills']['resolvability'] > 0

    def test_resolvability_cut_null_keeps_everything(self, fake_data_dir, tmp_path):
        d = catalog_spec_dict(fake_data_dir)
        d['population']['selection']['min_r50_over_psf_fwhm'] = None
        _, meta = build_population(spec_from_dict(tmp_path, d))
        assert meta['kills']['resolvability'] == 0

    def test_resolvability_cut_only_shrinks_selection(self, fake_data_dir, tmp_path):
        counts = []
        for floor in (None, 0.25, 0.5, 1.0):
            d = catalog_spec_dict(fake_data_dir)
            d['population']['sample']['n_galaxies'] = 10
            d['population']['selection']['min_r50_over_psf_fwhm'] = floor
            _, meta = build_population(spec_from_dict(tmp_path, d))
            counts.append(meta['n_selected'])
        assert counts == sorted(counts, reverse=True)

    def test_painted_index_spans_the_full_emulator_support(
        self, fake_data_dir, tmp_path
    ):
        # the classical component runs to the emulator's full 6.0 support;
        # cutting it at 4 would misspecify 10.7% of all bulges for no saving
        d = catalog_spec_dict(fake_data_dir)
        d['population']['sample']['n_galaxies'] = 300
        df, _ = build_population(spec_from_dict(tmp_path, d))
        n = df['bulge_nsersic'].to_numpy()
        assert n.max() <= population.BULGE_N_MAX
        assert n.min() >= population.BULGE_PSEUDO_N[2]
        assert population.BULGE_N_MAX == 6.0
        # the mixture must stay pseudobulge-dominated
        assert 0.6 < float((n <= 2.0).mean()) < 0.8
        # and the classical tail past 4 must actually be populated
        assert float((n > 4.0).mean()) > 0.03

    def test_scene_prior_is_the_generating_distribution(self):
        # the fit prior is the paint itself, not an approximation to it, so
        # the marginalization over bulge index is exact -- the same property
        # bulge_hlr has. Checked with a two-sample KS test.
        import jax
        import numpy as _np
        from kl_pipe.ensemble.scene import _bulge_nsersic_prior

        rng = _np.random.default_rng(0)
        painted = _np.array(
            [
                population._truncated_normal(
                    rng,
                    *(
                        population.BULGE_PSEUDO_N
                        if rng.uniform() < population.BULGE_PSEUDO_WEIGHT
                        else population.BULGE_CLASSICAL_N
                    ),
                )
                for _ in range(20000)
            ]
        )
        drawn = _np.asarray(
            _bulge_nsersic_prior().sample(jax.random.PRNGKey(0), (20000,))
        )
        # p > 0.01 with n = 2e4 per sample: the KS test resolves shape
        # differences far smaller than the unimodal approximation this
        # replaced (which shifted ~11% of the mass)
        assert stats.ks_2samp(painted, drawn).pvalue > 0.01

    def test_scene_prior_supports_every_painted_value(self, fake_data_dir, tmp_path):
        # a painted truth outside the prior support would be unrecoverable
        import jax.numpy as jnp
        from kl_pipe.ensemble.scene import _bulge_nsersic_prior

        d = catalog_spec_dict(fake_data_dir)
        d['population']['sample']['n_galaxies'] = 200
        df, _ = build_population(spec_from_dict(tmp_path, d))
        logp = _bulge_nsersic_prior().log_prob(
            jnp.asarray(df['bulge_nsersic'].to_numpy())
        )
        assert bool(jnp.all(jnp.isfinite(logp)))

    def test_nonpositive_resolvability_floor_raises(self, fake_data_dir, tmp_path):
        d = catalog_spec_dict(fake_data_dir)
        d['population']['selection']['min_r50_over_psf_fwhm'] = 0.0
        with pytest.raises(ValueError, match='min_r50_over_psf_fwhm'):
            spec_from_dict(tmp_path, d)

    def test_legacy_snr_line_min_key_raises(self, fake_data_dir, tmp_path):
        # the old key was the PER-PASS SNR applied as if it were the total,
        # so silently accepting it would keep selecting sqrt(N) deeper than
        # the number reads
        d = catalog_spec_dict(fake_data_dir)
        del d['population']['selection']['snr_line_total_min']
        d['population']['selection']['snr_line_min'] = 10.0
        with pytest.raises(ValueError, match='snr_line_total_min'):
            spec_from_dict(tmp_path, d)


# ==============================================================================
# Spec parsing (catalog branch)
# ==============================================================================


class TestCatalogSpecParsing:
    def test_yaml_round_trip(self, fake_data_dir, tmp_path):
        spec = spec_from_dict(tmp_path, catalog_spec_dict(fake_data_dir))
        cp = spec.catalog_population
        assert spec.population_type == 'catalog'
        assert cp.flux_variant == 'model3_ext'
        assert cp.z_range == (0.55, 1.9)
        assert cp.ring_members == 2
        assert cp.bulge_fraction_max is None
        assert spec.n_fits == 300 * 2 * 1  # n_galaxies * ring members * reps

    def test_sampled_only_keys_rejected_in_catalog_mode(self, fake_data_dir, tmp_path):
        d = catalog_spec_dict(fake_data_dir)
        d['population']['stratify'] = {'cosi': {'n_bins': 2, 'range': [0.3, 0.9]}}
        with pytest.raises(ValueError, match='unknown keys'):
            spec_from_dict(tmp_path, d)

    def test_catalog_only_keys_rejected_in_sampled_mode(self, tmp_path):
        d = yaml.safe_load(SAMPLED_SPEC.read_text())
        d['population']['paint'] = {'tfr': {}}
        with pytest.raises(ValueError, match='unknown keys'):
            spec_from_dict(tmp_path, d)

    def test_catalog_requires_shear_bias_measurement(self, fake_data_dir, tmp_path):
        d = catalog_spec_dict(fake_data_dir)
        d['run']['measurement'] = 'sigma_eps_vs_cosi'
        with pytest.raises(ValueError, match='shear_bias'):
            spec_from_dict(tmp_path, d)

    def test_sampled_rejects_shear_bias_measurement(self, tmp_path):
        d = yaml.safe_load(SAMPLED_SPEC.read_text())
        d['run']['measurement'] = 'shear_bias'
        with pytest.raises(ValueError, match='population.type: catalog'):
            spec_from_dict(tmp_path, d)

    def test_replace_true_not_implemented(self, fake_data_dir, tmp_path):
        d = catalog_spec_dict(fake_data_dir)
        d['population']['sample']['replace'] = True
        with pytest.raises(NotImplementedError, match='replace'):
            spec_from_dict(tmp_path, d)

    def test_missing_catalog_subblock_rejected(self, fake_data_dir, tmp_path):
        d = catalog_spec_dict(fake_data_dir)
        del d['population']['paint']
        with pytest.raises(ValueError, match='missing required keys'):
            spec_from_dict(tmp_path, d)

    def test_unknown_flux_variant_rejected(self, fake_data_dir, tmp_path):
        d = catalog_spec_dict(fake_data_dir)
        d['population']['preprocess']['flux_variant'] = 'model7'
        with pytest.raises(ValueError, match='flux_variant'):
            spec_from_dict(tmp_path, d)

    def test_bad_ring_members_rejected(self, fake_data_dir, tmp_path):
        d = catalog_spec_dict(fake_data_dir)
        d['population']['orientation']['ring']['members'] = 3
        with pytest.raises(ValueError, match='members'):
            spec_from_dict(tmp_path, d)

    def test_noise_reps_multiplies_n_fits(self, fake_data_dir, tmp_path):
        d = catalog_spec_dict(fake_data_dir)
        d['run']['noise_reps'] = 3
        spec = spec_from_dict(tmp_path, d)
        assert spec.n_fits == 300 * 2 * 3

    def test_example_spec_parses(self):
        spec = EnsembleSpec.from_yaml(EXAMPLE_SPEC)
        assert spec.run_name == 'flagship2_shear_dev'
        assert spec.measurement == 'shear_bias'
        assert spec.catalog_population.n_galaxies == 8
        assert spec.n_fits == 8 * 2 * 1


# ==============================================================================
# CLI
# ==============================================================================


class TestCLI:
    def test_population_subcommand(self, fake_data_dir, tmp_path, capsys):
        from kl_pipe.ensemble.__main__ import main

        d = catalog_spec_dict(fake_data_dir)
        d['population']['sample']['n_galaxies'] = 20
        spec_path = tmp_path / 'spec.yaml'
        spec_path.write_text(yaml.safe_dump(d))
        out_dir = tmp_path / 'out'
        rc = main(['population', str(spec_path), '--out-dir', str(out_dir)])
        assert rc == 0
        captured = capsys.readouterr().out
        assert 'stage counts' in captured
        df = pd.read_parquet(out_dir / 'population.parquet')
        assert len(df) == 20
        assert (out_dir / 'population_meta.json').exists()


class TestLineSnrReference:
    """The source F_LIM is referenced to (population.SNR_LINE_REFERENCE).

    Getting this wrong double-counts (or omits) the extended-source penalty.
    Wang et al. 2022 Sect. 5 derive the published Roman line-flux limits for
    a galaxy of half-light radius 0.25 arcsec at 1.5 micron, not for a point
    source; 'extended_fiducial' encodes that convention.
    """

    @pytest.fixture(autouse=True)
    def restore_reference(self):
        original = population.SNR_LINE_REFERENCE
        yield
        population.SNR_LINE_REFERENCE = original

    def test_default_is_extended_fiducial(self):
        # the ROTAC limit is an extended-source limit: the committee asked
        # the HLSS PIT for "line flux limits for a realistic extended source"
        # (Appendix B.2 item 1a) while asking separately for point-source and
        # extended imaging depths (B.1 item 1a), Sect. 3.1 labels only the
        # imaging depth "5 sigma point source", and Sect. 4.2.1 refers to
        # Wang et al. 2022, whose limits are for r50 = 0.25 arcsec. A
        # point-source reading would also put our yield 12x below the 14.2M
        # Halpha over 2400 deg2 that ROTAC forecasts in Sect. 4.4.3.
        assert population.SNR_LINE_REFERENCE == 'extended_fiducial'
        assert 0.0 < population.fiducial_compactness() < 1.0

    def test_point_source_reference_is_identity(self):
        # a point source (C = 1) at exactly the per-pass limit has
        # SNR = F_LIM_NSIGMA when the limit is referenced to a point source
        population.SNR_LINE_REFERENCE = 'point_source'
        assert population.fiducial_compactness() == 1.0
        snr = compute_line_snr_per_pass(
            np.array([population.F_LIM_PER_PASS_CGS]), np.array([1.0])
        )
        assert snr[0] == pytest.approx(population.F_LIM_NSIGMA, rel=1e-12)

    def test_fiducial_galaxy_lands_on_nsigma(self):
        # self-consistency: under the extended reference, the very galaxy the
        # limit was derived for must sit exactly at F_LIM_NSIGMA when its
        # flux equals F_LIM
        z_ref = population.F_LIM_REF_LAMBDA_A / population.HALPHA_REST_A - 1.0
        c_fid = matched_filter_compactness(
            np.array([population.F_LIM_REF_R50_ARCSEC]),
            np.array([1.0]),
            np.array([z_ref]),
        )
        snr = compute_line_snr_per_pass(
            np.array([population.F_LIM_PER_PASS_CGS]), c_fid
        )
        assert snr[0] == pytest.approx(population.F_LIM_NSIGMA, rel=1e-12)

    def test_extended_reference_rewards_compact_sources(self):
        # a point source beats the extended fiducial by exactly 1 / C_fid
        c_ref = population.fiducial_compactness()
        assert 0.0 < c_ref < 1.0
        snr = compute_line_snr_per_pass(
            np.array([population.F_LIM_PER_PASS_CGS]), np.array([1.0])
        )
        assert snr[0] == pytest.approx(population.F_LIM_NSIGMA / c_ref, rel=1e-12)

    def test_reference_choice_is_a_pure_rescaling(self):
        # switching reference must not change the RANKING of galaxies, only
        # the overall normalization -- the physics of C is untouched
        f = np.array([1e-16, 3e-16, 5e-16])
        c = np.array([0.2, 0.5, 0.9])
        c_fid = population.fiducial_compactness()
        extended = compute_line_snr_per_pass(f, c)
        population.SNR_LINE_REFERENCE = 'point_source'
        point = compute_line_snr_per_pass(f, c)
        ratio = extended / point
        assert np.allclose(ratio, ratio[0], rtol=1e-12)
        assert ratio[0] == pytest.approx(1.0 / c_fid)

    def test_unknown_reference_raises(self):
        population.SNR_LINE_REFERENCE = 'not_a_reference'
        with pytest.raises(ValueError, match='SNR_LINE_REFERENCE'):
            population.fiducial_compactness()


class TestLineSnrCoadd:
    """Per-pass vs coadded line SNR.

    The published limit is the coadd over N_GRISM_PASSES passes; a single
    pass is sqrt(N) shallower. Selection cuts belong on the total (the depth
    the joint multi-roll fit sees), while each roll's mock noise is
    normalized to one pass.
    """

    def test_total_is_sqrt_n_times_per_pass(self):
        f = np.array([1e-16, 4e-16])
        c = np.array([0.3, 0.8])
        ratio = compute_line_snr_total(f, c) / compute_line_snr_per_pass(f, c)
        expected = np.sqrt(population.N_GRISM_PASSES)
        np.testing.assert_allclose(ratio, expected, rtol=1e-12)

    def test_per_pass_limit_is_sqrt_n_shallower_than_coadd(self):
        assert population.F_LIM_PER_PASS_CGS == pytest.approx(
            population.F_LIM_COADD_CGS * np.sqrt(population.N_GRISM_PASSES), rel=1e-12
        )

    def test_reference_source_at_coadd_limit_hits_nsigma(self):
        # the source the published coadded limit was derived for, at exactly
        # that flux, sits at F_LIM_NSIGMA in the coadd
        snr = compute_line_snr_total(
            np.array([population.F_LIM_COADD_CGS]),
            np.array([population.fiducial_compactness()]),
        )
        assert snr[0] == pytest.approx(population.F_LIM_NSIGMA, rel=1e-12)


class TestPublishedConstants:
    """The survey numbers the whole selection chain hangs on.

    Every other test in this module references these symbolically, so
    without a literal pin an edit to any of them would rescale the selected
    number density with nothing failing. Each value is quoted from its
    source below; changing one means the source changed.
    """

    def test_flux_limit_matches_rotac(self):
        # ROTAC Final Report 2025-04-24 v3, Sect. 3.1: HLWAS medium tier
        # "1.5 x 10^-16 erg/cm2/sec (5 sigma line flux limit, texp ~ 1500
        # sec)", the coadd over the 4 grism passes of Sect. 4.4.3
        assert population.F_LIM_COADD_CGS == 1.5e-16
        assert population.F_LIM_NSIGMA == 5.0
        assert population.N_GRISM_PASSES == 4

    def test_extended_fiducial_matches_wang22(self):
        # Wang et al. 2022 (arXiv:2110.01829) Sect. 5.1: limits derived
        # "for galaxies with radius 0.25 arcsec at 1.5 micron"
        assert population.F_LIM_REF_R50_ARCSEC == 0.25
        assert population.F_LIM_REF_LAMBDA_A == 1.5e4
        # the resulting reference compactness, which divides every galaxy's
        # C and therefore sets the overall yield normalization
        assert population.fiducial_compactness() == pytest.approx(0.4148, abs=5e-5)

    def test_imaging_depths_match_rotac(self):
        # ROTAC Final Report Table 1 (arXiv:2505.10574): HLWAS medium
        # 5-sigma point-source coadded depths, F106 26.5 / F129 26.4 /
        # F158 26.4 AB ("26.4 (JH)" grouped in the table itself)
        assert population.IMAGING_DEPTH_AB == {
            'F106': 26.5,
            'F129': 26.4,
            'F158': 26.4,
        }
        assert population.IMAGING_DEPTH_NSIGMA == 5.0
        # AB <-> uJy pivot; f_lim(F129) = 10**((23.9 - 26.4)/2.5) = 0.1 uJy
        assert population.AB_MAG_UJY_PIVOT == 23.9
        assert population.band_flux_limit_ujy('F129') == pytest.approx(0.1)

    def test_band_effective_wavelengths_match_galsim(self):
        # pinned literals vs galsim.roman getBandpasses effective_wavelength
        # (nm), the source recorded on BAND_EFFECTIVE_LAMBDA_A
        galsim_roman = pytest.importorskip('galsim.roman')
        bps = galsim_roman.getBandpasses()
        legacy = {'F106': 'Y106', 'F129': 'J129', 'F158': 'H158'}
        for band, lam_a in population.BAND_EFFECTIVE_LAMBDA_A.items():
            expected_a = bps[legacy[band]].effective_wavelength * 10.0
            assert lam_a == pytest.approx(expected_a, rel=5e-4), band

    def test_extended_source_offset_reproduces_rotac_footnote(self):
        # ROTAC Table 1 caption: "r1/2 = 0.3 arcsec extended source
        # thresholds are typically ~1.1 mag brighter" than the point-source
        # depths. The Gaussian matched-filter proxy must reproduce that
        # statement; tolerance 0.3 mag covers the proxy-vs-ETC difference
        # (measured 2026-07-31: F129 1.27, F158 1.08 mag)
        for band in ('F129', 'F158'):
            c = population.imaging_compactness(np.array([0.3]), np.array([1.0]), band)[
                0
            ]
            offset_mag = -2.5 * np.log10(c)
            assert abs(offset_mag - 1.1) < 0.3, (band, offset_mag)

    def test_band_snr_and_sigma_are_consistent(self):
        # sigma_f = f / SNR by construction (the flux cancels): the two
        # helpers must agree exactly, and the flux-independence must hold
        reff = np.array([0.15, 0.3])
        cosi = np.array([0.4, 0.9])
        f = np.array([5.0, 50.0])
        snr = population.compute_band_snr(f, reff, cosi, 'F158')
        sigma = population.band_flux_sigma_ujy(reff, cosi, 'F158')
        np.testing.assert_allclose(f / snr, sigma, rtol=1e-12)

    def test_line_flux_sigma_is_coadd_referenced(self):
        # sigma_f = (F_LIM_COADD / 5) * C_ref / C: at the reference
        # compactness the sigma is exactly the published coadded limit / 5
        c_ref = population.fiducial_compactness()
        sigma = population.line_flux_sigma_cgs(np.array([c_ref]))[0]
        assert sigma == pytest.approx(population.F_LIM_COADD_CGS / 5.0, rel=1e-12)


class TestPaintConstants:
    """Literal pins for the literature paint constants.

    Same rationale as TestPublishedConstants: every truth the expander
    paints hangs on these numbers, the prior-provenance registry quotes
    them, and nothing else fails if one is silently edited. Each value's
    full provenance lives in the comment block above its definition in
    population.py; changing one here means that source changed.
    """

    def test_velocity_turnover_ratio(self):
        # PROBES-I derivation (r_t / R_d), Miller et al. 2011 Fig. 2 and
        # Catinella et al. 2006 Polyex conversions bracketing it
        assert population.VEL_RSCALE_RATIO_MEDIAN == 0.5
        assert population.VEL_RSCALE_RATIO_DEX == 0.3

    def test_halpha_extent_ratio(self):
        # Nelson et al. 2012: median 1.3, rms 0.2 dex at z ~ 1 (3D-HST)
        assert population.HALPHA_RSCALE_RATIO_MEDIAN == 1.3
        assert population.HALPHA_RSCALE_RATIO_DEX == 0.2

    def test_centroid_scatters(self):
        # about one Roman pixel per component; gas-vs-stars offset half a
        # pixel (clump-driven, ~11% of the median Halpha half-light radius)
        assert population.CENTROID_SCATTER_ARCSEC == 0.1
        assert population.CONT_CENTROID_OFFSET_ARCSEC == 0.055

    def test_systemic_velocity_scatter(self):
        # painted v0 offset, well inside the ~200 km/s per-grism-pixel scale
        assert population.V0_SCATTER_KMS == 25.0

    def test_bulge_sersic_mixture(self):
        # Fisher & Drory 2008 split at n = 2; Gadotti 2009 Table 3 component
        # medians/widths; Mendez-Abreu et al. 2010 mixture weight at
        # B/T <= 0.3; upper bound = Sersic emulator support
        assert population.BULGE_N_MAX == 6.0
        assert population.BULGE_PSEUDO_WEIGHT == 0.7
        assert population.BULGE_PSEUDO_N == (1.5, 0.9, 0.5, 2.0)
        assert population.BULGE_CLASSICAL_N == (3.4, 1.3, 2.0, 6.0)

    def test_bulge_size_ratio(self):
        # Lang et al. 2014 / Gadotti 2009 bracket, ln-scatter from Gadotti
        # Table 3, capped below 1 (bulge larger than its disk is unphysical)
        assert population.BULGE_SIZE_RATIO_MEDIAN == 0.3
        assert population.BULGE_SIZE_RATIO_LN_SCATTER == 0.4
        assert population.BULGE_SIZE_RATIO_MAX == 1.0
