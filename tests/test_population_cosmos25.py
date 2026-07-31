"""
CI-safe unit tests for the cosmos25 catalog adapter and its population
chain (kl_pipe/ensemble/catalogs/cosmos25.py).

Runs entirely on a synthetic cosmos25-schema catalog written to tmp dirs;
real-parquet checks live in scripts/audit_cosmos25_painting.py (handshake
against the delivering notebook) and are not part of CI.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from kl_pipe.ensemble.catalogs import (
    get_catalog_adapter,
    load_catalog,
    validate_contract,
)
from kl_pipe.ensemble.catalogs.cosmos25 import (
    COSMOS25_COLUMNS,
    F115W_F150W_EDGE_A,
    F150W_PIVOT_A,
    F150W_RED_EDGE_A,
    F277W_PIVOT_A,
    R50_OVER_RSCALE,
    UJY_TO_CGS,
)
from kl_pipe.ensemble.population import C_A_PER_S, HALPHA_REST_A, build_population
from kl_pipe.ensemble.spec import EnsembleSpec

COSMOS25 = get_catalog_adapter('cosmos25')


# ==============================================================================
# Fake cosmos25 catalog generator (full joined-parquet schema)
# ==============================================================================


def fake_cosmos25_rows(n: int = 400, seed: int = 4321) -> pd.DataFrame:
    """Physically-sensible fake joined rows with the full 29-column schema.

    Sentinel structure mirrors the real join: non-galaxies (``type != 0``)
    carry zfinal = 0 and NaN painted fluxes; a few percent of galaxies get
    warn flags; painted wavelengths are exactly 6562.8 * (1 + zfinal); ebv
    sits on a 0.1-step grid like the LePhare output.
    """
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.6, 1.8, n)
    obj_type = np.zeros(n, dtype=np.int64)
    n_star = max(1, n // 20)
    star_idx = rng.choice(n, size=n_star, replace=False)
    obj_type[star_idx] = 1
    z[star_idx] = 0.0

    warn = np.zeros(n, dtype=np.int64)
    warn_idx = rng.choice(
        np.where(obj_type == 0)[0], size=max(1, n // 25), replace=False
    )
    warn[warn_idx] = rng.integers(1, 6, size=warn_idx.size)

    ebv = rng.integers(0, 6, n) * 0.1
    f_ha = 10.0 ** rng.uniform(-16.3, -15.0, n)  # bright enough for SNR tests
    f_ha[obj_type != 0] = np.nan
    # a few galaxies without painted lines (failed SED rows in the real file)
    nofit_idx = rng.choice(
        np.where(obj_type == 0)[0], size=max(1, n // 25), replace=False
    )
    f_ha[nofit_idx] = np.nan

    r50_deg = rng.uniform(0.1, 0.6, n) / 3600.0
    f150 = 10.0 ** rng.uniform(-1.0, 1.5, n)  # uJy

    df = pd.DataFrame(
        {
            'id': np.arange(n, dtype=np.int64) + 1,
            'ra': rng.uniform(149.5, 150.5, n),
            'dec': rng.uniform(1.7, 2.7, n),
            'warn_flag': warn,
            'mag_model_f150w': rng.uniform(20.0, 26.0, n),
            'snr_f150w': 10.0 ** rng.uniform(1.0, 3.0, n),
            'radius_sersic': r50_deg,
            'radius_sersic_err': r50_deg * 0.05,
            'sersic': rng.uniform(0.5, 4.0, n),
            'axratio_sersic': rng.uniform(0.15, 0.95, n),
            'e1_err': rng.uniform(0.005, 0.1, n),
            'e2_err': rng.uniform(0.005, 0.1, n),
            'flux_model_f115w': f150 * rng.uniform(0.6, 1.1, n),
            'flux_model_f150w': f150,
            'flux_model_f277w': f150 * rng.uniform(0.8, 1.6, n),
            'zfinal': z,
            'type': obj_type,
            'ebv_minchi2': ebv,
            'mass_med': rng.uniform(9.0, 11.2, n),
            'sfr_med': rng.uniform(-0.5, 1.5, n),
            'F_Ha': f_ha,
            'F_OII': f_ha * 0.3,
            'F_OIII': f_ha * 0.1,
            'lambda_Ha_obs': HALPHA_REST_A * (1.0 + z),
            'lambda_OII_obs': 3727.5 * (1.0 + z),
            'lambda_OIII_obs': 5006.8 * (1.0 + z),
            'redshift': z,
            'sfr_young': 10.0 ** rng.uniform(-0.5, 1.5, n),
            'log_OH': rng.uniform(7.6, 8.85, n),
        }
    )
    return df[list(COSMOS25_COLUMNS)]


def test_band_flux_powerlaw_interpolation():
    """Roman band fluxes = power-law interpolation of the JWST pivots.

    Hand-computed per band: F106/F129 through (F115W, F150W), F158 through
    (F150W, F277W), evaluated at the Roman effective wavelengths.
    """
    from kl_pipe.ensemble.catalogs.cosmos25 import (
        F115W_PIVOT_A,
        F150W_PIVOT_A,
        F277W_PIVOT_A,
    )
    from kl_pipe.ensemble.population import BAND_EFFECTIVE_LAMBDA_A

    df = fake_cosmos25_rows(n=200, seed=99)
    out = COSMOS25.preprocess(df, preprocess_spec('as_delivered'))
    raw = df.set_index('id').loc[out['id'].to_numpy()]
    f115 = raw['flux_model_f115w'].to_numpy(float)
    f150 = raw['flux_model_f150w'].to_numpy(float)
    f277 = raw['flux_model_f277w'].to_numpy(float)

    for band, (fb, lb, fr, lr) in {
        'F106': (f115, F115W_PIVOT_A, f150, F150W_PIVOT_A),
        'F129': (f115, F115W_PIVOT_A, f150, F150W_PIVOT_A),
        'F158': (f150, F150W_PIVOT_A, f277, F277W_PIVOT_A),
    }.items():
        alpha = np.log(fr / fb) / np.log(lr / lb)
        expected = fb * (BAND_EFFECTIVE_LAMBDA_A[band] / lb) ** alpha
        np.testing.assert_allclose(
            out[f'flux_{band.lower()}_ujy'].to_numpy(float), expected, rtol=1e-10
        )


def write_fake_catalog(
    data_dir: Path, df: pd.DataFrame, name: str = 'cosmos25_fake'
) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / f'{name}.parquet'
    df.to_parquet(parquet_path, index=False)
    sidecar = {
        'name': name,
        'sha256': hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
    }
    (data_dir / f'{name}.provenance.json').write_text(json.dumps(sidecar))
    return parquet_path


def catalog_spec_dict(data_dir: Path, **population_overrides) -> dict:
    """A full catalog-mode ensemble spec dict pointing at a fake cosmos25."""
    d = {
        'run': {
            'name': 'pop_test_cosmos25',
            'version': 1,
            'description': 'cosmos25 population unit test',
            'seed': 778,
            'measurement': 'shear_bias',
            'noise_reps': 1,
        },
        'population': {
            'type': 'catalog',
            'catalog': {
                'kind': 'cosmos25',
                'download': 'cosmos25_fake',
                'data_dir': str(data_dir),
            },
            'preprocess': {'flux_variant': 'as_delivered', 'h': 0.70},
            'selection': {
                'z_range': [0.55, 1.9],
                'snr_line_total_min': 0.5,
                'bulge_fraction_max': None,
                'bulge_nsersic_range': None,
                'min_r50_over_psf_fwhm': None,
            },
            'sample': {'n_galaxies': 200, 'replace': False},
            'paint': {
                'bulge': False,
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
        # no snr block: catalog mode takes both channels' per-galaxy SNRs
        # from the population table
        'observation': {'config': 'hlwas_medium'},
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


def preprocess_spec(flux_variant: str = 'as_delivered'):
    """Minimal spec stand-in for direct preprocess calls."""

    class _Spec:
        pass

    s = _Spec()
    s.flux_variant = flux_variant
    s.h = 0.70
    return s


@pytest.fixture(scope='module')
def fake_data_dir(tmp_path_factory) -> Path:
    data_dir = tmp_path_factory.mktemp('cosmos25_fake')
    write_fake_catalog(data_dir, fake_cosmos25_rows(n=400, seed=4321))
    return data_dir


@pytest.fixture(scope='module')
def raw(fake_data_dir) -> pd.DataFrame:
    return load_catalog(COSMOS25, 'cosmos25_fake', fake_data_dir)


# ==============================================================================
# Adapter identity + contract
# ==============================================================================


class TestAdapter:
    def test_registry_and_identity(self):
        assert COSMOS25.kind == 'cosmos25'
        assert COSMOS25.id_columns == ('id',)
        assert COSMOS25.has_bulge is False
        assert COSMOS25.has_observed_redshift is False

    def test_flux_variant_vocabulary(self):
        # single variant: the regenerated painting is used verbatim; the
        # dust/IMF correction variants were retired 2026-07-29
        assert COSMOS25.flux_variants == ('as_delivered',)

    @pytest.mark.parametrize('variant', COSMOS25.flux_variants)
    def test_preprocess_output_matches_contract(self, raw, variant):
        pre = COSMOS25.preprocess(raw, preprocess_spec(variant))
        validate_contract(COSMOS25, pre)
        assert len(pre) > 0
        for stage in (
            'not_galaxy',
            'warn_flag',
            'no_painted_line',
            'bad_structure',
            'bad_mass',
            'continuum_nonpositive',
        ):
            assert stage in pre.attrs['kills']

    def test_kill_counts_match_construction(self, raw):
        pre = COSMOS25.preprocess(raw, preprocess_spec())
        kills = pre.attrs['kills']
        assert kills['not_galaxy'] == int((raw['type'] != 0).sum())
        galaxies = raw.loc[raw['type'] == 0]
        assert kills['warn_flag'] == int((galaxies['warn_flag'] != 0).sum())
        survivors = galaxies.loc[galaxies['warn_flag'] == 0]
        assert kills['no_painted_line'] == int(
            (~(np.isfinite(survivors['F_Ha']) & (survivors['F_Ha'] > 0))).sum()
        )


# ==============================================================================
# Flux-variant semantics
# ==============================================================================


class TestFluxVariants:
    def test_flux_used_verbatim(self, raw):
        pre = COSMOS25.preprocess(raw, preprocess_spec('as_delivered'))
        painted = raw.set_index('id').loc[pre['id'].to_numpy(), 'F_Ha'].to_numpy()
        np.testing.assert_array_equal(pre['f_line_cgs'].to_numpy(), painted)

    @pytest.mark.parametrize('retired', ['dust_fixed', 'dust_imf_fixed'])
    def test_retired_variants_raise_at_adapter(self, raw, retired):
        # the corrections moved upstream into the regenerated painting; a
        # spec still asking for them must fail loudly, not silently
        # double-correct
        with pytest.raises(ValueError, match='flux_variant'):
            COSMOS25.preprocess(raw, preprocess_spec(retired))

    def test_retired_variant_rejected_at_spec_validation(self, fake_data_dir, tmp_path):
        d = catalog_spec_dict(fake_data_dir)
        d['population']['preprocess']['flux_variant'] = 'dust_imf_fixed'
        with pytest.raises(ValueError, match="as_delivered"):
            spec_from_dict(tmp_path, d)

    def test_ebv_sentinels_do_not_matter(self):
        # as_delivered never touches ebv, so sentinel values must not raise
        df = fake_cosmos25_rows(n=50, seed=7)
        bad = df.index[(df['type'] == 0) & np.isfinite(df['F_Ha'])][0]
        df.loc[bad, 'ebv_minchi2'] = -99.0
        COSMOS25.preprocess(df, preprocess_spec('as_delivered'))


# ==============================================================================
# Continuum + derived-column physics
# ==============================================================================


class TestContinuum:
    def _one_row(self, z: float) -> pd.DataFrame:
        df = fake_cosmos25_rows(n=8, seed=11)
        df['zfinal'] = z
        df['redshift'] = z
        df['lambda_Ha_obs'] = HALPHA_REST_A * (1.0 + z)
        df['lambda_OII_obs'] = 3727.5 * (1.0 + z)
        df['lambda_OIII_obs'] = 5006.8 * (1.0 + z)
        df['type'] = 0
        df['warn_flag'] = 0
        df['F_Ha'] = 2.0e-16
        return df

    def test_ew_hand_round_trip(self):
        z = 1.0  # lambda_obs = 13125.6 A -> F150W branch
        df = self._one_row(z)
        pre = COSMOS25.preprocess(df, preprocess_spec('as_delivered'))
        lam = HALPHA_REST_A * (1.0 + z)
        f_nu = df['flux_model_f150w'].to_numpy()[0] * UJY_TO_CGS
        f_lambda = f_nu * C_A_PER_S / lam**2
        expected_ew = 2.0e-16 / f_lambda / (1.0 + z)
        assert pre['ew_rest_a'].to_numpy()[0] == pytest.approx(expected_ew, rel=1e-12)

    def test_band_assignment_edges(self):
        z_115 = F115W_F150W_EDGE_A / HALPHA_REST_A - 1.0 - 0.01
        z_150 = F115W_F150W_EDGE_A / HALPHA_REST_A - 1.0 + 0.01
        pre_115 = COSMOS25.preprocess(
            self._one_row(z_115), preprocess_spec('as_delivered')
        )
        pre_150 = COSMOS25.preprocess(
            self._one_row(z_150), preprocess_spec('as_delivered')
        )
        df = self._one_row(z_115)
        lam_115 = HALPHA_REST_A * (1.0 + z_115)
        f_nu_115 = df['flux_model_f115w'].to_numpy() * UJY_TO_CGS
        np.testing.assert_allclose(
            pre_115['f_lambda_cont_cgs'],
            f_nu_115 * C_A_PER_S / lam_115**2,
            rtol=1e-12,
        )
        lam_150 = HALPHA_REST_A * (1.0 + z_150)
        f_nu_150 = df['flux_model_f150w'].to_numpy() * UJY_TO_CGS
        np.testing.assert_allclose(
            pre_150['f_lambda_cont_cgs'],
            f_nu_150 * C_A_PER_S / lam_150**2,
            rtol=1e-12,
        )

    def test_gap_power_law_interpolation(self):
        z = 1.8  # lambda_obs = 18375.8 A > F150W red edge
        lam = HALPHA_REST_A * (1.0 + z)
        assert lam > F150W_RED_EDGE_A
        df = self._one_row(z)
        pre = COSMOS25.preprocess(df, preprocess_spec('as_delivered'))
        f150 = df['flux_model_f150w'].to_numpy()
        f277 = df['flux_model_f277w'].to_numpy()
        alpha = np.log(f277 / f150) / np.log(F277W_PIVOT_A / F150W_PIVOT_A)
        f_nu = f150 * (lam / F150W_PIVOT_A) ** alpha * UJY_TO_CGS
        np.testing.assert_allclose(
            pre['f_lambda_cont_cgs'], f_nu * C_A_PER_S / lam**2, rtol=1e-12
        )

    def test_rscale_is_r50_over_1p678(self, raw):
        pre = COSMOS25.preprocess(raw, preprocess_spec())
        np.testing.assert_allclose(
            pre['rscale_arcsec'],
            pre['disk_r50'].to_numpy() / R50_OVER_RSCALE,
            rtol=1e-12,
        )

    def test_lambda_z_mismatch_raises(self):
        df = self._one_row(1.0)
        df['lambda_Ha_obs'] = df['lambda_Ha_obs'] * 1.01
        with pytest.raises(ValueError, match='lambda_Ha_obs'):
            COSMOS25.preprocess(df, preprocess_spec('as_delivered'))


# ==============================================================================
# End-to-end population chain
# ==============================================================================


class TestBuildPopulation:
    def test_disk_only_build(self, fake_data_dir, tmp_path):
        spec = spec_from_dict(tmp_path, catalog_spec_dict(fake_data_dir))
        pop, meta = build_population(spec)
        assert len(pop) == 200
        # validation columns carried through under their output names
        assert 'catalog_axis_ratio' in pop.columns
        assert 'catalog_sersic_n' in pop.columns
        # no bulge columns in a disk-only, bulge-free-catalog build
        assert not any(c.startswith('bulge_') for c in pop.columns)

    def test_bulge_paint_rejected(self, fake_data_dir, tmp_path):
        d = catalog_spec_dict(fake_data_dir)
        d['population']['paint']['bulge'] = True
        spec = spec_from_dict(tmp_path, d)
        with pytest.raises(ValueError, match='paint.bulge'):
            build_population(spec)

    def test_bulge_fraction_cut_rejected(self, fake_data_dir, tmp_path):
        d = catalog_spec_dict(fake_data_dir)
        d['population']['selection']['bulge_fraction_max'] = 0.3
        spec = spec_from_dict(tmp_path, d)
        with pytest.raises(ValueError, match='bulge_fraction_max'):
            build_population(spec)

    def test_determinism_exact(self, fake_data_dir, tmp_path):
        spec = spec_from_dict(tmp_path, catalog_spec_dict(fake_data_dir))
        pop1, _ = build_population(spec)
        pop2, _ = build_population(spec)
        pd.testing.assert_frame_equal(pop1, pop2, check_exact=True)

    def test_dev_spec_parses(self):
        repo = Path(__file__).resolve().parent.parent
        spec = EnsembleSpec.from_yaml(
            repo / 'configs' / 'ensembles' / 'cosmos25_shear_dev.yaml'
        )
        cp = spec.catalog_population
        assert cp.catalog_kind == 'cosmos25'
        assert cp.flux_variant == 'as_delivered'
        assert cp.paint_bulge is False
        assert cp.bulge_fraction_max is None


class TestPriorProvenance:
    def test_registry_cites_the_cosmos25_catalog(self):
        from kl_pipe.ensemble.prior_provenance import catalog_registry
        from kl_pipe.ensemble.spec import ObservationConfig

        repo = Path(__file__).resolve().parent.parent
        spec = EnsembleSpec.from_yaml(
            repo / 'configs' / 'ensembles' / 'cosmos25_shear_dev.yaml'
        )
        config = ObservationConfig.from_yaml(
            repo / 'configs' / 'observation' / 'hlwas_medium_roman.yaml'
        )
        reg = catalog_registry(spec, config)
        assert len(reg) > 0
        used = {k for e in reg.values() for k in e.bibkeys}
        # catalog-sourced entries must cite this catalog, not Flagship2
        assert 'Castander2025' not in used
        assert 'Shuntov2025' in used
        # the redshift note reflects a photo-z truth, not grism spectroscopy
        assert 'photometric redshift' in reg['z'].notes
        # catalog-fit entries carry the measured cosmos25 constants
        # (n=294 census-selected sample of the regenerated catalog,
        # 2026-07-29: rscale log10 mu -0.774 -> median 0.17, sigma 0.161)
        assert 'TLN(0.17, 0.161' in reg[f'{config.bands[0]}.rscale'].fit_prior
