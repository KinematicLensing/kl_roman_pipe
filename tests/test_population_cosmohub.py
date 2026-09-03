"""
Real-data tier for the catalog population backend: runs against the
downloaded Flagship2 dev catalog (data/cosmohub/flagship2_dev.parquet,
'make download-cosmohub-dev'). Marked cosmohub; skipped when the parquet
is absent.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from kl_pipe.ensemble.population import build_population
from kl_pipe.ensemble.spec import EnsembleSpec

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / 'data' / 'cosmohub'
DEV_PARQUET = DATA_DIR / 'flagship2_dev.parquet'
EXAMPLE_SPEC = REPO_ROOT / 'configs' / 'ensembles' / 'flagship2_shear_dev.yaml'

pytestmark = [
    pytest.mark.cosmohub,
    pytest.mark.roman_ensemble,
    pytest.mark.skipif(
        not DEV_PARQUET.exists(),
        reason="flagship2_dev.parquet absent (make download-cosmohub-dev)",
    ),
]


def _dev_spec(tmp_path: Path, n_galaxies: int = 100) -> EnsembleSpec:
    """Example spec with a real-data-sized selection (snr_line_total_min 5 keeps
    ~400 of the dev box's ~280k disks; 10 keeps only ~50)."""
    d = yaml.safe_load(EXAMPLE_SPEC.read_text())
    d['population']['selection']['snr_line_total_min'] = 5.0
    d['population']['sample']['n_galaxies'] = n_galaxies
    d['population']['catalog']['data_dir'] = str(DATA_DIR)
    path = tmp_path / 'spec.yaml'
    path.write_text(yaml.safe_dump(d))
    return EnsembleSpec.from_yaml(path)


@pytest.fixture(scope='module')
def real_build(tmp_path_factory):
    spec = _dev_spec(tmp_path_factory.mktemp('spec'))
    return build_population(spec)


class TestRealCatalog:
    def test_all_finite_and_counts(self, real_build):
        df, meta = real_build
        assert len(df) == 100
        numeric = df.drop(columns=['pop_index', 'galaxy_id', 'halo_id'])
        assert np.isfinite(numeric.to_numpy(dtype=np.float64)).all()
        assert meta['n_raw'] == 288321
        assert meta['n_disk'] == meta['n_raw'] - meta['kills']['one_component']
        assert meta['n_selected'] >= 100
        print(
            f"stage counts: n_raw={meta['n_raw']} -> n_disk={meta['n_disk']} "
            f"-> n_selected={meta['n_selected']} -> n_sampled={meta['n_sampled']}"
        )
        print(f"kills: {meta['kills']}")
        print(
            f"medians: ew_rest={df['ew_rest_a'].median():.1f} A, "
            f"snr_line_total={df['snr_line_total'].median():.1f}, "
            f"vcirc={df['vcirc_kms'].median():.1f} km/s"
        )

    def test_determinism_double_build(self, real_build, tmp_path):
        df1, _ = real_build
        df2, _ = build_population(_dev_spec(tmp_path))
        pd.testing.assert_frame_equal(df1, df2, check_exact=True)

    def test_ew_median_physical_window(self, real_build):
        # loose physical window; provenance: gate-sample first-look median
        # 59 A over the full dev preprocess (2026-07), rising to ~90 A
        # under the snr >= 5 selection
        df, _ = real_build
        assert 30.0 <= df['ew_rest_a'].median() <= 120.0

    def test_physical_ranges(self, real_build):
        df, _ = real_build
        assert df['z'].between(0.55, 1.9).all()
        assert (df['snr_line_total'] >= 5.0).all()
        assert df['cosi'].between(0.05, 0.95).all()
        assert (df['vcirc_kms'] > 0).all()
        assert (df['sigma0_kms'] >= 5.0).all()
        assert (np.hypot(df['g1'], df['g2']) < 0.1).all()
