"""Static run dashboard: renders on a minimal run dir and degrades gracefully."""

import json
import time

import pandas as pd
import pytest

from kl_pipe.ensemble.dashboard import (
    GLOSSARY,
    NOT_AVAILABLE,
    _cell_class,
    _fmt,
    _repo_url,
    build_dashboard,
    speed_cell_classes,
    speed_table,
)
from test_bench import make_run_dir

SECTIONS = (
    'Progress',
    'Speed by galaxy property',
    'Failures and escalations',
    'Flags',
    'Early science',
    'Plots',
    'Notes',
    'Glossary',
)


def _add_status(run_dir, fit_ids, job='123'):
    now = time.time()
    for i, fid in enumerate(fit_ids):
        claim = run_dir / 'status' / 'claims' / fid
        claim.mkdir(parents=True)
        (claim / 'claim.json').write_text(
            json.dumps(
                {
                    'backend': 'slurm',
                    'hostname': 'node',
                    'pid': 1,
                    'slurm_job_id': job,
                    'ts': now - 3600 + 60 * i,
                }
            )
        )
        done = run_dir / 'status' / 'done'
        done.mkdir(exist_ok=True)
        (done / fid).write_text(json.dumps({'ts': now - 3000 + 60 * i}))


def test_dashboard_renders_minimal_run_dir(tmp_path):
    run_dir = make_run_dir(tmp_path)
    fit_ids = list(pd.read_parquet(run_dir / 'manifest.parquet')['fit_id'])
    _add_status(run_dir, fit_ids)
    # an older-style run: commit inferred from the job log first line
    (run_dir / 'prod_base_123.out').write_text('abcdef012 some commit subject\nmore\n')
    out = build_dashboard(run_dir)
    assert out == run_dir / 'diagnostics' / 'dashboard.html'
    text = out.read_text()
    for heading in SECTIONS:
        assert f'>{heading}</h2>' in text
        assert f'href="#' in text
    assert text.count('<details open') == text.count('</summary>')
    assert 'Escalated fits: 2 of 4' in text
    assert 'fits_per_node_hr' in text
    assert 'abcdef012' in text and '(inferred from job log)' in text
    # every glossary anchor referenced from a header exists in the page
    for key in GLOSSARY:
        assert f'id="g-{key}"' in text
    assert 'href="#g-max_rhat"' in text


def test_dashboard_degrades_without_optional_inputs(tmp_path):
    # no status dir, no provenance, no diagnostics, no truth/interval columns
    run_dir = make_run_dir(tmp_path, provenance=False)
    text = build_dashboard(run_dir).read_text()
    assert NOT_AVAILABLE in text
    assert 'no notes' in text
    for heading in SECTIONS:
        assert f'>{heading}</h2>' in text


def test_dashboard_missing_manifest_raises(tmp_path):
    (tmp_path / 'empty_run').mkdir()
    with pytest.raises(FileNotFoundError):
        build_dashboard(tmp_path / 'empty_run')
    with pytest.raises(FileNotFoundError):
        build_dashboard(tmp_path / 'does_not_exist')


def test_formatter_and_cell_classes():
    assert _fmt('pull_mean', 0.123456) == '0.12'
    assert _fmt('max_rhat', 1.04567) == '1.046'
    assert _fmt('min_ess', 123.7) == '124'
    assert _fmt('wall_min', 12.345) == '12.3'
    assert _fmt('sigma_med', 0.10234) == '0.102'
    assert _fmt('truth.cosi', 0.4567) == '0.46'
    assert _fmt('line_snr', 13.6) == '14'
    assert _fmt('max_rhat', float('nan')) == '-'
    row = pd.Series(dtype=float)
    assert _cell_class('pull_mean', 1.0, row) == 'good'
    assert _cell_class('pull_mean', -2.5, row) == 'warn'
    assert _cell_class('pull.g1', 3.5, row) == 'crit'
    assert _cell_class('max_rhat', 1.06, row) == 'crit'
    assert _cell_class('min_ess', 20.0, row) == 'crit'
    assert _cell_class('divergence_rate', 0.02, row) == 'crit'
    assert _cell_class('max_rhat', float('nan'), row) == 'crit'
    assert _cell_class('final_max_rhat', 1.02, row) == 'good'
    assert _cell_class('final_max_rhat', 1.06, row) == 'crit'
    assert _cell_class('final_min_ess', 120.0, row) == 'good'
    assert _cell_class('final_min_ess', 30.0, row) == 'crit'
    assert _cell_class('gate_pass', True, row) == 'good'
    assert _cell_class('gate_pass', False, row) == 'crit'
    assert _fmt('final_max_rhat', 1.04567) == '1.046'
    assert _fmt('final_min_ess', 123.7) == '124'
    assert _cell_class('fit_id', 'abc', row) == ''


def test_repo_url_is_https_or_none():
    url = _repo_url()
    assert url is None or url.startswith('https://')


def test_derived_shear_rotation_identities():
    import numpy as np

    from kl_pipe.ensemble.dashboard import derived_shear, gate_table

    g1, g2 = np.array([0.03, -0.01]), np.array([0.02, 0.05])
    gp, gx = derived_shear(g1, g2, np.zeros(2))
    np.testing.assert_allclose(gp, g1)
    np.testing.assert_allclose(gx, g2)
    gp, gx = derived_shear(g1, g2, np.full(2, np.pi / 4))
    np.testing.assert_allclose(gp, g2, atol=1e-15)
    np.testing.assert_allclose(gx, -g1, atol=1e-15)
    # rotation preserves |g|
    gp, gx = derived_shear(g1, g2, np.array([0.7, 2.1]))
    np.testing.assert_allclose(gp**2 + gx**2, g1**2 + g2**2)


def test_gate_table_counts_on_fixture(tmp_path):
    from kl_pipe.ensemble.dashboard import gate_table

    run_dir = make_run_dir(tmp_path)
    ok = pd.read_parquet(run_dir / 'results.parquet')
    # fixture: max_rhat [1.02, 1.01, 1.03, 1.08], min_ess [200, 400, 100, 30]
    final = gate_table(ok).set_index(['rhat_max', 'ess_min'])
    assert final.loc[(1.05, 50.0), 'n_fail'] == 1
    assert final.loc[(1.025, 50.0), 'n_fail'] == 2
    assert final.loc[(1.01, 100.0), 'n_fail'] == 3
    assert final.loc[(1.05, 200.0), 'n_fail'] == 2
    # first attempts: escalated fits had rhat 1.08 / 1.20 and ess 40 / 10
    first = gate_table(ok, first_attempt=True).set_index(['rhat_max', 'ess_min'])
    assert first.loc[(1.05, 50.0), 'n_fail'] == 2
    assert 'ess_g1' not in ok.columns and final['shear_ess_min'].isna().all()


def test_chain_derived_stats_and_ranks(tmp_path):
    import numpy as np

    from kl_pipe.ensemble.dashboard import _rank_mean_table, chain_derived_stats

    run_dir = make_run_dir(tmp_path)
    manifest = pd.read_parquet(run_dir / 'manifest.parquet')
    manifest['truth.theta_int'] = [0.0, np.pi / 4, 1.0, 2.0]
    manifest['truth.g1'] = 0.03
    manifest['truth.g2'] = -0.01
    manifest['truth.cosi'] = [0.1, 0.2, 0.8, 0.9]
    manifest.to_parquet(run_dir / 'manifest.parquet', index=False)
    (run_dir / 'chains').mkdir()
    rng = np.random.default_rng(1)
    # fit f0 (theta 0): g+ = g1 draws centred on truth -> rank ~0.5, inside 68%
    # fit f1 (theta pi/4): g+ = g2 draws shifted +3 sigma -> rank ~0, outside 95%
    for fid, shift in (('f0', 0.0), ('f1', 0.03)):
        g1 = 0.03 + 0.01 * rng.normal(size=200)
        g2 = -0.01 + shift + 0.01 * rng.normal(size=200)
        np.savez(
            run_dir / 'chains' / f'{fid}.npz',
            samples=np.column_stack([g1, g2]),
            param_names=np.array(['g1', 'g2']),
        )
    ok = pd.read_parquet(run_dir / 'results.parquet').merge(manifest, on='fit_id')
    chain = chain_derived_stats(run_dir, ok)
    assert list(chain['fit_id']) == ['f0', 'f1']
    r0 = chain.set_index('fit_id').loc['f0']
    r1 = chain.set_index('fit_id').loc['f1']
    assert 0.35 < r0['chain.g_plus.rank'] < 0.65 and r0['chain.g_plus.in68']
    assert r1['chain.g_plus.rank'] < 0.05 and not r1['chain.g_plus.in95']
    assert 0.35 < r1['chain.g_cross.rank'] < 0.65
    ranks = _rank_mean_table(ok, chain).set_index('subset')
    assert abs(ranks.loc['all', 'rank_mean.g_cross'] - 0.5) < 0.15
    text = build_dashboard(run_dir).read_text()
    assert '(from chains)' in text
    assert 'Derived g+, gx truth-rank histograms' in text
    # without chains the notes stay
    run2 = make_run_dir(tmp_path, name='no_chains')
    text2 = build_dashboard(run2).read_text()
    assert 'need a chains directory' in text2


def test_speed_table_shares_and_rates():
    ok = pd.DataFrame(
        {
            'fit_wallclock_s': [600.0, 600.0, 1800.0, 3600.0],
            'truth.cosi': [0.05, 0.2, 0.4, 0.8],
            'line_snr': [10.0, 20.0, 30.0, 40.0],
            'escalated': [False, False, False, True],
            'num_steps_total': [60e3, 60e3, 90e3, 180e3],
            'first_attempt_max_rhat': [1.01, 1.02, 1.03, 1.2],
            'first_attempt_min_ess': [200.0, 150.0, 90.0, 10.0],
            'ess_g1': [300.0, 250.0, 100.0, 60.0],
            'ess_g2': [280.0, 260.0, 120.0, 50.0],
        }
    )
    t = speed_table(ok, workers_per_node=8).set_index(['by', 'bin'])
    total = t.loc[('all', 'all')]
    assert total['n'] == 4
    assert abs(total['worker_h'] - (600 + 600 + 1800 + 3600) / 3600) < 1e-9
    assert abs(total['fits_per_worker_h'] - 4 / total['worker_h']) < 1e-9
    assert abs(total['fits_per_node_h'] - 8 * total['fits_per_worker_h']) < 1e-9
    face = t.loc[('truth cos i', '[0.70, 1.00)')]
    assert face['n'] == 1 and face['esc_%'] == 100.0 and face['fp_fail_frac'] == 1.0
    assert (
        list(t.columns).index('esc_%') == list(t.columns).index('fits_per_node_h') + 1
    )
    assert abs(face['frac_worker_h'] - 1.0 / total['worker_h']) < 1e-9
    assert face['shear_ess_med'] == 50.0
    cosi_rows = t.loc['truth cos i']
    assert abs(cosi_rows['frac_fits'].sum() - 1.0) < 1e-9
    assert abs(cosi_rows['frac_worker_h'].sum() - 1.0) < 1e-9
    assert t.loc[('escalation', 'first pass')]['n'] == 3
    assert speed_table(ok.iloc[:0]).empty
    assert (
        speed_table(ok).loc[0, 'fits_per_node_h']
        != speed_table(ok).loc[0, 'fits_per_node_h']
    )


def test_speed_cell_classes_relative_to_run():
    t = pd.DataFrame(
        {
            'by': ['all', 'truth cos i', 'truth cos i', 'truth cos i'],
            'bin': ['all', 'a', 'b', 'c'],
            'fits_per_worker_h': [2.0, 2.4, 2.0, 1.0],
            'fits_per_node_h': [16.0, 19.2, 16.0, 8.0],
            'esc_%': [10.0, 5.0, 10.0, 30.0],
        }
    )
    cls = speed_cell_classes(t)
    rows = [t.iloc[i] for i in range(4)]
    assert cls['fits_per_node_h'](16.0, rows[0]) == ''
    assert cls['fits_per_node_h'](19.2, rows[1]) == 'good'
    assert cls['fits_per_node_h'](16.0, rows[2]) == 'warn'
    assert cls['fits_per_node_h'](8.0, rows[3]) == 'crit'
    assert cls['esc_%'](5.0, rows[1]) == 'good'
    assert cls['esc_%'](30.0, rows[3]) == 'crit'
