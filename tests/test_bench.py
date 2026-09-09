"""
Tests for the sampler benchmark record system (kl_pipe.ensemble.bench).

A tiny synthetic run directory (manifest + results parquet + provenance) is
built in tmp_path; no sampler runs.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from kl_pipe.ensemble import bench

N_CHAINS = 4
N_SAMPLES = 300
DRAWS = N_CHAINS * N_SAMPLES

# 4 fits: 2 galaxies x ring pair. fit 1 escalated and rescued, fit 3 escalated
# and still failing the gate, fits 0 and 2 clean.
FITS = [
    dict(fit_id='f0', galaxy_id=0, ring_member=0, noise_seed=11),
    dict(fit_id='f1', galaxy_id=0, ring_member=90, noise_seed=12),
    dict(fit_id='f2', galaxy_id=1, ring_member=0, noise_seed=13),
    dict(fit_id='f3', galaxy_id=1, ring_member=90, noise_seed=14),
]
ESCALATED = [False, True, False, True]
FIRST_RHAT = [np.nan, 1.08, np.nan, 1.20]
FIRST_ESS = [np.nan, 40.0, np.nan, 10.0]
MAX_RHAT = [1.02, 1.01, 1.03, 1.08]
MIN_ESS = [200.0, 400.0, 100.0, 30.0]
STEPS = [60.0 * DRAWS, 300000.0, 80.0 * DRAWS, 250000.0]
WALL = [600.0, 2000.0, 800.0, 2500.0]
THETA_MEAN = [0.5, 1.0, 6.2, 2.0]


def _results_frame(theta_shift: float = 0.0, mode_column: bool = True) -> pd.DataFrame:
    rows = []
    for i, f in enumerate(FITS):
        row = {
            'fit_id': f['fit_id'],
            'status': 'succeeded',
            'escalated': ESCALATED[i],
            'first_attempt_max_rhat': FIRST_RHAT[i],
            'first_attempt_min_ess': FIRST_ESS[i],
            'max_rhat': MAX_RHAT[i],
            'min_ess': MIN_ESS[i],
            'min_ess_param': 'theta_int',
            'num_steps_total': STEPS[i],
            'fit_wallclock_s': WALL[i],
        }
        if mode_column:
            row['escalation_mode'] = 'restart' if ESCALATED[i] else ''
        for p in ('g1', 'g2', 'cosi', 'vel.vcirc'):
            row[f'post.{p}.mean'] = 0.1 * (i + 1)
            row[f'post.{p}.std'] = 0.02
        row['post.theta_int.mean'] = THETA_MEAN[i] + theta_shift
        row['post.theta_int.std'] = 0.2
        rows.append(row)
    return pd.DataFrame(rows)


def _spec_dict(name: str = 'bench_synthetic') -> dict:
    return {
        'run': {'name': name},
        'fit': {
            'n_warmup': 200,
            'n_samples': N_SAMPLES,
            'n_chains': N_CHAINS,
            'escalation': {'enabled': True, 'rhat_max': 1.05, 'ess_min': 50},
        },
    }


def make_run_dir(
    root: Path,
    name: str = 'bench_synthetic',
    provenance: bool = True,
    theta_shift: float = 0.0,
    mode_column: bool = True,
    fit_id_prefix: str = '',
) -> Path:
    run_dir = root / name
    run_dir.mkdir()
    manifest = pd.DataFrame(FITS)
    manifest['fit_id'] = fit_id_prefix + manifest['fit_id']
    manifest.to_parquet(run_dir / 'manifest.parquet', index=False)
    results = _results_frame(theta_shift=theta_shift, mode_column=mode_column)
    results['fit_id'] = fit_id_prefix + results['fit_id']
    results.to_parquet(run_dir / 'results.parquet', index=False)
    if provenance:
        prov = run_dir / 'provenance'
        prov.mkdir()
        (prov / 'ensemble_spec.yaml').write_text(yaml.safe_dump(_spec_dict(name)))
        (prov / 'expansion.json').write_text(
            json.dumps(
                {'run_name': name, 'git_commit': 'abcdef0123456789', 'n_fits': 4}
            )
        )
    return run_dir


@pytest.fixture
def run_dir(tmp_path):
    return make_run_dir(tmp_path)


def _fits(run_dir: Path) -> pd.DataFrame:
    return bench.prepare_fits(bench.load_results(run_dir), bench.load_manifest(run_dir))


# ==============================================================================
# Metrics
# ==============================================================================


class TestMetrics:
    def test_known_counts(self, run_dir):
        m = bench.compute_metrics(
            _fits(run_dir), N_CHAINS, N_SAMPLES, run_wall_s=5000.0
        )
        assert m['n_fits'] == 4
        assert m['first_pass_fail'] == 2
        assert m['final_fail'] == 1
        assert m['catastrophic'] == 0
        assert m['escalations_by_mode'] == {'restart': 2}
        # first-pass gate values: clean fits use their final values
        assert m['median_first_attempt_max_rhat'] == pytest.approx(
            np.median([1.02, 1.08, 1.03, 1.20])
        )
        assert m['median_first_attempt_min_ess'] == pytest.approx(
            np.median([200.0, 40.0, 100.0, 10.0])
        )
        assert m['median_worst_tau'] == pytest.approx(
            np.median(DRAWS / np.array([200.0, 40.0, 100.0, 10.0]))
        )
        assert m['steps_per_draw_median_clean'] == pytest.approx(70.0)
        assert m['fit_wall_median_clean_s'] == pytest.approx(700.0)
        assert m['fit_wall_max_s'] == pytest.approx(2500.0)
        assert m['fit_wall_sum_s'] == pytest.approx(5900.0)
        assert m['run_wall_s'] == pytest.approx(5000.0)

    def test_missing_escalation_mode_column_means_restart(self, tmp_path):
        run_dir = make_run_dir(tmp_path, mode_column=False)
        fits = _fits(run_dir)
        assert fits['escalation_mode'].tolist() == ['', 'restart', '', 'restart']

    def test_catastrophic_counted_and_excluded_from_medians(self, run_dir):
        results = bench.load_results(run_dir)
        results.loc[0, 'status'] = 'failed'
        results.loc[0, ['max_rhat', 'min_ess', 'num_steps_total']] = np.nan
        fits = bench.prepare_fits(results, bench.load_manifest(run_dir))
        m = bench.compute_metrics(fits, N_CHAINS, N_SAMPLES)
        assert m['catastrophic'] == 1
        assert m['n_fits'] == 4
        assert m['steps_per_draw_median_clean'] == pytest.approx(80.0)

    def test_nonfinite_required_column_on_succeeded_fit_raises(self, run_dir):
        results = bench.load_results(run_dir)
        results.loc[2, 'num_steps_total'] = np.nan
        with pytest.raises(ValueError, match='non-finite num_steps_total'):
            bench.prepare_fits(results, bench.load_manifest(run_dir))

    def test_escalated_fit_without_first_attempt_values_raises(self, run_dir):
        results = bench.load_results(run_dir)
        results.loc[1, 'first_attempt_max_rhat'] = np.nan
        with pytest.raises(ValueError, match='first-attempt gate values'):
            bench.prepare_fits(results, bench.load_manifest(run_dir))


# ==============================================================================
# Agreement vs reference
# ==============================================================================


class TestAgreement:
    def test_wrapped_theta_and_matching(self, tmp_path):
        arm = make_run_dir(tmp_path, name='arm', fit_id_prefix='a_')
        # reference shifted by 2 pi - 0.1 in theta: unwrapped |d| ~ 6.2 rad,
        # wrapped |d| = 0.1 rad = 0.5 sigma_ref
        ref = make_run_dir(
            tmp_path, name='ref', theta_shift=2 * np.pi - 0.1, fit_id_prefix='r_'
        )
        fits_arm, fits_ref = _fits(arm), _fits(ref)
        fits_ref.loc[fits_ref['fit_id'] == 'r_f1', 'escalated'] = False
        fits_ref.loc[fits_ref['fit_id'] == 'r_f3', 'final_fail'] = False
        agreement = bench.compute_agreement(fits_arm, fits_ref)
        assert agreement['n_matched'] == 4
        assert agreement['n_unmatched_arm'] == 0
        theta = agreement['params']['theta_int']
        assert theta['median_abs_dmean_over_sigma_ref'] == pytest.approx(0.5)
        assert theta['max_abs_dmean_over_sigma_ref'] == pytest.approx(0.5)
        assert theta['median_width_ratio'] == pytest.approx(1.0)
        assert agreement['params']['g1']['max_abs_dmean_over_sigma_ref'] == 0.0
        assert agreement['first_pass_fail_only_arm'] == ['g0_r90_s12']
        assert agreement['first_pass_fail_only_ref'] == []
        assert agreement['final_fail_only_arm'] == ['g1_r90_s14']

    def test_no_match_raises(self, tmp_path):
        arm = make_run_dir(tmp_path, name='arm')
        fits_arm = _fits(arm)
        fits_ref = fits_arm.copy()
        fits_ref['match_key'] = 'other_' + fits_ref['match_key']
        with pytest.raises(ValueError, match='no fits match'):
            bench.compute_agreement(fits_arm, fits_ref)

    def test_wrap_angle(self):
        assert bench.wrap_angle(2 * np.pi - 0.1) == pytest.approx(-0.1)
        assert bench.wrap_angle(0.3) == pytest.approx(0.3)
        assert bench.wrap_angle(-np.pi - 0.2) == pytest.approx(np.pi - 0.2)


# ==============================================================================
# Records
# ==============================================================================


def _record(run_dir: Path, **kwargs) -> dict:
    defaults = dict(
        job_id='123',
        hypothesis='h',
        reason='r',
        conclusion='c',
        date='2026-01-02',
    )
    defaults.update(kwargs)
    return bench.build_record(run_dir, **defaults)


class TestRecords:
    def test_write_read_round_trip(self, run_dir, tmp_path):
        out = tmp_path / 'bench' / 'runs'
        record = _record(run_dir)
        assert record['commit'] == 'abcdef0'
        assert record['arm'] == 'bench_synthetic'
        assert record['sampler'] == {
            'n_warmup': 200,
            'n_samples': N_SAMPLES,
            'n_chains': N_CHAINS,
        }
        assert record['spec_sha256'] == bench.sha256_of(
            run_dir / 'provenance' / 'ensemble_spec.yaml'
        )
        assert len(record['per_fit']) == 4
        assert record['per_fit'][0]['first_attempt_max_rhat'] == 1.02
        assert record['per_fit'][1]['first_attempt_max_rhat'] == 1.08
        path = bench.write_record(record, run_dir / 'results.parquet', out)
        assert path == out / '2026-01-02_123_bench_synthetic.json'
        assert (out / '2026-01-02_123_bench_synthetic.parquet').exists()
        assert bench.load_record(path) == record
        # strict JSON: no NaN tokens
        assert 'NaN' not in path.read_text()
        with pytest.raises(FileExistsError):
            bench.write_record(record, run_dir / 'results.parquet', out)
        df = bench.load_records(out.parent)
        assert len(df) == 1
        assert df.loc[0, 'first_pass_fail'] == 2
        assert df.loc[0, 'job_id'] == '123'

    def test_record_with_reference(self, tmp_path):
        arm = make_run_dir(tmp_path, name='arm', fit_id_prefix='a_')
        ref = make_run_dir(tmp_path, name='ref', fit_id_prefix='r_')
        ref_record = _record(ref, job_id='1')
        record = _record(arm, job_id='2', reference=ref_record)
        assert record['reference_arm'] == 'ref'
        assert record['reference_job_id'] == '1'
        agreement = record['metrics']['agreement_vs_reference']
        assert agreement['n_matched'] == 4
        assert agreement['params']['cosi']['max_abs_dmean_over_sigma_ref'] == 0.0

    def test_explicit_commit_overrides_provenance(self, run_dir):
        record = _record(run_dir, commit='1234567')
        assert record['commit'] == '1234567'
        assert record['expansion_commit'] == 'abcdef0'

    def test_annotate(self, run_dir, tmp_path):
        out = tmp_path / 'runs'
        path = bench.write_record(
            _record(run_dir, conclusion=''), run_dir / 'results.parquet', out
        )
        assert bench.load_record(path)['conclusion'] == ''
        bench.annotate(path, 'done')
        assert bench.load_record(path)['conclusion'] == 'done'
        with pytest.raises(ValueError):
            bench.annotate(path, '')

    def test_missing_results_raises(self, run_dir):
        (run_dir / 'results.parquet').unlink()
        with pytest.raises(FileNotFoundError, match='no results'):
            bench.load_results(run_dir)
        with pytest.raises(FileNotFoundError, match='results parquet not found'):
            bench.load_results(run_dir, run_dir / 'nope.parquet')

    def test_missing_commit_without_provenance_raises(self, tmp_path):
        run_dir = make_run_dir(tmp_path, provenance=False)
        spec = tmp_path / 'spec.yaml'
        spec.write_text(yaml.safe_dump(_spec_dict()))
        with pytest.raises(ValueError, match='no provenance'):
            _record(run_dir, spec_path=spec)
        with pytest.raises(ValueError, match='no provenance'):
            _record(run_dir, commit='1234567')
        record = _record(run_dir, commit='1234567', spec_path=spec)
        assert record['commit'] == '1234567'
        assert record['expansion_commit'] is None

    def test_missing_required_column_raises(self, run_dir):
        results = bench.load_results(run_dir).drop(columns=['min_ess_param'])
        with pytest.raises(KeyError, match='min_ess_param'):
            bench.prepare_fits(results, bench.load_manifest(run_dir))

    def test_results_manifest_count_mismatch_raises(self, run_dir):
        results = bench.load_results(run_dir).iloc[:3]
        results.to_parquet(run_dir / 'results.parquet', index=False)
        with pytest.raises(ValueError, match='provenance expects 4'):
            _record(run_dir)


# ==============================================================================
# Index
# ==============================================================================


class TestIndex:
    def test_index_regeneration_idempotent(self, tmp_path):
        bench_dir = tmp_path / 'bench'
        out = bench_dir / 'runs'
        a = make_run_dir(tmp_path, name='arm_a', fit_id_prefix='a_')
        b = make_run_dir(tmp_path, name='arm_b', fit_id_prefix='b_')
        bench.write_record(
            _record(a, job_id='2', date='2026-01-03', conclusion='x' * 200),
            a / 'results.parquet',
            out,
        )
        bench.write_record(
            _record(b, job_id='1', date='2026-01-02', conclusion='has | pipe'),
            b / 'results.parquet',
            out,
        )
        readme = bench_dir / 'README.md'
        header = 'header\n\n<!-- index:start -->\nstale\n<!-- index:end -->\ntail\n'
        readme.write_text(header)
        bench.write_index(bench_dir)
        first = readme.read_text()
        bench.write_index(bench_dir)
        assert readme.read_text() == first
        assert first.startswith('header\n\n<!-- index:start -->\n| date |')
        assert first.endswith('<!-- index:end -->\ntail\n')
        assert 'stale' not in first
        lines = [l for l in first.splitlines() if l.startswith('| 2026')]
        assert len(lines) == 2
        # sorted by date; pipes escaped; long conclusion truncated
        assert lines[0].startswith('| 2026-01-02 | arm_b | 1 | abcdef0 | 2/4 | 1/4 |')
        assert 'has / pipe' in lines[0]
        assert lines[1].endswith('...' + ' |')
        assert len(lines[1].split('|')[-2].strip()) == 80

    def test_index_requires_markers(self, tmp_path):
        bench_dir = tmp_path / 'bench'
        out = bench_dir / 'runs'
        a = make_run_dir(tmp_path)
        bench.write_record(_record(a), a / 'results.parquet', out)
        with pytest.raises(FileNotFoundError):
            bench.write_index(bench_dir)
        (bench_dir / 'README.md').write_text('no markers\n')
        with pytest.raises(ValueError, match='exactly one'):
            bench.write_index(bench_dir)


# ==============================================================================
# CLI
# ==============================================================================


class TestCLI:
    def test_record_annotate_index(self, run_dir, tmp_path, capsys):
        bench_dir = tmp_path / 'bench'
        bench_dir.mkdir()
        (bench_dir / 'README.md').write_text(
            '<!-- index:start -->\n<!-- index:end -->\n'
        )
        rc = bench.main(
            [
                'record',
                '--run-dir',
                str(run_dir),
                '--job-id',
                '77',
                '--date',
                '2026-01-02',
                '--hypothesis',
                'h',
                '--reason',
                'r',
                '--run-wall',
                '100',
                '--out',
                str(bench_dir / 'runs'),
            ]
        )
        assert rc == 0
        record_path = bench_dir / 'runs' / '2026-01-02_77_bench_synthetic.json'
        assert record_path.exists()
        assert bench.load_record(record_path)['metrics']['run_wall_s'] == 100.0
        assert bench.main(['annotate', str(record_path), '--conclusion', 'ok']) == 0
        assert bench.load_record(record_path)['conclusion'] == 'ok'
        assert bench.main(['index', '--dir', str(bench_dir)]) == 0
        assert '| 77 |' in (bench_dir / 'README.md').read_text()
        out = capsys.readouterr().out
        assert 'first-pass fail 2/4' in out

    def test_compare_prints_without_writing(self, tmp_path, capsys):
        a = make_run_dir(tmp_path, name='arm_a', fit_id_prefix='a_')
        b = make_run_dir(tmp_path, name='arm_b', fit_id_prefix='b_')
        before = sorted(p.name for p in tmp_path.rglob('*'))
        rc = bench.main(['compare', '--run-dir', str(a), '--run-dir-ref', str(b)])
        assert rc == 0
        assert sorted(p.name for p in tmp_path.rglob('*')) == before
        out = capsys.readouterr().out
        assert 'first_pass_fail' in out
        assert 'matched 4 fits' in out
