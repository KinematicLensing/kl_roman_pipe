"""
Unit tests for the quality-gated escalation retry (fit.escalation).

Covers: spec knob validation, the convergence gate (quality-based, NOT
divergence-based), same-fit metric donation pass-through into the retry's
sampler config, summary-row recording (escalated / first_attempt_* /
total wallclock), chains-file retention rules, and backward compatibility
(no escalation block = the legacy catastrophic fresh-seed retry with no new
summary columns). The sampler itself is mocked at the ``_run_fit_attempt``
boundary -- these tests are fast and fit-free; the real donated-metric
sampler path is exercised in test_numpyro.py.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from kl_pipe.ensemble import worker
from kl_pipe.ensemble.expander import expand, load_run
from kl_pipe.ensemble.spec import EnsembleSpec, EscalationSpec
from kl_pipe.ensemble.worker import needs_escalation, run_single_fit
from kl_pipe.sampling.configs import NumpyroSamplerConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = REPO_ROOT / 'configs' / 'observation'
DEV_SPEC = REPO_ROOT / 'configs' / 'ensembles' / 'sigma_eps_cosi_dev.yaml'


def _spec_dict() -> dict:
    return yaml.safe_load(DEV_SPEC.read_text())


def _escalation_spec_dict() -> dict:
    """Dev spec with the escalation ladder enabled (adapt-mass prerequisite)."""
    d = _spec_dict()
    d['fit']['unconstrained'] = True
    d['fit']['adapt_mass'] = True
    d['fit']['escalation'] = {'enabled': True}
    # chains saved (retention rules under test); mocks skipped (not under test)
    d['output'] = {'save_chains': 'all', 'save_mocks': 'none'}
    return d


def _write_spec(tmp_path: Path, spec_dict: dict) -> Path:
    path = tmp_path / 'spec.yaml'
    path.write_text(yaml.safe_dump(spec_dict))
    return path


# ==============================================================================
# Spec knob validation
# ==============================================================================


class TestEscalationSpecValidation:
    def test_absent_block_disabled_with_defaults(self, tmp_path):
        spec = EnsembleSpec.from_yaml(_write_spec(tmp_path, _spec_dict()))
        assert spec.escalation == EscalationSpec()
        assert not spec.escalation.enabled
        # documented defaults (census-gate thresholds + escalation rung)
        assert spec.escalation.rhat_max == 1.05
        assert spec.escalation.ess_min == 50.0
        assert spec.escalation.n_warmup == 800
        assert spec.escalation.n_samples == 1000

    def test_enabled_block_parses(self, tmp_path):
        d = _escalation_spec_dict()
        d['fit']['escalation'].update(
            rhat_max=1.02, ess_min=100, n_warmup=1200, n_samples=2000
        )
        spec = EnsembleSpec.from_yaml(_write_spec(tmp_path, d))
        assert spec.escalation.enabled
        assert spec.escalation.rhat_max == 1.02
        assert spec.escalation.ess_min == 100.0
        assert spec.escalation.n_warmup == 1200
        assert spec.escalation.n_samples == 2000

    def test_unknown_key_rejected(self, tmp_path):
        d = _escalation_spec_dict()
        d['fit']['escalation']['bogus'] = 1
        with pytest.raises(ValueError, match='unknown keys'):
            EnsembleSpec.from_yaml(_write_spec(tmp_path, d))

    def test_enabled_allows_frozen_first_pass(self, tmp_path):
        # adapt_mass false + escalation is the frozen-first-pass tier: the
        # retry re-enables mass adaptation instead of donating a metric
        # (requires-adapt-mass validation deliberately relaxed 2026-07-30)
        d = _escalation_spec_dict()
        d['fit']['adapt_mass'] = False
        spec = EnsembleSpec.from_yaml(_write_spec(tmp_path, d))
        assert spec.escalation.enabled and spec.adapt_mass is False

    def test_enabled_requires_laplace(self, tmp_path):
        d = _escalation_spec_dict()
        d['fit']['precondition'] = 'none'
        with pytest.raises(ValueError, match='laplace'):
            EnsembleSpec.from_yaml(_write_spec(tmp_path, d))

    def test_enabled_must_be_boolean(self, tmp_path):
        d = _escalation_spec_dict()
        d['fit']['escalation']['enabled'] = 'yes'
        with pytest.raises(ValueError, match='enabled must be a boolean'):
            EnsembleSpec.from_yaml(_write_spec(tmp_path, d))

    def test_rhat_max_must_exceed_one(self, tmp_path):
        d = _escalation_spec_dict()
        d['fit']['escalation']['rhat_max'] = 1.0
        with pytest.raises(ValueError, match='rhat_max'):
            EnsembleSpec.from_yaml(_write_spec(tmp_path, d))

    def test_ess_min_must_be_positive(self, tmp_path):
        d = _escalation_spec_dict()
        d['fit']['escalation']['ess_min'] = 0
        with pytest.raises(ValueError, match='ess_min'):
            EnsembleSpec.from_yaml(_write_spec(tmp_path, d))

    def test_n_warmup_must_be_int(self, tmp_path):
        d = _escalation_spec_dict()
        d['fit']['escalation']['n_warmup'] = 800.0
        with pytest.raises(ValueError, match='must be an integer'):
            EnsembleSpec.from_yaml(_write_spec(tmp_path, d))

    def test_retry_must_not_be_weaker_warmup(self, tmp_path):
        d = _escalation_spec_dict()
        # dev spec fit.n_warmup is 100
        d['fit']['escalation']['n_warmup'] = 50
        with pytest.raises(ValueError, match='must be >= fit.n_warmup'):
            EnsembleSpec.from_yaml(_write_spec(tmp_path, d))

    def test_retry_must_not_be_weaker_samples(self, tmp_path):
        d = _escalation_spec_dict()
        # dev spec fit.n_samples is 200
        d['fit']['escalation']['n_samples'] = 100
        with pytest.raises(ValueError, match='must be >= fit.n_samples'):
            EnsembleSpec.from_yaml(_write_spec(tmp_path, d))


# ==============================================================================
# Trigger logic
# ==============================================================================


class TestEscalationGate:
    ESC = EscalationSpec(enabled=True, rhat_max=1.05, ess_min=50.0)

    def test_high_rhat_triggers(self):
        assert needs_escalation({'max_rhat': 1.06, 'min_ess': 500.0}, self.ESC)

    def test_low_ess_triggers(self):
        assert needs_escalation({'max_rhat': 1.0, 'min_ess': 49.0}, self.ESC)

    def test_clean_fit_passes(self):
        assert not needs_escalation({'max_rhat': 1.01, 'min_ess': 500.0}, self.ESC)

    def test_thresholds_are_strict(self):
        # exactly at threshold = passing (gate is > / <)
        assert not needs_escalation({'max_rhat': 1.05, 'min_ess': 50.0}, self.ESC)


class TestDonorMassMatrix:
    def test_multi_chain_pooled_by_mean(self):
        per_chain = np.stack([np.eye(3) * 1.0, np.eye(3) * 3.0])
        donor = worker._donor_mass_matrix({'adapted_inverse_mass_matrix': per_chain})
        np.testing.assert_allclose(donor, np.eye(3) * 2.0)

    def test_single_chain_passthrough(self):
        m = np.diag([1.0, 2.0])
        donor = worker._donor_mass_matrix({'adapted_inverse_mass_matrix': m})
        np.testing.assert_array_equal(donor, m)

    def test_missing_matrix_raises(self):
        with pytest.raises(RuntimeError, match='adapted'):
            worker._donor_mass_matrix({})

    def test_bad_ndim_raises(self):
        with pytest.raises(RuntimeError, match='unexpected'):
            worker._donor_mass_matrix({'adapted_inverse_mass_matrix': np.ones(3)})


# ==============================================================================
# Worker flow (sampler mocked at the _run_fit_attempt boundary)
# ==============================================================================

_N_PARAMS = 3
_GOOD = {'max_rhat': 1.01, 'min_ess': 800.0}
_BAD_RHAT = {'max_rhat': 1.20, 'min_ess': 300.0}
_BAD_ESS = {'max_rhat': 1.02, 'min_ess': 12.0}


class _FakeResult:
    def __init__(self, diagnostics):
        self.samples = np.zeros((8, _N_PARAMS))
        self.log_prob = np.zeros(8)
        self.chains = None
        self.diagnostics = diagnostics


def _fake_summary(fit_id: str, quality: dict) -> dict:
    return {
        'fit_id': fit_id,
        'status': 'succeeded',
        'error_message': '',
        'max_rhat': float(quality['max_rhat']),
        'min_ess': float(quality['min_ess']),
        'n_divergences': int(quality.get('n_divergences', 0)),
        'divergence_rate': float(quality.get('divergence_rate', 0.0)),
        'fit_wallclock_s': 1.0,
        'precond_wallclock_s': 0.5,
    }


# a distinct per-chain adapted metric so the pooled donor is recognizable
_ADAPTED = np.stack([np.eye(_N_PARAMS) * 1.0, np.eye(_N_PARAMS) * 3.0])
_EXPECTED_DONOR = np.eye(_N_PARAMS) * 2.0


class _AttemptRecorder:
    """Replaces worker._run_fit_attempt; plays back scripted quality tiers."""

    def __init__(self, qualities):
        self.qualities = list(qualities)
        self.calls = []

    def __call__(
        self,
        row,
        spec,
        config,
        run_dir,
        truth,
        noise_seed,
        sampler_seed,
        n_warmup=None,
        n_samples=None,
        init_inverse_mass=None,
        adapt_mass=None,
        reuse=None,
    ):
        self.calls.append(
            {
                'sampler_seed': sampler_seed,
                'n_warmup': n_warmup,
                'n_samples': n_samples,
                'init_inverse_mass': init_inverse_mass,
                'adapt_mass': adapt_mass,
                'reuse': reuse,
            }
        )
        quality = self.qualities[len(self.calls) - 1]
        summary = _fake_summary(str(row['fit_id']), quality)
        summary['sampler_seed'] = sampler_seed
        summary['has_chains'] = bool(row['save_chains'])
        artifacts = worker._AttemptArtifacts(
            result=_FakeResult({'adapted_inverse_mass_matrix': _ADAPTED.copy()}),
            inputs=None,
            task=None,
            preconditioner=None,
            sampled_names=[f'p{i}' for i in range(_N_PARAMS)],
        )
        return summary, artifacts


@pytest.fixture
def escalation_run(tmp_path):
    """(run_dir, spec, config, first manifest row) with escalation enabled."""
    spec_path = _write_spec(tmp_path, _escalation_spec_dict())
    run_dir = expand(spec_path, REGISTRY, tmp_path / 'runs')
    spec, config, manifest = load_run(run_dir)
    return run_dir, spec, config, manifest.iloc[0]


@pytest.fixture
def frozen_escalation_run(tmp_path):
    """Escalation enabled with a frozen first-pass metric (adapt_mass off)."""
    d = _escalation_spec_dict()
    d['fit']['adapt_mass'] = False
    spec_path = _write_spec(tmp_path, d)
    run_dir = expand(spec_path, REGISTRY, tmp_path / 'runs')
    spec, config, manifest = load_run(run_dir)
    return run_dir, spec, config, manifest.iloc[0]


@pytest.fixture
def legacy_run(tmp_path):
    """Same scene with NO escalation block (legacy retry semantics)."""
    d = _spec_dict()
    d['output'] = {'save_chains': 'all', 'save_mocks': 'none'}
    spec_path = _write_spec(tmp_path, d)
    run_dir = expand(spec_path, REGISTRY, tmp_path / 'runs')
    spec, config, manifest = load_run(run_dir)
    return run_dir, spec, config, manifest.iloc[0]


class TestEscalationFlow:
    def test_bad_rhat_triggers_retry_with_donated_metric(
        self, escalation_run, monkeypatch
    ):
        run_dir, spec, config, row = escalation_run
        rec = _AttemptRecorder([_BAD_RHAT, _GOOD])
        monkeypatch.setattr(worker, '_run_fit_attempt', rec)

        summary = run_single_fit(row, spec, config, run_dir)

        assert len(rec.calls) == 2
        first, retry = rec.calls
        # first attempt runs the spec's baseline config
        assert first['n_warmup'] is None and first['n_samples'] is None
        assert first['init_inverse_mass'] is None and first['reuse'] is None
        # retry escalates warmup/samples, donates the pooled adapted metric,
        # and reuses the first attempt's inputs/task/preconditioner
        assert retry['n_warmup'] == spec.escalation.n_warmup
        assert retry['n_samples'] == spec.escalation.n_samples
        np.testing.assert_allclose(retry['init_inverse_mass'], _EXPECTED_DONOR)
        assert isinstance(retry['reuse'], worker._AttemptArtifacts)
        # config changes, seed stream advances deterministically
        assert retry['sampler_seed'] == first['sampler_seed'] + 1000

        # row recording
        assert summary['escalated'] is True
        assert summary['n_attempts'] == 2
        assert summary['first_attempt_max_rhat'] == _BAD_RHAT['max_rhat']
        assert summary['first_attempt_min_ess'] == _BAD_RHAT['min_ess']
        assert summary['first_attempt_n_divergences'] == 0
        assert summary['first_attempt_divergence_rate'] == 0.0
        assert summary['first_attempt_wallclock_s'] == 1.0
        # total wallclock covers both attempts (outer timer, mocked attempts
        # are near-instant)
        assert isinstance(summary['fit_wallclock_s'], float)
        assert summary['fit_wallclock_s'] >= 0.0

        # retry passed the gate: only the final chains file is kept
        fit_id = str(row['fit_id'])
        assert (run_dir / 'chains' / f'{fit_id}.npz').exists()
        assert not (run_dir / 'chains' / f'{fit_id}.attempt1.npz').exists()

    def test_frozen_first_pass_retries_with_adaptation(
        self, frozen_escalation_run, monkeypatch
    ):
        # frozen-metric tier: attempt 1 runs the spec's adapt_mass=False and
        # records no adapted metric, so the retry re-enables mass adaptation
        # instead of donating one
        run_dir, spec, config, row = frozen_escalation_run
        assert spec.adapt_mass is False
        rec = _AttemptRecorder([_BAD_RHAT, _GOOD])
        monkeypatch.setattr(worker, '_run_fit_attempt', rec)

        summary = run_single_fit(row, spec, config, run_dir)

        assert len(rec.calls) == 2
        first, retry = rec.calls
        assert first['adapt_mass'] is None and first['init_inverse_mass'] is None
        assert retry['adapt_mass'] is True
        assert retry['init_inverse_mass'] is None
        assert retry['n_warmup'] == spec.escalation.n_warmup
        assert retry['n_samples'] == spec.escalation.n_samples
        assert isinstance(retry['reuse'], worker._AttemptArtifacts)
        assert summary['escalated'] is True and summary['n_attempts'] == 2

    def test_low_ess_triggers_retry(self, escalation_run, monkeypatch):
        run_dir, spec, config, row = escalation_run
        rec = _AttemptRecorder([_BAD_ESS, _GOOD])
        monkeypatch.setattr(worker, '_run_fit_attempt', rec)
        summary = run_single_fit(row, spec, config, run_dir)
        assert len(rec.calls) == 2
        assert summary['escalated'] is True

    def test_clean_first_attempt_no_retry(self, escalation_run, monkeypatch):
        run_dir, spec, config, row = escalation_run
        rec = _AttemptRecorder([_GOOD])
        monkeypatch.setattr(worker, '_run_fit_attempt', rec)
        summary = run_single_fit(row, spec, config, run_dir)
        assert len(rec.calls) == 1
        assert summary['escalated'] is False
        assert summary['n_attempts'] == 1
        assert 'first_attempt_max_rhat' not in summary
        assert (run_dir / 'chains' / f"{row['fit_id']}.npz").exists()

    def test_divergences_alone_never_trigger(self, escalation_run, monkeypatch):
        # the silent catastrophic class has div=0, so the gate is quality-
        # based; conversely a high divergence rate with clean rhat/ess must
        # NOT trigger a retry (divergence-based triggers are useless here)
        run_dir, spec, config, row = escalation_run
        diverging = dict(_GOOD, n_divergences=900, divergence_rate=0.95)
        rec = _AttemptRecorder([diverging])
        monkeypatch.setattr(worker, '_run_fit_attempt', rec)
        summary = run_single_fit(row, spec, config, run_dir)
        assert len(rec.calls) == 1
        assert summary['escalated'] is False

    def test_retry_also_failing_keeps_both_chains(self, escalation_run, monkeypatch):
        run_dir, spec, config, row = escalation_run
        still_bad = {'max_rhat': 1.15, 'min_ess': 20.0}
        rec = _AttemptRecorder([_BAD_RHAT, still_bad])
        monkeypatch.setattr(worker, '_run_fit_attempt', rec)
        summary = run_single_fit(row, spec, config, run_dir)

        # recorded as-is: the retry's (still-bad) stats, flagged escalated
        assert summary['escalated'] is True
        assert summary['max_rhat'] == still_bad['max_rhat']
        assert summary['first_attempt_max_rhat'] == _BAD_RHAT['max_rhat']
        # both chains files kept for forensics
        fit_id = str(row['fit_id'])
        assert (run_dir / 'chains' / f'{fit_id}.npz').exists()
        assert (run_dir / 'chains' / f'{fit_id}.attempt1.npz').exists()


class TestBackwardCompatibility:
    def test_no_escalation_block_keeps_legacy_retry(self, legacy_run, monkeypatch):
        # catastrophic first attempt -> fresh-seed retry with the SAME config
        run_dir, spec, config, row = legacy_run
        assert not spec.escalation.enabled
        catastrophic = {
            'max_rhat': 1.5,
            'min_ess': 5.0,
            'divergence_rate': 0.0,
        }
        rec = _AttemptRecorder([catastrophic, _GOOD])
        monkeypatch.setattr(worker, '_run_fit_attempt', rec)
        summary = run_single_fit(row, spec, config, run_dir)

        assert len(rec.calls) == 2
        first, retry = rec.calls
        # legacy retry changes ONLY the seed -- no escalation knobs
        assert retry['sampler_seed'] == first['sampler_seed'] + 1000
        assert retry['n_warmup'] is None and retry['n_samples'] is None
        assert retry['init_inverse_mass'] is None and retry['reuse'] is None
        assert summary['n_attempts'] == 2

    def test_no_escalation_block_adds_no_new_columns(self, legacy_run, monkeypatch):
        # rows from escalation-off runs are byte-identical to the old schema
        run_dir, spec, config, row = legacy_run
        rec = _AttemptRecorder([_GOOD])
        monkeypatch.setattr(worker, '_run_fit_attempt', rec)
        summary = run_single_fit(row, spec, config, run_dir)
        assert len(rec.calls) == 1
        new_columns = [
            'escalated',
            'first_attempt_max_rhat',
            'first_attempt_min_ess',
            'first_attempt_n_divergences',
            'first_attempt_divergence_rate',
            'first_attempt_wallclock_s',
        ]
        for column in new_columns:
            assert column not in summary
        # chains still written exactly once for the final attempt
        assert (run_dir / 'chains' / f"{row['fit_id']}.npz").exists()


# ==============================================================================
# Tally reporting
# ==============================================================================


class TestEscalationTally:
    def test_count_escalated_tolerates_old_rows(self, legacy_run):
        import pandas as pd

        from kl_pipe.ensemble.collate import count_escalated

        run_dir, _, _, _ = legacy_run
        _, _, manifest = load_run(run_dir)
        ids = manifest['fit_id'].tolist()[:3]
        rows = [
            # old-schema row (no escalated column)
            {'fit_id': ids[0], 'status': 'succeeded'},
            {'fit_id': ids[1], 'status': 'succeeded', 'escalated': True},
            {'fit_id': ids[2], 'status': 'succeeded', 'escalated': False},
        ]
        for r in rows:
            pd.DataFrame([r]).to_parquet(
                run_dir / 'results' / f"{r['fit_id']}.parquet", index=False
            )
        assert count_escalated(run_dir, ids) == 1

    def test_print_tally_reports_escalations(self, legacy_run, capsys):
        import pandas as pd

        from kl_pipe.ensemble import ledger
        from kl_pipe.ensemble.collate import print_tally

        run_dir, _, _, _ = legacy_run
        _, _, manifest = load_run(run_dir)
        ids = manifest['fit_id'].tolist()[:2]
        quality = {'max_rhat': 1.005, 'divergence_rate': 0.0}
        for fid, escalated in zip(ids, [True, False]):
            ledger.try_claim(run_dir, fid)
            ledger.mark_done(run_dir, fid)
            pd.DataFrame(
                [
                    {
                        'fit_id': fid,
                        'status': 'succeeded',
                        'escalated': escalated,
                        **quality,
                    }
                ]
            ).to_parquet(run_dir / 'results' / f'{fid}.parquet', index=False)
        print_tally(run_dir)
        out = capsys.readouterr().out
        assert '(escalated: 1)' in out


# ==============================================================================
# Sampler-config knob (fast validation; sampler pass-through in test_numpyro)
# ==============================================================================


class TestInitInverseMassConfig:
    def _valid(self):
        return np.diag([1.0, 2.0, 3.0])

    def test_requires_laplace(self):
        with pytest.raises(ValueError, match="precondition='laplace'"):
            NumpyroSamplerConfig(init_inverse_mass_matrix=self._valid())

    def test_valid_matrix_accepted(self):
        config = NumpyroSamplerConfig(
            precondition='laplace', init_inverse_mass_matrix=self._valid()
        )
        np.testing.assert_array_equal(config.init_inverse_mass_matrix, self._valid())

    def test_non_square_rejected(self):
        with pytest.raises(ValueError, match='square'):
            NumpyroSamplerConfig(
                precondition='laplace',
                init_inverse_mass_matrix=np.ones((2, 3)),
            )

    def test_non_symmetric_rejected(self):
        m = self._valid()
        m[0, 1] = 0.5
        with pytest.raises(ValueError, match='symmetric'):
            NumpyroSamplerConfig(precondition='laplace', init_inverse_mass_matrix=m)

    def test_non_positive_definite_rejected(self):
        with pytest.raises(ValueError, match='positive definite'):
            NumpyroSamplerConfig(
                precondition='laplace',
                init_inverse_mass_matrix=np.diag([1.0, -1.0]),
            )

    def test_non_finite_rejected(self):
        with pytest.raises(ValueError, match='non-finite'):
            NumpyroSamplerConfig(
                precondition='laplace',
                init_inverse_mass_matrix=np.diag([1.0, np.nan]),
            )
