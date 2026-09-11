"""Spec knobs added for the census: escalation.ess_min_shear and dispatch.claim_order."""

import dataclasses
from pathlib import Path

import pytest
import yaml

import numpy as np
import pandas as pd

from kl_pipe.ensemble.spec import EnsembleSpec, EscalationSpec
from kl_pipe.ensemble.worker import claim_order_index, needs_escalation

pytestmark = pytest.mark.roman_ensemble

REPO_ROOT = Path(__file__).resolve().parents[1]
DEV_SPEC = REPO_ROOT / 'configs' / 'ensembles' / 'sigma_eps_cosi_dev.yaml'


def _write(tmp_path, d):
    p = tmp_path / 'spec.yaml'
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    return p


class TestEssMinShear:
    def test_default_none_and_parse(self, tmp_path):
        spec = EnsembleSpec.from_yaml(DEV_SPEC)
        assert spec.escalation.ess_min_shear is None
        d = yaml.safe_load(DEV_SPEC.read_text())
        d['fit'].setdefault('escalation', {})['ess_min_shear'] = 200
        spec2 = EnsembleSpec.from_yaml(_write(tmp_path, d))
        assert spec2.escalation.ess_min_shear == 200.0
        assert spec2.resolve_defaults(d)['fit']['escalation']['ess_min_shear'] == 200.0
        assert (
            spec.resolve_defaults(yaml.safe_load(DEV_SPEC.read_text()))['fit'][
                'escalation'
            ]['ess_min_shear']
            is None
        )

    def test_validation(self):
        with pytest.raises(ValueError, match='ess_min_shear'):
            EscalationSpec(ess_min_shear=0.0)
        with pytest.raises(ValueError, match='ess_min_shear'):
            EscalationSpec(ess_min_shear=-5.0)
        assert EscalationSpec(ess_min_shear=200.0).ess_min_shear == 200.0


class TestClaimOrder:
    def test_default_and_parse(self, tmp_path):
        spec = EnsembleSpec.from_yaml(DEV_SPEC)
        assert spec.claim_order == 'manifest'
        d = yaml.safe_load(DEV_SPEC.read_text())
        d['dispatch']['claim_order'] = 'hard_first'
        spec2 = EnsembleSpec.from_yaml(_write(tmp_path, d))
        assert spec2.claim_order == 'hard_first'
        resolved = spec2.resolve_defaults(d)['dispatch']
        assert resolved['claim_order'] == 'hard_first'
        assert resolved['workers_per_node'] == spec2.workers_per_node

    def test_validation(self, tmp_path):
        spec = EnsembleSpec.from_yaml(DEV_SPEC)
        with pytest.raises(ValueError, match='claim_order'):
            dataclasses.replace(spec, claim_order='random')
        d = yaml.safe_load(DEV_SPEC.read_text())
        d['dispatch']['claim_orderr'] = 'manifest'
        with pytest.raises(ValueError):
            EnsembleSpec.from_yaml(_write(tmp_path, d))


class TestGateAndOrder:
    def test_needs_escalation_shear_floor(self):
        healthy = {'max_rhat': 1.01, 'min_ess': 150.0, 'ess_g1': 260.0, 'ess_g2': 120.0}
        assert not needs_escalation(healthy, EscalationSpec())
        assert needs_escalation(healthy, EscalationSpec(ess_min_shear=200.0))
        assert not needs_escalation(healthy, EscalationSpec(ess_min_shear=100.0))
        # the generic gate still applies
        assert needs_escalation(
            {**healthy, 'max_rhat': 1.2}, EscalationSpec(ess_min_shear=100.0)
        )

    def test_claim_order_index(self):
        man = pd.DataFrame(
            {
                'fit_id': list('abcde'),
                'truth.cosi': [0.5, 0.06, 0.9, 0.06, 0.3],
                'line_snr': [10.0, 20.0, 10.0, 12.0, 10.0],
            }
        )
        assert list(claim_order_index(man, 'manifest')) == [0, 1, 2, 3, 4]
        order = list(man.fit_id.iloc[claim_order_index(man, 'hard_first')])
        # extremes first (0.06 pair, lower SNR member first), then 0.9, 0.3, 0.5
        assert order == ['d', 'b', 'c', 'e', 'a']
        with pytest.raises(ValueError, match='claim_order'):
            claim_order_index(man, 'random')
