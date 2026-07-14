"""
Unit tests for the ensemble pipeline infrastructure (kl_pipe/ensemble/).

Covers spec/observing-config validation, deterministic expansion, the CRN
noise-seed rule, ring pairing, the filesystem ledger, and collation. The
fit execution path (worker/mocks) is exercised by the local shakedown and
a small marked integration test, not here -- these tests are fast and
fit-free.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from kl_pipe.ensemble import ledger
from kl_pipe.ensemble.expander import (
    build_manifest,
    compute_fit_id,
    expand,
    load_run,
    truth_from_row,
)
from kl_pipe.ensemble.scene import scene_priors, scene_truth_defaults
from kl_pipe.ensemble.spec import EnsembleSpec, ObservingConfig
from kl_pipe.priors import LogNormal, Uniform

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = REPO_ROOT / 'configs' / 'observing'
DEV_SPEC = REPO_ROOT / 'configs' / 'ensembles' / 'sigma_eps_cosi_dev.yaml'


@pytest.fixture(scope='module')
def dev_spec() -> EnsembleSpec:
    return EnsembleSpec.from_yaml(DEV_SPEC)


@pytest.fixture(scope='module')
def canonical_q() -> ObservingConfig:
    return ObservingConfig.from_yaml(REGISTRY / 'canonical_Q.yaml')


def _spec_dict() -> dict:
    return yaml.safe_load(DEV_SPEC.read_text())


def _write_spec(tmp_path: Path, spec_dict: dict) -> Path:
    path = tmp_path / 'spec.yaml'
    path.write_text(yaml.safe_dump(spec_dict))
    return path


# ==============================================================================
# Spec + observing-config validation
# ==============================================================================


class TestSpecValidation:
    def test_dev_spec_loads(self, dev_spec):
        assert dev_spec.run_name == 'sigma_eps_cosi_dev'
        assert dev_spec.n_fits == 4
        assert 'vel.vcirc' in dev_spec.draw  # 'vcirc' alias resolved
        assert dev_spec.draw['vel.vcirc'].dist == 'lognormal_tf'

    def test_unknown_top_key_rejected(self, tmp_path):
        d = _spec_dict()
        d['typo_key'] = 1
        with pytest.raises(ValueError, match='unknown keys'):
            EnsembleSpec.from_yaml(_write_spec(tmp_path, d))

    def test_flux_fixed_rejected(self, tmp_path):
        d = _spec_dict()
        d['bank']['fixed']['flux'] = 1.0
        with pytest.raises(ValueError, match="'flux' is ambiguous"):
            EnsembleSpec.from_yaml(_write_spec(tmp_path, d))

    def test_unknown_draw_dist_rejected(self, tmp_path):
        d = _spec_dict()
        d['bank']['draw']['z'] = {'dist': 'hlss_halpha_nz', 'low': 1, 'high': 2}
        with pytest.raises(ValueError, match='unknown draw dist'):
            EnsembleSpec.from_yaml(_write_spec(tmp_path, d))

    def test_antithetic_g_rejected(self, tmp_path):
        d = _spec_dict()
        d['ring'] = {'enabled': True, 'antithetic_g': True}
        with pytest.raises(NotImplementedError, match='antithetic_g'):
            EnsembleSpec.from_yaml(_write_spec(tmp_path, d))

    def test_non_numpyro_sampler_rejected(self, tmp_path):
        d = _spec_dict()
        d['fit']['sampler'] = 'emcee'
        with pytest.raises(NotImplementedError, match='numpyro'):
            EnsembleSpec.from_yaml(_write_spec(tmp_path, d))

    def test_sampled_z_rejected(self, tmp_path):
        d = _spec_dict()
        d['fit']['pin_z_to_truth'] = False
        with pytest.raises(NotImplementedError, match='sampled-z'):
            EnsembleSpec.from_yaml(_write_spec(tmp_path, d))

    def test_grid_scheme_needs_component(self, tmp_path):
        d = _spec_dict()
        d['shear'] = {'scheme': 'grid', 'grid': [-0.05, 0.0, 0.05]}
        with pytest.raises(ValueError, match='missing required keys'):
            EnsembleSpec.from_yaml(_write_spec(tmp_path, d))


class TestObservingConfig:
    def test_canonical_q_loads(self, canonical_q):
        assert canonical_q.bands == ('F087',)
        assert canonical_q.grism_rolls_deg == (0.0,)
        assert canonical_q.lines == ('Halpha',)
        assert len(canonical_q.content_hash) == 64

    def test_multi_line_rejected(self, tmp_path):
        raw = yaml.safe_load((REGISTRY / 'canonical_Q.yaml').read_text())
        raw['lines'] = ['Halpha', 'NII6584']
        path = tmp_path / 'bad.yaml'
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(NotImplementedError, match='single-Halpha'):
            ObservingConfig.from_yaml(path)

    def test_non_gaussian_psf_rejected(self, tmp_path):
        raw = yaml.safe_load((REGISTRY / 'canonical_Q.yaml').read_text())
        raw['psf']['grism'] = {'type': 'roman_wfi_complex'}
        path = tmp_path / 'bad.yaml'
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(NotImplementedError, match='gaussian'):
            ObservingConfig.from_yaml(path)


# ==============================================================================
# Scene
# ==============================================================================


class TestScene:
    def test_truth_defaults_broadcast(self, dev_spec, canonical_q):
        truth = scene_truth_defaults(canonical_q, {'h_over_r': 0.2, 'x0': 0.1})
        for comp in ('F087', 'Halpha', 'Halpha.cont'):
            assert truth[f'{comp}.h_over_r'] == 0.2
            assert truth[f'{comp}.x0'] == 0.1
            assert truth[f'{comp}.y0'] == 0.0

    def test_unknown_fixed_key_rejected(self, canonical_q):
        with pytest.raises(ValueError, match='not a scene parameter'):
            scene_truth_defaults(canonical_q, {'F999.flux': 1.0})

    def test_priors_self_consistent(self, dev_spec, canonical_q):
        truth = scene_truth_defaults(canonical_q, dev_spec.fixed)
        truth.update(
            {
                'cosi': 0.6,
                'theta_int': 1.0,
                'g1': 0.05,
                'g2': 0.05,
                'vel.vcirc': 210.0,
                'z': 1.3,
            }
        )
        priors = scene_priors(truth, canonical_q, dev_spec)

        # drawn params carry their generating distribution
        assert isinstance(priors.get_prior('theta_int'), Uniform)
        assert priors.get_prior('theta_int').bounds == (0.0, np.pi)
        vc = priors.get_prior('vel.vcirc')
        assert isinstance(vc, LogNormal)
        assert np.isclose(vc.median, 200.0)
        assert np.isclose(vc.sigma, 0.08 * np.log(10.0))
        # stratified cosi gets the population prior over the stratify range
        assert priors.get_prior('cosi').bounds == dev_spec.stratify_range
        # z pinned to the per-fit truth
        assert priors.fixed_values['z'] == 1.3
        # every declared parameter has a truth value
        all_names = set(priors.sampled_names) | set(priors.fixed_names)
        assert all_names <= set(truth)


# ==============================================================================
# Expander
# ==============================================================================


class TestExpander:
    def test_fit_id_frozen(self):
        """fit_id scheme is load-bearing for resume; freeze one value."""
        # provenance: sha1('run|1|0|0|0|0|0')[:16] computed at implementation
        # time (2026-07-13); changing the hash scheme invalidates every
        # existing run dir
        assert compute_fit_id('run', 1, 0, 0, 0, 0, 0) == '7509f60d093524bd'

    def test_deterministic(self, dev_spec, canonical_q):
        m1 = build_manifest(dev_spec, canonical_q)
        m2 = build_manifest(dev_spec, canonical_q)
        pd.testing.assert_frame_equal(m1, m2)

    def test_row_count_and_unique_ids(self, dev_spec, canonical_q):
        m = build_manifest(dev_spec, canonical_q)
        assert len(m) == dev_spec.n_fits
        assert m['fit_id'].is_unique

    def test_truths_within_populations(self, dev_spec, canonical_q):
        m = build_manifest(dev_spec, canonical_q)
        assert m['truth.theta_int'].between(0, np.pi).all()
        assert m['truth.z'].between(1.0, 1.9).all()
        assert (m['truth.g1'] == 0.05).all()
        lo, hi = dev_spec.stratify_range
        assert m['truth.cosi'].between(lo, hi).all()

    def test_crn_noise_seed_constant_across_shear_grid(self, tmp_path, canonical_q):
        d = _spec_dict()
        d['shear'] = {
            'scheme': 'grid',
            'component': 'g1',
            'grid': [-0.05, -0.01, 0.01, 0.05],
        }
        spec = EnsembleSpec.from_yaml(_write_spec(tmp_path, d))
        m = build_manifest(spec, canonical_q)
        for _, group in m.groupby(['cosi_bin', 'galaxy_id', 'noise_rep']):
            assert group['noise_seed'].nunique() == 1  # CRN along shear
            assert group['shear_step'].nunique() == len(group)
        # but fresh noise across galaxies
        assert m.groupby('noise_seed').ngroups == (
            spec.stratify_n_bins * spec.n_gal_per_bin * spec.m_noise
        )

    def test_independent_noise_across_reps(self, tmp_path, canonical_q):
        d = _spec_dict()
        d['bank']['m_noise'] = 3
        spec = EnsembleSpec.from_yaml(_write_spec(tmp_path, d))
        m = build_manifest(spec, canonical_q)
        for _, group in m.groupby(['cosi_bin', 'galaxy_id']):
            assert group['noise_seed'].nunique() == 3

    def test_ring_pairs(self, tmp_path, canonical_q):
        d = _spec_dict()
        d['ring'] = {'enabled': True}
        spec = EnsembleSpec.from_yaml(_write_spec(tmp_path, d))
        m = build_manifest(spec, canonical_q)
        assert len(m) == 2 * 4
        by_id = m.set_index('fit_id')
        for _, row in m.iterrows():
            partner = by_id.loc[row['ring_partner_id']]
            assert partner['ring_partner_id'] == row['fit_id']  # reciprocal
            dtheta = (partner['truth.theta_int'] - row['truth.theta_int']) % np.pi
            assert np.isclose(dtheta, np.pi / 2)
            # identical intrinsics + shear, independent noise
            assert partner['truth.vel.vcirc'] == row['truth.vel.vcirc']
            assert partner['truth.g1'] == row['truth.g1']
            assert partner['noise_seed'] != row['noise_seed']

    def test_subset_policy(self, tmp_path, canonical_q):
        d = _spec_dict()
        d['output'] = {'save_chains': 'subset', 'save_mocks': 'none'}
        spec = EnsembleSpec.from_yaml(_write_spec(tmp_path, d))
        m = build_manifest(spec, canonical_q)
        # one diagnostic fit per cosi bin
        assert m['save_chains'].sum() == spec.stratify_n_bins
        assert not m['save_mocks'].any()

    def test_truth_from_row_roundtrip(self, dev_spec, canonical_q):
        m = build_manifest(dev_spec, canonical_q)
        truth = truth_from_row(m.iloc[0])
        assert truth['cosi'] == m.iloc[0]['truth.cosi']
        assert 'vel.vcirc' in truth
        assert not any(k.startswith('truth.') for k in truth)

    def test_expand_provenance_and_reload(self, tmp_path):
        run_dir = expand(DEV_SPEC, REGISTRY, tmp_path / 'runs')
        record = json.loads((run_dir / 'provenance' / 'expansion.json').read_text())
        assert record['n_fits'] == 4
        assert len(record['observing_config_hash']) == 64
        spec, config, manifest = load_run(run_dir)
        assert len(manifest) == 4
        # snapshot, not live registry: hash matches the copied file
        assert config.content_hash == record['observing_config_hash']

    def test_expand_refuses_overwrite(self, tmp_path):
        expand(DEV_SPEC, REGISTRY, tmp_path / 'runs')
        with pytest.raises(FileExistsError, match='overwrite'):
            expand(DEV_SPEC, REGISTRY, tmp_path / 'runs')


# ==============================================================================
# Ledger
# ==============================================================================


@pytest.fixture
def run_dir(tmp_path):
    return expand(DEV_SPEC, REGISTRY, tmp_path / 'runs')


class TestLedger:
    def test_claim_is_exclusive(self, run_dir):
        assert ledger.try_claim(run_dir, 'abc123')
        assert not ledger.try_claim(run_dir, 'abc123')
        ledger.release_claim(run_dir, 'abc123')
        assert ledger.try_claim(run_dir, 'abc123')

    def test_status_lifecycle(self, run_dir):
        fid = 'fit0001'
        assert ledger.fit_status(run_dir, fid, 30.0) == 'never_run'
        ledger.try_claim(run_dir, fid)
        assert ledger.fit_status(run_dir, fid, 30.0) == 'in_progress'
        ledger.mark_done(run_dir, fid)
        assert ledger.fit_status(run_dir, fid, 30.0) == 'succeeded'

    def test_failed_status(self, run_dir):
        fid = 'fit0002'
        ledger.try_claim(run_dir, fid)
        ledger.mark_failed(run_dir, fid, 'boom')
        assert ledger.fit_status(run_dir, fid, 30.0) == 'failed'
        ledger.clear_failed(run_dir, fid)
        ledger.release_claim(run_dir, fid)
        assert ledger.fit_status(run_dir, fid, 30.0) == 'never_run'

    def test_stale_claim_dead_pid(self, run_dir):
        fid = 'fit0003'
        ledger.try_claim(run_dir, fid)
        # rewrite the claim meta with a dead pid on this host
        claim = run_dir / 'status' / 'claims' / fid / 'claim.json'
        meta = json.loads(claim.read_text())
        meta['pid'] = _dead_pid()
        claim.write_text(json.dumps(meta))
        assert ledger.fit_status(run_dir, fid, 30.0) == 'stale'

    def test_tally(self, run_dir):
        _, _, manifest = load_run(run_dir)
        ids = manifest['fit_id'].tolist()
        ledger.try_claim(run_dir, ids[0])
        ledger.mark_done(run_dir, ids[0])
        ledger.try_claim(run_dir, ids[1])
        ledger.mark_failed(run_dir, ids[1], 'x')
        groups = ledger.tally(run_dir, ids, 30.0)
        assert groups['succeeded'] == [ids[0]]
        assert groups['failed'] == [ids[1]]
        assert sorted(groups['never_run']) == sorted(ids[2:])


def _dead_pid() -> int:
    # spawn-and-reap gives a pid guaranteed dead and unlikely reused yet
    import subprocess

    proc = subprocess.Popen(['true'])
    proc.wait()
    return proc.pid


# ==============================================================================
# Collate
# ==============================================================================


class TestCollate:
    def test_collate_and_join(self, run_dir):
        from kl_pipe.ensemble.collate import analysis_table, collate_results

        _, _, manifest = load_run(run_dir)
        for _, row in manifest.iterrows():
            pd.DataFrame(
                [{'fit_id': row['fit_id'], 'status': 'succeeded', 'post.g1.mean': 0.05}]
            ).to_parquet(run_dir / 'results' / f"{row['fit_id']}.parquet", index=False)
        results = collate_results(run_dir)
        assert len(results) == len(manifest)
        table = analysis_table(run_dir)
        assert len(table) == len(manifest)
        assert 'truth.g1' in table.columns
        assert 'post.g1.mean' in table.columns

    def test_collate_empty_raises(self, run_dir):
        from kl_pipe.ensemble.collate import collate_results

        with pytest.raises(FileNotFoundError, match='no per-fit result'):
            collate_results(run_dir)
