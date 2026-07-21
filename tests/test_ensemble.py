"""
Unit tests for the ensemble pipeline infrastructure (kl_pipe/ensemble/).

Covers spec/observation-config validation, deterministic expansion, the CRN
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
from kl_pipe.ensemble.mocks import build_fit_inputs
from kl_pipe.ensemble.scene import scene_priors, scene_truth_defaults
from kl_pipe.ensemble.spec import (
    EnsembleSpec,
    FoldingThresholdTier,
    ObservationConfig,
    PSFSpec,
)
from kl_pipe.priors import LogNormal, Uniform

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = REPO_ROOT / 'configs' / 'observation'
DEV_SPEC = REPO_ROOT / 'configs' / 'ensembles' / 'sigma_eps_cosi_dev.yaml'
P_GSNR_SPEC = REPO_ROOT / 'configs' / 'ensembles' / 'sigma_eps_gsnr_P_vista.yaml'


@pytest.fixture(scope='module')
def dev_spec() -> EnsembleSpec:
    return EnsembleSpec.from_yaml(DEV_SPEC)


@pytest.fixture(scope='module')
def canonical_q() -> ObservationConfig:
    return ObservationConfig.from_yaml(REGISTRY / 'canonical_Q.yaml')


def _spec_dict() -> dict:
    return yaml.safe_load(DEV_SPEC.read_text())


def _write_spec(tmp_path: Path, spec_dict: dict) -> Path:
    path = tmp_path / 'spec.yaml'
    path.write_text(yaml.safe_dump(spec_dict))
    return path


# ==============================================================================
# Spec + observation-config validation
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
        d['population']['fixed']['flux'] = 1.0
        with pytest.raises(ValueError, match="'flux' is ambiguous"):
            EnsembleSpec.from_yaml(_write_spec(tmp_path, d))

    def test_unknown_draw_dist_rejected(self, tmp_path):
        d = _spec_dict()
        d['population']['draw']['z'] = {'dist': 'hlss_halpha_nz', 'low': 1, 'high': 2}
        with pytest.raises(ValueError, match='unknown draw dist'):
            EnsembleSpec.from_yaml(_write_spec(tmp_path, d))

    def test_antithetic_g_rejected(self, tmp_path):
        d = _spec_dict()
        d['population']['ring'] = {'enabled': True, 'antithetic_g': True}
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
        d['population']['shear'] = {'scheme': 'grid', 'grid': [-0.05, 0.0, 0.05]}
        with pytest.raises(ValueError, match='missing required keys'):
            EnsembleSpec.from_yaml(_write_spec(tmp_path, d))

    def test_catalog_population_rejected(self, tmp_path):
        d = _spec_dict()
        d['population']['type'] = 'catalog'
        with pytest.raises(NotImplementedError, match='catalog population backend'):
            EnsembleSpec.from_yaml(_write_spec(tmp_path, d))

    def test_old_flat_schema_rejected(self, tmp_path):
        # pre-hierarchical flat schema (top-level run_name/bank/...) must
        # raise loudly on its unknown keys, not half-parse
        old = {
            'run_name': 'legacy',
            'version': 1,
            'description': 'old flat schema',
            'seed': 1,
            'measurement': 'sigma_eps_vs_cosi',
            'bank': {'stratify': {'cosi': {'n_bins': 2, 'range': [0.3, 0.9]}}},
            'observed_config': 'canonical_Q',
            'broadband_snr': 100,
            'line_snr': 100,
        }
        with pytest.raises(ValueError, match='unknown keys'):
            EnsembleSpec.from_yaml(_write_spec(tmp_path, old))


class TestObservationConfig:
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
            ObservationConfig.from_yaml(path)

    def test_non_gaussian_psf_rejected(self, tmp_path):
        raw = yaml.safe_load((REGISTRY / 'canonical_Q.yaml').read_text())
        raw['psf']['grism'] = {'type': 'roman_wfi_complex'}
        path = tmp_path / 'bad.yaml'
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(NotImplementedError, match='gaussian'):
            ObservationConfig.from_yaml(path)

    def test_gaussian_psf_specs(self, canonical_q):
        assert canonical_q.band_psf['F087'].psf_type == 'gaussian'
        assert canonical_q.band_psf['F087'].fwhm_arcsec == 0.18
        assert canonical_q.grism_psf.psf_type == 'gaussian'
        assert canonical_q.grism_psf.fwhm_arcsec == 0.18

    def test_canonical_p_roman_loads(self):
        config = ObservationConfig.from_yaml(REGISTRY / 'canonical_P_roman.yaml')
        assert config.bands == ('F158', 'F184')
        assert config.grism_rolls_deg == (0.0, 45.0, 90.0, 135.0)
        assert config.grism_psf.psf_type == 'roman_wfi'
        assert all(psf.psf_type == 'roman_wfi' for psf in config.band_psf.values())


# ==============================================================================
# z-tiered fit folding_threshold (PSFSpec.folding_threshold_tiers)
# ==============================================================================


class TestFoldingThresholdTiers:
    """The z-tiered fit-kernel folding_threshold schedule and its validation.

    The rigor audit (docs/sessions/2026-07-18_roman_psf_rigor_audit.md) found
    ft=0.01 fit kernels safe for z <= 1.2 but wing-truncation-biased at z=1.9;
    the ruling is a two-tier schedule (ft=0.01 for z<=1.2, ft=5e-3 above). At
    the tight tier ft == the mock default 5e-3, so mock == fit by construction.
    """

    def _roman_block(self, **overrides) -> dict:
        block = {'type': 'roman_wfi', 'sca': 10, 'pupil_bin': 4}
        block.update(overrides)
        return block

    def _load_grism_psf(self, tmp_path, grism_block) -> PSFSpec:
        raw = yaml.safe_load((REGISTRY / 'canonical_P_roman.yaml').read_text())
        raw['psf']['grism'] = grism_block
        path = tmp_path / 'obs.yaml'
        path.write_text(yaml.safe_dump(raw))
        return ObservationConfig.from_yaml(path).grism_psf

    def test_two_tier_resolution(self, tmp_path):
        # ruling schedule: ft=0.01 for z<=1.2, tighter 5e-3 above
        block = self._roman_block(
            folding_threshold_tiers=[
                {'z_max': 1.2, 'ft': 0.01},
                {'z_max': None, 'ft': 5.0e-3},
            ]
        )
        psf = self._load_grism_psf(tmp_path, block)
        assert psf.folding_threshold is None
        assert len(psf.folding_threshold_tiers) == 2
        # z <= 1.2 -> loose tier (boundary inclusive)
        assert psf.resolve_fit_folding_threshold(0.55) == 0.01
        assert psf.resolve_fit_folding_threshold(1.2) == 0.01
        # z > 1.2 -> tight tier == mock default (zero mismatch by construction)
        assert psf.resolve_fit_folding_threshold(1.9) == 5.0e-3
        assert psf.resolve_fit_folding_threshold(2.5) == 5.0e-3

    def test_scalar_option_unchanged(self, tmp_path):
        psf = self._load_grism_psf(tmp_path, self._roman_block(folding_threshold=0.01))
        assert psf.folding_threshold == 0.01
        assert psf.folding_threshold_tiers is None
        # scalar is z-independent
        assert psf.resolve_fit_folding_threshold(0.5) == 0.01
        assert psf.resolve_fit_folding_threshold(1.9) == 0.01
        # unset defaults resolve to None (galsim default) at any z
        default_psf = self._load_grism_psf(tmp_path, self._roman_block())
        assert default_psf.resolve_fit_folding_threshold(1.9) is None

    def test_scalar_and_tiers_mutually_exclusive(self):
        with pytest.raises(ValueError, match='mutually exclusive'):
            PSFSpec(
                psf_type='roman_wfi',
                sca=10,
                pupil_bin=4,
                folding_threshold=0.01,
                folding_threshold_tiers=(
                    FoldingThresholdTier(z_max=1.2, ft=0.01),
                    FoldingThresholdTier(z_max=None, ft=5.0e-3),
                ),
            )

    def test_final_tier_must_be_open(self):
        with pytest.raises(ValueError, match='final tier must have z_max: null'):
            PSFSpec(
                psf_type='roman_wfi',
                sca=10,
                pupil_bin=4,
                folding_threshold_tiers=(
                    FoldingThresholdTier(z_max=1.2, ft=0.01),
                    FoldingThresholdTier(z_max=2.0, ft=5.0e-3),
                ),
            )

    def test_only_final_tier_may_be_open(self):
        with pytest.raises(ValueError, match='only the final tier'):
            PSFSpec(
                psf_type='roman_wfi',
                sca=10,
                pupil_bin=4,
                folding_threshold_tiers=(
                    FoldingThresholdTier(z_max=None, ft=0.01),
                    FoldingThresholdTier(z_max=None, ft=5.0e-3),
                ),
            )

    def test_z_max_must_increase(self):
        with pytest.raises(ValueError, match='strictly increasing'):
            PSFSpec(
                psf_type='roman_wfi',
                sca=10,
                pupil_bin=4,
                folding_threshold_tiers=(
                    FoldingThresholdTier(z_max=1.5, ft=0.01),
                    FoldingThresholdTier(z_max=1.2, ft=8.0e-3),
                    FoldingThresholdTier(z_max=None, ft=5.0e-3),
                ),
            )

    def test_tier_ft_out_of_range_rejected(self):
        with pytest.raises(ValueError, match='ft must be a float in'):
            PSFSpec(
                psf_type='roman_wfi',
                sca=10,
                pupil_bin=4,
                folding_threshold_tiers=(
                    FoldingThresholdTier(z_max=1.2, ft=1.5),
                    FoldingThresholdTier(z_max=None, ft=5.0e-3),
                ),
            )

    def test_mock_must_be_at_least_as_accurate_as_tightest_tier(self):
        # mock (0.008) looser than the tight tier ft (5e-3) -> mock less
        # accurate than the fit at high z -> reject
        with pytest.raises(ValueError, match='at least as accurate'):
            PSFSpec(
                psf_type='roman_wfi',
                sca=10,
                pupil_bin=4,
                mock_folding_threshold=8.0e-3,
                folding_threshold_tiers=(
                    FoldingThresholdTier(z_max=1.2, ft=0.01),
                    FoldingThresholdTier(z_max=None, ft=5.0e-3),
                ),
            )

    def test_tiers_rejected_for_gaussian(self):
        with pytest.raises(ValueError, match='roman_wfi-only'):
            PSFSpec(
                psf_type='gaussian',
                fwhm_arcsec=0.18,
                folding_threshold_tiers=(FoldingThresholdTier(z_max=None, ft=5.0e-3),),
            )

    def test_resolve_requires_finite_z_for_tiers(self):
        psf = PSFSpec(
            psf_type='roman_wfi',
            sca=10,
            pupil_bin=4,
            folding_threshold_tiers=(
                FoldingThresholdTier(z_max=1.2, ft=0.01),
                FoldingThresholdTier(z_max=None, ft=5.0e-3),
            ),
        )
        with pytest.raises(ValueError, match='finite numeric z'):
            psf.resolve_fit_folding_threshold(None)

    def test_yaml_tier_missing_ft_rejected(self, tmp_path):
        block = self._roman_block(
            folding_threshold_tiers=[{'z_max': 1.2}, {'z_max': None, 'ft': 5.0e-3}]
        )
        with pytest.raises(ValueError, match='missing required keys'):
            self._load_grism_psf(tmp_path, block)

    def test_yaml_tier_unknown_key_rejected(self, tmp_path):
        block = self._roman_block(
            folding_threshold_tiers=[
                {'z_max': 1.2, 'ft': 0.01, 'bogus': 1},
                {'z_max': None, 'ft': 5.0e-3},
            ]
        )
        with pytest.raises(ValueError, match='unknown keys'):
            self._load_grism_psf(tmp_path, block)


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
        # provenance: sha1('run|1|0|0|0|0|0|0')[:16]. Scheme extended with
        # the sweep_step field 2026-07-14 (config-sweep axes), deliberately
        # invalidating pre-sweep scratch run dirs (no production campaigns
        # existed). Changing it again invalidates every existing run dir.
        assert compute_fit_id('run', 1, 0, 0, 0, 0, 0) == 'cd30f915609fe425'
        assert compute_fit_id('run', 1, 0, 0, 0, 0, 0, 0) == 'cd30f915609fe425'

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
        d['population']['shear'] = {
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
        d['run']['noise_reps'] = 3
        spec = EnsembleSpec.from_yaml(_write_spec(tmp_path, d))
        m = build_manifest(spec, canonical_q)
        for _, group in m.groupby(['cosi_bin', 'galaxy_id']):
            assert group['noise_seed'].nunique() == 3

    def test_ring_pairs(self, tmp_path, canonical_q):
        d = _spec_dict()
        d['population']['ring'] = {'enabled': True}
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
        assert len(record['observation_config_hash']) == 64
        spec, config, manifest = load_run(run_dir)
        assert len(manifest) == 4
        # snapshot, not live registry: hash matches the copied file
        assert config.content_hash == record['observation_config_hash']

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

    def test_write_collated_table_carries_priors(self, run_dir):
        from kl_pipe.ensemble.collate import collate_results, write_collated_table

        _, _, manifest = load_run(run_dir)
        for _, row in manifest.iterrows():
            pd.DataFrame(
                [
                    {
                        'fit_id': row['fit_id'],
                        'status': 'succeeded',
                        'post.g1.mean': 0.05,
                        'has_chains': True,
                        # flat prior columns as the worker emits them
                        'prior.g1.dist': 'gaussian',
                        'prior.g1.loc': 0.0,
                        'prior.g1.scale': 0.2,
                        'prior.g1.low': None,
                        'prior.g1.high': None,
                    }
                ]
            ).to_parquet(run_dir / 'results' / f"{row['fit_id']}.parquet", index=False)
        collate_results(run_dir)
        out_path = write_collated_table(run_dir)

        # self-identifying filename carries the run name
        assert out_path.name == f'{run_dir.name}_collated.parquet'
        assert out_path.exists()
        table = pd.read_parquet(out_path)
        assert len(table) == len(manifest)
        # truth (join) + fitted + priors + chains flag all present in one table
        assert 'truth.g1' in table.columns
        assert 'post.g1.mean' in table.columns
        assert 'has_chains' in table.columns
        assert table['prior.g1.dist'].iloc[0] == 'gaussian'
        assert table['prior.g1.scale'].iloc[0] == 0.2

    def test_scene_priors_describe_matches_config(self, dev_spec, canonical_q):
        # the production shear prior serializes to the configured width
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
        desc = scene_priors(truth, canonical_q, dev_spec).describe()
        assert desc['g1']['dist'] == 'gaussian'
        assert desc['g1']['loc'] == 0.0
        assert desc['g1']['scale'] == dev_spec.shear_fit_prior_sigma
        # a truth-pinned param is recorded as fixed
        assert desc['z']['dist'] == 'fixed'


# ==============================================================================
# Config-sweep axis (emission-line SNR)
# ==============================================================================

GSNR_SPEC = REPO_ROOT / 'configs' / 'ensembles' / 'sigma_eps_gsnr_dev.yaml'


class TestLineSnrSweep:
    @pytest.fixture(scope='class')
    def gsnr_spec(self) -> EnsembleSpec:
        return EnsembleSpec.from_yaml(GSNR_SPEC)

    def test_spec_loads(self, gsnr_spec):
        assert gsnr_spec.measurement == 'sigma_eps_vs_line_snr'
        assert gsnr_spec.sweep_values == (30.0, 50.0, 100.0)
        assert gsnr_spec.n_fits == 6
        assert 'cosi' in gsnr_spec.draw

    def test_top_level_line_snr_conflict(self, tmp_path, gsnr_spec):
        d = yaml.safe_load(GSNR_SPEC.read_text())
        d['observation']['snr']['line'] = 30
        with pytest.raises(ValueError, match='conflicts with the'):
            EnsembleSpec.from_yaml(_write_spec(tmp_path, d))

    def test_cosi_must_be_drawn(self, tmp_path):
        d = yaml.safe_load(GSNR_SPEC.read_text())
        del d['population']['draw']['cosi']
        with pytest.raises(ValueError, match='cosi in population.draw'):
            EnsembleSpec.from_yaml(_write_spec(tmp_path, d))

    def test_crn_across_sweep(self, gsnr_spec, canonical_q):
        """Same galaxy: identical truth + noise seed at every SNR step."""
        m = build_manifest(gsnr_spec, canonical_q)
        assert len(m) == 6
        assert sorted(m['line_snr'].unique()) == [30.0, 50.0, 100.0]
        assert (m['broadband_snr'] == 100.0).all()
        for _, group in m.groupby('galaxy_id'):
            assert len(group) == 3  # one row per SNR step
            assert group['noise_seed'].nunique() == 1  # CRN
            for col in ('truth.cosi', 'truth.theta_int', 'truth.vel.vcirc', 'truth.z'):
                assert group[col].nunique() == 1  # same galaxy
        # distinct galaxies still get independent noise
        assert m.groupby('noise_seed').ngroups == 2

    def test_cosi_drawn_prior(self, gsnr_spec, canonical_q):
        """Drawn cosi carries its generating distribution as the fit prior."""
        truth = scene_truth_defaults(canonical_q, gsnr_spec.fixed)
        truth.update(
            {
                'cosi': 0.5,
                'theta_int': 1.0,
                'g1': 0.05,
                'g2': 0.05,
                'vel.vcirc': 200.0,
                'z': 1.2,
            }
        )
        priors = scene_priors(truth, canonical_q, gsnr_spec)
        assert isinstance(priors.get_prior('cosi'), Uniform)
        assert priors.get_prior('cosi').bounds == (0.1, 1.0)

    def test_axis_helper(self, gsnr_spec, dev_spec):
        from kl_pipe.ensemble.diagnostics import measurement_axis

        assert measurement_axis(gsnr_spec)[:2] == ('sweep_step', 'line_snr')
        assert measurement_axis(dev_spec)[:2] == ('cosi_bin', 'truth.cosi')


# ==============================================================================
# Emission-line SNR convention
# ==============================================================================


def test_grism_noise_is_line_normalized(dev_spec, canonical_q):
    """Grism noise variance is the emission-LINE matched filter
    (var = ||line||^2 / line_snr^2), NOT the continuum-inflated whole stamp,
    and scales as 1/line_snr^2. Guards the line-SNR convention wiring so a
    regression back to whole-stamp normalization fails loudly. Deterministic
    (no sampling, no seed dependence)."""
    truth = scene_truth_defaults(canonical_q, dev_spec.fixed)
    truth.update(
        {
            'cosi': 0.5,
            'theta_int': 0.6,
            'g1': 0.02,
            'g2': -0.01,
            'vel.vcirc': 200.0,
            'z': 1.2,
        }
    )
    line_truth = {
        k: (0.0 if k.endswith('.cont.flux_per_nm') else v) for k, v in truth.items()
    }

    def first_grism_var(line_snr):
        inp = build_fit_inputs(
            truth, 12345, dev_spec, canonical_q, broadband_snr=100.0, line_snr=line_snr
        )
        obs = next(iter(inp.grism_obs.values()))
        var = float(np.asarray(obs.variance))
        full = np.asarray(inp.source.render_grism(truth, obs))
        line = np.asarray(inp.source.render_grism(line_truth, obs))
        return var, float(np.sum(full**2)), float(np.sum(line**2))

    var40, full_power, line_power = first_grism_var(40.0)
    # normalized on the LINE, not the whole stamp
    assert var40 == pytest.approx(line_power / 40.0**2, rel=1e-6)
    # continuum is present, so the line carries strictly less power than the
    # whole stamp -- i.e. this is demonstrably NOT the whole-stamp convention
    assert line_power < full_power

    var80, _, _ = first_grism_var(80.0)
    # matched-filter SNR: doubling line_snr quarters the variance
    assert var80 == pytest.approx(var40 / 4.0, rel=1e-6)


@pytest.mark.slow
def test_shear_information_increases_with_line_snr():
    """Data-only galaxy-frame shear widths (sigma_g+, sigma_gx) strictly
    decrease as emission-line SNR rises: the line-SNR knob controls the
    kinematic shear information. Fisher-based (likelihood curvature at truth) --
    fast and deterministic, no MCMC. Config P (multi-roll) so the rotational
    component gx is not degenerate with theta_int. No tolerance is tuned; the
    assertion is pure monotonicity (more information never widens a posterior)."""
    spec = EnsembleSpec.from_yaml(P_GSNR_SPEC)
    config = ObservationConfig.from_yaml(REGISTRY / 'canonical_P.yaml')
    truth = scene_truth_defaults(config, spec.fixed)
    truth.update(
        {
            'cosi': 0.5,
            'theta_int': 0.6,
            'g1': 0.0,
            'g2': 0.0,
            'vel.vcirc': 200.0,
            'z': 1.2,
        }
    )
    sub = [
        'g1',
        'g2',
        'cosi',
        'theta_int',
        'vel.vcirc',
        'vel.rscale',
        'Halpha.flux',
        'Halpha.rscale',
        'Halpha.dispersion',
        'F087.rscale',
        'F158.rscale',
    ]
    step = {
        'g1': 0.004,
        'g2': 0.004,
        'cosi': 0.004,
        'theta_int': 0.01,
        'vel.vcirc': 2.0,
        'vel.rscale': 0.01,
        'Halpha.flux': 1.0,
        'Halpha.rscale': 0.004,
        'Halpha.dispersion': 1.0,
        'F087.rscale': 0.004,
        'F158.rscale': 0.004,
    }

    def galaxy_frame_shear_sigmas(line_snr):
        inp = build_fit_inputs(
            truth, 7, spec, config, broadband_snr=100.0, line_snr=line_snr
        )
        s = inp.source
        chans = [
            (
                lambda th, o=o, b=b: np.asarray(s.render_broadband(th, o, b)).ravel(),
                float(np.asarray(o.variance)),
            )
            for b, o in inp.image_obs.items()
        ]
        chans += [
            (
                lambda th, o=o: np.asarray(s.render_grism(th, o)).ravel(),
                float(np.asarray(o.variance)),
            )
            for o in inp.grism_obs.values()
        ]
        th0 = {
            **inp.priors.fixed_values,
            **{n: truth[n] for n in inp.priors.sampled_names},
        }

        def fv(th):
            return np.concatenate([f(th) for f, _ in chans])

        var = np.concatenate([np.full(f(th0).size, v) for f, v in chans])
        jac = np.array(
            [
                (fv({**th0, n: th0[n] + step[n]}) - fv({**th0, n: th0[n] - step[n]}))
                / (2 * step[n])
                for n in sub
            ]
        )
        fisher = (jac / var) @ jac.T
        idx = {n: i for i, n in enumerate(sub)}
        # fold in only the NON-shear priors (TF + nuisances) so we isolate the
        # DATA constraint on shear (shear prior OFF)
        prior_prec = np.zeros(len(sub))
        for i, n in enumerate(sub):
            if n in ('g1', 'g2'):
                continue
            p = inp.priors.get_prior(n)
            cls = type(p).__name__
            if cls in ('Gaussian', 'TruncatedNormal'):
                prior_prec[i] = 1.0 / p.sigma**2
            elif cls == 'LogNormal':
                prior_prec[i] = 1.0 / (200.0 * np.log(10.0) * 0.08) ** 2
        cov = np.linalg.inv(fisher + np.diag(prior_prec))
        i1, i2, ti = idx['g1'], idx['g2'], truth['theta_int']
        cov_g = np.array([[cov[i1, i1], cov[i1, i2]], [cov[i2, i1], cov[i2, i2]]])
        c2, s2 = np.cos(2 * ti), np.sin(2 * ti)
        rot = np.array([[c2, s2], [-s2, c2]])
        cov_gal = rot @ cov_g @ rot.T
        return np.sqrt(cov_gal[0, 0]), np.sqrt(cov_gal[1, 1])

    sig = [galaxy_frame_shear_sigmas(ls) for ls in (30.0, 60.0, 120.0)]
    sig_gplus = [x[0] for x in sig]
    sig_gcross = [x[1] for x in sig]
    assert (
        sig_gplus[0] > sig_gplus[1] > sig_gplus[2]
    ), f"sigma_g+ not decreasing: {sig_gplus}"
    assert (
        sig_gcross[0] > sig_gcross[1] > sig_gcross[2]
    ), f"sigma_gx not decreasing: {sig_gcross}"


# ==============================================================================
# PA-stratified MAP starts + dispatch emission
# ==============================================================================


class TestPAStratifiedStarts:
    def test_grid_covers_prior_range(self, dev_spec, canonical_q):
        from kl_pipe.ensemble.worker import _pa_stratified_starts

        truth = scene_truth_defaults(canonical_q, dev_spec.fixed)
        truth.update(
            {
                'cosi': 0.5,
                'theta_int': 1.0,
                'g1': 0.05,
                'g2': 0.05,
                'vel.vcirc': 200.0,
                'z': 1.2,
            }
        )
        priors = scene_priors(truth, canonical_q, dev_spec)
        starts = _pa_stratified_starts(priors, seed=7, n_pa=4)
        names = list(priors.sampled_names)
        thetas = starts[:, names.index('theta_int')]
        # bin centers of a 4-way split of U(0, pi)
        assert np.allclose(thetas, (np.arange(4) + 0.5) * np.pi / 4)
        # non-theta columns are in-support prior draws
        for i, name in enumerate(names):
            lo, hi = priors.get_prior(name).bounds
            if lo is not None:
                assert (starts[:, i] >= lo).all()
            if hi is not None:
                assert (starts[:, i] <= hi).all()

    def test_n_map_starts_spec_knob(self, tmp_path):
        d = _spec_dict()
        d['fit']['n_map_starts'] = 8
        spec = EnsembleSpec.from_yaml(_write_spec(tmp_path, d))
        assert spec.n_map_starts == 8


class TestSlurmEmission:
    def test_gpu_packing_script(self, tmp_path):
        from kl_pipe.ensemble.dispatch import emit_slurm_script

        d = _spec_dict()
        d['dispatch'].update(
            {'backend': 'slurm', 'workers_per_node': 8, 'account': 'TEST-ALLOC'}
        )
        spec_path = _write_spec(tmp_path, d)
        run_dir = expand(spec_path, REGISTRY, tmp_path / 'runs')
        script = emit_slurm_script(run_dir).read_text()
        assert '#SBATCH -A TEST-ALLOC' in script
        # 0.75/N (not 0.85/N): headroom for CUDA contexts/kernel modules --
        # 0.85/N OOMed the first packed P ensemble run on GH200
        assert f'XLA_PYTHON_CLIENT_MEM_FRACTION={round(0.75 / 8, 3)}' in script
        assert 'nvidia-cuda-mps-control' in script
        assert 'KLPIPE_PYTHON' in script
        assert 'seq 0 7' in script

    def test_single_worker_no_mem_slice(self, tmp_path):
        from kl_pipe.ensemble.dispatch import emit_slurm_script

        run_dir = expand(DEV_SPEC, REGISTRY, tmp_path / 'runs')
        script = emit_slurm_script(run_dir).read_text()
        assert 'XLA_PYTHON_CLIENT_MEM_FRACTION=' not in script


# ==============================================================================
# Diagnostics
# ==============================================================================


def _write_fake_results(run_dir):
    """Synthetic summary rows: 3 healthy fits + 1 catastrophic."""
    rng = np.random.default_rng(0)
    _, _, manifest = load_run(run_dir)
    for i, (_, row) in enumerate(manifest.iterrows()):
        broken = i == 0
        summary = {
            'fit_id': row['fit_id'],
            'status': 'succeeded',
            'max_rhat': 5.0 if broken else 1.005,
            'min_ess': 2.0 if broken else 800.0,
            'n_divergences': 4000 if broken else 40,
            'divergence_rate': 1.0 if broken else 0.01,
            'mean_accept_prob': 0.85,
            'n_map_starts_converged': 4,
            'precond_condition_number': 100.0,
            'fit_wallclock_s': 100.0,
            'precond_wallclock_s': 5.0,
        }
        for p in ('g1', 'g2', 'cosi', 'theta_int', 'vel.vcirc'):
            std = 0.03 if p.startswith('g') else 0.1
            summary[f'post.{p}.mean'] = row[f'truth.{p}'] + rng.normal(0, std)
            summary[f'post.{p}.std'] = std
            summary[f'post.{p}.median'] = summary[f'post.{p}.mean']
        pd.DataFrame([summary]).to_parquet(
            run_dir / 'results' / f"{row['fit_id']}.parquet", index=False
        )


class TestDiagnostics:
    def test_tables(self, run_dir):
        from kl_pipe.ensemble.collate import collate_results
        from kl_pipe.ensemble.diagnostics import (
            augment_galaxy_frame,
            pull_table,
            quality_table,
            sigma_eps_table,
        )

        _write_fake_results(run_dir)
        collate_results(run_dir)
        from kl_pipe.ensemble.collate import analysis_table

        table = augment_galaxy_frame(run_dir, analysis_table(run_dir))

        q = quality_table(table)
        assert q['catastrophic'].sum() == 1
        assert q['low_quality'].sum() >= 1
        # the catastrophic fit is always low_quality too
        assert q.loc[q['catastrophic'], 'low_quality'].all()

        pulls = pull_table(table)
        good = pulls[~table['max_rhat'].gt(1.1).values]
        # healthy synthetic fits have pulls of order unity; shear reported in
        # the galaxy frame (g+, gx), not sky-frame g1/g2
        assert np.abs(good[['pull.g_plus', 'pull.g_cross']].values).max() < 5

        sig = sigma_eps_table(run_dir, table)
        excl = sig[sig['gate'] == 'exclude_catastrophic']
        head = excl[excl['axis_step'] == -1]
        assert head['n_fits'].iloc[0] == 3  # catastrophic fit masked
        # no chains saved -> marginal approx; widths are ~0.03 by
        # construction, so sigma_eps must land near 0.03
        assert 0.02 < head['sigma_eps'].iloc[0] < 0.04
