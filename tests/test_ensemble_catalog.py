"""
Catalog-population integration tests: population -> manifest -> priors ->
worker (kl_pipe/ensemble/ catalog mode).

The CI-safe tier runs on the synthetic Flagship2-schema catalog generator
from tests/test_population.py. The real-data tier (marked cosmohub + slow)
expands the example dev spec against the downloaded flagship2_dev catalog
and executes one full fit end-to-end.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from kl_pipe.ensemble.expander import (
    build_manifest,
    expand,
    load_run,
    truth_from_row,
)
from kl_pipe.ensemble.mocks import build_fit_inputs
from kl_pipe.ensemble.scene import scene_priors
from kl_pipe.ensemble.spec import EnsembleSpec, ObservationConfig
from kl_pipe.priors import LogNormal, TruncatedNormal, Uniform

from test_population import (
    catalog_spec_dict,
    fake_flagship2_rows,
    spec_from_dict,
    write_fake_catalog,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = REPO_ROOT / 'configs' / 'observation'
EXAMPLE_SPEC = REPO_ROOT / 'configs' / 'ensembles' / 'flagship2_shear_dev.yaml'
DATA_DIR = REPO_ROOT / 'data' / 'cosmohub'
DEV_PARQUET = DATA_DIR / 'flagship2_dev.parquet'


def _small_spec_dict(data_dir: Path) -> dict:
    """4 galaxies x 2 ring members x 1 noise rep = 8 fits."""
    d = catalog_spec_dict(data_dir)
    d['population']['sample']['n_galaxies'] = 4
    return d


@pytest.fixture(scope='module')
def fake_data_dir(tmp_path_factory) -> Path:
    data_dir = tmp_path_factory.mktemp('cosmohub_fake_e2e')
    write_fake_catalog(data_dir, fake_flagship2_rows(n=400, seed=1234))
    return data_dir


@pytest.fixture(scope='module')
def catalog_run(fake_data_dir, tmp_path_factory) -> Path:
    """Expanded run dir for the small catalog spec."""
    tmp = tmp_path_factory.mktemp('cat_run')
    spec_path = tmp / 'spec.yaml'
    spec_path.write_text(yaml.safe_dump(_small_spec_dict(fake_data_dir)))
    return expand(spec_path, REGISTRY, tmp / 'runs')


@pytest.fixture(scope='module')
def run_parts(catalog_run):
    spec, config, manifest = load_run(catalog_run)
    population = pd.read_parquet(catalog_run / 'population.parquet')
    return spec, config, manifest, population


# ==============================================================================
# Expansion: population + manifest artifacts
# ==============================================================================


class TestCatalogExpand:
    def test_run_dir_artifacts(self, catalog_run, run_parts):
        _, _, manifest, population = run_parts
        pop_parquet = catalog_run / 'population.parquet'
        assert pop_parquet.exists()
        assert (catalog_run / 'population_meta.json').exists()
        assert (catalog_run / 'manifest.parquet').exists()
        assert len(population) == 4
        assert len(manifest) == 8  # 4 galaxies x 2 ring members x 1 rep
        assert manifest['fit_id'].is_unique

        record = json.loads((catalog_run / 'provenance' / 'expansion.json').read_text())
        assert record['n_fits'] == 8
        assert (
            record['population_sha256']
            == hashlib.sha256(pop_parquet.read_bytes()).hexdigest()
        )
        assert record['population_stage_counts']['n_sampled'] == 4
        assert record['population_stage_counts']['n_raw'] == 400

    def test_ring_pairs(self, run_parts):
        _, _, manifest, _ = run_parts
        by_id = manifest.set_index('fit_id')
        for _, row in manifest.iterrows():
            partner = by_id.loc[row['ring_partner_id']]
            assert partner['ring_partner_id'] == row['fit_id']  # reciprocal
            dtheta = (partner['truth.theta_int'] - row['truth.theta_int']) % np.pi
            assert np.isclose(dtheta, np.pi / 2)
            # pair-shared shear + identical intrinsics, independent noise
            assert partner['truth.g1'] == row['truth.g1']
            assert partner['truth.g2'] == row['truth.g2']
            assert partner['truth.vel.vcirc'] == row['truth.vel.vcirc']
            assert partner['truth.z'] == row['truth.z']
            assert partner['noise_seed'] != row['noise_seed']

    def test_line_snr_is_population_snr_line(self, run_parts):
        # per-fit line SNR = the population's PHYSICAL per-exposure
        # matched-filter snr_line, not the (rejected) observation.snr.line
        _, _, manifest, population = run_parts
        pop = population.set_index('pop_index')
        for _, row in manifest.iterrows():
            expected = float(pop.loc[row['galaxy_id'], 'snr_line'])
            assert row['line_snr'] == expected
            assert row['pop.snr_line'] == expected

    def test_broadband_snr_is_spec_scalar(self, run_parts):
        spec, _, manifest, _ = run_parts
        assert (manifest['broadband_snr'] == spec.broadband_snr).all()

    def test_continuum_ew_conversion(self, run_parts):
        # flux_per_nm = line_flux / EW_obs with EW_obs [nm] =
        # ew_rest_a [A] * (1 + z) / 10; hand-computed per row
        _, _, manifest, _ = run_parts
        row = manifest.iloc[0]
        ew_obs_nm = row['pop.ew_rest_a'] * (1.0 + row['truth.z']) / 10.0
        expected = row['truth.Halpha.flux'] / ew_obs_nm
        assert row['truth.Halpha.cont.flux_per_nm'] == pytest.approx(
            expected, rel=1e-12
        )

    def test_rscale_sets_all_spatial_scales(self, run_parts):
        _, config, manifest, population = run_parts
        pop = population.set_index('pop_index')
        for _, row in manifest.iterrows():
            rscale = float(pop.loc[row['galaxy_id'], 'rscale_arcsec'])
            for comp in tuple(config.bands) + ('Halpha', 'Halpha.cont', 'vel'):
                assert row[f'truth.{comp}.rscale'] == rscale

    def test_kinematic_truths_from_population(self, run_parts):
        _, _, manifest, population = run_parts
        pop = population.set_index('pop_index')
        for _, row in manifest.iterrows():
            g = pop.loc[row['galaxy_id']]
            assert row['truth.vel.vcirc'] == float(g['vcirc_kms'])
            assert row['truth.Halpha.dispersion'] == float(g['sigma0_kms'])
            assert row['truth.cosi'] == float(g['cosi'])
            assert row['truth.z'] == float(g['z'])

    def test_determinism_double_expand(self, fake_data_dir, tmp_path, run_parts):
        _, _, manifest1, population1 = run_parts
        spec_path = tmp_path / 'spec.yaml'
        spec_path.write_text(yaml.safe_dump(_small_spec_dict(fake_data_dir)))
        run_dir2 = expand(spec_path, REGISTRY, tmp_path / 'runs')
        manifest2 = pd.read_parquet(run_dir2 / 'manifest.parquet')
        population2 = pd.read_parquet(run_dir2 / 'population.parquet')
        assert (
            pd.util.hash_pandas_object(manifest1).to_numpy()
            == pd.util.hash_pandas_object(manifest2).to_numpy()
        ).all()
        pd.testing.assert_frame_equal(population1, population2, check_exact=True)

    def test_build_manifest_requires_population(self, run_parts):
        spec, config, _, _ = run_parts
        with pytest.raises(ValueError, match='population table'):
            build_manifest(spec, config)

    def test_sampled_rejects_population(self, tmp_path):
        sampled_spec = EnsembleSpec.from_yaml(
            REPO_ROOT / 'configs' / 'ensembles' / 'sigma_eps_cosi_dev.yaml'
        )
        config = ObservationConfig.from_yaml(REGISTRY / 'canonical_Q.yaml')
        with pytest.raises(ValueError, match='catalog-mode only'):
            build_manifest(sampled_spec, config, population=pd.DataFrame())

    def test_snr_line_scalar_rejected(self, fake_data_dir, tmp_path):
        d = _small_spec_dict(fake_data_dir)
        d['observation']['snr']['line'] = 100
        with pytest.raises(ValueError, match='snr.line is not valid'):
            spec_from_dict(tmp_path, d)


# ==============================================================================
# Catalog-mode fit priors
# ==============================================================================


class TestCatalogPriors:
    def test_vcirc_prior_observable_conditioned(self, run_parts):
        # prior center = TFR at the NOISY logm_obs (pop.prior_vcirc_mu_kms),
        # NOT the fit's truth vcirc -- mis-centered by construction
        spec, config, manifest, _ = run_parts
        medians, truths = [], []
        for _, row in manifest.iterrows():
            truth = truth_from_row(row)
            priors = scene_priors(truth, config, spec, row=row)
            vc = priors.get_prior('vel.vcirc')
            assert isinstance(vc, LogNormal)
            assert vc.median == pytest.approx(
                float(row['pop.prior_vcirc_mu_kms']), rel=1e-9
            )
            assert vc.sigma == pytest.approx(
                float(row['pop.prior_vcirc_sigma_dex']) * np.log(10.0), rel=1e-9
            )
            medians.append(vc.median)
            truths.append(truth['vel.vcirc'])
        # mis-centering is real: with 0.25 dex logm_obs scatter the prior
        # centers must not track truth (deterministic on the fixed seed)
        assert np.max(np.abs(np.log10(np.array(medians) / np.array(truths)))) > 0.01

    def test_orientation_and_z_priors(self, run_parts):
        spec, config, manifest, _ = run_parts
        row = manifest.iloc[0]
        truth = truth_from_row(row)
        priors = scene_priors(truth, config, spec, row=row)
        cosi = priors.get_prior('cosi')
        assert isinstance(cosi, Uniform)
        assert cosi.bounds == spec.catalog_population.cosi_range == (0.05, 0.95)
        theta = priors.get_prior('theta_int')
        assert isinstance(theta, Uniform)
        assert theta.bounds == (0.0, np.pi)
        # z pinned to the per-fit truth
        assert priors.fixed_values['z'] == truth['z']

    def test_dispersion_population_prior(self, run_parts):
        # self-consistent with the paint: TruncatedNormal(intercept +
        # slope*z, scatter, min, 150) -- Ubler+2019 values from the spec
        spec, config, manifest, _ = run_parts
        cp = spec.catalog_population
        row = manifest.iloc[0]
        truth = truth_from_row(row)
        priors = scene_priors(truth, config, spec, row=row)
        disp = priors.get_prior('Halpha.dispersion')
        assert isinstance(disp, TruncatedNormal)
        assert disp.mu == pytest.approx(
            cp.sigma0_intercept_kms + cp.sigma0_slope_kms * truth['z'], rel=1e-12
        )
        assert disp.sigma == cp.sigma0_scatter_kms
        assert disp.bounds == (cp.sigma0_min_kms, 150.0)

    def test_all_truths_in_prior_support(self, run_parts):
        # every sampled parameter's truth has finite prior log-prob; a truth
        # outside support (e.g. catalog rscale beyond the bounds) fails here
        spec, config, manifest, _ = run_parts
        for _, row in manifest.iterrows():
            truth = truth_from_row(row)
            priors = scene_priors(truth, config, spec, row=row)
            for name in priors.sampled_names:
                lp = float(priors.get_prior(name).log_prob(truth[name]))
                assert np.isfinite(lp), f'{name}: truth {truth[name]} out of support'

    def test_row_required_in_catalog_mode(self, run_parts):
        spec, config, manifest, _ = run_parts
        truth = truth_from_row(manifest.iloc[0])
        with pytest.raises(ValueError, match='manifest row'):
            scene_priors(truth, config, spec)
        with pytest.raises(ValueError, match='manifest row'):
            build_fit_inputs(
                truth,
                12345,
                spec,
                config,
                broadband_snr=300.0,
                line_snr=float(manifest.iloc[0]['line_snr']),
            )


# ==============================================================================
# Real-data tier (cosmohub): expand + worker log-posterior smoke; the full
# single-fit end-to-end is kept but additionally marked slow (no sampling in
# routine runs)
# ==============================================================================


@pytest.fixture(scope='module')
def real_run_dir(tmp_path_factory) -> Path:
    """Expanded run dir for the example dev spec on the real catalog.

    Shared by the smoke gate and the full-fit tier; carries tiny sampler
    settings so the (opt-in) full fit is a wiring shakedown, not a
    convergence-quality fit.
    """
    tmp = tmp_path_factory.mktemp('real_e2e')
    d = yaml.safe_load(EXAMPLE_SPEC.read_text())
    d['population']['catalog']['data_dir'] = str(DATA_DIR)
    d['fit'].update(
        {'n_warmup': 50, 'n_samples': 50, 'n_chains': 1, 'precondition': 'none'}
    )
    d['output'] = {'save_chains': 'none', 'save_mocks': 'none'}
    spec_path = tmp / 'spec.yaml'
    spec_path.write_text(yaml.safe_dump(d))
    return expand(spec_path, REGISTRY, tmp / 'runs')


@pytest.mark.cosmohub
@pytest.mark.skipif(
    not DEV_PARQUET.exists(),
    reason="flagship2_dev.parquet absent (make download-cosmohub-dev)",
)
class TestRealCatalog:
    def test_worker_log_posterior_smoke(self, real_run_dir):
        """Cheap wiring gate on the real dev catalog: one manifest row
        through build_fit_inputs and the worker's log-posterior. Asserts a
        finite log-prob AND a finite gradient at truth -- no sampling."""
        import jax.numpy as jnp

        from kl_pipe.sampling import InferenceTask

        spec, config, manifest = load_run(real_run_dir)
        assert (real_run_dir / 'population.parquet').exists()
        assert len(manifest) == 16  # 8 galaxies x 2 ring members x 1 rep

        row = manifest.iloc[0]
        truth = truth_from_row(row)
        inputs = build_fit_inputs(
            truth,
            int(row['noise_seed']),
            spec,
            config,
            broadband_snr=float(row['broadband_snr']),
            line_snr=float(row['line_snr']),
            row=row,
        )
        # same task construction as worker._run_fit_attempt
        task = InferenceTask.from_obs(
            inputs.source,
            inputs.priors,
            image_obs=inputs.image_obs,
            grism_obs=inputs.grism_obs,
        )
        theta = jnp.array([truth[n] for n in inputs.priors.sampled_names])
        log_prob, grad = task.get_log_posterior_and_grad_fn()(theta)
        assert np.isfinite(float(log_prob)), f'log_prob at truth: {log_prob}'
        grad = np.asarray(grad)
        assert np.isfinite(grad).all(), (
            f'non-finite gradient components: '
            f'{[n for n, g in zip(inputs.priors.sampled_names, grad) if not np.isfinite(g)]}'
        )

    @pytest.mark.slow
    def test_expand_and_single_fit(self, real_run_dir):
        """Full single-fit end-to-end (expand -> worker -> results parquet).

        Deliberately excluded from routine runs (also marked slow): the
        infra phase gates on the no-sampling smoke above; run this
        explicitly with -m "cosmohub and slow" when a full NUTS shakedown
        is wanted."""
        from kl_pipe.ensemble.__main__ import main

        _, _, manifest = load_run(real_run_dir)
        rc = main(['run', '--run-dir', str(real_run_dir), '--max-fits', '1'])
        assert rc == 0
        results = list((real_run_dir / 'results').glob('*.parquet'))
        assert len(results) == 1
        summary = pd.read_parquet(results[0]).iloc[0]
        assert summary['status'] == 'succeeded'
        assert summary['fit_id'] in set(manifest['fit_id'])
        print(
            f"end-to-end fit: wallclock {summary['fit_wallclock_s']:.1f} s, "
            f"max_rhat {summary['max_rhat']:.3f}, "
            f"line_snr {float(manifest.set_index('fit_id').loc[summary['fit_id'], 'line_snr']):.1f}"
        )
