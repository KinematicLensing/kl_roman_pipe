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
from kl_pipe.ensemble.population import CONT_CENTROID_OFFSET_ARCSEC
from kl_pipe.ensemble.scene import scene_priors
from kl_pipe.ensemble.spec import EnsembleSpec, ObservationConfig
from kl_pipe.priors import (
    LogNormal,
    TruncatedLogNormal,
    TruncatedNormal,
    Uniform,
)


from test_population import (
    catalog_spec_dict,
    fake_flagship2_rows,
    spec_from_dict,
    write_fake_catalog,
)


def prior_logp_at_truth(priors, truth, name):
    """Prior log-prob of one parameter at its truth.

    A conditional prior has no density without its parent, so resolve the
    parent's truth for it. Per-parameter rather than the joint log-prior so a
    failure names the offending parameter.
    """
    prior = priors.get_prior(name)
    parent = priors.conditional_parents.get(name)
    if parent is None:
        return float(prior.log_prob(truth[name]))
    return float(prior.log_prob_given(truth[name], truth[parent]))


REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = REPO_ROOT / 'configs' / 'observation'
EXAMPLE_SPEC = REPO_ROOT / 'configs' / 'ensembles' / 'flagship2_shear_dev.yaml'
CENSUS_SPEC = REPO_ROOT / 'configs' / 'ensembles' / 'flagship2_shear_census_v1.yaml'
NOBULGE_CENSUS_SPEC = (
    REPO_ROOT / 'configs' / 'ensembles' / 'flagship2_shear_census_v1_nobulge.yaml'
)
DATA_DIR = REPO_ROOT / 'data' / 'cosmohub'
DEV_PARQUET = DATA_DIR / 'flagship2_dev.parquet'


def _small_spec_dict(data_dir: Path) -> dict:
    """4 galaxies x 2 ring members x 1 noise rep = 8 fits."""
    d = catalog_spec_dict(data_dir)
    d['population']['sample']['n_galaxies'] = 4
    return d


def _nobulge_spec_dict(data_dir: Path) -> dict:
    """The small spec with the bulge paint disabled (disk-only twin)."""
    d = _small_spec_dict(data_dir)
    d['run']['name'] = 'pop_test_nobulge'
    d['population']['paint']['bulge'] = False
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

    def test_line_snr_is_population_snr_line_per_pass(self, run_parts):
        # per-fit line SNR = the population's PHYSICAL PER-PASS matched-filter
        # SNR, not the (rejected) observation.snr.line and not the coadded
        # snr_line_total: the mock noise is drawn once per grism roll, so each
        # roll carries one pass's depth
        _, _, manifest, population = run_parts
        pop = population.set_index('pop_index')
        for _, row in manifest.iterrows():
            expected = float(pop.loc[row['galaxy_id'], 'snr_line_per_pass'])
            assert row['line_snr'] == expected
            assert row['pop.snr_line_per_pass'] == expected

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
            g = pop.loc[row['galaxy_id']]
            rscale = float(g['rscale_arcsec'])
            # broadband bands are BulgeDiskModel: the disk scale length is
            # disk_rscale, and the bulge carries the catalog fraction + size
            for band in config.bands:
                assert row[f'truth.{band}.disk_rscale'] == rscale
                assert row[f'truth.{band}.bulge_frac'] == float(g['bulge_fraction'])
                assert row[f'truth.{band}.bulge_hlr'] == float(g['bulge_r50_arcsec'])
            # the continuum under the line is the same stellar disk, so it
            # shares the catalog scale exactly; the line and the rotation
            # curve carry their painted ratios to it
            assert row['truth.Halpha.cont.rscale'] == rscale
            assert row['truth.Halpha.rscale'] == rscale * float(
                g['halpha_rscale_ratio']
            )
            assert row['truth.vel.rscale'] == rscale * float(g['vel_rscale_ratio'])

    def test_painted_ratios_follow_the_literature_distributions(self, run_parts):
        # the paint constants are the same objects the fit priors are built
        # from, so a drift here would silently desynchronise prior and truth
        from kl_pipe.ensemble.population import (
            HALPHA_RSCALE_RATIO_MEDIAN,
            VEL_RSCALE_RATIO_MEDIAN,
            V0_SCATTER_KMS,
        )

        _, _, _, population = run_parts
        vel = population['vel_rscale_ratio'].to_numpy()
        line = population['halpha_rscale_ratio'].to_numpy()
        v0 = population['v0_kms'].to_numpy()
        assert (vel > 0).all() and (line > 0).all()
        # loose bracket: 4 galaxies cannot pin a median, only catch a wrong
        # centre or a wrong units convention
        assert 0.1 < np.median(vel) < 1.6
        assert 0.4 < np.median(line) < 4.0
        assert np.median(vel) < np.median(line)  # turnover inside the line
        assert abs(np.mean(v0)) < 5 * V0_SCATTER_KMS
        assert VEL_RSCALE_RATIO_MEDIAN < HALPHA_RSCALE_RATIO_MEDIAN

    def test_component_centroids_are_drawn_not_pinned(self, run_parts):
        # every component keeps its own registration offset; pinning them all
        # at zero would leave the fit prior centred exactly on truth
        _, config, manifest, _ = run_parts
        row = manifest.iloc[0]
        offsets = [
            row[f'truth.{comp}.{axis}']
            for comp in list(config.bands) + ['Halpha']
            for axis in ('x0', 'y0')
        ]
        assert any(o != 0.0 for o in offsets)
        assert len(set(offsets)) == len(offsets)
        # the continuum sits near the line but not on it: clumpy star
        # formation need not centre on the older stellar disk, so the offset
        # is a distinct draw bounded by its own scatter
        for axis in ('x0', 'y0'):
            delta = abs(row[f'truth.Halpha.cont.{axis}'] - row[f'truth.Halpha.{axis}'])
            assert delta > 0.0
            assert delta < 5.0 * CONT_CENTROID_OFFSET_ARCSEC

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
                lp = prior_logp_at_truth(priors, truth, name)
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
# No-bulge catalog mode (paint.bulge: false -- disk-only twin)
# ==============================================================================


@pytest.fixture(scope='module')
def nobulge_run(fake_data_dir, tmp_path_factory) -> Path:
    """Expanded run dir for the small no-bulge catalog spec."""
    tmp = tmp_path_factory.mktemp('cat_run_nobulge')
    spec_path = tmp / 'spec.yaml'
    spec_path.write_text(yaml.safe_dump(_nobulge_spec_dict(fake_data_dir)))
    return expand(spec_path, REGISTRY, tmp / 'runs')


@pytest.fixture(scope='module')
def nobulge_parts(nobulge_run):
    spec, config, manifest = load_run(nobulge_run)
    population = pd.read_parquet(nobulge_run / 'population.parquet')
    return spec, config, manifest, population


class TestNoBulgeCatalog:
    def test_spec_knob(self, fake_data_dir, tmp_path):
        # default true (existing behavior), explicit false, non-bool rejected
        spec = spec_from_dict(tmp_path, _small_spec_dict(fake_data_dir))
        assert spec.catalog_population.paint_bulge is True
        d = _nobulge_spec_dict(fake_data_dir)
        spec = spec_from_dict(tmp_path, d)
        assert spec.catalog_population.paint_bulge is False
        d['population']['paint']['bulge'] = 'no'
        with pytest.raises(ValueError, match='paint.bulge'):
            spec_from_dict(tmp_path, d)

    def test_nobulge_census_spec_parses(self):
        # the committed disk-only twin of the census spec: same seed (paired
        # galaxy draws), same observation config, bulge paint off
        spec = EnsembleSpec.from_yaml(NOBULGE_CENSUS_SPEC)
        bulge_spec = EnsembleSpec.from_yaml(CENSUS_SPEC)
        assert spec.run_name == 'flagship2_shear_census_v1_nobulge'
        assert spec.catalog_population.paint_bulge is False
        assert bulge_spec.catalog_population.paint_bulge is True
        assert spec.seed == bulge_spec.seed
        assert spec.observed_config == bulge_spec.observed_config
        assert spec.catalog_population.n_galaxies == 100

    def test_population_omits_painted_bulge_columns(self, nobulge_parts):
        _, _, _, population = nobulge_parts
        assert 'bulge_nsersic' not in population.columns
        assert 'bulge_r50_arcsec' not in population.columns
        # catalog facts stay for diagnostics
        assert 'bulge_fraction' in population.columns
        assert 'catalog_bulge_nsersic' in population.columns
        assert 'catalog_bulge_r50_arcsec' in population.columns

    def test_population_matches_bulge_twin(self, run_parts, nobulge_parts):
        # same seed + selection -> identical galaxies and all non-bulge
        # draws; the no-bulge table is the bulge table minus painted columns
        _, _, _, pop_bulge = run_parts
        _, _, _, pop_nobulge = nobulge_parts
        pd.testing.assert_frame_equal(
            pop_bulge[list(pop_nobulge.columns)], pop_nobulge, check_exact=True
        )

    def test_manifest_single_disk_truth_keys(self, nobulge_parts):
        _, config, manifest, population = nobulge_parts
        pop = population.set_index('pop_index')
        for band in config.bands:
            for key in (
                'total_flux',
                'bulge_frac',
                'bulge_hlr',
                'disk_rscale',
                'disk_h_over_r',
                'bulge_h_over_hlr',
            ):
                assert f'truth.{band}.{key}' not in manifest.columns
            for key in ('flux', 'rscale', 'h_over_r', 'x0', 'y0'):
                assert f'truth.{band}.{key}' in manifest.columns
        # painted bulge passthrough columns absent; catalog fact retained
        assert 'pop.bulge_nsersic' not in manifest.columns
        assert 'pop.bulge_r50_arcsec' not in manifest.columns
        assert 'pop.bulge_fraction' in manifest.columns
        # the catalog disk scale still sets every spatial scale
        for _, row in manifest.iterrows():
            rscale = float(pop.loc[row['galaxy_id'], 'rscale_arcsec'])
            for band in config.bands:
                assert row[f'truth.{band}.rscale'] == rscale
            # continuum shares the disk scale; line and rotation curve carry
            # their painted ratios to it
            g = pop.loc[row['galaxy_id']]
            assert row['truth.Halpha.cont.rscale'] == rscale
            assert row['truth.Halpha.rscale'] == rscale * float(
                g['halpha_rscale_ratio']
            )
            assert row['truth.vel.rscale'] == rscale * float(g['vel_rscale_ratio'])

    def test_flux_matches_bulge_twin_total_flux(self, run_parts, nobulge_parts):
        # flux normalization: the single-disk band flux equals the bulge
        # twin's total_flux (the disk absorbs the whole catalog flux; rows
        # align because both manifests expand the same population in order)
        _, config, manifest_bulge, _ = run_parts
        _, _, manifest_nobulge, _ = nobulge_parts
        assert (
            manifest_bulge['galaxy_id'].to_numpy()
            == manifest_nobulge['galaxy_id'].to_numpy()
        ).all()
        for band in config.bands:
            assert (
                manifest_nobulge[f'truth.{band}.flux'].to_numpy()
                == manifest_bulge[f'truth.{band}.total_flux'].to_numpy()
            ).all()

    def test_priors_single_disk(self, nobulge_parts):
        spec, config, manifest, _ = nobulge_parts
        row = manifest.iloc[0]
        truth = truth_from_row(row)
        assert not any('bulge' in k for k in truth)
        priors = scene_priors(truth, config, spec, row=row)
        names = list(priors.sampled_names) + list(priors.fixed_names)
        assert not any('bulge' in n for n in names)
        for band in config.bands:
            assert f'{band}.flux' in priors.sampled_names
            rs = priors.get_prior(f'{band}.rscale')
            # population size distribution, not a truth-centered prior
            assert isinstance(rs, TruncatedLogNormal)
            # catalog-mode rscale bounds (scene._CATALOG_RSCALE_LOW/HIGH);
            # ceiling raised 2.0 -> 3.0 on 2026-07-27 so painted size-ratio
            # products keep prior support (see the constant's provenance)
            assert rs.bounds == (0.005, 3.0)

    def test_all_truths_in_prior_support(self, nobulge_parts):
        spec, config, manifest, _ = nobulge_parts
        for _, row in manifest.iterrows():
            truth = truth_from_row(row)
            priors = scene_priors(truth, config, spec, row=row)
            for name in priors.sampled_names:
                lp = prior_logp_at_truth(priors, truth, name)
                assert np.isfinite(lp), f'{name}: truth {truth[name]} out of support'

    def test_build_fit_inputs_single_disk_task(self, nobulge_parts):
        # end-to-end: mocks + priors -> InferenceTask with no bulge params
        # and single-disk broadband models
        from kl_pipe.intensity import InclinedExponentialModel
        from kl_pipe.sampling import InferenceTask

        spec, config, manifest, _ = nobulge_parts
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
        for band in config.bands:
            assert isinstance(
                inputs.source.broadband_models[band], InclinedExponentialModel
            )
        task = InferenceTask.from_obs(
            inputs.source,
            inputs.priors,
            image_obs=inputs.image_obs,
            grism_obs=inputs.grism_obs,
        )
        assert not any('bulge' in n for n in task.sampled_names)


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
    # cheapest sampler config the spec validator accepts: precondition='none'
    # requires unconstrained and adapt_mass off (both need the laplace metric)
    # and escalation off (it donates the first attempt's warmup-adapted mass
    # matrix, which only that path records), and the dev spec turns all three
    # on, so they come off together. Running the laplace path here instead
    # measured over 20 min for a single fit, too slow for a wiring gate; the
    # production sampler path is covered by the ensemble campaigns.
    d['fit'].update(
        {
            'n_warmup': 50,
            'n_samples': 50,
            'n_chains': 1,
            'precondition': 'none',
            'unconstrained': False,
            'adapt_mass': False,
            'escalation': {'enabled': False},
        }
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


class TestGalaxyIdSubset:
    """sample.galaxy_ids restricts the manifest to a paired subset."""

    def test_subset_rows_identical_to_full(self, fake_data_dir, catalog_run, tmp_path):
        d = _small_spec_dict(fake_data_dir)
        d['population']['sample']['galaxy_ids'] = [3, 1]
        spec_path = tmp_path / 'spec_subset.yaml'
        spec_path.write_text(yaml.safe_dump(d))
        run_dir = expand(spec_path, REGISTRY, tmp_path / 'runs')
        _, _, sub = load_run(run_dir)
        _, _, full = load_run(catalog_run)

        assert sorted(sub['galaxy_id'].unique()) == [1, 3]
        assert len(sub) == 4  # 2 galaxies x 2 ring members
        # same run name + seed: subset rows must be byte-identical to the
        # matching full-manifest rows (draws, noise seeds, fit_ids included)
        expected = full[full['galaxy_id'].isin([1, 3])].reset_index(drop=True)
        pd.testing.assert_frame_equal(sub.reset_index(drop=True), expected)

    def test_n_fits_accounts_for_subset(self, fake_data_dir, tmp_path):
        d = _small_spec_dict(fake_data_dir)
        d['population']['sample']['galaxy_ids'] = [0, 2]
        spec = spec_from_dict(tmp_path, d)
        assert spec.n_fits == 4

    @pytest.mark.parametrize(
        'bad, match',
        [
            ([1, 1], 'duplicates'),
            ([0, 99], 'outside'),
            ([], 'non-empty'),
            ('0,1', 'list of ints'),
            ([0, 1.5], 'ints'),
        ],
    )
    def test_invalid_galaxy_ids_raise(self, fake_data_dir, tmp_path, bad, match):
        d = _small_spec_dict(fake_data_dir)
        d['population']['sample']['galaxy_ids'] = bad
        with pytest.raises((ValueError, TypeError), match=match):
            spec_from_dict(tmp_path, d)


# ==============================================================================
# Sampled bulge Sersic index (fit.sample_bulge_nsersic)
# ==============================================================================


class TestSampledBulgeIndex:
    """Freeing the bulge index removes a truth pin from the fit.

    Census v1 handed the painted per-galaxy index to the model as a
    constructor argument, so the fit knew it exactly.
    """

    def _inputs(self, data_dir, tmp_path, sample_bulge_nsersic):
        d = _small_spec_dict(data_dir)
        d['run']['name'] = f'pop_test_bulgen_{int(sample_bulge_nsersic)}'
        d['fit']['sample_bulge_nsersic'] = sample_bulge_nsersic
        spec_path = tmp_path / f'spec_{int(sample_bulge_nsersic)}.yaml'
        spec_path.write_text(yaml.safe_dump(d))
        run_dir = expand(
            spec_path, REGISTRY, tmp_path / f'r{int(sample_bulge_nsersic)}'
        )
        spec, config, manifest = load_run(run_dir)
        row = manifest.iloc[0].to_dict()
        truth = {
            k[len('truth.') :]: v for k, v in row.items() if k.startswith('truth.')
        }
        return spec, config, row, truth

    def test_defaults_to_pinned(self, fake_data_dir, tmp_path):
        spec, _, _, _ = self._inputs(fake_data_dir, tmp_path, False)
        assert spec.sample_bulge_nsersic is False

    def test_adds_one_sampled_index_per_band(self, fake_data_dir, tmp_path):
        spec_off, cfg, row_off, truth_off = self._inputs(fake_data_dir, tmp_path, False)
        spec_on, _, row_on, truth_on = self._inputs(fake_data_dir, tmp_path, True)

        off = build_fit_inputs(
            truth_off,
            int(row_off['noise_seed']),
            spec_off,
            cfg,
            broadband_snr=float(row_off['broadband_snr']),
            line_snr=float(row_off['line_snr']),
            row=row_off,
        )
        on = build_fit_inputs(
            truth_on,
            int(row_on['noise_seed']),
            spec_on,
            cfg,
            broadband_snr=float(row_on['broadband_snr']),
            line_snr=float(row_on['line_snr']),
            row=row_on,
        )
        added = set(on.priors.sampled_names) - set(off.priors.sampled_names)
        assert added == {f'{band}.bulge_n_sersic' for band in cfg.bands}

    def test_truth_is_the_painted_index_not_the_model_default(
        self, fake_data_dir, tmp_path
    ):
        # the recovery target must be this galaxy's painted index; if the
        # expander failed to write it, truth would silently be the scene
        # default and every fit would appear to recover n = 4
        _, cfg, row, truth = self._inputs(fake_data_dir, tmp_path, True)
        painted = float(row['pop.bulge_nsersic'])
        for band in cfg.bands:
            assert truth[f'{band}.bulge_n_sersic'] == pytest.approx(painted)

    def test_painted_truth_lies_inside_the_prior_support(self, fake_data_dir, tmp_path):
        spec, cfg, row, truth = self._inputs(fake_data_dir, tmp_path, True)
        inputs = build_fit_inputs(
            truth,
            int(row['noise_seed']),
            spec,
            cfg,
            broadband_snr=float(row['broadband_snr']),
            line_snr=float(row['line_snr']),
            row=row,
        )
        for band in cfg.bands:
            name = f'{band}.bulge_n_sersic'
            logp = float(inputs.priors.get_prior(name).log_prob(truth[name]))
            assert np.isfinite(logp), f'{name} truth outside its own prior'

    def test_requires_a_bulge(self):
        # sampling an index for a scene that has no bulge is a spec error
        from kl_pipe.ensemble.scene import build_source_model

        with pytest.raises(ValueError, match='requires bulge_nsersic'):
            build_source_model(
                _observation_config(), bulge_nsersic=None, sample_bulge_nsersic=True
            )

    def test_spec_rejects_the_flag_without_the_bulge_paint(
        self, fake_data_dir, tmp_path
    ):
        # accepting it silently would leave the fit unchanged while the spec
        # claims the index is sampled
        d = _nobulge_spec_dict(fake_data_dir)
        d['fit']['sample_bulge_nsersic'] = True
        path = tmp_path / 'nobulge_sample_n.yaml'
        path.write_text(yaml.safe_dump(d))
        with pytest.raises(ValueError, match='population.paint.bulge'):
            EnsembleSpec.from_yaml(path)

    def test_spec_rejects_the_flag_in_sampled_mode(self, tmp_path):
        d = yaml.safe_load(
            (
                REPO_ROOT / 'configs' / 'ensembles' / 'sigma_eps_cosi_dev.yaml'
            ).read_text()
        )
        d['fit']['sample_bulge_nsersic'] = True
        path = tmp_path / 'sampled_sample_n.yaml'
        path.write_text(yaml.safe_dump(d))
        with pytest.raises(ValueError, match='requires a catalog population'):
            EnsembleSpec.from_yaml(path)


def _observation_config():
    from kl_pipe.ensemble.spec import ObservationConfig

    return ObservationConfig.from_yaml(REGISTRY / 'canonical_P.yaml')
