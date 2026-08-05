"""
noise_model wiring tests (kl_pipe/ensemble catalog mode).

The 'poisson' noise model replaces the label-normalized uniform variance
with a physical one: a flat per-pixel background anchored to the published
survey depths (rendered reference templates; see kl_pipe/surveys/roman.py)
plus the source's own shot noise. The default 'matched_filter' path is
byte-frozen by test_ensemble_catalog.TestMockBitIdentity; this module
covers the poisson branch and the snr_effective bookkeeping shared by both.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from kl_pipe.ensemble.expander import expand, load_run, truth_from_row
from kl_pipe.ensemble.mocks import _psf_l2_norm, build_fit_inputs
from kl_pipe.ensemble.spec import EnsembleSpec, ObservationConfig
from kl_pipe.noise import matched_filter_snr
from kl_pipe.surveys import roman

from test_population import (
    catalog_spec_dict,
    fake_flagship2_rows,
    spec_from_dict,
    write_fake_catalog,
)

pytestmark = pytest.mark.roman_ensemble

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = REPO_ROOT / 'configs' / 'observation'


def _row_band_snrs(row, config):
    return {b: float(row[f'broadband_snr_{b}']) for b in config.bands}


@pytest.fixture(scope='module')
def fake_data_dir(tmp_path_factory) -> Path:
    data_dir = tmp_path_factory.mktemp('cosmohub_fake_noise')
    write_fake_catalog(data_dir, fake_flagship2_rows(n=400, seed=1234))
    return data_dir


@pytest.fixture(scope='module')
def poisson_registry(tmp_path_factory) -> Path:
    """Config registry holding hlwas_medium plus a poisson twin of it."""
    registry = tmp_path_factory.mktemp('noise_registry')
    src = REGISTRY / 'hlwas_medium.yaml'
    shutil.copy(src, registry / 'hlwas_medium.yaml')
    raw = yaml.safe_load(src.read_bytes())
    raw['id'] = 'hlwas_medium_poisson'
    raw['noise_model'] = 'poisson'
    (registry / 'hlwas_medium_poisson.yaml').write_text(yaml.safe_dump(raw))
    return registry


@pytest.fixture(scope='module')
def poisson_run(fake_data_dir, poisson_registry, tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp('noise_run')
    d = catalog_spec_dict(fake_data_dir)
    d['population']['sample']['n_galaxies'] = 4
    d['observation']['config'] = 'hlwas_medium_poisson'
    spec_path = tmp / 'spec.yaml'
    spec_path.write_text(yaml.safe_dump(d))
    return expand(spec_path, poisson_registry, tmp / 'runs')


@pytest.fixture(scope='module')
def poisson_inputs(poisson_run):
    spec, config, manifest = load_run(poisson_run)
    row = manifest.iloc[0]
    inputs = build_fit_inputs(
        truth_from_row(row),
        int(row['noise_seed']),
        spec,
        config,
        band_snrs=_row_band_snrs(row, config),
        line_snr=float(row['line_snr']),
        row=row,
    )
    return spec, config, row, inputs


class TestNoiseModelKnob:
    def test_default_is_matched_filter(self):
        config = ObservationConfig.from_yaml(REGISTRY / 'hlwas_medium.yaml')
        assert config.noise_model == 'matched_filter'

    def test_poisson_twin_parses(self, poisson_registry):
        config = ObservationConfig.from_yaml(
            poisson_registry / 'hlwas_medium_poisson.yaml'
        )
        assert config.noise_model == 'poisson'

    def test_unknown_noise_model_raises(self, tmp_path):
        raw = yaml.safe_load((REGISTRY / 'hlwas_medium.yaml').read_bytes())
        raw['noise_model'] = 'shot'
        path = tmp_path / 'bad.yaml'
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match='noise_model'):
            ObservationConfig.from_yaml(path)

    def test_sampled_mode_rejects_poisson(
        self, fake_data_dir, poisson_registry, tmp_path
    ):
        # sampled-mode scenes carry no physical flux units, so shot noise
        # has no electron conversion; the build must refuse loudly
        sampled_spec = EnsembleSpec.from_yaml(
            REPO_ROOT / 'configs' / 'ensembles' / 'sigma_eps_cosi_dev.yaml'
        )
        config = ObservationConfig.from_yaml(
            poisson_registry / 'hlwas_medium_poisson.yaml'
        )
        with pytest.raises(ValueError, match='catalog population'):
            build_fit_inputs(
                {},
                1,
                sampled_spec,
                config,
                band_snrs={b: 100.0 for b in config.bands},
                line_snr=10.0,
            )


class TestPoissonMocks:
    def test_variance_is_a_map_on_every_channel(self, poisson_inputs):
        _, config, _, inputs = poisson_inputs
        shape_bb = (config.stamp_broadband_pix,) * 2
        shape_gr = (config.stamp_grism_pix,) * 2
        for band, obs in inputs.image_obs.items():
            assert np.asarray(obs.variance).shape == shape_bb, band
        for key, obs in inputs.grism_obs.items():
            assert np.asarray(obs.variance).shape == shape_gr, key

    def test_band_variance_floor_is_the_depth_anchor(self, poisson_inputs):
        # the variance floor (source-free pixels) is the background solved
        # from the published point-source depth through this config's PSF;
        # stamp corners still hold a little galaxy flux, hence the one-sided
        # tolerance
        _, config, _, inputs = poisson_inputs
        from kl_pipe.ensemble.mocks import _build_band_psf

        for band, obs in inputs.image_obs.items():
            psf = _build_band_psf(config.band_psf[band], band, None, mock=True)
            sigma_bg = roman.band_sigma_bg_ujy(
                band, _psf_l2_norm(psf, config.pixel_scale_arcsec)
            )
            floor = float(np.asarray(obs.variance).min())
            assert floor >= sigma_bg**2 * (1 - 1e-12), band
            assert floor <= sigma_bg**2 * 1.05, band

    def test_variance_has_shot_structure(self, poisson_inputs):
        # bright pixels must carry more variance than the background floor
        _, _, _, inputs = poisson_inputs
        for band, obs in inputs.image_obs.items():
            var = np.asarray(obs.variance)
            assert var.max() > 1.1 * var.min(), band

    def test_snr_effective_keys_and_coadd(self, poisson_inputs):
        _, config, _, inputs = poisson_inputs
        eff = inputs.snr_effective
        expected = set(config.bands)
        expected |= {f'line_roll{j}' for j in range(len(config.grism_rolls_deg))}
        expected |= {'line_total'}
        assert set(eff) == expected
        assert all(np.isfinite(v) and v > 0 for v in eff.values())
        quad = np.sqrt(
            sum(eff[f'line_roll{j}'] ** 2 for j in range(len(config.grism_rolls_deg)))
        )
        assert eff['line_total'] == pytest.approx(quad, rel=1e-12)

    def test_deterministic_rebuild(self, poisson_inputs):
        spec, config, row, inputs = poisson_inputs
        rebuilt = build_fit_inputs(
            truth_from_row(row),
            int(row['noise_seed']),
            spec,
            config,
            band_snrs=_row_band_snrs(row, config),
            line_snr=float(row['line_snr']),
            row=row,
        )
        for band in inputs.image_obs:
            np.testing.assert_array_equal(
                np.asarray(inputs.image_obs[band].data),
                np.asarray(rebuilt.image_obs[band].data),
            )
            np.testing.assert_array_equal(
                np.asarray(inputs.image_obs[band].variance),
                np.asarray(rebuilt.image_obs[band].variance),
            )
        for key in inputs.grism_obs:
            np.testing.assert_array_equal(
                np.asarray(inputs.grism_obs[key].data),
                np.asarray(rebuilt.grism_obs[key].data),
            )
        assert inputs.snr_effective == rebuilt.snr_effective


class TestMatchedFilterLabelsExact:
    """Under the default noise model, snr_effective must equal the labels
    exactly (a free wiring check: var = ||T||^2/label^2 by construction)."""

    @pytest.fixture(scope='class')
    def default_inputs(self, fake_data_dir, tmp_path_factory):
        tmp = tmp_path_factory.mktemp('mf_run')
        d = catalog_spec_dict(fake_data_dir)
        d['population']['sample']['n_galaxies'] = 4
        spec_path = tmp / 'spec.yaml'
        spec_path.write_text(yaml.safe_dump(d))
        run_dir = expand(spec_path, REGISTRY, tmp / 'runs')
        spec, config, manifest = load_run(run_dir)
        row = manifest.iloc[0]
        inputs = build_fit_inputs(
            truth_from_row(row),
            int(row['noise_seed']),
            spec,
            config,
            band_snrs=_row_band_snrs(row, config),
            line_snr=float(row['line_snr']),
            row=row,
        )
        return config, row, inputs

    def test_band_labels(self, default_inputs):
        config, row, inputs = default_inputs
        for band in config.bands:
            assert inputs.snr_effective[band] == pytest.approx(
                float(row[f'broadband_snr_{band}']), rel=1e-9
            )

    def test_line_labels(self, default_inputs):
        config, row, inputs = default_inputs
        n_rolls = len(config.grism_rolls_deg)
        for j in range(n_rolls):
            assert inputs.snr_effective[f'line_roll{j}'] == pytest.approx(
                float(row['line_snr']), rel=1e-9
            )
        assert inputs.snr_effective['line_total'] == pytest.approx(
            np.sqrt(n_rolls) * float(row['line_snr']), rel=1e-9
        )


class TestPointSourceClosure:
    """A point-like source at the published depth realizes MF SNR ~ 5.

    Non-tautological: sigma_bg comes from the galsim drawImage point-source
    template while the source below is rendered through kl_pipe's own
    k-space pipeline (profile FT x pixel FT x PSF FT), so agreement checks
    the whole anchoring chain, not the formula against itself.
    """

    def test_closure(self):
        import galsim

        from kl_pipe.intensity import InclinedExponentialModel
        from kl_pipe.observation import build_image_obs
        from kl_pipe.parameters import ImagePars
        from kl_pipe.render import RenderConfig

        band = 'F129'
        pixel_scale = 0.11
        psf = galsim.Gaussian(fwhm=0.18)
        sigma_bg = roman.band_sigma_bg_ujy(band, _psf_l2_norm(psf, pixel_scale))

        model = InclinedExponentialModel()
        pars = {
            'flux': roman.band_flux_limit_ujy(band),
            'rscale': 0.005,  # << PSF: effectively a point source
            'h_over_r': 0.0,
            'cosi': 1.0,
            'theta_int': 0.0,
            'g1': 0.0,
            'g2': 0.0,
            'x0': 0.0,
            'y0': 0.0,
        }
        import jax.numpy as jnp

        theta = jnp.array([pars[n] for n in model.PARAMETER_NAMES])
        obs = build_image_obs(
            ImagePars(shape=(32, 32), pixel_scale=pixel_scale, indexing='ij'),
            psf=psf,
            render_config=RenderConfig(oversample=5),
            int_model=model,
        )
        render = np.asarray(model.render_image(theta, obs=obs))
        snr = matched_filter_snr(render, sigma_bg**2)
        assert snr == pytest.approx(roman.IMAGING_DEPTH_NSIGMA, rel=0.02)


class TestMapVarianceArtifacts:
    def test_saved_mock_variance_is_a_map_and_plot_renders(
        self, poisson_run, poisson_inputs, tmp_path
    ):
        # _save_mocks round-trips the 2-D variance, and the datavector
        # diagnostic (whose chi2 used to read variance.ravel()[0]) renders
        # from it
        import matplotlib

        matplotlib.use('Agg')
        from kl_pipe.ensemble.diagnostics import plot_datavector_fit
        from kl_pipe.ensemble.worker import _save_mocks

        spec, config, row, inputs = poisson_inputs
        fit_id = str(row['fit_id'])
        (tmp_path / 'mocks').mkdir()
        _save_mocks(tmp_path, fit_id, inputs, None, list(inputs.priors.sampled_names))
        npz = np.load(tmp_path / 'mocks' / f'{fit_id}.npz')
        for band in config.bands:
            assert (
                npz[f'image.{band}.variance'].shape == (config.stamp_broadband_pix,) * 2
            )

        table = pd.DataFrame([{**dict(row), 'max_rhat': 1.0, 'divergence_rate': 0.0}])
        out = plot_datavector_fit(tmp_path, table, fit_id, tmp_path)
        assert out is not None and out.exists()
