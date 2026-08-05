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


class TestProductionSigmaBgPins:
    """Quotable background anchors for the production Roman-PSF config.

    Derived 2026-08-05 at the layer's landing commit from
    hlwas_medium_roman.yaml (roman_wfi PSF, 0.11 arcsec pixels) and the
    published point-source depths: sigma_bg = f_lim * ||K||_2 / 5. Pinned so
    an upstream change (galsim PSF model, depth constants, the derivation
    itself) moves these numbers loudly instead of silently re-anchoring the
    production noise. In electrons (x ELECTRONS_PER_UJY) these are ~41/41 e-
    rms per pixel, plausibly above a first-principles zodi+read estimate --
    expected, since published depths carry real-survey margins (see the
    background block in kl_pipe/surveys/roman.py).
    """

    SIGMA_BG_NJY = {'F129': 5.9396, 'F158': 5.7246}

    def test_pins(self):
        from kl_pipe.ensemble.mocks import _build_band_psf

        config = ObservationConfig.from_yaml(REGISTRY / 'hlwas_medium_roman.yaml')
        for band, pinned in self.SIGMA_BG_NJY.items():
            psf = _build_band_psf(config.band_psf[band], band, None, mock=True)
            sigma = roman.band_sigma_bg_ujy(
                band, _psf_l2_norm(psf, config.pixel_scale_arcsec)
            )
            assert sigma * 1e3 == pytest.approx(pinned, rel=1e-3), band


@pytest.mark.diagnostic_plots
class TestNoiseModelComparisonFigure:
    """Side-by-side datavector figure across the three noise pathways.

    One bright-ish galaxy (fluxes scaled up from the fake-catalog bank so
    the structure is visible over the noise): per channel (two bands + two
    grism rolls), the noiseless truth, a background-only draw, the
    matched_filter mock, the poisson mock, and all three variance maps on
    one shared color scale. Same noise deviates in every arm, so the
    panels differ only by the noise amplitude and structure; each noisy
    panel's title carries its realized matched-filter SNR next to the
    labeled target. Saved to tests/out/noise_model_comparison/.
    """

    # flux scalings applied to the bank galaxy: bright broadband, medium
    # grism (per-pass line label ~11 at the bank's ~2.2)
    K_BB = 4.0
    K_LINE = 5.0

    @staticmethod
    def _scaled_inputs(spec, config, row, k_bb, k_line):
        truth = truth_from_row(row)
        for band in config.bands:
            key = (
                f'{band}.total_flux'
                if f'{band}.total_flux' in truth
                else f'{band}.flux'
            )
            truth[key] *= k_bb
        truth['Halpha.flux'] *= k_line
        truth['Halpha.cont.flux_per_nm'] *= k_line  # preserve the EW
        band_snrs = {b: k_bb * v for b, v in _row_band_snrs(row, config).items()}
        line_snr = k_line * float(row['line_snr'])
        inputs = build_fit_inputs(
            truth,
            int(row['noise_seed']),
            spec,
            config,
            band_snrs=band_snrs,
            line_snr=line_snr,
            row=row,
        )
        return inputs, band_snrs, line_snr

    def test_figure(self, poisson_run, fake_data_dir, tmp_path_factory):
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        from kl_pipe.ensemble.mocks import _channel_seeds
        from kl_pipe.noise import add_map_noise

        spec_p, config_p, manifest = load_run(poisson_run)
        row = manifest.iloc[0]
        inputs_p, band_snrs, line_snr = self._scaled_inputs(
            spec_p, config_p, row, self.K_BB, self.K_LINE
        )

        # matched_filter twin: same bank, seeds, and scalings, default config
        tmp = tmp_path_factory.mktemp('mf_twin')
        d = catalog_spec_dict(fake_data_dir)
        d['population']['sample']['n_galaxies'] = 4
        spec_path = tmp / 'spec.yaml'
        spec_path.write_text(yaml.safe_dump(d))
        spec_m, config_m, manifest_m = load_run(
            expand(spec_path, REGISTRY, tmp / 'runs')
        )
        row_m = manifest_m.iloc[0]
        assert str(row_m['fit_id']) == str(row['fit_id'])
        inputs_m, _, _ = self._scaled_inputs(
            spec_m, config_m, row_m, self.K_BB, self.K_LINE
        )

        seeds = _channel_seeds(
            int(row['noise_seed']), len(config_p.bands) + len(config_p.grism_rolls_deg)
        )
        channels = [('image', b, i) for i, b in enumerate(config_p.bands)] + [
            ('grism', 'roll0', len(config_p.bands)),
            ('grism', 'roll1', len(config_p.bands) + 1),
        ]
        fig, axes = plt.subplots(len(channels), 7, figsize=(22, 3.1 * len(channels)))
        for i, (kind, key, seed_idx) in enumerate(channels):
            if kind == 'image':
                obs_p, obs_m = inputs_p.image_obs[key], inputs_m.image_obs[key]
                truth_img = np.asarray(
                    inputs_p.source.render_broadband(inputs_p.truth, obs_p, key)
                )
                template = truth_img
                target = band_snrs[key]
                eff_key = key
            else:
                obs_p, obs_m = inputs_p.grism_obs[key], inputs_m.grism_obs[key]
                truth_img = np.asarray(
                    inputs_p.source.render_grism(inputs_p.truth, obs_p)
                )
                # the SNR convention normalizes on the line template alone
                line_truth = {
                    k: (0.0 if k.endswith('.cont.flux_per_nm') else v)
                    for k, v in inputs_p.truth.items()
                }
                template = np.asarray(inputs_p.source.render_grism(line_truth, obs_p))
                target = line_snr
                eff_key = f'line_{key}'

            var_p = np.asarray(obs_p.variance)
            var_m = np.full_like(var_p, float(np.asarray(obs_m.variance)))
            sigma_bg = float(np.sqrt(var_p.min()))
            var_bg = np.full_like(var_p, sigma_bg**2)
            bg_only = add_map_noise(truth_img, var_bg, seed=int(seeds[seed_idx]))
            eff_bg = matched_filter_snr(template, var_bg)
            eff_m = inputs_m.snr_effective[eff_key]
            eff_p = inputs_p.snr_effective[eff_key]

            noisy = [bg_only, np.asarray(obs_m.data), np.asarray(obs_p.data)]
            dlo = min(a.min() for a in noisy)
            dhi = max(a.max() for a in noisy)
            vlo = min(var_bg.min(), var_m.min(), var_p.min())
            vhi = max(var_bg.max(), var_m.max(), var_p.max())
            panels = [
                (truth_img, f'truth (target snr={target:.0f})', {}),
                (
                    bg_only,
                    f'bg-only data (snr_eff={eff_bg:.1f})',
                    {'vmin': dlo, 'vmax': dhi},
                ),
                (
                    noisy[1],
                    f'matched_filter data (snr_eff={eff_m:.1f})',
                    {'vmin': dlo, 'vmax': dhi},
                ),
                (
                    noisy[2],
                    f'poisson data (snr_eff={eff_p:.1f})',
                    {'vmin': dlo, 'vmax': dhi},
                ),
                (var_bg, 'var: bg-only', {'vmin': vlo, 'vmax': vhi, 'cmap': 'viridis'}),
                (
                    var_m,
                    'var: matched_filter',
                    {'vmin': vlo, 'vmax': vhi, 'cmap': 'viridis'},
                ),
                (var_p, 'var: poisson', {'vmin': vlo, 'vmax': vhi, 'cmap': 'viridis'}),
            ]
            for j, (img, title, kwargs) in enumerate(panels):
                ax = axes[i, j]
                im = ax.imshow(img, origin='lower', **kwargs)
                ax.set_title(f'{key}: {title}', fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046)

        fig.suptitle('noise_model comparison (shared noise deviates)', fontsize=13)
        fig.text(
            0.5,
            0.955,
            'bg-only: flat background solved from the published survey depth, no '
            'source term  |  matched_filter: flat variance chosen so the labeled '
            'SNR is exact (the default)  |  poisson: that same published-depth '
            'background plus the source\'s own shot noise',
            ha='center',
            fontsize=9,
        )
        out_dir = Path('tests/out/noise_model_comparison')
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / 'noise_model_comparison.png'
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out, dpi=130)
        plt.close(fig)
        assert out.exists()
