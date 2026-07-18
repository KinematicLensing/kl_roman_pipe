"""
Tests for the roman_wfi PSF option in the ensemble pipeline.

Covers (a) PSF config parsing: the ``type: roman_wfi`` schema, defaults,
gaussian backward compatibility, and loud rejection of unknown/malformed
specs; (b) monochromatic ``galsim.roman.getPSF`` kernel construction:
unit normalization, the pinned kernel stamp shape across the ensemble z
range, and the wavelength-scaling physics sanity check; (c) obs
construction with a roman PSF plus the explicit-kernel-size validation
added to the kernel-render path.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from kl_pipe.ensemble.mocks import (
    _band_effective_wavelength_nm,
    _build_band_psf,
    _build_grism_psf,
    _get_roman_wfi_psf,
    _grism_psf_kernel_size,
    build_fit_inputs,
)
from kl_pipe.ensemble.scene import scene_truth_defaults
from kl_pipe.ensemble.spec import EnsembleSpec, ObservingConfig, PSFSpec
from kl_pipe.lines import LINE_LAMBDAS
from kl_pipe.observation import build_grism_obs, build_image_obs
from kl_pipe.parameters import ImagePars
from kl_pipe.psf import precompute_psf_fft
from kl_pipe.render import RenderConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = REPO_ROOT / 'configs' / 'observing'
DEV_SPEC = REPO_ROOT / 'configs' / 'ensembles' / 'sigma_eps_cosi_dev.yaml'

# canonical geometry (matches the observing-config registry)
PIXEL_SCALE = 0.11  # arcsec/pix

# ensemble z range endpoints (specs draw z uniform on [1.0, 1.9])
Z_LO = 0.55
Z_HI = 1.9

ROMAN_SPEC = PSFSpec(psf_type='roman_wfi', sca=10, pupil_bin=4)


def _config_dict() -> dict:
    return yaml.safe_load((REGISTRY / 'canonical_Q.yaml').read_text())


def _write_config(tmp_path: Path, config_dict: dict) -> Path:
    path = tmp_path / 'obs.yaml'
    path.write_text(yaml.safe_dump(config_dict))
    return path


def _pinned_kernel_size(pixel_scale: float, z_max: float = Z_HI) -> int:
    """Good image size at the largest ensemble wavelength, forced odd."""
    psf = _build_grism_psf(ROMAN_SPEC, z_max)
    size = int(psf.getGoodImageSize(pixel_scale))
    if size % 2 == 0:
        size += 1
    return size


# ==============================================================================
# Config parsing
# ==============================================================================


class TestRomanPSFConfig:
    def test_registry_roman_config_loads(self):
        config = ObservingConfig.from_yaml(REGISTRY / 'canonical_P_roman.yaml')
        assert config.bands == ('H158', 'F184')
        assert config.grism_rolls_deg == (0.0, 45.0, 90.0, 135.0)
        for band in config.bands:
            assert config.band_psf[band].psf_type == 'roman_wfi'
            assert config.band_psf[band].sca == 10
            assert config.band_psf[band].pupil_bin == 4
        assert config.grism_psf.psf_type == 'roman_wfi'

    def test_roman_defaults_applied(self, tmp_path):
        d = _config_dict()
        d['psf']['broadband'] = {'type': 'roman_wfi'}
        d['psf']['grism'] = {'type': 'roman_wfi'}
        config = ObservingConfig.from_yaml(_write_config(tmp_path, d))
        assert config.band_psf['F087'].sca == 10
        assert config.band_psf['F087'].pupil_bin == 4
        assert config.grism_psf.sca == 10
        assert config.grism_psf.pupil_bin == 4

    def test_explicit_sca_pupil_bin(self, tmp_path):
        d = _config_dict()
        d['psf']['grism'] = {'type': 'roman_wfi', 'sca': 4, 'pupil_bin': 8}
        config = ObservingConfig.from_yaml(_write_config(tmp_path, d))
        assert config.grism_psf.sca == 4
        assert config.grism_psf.pupil_bin == 8

    def test_mixed_gaussian_broadband_roman_grism(self, tmp_path):
        d = _config_dict()
        d['psf']['grism'] = {'type': 'roman_wfi'}
        config = ObservingConfig.from_yaml(_write_config(tmp_path, d))
        assert config.band_psf['F087'].psf_type == 'gaussian'
        assert config.band_psf_fwhm == {'F087': 0.18}
        assert config.grism_psf.psf_type == 'roman_wfi'

    def test_gaussian_configs_unchanged(self):
        config = ObservingConfig.from_yaml(REGISTRY / 'canonical_P.yaml')
        assert config.band_psf_fwhm == {'F087': 0.18, 'F158': 0.18}
        assert config.grism_psf_fwhm == 0.18

    def test_fwhm_properties_raise_for_roman(self):
        config = ObservingConfig.from_yaml(REGISTRY / 'canonical_P_roman.yaml')
        with pytest.raises(ValueError, match='gaussian PSFs only'):
            config.band_psf_fwhm
        with pytest.raises(ValueError, match='gaussian PSFs only'):
            config.grism_psf_fwhm

    def test_unknown_type_raises(self, tmp_path):
        for channel in ('broadband', 'grism'):
            d = _config_dict()
            d['psf'][channel] = {'type': 'imcom'}
            with pytest.raises(NotImplementedError, match='not supported'):
                ObservingConfig.from_yaml(_write_config(tmp_path, d))

    def test_roman_with_fwhm_key_raises(self, tmp_path):
        d = _config_dict()
        d['psf']['grism'] = {'type': 'roman_wfi', 'fwhm_arcsec': 0.18}
        with pytest.raises(ValueError, match='unknown keys'):
            ObservingConfig.from_yaml(_write_config(tmp_path, d))

    def test_bad_sca_raises(self, tmp_path):
        for sca in (0, 19):
            d = _config_dict()
            d['psf']['grism'] = {'type': 'roman_wfi', 'sca': sca}
            with pytest.raises(ValueError, match='sca'):
                ObservingConfig.from_yaml(_write_config(tmp_path, d))

    def test_bad_pupil_bin_raises(self, tmp_path):
        d = _config_dict()
        d['psf']['grism'] = {'type': 'roman_wfi', 'pupil_bin': 0}
        with pytest.raises(ValueError, match='pupil_bin'):
            ObservingConfig.from_yaml(_write_config(tmp_path, d))

    def test_psfspec_cross_field_validation(self):
        with pytest.raises(ValueError, match='roman_wfi-only'):
            PSFSpec(psf_type='gaussian', fwhm_arcsec=0.18, sca=10)
        with pytest.raises(ValueError, match='gaussian-only'):
            PSFSpec(psf_type='roman_wfi', fwhm_arcsec=0.18, sca=10, pupil_bin=4)
        with pytest.raises(ValueError, match='fwhm_arcsec'):
            PSFSpec(psf_type='gaussian')
        with pytest.raises(NotImplementedError, match='not supported'):
            PSFSpec(psf_type='airy')


# ==============================================================================
# Kernel construction (monochromatic getPSF)
# ==============================================================================


class TestRomanKernel:
    def test_grism_psf_wavelength_and_cache(self):
        lam = round(LINE_LAMBDAS['Halpha'] * (1.0 + Z_HI), 1)
        psf_a = _get_roman_wfi_psf(10, 4, lam, bandpass=None)
        # wavelengths rounding to the same 0.1 nm key hit the cache
        psf_b = _get_roman_wfi_psf(10, 4, lam + 0.04, bandpass=None)
        assert psf_a is psf_b
        # repeated builds at the same z reuse the cached object
        assert _build_grism_psf(ROMAN_SPEC, Z_HI) is _build_grism_psf(ROMAN_SPEC, Z_HI)

    def test_band_effective_wavelengths(self):
        # galsim.roman effective wavelengths are in nm; H158/F184 pivots
        # are ~1.58/1.84 um, a loose bracket guards a silent unit change
        lam_h158 = _band_effective_wavelength_nm('H158')
        lam_f184 = _band_effective_wavelength_nm('F184')
        assert 1500.0 < lam_h158 < 1650.0
        assert 1750.0 < lam_f184 < 1900.0

    def test_non_roman_band_raises(self):
        # flagship band names (F087/F158) are not Roman WFI bandpasses
        with pytest.raises(ValueError, match='not a Roman WFI bandpass'):
            _build_band_psf(ROMAN_SPEC, 'F087')

    def test_kernel_unit_normalized_and_finite(self):
        psf = _build_grism_psf(ROMAN_SPEC, 1.2)
        pin = _pinned_kernel_size(PIXEL_SCALE)
        psf_data = precompute_psf_fft(
            psf, image_shape=(32, 32), pixel_scale=PIXEL_SCALE, kernel_size=pin
        )
        kernel_fft = np.asarray(psf_data.kernel_fft)
        assert np.all(np.isfinite(kernel_fft))
        # unit-sum kernel: DC component of its FFT is exactly the sum
        assert kernel_fft[0, 0].real == pytest.approx(1.0, abs=1e-12)
        assert kernel_fft[0, 0].imag == pytest.approx(0.0, abs=1e-12)

    def test_pinned_shape_constant_across_z(self):
        pin = _pinned_kernel_size(PIXEL_SCALE)
        shapes = {}
        for z in (Z_LO, Z_HI):
            psf = _build_grism_psf(ROMAN_SPEC, z)
            psf_data = precompute_psf_fft(
                psf, image_shape=(32, 32), pixel_scale=PIXEL_SCALE, kernel_size=pin
            )
            shapes[z] = (psf_data.padded_shape, psf_data.kernel_fft.shape)
        assert shapes[Z_LO] == shapes[Z_HI]

    def test_higher_z_kernel_broader(self):
        # physics sanity: diffraction-limited size scales ~lambda/D, so the
        # z=1.9 kernel (1903 nm) must be measurably broader than z=0.55
        # (1017 nm); expected half-light-radius ratio ~1.87, assert > 1.3
        # to allow pupil-detail deviations while staying loudly physical.
        # Measured at the ensemble's fine pixel scale (oversample 3) -- the
        # coarse 0.11 arcsec pixels quantize the HLR below one pixel.
        fine_ps = PIXEL_SCALE / 3
        pin = _pinned_kernel_size(fine_ps)

        def half_light_radius(z: float) -> float:
            psf = _build_grism_psf(ROMAN_SPEC, z)
            image = psf.drawImage(nx=pin, ny=pin, scale=fine_ps).array
            image = image / image.sum()
            center = (pin - 1) / 2.0
            yy, xx = np.mgrid[0:pin, 0:pin]
            radius = np.hypot(xx - center, yy - center).ravel()
            order = np.argsort(radius)
            cumulative = np.cumsum(image.ravel()[order])
            return float(radius[order][np.searchsorted(cumulative, 0.5)])

        hlr_lo = half_light_radius(Z_LO)
        hlr_hi = half_light_radius(Z_HI)
        assert hlr_hi > 1.3 * hlr_lo

    def test_too_small_kernel_size_raises(self):
        psf = _build_grism_psf(ROMAN_SPEC, Z_HI)
        with pytest.raises(ValueError, match='truncate'):
            precompute_psf_fft(
                psf, image_shape=(32, 32), pixel_scale=PIXEL_SCALE, kernel_size=31
            )

    def test_even_kernel_size_raises(self):
        psf = _build_grism_psf(ROMAN_SPEC, Z_HI)
        pin = _pinned_kernel_size(PIXEL_SCALE) + 1  # even
        with pytest.raises(ValueError, match='odd'):
            precompute_psf_fft(
                psf, image_shape=(32, 32), pixel_scale=PIXEL_SCALE, kernel_size=pin
            )

    def test_pinning_helper(self, tmp_path):
        spec = EnsembleSpec.from_yaml(DEV_SPEC)
        d = _config_dict()
        d['psf']['grism'] = {'type': 'roman_wfi'}
        config = ObservingConfig.from_yaml(_write_config(tmp_path, d))
        size = _grism_psf_kernel_size(config, spec)
        fine_ps = config.pixel_scale_arcsec / config.oversample
        # dev spec draws z uniform on [1.0, 1.9]: pin at z=1.9
        assert size == _pinned_kernel_size(fine_ps, z_max=1.9)
        assert size % 2 == 1
        # gaussian grism psf needs no pinning
        gaussian_config = ObservingConfig.from_yaml(REGISTRY / 'canonical_Q.yaml')
        assert _grism_psf_kernel_size(gaussian_config, spec) is None


# ==============================================================================
# Obs construction
# ==============================================================================


class TestRomanObs:
    def test_image_obs_kernel_size_guards(self):
        image_pars = ImagePars(shape=(32, 32), pixel_scale=PIXEL_SCALE, indexing='ij')
        psf = _build_grism_psf(ROMAN_SPEC, 1.2)
        with pytest.raises(ValueError, match='without a psf'):
            build_image_obs(
                image_pars,
                render_config=RenderConfig(oversample=1),
                psf_kernel_size=101,
            )
        with pytest.raises(ValueError, match='explicit render_config'):
            build_image_obs(image_pars, psf=psf, psf_kernel_size=101)

    def test_image_obs_with_roman_psf(self):
        image_pars = ImagePars(shape=(32, 32), pixel_scale=PIXEL_SCALE, indexing='ij')
        psf = _build_band_psf(ROMAN_SPEC, 'H158')
        obs = build_image_obs(
            image_pars, psf=psf, render_config=RenderConfig(oversample=1)
        )
        assert obs.psf_data is not None
        assert np.all(np.isfinite(np.asarray(obs.psf_data.kernel_fft)))

    def test_grism_obs_pinned_shapes_match_across_z(self, tmp_path):
        from kl_pipe.dispersion import build_grism_pars_for_line

        pin = _pinned_kernel_size(PIXEL_SCALE)
        shapes = {}
        for z in (Z_LO, Z_HI):
            grism_pars = build_grism_pars_for_line(
                LINE_LAMBDAS['Halpha'],
                redshift=z,
                image_pars=ImagePars(
                    shape=(32, 32), pixel_scale=PIXEL_SCALE, indexing='ij'
                ),
                dispersion=1.1,
            )
            obs = build_grism_obs(
                grism_pars,
                z=z,
                psf=_build_grism_psf(ROMAN_SPEC, z),
                render_config=RenderConfig(oversample=1),
                psf_kernel_size=pin,
            )
            shapes[z] = (obs.psf_data.padded_shape, obs.psf_data.kernel_fft.shape)
        assert shapes[Z_LO] == shapes[Z_HI]


@pytest.mark.slow
def test_build_fit_inputs_roman_smoke(tmp_path):
    """End-to-end mock construction with roman_wfi PSFs on both channels:
    one H158 band + one grism roll, oversample 1 to keep the kernel grids
    small. Asserts valid PSFData on every obs, finite noisy data, and a
    finite truth render through the fit's own obs."""
    spec = EnsembleSpec.from_yaml(DEV_SPEC)
    d = _config_dict()
    d['id'] = 'roman_smoke'
    d['bands'] = ['H158']
    d['psf'] = {
        'broadband': {'type': 'roman_wfi'},
        'grism': {'type': 'roman_wfi'},
    }
    d['render'] = {'oversample': 1}
    config = ObservingConfig.from_yaml(_write_config(tmp_path, d))

    truth = scene_truth_defaults(config, spec.fixed)
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
    inputs = build_fit_inputs(
        truth, 11, spec, config, broadband_snr=100.0, line_snr=100.0
    )

    assert set(inputs.image_obs) == {'H158'}
    assert set(inputs.grism_obs) == {'roll0'}
    for obs in list(inputs.image_obs.values()) + list(inputs.grism_obs.values()):
        assert obs.psf_data is not None
        assert np.all(np.isfinite(np.asarray(obs.data)))
        assert float(np.asarray(obs.variance)) > 0

    band_render = np.asarray(
        inputs.source.render_broadband(truth, inputs.image_obs['H158'], 'H158')
    )
    grism_render = np.asarray(
        inputs.source.render_grism(truth, inputs.grism_obs['roll0'])
    )
    assert np.all(np.isfinite(band_render))
    assert np.all(np.isfinite(grism_render))
    assert band_render.sum() > 0
    assert grism_render.sum() > 0
