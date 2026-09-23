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
from kl_pipe.ensemble.spec import (
    EnsembleSpec,
    FoldingThresholdTier,
    ObservationConfig,
    PSFSpec,
)
from kl_pipe.lines import LINE_LAMBDAS
from kl_pipe.observation import build_grism_obs, build_image_obs
from kl_pipe.parameters import ImagePars
from kl_pipe.psf import precompute_psf_fft
from kl_pipe.render import RenderConfig

pytestmark = pytest.mark.roman_ensemble

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = REPO_ROOT / 'configs' / 'observation'
DEV_SPEC = REPO_ROOT / 'configs' / 'ensembles' / 'sigma_eps_cosi_dev.yaml'

# canonical geometry (matches the observation-config registry)
PIXEL_SCALE = 0.11  # arcsec/pix

# wavelength-scaling test endpoints: Z_HI matches the current spec z-draw
# upper bound (uniform on [1.0, 1.9], the pinning reference); Z_LO = 0.55 is
# the planned ensemble lower bound (Halpha enters G150 at z~0.52), used here
# as a low-z reference to exercise the full kernel-size spread
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
        config = ObservationConfig.from_yaml(REGISTRY / 'canonical_P_roman.yaml')
        assert config.bands == ('F158', 'F184')
        assert config.grism_rolls_deg == (0.0, 45.0, 90.0, 135.0)
        # 2026-07-19 ruling: fit kernels follow the z-tiered schedule
        # (ft=0.01 for z<=1.2, tighter 5e-3 above == the mock default, so
        # mock == fit at the tight tier); scalar folding_threshold retired
        # from this config. Pin the FULL schedule; mock kernels at default.
        ruling_tiers = (
            FoldingThresholdTier(z_max=1.2, ft=0.01),
            FoldingThresholdTier(z_max=None, ft=5.0e-3),
        )
        for band in config.bands:
            assert config.band_psf[band].psf_type == 'roman_wfi'
            assert config.band_psf[band].sca == 10
            assert config.band_psf[band].pupil_bin == 4
            assert config.band_psf[band].folding_threshold is None
            assert config.band_psf[band].folding_threshold_tiers == ruling_tiers
            assert config.band_psf[band].mock_folding_threshold is None
        assert config.grism_psf.psf_type == 'roman_wfi'
        assert config.grism_psf.folding_threshold is None
        assert config.grism_psf.folding_threshold_tiers == ruling_tiers
        assert config.grism_psf.mock_folding_threshold is None

    def test_roman_defaults_applied(self, tmp_path):
        d = _config_dict()
        d['psf']['broadband'] = {'type': 'roman_wfi'}
        d['psf']['grism'] = {'type': 'roman_wfi'}
        config = ObservationConfig.from_yaml(_write_config(tmp_path, d))
        assert config.band_psf['F087'].sca == 10
        assert config.band_psf['F087'].pupil_bin == 4
        assert config.grism_psf.sca == 10
        assert config.grism_psf.pupil_bin == 4

    def test_explicit_sca_pupil_bin(self, tmp_path):
        d = _config_dict()
        d['psf']['grism'] = {'type': 'roman_wfi', 'sca': 4, 'pupil_bin': 8}
        config = ObservationConfig.from_yaml(_write_config(tmp_path, d))
        assert config.grism_psf.sca == 4
        assert config.grism_psf.pupil_bin == 8

    def test_mixed_gaussian_broadband_roman_grism(self, tmp_path):
        d = _config_dict()
        d['psf']['grism'] = {'type': 'roman_wfi'}
        config = ObservationConfig.from_yaml(_write_config(tmp_path, d))
        assert config.band_psf['F087'].psf_type == 'gaussian'
        assert config.band_psf['F087'].fwhm_arcsec == 0.18
        assert config.grism_psf.psf_type == 'roman_wfi'

    def test_gaussian_configs_unchanged(self):
        config = ObservationConfig.from_yaml(REGISTRY / 'canonical_P.yaml')
        assert {b: p.fwhm_arcsec for b, p in config.band_psf.items()} == {
            'F087': 0.18,
            'F158': 0.18,
        }
        assert config.grism_psf.fwhm_arcsec == 0.18

    def test_unknown_type_raises(self, tmp_path):
        for channel in ('broadband', 'grism'):
            d = _config_dict()
            d['psf'][channel] = {'type': 'imcom'}
            with pytest.raises(NotImplementedError, match='not supported'):
                ObservationConfig.from_yaml(_write_config(tmp_path, d))

    def test_roman_with_fwhm_key_raises(self, tmp_path):
        d = _config_dict()
        d['psf']['grism'] = {'type': 'roman_wfi', 'fwhm_arcsec': 0.18}
        with pytest.raises(ValueError, match='unknown keys'):
            ObservationConfig.from_yaml(_write_config(tmp_path, d))

    def test_bad_sca_raises(self, tmp_path):
        for sca in (0, 19):
            d = _config_dict()
            d['psf']['grism'] = {'type': 'roman_wfi', 'sca': sca}
            with pytest.raises(ValueError, match='sca'):
                ObservationConfig.from_yaml(_write_config(tmp_path, d))

    def test_bad_pupil_bin_raises(self, tmp_path):
        d = _config_dict()
        d['psf']['grism'] = {'type': 'roman_wfi', 'pupil_bin': 0}
        with pytest.raises(ValueError, match='pupil_bin'):
            ObservationConfig.from_yaml(_write_config(tmp_path, d))

    def test_non_integer_sca_pupil_bin_rejected(self, tmp_path):
        # YAML floats/strings must raise, not silently truncate via int()
        for key, val in (('sca', 10.9), ('sca', '10'), ('pupil_bin', 4.5)):
            d = _config_dict()
            d['psf']['grism'] = {'type': 'roman_wfi', key: val}
            with pytest.raises(ValueError, match='must be an integer'):
                ObservationConfig.from_yaml(_write_config(tmp_path, d))

    def test_psfspec_cross_field_validation(self):
        with pytest.raises(ValueError, match='roman_wfi-only'):
            PSFSpec(psf_type='gaussian', fwhm_arcsec=0.18, sca=10)
        with pytest.raises(ValueError, match='gaussian-only'):
            PSFSpec(psf_type='roman_wfi', fwhm_arcsec=0.18, sca=10, pupil_bin=4)
        with pytest.raises(ValueError, match='fwhm_arcsec'):
            PSFSpec(psf_type='gaussian')
        with pytest.raises(NotImplementedError, match='not supported'):
            PSFSpec(psf_type='airy')

    def test_folding_threshold_parsing(self, tmp_path):
        d = _config_dict()
        d['psf']['grism'] = {'type': 'roman_wfi', 'folding_threshold': 0.02}
        config = ObservationConfig.from_yaml(_write_config(tmp_path, d))
        assert config.grism_psf.folding_threshold == 0.02
        d['psf']['grism'] = {'type': 'roman_wfi'}
        config = ObservationConfig.from_yaml(_write_config(tmp_path, d))
        assert config.grism_psf.folding_threshold is None  # galsim default

    def test_folding_threshold_roman_only(self, tmp_path):
        d = _config_dict()
        d['psf']['grism'] = {
            'type': 'gaussian',
            'fwhm_arcsec': 0.18,
            'folding_threshold': 0.02,
        }
        with pytest.raises(ValueError, match='unknown keys'):
            ObservationConfig.from_yaml(_write_config(tmp_path, d))
        with pytest.raises(ValueError, match='roman_wfi-only'):
            PSFSpec(psf_type='gaussian', fwhm_arcsec=0.18, folding_threshold=0.02)

    def test_folding_threshold_bad_value_raises(self):
        for ft in (0.0, 1.0, -0.01):
            with pytest.raises(ValueError, match='folding_threshold'):
                PSFSpec(psf_type='roman_wfi', sca=10, pupil_bin=4, folding_threshold=ft)

    def test_mock_folding_threshold_rules(self, tmp_path):
        # mock (truth-render) kernels must be at least as accurate as fit
        # kernels; None means the galsim default 5e-3
        with pytest.raises(ValueError, match='at least as accurate'):
            PSFSpec(
                psf_type='roman_wfi', sca=10, pupil_bin=4, mock_folding_threshold=0.02
            )
        with pytest.raises(ValueError, match='at least as accurate'):
            PSFSpec(
                psf_type='roman_wfi',
                sca=10,
                pupil_bin=4,
                folding_threshold=0.01,
                mock_folding_threshold=0.02,
            )
        with pytest.raises(ValueError, match='roman_wfi-only'):
            PSFSpec(psf_type='gaussian', fwhm_arcsec=0.18, mock_folding_threshold=0.005)
        # equal and tighter-than-fit are both allowed
        PSFSpec(
            psf_type='roman_wfi',
            sca=10,
            pupil_bin=4,
            folding_threshold=0.01,
            mock_folding_threshold=0.01,
        )
        spec = PSFSpec(
            psf_type='roman_wfi',
            sca=10,
            pupil_bin=4,
            folding_threshold=0.01,
            mock_folding_threshold=0.001,
        )
        assert spec.mock_folding_threshold == 0.001
        # YAML parsing round-trip
        d = _config_dict()
        d['psf']['grism'] = {
            'type': 'roman_wfi',
            'folding_threshold': 0.01,
            'mock_folding_threshold': 0.005,
        }
        config = ObservationConfig.from_yaml(_write_config(tmp_path, d))
        assert config.grism_psf.mock_folding_threshold == 0.005

    def test_dual_fidelity_kernel_split(self):
        # a spec with a loosened fit threshold builds DIFFERENT mock vs fit
        # PSF objects (cache keys differ); an unsplit spec shares one object
        split = PSFSpec(
            psf_type='roman_wfi', sca=10, pupil_bin=4, folding_threshold=0.01
        )
        assert _build_grism_psf(split, 1.2, mock=True) is not _build_grism_psf(
            split, 1.2
        )
        unsplit = PSFSpec(psf_type='roman_wfi', sca=10, pupil_bin=4)
        assert _build_grism_psf(unsplit, 1.2, mock=True) is _build_grism_psf(
            unsplit, 1.2
        )
        # mock kernels are the larger (more accurate) ones
        fine_ps = PIXEL_SCALE / 3
        assert _build_grism_psf(split, 1.2, mock=True).getGoodImageSize(
            fine_ps
        ) > _build_grism_psf(split, 1.2).getGoodImageSize(fine_ps)


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
        # galsim.roman effective wavelengths are in nm; F158/F184 pivots
        # are ~1.58/1.84 um, a loose bracket guards a silent unit change
        lam_f158 = _band_effective_wavelength_nm('F158')
        lam_f184 = _band_effective_wavelength_nm('F184')
        assert 1500.0 < lam_f158 < 1650.0
        assert 1750.0 < lam_f184 < 1900.0

    def test_official_and_galsim_band_names_agree(self):
        # official Roman filter names map onto galsim's legacy WFIRST keys
        assert _band_effective_wavelength_nm('F158') == (
            _band_effective_wavelength_nm('H158')
        )
        assert _band_effective_wavelength_nm('F087') == (
            _band_effective_wavelength_nm('Z087')
        )

    def test_non_roman_band_raises(self):
        with pytest.raises(ValueError, match='not a Roman WFI filter'):
            _build_band_psf(ROMAN_SPEC, 'F999')

    def test_kernel_unit_normalized_and_finite(self):
        psf = _build_grism_psf(ROMAN_SPEC, 1.2)
        pin = _pinned_kernel_size(PIXEL_SCALE)
        psf_data = precompute_psf_fft(
            psf, image_shape=(32, 32), pixel_scale=PIXEL_SCALE, kernel_size=pin
        )
        kernel_fft = np.asarray(psf_data.kernel_fft)
        assert np.all(np.isfinite(kernel_fft))
        # the DC component is the padded kernel's sum: the unit-normalized
        # drawn kernel's flux within half-width N - 1 of the stamp (the only
        # part that can reach the retained crop); the Roman wings put a few
        # percent beyond 31 coarse pixels
        full = psf.drawImage(nx=pin, ny=pin, scale=PIXEL_SCALE).array.astype(float)
        full /= full.sum()
        center = pin // 2
        reachable = full[center - 31 : center + 32, center - 31 : center + 32].sum()
        assert reachable < 1.0
        assert kernel_fft[0, 0].real == pytest.approx(reachable, abs=1e-12)
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
        config = ObservationConfig.from_yaml(_write_config(tmp_path, d))
        size = _grism_psf_kernel_size(config, spec)
        fine_ps = config.pixel_scale_arcsec / spec.render_oversample
        # dev spec draws z uniform on [1.0, 1.9]: pin at z=1.9
        assert size == _pinned_kernel_size(fine_ps, z_max=1.9)
        assert size % 2 == 1
        # gaussian grism psf needs no pinning
        gaussian_config = ObservationConfig.from_yaml(REGISTRY / 'canonical_Q.yaml')
        assert _grism_psf_kernel_size(gaussian_config, spec) is None


# ==============================================================================
# Kernel conventions: orientation, parity, centering, normalization, pixel
# contract -- with the real (asymmetric) roman kernel. Round-PSF tests are
# blind to a transpose/flip bug; the roman diffraction spikes are not.
# ==============================================================================


class TestRomanKernelConventions:
    """Both PSF pathways against GalSim references on an asymmetric kernel.

    Bounds provenance (measured 2026-07-18, SCA 10, pupil_bin 4, H158
    effective wavelength, 32 px stamp at 0.11 arcsec, oversample 3, sheared
    inclined-exponential scene rscale=0.35, g=(0.05,-0.03), offset
    (0.11,-0.055)):
    - k-space fused path vs galsim native: max|d|/peak = 7.0e-4; bound 2e-3
      (~3x measured) chosen deliberately BELOW the measured residual of a
      transposed kernel (8.0e-3) so a parity bug cannot pass; the transpose
      control asserts that failure mode stays detectable.
    - delta-image patch identity (real-space kernel path): 7.1e-8 (FFT
      roundoff); bound 1e-6 (~10x). Flipped/transposed comparisons measured
      6.6e-2 / 2.5e-2; controls assert > 1e-2.
    - real-space path end-to-end vs galsim: 2.7e-3 (dominated by the
      point-sampled-SB quadrature at oversample 3, not orientation);
      bound 3e-2 (10x, rounded up).
    """

    def _scene(self):
        return dict(
            cosi=0.6,
            theta_int=np.pi / 4,
            g1=0.05,
            g2=-0.03,
            flux=1.0,
            rscale=0.35,
            h_over_r=0.1,
            x0=0.11,
            y0=-0.055,
        )

    def _psf(self):
        return _build_band_psf(ROMAN_SPEC, 'F158')

    def test_kspace_path_matches_galsim_and_catches_transpose(self):
        import dataclasses

        import jax.numpy as jnp

        from kl_pipe.intensity import InclinedExponentialModel
        from kl_pipe.synthetic import _generate_sersic_galsim

        pars = self._scene()
        psf = self._psf()
        image_pars = ImagePars(shape=(32, 32), pixel_scale=PIXEL_SCALE, indexing='ij')
        model = InclinedExponentialModel()
        obs = build_image_obs(
            image_pars,
            psf=psf,
            render_config=RenderConfig(oversample=3),
            int_model=model,
        )
        theta = jnp.array(
            [
                pars['cosi'],
                pars['theta_int'],
                pars['g1'],
                pars['g2'],
                pars['flux'],
                pars['rscale'],
                pars['h_over_r'],
                pars['x0'],
                pars['y0'],
            ]
        )
        ours = np.array(model.render_image(theta, obs=obs))
        ref = _generate_sersic_galsim(
            image_pars,
            flux=pars['flux'],
            rscale=pars['rscale'],
            n_sersic=1.0,
            cosi=pars['cosi'],
            theta_int=pars['theta_int'],
            g1=pars['g1'],
            g2=pars['g2'],
            x0=pars['x0'],
            y0=pars['y0'],
            h_over_r=pars['h_over_r'],
            psf=psf,
            method='auto',
        )
        peak = np.max(np.abs(ref))
        assert np.max(np.abs(ours - ref)) / peak < 2e-3

        # parity control: a transposed fused kernel must fail the same bound,
        # proving the comparison is sensitive to an orientation bug
        obs_t = dataclasses.replace(obs, kspace_psf_fft=obs.kspace_psf_fft.T)
        ours_t = np.array(model.render_image(theta, obs=obs_t))
        assert np.max(np.abs(ours_t - ref)) / peak > 2e-3

    def test_realspace_kernel_delta_identity_and_parity(self):
        import jax.numpy as jnp

        from kl_pipe.psf import convolve_fft

        psf = self._psf()
        oversample = 3
        fine_ps = PIXEL_SCALE / oversample
        psf_data = precompute_psf_fft(
            psf, image_shape=(32, 32), pixel_scale=PIXEL_SCALE, oversample=oversample
        )
        fine_n = 32 * oversample
        delta = np.zeros((fine_n, fine_n))
        r0, c0 = fine_n // 2 + 7, fine_n // 2 - 5
        delta[r0, c0] = 1.0
        conv = np.array(convolve_fft(jnp.asarray(delta), psf_data, bin=False))

        kern_size = psf.getGoodImageSize(fine_ps)
        if kern_size % 2 == 0:
            kern_size += 1
        kimg = psf.drawImage(nx=kern_size, ny=kern_size, scale=fine_ps).array
        kimg = kimg / kimg.sum()
        h = 20
        kc = kern_size // 2
        patch = conv[r0 - h : r0 + h + 1, c0 - h : c0 + h + 1]
        kpatch = kimg[kc - h : kc + h + 1, kc - h : kc + h + 1]
        kpeak = np.max(kpatch)

        # a delta convolved with the kernel must reproduce the drawn kernel
        # stamp at the delta position: validates centering (no half-pixel
        # offset from the pad+roll) and array orientation of the FFT path
        assert np.max(np.abs(patch - kpatch)) / kpeak < 1e-6
        # parity controls: flipped or transposed stamps must NOT match
        assert np.max(np.abs(patch - kpatch[::-1, ::-1])) / kpeak > 1e-2
        assert np.max(np.abs(patch - kpatch.T)) / kpeak > 1e-2

    def test_realspace_path_matches_galsim(self):
        import galsim as gs
        import jax.numpy as jnp

        from kl_pipe.psf import convolve_fft

        pars = self._scene()
        psf = self._psf()
        oversample = 3
        fine_ps = PIXEL_SCALE / oversample
        fine_n = 32 * oversample
        psf_data = precompute_psf_fft(
            psf, image_shape=(32, 32), pixel_scale=PIXEL_SCALE, oversample=oversample
        )
        gal = (
            gs.InclinedExponential(
                inclination=np.arccos(pars['cosi']) * gs.radians,
                scale_radius=pars['rscale'],
                scale_h_over_r=pars['h_over_r'],
                flux=pars['flux'],
            )
            .rotate(pars['theta_int'] * gs.radians)
            .shear(g1=pars['g1'], g2=pars['g2'])
            .shift(pars['x0'], pars['y0'])
        )
        # point-sampled SB through our kernel: the drawImage kernel's own
        # fine-pixel factor is the quadrature weight that turns the discrete
        # convolution into the continuous one (NOT a pixel double-count; the
        # detector coarse pixel is applied exactly once, downstream)
        sb = gal.drawImage(nx=fine_n, ny=fine_n, scale=fine_ps, method='sb').array
        conv_sb = np.array(convolve_fft(jnp.asarray(sb), psf_data, bin=False))
        ref = (
            gs.Convolve(gal, psf)
            .drawImage(nx=fine_n, ny=fine_n, scale=fine_ps, method='auto')
            .array
            / fine_ps**2
        )
        assert np.max(np.abs(conv_sb - ref)) / np.max(np.abs(ref)) < 3e-2


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
        psf = _build_band_psf(ROMAN_SPEC, 'F158')
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

    def test_with_render_config_preserves_pinned_kernel(self):
        # regression: InferenceTask.from_obs rebuilds explicit-rc grism obs via
        # with_render_config to fill line_window_halfwidth on the analytic
        # path; the rebuild must keep the pinned kernel size, else shapes
        # vary with z again
        import dataclasses

        from kl_pipe.dispersion import build_grism_pars_for_line

        pin = _pinned_kernel_size(PIXEL_SCALE)
        z = Z_LO  # far from the pinning reference so auto size != pin
        grism_pars = build_grism_pars_for_line(
            LINE_LAMBDAS['Halpha'],
            redshift=z,
            image_pars=ImagePars(
                shape=(32, 32), pixel_scale=PIXEL_SCALE, indexing='ij'
            ),
            dispersion=1.1,
        )
        rc = RenderConfig(oversample=1)
        obs = build_grism_obs(
            grism_pars,
            z=z,
            psf=_build_grism_psf(ROMAN_SPEC, z),
            render_config=rc,
            psf_kernel_size=pin,
        )
        rebuilt = obs.with_render_config(
            dataclasses.replace(rc, line_window_halfwidth=3)
        )
        assert rebuilt.psf_kernel_size == pin
        assert rebuilt.psf_data.padded_shape == obs.psf_data.padded_shape
        assert rebuilt.psf_data.kernel_fft.shape == obs.psf_data.kernel_fft.shape

    def test_roman_psf_cache_bounded(self):
        from kl_pipe.ensemble import mocks as mocks_mod

        mocks_mod._ROMAN_PSF_CACHE.clear()
        n = mocks_mod._ROMAN_PSF_CACHE_MAX + 5
        for i in range(n):
            _build_grism_psf(ROMAN_SPEC, 1.0 + 0.01 * i)
        assert len(mocks_mod._ROMAN_PSF_CACHE) <= mocks_mod._ROMAN_PSF_CACHE_MAX

    def test_folding_threshold_shrinks_pinned_kernel(self):
        # a looser folding threshold trades far-wing truncation for a much
        # smaller kernel stamp (and padded convolution FFT); measured at
        # z=1.9, fine scale 0.11/3: 992 -> 252 px with 99.93% of flux still
        # inside the stamp
        loose = PSFSpec(
            psf_type='roman_wfi', sca=10, pupil_bin=4, folding_threshold=0.02
        )
        fine_ps = PIXEL_SCALE / 3
        size_default = _build_grism_psf(ROMAN_SPEC, Z_HI).getGoodImageSize(fine_ps)
        size_loose = _build_grism_psf(loose, Z_HI).getGoodImageSize(fine_ps)
        assert size_loose < size_default / 2
        # distinct cache entries: threshold is part of the key
        assert _build_grism_psf(loose, Z_HI) is not _build_grism_psf(ROMAN_SPEC, Z_HI)

    def test_effective_maxk_ignores_kvalue_sidelobes(self):
        # regression: the interpolated-pupil roman PSF's kValue bounces back
        # above the 1e-3 threshold beyond its physical band limit (psf.maxk:
        # the measured optical FT is ~5e-7 at the aperture cutoff but ~2e-2
        # at k=70). The grism scan must not follow those sidelobes, else the
        # priors-implied oversample explodes (7 vs the config's 3) and every
        # production roman fit raises in InferenceTask.from_obs.
        from kl_pipe.intensity import build_intensity_model
        from kl_pipe.render import compute_effective_maxk_grism

        psf = _build_grism_psf(ROMAN_SPEC, 1.2)
        model = build_intensity_model('default')
        params = {
            'flux': 1.0,
            'rscale': 0.35,
            'cosi': 0.5,
            'h_over_r': 0.1,
            'theta_int': 0.0,
            'g1': 0.0,
            'g2': 0.0,
            'x0': 0.0,
            'y0': 0.0,
        }
        # grad_v_max/sigma_v chosen so the bare cube bandwidth (~370
        # rad/arcsec) is far beyond the PSF band limit (~49): the scan
        # result must be capped by the PSF, not the sidelobes
        maxk = compute_effective_maxk_grism(
            model, params, sigma_v=20.0, grad_v_max=2000.0, psf=psf
        )
        assert maxk <= float(psf.maxk) + 1e-9


def _folding_scan_inputs(tmp_path, ft, seed=11):
    """FitInputs for the folding-threshold scan scene (single F158 band +
    single roll, roman PSFs, oversample 3, z=1.2, line SNR 100)."""
    d = _config_dict()
    d['id'] = f'roman_folding_{ft}'
    d['bands'] = ['F158']
    d['psf'] = {
        'broadband': {'type': 'roman_wfi'},
        'grism': {'type': 'roman_wfi'},
    }
    if ft is not None:
        d['psf']['broadband']['folding_threshold'] = ft
        d['psf']['grism']['folding_threshold'] = ft
    config = ObservationConfig.from_yaml(_write_config(tmp_path, d))
    spec = EnsembleSpec.from_yaml(DEV_SPEC)
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
    return (
        build_fit_inputs(
            truth,
            seed,
            spec,
            config,
            band_snrs={b: 100.0 for b in config.bands},
            line_snr=100.0,
        ),
        truth,
    )


@pytest.mark.slow
@pytest.mark.diagnostic_plots
def test_folding_threshold_diagnostic_plot(tmp_path):
    """Diagnostic figure for the roman_wfi folding_threshold knob: kernel
    stamp / padded-FFT size, log-posterior+gradient eval time, and render
    distortion vs the default-threshold reference, plus per-threshold grism
    residual maps in noise units (line SNR 100). Values are measured fresh
    each run; the figure and CSV land in tests/out/roman_psf_folding/."""
    import time

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    import jax

    from kl_pipe.sampling import InferenceTask

    out_dir = REPO_ROOT / 'tests' / 'out' / 'roman_psf_folding'
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [None, 0.01, 0.02, 0.05]
    fine_ps = PIXEL_SCALE / 3
    rows = []
    renders = {}
    for ft in thresholds:
        inputs, truth = _folding_scan_inputs(tmp_path, ft)
        roll0 = inputs.grism_obs['roll0']
        renders[ft] = np.asarray(inputs.source.render_grism(truth, roll0))

        spec_ft = (
            ROMAN_SPEC
            if ft is None
            else PSFSpec(
                psf_type='roman_wfi', sca=10, pupil_bin=4, folding_threshold=ft
            )
        )
        stamp = int(_build_grism_psf(spec_ft, Z_HI).getGoodImageSize(fine_ps))

        task = InferenceTask.from_obs(
            inputs.source,
            inputs.priors,
            image_obs=inputs.image_obs,
            grism_obs=inputs.grism_obs,
        )
        fn = task.get_log_posterior_and_grad_fn()
        theta0 = np.asarray(task.sample_prior(jax.random.PRNGKey(0), 1))[0]
        val, grad = fn(theta0)  # compile
        float(val), np.asarray(grad)
        times = []
        for i in range(10):
            t0 = time.perf_counter()
            v, g = fn(theta0 * (1.0 + 1e-4 * (i + 1)))
            float(v), np.asarray(g)
            times.append(time.perf_counter() - t0)

        rows.append(
            {
                'folding_threshold': 5e-3 if ft is None else ft,
                'kernel_stamp_fine_px': stamp,
                'padded_fft_side': int(roll0.psf_data.padded_shape[0]),
                'eval_median_ms': 1e3 * float(np.median(times)),
                'sigma_pix': float(np.sqrt(float(np.asarray(roll0.variance)))),
            }
        )

    ref = renders[None]
    sigma = rows[0]['sigma_pix']
    for row, ft in zip(rows, thresholds):
        diff = renders[ft] - ref
        row['maxabs_over_peak'] = float(np.max(np.abs(diff)) / np.max(ref))
        row['stamp_chi_snr100'] = float(np.sqrt(np.sum(diff**2)) / sigma)

    with open(out_dir / 'folding_scan.csv', 'w') as f:
        keys = list(rows[0])
        f.write(','.join(keys) + '\n')
        for row in rows:
            f.write(','.join(str(row[k]) for k in keys) + '\n')

    fts = [r['folding_threshold'] for r in rows]
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(fts, [r['kernel_stamp_fine_px'] for r in rows], 'o-', label='kernel stamp')
    ax.plot(fts, [r['padded_fft_side'] for r in rows], 's--', label='padded FFT side')
    ax.set_xscale('log')
    ax.set_xlabel('folding_threshold')
    ax.set_ylabel('size [fine px]')
    ax.legend()
    ax.set_title('grism kernel / FFT size (pin z=1.9)')

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(fts, [r['eval_median_ms'] for r in rows], 'o-')
    ax.set_xscale('log')
    ax.set_xlabel('folding_threshold')
    ax.set_ylabel('logpost+grad eval [ms]')
    ax.set_title('eval time (this host, this run)')

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(
        fts[1:], [r['maxabs_over_peak'] for r in rows[1:]], 'o-', label='max|d|/peak'
    )
    ax.plot(
        fts[1:],
        [r['stamp_chi_snr100'] for r in rows[1:]],
        's--',
        label='whole-stamp chi @ line SNR 100',
    )
    ax.axhline(1.0, color='k', lw=0.8, ls=':', label='chi = 1')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('folding_threshold')
    ax.set_title('render distortion vs default kernel')
    ax.legend()

    for i, ft in enumerate(thresholds[1:]):
        ax = fig.add_subplot(gs[1, i])
        resid = (renders[ft] - ref) / sigma
        vmax = max(1e-12, float(np.max(np.abs(resid))))
        im = ax.imshow(resid, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f'(render - ref) / sigma, ft={ft}')

    fig.suptitle(
        'roman_wfi PSF folding_threshold: cost vs fidelity '
        '(1-band + 1-roll scan scene, z=1.2, kernel pinned at z=1.9)'
    )
    fig.tight_layout()
    fig.savefig(out_dir / 'folding_threshold_diagnostics.png', dpi=130)
    plt.close(fig)

    assert (out_dir / 'folding_threshold_diagnostics.png').exists()
    assert (out_dir / 'folding_scan.csv').exists()


@pytest.mark.slow
def test_folding_threshold_render_fidelity(tmp_path):
    """Render distortion from a loosened kernel folding threshold, against
    the default-threshold reference on the same scene. Bounds are 10x the
    measured distortion (2026-07-18, canonical_P_roman geometry, z=1.2,
    kernel pinned at z=1.9, line SNR 100): grism max|d|/peak = 3.3e-4 ->
    4e-3; whole-stamp chi at SNR 100 = 0.51 -> 6.0. The knob does not touch
    the broadband path (its kernel FT is evaluated analytically in k-space,
    independent of the drawn stamp), pinned here by exact equality."""

    def _clean_renders(ft):
        inputs, truth = _folding_scan_inputs(tmp_path, ft)
        grism = np.asarray(inputs.source.render_grism(truth, inputs.grism_obs['roll0']))
        band = np.asarray(
            inputs.source.render_broadband(truth, inputs.image_obs['F158'], 'F158')
        )
        sigma = float(np.sqrt(float(np.asarray(inputs.grism_obs['roll0'].variance))))
        return grism, band, sigma

    grism_ref, band_ref, sigma = _clean_renders(None)
    grism_ft, band_ft, _ = _clean_renders(0.02)

    diff = grism_ft - grism_ref
    assert np.max(np.abs(diff)) / np.max(grism_ref) < 4e-3
    assert np.sqrt(np.sum(diff**2)) / sigma < 6.0
    np.testing.assert_array_equal(band_ft, band_ref)


@pytest.mark.slow
def test_dual_fidelity_data_uses_mock_kernel(tmp_path):
    """build_fit_inputs renders the mock DATA through the mock-fidelity
    kernel while the fit obs carries the loosened fit kernel: a split config
    (fit ft=0.01, mock default) and a matched config (both 0.01) with the
    same noise seed must produce different grism data but identical fit-obs
    kernel shapes."""

    def _inputs(mock_ft):
        d = _config_dict()
        d['id'] = f'dual_fid_{mock_ft}'
        d['bands'] = ['F158']
        d['psf'] = {
            'broadband': {'type': 'roman_wfi', 'folding_threshold': 0.01},
            'grism': {'type': 'roman_wfi', 'folding_threshold': 0.01},
        }
        if mock_ft is not None:
            d['psf']['grism']['mock_folding_threshold'] = mock_ft
            d['psf']['broadband']['mock_folding_threshold'] = mock_ft
        config = ObservationConfig.from_yaml(_write_config(tmp_path, d))
        spec = EnsembleSpec.from_yaml(DEV_SPEC)
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
        return build_fit_inputs(
            truth,
            11,
            spec,
            config,
            band_snrs={b: 100.0 for b in config.bands},
            line_snr=100.0,
        )

    split = _inputs(None)  # mock at galsim default, fit at 0.01
    matched = _inputs(0.01)  # both at 0.01

    roll_split = split.grism_obs['roll0']
    roll_matched = matched.grism_obs['roll0']
    # same fit-kernel shapes either way
    assert roll_split.psf_data.padded_shape == roll_matched.psf_data.padded_shape
    # but the data was rendered through different (mock) kernels
    assert not np.array_equal(
        np.asarray(roll_split.data), np.asarray(roll_matched.data)
    )


@pytest.mark.slow
def test_from_obs_roman_production_path(tmp_path):
    """The worker's exact fit-construction path with roman PSFs on both
    channels: build_fit_inputs at the config oversample, then
    InferenceTask.from_obs, then one finite log-posterior + gradient.
    Regression: the priors-implied render config once demanded oversample 7
    (kValue sidelobes beyond the PSF band limit) and from_obs raised on
    every production roman fit."""
    import jax

    from kl_pipe.sampling import InferenceTask

    spec = EnsembleSpec.from_yaml(DEV_SPEC)
    d = _config_dict()
    d['id'] = 'roman_prod_path'
    d['bands'] = ['F158']
    d['psf'] = {
        'broadband': {'type': 'roman_wfi'},
        'grism': {'type': 'roman_wfi'},
    }
    config = ObservationConfig.from_yaml(_write_config(tmp_path, d))

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
        truth,
        11,
        spec,
        config,
        band_snrs={b: 100.0 for b in config.bands},
        line_snr=100.0,
    )
    task = InferenceTask.from_obs(
        inputs.source,
        inputs.priors,
        image_obs=inputs.image_obs,
        grism_obs=inputs.grism_obs,
    )
    fn = task.get_log_posterior_and_grad_fn()
    theta0 = task.sample_prior(jax.random.PRNGKey(0), 1)[0]
    val, grad = fn(theta0)
    assert np.isfinite(float(val))
    assert np.all(np.isfinite(np.asarray(grad)))


@pytest.mark.slow
def test_build_fit_inputs_roman_smoke(tmp_path):
    """End-to-end mock construction with roman_wfi PSFs on both channels:
    one F158 band + one grism roll, oversample 1 to keep the kernel grids
    small. Asserts valid PSFData on every obs, finite noisy data, and a
    finite truth render through the fit's own obs."""
    # render oversample lives on the spec now: drop DEV_SPEC's 3 to 1
    spec_dict = yaml.safe_load(DEV_SPEC.read_text())
    spec_dict['model'] = {'render': {'oversample': 1}}
    spec_path = tmp_path / 'spec.yaml'
    spec_path.write_text(yaml.safe_dump(spec_dict))
    spec = EnsembleSpec.from_yaml(spec_path)
    d = _config_dict()
    d['id'] = 'roman_smoke'
    d['bands'] = ['F158']
    d['psf'] = {
        'broadband': {'type': 'roman_wfi'},
        'grism': {'type': 'roman_wfi'},
    }
    config = ObservationConfig.from_yaml(_write_config(tmp_path, d))

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
        truth,
        11,
        spec,
        config,
        band_snrs={b: 100.0 for b in config.bands},
        line_snr=100.0,
    )

    assert set(inputs.image_obs) == {'F158'}
    assert set(inputs.grism_obs) == {'roll0'}
    for obs in list(inputs.image_obs.values()) + list(inputs.grism_obs.values()):
        assert obs.psf_data is not None
        assert np.all(np.isfinite(np.asarray(obs.data)))
        assert float(np.asarray(obs.variance)) > 0

    band_render = np.asarray(
        inputs.source.render_broadband(truth, inputs.image_obs['F158'], 'F158')
    )
    grism_render = np.asarray(
        inputs.source.render_grism(truth, inputs.grism_obs['roll0'])
    )
    assert np.all(np.isfinite(band_render))
    assert np.all(np.isfinite(grism_render))
    assert band_render.sum() > 0
    assert grism_render.sum() > 0
