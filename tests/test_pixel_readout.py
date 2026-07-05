"""
Enforcement tests for the post-dispersion pixel-response readout.

The coarse-pixel BoxPixel sinc applied on the fine k-grid IS the
coarse-pixel integration: after the IFFT each fine cell holds the
coarse-box-averaged SB centered on that cell, and the readout must SAMPLE
that field at each coarse pixel center (the center fine cell of each
oversample block; exact for odd oversample). Mean-binning the block
instead convolves the already box-averaged field with a second
coarse-pixel-wide box: peaks of compact sources come out several percent
low while total flux is exactly conserved. That historical bug is pinned
by the canary test below so the fix can never silently regress.

Also enforced here: the grism wavelength-grid (n_lambda) velocity-
entanglement resolution canary. The `to_cube_pars` default spaces slices
by ~1 dispersion pixel (~250 km/s at the flagship config), which
quantizes the dispersed POSITION of spatially-varying Doppler shifts by
up to half a bin. The canary freezes the measured deviation of the
current default against a refined wavelength grid, so any silent change
to the default (better or worse) trips a test and forces a conscious
re-measure.
"""

import numpy as np
import pytest
import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from scipy.special import erf

from kl_pipe.grism import _apply_post_dispersion_pixel_response
from kl_pipe.observation import _fine_pixel_response_fft
from kl_pipe.parameters import ImagePars
from kl_pipe.render import RenderConfig


COARSE_PS = 0.11  # arcsec, Roman-like
N_COARSE = 32


def _gaussian_setup(oversample: int, sigma: float):
    """Analytic Gaussian SB field on the fine grid + erf closed form.

    Returns (sb_fine, pixel_response_fft, exact_coarse_flux) where
    ``exact_coarse_flux`` is the closed-form box integral of the Gaussian
    over each coarse pixel (flux per coarse pixel).
    """
    fine_ps = COARSE_PS / oversample
    n_fine = N_COARSE * oversample
    idx = (np.arange(n_fine) - (n_fine - 1) / 2.0) * fine_ps
    Yf, Xf = np.meshgrid(idx, idx, indexing='ij')
    sb = np.exp(-0.5 * (Xf**2 + Yf**2) / sigma**2)  # SB units

    fine_ip = ImagePars(shape=(n_fine, n_fine), pixel_scale=fine_ps, indexing='ij')
    prf = _fine_pixel_response_fft(fine_ip, COARSE_PS)

    ic = (np.arange(N_COARSE) - (N_COARSE - 1) / 2.0) * COARSE_PS
    Yc, Xc = np.meshgrid(ic, ic, indexing='ij')

    def box_int_1d(x0):
        a = (x0 - COARSE_PS / 2) / (sigma * np.sqrt(2))
        b = (x0 + COARSE_PS / 2) / (sigma * np.sqrt(2))
        return sigma * np.sqrt(np.pi / 2) * (erf(b) - erf(a))

    exact = box_int_1d(Xc) * box_int_1d(Yc)
    return sb, prf, exact


class TestClosedFormGate:
    """Readout == erf closed-form coarse-box integral of a Gaussian."""

    # measured agreement (2026-07-04, float64): 5.2e-8 .. 2.0e-7 relative
    # across oversample {3, 5} x sigma {0.08, 0.2}. Frozen with ~10x margin.
    GATE_REL = 2e-6

    @pytest.mark.parametrize('oversample', [3, 5])
    @pytest.mark.parametrize('sigma', [0.08, 0.2])
    def test_matches_erf_box_integral(self, oversample, sigma):
        sb, prf, exact = _gaussian_setup(oversample, sigma)
        out = np.asarray(
            _apply_post_dispersion_pixel_response(
                jnp.asarray(sb), prf, (N_COARSE, N_COARSE), oversample, COARSE_PS
            )
        )
        rel = np.abs(out - exact).max() / exact.max()
        assert rel < self.GATE_REL, (
            f'readout deviates from closed-form box integral by {rel:.3e} '
            f'relative (gate {self.GATE_REL:.0e}) at oversample={oversample}, '
            f'sigma={sigma}'
        )

    @pytest.mark.parametrize('oversample', [3, 5])
    def test_flux_conservation(self, oversample):
        # the BoxPixel sinc has exact zeros at nonzero multiples of the
        # coarse sampling frequency, so the coarse-center subsample sum
        # equals the DFT total flux exactly (up to FFT roundoff).
        # measured deviation <= 2.0e-7 relative; frozen with ~10x margin.
        sb, prf, _ = _gaussian_setup(oversample, sigma=0.08)
        fine_ps = COARSE_PS / oversample
        out = np.asarray(
            _apply_post_dispersion_pixel_response(
                jnp.asarray(sb), prf, (N_COARSE, N_COARSE), oversample, COARSE_PS
            )
        )
        total = sb.sum() * fine_ps**2
        assert abs(out.sum() - total) / total < 2e-6

    def test_oversample_one_is_area_conversion(self):
        rng = np.random.default_rng(0)
        img = rng.uniform(size=(N_COARSE, N_COARSE))
        out = np.asarray(
            _apply_post_dispersion_pixel_response(
                jnp.asarray(img), None, (N_COARSE, N_COARSE), 1, COARSE_PS
            )
        )
        np.testing.assert_allclose(out, img * COARSE_PS**2, rtol=1e-14)


class TestValidation:
    """Loud errors: even oversample has no coarse-center-aligned fine cell."""

    @pytest.mark.parametrize('oversample', [2, 4])
    def test_even_oversample_raises(self, oversample):
        n_fine = N_COARSE * oversample
        sb = jnp.ones((n_fine, n_fine))
        with pytest.raises(ValueError, match='odd'):
            _apply_post_dispersion_pixel_response(
                sb,
                jnp.ones((n_fine, n_fine)),
                (N_COARSE, N_COARSE),
                oversample,
                COARSE_PS,
            )

    @pytest.mark.parametrize('oversample', [2, 4, 0, -3])
    def test_render_config_rejects_non_positive_odd(self, oversample):
        with pytest.raises(ValueError, match='odd'):
            RenderConfig(oversample=oversample)

    def test_shape_mismatch_raises(self):
        sb = jnp.ones((N_COARSE * 3, N_COARSE * 3))
        with pytest.raises(ValueError, match='does not match'):
            _apply_post_dispersion_pixel_response(
                sb, jnp.ones_like(sb), (N_COARSE // 2, N_COARSE), 3, COARSE_PS
            )


class TestMeanBinCanary:
    """Pin the historical bug: mean-bin readout is >1% wrong on compact
    sources while conserving flux exactly.

    The old readout is reimplemented inline as the reference-of-what-was-
    wrong. If this test ever fails, either the sinc semantics changed or
    someone reintroduced block averaging -- both need explicit review.
    """

    @staticmethod
    def _old_mean_bin_readout(dispersed, prf, coarse_shape, oversample, coarse_ps):
        Nrow_c, Ncol_c = coarse_shape
        N = oversample
        img_fft = jnp.fft.fft2(dispersed)
        fine = jnp.fft.ifft2(img_fft * prf).real
        sb_coarse = fine.reshape(Nrow_c, N, Ncol_c, N).mean(axis=(1, 3))
        return sb_coarse * coarse_ps**2

    def test_old_readout_differs_from_closed_form(self):
        # measured: -6.9% peak error at sigma=0.08 (compact vs the 0.11"
        # pixel), flux conserved to <2e-7. The canary asserts >1%.
        sb, prf, exact = _gaussian_setup(oversample=3, sigma=0.08)
        old = np.asarray(
            self._old_mean_bin_readout(
                jnp.asarray(sb), prf, (N_COARSE, N_COARSE), 3, COARSE_PS
            )
        )
        rel = np.abs(old - exact).max() / exact.max()
        assert rel > 0.01, (
            f'mean-bin readout agrees with the closed form to {rel:.3e} -- '
            f'the double-convolution signature vanished; review the sinc '
            f'semantics before trusting this.'
        )
        # the bug conserves flux -- that is why flux tests never caught it
        assert abs(old.sum() / exact.sum() - 1.0) < 1e-5


class TestEntanglementCanary:
    """Freeze the CURRENT default n_lambda's velocity-entanglement error.

    A rotating disk maps different sky regions to different Doppler
    shifts; slices ~1 dispersion pixel apart (~250 km/s at this config)
    quantize the dispersed position of that structure. Measured deviation
    of the default wavelength grid vs a refined one (n_lambda=251, where
    the render is converged to ~0.3%): max|diff|/peak 5.3%, total-flux
    ratio 1.029 (the flux term is an O(dlam) wavelength-quadrature error
    on the CONTINUUM, also resolved by refining). Frozen as a two-sided
    window: a
    silent change to the `to_cube_pars` default (better OR worse) trips
    this test and must be consciously re-frozen with new measurements.
    """

    DEV_WINDOW = (0.03, 0.08)  # max|diff|/peak, measured 0.0533
    FLUX_WINDOW = (1.015, 1.045)  # flux ratio default/refined, measured 1.0294

    @pytest.fixture(scope='class')
    def renders(self):
        import dataclasses

        import galsim

        from kl_pipe.dispersion import build_grism_pars_for_line
        from kl_pipe.intensity import InclinedExponentialModel
        from kl_pipe.lines import EmissionLine, LINE_LAMBDAS
        from kl_pipe.observation import build_grism_obs
        from kl_pipe.source import SourceModel
        from kl_pipe.velocity import CenteredVelocityModel

        z = 1.0
        image_pars = ImagePars(
            shape=(N_COARSE, N_COARSE), pixel_scale=COARSE_PS, indexing='ij'
        )
        grism_pars = build_grism_pars_for_line(
            LINE_LAMBDAS['Halpha'],
            redshift=z,
            image_pars=image_pars,
            dispersion=1.1,
        )
        source = SourceModel(
            velocity_model=CenteredVelocityModel(),
            emission_lines={
                'Halpha': EmissionLine(
                    intensity=InclinedExponentialModel(),
                    continuum=InclinedExponentialModel(),
                )
            },
        )
        # flagship-like truth: rotating disk + flat continuum. The line
        # pins the position-quantization (entanglement) error; the
        # continuum pins the O(dlam) wavelength-quadrature flux error.
        pars = {
            'cosi': 0.6,
            'theta_int': np.pi / 4,
            'g1': 0.0,
            'g2': 0.0,
            'vel.v0': 0.0,
            'vel.vcirc': 200.0,
            'vel.rscale': 0.3,
            'Halpha.flux': 100.0,
            'Halpha.rscale': 0.25,
            'Halpha.h_over_r': 0.1,
            'Halpha.x0': 0.0,
            'Halpha.y0': 0.0,
            'Halpha.dispersion': 50.0,
            'Halpha.cont.flux_per_nm': 25.0,
            'Halpha.cont.rscale': 0.25,
            'Halpha.cont.h_over_r': 0.1,
            'Halpha.cont.x0': 0.0,
            'Halpha.cont.y0': 0.0,
            'z': z,
        }
        psf = galsim.Gaussian(fwhm=0.18)
        obs_default = build_grism_obs(
            grism_pars, z=z, psf=psf, render_config=RenderConfig(oversample=3)
        )
        obs_refined = dataclasses.replace(
            obs_default, cube_pars=grism_pars.to_cube_pars(z, n_lambda=251)
        )
        img_default = np.asarray(source.render_grism(pars, obs_default))
        img_refined = np.asarray(source.render_grism(pars, obs_refined))
        return img_default, img_refined

    def test_default_deviation_level_frozen(self, renders):
        img_default, img_refined = renders
        dev = np.abs(img_default - img_refined).max() / img_refined.max()
        lo, hi = self.DEV_WINDOW
        assert lo < dev < hi, (
            f'default-n_lambda entanglement deviation {dev:.4f} left the '
            f'frozen window ({lo}, {hi}). If the to_cube_pars default (or '
            f'the dispersion pipeline) changed, re-measure and re-freeze '
            f'consciously -- do not widen the window to silence this.'
        )

    def test_default_flux_excess_frozen(self, renders):
        img_default, img_refined = renders
        ratio = img_default.sum() / img_refined.sum()
        lo, hi = self.FLUX_WINDOW
        assert lo < ratio < hi, (
            f'default-n_lambda flux ratio {ratio:.4f} left the frozen '
            f'window ({lo}, {hi}); see test_default_deviation_level_frozen.'
        )
