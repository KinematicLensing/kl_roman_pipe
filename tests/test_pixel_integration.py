"""
Tests for pixel response and k-space pixel integration.

Tests:
A. Interface & defaults — BoxPixel, PixelResponse ABC, obs defaults
B. Point-sampling vs pixel response — verify sinc smooths images
C. Convergence: oversampling → sinc — oversample N→∞ converges to sinc
D. maxk / stepk validation vs GalSim — our formulas vs GalSim values
E. Grid adequacy — adaptive grid vs oversized reference
F. k-space power spectrum diagnostic — visual verification
G. Flux conservation — sum(image) * ps^2 ≈ flux
H. Edge cases — sub-pixel, extended, edge-on, high-n

GalSim pixel-integrated accuracy is validated by the existing regression
tests in test_intensity.py and test_intensity_spergel.py (method='auto',
1e-3 tolerance for non-cuspy profiles).
"""

import pytest
import numpy as np
import jax.numpy as jnp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from kl_pipe.pixel import PixelResponse, BoxPixel, _PIXEL_RESPONSE_UNSET
from kl_pipe.intensity import (
    InclinedExponentialModel,
    InclinedSpergelModel,
    InclinedSersicModel,
    _kspace_render_core,
)
from kl_pipe.render import (
    RenderConfig,
    compute_effective_maxk,
)
from kl_pipe.observation import build_image_obs, ImageObs
from kl_pipe.parameters import ImagePars
from kl_pipe.utils import get_test_dir


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def output_dir():
    out = get_test_dir() / 'out' / 'pixel_integration'
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture
def image_pars():
    return ImagePars((64, 64), 'ij', pixel_scale=0.11)


@pytest.fixture
def box_pixel():
    return BoxPixel(0.11)


@pytest.fixture
def exp_model():
    return InclinedExponentialModel()


@pytest.fixture
def exp_theta():
    # cosi, theta_int, g1, g2, flux, int_rscale, int_h_over_r, int_x0, int_y0
    return jnp.array([0.7, 0.5, 0.0, 0.0, 1e4, 0.3, 0.1, 0.0, 0.0])


# ==============================================================================
# A. Interface & default tests
# ==============================================================================


class TestPixelResponseInterface:
    """Test PixelResponse ABC and BoxPixel implementation."""

    def test_boxpixel_construction(self):
        bp = BoxPixel(0.11)
        assert bp.pixel_scale == 0.11

    def test_boxpixel_validation_negative(self):
        with pytest.raises(ValueError, match="pixel_scale must be > 0"):
            BoxPixel(-1.0)

    def test_boxpixel_validation_zero(self):
        with pytest.raises(ValueError, match="pixel_scale must be > 0"):
            BoxPixel(0.0)

    def test_boxpixel_ft_at_dc(self):
        bp = BoxPixel(0.11)
        KX = jnp.zeros((3, 3))
        KY = jnp.zeros((3, 3))
        ft = bp.ft(KX, KY)
        np.testing.assert_allclose(float(ft[0, 0]), 1.0, atol=1e-10)

    def test_boxpixel_ft_at_nyquist(self):
        bp = BoxPixel(0.11)
        k_nyq = np.pi / 0.11
        KX = jnp.array([[k_nyq]])
        KY = jnp.array([[0.0]])
        ft = bp.ft(KX, KY)
        # sinc(0.5) = 2/pi ≈ 0.6366
        np.testing.assert_allclose(float(ft[0, 0]), 2.0 / np.pi, atol=1e-6)

    def test_boxpixel_ft_first_zero(self):
        bp = BoxPixel(0.11)
        k_2nyq = 2 * np.pi / 0.11
        KX = jnp.array([[k_2nyq]])
        KY = jnp.array([[0.0]])
        ft = bp.ft(KX, KY)
        np.testing.assert_allclose(float(ft[0, 0]), 0.0, atol=1e-6)

    def test_boxpixel_maxk(self):
        bp = BoxPixel(0.11)
        maxk = bp.maxk(1e-3)
        assert maxk > 0
        # 2 / (1e-3 * 0.11) ≈ 18182
        np.testing.assert_allclose(maxk, 2.0 / (1e-3 * 0.11), rtol=1e-10)

    def test_boxpixel_maxk_validation(self):
        bp = BoxPixel(0.11)
        with pytest.raises(ValueError, match="threshold must be > 0"):
            bp.maxk(0.0)

    def test_abc_enforcement(self):
        with pytest.raises(TypeError):
            PixelResponse()

    def test_boxpixel_pytree_roundtrip(self):
        import jax

        bp = BoxPixel(0.11)
        leaves, treedef = jax.tree_util.tree_flatten(bp)
        bp2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert bp == bp2

    def test_boxpixel_repr(self):
        bp = BoxPixel(0.11)
        assert 'BoxPixel' in repr(bp)
        assert '0.11' in repr(bp)

    def test_boxpixel_equality(self):
        assert BoxPixel(0.11) == BoxPixel(0.11)
        assert BoxPixel(0.11) != BoxPixel(0.22)


class TestObsDefaults:
    """Test pixel response defaults on observation construction."""

    def test_default_boxpixel_from_pixel_scale(self, image_pars):
        obs = build_image_obs(image_pars)
        assert isinstance(obs.pixel_response, BoxPixel)
        assert obs.pixel_response.pixel_scale == image_pars.pixel_scale

    def test_explicit_none_disables(self, image_pars):
        obs = build_image_obs(image_pars, pixel_response=None)
        assert obs.pixel_response is None

    def test_custom_pixel_response(self, image_pars):
        bp = BoxPixel(0.22)
        obs = build_image_obs(image_pars, pixel_response=bp)
        assert obs.pixel_response.pixel_scale == 0.22

    def test_no_psf_preserves_oversample(self, image_pars):
        """Issue #38: oversample should NOT be forced to 1 without PSF."""
        obs = build_image_obs(image_pars, oversample=5)
        assert obs.oversample == 5
        assert obs.fine_X is not None

    def test_pixel_response_in_pytree(self, image_pars):
        import jax

        obs = build_image_obs(image_pars)
        leaves, treedef = jax.tree_util.tree_flatten(obs)
        obs2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert obs2.pixel_response == obs.pixel_response


# ==============================================================================
# B. Point-sampling vs pixel response
# ==============================================================================


class TestPointSamplingVsPixel:
    """Compare rendering with and without pixel response."""

    def test_pixel_response_smooths_peak(self, exp_model, exp_theta, image_pars):
        """Pixel-integrated image should have lower peak (sinc smooths cusp)."""
        obs_pix = build_image_obs(image_pars)
        obs_none = build_image_obs(image_pars, pixel_response=None)

        img_pix = exp_model.render_image(exp_theta, obs=obs_pix)
        img_none = exp_model.render_image(exp_theta, obs=obs_none)

        assert float(jnp.max(img_pix)) < float(jnp.max(img_none))

    def test_pixel_response_conserves_flux(self, exp_model, exp_theta, image_pars):
        """Sinc preserves DC → total flux should be nearly identical."""
        obs_pix = build_image_obs(image_pars)
        obs_none = build_image_obs(image_pars, pixel_response=None)

        img_pix = exp_model.render_image(exp_theta, obs=obs_pix)
        img_none = exp_model.render_image(exp_theta, obs=obs_none)

        flux_pix = float(jnp.sum(img_pix))
        flux_none = float(jnp.sum(img_none))
        np.testing.assert_allclose(flux_pix, flux_none, rtol=1e-4)

    def test_pixel_response_diagnostic(
        self, exp_model, exp_theta, image_pars, output_dir
    ):
        """Diagnostic: side-by-side comparison with fractional difference."""
        obs_pix = build_image_obs(image_pars)
        obs_none = build_image_obs(image_pars, pixel_response=None)

        img_pix = np.array(exp_model.render_image(exp_theta, obs=obs_pix))
        img_none = np.array(exp_model.render_image(exp_theta, obs=obs_none))

        peak = max(img_pix.max(), img_none.max())
        frac_diff = (img_pix - img_none) / peak

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        im0 = axes[0].imshow(img_none, origin='lower', vmax=peak)
        axes[0].set_title('Point-sampled')
        plt.colorbar(im0, ax=axes[0], label='SB')

        im1 = axes[1].imshow(img_pix, origin='lower', vmax=peak)
        axes[1].set_title('Pixel-integrated (sinc)')
        plt.colorbar(im1, ax=axes[1], label='SB')

        im2 = axes[2].imshow(
            frac_diff,
            origin='lower',
            cmap='RdBu_r',
            vmin=-np.max(np.abs(frac_diff)),
            vmax=np.max(np.abs(frac_diff)),
        )
        axes[2].set_title(
            f'(sinc - point) / peak\n'
            f'max|diff|/peak = {np.max(np.abs(frac_diff)):.4f}'
        )
        plt.colorbar(im2, ax=axes[2], label='frac')
        plt.tight_layout()
        plt.savefig(output_dir / 'point_vs_pixel.png', dpi=150)
        plt.close()


# ==============================================================================
# C. Convergence: oversampling → pixel integral
# ==============================================================================


class TestOversampleConvergence:
    """Verify oversample N→∞ converges to true pixel integral.

    Uses high-N oversampled rendering (N=21) as reference for the true
    pixel integral. Lower N should converge monotonically toward it.
    Also shows where sinc-only lands relative to the reference —
    the gap is the DFT aliasing floor of the sinc approach.
    """

    @pytest.fixture
    def _check_galsim(self):
        pytest.importorskip('galsim')

    def test_convergence(
        self, _check_galsim, exp_model, exp_theta, image_pars, output_dir
    ):
        """Compare pixel integration methods against GalSim reference.

        GalSim's drawImage(method='auto') is the authoritative pixel-
        integrated reference (uses the wrap operation internally).

        Both paths should converge to GalSim:
        - sinc + wrap converges quickly (by N=3)
        - oversample + bin (no sinc) converges more slowly (by N=15)
        """
        import galsim as gs
        from kl_pipe.synthetic import _generate_sersic_galsim

        ps = image_pars.pixel_scale
        gsp = gs.GSParams(
            folding_threshold=1e-4, maxk_threshold=1e-4, kvalue_accuracy=1e-6
        )
        params = exp_model.theta2pars(exp_theta)

        # GalSim reference
        gs_img = _generate_sersic_galsim(
            image_pars,
            flux=float(params['flux']),
            int_rscale=float(params['int_rscale']),
            n_sersic=1.0,
            cosi=float(params['cosi']),
            theta_int=float(params['theta_int']),
            g1=0.0,
            g2=0.0,
            int_x0=0.0,
            int_y0=0.0,
            int_h_over_r=float(params['int_h_over_r']),
            gsparams=gsp,
            method='auto',
        )
        gs_sb = gs_img / ps**2
        peak = np.max(np.abs(gs_sb))

        N_values = [1, 3, 5, 7, 9, 15, 21, 31]

        # path 1: oversample + bin (no sinc)
        resid_bin = []
        for N in N_values:
            img = np.array(
                exp_model.render_image(
                    exp_theta,
                    image_pars=image_pars,
                    render_config=RenderConfig(oversample=N),
                    pixel_response=None,
                )
            )
            resid_bin.append(np.max(np.abs(img - gs_sb)) / peak)

        # path 2: sinc + wrap at various oversample
        resid_wrap = []
        for N in N_values:
            img = np.array(
                exp_model.render_image(
                    exp_theta,
                    image_pars=image_pars,
                    render_config=RenderConfig(oversample=N),
                )
            )
            resid_wrap.append(np.max(np.abs(img - gs_sb)) / peak)

        # sinc+wrap at N>=3 should match GalSim to <1%
        assert (
            resid_wrap[1] < 0.01
        ), f"sinc+wrap at N=3 should match GalSim to <1%, got {resid_wrap[1]:.1%}"

        # bin-only at N=31 should converge to GalSim to <0.1%
        assert (
            resid_bin[-1] < 0.001
        ), f"bin-only at N=31 should match GalSim to <0.1%, got {resid_bin[-1]:.2%}"

        # diagnostic plot
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.semilogy(N_values, resid_bin, 'o-', label='oversample + bin (no sinc)')
        ax.semilogy(N_values, resid_wrap, 's-', label='sinc + wrap (production)')
        ax.set_xlabel('Oversample factor N')
        ax.set_ylabel("max|residual| / peak (vs GalSim method='auto')")
        ax.set_title(
            'Pixel integration accuracy vs GalSim\n'
            f'(exponential r_s=0.3" at {image_pars.pixel_scale}"/px)'
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'oversample_convergence.png', dpi=150)
        plt.close()


# ==============================================================================
# D. maxk / stepk validation vs GalSim
# ==============================================================================


class TestMaxkStepk:
    """Validate maxk/stepk methods against expected values."""

    def test_exponential_maxk_analytic(self):
        model = InclinedExponentialModel()
        params = {'int_rscale': 0.3, 'cosi': 1.0}
        maxk = model.maxk(params, threshold=1e-3)
        expected = np.sqrt(1e-3 ** (-2.0 / 3.0) - 1.0) / 0.3
        np.testing.assert_allclose(maxk, expected, rtol=1e-10)

    def test_exponential_maxk_inclined(self):
        """Inclination increases maxk by 1/cosi."""
        model = InclinedExponentialModel()
        maxk_fo = model.maxk({'int_rscale': 0.3, 'cosi': 1.0})
        maxk_inc = model.maxk({'int_rscale': 0.3, 'cosi': 0.5})
        np.testing.assert_allclose(maxk_inc, maxk_fo / 0.5, rtol=1e-10)

    def test_spergel_maxk_analytic(self):
        model = InclinedSpergelModel()
        params = {'int_rscale': 0.3, 'nu': 0.5, 'cosi': 1.0}
        maxk = model.maxk(params, threshold=1e-3)
        expected = np.sqrt(1e-3 ** (-1.0 / 1.5) - 1.0) / 0.3
        np.testing.assert_allclose(maxk, expected, rtol=1e-10)

    def test_sersic_maxk_rootfinding(self):
        model = InclinedSersicModel()
        for n in [1.0, 2.0, 4.0]:
            params = {'int_hlr': 0.5, 'n_sersic': n, 'cosi': 1.0}
            maxk = model.maxk(params, threshold=1e-3)
            assert maxk > 0, f"maxk should be positive for n={n}"
            assert np.isfinite(maxk), f"maxk should be finite for n={n}"

    def test_sersic_maxk_monotonic_in_n(self):
        """Higher Sersic n → slower FT decay → larger maxk."""
        model = InclinedSersicModel()
        maxks = []
        for n in [1.0, 2.0, 3.0, 4.0]:
            params = {'int_hlr': 0.5, 'n_sersic': n, 'cosi': 1.0}
            maxks.append(model.maxk(params))
        for i in range(1, len(maxks)):
            assert maxks[i] > maxks[i - 1]

    def test_maxk_requires_cosi(self):
        """maxk should raise KeyError if cosi missing."""
        model = InclinedExponentialModel()
        with pytest.raises(KeyError, match='cosi'):
            model.maxk({'int_rscale': 0.3})

    def test_stepk_exponential(self):
        model = InclinedExponentialModel()
        params = {'int_rscale': 0.3}
        stepk = model.stepk(params)
        assert stepk > 0
        hlr = 1.6783469900166605 * 0.3
        expected = np.pi / (5 * hlr)
        np.testing.assert_allclose(stepk, expected, rtol=1e-6)

    def test_effective_maxk_with_pixel(self):
        """Pixel sinc attenuation should reduce effective maxk vs bare."""
        model = InclinedExponentialModel()
        bp = BoxPixel(0.11)
        params = {'int_rscale': 0.3, 'cosi': 0.5}
        bare = model.maxk(params)
        eff = compute_effective_maxk(model, params, pixel_response=bp)
        assert eff < bare, "pixel sinc should reduce effective maxk"

    @pytest.fixture
    def _check_galsim(self):
        pytest.importorskip('galsim')

    def test_maxk_vs_galsim(self, _check_galsim, output_dir):
        """Compare our maxk to GalSim's for exponential profiles."""
        import galsim

        model = InclinedExponentialModel()
        rscales = np.linspace(0.1, 1.0, 10)
        our_maxks = []
        gs_maxks = []
        for r in rscales:
            our_maxks.append(model.maxk({'int_rscale': r, 'cosi': 1.0}))
            gs_profile = galsim.Exponential(scale_radius=r)
            gs_maxks.append(gs_profile.maxk)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rscales, our_maxks, 'o-', label='kl_pipe', alpha=0.7)
        ax.plot(rscales, gs_maxks, 's--', label='GalSim', alpha=0.7)
        ax.set_xlabel('Scale radius (arcsec)')
        ax.set_ylabel('maxk (rad/arcsec)')
        ax.set_title('maxk: kl_pipe vs GalSim (exponential)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'maxk_vs_galsim.png', dpi=150)
        plt.close()


# ==============================================================================
# E. Grid adequacy
# ==============================================================================


class TestGridAdequacy:
    """Test adaptive grid computation."""

    def test_grid_requirements_compact_galaxy(self):
        model = InclinedExponentialModel()
        params = {'int_rscale': 0.1, 'cosi': 1.0}
        rc = RenderConfig.for_model(model, params, 0.11)
        assert rc.oversample > 1, "compact galaxy should need oversample > 1"

    def test_grid_requirements_extended_galaxy(self):
        model = InclinedExponentialModel()
        params = {'int_rscale': 1.0, 'cosi': 1.0}
        rc = RenderConfig.for_model(model, params, 0.11)
        assert rc.oversample == 1, "extended galaxy should not need oversample"

    def test_grid_requirements_from_priors(self):
        from kl_pipe.priors import Uniform, PriorDict

        model = InclinedExponentialModel()
        priors = PriorDict(
            {
                'cosi': Uniform(0.1, 0.99),
                'theta_int': Uniform(0, 2 * np.pi),
                'g1': 0.0,
                'g2': 0.0,
                'flux': Uniform(1e3, 1e5),
                'int_rscale': Uniform(0.1, 1.0),
                'int_h_over_r': 0.1,
                'int_x0': 0.0,
                'int_y0': 0.0,
            }
        )
        rc = RenderConfig.for_priors(model, priors, 0.11)
        assert rc.effective_maxk > 0
        assert rc.oversample >= 1

    def test_pixel_response_reduces_effective_maxk(self):
        """Pixel sinc attenuation should reduce effective maxk."""
        model = InclinedExponentialModel()
        bp = BoxPixel(0.11)
        params = {'int_rscale': 0.3, 'cosi': 0.5}
        bare = compute_effective_maxk(model, params)
        with_pixel = compute_effective_maxk(model, params, pixel_response=bp)
        assert with_pixel < bare


# ==============================================================================
# F. k-space power spectrum diagnostic
# ==============================================================================


class TestKSpaceDiagnostic:
    """Visual diagnostic: k-space power spectrum."""

    def test_kspace_power_spectrum(self, exp_model, exp_theta, image_pars, output_dir):
        """Plot |I_hat(k)| along kx axis with sinc overlay."""
        ps = image_pars.pixel_scale
        Nrow = image_pars.Nrow
        bp = BoxPixel(ps)

        # render in k-space (extract FT before IFFT)
        from kl_pipe.intensity import _inclined_sech2_ft
        from scipy.fft import next_fast_len

        pad = next_fast_len(2 * Nrow)
        ky = 2 * np.pi * np.fft.fftfreq(pad, d=ps)
        kx = 2 * np.pi * np.fft.fftfreq(pad, d=ps)
        KY, KX = np.meshgrid(ky, kx, indexing='ij')

        # profile FT (face-on exponential for simplicity)
        rscale = 0.3
        k_sq = (KX * rscale) ** 2 + (KY * rscale) ** 2
        ft_profile = 1e4 / (1 + k_sq) ** 1.5

        # pixel FT
        ft_pixel = np.sinc(KX * ps / (2 * np.pi)) * np.sinc(KY * ps / (2 * np.pi))

        # 1D slice along kx (ky=0)
        kx_1d = kx[: pad // 2]
        ft_prof_1d = np.abs(ft_profile[0, : pad // 2])
        ft_pix_1d = np.abs(ft_pixel[0, : pad // 2])
        ft_combined_1d = ft_prof_1d * ft_pix_1d

        k_nyq = np.pi / ps

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(kx_1d, ft_prof_1d / ft_prof_1d[0], label='Profile FT')
        ax.semilogy(kx_1d, ft_pix_1d, label='Pixel FT (sinc)')
        ax.semilogy(
            kx_1d,
            ft_combined_1d / ft_combined_1d[0],
            label='Profile × Pixel',
            linewidth=2,
        )
        ax.axvline(k_nyq, color='red', linestyle='--', alpha=0.7, label='Nyquist')
        ax.axhline(1e-3, color='gray', linestyle=':', alpha=0.5, label='threshold=1e-3')
        ax.set_xlabel('kx (rad/arcsec)')
        ax.set_ylabel('|FT| (normalized)')
        ax.set_title(f'k-space power spectrum (r_s={rscale}", ps={ps}")')
        ax.legend()
        ax.axvspan(k_nyq, 1.5 * k_nyq, alpha=0.08, color='red', label='_nolegend_')
        ax.set_xlim(0, 1.5 * k_nyq)
        ax.set_ylim(1e-6, 2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'kspace_power_spectrum.png', dpi=150)
        plt.close()


# ==============================================================================
# G. Flux conservation
# ==============================================================================


class TestFluxConservation:
    """Verify pixel-integrated images conserve flux."""

    @pytest.mark.parametrize("rscale", [0.2, 0.5, 1.0])
    def test_flux_conservation_exponential(self, exp_model, image_pars, rscale):
        flux = 1e4
        theta = jnp.array([0.7, 0.5, 0.0, 0.0, flux, rscale, 0.1, 0.0, 0.0])
        obs = build_image_obs(image_pars, oversample=1)

        img = exp_model.render_image(theta, obs=obs)
        measured_flux = float(jnp.sum(img)) * image_pars.pixel_scale**2
        # flux conservation: extended profiles (rscale >= pixel_scale * Nrow/2)
        # lose flux beyond the image boundary via pad_factor wrapping
        rtol = 0.05 if rscale < 0.5 else 0.10
        np.testing.assert_allclose(measured_flux, flux, rtol=rtol)


# ==============================================================================
# H. Edge cases
# ==============================================================================


class TestEdgeCases:
    """Edge cases for pixel response."""

    def test_pixel_scale_one(self):
        """pixel_scale=1 (unitless) should work."""
        bp = BoxPixel(1.0)
        KX = jnp.array([[1.0]])
        KY = jnp.array([[0.0]])
        ft = bp.ft(KX, KY)
        assert float(ft[0, 0]) > 0

    def test_subpixel_galaxy(self, exp_model, image_pars, box_pixel):
        """Galaxy smaller than pixel — sinc should still work."""
        theta = jnp.array([0.9, 0.0, 0.0, 0.0, 1e4, 0.05, 0.1, 0.0, 0.0])
        obs = build_image_obs(image_pars, oversample=1)
        img = exp_model.render_image(theta, obs=obs)
        assert img.shape == (64, 64)
        assert float(jnp.max(img)) > 0

    def test_extended_galaxy(self, exp_model, image_pars):
        """Galaxy much larger than pixel — sinc effect should be negligible."""
        theta = jnp.array([0.9, 0.0, 0.0, 0.0, 1e4, 2.0, 0.1, 0.0, 0.0])
        obs_pix = build_image_obs(image_pars, oversample=1)
        obs_none = build_image_obs(image_pars, oversample=1, pixel_response=None)

        img_pix = exp_model.render_image(theta, obs=obs_pix)
        img_none = exp_model.render_image(theta, obs=obs_none)

        # for extended galaxies, sinc effect is tiny
        max_frac_diff = float(jnp.max(jnp.abs(img_pix - img_none)) / jnp.max(img_pix))
        assert max_frac_diff < 0.01, f"Extended galaxy should have <1% pixel effect"

    def test_sersic_high_n(self):
        """High Sersic n=5 should still compute maxk/stepk."""
        model = InclinedSersicModel()
        params = {'int_hlr': 0.5, 'n_sersic': 5.0, 'cosi': 1.0}
        maxk = model.maxk(params)
        stepk = model.stepk(params)
        assert maxk > 0
        assert stepk > 0


# ==============================================================================
# I. InferenceTask → RenderConfig integration
# ==============================================================================


class TestInferenceTaskRenderConfig:
    """Verify RenderConfig flows through InferenceTask → JIT'd likelihood."""

    @pytest.fixture
    def _check_galsim(self):
        pytest.importorskip('galsim')

    def test_inference_task_computes_render_config(self, _check_galsim):
        """InferenceTask.from_intensity_obs computes render_config from priors."""
        import galsim as gs
        from kl_pipe.priors import Uniform, PriorDict
        from kl_pipe.sampling.task import InferenceTask
        from kl_pipe.synthetic import generate_sersic_intensity_2d

        model = InclinedExponentialModel()
        ip = ImagePars((32, 32), 'ij', pixel_scale=0.11)
        psf = gs.Gaussian(fwhm=0.2)

        true_pars = {
            'cosi': 0.6,
            'theta_int': 0.5,
            'g1': 0.0,
            'g2': 0.0,
            'flux': 1e4,
            'int_rscale': 0.3,
            'int_h_over_r': 0.1,
            'int_x0': 0.0,
            'int_y0': 0.0,
        }
        theta_true = model.pars2theta(true_pars)

        # generate data with pixel response (default on)
        data = generate_sersic_intensity_2d(
            ip,
            **{k: v for k, v in true_pars.items() if k != 'int_h_over_r'},
            n_sersic=1.0,
            int_h_over_r=0.1,
        )
        # add simple Gaussian noise (avoid Poisson issues with negative pixels)
        rng = np.random.default_rng(42)
        noise_std = float(np.max(data)) / 100
        data_noisy = np.array(data) + rng.normal(0, noise_std, data.shape)

        obs = build_image_obs(
            ip,
            psf=psf,
            data=jnp.array(data_noisy),
            variance=noise_std**2,
            int_model=model,
        )

        priors = PriorDict(
            {
                'cosi': Uniform(0.3, 0.99),
                'theta_int': Uniform(0, 2 * np.pi),
                'g1': 0.0,
                'g2': 0.0,
                'flux': Uniform(5e3, 2e4),
                'int_rscale': Uniform(0.1, 0.5),
                'int_h_over_r': 0.1,
                'int_x0': 0.0,
                'int_y0': 0.0,
            }
        )

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            task = InferenceTask.from_intensity_obs(model, priors, obs)

        # render_config should be computed
        assert hasattr(task, '_render_configs')
        assert 'intensity' in task._render_configs
        rc = task._render_configs['intensity']
        assert rc.oversample >= 1
        assert rc.effective_maxk is not None

        # with cosi prior down to 0.3, should need oversample > 1
        assert rc.oversample > 1, (
            f"Expected oversample > 1 for cosi prior [0.3, 0.99], "
            f"got {rc.oversample}"
        )

        # likelihood should evaluate to a finite value at true params
        log_prob = task.likelihood_fn(theta_true)
        assert np.isfinite(float(log_prob)), "Likelihood at true params is not finite"


# ==============================================================================
# J. Wrap correctness (sinc + oversample)
# ==============================================================================


class TestWrapCorrectness:
    """Verify k-space wrap produces correct pixel integration when oversample > 1.

    The wrap operation (evaluate sinc at true k, fold onto base grid) must
    match GalSim's pixel integration for profiles where the auto-computed
    RenderConfig gives oversample > 1 (compact/inclined profiles).

    These tests would have caught the double-pixel-integration bug where
    IFFT+bin was used instead of wrap when sinc was active.
    """

    @pytest.fixture
    def _check_galsim(self):
        pytest.importorskip('galsim')

    def test_inclined_compact_vs_galsim(self, _check_galsim):
        """Inclined compact profile with auto RenderConfig vs GalSim.

        This is the case that triggered 16% residual from double pixel
        integration before the wrap fix.
        """
        import galsim as gs
        from kl_pipe.synthetic import _generate_sersic_galsim

        model = InclinedExponentialModel()
        ip = ImagePars((64, 64), 'ij', pixel_scale=0.11)
        ps = ip.pixel_scale
        gsp = gs.GSParams(
            folding_threshold=1e-4, maxk_threshold=1e-4, kvalue_accuracy=1e-6
        )

        # cosi=0.5, rscale=0.3 → auto oversample=3
        theta = jnp.array([0.5, 0.5, 0.0, 0.0, 1.0, 0.3, 0.1, 0.0, 0.0])
        img = np.array(model.render_image(theta, image_pars=ip))

        gs_img = _generate_sersic_galsim(
            ip,
            flux=1.0,
            int_rscale=0.3,
            n_sersic=1.0,
            cosi=0.5,
            theta_int=0.5,
            g1=0.0,
            g2=0.0,
            int_x0=0.0,
            int_y0=0.0,
            int_h_over_r=0.1,
            gsparams=gsp,
            method='auto',
        )
        gs_sb = gs_img / ps**2
        peak = np.max(np.abs(gs_sb))
        max_resid = np.max(np.abs(img - gs_sb)) / peak

        assert (
            max_resid < 0.01
        ), f"Inclined compact vs GalSim: {max_resid:.1%} (should be <1%)"

    def test_oversample_sinc_flux_conservation(self):
        """Flux must be conserved when wrap is active (oversample > 1 + sinc)."""
        model = InclinedExponentialModel()
        ip = ImagePars((64, 64), 'ij', pixel_scale=0.11)
        theta = jnp.array([0.3, 0.5, 0.0, 0.0, 1e4, 0.3, 0.1, 0.0, 0.0])

        # auto RenderConfig will give oversample > 1 for cosi=0.3
        img = model.render_image(theta, image_pars=ip)
        measured = float(jnp.sum(img)) * ip.pixel_scale**2
        np.testing.assert_allclose(measured, 1e4, rtol=0.01)

    def test_wrap_improves_over_sinc_only(self):
        """Wrap with oversample=3 should be more accurate than sinc-only.

        For inclined profiles, sinc-only (oversample=1) has residual
        aliasing from power above Nyquist. The wrap captures that power.
        """
        model = InclinedExponentialModel()
        ip = ImagePars((64, 64), 'ij', pixel_scale=0.11)
        theta = jnp.array([0.3, 0.5, 0.0, 0.0, 1e4, 0.3, 0.1, 0.0, 0.0])

        rc1 = RenderConfig(oversample=1)
        rc3 = RenderConfig(oversample=3)
        img1 = np.array(model.render_image(theta, image_pars=ip, render_config=rc1))
        img3 = np.array(model.render_image(theta, image_pars=ip, render_config=rc3))

        # both should have similar flux
        flux1 = float(jnp.sum(img1)) * ip.pixel_scale**2
        flux3 = float(jnp.sum(img3)) * ip.pixel_scale**2
        np.testing.assert_allclose(flux1, flux3, rtol=0.01)

        # wrap should change the image (more high-k captured)
        diff = np.max(np.abs(img1 - img3)) / np.max(img3)
        assert diff > 1e-4, "wrap with oversample=3 should differ from oversample=1"


# ==============================================================================
# J. Sub-pixel location accuracy vs GalSim
# ==============================================================================


class TestSubPixelLocation:
    """Verify rendering accuracy doesn't degrade at sub-pixel centroid offsets.

    Compares inclined exponential rendered at various fractional-pixel
    offsets against GalSim reference. Residuals should stay roughly
    constant across offsets — if they spike at certain sub-pixel
    positions, the k-space phase shift handling has a bug.
    """

    @pytest.fixture
    def _check_galsim(self):
        pytest.importorskip('galsim')

    @staticmethod
    @staticmethod
    def _render_subpixel_grid(
        model,
        ip,
        cosi,
        theta_int,
        rscale,
        h_over_r,
        gsp,
        maxk_threshold=5e-4,
    ):
        """Compute 4×4 sub-pixel residual grid vs GalSim.

        Uses tighter maxk_threshold than the production default (1e-3)
        to ensure the diagnostic validates rendering accuracy rather than
        being limited by the aliasing budget.
        """
        from kl_pipe.synthetic import _generate_sersic_galsim
        from kl_pipe.pixel import BoxPixel as _BP

        ps = ip.pixel_scale
        flux = 1.0
        fracs = [0.0, 0.25, 0.5, 0.75]

        residuals = np.zeros((4, 4))
        resid_images = {}
        baseline = {}

        for iy, fy in enumerate(fracs):
            for ix, fx in enumerate(fracs):
                x0, y0 = fx * ps, fy * ps
                theta = jnp.array(
                    [cosi, theta_int, 0.0, 0.0, flux, rscale, h_over_r, x0, y0]
                )
                # use tighter threshold for diagnostic accuracy
                params = model.theta2pars(theta)
                rc = RenderConfig.for_model(
                    model,
                    params,
                    ps,
                    pixel_response=_BP(ps),
                    maxk_threshold=maxk_threshold,
                )
                our_img = np.array(
                    model.render_image(theta, image_pars=ip, render_config=rc)
                )
                gs_img = _generate_sersic_galsim(
                    ip,
                    flux=flux,
                    int_rscale=rscale,
                    n_sersic=1.0,
                    cosi=cosi,
                    theta_int=theta_int,
                    g1=0.0,
                    g2=0.0,
                    int_x0=x0,
                    int_y0=y0,
                    int_h_over_r=h_over_r,
                    gsparams=gsp,
                    method='auto',
                )
                gs_sb = gs_img / ps**2
                peak = np.max(np.abs(gs_sb))
                # signed residual for diverging colormap
                signed_resid = (our_img - gs_sb) / peak
                residuals[iy, ix] = np.max(np.abs(signed_resid))
                resid_images[(iy, ix)] = signed_resid
                if ix == 0 and iy == 0:
                    baseline['ours'] = our_img
                    baseline['gs'] = gs_sb

        return residuals, resid_images, baseline

    @staticmethod
    def _plot_subpixel_panel(residuals, resid_images, baseline, title, path):
        """Plot 4×5 panel: baseline col + 4×4 signed residual grid."""
        fracs = [0.0, 0.25, 0.5, 0.75]
        vmax = np.max(np.abs(np.array(list(resid_images.values()))))

        fig, axes = plt.subplots(4, 5, figsize=(17, 13))

        # col 0: baseline context
        axes[0, 0].imshow(baseline['gs'], origin='lower')
        axes[0, 0].set_title('GalSim (centered)', fontsize=12)
        axes[1, 0].imshow(baseline['ours'], origin='lower')
        axes[1, 0].set_title('kl_pipe (centered)', fontsize=12)
        axes[2, 0].imshow(
            resid_images[(0, 0)],
            origin='lower',
            cmap='RdBu_r',
            vmin=-vmax,
            vmax=vmax,
        )
        axes[2, 0].set_title(
            f'centered resid\nmax|r|/peak={residuals[0, 0]:.3%}',
            fontsize=11,
        )
        axes[3, 0].axis('off')

        for iy in range(4):
            axes[iy, 0].set_ylabel(f'dy={fracs[iy]:.2f}px', fontsize=12)

        # cols 1-4: signed residual grid
        for ix in range(4):
            for iy in range(4):
                ax = axes[iy, ix + 1]
                im = ax.imshow(
                    resid_images[(iy, ix)],
                    origin='lower',
                    cmap='RdBu_r',
                    vmin=-vmax,
                    vmax=vmax,
                )
                ax.set_title(
                    f'({fracs[ix]:.2f}, {fracs[iy]:.2f})px\n'
                    f'max|r|/peak={residuals[iy, ix]:.3%}',
                    fontsize=10,
                )
                ax.set_xticks([])
                ax.set_yticks([])

        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 0.88, 0.95])
        # explicit colorbar axis to avoid layout conflicts
        cax = fig.add_axes([0.90, 0.08, 0.015, 0.82])
        fig.colorbar(im, cax=cax, label='(kl_pipe \u2212 GalSim) / peak').set_label(
            '(kl_pipe \u2212 GalSim) / peak',
            fontsize=12,
        )
        plt.savefig(path, dpi=150)
        plt.close()

    def test_subpixel_offset_stability(self, _check_galsim, output_dir):
        """Panel diagnostic: residuals at 4×4 sub-pixel offsets.

        Two panels: face-on and inclined, to separate pixel integration
        accuracy from inclination-dependent aliasing.
        """
        import galsim as gs

        model = InclinedExponentialModel()
        ip = ImagePars((64, 64), 'ij', pixel_scale=0.11)
        ps = ip.pixel_scale
        gsp = gs.GSParams(
            folding_threshold=1e-4, maxk_threshold=1e-4, kvalue_accuracy=1e-6
        )
        rscale = 0.3

        # face-on panel
        res_fo, imgs_fo, base_fo = self._render_subpixel_grid(
            model,
            ip,
            cosi=1.0,
            theta_int=0.0,
            rscale=rscale,
            h_over_r=0.0,
            gsp=gsp,
        )
        self._plot_subpixel_panel(
            res_fo,
            imgs_fo,
            base_fo,
            f'Sub-pixel residuals vs GalSim — face-on exponential\n'
            f'(cosi=1.0, r_s={rscale}", ps={ps}", r/ps={rscale/ps:.1f})',
            output_dir / 'subpixel_faceon.png',
        )

        # inclined panel
        res_inc, imgs_inc, base_inc = self._render_subpixel_grid(
            model,
            ip,
            cosi=0.6,
            theta_int=0.5,
            rscale=rscale,
            h_over_r=0.1,
            gsp=gsp,
        )
        self._plot_subpixel_panel(
            res_inc,
            imgs_inc,
            base_inc,
            f'Sub-pixel residuals vs GalSim — inclined exponential\n'
            f'(cosi=0.6, r_s={rscale}", ps={ps}", r/ps={rscale/ps:.1f})',
            output_dir / 'subpixel_inclined.png',
        )

        # assertion: no sub-pixel position should spike >3× centered
        for label, residuals in [('face-on', res_fo), ('inclined', res_inc)]:
            centered = residuals[0, 0]
            worst = np.max(residuals)
            assert worst < max(centered * 3, 0.03), (
                f"{label}: worst sub-pixel residual ({worst:.1%}) is >3× "
                f"centered ({centered:.1%})"
            )
