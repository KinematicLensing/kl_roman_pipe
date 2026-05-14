"""Regression tests for RenderConfig + obs single source of truth (Bug B).

Pins the structural fix from PR #41 (Commit 6, ee2f171): obs.render_config
is the single source of truth for grid sizing, and InferenceTask refuses to
silently drift from it.

Each test pins a specific failure mode that would have triggered the
Issue #42 crash under the old architecture.
"""

import jax

jax.config.update('jax_enable_x64', True)

import galsim
import jax.numpy as jnp
import numpy as np
import pytest

from kl_pipe.intensity import (
    InclinedDeVaucouleursModel,
    InclinedExponentialModel,
    InclinedSpergelModel,
)
from kl_pipe.observation import build_image_obs
from kl_pipe.parameters import ImagePars
from kl_pipe.pixel import BoxPixel
from kl_pipe.priors import LogUniform, PriorDict, Uniform
from kl_pipe.render import RenderConfig
from kl_pipe.sampling import InferenceTask


@pytest.fixture
def setup():
    """Build the Issue #42 reproducer state."""
    Nbins = 64
    pixel_scale = 0.1
    image_pars = ImagePars(shape=(Nbins, Nbins), pixel_scale=pixel_scale, indexing='ij')
    psf = galsim.Gaussian(fwhm=pixel_scale * 5)
    model = InclinedExponentialModel()
    rng = np.random.default_rng(42)
    data = rng.normal(size=image_pars.shape)
    variance = np.ones_like(data)
    return image_pars, psf, model, data, variance


@pytest.fixture
def tight_priors():
    """Priors that imply oversample > the default (5)."""
    return PriorDict(
        {
            'cosi': Uniform(0.01, 0.99),
            'theta_int': Uniform(0, np.pi),
            'flux': LogUniform(0.01, 1000.0),
            'int_rscale': Uniform(1.0, 2.0),
            'int_h_over_r': 0.2,
            'g1': 0.0,
            'g2': 0.0,
            'int_x0': 0.0,
            'int_y0': 0.0,
        }
    )


@pytest.fixture
def loose_priors():
    """Priors that fit within the default (5) oversample."""
    return PriorDict(
        {
            'cosi': Uniform(0.5, 0.99),
            'theta_int': Uniform(0, np.pi),
            'flux': LogUniform(0.01, 1000.0),
            'int_rscale': Uniform(2.0, 4.0),
            'int_h_over_r': 0.2,
            'g1': 0.0,
            'g2': 0.0,
            'int_x0': 0.0,
            'int_y0': 0.0,
        }
    )


def test_obs_rc_is_single_source_of_truth(setup):
    """obs.oversample property is read from obs.render_config — no drift possible."""
    image_pars, psf, model, data, variance = setup

    # default rc
    obs = build_image_obs(
        image_pars,
        psf=psf,
        data=jnp.array(data),
        variance=jnp.array(variance),
        int_model=model,
    )
    assert obs.oversample == obs.render_config.oversample
    assert obs.render_config is not None

    # explicit rc
    rc = RenderConfig(oversample=11)
    obs2 = build_image_obs(
        image_pars,
        psf=psf,
        data=jnp.array(data),
        variance=jnp.array(variance),
        int_model=model,
        render_config=rc,
    )
    assert obs2.oversample == 11
    assert obs2.render_config.oversample == 11


def test_obs_kspace_psf_fft_shape_matches_rc(setup):
    """PSF FFT shape is consistent with rc.oversample (wrap divisibility)."""
    image_pars, psf, model, data, variance = setup
    rc = RenderConfig(oversample=7)
    obs = build_image_obs(
        image_pars,
        psf=psf,
        data=jnp.array(data),
        variance=jnp.array(variance),
        int_model=model,
        render_config=rc,
    )
    # wrap path inside _kspace_render_core requires this divisibility
    pad_row, pad_col = obs.kspace_psf_fft.shape
    assert pad_row % rc.oversample == 0
    assert pad_col % rc.oversample == 0


def test_issue_42_does_not_crash(setup, tight_priors):
    """Issue #42 reproducer — must not crash mid-JIT.

    Old code: build obs with default oversample=5; InferenceTask recomputes
    oversample=13 from priors; PSF FFT (640, 640) cannot reshape to
    (64, 13, 64, 13) (832, 832); TypeError mid-JIT.

    New code: InferenceTask reads rc from obs and validates against priors;
    raises ValueError with rebuild instructions if priors imply more
    demanding rc than obs was built for.

    Built without PSF (psf=None) so PSF damping doesn't tame tight_priors'
    bare-profile maxk requirement. The original Issue #42 mid-JIT crash
    was a shape-mismatch bug independent of PSF presence; the validation
    path is exercised here in the bare-profile + pixel-sinc regime.
    """
    image_pars, _, model, data, variance = setup
    obs = build_image_obs(
        image_pars,
        psf=None,
        data=jnp.array(data),
        variance=jnp.array(variance),
        int_model=model,
    )

    # tight priors imply oversample > 5; expect loud raise, not crash mid-JIT
    with pytest.raises(ValueError, match='Priors imply oversample'):
        InferenceTask.from_intensity_obs(model, tight_priors, obs)


def test_priors_tighter_than_obs_rc_ok(setup, loose_priors):
    """Priors that fit within obs's pre-built rc — task constructs cleanly."""
    image_pars, psf, model, data, variance = setup
    obs = build_image_obs(
        image_pars,
        psf=psf,
        data=jnp.array(data),
        variance=jnp.array(variance),
        int_model=model,
    )
    # priors imply small oversample; obs was built with default 5 -> fine
    task = InferenceTask.from_intensity_obs(model, loose_priors, obs)
    assert task is not None
    # task uses the obs's rc (not a recomputed one)
    assert task._render_configs['intensity'].oversample == obs.render_config.oversample


def test_priors_wider_than_obs_rc_raises(setup, tight_priors):
    """Priors that demand larger rc than obs was built for — loud failure.

    Built without PSF (psf=None). With PSF in the worst-case scan, the
    Gaussian damping caps maxk far below the bare profile FT's reach, so
    realistic inference setups with real PSF + tight priors almost never
    trip this validation — the loud-failure path is mainly a safety net
    for the no-PSF case (or extreme priors). That's the intended behavior
    after the PSF-on-obs fix.
    """
    image_pars, _, model, data, variance = setup
    obs = build_image_obs(
        image_pars,
        psf=None,
        data=jnp.array(data),
        variance=jnp.array(variance),
        int_model=model,
    )
    with pytest.raises(ValueError) as excinfo:
        InferenceTask.from_intensity_obs(model, tight_priors, obs)
    msg = str(excinfo.value)
    assert 'oversample' in msg
    # message must point at the fix (for_priors)
    assert 'for_priors' in msg


def test_for_priors_obs_construction_works(setup, tight_priors):
    """Recommended fix from Issue #42 raise message — verify it actually works."""
    image_pars, psf, model, data, variance = setup
    rc = RenderConfig.for_priors(
        model,
        tight_priors,
        image_pars.pixel_scale,
        pixel_response=BoxPixel(image_pars.pixel_scale),
    )
    obs = build_image_obs(
        image_pars,
        psf=psf,
        data=jnp.array(data),
        variance=jnp.array(variance),
        int_model=model,
        render_config=rc,
    )
    # task construction succeeds
    task = InferenceTask.from_intensity_obs(model, tight_priors, obs)
    # likelihood evaluates without crashing
    theta_test = jnp.array([0.3, 0.0, 0.0, 0.0, 12.0, 1.5, 0.2, 0.0, 0.0])
    log_prob = task.likelihood_fn(theta_test)
    assert jnp.isfinite(log_prob)


# =============================================================================
# PSF damping of effective maxk (regression for commit eb9b8b5)
# =============================================================================
#
# eb9b8b5 fixed every production `RenderConfig.for_priors` call site to pass
# `psf=`. The load-bearing claim: for slow-decay profiles (Spergel cusp,
# DeVauc nu=-0.6) at high inclination, the bare-profile FT × pixel-sinc
# product overestimates the required grid by ~3-10x; folding the PSF into
# the scan caps the worst-case maxk before it blows up.
#
# These tests pin the claim numerically against silent regressions in the
# product-scan logic in `render.py:compute_effective_maxk` (e.g., someone
# accidentally drops the `if psf is not None` branch, or `_extract_worst_case_params`
# stops handing back the right worst-case nu/cosi).


class TestPSFEffectiveMaxk:
    """Pins eb9b8b5: PSF threading caps effective_maxk for cusp profiles."""

    @pytest.fixture
    def pixel_scale(self):
        return 0.1

    @pytest.fixture
    def devauc_priors(self):
        """DeVauc face-on prior set (cosi safely above the cusp guard)."""
        return PriorDict(
            {
                'cosi': Uniform(0.95, 0.99),
                'theta_int': Uniform(0, np.pi),
                'flux': LogUniform(0.01, 1000.0),
                'int_rscale': Uniform(1.0, 2.0),
                'int_h_over_r': 0.2,
                'g1': 0.0,
                'g2': 0.0,
                'int_x0': 0.0,
                'int_y0': 0.0,
            }
        )

    @pytest.fixture
    def spergel_priors(self):
        """Spergel prior set with nu just above the cusp guard (-0.5)."""
        return PriorDict(
            {
                'cosi': Uniform(0.95, 0.99),
                'theta_int': Uniform(0, np.pi),
                'flux': LogUniform(0.01, 1000.0),
                'int_rscale': Uniform(1.0, 2.0),
                'int_h_over_r': 0.2,
                'nu': Uniform(-0.4, 0.5),
                'g1': 0.0,
                'g2': 0.0,
                'int_x0': 0.0,
                'int_y0': 0.0,
            }
        )

    def test_devauc_psf_caps_oversample(self, pixel_scale, devauc_priors):
        """DeVauc (nu=-0.6, k^-0.4 decay): PSF fwhm=0.2 caps oversample vs bare."""
        model = InclinedDeVaucouleursModel()
        psf = galsim.Gaussian(fwhm=0.2)
        pixel_response = BoxPixel(pixel_scale)

        rc_bare = RenderConfig.for_priors(
            model,
            devauc_priors,
            pixel_scale,
            pixel_response=pixel_response,
            psf=None,
        )
        rc_psf = RenderConfig.for_priors(
            model,
            devauc_priors,
            pixel_scale,
            pixel_response=pixel_response,
            psf=psf,
        )

        # PSF damping must strictly reduce both effective_maxk and oversample
        assert rc_psf.effective_maxk < rc_bare.effective_maxk, (
            f"PSF must cap effective_maxk: bare={rc_bare.effective_maxk:.1f}, "
            f"psf={rc_psf.effective_maxk:.1f}"
        )
        assert rc_psf.oversample <= rc_bare.oversample, (
            f"PSF must not inflate oversample: bare={rc_bare.oversample}, "
            f"psf={rc_psf.oversample}"
        )
        # post-fix worst-case for DeVauc face-on with fwhm=0.2: oversample <= 7
        # (the commit message reports the original eb9b8b5 fix dropped this
        # from 17 to 5; we leave a small safety margin against tolerance drift)
        assert rc_psf.oversample <= 7, (
            f"PSF-damped oversample should be small for face-on DeVauc; got "
            f"{rc_psf.oversample}"
        )

    def test_spergel_psf_caps_oversample(self, pixel_scale, spergel_priors):
        """Spergel with nu in [-0.4, 0.5]: PSF damping must reduce oversample."""
        model = InclinedSpergelModel()
        psf = galsim.Gaussian(fwhm=0.2)
        pixel_response = BoxPixel(pixel_scale)

        rc_bare = RenderConfig.for_priors(
            model,
            spergel_priors,
            pixel_scale,
            pixel_response=pixel_response,
            psf=None,
        )
        rc_psf = RenderConfig.for_priors(
            model,
            spergel_priors,
            pixel_scale,
            pixel_response=pixel_response,
            psf=psf,
        )

        assert rc_psf.effective_maxk < rc_bare.effective_maxk
        assert rc_psf.oversample <= rc_bare.oversample

    def test_psf_only_path_consistent(self, pixel_scale, devauc_priors):
        """Without pixel_response, PSF alone must still tighten the maxk scan.

        Guards against a regression where the PSF branch is gated on
        ``pixel_response is not None``.
        """
        model = InclinedDeVaucouleursModel()
        psf = galsim.Gaussian(fwhm=0.2)

        rc_bare = RenderConfig.for_priors(
            model,
            devauc_priors,
            pixel_scale,
            pixel_response=None,
            psf=None,
        )
        rc_psf_only = RenderConfig.for_priors(
            model,
            devauc_priors,
            pixel_scale,
            pixel_response=None,
            psf=psf,
        )
        assert rc_psf_only.effective_maxk < rc_bare.effective_maxk
        assert rc_psf_only.oversample <= rc_bare.oversample

    def test_oversample_monotone_in_psf_width(self, pixel_scale, devauc_priors):
        """Wider PSF → more high-k damping → smaller worst-case oversample.

        Monotonic decrease pins the product-scan arithmetic: a regression that
        breaks the PSF factor (e.g., dropping ``abs(psf.kValue(...))``) would
        often flip or flatten this trend.
        """
        model = InclinedDeVaucouleursModel()
        pixel_response = BoxPixel(pixel_scale)
        fwhms = [0.10, 0.20, 0.40]

        oversamples = []
        for fwhm in fwhms:
            rc = RenderConfig.for_priors(
                model,
                devauc_priors,
                pixel_scale,
                pixel_response=pixel_response,
                psf=galsim.Gaussian(fwhm=fwhm),
            )
            oversamples.append(rc.oversample)

        # non-increasing in PSF width (allow equality for ceil/odd-rounding)
        assert (
            oversamples[0] >= oversamples[1] >= oversamples[2]
        ), f"oversample(fwhm={fwhms}) = {oversamples}; expected non-increasing"
        # strict somewhere across the 4x range
        assert oversamples[-1] < oversamples[0], (
            f"oversample(fwhm={fwhms[0]}) == oversample(fwhm={fwhms[-1]}); "
            f"PSF factor probably stuck at unity"
        )
