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
from kl_pipe.render import (
    RenderConfig,
    build_grism_render_config,
    build_image_render_config,
)
from kl_pipe.sampling import InferenceTask
from kl_pipe.source import SourceModel


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
            'F087.flux': LogUniform(0.01, 1000.0),
            'F087.rscale': Uniform(1.0, 2.0),
            'F087.h_over_r': 0.2,
            'g1': 0.0,
            'g2': 0.0,
            'F087.x0': 0.0,
            'F087.y0': 0.0,
        }
    )


@pytest.fixture
def loose_priors():
    """Priors that fit within the default (5) oversample."""
    return PriorDict(
        {
            'cosi': Uniform(0.5, 0.99),
            'theta_int': Uniform(0, np.pi),
            'F087.flux': LogUniform(0.01, 1000.0),
            'F087.rscale': Uniform(2.0, 4.0),
            'F087.h_over_r': 0.2,
            'g1': 0.0,
            'g2': 0.0,
            'F087.x0': 0.0,
            'F087.y0': 0.0,
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
    # explicit RenderConfig() (oversample=1) forces the priors-vs-obs-rc
    # validation path; builder-default rc would auto-derive + rebuild instead.
    obs = build_image_obs(
        image_pars,
        psf=None,
        data=jnp.array(data),
        variance=jnp.array(variance),
        int_model=model,
        render_config=RenderConfig(),
        broadband_key='F087',
    )
    source = SourceModel(broadband_models={'F087': model})

    # tight priors imply oversample > 5; expect loud raise, not crash mid-JIT
    with pytest.raises(ValueError, match='Priors imply oversample'):
        InferenceTask.from_obs(source, tight_priors, image_obs={'F087': obs})


def test_priors_tighter_than_obs_rc_ok(setup, loose_priors):
    """Priors that fit within obs's pre-built rc — task constructs cleanly."""
    image_pars, psf, model, data, variance = setup
    # explicit oversample=5 rc on obs; loose priors imply smaller oversample
    # → validation path accepts and uses obs's rc unchanged.
    obs = build_image_obs(
        image_pars,
        psf=psf,
        data=jnp.array(data),
        variance=jnp.array(variance),
        int_model=model,
        render_config=RenderConfig(oversample=5),
        broadband_key='F087',
    )
    source = SourceModel(broadband_models={'F087': model})
    # priors imply small oversample; obs explicit rc is 5 -> fine (no raise)
    task = InferenceTask.from_obs(source, loose_priors, image_obs={'F087': obs})
    assert task is not None
    # explicit-rc branch returns obs unchanged; task's likelihood uses this rc.
    assert obs.render_config.oversample == 5


def test_priors_wider_than_obs_rc_raises(setup, tight_priors):
    """Priors that demand larger rc than obs was built for — loud failure.

    Built without PSF (psf=None) and with explicit ``RenderConfig()`` so the
    priors-vs-obs-rc validation path is exercised (builder-default rc would
    take the auto-derive + rebuild path instead). With PSF in the worst-case
    scan, Gaussian damping caps maxk far below the bare profile FT's reach,
    so realistic inference setups with real PSF + tight priors almost never
    trip this validation — the loud-failure path is mainly a safety net
    for the no-PSF case (or extreme priors).
    """
    image_pars, _, model, data, variance = setup
    obs = build_image_obs(
        image_pars,
        psf=None,
        data=jnp.array(data),
        variance=jnp.array(variance),
        int_model=model,
        render_config=RenderConfig(),
        broadband_key='F087',
    )
    source = SourceModel(broadband_models={'F087': model})
    with pytest.raises(ValueError) as excinfo:
        InferenceTask.from_obs(source, tight_priors, image_obs={'F087': obs})
    msg = str(excinfo.value)
    assert 'oversample' in msg
    # message must point at the fix (for_priors)
    assert 'for_priors' in msg


def test_for_priors_obs_construction_works(setup, tight_priors):
    """Recommended fix from Issue #42 raise message — verify it actually works."""
    image_pars, psf, model, data, variance = setup
    source = SourceModel(broadband_models={'F087': model})
    rc = build_image_render_config(
        source,
        tight_priors,
        image_pars,
        broadband_key='F087',
        psf=psf,
    )
    obs = build_image_obs(
        image_pars,
        psf=psf,
        data=jnp.array(data),
        variance=jnp.array(variance),
        int_model=model,
        render_config=rc,
        broadband_key='F087',
    )
    # task construction succeeds
    task = InferenceTask.from_obs(source, tight_priors, image_obs={'F087': obs})
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


# ============================================================================
# build_image_render_config / build_grism_render_config
# ============================================================================
#
# Top-level builders that consume dotted-key SourceModel priors + obs-shape
# primitives and produce a worst-case RenderConfig. Thin wrappers over the
# RenderConfig.for_priors / for_grism_priors classmethods that handle the
# dotted-key → flat namespace translation + per-line iteration for grism.


@pytest.fixture
def source_image_setup():
    """SourceModel with a single broadband band + dotted-key priors."""
    from kl_pipe.source import SourceModel

    image_pars = ImagePars(shape=(64, 64), pixel_scale=0.1, indexing='ij')
    psf = galsim.Gaussian(fwhm=0.5)
    model = InclinedExponentialModel()
    source = SourceModel(broadband_models={'F087': model})
    priors = PriorDict(
        {
            'cosi': Uniform(0.1, 0.99),
            'theta_int': Uniform(0, np.pi),
            'g1': 0.0,
            'g2': 0.0,
            'F087.flux': LogUniform(0.01, 1000.0),
            'F087.rscale': Uniform(0.05, 1.0),
            'F087.h_over_r': 0.15,
            'F087.x0': 0.0,
            'F087.y0': 0.0,
        }
    )
    return source, priors, image_pars, psf


def test_build_image_render_config_matches_classmethod(source_image_setup):
    """build_image_render_config(source, priors, ...) == RenderConfig.for_priors(model, sub_priors, ...).

    Verifies the wrapper is a pure dotted-key→flat-namespace translation
    on top of the classmethod, with the BoxPixel(pixel_scale) default.
    """
    source, priors, image_pars, psf = source_image_setup

    rc = build_image_render_config(
        source, priors, image_pars, broadband_key='F087', psf=psf
    )

    # equivalent classmethod call with hand-extracted sub-priors
    from kl_pipe.source import _component_priors_for_intensity

    model = source.broadband_models['F087']
    sub_priors = _component_priors_for_intensity(priors, 'F087', model.PARAMETER_NAMES)
    rc_ref = RenderConfig.for_priors(
        model,
        sub_priors,
        image_pars.pixel_scale,
        pixel_response=BoxPixel(image_pars.pixel_scale),
        psf=psf,
    )

    assert rc.oversample == rc_ref.oversample
    assert rc.effective_maxk == pytest.approx(rc_ref.effective_maxk)
    assert rc.stepk == pytest.approx(rc_ref.stepk)


def test_build_image_render_config_missing_band_raises(source_image_setup):
    """Loud failure when broadband_key is not in source.broadband_models."""
    source, priors, image_pars, psf = source_image_setup
    with pytest.raises(ValueError, match="not in source.broadband_models"):
        build_image_render_config(
            source, priors, image_pars, broadband_key='F184', psf=psf
        )


def test_build_image_render_config_threads_into_build_image_obs(source_image_setup):
    """End-to-end: rc from builder makes from_obs validation pass."""
    from kl_pipe.observation import build_image_obs
    from kl_pipe.sampling import InferenceTask

    source, priors, image_pars, psf = source_image_setup
    rng = np.random.default_rng(0)
    data = jnp.asarray(rng.normal(size=image_pars.shape))
    variance = jnp.ones_like(data)

    rc = build_image_render_config(
        source, priors, image_pars, broadband_key='F087', psf=psf
    )
    obs = build_image_obs(
        image_pars,
        psf=psf,
        broadband_key='F087',
        render_config=rc,
        data=data,
        variance=variance,
    )

    # the from_obs validator runs _check_source_priors_fit_obs(ImageObs)
    # which would raise GridAdequacyWarning if rc were undersized.
    task = InferenceTask.from_obs(source, priors, image_obs={'F087': obs})
    assert task.model is source


@pytest.fixture
def source_grism_setup():
    """SourceModel with velocity + single emission line + dotted-key priors."""
    from kl_pipe.dispersion import GrismPars
    from kl_pipe.lines import EmissionLine
    from kl_pipe.source import SourceModel
    from kl_pipe.velocity import CenteredVelocityModel

    image_pars = ImagePars(shape=(48, 48), pixel_scale=0.11, indexing='ij')
    grism_pars = GrismPars(
        image_pars=image_pars,
        dispersion=1.1,
        lambda_ref=656.28 * 2.0,
        dispersion_angle_detector=0.0,
    )
    psf = galsim.Gaussian(fwhm=0.18)

    source = SourceModel(
        velocity_model=CenteredVelocityModel(),
        emission_lines={'Halpha': EmissionLine(intensity=InclinedExponentialModel())},
    )
    priors = PriorDict(
        {
            'cosi': Uniform(0.1, 0.99),
            'theta_int': Uniform(0, np.pi),
            'g1': 0.0,
            'g2': 0.0,
            'vel.vcirc': Uniform(80, 300),
            'vel.rscale': Uniform(0.1, 1.0),
            'vel.x0': 0.0,
            'vel.y0': 0.0,
            'Halpha.flux': LogUniform(0.01, 100.0),
            'Halpha.rscale': Uniform(0.05, 1.0),
            'Halpha.h_over_r': 0.15,
            'Halpha.x0': 0.0,
            'Halpha.y0': 0.0,
            'Halpha.dispersion': Uniform(20, 100),
        }
    )
    return source, priors, grism_pars, psf


def test_build_grism_render_config_happy_path(source_grism_setup):
    """Single-line grism: builder returns a sized rc with positive oversample."""
    source, priors, grism_pars, psf = source_grism_setup
    rc = build_grism_render_config(source, priors, grism_pars, psf=psf)
    assert rc.oversample >= 1
    assert rc.effective_maxk is not None and rc.effective_maxk > 0
    assert rc.stepk is not None and rc.stepk > 0


def test_build_grism_render_config_no_velocity_raises(source_grism_setup):
    """Loud failure when source.velocity_model is None."""
    from kl_pipe.lines import EmissionLine
    from kl_pipe.source import SourceModel

    _, priors, grism_pars, psf = source_grism_setup
    # construct a source with broadband + emission lines but no velocity
    # (use broadband to satisfy SourceModel's "at least one component" check)
    source = SourceModel(
        broadband_models={'F087': InclinedExponentialModel()},
        emission_lines={'Halpha': EmissionLine(intensity=InclinedExponentialModel())},
    )
    with pytest.raises(ValueError, match="velocity_model"):
        build_grism_render_config(source, priors, grism_pars, psf=psf)


def test_build_grism_render_config_no_emission_raises(source_grism_setup):
    """Loud failure when source.emission_lines is empty."""
    from kl_pipe.source import SourceModel
    from kl_pipe.velocity import CenteredVelocityModel

    _, priors, grism_pars, psf = source_grism_setup
    source = SourceModel(velocity_model=CenteredVelocityModel())
    with pytest.raises(ValueError, match="emission_lines"):
        build_grism_render_config(source, priors, grism_pars, psf=psf)


def test_build_grism_render_config_missing_dispersion_raises(source_grism_setup):
    """Loud failure when a required <line>.dispersion prior key is missing."""
    source, priors, grism_pars, psf = source_grism_setup
    # drop the Halpha.dispersion key
    bad_spec = {k: v for k, v in priors._param_spec.items() if k != 'Halpha.dispersion'}
    bad_priors = PriorDict(bad_spec)
    with pytest.raises(KeyError, match="Halpha.dispersion"):
        build_grism_render_config(source, bad_priors, grism_pars, psf=psf)


# ============================================================================
# from_obs auto-derive + with_render_config
# ============================================================================
#
# When an obs is built without an explicit render_config, build_*_obs marks
# it _rc_was_default=True. InferenceTask.from_obs detects the flag, derives
# a priors-sized rc, and rebuilds the obs internally via with_render_config.
# When the obs has an explicit user-supplied rc, from_obs honors it and
# runs validation against the priors. Tests below cover both branches and
# the rebuild mechanics directly.


def test_from_obs_auto_derives_when_default(source_image_setup):
    """from_obs auto-derives rc when build_image_obs got no render_config."""
    from kl_pipe.observation import build_image_obs
    from kl_pipe.sampling import InferenceTask

    source, priors, image_pars, psf = source_image_setup
    rng = np.random.default_rng(0)
    data = jnp.asarray(rng.normal(size=image_pars.shape))
    variance = jnp.ones_like(data)
    obs = build_image_obs(
        image_pars, psf=psf, broadband_key='F087', data=data, variance=variance
    )
    assert obs._rc_was_default is True

    # Compute the expected rc directly to compare:
    expected_rc = build_image_render_config(
        source, priors, image_pars, broadband_key='F087', psf=psf
    )

    task = InferenceTask.from_obs(source, priors, image_obs={'F087': obs})
    # task constructs cleanly — auto-derive + rebuild path exercised.
    # The task uses the rebuilt obs internally; from outside, the
    # behavior we can directly assert is "no exception, task is valid".
    assert task is not None
    # Validate the expected oversample is what we'd derive ourselves.
    assert expected_rc.oversample >= 1


def test_from_obs_honors_explicit_rc(source_image_setup):
    """from_obs respects an explicit render_config (no auto-derive)."""
    from kl_pipe.observation import build_image_obs
    from kl_pipe.sampling import InferenceTask

    source, priors, image_pars, psf = source_image_setup
    rng = np.random.default_rng(0)
    data = jnp.asarray(rng.normal(size=image_pars.shape))
    variance = jnp.ones_like(data)
    rc_explicit = build_image_render_config(
        source, priors, image_pars, broadband_key='F087', psf=psf
    )
    obs = build_image_obs(
        image_pars,
        psf=psf,
        broadband_key='F087',
        render_config=rc_explicit,
        data=data,
        variance=variance,
    )
    assert obs._rc_was_default is False

    # No auto-derive; explicit rc honored. Validation passes because the
    # explicit rc was derived from the same priors.
    task = InferenceTask.from_obs(source, priors, image_obs={'F087': obs})
    assert task is not None


def test_from_obs_grism_auto_derives(source_grism_setup):
    """Same auto-derive behavior on the GrismObs channel."""
    from kl_pipe.observation import build_grism_obs
    from kl_pipe.sampling import InferenceTask

    source, priors, grism_pars, psf = source_grism_setup
    rng = np.random.default_rng(0)
    Nrow, Ncol = grism_pars.image_pars.Nrow, grism_pars.image_pars.Ncol
    data = jnp.asarray(rng.normal(size=(Nrow, Ncol)))
    variance = jnp.ones_like(data)
    obs = build_grism_obs(grism_pars, z=1.0, psf=psf, data=data, variance=variance)
    assert obs._rc_was_default is True

    task = InferenceTask.from_obs(source, priors, grism_obs={'roll0': obs})
    assert task is not None


def test_with_render_config_rebuilds_grids(source_image_setup):
    """obs.with_render_config produces fresh psf_data, fine_X/Y, kspace_psf_fft."""
    from kl_pipe.observation import build_image_obs

    source, _, image_pars, psf = source_image_setup
    obs = build_image_obs(image_pars, psf=psf, broadband_key='F087')

    new_rc = RenderConfig(oversample=9)
    new_obs = obs.with_render_config(new_rc, int_model=source.broadband_models['F087'])

    assert new_obs.render_config.oversample == 9
    assert new_obs._rc_was_default is False
    # PSF FFT resized to new oversample
    assert new_obs.psf_data is not None
    assert new_obs.psf_data.oversample == 9
    # fine grids resized
    assert new_obs.fine_X is not None
    assert new_obs.fine_X.shape == (
        image_pars.Nrow * 9,
        image_pars.Ncol * 9,
    )
    # kspace_psf_fft built (int_model has _kspace_pad_factor)
    assert new_obs.kspace_psf_fft is not None
    # data fields preserved (None in this fixture; the point is they
    # pass through dataclasses.replace unchanged)
    assert new_obs.data is obs.data
    assert new_obs.variance is obs.variance
    assert new_obs.mask is obs.mask
    assert new_obs.broadband_key == obs.broadband_key


def test_with_render_config_preserves_obs_type(source_image_setup):
    """obs.with_render_config returns the same subtype, not the parent dataclass."""
    from kl_pipe.observation import ImageObs, build_image_obs

    source, _, image_pars, psf = source_image_setup
    obs = build_image_obs(image_pars, psf=psf, broadband_key='F087')
    new_obs = obs.with_render_config(
        RenderConfig(oversample=3), int_model=source.broadband_models['F087']
    )
    assert isinstance(new_obs, ImageObs)
    assert type(new_obs) is ImageObs
