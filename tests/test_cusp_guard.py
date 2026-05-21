"""Regression tests for Spergel cusp construction-time prior validation.

InclinedSpergelModel and InclinedDeVaucouleursModel produce unphysical
morphology along the minor axis when nu < -0.5 is combined with cosi <
0.9 (inclined orientations). PR #43 adds construction-time prior
validation in InferenceTask factories that raises ValueError before any
JIT trace, so misconfigured priors fail loudly upstream.

These tests pin the guard against regression.
"""

import jax

jax.config.update('jax_enable_x64', True)

import galsim
import numpy as np
import pytest

from kl_pipe.intensity import InclinedDeVaucouleursModel, InclinedSpergelModel
from kl_pipe.observation import build_image_obs
from kl_pipe.parameters import ImagePars
from kl_pipe.priors import PriorDict, Uniform
from kl_pipe.render import RenderConfig
from kl_pipe.sampling import InferenceTask


@pytest.fixture
def imaging_setup():
    """Minimal imaging setup for InferenceTask construction."""
    image_pars = ImagePars(shape=(32, 32), pixel_scale=0.1, indexing='ij')
    psf = galsim.Gaussian(fwhm=0.2)
    rng = np.random.default_rng(0)
    data = rng.normal(size=image_pars.shape)
    variance = np.ones_like(data)
    return image_pars, psf, data, variance


# ----------------------------------------------------------------------
# InclinedSpergelModel cusp guard
# ----------------------------------------------------------------------


class TestSpergelCuspGuard:
    """Construction-time validation for InclinedSpergelModel."""

    def _priors(self, nu_low, nu_high, cosi_low, cosi_high):
        return PriorDict(
            {
                'cosi': Uniform(cosi_low, cosi_high),
                'theta_int': Uniform(0, np.pi),
                'flux': Uniform(0.5, 2.0),
                'int_rscale': Uniform(0.3, 1.5),
                'int_h_over_r': 0.1,
                'nu': Uniform(nu_low, nu_high),
                'g1': 0.0,
                'g2': 0.0,
                'int_x0': 0.0,
                'int_y0': 0.0,
            }
        )

    def test_raises_for_unsafe_priors(self, imaging_setup):
        # nu lower bound -0.7 (below -0.5) + cosi lower bound 0.1 (below 0.9)
        # = unsafe cusp regime
        image_pars, psf, data, variance = imaging_setup
        model = InclinedSpergelModel()
        priors = self._priors(nu_low=-0.7, nu_high=0.5, cosi_low=0.1, cosi_high=0.99)
        obs = build_image_obs(
            image_pars=image_pars,
            psf=psf,
            data=data,
            variance=variance,
            int_model=model,
        )
        with pytest.raises(ValueError, match='cusp regime'):
            InferenceTask.from_intensity_obs(model, priors, obs)

    def test_check_priors_safe_no_raise_face_on(self):
        # nu lower bound -0.7 (cusp regime), but cosi bounded > 0.9 (face-on)
        # = safe. With the PSF-on-obs fix, full InferenceTask construction
        # also succeeds: PSF damping caps the worst-case maxk so the cusp
        # profile's slow FT decay no longer blows up the grid.
        model = InclinedSpergelModel()
        priors = self._priors(nu_low=-0.7, nu_high=0.5, cosi_low=0.95, cosi_high=0.99)
        # direct method does not raise
        model.check_priors_safe(priors)

    def test_face_on_inference_task_construction(self, imaging_setup):
        """Spergel cusp profile face-on: full from_intensity_obs succeeds."""
        image_pars, psf, data, variance = imaging_setup
        model = InclinedSpergelModel()
        priors = self._priors(nu_low=-0.7, nu_high=0.5, cosi_low=0.95, cosi_high=0.99)
        obs = build_image_obs(
            image_pars=image_pars,
            psf=psf,
            data=data,
            variance=variance,
            int_model=model,
        )
        # should not raise; PSF damping makes the worst-case grid tractable
        task = InferenceTask.from_intensity_obs(model, priors, obs)
        assert task is not None
        assert (
            obs.oversample <= 7
        ), f"face-on Spergel cusp should be tractable: oversample={obs.oversample}"

    def test_check_priors_safe_no_raise_safe_nu(self):
        # nu lower bound 0.0 (above -0.5), cosi can range freely = safe
        model = InclinedSpergelModel()
        priors = self._priors(nu_low=0.0, nu_high=1.0, cosi_low=0.1, cosi_high=0.99)
        # should not raise
        model.check_priors_safe(priors)


# ----------------------------------------------------------------------
# InclinedDeVaucouleursModel cusp guard (nu hardwired below -0.5)
# ----------------------------------------------------------------------


class TestDeVaucouleursCuspGuard:
    """Construction-time validation for InclinedDeVaucouleursModel."""

    def _priors(self, cosi_low, cosi_high):
        return PriorDict(
            {
                'cosi': Uniform(cosi_low, cosi_high),
                'theta_int': Uniform(0, np.pi),
                'flux': Uniform(0.5, 2.0),
                'int_rscale': Uniform(0.3, 1.5),
                'int_h_over_r': 0.1,
                'g1': 0.0,
                'g2': 0.0,
                'int_x0': 0.0,
                'int_y0': 0.0,
            }
        )

    def test_raises_for_inclined_priors(self, imaging_setup):
        # cosi lower bound 0.1 with hardwired nu=-0.6 → cusp regime
        image_pars, psf, data, variance = imaging_setup
        model = InclinedDeVaucouleursModel()
        priors = self._priors(cosi_low=0.1, cosi_high=0.99)
        obs = build_image_obs(
            image_pars=image_pars,
            psf=psf,
            data=data,
            variance=variance,
            int_model=model,
        )
        with pytest.raises(ValueError, match='cusp regime'):
            InferenceTask.from_intensity_obs(model, priors, obs)

    def test_check_priors_safe_no_raise_face_on(self):
        # cosi bounded > 0.9 -> safe regime; method returns without raising.
        # With the PSF-on-obs fix, the full InferenceTask path also succeeds:
        # PSF damping caps the bare nu=-0.6 profile FT's slow decay so the
        # face-on grid is tractable (~oversample=5 vs the formerly spurious
        # ~17 reported when PSF was dropped from the worst-case scan).
        model = InclinedDeVaucouleursModel()
        priors = self._priors(cosi_low=0.95, cosi_high=0.99)
        # direct method does not raise
        model.check_priors_safe(priors)

    def test_face_on_inference_task_construction(self, imaging_setup):
        """DeVauc face-on: full from_intensity_obs succeeds with PSF damping."""
        image_pars, psf, data, variance = imaging_setup
        model = InclinedDeVaucouleursModel()
        priors = self._priors(cosi_low=0.95, cosi_high=0.99)
        obs = build_image_obs(
            image_pars=image_pars,
            psf=psf,
            data=data,
            variance=variance,
            int_model=model,
        )
        # should not raise; PSF caps the slow nu=-0.6 FT decay
        task = InferenceTask.from_intensity_obs(model, priors, obs)
        assert task is not None
        assert obs.oversample <= 7, (
            f"face-on DeVauc with PSF should be tractable: "
            f"oversample={obs.oversample}"
        )
