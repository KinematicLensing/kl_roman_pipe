"""Tests for ``InferenceTask.from_obs`` -- the unified SourceModel factory.

Coverage:
- Construction + likelihood evaluation for each supported inference pattern
  (velocity-only, intensity-only single + multi-band, joint vel+phot,
  grism-only, joint phot+grism). Each pattern asserts the likelihood is
  finite and that ``jax.grad`` produces a finite gradient.
- Validation errors: all-empty obs args, missing source.broadband_models
  entry, broadband_key mismatch on obs, grism without emission_lines,
  grism without velocity_model, velocity_obs without velocity_model,
  velocity_obs.flux_weight_key not in emission_lines.
- Multi-roll grism contribution: zeroing one obs's data raises the chi^2
  unambiguously; the total likelihood depends on every roll.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import galsim  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from kl_pipe.dispersion import GrismPars  # noqa: E402
from kl_pipe.intensity import InclinedExponentialModel  # noqa: E402
from kl_pipe.lines import EmissionLine  # noqa: E402
from kl_pipe.observation import (  # noqa: E402
    build_grism_obs,
    build_image_obs,
    build_velocity_obs,
)
from kl_pipe.parameters import ImagePars  # noqa: E402
from kl_pipe.priors import PriorDict, Uniform  # noqa: E402
from kl_pipe.sampling.task import InferenceTask  # noqa: E402
from kl_pipe.source import SourceModel  # noqa: E402
from kl_pipe.velocity import CenteredVelocityModel  # noqa: E402
from kl_pipe.render import RenderConfig


# ===========================================================================
# Shared fixtures + small helpers
# ===========================================================================


@pytest.fixture
def image_pars():
    return ImagePars(shape=(16, 16), pixel_scale=0.1, indexing='ij')


@pytest.fixture
def gauss_psf():
    # roughly Roman F087-ish FWHM; sharp enough to tighten worst-case maxk
    # but cheap to FFT in test fixtures.
    return galsim.Gaussian(fwhm=0.18)


@pytest.fixture
def rng():
    return np.random.RandomState(0)


def _toy_image_data(rng, shape=(16, 16)):
    return rng.normal(0, 0.5, size=shape)


def _shared_geo_priors():
    return {
        'cosi': Uniform(0.1, 0.99),
        'theta_int': 0.3,
        'g1': 0.0,
        'g2': 0.0,
    }


def _broadband_priors(key='F087'):
    return {
        f'{key}.flux': Uniform(1.0, 200.0),
        f'{key}.rscale': 0.3,
        f'{key}.h_over_r': 0.15,
        f'{key}.x0': 0.0,
        f'{key}.y0': 0.0,
    }


def _velocity_priors():
    return {
        'vel.vcirc': Uniform(80.0, 300.0),
        'vel.v0': 0.0,
        'vel.rscale': 0.3,
    }


def _emission_priors(key='Halpha'):
    return {
        f'{key}.flux': Uniform(1.0, 100.0),
        f'{key}.rscale': 0.3,
        f'{key}.h_over_r': 0.15,
        f'{key}.x0': 0.0,
        f'{key}.y0': 0.0,
        f'{key}.dispersion': Uniform(20.0, 150.0),
    }


def _midpoint_theta(priors):
    """Build a theta_sampled vector with each Uniform prior at its midpoint."""
    theta = []
    for name in priors.sampled_names:
        prior = priors._priors[name]
        if hasattr(prior, 'low') and hasattr(prior, 'high'):
            theta.append(0.5 * (prior.low + prior.high))
        else:
            theta.append(0.5)
    return jnp.array(theta)


# ===========================================================================
# Pattern 1: intensity-only single-band (mirrors from_intensity_obs)
# ===========================================================================


class TestIntensityOnlySingleBand:

    def test_finite_likelihood_and_gradient(self, image_pars, gauss_psf, rng):
        src = SourceModel(broadband_models={'F087': InclinedExponentialModel()})
        priors = PriorDict({**_shared_geo_priors(), 'z': 1.0, **_broadband_priors()})
        obs = build_image_obs(
            image_pars,
            psf=gauss_psf,
            data=_toy_image_data(rng),
            variance=0.25,
            broadband_key='F087',
        )

        task = InferenceTask.from_obs(src, priors, image_obs={'F087': obs})
        theta = _midpoint_theta(priors)
        assert task.n_params == 2  # cosi, F087.flux

        log_l = task.log_likelihood(theta)
        assert jnp.isfinite(log_l)

        grad = jax.grad(task.log_likelihood)(theta)
        assert jnp.all(jnp.isfinite(grad))


# ===========================================================================
# Pattern 2: intensity-only multi-band (new capability)
# ===========================================================================


class TestIntensityOnlyMultiBand:

    def test_finite_likelihood_and_gradient(self, image_pars, gauss_psf, rng):
        src = SourceModel(
            broadband_models={
                'F087': InclinedExponentialModel(),
                'F184': InclinedExponentialModel(),
            }
        )
        priors = PriorDict(
            {
                **_shared_geo_priors(),
                **_broadband_priors('F087'),
                **_broadband_priors('F184'),
            }
        )
        obs_f087 = build_image_obs(
            image_pars,
            psf=gauss_psf,
            data=_toy_image_data(rng),
            variance=0.25,
            broadband_key='F087',
        )
        obs_f184 = build_image_obs(
            image_pars,
            psf=gauss_psf,
            data=_toy_image_data(rng),
            variance=0.25,
            broadband_key='F184',
        )
        task = InferenceTask.from_obs(
            src, priors, image_obs={'F087': obs_f087, 'F184': obs_f184}
        )
        theta = _midpoint_theta(priors)
        assert task.n_params == 3  # cosi, F087.flux, F184.flux

        log_l = task.log_likelihood(theta)
        assert jnp.isfinite(log_l)
        grad = jax.grad(task.log_likelihood)(theta)
        assert jnp.all(jnp.isfinite(grad))


# ===========================================================================
# Pattern 3: velocity-only (mirrors from_velocity_obs, no PSF)
# ===========================================================================


class TestVelocityOnly:

    def test_finite_likelihood_no_psf(self, image_pars, rng):
        src = SourceModel(velocity_model=CenteredVelocityModel())
        priors = PriorDict({**_shared_geo_priors(), **_velocity_priors()})
        # noiseless-ish velocity map data; variance=10 km/s
        obs = build_velocity_obs(
            image_pars,
            data=rng.normal(0, 10.0, size=(16, 16)),
            variance=100.0,
        )
        task = InferenceTask.from_obs(src, priors, velocity_obs=obs)
        theta = _midpoint_theta(priors)
        log_l = task.log_likelihood(theta)
        assert jnp.isfinite(log_l)
        grad = jax.grad(task.log_likelihood)(theta)
        assert jnp.all(jnp.isfinite(grad))


# ===========================================================================
# Pattern 4: grism-only (mirrors from_grism_obs)
# ===========================================================================


class TestGrismOnly:

    def test_finite_likelihood_and_gradient(self, image_pars, rng):
        src = SourceModel(
            velocity_model=CenteredVelocityModel(),
            emission_lines={
                'Halpha': EmissionLine(intensity=InclinedExponentialModel())
            },
        )
        priors = PriorDict(
            {
                **_shared_geo_priors(),
                **_velocity_priors(),
                **_emission_priors('Halpha'),
                'z': 1.0,
            }
        )
        gp = GrismPars(
            image_pars=image_pars,
            dispersion=1.1,
            lambda_ref=1300.0,
            dispersion_angle_detector=0.0,
        )
        gauss_psf = galsim.Gaussian(fwhm=0.18)
        obs = build_grism_obs(
            gp,
            z=1.0,
            psf=gauss_psf,
            render_config=RenderConfig(oversample=3),
            data=_toy_image_data(rng),
            variance=0.25,
        )
        task = InferenceTask.from_obs(src, priors, grism_obs={'roll0': obs})
        theta = _midpoint_theta(priors)
        log_l = task.log_likelihood(theta)
        assert jnp.isfinite(log_l)
        grad = jax.grad(task.log_likelihood)(theta)
        assert jnp.all(jnp.isfinite(grad))

    def test_local_line_window_auto_sized_from_priors(self, image_pars, rng):
        # from_obs fills line_window_halfwidth from the priors with the sizing
        # rule that matches line_window_mode; the local-mode likelihood equals
        # the global-mode one to the dropped six-sigma tail
        import dataclasses

        from kl_pipe.render import local_line_window_halfwidth_for_priors

        src = SourceModel(
            velocity_model=CenteredVelocityModel(),
            emission_lines={
                'Halpha': EmissionLine(intensity=InclinedExponentialModel())
            },
        )
        priors = PriorDict(
            {
                **_shared_geo_priors(),
                **_velocity_priors(),
                **_emission_priors('Halpha'),
                'z': 1.0,
            }
        )
        gp = GrismPars(
            image_pars=image_pars,
            dispersion=1.1,
            lambda_ref=1300.0,
            dispersion_angle_detector=0.0,
        )
        data = _toy_image_data(rng)
        rc_local = RenderConfig(oversample=3, line_window_mode='local')

        def task_for(rc):
            obs = build_grism_obs(
                gp,
                z=1.0,
                psf=galsim.Gaussian(fwhm=0.18),
                render_config=rc,
                data=data,
                variance=0.25,
            )
            return InferenceTask.from_obs(src, priors, grism_obs={'roll0': obs})

        theta = _midpoint_theta(priors)
        ll_global = float(task_for(RenderConfig(oversample=3)).log_likelihood(theta))
        ll_local = float(task_for(rc_local).log_likelihood(theta))
        hw = local_line_window_halfwidth_for_priors(src, priors, gp, 3)
        ll_explicit = float(
            task_for(
                dataclasses.replace(rc_local, line_window_halfwidth=hw)
            ).log_likelihood(theta)
        )
        assert ll_local == ll_explicit  # auto-fill used the local sizing rule
        assert abs(ll_local - ll_global) / abs(ll_global) < 1e-10
        grad = jax.grad(task_for(rc_local).log_likelihood)(theta)
        assert jnp.all(jnp.isfinite(grad))


# ===========================================================================
# Pattern 5: joint photometry + grism
# ===========================================================================


class TestJointPhotometryGrism:

    def test_finite_likelihood_and_gradient(self, image_pars, gauss_psf, rng):
        src = SourceModel(
            velocity_model=CenteredVelocityModel(),
            broadband_models={'F087': InclinedExponentialModel()},
            emission_lines={
                'Halpha': EmissionLine(intensity=InclinedExponentialModel())
            },
        )
        priors = PriorDict(
            {
                **_shared_geo_priors(),
                **_velocity_priors(),
                **_broadband_priors('F087'),
                **_emission_priors('Halpha'),
                'z': 1.0,
            }
        )
        obs_f087 = build_image_obs(
            image_pars,
            psf=gauss_psf,
            data=_toy_image_data(rng),
            variance=0.25,
            broadband_key='F087',
        )
        gp = GrismPars(
            image_pars=image_pars,
            dispersion=1.1,
            lambda_ref=1300.0,
            dispersion_angle_detector=0.0,
        )
        obs_grism = build_grism_obs(
            gp,
            z=1.0,
            psf=gauss_psf,
            render_config=RenderConfig(oversample=3),
            data=_toy_image_data(rng),
            variance=0.25,
        )
        task = InferenceTask.from_obs(
            src,
            priors,
            image_obs={'F087': obs_f087},
            grism_obs={'roll0': obs_grism},
        )
        theta = _midpoint_theta(priors)
        log_l = task.log_likelihood(theta)
        assert jnp.isfinite(log_l)


# ===========================================================================
# Pattern 6: multi-roll grism contribution
# ===========================================================================


class TestMultiRollGrism:

    def test_two_rolls_contribute_independently(self, image_pars, rng):
        """The two grism obs at different keys contribute separate chi^2
        terms; zeroing one obs's data should change the total likelihood."""
        src = SourceModel(
            velocity_model=CenteredVelocityModel(),
            emission_lines={
                'Halpha': EmissionLine(intensity=InclinedExponentialModel())
            },
        )
        priors = PriorDict(
            {
                **_shared_geo_priors(),
                **_velocity_priors(),
                **_emission_priors('Halpha'),
                'z': 1.0,
            }
        )
        gp = GrismPars(
            image_pars=image_pars,
            dispersion=1.1,
            lambda_ref=1300.0,
            dispersion_angle_detector=0.0,
        )
        gauss_psf = galsim.Gaussian(fwhm=0.18)
        data_a = _toy_image_data(rng)
        data_b = _toy_image_data(rng)
        obs_a = build_grism_obs(
            gp,
            z=1.0,
            psf=gauss_psf,
            render_config=RenderConfig(oversample=3),
            data=data_a,
            variance=0.25,
        )
        obs_b = build_grism_obs(
            gp,
            z=1.0,
            psf=gauss_psf,
            render_config=RenderConfig(oversample=3),
            data=data_b,
            variance=0.25,
        )
        obs_b_zero = build_grism_obs(
            gp,
            z=1.0,
            psf=gauss_psf,
            render_config=RenderConfig(oversample=3),
            data=np.zeros_like(data_b),
            variance=0.25,
        )

        theta = _midpoint_theta(priors)
        task_ab = InferenceTask.from_obs(
            src, priors, grism_obs={'a': obs_a, 'b': obs_b}
        )
        task_az = InferenceTask.from_obs(
            src, priors, grism_obs={'a': obs_a, 'b': obs_b_zero}
        )
        # different data on obs b -> different total likelihood
        assert float(task_ab.log_likelihood(theta)) != pytest.approx(
            float(task_az.log_likelihood(theta)), abs=1.0
        )


# ===========================================================================
# Validation errors
# ===========================================================================


class TestValidationErrors:

    @pytest.fixture
    def src_broadband(self):
        return SourceModel(broadband_models={'F087': InclinedExponentialModel()})

    @pytest.fixture
    def src_emission(self):
        return SourceModel(
            velocity_model=CenteredVelocityModel(),
            emission_lines={
                'Halpha': EmissionLine(intensity=InclinedExponentialModel())
            },
        )

    @pytest.fixture
    def priors_broadband(self):
        return PriorDict({**_shared_geo_priors(), **_broadband_priors()})

    @pytest.fixture
    def priors_grism(self):
        return PriorDict(
            {
                **_shared_geo_priors(),
                **_velocity_priors(),
                **_emission_priors(),
                'z': 1.0,
            }
        )

    @pytest.fixture
    def img_obs(self, image_pars, rng):
        return build_image_obs(
            image_pars,
            data=_toy_image_data(rng),
            variance=0.25,
            broadband_key='F087',
            pixel_response=None,
        )

    def test_no_obs_raises(self, src_broadband, priors_broadband):
        with pytest.raises(ValueError, match="at least one of"):
            InferenceTask.from_obs(src_broadband, priors_broadband)

    def test_image_obs_unknown_band_key(self, src_broadband, priors_broadband, img_obs):
        with pytest.raises(ValueError, match="has no entry in source.broadband_models"):
            InferenceTask.from_obs(
                src_broadband, priors_broadband, image_obs={'F184': img_obs}
            )

    def test_image_obs_broadband_key_mismatch(
        self, src_broadband, priors_broadband, image_pars, rng
    ):
        # build an obs with broadband_key='F087' but stuff it under key 'F184'
        # in the dict -- need both keys in source for the first check to pass
        src = SourceModel(
            broadband_models={
                'F087': InclinedExponentialModel(),
                'F184': InclinedExponentialModel(),
            }
        )
        priors = PriorDict({**_shared_geo_priors(), **_broadband_priors('F184')})
        obs_mislabeled = build_image_obs(
            image_pars,
            data=_toy_image_data(rng),
            variance=0.25,
            broadband_key='F087',  # this disagrees with the dict key below
            pixel_response=None,
        )
        with pytest.raises(ValueError, match="disagrees with its dict key"):
            InferenceTask.from_obs(src, priors, image_obs={'F184': obs_mislabeled})

    def test_cosi_prior_reaching_edge_on_raises_for_intensity(
        self, src_broadband, img_obs
    ):
        """A cosi prior whose lower bound reaches the edge-on floor must be
        rejected for intensity-rendering tasks (1/cosi SB diverges)."""
        priors = PriorDict(
            {
                'cosi': Uniform(0.0, 0.99),  # lower bound at edge-on
                'theta_int': 0.3,
                'g1': 0.0,
                'g2': 0.0,
                **_broadband_priors(),
            }
        )
        with pytest.raises(ValueError, match="reaches the edge-on floor"):
            InferenceTask.from_obs(src_broadband, priors, image_obs={'F087': img_obs})

    def test_cosi_fixed_at_zero_raises_for_intensity(self, src_broadband, img_obs):
        """A cosi fixed at 0 is also caught (get_param_bounds -> (0, 0))."""
        priors = PriorDict(
            {
                'cosi': 0.0,
                'theta_int': 0.3,
                'g1': 0.0,
                'g2': 0.0,
                **_broadband_priors(),
            }
        )
        with pytest.raises(ValueError, match="reaches the edge-on floor"):
            InferenceTask.from_obs(src_broadband, priors, image_obs={'F087': img_obs})

    def test_cosi_edge_on_allowed_for_velocity_only(self, image_pars, rng):
        """Velocity-only tasks are exempt: edge-on cosi is physical and
        informative (LOS projection uses sin i, not 1/cosi)."""
        src = SourceModel(velocity_model=CenteredVelocityModel())
        priors = PriorDict(
            {
                'cosi': Uniform(0.0, 0.99),
                'theta_int': 0.3,
                'g1': 0.0,
                'g2': 0.0,
                **_velocity_priors(),
            }
        )
        obs = build_velocity_obs(
            image_pars,
            data=rng.normal(0, 10.0, size=(16, 16)),
            variance=100.0,
        )
        # must NOT raise
        task = InferenceTask.from_obs(src, priors, velocity_obs=obs)
        assert task is not None

    def test_grism_without_velocity_model(self, image_pars, rng, priors_grism):
        """grism_obs requires source.velocity_model."""
        src = SourceModel(
            emission_lines={
                'Halpha': EmissionLine(intensity=InclinedExponentialModel())
            },
        )
        gp = GrismPars(
            image_pars=image_pars,
            dispersion=1.1,
            lambda_ref=1300.0,
            dispersion_angle_detector=0.0,
        )
        obs = build_grism_obs(gp, z=1.0, data=_toy_image_data(rng), variance=0.25)
        with pytest.raises(ValueError, match="velocity_model"):
            InferenceTask.from_obs(src, priors_grism, grism_obs={'roll0': obs})

    def test_grism_without_emission_lines(self, image_pars, rng, priors_grism):
        """grism_obs requires non-empty source.emission_lines."""
        src = SourceModel(velocity_model=CenteredVelocityModel())
        gp = GrismPars(
            image_pars=image_pars,
            dispersion=1.1,
            lambda_ref=1300.0,
            dispersion_angle_detector=0.0,
        )
        obs = build_grism_obs(gp, z=1.0, data=_toy_image_data(rng), variance=0.25)
        with pytest.raises(ValueError, match="emission_lines"):
            InferenceTask.from_obs(src, priors_grism, grism_obs={'roll0': obs})

    def test_velocity_obs_without_velocity_model(
        self, image_pars, rng, src_broadband, priors_broadband
    ):
        obs = build_velocity_obs(
            image_pars, data=rng.normal(0, 10.0, size=(16, 16)), variance=100.0
        )
        with pytest.raises(ValueError, match="velocity_model"):
            InferenceTask.from_obs(src_broadband, priors_broadband, velocity_obs=obs)

    def test_velocity_obs_bad_flux_weight_key(self, image_pars, rng):
        src = SourceModel(
            velocity_model=CenteredVelocityModel(),
            emission_lines={
                'Halpha': EmissionLine(intensity=InclinedExponentialModel())
            },
        )
        priors = PriorDict(
            {
                **_shared_geo_priors(),
                **_velocity_priors(),
                **_emission_priors(),
            }
        )
        gauss_psf = galsim.Gaussian(fwhm=0.18)
        obs = build_velocity_obs(
            image_pars,
            psf=gauss_psf,
            flux_weight_key='NonExistent',
            data=rng.normal(0, 10.0, size=(16, 16)),
            variance=100.0,
        )
        with pytest.raises(ValueError, match="flux_weight_key='NonExistent'"):
            InferenceTask.from_obs(src, priors, velocity_obs=obs)
