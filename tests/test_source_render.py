"""Tests for SourceModel rendering methods.

Covers:
- Theta-routing helpers (_lookup_param resolution order, prefix stripping,
  shear rotation invariance).
- render_broadband (smoke + rotation correctness via celestial-vs-detector
  invariance: rendering with theta_int=phi and image_rotation=phi must
  match rendering with theta_int=0 and image_rotation=0).
- render_velocity (velocity-only path with no PSF + flux-weighted path
  with PSF + flux_weight_key resolution including intensity_key sharing).
- build_cube (Halpha-only, Halpha+NII sharing, dispersion_key sharing,
  continuum + continuum_key, basic shape + flux validation).
- render_grism (full pipeline smoke + zero-roll equivalence).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import galsim  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

from kl_pipe.dispersion import GrismPars  # noqa: E402
from kl_pipe.intensity import InclinedExponentialModel  # noqa: E402
from kl_pipe.lines import EmissionLine  # noqa: E402
from kl_pipe.observation import (  # noqa: E402
    build_grism_obs,
    build_image_obs,
    build_velocity_obs,
)
from kl_pipe.parameters import ImagePars  # noqa: E402
from kl_pipe.source import (  # noqa: E402
    SourceModel,
    _apply_obs_rotation,
    _build_component_theta,
    _lookup_param,
    _strip_param_prefix,
)
from kl_pipe.spectral import CubePars  # noqa: E402
from kl_pipe.velocity import CenteredVelocityModel  # noqa: E402


# ===========================================================================
# Helpers
# ===========================================================================


def _wcs_with_rotation(shape, pixel_scale, phi):
    Nrow, Ncol = shape
    c, s = float(np.cos(phi)), float(np.sin(phi))
    wcs = WCS(naxis=2)
    wcs.wcs.pc = np.array([[c, -s], [s, c]])
    wcs.wcs.cdelt = np.array([pixel_scale, pixel_scale])
    wcs.wcs.crpix = np.array([Ncol / 2, Nrow / 2])
    wcs.wcs.crval = np.array([0.0, 0.0])
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.cunit = ['arcsec', 'arcsec']
    wcs.pixel_shape = (Ncol, Nrow)
    wcs.wcs.set()
    return wcs


_BASE_VEL_PARS = {
    'vel.v0': 0.0,
    'vel.vcirc': 200.0,
    'vel.rscale': 0.3,
}

_BASE_SHARED_PARS = {
    'cosi': 0.5,
    'theta_int': 0.3,
    'g1': 0.02,
    'g2': -0.01,
}


# ===========================================================================
# Theta routing helpers
# ===========================================================================


class TestStripParamPrefix:
    def test_int_prefix(self):
        assert _strip_param_prefix('int_rscale') == 'rscale'

    def test_vel_prefix(self):
        assert _strip_param_prefix('vel_rscale') == 'rscale'

    def test_no_prefix(self):
        assert _strip_param_prefix('cosi') == 'cosi'
        assert _strip_param_prefix('flux') == 'flux'


class TestLookupParam:
    def test_dotted_first(self):
        pars = {'cosi': 0.4, 'F087.rscale': 0.3}
        assert _lookup_param(pars, 'F087', 'int_rscale') == 0.3

    def test_top_level_fallback(self):
        pars = {'cosi': 0.5}
        assert _lookup_param(pars, 'F087', 'cosi') == 0.5

    def test_verbatim_fallback(self):
        pars = {'int_rscale': 0.3}
        assert _lookup_param(pars, 'F087', 'int_rscale') == 0.3

    def test_dotted_wins_over_top_level(self):
        """Per-component value overrides top-level when both present."""
        pars = {'cosi': 0.5, 'F087.cosi': 0.9}
        assert _lookup_param(pars, 'F087', 'cosi') == 0.9

    def test_missing_raises(self):
        with pytest.raises(KeyError, match="could not resolve"):
            _lookup_param({}, 'F087', 'int_rscale')


class TestApplyObsRotation:
    def test_zero_rotation_passthrough(self):
        theta = jnp.array([100.0, 0.3, 0.15, 0.0, 0.0, 0.0, 0.0, 0.3, 0.5])
        names = (
            'flux',
            'int_rscale',
            'int_h_over_r',
            'int_x0',
            'int_y0',
            'g1',
            'g2',
            'theta_int',
            'cosi',
        )
        out = _apply_obs_rotation(theta, names, 0.0)
        assert (out == theta).all()

    def test_theta_int_subtraction(self):
        theta = jnp.array([0.7, 0.5])
        names = ('theta_int', 'cosi')
        out = _apply_obs_rotation(theta, names, np.pi / 4)
        assert float(out[0]) == pytest.approx(0.7 - np.pi / 4, abs=1e-10)
        assert float(out[1]) == pytest.approx(0.5, abs=1e-10)

    def test_shear_rotation(self):
        """g1, g2 rotated by 2*phi (spin-2)."""
        theta = jnp.array([0.05, -0.02])
        names = ('g1', 'g2')
        out = _apply_obs_rotation(theta, names, np.pi / 4)
        # 2*phi = pi/2: g1' = g2, g2' = -g1
        assert float(out[0]) == pytest.approx(-0.02, abs=1e-10)
        assert float(out[1]) == pytest.approx(-0.05, abs=1e-10)


# ===========================================================================
# render_broadband
# ===========================================================================


class TestRenderBroadband:

    @pytest.fixture
    def image_pars(self):
        return ImagePars(shape=(32, 32), pixel_scale=0.1, indexing='ij')

    @pytest.fixture
    def obs(self, image_pars):
        return build_image_obs(image_pars, broadband_key='F087', pixel_response=None)

    @pytest.fixture
    def source(self):
        return SourceModel(broadband_models={'F087': InclinedExponentialModel()})

    @pytest.fixture
    def pars(self):
        return {
            **_BASE_SHARED_PARS,
            'F087.flux': 100.0,
            'F087.rscale': 0.3,
            'F087.h_over_r': 0.15,
            'F087.x0': 0.0,
            'F087.y0': 0.0,
        }

    def test_smoke(self, source, pars, obs):
        img = source.render_broadband(pars, obs, 'F087')
        assert img.shape == (32, 32)
        assert float(img.sum()) > 0

    def test_bad_band_key_raises(self, source, pars, obs):
        with pytest.raises(KeyError, match="band_key 'F184'"):
            source.render_broadband(pars, obs, 'F184')

    def test_celestial_to_detector_rotation_invariance(self, source, pars, image_pars):
        """Rendering with celestial theta_int=phi and image_rotation=phi
        equals rendering with theta_int=0 and image_rotation=0.

        After SourceModel rotation: theta_int_det = phi - phi = 0 vs 0 - 0 = 0.
        Both yield the same detector-frame model. Shear set to zero so the
        spin-2 rotation doesn't enter -- this isolates the theta_int rotation
        invariance.
        """
        phi = np.pi / 6
        pars_no_shear = dict(pars)
        pars_no_shear['g1'] = 0.0
        pars_no_shear['g2'] = 0.0

        # case 1: celestial theta_int = phi, obs rotated by phi
        wcs_rot = _wcs_with_rotation((32, 32), 0.1, phi)
        ip_rot = ImagePars(shape=(32, 32), wcs=wcs_rot, indexing='ij')
        obs_rot = build_image_obs(ip_rot, broadband_key='F087', pixel_response=None)
        pars_rot = dict(pars_no_shear)
        pars_rot['theta_int'] = phi
        img_rot = source.render_broadband(pars_rot, obs_rot, 'F087')

        # case 2: celestial theta_int = 0, identity WCS
        obs_id = build_image_obs(image_pars, broadband_key='F087', pixel_response=None)
        pars_id = dict(pars_no_shear)
        pars_id['theta_int'] = 0.0
        img_id = source.render_broadband(pars_id, obs_id, 'F087')

        np.testing.assert_allclose(np.asarray(img_rot), np.asarray(img_id), atol=1e-8)

    def test_shear_rotation_invariance(self, source, pars, image_pars):
        """Same invariance for shear: celestial (g1_cel, g2_cel) at
        image_rotation=phi must equal detector-frame shear (g1_det, g2_det)
        at image_rotation=0, where (g1_det, g2_det) = rotate_shear(g1_cel, g2_cel, phi).
        """
        from kl_pipe.coordinates import rotate_shear

        phi = np.pi / 5
        g1_cel, g2_cel = 0.03, -0.025
        g1_det, g2_det = rotate_shear(g1_cel, g2_cel, phi)

        # case 1: celestial shear, obs rotated by phi (theta_int set to phi so
        # theta_int_det = 0 in both cases)
        wcs_rot = _wcs_with_rotation((32, 32), 0.1, phi)
        ip_rot = ImagePars(shape=(32, 32), wcs=wcs_rot, indexing='ij')
        obs_rot = build_image_obs(ip_rot, broadband_key='F087', pixel_response=None)
        pars_rot = dict(pars)
        pars_rot['theta_int'] = phi
        pars_rot['g1'] = g1_cel
        pars_rot['g2'] = g2_cel
        img_rot = source.render_broadband(pars_rot, obs_rot, 'F087')

        # case 2: detector shear directly, identity WCS, theta_int=0
        obs_id = build_image_obs(image_pars, broadband_key='F087', pixel_response=None)
        pars_id = dict(pars)
        pars_id['theta_int'] = 0.0
        pars_id['g1'] = float(g1_det)
        pars_id['g2'] = float(g2_det)
        img_id = source.render_broadband(pars_id, obs_id, 'F087')

        np.testing.assert_allclose(np.asarray(img_rot), np.asarray(img_id), atol=1e-8)


# ===========================================================================
# render_velocity
# ===========================================================================


class TestRenderVelocity:

    @pytest.fixture
    def image_pars(self):
        return ImagePars(shape=(16, 16), pixel_scale=0.1, indexing='ij')

    @pytest.fixture
    def source(self):
        return SourceModel(
            velocity_model=CenteredVelocityModel(),
            emission_lines={
                'Halpha': EmissionLine(intensity=InclinedExponentialModel())
            },
        )

    @pytest.fixture
    def pars(self):
        return {
            **_BASE_SHARED_PARS,
            **_BASE_VEL_PARS,
            'Halpha.flux': 100.0,
            'Halpha.rscale': 0.3,
            'Halpha.h_over_r': 0.15,
            'Halpha.x0': 0.0,
            'Halpha.y0': 0.0,
        }

    def test_velocity_only_no_psf(self, image_pars, source, pars):
        """flux_weight_key=None + no PSF -> renders mean LOS velocity map."""
        obs = build_velocity_obs(image_pars)
        v_map = source.render_velocity(pars, obs)
        assert v_map.shape == (16, 16)
        # rotation curve at this geometry should produce non-trivial range
        assert float(v_map.max()) - float(v_map.min()) > 100

    def test_no_velocity_model_raises(self, image_pars, pars):
        src = SourceModel(broadband_models={'F087': InclinedExponentialModel()})
        obs = build_velocity_obs(image_pars)
        with pytest.raises(ValueError, match="velocity_model"):
            src.render_velocity(pars, obs)

    def test_flux_weighted_with_psf(self, image_pars, source, pars):
        """PSF + flux_weight_key -> SourceModel computes flux map and threads it."""
        gauss_psf = galsim.Gaussian(fwhm=0.18)
        obs = build_velocity_obs(
            image_pars, psf=gauss_psf, flux_weight_key='Halpha', oversample=3
        )
        v_map = source.render_velocity(pars, obs)
        assert v_map.shape == (16, 16)


# ===========================================================================
# build_cube
# ===========================================================================


class TestBuildCube:

    @pytest.fixture
    def image_pars(self):
        return ImagePars(shape=(16, 16), pixel_scale=0.11, indexing='ij')

    @pytest.fixture
    def cube_pars(self, image_pars):
        return CubePars.from_range(image_pars, 1300.0, 1320.0, 1.0)

    @pytest.fixture
    def pars(self):
        return {
            'z': 1.0,
            **_BASE_SHARED_PARS,
            **_BASE_VEL_PARS,
            'Halpha.flux': 100.0,
            'Halpha.rscale': 0.3,
            'Halpha.h_over_r': 0.15,
            'Halpha.x0': 0.0,
            'Halpha.y0': 0.0,
            'Halpha.dispersion': 50.0,
        }

    def test_halpha_only(self, cube_pars, pars):
        src = SourceModel(
            velocity_model=CenteredVelocityModel(),
            emission_lines={
                'Halpha': EmissionLine(intensity=InclinedExponentialModel())
            },
        )
        cube = src.build_cube(pars, cube_pars, spectral_oversample=3)
        assert cube.shape == (16, 16, cube_pars.n_lambda)
        assert float(cube.sum()) > 0

    def test_intensity_key_sharing(self, cube_pars, pars):
        """NII6584 shares Halpha's spatial profile via intensity_key + dispersion_key.
        NII6584's flux is independent (per-line)."""
        src = SourceModel(
            velocity_model=CenteredVelocityModel(),
            emission_lines={
                'Halpha': EmissionLine(intensity=InclinedExponentialModel()),
                'NII6584': EmissionLine(
                    intensity_key='Halpha', dispersion_key='Halpha'
                ),
            },
        )
        pars_shared = dict(pars)
        pars_shared['NII6584.flux'] = 30.0
        cube = src.build_cube(pars_shared, cube_pars, spectral_oversample=3)
        # both lines contribute; cube total > Halpha-only case
        src_alone = SourceModel(
            velocity_model=CenteredVelocityModel(),
            emission_lines={
                'Halpha': EmissionLine(intensity=InclinedExponentialModel())
            },
        )
        cube_alone = src_alone.build_cube(pars, cube_pars, spectral_oversample=3)
        assert float(cube.sum()) > float(cube_alone.sum())

    def test_continuum(self, image_pars, pars):
        """Continuum contribution adds flux to the cube uniformly in wavelength
        (modulo emission line broadening)."""
        # build a wider lambda range so the line broadening is far from edges
        cube_pars = CubePars.from_range(image_pars, 1280.0, 1340.0, 1.0)
        src = SourceModel(
            velocity_model=CenteredVelocityModel(),
            emission_lines={
                'Halpha': EmissionLine(
                    intensity=InclinedExponentialModel(),
                    continuum=InclinedExponentialModel(),
                )
            },
        )
        pars_cont = dict(pars)
        pars_cont.update(
            {
                'Halpha.cont.flux': 5.0,
                'Halpha.cont.rscale': 0.3,
                'Halpha.cont.h_over_r': 0.15,
                'Halpha.cont.x0': 0.0,
                'Halpha.cont.y0': 0.0,
            }
        )
        cube_with_cont = src.build_cube(pars_cont, cube_pars, spectral_oversample=3)

        # same model without continuum
        src_no = SourceModel(
            velocity_model=CenteredVelocityModel(),
            emission_lines={
                'Halpha': EmissionLine(intensity=InclinedExponentialModel())
            },
        )
        cube_no_cont = src_no.build_cube(pars, cube_pars, spectral_oversample=3)

        assert float(cube_with_cont.sum()) > float(cube_no_cont.sum())

    def test_no_velocity_model_raises(self, cube_pars, pars):
        src = SourceModel(
            emission_lines={
                'Halpha': EmissionLine(intensity=InclinedExponentialModel())
            }
        )
        with pytest.raises(ValueError, match="velocity_model"):
            src.build_cube(pars, cube_pars)

    def test_v0_shifts_cube(self, cube_pars, pars):
        """Non-zero ``vel.v0`` must shift the rendered cube in wavelength.

        Regression for a previous bug where build_cube subtracted v0
        before computing Doppler ("rotation-only Doppler"), making the
        systemic velocity observationally invisible. The physical
        behavior is that v_los = v0 + v_rotation, and the line wavelength
        at each pixel must shift by (1 + v_los/c), including the v0
        contribution.

        The v0/z degeneracy is real and is left to the caller to manage
        at the prior level -- but the *forward model* must include v0 or
        a non-zero v0 silently has no effect (a class of bug this test
        is designed to catch).
        """
        from kl_pipe.source import _C_KMS as c_kms

        src = SourceModel(
            velocity_model=CenteredVelocityModel(),
            emission_lines={
                'Halpha': EmissionLine(intensity=InclinedExponentialModel())
            },
        )

        pars_zero = dict(pars)
        pars_zero['vel.v0'] = 0.0

        pars_nonzero = dict(pars)
        pars_nonzero['vel.v0'] = 200.0  # km/s

        cube_zero = src.build_cube(pars_zero, cube_pars, spectral_oversample=3)
        cube_v0 = src.build_cube(pars_nonzero, cube_pars, spectral_oversample=3)

        # 1. cubes must differ
        assert not jnp.allclose(cube_zero, cube_v0), (
            "cube with v0=200 km/s identical to cube with v0=0 -- "
            "v0 is not entering the Doppler shift (renderer bug)"
        )

        # 2. spectral centroid must shift by (1 + v0/c) at fixed z.
        # Compute the wavelength-weighted mean per pixel and average over
        # nonzero pixels.
        lam_grid = cube_pars.lambda_grid

        def centroid(cube):
            num = (cube * lam_grid[None, None, :]).sum(axis=-1)
            den = cube.sum(axis=-1)
            # mean wavelength where there's significant flux
            mask = den > 0.01 * float(den.max())
            return float((num[mask] / den[mask]).mean())

        lam_mean_zero = centroid(cube_zero)
        lam_mean_v0 = centroid(cube_v0)
        shift = lam_mean_v0 - lam_mean_zero

        # Expected shift: lambda_rest * (1+z) * v0 / c
        # lambda_rest is the Halpha registry value (656.28 nm)
        from kl_pipe.lines import LINE_LAMBDAS

        z = pars_zero['z']
        expected = LINE_LAMBDAS['Halpha'] * (1.0 + z) * (200.0 / c_kms)
        # Tolerance 5% accounts for line-broadening + intensity weighting
        assert abs(shift - expected) / expected < 0.05, (
            f"v0=200 km/s should shift line centroid by ~{expected:.4f} nm; "
            f"got {shift:.4f} nm"
        )

    def test_no_emission_lines_raises(self, cube_pars, pars):
        src = SourceModel(velocity_model=CenteredVelocityModel())
        with pytest.raises(ValueError, match="emission_lines"):
            src.build_cube(pars, cube_pars)


# ===========================================================================
# render_grism
# ===========================================================================


class TestRenderGrism:

    @pytest.fixture
    def image_pars(self):
        return ImagePars(shape=(16, 16), pixel_scale=0.11, indexing='ij')

    @pytest.fixture
    def source(self):
        return SourceModel(
            velocity_model=CenteredVelocityModel(),
            emission_lines={
                'Halpha': EmissionLine(intensity=InclinedExponentialModel())
            },
        )

    @pytest.fixture
    def pars(self):
        return {
            'z': 1.0,
            **_BASE_SHARED_PARS,
            **_BASE_VEL_PARS,
            'Halpha.flux': 100.0,
            'Halpha.rscale': 0.3,
            'Halpha.h_over_r': 0.15,
            'Halpha.x0': 0.0,
            'Halpha.y0': 0.0,
            'Halpha.dispersion': 50.0,
        }

    def test_smoke(self, image_pars, source, pars):
        gp = GrismPars(
            image_pars=image_pars,
            dispersion=1.1,
            lambda_ref=1300.0,
            dispersion_angle_detector=0.0,
        )
        gauss_psf = galsim.Gaussian(fwhm=0.18)
        obs = build_grism_obs(gp, z=1.0, psf=gauss_psf, oversample=3)
        img = source.render_grism(pars, obs)
        assert img.shape == (16, 16)
        assert float(img.sum()) > 0
