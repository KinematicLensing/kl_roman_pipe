"""Tests verifying obs builders preserve image_pars.wcs through construction.

Regression guard: render methods derive the celestial-to-detector rotation
from ``obs.image_pars.wcs`` (or ``obs.grism_pars.image_pars.wcs``) at trace
time via ``image_rotation_from_wcs``. If the obs builder drops the WCS or
replaces it with one that's lost the original rotation, every rendered
output silently shifts. These tests pin that contract.

Existing tests construct ImagePars with only ``pixel_scale``, which yields
an identity-rotation WCS. All three obs types must therefore agree that
``image_rotation_from_wcs(obs.<image_pars>.wcs) == 0`` in that path.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.wcs import WCS

from kl_pipe.coordinates import image_rotation_from_wcs
from kl_pipe.dispersion import GrismPars
from kl_pipe.observation import (
    build_grism_obs,
    build_image_obs,
    build_velocity_obs,
)
from kl_pipe.parameters import ImagePars


def _wcs_with_pc(shape, pixel_scale, rotation_radians):
    Nrow, Ncol = shape
    c = float(np.cos(rotation_radians))
    s = float(np.sin(rotation_radians))
    pc = np.array([[c, -s], [s, c]])
    wcs = WCS(naxis=2)
    wcs.wcs.pc = pc
    wcs.wcs.cdelt = np.array([pixel_scale, pixel_scale])
    wcs.wcs.crpix = np.array([Ncol / 2, Nrow / 2])
    wcs.wcs.crval = np.array([0.0, 0.0])
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.cunit = ['arcsec', 'arcsec']
    wcs.pixel_shape = (Ncol, Nrow)
    wcs.wcs.set()
    return wcs


class TestDefaultWcsRotationIsZero:
    """Regression guard: existing tests that pass only pixel_scale must get 0.

    If this fails, the ImagePars default constructor has drifted and
    every legacy test is potentially broken at face-value.
    """

    def test_image_obs_default_wcs(self):
        ip = ImagePars(shape=(32, 32), pixel_scale=0.1, indexing='ij')
        obs = build_image_obs(ip)
        rotation = image_rotation_from_wcs(obs.image_pars.wcs)
        assert rotation == pytest.approx(0.0, abs=1e-12)

    def test_velocity_obs_default_wcs(self):
        ip = ImagePars(shape=(32, 32), pixel_scale=0.1, indexing='ij')
        obs = build_velocity_obs(ip)
        rotation = image_rotation_from_wcs(obs.image_pars.wcs)
        assert rotation == pytest.approx(0.0, abs=1e-12)

    def test_grism_obs_default_wcs(self):
        ip = ImagePars(shape=(32, 32), pixel_scale=0.11, indexing='ij')
        gp = GrismPars(
            image_pars=ip,
            dispersion=1.1,
            lambda_ref=1300.0,
            dispersion_angle_detector=0.0,
        )
        obs = build_grism_obs(gp, z=1.0)
        rotation = image_rotation_from_wcs(obs.grism_pars.image_pars.wcs)
        assert rotation == pytest.approx(0.0, abs=1e-12)


class TestCustomWcsRotationPropagates:
    """Non-trivial PC matrix must survive obs construction so render methods
    can recover the correct rotation via image_rotation_from_wcs."""

    @pytest.mark.parametrize("phi", [np.pi / 6, np.pi / 4, np.pi / 2, -np.pi / 3])
    def test_image_obs(self, phi):
        wcs = _wcs_with_pc((32, 32), 0.1, phi)
        ip = ImagePars(shape=(32, 32), wcs=wcs, indexing='ij')
        obs = build_image_obs(ip)
        rotation = image_rotation_from_wcs(obs.image_pars.wcs)
        assert rotation == pytest.approx(phi, abs=1e-12)

    @pytest.mark.parametrize("phi", [np.pi / 6, np.pi / 4, np.pi / 2])
    def test_velocity_obs(self, phi):
        wcs = _wcs_with_pc((32, 32), 0.1, phi)
        ip = ImagePars(shape=(32, 32), wcs=wcs, indexing='ij')
        obs = build_velocity_obs(ip)
        rotation = image_rotation_from_wcs(obs.image_pars.wcs)
        assert rotation == pytest.approx(phi, abs=1e-12)

    @pytest.mark.parametrize("phi", [np.pi / 6, np.pi / 4, np.pi / 2])
    def test_grism_obs(self, phi):
        wcs = _wcs_with_pc((32, 32), 0.11, phi)
        ip = ImagePars(shape=(32, 32), wcs=wcs, indexing='ij')
        gp = GrismPars(
            image_pars=ip,
            dispersion=1.1,
            lambda_ref=1300.0,
            dispersion_angle_detector=0.0,
        )
        obs = build_grism_obs(gp, z=1.0)
        rotation = image_rotation_from_wcs(obs.grism_pars.image_pars.wcs)
        assert rotation == pytest.approx(phi, abs=1e-12)


class TestObsBindingKwargs:
    """The new broadband_key / flux_weight_key kwargs propagate to the obs."""

    def test_image_obs_broadband_key(self):
        ip = ImagePars(shape=(32, 32), pixel_scale=0.1, indexing='ij')
        obs = build_image_obs(ip, broadband_key='F087')
        assert obs.broadband_key == 'F087'

    def test_image_obs_broadband_key_default_none(self):
        ip = ImagePars(shape=(32, 32), pixel_scale=0.1, indexing='ij')
        obs = build_image_obs(ip)
        assert obs.broadband_key is None

    def test_velocity_obs_flux_weight_key(self):
        ip = ImagePars(shape=(32, 32), pixel_scale=0.1, indexing='ij')
        obs = build_velocity_obs(ip, flux_weight_key='Halpha')
        assert obs.flux_weight_key == 'Halpha'
        # broadband_key is inherited from ImageObs and remains None
        assert obs.broadband_key is None

    def test_velocity_obs_flux_weight_key_default_none(self):
        ip = ImagePars(shape=(32, 32), pixel_scale=0.1, indexing='ij')
        obs = build_velocity_obs(ip)
        assert obs.flux_weight_key is None
