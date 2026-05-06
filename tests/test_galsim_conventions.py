"""GalSim parameter-convention regression tests.

GalSim's ``InclinedSersic`` reinterprets ``scale_h_over_r`` as
``h_z/scale_radius`` (NOT ``h_z/half_light_radius``) when ``half_light_radius``
is supplied. For Sersic n=4 the resulting physical h_z differs by
``b_n**n ~ 3463`` — silent thickness mismatch.

Tests:

1. Production-path correctness — ``kl_pipe.synthetic._generate_sersic_galsim``
   uses ``scale_radius`` parameterization (no ``half_light_radius``), so the
   convention is unambiguous. Verify renders match the kl_pipe Sersic model
   at matched physical h_z.
2. Convention regression guard — ``half_light_radius + scale_h_over_r`` and
   ``half_light_radius + scale_height`` produce different physical h_z. If
   GalSim ever changes this behavior, this test must be revisited.
3. Equivalence under matched physical h_z — three valid GalSim constructions
   produce equivalent renders.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

import galsim as gs

from kl_pipe.intensity import InclinedSersicModel, _sersic_bn
from kl_pipe.observation import build_image_obs
from kl_pipe.parameters import ImagePars
from kl_pipe.synthetic import _generate_sersic_galsim


# Test geometry — n=4 chosen to maximize the b_n^n convention sensitivity.
_N = 4.0
_HLR = 0.8  # arcsec
_H_OVER_HLR = 0.3  # h_z / hlr
_COSI = 0.7
_FLUX = 1.0
_PIXEL_SCALE = 0.15
_NPIX = 32
_PSF_FWHM = 0.15  # arcsec; matches typical Roman-like PSF


def _gs_params():
    return gs.GSParams(
        folding_threshold=1e-3,
        maxk_threshold=1e-3,
        kvalue_accuracy=1e-5,
        maximum_fft_size=32768,
    )


def _make_psf():
    return gs.Gaussian(fwhm=_PSF_FWHM, gsparams=_gs_params())


def _draw(profile, psf, npix=_NPIX, pixel_scale=_PIXEL_SCALE):
    convolved = gs.Convolve(profile, psf, gsparams=_gs_params())
    image = convolved.drawImage(nx=npix, ny=npix, scale=pixel_scale, method='auto')
    return image.array


def test_production_path_galsim_synthetic_matches_klpipe_sersic():
    """``_generate_sersic_galsim`` (scale_radius parameterization) must match
    kl_pipe's InclinedSersicModel render at matched physical h_z."""
    bn = float(_sersic_bn(_N))
    int_rscale = _HLR / bn**_N
    h_z = _H_OVER_HLR * _HLR
    int_h_over_r = h_z / int_rscale  # = _H_OVER_HLR * bn**_N

    image_pars = ImagePars(
        shape=(_NPIX, _NPIX), pixel_scale=_PIXEL_SCALE, indexing='ij'
    )
    psf = _make_psf()

    gs_image = _generate_sersic_galsim(
        image_pars=image_pars,
        flux=_FLUX,
        int_rscale=int_rscale,
        n_sersic=_N,
        cosi=_COSI,
        theta_int=0.0,
        g1=0.0,
        g2=0.0,
        int_x0=0.0,
        int_y0=0.0,
        int_h_over_r=int_h_over_r,
        gsparams=_gs_params(),
        psf=psf,
        method='auto',
    )

    model = InclinedSersicModel()
    theta = jnp.array([_COSI, 0.0, 0.0, 0.0, _FLUX, _HLR, _H_OVER_HLR, _N, 0.0, 0.0])
    obs = build_image_obs(image_pars, psf=psf, oversample=5)
    kl_image = np.array(model.render_image(theta, obs=obs))

    peak = np.max(np.abs(gs_image))
    rms_frac = np.sqrt(np.mean((kl_image - gs_image) ** 2)) / peak
    central_frac = (
        kl_image[_NPIX // 2, _NPIX // 2] - gs_image[_NPIX // 2, _NPIX // 2]
    ) / peak

    # Sersic n=4 emulator + 3D LOS quadrature accuracy: ~5% RMS at this geometry
    # with PSF + oversample=5. Tighter than 10% confirms the physical h_z is
    # matched and the production path is correctly parameterized.
    assert (
        rms_frac < 0.10
    ), f'RMS frac {rms_frac:.3f} too large; production path likely mis-parameterized'
    assert (
        abs(central_frac) < 0.10
    ), f'central frac {central_frac:+.3f} suggests h_z mismatch'


def test_galsim_scale_h_over_r_with_half_light_radius_uses_scale_radius_convention():
    """Regression guard for GalSim's documented convention.

    With ``half_light_radius`` supplied, GalSim treats ``scale_h_over_r`` as
    ``h_z / scale_radius`` (NOT ``h_z / half_light_radius``). For Sersic n=4
    that means the implied physical h_z is ~b_n^n times smaller than passing
    ``scale_height = h_over_hlr * hlr`` directly.

    If GalSim ever changes this — making both spellings physically equivalent
    — this test will newly fail and our convention fixes need revisiting.
    """
    psf = _make_psf()
    bn = float(_sersic_bn(_N))
    expected_thickness_ratio = bn**_N  # ~3463 for n=4
    inc = gs.Angle(np.arccos(_COSI), gs.radians)

    # Construction A: half_light_radius + scale_h_over_r (the bug pattern)
    prof_a = gs.InclinedSersic(
        n=_N,
        inclination=inc,
        half_light_radius=_HLR,
        scale_h_over_r=_H_OVER_HLR,
        flux=_FLUX,
        gsparams=_gs_params(),
    )

    # Construction B: half_light_radius + scale_height (the fix pattern)
    prof_b = gs.InclinedSersic(
        n=_N,
        inclination=inc,
        half_light_radius=_HLR,
        scale_height=_H_OVER_HLR * _HLR,
        flux=_FLUX,
        gsparams=_gs_params(),
    )

    # Confirm GalSim's reported scale_height differs by exactly bn^n
    ratio = prof_b.scale_height / prof_a.scale_height
    rel_err = abs(ratio - expected_thickness_ratio) / expected_thickness_ratio
    assert rel_err < 1e-6, (
        f'Expected scale_height(B)/scale_height(A) = bn^n = {expected_thickness_ratio:.1f}, '
        f'got {ratio:.3f}. GalSim convention may have changed.'
    )

    # And confirm the rendered images visibly disagree even after PSF blurring.
    # PSF FWHM (0.15) is comparable to projected h_z (0.24*sini~0.17), so the
    # PSF partially absorbs the thickness difference; threshold reflects what's
    # left after the PSF, not the underlying b_n^n mismatch.
    img_a = _draw(prof_a, psf)
    img_b = _draw(prof_b, psf)
    peak_b = np.max(np.abs(img_b))
    rms_diff_frac = np.sqrt(np.mean((img_a - img_b) ** 2)) / peak_b
    assert rms_diff_frac > 0.01, (
        f'half_light_radius+scale_h_over_r and half_light_radius+scale_height '
        f'produced equivalent renders (RMS diff {rms_diff_frac:.3f}). '
        f'GalSim convention may have changed; revisit fixes.'
    )


def test_galsim_scale_height_equivalent_constructions():
    """At matched physical h_z, three GalSim constructions render identically.

    Construction A: half_light_radius + scale_height
    Construction B: scale_radius + scale_height
    Construction C: scale_radius + scale_h_over_r (no half_light_radius)
    """
    psf = _make_psf()
    inc = gs.Angle(np.arccos(_COSI), gs.radians)
    bn = float(_sersic_bn(_N))
    scale_radius = _HLR / bn**_N
    h_z = _H_OVER_HLR * _HLR
    scale_h_over_r_at_rs = h_z / scale_radius  # = _H_OVER_HLR * bn**_N

    prof_a = gs.InclinedSersic(
        n=_N,
        inclination=inc,
        half_light_radius=_HLR,
        scale_height=h_z,
        flux=_FLUX,
        gsparams=_gs_params(),
    )
    prof_b = gs.InclinedSersic(
        n=_N,
        inclination=inc,
        scale_radius=scale_radius,
        scale_height=h_z,
        flux=_FLUX,
        gsparams=_gs_params(),
    )
    prof_c = gs.InclinedSersic(
        n=_N,
        inclination=inc,
        scale_radius=scale_radius,
        scale_h_over_r=scale_h_over_r_at_rs,
        flux=_FLUX,
        gsparams=_gs_params(),
    )

    img_a = _draw(prof_a, psf)
    img_b = _draw(prof_b, psf)
    img_c = _draw(prof_c, psf)
    peak = np.max(np.abs(img_a))
    rms_ab = np.sqrt(np.mean((img_a - img_b) ** 2)) / peak
    rms_ac = np.sqrt(np.mean((img_a - img_c) ** 2)) / peak

    assert rms_ab < 1e-4, f'A vs B RMS diff {rms_ab:.2e} (expected <1e-4)'
    assert rms_ac < 1e-4, f'A vs C RMS diff {rms_ac:.2e} (expected <1e-4)'
