"""
Pseudocode examples for the SourceModel API.

NOT meant to execute as a single script — this file illustrates the
design for collaborators. Sections marked as commented-out workflows
reference rendering / inference factory methods that may not yet exist.

Coverage:
  - Example A:  minimal grism+broadband (Halpha only)
  - Example B:  Halpha + [NII] sharing spatial profile + dispersion via *_key
  - Example C:  multi-band + multi-line + per-line continuum (stretch case)
  - Workflow A: grism-only, single roll
  - Workflow B: F087 + Ha+[NII] grism, single roll
  - Workflow C: multi-band + grism
  - Workflow D: broadband-only single band
  - Workflow D2: broadband-only multi-band
  - Workflow E: velocity-only (IFU / kinematics-only)
  - Workflow E2: velocity + photometry (IFU + broadband)
  - Workflow F: multi-roll grism (2 broadband + 2 grism rolls)
  - Workflow G: kitchen sink (4-band + 3-roll grism + velocity)

Conventions:
  - Sampled top-level params (g1, g2, cosi, theta_int) are in CELESTIAL frame.
    SourceModel rotates them into each obs's detector frame via the obs's WCS.
  - <line>.dispersion is the intrinsic kinematic velocity dispersion of the
    emitting gas in km/s (post-Phase-2 — no instrumental term).
  - dispersion_key on EmissionLine shares dispersion across lines, parallel
    to intensity_key / continuum_key for spatial / continuum profiles.
"""

from typing import Optional, Dict

import numpy as np
import jax.numpy as jnp
from astropy.wcs import WCS

from kl_pipe.lines import EmissionLine
from kl_pipe.priors import PriorDict, Uniform, Gaussian
from kl_pipe.source import SourceModel
from kl_pipe.velocity import CenteredVelocityModel, OffsetVelocityModel
from kl_pipe.intensity import (
    InclinedExponentialModel,
    InclinedSersicModel,
    BulgeDiskModel,
)
from kl_pipe.observation import (
    build_image_obs,
    build_grism_obs,
    build_velocity_obs,
)
from kl_pipe.sampling.task import InferenceTask


# ---------------------------------------------------------------------------
# Helper: build an astropy WCS with a non-trivial roll angle (for Workflow F/G)
# ---------------------------------------------------------------------------


def make_rolled_wcs(shape, pixel_scale_arcsec, roll_radians, crval=(0.0, 0.0)):
    """Construct a WCS with a celestial-to-detector rotation.

    roll_radians = 0 yields an identity-rotation WCS (detector +x aligned with
    celestial +RA). Non-zero values rotate the detector relative to celestial.
    """
    Nrow, Ncol = shape
    wcs = WCS(naxis=2)
    c, s = np.cos(roll_radians), np.sin(roll_radians)
    pc = np.array([[c, -s], [s, c]])
    wcs.wcs.pc = pc
    wcs.wcs.cdelt = np.array([pixel_scale_arcsec, pixel_scale_arcsec])
    wcs.wcs.crpix = np.array([Ncol / 2, Nrow / 2])
    wcs.wcs.crval = np.array(crval)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.cunit = ['arcsec', 'arcsec']
    wcs.pixel_shape = (Ncol, Nrow)
    wcs.wcs.set()
    return wcs


# ===========================================================================
# Example A — minimal: F087 imaging + G150 grism with Halpha only, z=1
# ===========================================================================

source_A = SourceModel(
    velocity_model=CenteredVelocityModel(),
    broadband_models={'F087': InclinedExponentialModel()},
    emission_lines={
        'Halpha': EmissionLine(intensity=InclinedExponentialModel()),
        # lambda_rest = 656.28 nm auto-resolved from LINE_LAMBDAS by dict key
    },
)

priors_A = PriorDict(
    {
        # --- velocity (no vel.dispersion — dispersion lives per emission line) ---
        'vel.vcirc': Uniform(80, 300),
        'vel.v0': 0.0,
        'vel.rscale': Uniform(0.05, 1.0),
        'vel.x0': Uniform(-0.2, 0.2),
        'vel.y0': Uniform(-0.2, 0.2),
        # --- broadband F087 ---
        'F087.flux': Uniform(1, 200),
        'F087.rscale': Uniform(0.05, 1.0),
        'F087.h_over_r': 0.15,
        'F087.x0': Uniform(-0.2, 0.2),
        'F087.y0': Uniform(-0.2, 0.2),
        # --- emission Halpha ---
        'Halpha.flux': Uniform(1, 100),
        'Halpha.rscale': Uniform(0.05, 1.0),
        'Halpha.h_over_r': 0.15,
        'Halpha.x0': Uniform(-0.2, 0.2),
        'Halpha.y0': Uniform(-0.2, 0.2),
        'Halpha.dispersion': Uniform(20, 150),  # km/s, intrinsic gas dispersion
        # --- shared (celestial-frame) ---
        'g1': Uniform(-0.1, 0.1),
        'g2': Uniform(-0.1, 0.1),
        'cosi': Uniform(0.1, 0.99),
        'theta_int': Uniform(0, jnp.pi),
        'z': 1.0,
    }
)


# ===========================================================================
# Example B — production-typical: F087 + Halpha+[NII] sharing spatial + dispersion
# ===========================================================================

source_B = SourceModel(
    velocity_model=OffsetVelocityModel(),
    broadband_models={'F087': InclinedExponentialModel()},
    emission_lines={
        'Halpha': EmissionLine(intensity=InclinedExponentialModel()),
        # NII6584 shares both spatial profile AND intrinsic dispersion with Halpha
        # via *_key references. NII6584.flux is independent (in priors).
        'NII6584': EmissionLine(
            intensity_key='Halpha',
            dispersion_key='Halpha',
        ),
    },
)

priors_B = PriorDict(
    {
        'vel.vcirc': Uniform(80, 300),
        'vel.v0': Gaussian(0.0, 30.0),
        'vel.rscale': Uniform(0.05, 1.0),
        'vel.x0': Gaussian(0.0, 0.1),
        'vel.y0': Gaussian(0.0, 0.1),
        'F087.flux': Uniform(1, 200),
        'F087.rscale': Uniform(0.05, 1.0),
        'F087.h_over_r': 0.15,
        'F087.x0': Uniform(-0.2, 0.2),
        'F087.y0': Uniform(-0.2, 0.2),
        'Halpha.flux': Uniform(1, 100),
        'Halpha.rscale': Uniform(0.05, 1.0),
        'Halpha.h_over_r': 0.15,
        'Halpha.x0': Uniform(-0.2, 0.2),
        'Halpha.y0': Uniform(-0.2, 0.2),
        'Halpha.dispersion': Uniform(20, 150),
        'NII6584.flux': Uniform(0.05, 50),  # NII/Ha typically 0.1-0.5
        # NB: NII6584.rscale, .x0, .y0, .dispersion are NOT in priors —
        # resolved via intensity_key + dispersion_key
        'g1': Uniform(-0.1, 0.1),
        'g2': Uniform(-0.1, 0.1),
        'cosi': Uniform(0.1, 0.99),
        'theta_int': Uniform(0, jnp.pi),
        'z': Gaussian(1.000, 0.005),
    }
)


# ===========================================================================
# Example C — stretch: F087 (bulge+disk) + F184 (disk) + Halpha (Sersic) +
#                       [OIII] (exp) + per-line continuum, z=1.5
# ===========================================================================

source_C = SourceModel(
    velocity_model=OffsetVelocityModel(),
    broadband_models={
        'F087': BulgeDiskModel(),
        'F184': InclinedExponentialModel(),
    },
    emission_lines={
        'Halpha': EmissionLine(
            intensity=InclinedSersicModel(),
            continuum=InclinedExponentialModel(),
        ),
        # OIII5007: independent spatial; shares continuum spatial with Halpha;
        # independent dispersion (Halpha and OIII trace different gas phases
        # in AGN; per-line dispersion captures that).
        'OIII5007': EmissionLine(
            intensity=InclinedExponentialModel(),
            continuum_key='Halpha',
        ),
    },
)

priors_C = PriorDict(
    {
        'vel.vcirc': Uniform(80, 300),
        'vel.v0': Gaussian(0.0, 30.0),
        'vel.rscale': Uniform(0.05, 1.0),
        'vel.x0': Gaussian(0.0, 0.1),
        'vel.y0': Gaussian(0.0, 0.1),
        # broadband F087: bulge + disk composite
        'F087.total_flux': Uniform(1, 200),
        'F087.bulge_frac': Uniform(0.0, 0.6),
        'F087.rscale': Uniform(0.05, 1.0),  # disk scale radius
        'F087.h_over_r': 0.15,
        'F087.bulge_rscale': Uniform(0.05, 0.4),
        'F087.bulge_n_sersic': 4.0,
        'F087.x0': Uniform(-0.2, 0.2),
        'F087.y0': Uniform(-0.2, 0.2),
        # broadband F184: pure disk
        'F184.flux': Uniform(1, 200),
        'F184.rscale': Uniform(0.05, 1.0),
        'F184.h_over_r': 0.15,
        'F184.x0': Uniform(-0.2, 0.2),
        'F184.y0': Uniform(-0.2, 0.2),
        # Halpha: Sersic intensity + own continuum (exp), per-line dispersion
        'Halpha.flux': Uniform(1, 100),
        'Halpha.rscale': Uniform(0.05, 1.0),
        'Halpha.n_sersic': Uniform(0.3, 1.5),
        'Halpha.h_over_r': 0.15,
        'Halpha.x0': Uniform(-0.2, 0.2),
        'Halpha.y0': Uniform(-0.2, 0.2),
        'Halpha.dispersion': Uniform(20, 150),
        'Halpha.cont.flux': Uniform(0.1, 20),
        'Halpha.cont.rscale': Uniform(0.05, 1.0),
        'Halpha.cont.h_over_r': 0.15,
        'Halpha.cont.x0': Uniform(-0.2, 0.2),
        'Halpha.cont.y0': Uniform(-0.2, 0.2),
        # OIII5007: independent spatial + own dispersion; continuum spatial
        # shared with Halpha via continuum_key
        'OIII5007.flux': Uniform(0.5, 80),
        'OIII5007.rscale': Uniform(0.05, 1.0),
        'OIII5007.h_over_r': 0.15,
        'OIII5007.x0': Uniform(-0.2, 0.2),
        'OIII5007.y0': Uniform(-0.2, 0.2),
        'OIII5007.dispersion': Uniform(40, 300),  # NLR can be broader than Halpha
        'OIII5007.cont.flux': Uniform(0.1, 20),
        # NOTE: no OIII5007.cont.rscale/x0/y0 — shared from Halpha.cont via continuum_key
        'g1': Uniform(-0.1, 0.1),
        'g2': Uniform(-0.1, 0.1),
        'cosi': Uniform(0.1, 0.99),
        'theta_int': Uniform(0, jnp.pi),
        'z': 1.500,
    }
)


# ===========================================================================
# InferenceTask.from_obs (new unified factory) signature
# ===========================================================================
#
# class InferenceTask:
#     @classmethod
#     def from_obs(
#         cls,
#         source:       SourceModel,
#         priors:       PriorDict,
#         *,
#         image_obs:    Optional[Dict[str, ImageObs]] = None,
#         grism_obs:    Optional[Dict[str, GrismObs]] = None,
#         velocity_obs: Optional[VelocityObs] = None,
#         meta_pars:    Optional[Dict] = None,
#     ) -> 'InferenceTask':
#         """Build inference task from any combination of observation channels.
#
#         Validation:
#           - At least one obs must be provided.
#           - Each image_obs key must exist in source.broadband_models.
#           - grism_obs (non-empty Dict) requires source.emission_lines non-empty
#             AND source.velocity_model is set.
#           - velocity_obs requires source.velocity_model is set. If
#             velocity_obs.flux_weight_key is not None, it must reference a key
#             in source.emission_lines. flux_weight_key=None is allowed and means
#             unweighted velocity rendering (velocity-only inference path).
#         """
#         ...


# ===========================================================================
# Workflow A — grism-only, single roll
# ===========================================================================
#
# obs_grism = build_grism_obs(
#     grism_pars, z=1.0, data=..., variance=..., psf=...,
# )
# task = InferenceTask.from_obs(source_A, priors_A, grism_obs={'roll0': obs_grism})


# ===========================================================================
# Workflow B — production: F087 + Halpha+[NII] grism (single roll)
# ===========================================================================
#
# obs_f087  = build_image_obs(image_pars, data=..., variance=..., psf=...,
#                             broadband_key='F087')
# obs_grism = build_grism_obs(grism_pars, z=1.0, data=..., variance=..., psf=...)
# task = InferenceTask.from_obs(
#     source_B, priors_B,
#     image_obs = {'F087': obs_f087},
#     grism_obs = {'roll0': obs_grism},
# )


# ===========================================================================
# Workflow C — multi-band + grism
# ===========================================================================
#
# task = InferenceTask.from_obs(
#     source_B, priors_B,
#     image_obs = {'F087': obs_f087, 'F184': obs_f184},
#     grism_obs = {'roll0': obs_grism},
# )


# ===========================================================================
# Workflow D — broadband-only (no kinematics, no emission)
# ===========================================================================

source_D = SourceModel(broadband_models={'F087': InclinedExponentialModel()})
priors_D = PriorDict(
    {
        'F087.flux': Uniform(1, 200),
        'F087.rscale': Uniform(0.05, 1.0),
        'F087.h_over_r': 0.15,
        'F087.x0': Uniform(-0.2, 0.2),
        'F087.y0': Uniform(-0.2, 0.2),
        'g1': Uniform(-0.1, 0.1),
        'g2': Uniform(-0.1, 0.1),
        'cosi': Uniform(0.1, 0.99),
        'theta_int': Uniform(0, jnp.pi),
    }
)
# obs_f087 = build_image_obs(image_pars, data=..., variance=..., psf=...,
#                            broadband_key='F087')
# task = InferenceTask.from_obs(source_D, priors_D, image_obs={'F087': obs_f087})


# ===========================================================================
# Workflow D2 — broadband-only multi-band (new capability)
# ===========================================================================

source_D2 = SourceModel(
    broadband_models={
        'F087': InclinedExponentialModel(),
        'F184': InclinedExponentialModel(),
    }
)
# priors_D2 = PriorDict({...})  # like priors_D but with F087.* and F184.* both
# task = InferenceTask.from_obs(
#     source_D2, priors_D2,
#     image_obs = {'F087': obs_f087, 'F184': obs_f184},
# )


# ===========================================================================
# Workflow E — velocity-only (IFU / kinematics-only, no intensity context)
# ===========================================================================

source_E = SourceModel(velocity_model=OffsetVelocityModel())
priors_E = PriorDict(
    {
        'vel.vcirc': Uniform(80, 300),
        'vel.v0': Gaussian(0.0, 30.0),
        'vel.rscale': Uniform(0.05, 1.0),
        'vel.x0': Gaussian(0.0, 0.1),
        'vel.y0': Gaussian(0.0, 0.1),
        'g1': Uniform(-0.1, 0.1),
        'g2': Uniform(-0.1, 0.1),
        'cosi': Uniform(0.1, 0.99),
        'theta_int': Uniform(0, jnp.pi),
    }
)
# obs_velocity built with flux_weight_key=None → no flux weighting
# (mirrors today's from_velocity_obs path)
# obs_velocity = build_velocity_obs(image_pars, data=..., variance=...,
#                                   flux_weight_key=None)
# task = InferenceTask.from_obs(source_E, priors_E, velocity_obs=obs_velocity)


# ===========================================================================
# Workflow E2 — velocity + photometry (IFU + broadband)
# ===========================================================================
#
# task = InferenceTask.from_obs(
#     source_B, priors_B,
#     image_obs    = {'F087': obs_f087},
#     velocity_obs = obs_velocity,   # flux_weight_key='Halpha' for PSF flux weighting
# )


# ===========================================================================
# Workflow F — multi-roll grism (new capability)
#
# Two broadband + two grism observations at different telescope roll angles.
# The per-obs WCS PC matrix encodes the celestial-to-detector rotation;
# SourceModel rotates the celestial-frame (theta_int, g1, g2) into each
# obs's detector frame at render time. GrismPars.dispersion_angle_detector
# stays = 0 (Roman convention; dispersion is fixed along detector +x).
# ===========================================================================
#
# pixel_scale = 0.11  # arcsec, Roman WFI
# shape = (64, 64)
# wcs_roll0  = make_rolled_wcs(shape, pixel_scale, roll_radians=0.0)
# wcs_roll90 = make_rolled_wcs(shape, pixel_scale, roll_radians=np.pi/2)
#
# image_pars_f087 = ImagePars(shape, indexing='ij', wcs=wcs_roll0)
# image_pars_f184 = ImagePars(shape, indexing='ij', wcs=wcs_roll90)
# grism_pars_g0   = GrismPars(image_pars=ImagePars(shape, indexing='ij', wcs=wcs_roll0),
#                             dispersion=1.1, lambda_ref=1300.0,
#                             dispersion_angle_detector=0.0)
# grism_pars_g90  = GrismPars(image_pars=ImagePars(shape, indexing='ij', wcs=wcs_roll90),
#                             dispersion=1.1, lambda_ref=1300.0,
#                             dispersion_angle_detector=0.0)
#
# obs_f087 = build_image_obs(image_pars_f087, data=..., variance=..., psf=...,
#                            broadband_key='F087')
# obs_f184 = build_image_obs(image_pars_f184, data=..., variance=..., psf=...,
#                            broadband_key='F184')
# obs_g0   = build_grism_obs(grism_pars_g0,  z=1.0, data=..., variance=..., psf=...)
# obs_g90  = build_grism_obs(grism_pars_g90, z=1.0, data=..., variance=..., psf=...)
#
# task = InferenceTask.from_obs(
#     source_B, priors_B,
#     image_obs = {'F087': obs_f087, 'F184': obs_f184},
#     grism_obs = {'roll0': obs_g0, 'roll90': obs_g90},
# )


# ===========================================================================
# Workflow G — kitchen sink: 4-band + 3-roll grism + velocity
# ===========================================================================
#
# task = InferenceTask.from_obs(
#     source_C, priors_C,
#     image_obs    = {
#         'F087': obs_f087, 'F129': obs_f129,
#         'F158': obs_f158, 'F184': obs_f184,
#     },
#     grism_obs    = {
#         'roll0': obs_g0, 'roll90': obs_g90, 'roll180': obs_g180,
#     },
#     velocity_obs = obs_velocity,
# )
