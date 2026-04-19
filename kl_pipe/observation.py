"""
Observation types for kinematic lensing models.

Bundles instrument state (PSF, grids, oversampling) and optional data
into frozen containers that models accept for rendering and likelihood.

Three types:
- ImageObs: broadband / narrowband 2D imaging
- VelocityObs(ImageObs): velocity map with flux weighting for PSF
- GrismObs: dispersed spectroscopy (grism)

Factory functions replace the old Model.configure_psf() family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp
from scipy.fft import next_fast_len
import galsim

if TYPE_CHECKING:
    from kl_pipe.model import IntensityModel

from kl_pipe.parameters import ImagePars
from kl_pipe.psf import PSFData
from kl_pipe.utils import build_map_grid_from_image_pars


# ============================================================================
# Observation types
# ============================================================================


@dataclass(frozen=True)
class ImageObs:
    """2D imaging observation (broadband, narrowband, etc.).

    Parameters
    ----------
    image_pars : ImagePars
        Pixel grid metadata (shape, pixel_scale).
    X, Y : jnp.ndarray
        Pre-computed coarse-scale coordinate grids.
    psf_data : PSFData, optional
        Pre-computed PSF FFT for convolve_fft.
    oversample : int
        Source oversampling factor (1 = no oversampling).
    fine_X, fine_Y : jnp.ndarray, optional
        Fine-scale grids when oversample > 1.
    data : jnp.ndarray, optional
        Observed data (None = rendering-only, required for likelihood).
    variance : jnp.ndarray or float, optional
        Noise variance (same shape as data, or scalar).
    mask : jnp.ndarray, optional
        Boolean mask (True=valid). Same shape as data.
    kspace_psf_fft : jnp.ndarray, optional
        Fused k-space PSF kernel for InclinedExponentialModel.
    """

    image_pars: ImagePars
    X: jnp.ndarray
    Y: jnp.ndarray
    psf_data: Optional[PSFData] = None
    oversample: int = 1
    fine_X: Optional[jnp.ndarray] = None
    fine_Y: Optional[jnp.ndarray] = None
    data: Optional[jnp.ndarray] = None
    variance: Optional[jnp.ndarray] = None
    mask: Optional[jnp.ndarray] = None
    kspace_psf_fft: Optional[jnp.ndarray] = None


@dataclass(frozen=True)
class VelocityObs(ImageObs):
    """2D velocity observation with flux weighting for PSF convolution.

    Velocity PSF requires: v_obs = Conv(I*v, PSF) / Conv(I, PSF).
    Three modes for intensity source:

    - flux_model + flux_theta: evaluate intensity model with fixed params
    - flux_image: pre-rendered intensity map (upsampled to fine scale if needed)
    - flux_model + flux_theta_override at render time: joint inference
    """

    flux_model: Optional['IntensityModel'] = None
    flux_theta: Optional[jnp.ndarray] = None
    flux_image: Optional[jnp.ndarray] = None


@dataclass(frozen=True)
class GrismObs:
    """Grism observation — dispersed spectroscopy.

    Parameters
    ----------
    grism_pars : GrismPars
        Dispersion parameters.
    cube_pars : CubePars
        Pre-computed wavelength grid at concrete redshift.
    psf_data : PSFData, optional
        Pre-computed PSF FFT for per-slice convolution.
    oversample : int
        Spatial oversampling factor.
    fine_image_pars : ImagePars, optional
        Fine spatial grid (oversample > 1).
    data : jnp.ndarray, optional
        Observed grism data.
    variance : jnp.ndarray or float, optional
        Noise variance.
    mask : jnp.ndarray, optional
        Boolean mask (True=valid).
    """

    grism_pars: object  # GrismPars — avoid circular import
    cube_pars: object  # CubePars — avoid circular import
    psf_data: Optional[PSFData] = None
    oversample: int = 1
    fine_image_pars: Optional[ImagePars] = None
    data: Optional[jnp.ndarray] = None
    variance: Optional[jnp.ndarray] = None
    mask: Optional[jnp.ndarray] = None

@dataclass(frozen=True)
class FiberObs: #wip
    fiber_pars: object #cubepars are within fiberpars for now
    cube_pars: object
    psf_data: Optional[PSFData] = None
    oversample: int = 1
    fine_image_pars: Optional[ImagePars] = None
    data: Optional[jnp.ndarray] = None
    variance: Optional[jnp.ndarray] = None
    ATMPSF_conv_fiber_mask: Optional[jnp.ndarray] = None
    resolution_matrix: Optional[jnp.ndarray] = None


# ============================================================================
# JAX pytree registration
# ============================================================================


def _image_obs_flatten(obs):
    children = (
        obs.X,
        obs.Y,
        obs.psf_data,
        obs.fine_X,
        obs.fine_Y,
        obs.data,
        obs.variance,
        obs.mask,
        obs.kspace_psf_fft,
    )
    aux = (obs.image_pars, obs.oversample)
    return children, aux


def _image_obs_unflatten(aux, children):
    return ImageObs(
        image_pars=aux[0],
        oversample=aux[1],
        X=children[0],
        Y=children[1],
        psf_data=children[2],
        fine_X=children[3],
        fine_Y=children[4],
        data=children[5],
        variance=children[6],
        mask=children[7],
        kspace_psf_fft=children[8],
    )


jax.tree_util.register_pytree_node(ImageObs, _image_obs_flatten, _image_obs_unflatten)


def _velocity_obs_flatten(obs):
    children = (
        obs.X,
        obs.Y,
        obs.psf_data,
        obs.fine_X,
        obs.fine_Y,
        obs.data,
        obs.variance,
        obs.mask,
        obs.kspace_psf_fft,
        obs.flux_theta,
        obs.flux_image,
    )
    aux = (obs.image_pars, obs.oversample, obs.flux_model)
    return children, aux


def _velocity_obs_unflatten(aux, children):
    return VelocityObs(
        image_pars=aux[0],
        oversample=aux[1],
        X=children[0],
        Y=children[1],
        psf_data=children[2],
        fine_X=children[3],
        fine_Y=children[4],
        data=children[5],
        variance=children[6],
        mask=children[7],
        kspace_psf_fft=children[8],
        flux_model=aux[2],
        flux_theta=children[9],
        flux_image=children[10],
    )


jax.tree_util.register_pytree_node(
    VelocityObs, _velocity_obs_flatten, _velocity_obs_unflatten
)


def _grism_obs_flatten(obs):
    children = (obs.psf_data, obs.data, obs.variance, obs.mask)
    aux = (obs.grism_pars, obs.cube_pars, obs.oversample, obs.fine_image_pars)
    return children, aux


def _grism_obs_unflatten(aux, children):
    return GrismObs(
        grism_pars=aux[0],
        cube_pars=aux[1],
        oversample=aux[2],
        fine_image_pars=aux[3],
        psf_data=children[0],
        data=children[1],
        variance=children[2],
        mask=children[3],
    )


jax.tree_util.register_pytree_node(GrismObs, _grism_obs_flatten, _grism_obs_unflatten)


# ============================================================================
# Factory functions
# ============================================================================


def build_image_obs(
    image_pars: ImagePars,
    *,
    psf=None,
    oversample: int = 5,
    gsparams=None,
    data=None,
    variance=None,
    mask=None,
    int_model=None,
) -> ImageObs:
    """Build imaging observation. Replaces Model.configure_psf().

    Parameters
    ----------
    image_pars : ImagePars
        Pixel grid metadata.
    psf : galsim.GSObject, optional
        PSF profile. None = no PSF convolution.
    oversample : int
        Oversampling factor for source evaluation (positive odd int).
        Only used when psf is not None. Default 5.
    gsparams : galsim.GSParams, optional
        GalSim rendering parameters.
    data : jnp.ndarray, optional
        Observed data. None = rendering-only.
    variance : jnp.ndarray or float, optional
        Noise variance.
    mask : jnp.ndarray, optional
        Boolean mask (True=valid).
    int_model : InclinedExponentialModel, optional
        When provided and has _kspace_pad_factor, also pre-compute
        fused k-space PSF kernel for the InclinedExponentialModel path.
    """
    X, Y = build_map_grid_from_image_pars(image_pars)

    psf_data = None
    fine_X = None
    fine_Y = None
    kspace_psf_fft = None

    if psf is not None:
        from kl_pipe.psf import precompute_psf_fft

        psf_data = precompute_psf_fft(
            psf,
            image_pars=image_pars,
            oversample=oversample,
            gsparams=gsparams,
        )

        if oversample > 1:
            fine_image_pars = image_pars.make_fine_scale(oversample)
            fine_X, fine_Y = build_map_grid_from_image_pars(fine_image_pars)

        # fused k-space PSF kernel for InclinedExponentialModel
        if int_model is not None and hasattr(int_model, '_kspace_pad_factor'):
            from kl_pipe.psf import precompute_psf_kspace_fft

            N = max(oversample, 1)
            fine_Nrow = image_pars.Nrow * N
            fine_Ncol = image_pars.Ncol * N
            fine_ps = image_pars.pixel_scale / N
            pad_sq = next_fast_len(
                int_model._kspace_pad_factor * max(fine_Nrow, fine_Ncol)
            )
            kspace_psf_fft = precompute_psf_kspace_fft(
                psf, (pad_sq, pad_sq), fine_ps, gsparams=gsparams
            )
    else:
        oversample = 1

    if data is not None:
        data = jnp.asarray(data)
    if variance is not None:
        variance = jnp.asarray(variance)
    if mask is not None:
        mask = jnp.asarray(mask, dtype=bool)

    return ImageObs(
        image_pars=image_pars,
        X=X,
        Y=Y,
        psf_data=psf_data,
        oversample=oversample,
        fine_X=fine_X,
        fine_Y=fine_Y,
        data=data,
        variance=variance,
        mask=mask,
        kspace_psf_fft=kspace_psf_fft,
    )


def build_velocity_obs(
    image_pars: ImagePars,
    *,
    psf=None,
    oversample: int = 5,
    gsparams=None,
    data=None,
    variance=None,
    mask=None,
    flux_model=None,
    flux_theta=None,
    flux_image=None,
    flux_image_pars=None,
) -> VelocityObs:
    """Build velocity observation. Replaces VelocityModel.configure_velocity_psf().

    Parameters
    ----------
    image_pars : ImagePars
        Pixel grid metadata.
    psf : galsim.GSObject, optional
        PSF profile.
    oversample : int
        Oversampling factor. Default 5.
    gsparams : galsim.GSParams, optional
        GalSim rendering parameters.
    data : jnp.ndarray, optional
        Observed velocity data.
    variance : jnp.ndarray or float, optional
        Noise variance.
    mask : jnp.ndarray, optional
        Boolean mask.
    flux_model : IntensityModel, optional
        Intensity model for PSF flux weighting.
    flux_theta : jnp.ndarray, optional
        Fixed intensity params (used with flux_model).
    flux_image : ndarray, optional
        Pre-rendered intensity map for PSF flux weighting.
    flux_image_pars : ImagePars, optional
        Image parameters of flux_image (for resampling if shape differs).
    """
    X, Y = build_map_grid_from_image_pars(image_pars)

    psf_data = None
    fine_X = None
    fine_Y = None
    processed_flux_image = None

    if psf is not None:
        from kl_pipe.psf import precompute_psf_fft

        psf_data = precompute_psf_fft(
            psf,
            image_pars=image_pars,
            oversample=oversample,
            gsparams=gsparams,
        )

        if oversample > 1:
            fine_image_pars = image_pars.make_fine_scale(oversample)
            fine_X, fine_Y = build_map_grid_from_image_pars(fine_image_pars)

        if flux_model is None and flux_image is None:
            raise ValueError(
                "Velocity PSF requires flux weighting. Provide flux_model + "
                "flux_theta, or flux_image. For joint inference use build_joint_obs."
            )

        # process flux_image: resample + upsample if needed
        if flux_image is not None:
            target_shape = (image_pars.Nrow, image_pars.Ncol)

            if flux_image.shape != target_shape:
                if flux_image_pars is None:
                    raise ValueError(
                        f"flux_image shape {flux_image.shape} != velocity grid "
                        f"{target_shape}. Provide flux_image_pars for resampling."
                    )
                from kl_pipe.psf import _resample_to_grid

                flux_image = _resample_to_grid(
                    flux_image,
                    flux_image_pars,
                    target_shape=target_shape,
                    target_pixel_scale=image_pars.pixel_scale,
                )

            if oversample > 1:
                from kl_pipe.psf import _resample_to_grid

                coarse_pars = ImagePars(
                    shape=target_shape,
                    pixel_scale=image_pars.pixel_scale,
                    indexing='ij',
                )
                fine_shape = (
                    target_shape[0] * oversample,
                    target_shape[1] * oversample,
                )
                fine_ps = image_pars.pixel_scale / oversample
                flux_image = _resample_to_grid(
                    flux_image,
                    coarse_pars,
                    target_shape=fine_shape,
                    target_pixel_scale=fine_ps,
                )

            processed_flux_image = jnp.asarray(flux_image)
    else:
        oversample = 1

    if data is not None:
        data = jnp.asarray(data)
    if variance is not None:
        variance = jnp.asarray(variance)
    if mask is not None:
        mask = jnp.asarray(mask, dtype=bool)
    if flux_theta is not None:
        flux_theta = jnp.asarray(flux_theta)

    return VelocityObs(
        image_pars=image_pars,
        X=X,
        Y=Y,
        psf_data=psf_data,
        oversample=oversample,
        fine_X=fine_X,
        fine_Y=fine_Y,
        data=data,
        variance=variance,
        mask=mask,
        kspace_psf_fft=None,
        flux_model=flux_model,
        flux_theta=flux_theta,
        flux_image=processed_flux_image,
    )


def build_joint_obs(
    image_pars_vel: ImagePars,
    image_pars_int: ImagePars,
    intensity_model: 'IntensityModel',
    *,
    psf_vel=None,
    psf_int=None,
    oversample: int = 5,
    gsparams=None,
    data_vel=None,
    variance_vel=None,
    mask_vel=None,
    data_int=None,
    variance_int=None,
    mask_int=None,
) -> tuple:
    """Build paired velocity+intensity obs for joint inference.

    Velocity gets flux_model=intensity_model (joint mode: flux_theta
    provided at render time via flux_theta_override).

    Returns
    -------
    obs_vel : VelocityObs
    obs_int : ImageObs
    """
    # velocity obs: flux_model set for joint mode, no flux_theta/flux_image
    obs_vel = _build_velocity_obs_joint(
        image_pars_vel,
        psf=psf_vel,
        oversample=oversample,
        gsparams=gsparams,
        data=data_vel,
        variance=variance_vel,
        mask=mask_vel,
        flux_model=intensity_model,
    )

    obs_int = build_image_obs(
        image_pars_int,
        psf=psf_int,
        oversample=oversample,
        gsparams=gsparams,
        data=data_int,
        variance=variance_int,
        mask=mask_int,
        int_model=intensity_model,
    )

    return obs_vel, obs_int


def _build_velocity_obs_joint(
    image_pars,
    *,
    psf=None,
    oversample=5,
    gsparams=None,
    data=None,
    variance=None,
    mask=None,
    flux_model=None,
):
    """Build VelocityObs for joint mode (flux_model set, no flux_theta/flux_image)."""
    X, Y = build_map_grid_from_image_pars(image_pars)

    psf_data = None
    fine_X = None
    fine_Y = None

    if psf is not None:
        from kl_pipe.psf import precompute_psf_fft

        psf_data = precompute_psf_fft(
            psf,
            image_pars=image_pars,
            oversample=oversample,
            gsparams=gsparams,
        )

        if oversample > 1:
            fine_image_pars = image_pars.make_fine_scale(oversample)
            fine_X, fine_Y = build_map_grid_from_image_pars(fine_image_pars)
    else:
        oversample = 1

    if data is not None:
        data = jnp.asarray(data)
    if variance is not None:
        variance = jnp.asarray(variance)
    if mask is not None:
        mask = jnp.asarray(mask, dtype=bool)

    return VelocityObs(
        image_pars=image_pars,
        X=X,
        Y=Y,
        psf_data=psf_data,
        oversample=oversample,
        fine_X=fine_X,
        fine_Y=fine_Y,
        data=data,
        variance=variance,
        mask=mask,
        kspace_psf_fft=None,
        flux_model=flux_model,
        flux_theta=None,
        flux_image=None,
    )


def build_grism_obs(
    grism_pars,
    z: float,
    *,
    psf=None,
    oversample: int = 5,
    gsparams=None,
    data=None,
    variance=None,
    mask=None,
) -> GrismObs:
    """Build grism observation. Replaces KLModel.configure_grism_psf().

    Parameters
    ----------
    grism_pars : GrismPars
        Dispersion parameters.
    z : float
        Concrete redshift for pre-computing cube_pars.
    psf : galsim.GSObject, optional
        PSF profile for per-slice convolution.
    oversample : int
        Spatial oversampling factor. Default 5.
    gsparams : galsim.GSParams, optional
        GalSim rendering parameters.
    data : jnp.ndarray, optional
        Observed grism data.
    variance : jnp.ndarray or float, optional
        Noise variance.
    mask : jnp.ndarray, optional
        Boolean mask.
    """
    cube_pars = grism_pars.to_cube_pars(z)

    psf_data = None
    fine_image_pars = None

    if psf is not None:
        from kl_pipe.psf import precompute_psf_fft

        psf_data = precompute_psf_fft(
            psf,
            image_pars=cube_pars.image_pars,
            oversample=oversample,
            gsparams=gsparams,
        )

        if oversample > 1:
            fine_image_pars = cube_pars.image_pars.make_fine_scale(oversample)
    else:
        oversample = 1

    if data is not None:
        data = jnp.asarray(data)
    if variance is not None:
        variance = jnp.asarray(variance)
    if mask is not None:
        mask = jnp.asarray(mask, dtype=bool)

    return GrismObs(
        grism_pars=grism_pars,
        cube_pars=cube_pars,
        psf_data=psf_data,
        oversample=oversample,
        fine_image_pars=fine_image_pars,
        data=data,
        variance=variance,
        mask=mask,
    )

def get_fiber_mask(fiber_pars):
    from photutils.geometry import (
        circular_overlap_grid as cog,
    )  # is it alright for me to still use this?

    mNx, mNy = fiber_pars.spatial_shape[1], fiber_pars.spatial_shape[0]
    mscale = fiber_pars.pix_scale
    if fiber_pars.is_dispersed:
        fiber_cen = [
            fiber_pars.obs_conf['FIBERDX'],
            fiber_pars.obs_conf['FIBERDY'],
        ]  # dx, dy in arcsec
        fiber_rad = fiber_pars.obs_conf['FIBERRAD']  # radius in arcsec
        xmin, xmax = -mNx / 2 * mscale, mNx / 2 * mscale
        ymin, ymax = -mNy / 2 * mscale, mNy / 2 * mscale
        mask = cog(
            xmin - fiber_cen[0],
            xmax - fiber_cen[0],
            ymin - fiber_cen[1],
            ymax - fiber_cen[1],
            mNx,
            mNy,
            fiber_rad,
            1,
            2,
        )
    else:
        mask = jnp.ones([mNy, mNx])
    return mask

def precompute_PSF_convolved_fiber_mask(fiber_pars, galsim_psf):  # precompute fiber mask and make it a jax array
    '''get atm-PSF convolved fiber mask'''
    mNx, mNy = fiber_pars.spatial_shape[1], fiber_pars.spatial_shape[0]
    mscale = fiber_pars.pix_scale

    #galsim_psf = _build_PSF_model_fiber(
    #fiber_pars.obs_conf, lam_mean=fiber_pars.lambda_eff
    #)

    mask = galsim.InterpolatedImage(
    galsim.Image(array=get_fiber_mask(fiber_pars)), scale=mscale
    )

    # convolve fiber mask with atmospheric PSF
    maskC = mask if galsim_psf is None else galsim.Convolve([mask, galsim_psf])
    ary = maskC.drawImage(nx=mNx, ny=mNy, scale=mscale).array

    # replace galsim convolution?
    # fiber_psf_data = self.configure_fiber_psf(galsim_psf, fiber_pars.cube_pars)
    # if self._fiber_psf_data is not None:
    # from kl_pipe.psf import convolve_fft
    # oversample = self._fiber_psf_data.oversample
    # maskC = convolve_fft(self.get_fiber_mask(fiber_pars), self._fiber_psf_data) #mask needs to be 5x bigger in size if oversampling = 5
    ##maskC = convolve_fft(self.get_fiber_mask(fiber_pars), fiber_psf_data)
    # else:
    # maskC = self.get_fiber_mask(fiber_pars)
    # print('maskC', maskC)
    # ary=maskC

    ATMPSF_conv_fiber_mask = jnp.array(ary)
    return ATMPSF_conv_fiber_mask

def get_resolution_matrix_fiber(fiber_pars):
    from scipy.sparse import dia_matrix

    if fiber_pars.is_dispersed:
        diameter_in_pixel = fiber_pars.obs_conf['FIBRBLUR']
        sigma = diameter_in_pixel / 4.0
        x_in_pixel = jnp.arange(-5, 6)
        # assume Gaussian for now
        kernel = jnp.exp(-0.5 * (x_in_pixel / sigma) ** 2) / (
            (2 * jnp.pi) ** 0.5 * sigma
        )
        # get the resolution matrix (sparse matrix)
        band = jnp.array([kernel]).repeat(fiber_pars.n_lambda, axis=0).T
        offset = jnp.arange(kernel.shape[0] // 2, -(kernel.shape[0] // 2) - 1, -1)
        Rmat = dia_matrix(
            (band, offset), shape=(fiber_pars.n_lambda, fiber_pars.n_lambda)
        )
    else:
        Rmat = None
    resolution_mat = jnp.array(Rmat.toarray())  # need to figure out how to make jnp array of sparse matrix directly. but oh well, for now this
    return resolution_mat


#should I have a separate fiber obs for each spectrum? yes I think so
#I could also put in the fiber mask here, and the resolution matrix, and stuff like that...
#actually yes I should, because right now fiber mask is a part of KLModel and that's no good. I need a different fiber mask for each spectrum
def build_fiber_obs(
    fiber_pars, #z: float, not being used rn, *, #what is this
    psf=None, #galsim object
    oversample: int = 1,
    gsparams=None,
    data=None,
    variance=None,
    ATMPSF_conv_fiber_mask=None,
    resolution_matrix=None,
) -> FiberObs:
    cube_pars = fiber_pars.cube_pars

    psf_data = None
    fine_image_pars = None

    if psf is not None:
        from kl_pipe.psf import precompute_psf_fft

        #added by me
        #psf = _build_PSF_model_fiber(fiber_pars.obs_conf, lam_mean=fiber_pars.lambda_eff)

        psf_data = precompute_psf_fft(
            psf,
            image_pars=cube_pars.image_pars,
            oversample=oversample,
            gsparams=gsparams,
        )

        if oversample > 1:
            fine_image_pars = cube_pars.image_pars.make_fine_scale(oversample)
    else:
        oversample = 1

    if data is not None:
        data = jnp.asarray(data)
    if variance is not None:
        variance = jnp.asarray(variance)
    #if mask is not None: #for masking out part of the data I'm guessing

    if ATMPSF_conv_fiber_mask is None and fiber_pars.is_dispersed:
        ATMPSF_conv_fiber_mask = precompute_PSF_convolved_fiber_mask(fiber_pars, psf)
        #fiber_mask = get_fiber_mask(fiber_pars)
        ATMPSF_conv_fiber_mask = jnp.asarray(ATMPSF_conv_fiber_mask)#, dtype=bool)
    if resolution_matrix is None and fiber_pars.is_dispersed:
        resolution_matrix = get_resolution_matrix_fiber(fiber_pars)
        resolution_matrix = jnp.asarray(resolution_matrix)

    return FiberObs(
        fiber_pars=fiber_pars,
        cube_pars=cube_pars,
        psf_data=psf_data,
        oversample=oversample,
        fine_image_pars=fine_image_pars,
        data=data,
        variance=variance,
        ATMPSF_conv_fiber_mask=ATMPSF_conv_fiber_mask,
        resolution_matrix=resolution_matrix,
    )
