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

if TYPE_CHECKING:
    from kl_pipe.model import IntensityModel

from kl_pipe.parameters import ImagePars
from kl_pipe.pixel import BoxPixel, PixelResponse, _PIXEL_RESPONSE_UNSET
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
    pixel_response : PixelResponse, optional
        Pixel response function for k-space intensity rendering.
        Default BoxPixel is created by build_image_obs. None disables
        pixel integration (for testing or point-sampled comparisons).
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
    pixel_response: Optional[PixelResponse] = None


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
        obs.pixel_response,
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
        pixel_response=children[9],
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
    pixel_response=_PIXEL_RESPONSE_UNSET,
    render_config=None,
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
        Used for velocity models (spatial oversampling) and as legacy
        anti-aliasing for k-space models. For k-space intensity models,
        pixel integration is handled by ``pixel_response`` in k-space;
        most users should rely on adaptive grid sizing via
        ``folding_threshold`` rather than manual ``oversample``.
        Default 5.
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
    pixel_response : PixelResponse or None, optional
        Pixel response function for k-space rendering. Default (sentinel):
        auto-construct ``BoxPixel(image_pars.pixel_scale)``. Pass
        ``pixel_response=None`` explicitly to disable pixel integration
        (for testing or point-sampled comparisons).
    render_config : RenderConfig, optional
        When provided, ``render_config.oversample`` takes precedence over
        the bare ``oversample`` parameter for PSF FFT sizing and fine-grid
        construction.
    """
    # render_config.oversample takes precedence when provided
    if render_config is not None:
        oversample = render_config.oversample

    X, Y = build_map_grid_from_image_pars(image_pars)

    # pixel response: default to BoxPixel from pixel_scale
    if pixel_response is _PIXEL_RESPONSE_UNSET:
        pixel_response = BoxPixel(image_pars.pixel_scale)

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

        # fused k-space PSF kernel for k-space intensity models
        if int_model is not None and hasattr(int_model, '_kspace_pad_factor'):
            from kl_pipe.psf import precompute_psf_kspace_fft

            N = max(oversample, 1)
            fine_ps = image_pars.pixel_scale / N
            # for wrap-compatible grids: compute base pad first, then
            # multiply by oversample so the fused PSF grid is an exact
            # multiple of the base grid (required by _wrap_kspace)
            base_pad_sq = next_fast_len(
                int_model._kspace_pad_factor * max(image_pars.Nrow, image_pars.Ncol)
            )
            pad_sq = base_pad_sq * N
            kspace_psf_fft = precompute_psf_kspace_fft(
                psf, (pad_sq, pad_sq), fine_ps, gsparams=gsparams
            )

    # fine grids: create when oversample > 1, regardless of PSF.
    # needed for velocity models (spatial oversampling) even without PSF.
    if oversample > 1:
        fine_image_pars = image_pars.make_fine_scale(oversample)
        fine_X, fine_Y = build_map_grid_from_image_pars(fine_image_pars)

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
        pixel_response=pixel_response,
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

    # fine grids: create when oversample > 1, regardless of PSF
    if oversample > 1:
        fine_image_pars = image_pars.make_fine_scale(oversample)
        fine_X, fine_Y = build_map_grid_from_image_pars(fine_image_pars)

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
    pixel_response=_PIXEL_RESPONSE_UNSET,
) -> tuple:
    """Build paired velocity+intensity obs for joint inference.

    Velocity gets flux_model=intensity_model (joint mode: flux_theta
    provided at render time via flux_theta_override).

    Parameters
    ----------
    pixel_response : PixelResponse or None, optional
        Passed through to build_image_obs for the intensity obs.
        Default (sentinel): auto-construct BoxPixel. Pass None to disable.

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
        pixel_response=pixel_response,
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

    # fine grids: create when oversample > 1, regardless of PSF
    if oversample > 1:
        fine_image_pars = image_pars.make_fine_scale(oversample)
        fine_X, fine_Y = build_map_grid_from_image_pars(fine_image_pars)

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

    # fine grid: create when oversample > 1, regardless of PSF
    if oversample > 1:
        fine_image_pars = cube_pars.image_pars.make_fine_scale(oversample)

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
