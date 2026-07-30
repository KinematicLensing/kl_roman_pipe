from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, Optional
import galsim as gs

import jax.numpy as jnp
import numpy as np
#import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from kl_pipe.spectral import CubePars

from kl_pipe.parameters import ImagePars

@dataclass(frozen=True)
class FiberPars:
    image_pars: ImagePars
    #obs_conf: dict  #not using. Would contain mirror diameter, exptime, gain...
    lambda_ref: float  # reference wavelength nm (zero-offset point) ???
    delta_lambda: float
    fiber_radius: float
    fiber_blur: float
    fiber_dx: float
    fiber_dy: float
    bandpass_path: Optional[str] = None #if throughput is not specified
    throughput: Optional[jnp.ndarray] = None  # T(lambda), shape (Nlambda,)

    def to_cube_pars( 
        self,
        z: float,
        velocity_window_kms: float = 3000.0,
        #delta_lambda: float = 0.1,
        n_lambda: int = None,
        line_lambdas_rest: tuple = None,
    ) -> 'CubePars':
        """Build CubePars centered on the emission line complex at redshift z.

        Parameters
        ----------
        z : float
            Galaxy redshift.
        velocity_window_kms : float
            Half-width of velocity window in km/s. Default 3000.
        n_lambda : int, optional
            Number of wavelength pixels. If None, computed from velocity window
            and dispersion.
        line_lambdas_rest : tuple of float, optional
            Rest-frame wavelengths (nm) of lines to cover. If None, uses
            H-alpha (656.28 nm).
        """
        from kl_pipe.spectral import CubePars

        if line_lambdas_rest is None:
            line_lambdas_rest = (656.28,)

        # observed wavelength range covering all lines + velocity window
        lam_obs = [(lam * (1.0 + z)) for lam in line_lambdas_rest]
        lam_min_line = min(lam_obs)
        lam_max_line = max(lam_obs)

        # velocity window in wavelength units
        c_kms = 299792.458
        lam_center = 0.5 * (lam_min_line + lam_max_line)
        dlam_vel = lam_center * velocity_window_kms / c_kms

        lam_min = lam_min_line - dlam_vel
        lam_max = lam_max_line + dlam_vel

        if n_lambda is None:
            n_lambda = int(np.ceil((lam_max - lam_min) / self.delta_lambda)) + 1
            n_lambda = max(n_lambda, 3)

        lambda_grid = jnp.linspace(lam_min, lam_max, n_lambda)
        return CubePars(image_pars=self.image_pars, lambda_grid=lambda_grid)
    
    @property
    def output_shape(self) -> Tuple[int, int]:
        """Output shape = same as input spatial grid (source cutout)."""
        return (self.image_pars.Nrow, self.image_pars.Ncol)
    
#without lambda_ref this seems kinda useless. what even is lambda_ref?
def build_fiber_pars_for_line(
    lambda_rest: float,
    redshift: float,
    delta_lambda: float,
    #obs_conf: dict,
    fiber_radius: float,
    fiber_blur: float,
    fiber_dx: float,
    fiber_dy: float,
    bandpass_path: Optional[str] = None,
    throughput: Optional[jnp.ndarray] = None,  # T(lambda), shape (Nlambda,)
    image_pars: ImagePars = None,
    pixel_scale: float = 0.262,
    Nrow: int = 32,
    Ncol: int = 32,
) -> FiberPars:
    """Convenience factory for fiber spectrum centered on a specific line."""
    if image_pars is None:
        image_pars = ImagePars(
            shape=(Nrow, Ncol), pixel_scale=pixel_scale, indexing='ij'
        )

    if bandpass_path is not None and throughput is not None:
        print("You can specify either a file (bandpass_path) or an array for the throughput, but not both")

    lambda_ref = lambda_rest * (1.0 + redshift)
    return FiberPars(
        image_pars=image_pars,
        #obs_conf=obs_conf,
        lambda_ref=lambda_ref,
        delta_lambda = delta_lambda,
        fiber_radius = fiber_radius,
        fiber_blur = fiber_blur,
        fiber_dx = fiber_dx,
        fiber_dy = fiber_dy,
        bandpass_path = bandpass_path,
        throughput = throughput
    )

#maybe put these in the fiber.py file
def get_fiber_mask(image_pars, fiber_pars):
    from photutils.geometry import (
        circular_overlap_grid as cog,
    )

    #print(Nrow_f, Ncol_f)
    #spatial_shape = fiber_pars.output_shape
    #mNx, mNy = spatial_shape[1], spatial_shape[0]

    mNx, mNy = image_pars.Nrow, image_pars.Ncol
    mscale = image_pars.pixel_scale #is this in arcsec/pixel?
    fiber_cen = [
        fiber_pars.fiber_dx,#fiber_pars.obs_conf['FIBERDX'],
        fiber_pars.fiber_dy,#fiber_pars.obs_conf['FIBERDY'],
    ]  # dx, dy in arcsec
    fiber_rad = fiber_pars.fiber_radius #fiber_pars.obs_conf['FIBERRAD']  # radius in arcsec
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
    return mask

def precompute_PSF_convolved_fiber_mask(image_pars, fiber_pars, galsim_psf):
    '''get atm-PSF convolved fiber mask'''
    mNx, mNy = image_pars.Nrow, image_pars.Ncol
    mscale = image_pars.pixel_scale

    mask = gs.InterpolatedImage(
    gs.Image(array=get_fiber_mask(image_pars, fiber_pars)), scale=mscale
    )

    # convolve fiber mask with atmospheric PSF
    maskC = mask if galsim_psf is None else gs.Convolve([mask, galsim_psf])
    ary = maskC.drawImage(nx=mNx, ny=mNy, scale=mscale).array

    # replace galsim convolution?
    # fiber_psf_data = self.configure_fiber_psf(galsim_psf, fiber_pars.cube_pars)
    # if self._fiber_psf_data is not None:
    # from kl_pipe.psf import convolve_fft
    # oversample = self._fiber_psf_data.oversample
    # maskC = convolve_fft(self.get_fiber_mask(fiber_pars), self._fiber_psf_data) #mask needs to be 5x bigger in size if oversampling = 5
    # else:
    # maskC = self.get_fiber_mask(fiber_pars)
    # print('maskC', maskC)
    # ary=maskC

    ATMPSF_conv_fiber_mask = jnp.array(ary)
    return ATMPSF_conv_fiber_mask

def get_resolution_matrix_fiber(fiber_pars, cube_pars):
    from scipy.sparse import dia_matrix

    diameter_in_pixel = fiber_pars.fiber_blur #fiber_pars.obs_conf['FIBRBLUR']
    sigma = diameter_in_pixel / 4.0
    x_in_pixel = jnp.arange(-5, 6)
    # assume Gaussian for now
    kernel = jnp.exp(-0.5 * (x_in_pixel / sigma) ** 2) / (
        (2 * jnp.pi) ** 0.5 * sigma
    )
    # get the resolution matrix (sparse matrix)
    band = jnp.array([kernel]).repeat(cube_pars.n_lambda, axis=0).T
    offset = jnp.arange(kernel.shape[0] // 2, -(kernel.shape[0] // 2) - 1, -1)
    Rmat = dia_matrix(
        (band, offset), shape=(cube_pars.n_lambda, cube_pars.n_lambda)
    )
    resolution_mat = jnp.array(Rmat.toarray())  # need to figure out how to make jnp array of sparse matrix directly. but oh well, for now this
    return resolution_mat