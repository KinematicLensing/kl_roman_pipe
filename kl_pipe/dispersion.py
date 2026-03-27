"""
Grism dispersion: project 3D datacube onto 2D dispersed image.

Uses pull-semantics map_coordinates for sub-pixel shifts.
Fully differentiable via JAX bilinear interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, Optional

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from kl_pipe.spectral import CubePars

from kl_pipe.parameters import ImagePars


@dataclass(frozen=True)
class GrismPars:
    """Defines grism observation parameters for dispersing a datacube.

    dispersion_angle is in the obs frame (same as theta_int).
    throughput is optional; None = flat 100% (OK for narrow windows).
    """

    image_pars: ImagePars  # spatial grid of source cutout
    dispersion: float  # nm/pixel (~1.1 for Roman)
    lambda_ref: float  # reference wavelength nm (zero-offset point)
    dispersion_angle: float  # radians, grism direction in obs frame
    throughput: Optional[jnp.ndarray] = None  # T(lambda), shape (Nlambda,)

    def __post_init__(self):
        if self.dispersion <= 0:
            raise ValueError(f"dispersion must be > 0, got {self.dispersion}")

    def to_cube_pars(
        self,
        z: float,
        velocity_window_kms: float = 3000.0,
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
            n_lambda = int(np.ceil((lam_max - lam_min) / self.dispersion)) + 1
            n_lambda = max(n_lambda, 3)

        lambda_grid = jnp.linspace(lam_min, lam_max, n_lambda)
        return CubePars(image_pars=self.image_pars, lambda_grid=lambda_grid)

    @property
    def output_shape(self) -> Tuple[int, int]:
        """Output shape = same as input spatial grid (source cutout)."""
        return (self.image_pars.Nrow, self.image_pars.Ncol)


def disperse_cube(
    cube: jnp.ndarray,
    grism_pars: GrismPars,
    lambda_grid: jnp.ndarray,
) -> jnp.ndarray:
    """Project 3D datacube onto 2D dispersed grism image.

    Uses pull-semantics map_coordinates: [Y - dy, X - dx] shifts content
    by (+dy, +dx). Fully differentiable via JAX bilinear interpolation.

    Parameters
    ----------
    cube : jnp.ndarray
        PSF-convolved datacube, shape (Nrow, Ncol, Nlambda).
    grism_pars : GrismPars
        Grism parameters.
    lambda_grid : jnp.ndarray
        Wavelength array nm, shape (Nlambda,).

    Returns
    -------
    jnp.ndarray
        Dispersed 2D image, shape (Nrow, Ncol).
    """
    Nrow, Ncol, Nlam = cube.shape
    angle = grism_pars.dispersion_angle

    # pixel offsets for each wavelength slice relative to reference
    pixel_offsets = (lambda_grid - grism_pars.lambda_ref) / grism_pars.dispersion

    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)

    # base pixel coordinates
    rows = jnp.arange(Nrow, dtype=jnp.float64)
    cols = jnp.arange(Ncol, dtype=jnp.float64)
    Y_base, X_base = jnp.meshgrid(rows, cols, indexing='ij')

    # throughput
    throughput = grism_pars.throughput
    if throughput is None:
        throughput = jnp.ones(Nlam)

    # delta_lambda for integration (nm per wavelength pixel)
    if Nlam >= 2:
        dlam = jnp.abs(lambda_grid[1] - lambda_grid[0])
    else:
        dlam = 1.0

    # accumulate dispersed image
    dispersed = jnp.zeros((Nrow, Ncol))

    for k in range(Nlam):
        offset_k = pixel_offsets[k]
        dx_k = offset_k * cos_a  # shift along x (cols)
        dy_k = offset_k * sin_a  # shift along y (rows)

        # pull semantics: sample source at (Y - dy, X - dx)
        coords = jnp.array([Y_base - dy_k, X_base - dx_k])

        shifted = jax.scipy.ndimage.map_coordinates(
            cube[:, :, k], coords, order=1, mode='constant', cval=0.0
        )

        dispersed = dispersed + shifted * throughput[k] * dlam

    return dispersed


def build_grism_pars_for_line(
    lambda_rest: float,
    redshift: float,
    image_pars: ImagePars = None,
    pixel_scale: float = 0.11,
    Nrow: int = 32,
    Ncol: int = 32,
    dispersion: float = 1.1,
    dispersion_angle: float = 0.0,
) -> GrismPars:
    """Convenience factory for Roman grism centered on a specific line."""
    if image_pars is None:
        image_pars = ImagePars(
            shape=(Nrow, Ncol), pixel_scale=pixel_scale, indexing='ij'
        )
    lambda_ref = lambda_rest * (1.0 + redshift)
    return GrismPars(
        image_pars=image_pars,
        dispersion=dispersion,
        lambda_ref=lambda_ref,
        dispersion_angle=dispersion_angle,
    )


# need jax import for map_coordinates
import jax
