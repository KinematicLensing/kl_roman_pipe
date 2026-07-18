#I should keep fiber-specific operations here, after the cube. Don't keep things in spectral.py
#I think I should also keep FiberPars here for now. I don't want to put it in dispersion.py
#photometric imaging needs to be kept separate

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, Optional

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from kl_pipe.spectral import CubePars

from kl_pipe.parameters import ImagePars

####WIP
@dataclass(frozen=True)
class FiberPars:
    #cube_pars: CubePars  # instead of taking cube_pars, I should have this take image_pars and then I build cube_pars myself
    image_pars: ImagePars
    obs_conf: dict  #not using. Would contain mirror diameter, exptime, gain...
    lambda_ref: float  # reference wavelength nm (zero-offset point) ???
    delta_lambda: float
    fiber_radius: float
    fiber_blur: float
    fiber_dx: float
    fiber_dy: float

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
    obs_conf: dict,
    fiber_radius: float,
    fiber_blur: float,
    fiber_dx: float,
    fiber_dy: float,
    image_pars: ImagePars = None,
    pixel_scale: float = 0.11,
    Nrow: int = 32,
    Ncol: int = 32,
) -> FiberPars:
    """Convenience factory for fiber spectrum centered on a specific line."""
    if image_pars is None:
        image_pars = ImagePars(
            shape=(Nrow, Ncol), pixel_scale=pixel_scale, indexing='ij'
        )
    lambda_ref = lambda_rest * (1.0 + redshift)
    return FiberPars(
        image_pars=image_pars,
        obs_conf=obs_conf,
        lambda_ref=lambda_ref,
        delta_lambda = delta_lambda,
        fiber_radius = fiber_radius,
        fiber_blur = fiber_blur,
        fiber_dx = fiber_dx,
        fiber_dy = fiber_dy,
    )
