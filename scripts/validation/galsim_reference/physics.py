"""Independent (numpy-only, no ``kl_pipe`` import) physics for the GalSim
chromatic grism reference render.

Conventions confirmed by READING (not importing) ``kl_pipe`` source, so the
comparison is meaningful pixel-for-pixel:
  - ``kl_pipe/transformation.py``: obs->cen->source->gal->disk plane chain;
    source->gal rotates by -theta_int; gal->disk divides y by cosi.
  - ``kl_pipe/velocity.py`` + ``kl_pipe/model.py`` (``VelocityModel.__call__``):
    v_circ(r) = (2/pi)*vcirc*arctan(r/rscale); v_los = v0 +
    sqrt(1-cosi^2)*cos(phi_disk)*v_circ(r_disk), evaluated at the FULL 2D
    deprojected disk position (no LOS averaging -- thin-disk kinematics).
  - ``kl_pipe/intensity.py`` (``InclinedExponentialModel.__call__``):
    3D LOS integral of rho(R,z) = rho0*exp(-R/rs)*sech^2(z/hz), rho0 =
    flux/(4*pi*hz*rs^2), integrated in the GALAXY frame (NOT deprojected to
    disk) via the inclination-angle rotation between (y_gal, ell) [sky
    minor-axis coord, LOS depth] and (y_disk3d, z_disk) [disk-plane
    radial-minor coord, height above disk]: y_disk3d = y_gal*cosi +
    ell*sini, z_disk = ell*cosi - y_gal*sini, x_disk3d = x_gal.
  - ``kl_pipe/utils.py`` (``build_map_grid_from_image_pars``): X=cols
    (horizontal), Y=rows (vertical), array[row,col], centered coord =
    (idx - (N-1)/2)*ps. Matches GalSim's own drawImage array/origin='lower'
    convention (no transpose needed).
  - ``kl_pipe/dispersion.py`` (``disperse_cube``): pixel_offset(lambda) =
    (lambda - lambda_ref)/dispersion; pull-semantics shift is +offset along
    +x (cols) for dispersion_angle_detector=0 -- i.e. redder wavelengths
    disperse to larger x. Matches the GalSim ``.shift(dx=+offset*pixel_scale)``
    convention used here (shifting the *object* by +dx moves its flux to
    +x in the final image, same sign).

This module implements the LOS integrals via a SIMPLE DENSE quadrature
(not the tanh-Gauss-Legendre substitution ``kl_pipe`` uses) so the two
numerical methods are independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C_KMS = 299792.458  # km/s, matches kl_pipe.constants.C_KMS (astropy CODATA)


@dataclass
class GalaxyParams:
    """Truth parameters for the reference scene (Halpha line only)."""

    cosi: float
    theta_int: float  # radians
    flux: float  # Halpha line flux
    rscale: float  # intensity exponential scale length (arcsec)
    h_over_r: float  # h_z = h_over_r * rscale
    v0: float  # km/s
    vcirc: float  # km/s
    vel_rscale: float  # arcsec, velocity arctan scale radius
    sigma_v: float  # km/s, intrinsic Gaussian velocity dispersion
    z: float  # redshift
    lambda_rest: float = 656.28  # nm, Halpha vacuum rest wavelength


def centered_coords_1d(n: int) -> np.ndarray:
    """1D pixel-index-centered coordinate array (arb. units), matching
    ``kl_pipe.utils._centered_coords``: arange(N) - (N-1)/2.
    """
    return np.arange(n) - (n - 1) / 2.0


def build_grid(nrow: int, ncol: int, pixel_scale: float):
    """(X, Y) arcsec grids, shape (nrow, ncol), matching
    ``kl_pipe.utils.build_map_grid_from_image_pars(unit='arcsec', centered=True)``:
    X varies along columns (axis 1), Y varies along rows (axis 0).
    """
    x = centered_coords_1d(ncol) * pixel_scale
    y = centered_coords_1d(nrow) * pixel_scale
    X, Y = np.meshgrid(x, y, indexing='xy')  # X.shape = Y.shape = (nrow, ncol)
    return X, Y


def sky_to_gal(x: np.ndarray, y: np.ndarray, theta_int: float):
    """source(=sky, since g1=g2=x0=y0=0) -> gal plane: rotate by -theta_int.

    Matches ``kl_pipe.transformation.source2gal``: c=cos(-theta_int),
    s=sin(-theta_int), [[c,-s],[s,c]] applied to (x,y).
    """
    c = np.cos(theta_int)
    s = np.sin(theta_int)
    x_gal = c * x + s * y
    y_gal = -s * x + c * y
    return x_gal, y_gal


def velocity_los_map(X: np.ndarray, Y: np.ndarray, p: GalaxyParams) -> np.ndarray:
    """LOS velocity field v_los(x,y) [km/s], thin-disk kinematics (no LOS
    averaging): full deprojection to the 2D disk plane, then
    v_los = v0 + sin(i)*cos(phi_disk)*v_circ(r_disk).
    """
    x_gal, y_gal = sky_to_gal(X, Y, p.theta_int)
    cosi = p.cosi
    sini = np.sqrt(max(1.0 - cosi**2, 0.0))
    x_disk = x_gal
    y_disk = y_gal / cosi
    r_disk = np.sqrt(x_disk**2 + y_disk**2)
    phi_disk = np.arctan2(y_disk, x_disk)
    v_circ = (2.0 / np.pi) * p.vcirc * np.arctan(r_disk / p.vel_rscale)
    return p.v0 + sini * np.cos(phi_disk) * v_circ


def intensity_map(
    X: np.ndarray, Y: np.ndarray, p: GalaxyParams, n_los: int = 4001, n_hz: float = 12.0
) -> np.ndarray:
    """3D inclined-exponential-disk surface brightness I(x,y) [flux/arcsec^2],
    via dense uniform-grid trapezoidal LOS quadrature (independent numerical
    method from ``kl_pipe``'s tanh-Gauss-Legendre substitution).

    Integration variable: ell (LOS depth in the galaxy frame, arcsec, same
    angular-unit convention ``kl_pipe`` uses -- NOT a physical proper
    distance). The sech^2(z_disk/hz) profile peaks where z_disk=0, i.e. at
    ell = ell_center(y_gal) = y_gal*sini/cosi (NOT at ell=0 -- z_disk =
    ell*cosi - y_gal*sini mixes both, so the quadrature window must be
    recentered per-pixel on ell_center or off-axis pixels miss the sech^2
    peak entirely). Window half-width: n_hz * h_z / cosi (>>1 vertical
    scale heights; sech^2 decays as exp(-2|z|/hz) so n_hz=12 gives a
    truncation floor of exp(-24) ~ 4e-11, negligible vs. target
    sub-percent accuracy).
    """
    cosi = p.cosi
    sini = np.sqrt(max(1.0 - cosi**2, 0.0))
    h_z = p.h_over_r * p.rscale
    rho0 = p.flux / (4.0 * np.pi * h_z * p.rscale**2)

    x_gal, y_gal = sky_to_gal(X, Y, p.theta_int)

    ell_half_width = n_hz * h_z / max(cosi, 1e-3)
    ell_center = y_gal * sini / cosi  # (nrow, ncol)
    u = np.linspace(-1.0, 1.0, n_los)  # (n_los,) normalized offset
    du = u[1] - u[0]
    ell = ell_center[..., None] + ell_half_width * u[None, None, :]

    # broadcast: (nrow, ncol, 1) vs (nrow, ncol, n_los) -> (nrow, ncol, n_los)
    y_disk3d = y_gal[..., None] * cosi + ell * sini
    z_disk = ell * cosi - y_gal[..., None] * sini
    x_disk3d = x_gal[..., None]

    R = np.sqrt(x_disk3d**2 + y_disk3d**2)
    radial = np.exp(-R / p.rscale)
    vertical = 1.0 / np.cosh(z_disk / h_z) ** 2
    integrand = rho0 * radial * vertical

    # trapezoidal rule along the LOS axis (dell = ell_half_width * du)
    I = np.trapz(integrand, dx=1.0, axis=-1) * du * ell_half_width
    return I


def assign_velocity_bins(
    v_map: np.ndarray, n_v: int, v_margin_sigma: float, sigma_v: float
):
    """Partition v_map into n_v equal-width isovelocity bins.

    Bin range = [v_map.min(), v_map.max()] padded by
    v_margin_sigma * sigma_v on each side (keeps bin edges away from the
    map extrema so no galaxy flux sits exactly at a bin boundary due to
    floating-point ties).

    Returns
    -------
    bin_index : ndarray[int], shape v_map.shape
        Bin index (0..n_v-1) of each pixel.
    v_centers : ndarray, shape (n_v,)
        Bin-center velocity [km/s].
    edges : ndarray, shape (n_v+1,)
        Bin edge velocities [km/s].
    """
    vmin = v_map.min() - v_margin_sigma * sigma_v
    vmax = v_map.max() + v_margin_sigma * sigma_v
    edges = np.linspace(vmin, vmax, n_v + 1)
    bin_index = np.clip(np.digitize(v_map, edges[1:-1]), 0, n_v - 1)
    v_centers = 0.5 * (edges[:-1] + edges[1:])
    return bin_index, v_centers, edges
