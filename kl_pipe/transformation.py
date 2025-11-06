'''
This file contains transformation functions. These
are all static functions so that numba can be used
efficiently.

Definition of each plane:

    disk: Face-on view of the galactic disk, no inclination angle.
          This will be cylindrically symmetric for most models

    gal:  Galaxy major/minor axis frame with inclination angle same as
          source plane. Will now be ~ellipsoidal for sini!=0

    source: View from the lensing source plane, rotated version of gal
            plane with theta = theta_intrinsic

    cen: View from the object-centered observed plane. Sheared version of
         source plane

    obs:  Observed image plane. Offset version of cen plane
'''

# transform.py - JAX-friendly coordinate transformations
import jax.numpy as jnp
from typing import Tuple


SUPPORTED_PLANES = ['disk', 'gal', 'source', 'cen', 'obs']


def _multiply(transform: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray):
    """
    Apply 2x2 transformation matrix to coordinate arrays.

    Parameters
    ----------
    transform : jnp.ndarray
        2x2 transformation matrix.
    x, y : jnp.ndarray
        Coordinate arrays (must have same shape).

    Returns
    -------
    xp, yp : jnp.ndarray
        Transformed coordinates.
    """
    coords = jnp.stack([x.ravel(), y.ravel()], axis=0)
    transformed = jnp.matmul(transform, coords)

    xp = transformed[0].reshape(x.shape)
    yp = transformed[1].reshape(y.shape)

    return xp, yp


def obs2cen(
    x0: float, y0: float, x: jnp.ndarray, y: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Transform from obs to cen plane (remove offset).

    Parameters
    ----------
    x0, y0 : float
        Centroid offsets.
    x, y : jnp.ndarray
        Coordinates in obs plane.

    Returns
    -------
    xp, yp : jnp.ndarray
        Coordinates in cen plane.
    """
    return x - x0, y - y0


def cen2source(
    g1: float, g2: float, x: jnp.ndarray, y: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Transform from cen to source plane (inverse lensing shear).

    Parameters
    ----------
    g1, g2 : float
        Shear components.
    x, y : jnp.ndarray
        Coordinates in cen plane.

    Returns
    -------
    xp, yp : jnp.ndarray
        Coordinates in source plane.
    """

    norm = 1.0 / (1.0 - (g1**2 + g2**2))
    transform = norm * jnp.array([[1.0 + g1, g2], [g2, 1.0 - g1]])

    return _multiply(transform, x, y)


def source2gal(
    theta_int: float, x: jnp.ndarray, y: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Transform from source to gal plane (remove intrinsic rotation).

    Parameters
    ----------
    theta_int : float
        Intrinsic position angle.
    x, y : jnp.ndarray
        Coordinates in source plane.

    Returns
    -------
    xp, yp : jnp.ndarray
        Coordinates in gal plane.
    """

    c = jnp.cos(-theta_int)
    s = jnp.sin(-theta_int)
    transform = jnp.array([[c, -s], [s, c]])

    return _multiply(transform, x, y)


def gal2disk(
    sini: float, x: jnp.ndarray, y: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Transform from gal to disk plane (remove inclination).

    Parameters
    ----------
    sini : float
        Sine of inclination angle.
    x, y : jnp.ndarray
        Coordinates in gal plane.

    Returns
    -------
    xp, yp : jnp.ndarray
        Coordinates in disk plane.
    """

    cosi = jnp.sqrt(1.0 - sini**2)
    transform = jnp.array([[1.0, 0.0], [0.0, 1.0 / cosi]])

    return _multiply(transform, x, y)


def transform_to_disk_plane(
    x: jnp.ndarray,
    y: jnp.ndarray,
    plane: str,
    x0: float,
    y0: float,
    g1: float,
    g2: float,
    theta_int: float,
    sini: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Transform coordinates from specified plane to disk plane.

    The disk plane is the face-on, intrinsic coordinate system where
    models are naturally evaluated.

    Parameters
    ----------
    x, y : jnp.ndarray
        Coordinates in the starting plane.
    plane : str
        Starting plane name ('obs', 'cen', 'source', 'gal', or 'disk').
    x0, y0 : float
        Centroid offsets (obs plane).
    g1, g2 : float
        Lensing shear components.
    theta_int : float
        Intrinsic position angle (radians).
    sini : float
        Sine of inclination angle.

    Returns
    -------
    x_disk, y_disk : jnp.ndarray
        Coordinates in the disk plane.
    """

    if plane not in SUPPORTED_PLANES:
        raise ValueError(
            f"Plane '{plane}' not supported. Must be one of {SUPPORTED_PLANES}"
        )

    if plane == 'disk':
        return x, y

    xp, yp = x, y

    if plane == 'obs':
        xp, yp = obs2cen(x0, y0, xp, yp)
        plane = 'cen'

    if plane == 'cen':
        xp, yp = cen2source(g1, g2, xp, yp)
        plane = 'source'

    if plane == 'source':
        xp, yp = source2gal(theta_int, xp, yp)
        plane = 'gal'

    if plane == 'gal':
        xp, yp = gal2disk(sini, xp, yp)

    return xp, yp
