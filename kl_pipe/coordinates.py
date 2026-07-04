"""
Celestial / detector frame coordinate utilities.

Slim JAX-friendly helpers for handling per-observation celestial-to-detector
rotations driven by an astropy WCS. Used by SourceModel to rotate celestial-
frame sampled parameters (theta_int, g1, g2) into the detector frame for each
observation at render time.

The math is the same as the kl-tools OrientedAngle class (we use the term
"celestial" instead of "sky" but the conversions are identical). astropy
Angle is not JAX-traceable, so we extract scalar rotations at obs
construction time and operate on plain floats / jnp arrays thereafter.

Sign convention:
- ``image_rotation`` is the angle (radians) that rotates celestial axes into
  detector pixel axes. For a WCS with PC matrix P, ``image_rotation =
  atan2(P[1,0], P[0,0])`` (with +pi added if det(P) < 0 to absorb mirroring).
- ``theta_int_detector = theta_int_celestial - image_rotation`` (matches
  ``OrientedAngle._sky2cartesian``).
- ``(g1_det, g2_det) = rotate_shear(g1_cel, g2_cel, image_rotation)``; shear
  is a spin-2 quantity that rotates by ``2 * image_rotation``.
- ``(x_det, y_det) = rotate_position(x_cel, y_cel, image_rotation)``;
  positions are spin-1 (component rotation by ``image_rotation``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

if TYPE_CHECKING:
    from astropy.wcs import WCS


def image_rotation_from_wcs(wcs: 'WCS') -> float:
    """Extract celestial-to-detector rotation (radians) from a WCS.

    Reads the WCS PC matrix (preferred) or CD matrix. The rotation is
    ``atan2(R[1,0], R[0,0])``. If the matrix has negative determinant
    (mirroring, e.g. a parity flip), pi is added so the rotation absorbs
    the sign-flip — matches ``OrientedAngle._compute_image_rotation``.

    Returned as a plain Python float so it can be stored on frozen
    obs dataclasses and broadcast into JAX arrays at render time.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        Two-axis WCS object. astropy WCS instances always expose a PC
        matrix (defaulting to identity) or a CD matrix; this function
        prefers PC when present.
    """
    if wcs.wcs.has_pc():
        R = wcs.wcs.get_pc()
    else:
        R = wcs.wcs.get_cd()

    theta = float(np.arctan2(R[1, 0], R[0, 0]))
    if np.linalg.det(R) < 0:
        theta += np.pi

    return theta


def wcs_is_flipped(wcs: 'WCS') -> bool:
    """True when the WCS PC/CD matrix has negative determinant (a parity
    flip / mirroring).

    A flip is NOT a rotation: ``image_rotation_from_wcs`` absorbs it by
    adding pi (the kl-tools OrientedAngle convention), which is only
    meaningful for spin-2 / axis-like quantities. Pathways that rotate
    positions or sampling grids (e.g. the shared-cube grism path) must
    reject flipped WCSs loudly instead.
    """
    if wcs.wcs.has_pc():
        R = wcs.wcs.get_pc()
    else:
        R = wcs.wcs.get_cd()
    return bool(np.linalg.det(R) < 0)


def rotate_shear(g1, g2, phi) -> Tuple:
    """Spin-2 rotation of shear components by ``phi`` radians.

    Shear is a rank-2 traceless symmetric tensor that transforms as spin-2:
    rotating coordinates by ``phi`` rotates the components by ``2 * phi``.

    .. math::
        g1' = g1 \\cos(2\\phi) + g2 \\sin(2\\phi)
        g2' = -g1 \\sin(2\\phi) + g2 \\cos(2\\phi)

    JAX-traceable: all inputs may be JAX scalars or arrays.

    Parameters
    ----------
    g1, g2 : float or jnp.ndarray
        Celestial-frame shear components.
    phi : float or jnp.ndarray
        Rotation angle in radians (celestial-to-detector, i.e. the obs's
        ``image_rotation``).

    Returns
    -------
    (g1_rot, g2_rot) : tuple of same type as inputs
        Detector-frame shear components.
    """
    c = jnp.cos(2.0 * phi)
    s = jnp.sin(2.0 * phi)
    g1_rot = g1 * c + g2 * s
    g2_rot = -g1 * s + g2 * c
    return g1_rot, g2_rot


def rotate_position(x, y, phi) -> Tuple:
    """Spin-1 rotation of position components by ``phi`` radians.

    A position vector fixed on the sky has detector-frame components
    rotated by the celestial-to-detector angle: a direction at celestial
    angle ``alpha`` appears at ``alpha - phi`` in the detector frame,
    consistent with ``theta_int_det = theta_int_cel - phi`` and the spin-2
    ``rotate_shear``.

    .. math::
        x' = x \\cos(\\phi) + y \\sin(\\phi)
        y' = -x \\sin(\\phi) + y \\cos(\\phi)

    JAX-traceable: all inputs may be JAX scalars or arrays.

    Parameters
    ----------
    x, y : float or jnp.ndarray
        Celestial-frame position components.
    phi : float or jnp.ndarray
        Rotation angle in radians (celestial-to-detector, i.e. the obs's
        ``image_rotation``).

    Returns
    -------
    (x_rot, y_rot) : tuple of same type as inputs
        Detector-frame position components.
    """
    c = jnp.cos(phi)
    s = jnp.sin(phi)
    return x * c + y * s, -x * s + y * c
