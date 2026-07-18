'''
A place for utility functions used throughout kl_pipe.
'''

import jax.numpy as jnp
from pathlib import Path
from typing import Tuple, Literal


def _build_pixel_grid(
    N1: int,
    N2: int,
    indexing: Literal['ij', 'xy'],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Internal helper for centered pixel grids. Not part of the public API.

    Use ``build_map_grid_from_image_pars`` instead, which returns grids in
    standard Cartesian convention (X=horizontal=cols, Y=vertical=rows).

    Parameters
    ----------
    N1 : int
        Number of pixels along first axis.
    N2 : int
        Number of pixels along second axis.
    indexing : {'ij', 'xy'}
        Passed directly to ``jnp.meshgrid``.

    Returns
    -------
    X, Y : jnp.ndarray
        2D coordinate grids in pixel units, centered at (0, 0).
    """
    if indexing not in ['ij', 'xy']:
        raise ValueError(f"indexing must be 'ij' or 'xy', got '{indexing}'")

    # Maximum distance along each axis
    # For even counts, offset by 0.5 pixels (center falls on corner)
    # For odd counts, no offset (center falls on pixel center)
    R1 = (N1 // 2) - 0.5 * ((N1 - 1) % 2)
    R2 = (N2 // 2) - 0.5 * ((N2 - 1) % 2)

    # Create 1D coordinate arrays
    coord1 = jnp.arange(-R1, R1 + 1, 1)
    coord2 = jnp.arange(-R2, R2 + 1, 1)

    # Verify correct lengths
    assert len(coord1) == N1, f"coord1 length {len(coord1)} != N1 {N1}"
    assert len(coord2) == N2, f"coord2 length {len(coord2)} != N2 {N2}"

    # Create 2D meshgrid
    X, Y = jnp.meshgrid(coord1, coord2, indexing=indexing)

    # Verify output shape
    expected_shape = (N1, N2) if indexing == 'ij' else (N2, N1)
    assert X.shape == expected_shape, f"X.shape {X.shape} != expected {expected_shape}"
    assert Y.shape == expected_shape, f"Y.shape {Y.shape} != expected {expected_shape}"

    return X, Y


def build_map_grid_from_image_pars(
    image_pars, unit: Literal['arcsec', 'pixel'] = 'arcsec', centered: bool = True
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build coordinate grid from ImagePars instance.

    Returns (X, Y) in standard Cartesian convention:
    X = horizontal coordinate (varies along cols, axis 1)
    Y = vertical coordinate (varies along rows, axis 0)
    Shape is always (Nrow, Ncol).

    Parameters
    ----------
    image_pars : ImagePars
        Image parameters containing shape, pixel_scale, and indexing.
    unit : {'arcsec', 'pixel'}
        Coordinate units for output grid:
        - 'arcsec': Scale coordinates by pixel_scale (physical units)
        - 'pixel': Use pixel units (integer-spaced)
        Default is 'arcsec'.
    centered : bool
        If True, center grid at (0, 0).
        If False, use pixel indices starting from 0.

    Returns
    -------
    X, Y : jnp.ndarray
        2D coordinate grids in specified units, shape (Nrow, Ncol).
        X[i,j] = x_j (horizontal, constant along rows, varies along cols).
        Y[i,j] = y_i (vertical, varies along rows, constant along cols).

    Examples
    --------
    >>> from kl_pipe.parameters import ImagePars
    >>>
    >>> # Rectangular image: 60 rows x 100 columns
    >>> image_pars = ImagePars(shape=(60, 100), pixel_scale=0.1, indexing='ij')
    >>>
    >>> X, Y = build_map_grid_from_image_pars(image_pars, unit='arcsec', centered=True)
    >>> X.shape
    (60, 100)
    >>> # X is horizontal (varies along cols = axis 1)
    >>> float(X[0, 49])  # just left of center
    -0.05
    >>> float(X[0, 50])  # just right of center
    0.05
    >>> # Y is vertical (varies along rows = axis 0)
    >>> float(Y[29, 0])  # just below center
    -0.05
    >>> float(Y[30, 0])  # just above center
    0.05

    Notes
    -----
    Uses standard Cartesian convention matching GalSim, matplotlib imshow
    with ``origin='lower'``, and FITS WCS (NAXIS1=Ncol=Nx, NAXIS2=Nrow=Ny).
    No transposes are needed when comparing with these tools.
    """
    if unit not in ['arcsec', 'pixel']:
        raise ValueError(f"unit must be 'arcsec' or 'pixel', got '{unit}'")

    Nrow = image_pars.Nrow
    Ncol = image_pars.Ncol

    if centered:
        # standard convention: X=cols (horizontal), Y=rows (vertical)
        x_coords = _centered_coords(Ncol)
        y_coords = _centered_coords(Nrow)
        X, Y = jnp.meshgrid(x_coords, y_coords, indexing='xy')

        if unit == 'arcsec':
            X = X * image_pars.pixel_scale
            Y = Y * image_pars.pixel_scale
    else:
        # non-centered: pixel indices starting from 0
        idx_x = jnp.arange(Ncol)  # horizontal pixel indices
        idx_y = jnp.arange(Nrow)  # vertical pixel indices
        X, Y = jnp.meshgrid(idx_x, idx_y, indexing='xy')

        if unit == 'arcsec':
            X = X * image_pars.pixel_scale
            Y = Y * image_pars.pixel_scale

    return X, Y


def _centered_coords(N: int) -> jnp.ndarray:
    """
    1D centered pixel coordinates for N pixels.

    Even N: center on corner, coords = [-N/2+0.5, ..., N/2-0.5]
    Odd N:  center on pixel, coords = [-(N-1)/2, ..., (N-1)/2]
    """
    return jnp.arange(N) - (N - 1) / 2.0


def get_base_dir() -> Path:
    '''
    base dir is parent repo dir
    '''
    module_dir = get_module_dir()
    return module_dir.parent


def get_module_dir() -> Path:
    return Path(__file__).parent


def get_test_dir() -> Path:
    base_dir = get_base_dir()
    return base_dir / 'tests'


def get_script_dir() -> Path:
    base_dir = get_base_dir()
    return base_dir / 'scripts'
