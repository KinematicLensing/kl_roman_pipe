'''
A place for utility functions used throughout kl_pipe.
'''

import jax.numpy as jnp
from pathlib import Path
from typing import Tuple, Literal


def build_pixel_grid(
    N1: int,
    N2: int,
    indexing: Literal['ij', 'xy'],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build a coordinate grid for a 2D map with pixel-centered coordinates.

    Grid positions are defined at pixel centers. For even pixel counts,
    the image center falls on pixel corners (between 4 pixels). For odd
    counts, it falls on a pixel center.

    Parameters
    ----------
    N1 : int
        Number of pixels along first axis.
    N2 : int
        Number of pixels along second axis.
    indexing : {'ij', 'xy'}
        Indexing convention for output arrays:
        - 'ij': Matrix indexing where X[i,j] corresponds to position (i,j).
                Output shape is (N1, N2). First axis is x, second is y.
        - 'xy': Cartesian indexing where X[i,j] corresponds to position (j,i).
                Output shape is (N2, N1). First axis is y, second is x.

    Returns
    -------
    X, Y : jnp.ndarray
        2D coordinate grids in pixel units, centered at (0, 0).
        Shape is (N1, N2) if indexing='ij', or (N2, N1) if indexing='xy'.

    Examples
    --------
    >>> # Even dimensions: center falls between pixels
    >>> X, Y = build_pixel_grid(120, 80, indexing='ij')
    >>> X.shape
    (120, 80)
    >>> # No pixel is exactly at (0, 0) - center is between pixels
    >>> X[59, 39]  # pixel just below/left of center
    -0.5
    >>> X[60, 40]  # pixel just above/right of center
    0.5
    >>> Y[59, 39]
    -0.5
    >>> Y[60, 40]
    0.5
    >>> # Corners
    >>> X[0, 0]
    -59.5
    >>> Y[0, 0]
    -39.5

    >>> # Odd dimensions: center pixel is exactly at (0, 0)
    >>> X, Y = build_pixel_grid(101, 51, indexing='ij')
    >>> X.shape
    (101, 51)
    >>> X[50, 25]  # center pixel
    0.0
    >>> Y[50, 25]
    0.0
    >>> # Corners (no half-pixel offset)
    >>> X[0, 0]
    -50.0
    >>> Y[0, 0]
    -25.0

    >>> # Cartesian indexing with rectangular grid (75 height, 125 width)
    >>> # Note: with indexing='xy', first arg is width (x-direction)
    >>> X, Y = build_pixel_grid(125, 75, indexing='xy')
    >>> X.shape  # Shape is (N2, N1) for 'xy' indexing
    (75, 125)
    >>> # Center pixel (odd x odd)
    >>> X[37, 62]
    0.0
    >>> Y[37, 62]
    0.0

    Notes
    -----
    The coordinate system is centered at (0, 0):
    - For even N: pixels span from -(N/2 - 0.5) to (N/2 - 0.5)
      Center falls on pixel corner at (0, 0)
    - For odd N: pixels span from -(N-1)/2 to (N-1)/2
      Center falls on pixel center at (0, 0)
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

    This function always returns grids in matrix indexing ('ij') convention,
    using the unambiguous Nrow and Ncol properties from ImagePars.

    Parameters
    ----------
    image_pars : ImagePars
        Image parameters containing shape, pixel_scale, and indexing.
        The output grid uses image_pars.Nrow and image_pars.Ncol which
        are guaranteed to represent rows and columns correctly.
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
        2D coordinate grids in specified units.
        Always uses 'ij' indexing convention where X represents the row
        coordinate and Y represents the column coordinate.
        Shape is (Nrow, Ncol).

    Examples
    --------
    >>> from kl_pipe.parameters import ImagePars
    >>>
    >>> # Rectangular image: 150 rows x 200 columns
    >>> # Regardless of how ImagePars was created, Nrow and Ncol are unambiguous
    >>> image_pars = ImagePars(shape=(150, 200), pixel_scale=0.1, indexing='ij')
    >>>
    >>> # Physical coordinates in arcsec, centered at origin
    >>> X, Y = build_map_grid_from_image_pars(image_pars, unit='arcsec', centered=True)
    >>> X.shape
    (150, 200)
    >>> # Center is between pixels for even dimensions
    >>> X[74, 99]  # pixel just below/left of center
    -0.05
    >>> X[75, 100]  # pixel just above/right of center
    0.05
    >>> # Corners
    >>> X[0, 0]  # corner in arcsec
    -7.45
    >>> Y[0, 0]
    -9.95
    >>>
    >>> # Pixel coordinates, centered at origin
    >>> X, Y = build_map_grid_from_image_pars(image_pars, unit='pixel', centered=True)
    >>> X[74, 99]
    -0.5
    >>> X[75, 100]
    0.5
    >>>
    >>> # Odd dimensions (101 rows x 51 columns) - center pixel at exactly (0, 0)
    >>> image_pars2 = ImagePars(shape=(101, 51), pixel_scale=0.2, indexing='ij')
    >>> X, Y = build_map_grid_from_image_pars(image_pars2, unit='arcsec', centered=True)
    >>> X[50, 25]  # exact center
    0.0
    >>> Y[50, 25]
    0.0
    >>> X[0, 0]  # corner
    -10.0
    >>> Y[0, 0]
    -5.0
    >>>
    >>> # Non-centered pixel indices (80 rows x 120 columns)
    >>> image_pars3 = ImagePars(shape=(80, 120), pixel_scale=0.05, indexing='ij')
    >>> X, Y = build_map_grid_from_image_pars(image_pars3, unit='pixel', centered=False)
    >>> X[0, 0]
    0.0
    >>> X[79, 119]
    79.0
    >>> Y[79, 119]
    119.0
    >>>
    >>> # Same grid but in arcsec (non-centered)
    >>> X, Y = build_map_grid_from_image_pars(image_pars3, unit='arcsec', centered=False)
    >>> X[0, 0]
    0.0
    >>> X[79, 119]
    3.95
    >>> Y[79, 119]
    5.95

    Notes
    -----
    This function intentionally uses only 'ij' indexing internally to avoid
    confusion. The ImagePars.Nrow and ImagePars.Ncol properties handle any
    indexing conversion needed from the original ImagePars creation.
    """
    if unit not in ['arcsec', 'pixel']:
        raise ValueError(f"unit must be 'arcsec' or 'pixel', got '{unit}'")

    if centered:
        # Build centered grid in pixel units
        # Always use 'ij' indexing with unambiguous Nrow, Ncol
        X, Y = build_pixel_grid(image_pars.Nrow, image_pars.Ncol, indexing='ij')

        # Scale to physical units if requested
        if unit == 'arcsec':
            X = X * image_pars.pixel_scale
            Y = Y * image_pars.pixel_scale
    else:
        # Non-centered: pixel indices starting from 0
        # Always use 'ij' convention
        idx_row = jnp.arange(image_pars.Nrow)
        idx_col = jnp.arange(image_pars.Ncol)

        X, Y = jnp.meshgrid(idx_row, idx_col, indexing='ij')

        # Scale to physical units if requested
        if unit == 'arcsec':
            X = X * image_pars.pixel_scale
            Y = Y * image_pars.pixel_scale

    return X, Y


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
