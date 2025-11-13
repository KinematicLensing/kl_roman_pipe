'''
Various plotting utilities for velocity and intensity maps, mostly for visualization &
testing purposes.
'''

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from kl_pipe.parameters import ImagePars
from kl_pipe.utils import build_map_grid_from_image_pars


def create_default_image_pars(
    rmax: float,
    Ngrid: int = 100,
    pixel_scale: Optional[float] = None,
    indexing: str = 'ij',
) -> ImagePars:
    """
    Create default ImagePars for plotting with a square grid centered at origin.

    Parameters
    ----------
    rmax : float
        Maximum radius from center in arcsec.
    Ngrid : int
        Number of pixels along each axis.
    pixel_scale : float, optional
        Pixel scale in arcsec/pixel. If None, automatically computed from rmax and Ngrid.
    indexing : str
        Indexing convention ('ij' or 'xy'). Default is 'ij'.

    Returns
    -------
    ImagePars
        Image parameters for the grid.
    """
    if pixel_scale is None:
        pixel_scale = 2 * rmax / Ngrid

    shape = (Ngrid, Ngrid)
    return ImagePars(shape=shape, pixel_scale=pixel_scale, indexing=indexing)


def plot_velocity_map(
    model,
    theta: jnp.ndarray,
    image_pars: Optional[ImagePars] = None,
    plane: str = 'obs',
    rmax: Optional[float] = None,
    Ngrid: int = 100,
    speed: bool = False,
    show: bool = True,
    outfile: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 8),
    title: Optional[str] = None,
    mark_center: bool = True,
    **kwargs,
):
    """
    Plot velocity or speed map in a given plane.

    Parameters
    ----------
    model : VelocityModel
        Velocity model instance.
    theta : jnp.ndarray
        Parameter array.
    image_pars : ImagePars, optional
        Image parameters defining the grid. If None, creates a default grid
        using rmax and Ngrid.
    plane : str
        Plane to plot in ('obs', 'cen', 'source', 'gal', 'disk').
    rmax : float, optional
        Maximum radius for plot (used only if image_pars is None).
        If None, uses 5 * rscale.
    Ngrid : int
        Grid resolution (used only if image_pars is None).
    speed : bool
        If True, plot speed instead of line-of-sight velocity.
    show : bool
        Whether to display the plot.
    outfile : str, optional
        Path to save figure.
    figsize : tuple
        Figure size.
    title : str, optional
        Plot title.
    mark_center : bool
        Whether to mark the center with an 'x'.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis objects.
    """

    # create ImagePars if not provided
    if image_pars is None:
        if rmax is None:
            rscale = model.get_param('vel_rscale', theta)
            rmax = 5.0 * float(rscale)

        image_pars = create_default_image_pars(rmax, Ngrid)

    # get coordinate grids in arcsec
    X, Y = build_map_grid_from_image_pars(image_pars, unit='arcsec', centered=True)

    # evaluate model
    V = model(theta, plane, X, Y, return_speed=speed)

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(X, Y, V, shading='auto')
    cbar = plt.colorbar(im, ax=ax, label='km/s')

    ax.set_xlabel('arcsec')
    ax.set_ylabel('arcsec')
    ax.set_aspect('equal')

    if mark_center:
        ax.plot(0, 0, 'rx', ms=10, markeredgewidth=2)

    if title is None:
        map_type = 'Speed' if speed else 'Velocity'
        title = f'{map_type} map in {plane} plane'
    ax.set_title(title)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show:
        plt.show()

    return fig, ax


def plot_all_planes(
    model,
    theta: jnp.ndarray,
    image_pars: Optional[ImagePars] = None,
    rmax: Optional[float] = None,
    Ngrid: int = 100,
    speed: bool = False,
    show: bool = True,
    outfile: Optional[str] = None,
    figsize: Tuple[int, int] = (13, 8),
    mark_center: bool = True,
):
    """
    Plot velocity/speed maps in all coordinate planes.

    Parameters
    ----------
    model : VelocityModel
        Velocity model instance.
    theta : jnp.ndarray
        Parameter array.
    image_pars : ImagePars, optional
        Image parameters. If None, creates default grid from rmax and Ngrid.
    rmax : float, optional
        Maximum radius (used only if image_pars is None).
    Ngrid : int
        Grid resolution (used only if image_pars is None).
    speed : bool
        Plot speed instead of velocity.
    show : bool
        Display the plot.
    outfile : str, optional
        Save path.
    figsize : tuple
        Figure size.
    mark_center : bool
        Mark centers.

    Returns
    -------
    fig, axes
        Matplotlib figure and axes array.
    """
    from kl_pipe.transformation import SUPPORTED_PLANES

    # Create ImagePars if not provided
    if image_pars is None:
        if rmax is None:
            rscale = model.get_param('vel_rscale', theta)
            rmax = 5.0 * float(rscale)

        image_pars = create_default_image_pars(rmax, Ngrid)

    # Get coordinate grid in arcsec
    X, Y = build_map_grid_from_image_pars(image_pars, unit='arcsec', centered=True)

    # Check if we should skip 'cen' plane (no offset)
    planes = SUPPORTED_PLANES.copy()
    has_offset = 'x0' in model._param_indices
    if not has_offset and 'cen' in planes:
        planes.remove('cen')

    # Setup subplots
    nplanes = len(planes)
    ncols = 2 if nplanes == 4 else 3
    nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    map_type = 'Speed' if speed else 'Velocity'

    for idx, plane in enumerate(planes):
        ax = axes[idx]

        # evaluate model
        V = model(theta, plane, X, Y, return_speed=speed)

        im = ax.pcolormesh(X, Y, V, shading='auto')
        plt.colorbar(im, ax=ax, label='km/s')
        ax.set_xlabel('arcsec')
        ax.set_ylabel('arcsec')
        ax.set_title(f'{map_type} in {plane} plane')
        ax.set_aspect('equal')

        if mark_center:
            ax.plot(0, 0, 'rx', ms=10, markeredgewidth=2)

    # Hide unused subplots
    for idx in range(nplanes, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show:
        plt.show()

    return fig, axes


def plot_rotation_curve(
    model,
    theta: jnp.ndarray,
    image_pars: ImagePars,
    plane: str = 'obs',
    mask: Optional[np.ndarray] = None,
    threshold_dist: float = 5.0,
    Nrbins: int = 20,
    show: bool = True,
    outfile: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
):
    """
    Plot rotation curve extracted from velocity map.

    Extracts the rotation curve by averaging velocities within a strip
    along the major axis, then plots both the velocity map and the
    resulting rotation curve.

    Parameters
    ----------
    model : VelocityModel
        Velocity model instance.
    theta : jnp.ndarray
        Parameter array.
    image_pars : ImagePars
        Image parameters defining the grid and pixel scale.
    plane : str
        Coordinate plane.
    mask : np.ndarray, optional
        Boolean mask for valid pixels. If None, uses all pixels.
    threshold_dist : float
        Maximum distance from major axis in pixels.
    Nrbins : int
        Number of radial bins for rotation curve.
    show : bool
        Display plot.
    outfile : str, optional
        Save path.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig, (ax1, ax2)
        Figure and axes.
    bin_centers : np.ndarray
        Radial bin centers (pixels).
    rotation_curve : np.ndarray
        Extracted circular velocities (km/s).
    rotation_curve_err : np.ndarray
        Standard deviations in each bin (km/s).
    """
    # Get coordinate grids (in arcsec, centered at 0)
    X_arcsec, Y_arcsec = build_map_grid_from_image_pars(
        image_pars, unit='arcsec', centered=True
    )

    # Also get pixel coordinates for display
    X_pix, Y_pix = build_map_grid_from_image_pars(
        image_pars, unit='pixel', centered=False
    )

    if mask is None:
        mask = np.ones(image_pars.shape, dtype=bool)

    # Evaluate velocity map
    vmap = np.array(model(theta, plane, X_arcsec, Y_arcsec))

    # Get model parameters
    vcirc = float(model.get_param('vcirc', theta))
    rscale = float(model.get_param('vel_rscale', theta))
    cosi = float(model.get_param('cosi', theta))
    theta_int = float(model.get_param('theta_int', theta))

    x0 = float(model.get_param('x0', theta)) if 'x0' in model._param_indices else 0.0
    y0 = float(model.get_param('y0', theta)) if 'y0' in model._param_indices else 0.0

    # Convert center from arcsec to pixels
    x0_pix = x0 / image_pars.pixel_scale + image_pars.Nrow / 2
    y0_pix = y0 / image_pars.pixel_scale + image_pars.Ncol / 2

    # Distance along major axis (in pixels)
    dx = np.cos(theta_int)
    dy = np.sin(theta_int)
    R_signed = (X_pix - x0_pix) * dx + (Y_pix - y0_pix) * dy

    # Distance from major axis (in pixels)
    dist_major = np.abs(-(X_pix - x0_pix) * dy + (Y_pix - y0_pix) * dx)

    # Find pixels near major axis
    major_mask = (dist_major <= threshold_dist) & mask
    R_in_major = R_signed[major_mask]

    if len(R_in_major) == 0:
        raise ValueError(f'No pixels within {threshold_dist} pixels of major axis')

    # Create radial bins
    Rmin, Rmax = R_in_major.min(), R_in_major.max()
    rbins = np.linspace(Rmin, Rmax, Nrbins + 1)
    bin_centers = (rbins[:-1] + rbins[1:]) / 2

    # Extract rotation curve
    rotation_curve = []
    rotation_curve_err = []

    for n in range(Nrbins):
        radial_mask = (R_signed >= rbins[n]) & (R_signed < rbins[n + 1])
        combined_mask = radial_mask & (dist_major <= threshold_dist) & mask

        if np.any(combined_mask):
            # Deproject to circular velocity
            avg_vcirc = np.mean(vmap[combined_mask] / cosi)
            std_vcirc = np.std(vmap[combined_mask] / cosi)
            rotation_curve.append(avg_vcirc)
            rotation_curve_err.append(std_vcirc)
        else:
            rotation_curve.append(np.nan)
            rotation_curve_err.append(np.nan)

    rotation_curve = np.array(rotation_curve)
    rotation_curve_err = np.array(rotation_curve_err)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Velocity map (show in pixel coordinates for clarity)
    vmin, vmax = np.percentile(vmap[mask], [1, 99])
    im = ax1.imshow(
        vmap.T,
        origin='lower',
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
        extent=[0, image_pars.Nrow, 0, image_pars.Ncol],
    )
    plt.colorbar(im, ax=ax1, label='km/s')

    # Mark major axis
    ax1.axline(
        (x0_pix, y0_pix),
        slope=dy / dx if dx != 0 else np.inf,
        color='k',
        ls='-',
        label='Major axis',
        lw=1.5,
    )

    # Show threshold region
    offset_dist = threshold_dist
    for sign in [-1, 1]:
        x_offset = x0_pix + sign * offset_dist * (-dy)
        y_offset = y0_pix + sign * offset_dist * dx
        ax1.axline(
            (x_offset, y_offset),
            slope=dy / dx if dx != 0 else np.inf,
            color='k',
            ls=':',
            alpha=0.5,
            lw=1,
        )

    ax1.set_xlim(0, image_pars.Nrow)
    ax1.set_ylim(0, image_pars.Ncol)
    ax1.set_title(f'Velocity Map ({plane} plane)')
    ax1.set_xlabel('x (pixels)')
    ax1.set_ylabel('y (pixels)')
    ax1.legend()

    # Rotation curve
    ax2.errorbar(
        bin_centers,
        rotation_curve,
        rotation_curve_err,
        marker='o',
        label='Extracted',
        capsize=3,
    )
    ax2.axhline(vcirc, c='k', ls='--', label=f'v_circ = {vcirc:.1f}')
    ax2.axhline(-vcirc, c='k', ls='--')

    rscale_pix = rscale / image_pars.pixel_scale
    ax2.axvline(rscale_pix, c='g', ls=':', label='vel_rscale')
    ax2.axvline(-rscale_pix, c='g', ls=':')

    ax2.set_xlabel('Radial Distance (pixels)')
    ax2.set_ylabel('Circular Velocity (km/s)')
    ax2.set_title('Rotation Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show:
        plt.show()

    return fig, (ax1, ax2), bin_centers, rotation_curve, rotation_curve_err
