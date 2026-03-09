"""Grism diagnostic plots."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Callable, Sequence

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection

from kl_pipe.plotting import MidpointNormalize
from kl_pipe.diagnostics.datacube import _add_colorbar, _pick_channel_indices

_C_KMS = 299792.458


def plot_grism_overview(
    cube,
    grism_image,
    lambda_grid,
    grism_pars,
    imap=None,
    vmap=None,
    grism_norot=None,
    lam_center=None,
    n_channels=5,
    v0=0.0,
    title=None,
    save_path=None,
):
    """Master 3-row grism diagnostic.

    Row 1: [Intensity] [Velocity] [Stacked cube flux]
    Row 2: [channel_1] ... [channel_n]
    Row 3: [Stacked + dispersion arrow] [Grism image] [Velocity signature]

    Parameters
    ----------
    cube : ndarray
        Datacube (Nrow, Ncol, Nlambda).
    grism_image : ndarray
        Dispersed 2D grism image (Nrow, Ncol).
    lambda_grid : ndarray
        Wavelength array (nm), shape (Nlambda,).
    grism_pars : GrismPars
        Grism parameters (dispersion_angle, lambda_ref, etc).
    imap : ndarray, optional
        True intensity map.
    vmap : ndarray, optional
        True velocity map.
    grism_norot : ndarray, optional
        Grism image for non-rotating galaxy (vcirc=0). If provided,
        row 3 panel 3 shows velocity signature = grism - grism_norot.
    lam_center : float, optional
        Center wavelength. Default: middle of lambda_grid.
    n_channels : int
        Number of wavelength channels in row 2.
    v0 : float
        Systemic velocity for vmap centering.
    title : str, optional
        Figure title.
    save_path : str or Path, optional
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cube = np.asarray(cube)
    grism_image = np.asarray(grism_image)
    lambda_grid = np.asarray(lambda_grid)
    Nlam = len(lambda_grid)

    if lam_center is None:
        lam_center = float(lambda_grid[Nlam // 2])

    dlam = float(lambda_grid[1] - lambda_grid[0]) if Nlam >= 2 else 1.0
    stacked = np.sum(cube, axis=2) * dlam

    ncols = max(5, n_channels)
    fig, axes = plt.subplots(3, ncols, figsize=(3.5 * ncols, 10))

    # === Row 1 ===
    # intensity map
    ax = axes[0, 0]
    if imap is not None:
        im = ax.imshow(np.asarray(imap), origin='lower')
        ax.set_title('Intensity I(x,y)')
        _add_colorbar(ax, im)
    else:
        ax.set_visible(False)

    # velocity map
    ax = axes[0, 1]
    if vmap is not None:
        vmap_np = np.asarray(vmap)
        norm_v = MidpointNormalize(
            vmin=float(np.nanmin(vmap_np)),
            vmax=float(np.nanmax(vmap_np)),
            midpoint=v0,
        )
        im = ax.imshow(vmap_np, origin='lower', cmap='RdBu_r', norm=norm_v)
        ax.set_title('Velocity V(x,y)')
        _add_colorbar(ax, im)
    else:
        ax.set_visible(False)

    # stacked cube flux
    ax = axes[0, 2]
    im = ax.imshow(stacked, origin='lower')
    ax.set_title(r'Stacked $\Sigma_\lambda$ flux')
    _add_colorbar(ax, im)

    for j in range(3, ncols):
        axes[0, j].set_visible(False)

    # === Row 2: wavelength channels ===
    channel_indices = _pick_channel_indices(Nlam, n_channels, lam_center, lambda_grid)
    channel_data = [np.asarray(cube[:, :, idx]) for idx in channel_indices]
    vmin_ch = min(d.min() for d in channel_data)
    vmax_ch = max(d.max() for d in channel_data)

    for i, idx in enumerate(channel_indices):
        ax = axes[1, i]
        im = ax.imshow(channel_data[i], origin='lower', vmin=vmin_ch, vmax=vmax_ch)
        offset_nm = float(lambda_grid[idx]) - lam_center
        offset_kms = offset_nm / lam_center * _C_KMS
        ax.set_title(
            f'$\\Delta\\lambda$ = {offset_nm:.1f} nm\n({offset_kms:.0f} km/s)',
            fontsize=9,
        )
        _add_colorbar(ax, im)

    for j in range(n_channels, ncols):
        axes[1, j].set_visible(False)

    # === Row 3 ===
    # panel 1: stacked + dispersion direction arrow
    ax = axes[2, 0]
    im = ax.imshow(stacked, origin='lower')
    ax.set_title('Stacked + disp. dir.')
    _add_colorbar(ax, im)
    _draw_dispersion_arrow(ax, grism_pars, stacked.shape, lambda_grid)

    # panel 2: grism image
    ax = axes[2, 1]
    im = ax.imshow(grism_image, origin='lower')
    ax.set_title('Grism image')
    _add_colorbar(ax, im)

    # panel 3: velocity signature
    ax = axes[2, 2]
    if grism_norot is not None:
        vel_sig = np.asarray(grism_image) - np.asarray(grism_norot)
        vmax_sig = float(np.max(np.abs(vel_sig)))
        if vmax_sig == 0:
            vmax_sig = 1.0
        im = ax.imshow(
            vel_sig, origin='lower', cmap='RdBu_r', vmin=-vmax_sig, vmax=vmax_sig
        )
        ax.set_title('Velocity signature')
        _add_colorbar(ax, im)
    else:
        ax.text(
            0.5,
            0.5,
            'No grism_norot\nprovided',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=10,
        )
        ax.set_title('Velocity signature (N/A)')

    for j in range(3, ncols):
        axes[2, j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_dispersion_angles(
    build_grism_fn,
    angles=(0, np.pi / 2, np.pi, 3 * np.pi / 2),
    labels=('0', '90', '180', '270'),
    title=None,
    save_path=None,
):
    """1xN grid of grism images at different dispersion angles.

    Parameters
    ----------
    build_grism_fn : callable
        ``build_grism_fn(angle) -> (Nrow, Ncol)`` grism image.
    angles : sequence of float
        Dispersion angles in radians.
    labels : sequence of str
        Labels for each angle.
    title : str, optional
        Figure title.
    save_path : str or Path, optional
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(angles)
    images = [np.asarray(build_grism_fn(a)) for a in angles]

    vmin = min(img.min() for img in images)
    vmax = max(img.max() for img in images)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for i, (img, label) in enumerate(zip(images, labels)):
        im = axes[i].imshow(img, origin='lower', vmin=vmin, vmax=vmax)
        axes[i].set_title(f'Disp. angle = {label}')
        _add_colorbar(axes[i], im)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_dispersion_angle_study(
    build_grism_fn,
    broadband_stacked,
    angles=(0, np.pi / 2, np.pi, 3 * np.pi / 2),
    labels=('0', '90', '180', '270'),
    title=None,
    save_path=None,
):
    """Deep-dive: 4x3 grid showing grism image, residual, 1D cross-section per angle.

    Col 1: Grism image
    Col 2: Grism - broadband stacked (diverging)
    Col 3: 1D cross-section through center along dispersion direction

    Parameters
    ----------
    build_grism_fn : callable
        ``build_grism_fn(angle) -> (Nrow, Ncol)`` grism image.
    broadband_stacked : ndarray
        Reference undispersed image (Nrow, Ncol).
    angles : sequence of float
        Dispersion angles in radians.
    labels : sequence of str
        Labels for each angle.
    title : str, optional
        Figure title.
    save_path : str or Path, optional
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    broadband_stacked = np.asarray(broadband_stacked)
    n = len(angles)
    images = [np.asarray(build_grism_fn(a)) for a in angles]
    residuals = [img - broadband_stacked for img in images]

    # shared colorscale for grism images
    vmin_g = min(img.min() for img in images)
    vmax_g = max(img.max() for img in images)

    # shared symmetric colorscale for residuals
    abs_max_r = max(np.abs(r).max() for r in residuals)
    if abs_max_r == 0:
        abs_max_r = 1.0

    fig, axes = plt.subplots(n, 3, figsize=(14, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    Nrow, Ncol = broadband_stacked.shape
    cy, cx = Nrow // 2, Ncol // 2

    for i, (img, resid, angle, label) in enumerate(
        zip(images, residuals, angles, labels)
    ):
        # col 1: grism image
        ax = axes[i, 0]
        im = ax.imshow(img, origin='lower', vmin=vmin_g, vmax=vmax_g)
        ax.set_title(f'Grism ({label})')
        _add_colorbar(ax, im)

        # col 2: residual
        ax = axes[i, 1]
        im = ax.imshow(
            resid, origin='lower', cmap='RdBu_r', vmin=-abs_max_r, vmax=abs_max_r
        )
        ax.set_title(f'Grism - Broadband ({label})')
        _add_colorbar(ax, im)

        # col 3: 1D cross-section along dispersion direction
        ax = axes[i, 2]
        grism_profile, bb_profile, offsets = _extract_cross_section(
            img, broadband_stacked, angle, cy, cx
        )
        ax.plot(offsets, grism_profile, '-', label='Grism', linewidth=1.5)
        ax.plot(offsets, bb_profile, '--', label='Broadband', linewidth=1.5)
        ax.set_xlabel('Pixel offset from center')
        ax.set_ylabel('Flux')
        ax.set_title(f'Cross-section ({label})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _draw_dispersion_arrow(ax, grism_pars, shape, lambda_grid):
    """Draw a blue-to-red gradient arrow showing dispersion direction."""
    Nrow, Ncol = shape
    cy, cx = Nrow / 2, Ncol / 2

    angle = grism_pars.dispersion_angle
    lambda_grid = np.asarray(lambda_grid)
    lam_range = float(lambda_grid[-1] - lambda_grid[0])
    arrow_len_pix = lam_range / grism_pars.dispersion  # pixels
    # scale to fit in image
    max_arrow = min(Nrow, Ncol) * 0.4
    arrow_len = min(arrow_len_pix, max_arrow)

    dx = arrow_len * np.cos(angle)
    dy = arrow_len * np.sin(angle)

    # gradient arrow via LineCollection
    n_seg = 20
    t = np.linspace(0, 1, n_seg + 1)
    x_pts = cx - dx / 2 + dx * t
    y_pts = cy - dy / 2 + dy * t

    points = np.column_stack([x_pts, y_pts]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # blue (short lambda) to red (long lambda)
    colors = plt.cm.coolwarm(t[:-1])
    lc = LineCollection(segments, colors=colors, linewidths=3)
    ax.add_collection(lc)

    # arrowhead at red end
    ax.annotate(
        '',
        xy=(x_pts[-1], y_pts[-1]),
        xytext=(x_pts[-2], y_pts[-2]),
        arrowprops=dict(arrowstyle='->', color='red', lw=2),
    )


def _extract_cross_section(image, reference, angle, cy, cx):
    """Extract 1D cross-section through (cy,cx) along direction `angle`.

    Uses bilinear interpolation via scipy.ndimage.map_coordinates.
    """
    from scipy.ndimage import map_coordinates

    Nrow, Ncol = image.shape
    max_len = min(Nrow, Ncol) // 2

    offsets = np.arange(-max_len, max_len + 1, dtype=float)
    # sample along dispersion direction
    y_coords = cy + offsets * np.sin(angle)
    x_coords = cx + offsets * np.cos(angle)

    # bilinear interpolation
    grism_profile = map_coordinates(
        image, [y_coords, x_coords], order=1, mode='constant'
    )
    bb_profile = map_coordinates(
        reference, [y_coords, x_coords], order=1, mode='constant'
    )

    return grism_profile, bb_profile, offsets
