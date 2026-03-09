"""Datacube diagnostic plots."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from kl_pipe.plotting import MidpointNormalize

# speed of light in km/s
_C_KMS = 299792.458


def plot_datacube_overview(
    cube,
    lambda_grid,
    imap=None,
    vmap=None,
    lam_center=None,
    n_channels=5,
    v0=0.0,
    title=None,
    save_path=None,
):
    """Multi-panel datacube overview.

    Row 1: [Intensity I(x,y)] [Velocity V(x,y)] [Stacked cube flux]
    Row 2: [lambda_1] [lambda_2] ... [lambda_n_channels]

    Parameters
    ----------
    cube : ndarray
        Datacube of shape (Nrow, Ncol, Nlambda).
    lambda_grid : ndarray
        Wavelength array (nm), shape (Nlambda,).
    imap : ndarray, optional
        True intensity map (Nrow, Ncol).
    vmap : ndarray, optional
        True velocity map (Nrow, Ncol).
    lam_center : float, optional
        Center wavelength (nm). Default: middle of lambda_grid.
    n_channels : int
        Number of wavelength channel panels in row 2.
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
    lambda_grid = np.asarray(lambda_grid)
    Nlam = len(lambda_grid)

    if lam_center is None:
        lam_center = float(lambda_grid[Nlam // 2])

    # stacked cube flux
    if Nlam >= 2:
        dlam = float(lambda_grid[1] - lambda_grid[0])
    else:
        dlam = 1.0
    stacked = np.sum(cube, axis=2) * dlam

    # figure layout
    n_top = 3
    fig, axes = plt.subplots(
        2, max(n_top, n_channels), figsize=(3.5 * max(n_top, n_channels), 7)
    )

    # --- Row 1 ---
    # panel 1: intensity map
    ax = axes[0, 0]
    if imap is not None:
        im = ax.imshow(np.asarray(imap), origin='lower')
        ax.set_title('Intensity I(x,y)')
        _add_colorbar(ax, im)
    else:
        ax.set_visible(False)

    # panel 2: velocity map
    ax = axes[0, 1]
    if vmap is not None:
        vmap_np = np.asarray(vmap)
        vmin_v = np.nanmin(vmap_np)
        vmax_v = np.nanmax(vmap_np)
        norm_v = MidpointNormalize(vmin=vmin_v, vmax=vmax_v, midpoint=v0)
        im = ax.imshow(vmap_np, origin='lower', cmap='RdBu_r', norm=norm_v)
        ax.set_title('Velocity V(x,y)')
        _add_colorbar(ax, im)
    else:
        ax.set_visible(False)

    # panel 3: stacked cube flux
    ax = axes[0, 2]
    im = ax.imshow(stacked, origin='lower')
    ax.set_title(r'Stacked $\Sigma_\lambda$ flux')
    _add_colorbar(ax, im)

    # hide unused top-row panels
    for j in range(n_top, max(n_top, n_channels)):
        axes[0, j].set_visible(False)

    # --- Row 2: wavelength channels ---
    channel_indices = _pick_channel_indices(Nlam, n_channels, lam_center, lambda_grid)

    # shared colorscale across channels
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

    # hide unused bottom-row panels
    for j in range(n_channels, max(n_top, n_channels)):
        axes[1, j].set_visible(False)

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


def _add_colorbar(ax, im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


def _pick_channel_indices(Nlam, n_channels, lam_center, lambda_grid):
    """Pick n_channels evenly spaced indices centered near lam_center."""
    lambda_grid = np.asarray(lambda_grid)
    center_idx = int(np.argmin(np.abs(lambda_grid - lam_center)))

    half = n_channels // 2
    start = max(0, center_idx - half)
    end = min(Nlam - 1, center_idx + half)

    # adjust if near edges
    if end - start + 1 < n_channels:
        if start == 0:
            end = min(Nlam - 1, start + n_channels - 1)
        else:
            start = max(0, end - n_channels + 1)

    indices = np.linspace(start, end, n_channels, dtype=int)
    return indices
