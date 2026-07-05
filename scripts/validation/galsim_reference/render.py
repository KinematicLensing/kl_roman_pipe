"""GalSim-chromatic reference render of a dispersed slitless-grism image.

Builds the dispersed grism image independently of ``kl_pipe``'s rendering
pipeline (no ``kl_pipe`` import in this module -- only ``galsim`` + numpy +
``galsim_reference.physics``), following the recipe:

  1. Dense fine spatial grid (oversampled pixel, ~2x the coarse stamp).
  2. I(x,y), v_los(x,y) via ``physics`` (independent numpy LOS quadrature
     + arctan rotation curve).
  3. Partition v_los into N_v isovelocity-bin channel images S_b(x,y).
  4. Per-channel Gaussian line SED centered at the Doppler-shifted
     wavelength of the bin center, width from the intrinsic velocity
     dispersion.
  5. ``galsim.ChromaticSum([S_b * SED_b])``, shifted by the
     wavelength-dependent dispersion offset via
     ``ChromaticObject.shift(callable_of_wavelength)``, convolved with the
     (achromatic) PSF, drawn through a top-hat bandpass spanning the full
     line window.

This construction is independent of ``kl_pipe``'s cube-assembly/dispersal
code path (``kl_pipe/spectral.py``, ``kl_pipe/dispersion.py``): rather than
building an (x, y, lambda) datacube and shifting wavelength slices, it sums
chromatic objects whose spatial support already encodes the Doppler field
and lets GalSim integrate over wavelength internally.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import galsim
import numpy as np

from .physics import (
    GalaxyParams,
    C_KMS,
    assign_velocity_bins,
    build_grid,
    intensity_map,
    velocity_los_map,
)


@dataclass
class GalSimReferenceConfig:
    """Rendering-density knobs for the GalSim-chromatic reference render.

    Not physical parameters -- convergence of these against ``kl_pipe`` was
    checked directly (N_v in {32,64,128}: flat residual; pts_per_sigma in
    {5,10,20,40}: flat residual). See
    ``docs/validation/galsim_reference_gate.md``.
    """

    n_v: int = 64
    pts_per_sigma: int = 20  # SED/wave_list sampling density per sigma_lambda
    nsigma_line: float = 6.0  # SED window half-width in sigma_lambda
    v_margin_sigma: float = 3.0  # velocity-bin range padding, in sigma_v
    fine_oversample: int = 5  # fine pixel = coarse_pixel_scale / fine_oversample
    stamp_pad: float = 2.0  # fine grid spans stamp_pad x the coarse stamp width
    # LOS quadrature density for physics.intensity_map. Converged (5e-5
    # rel. vs kl_pipe's independent tanh-GL quadrature) by n_los=41; 101
    # kept as a safety margin at negligible extra cost.
    n_los: int = 101
    n_hz: float = 12.0  # LOS quadrature half-width, in h_z/cosi units


def render_galsim_reference(
    p: GalaxyParams,
    coarse_pixel_scale: float,
    coarse_shape: tuple,
    psf_fwhm: float,
    dispersion_nm_per_pix: float,
    lambda_ref: float,
    cfg: GalSimReferenceConfig,
):
    """Render the dispersed grism image via the GalSim-chromatic reference.

    Returns
    -------
    dict with keys: 'image' (ndarray, coarse_shape), 'timing' (dict),
    'total_flux_fine', 'n_active_channels', 'v_centers', 'v_edges'.
    """
    t_start = time.time()
    nrow_c, ncol_c = coarse_shape
    stamp_h = nrow_c * coarse_pixel_scale
    stamp_w = ncol_c * coarse_pixel_scale
    fine_scale = coarse_pixel_scale / cfg.fine_oversample
    nrow_f = int(round(stamp_h * cfg.stamp_pad / fine_scale))
    ncol_f = int(round(stamp_w * cfg.stamp_pad / fine_scale))
    # keep fine grid odd-sized so the coarse stamp center coincides with a
    # fine pixel center (avoids a half-fine-pixel centroid offset)
    if nrow_f % 2 == 0:
        nrow_f += 1
    if ncol_f % 2 == 0:
        ncol_f += 1

    X, Y = build_grid(nrow_f, ncol_f, fine_scale)
    I_fine = intensity_map(X, Y, p, n_los=cfg.n_los, n_hz=cfg.n_hz)
    v_fine = velocity_los_map(X, Y, p)

    bin_index, v_centers, v_edges = assign_velocity_bins(
        v_fine, cfg.n_v, cfg.v_margin_sigma, p.sigma_v
    )

    lam_centers = lambda_ref * (1.0 + v_centers / C_KMS)
    sigma_lambdas = lam_centers * p.sigma_v / C_KMS
    global_blue = float((lam_centers - cfg.nsigma_line * sigma_lambdas).min())
    global_red = float((lam_centers + cfg.nsigma_line * sigma_lambdas).max())

    t_setup0 = time.time()

    comps = []
    for b in range(cfg.n_v):
        mask = bin_index == b
        if not np.any(mask):
            continue
        Sb = np.where(mask, I_fine, 0.0)
        img = galsim.Image(np.ascontiguousarray(Sb), scale=fine_scale)
        gsobj = galsim.InterpolatedImage(img, normalization='sb')

        lam_b = lam_centers[b]
        sigma_lambda = sigma_lambdas[b]
        npts = cfg.pts_per_sigma * int(2 * cfg.nsigma_line) + 1
        waves_local = np.linspace(
            lam_b - cfg.nsigma_line * sigma_lambda,
            lam_b + cfg.nsigma_line * sigma_lambda,
            npts,
        )
        vals_local = np.exp(-0.5 * ((waves_local - lam_b) / sigma_lambda) ** 2)
        # hard cutoff at the +/-nsigma edge (already negligible), then pad
        # the SED's defined domain out to the GLOBAL blue/red limits with
        # exact zeros -- galsim probes SEDs at bandpass edges (FFT-size
        # caching) even when they contribute no flux there, and raises
        # GalSimRangeError outside a LookupTable-SED's declared domain.
        vals_local[0] = 0.0
        vals_local[-1] = 0.0
        eps = 1e-9 * (waves_local[-1] - waves_local[0])
        pre = [global_blue] if global_blue < waves_local[0] - eps else []
        post = [global_red] if global_red > waves_local[-1] + eps else []
        waves = np.concatenate([pre, waves_local, post])
        vals = np.concatenate([[0.0] * len(pre), vals_local, [0.0] * len(post)])

        lut = galsim.LookupTable(waves, vals, interpolant='linear')
        sed = galsim.SED(lut, wave_type='nm', flux_type='flambda')
        bp_local = galsim.Bandpass(
            galsim.LookupTable(waves_local, np.ones_like(waves_local)), wave_type='nm'
        )
        # normalize to unit wavelength-integrated weight: total flux
        # contributed by this channel across all wavelengths equals its
        # (wavelength-independent) spatial flux, matching kl_pipe's
        # per-pixel normalized Gaussian-in-wavelength kernel convention.
        sed = sed.withFlux(1.0, bp_local)

        comps.append(gsobj * sed)

    chrom_sum = galsim.ChromaticSum(comps)

    def shift_fn(w):
        dx = (w - lambda_ref) / dispersion_nm_per_pix * coarse_pixel_scale
        return (dx, 0.0)

    shifted = chrom_sum.shift(shift_fn)
    psf = galsim.Gaussian(fwhm=psf_fwhm)
    final = galsim.Convolve(shifted, psf)

    bp = galsim.Bandpass(
        galsim.LookupTable([global_blue, global_red], [1.0, 1.0]),
        wave_type='nm',
        blue_limit=global_blue,
        red_limit=global_red,
    )

    t_setup1 = time.time()

    image = final.drawImage(
        bandpass=bp, scale=coarse_pixel_scale, nx=ncol_c, ny=nrow_c, method='auto'
    )

    t_end = time.time()

    return {
        'image': image.array.copy(),
        'timing': {
            'grid_and_physics': t_setup0 - t_start,
            'sed_and_chromatic_setup': t_setup1 - t_setup0,
            'drawImage': t_end - t_setup1,
            'total': t_end - t_start,
        },
        'n_active_channels': len(comps),
        'total_flux_fine': float(I_fine.sum() * fine_scale**2),
        'fine_shape': (nrow_f, ncol_f),
        'fine_scale': fine_scale,
        'v_centers': v_centers,
        'v_edges': v_edges,
    }
