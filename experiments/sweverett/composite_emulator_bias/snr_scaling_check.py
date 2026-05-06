"""Run BulgeDisk likelihood slices at SNR=1000 and SNR=10000 (same seed) to test
the Fisher hypothesis: noise-driven slice peak shifts should scale as 1/SNR.

Reuses test_composite_intensity helpers via the imports below.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

from tests.test_composite_intensity import (
    _generate_composite_synthetic,
    _IMAGE_PARS,
    _TRUE_PARS_SHARED,
    _TEST_PSF,
    composite_grids as _composite_grids,
)
from kl_pipe.intensity import BulgeDiskModel
from kl_pipe.observation import build_image_obs
from kl_pipe.utils import build_map_grid_from_image_pars
from kl_pipe.likelihood import create_jitted_likelihood_intensity


PARAMS_TO_SCAN = [
    'cosi',
    'theta_int',
    'total_flux',
    'bulge_frac',
    'disk_rscale',
    'disk_h_over_r',
    'bulge_hlr',
]
N_POINTS = 401  # finer than e6's 201 to resolve small biases


def find_slice_peak(log_like_fn, theta_true, idx, lo, hi, n=N_POINTS):
    grid = jnp.linspace(lo, hi, n)
    thetas = theta_true[None, :].repeat(n, axis=0)
    thetas = thetas.at[:, idx].set(grid)
    lps = np.array([float(log_like_fn(t)) for t in thetas])
    peak_idx = int(np.argmax(lps))
    return float(grid[peak_idx]), peak_idx in (0, n - 1)


def run_at_snr(snr: int, seed: int = 42):
    pars = dict(_TRUE_PARS_SHARED)
    model = BulgeDiskModel(shared_centroids=True)
    theta_true = model.pars2theta(pars)

    data_true, data_noisy, variance = _generate_composite_synthetic(
        pars, _IMAGE_PARS, snr, seed=seed, psf=_TEST_PSF
    )

    obs = build_image_obs(
        _IMAGE_PARS,
        psf=_TEST_PSF,
        oversample=5,
        int_model=model,
        data=data_noisy,
        variance=variance,
    )
    log_like_fn = create_jitted_likelihood_intensity(model, obs)

    name_to_idx = {n: i for i, n in enumerate(model.PARAMETER_NAMES)}
    biases = {}
    for name in PARAMS_TO_SCAN:
        idx = name_to_idx[name]
        truth = float(theta_true[idx])
        # ±25% scan window (with floor for near-zero params)
        scan = max(0.25 * abs(truth), 0.05)
        lo, hi = truth - scan, truth + scan
        peak, hit_bound = find_slice_peak(log_like_fn, theta_true, idx, lo, hi)
        biases[name] = {
            'truth': truth,
            'peak': peak,
            'delta': peak - truth,
            'rel': (peak - truth) / truth if abs(truth) > 0 else float('nan'),
            'bound': hit_bound,
        }
    return biases


def main():
    print('Running BulgeDisk slice scan at SNR=1000 and SNR=10000 (seed=42)...\n')
    snr_results = {snr: run_at_snr(snr) for snr in [1000, 10000]}

    print(
        f'{"param":>20s}  {"truth":>10s}  '
        f'{"snr=1000 bias":>20s}  {"snr=10000 bias":>20s}  {"ratio":>8s}'
    )
    for name in PARAMS_TO_SCAN:
        b1k = snr_results[1000][name]
        b10k = snr_results[10000][name]
        ratio = (
            abs(b1k['delta'] / b10k['delta'])
            if abs(b10k['delta']) > 1e-12
            else float('inf')
        )
        print(
            f'{name:>20s}  {b1k["truth"]:+10.4f}  '
            f'{b1k["delta"]:+10.4f} ({b1k["rel"]*100:+6.2f}%)  '
            f'{b10k["delta"]:+10.4f} ({b10k["rel"]*100:+6.2f}%)  '
            f'{ratio:8.2f}'
        )

    out_path = Path(__file__).resolve().parent / 'out' / 'snr_scaling.csv'
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['param', 'snr', 'truth', 'peak', 'delta', 'rel'])
        for snr, results in snr_results.items():
            for name, d in results.items():
                w.writerow([name, snr, d['truth'], d['peak'], d['delta'], d['rel']])
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
