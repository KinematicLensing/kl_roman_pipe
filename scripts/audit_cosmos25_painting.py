"""Audit the COSMOS25 painted-emission-line catalog.

Runs on the joined parquet from scripts/build_cosmos25_catalog.py (which
already gates the row-order join) and reports, per stage:

  1  handshake: reproduce the delivering notebook's selections verbatim and
     require its exact printed densities (deep KL 6.8307, deep WL 58.4793,
     medium KL 1.9276 per arcmin^2; the revised 2026-07-29 notebook, from
     which scripts/regen_cosmos25_painting.py regenerates the painted
     section). A mismatch means the join, the painted file, or this
     reimplementation drifted -- hard failure.
  2  medium-tier density and the any-line/Halpha-only selection ratio
  3  medium-sample dN/dz (CSV + png)
  4  rest-frame Halpha EW distribution against the catalog's own
     photometry -- the physicality check (literature anchor: median rest
     EW(Ha) ~ 100-300 A for star-forming galaxies at z ~ 1-1.5)
  5  kl_pipe production waterfall using the actual population.py selection
     functions (census cuts; no reimplementation drift possible)

Usage:
    python scripts/audit_cosmos25_painting.py
    python scripts/audit_cosmos25_painting.py --out /tmp/audit
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kl_pipe.ensemble.population import (  # noqa: E402
    C_A_PER_S,
    compute_line_snr_total,
    matched_filter_compactness,
    psf_fwhm_arcsec,
)

# delivering-notebook constants (IAEstimate.ipynb, exact)
AREA_ARCMIN2 = 0.43 * 3600.0
GRISM_MIN_A, GRISM_MAX_A = 1.00e4, 1.93e4
MEDIUM_GRISM_LIM = 1.5e-16
DEEP_GRISM_LIM = 5.8e-17
MEDIUM_JH_DEPTH = 26.4
DEEP_JH_DEPTH = 27.5
JH_SNR_THRESHOLD = 18
R_SPATIAL_THRESHOLD = 0.4
SHAPE_ERROR_THRESHOLD = 0.2
ROMAN_EFF50PSF = 0.11

# the notebook's printed densities [per arcmin^2]; the handshake targets
# (2026-07-29 revised notebook; the original 2026-07-28 delivery printed
# 12.3004 / 58.4793 / 5.7377 with the dust-sign bug)
HANDSHAKE_TARGETS = {'deep_kl': 6.8307, 'deep_wl': 58.4793, 'medium_kl': 1.9276}

# kl_pipe census selection values (configs/ensembles/flagship2_shear_census*)
KL_Z_RANGE = (0.55, 1.9)
KL_SNR_MIN = 20.0
KL_MIN_R50_OVER_PSF = 1.0

LINES = ('Ha', 'OII', 'OIII')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--parquet',
        type=Path,
        default=REPO_ROOT / 'data/cosmos2025/private/cosmos25_v1.parquet',
    )
    ap.add_argument('--out', type=Path, default=REPO_ROOT / 'data/cosmos2025/audit')
    args = ap.parse_args()
    if not args.parquet.exists():
        raise FileNotFoundError(
            f"{args.parquet} not found; run scripts/build_cosmos25_catalog.py"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.parquet)
    n = len(df)
    print(f'loaded {n} rows from {args.parquet.name}')

    # --- notebook selection, verbatim reimplementation ---
    r_eff_arcsec = df['radius_sersic'].to_numpy() * 3600.0
    with np.errstate(divide='ignore', invalid='ignore'):
        r_spatial = (1.0 + (ROMAN_EFF50PSF / r_eff_arcsec) ** 2) ** (-1)
    shape_err = np.sqrt((df['e1_err'] ** 2 + df['e2_err'] ** 2) / 2.0).to_numpy()
    is_galaxy = df['type'].to_numpy() == 0
    lam = {line: df[f'lambda_{line}_obs'].to_numpy() for line in LINES}
    fluxes = {line: df[f'F_{line}'].to_numpy() for line in LINES}

    def in_grism(line):
        return (lam[line] > GRISM_MIN_A) & (lam[line] < GRISM_MAX_A)

    def image_cuts(depth):
        snr_proxy = 10 ** (-0.4 * (df['mag_model_f150w'].to_numpy() - depth))
        with np.errstate(invalid='ignore'):
            return (
                (snr_proxy > JH_SNR_THRESHOLD)
                & (r_spatial > R_SPATIAL_THRESHOLD)
                & (shape_err < SHAPE_ERROR_THRESHOLD)
                & is_galaxy
            )

    def line_selected(lim):
        m = np.zeros(n, dtype=bool)
        for line in LINES:
            with np.errstate(invalid='ignore'):
                m |= (fluxes[line] > lim) & in_grism(line)
        return m

    print('\n=== stage 1: handshake vs notebook printed densities ===')
    got = {
        'deep_kl': (image_cuts(DEEP_JH_DEPTH) & line_selected(DEEP_GRISM_LIM)).sum()
        / AREA_ARCMIN2,
        'deep_wl': image_cuts(DEEP_JH_DEPTH).sum() / AREA_ARCMIN2,
        'medium_kl': (
            image_cuts(MEDIUM_JH_DEPTH) & line_selected(MEDIUM_GRISM_LIM)
        ).sum()
        / AREA_ARCMIN2,
    }
    for key, target in HANDSHAKE_TARGETS.items():
        status = 'match' if abs(got[key] - target) < 5e-5 else 'mismatch'
        print(
            f'  {key:10s} {got[key]:9.4f} /arcmin^2  '
            f'(notebook {target:9.4f})  {status}'
        )
        if status == 'mismatch':
            raise RuntimeError(f'handshake failed on {key}')

    print('\n=== stage 2: medium-tier density ===')
    img = image_cuts(MEDIUM_JH_DEPTH)
    med_mask = img & line_selected(MEDIUM_GRISM_LIM)
    with np.errstate(invalid='ignore'):
        ha_only = img & (fluxes['Ha'] > MEDIUM_GRISM_LIM) & in_grism('Ha')
    print(
        f'  n = {med_mask.sum():6d}  '
        f'density = {med_mask.sum()/AREA_ARCMIN2:6.3f} /arcmin^2  '
        f'(any-line/Ha-only = {med_mask.sum()/max(ha_only.sum(), 1):.3f})'
    )

    print('\n=== stage 3: medium KL dN/dz ===')
    bins = np.linspace(0, 4, 81)
    z = df['zfinal'].to_numpy()
    h, _ = np.histogram(z[med_mask], bins=bins)
    table = {'z_lo': bins[:-1], 'z_hi': bins[1:], 'dn_per_arcmin2': h / AREA_ARCMIN2}
    pd.DataFrame(table).to_csv(args.out / 'medium_dndz.csv', index=False)
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    zc = 0.5 * (bins[1:] + bins[:-1])
    ax.step(zc, table['dn_per_arcmin2'], where='mid')
    ax.set_xlabel('photo-z (v1 zfinal)')
    ax.set_ylabel('dN/dz per arcmin$^2$ (bin width 0.05)')
    fig.tight_layout()
    fig.savefig(args.out / 'medium_dndz.png', dpi=150)
    print(f'  wrote {args.out}/medium_dndz.csv/.png')

    print('\n=== stage 4: rest-frame Halpha EW (medium sample) ===')
    # continuum from SE++ model photometry [uJy -> cgs] at lambda_obs;
    # restricted to F115W/F150W coverage (lambda_obs < 1.668 um)
    lam_ha = lam['Ha']
    f_nu_ujy = np.where(
        lam_ha < 1.30e4,
        df['flux_model_f115w'].to_numpy(),
        df['flux_model_f150w'].to_numpy(),
    )
    cov = lam_ha < 1.668e4
    m = med_mask & cov & (f_nu_ujy > 0) & in_grism('Ha')
    f_lambda = 1e-29 * f_nu_ujy[m] * C_A_PER_S / lam_ha[m] ** 2
    ew_rest = fluxes['Ha'][m] / f_lambda / (1.0 + z[m])
    q = np.percentile(ew_rest, [16, 50, 84])
    print(
        f'  n={m.sum():6d}  EW_rest(Ha) 16/50/84% = '
        f'{q[0]:7.1f} / {q[1]:7.1f} / {q[2]:7.1f} A'
    )
    print(
        '  (anchor: median rest EW(Ha) ~ 100-300 A for SF galaxies at '
        'z ~ 1-1.5; the regenerated painting measured 181 A on 2026-07-29, '
        'vs ~300 A for the retired dust-boosted delivery)'
    )

    print('\n=== stage 5: kl_pipe census waterfall ===')
    print(
        f'  cuts: galaxy quality -> z in {KL_Z_RANGE} -> '
        f'r50/PSF >= {KL_MIN_R50_OVER_PSF} -> line SNR_total >= {KL_SNR_MIN}'
    )
    rng = np.random.default_rng(42)
    cosi_iso = rng.uniform(0.05, 1.0, n)
    warn_ok = df['warn_flag'].to_numpy() == 0
    f_ha = fluxes['Ha']
    with np.errstate(invalid='ignore'):
        quality = (
            is_galaxy
            & warn_ok
            & np.isfinite(f_ha)
            & (f_ha > 0)
            & np.isfinite(r_eff_arcsec)
            & (r_eff_arcsec > 0)
        )
    zin = quality & (z >= KL_Z_RANGE[0]) & (z <= KL_Z_RANGE[1])
    res = zin.copy()
    res[zin] = r_eff_arcsec[zin] / psf_fwhm_arcsec(z[zin]) >= KL_MIN_R50_OVER_PSF
    final = res.copy()
    final[res] = (
        compute_line_snr_total(
            f_ha[res],
            matched_filter_compactness(r_eff_arcsec[res], cosi_iso[res], z[res]),
        )
        >= KL_SNR_MIN
    )
    print(
        f'  quality {quality.sum():6d} -> z {zin.sum():6d} '
        f'-> resolvable {res.sum():6d} -> selected {final.sum():5d}  '
        f'= {final.sum()/AREA_ARCMIN2:6.3f} /arcmin^2'
    )

    print('\naudit complete')
    return 0


if __name__ == '__main__':
    sys.exit(main())
