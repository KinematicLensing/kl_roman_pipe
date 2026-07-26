#!/usr/bin/env python
"""Attribute the catalog line-SNR selection to its levers, without fitting.

Runs the population selection chain up through the line-SNR computation
(disk + z + B/T cuts, isotropic cos i redraw, matched-filter compactness,
per-exposure SNR) ONCE, then sweeps the yield over three levers:

  * depth       : per-pass vs coadded (SNR scales by sqrt(n_coadd_passes))
  * SNR cut     : a grid of minimum-SNR thresholds
  * compactness : the matched-filter extended-source penalty on vs off
                  (off = point-source SNR, C = 1)

Also prints the coadded-SNR histogram over the threshold grid so the
high-SNR bin populations (which a faithful draw leaves sparse) are visible.

Read-only diagnostic: it does NOT change the pipeline selection or paint
anything. Use it to decide the production depth/threshold before wiring
them into the spec.

Must run under a Python with kl_pipe importable (Vista: $KLPIPE_PYTHON in
an idev session). For the v1 bank, pass --download flagship2_v1.

Usage:
    $KLPIPE_PYTHON scripts/selection_sensitivity.py <spec.yaml> \\
        [--download NAME] [--n-coadd-passes 4]
"""

import argparse
import re
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from kl_pipe.ensemble.population import (
    F_LIM_PER_PASS_CGS,
    F_LIM_NSIGMA,
    _draw_geometry,
    compute_line_snr_per_pass,
    load_flagship2_catalog,
    matched_filter_compactness,
    preprocess,
)
from kl_pipe.ensemble.spec import EnsembleSpec

SNR_GRID = (5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0)


def _sky_area_deg2(download: str, data_dir: Path) -> Optional[float]:
    spec_path = data_dir / f'{download}.yaml'
    if not spec_path.exists():
        return None
    cuts = yaml.safe_load(spec_path.read_text()).get('cuts', [])
    spans = {}
    for cut in cuts:
        m = re.search(
            r'(ra|dec)_gal\s+BETWEEN\s+([-\d.]+)\s+AND\s+([-\d.]+)',
            cut,
            re.IGNORECASE,
        )
        if m:
            spans[m.group(1).lower()] = abs(float(m.group(3)) - float(m.group(2)))
    if 'ra' in spans and 'dec' in spans:
        return spans['ra'] * spans['dec']
    return None


def _load_spec(spec_path: str, download: Optional[str]) -> EnsembleSpec:
    if download is None:
        return EnsembleSpec.from_yaml(Path(spec_path))
    d = yaml.safe_load(Path(spec_path).read_text())
    d['population']['catalog']['download'] = download
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as handle:
        yaml.safe_dump(d, handle)
        tmp = handle.name
    return EnsembleSpec.from_yaml(Path(tmp))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('spec', help='catalog-mode ensemble spec YAML')
    ap.add_argument('--download', default=None, help='override catalog bank')
    ap.add_argument(
        '--n-coadd-passes',
        type=int,
        default=4,
        help='coadded depth = per-pass / sqrt(n); HLWAS reference = 4 rolls',
    )
    args = ap.parse_args()

    spec = _load_spec(args.spec, args.download)
    cp = spec.catalog_population
    data_dir = Path(cp.catalog_data_dir)

    # selection chain up to the SNR computation (mirrors build_population)
    raw = load_flagship2_catalog(cp.catalog_download, data_dir)
    pre = preprocess(raw, cp)
    n_disk = len(pre)

    z = pre['true_redshift_gal'].to_numpy(dtype=np.float64)
    zmask = (z >= cp.z_range[0]) & (z <= cp.z_range[1])
    pre = pre.loc[zmask].reset_index(drop=True)
    if cp.bulge_fraction_max is not None:
        bfmask = pre['bulge_fraction'].to_numpy() <= cp.bulge_fraction_max
        pre = pre.loc[bfmask].reset_index(drop=True)
    n_pool = len(pre)  # disks passing z + B/T, before any SNR cut

    cosi, _ = _draw_geometry(
        spec.seed,
        pre['halo_id'].to_numpy(),
        pre['galaxy_id'].to_numpy(),
        cp.cosi_range,
    )
    reff = pre['disk_r50'].to_numpy(dtype=np.float64)
    zz = pre['true_redshift_gal'].to_numpy(dtype=np.float64)
    f_line = pre['f_line_cgs'].to_numpy()

    compact = matched_filter_compactness(reff, cosi, zz)
    ones = np.ones_like(compact)
    sqrt_n = np.sqrt(args.n_coadd_passes)

    # four (depth, compactness) SNR arrays
    variants = {
        ('per-pass', 'on '): compute_line_snr_per_pass(f_line, compact),
        ('per-pass', 'off'): compute_line_snr_per_pass(f_line, ones),
        ('coadded ', 'on '): compute_line_snr_per_pass(f_line, compact) * sqrt_n,
        ('coadded ', 'off'): compute_line_snr_per_pass(f_line, ones) * sqrt_n,
    }

    area = _sky_area_deg2(cp.catalog_download, data_dir)

    print(f"catalog        : {cp.catalog_download}")
    print(f"n_disk (2-comp): {n_disk:,}")
    print(
        f"pool (z + B/T) : {n_pool:,}"
        + (f"   ({n_pool / area:.0f}/deg^2)" if area else "")
    )
    print(
        f"F_LIM (per-pass): {F_LIM_PER_PASS_CGS:.3e} erg/s/cm^2 @ {F_LIM_NSIGMA:.0f}-sigma"
    )
    print(
        f"n_coadd_passes : {args.n_coadd_passes}  (coadded SNR = per-pass x {sqrt_n:.2f})"
    )
    print(
        f"median compactness C: {np.median(compact):.3f} "
        f"(range {compact.min():.3f}-{compact.max():.3f})\n"
    )

    header = "depth     C    " + "".join(f"  >={int(s):>4d}" for s in SNR_GRID)
    print(header)
    print("-" * len(header))
    for (depth, comp), snr in variants.items():
        counts = [int((snr >= s).sum()) for s in SNR_GRID]
        print(f"{depth}  {comp}  " + "".join(f"  {c:>6d}" for c in counts))

    if area:
        print("\nyield per deg^2 (same rows / area):")
        print(header)
        print("-" * len(header))
        for (depth, comp), snr in variants.items():
            dens = [(snr >= s).sum() / area for s in SNR_GRID]
            print(f"{depth}  {comp}  " + "".join(f"  {d:>6.1f}" for d in dens))

    # coadded, compactness-on SNR histogram (the production-like case)
    snr_c = variants[('coadded ', 'on ')]
    edges = list(SNR_GRID) + [np.inf]
    print("\ncoadded + compactness-on SNR histogram (faithful-draw bin populations):")
    for lo, hi in zip(edges[:-1], edges[1:]):
        n = int(((snr_c >= lo) & (snr_c < hi)).sum())
        label = (
            f"[{int(lo):>3d}, {int(hi):>3d})"
            if np.isfinite(hi)
            else f"[{int(lo):>3d}, inf)"
        )
        print(f"  {label}: {n:,}")


if __name__ == '__main__':
    main()
