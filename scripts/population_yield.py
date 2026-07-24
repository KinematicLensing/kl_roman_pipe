#!/usr/bin/env python
"""Report the catalog-mode selection yield for an ensemble spec.

Runs ``build_population``'s selection chain only (no fits): prints the
per-stage kill counts, the total selected count, and the yield per deg^2
(sky area read from the catalog query-spec's ra/dec cuts). Use to size a
campaign against a catalog bank without launching any fits.

Must run under a Python with kl_pipe importable (on Vista: $KLPIPE_PYTHON,
inside an idev session -- the login base env has no jax).

Usage:
    $KLPIPE_PYTHON scripts/population_yield.py <spec.yaml> [--download NAME]

``--download`` overrides ``population.catalog.download`` so one spec's cuts
can be pointed at a different bank (e.g. the dev spec against the v1 bank):

    $KLPIPE_PYTHON scripts/population_yield.py \\
        configs/ensembles/flagship2_shear_dev.yaml --download flagship2_v1
"""

import argparse
import re
import tempfile
from pathlib import Path
from typing import Optional

import yaml

from kl_pipe.ensemble.population import build_population
from kl_pipe.ensemble.spec import EnsembleSpec


def _sky_area_deg2(download: str, data_dir: Path) -> Optional[float]:
    """(ra span) x (dec span) in deg^2 from the query-spec's BETWEEN cuts."""
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
    # override the catalog bank in a throwaway copy, then load normally
    d = yaml.safe_load(Path(spec_path).read_text())
    d['population']['catalog']['download'] = download
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as handle:
        yaml.safe_dump(d, handle)
        tmp = handle.name
    return EnsembleSpec.from_yaml(Path(tmp))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('spec', help='catalog-mode ensemble spec YAML')
    ap.add_argument(
        '--download',
        default=None,
        help='override population.catalog.download (e.g. flagship2_v1)',
    )
    args = ap.parse_args()

    spec = _load_spec(args.spec, args.download)
    _, meta = build_population(spec)

    data_dir = Path(spec.catalog_population.catalog_data_dir)
    area = _sky_area_deg2(meta['catalog_name'], data_dir)

    print(f"catalog        : {meta['catalog_name']}  {meta['catalog_sha256'][:12]}")
    print(f"n_raw          : {meta['n_raw']:,}")
    print(f"n_disk (2-comp): {meta['n_disk']:,}")
    for stage, n in meta['kills'].items():
        print(f"  kill {stage:16s}: {n:,}")
    print(f"n_selected     : {meta['n_selected']:,}")
    if area is not None:
        print(f"sky area       : {area:.1f} deg^2")
        print(f"yield / deg^2  : {meta['n_selected'] / area:.1f}")
    print(f"selected/disk  : {meta['n_selected'] / meta['n_disk']:.4f}")


if __name__ == '__main__':
    main()
