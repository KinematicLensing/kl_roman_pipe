"""Build the joined cosmos25 catalog parquet from the downloaded sections.

Joins the public COSMOS2025 v1 sections (photom_primary, lephare; see
scripts/download_cosmos2025.py) with the private painted emission-line
section by row order -- the COSMOS2025 master-catalog convention -- after
gating that the join is valid: equal row counts everywhere, unique ``id``,
and the painted ``redshift`` column bitwise-equal to LePhare ``zfinal``
(which also pins the section version to the one the painting used).

Output is the exact raw schema of the ``cosmos25`` catalog adapter
(kl_pipe/ensemble/catalogs/cosmos25.py) plus a provenance sidecar carrying
the output sha256 (mandatory for ``load_catalog``) and the input files'
sha256s. The parquet contains the private painted columns, so it lives
under data/cosmos2025/private/ -- never commit or redistribute it.

Usage:
    python scripts/build_cosmos25_catalog.py
    python scripts/build_cosmos25_catalog.py --data-dir data/cosmos2025 --force
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kl_pipe.ensemble.catalogs.cosmos25 import (  # noqa: E402
    COSMOS25_COLUMNS,
    COSMOS25_SOURCE_VERSION,
)

OUTPUT_NAME = 'cosmos25_v1'
EXPECTED_ROWS = 784016

PHOTOM_COLUMNS = (
    'id',
    'ra',
    'dec',
    'warn_flag',
    'mag_model_f150w',
    'snr_f150w',
    'radius_sersic',
    'radius_sersic_err',
    'sersic',
    'axratio_sersic',
    'e1_err',
    'e2_err',
    'flux_model_f115w',
    'flux_model_f150w',
    'flux_model_f277w',
)
LEPHARE_COLUMNS = ('zfinal', 'type', 'ebv_minchi2', 'mass_med', 'sfr_med')
MOCK_COLUMNS = (
    'F_Ha',
    'F_OII',
    'F_OIII',
    'lambda_Ha_obs',
    'lambda_OII_obs',
    'lambda_OIII_obs',
    'redshift',
    'sfr_young',
    'log_U',
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_section(path: Path, columns) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run 'make download-cosmos2025' (public "
            f"sections) or obtain the private painted section"
        )
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        if len(data) != EXPECTED_ROWS:
            raise RuntimeError(f"{path}: {len(data)} rows != {EXPECTED_ROWS}")
        # native byte order for pandas/parquet
        return {
            c: np.asarray(data[c]).astype(np.asarray(data[c]).dtype.newbyteorder('='))
            for c in columns
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, default=REPO_ROOT / 'data/cosmos2025')
    ap.add_argument('--force', action='store_true', help='rebuild even if current')
    args = ap.parse_args()

    ver = COSMOS25_SOURCE_VERSION
    photom_path = args.data_dir / f'COSMOSWeb_mastercatalog_{ver}_photom_primary.fits'
    lephare_path = args.data_dir / f'COSMOSWeb_mastercatalog_{ver}_lephare.fits'
    mock_path = (
        args.data_dir
        / 'private'
        / f'COSMOSWeb_mastercatalog_{ver}_mock_emission_lines.fits'
    )
    out_path = args.data_dir / 'private' / f'{OUTPUT_NAME}.parquet'
    sidecar_path = args.data_dir / 'private' / f'{OUTPUT_NAME}.provenance.json'

    input_shas = {
        p.name: sha256_file(p) for p in (photom_path, lephare_path, mock_path)
    }
    if not args.force and out_path.exists() and sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text())
        if sidecar.get('input_sha256') == input_shas and sidecar.get(
            'sha256'
        ) == sha256_file(out_path):
            print(f"{out_path} up to date (input + output sha256 verified)")
            return 0

    photom = load_section(photom_path, PHOTOM_COLUMNS)
    lephare = load_section(lephare_path, LEPHARE_COLUMNS)
    mock = load_section(mock_path, MOCK_COLUMNS)

    # join gates: the sections carry no shared key, so row order IS the join;
    # these checks are what make that defensible
    ids = photom['id'].astype(np.int64)
    if len(np.unique(ids)) != EXPECTED_ROWS:
        raise RuntimeError("photom 'id' is not unique; row-order join unsafe")
    z_eq = (mock['redshift'] == lephare['zfinal']) | (
        np.isnan(mock['redshift']) & np.isnan(lephare['zfinal'])
    )
    if not z_eq.all():
        raise RuntimeError(
            f"painted 'redshift' != {ver} 'zfinal' on {(~z_eq).sum()} rows; "
            f"the painted section was built on a different catalog version -- "
            f"do NOT join these files"
        )

    df = pd.DataFrame({**photom, **lephare, **mock})
    df['id'] = ids
    missing = [c for c in COSMOS25_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in COSMOS25_COLUMNS]
    if missing or extra:
        raise RuntimeError(
            f"builder/adapter schema drift: missing {missing}, extra {extra}"
        )
    df = df[list(COSMOS25_COLUMNS)]
    df.to_parquet(out_path, index=False)

    sidecar = {
        'name': OUTPUT_NAME,
        'source_version': ver,
        'n_rows': int(len(df)),
        'columns': list(COSMOS25_COLUMNS),
        'input_sha256': input_shas,
        'join': 'row-order; gated on unique id and painted redshift == zfinal',
        'ts_built': datetime.now(timezone.utc).isoformat(),
        'builder': 'scripts/build_cosmos25_catalog.py',
        'local_size_bytes': out_path.stat().st_size,
        'sha256': sha256_file(out_path),
        'private': 'contains the painted emission-line columns; do not redistribute',
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=1))
    print(
        f"wrote {out_path} ({sidecar['local_size_bytes']/1e6:.1f} MB, {len(df)} rows)"
    )
    print(f"wrote {sidecar_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
