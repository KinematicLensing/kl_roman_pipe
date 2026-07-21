"""Download the Euclid Q1 Halpha bright sample from IRSA TAP (no auth).

This is the real-data validation anchor for the Flagship2-backed ensemble:
all rank-0 Halpha line detections with flux > 2e-16 erg/s/cm2 and
matched-filter line SNR >= 5, LEFT JOINed to MER morphology (VIS Sersic
fits) and the SPE galaxy-candidate redshift catalog. One synchronous TAP
query re-creates the sample exactly; a provenance sidecar records the
verbatim ADQL and sha256.

Usage:
    python scripts/download_q1.py            # writes data/q1/q1_halpha_bright.csv
    python scripts/download_q1.py --force
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import requests

TAP_SYNC = 'https://irsa.ipac.caltech.edu/TAP/sync'
OUT_DIR = Path(__file__).resolve().parent.parent / 'data' / 'q1'
OUT_NAME = 'q1_halpha_bright'

ADQL = (
    "SELECT lines.object_id, "
    "lines.spe_line_flux_gf AS flux, lines.spe_line_flux_err_gf AS flux_err, "
    "lines.spe_line_snr_gf AS lsnr, lines.spe_line_ew_gf AS ew, "
    "galaxy.spe_z, galaxy.spe_z_prob, "
    "morph.sersic_sersic_vis_radius AS reff, "
    "morph.sersic_sersic_vis_axis_ratio AS q, "
    "morph.sersic_sersic_vis_index AS nser, "
    "morph.sersic_visnir_flags AS mflags "
    "FROM euclid_q1_spe_lines_line_features AS lines "
    "JOIN euclid_q1_spectro_zcatalog_spe_galaxy_candidates AS galaxy "
    "ON lines.object_id = galaxy.object_id AND lines.spe_rank = galaxy.spe_rank "
    "LEFT JOIN euclid_q1_mer_morphology AS morph "
    "ON lines.object_id = morph.object_id "
    "WHERE lines.spe_line_name = 'Halpha' "
    "AND lines.spe_rank = 0 "
    "AND lines.spe_line_flux_gf > 2E-16 "
    "AND lines.spe_line_snr_gf >= 5"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--force', action='store_true', help='re-download even if present')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f'{OUT_NAME}.csv'
    sidecar_path = OUT_DIR / f'{OUT_NAME}.provenance.json'

    if out_path.exists() and sidecar_path.exists() and not args.force:
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        if sidecar.get('adql') == ADQL and sha256_file(out_path) == sidecar.get(
            'sha256'
        ):
            print(f"{out_path} up to date (sha256 verified); nothing to do")
            return 0

    print("querying IRSA TAP (sync; typically ~1 min) ...")
    r = requests.post(
        TAP_SYNC,
        data={'QUERY': ADQL, 'FORMAT': 'CSV', 'LANG': 'ADQL'},
        timeout=1800,
        stream=True,
    )
    if r.status_code != 200:
        raise RuntimeError(f"IRSA TAP failed: HTTP {r.status_code}: {r.text[:500]}")
    tmp = out_path.with_suffix('.csv.tmp')
    with open(tmp, 'wb') as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    n_rows = sum(1 for _ in open(tmp)) - 1
    if n_rows <= 0:
        tmp.unlink()
        raise RuntimeError("IRSA TAP returned zero rows -- query or service issue")
    tmp.rename(out_path)

    sidecar = {
        'name': OUT_NAME,
        'service': TAP_SYNC,
        'adql': ADQL,
        'n_rows': n_rows,
        'local_size_bytes': out_path.stat().st_size,
        'sha256': sha256_file(out_path),
    }
    with open(sidecar_path, 'w') as f:
        json.dump(sidecar, f, indent=1)
    print(f"wrote {out_path} ({n_rows} rows)")
    print(f"wrote {sidecar_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
