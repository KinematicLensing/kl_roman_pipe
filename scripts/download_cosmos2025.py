"""Download COSMOS2025 (COSMOS-Web master catalog) split FITS sections.

Static-file downloads from the COSMOS2025 portal with sha256 provenance
sidecars, mirroring the CosmoHub downloader's discipline. Re-running is
idempotent: a file whose sidecar records a matching URL and sha256 is
skipped. Each download is validated against the published master-catalog
row count before being accepted.

The files are world-readable; the portal's registration form only gates
the HTML download page. Please register once at
https://cosmos2025.iap.fr/catalog_download.html anyway (it is how the
team tracks usage), and cite Shuntov et al. (2025, arXiv:2506.03243) and
Casey et al. (2023, ApJ 954, 31) in any publication.

Usage:
    python scripts/download_cosmos2025.py data/cosmos2025
    python scripts/download_cosmos2025.py data/cosmos2025 --version v1 --files lephare
    python scripts/download_cosmos2025.py data/cosmos2025 --force
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from astropy.io import fits

BASE_URL = 'https://cosmos2025.iap.fr/data/catalog'
SECTIONS = ('photom_primary', 'lephare', 'cigale')
VERSIONS = ('v1', 'v1.1')
# all COSMOS2025 master-catalog sections are row-matched with exactly this
# many rows (Shuntov et al. 2025); a section with any other count is not the
# catalog we think it is
EXPECTED_ROWS = 784016
CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(CHUNK), b''):
            h.update(chunk)
    return h.hexdigest()


def already_downloaded(out_path: Path, sidecar_path: Path, url: str) -> bool:
    if not (out_path.exists() and sidecar_path.exists()):
        return False
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    if sidecar.get('url') != url:
        print(f"  sidecar URL changed for {out_path.name}; re-downloading")
        return False
    actual = sha256_file(out_path)
    if actual != sidecar.get('sha256'):
        raise RuntimeError(
            f"{out_path}: sha256 mismatch vs sidecar ({actual} != "
            f"{sidecar.get('sha256')}); file corrupted or modified -- "
            f"delete it and re-run, or pass --force"
        )
    return True


def validate_rows(path: Path) -> int:
    with fits.open(path, memmap=True) as hdul:
        n_rows = hdul[1].header.get('NAXIS2')
    if n_rows != EXPECTED_ROWS:
        raise RuntimeError(
            f"{path}: HDU1 has {n_rows} rows, expected {EXPECTED_ROWS} "
            f"(COSMOS2025 master-catalog sections are row-matched); the "
            f"download is truncated or the file is not what we expect"
        )
    return n_rows


def download_one(name: str, version: str, out_dir: Path, force: bool) -> None:
    filename = f'COSMOSWeb_mastercatalog_{version}_{name}.fits'
    url = f'{BASE_URL}/{filename}'
    out_path = out_dir / filename
    sidecar_path = out_dir / f'{filename}.provenance.json'

    if not force and already_downloaded(out_path, sidecar_path, url):
        print(f"{filename} up to date (sha256 verified); nothing to do")
        return

    print(f"downloading {url} ...")
    ts_started = datetime.now(timezone.utc).isoformat()
    tmp = out_path.with_suffix(out_path.suffix + '.tmp')
    with requests.get(url, stream=True, timeout=300) as r:
        if r.status_code != 200:
            raise RuntimeError(f"{url}: HTTP {r.status_code}")
        expected_bytes = int(r.headers.get('content-length', 0))
        last_modified = r.headers.get('last-modified')
        done = 0
        with open(tmp, 'wb') as f:
            for chunk in r.iter_content(CHUNK):
                f.write(chunk)
                done += len(chunk)
                if expected_bytes and done % (256 * CHUNK) < CHUNK:
                    print(f"  {done/1e9:.2f} / {expected_bytes/1e9:.2f} GB")
    if expected_bytes and tmp.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"{filename}: got {tmp.stat().st_size} bytes, server advertised "
            f"{expected_bytes}; incomplete download -- re-run"
        )
    n_rows = validate_rows(tmp)
    tmp.rename(out_path)

    sidecar = {
        'file': filename,
        'url': url,
        'version': version,
        'section': name,
        'server_last_modified': last_modified,
        'ts_downloaded': ts_started,
        'local_size_bytes': out_path.stat().st_size,
        'n_rows': n_rows,
        'sha256': sha256_file(out_path),
    }
    with open(sidecar_path, 'w') as f:
        json.dump(sidecar, f, indent=1)
    print(
        f"wrote {out_path} ({sidecar['local_size_bytes']/1e9:.2f} GB, "
        f"{n_rows} rows)"
    )
    print(f"wrote {sidecar_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out_dir', type=Path, help='download directory')
    ap.add_argument('--version', choices=VERSIONS, default='v1.1')
    ap.add_argument(
        '--files',
        nargs='+',
        choices=SECTIONS,
        default=list(SECTIONS),
        help='catalog sections to download (default: all)',
    )
    ap.add_argument(
        '--force', action='store_true', help='re-download even if up to date'
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.files:
        download_one(name, args.version, args.out_dir, args.force)
    return 0


if __name__ == '__main__':
    sys.exit(main())
