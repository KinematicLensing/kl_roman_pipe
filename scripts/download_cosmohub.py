"""Download a named CosmoHub catalog subset defined by a query-spec YAML.

Submits an async Hive query to the CosmoHub API, polls until completion,
downloads the result, and writes a provenance sidecar (query id, verbatim
SQL, timestamps, sha256). Re-running is idempotent: if the output file
exists and its sidecar matches both the current spec's SQL and the file's
sha256, the download is skipped.

Authentication: HTTP Basic via ~/.netrc, e.g.

    machine api.cosmohub.pic.es
      login <email>
      password <password>

Usage:
    python scripts/download_cosmohub.py data/cosmohub/flagship2_dev.yaml
    python scripts/download_cosmohub.py data/cosmohub/flagship2_v1.yaml --force

CosmoHub data is CC BY-NC licensed; downloads stay local (gitignored).
See data/cosmohub/README.md for the required acknowledgement and citations.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import requests
import yaml

API = 'https://api.cosmohub.pic.es'
POLL_SECONDS = 30
TIMEOUT_SECONDS = 4 * 3600
# CosmoHub asks for up to 10 minutes between job completion and the
# result file becoming downloadable
DOWNLOAD_RETRY_SECONDS = 60
DOWNLOAD_MAX_RETRIES = 12

REQUIRED_SPEC_KEYS = ('name', 'catalog_id', 'table', 'format', 'columns', 'cuts')


def load_spec(path: Path) -> dict:
    with open(path) as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, dict):
        raise ValueError(f"{path}: query spec must be a mapping")
    missing = [k for k in REQUIRED_SPEC_KEYS if k not in spec]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")
    unknown = [k for k in spec if k not in REQUIRED_SPEC_KEYS]
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}")
    if not spec['columns'] or not isinstance(spec['columns'], list):
        raise ValueError(f"{path}: 'columns' must be a non-empty list")
    if not spec['cuts'] or not isinstance(spec['cuts'], list):
        raise ValueError(f"{path}: 'cuts' must be a non-empty list")
    return spec


def build_sql(spec: dict) -> str:
    cols = ', '.join(spec['columns'])
    where = ' AND '.join(f"({c})" for c in spec['cuts'])
    return f"SELECT {cols} FROM {spec['table']} WHERE {where}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def already_downloaded(out_path: Path, sidecar_path: Path, sql: str) -> bool:
    if not (out_path.exists() and sidecar_path.exists()):
        return False
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    if sidecar.get('sql') != sql:
        print(f"spec SQL changed since last download of {out_path.name}")
        return False
    actual = sha256_file(out_path)
    if actual != sidecar.get('sha256'):
        raise RuntimeError(
            f"{out_path}: sha256 mismatch vs sidecar "
            f"({actual} != {sidecar.get('sha256')}); file corrupted or "
            f"modified -- delete it and re-run, or pass --force"
        )
    return True


def submit(session: requests.Session, sql: str, fmt: str) -> int:
    r = session.post(f'{API}/queries', json={'sql': sql, 'format': fmt})
    if r.status_code != 201:
        raise RuntimeError(
            f"query submission failed: HTTP {r.status_code}: {r.text[:500]}"
        )
    return r.json()['id']


def poll(session: requests.Session, query_id: int) -> dict:
    deadline = time.time() + TIMEOUT_SECONDS
    while time.time() < deadline:
        r = session.get(f'{API}/queries')
        r.raise_for_status()
        matches = [q for q in r.json() if q['id'] == query_id]
        if not matches:
            raise RuntimeError(f"query {query_id} vanished from /queries")
        q = matches[0]
        status = q['status']
        if status == 'SUCCEEDED':
            return q
        if status in ('FAILED', 'ERROR', 'KILLED'):
            raise RuntimeError(f"query {query_id} terminal status {status}: {q}")
        print(f"  query {query_id}: {status} ...")
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"query {query_id} still running after {TIMEOUT_SECONDS}s")


def download(session: requests.Session, record: dict, out_path: Path) -> None:
    url = record.get('download_results')
    if not url:
        raise RuntimeError(f"query {record['id']} has no download_results URL")
    tmp = out_path.with_suffix(out_path.suffix + '.tmp')
    for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
        r = session.get(url, stream=True)
        if r.status_code == 200:
            with open(tmp, 'wb') as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            if tmp.stat().st_size == 0:
                tmp.unlink()
                raise RuntimeError(
                    f"query {record['id']} returned an empty result file; "
                    f"the SQL matched zero rows -- check the spec's cuts"
                )
            tmp.rename(out_path)
            return
        print(
            f"  download attempt {attempt}: HTTP {r.status_code}; "
            f"retrying in {DOWNLOAD_RETRY_SECONDS}s"
        )
        time.sleep(DOWNLOAD_RETRY_SECONDS)
    raise RuntimeError(f"result download failed after {DOWNLOAD_MAX_RETRIES} attempts")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('spec', type=Path, help='query-spec YAML path')
    ap.add_argument(
        '--force', action='store_true', help='re-download even if up to date'
    )
    args = ap.parse_args()

    spec = load_spec(args.spec)
    sql = build_sql(spec)
    fmt = spec['format']
    ext = {'parquet': '.parquet', 'csv.bz2': '.csv.bz2', 'fits': '.fits'}.get(fmt)
    if ext is None:
        raise ValueError(f"unsupported format '{fmt}'")

    out_dir = args.spec.parent
    out_path = out_dir / f"{spec['name']}{ext}"
    sidecar_path = out_dir / f"{spec['name']}.provenance.json"

    if not args.force and already_downloaded(out_path, sidecar_path, sql):
        print(f"{out_path} up to date (sha256 verified); nothing to do")
        return 0

    session = requests.Session()  # HTTP Basic auth resolved via ~/.netrc
    user = session.get(f'{API}/user')
    if user.status_code != 200:
        raise RuntimeError(
            f"CosmoHub auth failed (HTTP {user.status_code}). Add credentials "
            f"to ~/.netrc for machine api.cosmohub.pic.es"
        )

    print(f"submitting {spec['name']} ({fmt}) ...")
    query_id = submit(session, sql, fmt)
    print(f"  submitted as query {query_id}; polling every {POLL_SECONDS}s")
    record = poll(session, query_id)
    size_mb = (record.get('size') or 0) / 1e6
    print(f"  SUCCEEDED (server size {size_mb:.1f} MB); downloading ...")
    download(session, record, out_path)

    sidecar = {
        'name': spec['name'],
        'query_id': query_id,
        'catalog_id': spec['catalog_id'],
        'table': spec['table'],
        'format': fmt,
        'sql': sql,
        'ts_submitted': record.get('ts_submitted'),
        'ts_finished': record.get('ts_finished'),
        'server_size_bytes': record.get('size'),
        'local_size_bytes': out_path.stat().st_size,
        'sha256': sha256_file(out_path),
        'spec_file': args.spec.name,
        'spec_sha256': hashlib.sha256(args.spec.read_bytes()).hexdigest(),
    }
    with open(sidecar_path, 'w') as f:
        json.dump(sidecar, f, indent=1)
    print(f"wrote {out_path} ({sidecar['local_size_bytes']/1e6:.1f} MB)")
    print(f"wrote {sidecar_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
