"""
Filesystem-derived, lock-free status ledger.

Per-fit status is a pure function of which markers exist under
``run_dir/status/``:

- ``claims/<fit_id>/``  atomic-mkdir claim (first-wins across workers/nodes);
  contains ``claim.json`` with {backend, hostname, pid, slurm_job_id, ts}.
- ``done/<fit_id>``     written after the per-fit result file is durably in
  place (write-tmp + atomic rename happens first).
- ``failed/<fit_id>``   written on a caught fit error; contains the message.

Derived status: ``succeeded`` (done) / ``failed`` (failed, no done) /
``in_progress`` (claim, claimant alive, no done) / ``stale`` (claim,
claimant dead, no done) / ``never_run`` (no markers).

Liveness: a local claim is alive when its pid exists on this host; a SLURM
claim when its job id appears in squeue. Liveness that cannot be determined
(other host, no squeue) falls back to the claim's age vs the spec's
``max_fit_walltime_min`` -- never reclaim a fit whose owner may be running.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

STATUSES = ('succeeded', 'failed', 'in_progress', 'stale', 'never_run')


def _status_dir(run_dir: Path, kind: str) -> Path:
    return Path(run_dir) / 'status' / kind


def try_claim(run_dir: Path, fit_id: str) -> bool:
    """Atomically claim a fit. Returns False if already claimed."""
    claim = _status_dir(run_dir, 'claims') / fit_id
    try:
        claim.mkdir()
    except FileExistsError:
        return False
    meta = {
        'backend': 'slurm' if os.environ.get('SLURM_JOB_ID') else 'local',
        'hostname': socket.gethostname(),
        'pid': os.getpid(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'ts': time.time(),
    }
    (claim / 'claim.json').write_text(json.dumps(meta) + '\n')
    return True


def release_claim(run_dir: Path, fit_id: str) -> None:
    """Remove a claim (stale recovery or voluntary release on failure)."""
    claim = _status_dir(run_dir, 'claims') / fit_id
    if claim.exists():
        shutil.rmtree(claim, ignore_errors=True)


def mark_done(run_dir: Path, fit_id: str) -> None:
    (_status_dir(run_dir, 'done') / fit_id).write_text(
        json.dumps({'ts': time.time()}) + '\n'
    )


def mark_failed(run_dir: Path, fit_id: str, message: str) -> None:
    (_status_dir(run_dir, 'failed') / fit_id).write_text(message + '\n')


def clear_failed(run_dir: Path, fit_id: str) -> None:
    path = _status_dir(run_dir, 'failed') / fit_id
    if path.exists():
        path.unlink()


def is_done(run_dir: Path, fit_id: str) -> bool:
    return (_status_dir(run_dir, 'done') / fit_id).exists()


def read_claim(run_dir: Path, fit_id: str) -> Optional[dict]:
    path = _status_dir(run_dir, 'claims') / fit_id / 'claim.json'
    if not path.exists():
        # claim dir may exist with the meta not yet flushed; treat as live-ish
        if (_status_dir(run_dir, 'claims') / fit_id).exists():
            return {}
        return None
    return json.loads(path.read_text())


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _squeue_job_ids() -> Optional[set]:
    """Active SLURM job ids, or None when squeue is unavailable."""
    if shutil.which('squeue') is None:
        return None
    try:
        out = subprocess.run(
            ['squeue', '-h', '-o', '%A'],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


def claim_is_alive(
    claim: dict,
    max_fit_walltime_min: float,
    squeue_ids: Optional[set] = None,
) -> bool:
    """Whether a claim's owner is (or may still be) running.

    Deterministic liveness where possible (local pid on this host, SLURM job
    in squeue); otherwise the age fallback: alive until the claim is older
    than max_fit_walltime_min.
    """
    if not claim:
        return True  # claim meta not yet flushed; err on the safe side
    if claim.get('backend') == 'slurm':
        job_id = str(claim.get('slurm_job_id'))
        if squeue_ids is not None:
            return job_id in squeue_ids
    elif claim.get('hostname') == socket.gethostname():
        return _pid_alive(int(claim['pid']))
    # indeterminate (other host / no squeue): age fallback
    age_min = (time.time() - float(claim.get('ts', 0.0))) / 60.0
    return age_min < max_fit_walltime_min


def fit_status(
    run_dir: Path,
    fit_id: str,
    max_fit_walltime_min: float,
    squeue_ids: Optional[set] = None,
) -> str:
    if is_done(run_dir, fit_id):
        return 'succeeded'
    if (_status_dir(run_dir, 'failed') / fit_id).exists():
        return 'failed'
    claim = read_claim(run_dir, fit_id)
    if claim is None:
        return 'never_run'
    if claim_is_alive(claim, max_fit_walltime_min, squeue_ids):
        return 'in_progress'
    return 'stale'


def tally(
    run_dir: Path,
    fit_ids,
    max_fit_walltime_min: float,
) -> Dict[str, list]:
    """Derived status for every fit, grouped: {status: [fit_id, ...]}."""
    squeue_ids = _squeue_job_ids()
    groups: Dict[str, list] = {s: [] for s in STATUSES}
    for fit_id in fit_ids:
        groups[fit_status(run_dir, fit_id, max_fit_walltime_min, squeue_ids)].append(
            fit_id
        )
    return groups
