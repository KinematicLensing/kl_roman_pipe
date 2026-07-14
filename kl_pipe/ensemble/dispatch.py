"""
Dispatch backends.

Local backend (first-class): runs the same claim -> fit -> write worker loop
as any cluster node, either in-process (workers=1) or as N concurrent worker
subprocesses. The SLURM layer is just a launcher around the identical worker
entrypoint (``python -m kl_pipe.ensemble worker --run-dir ...``); the worker
itself never imports or assumes anything SLURM-specific.

SLURM backend: emits a turnkey ``submit.slurm`` job-array script into the run
directory. Static mode partitions manifest rows by array index; dynamic mode
lets every task claim from the shared ledger (better load balance).
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from kl_pipe.ensemble.expander import load_run
from kl_pipe.ensemble.worker import worker_loop


def run_local(
    run_dir: Path,
    workers: Optional[int] = None,
    max_fits: Optional[int] = None,
) -> Dict[str, int]:
    """
    Run the campaign locally.

    Parameters
    ----------
    run_dir : Path
        Expanded run directory.
    workers : int, optional
        Concurrent worker processes; defaults to the spec's
        dispatch.workers_per_node. 1 = in-process serial loop.
    max_fits : int, optional
        Per-worker cap on fit attempts (dev knob).

    Returns
    -------
    dict
        Aggregate {'succeeded', 'failed', 'skipped'} counts (in-process mode)
        or per-worker exit summary (subprocess mode).
    """
    run_dir = Path(run_dir)
    spec, _, _ = load_run(run_dir)
    n_workers = workers if workers is not None else spec.workers_per_node

    if n_workers == 1:
        return worker_loop(run_dir, worker_label='local0', max_fits=max_fits)

    # slice HBM across workers (same rule as the emitted submit.slurm):
    # each JAX process otherwise preallocates 75% of the device and N > 1
    # workers OOM immediately. An explicit caller setting wins.
    child_env = os.environ.copy()
    if 'XLA_PYTHON_CLIENT_MEM_FRACTION' not in child_env:
        frac = round(0.85 / n_workers, 3)
        child_env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(frac)
        print(
            f'[dispatch] XLA_PYTHON_CLIENT_MEM_FRACTION={frac} per worker '
            f'({n_workers} workers share the device)'
        )

    procs = []
    for w in range(n_workers):
        cmd = [
            sys.executable,
            '-m',
            'kl_pipe.ensemble',
            'worker',
            '--run-dir',
            str(run_dir),
            '--label',
            f'local{w}',
        ]
        if max_fits is not None:
            cmd += ['--max-fits', str(max_fits)]
        procs.append(subprocess.Popen(cmd, env=child_env))
        # stagger jax inits (simultaneous startups spike threads/allocs)
        time.sleep(0.5)

    counts = {'workers': n_workers, 'nonzero_exits': 0}
    for w, proc in enumerate(procs):
        rc = proc.wait()
        if rc != 0:
            counts['nonzero_exits'] += 1
            print(f'[dispatch] worker local{w} exited with code {rc}')
    if counts['nonzero_exits']:
        raise RuntimeError(
            f"{counts['nonzero_exits']}/{n_workers} local workers exited "
            f"nonzero -- check worker logs above"
        )
    return counts


_SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH -J {run_name}
#SBATCH -p {queue}{account_line}
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t {walltime}
#SBATCH -a 0-{array_max}
#SBATCH -o {run_dir}/slurm-%A_%a.out
#SBATCH -e {run_dir}/slurm-%A_%a.err

# Emitted by kl_pipe.ensemble.dispatch -- one node per array task; each task
# runs {workers} concurrent worker process(es) sharing the claim ledger.
#
# Environment: KLPIPE_PYTHON is the full launcher command for a python that
# can import kl_pipe + jax (e.g. the containerized Vista launcher from
# docs/ensemble_workflow.md). If it was exported when this script was
# emitted, it is baked in below; an export in the sbatch environment
# overrides the baked value.

set -u

{launcher_block}

PY="${{KLPIPE_PYTHON:-python}}"

# preflight: one loud failure instead of {workers} import tracebacks
if ! $PY -c "import kl_pipe, jax"; then
  echo "ERROR: launcher '$PY' cannot import kl_pipe + jax." >&2
  echo "Export KLPIPE_PYTHON (containerized launcher, see" >&2
  echo "docs/ensemble_workflow.md) and re-emit or resubmit." >&2
  exit 1
fi

export JAX_COMPILATION_CACHE_DIR="${{SCRATCH:-$HOME}}/jax_cache"

# GPU packing: workers share the device via bounded HBM slices (harmless on
# CPU-only nodes -- the variable is ignored without a GPU backend)
{mem_fraction_line}

for w in $(seq 0 {workers_minus_one}); do
  $PY -m kl_pipe.ensemble worker \\
    --run-dir {run_dir} \\
    --label "task${{SLURM_ARRAY_TASK_ID}}_w${{w}}"{shard_args} &
  sleep 2  # stagger jax inits (simultaneous startup spikes threads/allocs)
done
wait
"""


def emit_slurm_script(run_dir: Path) -> Path:
    """Write a turnkey submit.slurm for the campaign into the run dir."""
    run_dir = Path(run_dir).resolve()
    spec, _, manifest = load_run(run_dir)

    n_fits = len(manifest)
    per_fit_min = spec.max_fit_walltime_min
    fits_per_task = max(
        1,
        math.floor(spec.workers_per_node * spec.target_task_walltime_min / per_fit_min),
    )
    n_tasks = math.ceil(n_fits / fits_per_task)
    # walltime: target + one max-length fit of slack, rounded up to 5 min
    walltime_min = int(
        math.ceil((spec.target_task_walltime_min + per_fit_min) / 5.0) * 5
    )
    hh, mm = divmod(walltime_min, 60)

    if spec.mode == 'static':
        # strided partition over (task, worker) shards; the claim ledger
        # still guards against any accidental overlap
        total_shards = n_tasks * spec.workers_per_node
        shard_args = (
            ' \\\n    --shard-index '
            f'"$((SLURM_ARRAY_TASK_ID * {spec.workers_per_node} + w))"'
            f' \\\n    --shard-count {total_shards}'
        )
    else:
        shard_args = ''

    if spec.workers_per_node > 1:
        frac = round(0.85 / spec.workers_per_node, 3)
        mem_fraction_line = f'export XLA_PYTHON_CLIENT_MEM_FRACTION={frac}'
    else:
        mem_fraction_line = '# (single worker: full device)'

    # bake the launcher into the script when available: sbatch does not see
    # the emitting (idev) shell's exports, so relying on a manual edit or on
    # the submit shell's environment silently falls back to the host python.
    launcher = os.environ.get('KLPIPE_PYTHON', '').strip()
    if '"' in launcher:
        raise ValueError(
            "KLPIPE_PYTHON contains a double quote -- cannot bake it into "
            "submit.slurm safely; simplify the launcher command"
        )
    if launcher:
        module_line = (
            'set +u; module load tacc-apptainer 2>/dev/null || true; set -u\n'
            if 'apptainer' in launcher
            else ''
        )
        launcher_block = (
            '# launcher baked from the emitting shell\'s $KLPIPE_PYTHON; an\n'
            '# export in the sbatch environment overrides it.\n'
            f'{module_line}'
            f'export KLPIPE_PYTHON="${{KLPIPE_PYTHON:-{launcher}}}"'
        )
    else:
        launcher_block = (
            '# WARNING: KLPIPE_PYTHON was not set when this script was\n'
            '# emitted -- export it in the sbatch environment or re-emit\n'
            '# from a shell where it is set (the preflight below fails loud\n'
            '# otherwise).'
        )

    script = _SLURM_TEMPLATE.format(
        run_name=spec.run_name,
        queue=spec.queue,
        account_line=(f'\n#SBATCH -A {spec.account}' if spec.account else ''),
        walltime=f'{hh:02d}:{mm:02d}:00',
        array_max=n_tasks - 1,
        run_dir=run_dir,
        workers=spec.workers_per_node,
        workers_minus_one=spec.workers_per_node - 1,
        shard_args=shard_args,
        mem_fraction_line=mem_fraction_line,
        launcher_block=launcher_block,
    )
    path = run_dir / 'submit.slurm'
    path.write_text(script)
    print(
        f'wrote {path} ({n_tasks} array tasks x {spec.workers_per_node} '
        f'workers, {n_fits} fits, walltime {hh:02d}:{mm:02d}:00)'
    )
    if launcher:
        print(f'baked KLPIPE_PYTHON launcher into script: {launcher}')
    else:
        print(
            'WARNING: KLPIPE_PYTHON not set at emission -- submit.slurm '
            'requires it in the sbatch environment'
        )
    return path
