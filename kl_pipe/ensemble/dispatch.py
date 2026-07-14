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
import subprocess
import sys
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
        procs.append(subprocess.Popen(cmd))

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
# Environment: either activate a python env with kl_pipe installed before the
# loop below, or set KLPIPE_PYTHON to a full launcher command. Containerized
# Vista example (paths from vista_kit/provision_vista.sh):
#   module load tacc-apptainer
#   export KLPIPE_PYTHON="apptainer exec --nv \\
#     -B $STOCKYARD/repos/kl_roman_pipe -B $WORK/klpipe_pipdeps -B $SCRATCH \\
#     -B $WORK/fftw3 \\
#     --env PYTHONPATH=$STOCKYARD/repos/kl_roman_pipe:$WORK/klpipe_pipdeps \\
#     $WORK/containers/jax_26.06-py3.sif python"

set -u

PY="${{KLPIPE_PYTHON:-python}}"

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
    )
    path = run_dir / 'submit.slurm'
    path.write_text(script)
    print(
        f'wrote {path} ({n_tasks} array tasks x {spec.workers_per_node} '
        f'workers, {n_fits} fits, walltime {hh:02d}:{mm:02d}:00)'
    )
    return path
