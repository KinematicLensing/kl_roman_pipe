"""
Ensemble pipeline CLI.

Commands
--------
expand   spec.yaml -> run directory (manifest + provenance + ledger dirs)
run      execute the campaign locally (serial or N worker subprocesses)
worker   single worker loop (the SLURM/subprocess entrypoint)
status   derived-status tally (succeeded/failed/in_progress/stale/never_run)
collate  merge per-fit results -> results.parquet + <run>_collated.parquet (+ tally)
slurm    emit submit.slurm into the run directory
reclaim  release stale claims (and optionally clear failed markers) so a
         re-run picks those fits up again
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog='python -m kl_pipe.ensemble',
        description='Ensemble fitting pipeline driver',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    p_expand = sub.add_parser('expand', help='expand a spec into a run dir')
    p_expand.add_argument('spec', type=Path, help='ensemble spec YAML')
    p_expand.add_argument(
        '--registry',
        type=Path,
        default=Path('configs/observing'),
        help='observing-config registry dir (default: configs/observing)',
    )
    p_expand.add_argument(
        '--runs-dir',
        type=Path,
        default=Path('runs'),
        help='parent dir for run outputs (default: runs/)',
    )
    p_expand.add_argument(
        '--overwrite',
        action='store_true',
        help='replace an existing run directory',
    )

    p_run = sub.add_parser('run', help='run the campaign locally')
    p_run.add_argument('--run-dir', type=Path, required=True)
    p_run.add_argument(
        '--workers',
        type=int,
        default=None,
        help='override spec dispatch.workers_per_node',
    )
    p_run.add_argument(
        '--max-fits',
        type=int,
        default=None,
        help='per-worker cap on fit attempts (dev knob)',
    )

    p_worker = sub.add_parser('worker', help='single worker loop')
    p_worker.add_argument('--run-dir', type=Path, required=True)
    p_worker.add_argument('--label', default='worker0')
    p_worker.add_argument('--max-fits', type=int, default=None)
    p_worker.add_argument('--shard-index', type=int, default=0)
    p_worker.add_argument('--shard-count', type=int, default=1)

    p_status = sub.add_parser('status', help='derived-status tally')
    p_status.add_argument('--run-dir', type=Path, required=True)

    p_collate = sub.add_parser('collate', help='merge per-fit results')
    p_collate.add_argument('--run-dir', type=Path, required=True)

    p_slurm = sub.add_parser('slurm', help='emit submit.slurm')
    p_slurm.add_argument('--run-dir', type=Path, required=True)

    p_reclaim = sub.add_parser('reclaim', help='release stale claims')
    p_reclaim.add_argument('--run-dir', type=Path, required=True)
    p_reclaim.add_argument(
        '--clear-failed',
        action='store_true',
        help='also clear failed markers so those fits re-run',
    )

    args = parser.parse_args(argv)

    if args.command == 'expand':
        from kl_pipe.ensemble.expander import expand

        run_dir = expand(
            args.spec, args.registry, args.runs_dir, overwrite=args.overwrite
        )
        print(f'expanded to {run_dir}')
        return 0

    if args.command == 'run':
        from kl_pipe.ensemble.collate import print_tally
        from kl_pipe.ensemble.dispatch import run_local

        counts = run_local(args.run_dir, workers=args.workers, max_fits=args.max_fits)
        print(f'run complete: {counts}')
        print_tally(args.run_dir)
        return 0

    if args.command == 'worker':
        from kl_pipe.ensemble.worker import worker_loop

        counts = worker_loop(
            args.run_dir,
            worker_label=args.label,
            max_fits=args.max_fits,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
        print(f'[{args.label}] done: {counts}')
        return 0 if counts['failed'] == 0 else 1

    if args.command == 'status':
        from kl_pipe.ensemble.collate import print_tally

        groups = print_tally(args.run_dir)
        return 0 if not (groups['failed'] or groups['stale']) else 1

    if args.command == 'collate':
        from kl_pipe.ensemble.collate import (
            collate_results,
            print_tally,
            write_collated_table,
        )

        results = collate_results(args.run_dir)
        print(f'collated {len(results)} rows -> results.parquet')
        collated_path = write_collated_table(args.run_dir)
        print(f'wrote analysis table -> {collated_path.name}')
        print_tally(args.run_dir)
        return 0

    if args.command == 'slurm':
        from kl_pipe.ensemble.dispatch import emit_slurm_script

        emit_slurm_script(args.run_dir)
        return 0

    if args.command == 'reclaim':
        from kl_pipe.ensemble import ledger
        from kl_pipe.ensemble.collate import run_tally

        groups = run_tally(args.run_dir)
        for fit_id in groups['stale']:
            ledger.release_claim(args.run_dir, fit_id)
            print(f'released stale claim {fit_id}')
        if args.clear_failed:
            for fit_id in groups['failed']:
                ledger.clear_failed(args.run_dir, fit_id)
                ledger.release_claim(args.run_dir, fit_id)
                print(f'cleared failed marker {fit_id}')
        return 0

    raise ValueError(f'unhandled command {args.command}')


if __name__ == '__main__':
    sys.exit(main())
