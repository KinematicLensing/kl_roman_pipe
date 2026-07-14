"""
Collate per-fit results into the run catalog + analysis table.

- ``collate_results``: concatenates ``results/<fit_id>.parquet`` into
  ``results.parquet`` (the catalog of summary rows).
- ``analysis_table``: joins the catalog onto the manifest by fit_id --
  truth + recovered per fit, the calibration module's input.
- ``print_tally``: the between-submission-rounds human checkpoint --
  succeeded / failed / in_progress / stale / never_run counts + the
  incomplete list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from kl_pipe.ensemble import ledger
from kl_pipe.ensemble.expander import load_run


def collate_results(run_dir: Path) -> pd.DataFrame:
    """Merge per-fit result files into run_dir/results.parquet."""
    run_dir = Path(run_dir)
    files = sorted((run_dir / 'results').glob('*.parquet'))
    if not files:
        raise FileNotFoundError(f"no per-fit result files under {run_dir / 'results'}")
    frames = [pd.read_parquet(f) for f in files]
    results = pd.concat(frames, ignore_index=True)
    if results['fit_id'].duplicated().any():
        dupes = results.loc[results['fit_id'].duplicated(), 'fit_id'].tolist()
        raise RuntimeError(f"duplicate fit_id in per-fit results: {dupes}")
    results.to_parquet(run_dir / 'results.parquet', index=False)
    return results


def analysis_table(run_dir: Path) -> pd.DataFrame:
    """Manifest joined with collated results on fit_id (inner join)."""
    run_dir = Path(run_dir)
    _, _, manifest = load_run(run_dir)
    results_path = run_dir / 'results.parquet'
    if not results_path.exists():
        raise FileNotFoundError(f"{results_path} missing -- run collate_results first")
    results = pd.read_parquet(results_path)
    table = manifest.merge(results, on='fit_id', how='inner', validate='1:1')
    return table


def run_tally(run_dir: Path) -> Dict[str, List[str]]:
    run_dir = Path(run_dir)
    spec, _, manifest = load_run(run_dir)
    return ledger.tally(run_dir, manifest['fit_id'].tolist(), spec.max_fit_walltime_min)


def print_tally(run_dir: Path) -> Dict[str, List[str]]:
    groups = run_tally(run_dir)
    total = sum(len(v) for v in groups.values())
    print(f'run: {Path(run_dir).name}  ({total} fits)')
    for status in ledger.STATUSES:
        print(f'  {status:12s} {len(groups[status])}')
    incomplete = groups['never_run'] + groups['failed'] + groups['stale']
    if incomplete:
        print(f'incomplete ({len(incomplete)}):')
        for fit_id in incomplete[:20]:
            print(f'  {fit_id}')
        if len(incomplete) > 20:
            print(f'  ... and {len(incomplete) - 20} more')
    return groups
