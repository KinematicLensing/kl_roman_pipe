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

# a fit whose chains come back broken (a sampler stuck in a spurious mode --
# not mere low quality) is catastrophic. Canonical thresholds live here so the
# worker's retry policy and the status tally always agree.
CATASTROPHIC_RHAT = 1.1
CATASTROPHIC_DIV_RATE = 0.9


def is_catastrophic(summary) -> bool:
    """True if a fit's summary shows broken chains.

    Accepts any mapping with ``max_rhat`` and ``divergence_rate`` (a worker
    summary dict during a run, or a result row read back from parquet).
    """
    return (
        summary['max_rhat'] > CATASTROPHIC_RHAT
        or summary['divergence_rate'] > CATASTROPHIC_DIV_RATE
    )


def count_catastrophic(run_dir: Path, fit_ids: List[str]) -> int:
    """Number of the given (succeeded) fits whose recorded summary is catastrophic.

    Reads each fit's per-fit result file. A succeeded fit missing its result
    file is an inconsistency and raises rather than being silently skipped.
    """
    run_dir = Path(run_dir)
    n = 0
    for fit_id in fit_ids:
        path = run_dir / 'results' / f'{fit_id}.parquet'
        if not path.exists():
            raise FileNotFoundError(
                f"succeeded fit {fit_id} has no result file at {path}"
            )
        if is_catastrophic(pd.read_parquet(path).iloc[0]):
            n += 1
    return n


def count_escalated(run_dir: Path, fit_ids: List[str]) -> int:
    """Number of the given (succeeded) fits that ran the escalation retry.

    Reads each fit's per-fit result file. Rows from runs without the
    ``fit.escalation`` feature lack the ``escalated`` column and count as
    not escalated. A succeeded fit missing its result file raises.
    """
    run_dir = Path(run_dir)
    n = 0
    for fit_id in fit_ids:
        path = run_dir / 'results' / f'{fit_id}.parquet'
        if not path.exists():
            raise FileNotFoundError(
                f"succeeded fit {fit_id} has no result file at {path}"
            )
        row = pd.read_parquet(path).iloc[0]
        if 'escalated' in row.index and bool(row['escalated']):
            n += 1
    return n


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


def write_collated_table(run_dir: Path) -> Path:
    """
    Persist the manifest-joined analysis table to disk.

    Writes ``<run_name>_collated.parquet`` -- the one-stop queryable catalog:
    truth + observable (SNR, cosi bin, seeds) + fitted (post.*, map.*, g+/gx)
    + convergence (rhat/ess/divergences) + priors (prior.<param>.*) + has_chains,
    one row per fit. Filename carries the run name so it stays self-identifying
    when copied out of the run directory.
    """
    run_dir = Path(run_dir)
    table = analysis_table(run_dir)
    out_path = run_dir / f'{run_dir.name}_collated.parquet'
    table.to_parquet(out_path, index=False)
    return out_path


def run_tally(run_dir: Path) -> Dict[str, List[str]]:
    run_dir = Path(run_dir)
    spec, _, manifest = load_run(run_dir)
    return ledger.tally(run_dir, manifest['fit_id'].tolist(), spec.max_fit_walltime_min)


def print_tally(run_dir: Path) -> Dict[str, List[str]]:
    groups = run_tally(run_dir)
    total = sum(len(v) for v in groups.values())
    print(f'run: {Path(run_dir).name}  ({total} fits)')
    n_catastrophic = count_catastrophic(run_dir, groups['succeeded'])
    n_escalated = count_escalated(run_dir, groups['succeeded'])
    for status in ledger.STATUSES:
        line = f'  {status:12s} {len(groups[status])}'
        if status == 'succeeded' and groups['succeeded']:
            line += f'  (catastrophic: {n_catastrophic})'
            if n_escalated:
                line += f'  (escalated: {n_escalated})'
        print(line)
    incomplete = groups['never_run'] + groups['failed'] + groups['stale']
    if incomplete:
        print(f'incomplete ({len(incomplete)}):')
        for fit_id in incomplete[:20]:
            print(f'  {fit_id}')
        if len(incomplete) > 20:
            print(f'  ... and {len(incomplete) - 20} more')
    return groups
