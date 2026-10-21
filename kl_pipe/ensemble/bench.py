"""
Benchmark records for sampler A/B runs of the canonical 32-fit bank.

One JSON record per benchmark run under ``docs/benchmarks/runs/`` (named
``<date>_<job_id>_<arm>.json``) with a copy of that run's summary-row parquet
beside it; ``docs/benchmarks/README.md`` carries a generated index table
between ``<!-- index:start -->`` and ``<!-- index:end -->``. Chains and mocks
stay in the run directory.

Metrics are computed from the run's ``results.parquet`` (or the per-fit
``results/<fit_id>.parquet`` files) joined to ``manifest.parquet`` on
``fit_id``. The first attempt of a non-escalated fit is the only attempt, so
its ``first_attempt_*`` values are the final ``max_rhat`` / ``min_ess``; the
escalated fits carry the values the worker recorded before the retry.
Medians are taken over fits with ``status == 'succeeded'``.

Command line::

    python -m kl_pipe.ensemble.bench record --run-dir R --job-id J \\
        --hypothesis "..." --reason "..." [--reference docs/benchmarks/runs/X.json]
    python -m kl_pipe.ensemble.bench compare --run-dir A --run-dir-ref B
    python -m kl_pipe.ensemble.bench annotate docs/benchmarks/runs/X.json --conclusion "..."
    python -m kl_pipe.ensemble.bench index
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from kl_pipe.ensemble.quality import FLAG_COLUMNS

DEFAULT_BENCH_DIR = Path('docs/benchmarks')
INDEX_START = '<!-- index:start -->'
INDEX_END = '<!-- index:end -->'

# escalation-gate defaults, used when the spec has no fit.escalation block
GATE_RHAT_MAX = 1.05
GATE_ESS_MIN = 50.0

AGREEMENT_PARAMS = ('g1', 'g2', 'cosi', 'theta_int', 'vel.vcirc')
PER_FIT_POST_PARAMS = ('g1', 'g2', 'cosi', 'theta_int', 'vel.vcirc')

MANIFEST_COLUMNS = ('fit_id', 'galaxy_id', 'ring_member', 'noise_seed')
RESULTS_COLUMNS = (
    'fit_id',
    'status',
    'escalated',
    'first_attempt_max_rhat',
    'first_attempt_min_ess',
    'max_rhat',
    'min_ess',
    'min_ess_param',
    'num_steps_total',
    'fit_wallclock_s',
) + tuple(f'post.{p}.{stat}' for p in PER_FIT_POST_PARAMS for stat in ('mean', 'std'))
# per-fit numeric columns that must be finite on every succeeded fit
REQUIRED_FINITE = (
    'first_pass_max_rhat',
    'first_pass_min_ess',
    'max_rhat',
    'min_ess',
    'num_steps_total',
    'fit_wallclock_s',
)

PER_FIT_FIELDS = (
    'fit_id',
    'galaxy_id',
    'ring_member',
    'noise_seed',
    'first_attempt_max_rhat',
    'first_attempt_min_ess',
    'max_rhat',
    'min_ess',
    'min_ess_param',
    'escalated',
    'escalation_mode',
    'final_fail',
    'num_steps_total',
    'fit_wallclock_s',
) + tuple(f'post.{p}.{stat}' for p in PER_FIT_POST_PARAMS for stat in ('mean', 'std'))


# ==============================================================================
# Inputs
# ==============================================================================


def _require_columns(df: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{what} is missing required columns: {missing}")


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_results(run_dir: Path, results: Optional[Path] = None) -> pd.DataFrame:
    """Summary rows: explicit path, else run_dir/results.parquet, else results/*.parquet."""
    if results is not None:
        results = Path(results)
        if not results.exists():
            raise FileNotFoundError(f"results parquet not found: {results}")
        df = pd.read_parquet(results)
    else:
        run_dir = Path(run_dir)
        collated = run_dir / 'results.parquet'
        per_fit_dir = run_dir / 'results'
        if collated.exists():
            df = pd.read_parquet(collated)
        elif per_fit_dir.is_dir() and any(per_fit_dir.glob('*.parquet')):
            df = pd.concat(
                [pd.read_parquet(p) for p in sorted(per_fit_dir.glob('*.parquet'))],
                ignore_index=True,
            )
        else:
            raise FileNotFoundError(
                f"no results in {run_dir}: expected results.parquet or "
                f"results/*.parquet (or pass --results PATH)"
            )
    # the worker writes first_attempt_* only on escalated fits; a run with no
    # escalation has no such column and the first attempt is the final one
    for col, final in (
        ('first_attempt_max_rhat', 'max_rhat'),
        ('first_attempt_min_ess', 'min_ess'),
    ):
        if col not in df.columns and final in df.columns:
            df[col] = df[final]
    _require_columns(df, RESULTS_COLUMNS, 'results')
    if df['fit_id'].duplicated().any():
        raise ValueError("results contain duplicated fit_id rows")
    return df


def load_manifest(run_dir: Path, manifest: Optional[Path] = None) -> pd.DataFrame:
    """Per-fit manifest: explicit path, else run_dir/manifest.parquet."""
    path = (
        Path(manifest) if manifest is not None else Path(run_dir) / 'manifest.parquet'
    )
    if not path.exists():
        raise FileNotFoundError(f"manifest parquet not found: {path}")
    df = pd.read_parquet(path)
    _require_columns(df, MANIFEST_COLUMNS, 'manifest')
    return df


def load_provenance(run_dir: Path) -> Optional[dict]:
    """Run provenance (expansion.json + ensemble_spec.yaml), or None when absent.

    Returns ``{'run_name', 'git_commit', 'n_fits', 'spec_path', 'spec_sha256'}``.
    A provenance directory that exists but lacks either file is an error.
    """
    prov_dir = Path(run_dir) / 'provenance'
    if not prov_dir.is_dir():
        return None
    expansion = prov_dir / 'expansion.json'
    spec = prov_dir / 'ensemble_spec.yaml'
    for path in (expansion, spec):
        if not path.exists():
            raise FileNotFoundError(f"provenance dir {prov_dir} lacks {path.name}")
    meta = json.loads(expansion.read_text())
    for key in ('run_name', 'git_commit', 'n_fits'):
        if key not in meta:
            raise KeyError(f"{expansion} lacks key {key!r}")
    return {
        'run_name': str(meta['run_name']),
        'git_commit': str(meta['git_commit']),
        'n_fits': int(meta['n_fits']),
        'spec_path': spec,
        'spec_sha256': sha256_of(spec),
    }


def read_fit_config(spec_path: Path) -> dict:
    """Sampler draw budget and gate from a spec YAML (no full spec validation).

    Returns ``{'run_name', 'n_warmup', 'n_samples', 'n_chains', 'gate_rhat_max',
    'gate_ess_min'}``; the gate falls back to the module defaults when the spec
    has no ``fit.escalation`` block.
    """
    spec_path = Path(spec_path)
    if not spec_path.exists():
        raise FileNotFoundError(f"spec not found: {spec_path}")
    raw = yaml.safe_load(spec_path.read_text())
    if not isinstance(raw, dict) or 'fit' not in raw or 'run' not in raw:
        raise ValueError(f"{spec_path}: expected a spec mapping with run and fit")
    fit = raw['fit']
    out = {'run_name': str(raw['run']['name'])}
    for key in ('n_warmup', 'n_samples', 'n_chains'):
        if key not in fit:
            raise KeyError(f"{spec_path}: fit block lacks {key!r}")
        out[key] = int(fit[key])
    esc = fit.get('escalation') or {}
    out['gate_rhat_max'] = float(esc.get('rhat_max', GATE_RHAT_MAX))
    out['gate_ess_min'] = float(esc.get('ess_min', GATE_ESS_MIN))
    return out


# ==============================================================================
# Per-fit table
# ==============================================================================


def prepare_fits(
    results: pd.DataFrame,
    manifest: pd.DataFrame,
    gate_rhat_max: float = GATE_RHAT_MAX,
    gate_ess_min: float = GATE_ESS_MIN,
) -> pd.DataFrame:
    """Join results to the manifest and derive the benchmark columns.

    Adds ``first_pass_max_rhat`` / ``first_pass_min_ess`` (first-attempt gate
    values, equal to the final values for non-escalated fits), ``final_fail``
    (gate failed after any escalation), ``catastrophic`` (status not
    succeeded), ``escalation_mode`` (restart for results written before the
    column existed) and ``match_key``.
    """
    _require_columns(results, RESULTS_COLUMNS, 'results')
    _require_columns(manifest, MANIFEST_COLUMNS, 'manifest')
    extra = [c for c in MANIFEST_COLUMNS if c not in results.columns]
    fits = results.merge(manifest[['fit_id'] + extra], on='fit_id', how='left')
    unmatched = fits['galaxy_id'].isna()
    if unmatched.any():
        raise ValueError(
            f"{int(unmatched.sum())} result rows have no manifest row: "
            f"{fits.loc[unmatched, 'fit_id'].tolist()}"
        )
    fits = fits.sort_values(['galaxy_id', 'ring_member', 'noise_seed']).reset_index(
        drop=True
    )

    escalated = fits['escalated'].astype(bool)
    if 'escalation_mode' not in fits.columns:
        # restart was the only escalation mode before the column existed
        fits['escalation_mode'] = np.where(escalated, 'restart', '')
    fits['escalation_mode'] = fits['escalation_mode'].fillna('').astype(str)

    missing_first = escalated & (
        fits['first_attempt_max_rhat'].isna() | fits['first_attempt_min_ess'].isna()
    )
    if missing_first.any():
        raise ValueError(
            "escalated fits without first-attempt gate values: "
            f"{fits.loc[missing_first, 'fit_id'].tolist()}"
        )
    fits['first_pass_max_rhat'] = np.where(
        escalated, fits['first_attempt_max_rhat'], fits['max_rhat']
    )
    fits['first_pass_min_ess'] = np.where(
        escalated, fits['first_attempt_min_ess'], fits['min_ess']
    )
    fits['catastrophic'] = fits['status'].astype(str) != 'succeeded'
    fits['final_fail'] = (fits['max_rhat'] > gate_rhat_max) | (
        fits['min_ess'] < gate_ess_min
    )

    ok = ~fits['catastrophic']
    for col in REQUIRED_FINITE:
        bad = ok & ~np.isfinite(fits[col].astype(float))
        if bad.any():
            raise ValueError(
                f"succeeded fits with non-finite {col}: {fits.loc[bad, 'fit_id'].tolist()}"
            )
    fits['match_key'] = [
        _match_key(g, r, s)
        for g, r, s in zip(fits['galaxy_id'], fits['ring_member'], fits['noise_seed'])
    ]
    return fits


def _match_key(galaxy_id, ring_member, noise_seed) -> str:
    return f"g{int(galaxy_id)}_r{int(ring_member)}_s{int(noise_seed)}"


def per_fit_rows(fits: pd.DataFrame) -> List[dict]:
    """The per-fit block of a record (stable order: galaxy, ring, seed)."""
    rows = []
    for _, r in fits.iterrows():
        row = {}
        for field in PER_FIT_FIELDS:
            if field == 'first_attempt_max_rhat':
                row[field] = r['first_pass_max_rhat']
            elif field == 'first_attempt_min_ess':
                row[field] = r['first_pass_min_ess']
            else:
                row[field] = r[field]
        rows.append({k: _jsonable(v) for k, v in row.items()})
    return rows


def fits_from_per_fit(rows: List[dict]) -> pd.DataFrame:
    """Rebuild the benchmark per-fit table from a record's per_fit block."""
    fits = pd.DataFrame(rows)
    _require_columns(fits, PER_FIT_FIELDS, 'record per_fit')
    fits = fits.rename(
        columns={
            'first_attempt_max_rhat': 'first_pass_max_rhat',
            'first_attempt_min_ess': 'first_pass_min_ess',
        }
    )
    fits['match_key'] = [
        _match_key(g, r, s)
        for g, r, s in zip(fits['galaxy_id'], fits['ring_member'], fits['noise_seed'])
    ]
    return fits


# ==============================================================================
# Metrics
# ==============================================================================


def compute_metrics(
    fits: pd.DataFrame,
    n_chains: int,
    n_samples: int,
    run_wall_s: Optional[float] = None,
    gate_rhat_max: float = GATE_RHAT_MAX,
    gate_ess_min: float = GATE_ESS_MIN,
) -> dict:
    """Run-level convergence and cost metrics from a prepared per-fit table.

    ``median_worst_tau`` is ``n_chains * n_samples / first_pass_min_ess``
    (draws per effective sample of the worst parameter on the first attempt);
    ``steps_per_draw_median_clean`` is ``num_steps_total`` per first-attempt
    draw over non-escalated fits.
    """
    if len(fits) == 0:
        raise ValueError("no fits to compute metrics on")
    ok = fits[~fits['catastrophic']]
    if len(ok) == 0:
        raise ValueError("every fit is catastrophic; no convergence metrics")
    escalated = ok['escalated'].astype(bool)
    clean = ok[~escalated]
    draws = n_chains * n_samples
    modes = ok.loc[escalated, 'escalation_mode'].value_counts().sort_index()
    metrics = {
        'n_fits': int(len(fits)),
        'gate_rhat_max': float(gate_rhat_max),
        'gate_ess_min': float(gate_ess_min),
        'first_pass_fail': int(escalated.sum()),
        'final_fail': int(ok['final_fail'].sum()),
        'escalations_by_mode': {str(k): int(v) for k, v in modes.items()},
        'catastrophic': int(fits['catastrophic'].sum()),
        'median_first_attempt_max_rhat': float(ok['first_pass_max_rhat'].median()),
        'median_first_attempt_min_ess': float(ok['first_pass_min_ess'].median()),
        'median_worst_tau': float((draws / ok['first_pass_min_ess']).median()),
        'steps_per_draw_median_clean': (
            float((clean['num_steps_total'] / draws).median()) if len(clean) else None
        ),
        'fit_wall_median_clean_s': (
            float(clean['fit_wallclock_s'].median()) if len(clean) else None
        ),
        'fit_wall_max_s': float(ok['fit_wallclock_s'].max()),
        'fit_wall_sum_s': float(ok['fit_wallclock_s'].sum()),
        'run_wall_s': None if run_wall_s is None else float(run_wall_s),
    }
    # failure-flag counts over the non-catastrophic fits; None (printed '-')
    # for results written before the flag columns existed
    for col in FLAG_COLUMNS:
        metrics[f'n_{col}'] = (
            int(ok[col].fillna(False).astype(bool).sum()) if col in ok.columns else None
        )
    return metrics


def wrap_angle(delta: np.ndarray) -> np.ndarray:
    """Wrap an angle difference to (-pi, pi]."""
    return (np.asarray(delta, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def compute_agreement(fits: pd.DataFrame, fits_ref: pd.DataFrame) -> dict:
    """Posterior agreement between two arms matched on (galaxy, ring, seed).

    For each parameter: median and max of ``|dmean| / sigma_ref`` and the
    median width ratio ``sigma / sigma_ref``; ``theta_int`` differences are
    wrapped modulo 2 pi. Also lists fits (match keys) that fail only in this
    arm or only in the reference, at first pass and after escalation.
    """
    cols = ['match_key', 'fit_id', 'escalated', 'final_fail'] + [
        f'post.{p}.{s}' for p in AGREEMENT_PARAMS for s in ('mean', 'std')
    ]
    _require_columns(fits, cols, 'arm fits')
    _require_columns(fits_ref, cols, 'reference fits')
    both = fits[cols].merge(
        fits_ref[cols], on='match_key', how='inner', suffixes=('', '_ref')
    )
    if len(both) == 0:
        raise ValueError(
            "no fits match between arm and reference on (galaxy_id, ring_member, "
            "noise_seed)"
        )
    keys = set(fits['match_key'])
    keys_ref = set(fits_ref['match_key'])
    # posterior columns are NaN for a fit that did not succeed (infrastructure
    # failure); the agreement statistics use the fits finite on both sides
    post_cols = [
        f'post.{p}.{s}{suf}'
        for p in AGREEMENT_PARAMS
        for s in ('mean', 'std')
        for suf in ('', '_ref')
    ]
    finite = np.isfinite(both[post_cols].to_numpy(float)).all(axis=1)
    out = {
        'n_matched': int(len(both)),
        'n_unmatched_arm': int(len(keys - keys_ref)),
        'n_unmatched_ref': int(len(keys_ref - keys)),
        'n_agreement_fits': int(finite.sum()),
        'params': {},
    }
    if not finite.any():
        raise ValueError(
            "no matched fit has finite posterior means and widths on both sides"
        )
    good = both[finite]
    for p in AGREEMENT_PARAMS:
        d = good[f'post.{p}.mean'].to_numpy(float) - good[
            f'post.{p}.mean_ref'
        ].to_numpy(float)
        if p == 'theta_int':
            d = wrap_angle(d)
        sigma_ref = good[f'post.{p}.std_ref'].to_numpy(float)
        if not np.all(sigma_ref > 0):
            raise ValueError(f"reference posterior std of {p} must be > 0")
        pull = np.abs(d) / sigma_ref
        ratio = good[f'post.{p}.std'].to_numpy(float) / sigma_ref
        out['params'][p] = {
            'median_abs_dmean_over_sigma_ref': float(np.median(pull)),
            'max_abs_dmean_over_sigma_ref': float(np.max(pull)),
            'median_width_ratio': float(np.median(ratio)),
        }
    esc = both['escalated'].astype(bool)
    esc_ref = both['escalated_ref'].astype(bool)
    fin = both['final_fail'].astype(bool)
    fin_ref = both['final_fail_ref'].astype(bool)
    out['first_pass_fail_only_arm'] = both.loc[esc & ~esc_ref, 'match_key'].tolist()
    out['first_pass_fail_only_ref'] = both.loc[~esc & esc_ref, 'match_key'].tolist()
    out['final_fail_only_arm'] = both.loc[fin & ~fin_ref, 'match_key'].tolist()
    out['final_fail_only_ref'] = both.loc[~fin & fin_ref, 'match_key'].tolist()
    return out


# ==============================================================================
# Records
# ==============================================================================


def _jsonable(value):
    """Convert numpy scalars to Python; non-finite floats to None."""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, str):
        return value
    return str(value)


def record_stem(date: str, job_id: str, arm: str) -> str:
    return f"{date}_{job_id}_{arm}"


def build_record(
    run_dir: Path,
    job_id: str,
    hypothesis: str,
    reason: str,
    conclusion: str = '',
    commit: Optional[str] = None,
    spec_path: Optional[Path] = None,
    results: Optional[Path] = None,
    manifest: Optional[Path] = None,
    precision: str = 'fp64',
    node: Optional[str] = None,
    date: Optional[str] = None,
    run_wall_s: Optional[float] = None,
    reference: Optional[dict] = None,
) -> dict:
    """Assemble a benchmark record from a run directory.

    Provenance (``run_dir/provenance``) supplies the arm name, commit and
    spec; without it ``commit`` and ``spec_path`` are required. An explicit
    ``commit`` always wins over the provenance expansion commit.
    """
    if precision not in ('fp64', 'fp32'):
        raise ValueError(f"precision must be 'fp64' or 'fp32', got {precision!r}")
    if not hypothesis or not reason:
        raise ValueError("hypothesis and reason are required")
    run_dir = Path(run_dir)
    prov = load_provenance(run_dir)
    if prov is None:
        if commit is None or spec_path is None:
            raise ValueError(
                f"{run_dir} has no provenance/ directory: pass --commit and "
                f"--spec-path explicitly"
            )
        spec_used = Path(spec_path)
        expansion_commit = None
    else:
        spec_used = Path(spec_path) if spec_path is not None else prov['spec_path']
        expansion_commit = prov['git_commit'][:7]
        if commit is None:
            commit = expansion_commit
        elif not prov['git_commit'].startswith(commit):
            print(
                f"note: --commit {commit} differs from the expansion commit "
                f"{expansion_commit}; recording {commit}",
                file=sys.stderr,
            )
    fit_cfg = read_fit_config(spec_used)
    arm = prov['run_name'] if prov is not None else fit_cfg['run_name']
    if prov is not None and prov['run_name'] != fit_cfg['run_name']:
        raise ValueError(
            f"spec run name {fit_cfg['run_name']!r} != provenance run name "
            f"{prov['run_name']!r}"
        )

    fits = prepare_fits(
        load_results(run_dir, results),
        load_manifest(run_dir, manifest),
        gate_rhat_max=fit_cfg['gate_rhat_max'],
        gate_ess_min=fit_cfg['gate_ess_min'],
    )
    if prov is not None and len(fits) != prov['n_fits']:
        raise ValueError(
            f"results have {len(fits)} rows but provenance expects {prov['n_fits']}"
        )
    metrics = compute_metrics(
        fits,
        n_chains=fit_cfg['n_chains'],
        n_samples=fit_cfg['n_samples'],
        run_wall_s=run_wall_s,
        gate_rhat_max=fit_cfg['gate_rhat_max'],
        gate_ess_min=fit_cfg['gate_ess_min'],
    )
    reference_arm = None
    reference_job_id = None
    if reference is not None:
        reference_arm = reference['arm']
        reference_job_id = reference['job_id']
        metrics['agreement_vs_reference'] = compute_agreement(
            fits, fits_from_per_fit(reference['per_fit'])
        )
    date = date or _dt.date.today().isoformat()
    _dt.date.fromisoformat(date)
    record = {
        'arm': arm,
        'job_id': str(job_id),
        'date': date,
        'commit': str(commit),
        'expansion_commit': expansion_commit,
        'spec_path': str(spec_used),
        'spec_sha256': sha256_of(spec_used),
        'precision': precision,
        'node': node,
        'n_fits': int(len(fits)),
        'sampler': {k: fit_cfg[k] for k in ('n_warmup', 'n_samples', 'n_chains')},
        'reference_arm': reference_arm,
        'reference_job_id': reference_job_id,
        'hypothesis': hypothesis,
        'reason': reason,
        'conclusion': conclusion,
        'metrics': metrics,
        'per_fit': per_fit_rows(fits),
        'results_parquet': record_stem(date, str(job_id), arm) + '.parquet',
    }
    return _jsonable(record)


def write_record(record: dict, results_path: Path, out_dir: Path) -> Path:
    """Write ``<stem>.json`` and copy the summary parquet to ``<stem>.parquet``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = record_stem(record['date'], record['job_id'], record['arm'])
    json_path = out_dir / f'{stem}.json'
    if json_path.exists():
        raise FileExistsError(f"record already exists: {json_path}")
    parquet_path = out_dir / record['results_parquet']
    if parquet_path.name != f'{stem}.parquet':
        raise ValueError(
            f"record results_parquet {record['results_parquet']!r} does not "
            f"match the record stem {stem!r}"
        )
    shutil.copyfile(results_path, parquet_path)
    json_path.write_text(json.dumps(record, indent=2) + '\n')
    return json_path


def load_record(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"record not found: {path}")
    record = json.loads(path.read_text())
    for key in ('arm', 'job_id', 'date', 'commit', 'metrics', 'per_fit'):
        if key not in record:
            raise KeyError(f"{path} lacks key {key!r}")
    return record


def annotate(path: Path, conclusion: str) -> None:
    """Set a record's conclusion in place."""
    if not conclusion:
        raise ValueError("conclusion must be non-empty")
    record = load_record(path)
    record['conclusion'] = conclusion
    Path(path).write_text(json.dumps(record, indent=2) + '\n')


def load_records(bench_dir: Path = DEFAULT_BENCH_DIR) -> pd.DataFrame:
    """One row per record: top-level scalars plus flattened metrics.

    Nested agreement metrics are flattened as
    ``agreement.<param>.<statistic>``; ``per_fit`` is left out.
    """
    runs_dir = Path(bench_dir) / 'runs'
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"no records directory: {runs_dir}")
    rows = []
    for path in sorted(runs_dir.glob('*.json')):
        record = load_record(path)
        row = {k: v for k, v in record.items() if k not in ('metrics', 'per_fit')}
        row['sampler'] = json.dumps(record.get('sampler'))
        row['record_path'] = str(path)
        for k, v in record['metrics'].items():
            if k == 'agreement_vs_reference':
                for p, stats in v['params'].items():
                    for s, val in stats.items():
                        row[f'agreement.{p}.{s}'] = val
                for k2 in ('n_matched', 'n_unmatched_arm', 'n_unmatched_ref'):
                    row[f'agreement.{k2}'] = v[k2]
            elif k == 'escalations_by_mode':
                row[k] = json.dumps(v)
            else:
                row[k] = v
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no records in {runs_dir}")
    return pd.DataFrame(rows).sort_values(['date', 'job_id']).reset_index(drop=True)


# ==============================================================================
# Index
# ==============================================================================


def _fmt(value, spec: str) -> str:
    return '-' if value is None else format(value, spec)


def _truncate(text: str, width: int = 80) -> str:
    text = ' '.join(str(text or '').split()).replace('|', '/')
    return text if len(text) <= width else text[: width - 3].rstrip() + '...'


def render_index(records: List[dict]) -> str:
    """Markdown index table for the README."""
    header = (
        '| date | arm | job | commit | first-pass fails | final fails | '
        'steps/draw clean | clean fit wall median [min] | sum wall [h] | conclusion |\n'
        '|---|---|---|---|---|---|---|---|---|---|'
    )
    lines = [header]
    for r in sorted(records, key=lambda r: (r['date'], r['job_id'])):
        m = r['metrics']
        n = m['n_fits']
        wall_med = m['fit_wall_median_clean_s']
        lines.append(
            f"| {r['date']} | {r['arm']} | {r['job_id']} | {r['commit']} | "
            f"{m['first_pass_fail']}/{n} | {m['final_fail']}/{n} | "
            f"{_fmt(m['steps_per_draw_median_clean'], '.1f')} | "
            f"{_fmt(None if wall_med is None else wall_med / 60.0, '.1f')} | "
            f"{m['fit_wall_sum_s'] / 3600.0:.2f} | {_truncate(r.get('conclusion', ''))} |"
        )
    return '\n'.join(lines)


def write_index(bench_dir: Path = DEFAULT_BENCH_DIR) -> Path:
    """Regenerate the README table between the index markers (idempotent)."""
    bench_dir = Path(bench_dir)
    readme = bench_dir / 'README.md'
    if not readme.exists():
        raise FileNotFoundError(
            f"{readme} not found; write the header with {INDEX_START} and "
            f"{INDEX_END} markers first"
        )
    text = readme.read_text()
    if text.count(INDEX_START) != 1 or text.count(INDEX_END) != 1:
        raise ValueError(
            f"{readme} must contain exactly one {INDEX_START} and one {INDEX_END}"
        )
    start = text.index(INDEX_START) + len(INDEX_START)
    end = text.index(INDEX_END)
    if end < start:
        raise ValueError(f"{readme}: {INDEX_END} precedes {INDEX_START}")
    records = [load_record(p) for p in sorted((bench_dir / 'runs').glob('*.json'))]
    if not records:
        raise FileNotFoundError(f"no records under {bench_dir / 'runs'}")
    table = render_index(records)
    readme.write_text(text[:start] + '\n' + table + '\n' + text[end:])
    return readme


# ==============================================================================
# CLI
# ==============================================================================


def _metrics_table(metrics_a: dict, metrics_b: dict, label_a: str, label_b: str) -> str:
    keys = [
        k
        for k in metrics_a
        if k not in ('agreement_vs_reference', 'escalations_by_mode')
    ]
    width = max(len(k) for k in keys)
    col = max(14, len(label_a), len(label_b))
    lines = [f"{'metric':<{width}}  {label_a:>{col}}  {label_b:>{col}}"]
    for k in keys:
        a, b = metrics_a.get(k), metrics_b.get(k)
        lines.append(f"{k:<{width}}  {_fmt_cell(a):>{col}}  {_fmt_cell(b):>{col}}")
    lines.append(
        f"{'escalations_by_mode':<{width}}  "
        f"{json.dumps(metrics_a['escalations_by_mode']):>{col}}  "
        f"{json.dumps(metrics_b['escalations_by_mode']):>{col}}"
    )
    return '\n'.join(lines)


def _fmt_cell(value) -> str:
    if value is None:
        return '-'
    if isinstance(value, float):
        return f'{value:.4g}'
    return str(value)


def _agreement_table(agreement: dict) -> str:
    lines = [
        f"matched {agreement['n_matched']} fits "
        f"(unmatched arm {agreement['n_unmatched_arm']}, "
        f"ref {agreement['n_unmatched_ref']}; "
        f"{_fmt_cell(agreement.get('n_agreement_fits'))} finite on both sides)",
        f"{'param':<10} {'med|dm|/sig':>12} {'max|dm|/sig':>12} {'med width':>10}",
    ]
    for p, s in agreement['params'].items():
        lines.append(
            f"{p:<10} {_fmt_cell(s['median_abs_dmean_over_sigma_ref']):>12} "
            f"{_fmt_cell(s['max_abs_dmean_over_sigma_ref']):>12} "
            f"{_fmt_cell(s['median_width_ratio']):>10}"
        )
    for key in (
        'first_pass_fail_only_arm',
        'first_pass_fail_only_ref',
        'final_fail_only_arm',
        'final_fail_only_ref',
    ):
        lines.append(f"{key}: {agreement[key]}")
    return '\n'.join(lines)


def _prepare_run(run_dir: Path, results, manifest, spec_path) -> tuple:
    """(fits, fit_cfg, arm label) for a run dir, honoring the CLI overrides."""
    prov = load_provenance(run_dir)
    if prov is None and spec_path is None:
        raise ValueError(f"{run_dir} has no provenance/: pass the spec path explicitly")
    spec_used = Path(spec_path) if spec_path is not None else prov['spec_path']
    fit_cfg = read_fit_config(spec_used)
    fits = prepare_fits(
        load_results(run_dir, results),
        load_manifest(run_dir, manifest),
        gate_rhat_max=fit_cfg['gate_rhat_max'],
        gate_ess_min=fit_cfg['gate_ess_min'],
    )
    return fits, fit_cfg, fit_cfg['run_name']


def _add_run_inputs(parser: argparse.ArgumentParser, suffix: str = '') -> None:
    parser.add_argument(f'--run-dir{suffix}', type=Path, required=True)
    parser.add_argument(
        f'--results{suffix}',
        type=Path,
        default=None,
        help='summary-row parquet (default: <run-dir>/results.parquet)',
    )
    parser.add_argument(
        f'--manifest{suffix}',
        type=Path,
        default=None,
        help='manifest parquet (default: <run-dir>/manifest.parquet)',
    )
    parser.add_argument(
        f'--spec-path{suffix}',
        type=Path,
        default=None,
        help='spec YAML (default: <run-dir>/provenance/ensemble_spec.yaml)',
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog='python -m kl_pipe.ensemble.bench',
        description='Benchmark records for sampler A/B runs',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    p_rec = sub.add_parser('record', help='write a record + parquet copy')
    _add_run_inputs(p_rec)
    p_rec.add_argument('--job-id', required=True)
    p_rec.add_argument('--commit', default=None, help='git short sha the run used')
    p_rec.add_argument('--precision', default='fp64', choices=('fp64', 'fp32'))
    p_rec.add_argument('--node', default=None)
    p_rec.add_argument('--date', default=None, help='ISO date (default: today)')
    p_rec.add_argument(
        '--reference', type=Path, default=None, help='record JSON to compare against'
    )
    p_rec.add_argument('--hypothesis', required=True)
    p_rec.add_argument('--reason', required=True)
    p_rec.add_argument('--conclusion', default='')
    p_rec.add_argument('--run-wall', type=float, default=None, help='run wall [s]')
    p_rec.add_argument('--out', type=Path, default=DEFAULT_BENCH_DIR / 'runs')

    p_cmp = sub.add_parser('compare', help='print metrics for two runs (no write)')
    _add_run_inputs(p_cmp)
    _add_run_inputs(p_cmp, suffix='-ref')

    p_ann = sub.add_parser('annotate', help='set a record conclusion')
    p_ann.add_argument('record', type=Path)
    p_ann.add_argument('--conclusion', required=True)

    p_idx = sub.add_parser('index', help='regenerate the README table')
    p_idx.add_argument('--dir', type=Path, default=DEFAULT_BENCH_DIR)

    args = parser.parse_args(argv)

    if args.command == 'record':
        reference = load_record(args.reference) if args.reference else None
        record = build_record(
            args.run_dir,
            job_id=args.job_id,
            hypothesis=args.hypothesis,
            reason=args.reason,
            conclusion=args.conclusion,
            commit=args.commit,
            spec_path=args.spec_path,
            results=args.results,
            manifest=args.manifest,
            precision=args.precision,
            node=args.node,
            date=args.date,
            run_wall_s=args.run_wall,
            reference=reference,
        )
        results_path = (
            args.results
            if args.results is not None
            else args.run_dir / 'results.parquet'
        )
        if not Path(results_path).exists():
            raise FileNotFoundError(
                f"{results_path} not found; pass --results to name the parquet to copy"
            )
        path = write_record(record, results_path, args.out)
        print(f'wrote {path}')
        m = record['metrics']
        print(
            f"{record['arm']} job {record['job_id']}: first-pass fail "
            f"{m['first_pass_fail']}/{m['n_fits']}, final fail {m['final_fail']}, "
            f"steps/draw clean {_fmt_cell(m['steps_per_draw_median_clean'])}, "
            f"sum wall {m['fit_wall_sum_s'] / 3600.0:.2f} h"
        )
        if 'agreement_vs_reference' in m:
            print(_agreement_table(m['agreement_vs_reference']))
        return 0

    if args.command == 'compare':
        fits_a, cfg_a, arm_a = _prepare_run(
            args.run_dir, args.results, args.manifest, args.spec_path
        )
        fits_b, cfg_b, arm_b = _prepare_run(
            args.run_dir_ref, args.results_ref, args.manifest_ref, args.spec_path_ref
        )
        metrics_a = compute_metrics(
            fits_a,
            cfg_a['n_chains'],
            cfg_a['n_samples'],
            gate_rhat_max=cfg_a['gate_rhat_max'],
            gate_ess_min=cfg_a['gate_ess_min'],
        )
        metrics_b = compute_metrics(
            fits_b,
            cfg_b['n_chains'],
            cfg_b['n_samples'],
            gate_rhat_max=cfg_b['gate_rhat_max'],
            gate_ess_min=cfg_b['gate_ess_min'],
        )
        print(_metrics_table(metrics_a, metrics_b, arm_a, arm_b))
        print()
        print(_agreement_table(compute_agreement(fits_a, fits_b)))
        return 0

    if args.command == 'annotate':
        annotate(args.record, args.conclusion)
        print(f'annotated {args.record}')
        return 0

    if args.command == 'index':
        path = write_index(args.dir)
        print(f'regenerated index in {path}')
        return 0

    raise ValueError(f'unhandled command {args.command}')


if __name__ == '__main__':
    sys.exit(main())
