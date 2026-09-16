"""
Static HTML dashboard for an ensemble run directory.

``build_dashboard(run_dir)`` writes ``<run_dir>/diagnostics/dashboard.html``:
one self-contained page (inline CSS, base64 PNGs) with progress, failures and
escalations, quality flags, early science tables, plots, notes and a glossary.
Every optional input degrades to a one-line "not available" note; only a
missing run directory or manifest raises.
"""

from __future__ import annotations

import base64
import html
import io
import json
import re
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from kl_pipe.ensemble.quality import FLAG_COLUMNS, PERIODIC_PARAMS

SCIENCE_PARAMS = ('g1', 'g2', 'cosi', 'theta_int', 'vel.vcirc')
DERIVED_SHEAR = ('g_plus', 'g_cross')
PHOTOMETRIC_SHAPE_NOISE = 0.26
COSI_BINS = ((0.0, 0.3), (0.3, 0.7), (0.7, 1.0001))
NOT_AVAILABLE = 'not available'
GATE_RHAT_MAX = 1.05
GATE_ESS_MIN = 50.0
ROTATION_FLAG_NATS = 3.0

# plot colours: categorical slots 1-3 (all-pairs safe), a blue sequential ramp
# for ordered cos i bins, status colours for pass / warn / fail
C_BLUE, C_ORANGE, C_AQUA = '#2a78d6', '#eb6834', '#1baf7a'
C_SEQ = ('#86b6ef', '#2a78d6', '#0d366b')
C_GOOD, C_WARN, C_CRIT = '#0ca30c', '#fab219', '#d03b3b'
C_TEXT2 = '#52514e'
C_INK = '#1c1c1b'

_CSS = """
:root { --ink: #1c1c1b; --ink2: #52514e; --line: #e2e1dd; --surface: #fcfcfb;
        --head: #f3f2ef; --good-bg: #e2f3e2; --warn-bg: #fdf0cf; --crit-bg: #f9dada; }
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
       color: var(--ink); background: var(--surface); margin: 0; }
main { max-width: 1280px; margin: 0 auto; padding: 1.5em 2em 4em; }
nav { position: sticky; top: 0; background: var(--surface); border-bottom: 1px solid var(--line);
      padding: 0.5em 2em; font-size: 0.9em; z-index: 10; }
nav a { margin-right: 1.2em; color: var(--ink2); text-decoration: none; }
nav a:hover { color: var(--ink); text-decoration: underline; }
h1 { font-size: 1.45em; font-weight: 600; margin: 0.4em 0 0.2em; }
h2 { font-size: 1.15em; font-weight: 600; margin: 2.2em 0 0.6em; padding-bottom: 0.25em;
     border-bottom: 1px solid var(--line); }
h3 { font-size: 0.98em; font-weight: 600; margin: 1.2em 0 0.3em; color: var(--ink); }
p, li { line-height: 1.45; } .meta { color: var(--ink2); font-size: 0.9em; }
table { border-collapse: collapse; font-size: 0.84em; margin: 0.4em 0 0.8em; }
th, td { border-bottom: 1px solid var(--line); padding: 3px 9px; text-align: right;
         white-space: nowrap; }
th { background: var(--head); font-weight: 600; color: var(--ink2); position: sticky; top: 2.3em; }
th a { color: inherit; text-decoration: none; border-bottom: 1px dotted #999; }
td:first-child, th:first-child { text-align: left; }
tr:hover td { background: #f6f5f2; }
.good { background: var(--good-bg); } .warn { background: var(--warn-bg); }
.crit { background: var(--crit-bg); font-weight: 600; }
details > summary { cursor: pointer; list-style: none; }
details > summary::before { content: '\\25BE'; display: inline-block; width: 1em; color: var(--ink2); }
details:not([open]) > summary::before { content: '\\25B8'; }
details > summary h2 { display: inline-block; margin: 2.2em 0 0.6em; width: calc(100% - 1.2em); }
details > summary + * { margin-top: 0.4em; }
.na { color: var(--ink2); font-style: italic; }
.bad { color: #a12b2b; font-weight: 600; }
img { max-width: 100%; margin: 0.4em 0 0.2em; border: 1px solid var(--line); background: #fff; }
.cap { color: var(--ink2); font-size: 0.86em; margin: 0 0 1.2em; }
pre { background: var(--head); padding: 0.8em; white-space: pre-wrap; font-size: 0.86em; }
.bar { background: var(--line); height: 14px; width: 560px; border-radius: 3px; overflow: hidden; }
.bar div { height: 100%; background: #2a78d6; }
dl dt { font-weight: 600; margin-top: 0.6em; } dl dd { margin-left: 1.2em; color: var(--ink); }
code { background: var(--head); padding: 0 3px; font-size: 0.92em; }
"""

# ==============================================================================
# Glossary (wording follows docs/sampler_failure_ledger.md)
# ==============================================================================

GLOSSARY: Dict[str, Tuple[str, str]] = {
    'pull': (
        'pull',
        '(posterior mean - truth) / posterior std per fit; theta_int residuals '
        'wrapped to (-pi, pi]. A calibrated fit gives mean 0 and sd 1; |pull| > 3 '
        'marks a wrong mode or a broken fit.',
    ),
    'coverage': (
        'coverage (cov68 / cov95)',
        'fraction of truths inside the posterior 16-84 and 2.5-97.5 percentile '
        'intervals (edges relative to the posterior median, wrapped for theta_int); '
        'a calibrated posterior gives 0.68 and 0.95 within the binomial error.',
    ),
    'shear_ess': (
        'shear ESS',
        'min(ess_g1, ess_g2): effective sample size of the shear components, the '
        'quantities the science uses; below ~200 the posterior width carries '
        '~5% Monte-Carlo noise.',
    ),
    'steps': (
        'leapfrog steps / steps per draw',
        'num_steps_total: leapfrog steps summed over chains and draws (steps per '
        'draw = total / (n_chains x n_samples)); the sampler cost per fit.',
    ),
    'max_rhat': (
        'max_rhat, min_ess',
        'escalation-gate quantities over all sampled parameters; production gate '
        'rhat_max 1.05, ess_min 50.',
    ),
    'escalated': (
        'escalated / escalation_mode',
        "a first attempt failed the gate and more sampling ran: 'restart' (fresh "
        "warmup with the donated adapted metric), 'continue' (more draws from the "
        "warm chains in blocks of 300 per chain, first-attempt draws kept), '' "
        '(no escalation).',
    ),
    'restart_reason': (
        'restart_reason',
        "why more draws would not (or did not) rescue the first attempt: '' "
        "(marginal), 'rhat' (above continue_rhat_max), 'divergences' (above "
        "continue_divergence_max), 'blocks_exhausted' (continued to the block cap, "
        'still below the gate).',
    ),
    'final_gate': (
        'final gate failure',
        'max_rhat > 1.05 or min_ess < 50 after escalation; the fit is kept and '
        'flagged catastrophic.',
    ),
    'flag_gate': (
        'flag_gate',
        'gate failed after escalation (see final gate failure).',
    ),
    'flag_map_dev': (
        'flag_map_dev',
        'map_postmean_max_dev > 5: largest |MAP - posterior mean| / sigma over the '
        'sampled parameters (healthy 0.5-2.7, a MAP left in a wrong basin 11-25).',
    ),
    'flag_chi2_excess': (
        'flag_chi2_excess',
        'postmean_chi2 - n_data > 5 sqrt(2 n_data): the posterior mean fits the data '
        'far worse than a chi-square with n_data degrees of freedom allows, i.e. a '
        'posterior stuck in a wrong basin.',
    ),
    'flag_rotation_ambiguous': (
        'flag_rotation_ambiguous',
        'map_pa_flip_margin < 3 nats: the MAP beats the counter-rotating solution '
        '(theta + pi) by less than 3 nats, so the two rotation directions are '
        'competing modes.',
    ),
    'map_pa_flip_margin': (
        'map_pa_flip_margin',
        'negative-log-posterior margin of the MAP over the best optimisation start '
        'that settled in the counter-rotating PA basin (inf: no start settled there).',
    ),
    'divergence_rate': (
        'divergence_rate',
        'fraction of NUTS transitions that diverged; above 0.01 the geometry is '
        'hurting the sampler.',
    ),
    'git_commit': (
        'git_commit',
        'short sha of the kl_pipe checkout the worker ran at (-dirty: uncommitted '
        'changes). Older runs lack the column; it is then inferred per SLURM job '
        'from the first line of the job log.',
    ),
    'g_plus_cross': (
        'g+, gx (derived disk-frame shear)',
        'g+ = g1 cos 2theta + g2 sin 2theta, gx = -g1 sin 2theta + g2 cos 2theta with '
        'theta = the TRUE theta_int for both truth and measured values, so both sit in '
        'one fixed frame; measured = rotated posterior means, sigma by linear '
        'propagation of post.g1.std / post.g2.std assuming independence. Coverage, '
        'truth rank and skew of g+, gx come from the saved posterior draws rotated by '
        'the true theta_int when a chains directory exists (from chains).',
    ),
    'shape_noise': (
        'effective shape noise',
        'the posterior sigma of one shear component for one galaxy; in kinematic '
        'lensing this per-galaxy width is the effective per-component shape noise '
        '(photometric weak lensing: ~0.26 per component).',
    ),
    'gate_table': (
        'gate table',
        'fraction (count) of fits whose final max_rhat exceeds the row threshold OR '
        'whose min_ess falls below the column threshold; the first-attempt version '
        'uses first_attempt_max_rhat / first_attempt_min_ess and shows how many fits '
        'a stricter gate would have sent to escalation. Shear-ESS rows use '
        'min(ess_g1, ess_g2) alone.',
    ),
    'wrong_rotation': (
        'wrong rotation',
        'a fit whose wrapped theta_int pull exceeds 3: the posterior sits in the '
        'counter-rotating mode (theta + pi, velocity sign flipped).',
    ),
}

_HEADER_ANCHORS = {
    'pull': 'pull',
    'pull_mean': 'pull',
    'pull_sem': 'pull',
    'pull_sd': 'pull',
    'n_|pull|>3': 'pull',
    'cov68': 'coverage',
    'cov95': 'coverage',
    'shear_ess': 'shear_ess',
    'min_shear_ess': 'shear_ess',
    'ess_g1': 'shear_ess',
    'ess_g2': 'shear_ess',
    'num_steps_total': 'steps',
    'steps_per_draw': 'steps',
    'max_rhat': 'max_rhat',
    'min_ess': 'max_rhat',
    'first_attempt_max_rhat': 'max_rhat',
    'first_attempt_min_ess': 'max_rhat',
    'final_max_rhat': 'max_rhat',
    'final_min_ess': 'max_rhat',
    'gate_pass': 'max_rhat',
    'escalated': 'escalated',
    'escalation_mode': 'escalated',
    'escalation_n_blocks': 'escalated',
    'restart_reason': 'restart_reason',
    'divergence_rate': 'divergence_rate',
    'map_pa_flip_margin': 'map_pa_flip_margin',
    'git_commit': 'git_commit',
    'flags': 'flag_gate',
}
for _col in ('rhat_max', 'ess_min', 'shear_ess_min', 'n_fail', 'frac_fail'):
    _HEADER_ANCHORS[_col] = 'gate_table'
for _col in ('sigma_med', 'sigma_g1', 'sigma_g2', 'sigma_g_plus', 'sigma_g_cross'):
    _HEADER_ANCHORS[_col] = 'shape_noise'
for _flag in FLAG_COLUMNS:
    _HEADER_ANCHORS[_flag] = _flag


# ==============================================================================
# Inputs
# ==============================================================================


def _read_results(run_dir: Path) -> Optional[pd.DataFrame]:
    """Collated results, re-collated when per-fit files are newer or missing."""
    from kl_pipe.ensemble.collate import collate_results

    results_path = run_dir / 'results.parquet'
    per_fit = sorted((run_dir / 'results').glob('*.parquet'))
    if per_fit:
        newest = max(p.stat().st_mtime for p in per_fit)
        if not results_path.exists() or results_path.stat().st_mtime < newest:
            return collate_results(run_dir)
    if results_path.exists():
        return pd.read_parquet(results_path)
    return None


def _read_spec(run_dir: Path) -> Tuple[Optional[dict], str]:
    prov = run_dir / 'provenance'
    for name in ('ensemble_spec_resolved.yaml', 'ensemble_spec.yaml'):
        p = prov / name
        if p.exists():
            return yaml.safe_load(p.read_text()) or {}, str(p)
    return None, NOT_AVAILABLE


def _workers_per_node(spec: Optional[dict]) -> Optional[int]:
    """dispatch.workers_per_node from the resolved spec, None when absent."""
    try:
        v = (spec or {}).get('dispatch', {}).get('workers_per_node')
    except AttributeError:
        return None
    return int(v) if v else None


def _expansion_commit(run_dir: Path) -> Optional[str]:
    exp = run_dir / 'provenance' / 'expansion.json'
    if not exp.exists():
        return None
    try:
        c = json.loads(exp.read_text()).get('git_commit')
    except json.JSONDecodeError:
        return None
    return str(c)[:9] if c else None


def _job_log_commits(run_dir: Path) -> Dict[str, str]:
    """{slurm job id: short sha} from ``prod_base_<jobid>.out`` first lines."""
    out = {}
    for p in run_dir.glob('prod_base_*.out'):
        m = re.fullmatch(r'prod_base_(\d+)\.out', p.name)
        if not m:
            continue
        try:
            first = p.open(errors='replace').readline().strip()
        except OSError:
            continue
        sha = first.split(' ', 1)[0] if first else ''
        if re.fullmatch(r'[0-9a-f]{7,40}', sha):
            out[m.group(1)] = sha[:9]
    return out


def _repo_url() -> Optional[str]:
    """https URL of the origin remote (git@github.com: form converted)."""
    try:
        url = subprocess.run(
            ['git', 'remote', 'get-url', 'origin'],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
            timeout=10,
        ).stdout.strip()
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return None
    m = re.fullmatch(r'git@([^:]+):(.+?)(\.git)?', url)
    if m:
        return f'https://{m.group(1)}/{m.group(2)}'
    m = re.fullmatch(r'(https?://.+?)(\.git)?', url)
    return m.group(1) if m else None


def _status_frame(run_dir: Path) -> pd.DataFrame:
    """One row per claimed fit: fit_id, job, start ts, done ts, state."""
    rows = []
    claims = run_dir / 'status' / 'claims'
    done_dir = run_dir / 'status' / 'done'
    failed_dir = run_dir / 'status' / 'failed'
    columns = ['fit_id', 'job', 'start', 'end', 'state']
    if not claims.is_dir():
        return pd.DataFrame(columns=columns)
    for claim in sorted(claims.iterdir()):
        meta_path = claim / 'claim.json'
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                meta = {}
        fid = claim.name
        end = None
        if (done_dir / fid).exists():
            state = 'done'
            try:
                end = json.loads((done_dir / fid).read_text()).get('ts')
            except (json.JSONDecodeError, OSError):
                end = (done_dir / fid).stat().st_mtime
        elif (failed_dir / fid).exists():
            state = 'failed'
        else:
            state = 'in_flight'
        rows.append(
            {
                'fit_id': fid,
                'job': str(meta.get('slurm_job_id') or meta.get('hostname') or '?'),
                'start': float(meta['ts']) if 'ts' in meta else np.nan,
                'end': float(end) if end is not None else np.nan,
                'state': state,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _succeeded(results: Optional[pd.DataFrame], manifest: pd.DataFrame) -> pd.DataFrame:
    """Succeeded rows joined with the manifest's extra columns (truth, SNR)."""
    if results is None:
        return pd.DataFrame()
    ok = results[results['status'] == 'succeeded'] if 'status' in results else results
    extra = [c for c in manifest.columns if c not in ok.columns and c != 'fit_id']
    return ok.merge(manifest[['fit_id'] + extra], on='fit_id', how='left')


# ==============================================================================
# Formatting
# ==============================================================================

_FORMAT_RULES: Tuple[Tuple[str, Callable[[float], str]], ...] = (
    (
        r'^(pull|pull_mean|pull_sem|pull_sd|pull\..*|.*_ratio|cov68|cov95|frac.*)$',
        lambda v: f'{v:.2f}',
    ),
    (
        r'^(max_rhat|min_rhat|first_attempt_max_rhat|final_max_rhat|rhat.*)$',
        lambda v: f'{v:.3f}',
    ),
    (
        r'^(min_ess|ess_.*|first_attempt_min_ess|final_min_ess|shear_ess|shear_ess_med|min_shear_ess|num_steps_total|steps.*|n|n_.*|claimed|done|failed|unfinished|escalation_n_blocks|n_data)$',
        lambda v: f'{v:.0f}',
    ),
    (
        r'^(wall_min|wall_med_min|wall_mean_min|wall_p90_min|fit_wall_min|span_h|fits_per_node_hr|fits_per_node_h|fits_per_worker_h|worker_h|steps_med_k)$',
        lambda v: f'{v:.1f}',
    ),
    (r'^(sigma_med|post\.g[12]\.std|shear_sigma.*)$', lambda v: f'{v:.3f}'),
    (
        r'^(truth\.cosi|cosi|post\.cosi\..*|map_pa_flip_margin|map_postmean_max_dev|chi2_excess)$',
        lambda v: f'{v:.2f}',
    ),
    (r'^(line_snr|.*snr.*)$', lambda v: f'{v:.0f}'),
    (r'^(divergence_rate|first_attempt_divergence_rate)$', lambda v: f'{v:.4f}'),
    (r'^ms_per_step$', lambda v: f'{v:.2f}'),
    (r'^esc_%$', lambda v: f'{v:.0f}'),
)


def _fmt(col: str, v) -> str:
    """One formatter for every table cell, decimals chosen by column name."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '-'
    if isinstance(v, (bool, np.bool_)):
        return 'yes' if v else 'no'
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return 'inf' if v > 0 else '-inf'
        for pattern, fn in _FORMAT_RULES:
            if re.fullmatch(pattern, col):
                return fn(float(v))
        return f'{v:.4g}' if abs(v) < 1e4 else f'{v:.3e}'
    return html.escape(str(v))


def _cell_class(col: str, v, row: pd.Series) -> str:
    """Semantic colour: pulls by magnitude, pathological sampler values red."""
    is_num = isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(
        v, (bool, np.bool_)
    )
    numeric_col = re.fullmatch(
        r'(pull.*|max_rhat|min_ess|first_attempt_.*|divergence_rate|num_steps_total|'
        r'fit_wallclock_s|wall_min|post\..*|truth\..*|ess_g[12]|shear_ess|cov68|cov95)',
        col,
    )
    if numeric_col and (v is None or (is_num and np.isnan(v))):
        return 'crit'
    if col == 'gate_pass' and isinstance(v, (bool, np.bool_)):
        return 'good' if v else 'crit'
    if not is_num:
        return ''
    if col == 'final_max_rhat':
        return 'crit' if v > GATE_RHAT_MAX else 'good'
    if col == 'final_min_ess':
        return 'crit' if v < GATE_ESS_MIN else 'good'
    if col.startswith('pull') or col == 'pull':
        a = abs(float(v))
        return 'good' if a < 2 else ('warn' if a < 3 else 'crit')
    if col in ('max_rhat', 'first_attempt_max_rhat') and v > GATE_RHAT_MAX:
        return 'crit'
    if col in ('min_ess', 'first_attempt_min_ess') and v < GATE_ESS_MIN:
        return 'crit'
    if col in ('divergence_rate', 'first_attempt_divergence_rate') and v > 0.01:
        return 'crit'
    if col in ('cov68',) and abs(v - 0.68) > 0.15:
        return 'warn'
    if col in ('cov95',) and v < 0.85:
        return 'warn'
    return ''


# ==============================================================================
# HTML helpers
# ==============================================================================


def _esc(x) -> str:
    return html.escape(str(x))


def _na(msg: str = NOT_AVAILABLE) -> str:
    return f'<p class="na">{_esc(msg)}</p>'


def _header(col: str) -> str:
    key = _HEADER_ANCHORS.get(col)
    label = _esc(col)
    return f'<a href="#g-{key}">{label}</a>' if key else label


def _commit_html(
    sha: Optional[str], note: str = '', *, repo_url: Optional[str] = None
) -> str:
    if not sha or sha in ('unknown', '?', '-'):
        return '-'
    core = sha.replace('-dirty', '')
    text = _esc(sha) + (f' <span class="meta">{_esc(note)}</span>' if note else '')
    if repo_url and re.fullmatch(r'[0-9a-f]{7,40}', core):
        return f'<a href="{_esc(repo_url)}/commit/{core}">{text}</a>'
    return text


def _table(
    df: pd.DataFrame,
    max_rows: int = 60,
    raw_html_cols: Tuple[str, ...] = (),
    cell_classes: Optional[Dict[str, Callable]] = None,
) -> str:
    if df is None or len(df) == 0:
        return _na('no rows')
    head = ''.join(f'<th>{_header(c)}</th>' for c in df.columns)
    body = []
    for _, row in df.head(max_rows).iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if c in raw_html_cols:
                cells.append(f'<td>{v}</td>')
                continue
            cls = (
                cell_classes[c](v, row)
                if cell_classes and c in cell_classes
                else _cell_class(c, v, row)
            )
            attr = f' class="{cls}"' if cls else ''
            cells.append(f'<td{attr}>{_fmt(c, v)}</td>')
        body.append('<tr>' + ''.join(cells) + '</tr>')
    more = (
        f'<p class="na">... {len(df) - max_rows} more rows</p>'
        if len(df) > max_rows
        else ''
    )
    return f'<table><tr>{head}</tr>{"".join(body)}</table>{more}'


def _img_from_path(path: Path, title: str = '') -> str:
    data = base64.b64encode(path.read_bytes()).decode('ascii')
    return (
        f'<h3>{_esc(title or path.name)}</h3>'
        f'<img src="data:image/png;base64,{data}" alt="{_esc(path.name)}">'
    )


def _img_from_fig(fig, title: str, caption: str = '') -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode('ascii')
    cap = f'<p class="cap">{_esc(caption)}</p>' if caption else ''
    return (
        f'<h3>{_esc(title)}</h3>'
        f'<img src="data:image/png;base64,{data}" alt="{_esc(title)}">{cap}'
    )


def _guard(fn, *args, **kwargs) -> str:
    """Render a section; any exception becomes a visible note, not a crash."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - the page must always render
        return _na(f'{NOT_AVAILABLE}: {type(exc).__name__}: {exc}')


def _style_axes(ax) -> None:
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(True, color='#e6e5e1', lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors=C_TEXT2)


# ==============================================================================
# Science helpers
# ==============================================================================


def _pull(ok: pd.DataFrame, p: str) -> Optional[pd.Series]:
    cols = (f'truth.{p}', f'post.{p}.mean', f'post.{p}.std')
    if any(c not in ok for c in cols):
        return None
    resid = ok[f'post.{p}.mean'] - ok[f'truth.{p}']
    period = PERIODIC_PARAMS.get(p)
    if period is not None:
        resid = resid - period * np.round(resid / period)
    return resid / ok[f'post.{p}.std']


def _wrong_rotation(ok: pd.DataFrame) -> pd.Series:
    pull = _pull(ok, 'theta_int')
    if pull is None:
        return pd.Series(False, index=ok.index)
    return pull.abs() > 3


def _coverage(ok: pd.DataFrame, p: str, lo: str, hi: str) -> Tuple[float, float, int]:
    """(fraction inside, binomial error, n)."""
    med = ok[f'post.{p}.median'] if f'post.{p}.median' in ok else ok[f'post.{p}.mean']
    resid = ok[f'truth.{p}'] - med
    period = PERIODIC_PARAMS.get(p)
    if period is not None:
        resid = resid - period * np.round(resid / period)
    inside = (resid >= ok[f'post.{p}.{lo}'] - med) & (
        resid <= ok[f'post.{p}.{hi}'] - med
    )
    inside = inside.dropna()
    n = int(len(inside))
    f = float(inside.mean()) if n else np.nan
    err = float(np.sqrt(f * (1 - f) / n)) if n else np.nan
    return f, err, n


def _cosi_bin_index(cosi: pd.Series) -> np.ndarray:
    idx = np.full(len(cosi), -1)
    for i, (lo, hi) in enumerate(COSI_BINS):
        idx[((cosi >= lo) & (cosi < hi)).values] = i
    return idx


# ==============================================================================
# Sections
# ==============================================================================


def _section_progress(
    manifest: pd.DataFrame,
    status: pd.DataFrame,
    results: Optional[pd.DataFrame],
    commits: Dict[str, Tuple[str, str]],
    repo_url: Optional[str],
) -> str:
    n = len(manifest)
    counts = status['state'].value_counts().to_dict() if len(status) else {}
    done = int(counts.get('done', 0))
    failed = int(counts.get('failed', 0))
    in_flight = int(counts.get('in_flight', 0))
    open_ = n - done - failed - in_flight
    frac = done / n if n else 0.0
    out = [
        f'<div class="bar"><div style="width:{100 * frac:.1f}%"></div></div>'
        f'<p>{done} / {n} done ({100 * frac:.0f}%); failed {failed}; '
        f'claimed but unfinished {in_flight}; open {open_}</p>'
    ]
    if len(status) == 0 or status['start'].isna().all():
        return ''.join(out) + _na(
            'per-job progress not available (no claim timestamps)'
        )
    rows = []
    for job, g in status.groupby('job'):
        d = g[g['state'] == 'done']
        t0 = g['start'].min()
        t1 = np.nanmax(np.concatenate([g['end'].values, g['start'].values]))
        span_h = max((t1 - t0) / 3600.0, 1e-6)
        sha, note = commits.get(job, (None, ''))
        rows.append(
            {
                'job': job,
                'git_commit': _commit_html(sha, note, repo_url=repo_url),
                'claimed': len(g),
                'done': len(d),
                'failed': int((g['state'] == 'failed').sum()),
                'unfinished': int((g['state'] == 'in_flight').sum()),
                'span_h': span_h,
                'fits_per_node_hr': len(d) / span_h,
                'wall_med_min': (
                    float(np.nanmedian((d['end'] - d['start']) / 60.0))
                    if len(d)
                    else np.nan
                ),
                'first_claim': time.strftime('%Y-%m-%d %H:%M', time.localtime(t0)),
            }
        )
    jobs = pd.DataFrame(rows).sort_values('first_claim')
    out.append(_table(jobs, raw_html_cols=('git_commit',)))
    if results is not None and 'git_commit' in results.columns:
        per = results['git_commit'].fillna('-').value_counts()
        out.append('<h3>Fits per code version</h3>')
        out.append(
            _table(
                pd.DataFrame(
                    {
                        'git_commit': [
                            _commit_html(s, repo_url=repo_url) for s in per.index
                        ],
                        'fits': per.values,
                    }
                ),
                raw_html_cols=('git_commit',),
            )
        )
    latest = jobs.iloc[-1]
    remaining = open_ + failed + in_flight
    if latest['fits_per_node_hr'] > 0:
        rate = latest['fits_per_node_hr']
        out.append(
            f'<p>Remaining {remaining} fits at the latest job rate '
            f'({rate:.1f} fits/node-hr) = {remaining / rate:.1f} node-hours '
            f'(~{np.ceil(remaining / rate / 3.0):.0f} jobs of 3 h).</p>'
        )
    return ''.join(out)


def _job_commit(commits: Dict[str, Tuple[str, str]], job) -> Tuple[Optional[str], str]:
    return commits.get(str(job), (None, ''))


def _with_commit(
    df: pd.DataFrame, status: pd.DataFrame, commits, repo_url
) -> pd.DataFrame:
    """Attach a job column and an html git_commit column to a per-fit table."""
    if len(status):
        df = df.merge(status[['fit_id', 'job']], on='fit_id', how='left')
    if 'git_commit' in df.columns:
        df['git_commit'] = [
            _commit_html(s, repo_url=repo_url) for s in df['git_commit']
        ]
    elif 'job' in df.columns:
        df['git_commit'] = [
            _commit_html(*_job_commit(commits, j), repo_url=repo_url) for j in df['job']
        ]
    return df


SPEED_COSI_BINS = ((0.0, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0001))


def speed_table(
    ok: pd.DataFrame, workers_per_node: Optional[int] = None
) -> pd.DataFrame:
    """Per-class throughput of succeeded fits: one row per truth cos i bin, per
    line-SNR tercile, per escalation state, plus the whole run.

    ``fits_per_worker_h`` is the class's fit count over its summed fit
    wallclock; ``fits_per_node_h`` multiplies by ``workers_per_node`` (the
    rate a node running only that class would sustain). Shares are of the
    whole run's fits and worker-hours.
    """
    if len(ok) == 0 or 'fit_wallclock_s' not in ok.columns:
        return pd.DataFrame()
    wall_h = ok['fit_wallclock_s'] / 3600.0
    total_fits, total_wall_h = len(ok), float(wall_h.sum())
    groups: List[Tuple[str, str, np.ndarray]] = [('all', 'all', np.ones(len(ok), bool))]
    if 'truth.cosi' in ok.columns:
        cosi = ok['truth.cosi'].values.astype(float)
        for lo, hi in SPEED_COSI_BINS:
            groups.append(
                (
                    'truth cos i',
                    f'[{lo:.2f}, {min(hi, 1.0):.2f})',
                    (cosi >= lo) & (cosi < hi),
                )
            )
    snr_col = 'line_snr' if 'line_snr' in ok.columns else None
    if snr_col is not None and ok[snr_col].notna().sum() >= 3:
        snr = ok[snr_col].values.astype(float)
        edges = np.nanquantile(snr, [0.0, 1 / 3, 2 / 3, 1.0])
        for k in range(3):
            hi_inclusive = k == 2
            m = (snr >= edges[k]) & (
                (snr <= edges[k + 1]) if hi_inclusive else (snr < edges[k + 1])
            )
            groups.append(
                (
                    'line SNR (per roll) tercile',
                    f'[{edges[k]:.0f}, {edges[k + 1]:.0f}{"]" if hi_inclusive else ")"}',
                    m,
                )
            )
    if 'escalated' in ok.columns:
        esc = ok['escalated'].fillna(False).astype(bool).values
        groups.append(('escalation', 'first pass', ~esc))
        groups.append(('escalation', 'escalated', esc))
    fp_fail = None
    if {'first_attempt_max_rhat', 'first_attempt_min_ess'} <= set(ok.columns):
        fp_fail = (ok['first_attempt_max_rhat'] > 1.05) | (
            ok['first_attempt_min_ess'] < 50
        )
    rows = []
    for by, label, m in groups:
        n = int(m.sum())
        if n == 0:
            continue
        w = wall_h[m]
        sub = ok[m]
        row = {
            'by': by,
            'bin': label,
            'n': n,
            'frac_fits': n / total_fits,
            'worker_h': float(w.sum()),
            'frac_worker_h': (
                float(w.sum()) / total_wall_h if total_wall_h > 0 else np.nan
            ),
            'fits_per_worker_h': n / float(w.sum()) if w.sum() > 0 else np.nan,
            'fits_per_node_h': (
                workers_per_node * n / float(w.sum())
                if workers_per_node and w.sum() > 0
                else np.nan
            ),
            'esc_%': (
                100.0 * float(sub['escalated'].fillna(False).astype(bool).mean())
                if 'escalated' in sub.columns
                else np.nan
            ),
            'wall_med_min': float(np.median(w)) * 60.0,
            'wall_mean_min': float(np.mean(w)) * 60.0,
            'wall_p90_min': float(np.quantile(w, 0.9)) * 60.0,
        }
        if 'num_steps_total' in sub.columns:
            steps = sub['num_steps_total'].astype(float)
            row['steps_med_k'] = float(np.median(steps)) / 1e3
            row['ms_per_step'] = float(np.median(1e3 * sub['fit_wallclock_s'] / steps))
        if fp_fail is not None:
            row['fp_fail_frac'] = float(fp_fail[m].mean())
        if {'ess_g1', 'ess_g2'} <= set(sub.columns):
            row['shear_ess_med'] = float(
                np.median(np.minimum(sub['ess_g1'], sub['ess_g2']))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_speed(ok: pd.DataFrame) -> str:
    """Wall and leapfrog steps per fit against truth cos i and line SNR, and the
    fit share versus worker-hour share per cos i bin."""
    import matplotlib.pyplot as plt

    esc = (
        ok['escalated'].fillna(False).astype(bool).values
        if 'escalated' in ok
        else np.zeros(len(ok), bool)
    )
    wall_min = ok['fit_wallclock_s'].values / 60.0
    specs = [
        (
            wall_min,
            'truth.cosi',
            'truth cos i',
            'fit wall (min)',
            False,
            True,
            True,
            None,
        )
    ]
    if 'num_steps_total' in ok.columns:
        specs.append(
            (
                ok['num_steps_total'].values / 1e3,
                'truth.cosi',
                'truth cos i',
                'leapfrog steps (k)',
                False,
                True,
                True,
                None,
            )
        )
    if 'line_snr' in ok.columns:
        specs.append(
            (
                wall_min,
                'line_snr',
                'line SNR (per roll)',
                'fit wall (min)',
                True,
                True,
                True,
                None,
            )
        )
    fig = _small_multiples(ok, specs, esc, ncols=len(specs))
    for ax, (y, xcol, *_rest) in zip(fig.axes, specs):
        x = ok[xcol].values.astype(float)
        good = (
            np.isfinite(x)
            & np.isfinite(np.asarray(y, float))
            & (np.asarray(y, float) > 0)
        )
        if good.sum() >= 12:
            xs, ys, es = _binned_median(x[good], np.asarray(y, float)[good])
            ax.errorbar(
                xs,
                ys,
                yerr=es,
                color=C_INK,
                marker='o',
                ms=4,
                lw=1.2,
                zorder=5,
                label='binned median',
            )
            ax.legend(fontsize=7, frameon=False)
    html = _img_from_fig(
        fig,
        'Cost per fit vs galaxy properties',
        'Wall and steps per succeeded fit (final attempt included); squares are escalated fits; black points are quantile-binned medians.',
    )
    if 'truth.cosi' not in ok.columns:
        return html
    t = speed_table(ok)
    t = t[t['by'] == 'truth cos i']
    if len(t) == 0:
        return html
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    xs = np.arange(len(t))
    ax.bar(xs - 0.2, t['frac_fits'], width=0.4, color=C_BLUE, label='share of fits')
    ax.bar(
        xs + 0.2,
        t['frac_worker_h'],
        width=0.4,
        color=C_ORANGE,
        label='share of worker-hours',
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(t['bin'], fontsize=8)
    ax.set_xlabel('truth cos i bin', fontsize=8)
    ax.set_ylabel('fraction of run', fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    html += _img_from_fig(
        fig,
        'Where the worker-hours went',
        'A bin whose worker-hour share exceeds its fit share is slower than the run average.',
    )
    return html


def speed_cell_classes(t: pd.DataFrame) -> Dict[str, Callable]:
    """Colour rules for the speed table: the fit rate green at or above 1.15x
    the whole-run rate, red below 0.85x, yellow between; esc_% green at or
    below the whole-run value, red above."""
    if len(t) == 0:
        return {}
    ref = t[t['by'] == 'all'].iloc[0]

    def rate(col):
        def cls(v, row):
            if row['by'] == 'all' or not np.isfinite(v) or not ref[col] > 0:
                return ''
            x = v / ref[col]
            return 'good' if x >= 1.15 else ('crit' if x < 0.85 else 'warn')

        return cls

    def esc(v, row):
        if row['by'] == 'all' or not np.isfinite(v) or not np.isfinite(ref['esc_%']):
            return ''
        return 'good' if v <= ref['esc_%'] else 'crit'

    out = {'fits_per_worker_h': rate('fits_per_worker_h'), 'esc_%': esc}
    if 'fits_per_node_h' in t.columns:
        out['fits_per_node_h'] = rate('fits_per_node_h')
    return out


def _section_speed(ok: pd.DataFrame, workers_per_node: Optional[int]) -> str:
    if len(ok) == 0 or 'fit_wallclock_s' not in ok.columns:
        return _na('no succeeded fits with wallclock')
    t = speed_table(ok, workers_per_node)
    note = (
        f'fits_per_node_h assumes {workers_per_node} packed workers per node (from the spec); '
        if workers_per_node
        else 'fits_per_node_h needs dispatch.workers_per_node in the spec; '
    )
    out = [
        '<p>'
        + note
        + 'fits_per_worker_h is the class fit count over its summed fit wallclock, '
        'so it is the rate a node would sustain on that class alone. Shares are of the whole run. '
        'Colours: rate green at or above 1.15x the whole-run rate, red below 0.85x; '
        'escalation % green at or below the whole-run value.</p>',
        _table(t, max_rows=40, cell_classes=speed_cell_classes(t)),
        _guard(_plot_speed, ok),
    ]
    return ''.join(out)


def _section_failures(
    run_dir: Path,
    results: Optional[pd.DataFrame],
    status: pd.DataFrame,
    commits,
    repo_url,
) -> str:
    out = []
    if results is None:
        return _na('no results yet')
    if 'status' in results.columns:
        out.append(
            '<p>result status: '
            + _esc(results['status'].value_counts().to_dict())
            + '</p>'
        )
    ok = results[results['status'] == 'succeeded'] if 'status' in results else results
    if 'escalated' in ok.columns:
        esc = ok[ok['escalated'].fillna(False).astype(bool)].copy()
        out.append(
            f'<h3>Escalated fits: {len(esc)} of {len(ok)} '
            f'({100 * len(esc) / max(len(ok), 1):.0f}%)</h3>'
        )
        esc = _with_commit(esc, status, commits, repo_url)
        if 'fit_wallclock_s' in esc.columns:
            esc['wall_min'] = esc['fit_wallclock_s'] / 60.0
        if {'max_rhat', 'min_ess'} <= set(esc.columns):
            esc['final_max_rhat'] = esc['max_rhat']
            esc['final_min_ess'] = esc['min_ess']
            esc['gate_pass'] = (esc['max_rhat'] <= GATE_RHAT_MAX) & (
                esc['min_ess'] >= GATE_ESS_MIN
            )
        cols = [
            c
            for c in (
                'fit_id',
                'job',
                'git_commit',
                'escalation_mode',
                'escalation_n_blocks',
                'first_attempt_max_rhat',
                'first_attempt_min_ess',
                'final_max_rhat',
                'final_min_ess',
                'gate_pass',
                'divergence_rate',
                'wall_min',
                'truth.cosi',
                'line_snr',
                'restart_reason',
            )
            if c in esc.columns
        ]
        out.append(_table(esc[cols], raw_html_cols=('git_commit',)))
    else:
        out.append(_na('escalation columns not available'))
    if {'max_rhat', 'min_ess'} <= set(ok.columns):
        gate = ok[(ok['max_rhat'] > GATE_RHAT_MAX) | (ok['min_ess'] < GATE_ESS_MIN)]
        tag = ' class="bad"' if len(gate) else ''
        out.append(
            f'<h3{tag}><a href="#g-final_gate">Final gate failures</a> '
            f'(rhat > {GATE_RHAT_MAX} or ESS < {GATE_ESS_MIN:.0f}): {len(gate)}</h3>'
        )
        if len(gate):
            gate = _with_commit(gate.copy(), status, commits, repo_url)
            cols = [
                c
                for c in (
                    'fit_id',
                    'job',
                    'git_commit',
                    'max_rhat',
                    'max_rhat_param',
                    'min_ess',
                    'min_ess_param',
                    'divergence_rate',
                )
                if c in gate.columns
            ]
            out.append(_table(gate[cols], raw_html_cols=('git_commit',)))
    failed_dir = run_dir / 'status' / 'failed'
    failed = sorted(failed_dir.iterdir()) if failed_dir.is_dir() else []
    tag = ' class="bad"' if failed else ''
    out.append(f'<h3{tag}>Failed markers: {len(failed)}</h3>')
    if failed:
        rows = []
        for p in failed:
            text = p.read_text(errors='replace').strip().splitlines()
            last = next((ln for ln in reversed(text) if ln.strip()), '') if text else ''
            rows.append({'fit_id': p.name, 'last_line': last[:160]})
        out.append(_table(pd.DataFrame(rows)))
    return ''.join(out)


def _section_flags(results, status, commits, repo_url) -> str:
    if results is None:
        return _na('no results yet')
    ok = results[results['status'] == 'succeeded'] if 'status' in results else results
    present = [c for c in FLAG_COLUMNS if c in ok.columns]
    if not present:
        return _na('flag columns not available (run predates quality flags)')
    counts = {c: int(ok[c].fillna(False).astype(bool).sum()) for c in present}
    if 'n_flags' in ok.columns:
        counts['n_flags > 0'] = int((ok['n_flags'].fillna(0) > 0).sum())
    summary = pd.DataFrame([counts])
    flagged = ok[ok[present].fillna(False).astype(bool).any(axis=1)].copy()
    flagged['flags'] = [
        ','.join(c[len('flag_') :] for c in present if bool(row[c]))
        for _, row in flagged.iterrows()
    ]
    if {'postmean_chi2', 'n_data'} <= set(flagged.columns):
        flagged['chi2_excess'] = flagged['postmean_chi2'] - flagged['n_data']
    flagged = _with_commit(flagged, status, commits, repo_url)
    cols = [
        c
        for c in (
            'fit_id',
            'job',
            'git_commit',
            'flags',
            'map_pa_flip_margin',
            'map_postmean_max_dev',
            'chi2_excess',
            'max_rhat',
            'min_ess',
            'line_snr',
            'truth.cosi',
        )
        if c in flagged.columns
    ]
    return _table(summary) + _table(flagged[cols], raw_html_cols=('git_commit',))


def derived_shear(
    g1: np.ndarray, g2: np.ndarray, theta: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate sky-frame shear into the disk frame of position angle ``theta``.

    Returns ``(g_plus, g_cross)`` with ``g+ = g1 cos 2theta + g2 sin 2theta``
    and ``gx = -g1 sin 2theta + g2 cos 2theta``.
    """
    c, s = np.cos(2.0 * np.asarray(theta)), np.sin(2.0 * np.asarray(theta))
    g1, g2 = np.asarray(g1), np.asarray(g2)
    return g1 * c + g2 * s, -g1 * s + g2 * c


def with_derived_shear(ok: pd.DataFrame) -> pd.DataFrame:
    """
    Add truth / posterior columns for the derived disk-frame shear.

    Truth and posterior means are both rotated by the TRUE ``theta_int``;
    ``post.g_plus.std`` / ``post.g_cross.std`` follow from linear propagation of
    the g1 / g2 posterior widths assuming independence.
    """
    need = ('truth.g1', 'truth.g2', 'truth.theta_int', 'post.g1.mean', 'post.g2.mean')
    if any(c not in ok for c in need):
        return ok
    out = ok.copy()
    th = out['truth.theta_int'].values
    tp, tx = derived_shear(out['truth.g1'].values, out['truth.g2'].values, th)
    mp, mx = derived_shear(out['post.g1.mean'].values, out['post.g2.mean'].values, th)
    out['truth.g_plus'], out['truth.g_cross'] = tp, tx
    out['post.g_plus.mean'], out['post.g_cross.mean'] = mp, mx
    if 'post.g1.std' in out and 'post.g2.std' in out:
        c2, s2 = np.cos(2 * th) ** 2, np.sin(2 * th) ** 2
        v1, v2 = out['post.g1.std'].values ** 2, out['post.g2.std'].values ** 2
        out['post.g_plus.std'] = np.sqrt(c2 * v1 + s2 * v2)
        out['post.g_cross.std'] = np.sqrt(s2 * v1 + c2 * v2)
    return out


DERIVED_NOTE = (
    'Derived disk-frame shear: g+ = g1 cos 2theta + g2 sin 2theta, gx = -g1 sin 2theta '
    '+ g2 cos 2theta, with theta = the true theta_int for both truth and measured '
    'values (one fixed frame); measured = rotated posterior means, sigma by linear '
    'propagation of post.g1.std / post.g2.std assuming independence.'
)


def chain_derived_stats(run_dir: Path, ok: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Per-fit derived-shear statistics from the saved posterior draws.

    For every fit with ``chains/<fit_id>.npz`` the g1, g2 draws are rotated by
    the TRUE theta_int into g+, gx; returned per fit: ``chain.<p>.rank`` (fraction
    of draws below the derived truth), ``chain.<p>.in68`` / ``.in95`` (truth inside
    the 16-84 / 2.5-97.5 percentile interval) and ``chain.<p>.skew``. None when
    the run has no chains directory or no usable file.
    """
    from scipy.stats import skew

    chains_dir = Path(run_dir) / 'chains'
    need = ('truth.g1', 'truth.g2', 'truth.theta_int')
    if not chains_dir.is_dir() or any(c not in ok.columns for c in need):
        return None
    rows = []
    for _, row in ok.iterrows():
        path = chains_dir / f"{row['fit_id']}.npz"
        if not path.exists():
            continue
        saved = np.load(path)
        names = [str(n) for n in saved['param_names']]
        if 'g1' not in names or 'g2' not in names:
            continue
        draws = np.asarray(saved['samples'], dtype=float)
        th = float(row['truth.theta_int'])
        gp, gx = derived_shear(
            draws[:, names.index('g1')], draws[:, names.index('g2')], th
        )
        tp, tx = derived_shear(float(row['truth.g1']), float(row['truth.g2']), th)
        out = {'fit_id': row['fit_id'], 'chain.n_draws': int(draws.shape[0])}
        for p, d, t in (('g_plus', gp, float(tp)), ('g_cross', gx, float(tx))):
            q025, q16, q84, q975 = np.percentile(d, [2.5, 16, 84, 97.5])
            out[f'chain.{p}.rank'] = float(np.mean(d < t))
            out[f'chain.{p}.in68'] = bool(q16 <= t <= q84)
            out[f'chain.{p}.in95'] = bool(q025 <= t <= q975)
            out[f'chain.{p}.skew'] = float(skew(d))
        rows.append(out)
    return pd.DataFrame(rows) if rows else None


def _rank_mean_table(ok: pd.DataFrame, chain: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Mean truth rank per cos i bin: results columns for cosi/g1/g2, chains for g+/gx."""
    t = ok
    if chain is not None:
        t = ok.merge(chain, on='fit_id', how='left')
    rows = []
    for label, sub in _cosi_subsets(t):
        row = {'subset': label, 'n': len(sub)}
        for p in ('cosi', 'g1', 'g2'):
            col = f'truth_rank.{p}'
            row[f'rank_mean.{p}'] = float(sub[col].mean()) if col in sub else np.nan
        for p in ('g_plus', 'g_cross'):
            col = f'chain.{p}.rank'
            row[f'rank_mean.{p}'] = float(sub[col].mean()) if col in sub else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _science_rows(ok: pd.DataFrame, label: str, params=SCIENCE_PARAMS) -> List[dict]:
    rows = []
    for p in params:
        pull = _pull(ok, p)
        if pull is None:
            continue
        pull = pull.dropna()
        if len(pull) == 0:
            continue
        row = {
            'subset': label,
            'param': p,
            'n': len(pull),
            'pull_mean': float(pull.mean()),
            'pull_sem': float(pull.std() / np.sqrt(len(pull))),
            'pull_sd': float(pull.std()),
            'n_|pull|>3': int((pull.abs() > 3).sum()),
            'sigma_med': float(ok[f'post.{p}.std'].median()),
        }
        have_q = all(f'post.{p}.{q}' in ok for q in ('q16', 'q84', 'q025', 'q975'))
        row['cov68'] = _coverage(ok, p, 'q16', 'q84')[0] if have_q else np.nan
        row['cov95'] = _coverage(ok, p, 'q025', 'q975')[0] if have_q else np.nan
        rows.append(row)
    return rows


def _cosi_subsets(ok: pd.DataFrame):
    yield 'all', ok
    if 'truth.cosi' in ok.columns:
        for lo, hi in COSI_BINS:
            sub = ok[(ok['truth.cosi'] >= lo) & (ok['truth.cosi'] < hi)]
            if len(sub):
                yield f'cosi [{lo:.1f}, {min(hi, 1.0):.1f})', sub


def _section_science(ok: pd.DataFrame, chain: Optional[pd.DataFrame] = None) -> str:
    if len(ok) == 0:
        return _na('no succeeded fits yet')
    out = []
    rows = []
    for label, sub in _cosi_subsets(ok):
        rows += _science_rows(sub, label)
    out.append('<h3>Pulls and coverage</h3>')
    out.append(
        _table(pd.DataFrame(rows))
        if rows
        else _na('truth/posterior columns not available')
    )
    okd = with_derived_shear(ok)
    drows = []
    for label, sub in _cosi_subsets(okd):
        drows += _science_rows(sub, label, DERIVED_SHEAR)
    out.append('<h3>Derived: disk-frame shear (g+, gx)</h3>')
    if drows:
        d = pd.DataFrame(drows).drop(columns=['cov68', 'cov95'])
        if chain is not None:
            merged = okd.merge(chain, on='fit_id', how='inner')
            for i, r in d.iterrows():
                sub = merged
                if r['subset'] != 'all':
                    sub = [g for lab, g in _cosi_subsets(merged) if lab == r['subset']]
                    sub = sub[0] if sub else merged.iloc[0:0]
                p = r['param']
                if len(sub):
                    d.loc[i, 'cov68 (from chains)'] = float(
                        sub[f'chain.{p}.in68'].mean()
                    )
                    d.loc[i, 'cov95 (from chains)'] = float(
                        sub[f'chain.{p}.in95'].mean()
                    )
                    d.loc[i, 'rank_mean (from chains)'] = float(
                        sub[f'chain.{p}.rank'].mean()
                    )
                    d.loc[i, 'skew_med (from chains)'] = float(
                        sub[f'chain.{p}.skew'].median()
                    )
            out.append(_table(d))
            out.append(
                f'<p class="cap">{_esc(DERIVED_NOTE)} Coverage, truth rank and skew of '
                'g+, gx are computed from the saved posterior draws rotated by the true '
                'theta_int (from chains).</p>'
            )
        else:
            out.append(_table(d))
            out.append(
                f'<p class="cap">{_esc(DERIVED_NOTE)} Coverage is not available for '
                'derived quantities (no interval columns; no chains directory).</p>'
            )
        out.append('<h3>Mean truth rank per cos i bin (0.5 is calibrated)</h3>')
        out.append(_table(_rank_mean_table(ok, chain)))
        if chain is None:
            out.append(_na('g+, gx ranks not available without a chains directory'))
    else:
        out.append(_na('derived shear needs truth.theta_int and g1/g2 columns'))
    stats = {}
    if {'ess_g1', 'ess_g2'} <= set(ok.columns):
        mse = np.minimum(ok['ess_g1'], ok['ess_g2'])
        stats['shear_ess'] = float(mse.median())
        stats['shear ESS < 200'] = f'{100 * (mse < 200).mean():.0f}%'
    if 'fit_wallclock_s' in ok.columns:
        w = ok['fit_wallclock_s'] / 60.0
        stats['wall median / mean / max [min]'] = (
            f'{w.median():.1f} / {w.mean():.1f} / {w.max():.1f}'
        )
    if 'num_steps_total' in ok.columns:
        stats['num_steps_total'] = float(ok['num_steps_total'].median())
        if 'fit_wallclock_s' in ok.columns:
            stats['ms per step (median)'] = (
                f'{(1e3 * ok["fit_wallclock_s"] / ok["num_steps_total"]).median():.2f}'
            )
    if 'precond_wallclock_s' in ok.columns:
        stats['preconditioner wall median [s]'] = (
            f'{ok["precond_wallclock_s"].median():.0f}'
        )
    if 'divergence_rate' in ok.columns:
        stats['fits with divergences'] = (
            f'{100 * (ok["divergence_rate"] > 0).mean():.0f}%'
        )
        stats['fits with divergence_rate > 0.01'] = (
            f'{int((ok["divergence_rate"] > 0.01).sum())}'
        )
    out.append('<h3>Sampler cost (medians)</h3>')
    out.append(_table(pd.DataFrame([stats])) if stats else _na())
    return ''.join(out)


# ------------------------------------------------------------------------------
# Convergence gate
# ------------------------------------------------------------------------------

GATE_RHATS = (1.05, 1.025, 1.01)
GATE_ESSES = (50.0, 100.0, 200.0)
SHEAR_ESS_THRESHOLDS = (100.0, 200.0, 300.0)


def gate_table(ok: pd.DataFrame, first_attempt: bool = False) -> pd.DataFrame:
    """
    Fraction and count of fits failing each (rhat, ESS) gate, plus shear-ESS gates.

    ``first_attempt=True`` uses ``first_attempt_max_rhat`` / ``first_attempt_min_ess``
    where present (escalated fits) and the final values elsewhere, i.e. what the
    gate saw before any escalation ran.
    """
    rhat_col, ess_col = 'max_rhat', 'min_ess'
    rhat = ok[rhat_col].astype(float).copy()
    ess = ok[ess_col].astype(float).copy()
    if first_attempt:
        for col, target in (
            ('first_attempt_max_rhat', rhat),
            ('first_attempt_min_ess', ess),
        ):
            if col in ok.columns:
                fa = ok[col].astype(float)
                target[fa.notna()] = fa[fa.notna()]
    n = int(len(ok))
    rows = []
    for r in GATE_RHATS:
        for e in GATE_ESSES:
            fail = int(((rhat > r) | (ess < e)).sum())
            rows.append(
                {
                    'rhat_max': r,
                    'ess_min': e,
                    'shear_ess_min': np.nan,
                    'n_fail': fail,
                    'frac_fail': fail / n if n else np.nan,
                }
            )
    if {'ess_g1', 'ess_g2'} <= set(ok.columns):
        mse = np.minimum(ok['ess_g1'], ok['ess_g2']).astype(float)
        for t in SHEAR_ESS_THRESHOLDS:
            fail = int((mse < t).sum())
            rows.append(
                {
                    'rhat_max': np.nan,
                    'ess_min': np.nan,
                    'shear_ess_min': t,
                    'n_fail': fail,
                    'frac_fail': fail / n if n else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _plot_gate(ok: pd.DataFrame) -> str:
    import matplotlib.pyplot as plt

    if not {'max_rhat', 'min_ess'} <= set(ok.columns):
        raise KeyError('max_rhat / min_ess missing')
    esc = (
        ok['escalated'].fillna(False).astype(bool).values
        if 'escalated' in ok
        else np.zeros(len(ok), bool)
    )
    have_shear = {'ess_g1', 'ess_g2'} <= set(ok.columns)
    fig, axes = plt.subplots(
        1, 2 if have_shear else 1, figsize=(10.5, 3.6), squeeze=False
    )
    ax = axes[0][0]
    ax.scatter(
        ok['max_rhat'][~esc],
        ok['min_ess'][~esc],
        s=14,
        color=C_BLUE,
        label='first pass',
    )
    ax.scatter(
        ok['max_rhat'][esc],
        ok['min_ess'][esc],
        s=26,
        marker='s',
        color=C_ORANGE,
        label='escalated',
    )
    ax.axvline(GATE_RHAT_MAX, color=C_CRIT, lw=1.0, ls='--', label='gate rhat 1.05')
    ax.axhline(GATE_ESS_MIN, color=C_CRIT, lw=1.0, ls=':', label='gate ESS 50')
    ax.set_yscale('log')
    ax.set_xlabel('final max_rhat', fontsize=8)
    ax.set_ylabel('final min_ess', fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    _style_axes(ax)
    if have_shear:
        ax = axes[0][1]
        mse = np.minimum(ok['ess_g1'], ok['ess_g2'])
        ax.scatter(
            ok['max_rhat'][~esc], mse[~esc], s=14, color=C_BLUE, label='first pass'
        )
        ax.scatter(
            ok['max_rhat'][esc],
            mse[esc],
            s=26,
            marker='s',
            color=C_ORANGE,
            label='escalated',
        )
        ax.axvline(GATE_RHAT_MAX, color=C_CRIT, lw=1.0, ls='--')
        ax.axhline(200, color='#555', lw=1.0, ls='--', label='shear ESS 200')
        ax.set_yscale('log')
        ax.set_xlabel('final max_rhat', fontsize=8)
        ax.set_ylabel('min(ess_g1, ess_g2)', fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        _style_axes(ax)
    fig.tight_layout()
    return _img_from_fig(
        fig,
        'Final convergence diagnostics per fit',
        'Healthy: every point right of nothing and above both gate lines (rhat below '
        '1.05, ESS above 50) after escalation; escalated fits (orange squares) show '
        'where the first pass fell short. Right: shear ESS above 200 keeps the '
        'posterior width error under ~5%.',
    )


def _section_gate(ok: pd.DataFrame) -> str:
    if len(ok) == 0:
        return _na('no succeeded fits yet')
    out = [_guard(_plot_gate, ok)]
    final = gate_table(ok)
    first = gate_table(ok, first_attempt=True)
    tbl = final.rename(
        columns={'n_fail': 'n_fail_final', 'frac_fail': 'frac_fail_final'}
    )
    tbl['n_fail_first_attempt'] = first['n_fail']
    tbl['frac_fail_first_attempt'] = first['frac_fail']
    out.append(
        '<h3><a href="#g-gate_table">Gate table</a>: fits failing stricter gates</h3>'
    )
    out.append(_table(tbl))
    out.append(
        '<p class="cap">Read: a row is the fraction (and count) of the succeeded fits '
        'with final max_rhat above rhat_max OR min_ess below ess_min (shear rows: '
        'min(ess_g1, ess_g2) below shear_ess_min alone); the first-attempt columns '
        'apply the same gate to the values before escalation, i.e. the escalation '
        'load a stricter production gate would have produced.</p>'
    )
    return ''.join(out)


# ------------------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------------------


def _gpu_csv(run_dir: Path) -> Optional[pd.DataFrame]:
    files = [
        p
        for p in sorted(run_dir.glob('gpu_*.csv'))
        if not p.name.endswith('_procs.csv')
    ]
    if not files:
        return None
    frames = []
    for p in files:
        g = pd.read_csv(
            p,
            header=None,
            names=['ts', 'used', 'total', 'util', 'memutil'],
            skipinitialspace=True,
        )
        for c in ('used', 'total'):
            g[c] = g[c].astype(str).str.replace(' MiB', '').astype(float) / 1024.0
        for c in ('util', 'memutil'):
            g[c] = g[c].astype(str).str.replace(' %', '').astype(float)
        g['t'] = pd.to_datetime(g['ts'], format='%Y/%m/%d %H:%M:%S.%f')
        g['file'] = p.name
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def _plot_pulls(ok: pd.DataFrame, params=SCIENCE_PARAMS, title_suffix: str = '') -> str:
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    params = [p for p in params if _pull(ok, p) is not None]
    if not params:
        raise KeyError('no pull columns')
    fig, axes = plt.subplots(
        1, len(params), figsize=(3.4 * len(params), 3.0), squeeze=False
    )
    edges = np.linspace(-5, 5, 41)
    x = np.linspace(-5, 5, 201)
    for ax, p in zip(axes[0], params):
        pull = _pull(ok, p).dropna()
        clipped = np.clip(pull, -5, 5)
        ax.hist(
            clipped,
            bins=edges,
            color=C_BLUE,
            alpha=0.85,
            label=f'pulls (n={len(pull)})',
        )
        ax.plot(
            x,
            len(pull) * (edges[1] - edges[0]) * norm.pdf(x),
            color=C_TEXT2,
            lw=1.6,
            label='N(0,1)',
        )
        ax.axvline(0, color='#999', lw=0.8)
        ax.set_title(f'{p}: mean {pull.mean():+.2f}, sd {pull.std():.2f}', fontsize=9)
        ax.set_xlabel('pull', fontsize=8)
        ax.legend(
            fontsize=7, frameon=False, title=f'sd = {pull.std():.2f}', title_fontsize=7
        )
        _style_axes(ax)
    fig.tight_layout()
    return _img_from_fig(
        fig,
        'Pull histograms with the unit normal' + title_suffix,
        'Healthy: the histogram follows the N(0,1) curve (mean 0, sd 1), no |pull| > 3 '
        'tail. Values beyond 5 are clipped into the edge bins.',
    )


def _plot_measured_vs_true(
    ok: pd.DataFrame, params=SCIENCE_PARAMS, title_suffix: str = ''
) -> str:
    import matplotlib.pyplot as plt

    params = [
        p
        for p in params
        if all(c in ok for c in (f'truth.{p}', f'post.{p}.mean', f'post.{p}.std'))
    ]
    if not params or 'truth.cosi' not in ok:
        raise KeyError('no truth/posterior columns')
    wrong = _wrong_rotation(ok).values
    bins = _cosi_bin_index(ok['truth.cosi'])
    fig, axes = plt.subplots(
        1, len(params), figsize=(3.4 * len(params), 3.3), squeeze=False
    )
    for ax, p in zip(axes[0], params):
        t = ok[f'truth.{p}'].values.astype(float)
        mu = ok[f'post.{p}.mean'].values.astype(float).copy()
        sd = ok[f'post.{p}.std'].values.astype(float)
        period = PERIODIC_PARAMS.get(p)
        if period is not None:
            mu = t + ((mu - t + period / 2) % period - period / 2)
        for i, (lo, hi) in enumerate(COSI_BINS):
            m = (bins == i) & ~wrong
            if m.any():
                ax.errorbar(
                    t[m],
                    mu[m],
                    yerr=sd[m],
                    fmt='o',
                    ms=3.5,
                    lw=0.7,
                    color=C_SEQ[i],
                    alpha=0.85,
                    label=f'cos i [{lo:.1f}, {min(hi, 1):.1f})',
                )
        if wrong.any():
            ax.errorbar(
                t[wrong],
                mu[wrong],
                yerr=sd[wrong],
                fmt='x',
                ms=7,
                lw=0.8,
                color=C_CRIT,
                label='wrong rotation',
            )
        # x axis follows the truth range only; the 1:1 line is clipped to it
        finite = np.isfinite(t)
        lo_, hi_ = float(np.min(t[finite])), float(np.max(t[finite]))
        pad = 0.1 * (hi_ - lo_ if hi_ > lo_ else max(abs(hi_), 1e-3))
        ax.plot(
            [lo_ - pad, hi_ + pad],
            [lo_ - pad, hi_ + pad],
            color=C_INK,
            lw=1.2,
            ls='--',
            label='truth',
        )
        ax.set_xlim(lo_ - pad, hi_ + pad)
        ax.set_xlabel(f'true {p}', fontsize=8)
        ax.set_ylabel(f'posterior mean {p}', fontsize=8)
        _style_axes(ax)
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return _img_from_fig(
        fig,
        'Measured vs true (posterior mean with 1-sigma bars)' + title_suffix,
        'Healthy: points scatter about the dashed 1:1 line with error bars covering '
        'it ~68% of the time; the wrong-rotation markers (wrapped theta pull > 3) '
        'should be rare and confined to theta_int. x range = truth range padded 10%; '
        'shear is sky-frame g1, g2.',
    )


def _plot_shear_bias(
    ok: pd.DataFrame, comps=('g1', 'g2'), title_suffix: str = ''
) -> str:
    import matplotlib.pyplot as plt

    comps = [
        c
        for c in comps
        if all(k in ok for k in (f'truth.{c}', f'post.{c}.mean', f'post.{c}.std'))
    ]
    if not comps:
        raise KeyError('no shear columns')
    fig, axes = plt.subplots(
        2, len(comps), figsize=(4.2 * len(comps), 5.6), squeeze=False
    )
    for j, c in enumerate(comps):
        t = ok[f'truth.{c}']
        resid = ok[f'post.{c}.mean'] - t
        pull = resid / ok[f'post.{c}.std']
        n_bins = min(6, max(2, len(t) // 8))
        q = pd.qcut(t, n_bins, duplicates='drop')
        for ax, y, ylabel in (
            (axes[0][j], resid, f'mean(post.mean - truth) {c}'),
            (axes[1][j], pull, f'mean pull {c}'),
        ):
            grp = y.groupby(q, observed=True)
            xc = t.groupby(q, observed=True).mean()
            ax.errorbar(
                xc,
                grp.mean(),
                yerr=grp.std() / np.sqrt(grp.count()),
                fmt='o',
                color=C_BLUE,
                ms=5,
                lw=1.2,
            )
            ax.axhline(0, color='#999', lw=0.8)
            ax.set_xlabel(f'true {c}', fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            _style_axes(ax)
    fig.tight_layout()
    return _img_from_fig(
        fig,
        'Shear bias vs true shear' + title_suffix,
        'Healthy: every bin consistent with zero within its standard error, on the '
        'residual scale (top) and the pull scale (bottom); a slope would be a '
        'multiplicative bias, an offset an additive one.',
    )


def _small_multiples(ok: pd.DataFrame, specs, esc, ncols: int = 3):
    """Scatter grid; each spec = (ycol_or_array, xcol, xlabel, ylabel, logx, logy, colour_esc, hline)."""
    import matplotlib.pyplot as plt

    n = len(specs)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.8 * ncols, 3.1 * nrows), squeeze=False
    )
    for k, (y, xcol, xlabel, ylabel, logx, logy, colour_esc, hline) in enumerate(specs):
        ax = axes[k // ncols][k % ncols]
        x = ok[xcol].values.astype(float)
        y = np.asarray(y, dtype=float)
        if colour_esc:
            ax.scatter(x[~esc], y[~esc], s=12, color=C_BLUE, label='first pass')
            ax.scatter(
                x[esc], y[esc], s=22, color=C_ORANGE, marker='s', label='escalated'
            )
            ax.legend(fontsize=7, frameon=False)
        else:
            ax.scatter(x, y, s=12, color=C_BLUE)
        if hline is not None:
            ax.axhline(hline, color='#999', lw=0.8, ls='--')
        if logx:
            ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        _style_axes(ax)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    fig.tight_layout()
    return fig


def _plot_scaling(ok: pd.DataFrame) -> str:
    need = ('line_snr', 'truth.cosi')
    if any(c not in ok for c in need):
        raise KeyError('line_snr / truth.cosi missing')
    esc = (
        ok['escalated'].fillna(False).astype(bool).values
        if 'escalated' in ok
        else np.zeros(len(ok), bool)
    )
    specs = []
    for c in ('g1', 'g2'):
        if f'post.{c}.std' in ok:
            specs.append(
                (
                    ok[f'post.{c}.std'],
                    'line_snr',
                    'line SNR',
                    f'sigma {c}',
                    True,
                    False,
                    False,
                    None,
                )
            )
            specs.append(
                (
                    ok[f'post.{c}.std'],
                    'truth.cosi',
                    'true cos i',
                    f'sigma {c}',
                    False,
                    False,
                    False,
                    None,
                )
            )
    for c in ('g1', 'g2'):
        if f'ess_{c}' in ok:
            specs.append(
                (
                    ok[f'ess_{c}'],
                    'line_snr',
                    'line SNR',
                    f'ess_{c}',
                    True,
                    True,
                    False,
                    200,
                )
            )
            specs.append(
                (
                    ok[f'ess_{c}'],
                    'truth.cosi',
                    'true cos i',
                    f'ess_{c}',
                    False,
                    True,
                    False,
                    200,
                )
            )
    if 'num_steps_total' in ok:
        specs.append(
            (
                ok['num_steps_total'],
                'truth.cosi',
                'true cos i',
                'leapfrog steps (total)',
                False,
                True,
                False,
                None,
            )
        )
    if 'fit_wallclock_s' in ok:
        specs.append(
            (
                ok['fit_wallclock_s'] / 60.0,
                'truth.cosi',
                'true cos i',
                'fit wall [min]',
                False,
                True,
                True,
                None,
            )
        )
    if not specs:
        raise KeyError('no scaling columns')
    fig = _small_multiples(ok, specs, esc, ncols=4)
    return _img_from_fig(
        fig,
        'Precision, ESS and cost scaling (g1 and g2 separately)',
        'Healthy: each shear sigma falls with line SNR and is flat or mildly rising '
        'toward face-on; ESS mostly above the dashed 200 line; steps and wall rise '
        'smoothly toward face-on without a detached slow tail; escalated fits sit in '
        'that tail, not scattered across the plane.',
    )


def _plot_scaling_derived(okd: pd.DataFrame) -> str:
    if any(c not in okd for c in ('post.g_plus.std', 'post.g_cross.std', 'line_snr')):
        raise KeyError('derived shear sigma missing')
    esc = np.zeros(len(okd), bool)
    specs = []
    for c, label in (('g_plus', 'sigma g+'), ('g_cross', 'sigma gx')):
        specs.append(
            (
                okd[f'post.{c}.std'],
                'line_snr',
                'line SNR',
                label,
                True,
                False,
                False,
                None,
            )
        )
        specs.append(
            (
                okd[f'post.{c}.std'],
                'truth.cosi',
                'true cos i',
                label,
                False,
                False,
                False,
                None,
            )
        )
    fig = _small_multiples(okd, specs, esc, ncols=4)
    return _img_from_fig(
        fig,
        'Derived: disk-frame shear precision (sigma g+, sigma gx)',
        DERIVED_NOTE
        + ' Healthy: gx (the component the velocity field constrains) at or below g+.',
    )


def _plot_coverage(ok: pd.DataFrame) -> str:
    import matplotlib.pyplot as plt

    rows = []
    for p in SCIENCE_PARAMS:
        if (
            not all(f'post.{p}.{q}' in ok for q in ('q16', 'q84', 'q025', 'q975'))
            or f'truth.{p}' not in ok
        ):
            continue
        f68, e68, n = _coverage(ok, p, 'q16', 'q84')
        f95, e95, _ = _coverage(ok, p, 'q025', 'q975')
        rows.append((p, f68, e68, f95, e95, n))
    if not rows:
        raise KeyError('no interval columns')
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    x = np.arange(len(rows))
    w = 0.36
    ax.bar(
        x - w / 2,
        [r[1] for r in rows],
        w,
        yerr=[r[2] for r in rows],
        color=C_BLUE,
        label='68% interval',
        capsize=2,
    )
    ax.bar(
        x + w / 2,
        [r[3] for r in rows],
        w,
        yerr=[r[4] for r in rows],
        color=C_AQUA,
        label='95% interval',
        capsize=2,
    )
    ax.axhline(0.68, color=C_BLUE, lw=0.9, ls='--', label='nominal 0.68')
    ax.axhline(0.95, color=C_AQUA, lw=0.9, ls='--', label='nominal 0.95')
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('fraction of truths inside', fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5))
    _style_axes(ax)
    fig.tight_layout()
    return _img_from_fig(
        fig,
        f'Interval coverage per parameter (n = {rows[0][5]})',
        'Healthy: bars reach the dashed nominal lines (0.68, 0.95) within their '
        'binomial error bars; a low bar means over-confident intervals, a high bar '
        'conservative ones. Coverage is not available for the derived g+, gx.',
    )


def _plot_derived_rank_hist(ok: pd.DataFrame, chain: pd.DataFrame) -> str:
    import matplotlib.pyplot as plt
    from scipy.stats import kstest

    t = ok.merge(chain, on='fit_id', how='inner')
    if len(t) == 0 or 'truth.cosi' not in t:
        raise KeyError('no chain-based ranks')
    subsets = (
        ('all', t),
        ('edge-on (cos i < 0.3)', t[t['truth.cosi'] < 0.3]),
        ('face-on (cos i > 0.7)', t[t['truth.cosi'] > 0.7]),
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6))
    edges = np.linspace(0, 1, 11)
    for i, (p, lab) in enumerate((('g_plus', 'g+'), ('g_cross', 'gx'))):
        for j, (name, sub) in enumerate(subsets):
            ax = axes[i][j]
            r = sub[f'chain.{p}.rank'].dropna().values
            if len(r) == 0:
                ax.set_visible(False)
                continue
            pval = kstest(r, 'uniform').pvalue
            ax.hist(
                r,
                bins=edges,
                color=C_BLUE,
                alpha=0.85,
                label=f'n={len(r)}, KS p={pval:.2g}, mean {r.mean():.2f}',
            )
            ax.axhline(len(r) / 10, color='#999', lw=0.8, ls='--')
            ax.set_title(f'{lab} truth rank, {name}', fontsize=9)
            ax.set_xlabel('rank of truth among draws', fontsize=8)
            ax.legend(fontsize=7, frameon=False)
            _style_axes(ax)
    fig.tight_layout()
    return _img_from_fig(
        fig,
        'Derived g+, gx truth-rank histograms (from chains)',
        'One entry per fit: the fraction of that fit\'s posterior draws (rotated into '
        'the disk frame by the true theta_int) that fall below the true g+ or gx, i.e. '
        'the truth\'s percentile in its own posterior. A calibrated posterior puts the '
        'truth at a uniformly random percentile, so the histogram is flat at the dashed '
        'level with mean 0.5 and KS p above ~0.05. A pile-up at low rank means most '
        'draws sit above the truth (posterior biased high), at high rank biased low; a '
        'U shape means the posteriors are too narrow, a central hump too wide.',
    )


def _plot_rotation(ok: pd.DataFrame) -> str:
    import matplotlib.pyplot as plt

    if 'map_pa_flip_margin' not in ok:
        raise KeyError('map_pa_flip_margin missing')
    m = ok['map_pa_flip_margin'].astype(float)
    wrong = _wrong_rotation(ok)
    finite = m[np.isfinite(m) & (m > 0)]
    n_inf = int((~np.isfinite(m)).sum())
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    if len(finite):
        edges = np.logspace(
            np.log10(max(finite.min(), 1e-3)), np.log10(finite.max() * 1.1), 30
        )
        ax.hist(
            finite,
            bins=edges,
            color=C_BLUE,
            alpha=0.85,
            label=f'all fits (n={len(finite)}, {n_inf} with no flipped basin)',
        )
        fw = finite[wrong.reindex(finite.index).fillna(False).astype(bool)]
        if len(fw):
            ax.hist(fw, bins=edges, color=C_CRIT, label=f'wrong rotation (n={len(fw)})')
        ax.set_xscale('log')
    ax.axvline(
        ROTATION_FLAG_NATS,
        color=C_WARN,
        lw=1.4,
        ls='--',
        label=f'flag threshold {ROTATION_FLAG_NATS:.0f} nats',
    )
    ax.set_xlabel('map_pa_flip_margin [nats]', fontsize=8)
    ax.set_ylabel('fits', fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    return _img_from_fig(
        fig,
        'Rotation-direction ambiguity',
        'Healthy: wrong-rotation fits (red) sit left of the dashed flag threshold, so '
        'the flag catches them; any red bar right of the line is a wrong mode the '
        'flag misses.',
    )


def _plot_gpu(run_dir: Path) -> str:
    import matplotlib.pyplot as plt

    g = _gpu_csv(run_dir)
    if g is None:
        raise FileNotFoundError('no gpu_*.csv in run dir')
    fig, (ax_m, ax_u) = plt.subplots(2, 1, figsize=(11, 4.8), sharex=True)
    total = float(g['total'].iloc[0])
    for k, (name, sub) in enumerate(g.groupby('file')):
        t = (sub['t'] - sub['t'].iloc[0]).dt.total_seconds() / 60
        colour = (C_BLUE, C_ORANGE, C_AQUA)[k % 3]
        ax_m.plot(t, sub['used'], color=colour, lw=1.6, label=f'{name}: memory used')
        ax_u.plot(
            t, sub['util'], color=colour, lw=1.6, label=f'{name}: GPU utilisation'
        )
        ax_u.plot(
            t,
            sub['memutil'],
            color=colour,
            lw=1.2,
            ls=':',
            label=f'{name}: memory-bandwidth utilisation',
        )
    ax_m.axhline(
        total, color='#555', lw=1.0, ls='--', label=f'total GPU memory {total:.0f} GiB'
    )
    ax_m.set_ylabel('GiB', fontsize=8)
    ax_m.set_ylim(0, total * 1.08)
    ax_u.set_ylabel('%', fontsize=8)
    ax_u.set_ylim(0, 105)
    ax_u.set_xlabel('minutes since job start', fontsize=8)
    for ax in (ax_m, ax_u):
        ax.legend(fontsize=7, frameon=False, loc='center right')
        _style_axes(ax)
    ax_m.set_title(
        f'GPU memory used: median {g["used"].median():.1f} GiB, max '
        f'{g["used"].max():.1f} of {total:.0f} GiB; utilisation median '
        f'{g["util"].median():.0f}%, memory bandwidth {g["memutil"].median():.0f}%',
        fontsize=9,
    )
    fig.tight_layout()
    return _img_from_fig(
        fig,
        'GPU memory and utilisation',
        'Memory (top) and utilisation (bottom) share the time axis; two panels '
        'rather than two y-scales on one plot. Healthy: memory flat and well below '
        'the dashed total, no spikes at fit ends; utilisation near 100% means the '
        'workers are compute-bound, so more workers add little.',
    )


def _section_plots(
    run_dir: Path, ok: pd.DataFrame, chain: Optional[pd.DataFrame] = None
) -> str:
    import matplotlib

    matplotlib.use('Agg')
    out = []
    if len(ok):
        okd = with_derived_shear(ok)
        out.append(_guard(_plot_pulls, ok))
        out.append(_guard(_plot_measured_vs_true, ok))
        out.append(_guard(_plot_shear_bias, ok))
        out.append(_guard(_plot_coverage, ok))
        out.append(_guard(_plot_scaling, ok))
        out.append(_guard(_plot_rotation, ok))
        out.append('<h3>Derived: disk-frame shear (g+, gx)</h3>')
        out.append(f'<p class="cap">{_esc(DERIVED_NOTE)}</p>')
        out.append(_guard(_plot_pulls, okd, DERIVED_SHEAR, ' (derived g+, gx)'))
        out.append(
            _guard(_plot_measured_vs_true, okd, DERIVED_SHEAR, ' (derived g+, gx)')
        )
        out.append(_guard(_plot_shear_bias, okd, DERIVED_SHEAR, ' (derived g+, gx)'))
        out.append(_guard(_plot_scaling_derived, okd))
        if chain is not None:
            out.append(_guard(_plot_derived_rank_hist, ok, chain))
        else:
            out.append(_na('derived rank histograms need a chains directory'))
    else:
        out.append(_na('no succeeded fits yet'))
    out.append(_guard(_plot_gpu, run_dir))
    diag = run_dir / 'diagnostics'
    if diag.is_dir():
        pngs = sorted(diag.glob('*.png'))
        if pngs:
            out.append('<h3>Report figures already in diagnostics/</h3>')
            for png in pngs:
                out.append(_guard(_img_from_path, png))
    return ''.join(out)


# ------------------------------------------------------------------------------
# Headline science
# ------------------------------------------------------------------------------


def _binned_median(x: np.ndarray, y: np.ndarray, n_bins: int = 6):
    """Quantile-binned medians with 1.253 MAD / sqrt(n) error bars."""
    q = pd.qcut(pd.Series(x), min(n_bins, max(2, len(x) // 6)), duplicates='drop')
    xs, ys, es = [], [], []
    for _, idx in pd.Series(np.arange(len(x))).groupby(q, observed=True):
        yy = y[idx.values]
        xs.append(float(np.median(x[idx.values])))
        ys.append(float(np.median(yy)))
        mad = float(np.median(np.abs(yy - np.median(yy))))
        es.append(1.253 * mad / np.sqrt(len(yy)))
    return np.array(xs), np.array(ys), np.array(es)


def _plot_shape_noise(okd: pd.DataFrame, xcol: str, xlabel: str, logx: bool) -> str:
    import matplotlib.pyplot as plt

    comps = [
        (c, lab)
        for c, lab in (
            ('g1', 'sigma g1'),
            ('g2', 'sigma g2'),
            ('g_plus', 'sigma g+'),
            ('g_cross', 'sigma gx'),
        )
        if f'post.{c}.std' in okd
    ]
    if not comps or xcol not in okd:
        raise KeyError('shear sigma or axis column missing')
    x = okd[xcol].values.astype(float)
    fig, axes = plt.subplots(
        1, len(comps), figsize=(3.5 * len(comps), 3.2), squeeze=False
    )
    for ax, (c, lab) in zip(axes[0], comps):
        y = okd[f'post.{c}.std'].values.astype(float)
        ax.scatter(x, y, s=10, color=C_SEQ[0], label='per fit')
        bx, by, be = _binned_median(x, y)
        ax.errorbar(
            bx, by, yerr=be, fmt='o-', color=C_INK, ms=5, lw=1.4, label='binned median'
        )
        ax.axhline(
            PHOTOMETRIC_SHAPE_NOISE,
            color=C_ORANGE,
            lw=1.2,
            ls='--',
            label='photometric WL per-component shape noise 0.26',
        )
        if logx:
            ax.set_xscale('log')
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(lab, fontsize=8)
        ax.set_ylim(0, max(0.3, float(np.nanmax(y)) * 1.05))
        _style_axes(ax)
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return _img_from_fig(
        fig,
        f'Effective per-component shape noise vs {xlabel}',
        'In kinematic lensing the per-galaxy posterior sigma of a shear component IS '
        'the effective shape noise (the paper-1 headline); the dashed line is the '
        'photometric weak-lensing value 0.26. Error bars: 1.253 MAD / sqrt(n) per '
        'quantile bin. The sample so far is weighted toward the hard tail '
        '(hard_first claim order).',
    )


def _plot_shape_noise_grid(ok: pd.DataFrame) -> str:
    import matplotlib.pyplot as plt

    if any(
        c not in ok for c in ('post.g1.std', 'post.g2.std', 'line_snr', 'truth.cosi')
    ):
        raise KeyError('shear sigma / axes missing')
    sig = np.sqrt(0.5 * (ok['post.g1.std'] ** 2 + ok['post.g2.std'] ** 2)).values
    cosi_idx = _cosi_bin_index(ok['truth.cosi'])
    snr_bins = pd.qcut(ok['line_snr'], min(3, max(1, len(ok) // 10)), duplicates='drop')
    cats = list(snr_bins.cat.categories)
    grid = np.full((len(COSI_BINS), len(cats)), np.nan)
    counts = np.zeros_like(grid, dtype=int)
    for i in range(len(COSI_BINS)):
        for j, cat in enumerate(cats):
            m = (cosi_idx == i) & (snr_bins == cat).values
            if m.any():
                grid[i, j] = np.median(sig[m])
                counts[i, j] = int(m.sum())
    fig, ax = plt.subplots(figsize=(1.6 * len(cats) + 3.5, 3.0))
    im = ax.imshow(grid, cmap='Blues', aspect='auto', origin='lower')
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if counts[i, j]:
                ax.text(
                    j,
                    i,
                    f'{grid[i, j]:.3f}\nn={counts[i, j]}',
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='#111' if grid[i, j] < np.nanmax(grid) * 0.7 else '#fff',
                )
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([f'SNR {c.left:.0f}-{c.right:.0f}' for c in cats], fontsize=8)
    ax.set_yticks(range(len(COSI_BINS)))
    ax.set_yticklabels(
        [f'cos i [{lo:.1f}, {min(hi, 1):.1f})' for lo, hi in COSI_BINS], fontsize=8
    )
    fig.colorbar(im, ax=ax, label='median per-component sigma_g')
    ax.set_title('median sqrt((sigma_g1^2 + sigma_g2^2)/2) per cell', fontsize=9)
    fig.tight_layout()
    return _img_from_fig(
        fig,
        'Shape noise per (cos i bin x line-SNR bin)',
        'Median per-component shear sigma in each cell with the fit count; healthy: '
        'darker (larger) toward low SNR, without a cell far off its neighbours.',
    )


def _section_headline(ok: pd.DataFrame) -> str:
    if len(ok) == 0:
        return _na('no succeeded fits yet')
    okd = with_derived_shear(ok)
    return ''.join(
        [
            _guard(_plot_shape_noise, okd, 'line_snr', 'line SNR', True),
            _guard(_plot_shape_noise, okd, 'truth.cosi', 'true cos i', False),
            _guard(_plot_shape_noise_grid, ok),
        ]
    )


# ==============================================================================
# vcirc constraint beyond the TF prior
# ==============================================================================

VCIRC_PRIOR_INFORMATIVE = (
    0.8  # posterior/prior width ratio below which the data constrain vcirc
)


def vcirc_prior_table(ok: pd.DataFrame) -> pd.DataFrame:
    """Posterior vs prior width of vcirc per cos i subset.

    The vcirc fit prior is log-normal with width ``pop.prior_vcirc_sigma_dex``;
    the posterior width in dex is ``std / (mean ln 10)``. A ratio near 1 means
    the kinematics add nothing to vcirc and the inclination (hence the shear)
    is set by the prior width.
    """
    need = ('post.vel.vcirc.mean', 'post.vel.vcirc.std', 'pop.prior_vcirc_sigma_dex')
    if any(c not in ok for c in need):
        raise KeyError('vcirc posterior or prior columns missing')
    rows = []
    for label, sub in _cosi_subsets(ok):
        post_dex = sub['post.vel.vcirc.std'] / (sub['post.vel.vcirc.mean'] * np.log(10))
        ratio = post_dex / sub['pop.prior_vcirc_sigma_dex']
        row = {
            'subset': label,
            'n': int(len(sub)),
            'prior_sigma_dex': float(sub['pop.prior_vcirc_sigma_dex'].median()),
            'post_sigma_dex_med': float(post_dex.median()),
            'ratio_med': float(ratio.median()),
            'ratio_p10': float(ratio.quantile(0.1)),
            f'frac_ratio_lt_{VCIRC_PRIOR_INFORMATIVE}': float(
                (ratio < VCIRC_PRIOR_INFORMATIVE).mean()
            ),
        }
        for p, name in (
            ('cosi', 'sigma_cosi_med'),
            ('theta_int', 'sigma_theta_med'),
            ('g_plus', 'sigma_gplus_med'),
            ('g_cross', 'sigma_gcross_med'),
        ):
            if f'post.{p}.std' in sub:
                row[name] = float(sub[f'post.{p}.std'].median())
        rows.append(row)
    return pd.DataFrame(rows)


def _vcirc_ratio(ok: pd.DataFrame) -> np.ndarray:
    post_dex = ok['post.vel.vcirc.std'] / (ok['post.vel.vcirc.mean'] * np.log(10))
    return (post_dex / ok['pop.prior_vcirc_sigma_dex']).values.astype(float)


def _plot_vcirc_ratio(ok: pd.DataFrame, xcol: str, xlabel: str, logx: bool) -> str:
    import matplotlib.pyplot as plt

    if xcol not in ok:
        raise KeyError(f'{xcol} missing')
    x = ok[xcol].values.astype(float)
    ratio = _vcirc_ratio(ok)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bins = _cosi_bin_index(ok['truth.cosi'])
    for i, (lo, hi) in enumerate(COSI_BINS):
        m = bins == i
        if m.any():
            ax.scatter(
                x[m],
                ratio[m],
                s=12,
                color=C_SEQ[i],
                label=f'true cos i [{lo:.1f}, {min(hi, 1.0):.1f})',
            )
    bx, by, be = _binned_median(x, ratio)
    ax.errorbar(
        bx, by, yerr=be, fmt='o-', color=C_INK, ms=5, lw=1.4, label='binned median'
    )
    ax.axhline(1.0, color=C_ORANGE, lw=1.2, ls='--', label='prior only (ratio 1)')
    ax.axhline(VCIRC_PRIOR_INFORMATIVE, color=C_TEXT2, lw=0.8, ls=':')
    if logx:
        ax.set_xscale('log')
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel('sigma(vcirc) posterior / prior', fontsize=8)
    ax.set_ylim(0, 1.15)
    _style_axes(ax)
    ax.legend(fontsize=7, frameon=False, loc='lower left')
    fig.tight_layout()
    return _img_from_fig(
        fig,
        f'vcirc posterior/prior width ratio vs {xlabel}',
        'Ratio of the posterior width of vcirc (dex) to the TFR-implied fit prior '
        'width (pop.prior_vcirc_sigma_dex). 1 = the kinematics add nothing to vcirc; '
        f'the dotted line at {VCIRC_PRIOR_INFORMATIVE} marks the informative threshold '
        'used in the table. Error bars: 1.253 MAD / sqrt(n) per quantile bin.',
    )


def _plot_gplus_vs_cosi_sigma(okd: pd.DataFrame) -> str:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    from scipy.stats import spearmanr

    need = ('post.g_plus.std', 'post.cosi.std', 'line_snr')
    if any(c not in okd for c in need):
        raise KeyError('g+ / cos i sigma or line_snr missing')
    sx = okd['post.cosi.std'].values.astype(float)
    sy = okd['post.g_plus.std'].values.astype(float)
    snr = okd['line_snr'].values.astype(float)
    rho = spearmanr(sx, sy).statistic
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    sc = axes[0].scatter(
        sx, sy, c=np.log10(snr), s=12, cmap='Blues', vmin=np.log10(snr).min()
    )
    cb = fig.colorbar(sc, ax=axes[0])
    cb.set_label('log10 line SNR (per roll)', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    axes[0].set_xlabel('sigma cos i (posterior)', fontsize=8)
    axes[0].set_ylabel('sigma g+ (posterior)', fontsize=8)
    axes[0].set_title(f'Spearman rho {rho:.2f}', fontsize=8)
    _style_axes(axes[0])
    bins = _cosi_bin_index(okd['truth.cosi'])
    for i, (lo, hi) in enumerate(COSI_BINS):
        m = bins == i
        if m.sum() >= 6:
            bx, by, be = _binned_median(snr[m], sx[m], n_bins=4)
            axes[1].errorbar(
                bx,
                by,
                yerr=be,
                fmt='o-',
                color=C_SEQ[i],
                ms=4,
                lw=1.2,
                label=f'true cos i [{lo:.1f}, {min(hi, 1.0):.1f})',
            )
    axes[1].set_xscale('log')
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].xaxis.set_major_formatter(ScalarFormatter())
    axes[1].set_xlabel('line SNR (per roll)', fontsize=8)
    axes[1].set_ylabel('sigma cos i, binned median', fontsize=8)
    _style_axes(axes[1])
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return _img_from_fig(
        fig,
        'Shear noise tracks the inclination noise',
        'Left: per-fit posterior sigma of the disk-frame g+ against the posterior '
        'sigma of cos i, coloured by per-roll line SNR. Right: sigma cos i vs line SNR '
        'per true cos i bin. With vcirc pinned by the prior, cos i comes from '
        'v sin i / vcirc, so its width (and the shear width) is set by the prior '
        'width and the line SNR.',
    )


def _section_vcirc_prior(ok: pd.DataFrame, spec: Optional[dict]) -> str:
    if len(ok) == 0:
        return _na('no succeeded fits yet')
    table = vcirc_prior_table(with_derived_shear(ok))
    out = []
    pop = (spec or {}).get('population', {}) or {}
    tfr = (pop.get('paint', {}) or {}).get('tfr', {}) or {}
    mass_err = (pop.get('priors', {}) or {}).get('logm_obs_scatter_dex')
    if tfr and mass_err is not None:
        s_tfr = float(tfr['scatter_dex'])
        s_mass = float(mass_err) / float(tfr['slope'])
        out.append(
            f'<p>Fit prior width {np.hypot(s_tfr, s_mass):.3f} dex = TFR scatter '
            f'{s_tfr:.3f} dex (+) mass error {float(mass_err):.2f} dex / slope '
            f'{float(tfr["slope"]):.2f} = {s_mass:.3f} dex, the same for every galaxy.</p>'
        )
    allrow = table.iloc[0]
    out.append(
        f'<p>Median vcirc posterior/prior width ratio {allrow["ratio_med"]:.2f}; '
        f'{100 * allrow[f"frac_ratio_lt_{VCIRC_PRIOR_INFORMATIVE}"]:.0f}% of fits '
        f'below {VCIRC_PRIOR_INFORMATIVE}. "line SNR" here is the per-roll (single '
        'pass) matched-filter line SNR from the manifest, not the coadded total used '
        'for selection.</p>'
    )
    out.append(_table(table))
    out.append(_guard(_plot_vcirc_ratio, ok, 'line_snr', 'line SNR (per roll)', True))
    out.append(_guard(_plot_vcirc_ratio, ok, 'truth.cosi', 'true cos i', False))
    out.append(_guard(_plot_gplus_vs_cosi_sigma, with_derived_shear(ok)))
    return ''.join(out)


def _section_notes(run_dir: Path) -> str:
    p = run_dir / 'diagnostics' / 'notes.md'
    if not p.exists():
        return _na('no notes (write diagnostics/notes.md to show them here)')
    return f'<pre>{_esc(p.read_text())}</pre>'


def _section_glossary() -> str:
    items = ''.join(
        f'<dt id="g-{key}">{_esc(term)}</dt><dd>{_esc(text)}</dd>'
        for key, (term, text) in GLOSSARY.items()
    )
    return (
        '<p class="meta">Wording follows docs/sampler_failure_ledger.md. Table headers '
        'link here.</p>'
        f'<dl>{items}</dl>'
    )


# ==============================================================================
# Entry point
# ==============================================================================

_SECTIONS = (
    ('progress', 'Progress'),
    ('speed', 'Speed by galaxy property'),
    ('failures', 'Failures and escalations'),
    ('gate', 'Convergence gate summary'),
    ('flags', 'Flags'),
    ('science', 'Early science'),
    ('headline', 'Headline science (in progress)'),
    ('vcirc_prior', 'vcirc constraint beyond the TF prior'),
    ('plots', 'Plots'),
    ('notes', 'Notes'),
    ('glossary', 'Glossary'),
)


def build_dashboard(run_dir: Path, open_browser: bool = False) -> Path:
    """
    Write ``<run_dir>/diagnostics/dashboard.html`` and return its path.

    Parameters
    ----------
    run_dir : Path
        Ensemble run directory (``manifest.parquet`` required).
    open_browser : bool
        Open the written page in the default browser.

    Raises
    ------
    FileNotFoundError
        Run directory or manifest missing.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f'run directory {run_dir} does not exist')
    manifest_path = run_dir / 'manifest.parquet'
    if not manifest_path.exists():
        raise FileNotFoundError(f'{manifest_path} missing')
    manifest = pd.read_parquet(manifest_path)
    results = _read_results(run_dir)
    status = _status_frame(run_dir)
    ok = _succeeded(results, manifest)
    chain = chain_derived_stats(run_dir, ok) if len(ok) else None
    spec, spec_path = _read_spec(run_dir)
    run_name = (spec or {}).get('run', {}).get('name') or run_dir.name
    repo_url = _repo_url()

    # per-job commit: from the per-fit column when present, else the job log
    commits: Dict[str, Tuple[str, str]] = {}
    if results is not None and 'git_commit' in results.columns and len(status):
        joined = status.merge(
            results[['fit_id', 'git_commit']], on='fit_id', how='left'
        )
        for job, g in joined.groupby('job'):
            shas = sorted({str(s) for s in g['git_commit'].dropna()})
            if shas:
                commits[str(job)] = (', '.join(shas), '')
    for job, sha in _job_log_commits(run_dir).items():
        commits.setdefault(job, (sha, '(inferred from job log)'))
    exp_commit = _expansion_commit(run_dir)

    header_commits = []
    if exp_commit:
        header_commits.append(
            'expansion ' + _commit_html(exp_commit, repo_url=repo_url)
        )
    fit_shas = sorted({v[0] for v in commits.values()})
    if fit_shas:
        header_commits.append(
            'fits ' + ', '.join(_commit_html(s, repo_url=repo_url) for s in fit_shas)
        )
    commit_line = '; '.join(header_commits) if header_commits else NOT_AVAILABLE

    nav = ''.join(f'<a href="#{key}">{_esc(title)}</a>' for key, title in _SECTIONS)
    bodies = {
        'progress': _guard(
            _section_progress, manifest, status, results, commits, repo_url
        ),
        'speed': _guard(_section_speed, ok, _workers_per_node(spec)),
        'failures': _guard(
            _section_failures, run_dir, results, status, commits, repo_url
        ),
        'gate': _guard(_section_gate, ok),
        'flags': _guard(_section_flags, results, status, commits, repo_url),
        'science': _guard(_section_science, ok, chain),
        'headline': _guard(_section_headline, ok),
        'vcirc_prior': _guard(_section_vcirc_prior, ok, spec),
        'plots': _guard(_section_plots, run_dir, ok, chain),
        'notes': _guard(_section_notes, run_dir),
        'glossary': _guard(_section_glossary),
    }
    parts = [
        '<!doctype html><html><head><meta charset="utf-8">',
        f'<title>{_esc(run_name)}</title><style>{_CSS}</style></head><body>',
        f'<nav>{nav}</nav><main>',
        f'<h1>{_esc(run_name)}</h1>',
        f'<p class="meta">run dir {_esc(run_dir)}<br>spec {_esc(spec_path)}<br>'
        f'commits {commit_line}<br>generated {time.strftime("%Y-%m-%d %H:%M:%S")}</p>',
    ]
    for key, title in _SECTIONS:
        parts.append(
            f'<details open id="{key}"><summary><h2>{_esc(title)}</h2></summary>'
            f'{bodies[key]}</details>'
        )
    parts.append('</main></body></html>')
    out_dir = run_dir / 'diagnostics'
    out_dir.mkdir(exist_ok=True)
    out = out_dir / 'dashboard.html'
    out.write_text('\n'.join(parts))
    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return out
