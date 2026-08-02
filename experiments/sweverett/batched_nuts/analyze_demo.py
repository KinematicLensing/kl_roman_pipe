"""Analyze a batched-NUTS demo run against the solo bb32gr32 baselines.

Reports, per gate:
  parity   per-fit, per-param: |mean_b - mean_s| / MC-error (z-score using
           solo min_ess as the conservative ESS for all params), and the
           posterior-width ratio std_b/std_s. Human-reviewed, no hard fail.
  quality  per-fit R-hat (4 chains) and ESS from the batched draws,
           divergence rate.
  cost     wall per fit-equivalent; per-iteration integration-step
           distribution across lanes (straggler overhead = mean over iters
           of [max_lanes / mean_lanes]).

Usage:
  python analyze_demo.py <demo_out_dir> <solo_results_dir>
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def rhat_and_ess(x):
    # x: (n_chains, n_draws); production's own estimators (numpyro summary
    # split_gelman_rubin + n_eff) so gate results are directly comparable
    # to the ensemble worker's -- arviz bulk-ESS differs enough to flip
    # hair's-breadth fits
    from numpyro.diagnostics import summary

    s = summary({'x': x})['x']
    return float(s['r_hat']), float(s['n_eff'])


def main():
    out_dir = Path(sys.argv[1])
    solo_dir = Path(sys.argv[2])
    d = np.load(out_dir / 'demo_results.npz')
    meta = json.loads((out_dir / 'demo_meta.json').read_text())
    names = meta['param_names']
    n_fits, n_chains = meta['n_fits'], meta['n_chains']
    pos = d['positions']  # (lanes, draws, n_params), fit-major
    steps = d['num_integration_steps']  # (lanes, draws)
    div = d['divergent']

    wall = meta['wall_s']
    # map phase absent from pre-map-init runs (and zero in truth mode)
    t_map = wall.get('map', 0.0)
    total = t_map + wall['warmup'] + wall['sampling']
    print(
        f"wall: build {wall['build']:.0f} s, map {t_map:.0f} s, "
        f"warmup {wall['warmup']:.0f} s, sampling {wall['sampling']:.0f} s "
        f"(init mode {meta.get('init_mode', 'truth')})"
    )
    print(
        f"fit-equivalent: {total / n_fits / 60:.2f} min/fit "
        f"({3600 / (total / n_fits):.1f} fits/node-hr on this device, "
        f"excluding build)"
    )

    # straggler accounting: lockstep pays max-over-lanes each iteration
    mx = steps.max(axis=0).astype(float)
    mean_ = steps.mean(axis=0)
    print(
        f"integration steps/iter: lane-mean {steps.mean():.0f}, "
        f"lockstep-max mean {mx.mean():.0f}, straggler overhead "
        f"x{(mx / np.maximum(mean_, 1)).mean():.2f}"
    )
    print(f"divergence rate: {float(div.mean()):.3%}")

    rows = []
    for i, fid in enumerate(meta['fit_ids']):
        lanes = slice(i * n_chains, (i + 1) * n_chains)
        solo_path = solo_dir / f'{fid}.parquet'
        solo = pd.read_parquet(solo_path).iloc[0] if solo_path.exists() else None
        chains = pos[lanes]  # (n_chains, draws, P)
        for j, name in enumerate(names):
            x = chains[:, :, j]
            mb, sb = float(x.mean()), float(x.std(ddof=1))
            r, e = rhat_and_ess(x)
            row = {
                'fit_id': fid,
                'param': name,
                'mean_b': mb,
                'std_b': sb,
                'rhat_b': r,
                'ess_b': e,
            }
            if solo is not None and f'post.{name}.mean' in solo.index:
                ms = float(solo[f'post.{name}.mean'])
                ss = float(solo[f'post.{name}.std'])
                ess_s = float(solo.get('min_ess', np.nan))
                mc = np.sqrt(ss**2 / max(ess_s, 1.0) + sb**2 / max(e, 1.0))
                row.update(
                    mean_s=ms,
                    std_s=ss,
                    z=(mb - ms) / mc if mc > 0 else np.nan,
                    width_ratio=sb / ss if ss > 0 else np.nan,
                )
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'parity.csv', index=False)

    have = df.dropna(subset=['z']) if 'z' in df else pd.DataFrame()
    print(
        f"\nbatched quality: max R-hat {df['rhat_b'].max():.3f}, "
        f"min ESS {df['ess_b'].min():.0f}"
    )

    # per-fit gate table: production escalation gate (rhat_max 1.05,
    # ess_min 50, worst param per fit)
    RHAT_MAX, ESS_MIN = 1.05, 50.0
    gate = (
        df.groupby('fit_id')
        .agg(max_rhat=('rhat_b', 'max'), min_ess=('ess_b', 'min'))
        .reset_index()
    )
    gate['pass'] = (gate['max_rhat'] <= RHAT_MAX) & (gate['min_ess'] >= ESS_MIN)
    print(f'\nper-fit gates (rhat<={RHAT_MAX}, ess>={ESS_MIN:.0f}):')
    for _, g in gate.iterrows():
        print(
            f"  {g['fit_id']}  max_rhat {g['max_rhat']:.3f}  "
            f"min_ess {g['min_ess']:6.1f}  {'PASS' if g['pass'] else 'FAIL'}"
        )
    print(f"gate summary: {int(gate['pass'].sum())}/{len(gate)} pass")
    if len(have):
        print(
            f"parity vs solo ({have['fit_id'].nunique()} fits matched): "
            f"|z| median {have['z'].abs().median():.2f}, "
            f"max {have['z'].abs().max():.2f}; width ratio "
            f"median {have['width_ratio'].median():.2f}, "
            f"range {have['width_ratio'].min():.2f}-{have['width_ratio'].max():.2f}"
        )
        worst = have.loc[have['z'].abs().idxmax()]
        print(
            f"worst |z|: {worst['fit_id']} {worst['param']} "
            f"(b {worst['mean_b']:.4g} vs s {worst['mean_s']:.4g})"
        )
    else:
        print('no solo results matched -- parity table has batched stats only')
    print(f"full table: {out_dir / 'parity.csv'}")


if __name__ == '__main__':
    main()
