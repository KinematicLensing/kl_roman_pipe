"""Difficulty-stratified packing counterfactual for the vista_demo_v3 batched-nuts run.

Reads runs/vista_demo_v3/demo_results.npz + demo_meta.json (64 lanes, fit-major,
chains contiguous) and runs/vista_demo_v1/fresh_manifest.parquet (galaxy truths),
builds a per-fit difficulty table, checks whether cap-hit fraction is predictable
from galaxy properties or an early-iteration probe, and computes leapfrog-step
counterfactuals for repacking schemes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RUN_DIR = Path("runs/vista_demo_v3")
MANIFEST_PATH = Path("runs/vista_demo_v1/fresh_manifest.parquet")
N_CHAINS = 4
N_FITS = 16
CAP_STEPS = 127  # 2**max_num_doublings - 1
EARLY_WINDOW = 50  # first N sampling iterations used as an early-online probe


def load_results():
    d = np.load(RUN_DIR / "demo_results.npz", allow_pickle=True)
    meta = json.loads((RUN_DIR / "demo_meta.json").read_text())
    return d, meta


def fit_lane_slices(n_fits: int, n_chains: int):
    """Fit-major, chains-contiguous lane order -> slice(4*i, 4*i+4) per fit."""
    return [slice(i * n_chains, (i + 1) * n_chains) for i in range(n_fits)]


def per_fit_difficulty_table(steps, divergent, fit_ids, manifest):
    """cap-hit fraction, mean steps, divergence rate, min-ess proxy per fit."""
    slices = fit_lane_slices(N_FITS, N_CHAINS)
    rows = []
    for i, (fid, sl) in enumerate(zip(fit_ids, slices)):
        s = steps[sl]  # (4, 600)
        dv = divergent[sl]
        rows.append(
            {
                "fit_id": fid,
                "cap_hit_frac": float((s == CAP_STEPS).mean()),
                "mean_steps": float(s.mean()),
                "median_steps": float(np.median(s)),
                "divergence_rate": float(dv.mean()),
                "early_cap_hit_frac": float((s[:, :EARLY_WINDOW] == CAP_STEPS).mean()),
                "late_cap_hit_frac": float((s[:, EARLY_WINDOW:] == CAP_STEPS).mean()),
            }
        )
    df = pd.DataFrame(rows)
    m = manifest.set_index("fit_id")
    join_cols = [
        "truth.cosi",
        "truth.z",
        "truth.g1",
        "truth.g2",
        "truth.theta_int",
        "truth.vel.vcirc",
        "truth.vel.rscale",
        "truth.F129.rscale",
        "broadband_snr_F129",
        "broadband_snr_F158",
        "line_snr",
        "pop.snr_bb_f129",
        "pop.snr_line_total",
        "galaxy_id",
        "ring_member",
    ]
    df = df.join(m[join_cols], on="fit_id")
    return df


def ess_proxy(positions_eta, fit_ids):
    """min-ess proxy: per-fit min over params of arviz ess on the reparam trace."""
    import arviz as az

    slices = fit_lane_slices(N_FITS, N_CHAINS)
    out = {}
    for fid, sl in zip(fit_ids, slices):
        chunk = positions_eta[sl]  # (4, 600, 24)
        idata = az.convert_to_inference_data(chunk)
        ess = az.ess(idata).to_array().values  # (24,)
        out[fid] = float(np.nanmin(ess))
    return out


def predictability(df):
    """spearman rank correlation of cap-hit fraction vs galaxy properties."""
    props = [
        "truth.cosi",
        "truth.z",
        "truth.g1",
        "truth.g2",
        "truth.theta_int",
        "truth.vel.vcirc",
        "truth.vel.rscale",
        "truth.F129.rscale",
        "broadband_snr_F129",
        "broadband_snr_F158",
        "line_snr",
        "pop.snr_bb_f129",
        "pop.snr_line_total",
    ]
    rows = []
    for p in props:
        rho, pval = stats.spearmanr(df[p], df["cap_hit_frac"])
        rows.append({"property": p, "spearman_rho": rho, "p_value": pval})
    return pd.DataFrame(rows).sort_values(
        "spearman_rho", key=lambda s: s.abs(), ascending=False
    )


def leapfrog_cost(steps_subset):
    """lockstep cost: every lane in the batch runs the per-iter max, so total
    leapfrog-step work = n_lanes * sum over iters of max over those lanes."""
    n_lanes = steps_subset.shape[0]
    return float(n_lanes * steps_subset.max(axis=0).sum())


def packing_counterfactuals(steps, df):
    """(a) actual 64-lane, (b) hard/easy 32-lane split, (c) ideal per-fit,
    each repeated under a cap-63 (2**6-1) truncation approximation."""
    slices = fit_lane_slices(N_FITS, N_CHAINS)

    order = df.sort_values("cap_hit_frac", ascending=False)["fit_id"].tolist()
    fid_to_idx = {fid: i for i, fid in enumerate(df["fit_id"])}
    hard_fits = order[:8]
    easy_fits = order[8:]
    hard_lanes = np.concatenate(
        [np.arange(*slices[fid_to_idx[f]].indices(64)) for f in hard_fits]
    )
    easy_lanes = np.concatenate(
        [np.arange(*slices[fid_to_idx[f]].indices(64)) for f in easy_fits]
    )

    def costs(s):
        actual = leapfrog_cost(s)
        hard_easy = leapfrog_cost(s[hard_lanes]) + leapfrog_cost(s[easy_lanes])
        ideal = sum(leapfrog_cost(s[sl]) for sl in slices)
        return actual, hard_easy, ideal

    actual, hard_easy, ideal = costs(steps)

    cap6_steps = np.minimum(steps, 63)
    actual_c6, hard_easy_c6, ideal_c6 = costs(cap6_steps)

    def reclaim(scheme_cost, actual_cost, ideal_cost):
        gap = actual_cost - ideal_cost
        if gap == 0:
            return float("nan")
        return (actual_cost - scheme_cost) / gap

    rows = [
        {
            "cap_regime": "cap-127 (actual)",
            "scheme": "64-lane (actual)",
            "leapfrog_steps": actual,
            "vs_ideal_ratio": actual / ideal,
            "reclaimed_frac": reclaim(actual, actual, ideal),
        },
        {
            "cap_regime": "cap-127 (actual)",
            "scheme": "2x32-lane hard/easy split",
            "leapfrog_steps": hard_easy,
            "vs_ideal_ratio": hard_easy / ideal,
            "reclaimed_frac": reclaim(hard_easy, actual, ideal),
        },
        {
            "cap_regime": "cap-127 (actual)",
            "scheme": "ideal per-fit (4-lane)",
            "leapfrog_steps": ideal,
            "vs_ideal_ratio": 1.0,
            "reclaimed_frac": 1.0,
        },
        {
            "cap_regime": "cap-63 (approx, clipped)",
            "scheme": "64-lane (actual)",
            "leapfrog_steps": actual_c6,
            "vs_ideal_ratio": actual_c6 / ideal_c6,
            "reclaimed_frac": reclaim(actual_c6, actual_c6, ideal_c6),
        },
        {
            "cap_regime": "cap-63 (approx, clipped)",
            "scheme": "2x32-lane hard/easy split",
            "leapfrog_steps": hard_easy_c6,
            "vs_ideal_ratio": hard_easy_c6 / ideal_c6,
            "reclaimed_frac": reclaim(hard_easy_c6, actual_c6, ideal_c6),
        },
        {
            "cap_regime": "cap-63 (approx, clipped)",
            "scheme": "ideal per-fit (4-lane)",
            "leapfrog_steps": ideal_c6,
            "vs_ideal_ratio": 1.0,
            "reclaimed_frac": 1.0,
        },
    ]
    return pd.DataFrame(rows), hard_fits, easy_fits


def main():
    d, meta = load_results()
    fit_ids = meta["fit_ids"]
    manifest = pd.read_parquet(MANIFEST_PATH)

    steps = d["num_integration_steps"]
    divergent = d["divergent"]

    df = per_fit_difficulty_table(steps, divergent, fit_ids, manifest)

    try:
        ess = ess_proxy(d["positions_eta"], fit_ids)
        df["min_ess_proxy"] = df["fit_id"].map(ess)
    except ImportError:
        df["min_ess_proxy"] = np.nan
        print("arviz unavailable, min_ess_proxy left as nan")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("=== per-fit difficulty table ===")
    print(
        df[
            [
                "fit_id",
                "cap_hit_frac",
                "mean_steps",
                "divergence_rate",
                "min_ess_proxy",
                "truth.cosi",
                "truth.z",
                "broadband_snr_F129",
                "line_snr",
                "ring_member",
            ]
        ].to_string(index=False)
    )

    print("\n=== spearman rank correlation: cap_hit_frac vs galaxy properties ===")
    print(predictability(df).to_string(index=False))

    print(
        "\n=== early-window (first %d iters) vs late-window cap-hit fraction ==="
        % EARLY_WINDOW
    )
    rho, pval = stats.spearmanr(df["early_cap_hit_frac"], df["late_cap_hit_frac"])
    print(f"spearman rho = {rho:.3f}, p = {pval:.4f}")
    print(
        df[
            ["fit_id", "early_cap_hit_frac", "late_cap_hit_frac", "cap_hit_frac"]
        ].to_string(index=False)
    )

    print(
        "\n=== ring-pair check: same galaxy_id, same cosi/z/snr/shear, different theta_int + noise_seed ==="
    )
    pair_rows = []
    for gid, grp in df.groupby("galaxy_id"):
        if len(grp) != 2:
            continue
        pair_rows.append(
            {
                "galaxy_id": gid,
                "cap_hit_fracs": grp["cap_hit_frac"].tolist(),
                "abs_within_pair_diff": abs(
                    grp["cap_hit_frac"].iloc[0] - grp["cap_hit_frac"].iloc[1]
                ),
            }
        )
    pair_df = pd.DataFrame(pair_rows)
    print(pair_df.to_string(index=False))
    print(f"mean within-pair |diff| = {pair_df['abs_within_pair_diff'].mean():.3f}")
    print(
        f"between-galaxy std of per-pair mean cap_hit_frac = {df.groupby('galaxy_id')['cap_hit_frac'].mean().std():.3f}"
    )

    print("\n=== simple 1-2 variable low-cosi/low-snr rule ===")
    thr_cosi = df["truth.cosi"].median()
    thr_snr = df["broadband_snr_F129"].median()
    df["low_cosi_or_low_snr"] = (df["truth.cosi"] < thr_cosi) | (
        df["broadband_snr_F129"] < thr_snr
    )
    print(
        df.groupby("low_cosi_or_low_snr")["cap_hit_frac"]
        .agg(["mean", "median", "count"])
        .to_string()
    )

    print("\n=== packing counterfactuals (leapfrog step totals) ===")
    cost_df, hard_fits, easy_fits = packing_counterfactuals(steps, df)
    print(cost_df.to_string(index=False))
    print(f"\nhard-8 fit_ids (by cap_hit_frac desc): {hard_fits}")
    print(f"easy-8 fit_ids: {easy_fits}")

    out_dir = Path(__file__).parent
    df.to_csv(out_dir / "per_fit_difficulty.csv", index=False)
    cost_df.to_csv(out_dir / "packing_counterfactuals.csv", index=False)
    print(
        f"\nwrote {out_dir / 'per_fit_difficulty.csv'} and {out_dir / 'packing_counterfactuals.csv'}"
    )


if __name__ == "__main__":
    main()
