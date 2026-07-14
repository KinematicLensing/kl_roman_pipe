"""
Per-fit worker: claim -> mock -> fit -> per-fit result file.

The worker loop is backend-agnostic (no SLURM imports or assumptions): it
walks the manifest, claims unowned fits via the atomic-mkdir ledger, runs
each claimed fit, and writes one single-row parquet per fit
(``results/<fit_id>.parquet``, written to a tmp name then atomically
renamed, THEN the done marker). Fit errors are caught per fit: a failed
marker + a status='failed' summary row are written and the loop continues --
failures are loud, explicit, and re-runnable, and never abort the worker.

The summary row records recovered posteriors, the MAP, and the inclusive
quality-column set (rhat/ess/divergences/acceptance/precond diagnostics/
wallclock) -- gate policy is applied downstream, post hoc.
"""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from kl_pipe.ensemble import ledger
from kl_pipe.ensemble.expander import truth_from_row
from kl_pipe.ensemble.mocks import build_fit_inputs
from kl_pipe.ensemble.spec import EnsembleSpec, ObservingConfig

# sampler-seed stream tag: distinct from the expander's galaxy/noise streams
_SAMPLER_STREAM = 3


def _sampler_seed(noise_seed: int) -> int:
    ss = np.random.SeedSequence([int(noise_seed), _SAMPLER_STREAM])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(f'.tmp.{os.getpid()}.{path.name}')
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _atomic_savez(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    # tmp name must keep the .npz suffix LAST: np.savez appends '.npz' to any
    # filename that lacks it, which would break the atomic rename
    tmp = path.with_name(f'.tmp.{os.getpid()}.{path.name}')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


# PA basins are the known multimodality: every fit's MAP multi-start gets
# this many extra starts with theta_int stratified across its prior range
# (random prior draws can all land in the wrong basin, whose shape-shear-
# compensated mode traps the sampler)
_N_PA_STRATIFIED_STARTS = 4

# a fit whose chains come back broken (stuck wrong mode) gets ONE retry with
# a fresh sampler seed; the retry outcome is recorded, never hidden
_CATASTROPHIC_RHAT = 1.1
_CATASTROPHIC_DIV_RATE = 0.9
_MAX_ATTEMPTS = 2


def _pa_stratified_starts(priors, seed: int, n_pa: int = _N_PA_STRATIFIED_STARTS):
    """Prior-draw start points with theta_int overridden by a PA grid."""
    import jax

    names = list(priors.sampled_names)
    if 'theta_int' not in names:
        return None
    lo, hi = priors.get_prior('theta_int').bounds
    if lo is None or hi is None:
        return None
    # explicit copy: np.asarray on a jax array returns a read-only view
    starts = np.array(priors.sample(jax.random.PRNGKey(seed + 2), n_pa))
    centers = lo + (np.arange(n_pa) + 0.5) * (hi - lo) / n_pa
    starts[:, names.index('theta_int')] = centers
    return starts


def _is_catastrophic(summary: dict) -> bool:
    return (
        summary['max_rhat'] > _CATASTROPHIC_RHAT
        or summary['divergence_rate'] > _CATASTROPHIC_DIV_RATE
    )


def run_single_fit(
    row: Dict,
    spec: EnsembleSpec,
    config: ObservingConfig,
    run_dir: Path,
) -> dict:
    """Run one fit and return its summary row (also persisted by the caller).

    A catastrophically unconverged result (broken chains: max_rhat > 1.1 or
    divergence rate > 0.9, i.e. a sampler stuck in a spurious mode -- not
    mere low quality) is retried once with a fresh sampler seed; the attempt
    count is recorded in the summary. Raises on any error -- the caller
    decides how to record the failure.
    """
    fit_id = str(row['fit_id'])
    truth = truth_from_row(row)
    noise_seed = int(row['noise_seed'])

    summary = None
    for attempt in range(_MAX_ATTEMPTS):
        sampler_seed = _sampler_seed(noise_seed) + 1000 * attempt
        summary = _run_fit_attempt(
            row, spec, config, run_dir, truth, noise_seed, sampler_seed
        )
        summary['n_attempts'] = attempt + 1
        if not _is_catastrophic(summary):
            break
        print(
            f'[fit {fit_id}] attempt {attempt + 1} catastrophic '
            f"(max_rhat={summary['max_rhat']:.2f}, "
            f"div={summary['divergence_rate']:.0%}) -- "
            + (
                'retrying with fresh seed'
                if attempt + 1 < _MAX_ATTEMPTS
                else 'giving up; recorded as-is'
            ),
            flush=True,
        )
    return summary


def _run_fit_attempt(
    row: Dict,
    spec: EnsembleSpec,
    config: ObservingConfig,
    run_dir: Path,
    truth: Dict[str, float],
    noise_seed: int,
    sampler_seed: int,
) -> dict:
    from kl_pipe.sampling import InferenceTask
    from kl_pipe.sampling.configs import NumpyroSamplerConfig
    from kl_pipe.sampling.numpyro import NumpyroSampler

    fit_id = str(row['fit_id'])
    t_start = time.time()

    inputs = build_fit_inputs(
        truth,
        noise_seed,
        spec,
        config,
        broadband_snr=float(row['broadband_snr']),
        grism_snr=float(row['grism_snr']),
    )
    task = InferenceTask.from_obs(
        inputs.source,
        inputs.priors,
        image_obs=inputs.image_obs,
        grism_obs=inputs.grism_obs,
    )

    sampler_config = NumpyroSamplerConfig(
        n_samples=spec.n_samples,
        n_warmup=spec.n_warmup,
        n_chains=spec.n_chains,
        target_accept_prob=spec.target_accept,
        precondition=spec.precondition,
        n_map_starts=spec.n_map_starts,
        seed=sampler_seed,
    )

    preconditioner = None
    if spec.precondition == 'laplace':
        # build explicitly (rather than letting the sampler build it
        # internally) so the MAP point and precond diagnostics reach the
        # summary row; PA-stratified extra starts guarantee the position-
        # angle basins are all visited
        preconditioner = task.laplace_preconditioner(
            n_starts=sampler_config.n_map_starts,
            seed=sampler_seed,
            hessian_method=sampler_config.hessian_method,
            extra_starts=_pa_stratified_starts(inputs.priors, sampler_seed),
        )
    t_precond = time.time()

    sampler = NumpyroSampler(task, sampler_config, preconditioner=preconditioner)
    result = sampler.run()
    t_end = time.time()

    sampled_names = list(inputs.priors.sampled_names)
    summary = _summary_row(
        row,
        spec,
        result,
        preconditioner,
        sampled_names,
        wallclock_s=t_end - t_start,
        precond_s=t_precond - t_start,
    )
    summary['sampler_seed'] = sampler_seed

    if bool(row['save_chains']):
        _save_chains(run_dir, fit_id, result, sampled_names)
    if bool(row['save_mocks']):
        _save_mocks(run_dir, fit_id, inputs, preconditioner, sampled_names)

    return summary


def _summary_row(
    row: Dict,
    spec: EnsembleSpec,
    result,
    preconditioner,
    sampled_names,
    wallclock_s: float,
    precond_s: float,
) -> dict:
    diag = result.diagnostics
    r_hat = diag.get('r_hat', {})
    ess = diag.get('ess', {})
    if not r_hat or not ess:
        raise RuntimeError(
            "sampler diagnostics missing r_hat/ess -- cannot write an honest "
            "summary row"
        )

    stats = result.get_summary()
    map_theta = (
        np.asarray(preconditioner.map_point) if preconditioner is not None else None
    )

    out: dict = {
        'fit_id': str(row['fit_id']),
        'status': 'succeeded',
        'error_message': '',
        # quality columns (inclusive -- gate policy applied post hoc)
        'max_rhat': float(max(r_hat.values())),
        'min_ess': float(min(ess.values())),
        'ess_g1': float(ess.get('g1', np.nan)),
        'ess_g2': float(ess.get('g2', np.nan)),
        'n_divergences': int(diag.get('n_divergences', -1)),
        'divergence_rate': float(diag.get('divergence_rate', np.nan)),
        'mean_accept_prob': float(diag.get('mean_accept_prob', np.nan)),
        'num_steps_total': float(np.sum(diag.get('num_steps', np.nan))),
        'converged': bool(result.converged),
        'chain_method': str(diag.get('chain_method', '')),
        'n_map_starts_converged': (
            int(preconditioner.n_starts_converged) if preconditioner is not None else -1
        ),
        'precond_condition_number': (
            float(preconditioner.condition_number)
            if preconditioner is not None
            else np.nan
        ),
        'fit_wallclock_s': float(wallclock_s),
        'precond_wallclock_s': float(precond_s),
    }

    for i, name in enumerate(sampled_names):
        s = stats[name]
        out[f'post.{name}.mean'] = float(s['mean'])
        out[f'post.{name}.std'] = float(s['std'])
        out[f'post.{name}.median'] = float(s['quantiles'][0.5])
        if map_theta is not None:
            out[f'map.{name}'] = float(map_theta[i])
            out[f'map_minus_postmean_over_sigma.{name}'] = float(
                (map_theta[i] - s['mean']) / s['std']
            )
    return out


def failed_summary_row(row: Dict, message: str) -> dict:
    return {
        'fit_id': str(row['fit_id']),
        'status': 'failed',
        'error_message': message,
    }


def _save_chains(run_dir: Path, fit_id: str, result, sampled_names) -> None:
    path = Path(run_dir) / 'chains' / f'{fit_id}.npz'
    arrays = {
        'samples': np.asarray(result.samples),
        'log_prob': np.asarray(result.log_prob),
        'param_names': np.array(sampled_names),
    }
    if result.chains is not None:
        arrays['chains'] = np.asarray(result.chains)
    _atomic_savez(path, arrays)


def _save_mocks(
    run_dir: Path, fit_id: str, inputs, preconditioner, sampled_names
) -> None:
    """Persist the mock datavectors (+ MAP renders when available)."""
    arrays: Dict[str, np.ndarray] = {}
    map_pars: Optional[Dict[str, float]] = None
    if preconditioner is not None:
        map_pars = dict(inputs.priors.fixed_values)
        for i, name in enumerate(sampled_names):
            map_pars[name] = float(np.asarray(preconditioner.map_point)[i])

    for band, obs in inputs.image_obs.items():
        arrays[f'image.{band}.data'] = np.asarray(obs.data)
        arrays[f'image.{band}.variance'] = np.asarray(obs.variance)
        arrays[f'image.{band}.truth_render'] = np.asarray(
            inputs.source.render_broadband(inputs.truth, obs, band)
        )
        if map_pars is not None:
            arrays[f'image.{band}.map_render'] = np.asarray(
                inputs.source.render_broadband(map_pars, obs, band)
            )
    for key, obs in inputs.grism_obs.items():
        arrays[f'grism.{key}.data'] = np.asarray(obs.data)
        arrays[f'grism.{key}.variance'] = np.asarray(obs.variance)
        arrays[f'grism.{key}.truth_render'] = np.asarray(
            inputs.source.render_grism(inputs.truth, obs)
        )
        if map_pars is not None:
            arrays[f'grism.{key}.map_render'] = np.asarray(
                inputs.source.render_grism(map_pars, obs)
            )

    path = Path(run_dir) / 'mocks' / f'{fit_id}.npz'
    _atomic_savez(path, arrays)


def worker_loop(
    run_dir: Path,
    worker_label: str = 'worker0',
    max_fits: Optional[int] = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> Dict[str, int]:
    """
    Claim-and-fit loop over the manifest. Backend-agnostic.

    Walks manifest rows in order; for each row: skip if a done marker exists,
    skip if claimed by someone else, otherwise claim -> fit -> write result ->
    done marker. A fit error writes a failed marker + failed summary row and
    the loop continues.

    Parameters
    ----------
    run_dir : Path
        The expanded run directory.
    worker_label : str
        Identifies this worker in log lines.
    max_fits : int, optional
        Stop after this many completed fit attempts (dev/testing knob).
    shard_index, shard_count : int
        Static-dispatch partition: this worker walks manifest rows
        ``[shard_index::shard_count]`` (strided). Dynamic dispatch leaves the
        default (every worker walks all rows; the claim ledger arbitrates).

    Returns
    -------
    dict
        {'succeeded': int, 'failed': int, 'skipped': int}
    """
    from kl_pipe.ensemble.expander import load_run

    run_dir = Path(run_dir)
    spec, config, manifest = load_run(run_dir)
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError(f"invalid shard {shard_index}/{shard_count}")
    if shard_count > 1:
        manifest = manifest.iloc[shard_index::shard_count]
    counts = {'succeeded': 0, 'failed': 0, 'skipped': 0}

    for _, row in manifest.iterrows():
        fit_id = str(row['fit_id'])
        if max_fits is not None and (
            counts['succeeded'] + counts['failed'] >= max_fits
        ):
            break
        if ledger.is_done(run_dir, fit_id):
            counts['skipped'] += 1
            continue
        if not ledger.try_claim(run_dir, fit_id):
            counts['skipped'] += 1
            continue

        print(f'[{worker_label}] fit {fit_id} starting', flush=True)
        try:
            summary = run_single_fit(row, spec, config, run_dir)
        except Exception:
            message = traceback.format_exc()
            print(f'[{worker_label}] fit {fit_id} FAILED\n{message}', flush=True)
            ledger.mark_failed(run_dir, fit_id, message)
            _atomic_write_parquet(
                pd.DataFrame([failed_summary_row(dict(row), message)]),
                run_dir / 'results' / f'{fit_id}.parquet',
            )
            counts['failed'] += 1
            continue

        _atomic_write_parquet(
            pd.DataFrame([summary]),
            run_dir / 'results' / f'{fit_id}.parquet',
        )
        ledger.mark_done(run_dir, fit_id)
        counts['succeeded'] += 1
        print(
            f"[{worker_label}] fit {fit_id} done "
            f"({summary['fit_wallclock_s']:.1f} s, "
            f"max_rhat={summary['max_rhat']:.3f})",
            flush=True,
        )

    return counts
