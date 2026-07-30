"""Cross-galaxy batching decision gate: vmapped value_and_grad amortization.

Times jit(vmap(value_and_grad(log_posterior))) over B stacked theta draws
on ONE fit's likelihood, B = 1..256. Shared-obs proxy for true cross-galaxy
batching: per-eval cost is render-dominated and theta-dependent, so the
amortization curve measures GPU-overhead recovery without the
data-as-argument refactor. True batching adds per-galaxy data/PSF-kernel
loads (small arrays; second-order correction).

Decision rule (2026-07-29): if per-galaxy ms/eval at B=64 is < ~1/3 of the
B=1 cost, in-kernel batching pays beyond process packing and the
data-as-argument refactor is warranted; if the curve is flat, drop it and
process packing is the scaling story.

Usage: batch_probe.py [row_idx] [max_B]
"""

import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from kl_pipe.ensemble.expander import load_run, truth_from_row
from kl_pipe.ensemble.spec import ObservationConfig
from kl_pipe.ensemble.mocks import build_fit_inputs
from kl_pipe.sampling import InferenceTask

RUN_DIR = Path(os.environ.get('KLPIPE_RUN_DIR', 'runs/cosmos25_shear_dev'))
BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def build_task(row, spec) -> InferenceTask:
    config = ObservationConfig.from_yaml(
        Path('configs/observation') / f'{spec.observed_config}.yaml'
    )
    inputs = build_fit_inputs(
        truth_from_row(row),
        int(row['noise_seed']),
        spec,
        config,
        broadband_snr=float(row['broadband_snr']),
        line_snr=float(row['line_snr']),
        row=row,
    )
    task = InferenceTask.from_obs(
        inputs.source,
        inputs.priors,
        image_obs=inputs.image_obs,
        grism_obs=inputs.grism_obs,
    )
    return task, inputs


def main() -> None:
    spec, config, manifest = load_run(RUN_DIR)
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    max_b = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    row = manifest.iloc[idx]
    print(f'fit_id={row["fit_id"]}  z={row["truth.z"]}  cosi={row["truth.cosi"]}')
    print(f'devices: {jax.devices()}', flush=True)

    t0 = time.time()
    task, inputs = build_task(row, spec)
    print(f'build: {time.time() - t0:.1f} s', flush=True)

    n_dim = len(inputs.priors.sampled_names)
    vg = jax.value_and_grad(task.log_posterior)

    results = []
    t_ms_b1 = None
    for b in [x for x in BATCH_SIZES if x <= max_b]:
        thetas = jnp.asarray(
            inputs.priors.sample(jax.random.PRNGKey(1), n_samples=b)
        ).reshape(b, n_dim)
        fn = jax.jit(jax.vmap(vg)) if b > 1 else jax.jit(vg)
        arg = thetas if b > 1 else thetas[0]
        try:
            t0 = time.time()
            out = fn(arg)
            jax.block_until_ready(out)
            t_compile = time.time() - t0
            n_rep = max(3, min(20, 256 // b))
            t0 = time.time()
            for _ in range(n_rep):
                out = fn(arg)
            jax.block_until_ready(out)
            dt_ms = (time.time() - t0) / n_rep * 1000.0
        except Exception as e:
            print(f'B={b:4d}  FAILED: {type(e).__name__}: {e}', flush=True)
            break
        per_gal = dt_ms / b
        if t_ms_b1 is None:
            t_ms_b1 = per_gal
        results.append((b, t_compile, dt_ms, per_gal, t_ms_b1 / per_gal))
        print(
            f'B={b:4d}  compile {t_compile:6.1f} s  batch {dt_ms:8.1f} ms  '
            f'per-galaxy {per_gal:7.2f} ms  speedup x{t_ms_b1 / per_gal:5.2f}',
            flush=True,
        )

    if len(results) > 1:
        b, _, _, per_gal, speedup = results[-1]
        print(
            f'\nverdict input: per-galaxy eval at B={b} is x{speedup:.2f} '
            f'cheaper than B=1 (gate: >x3 at B=64 -> invest in '
            f'data-as-argument batching)',
            flush=True,
        )


if __name__ == '__main__':
    main()
