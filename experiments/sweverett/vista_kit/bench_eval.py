"""Per-eval likelihood cost: cosmos25_shear_dev fit 0, roman vs gaussian PSF.

Times jitted log_posterior value+grad (steady state, post-compile), reports
compile time and PSF kernel / render grid sizes. Run inside klpipe env.
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve()))

from kl_pipe.ensemble.expander import load_run, truth_from_row
from kl_pipe.ensemble.spec import ObservationConfig
from kl_pipe.ensemble.mocks import build_fit_inputs
from kl_pipe.sampling import InferenceTask

RUN_DIR = Path('runs/cosmos25_shear_dev')


def bench(config_id: str, row, spec) -> None:
    print(f'\n=== observation config: {config_id} ===', flush=True)
    config = ObservationConfig.from_yaml(
        Path('configs/observation') / f'{config_id}.yaml'
    )

    t0 = time.time()
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
    t_build = time.time() - t0
    print(f'build_fit_inputs + task: {t_build:.1f} s', flush=True)

    # kernel / stamp shapes
    for label, obs_entry in [('image', inputs.image_obs), ('grism', inputs.grism_obs)]:
        obs_list = obs_entry if isinstance(obs_entry, (list, tuple)) else [obs_entry]
        for i, obs in enumerate(obs_list):
            desc = []
            for attr in ('data', 'psf_data'):
                v = getattr(obs, attr, None)
                if v is not None:
                    arr = getattr(v, 'kernel', v)
                    if hasattr(arr, 'shape'):
                        desc.append(f'{attr}.shape={tuple(arr.shape)}')
            print(f'  {label}[{i}]: ' + ', '.join(desc), flush=True)

    n_dim = len(inputs.priors.sampled_names)
    print(f'n sampled params: {n_dim}', flush=True)

    theta0 = jnp.asarray(
        inputs.priors.sample(jax.random.PRNGKey(0), n_samples=1)
    ).reshape(-1)

    vg = jax.jit(jax.value_and_grad(task.log_posterior))

    t0 = time.time()
    v, g = vg(theta0)
    jax.block_until_ready((v, g))
    t_compile = time.time() - t0
    print(f'value_and_grad compile+first eval: {t_compile:.1f} s', flush=True)
    print(f'log_posterior at draw: {float(v):.3f}', flush=True)

    n_rep = 10
    t0 = time.time()
    for _ in range(n_rep):
        v, g = vg(theta0)
    jax.block_until_ready((v, g))
    dt = (time.time() - t0) / n_rep
    print(f'steady-state value_and_grad: {dt * 1000:.0f} ms/eval', flush=True)

    # value-only (MAP fd paths and NUTS both use grad, but record anyway)
    f = jax.jit(task.log_posterior)
    v = f(theta0)
    jax.block_until_ready(v)
    t0 = time.time()
    for _ in range(n_rep):
        v = f(theta0)
    jax.block_until_ready(v)
    dtv = (time.time() - t0) / n_rep
    print(f'steady-state value only:     {dtv * 1000:.0f} ms/eval', flush=True)


def main() -> None:
    spec, config, manifest = load_run(RUN_DIR)
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    row = manifest.iloc[idx]
    print(f'fit_id={row["fit_id"]}  z={row["truth.z"]}')
    bench(spec.observed_config, row, spec)
    if len(sys.argv) <= 2:
        bench('hlwas_medium', row, spec)


if __name__ == '__main__':
    main()
