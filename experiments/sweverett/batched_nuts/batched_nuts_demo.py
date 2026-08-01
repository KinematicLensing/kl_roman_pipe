"""Vista demo: batched NUTS over 16 production fits x 4 chains = 64 lanes.

One compiled program (experiments 1-3 chain of evidence, see NOTES.md):
shared posterior with per-fit data/PSF/lambda_ref/fixed/prior parameters as
traced inputs; blackjax NUTS with per-lane dense window adaptation, vmapped
over all lanes; tree depth capped (straggler bound, BPD precedent).

Gates this run feeds (analyze_demo.py):
  parity  per-galaxy posterior mean/std vs the solo bb32gr32 results
          (chains pooled per fit), within MC error
  cost    wall per fit-equivalent vs the 16-18 min solo baseline
  shape   per-iteration integration-step distribution across lanes
          (straggler cost of lockstep vmap)

Outputs: <out_dir>/demo_results.npz + demo_meta.json.

Env knobs: KLPIPE_RUN_DIR (default runs/cosmos25_ab_bb32gr32),
KLPIPE_DEMO_OUT (default runs/batched_nuts_demo), KLPIPE_N_FITS (16),
KLPIPE_N_CHAINS (4), KLPIPE_NUTS_WARMUP (400), KLPIPE_NUTS_SAMPLES (300),
KLPIPE_MAX_DOUBLINGS (7).

Usage (worktree root, GPU node):
  python experiments/sweverett/batched_nuts/batched_nuts_demo.py
"""

import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from proto_v2_shared_program import build_inputs, pin_halfwidth
from proto_v3_batched_posterior import extract_dyn, make_shared_posterior

from kl_pipe.ensemble.expander import load_run

RUN_DIR = Path(os.environ.get('KLPIPE_RUN_DIR', 'runs/cosmos25_ab_bb32gr32'))
OUT_DIR = Path(os.environ.get('KLPIPE_DEMO_OUT', 'runs/batched_nuts_demo'))
N_FITS = int(os.environ.get('KLPIPE_N_FITS', '16'))
N_CHAINS = int(os.environ.get('KLPIPE_N_CHAINS', '4'))
N_WARMUP = int(os.environ.get('KLPIPE_NUTS_WARMUP', '400'))
N_SAMPLES = int(os.environ.get('KLPIPE_NUTS_SAMPLES', '300'))
MAX_DOUBLINGS = int(os.environ.get('KLPIPE_MAX_DOUBLINGS', '7'))
SEED = int(os.environ.get('KLPIPE_DEMO_SEED', '20260801'))


def main():
    import blackjax

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec, config, manifest = load_run(RUN_DIR)
    rows = [manifest.iloc[i] for i in range(min(N_FITS, len(manifest)))]
    fit_ids = [r['fit_id'] for r in rows]
    print(
        f'{len(rows)} fits x {N_CHAINS} chains = {len(rows) * N_CHAINS} lanes; '
        f'warmup {N_WARMUP} + samples {N_SAMPLES}, max_doublings {MAX_DOUBLINGS}'
    )
    print(f'devices: {jax.devices()}', flush=True)

    t0 = time.time()
    built = [build_inputs(r, spec, config) for r in rows]
    t_build = time.time() - t0
    print(f'build x{len(rows)}: {t_build:.0f} s', flush=True)
    names = tuple(built[0][0].priors.sampled_names)

    hw_max = max(
        o.render_config.line_window_halfwidth for _, _, g in built for o in g.values()
    )
    built = [(inp, img, pin_halfwidth(grm, hw_max)) for inp, img, grm in built]

    # continuum kernel: galaxy-independent under flat throughput (asserted
    # in experiment 2); precompute once and pin
    import kl_pipe.dispersion as kdisp

    tmpl_inp, tmpl_img, tmpl_grm = built[0]
    roll0 = next(iter(tmpl_grm.values()))
    gp0 = roll0.grism_pars
    half_nm = (gp0.image_pars.Ncol + 2) * gp0.dispersion
    kern0, m_lo0 = kdisp.continuum_trace_kernel(
        gp0,
        roll0.cube_pars.lambda_grid,
        roll0.oversample,
        integration_window=(gp0.lambda_ref - half_nm, gp0.lambda_ref + half_nm),
    )
    gp_last = next(iter(built[-1][2].values())).grism_pars
    kern_l, m_lo_l = kdisp.continuum_trace_kernel(
        gp_last,
        next(iter(built[-1][2].values())).cube_pars.lambda_grid,
        next(iter(built[-1][2].values())).oversample,
        integration_window=(gp_last.lambda_ref - half_nm, gp_last.lambda_ref + half_nm),
    )
    assert m_lo0 == m_lo_l and np.array_equal(kern0, kern_l)
    kdisp.continuum_trace_kernel = lambda *a, **kw: (kern0, m_lo0)

    shared = make_shared_posterior(
        tmpl_inp.source, names, tmpl_img, tmpl_grm, tmpl_inp.priors
    )

    # lanes: fit-major ordering, chains contiguous per fit
    dyns = [
        extract_dyn(img, grm, dict(inp.priors.fixed_values), inp.priors, names)
        for inp, img, grm in built
    ]
    lane_dyns = [dyns[i] for i in range(len(built)) for _ in range(N_CHAINS)]
    dyn_batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *lane_dyns)

    # init: truth + 1 percent-of-prior-std jitter per lane (production uses
    # MAP + 1 percent jitter; truth-adjacent is the demo stand-in -- both
    # must converge to the same posterior, which is what the parity gate
    # checks)
    key = jax.random.PRNGKey(SEED)
    lane_inits = []
    for i, (inp, _, _) in enumerate(built):
        truth = jnp.asarray(
            [float(rows[i][f'truth.{n}']) for n in names], dtype=jnp.float64
        )
        stds = jnp.asarray(
            np.std(
                np.asarray(inp.priors.sample(jax.random.PRNGKey(1), n_samples=256)),
                axis=0,
            ),
            dtype=jnp.float64,
        )
        for c in range(N_CHAINS):
            key, sub = jax.random.split(key)
            lane_inits.append(truth + 0.01 * stds * jax.random.normal(sub, truth.shape))
    theta0 = jnp.stack(lane_inits)

    def warmup_one(k, theta_init, dyn):
        logdensity = lambda th: shared(th, dyn)
        warm = blackjax.window_adaptation(
            blackjax.nuts,
            logdensity,
            target_acceptance_rate=0.8,
            is_mass_matrix_diagonal=False,
            max_num_doublings=MAX_DOUBLINGS,
        )
        (state, params), _ = warm.run(k, theta_init, num_steps=N_WARMUP)
        return (
            state.position,
            params['step_size'],
            params['inverse_mass_matrix'],
        )

    def sample_one(k, position, step_size, inverse_mass_matrix, dyn):
        logdensity = lambda th: shared(th, dyn)
        kernel = blackjax.nuts(
            logdensity,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            max_num_doublings=MAX_DOUBLINGS,
        )
        state = kernel.init(position)

        def step(st, kk):
            st, info = kernel.step(kk, st)
            return st, (
                st.position,
                info.acceptance_rate,
                info.num_integration_steps,
                info.is_divergent,
                st.logdensity,
            )

        keys = jax.random.split(k, N_SAMPLES)
        _, out = jax.lax.scan(step, state, keys)
        return out

    n_lanes = len(lane_inits)
    key, kw, ks = jax.random.split(key, 3)
    warm_keys = jax.random.split(kw, n_lanes)
    samp_keys = jax.random.split(ks, n_lanes)

    print('warmup (compile + run)...', flush=True)
    t0 = time.time()
    warm_fn = jax.jit(jax.vmap(warmup_one))
    pos_w, step_sizes, inv_mass = warm_fn(warm_keys, theta0, dyn_batch)
    jax.block_until_ready(pos_w)
    t_warmup = time.time() - t0
    print(f'warmup wall {t_warmup:.0f} s', flush=True)

    print('sampling (compile + run)...', flush=True)
    t0 = time.time()
    samp_fn = jax.jit(jax.vmap(sample_one))
    positions, acc, n_steps, divergent, logdens = samp_fn(
        samp_keys, pos_w, step_sizes, inv_mass, dyn_batch
    )
    jax.block_until_ready(positions)
    t_sample = time.time() - t0
    print(f'sampling wall {t_sample:.0f} s', flush=True)

    np.savez_compressed(
        OUT_DIR / 'demo_results.npz',
        positions=np.asarray(positions),
        acceptance=np.asarray(acc),
        num_integration_steps=np.asarray(n_steps),
        divergent=np.asarray(divergent),
        logdensity=np.asarray(logdens),
        step_sizes=np.asarray(step_sizes),
        theta0=np.asarray(theta0),
    )
    meta = {
        'fit_ids': list(fit_ids),
        'param_names': list(names),
        'n_fits': len(rows),
        'n_chains': N_CHAINS,
        'n_warmup': N_WARMUP,
        'n_samples': N_SAMPLES,
        'max_num_doublings': MAX_DOUBLINGS,
        'seed': SEED,
        'run_dir': str(RUN_DIR),
        'line_window_halfwidth_pin': int(hw_max),
        'wall_s': {
            'build': t_build,
            'warmup': t_warmup,
            'sampling': t_sample,
        },
        'devices': [str(d) for d in jax.devices()],
        'lane_order': 'fit-major, chains contiguous',
    }
    (OUT_DIR / 'demo_meta.json').write_text(json.dumps(meta, indent=2))
    total = t_warmup + t_sample
    print(
        f'\nTOTAL sampling wall {total:.0f} s for {len(rows)} fits '
        f'= {total / len(rows) / 60:.1f} min/fit-equivalent '
        f'(solo baseline 16-18 min/fit)'
    )
    print(f'results in {OUT_DIR}/')


if __name__ == '__main__':
    main()
