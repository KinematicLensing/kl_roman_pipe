"""Vista demo v2: batched NUTS over 16 production fits x 4 chains = 64 lanes.

v2 (after demo v1's 22.8% divergences): sampling runs in UNCONSTRAINED
coordinates via the production UnconstrainingTransform -- the exact
component that cured the constrained-space prior-wall divergences in the
solo pipeline (44% -> 0.33%). Transform kinds are static (identical across
galaxies, asserted); the per-galaxy bounds ride in the dynamic pytree.

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
KLPIPE_MAX_DOUBLINGS (7), KLPIPE_INIT (map_laplace | map_window | truth),
KLPIPE_TARGET_ACCEPT (0.9), KLPIPE_MAP_STARTS (4), KLPIPE_MAP_MAXITER
(2000), KLPIPE_INIT_STEP (0.01).

Usage (worktree root, GPU node):
  python experiments/sweverett/batched_nuts/batched_nuts_demo.py
"""

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from proto_v2_shared_program import (
    build_inputs,
    pin_halfwidth,
    precompute_continuum_kernel,
)
from proto_v3_batched_posterior import (
    assert_prior_structure,
    extract_dyn,
    make_shared_posterior,
)

from kl_pipe.ensemble.expander import load_run
from kl_pipe.sampling.transforms import UnconstrainingTransform

RUN_DIR = Path(os.environ.get('KLPIPE_RUN_DIR', 'runs/cosmos25_ab_bb32gr32'))
OUT_DIR = Path(os.environ.get('KLPIPE_DEMO_OUT', 'runs/batched_nuts_demo'))
N_FITS = int(os.environ.get('KLPIPE_N_FITS', '16'))
N_CHAINS = int(os.environ.get('KLPIPE_N_CHAINS', '4'))
N_WARMUP = int(os.environ.get('KLPIPE_NUTS_WARMUP', '400'))
N_SAMPLES = int(os.environ.get('KLPIPE_NUTS_SAMPLES', '300'))
MAX_DOUBLINGS = int(os.environ.get('KLPIPE_MAX_DOUBLINGS', '7'))
SEED = int(os.environ.get('KLPIPE_DEMO_SEED', '20260801'))
# init modes: map_laplace = production recipe (host MAP through the shared
# program, fixed Laplace metric, step-size-only warmup); map_window = MAP
# init + dense window adaptation; truth = truth-adjacent init + window
# adaptation (the v1-v3 demo behavior; NOT production-achievable timing)
INIT_MODE = os.environ.get('KLPIPE_INIT', 'map_laplace')
TARGET_ACCEPT = float(os.environ.get('KLPIPE_TARGET_ACCEPT', '0.9'))
MAP_STARTS = int(os.environ.get('KLPIPE_MAP_STARTS', '4'))
MAP_MAXITER = int(os.environ.get('KLPIPE_MAP_MAXITER', '2000'))
INIT_STEP = float(os.environ.get('KLPIPE_INIT_STEP', '0.01'))


def main():
    import blackjax

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec, config, manifest = load_run(RUN_DIR)
    # retry passes rerun an explicit subset (comma-separated fit_ids);
    # default = first N_FITS manifest rows
    fit_id_filter = os.environ.get('KLPIPE_FIT_IDS', '')
    if fit_id_filter:
        wanted = [f.strip() for f in fit_id_filter.split(',') if f.strip()]
        missing = set(wanted) - set(manifest['fit_id'])
        if missing:
            raise ValueError(f'KLPIPE_FIT_IDS not in manifest: {sorted(missing)}')
        rows = [manifest[manifest['fit_id'] == f].iloc[0] for f in wanted]
    else:
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

    # continuum kernel: precompute once and pin, with galaxy-independence
    # asserted across every fit and roll in the bank
    import kl_pipe.dispersion as kdisp

    tmpl_inp, tmpl_img, tmpl_grm = built[0]
    kern0, m_lo0 = precompute_continuum_kernel(built)
    kdisp.continuum_trace_kernel = lambda *a, **kw: (kern0, m_lo0)

    # prior structure (types, attr sets, parent wiring) must match the
    # template exactly -- only numeric prior attributes travel per galaxy
    assert_prior_structure(built, names)

    shared = make_shared_posterior(
        tmpl_inp.source, names, tmpl_img, tmpl_grm, tmpl_inp.priors
    )

    # production unconstrained reparam: same bijection kinds everywhere
    # (spec-level prior structure), per-galaxy bounds as data
    transforms = [
        UnconstrainingTransform.from_priors(inp.priors) for inp, _, _ in built
    ]
    tmpl_tf = transforms[0]
    for t in transforms:
        assert np.array_equal(t.kinds, tmpl_tf.kinds), 'transform kinds differ'
    print(f'unconstrained kinds: {tmpl_tf.kind_names}')

    def logdensity_eta(eta, dyn):
        tf = dataclasses.replace(tmpl_tf, lows=dyn['tf_lows'], highs=dyn['tf_highs'])
        theta = tf.inverse(eta)
        return shared(theta, dyn) + tf.log_jacobian(eta)

    # lanes: fit-major ordering, chains contiguous per fit
    dyns = []
    for (inp, img, grm), tf in zip(built, transforms):
        dyn = extract_dyn(img, grm, dict(inp.priors.fixed_values), inp.priors, names)
        dyn['tf_lows'] = jnp.asarray(tf.lows)
        dyn['tf_highs'] = jnp.asarray(tf.highs)
        dyns.append(dyn)
    # every fit's dynamic pytree must mirror the template's structure and
    # leaf shapes; name the offending fit and leaf instead of letting
    # jnp.stack fail cryptically (PSFData shape metadata lives in aux, so a
    # kernel-shape mismatch surfaces here as a treedef difference)
    tmpl_paths = {
        jax.tree_util.keystr(p): jnp.shape(leaf)
        for p, leaf in jax.tree_util.tree_flatten_with_path(dyns[0])[0]
    }
    tmpl_treedef = jax.tree_util.tree_structure(dyns[0])
    for i, dyn in enumerate(dyns):
        treedef = jax.tree_util.tree_structure(dyn)
        if treedef != tmpl_treedef:
            raise AssertionError(
                f'fit {fit_ids[i]}: dyn pytree structure differs from template '
                f'(e.g. PSF kernel shape or missing leaf)'
            )
        for p, leaf in jax.tree_util.tree_flatten_with_path(dyn)[0]:
            key = jax.tree_util.keystr(p)
            if jnp.shape(leaf) != tmpl_paths[key]:
                raise AssertionError(
                    f'fit {fit_ids[i]}: leaf {key} shape {jnp.shape(leaf)} '
                    f'!= template {tmpl_paths[key]}'
                )

    lane_dyns = [dyns[i] for i in range(len(built)) for _ in range(N_CHAINS)]
    dyn_batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *lane_dyns)

    # per-fit init points (and, for map modes, the Laplace metric)
    print(f'init mode: {INIT_MODE}, target accept {TARGET_ACCEPT}', flush=True)
    key = jax.random.PRNGKey(SEED)
    lane_inits = []
    t_map = 0.0
    map_eta = map_inv_eta = None
    map_conds, map_nconv = [], []
    if INIT_MODE in ('map_laplace', 'map_window'):
        from map_init import laplace_for_fit

        # host MAP through the ONE shared compiled program: per-fit
        # value_and_grad with that galaxy's dynamic pytree as data
        vg_shared = jax.jit(jax.value_and_grad(shared))
        t0 = time.time()
        map_eta, map_inv_eta = [], []
        for i, (inp, _, _) in enumerate(built):
            vg = lambda th, _d=dyns[i]: vg_shared(th, _d)
            theta_map, inv_th, n_conv, cond = laplace_for_fit(
                vg,
                inp.priors,
                n_starts=MAP_STARTS,
                maxiter=MAP_MAXITER,
                seed=SEED + 1000 + i,
            )
            tf = transforms[i]
            eta_m, clipped = tf.forward_clipped(theta_map, u_margin=1e-6)
            if clipped.any():
                clip_names = [names[j] for j in np.flatnonzero(np.asarray(clipped))]
                print(
                    f'WARNING: fit {fit_ids[i]}: MAP clipped into prior '
                    f'support for {clip_names}'
                )
            map_eta.append(np.asarray(eta_m))
            map_inv_eta.append(np.asarray(tf.transform_inverse_mass(inv_th, eta_m)))
            map_conds.append(cond)
            map_nconv.append(n_conv)
            print(
                f'  map fit {i} ({fit_ids[i]}): starts converged '
                f'{n_conv}/{MAP_STARTS}, metric condition {cond:.2e}',
                flush=True,
            )
        t_map = time.time() - t0
        print(f'map + laplace x{len(built)}: {t_map:.0f} s', flush=True)

        # production chain jitter: 1 percent of the per-dim posterior scale
        for i in range(len(built)):
            post_scale = np.sqrt(np.diag(map_inv_eta[i]))
            for c in range(N_CHAINS):
                key, sub = jax.random.split(key)
                jit = (
                    0.01
                    * post_scale
                    * np.asarray(jax.random.normal(sub, (len(names),)))
                )
                lane_inits.append(jnp.asarray(map_eta[i] + jit))
    elif INIT_MODE == 'truth':
        # truth + 1 percent-of-prior-std jitter (v1-v3 behavior; timing is
        # NOT production-achievable -- no MAP phase is paid)
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
            tf = transforms[i]
            for c in range(N_CHAINS):
                key, sub = jax.random.split(key)
                th0 = np.asarray(
                    truth + 0.01 * stds * jax.random.normal(sub, truth.shape)
                )
                eta0, clipped = tf.forward_clipped(th0, u_margin=1e-6)
                if clipped.any():
                    clip_names = [names[j] for j in np.flatnonzero(np.asarray(clipped))]
                    print(
                        f'WARNING: fit {fit_ids[i]} chain {c}: init clipped '
                        f'to prior support for {clip_names}'
                    )
                lane_inits.append(jnp.asarray(eta0))
    else:
        raise ValueError(f'unknown KLPIPE_INIT mode {INIT_MODE!r}')
    theta0 = jnp.stack(lane_inits)

    def warmup_one(k, theta_init, dyn):
        logdensity = lambda et: logdensity_eta(et, dyn)
        warm = blackjax.window_adaptation(
            blackjax.nuts,
            logdensity,
            target_acceptance_rate=TARGET_ACCEPT,
            is_mass_matrix_diagonal=False,
            max_num_doublings=MAX_DOUBLINGS,
        )
        (state, params), _ = warm.run(k, theta_init, num_steps=N_WARMUP)
        return (
            state.position,
            params['step_size'],
            params['inverse_mass_matrix'],
        )

    def warmup_da_one(k, theta_init, dyn, inv_mass):
        # production preconditioned recipe: FIXED Laplace metric, only the
        # step size adapts (dual averaging), so warmup is a fraction of the
        # dense window-adaptation cost
        from blackjax.adaptation.step_size import dual_averaging_adaptation
        from blackjax.mcmc import nuts as bj_nuts

        logdensity = lambda et: logdensity_eta(et, dyn)
        kernel = bj_nuts.build_kernel()
        da_init, da_update, da_final = dual_averaging_adaptation(target=TARGET_ACCEPT)
        state0 = bj_nuts.init(theta_init, logdensity)

        def step(carry, kk):
            st, da_st = carry
            ss = jnp.exp(da_st.log_step_size)
            st, info = kernel(kk, st, logdensity, ss, inv_mass, MAX_DOUBLINGS)
            return (st, da_update(da_st, info.acceptance_rate)), None

        keys = jax.random.split(k, N_WARMUP)
        (state, da_st), _ = jax.lax.scan(step, (state0, da_init(INIT_STEP)), keys)
        return state.position, da_final(da_st), inv_mass

    def sample_one(k, position, step_size, inverse_mass_matrix, dyn):
        logdensity = lambda et: logdensity_eta(et, dyn)
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
    if INIT_MODE == 'map_laplace':
        inv_mass_lanes = jnp.stack(
            [
                jnp.asarray(map_inv_eta[i])
                for i in range(len(built))
                for _ in range(N_CHAINS)
            ]
        )
        warm_fn = jax.jit(jax.vmap(warmup_da_one))
        pos_w, step_sizes, inv_mass = warm_fn(
            warm_keys, theta0, dyn_batch, inv_mass_lanes
        )
    else:
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

    # positions are eta; store physical theta for analysis (eta kept too)
    def _inv(eta_lane, lows, highs):
        tf = dataclasses.replace(tmpl_tf, lows=lows, highs=highs)
        return tf.inverse(eta_lane)

    positions_eta = positions
    positions = jax.jit(jax.vmap(_inv))(
        positions_eta, dyn_batch['tf_lows'], dyn_batch['tf_highs']
    )
    jax.block_until_ready(positions)

    extra = {}
    if map_eta is not None:
        extra = {
            'map_eta': np.asarray(map_eta),
            'map_inv_mass_eta': np.asarray(map_inv_eta),
            'map_condition': np.asarray(map_conds),
            'map_n_converged': np.asarray(map_nconv),
        }
    np.savez_compressed(
        OUT_DIR / 'demo_results.npz',
        positions=np.asarray(positions),
        positions_eta=np.asarray(positions_eta),
        acceptance=np.asarray(acc),
        num_integration_steps=np.asarray(n_steps),
        divergent=np.asarray(divergent),
        logdensity=np.asarray(logdens),
        step_sizes=np.asarray(step_sizes),
        theta0=np.asarray(theta0),
        **extra,
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
        'init_mode': INIT_MODE,
        'target_accept': TARGET_ACCEPT,
        'map_starts': MAP_STARTS if map_eta is not None else None,
        'wall_s': {
            'build': t_build,
            'map': t_map,
            'warmup': t_warmup,
            'sampling': t_sample,
        },
        'devices': [str(d) for d in jax.devices()],
        'lane_order': 'fit-major, chains contiguous',
    }
    (OUT_DIR / 'demo_meta.json').write_text(json.dumps(meta, indent=2))
    # honest fit-equivalent includes the MAP phase (zero in truth mode,
    # which is exactly why truth-mode timing is not production-achievable)
    total = t_map + t_warmup + t_sample
    print(
        f'\nTOTAL wall (map + warmup + sampling) {total:.0f} s for '
        f'{len(rows)} fits = {total / len(rows) / 60:.1f} min/fit-equivalent '
        f'(solo baseline 16-18 min/fit)'
    )
    print(f'results in {OUT_DIR}/')


if __name__ == '__main__':
    main()
