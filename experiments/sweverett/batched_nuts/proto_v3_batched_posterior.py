"""Experiment 3: shared POSTERIOR (priors as data) + grad parity + batched-NUTS smoke.

Extends experiment 2 from likelihood to the full sampling target:

  - Priors as data: per-galaxy prior parameters (all numeric attributes of
    each Prior's __dict__, INCLUDING derived caches) travel in the dynamic
    pytree; inside the trace each prior is a shallow copy of the template
    with those attributes overridden. Assumes (and asserts) identical prior
    types + attribute sets across galaxies -- true by spec construction.
  - Mirrors task._log_posterior_jittable exactly (jnp.where -inf guard).

Hypotheses:
  H4  shared posterior == task.log_posterior per galaxy at ulp level.
  H5  value_and_grad through the shared program: one compile, grads match
      each galaxy's own task grad at ulp-ish level (<1e-12 rel).
  H6  vmapped blackjax NUTS (window adaptation, BPD pattern) runs the
      machinery across lanes with per-lane step sizes, finite states, no
      recompile per lane. Smoke test only (tiny step counts, CPU);
      posterior parity gates are the vista demo's job.

Usage (worktree root):
  python experiments/sweverett/batched_nuts/proto_v3_batched_posterior.py [n_fits]
"""

import copy
import dataclasses
import logging
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from kl_pipe.ensemble.expander import load_run
from kl_pipe.likelihood import _log_likelihood_total_source
from kl_pipe.sampling import InferenceTask

from proto_v2_shared_program import (
    build_inputs,
    pin_halfwidth,
    precompute_continuum_kernel,
)

RUN_DIR = Path(os.environ.get('KLPIPE_RUN_DIR', 'runs/cosmos25_ab_bb32gr32'))


def numeric_attrs(prior):
    out = {}
    for k, v in vars(prior).items():
        if isinstance(v, (bool, str)) or v is None:
            continue
        if isinstance(v, (int, float)) or hasattr(v, 'dtype'):
            out[k] = jnp.asarray(v, dtype=jnp.float64)
    return out


def assert_prior_structure(built, names):
    """Require identical prior STRUCTURE across the bank.

    The shared posterior takes prior types, attribute sets, non-numeric
    attribute values (e.g. a conditional prior's ``parent`` name), and the
    PriorDict conditional-parent index from the template; only numeric
    attributes travel per galaxy. Anything structural that differs would be
    silently overridden by the template, so check all of it here.
    """
    tmpl_priors = built[0][0].priors
    tmpl_index = dict(tmpl_priors._parent_index)
    for i, (inp, _, _) in enumerate(built):
        assert (
            dict(inp.priors._parent_index) == tmpl_index
        ), f'fit {i}: conditional parent index differs from template'
        for n in names:
            pa, pb = tmpl_priors.get_prior(n), inp.priors.get_prior(n)
            assert type(pa) is type(pb), f'fit {i} {n}: {type(pa)} vs {type(pb)}'
            assert set(vars(pa)) == set(vars(pb)), f'fit {i} {n}: attr sets differ'
            for attr, va in vars(pa).items():
                if isinstance(va, (bool, str)) or va is None:
                    vb = vars(pb)[attr]
                    assert va == vb, f'fit {i} {n}.{attr}: {va!r} vs {vb!r}'


def extract_dyn(image_obs, grism_obs, fixed_pars, priors, sampled_names):
    # broadband PSF rides as the traced k-space kernel (same grid across the
    # bank; per-galaxy values); psf_data's padded shape is z-tier-dependent
    # and unreachable behind the k-space path, so require that path loudly
    for k, o in image_obs.items():
        if o.kspace_psf_fft is None:
            raise AssertionError(
                f'image obs {k}: kspace_psf_fft is None; the shared program '
                'requires the k-space PSF path'
            )
    dyn = {
        'image': {
            k: {
                'data': o.data,
                'variance': o.variance,
                'kspace_psf_fft': o.kspace_psf_fft,
            }
            for k, o in image_obs.items()
        },
        'grism': {
            k: {
                'data': o.data,
                'variance': o.variance,
                'psf_data': o.psf_data,
                'lambda_ref': jnp.asarray(o.grism_pars.lambda_ref),
            }
            for k, o in grism_obs.items()
        },
        'fixed': {k: jnp.asarray(v) for k, v in fixed_pars.items()},
        'prior': {
            name: numeric_attrs(priors.get_prior(name)) for name in sampled_names
        },
    }
    return dyn


def make_shared_posterior(source, sampled_names, tmpl_image, tmpl_grism, tmpl_priors):
    tmpl_prior_objs = {n: tmpl_priors.get_prior(n) for n in sampled_names}
    # mirror PriorDict.log_prior's conditional-parent resolution (static
    # structure -- identical across galaxies by spec construction)
    parent_index = dict(tmpl_priors._parent_index)

    def log_prior(theta, prior_dyn):
        total = 0.0
        for i, name in enumerate(sampled_names):
            p = copy.copy(tmpl_prior_objs[name])
            for attr, val in prior_dyn[name].items():
                object.__setattr__(p, attr, val)
            if name in parent_index:
                total = total + p.log_prob_given(theta[i], theta[parent_index[name]])
            else:
                total = total + p.log_prob(theta[i])
        return total

    def fn(theta, dyn):
        image_obs = {
            # psf_data=None: unreachable behind the asserted k-space path;
            # keeps the template's z-tier fallback kernel out of the trace
            k: dataclasses.replace(
                tmpl_image[k],
                data=dyn['image'][k]['data'],
                variance=dyn['image'][k]['variance'],
                psf_data=None,
                kspace_psf_fft=dyn['image'][k]['kspace_psf_fft'],
            )
            for k in tmpl_image
        }
        grism_obs = {
            k: dataclasses.replace(
                tmpl_grism[k],
                data=dyn['grism'][k]['data'],
                variance=dyn['grism'][k]['variance'],
                psf_data=dyn['grism'][k]['psf_data'],
                grism_pars=dataclasses.replace(
                    tmpl_grism[k].grism_pars,
                    lambda_ref=dyn['grism'][k]['lambda_ref'],
                ),
            )
            for k in tmpl_grism
        }
        ll = _log_likelihood_total_source(
            theta,
            source=source,
            image_obs=image_obs,
            grism_obs=grism_obs,
            velocity_obs=None,
            sampled_names=sampled_names,
            fixed_pars=dyn['fixed'],
            grism_groups=None,
            grism_group_operators=None,
        )
        lp = log_prior(theta, dyn['prior'])
        return jnp.where(jnp.isfinite(lp), lp + ll, -jnp.inf)

    return fn


def main():
    n_fits = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    n_nuts = int(os.environ.get('KLPIPE_NUTS_LANES', '4'))
    spec, config, manifest = load_run(RUN_DIR)
    rows = [manifest.iloc[i] for i in range(min(n_fits, len(manifest)))]

    t0 = time.time()
    built = [build_inputs(r, spec, config) for r in rows]
    print(f'build x{len(rows)}: {time.time() - t0:.0f} s', flush=True)
    names = tuple(built[0][0].priors.sampled_names)

    hw_max = max(
        o.render_config.line_window_halfwidth for _, _, g in built for o in g.values()
    )
    built = [(inp, img, pin_halfwidth(grm, hw_max)) for inp, img, grm in built]

    # prior structure must be identical across galaxies
    assert_prior_structure(built, names)

    # continuum kernel precompute (asserted galaxy-independent) + monkeypatch
    import kl_pipe.dispersion as kdisp

    tmpl_inp, tmpl_img, tmpl_grm = built[0]
    kern0, m_lo0 = precompute_continuum_kernel(built)
    real_kernel = kdisp.continuum_trace_kernel
    kdisp.continuum_trace_kernel = lambda *a, **kw: (kern0, m_lo0)

    # ---- reference tasks (closure path) --------------------------------------
    tasks, thetas, dyns = [], [], []
    for i, (inp, img, grm) in enumerate(built):
        kdisp.continuum_trace_kernel = real_kernel
        task = InferenceTask.from_obs(
            inp.source, inp.priors, image_obs=img, grism_obs=grm
        )
        kdisp.continuum_trace_kernel = lambda *a, **kw: (kern0, m_lo0)
        tasks.append(task)
        thetas.append(
            jnp.asarray(
                inp.priors.sample(jax.random.PRNGKey(300 + i), n_samples=1)
            ).ravel()
        )
        dyns.append(
            extract_dyn(img, grm, dict(inp.priors.fixed_values), inp.priors, names)
        )

    shared = make_shared_posterior(
        tmpl_inp.source, names, tmpl_img, tmpl_grm, tmpl_inp.priors
    )

    n_compiles = {'count': 0}

    class _Counter(logging.Handler):
        def emit(self, record):
            if 'Compiling' in record.getMessage():
                n_compiles['count'] += 1

    counter = _Counter()
    jax.config.update('jax_log_compiles', True)
    logging.getLogger('jax').addHandler(counter)

    # H4: posterior values; H5: value_and_grad
    vg_shared = jax.jit(jax.value_and_grad(shared))
    rel_v, rel_g = [], []
    marks = []
    for i, task in enumerate(tasks):
        v_ref, g_ref = task.get_log_posterior_and_grad_fn()(thetas[i])
        v_s, g_s = vg_shared(thetas[i], dyns[i])
        jax.block_until_ready((v_s, g_s))
        rel_v.append(abs(float(v_s) - float(v_ref)) / abs(float(v_ref)))
        gr = np.asarray(g_ref)
        gs = np.asarray(g_s)
        denom = np.maximum(np.abs(gr), 1e-30)
        rel_g.append(float(np.max(np.abs(gs - gr) / denom)))
        marks.append(n_compiles['count'])
    print(f'H4 posterior rel diff: max {max(rel_v):.2e}')
    print(
        f'H5 grad rel diff: max {max(rel_g):.2e}; shared-program compiles at '
        f'each fit: {marks} (first includes the one compile)'
    )

    jax.config.update('jax_log_compiles', False)
    logging.getLogger('jax').removeHandler(counter)

    # ---- H6: vmapped blackjax NUTS smoke (tiny counts) -----------------------
    import blackjax

    lanes = min(n_nuts, len(built))
    dyn_batch = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs), *[dyns[i] for i in range(lanes)]
    )
    theta0 = jnp.stack([thetas[i] for i in range(lanes)])

    n_warm = int(os.environ.get('KLPIPE_NUTS_WARMUP', '20'))
    n_samp = int(os.environ.get('KLPIPE_NUTS_SAMPLES', '10'))

    def one_fit(key, theta_init, dyn):
        logdensity = lambda th: shared(th, dyn)
        warm = blackjax.window_adaptation(
            blackjax.nuts,
            logdensity,
            target_acceptance_rate=0.8,
            max_num_doublings=5,
        )
        key_w, key_s = jax.random.split(key)
        (state, params), _ = warm.run(key_w, theta_init, num_steps=n_warm)
        kernel = blackjax.nuts(logdensity, **params).step

        def step(carry, k):
            st, _ = carry, None
            st, info = kernel(k, st)
            return st, (st.position, info.acceptance_rate)

        keys = jax.random.split(key_s, n_samp)
        final, (positions, acc) = jax.lax.scan(step, state, keys)
        return positions, acc, params['step_size']

    keys = jax.random.split(jax.random.PRNGKey(0), lanes)
    t0 = time.time()
    run = jax.jit(jax.vmap(one_fit))
    pos, acc, step_sizes = run(keys, theta0, dyn_batch)
    jax.block_until_ready(pos)
    t_total = time.time() - t0
    kdisp.continuum_trace_kernel = real_kernel

    finite = bool(np.all(np.isfinite(np.asarray(pos))))
    ss = np.asarray(step_sizes)
    print(
        f'H6 batched NUTS smoke: lanes {lanes}, warmup {n_warm} + samples '
        f'{n_samp}, wall {t_total:.0f} s (incl. one compile)'
    )
    print(
        f'   finite positions: {finite}; per-lane adapted step sizes: '
        f'{np.array2string(ss, precision=4)}; mean accept '
        f'{np.asarray(acc).mean(axis=1).round(2)}'
    )
    per_lane_distinct = len(set(np.round(ss, 10))) > 1
    print(f'   per-lane adaptation distinct: {per_lane_distinct}')

    print('\nsummary:')
    print(f'  H4 {"PASS" if max(rel_v) < 5e-14 else "FAIL"}')
    print(f'  H5 {"PASS" if max(rel_g) < 1e-11 else "FAIL"}')
    print(f'  H6 {"PASS" if finite else "FAIL"} (smoke only)')


if __name__ == '__main__':
    main()
