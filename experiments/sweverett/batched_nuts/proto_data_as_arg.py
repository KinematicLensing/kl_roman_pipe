"""Experiment 1: data-as-argument likelihood mechanics on the real production task.

Hypotheses (2026-08-01, cc/batched-nuts):
  H1  A jitted likelihood taking the obs pytrees as ARGUMENTS (instead of the
      current functools.partial closure) returns bit-identical values.
  H2  Two galaxies from the same run spec share ONE compiled program: the obs
      treedefs match and the second call triggers no recompile. Expected
      failure modes this script must name explicitly: unhashable aux
      (e.g. arrays inside cube_pars), or per-galaxy values living in aux
      (lambda_ref, WCS, kernel size) forcing a treedef mismatch.
  H3  vmap over stacked obs leaves reproduces per-galaxy solo evals.

Success: H1 exact, H2 zero recompiles galaxy-to-galaxy, H3 matches to fp64
roundoff. Any failure -> print the offending aux diff, do not paper over.

Usage (from the worktree root):
  python experiments/sweverett/batched_nuts/proto_data_as_arg.py [row_a] [row_b]
"""

import logging
import os
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from kl_pipe.ensemble.expander import load_run, truth_from_row
from kl_pipe.ensemble.mocks import build_fit_inputs
from kl_pipe.ensemble.spec import ObservationConfig
from kl_pipe.likelihood import (
    _log_likelihood_total_source,
    create_jitted_likelihood_from_obs,
)

RUN_DIR = Path(os.environ.get('KLPIPE_RUN_DIR', 'runs/cosmos25_ab_bb32gr32'))


def build_inputs(row, spec, config):
    band_snrs = {b: float(row[f'broadband_snr_{b}']) for b in config.bands}
    inputs = build_fit_inputs(
        truth_from_row(row),
        int(row['noise_seed']),
        spec,
        config,
        band_snrs=band_snrs,
        line_snr=float(row['line_snr']),
        row=row,
    )
    # replicate InferenceTask.from_obs's per-channel rc handling so the obs
    # carry a concrete line_window_halfwidth (from_obs does not retain the
    # rebuilt obs -- only the closure sees them)
    from kl_pipe.sampling.task import _check_source_priors_fit_obs

    image_obs = {
        k: _check_source_priors_fit_obs(
            inputs.source, inputs.priors, o, component_key=k
        )
        for k, o in inputs.image_obs.items()
    }
    grism_obs = {
        k: _check_source_priors_fit_obs(inputs.source, inputs.priors, o)
        for k, o in inputs.grism_obs.items()
    }
    return inputs, image_obs, grism_obs


def tree_structs(inputs):
    return {
        'image_obs': jax.tree_util.tree_structure(inputs[0]),
        'grism_obs': jax.tree_util.tree_structure(inputs[1]),
    }


def diff_aux(name, obs_a, obs_b):
    """Name every aux field that differs between two same-type obs objects."""
    from kl_pipe.observation import GrismObs

    flat_a, aux_a = (
        jax.tree_util.tree_flatten(obs_a)[0],
        jax.tree_util.tree_structure(obs_a),
    )
    # compare via the registered flatten fns directly for field-level names
    if isinstance(obs_a, GrismObs):
        fields = (
            'grism_pars',
            'cube_pars',
            'render_config',
            'fine_image_pars',
            'psf',
            '_rc_was_default',
            'psf_kernel_size',
            'flux_unit',
        )
        from kl_pipe.observation import _grism_obs_flatten as flatten
    else:
        fields = (
            'image_pars',
            'render_config',
            'psf',
            'broadband_key',
            '_rc_was_default',
            'psf_kernel_size',
            'flux_unit',
        )
        from kl_pipe.observation import _image_obs_flatten as flatten
    _, aux_ta = flatten(obs_a)
    _, aux_tb = flatten(obs_b)
    for f, va, vb in zip(fields, aux_ta, aux_tb):
        try:
            same = va == vb
        except Exception as e:  # noqa: BLE001 - report, never hide
            same = f'EQ-RAISED {type(e).__name__}: {e}'
        if same is not True:
            ra_, rb_ = repr(va)[:120], repr(vb)[:120]
            ident = ' [IDENTITY-EQ? same-looking repr]' if ra_ == rb_ else ''
            print(f'    aux diff [{name}].{f} (eq={same}){ident}:')
            print(f'        A: {ra_}')
            print(f'        B: {rb_}')


def leaf_shapes_match(tree_a, tree_b):
    la = jax.tree_util.tree_leaves(tree_a)
    lb = jax.tree_util.tree_leaves(tree_b)
    if len(la) != len(lb):
        return False
    return all(
        getattr(a, 'shape', None) == getattr(b, 'shape', None) for a, b in zip(la, lb)
    )


def main():
    row_a = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    row_b = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    spec, config, manifest = load_run(RUN_DIR)
    ra, rb = manifest.iloc[row_a], manifest.iloc[row_b]
    print(
        f'galaxy A: {ra["fit_id"]}  z={ra["truth.z"]:.3f} cosi={ra["truth.cosi"]:.3f}'
    )
    print(
        f'galaxy B: {rb["fit_id"]}  z={rb["truth.z"]:.3f} cosi={rb["truth.cosi"]:.3f}'
    )
    print(f'devices: {jax.devices()}')

    t0 = time.time()
    ia, img_a, grm_a = build_inputs(ra, spec, config)
    ib, img_b, grm_b = build_inputs(rb, spec, config)
    print(f'build_fit_inputs x2: {time.time() - t0:.1f} s', flush=True)

    names_a = tuple(ia.priors.sampled_names)
    names_b = tuple(ib.priors.sampled_names)
    assert names_a == names_b, f'sampled_names differ: {names_a} vs {names_b}'
    fixed_a = dict(ia.priors.fixed_values)
    fixed_b = dict(ib.priors.fixed_values)
    print(f'n sampled: {len(names_a)}; fixed keys: {sorted(fixed_a)}')
    fixed_diff = {
        k for k in fixed_a if not np.isclose(fixed_a[k], fixed_b[k], rtol=0, atol=0)
    }
    print(f'fixed values differing A vs B: {sorted(fixed_diff)}')

    theta_a = jnp.asarray(ia.priors.sample(jax.random.PRNGKey(7), n_samples=1)).ravel()
    theta_b = jnp.asarray(ib.priors.sample(jax.random.PRNGKey(8), n_samples=1)).ravel()

    # ---- baseline: current closure path, one jit per galaxy -----------------
    base_a = create_jitted_likelihood_from_obs(
        ia.source, names_a, fixed_a, image_obs=img_a, grism_obs=grm_a
    )
    base_b = create_jitted_likelihood_from_obs(
        ib.source, names_b, fixed_b, image_obs=img_b, grism_obs=grm_b
    )
    t0 = time.time()
    va = base_a(theta_a).block_until_ready()
    t_compile_a = time.time() - t0
    t0 = time.time()
    vb = base_b(theta_b).block_until_ready()
    t_compile_b = time.time() - t0
    print(
        f'closure path: compile+eval A {t_compile_a:.1f} s, B {t_compile_b:.1f} s '
        f'(B pays full recompile today)',
        flush=True,
    )

    # ---- treedef comparison -------------------------------------------------
    sa, sb = tree_structs((img_a, grm_a)), tree_structs((img_b, grm_b))
    for k in sa:
        if sa[k] == sb[k]:
            print(f'treedef[{k}]: MATCH')
        else:
            print(f'treedef[{k}]: MISMATCH -- aux diffs follow')
            if k == 'image_obs':
                for band in img_a:
                    diff_aux(band, img_a[band], img_b[band])
            else:
                for roll in grm_a:
                    diff_aux(roll, grm_a[roll], grm_b[roll])
    print(
        'leaf shapes match: '
        f'image={leaf_shapes_match(img_a, img_b)} '
        f'grism={leaf_shapes_match(grm_a, grm_b)}'
    )

    # ---- H1/H2: obs-as-argument path ---------------------------------------
    # source/sampled_names stay closed over (galaxy A's); obs + fixed_pars are
    # arguments. Calling with galaxy B's obs+fixed tests BOTH value equality
    # (H1, vs base_b) and program reuse (H2). grism_groups=None: analytic
    # dispersal groups are singletons, identical math to the per-obs path.
    n_compiles = {'count': 0}

    def _count_compiles(record):
        n_compiles['count'] += 1

    arg_fn = jax.jit(
        partial(
            _log_likelihood_total_source,
            source=ia.source,
            sampled_names=names_a,
            grism_groups=None,
            grism_group_operators=None,
            velocity_obs=None,
        )
    )

    logger = logging.getLogger('jax._src.dispatch')
    jax.config.update('jax_log_compiles', True)

    class _Counter(logging.Handler):
        def emit(self, record):
            if 'Compiling' in record.getMessage():
                n_compiles['count'] += 1

    counter = _Counter()
    logging.getLogger('jax').addHandler(counter)

    t0 = time.time()
    wa = arg_fn(
        theta_a, image_obs=img_a, grism_obs=grm_a, fixed_pars=fixed_a
    ).block_until_ready()
    t_arg_a = time.time() - t0
    compiles_after_a = n_compiles['count']
    t0 = time.time()
    wb = arg_fn(
        theta_b, image_obs=img_b, grism_obs=grm_b, fixed_pars=fixed_b
    ).block_until_ready()
    t_arg_b = time.time() - t0
    compiles_after_b = n_compiles['count']
    jax.config.update('jax_log_compiles', False)
    logging.getLogger('jax').removeHandler(counter)

    print(
        f'arg path: call A {t_arg_a:.1f} s ({compiles_after_a} compiles), '
        f'call B {t_arg_b:.3f} s ({compiles_after_b - compiles_after_a} new compiles)'
    )

    ok_h1a = bool(np.array_equal(np.asarray(wa), np.asarray(va)))
    ok_h1b = bool(np.array_equal(np.asarray(wb), np.asarray(vb)))
    print(
        f'H1 bit-identical: A {ok_h1a} (arg {wa} vs closure {va}), '
        f'B {ok_h1b} (arg {wb} vs closure {vb})'
    )
    ok_h2 = compiles_after_b == compiles_after_a
    print(f'H2 no recompile galaxy-to-galaxy: {ok_h2}')

    # ---- H3: vmap over stacked obs ------------------------------------------
    if sa['image_obs'] == sb['image_obs'] and sa['grism_obs'] == sb['grism_obs']:
        stack = lambda *xs: jnp.stack([jnp.asarray(x) for x in xs])
        img_batch = jax.tree_util.tree_map(stack, img_a, img_b)
        grm_batch = jax.tree_util.tree_map(stack, grm_a, grm_b)
        fix_batch = jax.tree_util.tree_map(stack, fixed_a, fixed_b)
        theta_batch = jnp.stack([theta_a, theta_b])

        vmapped = jax.jit(
            jax.vmap(
                partial(
                    _log_likelihood_total_source,
                    source=ia.source,
                    sampled_names=names_a,
                    grism_groups=None,
                    grism_group_operators=None,
                    velocity_obs=None,
                )
            )
        )
        t0 = time.time()
        wv = vmapped(
            theta_batch, image_obs=img_batch, grism_obs=grm_batch, fixed_pars=fix_batch
        ).block_until_ready()
        print(f'vmap compile+eval: {time.time() - t0:.1f} s')
        solo = np.array([wa, wb])
        batch = np.asarray(wv)
        rel = np.abs(batch - solo) / np.abs(solo)
        print(
            f'H3 vmap vs solo: batch={batch} solo={solo} max rel diff {rel.max():.2e}'
        )
    else:
        print('H3 SKIPPED: treedef mismatch must be resolved first (see aux diffs)')

    print('\nsummary:')
    print(f'  H1 {"PASS" if ok_h1a and ok_h1b else "FAIL"}')
    print(f'  H2 {"PASS" if ok_h2 else "FAIL"}')


if __name__ == '__main__':
    main()
