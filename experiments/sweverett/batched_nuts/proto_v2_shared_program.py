"""Experiment 2: one compiled likelihood serving every galaxy in the run.

Design under test (from experiment 1's conclusions):
  - static template = galaxy A's obs, closed over; per-galaxy quantities
    enter as a dynamic pytree argument:
      image bands: data, variance
      grism rolls: data, variance, psf_data, kspace_psf_fft (z-dependent
                   Roman PSF values; shapes ensemble-pinned already),
                   lambda_ref (traced; enters xi arithmetically)
      fixed_pars (z), passed per galaxy
  - line_window_halfwidth pinned to the run-wide max (wider = exact).
  - continuum trace kernel precomputed once outside the trace (with flat
    throughput it is lambda_ref-independent: the window is
    lambda_ref +/- (Ncol+2)*dispersion, so lambda_ref cancels in s_min/max).
    Prototype: monkeypatch kl_pipe.dispersion.continuum_trace_kernel inside
    source's render with the precomputed pair; infra version hoists it like
    grism_group_operators.

Hypotheses:
  H1  per-galaxy values match each galaxy's own closure-path likelihood to
      a few ulp (constant-folding differences only).
  H2  ONE compile serves all 16 fits (zero recompiles after the first).
  H3  vmap over the stacked dynamic pytree reproduces solo evals.

Usage (worktree root): python experiments/sweverett/batched_nuts/proto_v2_shared_program.py
"""

import dataclasses
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
from kl_pipe.likelihood import (
    _log_likelihood_total_source,
    create_jitted_likelihood_from_obs,
)

RUN_DIR = Path(os.environ.get('KLPIPE_RUN_DIR', 'runs/cosmos25_ab_bb32gr32'))
N_FITS = int(os.environ.get('KLPIPE_N_FITS', '16'))


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


def pin_halfwidth(grism_obs, hw):
    out = {}
    for k, o in grism_obs.items():
        rc = dataclasses.replace(o.render_config, line_window_halfwidth=hw)
        out[k] = o.with_render_config(rc)
    return out


def precompute_continuum_kernel(built):
    """Continuum trace kernel for the bank, asserting galaxy-independence.

    With flat throughput the window lambda_ref +/- (Ncol+2)*dispersion makes
    lambda_ref cancel in the kernel's s-range, so one kernel can serve every
    fit -- but only while Ncol/dispersion/oversample also match. Compute the
    kernel for EVERY fit and roll and require exact equality, naming the
    offender (the earlier fit-0-vs-last spot check would miss a middle fit).
    """
    import kl_pipe.dispersion as kdisp

    kern0 = m_lo0 = None
    for i, (_, _, grm) in enumerate(built):
        for roll_key, roll in grm.items():
            gp = roll.grism_pars
            assert gp.throughput is None, 'kernel precompute assumes flat T'
            half_nm = (gp.image_pars.Ncol + 2) * gp.dispersion
            kern, m_lo = kdisp.continuum_trace_kernel(
                gp,
                roll.cube_pars.lambda_grid,
                roll.oversample,
                integration_window=(gp.lambda_ref - half_nm, gp.lambda_ref + half_nm),
            )
            if kern0 is None:
                kern0, m_lo0 = kern, m_lo
            elif m_lo != m_lo0 or not np.array_equal(kern, kern0):
                raise AssertionError(
                    f'continuum kernel differs at fit {i} roll {roll_key}: '
                    'the shared-program kernel pin would be silently wrong'
                )
    return kern0, m_lo0


def extract_dyn(image_obs, grism_obs, fixed_pars):
    return {
        'image': {
            k: {
                'data': o.data,
                'variance': o.variance,
                # broadband PSF is z-dependent (folding-tier split at z=1.2);
                # serving it from the template would be silently wrong for a
                # bank that crosses the tier, so it rides as traced leaves
                'psf_data': o.psf_data,
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
    }


def make_shared_fn(source, sampled_names, tmpl_image, tmpl_grism):
    """Jitted (theta, dyn) -> log_like, closing over the static template."""

    def fn(theta, dyn):
        image_obs = {
            k: dataclasses.replace(
                tmpl_image[k],
                data=dyn['image'][k]['data'],
                variance=dyn['image'][k]['variance'],
                psf_data=dyn['image'][k]['psf_data'],
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
        return _log_likelihood_total_source(
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

    return jax.jit(fn)


def main():
    spec, config, manifest = load_run(RUN_DIR)
    rows = [manifest.iloc[i] for i in range(min(N_FITS, len(manifest)))]
    print(
        f'{len(rows)} fits, z range '
        f'{min(r["truth.z"] for r in rows):.2f}-{max(r["truth.z"] for r in rows):.2f}'
    )

    t0 = time.time()
    built = [build_inputs(r, spec, config) for r in rows]
    print(f'build_fit_inputs x{len(rows)}: {time.time() - t0:.1f} s', flush=True)

    names = tuple(built[0][0].priors.sampled_names)
    for inp, _, _ in built:
        assert tuple(inp.priors.sampled_names) == names

    # pin the line window to the run-wide max (wider = exact, extra columns
    # carry ~zero line flux)
    hws = [
        o.render_config.line_window_halfwidth for _, _, g in built for o in g.values()
    ]
    hw_max = max(hws)
    print(f'line_window_halfwidth: per-fit range {min(hws)}-{max(hws)}, pin {hw_max}')
    built = [(inp, img, pin_halfwidth(grm, hw_max)) for inp, img, grm in built]

    # precompute the continuum trace kernel, asserting galaxy-independence
    # across the whole bank (flat throughput -> lambda_ref cancels)
    import kl_pipe.dispersion as kdisp

    tmpl_inp, tmpl_img, tmpl_grm = built[0]
    kern0, m_lo0 = precompute_continuum_kernel(built)
    print(
        f'continuum kernel precomputed: {kern0.shape[0]} taps, m_lo {m_lo0}, '
        f'galaxy-independence verified across all {len(built)} fits'
    )

    # ---- per-galaxy closure baselines (reference values) --------------------
    thetas, refs = [], []
    t0 = time.time()
    for i, (inp, img, grm) in enumerate(built):
        theta = jnp.asarray(
            inp.priors.sample(jax.random.PRNGKey(100 + i), n_samples=1)
        ).ravel()
        base = create_jitted_likelihood_from_obs(
            inp.source,
            names,
            dict(inp.priors.fixed_values),
            image_obs=img,
            grism_obs=grm,
        )
        refs.append(float(base(theta)))
        thetas.append(theta)
    print(
        f'closure baselines x{len(built)}: {time.time() - t0:.1f} s '
        f'(one full compile each)',
        flush=True,
    )

    # ---- shared program ------------------------------------------------------
    # monkeypatch the kernel with the precomputed pair so a traced
    # lambda_ref never reaches the numpy precompute (prototype-only; the
    # infra version hoists the kernel to likelihood construction)
    import kl_pipe.source as ksource

    real_kernel = kdisp.continuum_trace_kernel

    def _pinned_kernel(grism_pars, lambda_grid, oversample, integration_window=None):
        return kern0, m_lo0

    kdisp.continuum_trace_kernel = _pinned_kernel
    # source.py imports it inside the render fn from kl_pipe.dispersion, so
    # patching the module attribute is sufficient

    n_compiles = {'count': 0}

    class _Counter(logging.Handler):
        def emit(self, record):
            if 'Compiling' in record.getMessage():
                n_compiles['count'] += 1

    counter = _Counter()
    jax.config.update('jax_log_compiles', True)
    logging.getLogger('jax').addHandler(counter)

    shared = make_shared_fn(tmpl_inp.source, names, tmpl_img, tmpl_grm)

    vals, compile_marks, t_calls = [], [], []
    for i, (inp, img, grm) in enumerate(built):
        dyn = extract_dyn(img, grm, dict(inp.priors.fixed_values))
        t0 = time.time()
        v = shared(thetas[i], dyn).block_until_ready()
        t_calls.append(time.time() - t0)
        vals.append(float(v))
        compile_marks.append(n_compiles['count'])

    jax.config.update('jax_log_compiles', False)
    logging.getLogger('jax').removeHandler(counter)
    kdisp.continuum_trace_kernel = real_kernel

    rel = np.abs(np.array(vals) - np.array(refs)) / np.abs(np.array(refs))
    print(
        f'\nH1 shared vs closure per fit: max rel diff {rel.max():.2e} '
        f'(ulp scale ~2e-16)'
    )
    for i in (int(rel.argmax()),):
        print(f'   worst fit {i}: shared {vals[i]!r} vs closure {refs[i]!r}')
    new_compiles_after_first = compile_marks[-1] - compile_marks[0]
    print(
        f'H2 compiles: first call {compile_marks[0]}, '
        f'subsequent new compiles {new_compiles_after_first}; '
        f'call times first {t_calls[0]:.1f} s then '
        f'{np.median(t_calls[1:]) * 1000:.0f} ms median'
    )

    # ---- H3: vmap over all fits ---------------------------------------------
    kdisp.continuum_trace_kernel = _pinned_kernel
    dyns = [
        extract_dyn(img, grm, dict(inp.priors.fixed_values)) for inp, img, grm in built
    ]
    dyn_batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *dyns)
    theta_batch = jnp.stack(thetas)
    inner = make_shared_fn(tmpl_inp.source, names, tmpl_img, tmpl_grm)
    vfn = jax.jit(jax.vmap(inner._fun if hasattr(inner, '_fun') else inner))
    t0 = time.time()
    vv = np.asarray(vfn(theta_batch, dyn_batch).block_until_ready())
    t_vmap = time.time() - t0
    kdisp.continuum_trace_kernel = real_kernel
    rel_v = np.abs(vv - np.array(vals)) / np.abs(np.array(vals))
    print(
        f'H3 vmap ({len(built)} fits, compile+eval {t_vmap:.1f} s): '
        f'max rel diff vs solo shared {rel_v.max():.2e}'
    )

    print('\nsummary:')
    print(f'  H1 {"PASS" if rel.max() < 5e-14 else "FAIL"} (bar: few-ulp, 5e-14)')
    print(f'  H2 {"PASS" if new_compiles_after_first == 0 else "FAIL"}')
    print(f'  H3 {"PASS" if rel_v.max() < 5e-14 else "FAIL"}')


if __name__ == '__main__':
    main()
