"""Vista GH200 benchmark matrix for the kl_pipe grism posterior.

One self-contained runner; writes one timestamped JSON. Sections:

  a  posterior primal / grad / value_and_grad, configs Q (flagship) and
     P (production), min-of-N with block_until_ready
  b  numpyro chain_method comparison on Q: vectorized vs sequential vs
     parallel (parallel needs >= n_chains devices; on CPU the runner sets
     jax_num_cpu_devices BEFORE JAX init, on GPU it uses the real count)
  c  jax.checkpoint variants on build_cube (ckpt_port.py monkeypatches),
     Q and P, with gradient-equivalence checks
  d  dispersion-leg layout matrix: BCOO vs dense-gather vs scan-streaming
     (stream_fusion_port.py), 1 and 4 rolls, forward + value_and_grad
  e  fp32 vs fp64: two subprocess self-invocations (precision must be set
     before the kl_pipe import chain), pixel deltas (max|diff|/peak on the
     flagship grism + broadband renders) + posterior timing per precision
  f  environment record (always runs): jax/jaxlib/numpyro versions,
     devices, platform, hostname, git commit
  g  galaxy-batching throughput (opt-in): vmap value_and_grad over N fits,
     sweep N -> per-fit time + fits/s (the 30M-scale throughput lever;
     batches theta only, a valid compute proxy pre-keystone-B)
  h  Laplace preconditioner cost (opt-in): cold vs warm (one-time compile vs
     per-galaxy compute) + n_starts sweep -- isolates the survey bottleneck

Every section degrades gracefully: failures are caught and the traceback
is recorded in the JSON under the section (never silently skipped);
deliberate skips record the reason.

Usage:
  python bench_matrix.py                          # everything, fp64
  python bench_matrix.py --sections a,d --nreps 10
  python bench_matrix.py --x64 0                  # full matrix in fp32
  python bench_matrix.py --force-no-galsim        # simulate Vista (no galsim)

Run from the vista_kit directory, or from anywhere (kit dir is added to
sys.path). Requires kl_pipe importable (pip install -e / PYTHONPATH).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback

KIT_DIR = os.path.dirname(os.path.abspath(__file__))
if KIT_DIR not in sys.path:
    sys.path.insert(0, KIT_DIR)


# ---------------------------------------------------------------------------
# CLI + JAX configuration (MUST happen before any jax array op / kl_pipe import)
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--sections',
        default='a,b,c,d,e',
        help='comma list from {a,b,c,d,e,g,h} (f always runs; g,h opt-in)',
    )
    p.add_argument(
        '--configs', default='Q,P', help='comma list from {Q,P} for sections a/c'
    )
    p.add_argument(
        '--nreps', type=int, default=30, help='timing repetitions (min-of-N reported)'
    )
    p.add_argument('--out', default=None, help='output JSON path')
    p.add_argument(
        '--x64',
        type=int,
        default=1,
        choices=(0, 1),
        help='1 = fp64 (default), 0 = fp32 pass',
    )
    p.add_argument(
        '--cpu-devices',
        type=int,
        default=2,
        help='jax_num_cpu_devices for CPU parallel-chains fallback',
    )
    p.add_argument('--mcmc-samples', type=int, default=200)
    p.add_argument('--mcmc-warmup', type=int, default=50)
    p.add_argument('--mcmc-chains', type=int, default=2)
    p.add_argument(
        '--batch-sizes',
        default='1,2,4,8,16,32,64,128,256',
        help='section g: galaxy-batch sizes (vmap width); stops on OOM',
    )
    p.add_argument(
        '--batch-nreps', type=int, default=10, help='section g: reps per batch size'
    )
    p.add_argument(
        '--precond-starts',
        default='1,4',
        help='section h: laplace_preconditioner n_starts values to sweep',
    )
    p.add_argument(
        '--force-no-galsim',
        action='store_true',
        help='block real galsim to simulate the Vista container',
    )
    p.add_argument(
        '--precision-child', action='store_true', help=argparse.SUPPRESS
    )  # internal: section-e worker
    return p.parse_args()


def configure_jax(args) -> dict:
    """All pre-init JAX config. Returns notes recorded in the JSON."""
    notes = {'x64_requested': bool(args.x64)}
    # route the choice through kl_pipe's central precision control too, so
    # a kl_pipe import (whenever it happens) agrees with the flag
    os.environ['KLPIPE_FP32'] = '' if args.x64 else '1'
    import jax

    jax.config.update('jax_enable_x64', bool(args.x64))

    # forbid the tf32 tensor-core shortcut for fp32 matmuls (GH200/Ampere+):
    # without this pin an fp32 pass silently runs 10-bit-mantissa matmuls,
    # invalidating any fp32-vs-fp64 accuracy comparison. No-op on CPU/fp64.
    jax.config.update('jax_default_matmul_precision', 'highest')
    notes['jax_default_matmul_precision'] = 'highest'

    # CPU multi-device fallback for chain_method='parallel'. Must run before
    # backend init. Harmless on GPU (only sizes the unused CPU client).
    try:
        jax.config.update('jax_num_cpu_devices', args.cpu_devices)
        notes['jax_num_cpu_devices'] = args.cpu_devices
    except Exception as err:  # older jax: use XLA_FLAGS instead
        notes['jax_num_cpu_devices'] = (
            f'FAILED ({err}); set XLA_FLAGS='
            f'"--xla_force_host_platform_device_count={args.cpu_devices}"'
        )

    if not args.x64:
        # kl_pipe force-enables x64 at import (coordinates/source/lines);
        # intercept those calls so the fp32 pass stays fp32. LOUD by design.
        _orig_update = jax.config.update

        def _guarded_update(name, value):
            if name == 'jax_enable_x64' and value:
                print(
                    '[bench] fp32 mode: intercepted jax_enable_x64=True '
                    '(kl_pipe import-time force)'
                )
                return None
            return _orig_update(name, value)

        jax.config.update = _guarded_update
        notes['fp32_intercept'] = True
    return notes


# ---------------------------------------------------------------------------
# timing helper
# ---------------------------------------------------------------------------


def time_fn(fn, *args, nreps=30):
    """Min-of-N timing with block_until_ready; first call = compile."""
    import jax

    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0

    times = []
    for _ in range(nreps):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    import numpy as np

    arr = np.asarray(times)
    return {
        'compile_ms': compile_s * 1e3,
        'min_ms': float(arr.min()) * 1e3,
        'mean_ms': float(arr.mean()) * 1e3,
        'std_ms': float(arr.std()) * 1e3,
        'nreps': nreps,
    }, out


# ---------------------------------------------------------------------------
# lazy task-state cache (built once, shared across sections)
# ---------------------------------------------------------------------------

_STATE: dict = {}


def get_state(config_name: str):
    if config_name in _STATE:
        return _STATE[config_name]
    import tasks_vista as tv

    t0 = time.perf_counter()
    if config_name == 'Q':
        source, priors, task, obs_f087, obs_grism, true = tv.build_flagship_task()
        nlam = int(len(obs_grism.cube_pars.lambda_grid))
        # nlam is vestigial under analytic dispersal (no wavelength grid for the
        # line); record the actual pathway so the JSON is self-documenting.
        meta = {
            'rolls': 1,
            'bands': 1,
            'nlam': nlam,
            'dispersal_method': obs_grism.dispersal_method,
        }
    elif config_name == 'P':
        source, priors, task, image_obs, grism_obs, true = tv.build_production_task()
        obs_g0 = next(iter(grism_obs.values()))
        nlam = int(len(obs_g0.cube_pars.lambda_grid))
        meta = {
            'rolls': len(grism_obs),
            'bands': len(image_obs),
            'nlam': nlam,
            'dispersal_method': obs_g0.dispersal_method,
        }
    else:
        raise ValueError(f'unknown config {config_name!r}')
    build_s = time.perf_counter() - t0
    theta, sampled_names = tv.perturbed_theta(priors, true)
    meta.update({'n_sampled': len(sampled_names), 'build_s': build_s})
    _STATE[config_name] = {
        'source': source,
        'priors': priors,
        'task': task,
        'true': true,
        'theta': theta,
        'meta': meta,
    }
    print(f'[bench] built config {config_name}: {meta}')
    return _STATE[config_name]


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def section_a(args):
    """Posterior primal / grad / value_and_grad for each config."""
    import jax

    out = {}
    for cfg in args.configs.split(','):
        st = get_state(cfg)
        task, theta = st['task'], st['theta']
        rows = {}
        fns = {
            'primal': jax.jit(task._log_posterior_jittable),
            'grad': jax.jit(jax.grad(task._log_posterior_jittable)),
            'value_and_grad': task.get_log_posterior_and_grad_fn(),
        }
        for label, fn in fns.items():
            r, val = time_fn(fn, theta, nreps=args.nreps)
            rows[label] = r
            print(
                f'  [a:{cfg}] {label:<16} min {r["min_ms"]:8.3f} ms  '
                f'mean {r["mean_ms"]:8.3f}  compile {r["compile_ms"]:.0f}'
            )
        if 'primal' in rows:
            _, v = time_fn(fns['primal'], theta, nreps=1)
            rows['posterior_value'] = float(v)
        out[cfg] = {'meta': st['meta'], 'timings': rows}
    return out


def section_b(args):
    """chain_method comparison (numpyro NUTS, shared Laplace preconditioner)."""
    import jax
    import numpy as np

    from kl_pipe.sampling import NumpyroSamplerConfig, build_sampler

    st = get_state('Q')
    task = st['task']

    n_dev = len(jax.devices())
    base = dict(
        n_samples=args.mcmc_samples,
        n_warmup=args.mcmc_warmup,
        n_chains=args.mcmc_chains,
        seed=42,
        progress=False,
        reparam_strategy='prior',
        dense_mass=True,
        precondition='laplace',
        target_accept_prob=0.8,
        max_tree_depth=8,
        init_strategy='prior',
    )

    # warm the posterior jit, then share ONE preconditioner across methods
    vg = task.get_log_posterior_and_grad_fn()
    jax.block_until_ready(vg(st['theta']))
    t0 = time.perf_counter()
    pre = task.laplace_preconditioner(n_starts=4, seed=42)
    pre_s = time.perf_counter() - t0
    print(f'  [b] laplace preconditioner: {pre_s:.1f} s (shared)')

    out = {
        'n_devices_default_backend': n_dev,
        'preconditioner_s': pre_s,
        'sampler_config': base,
        'methods': {},
    }
    summaries = {}
    for method in ('vectorized', 'sequential', 'parallel'):
        if method == 'parallel' and n_dev < base['n_chains']:
            out['methods'][method] = {
                'status': 'skipped',
                'reason': (
                    f'{n_dev} device(s) on default backend '
                    f'{jax.default_backend()!r} < n_chains={base["n_chains"]}; '
                    f'parallel pmap needs one device per chain'
                ),
            }
            print(f'  [b] {method}: SKIPPED ({out["methods"][method]["reason"]})')
            continue
        try:
            config = NumpyroSamplerConfig(chain_method=method, **base)
            sampler = build_sampler('numpyro', task, config)
            sampler._preconditioner = pre
            t0 = time.perf_counter()
            res = sampler.run()
            wall = time.perf_counter() - t0
            num_steps = np.asarray(res.diagnostics['num_steps'])
            rhat = res.diagnostics['r_hat']
            out['methods'][method] = {
                'status': 'ok',
                'wallclock_s': wall,
                'sum_num_steps': int(num_steps.sum()),
                'mean_tree_depth': float(np.log2(num_steps + 1).mean()),
                'n_divergences': int(res.diagnostics['n_divergences']),
                'max_rhat': float(max(rhat.values())),
                'mean_accept': float(
                    np.mean(np.asarray(res.diagnostics['mean_accept_prob']))
                ),
            }
            summaries[method] = res.get_summary()
            print(
                f'  [b] {method:>11}: {wall:7.1f} s, '
                f'{int(num_steps.sum())} leapfrogs, '
                f'max R-hat {max(rhat.values()):.3f}'
            )
        except Exception:
            out['methods'][method] = {
                'status': 'error',
                'error': traceback.format_exc(),
            }
            print(f'  [b] {method}: ERROR (recorded)')

    # posterior agreement across methods (worst mean shift in combined sigma)
    ok = [m for m in summaries if out['methods'][m]['status'] == 'ok']
    if len(ok) >= 2:
        ref = summaries[ok[0]]
        worst, worst_id = 0.0, ''
        for other in ok[1:]:
            for name in ref:
                s1, s2 = ref[name], summaries[other][name]
                denom = np.sqrt(s1['std'] ** 2 + s2['std'] ** 2)
                d = abs(s1['mean'] - s2['mean']) / denom if denom > 0 else 0.0
                if d > worst:
                    worst, worst_id = d, f'{ok[0]} vs {other}: {name}'
        out['worst_posterior_mean_shift_sigma'] = worst
        out['worst_posterior_mean_shift_id'] = worst_id
        print(f'  [b] worst posterior-mean shift: {worst:.3f} sigma ({worst_id})')
    return out


def section_c(args):
    """jax.checkpoint variants on build_cube (monkeypatch, always unpatched)."""
    import jax
    import jax.numpy as jnp

    import ckpt_port

    match_rtol = 1e-10 if args.x64 else 1e-3
    match_atol = 1e-12 if args.x64 else 1e-5
    out = {'match_rtol': match_rtol}
    try:
        for cfg in args.configs.split(','):
            st = get_state(cfg)
            task, theta = st['task'], st['theta']
            rows = {}
            base_val = base_grad = None
            for name, apply_patch in ckpt_port.VARIANTS.items():
                ckpt_port.unpatch()
                apply_patch()
                fwd = jax.jit(task._log_posterior_jittable)
                vg = jax.jit(jax.value_and_grad(task._log_posterior_jittable))
                r_f, _ = time_fn(fwd, theta, nreps=args.nreps)
                r_vg, (val, grad) = time_fn(vg, theta, nreps=args.nreps)
                if name == 'base':
                    base_val, base_grad = val, grad
                    ok = True
                else:
                    ok = bool(
                        jnp.allclose(val, base_val, rtol=match_rtol)
                        and jnp.allclose(
                            grad, base_grad, rtol=match_rtol, atol=match_atol
                        )
                    )
                rows[name] = {
                    'fwd_min_ms': r_f['min_ms'],
                    'vg_min_ms': r_vg['min_ms'],
                    'vg_mean_ms': r_vg['mean_ms'],
                    'vg_compile_ms': r_vg['compile_ms'],
                    'grad_matches_base': ok,
                }
                if not ok:
                    rows[name]['max_rel_grad_diff'] = float(
                        jnp.max(
                            jnp.abs(grad - base_grad) / (jnp.abs(base_grad) + 1e-30)
                        )
                    )
                print(
                    f'  [c:{cfg}] {name:<10} fwd {r_f["min_ms"]:7.2f} ms  '
                    f'vg {r_vg["min_ms"]:7.2f} ms  match={ok}'
                )
            out[cfg] = rows
    finally:
        ckpt_port.unpatch()
    return out


def section_d(args):
    """BCOO vs gather vs scan-streaming dispersion leg (synthetic inputs)."""
    import numpy as np
    import jax
    import jax.numpy as jnp

    import stream_fusion_port as sfp

    match_rtol = 1e-9 if args.x64 else 1e-3
    edges, dl, I, lam, sig, ct = sfp.build_problem()
    out = {
        'match_rtol': match_rtol,
        'geometry': 'fine 96x96, Nlam 33, oversample-3 flagship dispersion',
    }

    for label, rolls in {'1_roll': [0.0], '4_rolls': [0, 45, 90, 135]}.items():
        ops = [
            sfp.build_operators(96, 96, 33, 3.0, np.deg2rad(0.0), np.deg2rad(a))
            for a in rolls
        ]
        variants_per_roll = [
            sfp.make_variants(b, gi, gw, edges, dl, (96, 96)) for (b, gi, gw) in ops
        ]
        rows = {}
        base_val = base_grads = None
        for name in ('A_bcoo', 'A2_gather', 'B_stream', 'C_stream_remat'):

            def loss(I_, lam_, sig_, _name=name):
                total = 0.0
                for v in variants_per_roll:
                    total = total + jnp.sum((v[_name](I_, lam_, sig_) * ct) ** 2)
                return total

            fwd = jax.jit(loss)
            vg = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))
            r_f, _ = time_fn(fwd, I, lam, sig, nreps=args.nreps)
            r_vg, (val, grads) = time_fn(vg, I, lam, sig, nreps=args.nreps)
            if name == 'A_bcoo':
                base_val, base_grads = val, grads
                ok = True
                max_dev = 0.0
            else:
                # atol scaled to each gradient's magnitude: near-zero
                # elements sit at eps of the LARGE entries under a different
                # summation schedule (matters in fp32; fp64 stays ~exact)
                atol_scale = 1e-12 if args.x64 else 1e-6
                devs = [
                    float(jnp.max(jnp.abs(g - b)) / (match_rtol * jnp.max(jnp.abs(b))))
                    for g, b in zip(grads, base_grads)
                ]
                max_dev = max(devs)
                ok = bool(jnp.allclose(val, base_val, rtol=match_rtol)) and all(
                    bool(
                        jnp.allclose(
                            g,
                            b,
                            rtol=match_rtol,
                            atol=atol_scale * float(jnp.max(jnp.abs(b))),
                        )
                    )
                    for g, b in zip(grads, base_grads)
                )
            rows[name] = {
                'fwd_min_ms': r_f['min_ms'],
                'vg_min_ms': r_vg['min_ms'],
                'vg_mean_ms': r_vg['mean_ms'],
                'match': ok,
                'max_grad_dev_over_rtol_x_peak': max_dev,
            }
            print(
                f'  [d:{label}] {name:<15} fwd {r_f["min_ms"]:8.2f} ms  '
                f'vg {r_vg["min_ms"]:8.2f} ms  match={ok}'
            )
        out[label] = rows
    return out


def section_e(args):
    """fp64 vs fp32: subprocess per precision (x64 set before kl_pipe import)."""
    import numpy as np

    out = {'children': {}}
    prefixes = {}
    for x64 in (1, 0):
        tag = 'fp64' if x64 else 'fp32'
        child_json = args.out + f'.{tag}_child.json'
        prefix = args.out + f'.{tag}_child'
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            '--precision-child',
            '--x64',
            str(x64),
            '--out',
            child_json,
            '--nreps',
            str(args.nreps),
        ]
        if args.force_no_galsim:
            cmd.append('--force-no-galsim')
        print(f'  [e] spawning {tag} child ...')
        proc = subprocess.run(
            cmd, cwd=KIT_DIR, capture_output=True, text=True, env=os.environ.copy()
        )
        if proc.returncode != 0:
            out['children'][tag] = {
                'status': 'error',
                'returncode': proc.returncode,
                'stdout_tail': proc.stdout[-3000:],
                'stderr_tail': proc.stderr[-3000:],
            }
            print(f'  [e] {tag} child FAILED (recorded)')
            continue
        with open(child_json) as f:
            out['children'][tag] = json.load(f)
        out['children'][tag]['status'] = 'ok'
        prefixes[tag] = prefix
        print(
            f'  [e] {tag} child ok: vg min '
            f'{out["children"][tag]["vg_min_ms"]:.2f} ms, dtype '
            f'{out["children"][tag]["grism_dtype"]}'
        )

    if set(prefixes) == {'fp64', 'fp32'}:
        for obs_name in ('grism', 'broadband'):
            a = np.load(prefixes['fp64'] + f'_{obs_name}.npy').astype(np.float64)
            b = np.load(prefixes['fp32'] + f'_{obs_name}.npy').astype(np.float64)
            peak = float(np.max(np.abs(a)))
            out[f'{obs_name}_max_abs_diff_over_peak'] = float(
                np.max(np.abs(a - b)) / peak
            )
            out[f'{obs_name}_rms_diff_over_peak'] = float(
                np.sqrt(np.mean((a - b) ** 2)) / peak
            )
            print(
                f'  [e] {obs_name}: max|diff|/peak = '
                f'{out[f"{obs_name}_max_abs_diff_over_peak"]:.3e}'
            )
        v64 = out['children']['fp64']['vg_min_ms']
        v32 = out['children']['fp32']['vg_min_ms']
        out['vg_speedup_fp32_over_fp64'] = v64 / v32
        out['posterior_value_fp64'] = out['children']['fp64']['posterior_value']
        out['posterior_value_fp32'] = out['children']['fp32']['posterior_value']
        print(f'  [e] value_and_grad fp32 speedup: {v64 / v32:.2f}x')
    else:
        out['comparison'] = 'skipped: one or both children failed (see children)'
    return out


def run_precision_child(args):
    """Section-e worker: build Q, render truth images, time posterior vg."""
    import jax
    import numpy as np

    import tasks_vista as tv

    source, priors, task, obs_f087, obs_grism, true = tv.build_flagship_task()
    theta, _ = tv.perturbed_theta(priors, true)

    grism = np.asarray(source.render_grism(true, obs_grism))
    broadband = np.asarray(source.render_broadband(true, obs_f087, 'F087'))
    prefix = args.out[:-5] if args.out.endswith('.json') else args.out
    np.save(prefix + '_grism.npy', grism)
    np.save(prefix + '_broadband.npy', broadband)

    vg = task.get_log_posterior_and_grad_fn()
    r_vg, (val, _) = time_fn(vg, theta, nreps=args.nreps)

    payload = {
        'x64': bool(args.x64),
        'grism_dtype': str(grism.dtype),
        'default_dtype': str(jax.numpy.zeros(1).dtype),
        'posterior_value': float(val),
        'vg_min_ms': r_vg['min_ms'],
        'vg_mean_ms': r_vg['mean_ms'],
        'vg_compile_ms': r_vg['compile_ms'],
        'nreps': args.nreps,
    }
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(
        f'[child x64={args.x64}] wrote {args.out} '
        f'(dtype {grism.dtype}, vg min {r_vg["min_ms"]:.2f} ms)'
    )


def section_f(args, jax_notes):
    import jax
    import jaxlib

    env = {
        'timestamp': datetime.datetime.now().isoformat(),
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'machine': platform.machine(),
        'python': sys.version.split()[0],
        'jax': jax.__version__,
        'jaxlib': jaxlib.__version__,
        'default_backend': jax.default_backend(),
        'devices': [str(d) for d in jax.devices()],
        'local_device_count': jax.local_device_count(),
        'x64_active': bool(jax.config.jax_enable_x64),
        'jax_config_notes': jax_notes,
        'argv': sys.argv,
    }
    try:
        import numpyro

        env['numpyro'] = numpyro.__version__
    except Exception as err:
        env['numpyro'] = f'unavailable: {err}'
    try:
        env['git_commit'] = subprocess.run(
            ['git', '-C', KIT_DIR, 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        env['git_branch'] = subprocess.run(
            ['git', '-C', KIT_DIR, 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception as err:
        env['git_commit'] = f'unavailable: {err}'
    try:
        import galsim as _gs

        env['galsim'] = (
            'STUB (kit shim)'
            if getattr(_gs, '__kl_vista_stub__', False)
            else getattr(_gs, '__version__', 'real, version unknown')
        )
    except ImportError:
        env['galsim'] = 'absent'
    return env


# ---------------------------------------------------------------------------
# section g: galaxy-batching throughput
# ---------------------------------------------------------------------------


def _device_peak_mem_bytes():
    """Peak device bytes in use, or None (CPU / unsupported)."""
    import jax

    try:
        stats = jax.devices()[0].memory_stats()
        return int(stats.get('peak_bytes_in_use')) if stats else None
    except Exception:
        return None


def section_g(args):
    """Galaxy-batching throughput: vmap value_and_grad over N independent fits.

    THE 30M-scale lever. Section a showed single fits are launch/bandwidth-
    bound (fp32/remat/2-chain all near-flat), so throughput must come from
    packing many galaxies onto one device. This vmaps the posterior
    value_and_grad over a batch of N theta and sweeps N, reporting per-fit
    time and fits/s -- if batching amortizes overhead, per-fit time drops
    with N until the device saturates (then it flattens / OOMs).

    Batches over THETA only: data is baked into the likelihood closure
    (pre-keystone-B), but the model render -- the dominant cost -- depends on
    theta, so this is a valid compute-throughput proxy. True multi-DATA
    batching (different galaxies' images) needs the data-as-argument refactor
    (PRODUCTION_SPEEDUPS Tier B); this measures the compute ceiling that
    refactor would unlock.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    sizes = [int(x) for x in args.batch_sizes.split(',')]
    out = {}
    for cfg in args.configs.split(','):
        st = get_state(cfg)
        task, theta0 = st['task'], st['theta']
        batched_vg = jax.jit(jax.vmap(jax.value_and_grad(task._log_posterior_jittable)))
        D = int(np.asarray(theta0).shape[0])
        theta_np = np.asarray(theta0)
        rows = {}
        per_fit_n1 = None
        for N in sizes:
            rng = np.random.default_rng(1000 + N)
            # tiny relative jitter -> N distinct, in-support thetas
            batch = jnp.asarray(
                theta_np[None, :] * (1.0 + 1e-3 * rng.standard_normal((N, D)))
            )
            try:
                r, _ = time_fn(batched_vg, batch, nreps=args.batch_nreps)
            except Exception as err:
                rows[str(N)] = {'status': 'oom_or_error', 'error': repr(err)[:300]}
                print(f'  [g:{cfg}] N={N}: OOM/error -> stop ({repr(err)[:70]})')
                break
            per_fit = r['min_ms'] / N
            if N == 1:
                per_fit_n1 = per_fit
            speedup = (per_fit_n1 / per_fit) if per_fit_n1 else None
            rows[str(N)] = {
                'batch_min_ms': r['min_ms'],
                'batch_mean_ms': r['mean_ms'],
                'compile_ms': r['compile_ms'],
                'per_fit_ms': per_fit,
                'throughput_fits_per_s': N / (r['min_ms'] / 1e3),
                'per_fit_speedup_vs_N1': speedup,
                'peak_mem_bytes': _device_peak_mem_bytes(),
            }
            msg = (
                f'  [g:{cfg}] N={N:5d}: batch {r["min_ms"]:8.2f} ms  '
                f'per-fit {per_fit:8.4f} ms  {N / (r["min_ms"] / 1e3):9.1f} fits/s'
            )
            if speedup:
                msg += f'  ({speedup:5.1f}x/N1)'
            print(msg)
        out[cfg] = {'meta': st['meta'], 'batch_nreps': args.batch_nreps, 'sizes': rows}
    return out


# ---------------------------------------------------------------------------
# section h: Laplace preconditioner cost
# ---------------------------------------------------------------------------


def section_h(args):
    """Laplace preconditioner cost + compile-vs-compute split.

    Section b showed the Laplace preconditioner is ~13 s/fit -- a fixed
    per-fit cost that dominates survey wallclock far more than the ~ms
    gradient. This isolates it: COLD (first call, incl. JIT compile) vs WARM
    (second call, same shapes, compile cached). warm ~= true per-galaxy
    amortized cost; (cold - warm) ~= one-time compile that amortizes across
    all galaxies of the same model shape. If warm is small, the 13 s is
    mostly compile and NOT a per-fit survey bottleneck; if warm ~ cold, it is
    a real per-galaxy cost and the top survey-scale target. Also sweeps
    n_starts to expose the multi-start L-BFGS scaling.
    """
    import jax

    out = {}
    for cfg in args.configs.split(','):
        st = get_state(cfg)
        task = st['task']
        # warm the posterior jit first so we isolate the preconditioner's own cost
        vg = task.get_log_posterior_and_grad_fn()
        jax.block_until_ready(vg(st['theta']))
        by_starts = {}
        for ns in [int(x) for x in args.precond_starts.split(',')]:
            t0 = time.perf_counter()
            task.laplace_preconditioner(n_starts=ns, seed=1)
            cold = time.perf_counter() - t0
            t0 = time.perf_counter()
            task.laplace_preconditioner(n_starts=ns, seed=2)
            warm = time.perf_counter() - t0
            by_starts[str(ns)] = {
                'cold_s': cold,
                'warm_s': warm,
                'compile_s_est': cold - warm,
            }
            print(
                f'  [h:{cfg}] n_starts={ns}: cold {cold:6.1f}s  warm {warm:6.1f}s'
                f'  (compile~{cold - warm:.1f}s)'
            )
        out[cfg] = {'meta': st['meta'], 'n_starts': by_starts}
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

SECTION_FNS = {
    'a': section_a,
    'b': section_b,
    'c': section_c,
    'd': section_d,
    'e': section_e,
    'g': section_g,
    'h': section_h,
}


def main():
    args = parse_args()
    jax_notes = configure_jax(args)

    if args.force_no_galsim:
        from psf_numpy import block_galsim

        block_galsim()

    if args.out is None:
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        tag = 'fp64' if args.x64 else 'fp32'
        args.out = os.path.join(KIT_DIR, f'results_vista_{stamp}_{tag}.json')

    if args.precision_child:
        run_precision_child(args)
        return

    results = {'sections': {}}
    results['env'] = section_f(args, jax_notes)
    print(
        f'[bench] env: {results["env"]["default_backend"]} '
        f'{results["env"]["devices"]}  x64={results["env"]["x64_active"]}'
    )

    requested = [s.strip() for s in args.sections.split(',') if s.strip()]
    unknown = [s for s in requested if s not in SECTION_FNS]
    if unknown:
        raise ValueError(f'unknown sections {unknown}; valid: a,b,c,d,e,g,h')

    t_total = time.perf_counter()
    for name in ('a', 'b', 'c', 'd', 'e', 'g', 'h'):
        if name not in requested:
            results['sections'][name] = {
                'status': 'skipped',
                'reason': 'not in --sections',
            }
            continue
        print(f'\n[bench] === section {name} ===')
        t0 = time.perf_counter()
        try:
            payload = SECTION_FNS[name](args)
            results['sections'][name] = {'status': 'ok', 'result': payload}
        except Exception:
            results['sections'][name] = {
                'status': 'error',
                'error': traceback.format_exc(),
            }
            print(f'[bench] section {name} ERRORED (recorded in JSON):')
            print(results['sections'][name]['error'].splitlines()[-1])
        results['sections'][name]['wallclock_s'] = time.perf_counter() - t0
        # persist incrementally so a crash never loses completed sections
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    results['total_wallclock_s'] = time.perf_counter() - t_total
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(
        f'\n[bench] wrote {args.out} ' f'(total {results["total_wallclock_s"]:.0f} s)'
    )


if __name__ == '__main__':
    main()
