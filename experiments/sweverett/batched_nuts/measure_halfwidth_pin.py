"""Direct measurement: pinned vs per-fit line_window_halfwidth (grism analytic dispersal).

Hypothesis (audit finding, see NOTES.md): the batched-NUTS prototypes pin
GrismObs.render_config.line_window_halfwidth to the run-wide max across a
16-fit bank so one compiled program serves all galaxies. The prototype
session called this pin "exact"; an audit found it is an erfc-tail
approximation instead -- the wider (pinned) window deposits extra flux taps
that the per-fit sizing (line_window_halfwidth_for_priors, n_sigma=4,
render.py:1037-1107) intentionally drops, at a level bounded by
erfc(4/sqrt(2)) ~ 6e-5 of the line flux (one-sided). Nobody had directly
compared pinned vs per-fit rendering before pinning -- this script does that
comparison with no monkeypatches, using the production render/likelihood
path.

H0: |pinned - native| <= 1e-4 of the line flux at truth theta, and
    delta log-likelihood at truth is negligible against unit-scale chi2
    fluctuations (|delta logL| << 1).

Target fit: the one with the SMALLEST native halfwidth in the bank (max
contrast against the pinned run-wide max) -- the widest relative widening,
so the largest expected pin-vs-native difference.

Usage (worktree root):
  python experiments/sweverett/batched_nuts/measure_halfwidth_pin.py
"""

import dataclasses
import time
from pathlib import Path

import jax
import numpy as np

from kl_pipe.ensemble.expander import load_run, truth_from_row
from kl_pipe.ensemble.mocks import build_fit_inputs
from kl_pipe.likelihood import _build_pars_dict, create_jitted_likelihood_from_obs
from kl_pipe.sampling.task import _check_source_priors_fit_obs

# same run + fit count as proto_v2_shared_program.py (NOTES.md experiment 2)
RUN_DIR = Path('runs/cosmos25_ab_bb32gr32')
N_FITS = 16


def build_inputs(row, spec, config):
    """Mirror proto_v2_shared_program.py's build_inputs (production fit-input path)."""
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
    """Widen every roll's render_config to the run-wide max halfwidth (data untouched)."""
    out = {}
    for k, o in grism_obs.items():
        rc = dataclasses.replace(o.render_config, line_window_halfwidth=hw)
        out[k] = o.with_render_config(rc)
    return out


def main():
    t_start = time.time()
    spec, config, manifest = load_run(RUN_DIR)
    rows = [manifest.iloc[i] for i in range(min(N_FITS, len(manifest)))]
    built = [build_inputs(r, spec, config) for r in rows]
    print(f'build_fit_inputs x{len(rows)}: {time.time() - t_start:.1f} s')

    # run-wide halfwidth range + the (fit, roll) with the smallest native value
    recs = []
    for i, (_, _, grm) in enumerate(built):
        for k, o in grm.items():
            recs.append((o.render_config.line_window_halfwidth, i, k))
    hw_max = max(r[0] for r in recs)
    hw_min, fit_idx, roll_key = min(recs, key=lambda r: r[0])
    print(
        f'run halfwidth range {min(r[0] for r in recs)}-{hw_max} fine px; '
        f'target fit {fit_idx} (z={rows[fit_idx]["truth.z"]:.3f}), '
        f'native halfwidth {hw_min} -> pinned {hw_max} '
        f'({hw_max - hw_min} extra fine px per side)'
    )

    inp, image_obs, grism_obs_native = built[fit_idx]
    row = rows[fit_idx]
    names = tuple(inp.priors.sampled_names)
    fixed_pars = dict(inp.priors.fixed_values)
    source = inp.source
    grism_obs_pinned = pin_halfwidth(grism_obs_native, hw_max)

    # sizing-formula context: what v_max/dispersion_max line_window_halfwidth_for_priors
    # actually used for this fit (render.py:1037-1107)
    from kl_pipe.render import _prior_abs_max, _prior_upper

    fit_spec = inp.priors._param_spec
    v0_abs_max = _prior_abs_max(fit_spec['vel.v0'], 6.0)
    vcirc_upper_asused = _prior_upper(fit_spec['vel.vcirc'], 6.0)
    disp_upper = _prior_upper(fit_spec['Halpha.dispersion'], 6.0)
    print(
        f'sizing formula for fit {fit_idx}: v0_abs_max={v0_abs_max:.1f}, '
        f'vcirc_upper (as coded)={vcirc_upper_asused:.2f}, disp_upper={disp_upper:.1f} '
        f'-> v_max={v0_abs_max + vcirc_upper_asused:.1f} km/s (hw={hw_min})'
    )

    # ---- thetas: truth, plus prior-tail draws where the sizing math
    # (line_window_halfwidth_for_priors: window ~ |v0|+vcirc, +4*sigma_v)
    # bites hardest -- near edge-on so sin(i)~1 realizes the v_los bound. ----
    truth = truth_from_row(row)
    theta_truth = inp.priors.full_pars_to_theta(truth)

    def theta_variant(**overrides):
        pars = dict(truth)
        pars.update(overrides)
        return inp.priors.full_pars_to_theta(pars)

    # designed_edge: v0 at its Gaussian _prior_abs_max (1200 = 6 sigma),
    # Halpha.dispersion at its TruncatedNormal upper bound (150), edge-on
    # (cosi near the prior's low bound 0.05) so sin(i)~1 realizes the v_los
    # bound -- this is the literal worst case line_window_halfwidth_for_priors
    # sized fit 12's window against (v_max used there = 1206.2 km/s, driven
    # almost entirely by the v0 term; see below).
    theta_designed_edge = theta_variant(
        cosi=0.05, **{'vel.v0': 1200.0, 'vel.vcirc': 138.0, 'Halpha.dispersion': 150.0}
    )
    # beyond_edge: same, but vel.vcirc at the true LogNormal 6-sigma tail
    # (497 = exp(mu + 6*sigma)) instead of a typical value. _prior_upper's
    # unbounded branch ("mu + n_sd*sigma") is written for linear-space
    # (Gaussian) priors; applied to vel.vcirc's LogNormal(mu=4.93 in
    # log-space, sigma=0.213) it returns 6.2 km/s, not exp(6.21)=497 km/s --
    # so the window sizing silently under-covers this axis. This theta
    # exposes that gap directly (unrelated to the halfwidth-pin question,
    # flagged separately below).
    theta_beyond_edge = theta_variant(
        cosi=0.05, **{'vel.v0': 1200.0, 'vel.vcirc': 497.0, 'Halpha.dispersion': 150.0}
    )
    thetas = {
        'truth': theta_truth,
        'designed_edge (v0=1200,vcirc=138,disp=150,cosi=0.05)': theta_designed_edge,
        'beyond_edge (v0=1200,vcirc=497,disp=150,cosi=0.05)': theta_beyond_edge,
    }

    line_flux = truth['Halpha.flux']
    print(
        f'\ntarget fit truth Halpha.flux (line flux) = {line_flux:.4g} [1e-17 erg/s/cm2]'
    )

    # ---- (a) rendered model images, all 4 rolls of the target fit ----
    print('\n(a) rendered grism model image, pinned vs native, per roll:')
    for label, theta in thetas.items():
        pars = _build_pars_dict(theta, names, fixed_pars)
        print(f'  theta={label}:')
        for k in grism_obs_native:
            img_native = np.asarray(source.render_grism(pars, grism_obs_native[k]))
            img_pinned = np.asarray(source.render_grism(pars, grism_obs_pinned[k]))
            diff = img_pinned - img_native
            peak = np.abs(img_native).max()
            print(
                f'    {k}: max|diff|={np.abs(diff).max():.3e} flux/pix, '
                f'max|diff|/peak={np.abs(diff).max() / peak:.3e}, '
                f'sum(diff)/line_flux={diff.sum() / line_flux:.3e}, '
                f'peak_native={peak:.4g} flux/pix'
            )

    # ---- (b)+(c) log-likelihood + gradient at truth (same data both ways) ----
    ll_grism_native = create_jitted_likelihood_from_obs(
        source, names, fixed_pars, grism_obs={roll_key: grism_obs_native[roll_key]}
    )
    ll_grism_pinned = create_jitted_likelihood_from_obs(
        source, names, fixed_pars, grism_obs={roll_key: grism_obs_pinned[roll_key]}
    )
    ll_joint_native = create_jitted_likelihood_from_obs(
        source, names, fixed_pars, image_obs=image_obs, grism_obs=grism_obs_native
    )
    ll_joint_pinned = create_jitted_likelihood_from_obs(
        source, names, fixed_pars, image_obs=image_obs, grism_obs=grism_obs_pinned
    )
    grad_grism_native = jax.jit(jax.grad(ll_grism_native))
    grad_grism_pinned = jax.jit(jax.grad(ll_grism_pinned))

    print(
        f'\n(b)+(c) log-likelihood (target roll {roll_key} only, and joint '
        'all-channel) + gradient, pinned vs native:'
    )
    for label, theta in thetas.items():
        t0 = time.time()
        lg_n = float(ll_grism_native(theta))
        lg_p = float(ll_grism_pinned(theta))
        lj_n = float(ll_joint_native(theta))
        lj_p = float(ll_joint_pinned(theta))
        g_n = np.asarray(grad_grism_native(theta))
        g_p = np.asarray(grad_grism_pinned(theta))
        rel_g = np.abs(g_p - g_n) / np.maximum(np.abs(g_n), 1e-8)
        dt = time.time() - t0
        worst_name = names[int(np.argmax(rel_g))]
        print(f'  theta={label} ({dt:.1f} s):')
        print(
            f'    grism-only logL  native={lg_n:.6f} pinned={lg_p:.6f} '
            f'delta={lg_p - lg_n:.3e}'
        )
        print(
            f'    joint logL       native={lj_n:.6f} pinned={lj_p:.6f} '
            f'delta={lj_p - lj_n:.3e}'
        )
        print(
            f'    grad(grism-only logL) max rel diff={rel_g.max():.3e} '
            f'at {worst_name!r} (native grad there={g_n[int(np.argmax(rel_g))]:.4g})'
        )

    print(f'\ntotal runtime: {time.time() - t_start:.1f} s')


if __name__ == '__main__':
    main()
