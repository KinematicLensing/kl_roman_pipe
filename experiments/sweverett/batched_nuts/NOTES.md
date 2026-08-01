# Batched-NUTS track — experiment notes

Branch `cc/batched-nuts`, worktree `~/repos/kl_roman_pipe-wt/batched-nuts`.
Goal: one compiled program serving all galaxies (data-as-argument likelihood),
then NUTS vmapped over galaxies x chains. Run everything from the worktree
root (cwd resolves `kl_pipe`; the editable install points at the main
checkout otherwise).

## Experiment 1: data-as-argument mechanics (2026-08-01) — CONCLUDED

Script: `proto_data_as_arg.py`. Scene: `runs/cosmos25_ab_bb32gr32` expansion
(16 fits, 8 galaxies, ring pairs; catalog symlinked from the main checkout).

Hypotheses and results:

- **H1 (arg-path values == closure-path values): SUPPORTED at ulp level, not
  bit-identical.** Passing obs through the jit boundary instead of closing
  over them changes constant folding; observed relative differences ~2-4e-16
  (last ulp). Acceptance bar for the refactor: exact to a few ulp at the
  likelihood, plus posterior-level parity gates downstream. Not a silent
  tolerance loosening — recorded here deliberately.
- **H2 (two galaxies share one program): REFUTED as-is — the audit's "shapes
  nearly pinned" story was too optimistic.** Compile-cache reuse fails for
  ring partners (identical configs!) purely because `ImagePars`, `GrismPars`,
  `CubePars` compare by object identity inside the obs aux. For genuinely
  different galaxies the additional real variation is: `lambda_ref` /
  `lambda_grid` (z-dependent; Nlam 30 vs 34 across the bank),
  `render_config.line_window_halfwidth` (per-fit static int from priors),
  and the z-dependent grism PSF object sitting in aux.
- **H3 (vmap across galaxies): BLOCKED on H2**, not yet run.

Code-reading conclusions for the production config (analytic dispersal,
`throughput=None`, `continuum_fills_stamp`, post-dispersion PSF):

1. `lambda_ref` enters the line term only arithmetically
   (`xi = (lam_obs - lambda_ref)/dispersion`, source.py:589) — can be a
   traced per-galaxy input.
2. `lambda_grid` is UNUSED in the line term when throughput is None
   (only feeds `jnp.interp` for the throughput weight) and serves only as a
   lookup table for the continuum kernel.
3. The continuum trace kernel (dispersion.py:632) is a trace-time numpy
   precompute whose window is `lambda_ref +/- (Ncol+2)*dispersion` — the
   lambda_ref cancels in s_min/s_max, so with flat throughput the kernel is
   IDENTICAL across galaxies. Needs a small refactor to compute the window
   relative (without touching a traced lambda_ref) — exactness-preserving.
4. `line_window_halfwidth` must be pinned ensemble-wide (max over the
   manifest), exactly like the grism PSF kernel size already is. Wider
   window = exact (extra columns carry ~zero flux), modest cost.
5. Grism PSF per-galaxy VALUES flow through traced leaves
   (`psf_data`/`kspace_psf_fft`); shape already ensemble-pinned. The galsim
   `psf` object in aux is metadata for the jitted path and must be excluded
   from the program cache key (dropped/None'd in the data-args mode).
6. Broadband PSF has exactly 2 shape classes (folding-threshold tier split
   at z=1.2) -> 2 compiled programs, acceptable (or pin one tier).
7. `z` is a fixed par (pin_z_to_truth) -> already flows as a traced arg once
   fixed_pars becomes an argument.

Refactor shape chosen (prototype v2): surgical split at the likelihood seam —
static obs template closed over (galaxy-0's obs with per-galaxy fields
stripped), dynamic pytree argument
`{channel: {data, variance, psf leaves, lambda_ref}, fixed_pars, prior params}`
recombined via `dataclasses.replace` inside the traced function, with loud
equality asserts at batch-build time for every aux field that must be
galaxy-independent. Avoids touching ImagePars equality semantics repo-wide.

## Experiment 2: shared program across the run (2026-08-01) — ALL PASS

Script: `proto_v2_shared_program.py`, all 16 fits of the bb32gr32 expansion
(z 0.95-1.69). Static template (fit 0's obs) closed over; dynamic pytree arg
carries per-fit {image data/variance, grism data/variance/psf_data/lambda_ref,
fixed_pars}; halfwidth pinned to run max (23-31 -> 31); continuum kernel
precomputed once (galaxy-independence ASSERTED against a second galaxy's
kernel, exact match).

- H1 PASS: max rel diff vs per-galaxy closure baselines 1.57e-15.
- H2 PASS: 1 compile total, 0 recompiles across the other 15 fits
  (2.5 s once + 30 ms/call CPU; closure path = 16 full compiles, 40 s).
- H3 PASS: vmap over all 16 stacked fits, max rel diff 8.5e-16, one 3.4 s
  compile.

Robustness flags for the infra version (not exercised as failures here):
1. Broadband PSF leaves were NOT in the dynamic pytree (template's PSF used
   for all fits) and H1 still passed at ulp level -- so the z<=1.2 vs >1.2
   folding-threshold tier did not change kernel values for this bank. Do
   NOT rely on that: include broadband psf_data/kspace_psf_fft in the
   dynamic leaves (cheap, always correct), and assert leaf-shape equality
   at batch build with a loud error naming the offending fit.
2. The kernel monkeypatch is prototype-only; infra hoists the continuum
   kernel to likelihood construction (grism_group_operators precedent).

## Next

- v2: implement split/merge + traced lambda_ref + halfwidth pin; H1/H2 on
  rows 0 vs 2, then all 16; then H3 vmap.
- Then: batched potential_fn (priors as data), CPU mini-config NUTS parity
  (per-galaxy posteriors vs solo fits), then the vista B={16,64} demo
  (pre-submission math check-in first).
- XLA "algebraic simplifier circular loop" warnings reappear on these
  compiles (known from 07-29, watch compile times).
