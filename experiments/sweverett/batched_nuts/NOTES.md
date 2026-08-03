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

## Experiment 3: shared posterior + grads + batched-NUTS smoke (2026-08-01) — ALL PASS

Script: `proto_v3_batched_posterior.py` (4 fits, CPU). Priors travel as data:
all numeric attrs of each Prior's `__dict__` (incl. derived caches) traced,
template shallow-copied inside the trace; conditional log-normal parent
resolution mirrored from `PriorDict.log_prior` (static structure asserted
identical across galaxies).

- H4 PASS: shared posterior vs `task.get_log_posterior_and_grad_fn()` value,
  max rel 3.24e-15.
- H5 PASS: grads max rel 8.96e-14; ONE shared compile across fits (the +1
  per fit in the compile log is each reference task's own grad compile).
- H6 PASS (machinery only): `jit(vmap(warmup+scan))` blackjax NUTS, dense
  window adaptation per lane, `max_num_doublings=5`, 2 lanes x (10+5) steps,
  82 s CPU incl. compile; finite states, per-lane step sizes distinct.
  NOTE: blackjax `window_adaptation(...).run` returns extra kernel kwargs in
  its params dict — do not pass `max_num_doublings` again at kernel build.
- CPU lockstep NUTS at production depth is hours-slow (first attempt at
  4 lanes x 30 steps uncapped blew a 10-min budget) — real sampling
  measurements belong on vista, as ruled.

Next: vista gh-dev demo (pre-submission check-in first): 16 fits x 4 chains
= 64 lanes batched NUTS, parity gates vs the existing bb32gr32 solo results
already on vista disk.

## Vista demo v1 (2026-08-01, gh-dev, 16 fits x 4 chains = 64 lanes) — throughput PASS, quality FAIL for a known reason

Wall: build 36 s, warmup 2242 s, sampling 1716 s -> 4.1 min/fit-equivalent =
14.6 fits/node-hr (vs ~6 effective today, ~4x the 16-18 min solo).
Per-lockstep leapfrog at 64 lanes: 45 ms = 0.70 ms/lane vs solo 3.4-5.6
ms/step — consistent with the x5.7 batching gate. Straggler overhead x2.04
(lane-mean 62 steps/iter vs lockstep-max 127; 127 = depth-7 saturation).

Quality as-run: 22.8% divergences, max R-hat 1.82, min ESS 7. Cause: the
demo sampled in CONSTRAINED space — the production cure for exactly this
(unconstrained bijection reparam + mass adaptation, 44%->0.33% divergences,
memory project_reparam_wall_fix_status) was not ported into the batched
path. Evidence it is the same pathology: divergences concentrate at the
near-edge-on pair (cosi 0.085: 72-75%) and half the fits are <10%;
recovery is NOT broken — mean|pull| vs truth 0.5-1.2 per fit (|pull| ~0.8
expected for unit-scale), shear pulls g1 +0.13 rms 1.06 / g2 +0.11 rms
0.57, no wrong modes visible.

Also: the old vista A/B run dir failed spec validation under current code
(observation.snr rejected post-flux-units) — those solo results are
OLD-noise-convention fits and are NOT valid parity baselines. Parity needs
a small current-code solo baseline run.

Demo v2 punch list:
1. Port the unconstrained transform (kl_pipe/sampling/transforms.py) into
   the shared posterior; per-galaxy bound params join the dynamic pytree.
2. Current-code solo baselines: ~4 production-path fits on the fresh
   manifest (one gh-dev session) for the parity gate.
3. After reparam, re-measure straggler/tree stats (walls gone -> shorter
   trees expected) and revisit depth cap.
Data pulled to runs/vista_demo_v1/ (worktree, untracked).

## Vista demo v2 (2026-08-01) — reparam cure CONFIRMED; residual = draw count, not pathology

Same cost as v1 (4.1 min/fit, 14.5 fits/node-hr). Reparam effect:
divergences 22.8% -> 3.6%, max R-hat 1.82 -> 1.33, edge-on pair cured
(75% -> 2-6%). Recovery clean: shear pulls g1 +0.13 rms 1.15 / g2 +0.03
rms 0.60, no wrong modes.

Production gates (rhat<1.05, ESS>=50): 7/16 pass. ALL failures are
marginal (rhat 1.08-1.33, ESS 11-47) at 4x300 draws vs production's
~1000-draw chains -- an ESS-budget miss, not a sampler failure. Trees:
lane-mean 57 steps/iter but lockstep-max saturates the depth-7 cap on
100% of iterations (straggler x2.25) -- the cap is binding; depth vs ESS
tradeoff unresolved.

Next step ruled cheapest-first: KLPIPE_NUTS_SAMPLES=600 (zero code, ~1.6 h,
fits gh-dev): ESS doubles -> expect most marginals to pass. Then, in order
of expected value: Laplace-metric warmup init (shared-program host MAP
~6 s/galaxy), batched retry pass for residual failures (the production
escalation pattern), depth-cap A/B.

## Vista demo v3 (2026-08-01, samples 600) — production-parity first-pass quality CONFIRMED

ESS scaled exactly as predicted (median x2.10 vs predicted ~x2; the
draw-budget hypothesis is SUPPORTED). Pass rate 12/16 at production gates
(rhat<1.05, ESS>=50) at 5.8 min/fit = 10.4 fits/node-hr -- statistically
consistent with production's ~70% solo first-pass escalation rate. The 4
failures: two hair's-breadth (rhat 1.057), one genuinely hard GALAXY (both
ring members of z=1.37/cosi=0.238: rhat 1.15-1.20, ESS 16-20, sublinear
ESS scaling x1.4-1.5 -- a window-adaptation-resistant posterior; production
would likely escalate it too), one marginal (fit 15 rhat 1.057).

Bottom line vs solo baseline: ~3x per-fit (5.8 vs 16-18 min) and ~1.7x
node throughput vs today's effective ~6 fits/node-hr, at B=16 with
production-parity quality. Headroom not yet banked: batched retry pass
(the escalation analog, handles the 4/16), Laplace-metric warmup init,
B=32-64 fits/batch (amortization plateau is at B>=64 LANES; we ran 64
lanes but only 16 distinct fits), depth-cap tuning, fp32.

## Local audit + robustness wave (2026-08-02, CPU only)

Independent re-analysis of the v1-v3 draws (from-scratch split-Rhat/ESS,
scripts in the session scratchpad; committed artifacts below): every v3
headline number reproduces exactly (12/16 with the same four fits, ESS
scaling x2.0, shear pull rms 1.10/0.58, zero wrong modes by both a pull
screen and a chain-disagreement screen, straggler x2.25). New findings:

1. The straggler tax is CONCENTRATED, not diffuse: fit 01302235 hits the
   depth-7 cap on 71% of its iterations, three more fits sit at 31-36%,
   nine of sixteen below 10%. A handful of hot fits cause essentially the
   whole x2.25.
2. Capped iterations are the HEALTHIEST (acceptance 0.91 vs 0.83,
   divergence ~0% vs 4.1%): the cap truncates productive trajectories,
   not waste. A cap-6 would additionally truncate <1% of naturally
   terminating trajectories; cap-5 ~49%. So cap-6 is a gentle cut, cap-5
   an aggressive one -- and "cap frees wasted compute" is the wrong
   framing. Only a real cap-6 run settles ESS-per-leapfrog.
3. Difficulty is NOT predictable from catalog columns: ring pairs with
   identical cosi/z/SNR/shear differ in cap-hit fraction by up to 0.62
   (orientation + noise realization drive it); no property survives
   multiple-comparison correction at N=16. REFUTES pre-run stratified
   packing. BUT the first 50 sampling iterations predict the remaining
   550 at spearman 0.990 -> probe-then-repack is the viable design. The
   observed hard/easy 2x32 split reclaims 46% of the actual-vs-ideal
   leapfrog gap (16.5% of total steps). analyze_packing.py + CSVs.
4. The hard galaxy (z=1.37/cosi=0.238 pair) mixes slowly with AGREEING
   chains (between-chain share of the variance estimate only 12-30%):
   metric quality, not multimodality -- supports the Laplace-metric
   warmup lever as the targeted fix.
5. Halfwidth pin measured directly (measure_halfwidth_pin.py; fit 12,
   native 23 vs pin 31 fine px): delta logL at truth exactly 0; at the
   sizing formula's own designed worst case ~1e-5; only a beyond-design
   double-tail theta (v0 AND vcirc at 6-sigma simultaneously) reaches
   6e-5 of line flux. The pin is safe for production; the earlier
   "exact" claim is corrected to "erfc-negligible, now measured".
6. BUG in generic code found along the way (kl_pipe/render.py
   _prior_upper/_prior_abs_max): any prior with high=None falls into the
   linear-Gaussian branch, so LogNormal returns log-space mu+6sigma
   (~6.2 "km/s" for vel.vcirc) instead of exp(mu+6sigma) ~497 km/s.
   Masked today because vel.v0's +/-1200 km/s dominates v_max, but a
   spec with a tight v0 and wide vcirc LogNormal could undersize the
   line window silently. Present on the main line too. Needs a main-line
   fix + ruling (changes window sizes for LogNormal banks); NOT fixed on
   this branch.

Robustness wave (this commit):
- broadband PSF (psf_data + kspace_psf_fft) now traced leaves in both
  extract_dyn variants and shared fns -- closes the folding-tier
  silent-wrongness channel (a z=1.2-crossing bank would have reused the
  template's kernels with no error).
- continuum-kernel galaxy-independence asserted across EVERY fit and
  roll (precompute_continuum_kernel), replacing the fit-0-vs-last spot
  check.
- prior STRUCTURE assert including non-numeric attributes (conditional
  parent names) and the PriorDict parent index (assert_prior_structure);
  the old check covered types + numeric attr sets only.
- per-fit dyn treedef/leaf-shape check that names the offending fit
  before jnp.stack can fail cryptically; init clip mask surfaced as a
  warning instead of discarded.
Parity re-verified after the wave: proto_v3 CPU 2-fit run H4 2.24e-15 /
H5 3.75e-14 PASS; demo driver CPU smoke (2 fits x 2 chains) end-to-end
PASS with all asserts active.

analyze_demo.py now scores with production's own estimators (numpyro
summary split_gelman_rubin + n_eff, replacing hand-rolled rhat + arviz
bulk ESS) and prints the per-fit gate table itself. v3 re-scored: STILL
12/16, same four fits (the hard pair's min ESS drops to ~9-18 under
n_eff; no gate flips) -- the headline is robust to the estimator.

Literature status checks: MAMS ships in blackjax stable as
adjusted_mclmc (1.6.2), so the probe is config-only, no PR wait. BPD
(arXiv:2604.22048) actually varies its depth cap 2-7 by phase (not a
fixed 5) and never states fp32; it reports the same lockstep straggler
effect. The nested-Rhat literature (arXiv:2110.13017) supports
many-short-chains diagnostics, not fewer-longer chains, and nothing
endorses 2-chain rhat gating (floor is 4) -> keep 4 chains/fit.

Still open: the solo-baseline parity gate -- runs/cosmos25_ab_bb32gr32
has population+manifest expanded but chains/ and results/ are EMPTY; the
~4 fresh current-code solo fits remain the blocking vista item.

## Production-honest init (2026-08-02, local) — map_laplace mode + campaign v4

User ruling: v1-v3 truth-init timing is not production-achievable (no MAP
phase, no MAP init, 600 draws vs production's 300+escalation, target 0.8
vs 0.9) and must not be quoted as such. Changes:

- map_init.py: per-galaxy MAP + Laplace metric mirroring
  InferenceTask.laplace_preconditioner exactly (multi-start scaled
  L-BFGS-B from prior draws, best-of-finite, fd Hessian, scale-aware
  eig_floor 1e-4) but evaluated through the ONE shared compiled program
  (per-fit value_and_grad with the galaxy's dyn as data) -- truth-free, no
  per-fit compilation.
- batched_nuts_demo.py: KLPIPE_INIT = map_laplace (default; fixed Laplace
  metric in eta space via transform_inverse_mass, production 1%-of-
  posterior-scale chain jitter, step-size-only dual-averaging warmup) |
  map_window (MAP init + dense window adaptation) | truth (v1-v3
  behavior). KLPIPE_TARGET_ACCEPT default 0.9 (production spec value;
  v1-v3 ran 0.8). KLPIPE_FIT_IDS = explicit fit subset for retry passes.
  MAP wall + metric condition numbers + eta MAPs saved to npz/meta; the
  fit-equivalent print now includes the MAP phase.
- configs/ensembles/cosmos25_batched_b32.yaml: 16 galaxies x 2 ring
  members = 32 fits (128 lanes), same seed/selection as the A/B arm.
- RUNBOOK_DEMO.md rewritten as the two-session v4 campaign (T1 honest
  B=16 / T3 solo parity fits via the production worker / T2 B=32 at
  production-matched 300 draws / T2r retry via KLPIPE_FIT_IDS).

CPU smoke (2 fits x 2 chains, MAP starts 2 maxiter 60): full map_laplace
path end-to-end PASS -- MAP converged, metric conditions 5e3/1e4 (floor
cap 1e4), per-lane DA step sizes distinct, finite draws, npz carries the
map block. Solo-vs-batched estimator parity already handled (numpyro
summary in analyze_demo).

## T1 first launch: shape assert fired — image PSF tracing corrected (2026-08-02)

The new per-fit dyn shape assert stopped T1 at fit 4a7b8f69 (galaxy 6,
z=0.946 — the bank's only sub-1.2 pair): its image psf_data PADDED shapes
(270/308) differ from the template's (440/512; z-tier grid sizing). v1-v3
silently served the template's kernels here — benign for values (v2 H1
1e-15) but exactly the reuse channel the assert exists to catch.

Resolution measured, not assumed: in this config the image render always
takes the k-space branch (render_image uses kspace_psf_fft and never
touches psf_data when it is set), and kspace_psf_fft is the SAME 192x192
grid for every fit with per-galaxy values. So the dynamic pytree now
carries kspace_psf_fft only; psf_data is excluded and set to None inside
the trace, with a loud extract_dyn assert that every image obs has
kspace_psf_fft (the real-space fallback cannot silently engage).

Validation: mixed-tier parity (template z=1.42 + fit z=0.946, 3 prior
thetas each, production InferenceTask reference): values <= 4.7e-14,
grads <= 3.6e-13, PASS. Demo-driver CPU smoke on exactly that pair via
KLPIPE_FIT_IDS (map_laplace, 2x2 lanes): end-to-end PASS.

## T1 (demo v4, 2026-08-02 vista): fixed-Laplace metric REFUTED at batch scale; parity + honest cost excellent

Numbers (16 fits, map_laplace, warmup 100 DA-only, 600 draws, accept 0.9):
map 103 s (6.4 s/fit, 4/4 starts converged every fit), warmup 819 s,
sampling 4505 s -> 5.65 min/fit HONEST (10.6 fits/node-hr). Solo
baselines (production worker, same bank): 9.5 min/fit, 2/2 first-pass.
Parity vs those solos: |z| median 0.28 max 1.78, width ratio 0.99
(0.85-1.07) -- batched posteriors match production.

BUT gates 3/16 (v3: 12/16). Per-chain forensics: acceptance healthy
(0.7-0.96), divergences ~2%, yet failing fits all sit at step size
0.01-0.02 vs 0.1-0.2 for passing ones -> the FIXED Laplace metric
mis-measures soft directions; DA compensates with tiny steps; chains
crawl (ESS ~2, rhat up to 3.4). Fits that were EASY under v3's window
adaptation are among the casualties (0aab15fc, c3911e44). This is the
same reason production dropped fixed-Laplace over degeneracies and runs
precondition + adapt_mass: true (spec). Fixed-metric map_laplace is
REFUTED for this posterior class; not a bug, a finding.

Silver linings: v3's window-adaptation-resistant galaxy (8cea733e) now
PASSES (rhat 1.035, ESS 79) and its ring partner improved 1.20->1.11 --
MAP init + Laplace-scale information helps exactly where window
adaptation failed. The two components are complementary.

Also: warmup XLA compile pathological on first build (17 min on one
fusion op, session 1) but persistent-cache hit on rerun; straggler
overhead x1.57 at accept 0.9 (lane-mean 81 vs cap 127).

NEXT (T1b): KLPIPE_INIT=map_window KLPIPE_NUTS_WARMUP=200 -- MAP init +
200-step dense window adaptation, the closest blackjax analog of
production's Laplace-init + adapt-mass w200. Hypothesis: >= v3's 12/16
at roughly v3's sampling cost, with the honest MAP phase included and
warmup halved vs v3 (400 -> 200 thanks to MAP-adjacent starts).
