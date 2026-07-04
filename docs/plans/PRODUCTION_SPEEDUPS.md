# Production Speedups -- Findings, Strategy, Decisions

Status: ACTIVE planning doc. Branch `se/speedups` (on top of `se/source-model`,
PR #53). Created 2026-07-02 from a six-way audit (codebase audits: fp32
dependence, sampler batching readiness, grism hot-path anatomy; empirical
flagship profile; external research: Apple-silicon GPU, TACC/HPC GPU + batched
MCMC). Companion to `docs/papers/kl-roman-pipeline/planning/COMPUTE.md`
(hardware inventory + benchmark plan, separate paper repo); this doc is the
pipeline-side source of truth for speedup work.

Context: the paper needs O(few 1e3 - 1.5e4) MCMC fits by ~Aug 30 2026; survey
scale is 10-30M fits. Per-fit cost today (flagship, 1 roll, CPU): ~8.5 min with
Laplace preconditioning. Canonical config (2 bands + 4-roll grism) projected
~25-45 min/fit CPU without the work below.

---

## 1. Measured baseline (2026-07-02, M3 Max 16-core, fp64, JAX 0.7.1 CPU)

Flagship config exactly as `tests/test_flagship.py`: 32x32 stamp @ 0.11"/pix,
spatial oversample 3 (96x96 fine grid), Nlam=25 coarse, spectral_oversample=15
(375 fine lambda), 1 line (Halpha) + flat continuum, 1 grism roll, Gaussian PSF
FWHM 0.18" (PSF FFT pad 192x192), 17 sampled params.

| component | compile ms | steady ms | share |
|---|---:|---:|---|
| log_posterior (total, fused) | 620 | 8.8 | 100% |
| grad(log_posterior) | 1623 | 44.9 | 5.1x primal |
| render_broadband (1 band) | 261 | 0.32 | ~4% |
| render_grism (1 roll) | 421 | 9.8 | ~all |
| -- build_cube | 96 | 5.1 | 58% of grism |
| -- PSF conv (vmap over 25 slices, 50 FFTs @192x192) | 44 | 3.9 | 45% of grism |
| -- disperse_cube (25-iter loop) | 294 | 0.19 | 2% |
| -- post-dispersion pixel response + bin | 19 | 0.09 | 1% |

Key facts:
- Single-thread == 16-thread on CPU (arrays too small for threading). Implies
  1-fit-per-core node packing on Stampede3 has no contention penalty.
- grad/primal = 5.1x (NOT 2-4x as previously assumed). The fine cube
  (96x96x375 fp64 ~= 28 MB/line) is retained for the backward pass.
- CORRECTION to `experiments/sweverett/flagship_speedup/speedup_ideas.md`
  (2026-06): build_cube is 58% of grism, NOT ~90%; per-slice PSF convolution is
  a co-equal 45%. Use the numbers above.
- Compile cost per (shape, config): ~0.6 s primal + ~1.6 s grad + NUTS kernel.
  Matters at 1e4+ fits if recompiled per galaxy (see Sec 3.2 item 3).
- Profiling script (untracked working notes):
  `experiments/sweverett/production_speedups/profile_flagship.py` (+ JSON).

Caveat: flagship wallclock is partly sampler-conditioning-bound (Laplace
preconditioner took 42.5 -> 8.5 min with zero forward-model changes), so
per-eval FLOP wins do not convert 1:1 at flagship scale. They dominate at
production scale (4 rolls, 2 bands, [NII] blend => FLOPs up ~5x) and on GPU.

---

## 2. Grism forward-model speedups (Tier A -- do first; helps every platform)

Ranked. Combined realistic estimate: ~5-10x per-eval before any GPU.

### A1. Convolve once after dispersion (kills the 45%)
Replace 25 per-slice PSF convolutions (50 FFTs @ 192x192) with ONE convolution
of the dispersed 2D image. Valid because dispersion (`dispersion.py:124-222`)
is a linear operator (throughput-weighted sum of bilinear sub-pixel shifts;
bilinear shift = convolution with a fixed triangle kernel) and convolution
commutes with it when the PSF is wavelength-independent across the cube.

DECISION (user, 2026-07-02): interested, but requires ROCK-SOLID math/physics
justification before implementation. Two separable claims to nail down:
1. Mathematical equivalence GIVEN a lambda-independent PSF: exact in the image
   interior; boundary (cval=0 edge handling) differs. Note the CURRENT code
   already uses a single shared `psf_data` for all slices (`source.py:219-223`,
   issue #51 defers lambda-dependent PSF), so A1 is a pure reordering of the
   existing model, not a new physics approximation. Deliverable: written
   derivation + equivalence test (old path vs new path) with tolerance set
   BEFORE implementation, incl. quantified edge effect vs stamp size.
2. Physical adequacy of lambda-independence over the relevant bandwidth: the
   single-emission-line cube spans ~25 coarse pix * 1.1 nm ~= 27 nm at ~1.31 um
   (~2% fractional bandwidth). Quantify Roman G150 PSF variation (size/shape
   moments) over that window from galsim.roman getPSF; if negligible at our
   accuracy floor, A1 is justified PER LINE. Multi-line configs (Halpha+[NII]
   blended: same window; [SII] at +30 nm: still ~5% total) may need per-line
   PSFs -- A1 generalizes to convolve-per-line-after-dispersion (disperse each
   line's cube, convolve each with its own PSF, sum), still O(N_lines) FFTs
   instead of O(Nlam).
Expected gain: grism leg 9.8 -> ~6 ms (1.6x); larger under grad; compounds
with A3 (per-roll cost collapses).

Implementation constraints (user, 2026-07-04): the per-slice convolution
pathway is KEPT as the general path (future lambda-varying PSFs); the shortcut
is an opt-in/configurable fast path. Justification deliverable = LaTeX
derivation document + empirical equivalence tests.

Edge-tolerance choice (agent, 2026-07-04, user may override): tie the
equivalence-test tolerance to `folding_threshold` rather than an ad-hoc
absolute flux fraction. Rationale: the commutation discrepancy is confined to
within one PSF support of the stamp boundary and is proportional to the
profile flux near that boundary -- the SAME flux class that
`folding_threshold` (default 5e-3) already budgets for k-space aliasing in
RenderConfig grid sizing. Tying to it (i) makes the new error provably the
same order as an error the pipeline already tolerates, (ii) auto-tightens if
a user requests higher rendering accuracy, (iii) avoids inventing a second
independent accuracy knob. Tradeoff: an absolute flux-fraction bound would be
simpler to state and independent of render settings, but is arbitrary and can
silently diverge from the actual rendering accuracy regime.

IMPLEMENTED (2026-07-04), user sign-off after independent cross-check of the
derivation (agent audit vs code + numerical verification with real pipeline
operators: interior-source A/B 5e-5 L1/Ftot; theta-gradient agreement ~2e-5).
- `RenderConfig.psf_mode`: 'post_dispersion' (DEFAULT) | 'per_slice'
  (reference path, retained for future lambda-varying PSFs + verification).
  Threads RenderConfig -> GrismObs -> render_grism -> likelihood ->
  InferenceTask.from_obs (per-roll mismatch guard), mirroring spectral_method.
- Variant: SEPARATE exact padded convolution of the dispersed image via the
  existing `convolve_fft` (sinc pixel-response pass untouched) -- 4 FFTs/roll
  vs 52, deliberately isolating the commutation change; full fusion into the
  sinc pass (2 FFTs) remains an optional micro-opt gated on the
  circular-vs-linear check (proof doc Sec 5).
- Validation protocol (proof doc Sec 6) executed: (1) image A/B grid
  (angles 0/30/90, FWHM 0.11/0.18/0.30, high-shift, high-velocity,
  continuum): measured C in [0.03, 1.33] (worst = high-shift stress, line
  near stamp edge); frozen C=2.0 in `tests/test_grism_psf_mode.py`;
  (2) gradient equivalence: max 1.4e-3 rel (frozen 5e-3); (3) seeded
  posterior A/B (SNR~50, 4 params, 300+300 NUTS, single roll 24x24): mean
  shifts cosi 0.003 / g1 0.070 / g2 0.006 / vcirc 0.018 sigma, widths within
  12%, divergences 1 vs 2 -- flagship-depth A/B deferred to GPU era per
  short-run constraint; (4) per_slice path asserted bit-identical to the
  manually-spelled pre-refactor chain.
- MEASURED GAIN (32x32/os=3/Nlam=25, single-roll grism chi2, M3 Max fp64):
  forward 7.15 -> 3.68 ms (1.94x); grad 24.7 -> 22.9 ms (1.08x ONLY).
  The backward pass is dominated by build_cube + disperse backward, not the
  per-slice PSF FFTs, so A1's NUTS-wallclock win is ~8% at this config
  despite the ~2x primal win. The "kills 45%" expectation was a
  primal-leg statement. CONSEQUENCE: A3 (share cube across rolls) and the
  cube/disperse backward path are now the dominant Tier-A levers for
  sampler wallclock.

### A2. erf-exact spectral integration (attacks the 58%)
Issue #52 prototype (`experiments/sweverett/erf_spectral_integration/`):
replace the dense Gaussian exp over 375 fine-lambda points per spatial pixel
(3.46M-element exp, the build_cube hot spot) with exact per-bin integrals via
erf at 26 bin edges. ~14x fewer transcendentals, eliminates the 28 MB fine-cube
intermediate (=> smaller backward pass; grad is 5.1x primal today), and REMOVES
the spectral_oversample=15 knob entirely while being MORE accurate (no osf
bias; osf was raised 5->15 for accuracy, see docs/oversampling_convergence.md).

Repo previously filed this "speed-neutral on CPU" -- that was forward-only.
MUST re-benchmark under grad (and later GPU) before accepting the old verdict.
Numerics change slightly (more accurate): expect small shifts in
grism/datacube test expectations; any tolerance change needs explicit user
sign-off per project rules.

Implementation constraint (user, 2026-07-04): do NOT remove the
spectral-oversampling pathway even if erf wins -- keep both selectable for
comparison tests; a default change requires user sign-off.

BENCHMARK RESULT (2026-07-04, experiments/sweverett/production_speedups/
erf_grad_bench/, flagship config, cube stage only):
- Accuracy: erf ~45x more accurate than osf=15 vs an osf=101 reference
  (1.1e-5 vs 5e-4 max rel err; similar at a high-velocity-gradient point).
- Speed vs osf=15 as-is: only 1.1-1.3x (forward 5.8 vs 5.2 ms; grad ~28 vs
  ~35 ms) -- the naive >=2x hypothesis REFUTED; XLA CPU erf kernel cost
  dominates (76% of erf variant).
- INITIAL "gradient pathology" claim, CORRECTED after script verification
  (2026-07-04): the agent's 77-111%-error / sign-flip numbers came from
  comparing chi2 gradients AT THE TRUTH POINT with mock data generated
  from the osf=15 model itself -- there both gradient vectors are
  near-zero (noise- and discretization-offset-dominated), so relative
  errors between them are not a valid "gradient wrongness" metric, and
  the "needs osf~201-401" figure inherits the same artifact. AD == FD
  (autodiff is fine). What the experiment DOES validly show: the osf=15
  and converged surfaces have slightly displaced optima (detectable
  against 1%-noise data), and erf gradients match the converged reference
  to ~9e-5.
- The REAL osf failure mode (verified, tests/test_spectral_methods.py
  washboard diagnostic): for NARROW lines, sigma_lambda < fine sub-bin
  (dispersion < ~17 km/s at osf=15; the flagship prior extends to 5 km/s),
  midpoint sampling under-resolves the line: VALUE error reaches ~5% of
  the peak voxel (measured at 10 km/s) oscillating with lam_obs at the
  fine-bin period, and gradients ripple accordingly. At the 50 km/s
  fiducial the error is the smooth ~5e-4 Euler-Maclaurin edge term --
  benign per-fit, but a coherent systematic across an ensemble (relevant
  at the m~1e-3 bias level when data comes from a different renderer).
  erf eliminates the entire axis (exact for every line width).
- MCMC-level impact of osf=15 vs erf: still UNTESTED (deferred with the
  flagship-run moratorium); the narrow-line value error is now the
  motivated mechanism, not "wrong gradients" generically.

### A3. Single model-space cube for all roll angles
DECISION (user, 2026-07-02): approved direction ("quite like").
DESIGN GRILLED + SIGNED OFF (user, 2026-07-04): full decision record below
(decisions 21-27). Implementation in progress this session.

**Backward-pass profile (2026-07-04, M3 Max fp64, 30 reps, perturbed-theta
eval; `experiments/sweverett/production_speedups/profile_backward_split.py`):**

| leg (1 roll, post_dispersion) | grad ms | share |
|---|---|---|
| build_cube | 28.2 | ~81% |
| disperse | 2.8 | ~8% |
| post-disp PSF conv | 1.6 | ~4% |
| pixel-response + chi2 | 2.5 | ~7% |

4-roll grad scales 3.99x (136.8 vs 34.3 ms); posterior grad ~= grism chi2
grad (broadband+prior negligible). **A3 ceiling (shared-cube timing sim,
real rotated-WCS obs at 0/45/90/135 deg): 2.06x on the 4-roll gradient**
(136.8 -> 66.4 ms), 3.40x forward -- NOT the naive ~4x: the shared cube's
backward accumulates 4 cotangent cubes (96x96x50 fp64) and loses XLA fusion
(~11 ms). Post-A3, build_cube's own backward (~28 ms) is ~60% of the 4-roll
gradient -- the next native lever lives INSIDE build_cube backward (erf bin
integral + intensity/velocity eval chain), not disperse/PSF.

**Mechanism (fused rotated sampling).** Sky fixed; detector rotates per roll;
Roman disperses along det +x always (`dispersion_angle_detector=0`, roll in
obs WCS). The unconvolved cube C(x_sky, y_sky, lambda) is roll-independent
by physics (LOS velocity is a scalar field on the sky; PA/shear/centroid are
sky-frame). Exact change of variables:

    dispersed_det_r(x) = sum_k C_sky(R_r (x - s_k)) T_k dlam,  s_k = det +x shift

`disperse_cube` already does one bilinear map_coordinates per lambda-slice at
`[Y-dy_k, X-dx_k]` -- rotating those pull-coordinates about the stamp center
costs ZERO extra interpolation passes and lands output directly in det frame.
Rejected alternatives: rotate dispersed 2D image (+1 interp pass/roll),
rotate cube slices (+Nlam passes/roll).

**PSF.** The det-frame PSF never touches the shared cube: A1's
post-dispersion convolution applies per roll, per obs kernel (PSFs MAY
differ across rolls -- sharing unaffected). Shared path REQUIRES
psf_mode='post_dispersion'; per_slice + shared group raises loudly.
Static-PSF validity remains per emission-line complex only (issue #51;
close blends like [NII]+Halpha in one window are fine). Near-term workflow:
one line complex per fit, combine posteriors post hoc (divide out shared-
param prior N-1 times). Designated issue-51 resolution path (deferred,
A3-compatible): per-line sub-cube dispersion + per-line static kernels
summed pre-pixel-response (A1 proof already has the per-line
generalization); full-bandpass continuum w/ continuously varying PSF is the
only genuinely long-term case.

**x0/y0.** Sky-frame convention: `_apply_obs_rotation` currently rotates
theta_int + (g1,g2) but NOT (x0,y0) -- physically wrong at rotation != 0
(a fixed sky offset appears rotated in each roll's det frame); latent
because all current usage has rotation=0. Pre-A3 fix commits the rotation;
shared path then agrees by construction (offset baked into sky cube).
Per-exposure astrometric nuisance offsets (det-frame delta_e) DEFERRED:
Roman per-exposure residuals ~mas vs 110 mas pixels; if ever needed they
compose as a translation of the sampling coords -- zero cube rebuild, +2
params/exposure. Open science question: quantify residual vs shear-bias
budget before survey scale.

**API.** User contract unchanged: pass more grism obs to
`InferenceTask.from_obs(grism_obs={...})` like broadband. New public
`SourceModel.render_grism_group(pars, obs_group) -> dict`: build sky cube
once, per roll rotated-disperse + PSF + pixel response. from_obs groups
grism obs by cube-compat key (identical lambda_grid + shape/scale +
oversample + spectral_method; WCS rotation/data/variance/PSF excluded);
different lines -> different groups -> own cubes (correct, not an error).
`render_grism(pars, obs)` untouched as single-obs/reference path.
Knob: `RenderConfig.cube_mode` = 'shared' (default) | 'per_roll' (reference),
mirroring spectral_method/psf_mode plumbing.

**Guards (all loudly unit-tested):** per_slice + shared group; flipped WCS
(det(PC) < 0 -- parity is not a rotation; image_rotation_from_wcs's +pi
convention is unsound for it); near-miss lambda grids (same shape, tiny rel
diff -> "float drift? make identical"); cube_mode mismatch across a group.

**Tolerance model (two terms, both measure-then-freeze per A1 protocol):**
(i) rotated-corner/edge truncation -- C x folding_threshold budget, C frozen
after measuring the A/B grid; (ii) bilinear interpolation error at rotated
(both-axes-fractional) coords -- NOT folding-bounded; separate
convergence-vs-oversample test, tolerance frozen at production oversample=3.
Gradient equivalence on vector norms at perturbed theta (both A1 traps
apply). Padding: implement UNPADDED first, measure 45-deg worst case, add
sky-grid padding only if the budget is exceeded (minimal sufficient
complexity).

### A4. Precomputed fixed-dispersion gather operator (GPU enabler)
The 25-iter Python disperse loop's geometry (pixel offsets, dx/dy) is static
per obs. Bake into one sparse gather/matmul applied to the flattened cube.
Negligible CPU win (disperse is 2%) but turns the whole grism path into dense
XLA ops -- do it as part of the GPU port, not before. In-code note at
`dispersion.py:200-205`: vmap/scan variants were 2.5x SLOWER on CPU (keep the
loop on CPU); GPU untested.

---

## 3. GPU production path (Tier B -- TACC Vista GH200)

DECISION (user, 2026-07-02): TACC GPU node setup is the production pathway and
comes BEFORE any macbook-GPU work; grism speedups (Tier A) first, but do not
lose sight of GPU gains. User has TACC access NOW; start small-scale tests,
no overboard spending until confident.

### 3.1 Deployment (solved problem)
- Vista GH nodes: 1x GH200 (96 GB HBM3) + 72-core Grace ARM host per node;
  `gh` queue 1 SU/node-hr; aarch64 => existing x86 conda-lock env unusable.
- Primary: NGC JAX container `nvcr.io/nvidia/jax:<pinned-tag>` (arm64) via
  `module load tacc-apptainer`, `apptainer exec --nv`. Fallback: venv +
  `pip install "jax[cuda12]"` (official aarch64 wheels exist) + pip numpyro.
  NEVER build jaxlib from source on GH200 (known failure history).
- Keep the GPU fit env minimal: jax + numpyro (+ blackjax only if Sec 3.3
  benchmarks force it). GalSim/synthetic-data deps stay off the GPU env
  (generate mocks elsewhere or in a separate CPU env).
- Memory: tune `XLA_PYTHON_CLIENT_MEM_FRACTION`; 96 GB HBM is the budget
  (JAX does not auto-use Grace unified memory).

### 3.2 Codebase blockers, ranked smallest-first (from batching audit)
1. Progress bar off in batched paths (host callback breaks vmap) -- config
   already supports it (`numpyro.py:515,656`).
2. Pin ONE shared RenderConfig across the ensemble: `InferenceTask.from_obs`
   re-derives RenderConfig per task from priors (`task.py:171-224`); any
   derived-shape drift across galaxies breaks vmap and forces recompiles.
3. KEYSTONE: data-as-argument likelihood. Data/variance/mask are currently
   baked into the jit closure via `partial` (`likelihood.py:195-206`) => every
   galaxy re-traces + recompiles likelihood, grad, AND NUTS kernel (~several s
   each). Refactor: obs pytree splits into static aux (grids, psf_data,
   RenderConfig, Nlam, sampled_names) closed over vs dynamic leaves (data,
   variance, mask, varying prior loc/scale) passed as a small pytree arg.
   One compiled `potential_fn(theta, data)` reused for all galaxies; vmap over
   the leading batch dim. The preconditioned numpyro path already builds a
   bare potential_fn (`numpyro.py:621`) -- use as template.
4. JAX-native Laplace preconditioner: current impl is host-side scipy
   L-BFGS-B multi-start + numpy eigh/inv (`task.py:552-670`) -- not
   vmappable. Replace with optax/jax.scipy MAP + `vmap(jax.hessian)`.
   DECISION (user): no backend preference IF equal-or-better accuracy AND
   speed -- gate the swap on a head-to-head (same multi-start protocol,
   compare MAP quality, Hessian conditioning, wallclock, and downstream
   sampler ESS/wallclock on ~10 galaxies).
5. Result storage. DECISION (user): keep FULL chains until we are very
   confident; reduce detail only for full-scale paper-plot runs.
   ~1.3 MB/fit at defaults => ~13 GB per 1e4 fits: fine on disk, not in RAM.
   Stream per-galaxy results to disk as they complete (npz/parquet shards);
   an ensemble aggregator reads summaries.
6. Real-space Spergel uses `jax.pure_callback` (host round-trip,
   non-differentiable; `intensity.py:796-801`) -- must never appear in the
   batched hot path. K-space inference path is safe (analytic FT). Add a loud
   guard in the batched driver.

### 3.3 Sampler / batching strategy
DECISION (user, 2026-07-02): numpyro stays the first-class backend; blackjax
only if a specific measured speed/GPU reason emerges (prior mixed results).

Architecture (two-tier, embarrassingly parallel):
- Across GPUs: SLURM job array, 1 process = 1 node = 1 GPU = 1 galaxy shard.
  No jax.distributed / NCCL. Restartable per shard.
- Within a GPU: batch galaxies via the data-as-argument potential_fn.
  Numpyro-first implementation ladder:
  a. Baseline: sequential per-galaxy numpyro NUTS on GPU (no batching) --
     already works once fp64/closure issues fixed; measures raw GPU-vs-CPU
     per-fit gain. (Precedent: geko -- our grism validation reference -- runs
     per-galaxy numpyro NUTS on GPU for JWST grism kinematics,
     arXiv:2510.07369.)
  b. `chain_method='vectorized'` with chains = (galaxies x chains) via a
     batch-aware potential_fn; progress bar off; cap `max_tree_depth` (6-8)
     to bound the slowest-chain sync penalty (COMPUTE.md L1).
  c. Only if (b) leaves large measured idle fractions: hand-rolled
     vmap(warmup+sample) loop (blackjax or numpyro internals), and/or
     fixed-length samplers (ChEES-HMC / adjusted MCLMC, blackjax >= 1.5 only
     -- MEADS formula bug fixed in 1.5) benchmarked against (b). Literature:
     fixed-length samplers win 2-3x at high batch width; our Laplace
     preconditioner already plays the role of their cross-chain adaptation.
- Bucket galaxies by expected difficulty (SNR, cosi near edge-on) per batch to
  reduce slowest-lane waste; measure trajectory-length variance FIRST (it
  decides whether (b) suffices).
- Failure handling at scale: per-galaxy divergence/R-hat/ESS thresholds ->
  flag-and-refit queue (CPU/fp64 fallback), never silent. One bad batch member
  must not contaminate ensemble diagnostics.

### 3.4 First TACC milestone (small-scale, uses existing allocation)
1. Vista dev-queue smoke test: NGC container + flagship single fit on GPU
   (fp64 first -- GH200 fp64 is strong, 34 TF, so fp64 GPU is only ~2x slower
   than fp32, not a blocker for the pilot).
2. COMPUTE.md Sec 4 benchmark matrix, reduced: per-fit wallclock CPU(SPR core)
   vs GH200; Laplace on/off; then batched (b) at batch {1, 32, 256}.
3. Feed numbers back into COMPUTE.md Sec 3 and the paper Sec 8 plan.

---

## 4. fp32 plan (Tier C -- gates GPU throughput and any Metal experiment)

Audit summary: the switch is small and well-defined.
- x64 forced at import in exactly 3 modules: `coordinates.py:30`,
  `source.py:29`, `lines.py:28`. Centralize in the (currently empty)
  `kl_pipe/__init__.py` behind an env flag (e.g. `KL_PIPE_X64`, default ON --
  fp64 stays default), delete the scattered calls.
- Relax the hard guard `psf.py:258-262` (gate on the same flag).
- One true in-graph fp64 pin to fix: `dispersion.py:181-182`
  (`jnp.arange(..., dtype=jnp.float64)`).
- Host-side numpy fp64 (Laplace eigh/inv, numpy PSF path, TNG loaders) stays
  fp64 -- protective, free.
- Cheap hardening regardless of mode: fp64 accumulation of the chi2/log_det
  reductions (`likelihood.py:82-93`) -- one cast, guards the ~1e4 intensity/
  velocity dynamic range and the fp32 MH-accept precision problem.
- Known fp32-sensitive spots to watch: `gammaln` flux normalization overflow
  (`intensity.py:864,1096`), u/sinh Taylor crossover thresholds
  (`intensity.py:576-584`), log-space Spergel kve arithmetic.

Acceptance criteria -- DECISION (user, 2026-07-02): the requirement is on
LIKELIHOOD-SLICE and SAMPLER RECOVERY (particularly but not exclusively
shear), NOT max-pixel residuals vs GalSim (those tests are defensive; expect
the ~7 exact-FFT-identity test files pinned at 1e-10..1e-15 to need
fp32-conditional tolerances, which is acceptable BY DESIGN here, with the
change flagged loudly per project rules). Concretely, explore the contour:
1. Likelihood-slice tier passes at fp32 within EXISTING tolerances (no
   loosening) -- this is the hard gate.
2. Sampler A/B on N~10-20 representative galaxies (span SNR, cosi incl. near
   edge-on): fp32 vs fp64 posterior means/widths for (g1, g2) within a small
   fraction of the posterior sigma (propose |dmu| < 0.1 sigma, widths within
   ~5%; finalize with user); divergence rate and R-hat/ESS not degraded.
3. Map WHERE fp32 breaks (if anywhere): scan SNR up + stamp size up until
   criteria fail; document the boundary. Mitigations in order: constant-term
   subtraction in logL, fp64 reductions, Kahan summation.

---

## 5. Apple-silicon local GPU (Tier D -- optional, local dev only)

DECISION (user, 2026-07-02): deprioritized below TACC; local-dev convenience
only.
- jax-metal is DEAD (last release 2024-10, no FFT/complex/fp64, segfaults).
  Do not touch.
- Do NOT port to MLX / mlxmc (github.com/jrcheshire/mlxmc): credible young
  sampler library, but the sampler is the cheap part -- porting the JAX
  forward model to MLX (fp32-only, dual codebase, full revalidation) is not
  sane.
- IF fp32 mode (Sec 4) lands: half-day throwaway-venv experiment with a
  community JAX Metal plugin -- `applejax` (claims FFT/complex + numpyro NUTS
  tested; jaxlib 0.9.x pin) or `jax-mps` (MLX-backed, active, ~95% upstream
  test pass; jaxlib 0.10.x pin). Gate: FFT ops run, no divergence/R-hat
  regressions vs fp64 CPU, wallclock beats the 16 M-series cores. Expect <=3x
  best case; possibly <2x or a loss (small FFTs, launch-bound). Skip without
  regret if it disappoints.

---

## 6. Decisions log

2026-07-02 (user):
1. Order: core grism speedups (Tier A) first -- helps all platforms; GPU
   production path (Tier B) is real and must not stall; TACC before macbook.
2. A1 convolve-after-dispersion: pursue ONLY with rock-solid math/physics
   justification (see A1 deliverables); PSF constancy is per single-line cube
   bandwidth, not full spectrum.
3. A3 single model-space cube for all rolls: approved direction.
4. fp32 acceptance: likelihood-slice + sampler recovery (esp. shear) is the
   requirement; GalSim pixel-residual tests are defensive; explore the exact
   contour.
5. Samplers: numpyro remains first-class; NO blackjax as first-class backend
   absent a specific measured speed/GPU reason.
6. Laplace backend: swap to JAX-native acceptable iff equal-or-better accuracy
   AND speed, head-to-head verified.
7. TACC: access exists now; start small-scale; scale only when confident.
8. Branching: this work on `se/speedups` (new, on ~final `se/source-model`)
   while PR #53 is in review; rebase over review commits as needed.
9. Chains: keep full sampler chains until very confident; reduced storage only
   for full-scale paper-plot runs.

2026-07-04 (user):
10. PSF lambda-dependence: PSF IS lambda-dependent physically; the question is
    only whether constancy across a SINGLE line matters at the ~0.1 sigma
    parameter-shift level (doubted; estimate from basic physics). The current
    per-slice-convolution pathway must be KEPT (not removed) when the A1
    shortcut lands, so lambda-varying PSFs can be implemented later. Shortcut
    = fast path; per-slice = general path.
11. A1 justification: empirical tests PLUS a LaTeX document deriving and
    proving the commutation mathematically.
12. A2 erf: pursue; do NOT remove the spectral-oversampling pathway even if
    slower/worse -- keep both for comparison tests; default may change later
    with user sign-off.
13. fp32: implement as a high-level configuration choice (user-selectable per
    tolerance needs). Required documentation of impact: (a) pixel-level
    residuals vs GalSim for common cases; (b) likelihood-slice deltas;
    (c) posterior differences (bias, sigma, MAP deltas) on the seeded flagship
    test. Thresholds decided later; target at least < 0.1 sigma on shear.
14. A1 edge tolerance: agent chooses the most-justified option now (chosen:
    tie to folding_threshold -- see A1 section rationale), user may swap
    later. Do not delay initial work on this.
15. SU budgets: out of scope for now; focus on speedups + implementations for
    the user to test.
16. max_tree_depth: no user intuition; decide via ESS/wallclock A/B when the
    batched path exists.
17. Chain storage: no format preference; pick for disk/I-O efficiency. No
    significant quota constraints. (Chosen: one compressed .npz per fit,
    directory-sharded, + a single parquet summary table per ensemble; no new
    heavy deps, streamable, trivially restartable.)
18. (2026-07-04, user) A1 SIGNED OFF after independent cross-check of the
    derivation; implement with post-dispersion convolution as DEFAULT,
    per-slice retained (future evolving PSF + verification), with clear unit
    tests + visual diagnostics. Agent choices accepted by default (user AFK
    at ask): variant = separate padded conv (not fused into sinc pass);
    knob = `RenderConfig.psf_mode` ('post_dispersion'|'per_slice').
19. (2026-07-04, user) B keystone (data-as-argument likelihood) DEFERRED:
    near-term target is per-galaxy end-to-end sampler wallclock (local +
    single CPU/GPU node), not batched mass-fitting. Per-galaxy compile
    (~seconds) is acceptable at the 1e2-1e4 fit scale; galaxies are not
    guaranteed shape/config-uniform anyway. Revisit for the ~30M-scale
    production runs. (Supporting measurement: single fit is not helped by
    multi-core CPU -- node throughput = N independent fits in parallel,
    which needs no batching refactor.)
20. (2026-07-04, user) Local run budget: SHORT runs only (<5-10 min
    wallclock) until GPU-era hardware -- battery constraints. Validation
    via likelihood slices / optimizer / minimal-sampler configs is fine;
    flagship-depth runs are not.
21. (2026-07-04, user, A3 grill) Mechanism: fuse the roll rotation into
    disperse_cube's pull-coordinates (sample sky-frame cube at
    R(x - s_k)); output directly det-frame with dispersion along det +x.
    Requires rock-solid unit tests + visual diagnostics: regression vs
    current path first, standalone multi-roll-of-one-sky-model
    verification by the end.
22. (2026-07-04, user, A3 grill) Shared lambda grid: all rolls of a group
    must have identical CubePars ("no easy way otherwise"); enforce loudly.
23. (2026-07-04, user, A3 grill) x0/y0 are sky-frame; commit the pre-A3
    _apply_obs_rotation fix rotating them. Short term: single sky-frame
    offset models static astrometric error in all cutouts; per-exposure
    det-frame nuisances acknowledged as a possible long-term need
    (extension path recorded in A3 section, composes w/o cube rebuild).
24. (2026-07-04, user, A3 grill) API: keep the broadband-like contract --
    "just pass more grism obs"; sharing is an invisible internal
    optimization (render_grism_group + from_obs auto-grouping).
25. (2026-07-04, user, A3 grill) per_slice psf_mode with a shared group:
    loud error. cube_mode default = 'shared'. WCS/rotation guards at
    author's discretion but all rigorously unit-tested.
26. (2026-07-04, user, A3 grill) Multi-line static-PSF limitation (issue
    #51) stays out of A3 scope: near-term = one line complex per fit,
    combine chains post hoc; tutorial/docs warning REQUIRED at the
    end-of-branch doc pass (TODO Sec 7). Per-line post-dispersion kernels
    recorded as the designated future resolution.
27. (2026-07-04, user, A3 grill) Tolerances: A1-style measure-then-freeze;
    two-term budget (truncation C x folding_threshold + interpolation
    convergence-vs-oversample). Implement unpadded first; padding only if
    the measured 45-deg budget demands it.

## 7. TODO (rank-ordered)

- [x] A1 justification: derivation note + PSF-vs-lambda variation quantified
      over the line window (galsim.roman getPSF moments); equivalence-test
      spec w/ tolerances BEFORE implementation. (Proof doc committed;
      independent cross-check 2026-07-04: math CONFIRMED, 3 doc corrections
      applied -- pad-size figure, kernel-negativity caveat, m-estimate
      convention note; streamlined-argument section added.)
- [x] A2 re-benchmark erf integration under jax.grad (2026-07-04; see A2
      section: 45x accuracy win, 1.1-1.3x vs osf=15 as-is, 7-19x vs the
      osf~201-401 actually needed for correct gradients; osf=15 gradient
      pathology discovered).
- [x] A2 IMPLEMENTED as default (f8eb3d0): spectral_method='erf' throughout
      (RenderConfig -> obs -> render_grism/build_cube -> likelihood -> task);
      'oversample' retained + explicitly selectable; 13 tests incl. washboard
      visual diagnostic (tests/test_spectral_methods.py); test-basic 820 pass.
- [ ] A2-FOLLOW-UP (deferred w/ flagship-run moratorium; GPU first): seeded
      flagship NUTS erf vs osf=15 A/B -- posterior deltas (esp. shear), ESS,
      divergences. Motivated mechanism = narrow-line value error, NOT the
      retracted "wrong gradients" claim.
- [x] A1 implementation behind equivalence tests (per-slice path stays as
      the general/lambda-varying-PSF path); re-profiled. See A1 section
      IMPLEMENTED block: primal 1.94x, grad 1.08x -- grad bottleneck is
      build_cube/disperse backward, raising A3's priority.
- [x] Backward-pass profile split (2026-07-04; see A3 section): build_cube
      backward = ~81% of grism grad; 4-roll scales 3.99x; A3 measured
      ceiling 2.06x on 4-roll grad (cotangent-accumulation overhead eats
      the naive 4x). Next post-A3 native lever = build_cube backward.
- [ ] A3 shared-cube-across-rolls refactor: design GRILLED + signed off
      (decisions 21-27; full record in A3 section). Implementation in
      progress: pre-A3 x0/y0 rotation fix -> disperse_cube rotation kwarg
      -> RenderConfig.cube_mode -> render_grism_group + from_obs grouping
      -> two-term-tolerance equivalence/gradient/posterior tests + visual
      diagnostics -> re-profile.
- [ ] A3-FOLLOW-UP (end-of-branch doc pass, DO NOT FORGET): tutorial +
      docs warning that the static-PSF grism pathway is valid per emission
      line complex only (issue #51); recommended workflow for now = separate
      per-line posterior fits, combine chains post hoc (divide out the
      shared-param prior N-1 times when multiplying posteriors).
- [ ] (DEFERRED, decision 19 -- revisit at ~30M scale) B: RenderConfig
      pinning + data-as-argument likelihood refactor
      (potential_fn(theta, data)); kill per-galaxy recompilation.
- [ ] B: Vista smoke test (NGC container, flagship fit, fp64) -> reduced
      benchmark matrix -> update COMPUTE.md numbers.
- [ ] B: numpyro vectorized batched path w/ max_tree_depth cap; measure
      trajectory-length variance + idle fraction.
- [ ] B: JAX-native Laplace head-to-head (accuracy + speed gate).
- [ ] C: fp32 env-flag mode + fp64 reductions; acceptance protocol Sec 4.
- [ ] D (optional): applejax/jax-mps half-day trial after C.
- [ ] Ensemble driver + calibration module (paper P0; STRATEGY.md Sec 10) --
      design batch-first: consumes the data-as-argument API from day one.

## 8. Open questions

(1-5 of 2026-07-02 resolved -- see decisions 13-17.)
1. fp32 exact thresholds: deferred until the impact documentation (decision
   13 a/b/c) exists; floor = < 0.1 sigma on shear.
2. max_tree_depth for batched runs: decide via ESS/wallclock A/B (decision 16).
3. Lambda-varying PSF API shape (per-slice PSF list? callable psf(lambda)?)
   -- design when the general path is revisited; A1 must not preclude it.

## 9. Pointers

- Paper-side compute doc: `docs/papers/kl-roman-pipeline/planning/COMPUTE.md`
  (hardware tables, SU rates, lessons L1-L5, benchmark plan).
- Stale-but-useful: `experiments/sweverett/flagship_speedup/speedup_ideas.md`
  (sampler-side ladder; its build_cube 90% figure is superseded by Sec 1).
- erf prototype: `experiments/sweverett/erf_spectral_integration/` (issue #52).
- Lambda-dependent PSF deferral: issue #51 (`source.py:219-223`).
- GPU-revisit negative results (CPU-only verdicts): memory
  `project_gpu_speedup_todo.md`; `dispersion.py:200-205` comment.
- Profiling script + raw numbers (untracked):
  `experiments/sweverett/production_speedups/`.
