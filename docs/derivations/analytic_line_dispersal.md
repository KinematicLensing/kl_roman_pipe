# Analytic per-spaxel line dispersal: derivation and design audit

Status: 2026-07-06. Derivation validated by standalone prototype (section
11); literature search found no prior art (section 12); pipeline
implementation pending. Companion to the wavelength-grid production rule
and the two-tier grid / analytic dispersal follow-ups in
docs/plans/PRODUCTION_SPEEDUPS.md (section 7).

## 1. Problem statement

The grism forward model builds an (x, y, lambda) cube and disperses it by
rigidly shifting each wavelength slice along the dispersion axis
(kl_pipe/source.py `_build_cube_impl`, kl_pipe/dispersion.py
`disperse_cube`). Accuracy demands a dense wavelength grid: the production
rule is slice_width_kms = 40 (n_lambda ~ 151), because each slice deposits
at exactly one detector offset and a coarse grid quantizes the
velocity-to-position mapping of a rotating disk (the entanglement bias
frozen in tests/test_pixel_readout.py).

Measured cost (M3 Max fp64, flagship config, ms per value_and_grad, from
experiments/sweverett/production_speedups/results_decision_grid.json):

| config          | grad ms | note                          |
|-----------------|---------|-------------------------------|
| n=25,  os=3     | 11.9    | old cheap default (biased)    |
| n=151, os=3     | 37.5    | slice quantization fixed      |
| n=25,  os=5     | 30.2    | interpolation bias fixed      |
| n=151, os=5     | 96.7    | production-safe               |

The n_lambda axis alone is a ~3.2x multiplier; combined with the
oversample floor it is ~8x over the old cheap default. This document
derives a way to remove the n_lambda axis for emission lines entirely.

## 2. The current method, formalized

Notation. Fine-grid spatial samples sit at integer positions q = (q_x, q_y)
in fine pixels (oversample `os` fine pixels per coarse pixel). The
dispersion shift for wavelength lambda is, in fine pixels along the
detector dispersion direction d_hat = (cos a, sin a):

    s(lambda) = (lambda - lambda_ref) / D * os,        D = nm per coarse pixel

For one emission line, the cube voxel at spaxel j and slice k is

    C_jk = I_j * K_k(lambda_obs_j, sigma_j)

where I_j is the line's spatial surface brightness at spaxel j,
lambda_obs_j = lambda_rest (1+z)(1 + v_los_j / c) the Doppler-shifted
center, sigma_j = lambda_obs_j sigma_v / c the intrinsic width, and K_k the
exact bin average of the normalized Gaussian over slice k's wavelength bin
(the erf spectral method; kl_pipe/source.py). The dispersed image is

    Dsp(q) = sum_k T_k w_k dlam * sum_j C_jk * tent(q - s_k d_hat - x_j)

with T the throughput, w_k the trapezoid endpoint weights, and tent the
bilinear interpolation kernel of `map_coordinates(order=1)`,
tent(u, v) = tri(u) tri(v), tri the unit triangle.

Because K_k is an exact bin integral, the only lambda-discretization error
is that the deposit position is frozen at s_k across a bin of width
ds = os * dlam / D fine pixels instead of sliding continuously. At n=151
(flagship window ~26.2 nm), ds = 0.79 fine px; at the n=25 default,
ds = 5 fine px (one full coarse pixel) -- hence the entanglement bias. The
n ~ 151 rule is exactly the requirement "position quantization below one
fine pixel."

## 3. The n_lambda -> infinity limit is closed-form per spaxel

Take the continuum limit of the slice sum (the exact target that n=251
references approximate). The trapezoid sum converges to

    Dsp(q) = sum_j I_j * Integral dlam T(lambda) N(lambda; lambda_obs_j, sigma_j)
                          * tent(q - s(lambda) d_hat - x_j)

with N the normalized Gaussian. Change variables to the deposit coordinate
xi = s(lambda) (linear in lambda, so exactly a Gaussian in xi):

    Dsp(q) = sum_j I_j * Integral dxi T(lambda(xi))
                          * N(xi; xi_j, sigma_s_j) * tent(q - xi d_hat - x_j)

where

    xi_j      = s(lambda_obs_j)          (deposit center, fine px)
    sigma_s_j = sigma_j * os / D         (deposit width, fine px)

This is, per spaxel, a 1D Gaussian convolved with the interpolation kernel
along the dispersion direction -- and that convolution is closed-form.

### 3.1 Axis-aligned dispersion (single-roll path, a = 0)

Roman G150 disperses along detector +x, so tent separates:

    Dsp(q) = sum_j I_j * T_j * tri(q_y - y_j) * [N conv tri](q_x - x_j - xi_j)

tri is the convolution of two unit boxes, and a Gaussian convolved with a
box is an erf difference, so with Phi the standard normal CDF,
phi the standard normal PDF, and Psi(z) = z Phi(z) + phi(z)
(the antiderivative of Phi):

    [N_sigma conv tri](u) = sigma * [ Psi((u+1)/sigma) - 2 Psi(u/sigma)
                                      + Psi((u-1)/sigma) ]

Each spaxel therefore deposits a closed-form profile (erf + exp terms)
onto the fine pixels of its own row, within a window of
+/- (4 sigma_s + 1) fine pixels around q_x = x_j + xi_j. No wavelength
grid appears anywhere. tri(q_y - y_j) is nonzero only at q_y = y_j on the
integer grid, so the deposit is exactly one row wide.

Two properties worth stating explicitly:

- This is exactly the current model at n_lambda -> infinity. Every other
  ingredient -- tent interpolation, spatial oversampling, post-dispersion
  PSF, sinc pixel response, center-sample readout -- is untouched. The
  method removes one discretization axis; it does not change the model.
- Velocity gradients are handled with no extra machinery. The apparent
  compression or stretching of the line profile where v_los varies along
  the dispersion direction (space-velocity entanglement) emerges from the
  sum over spaxels with sliding centers xi_j, which is the same mechanism
  as in the slice method, minus the quantization.

### 3.2 Rotated frames (shared-cube roll path)

The shared-cube pathway (image_rotation != 0) pulls rotated coordinates,
so the tent kernel is applied in the celestial frame while the deposit ray
runs along the detector dispersion direction. The per-spaxel integral
becomes a 1D Gaussian convolved with a 2D tent along a diagonal direction:

    Integral dxi N(xi; xi_j, sigma_s_j) * tri(u_j - xi cos r) * tri(v_j - xi sin r)

The product of two triangles along a line is piecewise cubic in xi with at
most 8 knots, so the integral is a sum of truncated-Gaussian moments of
order 0..3 -- all closed-form in erf and exp. This costs roughly 2-3x the
axis-aligned case per spaxel and deposits into a 2D window (the tent
footprint around the ray) rather than one row. Still far below 151 slices.

An alternative is to use a different (rotationally friendlier) deposit
kernel for the roll path, as the precomputed operator already does with
Catmull-Rom cubic. That changes the interpolation kernel relative to the
loop path and would need its own bias gate; the piecewise-moment tent form
is the conservative default because it reproduces the existing pathway's
limit exactly.

### 3.3 Throughput

The exact integrand carries T(lambda(xi)) inside the xi integral. Over the
line's deposit width (sigma_s ~ 1 fine px = 0.04 nm at sigma_v = 50 km/s)
any realistic Roman throughput is constant to well below the reference
floor: a factor-3 ramp across the flagship 26 nm window has
T'/T ~ 8 %/nm, giving a profile-weighted centroid shift of
sigma_lambda^2 T'/T ~ 0.004 nm ~ 0.02 fine px and a flux error of order
(T''/T) sigma_lambda^2, both negligible. Zeroth order T(lambda_obs_j) per
spaxel is the design choice; the first-order correction
(T' times the first moment) is closed-form if ever needed. The continuum
path keeps the full T(lambda) weighting (section 4).

### 3.4 Window truncation

The deposit window must be static for JIT. Truncating at k sigma leaves
erfc(k/sqrt(2)) of the line flux: k = 4 -> 6e-5, k = 5 -> 6e-7, against
the GalSim dynamic-scene mean floor of 3e-4 of peak. Size the window from
the prior upper bound on the dispersion parameter (the same
worst-case-from-priors pattern as RenderConfig.for_priors), plus one pixel
for the tent support. Flagship prior sigma_v <= 150 km/s gives
sigma_s <= 3.0 fine px at os = 5, so a +/- 13 fine-px window (27 taps)
covers 4 sigma; typical sigma_v = 50 km/s needs +/- 5.

## 4. Continuum: closed form as well (coarse slices refuted)

The continuum is flat in lambda across the window (kl_pipe/source.py adds
I_cont to every slice), so its dispersed image is the continuum spatial
profile convolved with a T-weighted box along the dispersion axis.

A first draft of this section proposed keeping the slice path with a
coarse grid (nc ~ 9-15), reasoning that the integrand's smoothness in
lambda is set by T. The pipeline experiment refuted that: the slice
count is set by the source's spatial structure along the trace --
I_cont(q - s(lambda)) varies with lambda at the galaxy's rscale (~11
fine px, equivalent to ~2.5 nm), so nc = 11 leaves 2e-2 of peak and even
nc = 25 leaves 3e-3. Coarse continuum slices are a dead end at
production accuracy.

The correct move is closed form here too. For flat T the n -> infinity
limit is a box convolution of the tent-reconstructed image:

    Dsp_cont(q) = (D/os) * [ C(q_x - s_min) - C(q_x - s_max) ]

where C is the running integral along x of the piecewise-linear
reconstruction of I_cont: the cumulative trapezoid of the samples,
evaluated at fractional positions as a piecewise quadratic, with
half-tent corrections within one pixel outside the grid border. Exact,
O(N) per row, no slices, validated to the same floor as the line path.

For lambda-dependent throughput, T(lambda(s)) enters the kernel. Options
(design decision deferred): treat T as piecewise linear over the trace
and keep an exact closed form (products of piecewise-linear functions
integrate in closed form from the same cumulative machinery), or a
per-row 1D convolution with a finely sampled T-weighted box kernel
(error set by T curvature per fine pixel -- negligible for any smooth
Roman bandpass). The flat-T case is what current tests exercise.

## 5. What does not change

- Spatial oversampling and its floor. The os = 5 floor came from dispersal
  shift-interpolation plus readout bias at os = 3 (coherent,
  inclination-shaped residual). Line deposits still land at fractional
  positions through the same tent kernel, so the floor logic stands until
  measured otherwise. Any os relaxation is a separate, gated question; the
  static-scene GalSim floor (bilinear sub-pixel shift bias, ~0.8-1 % for
  the continuum path) is unaffected by this change.
- PSF handling. post_dispersion commutation is unchanged (the analytic sum
  is still a T-weighted linear combination of shifted images).
- Pixel response and center-sample readout semantics.
- The erf spectral method for any code path that still builds cubes
  (velocity-map observables, diagnostics).

## 6. Assumption audit (what the analytic form bakes in)

| assumption                                | already assumed today? |
|-------------------------------------------|------------------------|
| Gaussian intrinsic line profile           | yes (erf cube kernel)  |
| linear dispersion s(lambda) across window | yes (GrismPars.dispersion scalar) |
| lambda-independent PSF across window      | yes (post_dispersion default) |
| T smooth across the line width            | new, but quantified (3.3) and correctable to first order |

No new strong physical assumptions. A future nonlinear dispersion relation
s(lambda) would enter through xi_j = s(lambda_obs_j) and a local
ds/dlambda in sigma_s_j -- the analytic form handles smooth nonlinearity
at least as gracefully as slicing does.

The one representational choice is matching the tent (bilinear) kernel so
the method is bit-comparable to the current pathway's limit. If we ever
want a better interpolant, that is an orthogonal decision with its own
gate (as the Catmull-Rom operator precedent shows).

## 7. Cost model

Per line, per fine spaxel, per gradient pass:

- Current (production n = 151): 152 erf for the cube kernel, 151
  multiply-adds into the cube, and the spaxel's share of 151 bilinear
  4-tap gathers in disperse_cube, plus a (Nrow_f, Ncol_f, 151) cube in
  memory (and its rematerialized backward pass).
- Analytic (axis-aligned): about W+2 Psi evaluations (erf + exp each) and
  W scatter-adds with W ~ 11-27; no line cube at all. The continuum keeps
  nc ~ 11 slices of the old path.

Rough per-slice grad cost from the decision grid is ~0.2 ms/slice at
os = 3 (25 -> 151 slices: 11.9 -> 37.5 ms). Removing ~140 slices and
adding the deposit work should land the production-safe configuration near
the old n = 25 cost at the same os:

- expectation at os = 5: 96.7 ms -> roughly 35-45 ms (~2.5x), bounded
  below by the LOS spatial evaluation (which this change does not touch).
- The 151-iteration unrolled dispersal loop disappears from the trace,
  which should also cut compile time and helps the GPU story (scatter/
  gather instead of a long sequential loop).

These are estimates; the prototype timing gate decides.

## 8. Hypotheses and gates

H-A (primary). Analytic per-spaxel line dispersal + nc ~ 11 continuum
slices reproduces the current pathway's dense reference and removes the
n_lambda cost axis.

- Numerical gate: against the current code at n_lambda in {51, 101, 251,
  501, 1001} (fixed os, fixed scene), the slice method converges to the
  analytic image as O(ds^2), i.e. the max|diff| falls ~4x per n doubling
  toward machine-level floor. This is sharp: an error in the closed form
  shows up as a convergence plateau.
- External gate: GalSim reference scenes (static, dynamic, sheared,
  ramp-throughput) pass at their frozen floors with the analytic path
  swapped in.
- Science gate: Fisher bias vs the os = 9 / n = 251 truth below 0.1 sigma
  on all parameters (same instrument as the oversample audit); seeded
  posterior A/B within the established band.
- Cost gate: production-safe value_and_grad at or below the n = 101
  configuration's cost (i.e. >= 2x saving vs n = 151), measured min-of-N
  on a quiet machine.

Failure criteria: convergence plateau above the GalSim dynamic floor
(closed form wrong or mis-registered by a half-pixel class bug); Fisher
bias > 0.1 sigma attributable to the swap; cost within 20 % of n = 151
(no point).

H-B (fallback). Two-tier grid alone (dense over the line span, coarse over
continuum wings) achieves 2-3x at equal accuracy. Only pursued if H-A
fails its gates.

H-C (control, cheap). The n ~ 151 rule may be over-conservative now that
trapezoid weights landed: re-measure the entanglement deviation and Fisher
bias at n in {51, 75, 101} with current code. This calibrates the gate for
H-A and is worth having even if H-A ships.

H-D (skeptic's null). At production settings the LOS spatial evaluation
dominates and dispersal work is secondary, capping any dispersal-side win
below 1.5x. Test first by profiling the 96.7 ms configuration into
build_cube-lambda / disperse / LOS-eval / PSF+pixel components; the
decision-grid slope (0.2 ms/slice) already argues against H-D but the
decomposition pins the ceiling for H-A's cost gate.

## 9. Prototype plan

1. Standalone script (scratchpad): implement the axis-aligned closed form
   with numpy/JAX on a synthetic rotating-disk scene; run the O(ds^2)
   convergence gate against `disperse_cube` + erf cubes at n up to 1001.
   This validates the math before touching kl_pipe.
2. If (1) passes: implement in a worktree as a new dispersal path --
   sketch: `build_cube` splits into line-deposit (new, no cube) and
   continuum-cube (existing, nc slices) contributions; `render_grism`
   sums both dispersed images before the shared PSF + pixel stage. The
   deposit window W is static, sized from priors at task construction
   (RenderConfig pattern). Scatter via segment_sum or .at[].add with
   clipped-and-dropped out-of-stamp indices (match mode='constant').
3. Rotated path (piecewise moments) second; single-roll axis-aligned
   first, since the flagship and the GalSim gates run there.
4. Gates in order: numerical convergence, GalSim scenes, Fisher, seeded
   posterior A/B, timing grid. Record everything (supported or refuted)
   per the experiment manifest.

## 10. Prototype results (2026-07-06): numerical gate supported

Standalone fp64 numpy implementation, independent of kl_pipe (session
scratchpad, analytic_dispersal_prototype.py). Scene mirrors the flagship
grism config: 32x32 coarse, os = 5, Halpha at z = 1, D = 1.1 nm/coarse px,
axis-aligned dispersion, inclined exponential disk with an arctan rotation
curve (vcirc = 200, cosi = 0.55).

- Closed-form kernel vs adaptive quadrature over (u, sigma) in
  [-8, 8] x [0.01, 5] fine px: worst abs err 2.6e-15. The closed form is
  exact.
- Slice-method convergence toward the analytic image (max|diff|/peak):

  | n_lambda | sigma_v=20 | sigma_v=50 | sigma_v=100 |
  |----------|-----------|-----------|------------|
  | 51       | 1.5e-1    | 3.7e-2    | 8.4e-3     |
  | 101      | 5.1e-2    | 3.9e-3    | 2.2e-3     |
  | 251      | 2.1e-3    | 6.3e-4    | 3.8e-4     |
  | 501      | 5.8e-4    | 1.8e-4    | 9.0e-5     |
  | 1001     | 1.7e-4    | 4.2e-5    | 2.1e-5     |
  | 2001     | 4.1e-5    | 1.1e-5    | 5.0e-6     |

  Clean O(ds^2) (~4x error drop per n doubling) with no plateau; total
  flux agrees to 1e-8 at n = 2001. An error in the closed form or a
  half-pixel registration bug would plateau far above this. H-A's
  numerical gate: supported.
- One trap found and root-caused on the way: a first reference built on
  scipy.ndimage.shift plateaued at ~3e-4. scipy's mode='constant' zeroes
  any output whose pull coordinate leaves [0, N-1], while
  jax.scipy.ndimage.map_coordinates (what disperse_cube uses) blends the
  in-range neighbor with cval by its tent weight. The analytic tent form
  matches the JAX convention exactly; the scipy reference was the
  outlier. Any future external reference implementation must replicate
  the JAX edge semantics or mask the outermost column.
- H-D (dispersal cost is secondary) is refuted by existing decision-grid
  data without new runs: the n_lambda slope at os = 5 is
  (96.7 - 30.2) / 126 = 0.53 ms/slice, so ~68 % of the production-safe
  grad cost is n_lambda-dependent.

Stage 2, real forward model (experiments/sweverett/analytic_dispersal/
bench_pipeline.py): the analytic line deposit + closed-form flat
continuum, sharing render_grism's PSF and pixel-response stages, on the
flagship grism scene at os = 5, against a slice n = 1001 reference:

- Accuracy: analytic max 1.9e-6 / mean 1.3e-7 of peak (at the
  reference's own convergence floor, so this is an upper bound);
  production slice n = 151: 2.3e-4 / 7.0e-6; default n = 25: 2.8e-2.
- Cost: chi2 value_and_grad (11 free params, min of 20): analytic
  32.6 ms vs 98.2 ms production slice n = 151 (3.0x), equal to the old
  biased n = 25 default (32.0 ms). Compile time drops as well (no
  unrolled 151-slice loop).
- Gradients: analytic agrees with the dense reference to <= 5.9e-4
  relative (cosi, worst) and <= 3.4e-4 of max|grad| on all 11
  components; slice n = 151 is ~100-1000x further from the converged
  gradient on every component.
- The coarse-slice continuum variant (first draft of section 4) was
  refuted here: nc = 11 leaves 2.1e-2 max. The closed-form continuum
  replaced it (section 4).

Full record: experiments/sweverett/analytic_dispersal/results.md.

## 11. Literature and cross-code survey (2026-07-06)

Full agent survey in the session record; conclusions:

- No published code performs per-spaxel analytic line dispersal (Gaussian
  in lambda deposited as a closed-form profile along detector x). This
  appears to be unexplored in the open literature.
- geko (Danhaive et al. 2025, arXiv:2510.07369), our cross-validation
  reference, is in the same family as our current method: erf-CDF bin
  integration in lambda into a cube, then collapse. It bakes the trace
  offset into per-pixel wavelength centers and uses spectral oversample 9
  on NIRCam's R~1600; it still needs a wavelength grid.
- Outini & Copin 2020 (arXiv:1910.07803) is the nearest relative: they
  formalize dispersal as the same continuous integral (section 3 here)
  and evaluate it by global Fourier methods, citing an implementation
  paper ("Copin, in prep.") that was never published. Their R_kin metric
  (km/s per detector pixel) is an independent way to express the
  kinematic sampling requirement.
- aXe/grizli/pyLINEAR: discrete trace-matrix accumulation at native
  detector resolution, no sub-resolution velocity structure, no
  interpolation along the dispersion axis; no precedent either way.
- No published quantitative study of wavelength-slice quantization error
  for dispersed velocity fields exists; our entanglement tests appear to
  be the most rigorous treatment of that question anywhere.

Adversarial re-survey (2026-07-07, independent agent instructed to refute
the novelty claim): claim survived. One new candidate found and
eliminated -- Griggio, Ryan, Pirzkal et al. 2026 (arXiv:2606.09974,
Roman flux-cube reconstruction) is a discretized sparse-operator INVERSE
method, wavelength-gridded; direct full-text read (an LLM summarizer
initially fabricated a "closed-form" description of it -- always verify
prior-art determinations against the paper text). Correct scoping for a
paper: the kernel identity (Gaussian conv linear B-spline via second
differences of Psi) is standard math -- cite Unser 1999 (IEEE Sig. Proc.
Mag. 16(6), B-spline framework) and derive the specific form in an
appendix; the novel contribution is eliminating the wavelength grid from
slitless forward models by depositing the line as that closed form along
the detector dispersion axis. Outini & Copin 2020 = same continuous
integral, unpublished Fourier-space evaluation, cite as
related-but-distinct. kl-tools (our own predecessor) is grid-based
slice-and-shift with an in-code fine-grid caveat.

## 12. Integration (2026-07-06)

Implemented behind ``RenderConfig(dispersal_method='analytic',
deposit_halfwidth=N)``:

- kl_pipe/dispersion.py: ``gaussian_tent_profile``,
  ``disperse_line_analytic``, ``continuum_trace_kernel`` (exact closed
  form for flat throughput; per-segment Gauss-Legendre for a sampled
  smooth throughput), ``disperse_continuum_analytic``.
- SourceModel.render_grism dispatches to ``_render_grism_analytic``;
  spatial-owner dedupe, throughput interpolated at each spaxel's
  observed wavelength, shared post-dispersion PSF + pixel-response
  stages.
- Per-obs roll rotation (nonzero WCS) is supported: model parameters
  rotate into the detector frame exactly as the per-obs slice path does,
  and the dispersal stays along detector +x. Only the multi-roll
  shared-cube fast path (one celestial cube, rotation fused into the
  dispersal sampling) raises NotImplementedError -- its closed form
  needs the section 3.2 tent-moment machinery.
- ``deposit_halfwidth`` is required for jitted/inference use (loud
  ValueError otherwise); standalone renders size it from concrete
  parameter values.
- Gates: tests/test_analytic_dispersal.py (closed form vs quadrature,
  dense-slice equivalence incl. rolled obs and ramp throughput, jit
  equivalence, finite gradients, guard errors) and two GalSim reference
  scenes in tests/test_galsim_reference.py -- dynamic 0.206%/0.019%,
  ramp-static 0.448%/0.025% of peak, both below the slice-path floors
  (slice quantization and most shift-interpolation error vanish).

## 13. Open questions (status 2026-07-07)

1. Default flip: DONE (2026-07-07). ``dispersal_method='analytic'`` is
   the RenderConfig default; 'slice' remains fully supported (reference
   path, per-slice PSF, rotated dispersion axis, shared-cube groups).
   Gates that cleared it: Fisher shift 0.000-0.001 sigma vs the n=501
   os=5 reference at 3 anchors (slice n=151: 0.003-0.006; old n=25
   default: 0.54-1.05, fails); seeded posterior A/B within the
   trajectory-noise band with equal sampler health. Analytic obs form
   singleton groups in ``group_grism_obs_by_cube_compat`` (no cube to
   share), so multi-roll works with default configs via independent
   per-obs renders. Fixture sweep: shared-cube tests pinned to
   'slice' (they test that machinery).
2. deposit_halfwidth auto-sizing: DONE --
   ``render.line_window_halfwidth_for_priors`` (worst case
   ``|v0| + vcirc`` from prior bounds, per-line dispersion bound,
   unbounded Gaussians at 6 sd), filled in by
   ``InferenceTask.from_obs`` when the analytic rc leaves it None.
3. Rotated shared-cube kernel: tent moments derived (section 3.2), not
   implemented. Cost analysis now disfavors it on CPU: the deposit loop
   (not the spatial LOS eval) is the majority of analytic cost at the
   prior-safe halfwidth, and the rotated deposit costs 2-3x the
   axis-aligned one, so sharing the spatial eval across rolls while
   paying 4 rotated deposits loses to 4 independent axis-aligned
   renders. Re-evaluate on GPU (scatter/gather economics differ).
4. os = 5 floor relaxation: REFUTED (2026-07-07). Fisher instrument at
   os = 3 vs an analytic os = 9 reference: analytic worst shifts
   0.048/0.082/0.116 sigma across anchors vs slice n=151's
   0.052/0.088/0.117 -- statistically identical, both above the 0.05
   gate at the stress anchors. The os = 3 bias lives in the spatial
   axis (LOS evaluation / PSF grid / readout), not in dispersal
   interpolation. os = 5 remains the inference floor for both paths
   (results_fisher_gate.json Part B).

New (2026-07-07): ``disperse_line_analytic`` computes the tap profile
as a rolling second difference of Psi (one new erf+exp per tap instead
of three -- identical values, ~3x fewer special-function evaluations in
the deposit loop). A ``jax.checkpoint`` wrap of the analytic assembly
(mirroring the slice path's cube remat) was tried and REVERTED:
gradients identical (1e-13) but 1.04-1.29x slower on CPU -- the
analytic path has no cube-sized intermediate to rematerialize, so the
checkpoint only forces recompute. An apparent multi-roll backward
blowup that motivated it was machine-contention noise; on a quiet
machine the 4-roll analytic gradient is 97.9 ms (sublinear in rolls,
3.1x the 31.9 ms single-roll cost).

Quiet-machine benchmark (2026-07-07, os=5, prior-safe halfwidth 23,
value-and-grad min-of-30, experiments/sweverett/analytic_dispersal/
results_bench_final.json):

| config                       | slice n=151 | analytic | speedup |
|------------------------------|-------------|----------|---------|
| 1 band + 1 roll (quick-dev)  | 99.6 ms     | 31.9 ms  | 3.1x    |
| 4 rolls grism-only           | 310.5 ms    | 97.9 ms  | 3.2x    |
| 2 bands + 4 rolls (prod)     | 328.5 ms    | 119.0 ms | 2.8x    |

Gradient compile times drop 3-5x (e.g. 58 s -> 11 s at 4 rolls). The
shared-cube operator's advantage collapses at production settings:
slice shared at n=151/os=5 is 322.3 ms on the 4-roll gradient vs 310.5
per-roll (the 2.9x A3 win was measured at n=25/os=3 and does not
transfer -- the operator grows to ~7.5M nonzeros per roll).
