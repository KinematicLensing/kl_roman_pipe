# Fitting lessons

What the ensemble sampler campaigns taught us about fitting joint
photometry + grism kinematic-lensing posteriors with NUTS, one lesson per
entry: symptom, cause, evidence (job ids on the cosmos25 benchmark bank
unless noted), fix, status. The tally of every arm lives in
`docs/sampler_failure_ledger.md` and `docs/benchmarks/README.md`; this page
is the narrative those tables support, written for the paper's methods and
for whoever next changes the fitting procedure.

## 1. A bounded position-angle prior manufactures a second mode

**Symptom.** First-pass convergence failures concentrated on galaxies whose
true position angle sat within ~0.3 rad of the `Uniform(0, pi)` prior
edges: fail rate 33-53% there vs 15% elsewhere (census v1, 200 fits), median
leapfrog steps 2x, theta pull 1.14 sigma at the wall.

**Cause.** With `vcirc > 0` the counter-rotating solution `theta + pi` lies
outside the prior; it reappears at the opposite wall as a shape-shear
compensated mode (theta near the other edge, g2 off by 0.2). Chains cannot
cross the wall, so r-hat fails and the truncated marginal is biased.

**Fix.** Sample theta on the circle (`CircularUniform`, period 2 pi,
periodic sampling coordinate wrapped about the MAP for diagnostics, MAP
starts over both rotation directions). A/B 983939 vs 968810: steps -30%,
near-wall fits x0.38 steps, the 460k-step tail fit 104k first try; shear
means unchanged. Default since 4a4a317. A fit whose MAP margin over the
counter-rotating basin is below ~3 nats has a genuinely ambiguous rotation
direction and the circle samples both.

## 2. The convergence gate is noisy at 4 x 300 draws; escalate by continuing

**Symptom.** Every adaptive arm on the 32-fit bank fails 2-4 first passes,
but a *different* set each time (983442 vs 985872 vs 985873); autocorrelation
times 7-14 on the slow directions put rhat 1.05 inside estimator noise.

**Fix.** Escalate marginal first attempts (rhat <= 1.2, no divergences) by
continuing the warm chains in 300-draw blocks with the gate re-checked per
block (`escalation.mode: auto`), restart only for chains in different
basins. 985582 vs 983939: escalated fits x0.81 wall despite 1.6x more
draws, posteriors agree to 0.12 sigma. The gate threshold itself is left at
1.05 until the initialization and metric work settles (a gate on the
Monte-Carlo error of the shear means is the candidate replacement).

## 3. The warmup metric is a noisy covariance estimate, not a bad Hessian

**Symptom.** Per-chain adapted metrics disagree by 0.21-2.36 in generalized
eigenvalue over the full 24 x 24 matrix (985872 chains); chains in the same
fit take 2-12x different autocorrelation times.

**Cause.** numpyro's Stan schedule (verified 0.20.1) estimates the dense
metric from 50 (n_warmup 200) or 100 (250) correlated draws per chain and
shrinks it toward `1e-3 I`; the Laplace metric is *replaced* after the first
window, never used as the shrinkage target. Marchenko-Pastur for p = 24,
n = 100 predicts a 0.26-2.22 eigenvalue band: the measured spread is pure
sample-covariance noise.

**Evidence.** Frozen Laplace metric (985912): 17 first-pass escalations of 23
fits, i.e. the MAP Hessian alone is not good enough (the relative eigenvalue
floor clips its soft directions, lesson 5). n_warmup 400 (985873): steps/draw
32 vs 42 but no fewer first-pass fails and a longer sum wall. Clean fits under
the frozen metric are the fastest of any arm (427 s median): a *good* fixed
metric is the cheapest regime.

**Direction.** Pool the covariance across chains (staged warmup: adaptive
warmup, pooled window covariance, restart every chain from that metric,
production). Landed opt-in as `fit.warmup_metric: pooled`. A four-fit
prototype (2026-09-14, cosmos25_bank32_robust fits on gh-dev) found the
pooled metric 2-3x closer to the reference posterior covariance than any
single chain's, the frozen Laplace metric the worst option on every fit,
and a Laplace admixture unsafe without a guard: at a cos i prior-wall MAP the
unconstrained-coordinate Laplace metric has a 2e10 eigenvalue that a 25%
blend inherits, and a frozen blended metric sent three chains to the tree
cap (r-hat 400) where adaptation had discarded it. The 4-fit x 2-seed
follow-up (997155) ranked the stage-2 variants by shear ESS per leapfrog
step relative to the status quo: frozen pooled metric 1.40x median (worst
0.88x, r-hat <= 1.03, the wall fit included); re-adapting from the pooled
metric 0.99x (one seed at r-hat 1.23: adaptation re-noises the metric per
chain); Laplace blend with its eigenvalues clipped to within 10x of the
pooled covariance 0.85x frozen, 0.71x adapting (the blend knob was dropped
from the code); 150 stage-2 draws 0.84x. The status quo itself varies 35-85 steps/draw between seeds on
one fit, so per-fit comparisons need seeds; the bank32 A/B decides the
default. A third run (998895, same 7 fit-seeds; GPU reruns reproduce
experiment 2 exactly) tested whether the 2-5x spread of adapted step sizes
between chains sharing the frozen pooled metric comes from numpyro's default
initial step size of 1.0: seeding the stage-2 step size from the median
stage-1 value left the spread unchanged (2.6x median vs 2.5x) and produced
one r-hat 1.42 failure; 100 stage-2 draws did not shrink it either (3.0x);
numpyro's doubling/halving heuristic drove the step size to 0.01-0.02 and
cost 3.7x the steps; one common production step size (chain median) cut wall
40% but failed 2 of 7 fit-seeds (r-hat 1.07 / 1.17), so part of the spread is
real chain-local curvature, not adaptation noise. The unseeded frozen
50-draw variant stays the only one with r-hat <= 1.03 and min ESS >= 93 on
every fit-seed.

## 4. "Wrong-basin MAPs" were optimizer false convergence

**Symptom.** 3/32 (983442) and 2/32 (985872) fits had a MAP thousands of
nats below the truth basin (theta + pi, 3x too large line disk, shifted
centroid, v0 off by 100 km/s). NUTS warmup escaped in 4 of 5; in one
(985872, fit fcfb5651) the posterior stayed there and passed the gate at
rhat 1.018. Every L-BFGS start reported "converged".

**Cause.** Not multimodality. At the trapped endpoints the scaled gradient
norm is 150-3000 and the prior-scaled Hessian has 1-4 negative eigenvalues.
L-BFGS-B, run unbounded with the `-inf` prior as a barrier, takes a first
line-search trial through a prior wall (cosi 0.009 from its lower bound),
the search collapses to a zero step and the relative-reduction test declares
convergence at iteration 1. The straight line from a stalled endpoint to the
truth basin is monotonically downhill (55 nats over 20 samples, fit
1803a44c). Prior-draw starts make it worse: a centroid prior of 0.11" against
a 0.005" posterior and a line-size prior 3x the truth put the starts in the
smeared region where these stalls live.

**Fix (toolkit, `docs/fit_initialization.md`).** Hand the support bounds to
L-BFGS-B. On the three flagged baseline fits (1803a44c, 05c84f8d, 188b1f83)
the bounded search alone reaches the truth-start optimum from the production
starts (gap 0.00 nats; the 11-12 "basins" of the unbounded search collapse
to 2-5), at 3000-6000 gradient evaluations per fit instead of the 100-750
the stalled search spent. A regularized Newton polish of the leading basins
(+700 evaluations) certifies the result: scaled gradient norm 1e-4 to 1e-6,
smallest prior-scaled Hessian eigenvalue ~1.0. Image-moment starts (centroid,
flux, size, cosi, PA from adaptive moments) are available but did not change
the basin reached; alone, without bounds, they stall 13-451 nats short.
Recorded per fit: `map_grad_norm`, `map_min_eigenvalue`, `map_n_basins`,
`map_basin_margin`, `map_polish_gain`. Detection after the fact:
`map_postmean_max_dev` is 0.5-2.7 for healthy fits and 11-25 for the four
bad MAPs; a stuck posterior shows a chi-square excess (`postmean_chi2 -
n_data`).

**Status.** Bank A/B read (988356 `cosmos25_bank32_mapfix` vs 986080):
every MAP interior and stationary except at the cosi prior wall (28/32 with
gradient norm < 1e-3, smallest eigenvalue 0.6-1.2; `map_postmean_max_dev`
0.5-1.2 on all 32, was 24.7 on g5_r90). The w200 reference posterior of
g5_r90 turns out to have been in the counter-rotating mode on all four
chains (v0 -104 vs truth +15) after passing the gate on escalation; the
bounded search recovers the truth basin. Cost: preconditioner wall 22 -> 43
s median, sum wall +14%, of which most is the two fits with truth cosi 0.054:
their MAP sits on the cosi 0.05 wall (nonzero projected gradient, negative
Hessian eigenvalue) and the Laplace metric built there costs 1.6-2.1x the
steps. Open: metric at a boundary MAP; g8_r90 has two rotation-direction
modes 0.14 nats apart and each run sampled a single one (chain-per-basin arm
988826 pending).

Combined with the prior-unit floor (990891 `cosmos25_bank32_robust`, the
reference for later arms): the two fixes compose. 2/32 first-pass fails on a
different pair of fits (gate noise), g5_r90 in the truth basin, posteriors
identical to the mapfix arm, and the cost is neutral (steps 1.00, wall 0.98
of 986080) once the two cosi-0.054 wall fits are set aside; those two cost
2.3x and are the target of the wider cosi fit prior.

## 5. The relative eigenvalue floor clips real posterior directions

**Symptom.** `precond_condition_number` pegged at 1e4 (= 1/eig_floor) in
20/32 bank fits.

**Cause.** In prior-scaled coordinates the posterior's stiffest direction
(centroids, sigma_post/sigma_prior ~ 0.05) is 1e4-2e4 times stiffer than
its softest (fluxes, vcirc, dispersion: prior-dominated by design,
sigma_post/sigma_prior 0.8-1.0). A floor at `1e-4 x max` therefore raises
1-9 genuine eigenvalues in 13/32 fits, making the metric up to 2x too stiff
exactly along the directions that cost leapfrog steps.

**Fix.** `EigenFloor('prior', 0.5)`: an absolute floor in prior-width units;
the softest true eigenvalue is 0.6-1.1 in every healthy fit, so the floor
never touches a data-constrained direction. The one fit with a softer
direction (0.11) is the stuck wrong-basin posterior. Status: bank A/B read
(988824 vs 986080): the 14 fits where the relative floor never engaged are
bit-identical; on the 18 edge-on fits the condition number runs free (up to
2.4e5) with steps 0.42-1.45x (median 1.00), both reference escalations pass
first try at 0.42-0.44x the steps, 0/32 first-pass fails, sum wall -9%,
posteriors unchanged (max 0.2 sigma). Candidate default.

## 6. Four chains started at one point cannot see a basin error

**Symptom.** The stuck posterior of lesson 4 passed r-hat because all chains
began at the same (wrong) MAP with 1% jitter.

**Fix.** `chain_init: map_basins` starts one chain per competing optimizer
basin within 20 nats of the MAP. Costs nothing when the basins agree.

**Status.** Bank A/B (988826) with the legacy unbounded search REFUTED the
knob in that combination: the 3-12 "basins" per fit are mostly the stall
points of lesson 4, so chains start at non-optima; 8/31 escalations, 5
final gate failures, sum wall +37%, job timed out. The knob is only
meaningful after the bounded search leaves real optima (1-5 per fit in
988356); rerun as bounded + polish + map_basins.

Rerun on the bounded search (990892): still refuted as a default. 21/32
fits keep 2-5 distinct optima after polish; the chains started in optima
3-20 nats below the MAP never join the main chains in 500 draws (9 final
gate failures, sum wall 1.58x). Those optima carry e^-3 to e^-20 of the
posterior mass, so this measures the trap, not the posterior. The useful
product is the basin margin from the ordinary run: three fits have a
competitor within 3 nats (the genuinely rotation-ambiguous g8_r90 at 0.14
nats among them), and the census flags margin < 3 nats rather than starting
chains there.
Status: implemented, A/B pending.

## 7. Levers that did not work

- **Flat shear prior** (974729): +34% steps, 7 vs 4 escalations, posteriors
  unchanged; the Gaussian prior's curvature regularizes the shear
  directions. Kept Gaussian.
- **Tree-depth cap 8** (976034): -3% steps, one extra catastrophic fit.
- **Linear reparameterizations** measured on saved chains: a fixed-angle
  spin-2 rotation of the shear gains ESS x1.01 (a dense metric already
  absorbs any linear map); `vcirc sin i` *raises* the cosi-vcirc correlation.
- **Sampling in the disk-frame shear (g+, gx)**: rotating a sky-pinned
  quantity by an uncertain theta manufactures a ridge (r = -1.00 between
  theta and gx); keep shear in the sky frame and use g+/gx only for
  interpretation.
- **float32**: means agree with fp64 to 0.1 sigma, widths +3-5%, but no wall
  gain on this problem and the cuFFT plan failure is node-dependent. Parked.
- **Longer warmup** (lesson 3): fewer steps per draw, same failure count,
  longer wall.

## 8. Where the sampler's time goes

Every fit has the same two slow directions (983442 chains): inclination vs
the three disk sizes vs plus-shear (|r| 0.97; a rounder image is either more
face-on or sheared), and position angle vs cross-shear vs systemic velocity
(|r| 0.98; a cross-shear rotates the isophotes). The other ten geometric
directions decorrelate in 1-2 draws. Failing fits are the slow tail of this
one continuum, not a different population. The position-angle width grows
as 1/sin i toward face-on (sigma_theta sin i ~ 0.16 rad across the bank),
which no single mass matrix can represent: the nonlinear reparameterization
target is the (cosi, theta) funnel, e.g. the projected spin vector
`sin i (cos theta, sin theta)`, which keeps the 2 pi rotation direction.

## 9. Provenance traps

- A saved spec YAML records only the keys the author wrote; a rebuild
  resolves the missing ones at the *current* code defaults. The 983442
  baseline (half-turn PA prior) rebuilds today with the full-circle prior.
  Since 48859df expansion writes `provenance/ensemble_spec_resolved.yaml`
  with every fit, escalation and render knob written out, and rebuilds read
  it; run dirs from before it warn on load.
- Two jobs sharing a node and the JAX persistent-cache directory race on the
  per-fusion autotune files (986384 vs 986080 on c642-001; one fit lost).
- Escalation-restart fits under float32 rejected their donor metric on
  roundoff asymmetry until it was symmetrized at working precision.
## 10. Edge-on inclination is prior-dominated at any lower bound

Widening the cos i fit prior from the generating [0.05, 0.95] to [0.02, 1.0]
(990979 vs 990891) moved the two truth-0.054 MAPs from the 0.05 wall to the
0.02 wall with the same projected gradient, cost every fit 1.14x per step
(the lower bound sets the worst-case k-space grid), and moved cos i means by
0.03 with unchanged widths and identical shear posteriors. For a thick disk
(h/r 0.25) the observed axis ratio is flat in cos i below about 0.15, so the
likelihood there is flat and the posterior mean is set by where the prior
stops. The coherent +0.6 sigma cos i pull at edge-on and -0.4 sigma face-on
is truncation of a skewed, bounded posterior, not a sampler bias: judge
inclination recovery by coverage and rank, not by the mean pull. A face-on
only widening [0.05, 1.0] would cost no grid and is the remaining untested
variant.

## 11. The 12-worker packing failure was the log-posterior chunk, not the sampler

Twelve workers per node (991151): 22 of 32 fits died allocating ~5 GiB and
the survivors ran 1.36x slower per step. The 5 GiB was the end-of-run
log-posterior evaluated over 256 draws at once; 9cbf221 evaluates 16 (the
NUTS gradient itself needs 0.16 GiB). The same transient killed 20 of 64
census v2 fits at 8 workers. With the chunk fixed, 8 workers hold 3 GiB each
of 95.6 at 100% GPU utilisation and 51% memory bandwidth (992869), so the
packing ceiling is compute, not memory. 12 and 16 workers on the bank are
untested since the fix; the 1.36x per-step slowdown at 12 is the expected
price of oversubscribing a saturated GPU.

## 12. Disk-frame plus-shear is skewed at edge-on under the wide fit shear prior

Sky-frame g1, g2 truth ranks are uniform in every cos i bin of census v2
(121 fits), but the shear rotated into the disk frame is not: the g+ truth
rank averages 0.26 at cos i < 0.15 (68% coverage 0.55, mean pull +0.78
sigma) and 0.47 face-on; gx is uniform everywhere. Spearman(rank g+,
rank cos i) is 0.87, so this is the inclination / size / plus-shear ridge
of lesson 8. The truth sits at the likelihood peak (2 dlogP median 23 for
24 parameters, edge-on and face-on alike), so it is not model mismatch.
The mechanism is the two prior walls: the cos i lower bound cuts the ridge
on one side (lesson 10) while the N(0, 0.2) fit shear prior lets it run to
positive g+ on the other. Importance-reweighting the saved chains to the
N(0, 0.03) population prior restores edge-on rank 0.46, coverage 0.72 and
pull +0.15, with sigma(g+) 0.089 -> 0.027: the population prior dominates
g+ per galaxy at this SNR. Random position angles average the skew out of
the sky-frame means, so the mean-based estimator is calibrated but pays in
scatter, and any PA-shear correlation (intrinsic alignment) would turn the
skew into a bias. A hierarchical or population prior (equivalently the
reweighting) removes it at the cost of the shrinkage A_ivw already corrects.
To quantify on the full census; the paper reports the sky-frame calibration
and states this mechanism.
