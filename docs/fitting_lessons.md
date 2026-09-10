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

**Direction.** Pool the covariance across chains and shrink toward the
Laplace metric (staged warmup: Laplace metric, N draws, pooled shrunk
estimate, freeze, step-size warmup, production). Not yet run.

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

**Status.** Toolkit landed; bank A/B (`cosmos25_bank32_mapfix`: bounded +
polish) pending.

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
direction (0.11) is the stuck wrong-basin posterior. Status: implemented,
A/B pending.

## 6. Four chains started at one point cannot see a basin error

**Symptom.** The stuck posterior of lesson 4 passed r-hat because all chains
began at the same (wrong) MAP with 1% jitter.

**Fix.** `chain_init: map_basins` starts one chain per competing optimizer
basin within 20 nats of the MAP. Costs nothing when the basins agree.
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
  Record the resolved fit settings with the run.
- Two jobs sharing a node and the JAX persistent-cache directory race on the
  per-fusion autotune files (986384 vs 986080 on c642-001; one fit lost).
- Escalation-restart fits under float32 rejected their donor metric on
  roundoff asymmetry until it was symmetrized at working precision.
