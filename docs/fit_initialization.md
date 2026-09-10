# Fit initialization toolkit

`kl_pipe.sampling.initialization` builds everything the NUTS sampler needs
before its first step -- optimizer starts, the MAP, the Laplace metric and
the chain initial points -- from small composable pieces, each of which
leaves a record in the fit summary. This page is the user pathway: what the
pieces are, how to use them directly on an `InferenceTask`, how the ensemble
worker exposes them as spec knobs, and which columns tell you what happened.

## Why a toolkit

Two failure modes of the production fits trace to the initialization stage,
not to the posterior:

- **Wrong-basin MAP.** L-BFGS from prior draws settles in a smeared local
  optimum (counter-rotating position angle, 3x too large line disk, shifted
  centroid) thousands of nats below the truth basin; the Laplace metric built
  there is garbage and, once in 64 fits, the chains stayed there and passed
  the convergence gate. Diagnosed on the cosmos25 bank (2026-09-09/10): the
  endpoints are not local optima at all. L-BFGS-B reports convergence with a
  scaled gradient norm of 150-3000 because its first line-search trial steps
  through a prior wall (`-inf`) and the search collapses to a zero step.
  Handing the bounds to L-BFGS-B reaches the truth basin on every flagged
  fit; a regularized Newton polish certifies stationarity (and descends
  through the stall on its own: 3089 -> 3033 nats in 4 steps).
- **Clipped soft directions.** The Laplace metric's relative eigenvalue
  floor (`1e-4 x max`) pegs the condition number at 1e4 in 20/32 bank fits
  and clips 1-9 real posterior directions (the prior-dominated ones: fluxes,
  vcirc, dispersion) by up to 2x in variance. An absolute floor in prior
  units never touches a direction the data constrain.

## Pieces

```python
from kl_pipe.sampling import initialization as ini

# 1. start proposals (each returns a StartSet: points + family labels)
starts = ini.StartSet.concat(
    ini.prior_starts(task, n=4, seed=seed),                 # prior draws
    ini.pa_stratified_starts(task.priors, n_pa=4, seed=seed),  # PA grid (both directions on the circle)
    ini.moment_starts(task, image_obs, seed=seed),          # image moments -> centroid, flux, size, cosi, PA (+pi)
)
print(starts.families())          # {'prior': 4, 'pa_stratified': 8, 'moments': 2}

# 2. MAP finder: every endpoint kept, clustered into basins
r = ini.find_map(task, starts, seed=seed, bounded=True, polish_steps=8, polish_basins=3)
print(r.format_summary())         # per-start table: family, converged, -logpost, gap to MAP, basin
r.theta_map, r.neg_logpost, r.n_basins, r.basin_margin, r.winning_label
r.map_grad_norm, r.map_min_eigenvalue   # stationarity of the final MAP (polish stage)

# 3. Laplace metric with an explicit floor rule
pre = ini.build_preconditioner(task, r, floor=ini.EigenFloor('prior', 0.5))
pre.n_floored_eigenvalues, pre.condition_number

# 4. chain initial points (sampling coordinates)
inits = ini.chain_inits(pre, inv_mass, n_chains=4, mode='map_basins', seed=seed, transform=transform)
```

`InferenceTask.laplace_preconditioner(...)` composes 1-3 with the historical
defaults (prior draws + `extra_starts`, unbounded L-BFGS, no polish, relative
floor) and accepts every knob above (`starts=`, `bounded=`, `polish_steps=`,
`polish_basins=`, `eig_floor_mode=`, `eig_floor=`). `NumpyroSamplerConfig`
carries `chain_init` / `chain_init_max_margin` for step 4.

### Start proposals

| family | what | randomness |
|---|---|---|
| `prior` | independent prior draws | `PRNGKey(seed + 1)` (historical stream) |
| `pa_stratified` | prior draws with `theta_int` on a grid: `2 n_pa` points over the circle (periodic prior) or `n_pa` over the bounds | `PRNGKey(seed + 2)` |
| `moments` | adaptive elliptical-Gaussian moments of each broadband stamp, PSF-deconvolved (PSF second moments measured through the galaxy's own weight) and pixel-corrected: centroid for every component, band pixel sums for fluxes, size for every `rscale` (matched-Gaussian sigma / 1.164 for an exponential), inclination from the axis ratio with the sech^2 thickness correction `cosi^2 = (q^2 - q0^2) / (1 - q0^2)`, `q0 = 1.15 h_over_r`, position angle for `theta_int` at PA and PA + pi; shear 0; everything else at the prior median | none |

Moment accuracy on the 32 noisy cosmos25 bank stamps (SNR 40-50 per band):
size within 10-25% (Roman PSF wings bias it high), cosi within 0.1, PA within
0.15 rad, centroid within 1-1.5 pixels, flux within 20%. A stamp with no
usable object raises `MomentsError`; the worker then falls back to the other
families and records `map_moment_starts_ok = no`.

### MAP finder

`find_map` runs L-BFGS-B in prior-scaled coordinates (`theta = loc + scale
u`, `loc`/`scale` from 512 prior draws) from every start, keeps every finite
endpoint (converged or not; the lowest objective wins), and clusters the
endpoints into basins by single linkage at 0.25 prior sigma (periodic
parameters wrapped). Options:

- `bounded=True`: the prior support bounds go to L-BFGS-B so the projected
  gradient slides along a wall instead of stepping through it.
- `polish_steps=n`: regularized Newton descent (|eigenvalue| of the scaled
  Hessian, floored at 1e-6 of the largest; backtracking line search) from the
  best endpoint of each of the `polish_basins` best basins. Costs
  `2 n_params + 1` gradient evaluations per step with the finite-difference
  Hessian (~1.5 s per step on the production task, CPU). Stops when the
  scaled gradient norm is below 1e-3 and the Hessian is positive definite.

The result carries `map_grad_norm` and `map_min_eigenvalue`: a MAP with a
large gradient or a negative eigenvalue is not a local maximum and the
metric built on it is not a Laplace approximation.

### Eigenvalue floor

`EigenFloor('relative', 1e-4)` (historical): eigenvalues of the prior-scaled
Hessian below `1e-4 x max` are raised to it. `EigenFloor('prior', 0.5)`:
eigenvalues below 0.5 are raised to it, in units where 1 means the posterior
is as wide as the prior along that direction. Measured on the bank chains,
the softest true posterior eigenvalue is 0.6-1.1 in every healthy fit, so
the prior floor is inactive there while the relative floor clips up to 9
directions.

### Chain initial points

`'map_jitter'` (historical): every chain at the MAP plus 1% of the metric
scale times a standard normal. `'map_basins'`: chain 0 at the MAP, the next
chains at the best endpoint of each competing basin within
`chain_init_max_margin` nats (default 20), the rest jittered. With every
chain started in the MAP basin r-hat cannot see a basin disagreement; the one
stuck wrong-basin posterior on record (985872, fit fcfb5651) passed the gate
at rhat 1.018.

## Ensemble spec knobs

All default to the historical procedure, so each can be A/B'd alone on the
benchmark bank (see `docs/benchmarks/README.md`).

```yaml
fit:
  n_map_starts: 4            # prior-draw starts (PA-stratified starts are always added)
  map_moment_starts: false   # add image-moment starts
  map_bounded: false         # projected L-BFGS-B on the prior support
  map_polish_steps: 0        # Newton polish steps (0 = off)
  map_polish_basins: 1       # basins polished
  eig_floor_mode: relative   # relative | prior
  eig_floor: null            # mode default (1e-4 relative, 0.5 prior)
  chain_init: map_jitter     # map_jitter | map_basins
  chain_init_max_margin: 20.0
```

## Summary columns

| column | meaning |
|---|---|
| `map_n_basins`, `map_basin_margin` | endpoint basins found and the nats between the MAP basin and the runner-up (inf: one basin) |
| `map_winning_start` | start family that produced the MAP |
| `map_moment_starts_ok` | `yes` / `no` (images yielded no usable moments; fit proceeded without) / `n/a` (not requested) |
| `map_grad_norm`, `map_min_eigenvalue`, `map_polish_gain` | stationarity of the final MAP and the nats the polish stage gained (nan without a polish stage) |
| `precond_n_floored_eigenvalues`, `precond_eig_floor_mode` | how many metric directions the floor touched, and which rule |
| `chain_init` | chain initial-point mode |
| `map_postmean_max_dev`, `map_chi2`, `postmean_chi2` | the after-the-fact checks (MAP vs posterior, chi-squares); a MAP left in a wrong basin sits at 11-25 on the first, a stuck posterior shows a chi-square excess |

## Inspecting a fit's MAP search after the fact

```python
from kl_pipe.diagnostics.posterior_slices import rebuild_fit_posterior
from kl_pipe.sampling import initialization as ini

task, map_theta, chains, truth, names = rebuild_fit_posterior(run_dir, fit_id)
starts = ini.StartSet.concat(ini.prior_starts(task, 4, seed), ini.pa_stratified_starts(task.priors, 4, seed))
print(ini.find_map(task, starts, seed=seed).format_summary())
```

Note that a rebuild resolves spec defaults at the current code version; a
run made before a default changed (e.g. the full-circle PA prior) rebuilds
with the new default unless the saved spec pins the old value.
