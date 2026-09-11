# Fit initialization

`kl_pipe.sampling.initialization` builds everything the NUTS sampler needs
before its first step: optimizer starts, the MAP, the Laplace metric and the
chain initial points. One call runs the whole procedure and returns one
record; the pieces stay public for refining it one at a time. This page is
the user pathway: the one-call API, the pieces behind it, the ensemble spec
knobs (same names) and the summary columns that tell you what happened.

## One call

```python
from kl_pipe.sampling.initialization import InitConfig, Initializer

config = InitConfig()                       # the robust defaults, see below
result = Initializer(task, config, seed=seed, image_obs=image_obs).run()

result.preconditioner      # LaplacePreconditioner for NumpyroSampler(task, cfg, preconditioner=...)
result.map                 # MapResult: every start's endpoint, basins, stationarity
result.map.format_summary()
result.metric              # MetricResult: floored spectrum, condition number
result.starts.families()   # {'prior': 4, 'pa_stratified': 8}
result.columns()           # the per-fit summary columns the ensemble worker writes
result.chain_points(inverse_mass_matrix, n_chains, seed=seed, transform=transform)
```

`InitConfig` fields are the ensemble spec's `fit.*` initialization knobs
under the same names (`InitConfig.from_spec(spec)` copies them), plus
`n_pa_starts` (position-angle grid points per rotation direction, 4) and the
optimizer internals `maxiter` (2000) and `fd_rel_step` (1e-5). The stages are
also methods on `Initializer` (`starts()`, `find_map(starts)`,
`metric(map_result)`, `preconditioner(map_result, metric)`) for inspecting
one at a time. `InferenceTask.laplace_preconditioner(...)` is the same
procedure behind a keyword interface with the same defaults.

## Defaults and why

| knob | default | why |
|---|---|---|
| `map_bounded` | `true` | Unbounded L-BFGS stalls where its first line-search trial crosses a prior wall: the endpoint reports convergence with a scaled gradient norm of 150-3000 and sits thousands of nats below the truth basin. On the cosmos25 bank the bounded search alone reached the truth basin on every flagged fit; A/B 988356 vs 986080 cured a wrong-mode posterior the reference had passed. |
| `map_polish_steps` / `map_polish_basins` | 8 / 3 | Regularized Newton descent certifies stationarity (`map_grad_norm` < 1e-3, positive Hessian) and descends through saddles the L-BFGS stall left; 3 basins so a runner-up basin is polished before the margin is read. Costs `2 n_params + 1` gradients per step taken. |
| `eig_floor_mode` / `eig_floor` | `prior` / 0.5 | The relative floor (1e-4 x max) pegged the condition number at 1e4 in 20/32 bank fits and clipped 1-9 real posterior directions by up to 2x in variance. In prior units the softest true eigenvalue is 0.6-1.1 in every healthy fit, so a floor at 0.5 never touches a direction the data constrain. A/B 988824 vs 986080: 0/32 first-pass fails, sum wall -9%, posteriors unchanged. |
| both together | | 990891 `cosmos25_bank32_robust`: posteriors identical to 988356, cost neutral outside the two fits whose truth sits on the cosi prior wall. |
| `map_moment_starts` | `false` | Image-moment starts landed near the truth but did not change the basin reached once the search was bounded; kept opt-in. |
| `chain_init` | `map_jitter` | `map_basins` on the legacy search (988826) started chains at optimizer stall points and was refuted; its retest on the bounded search is arm 990892. |

The historical procedure is reproduced by `InitConfig(map_bounded=False,
map_polish_steps=0, eig_floor_mode='relative')` (relative floor default
1e-4) and by the same keywords on `laplace_preconditioner`.

## Pieces

```python
from kl_pipe.sampling.initialization import (
    EigenFloor, build_preconditioner, chain_inits, combine_starts, find_map,
    moment_starts, pa_stratified_starts, prior_starts,
)

# 1. start proposals (each a StartSet: points + family labels)
starts = combine_starts(
    prior_starts(task, n=4, seed=seed),                     # prior draws
    pa_stratified_starts(task.priors, n_pa=4, seed=seed),   # PA grid, both rotation directions on the circle
    moment_starts(task, image_obs, seed=seed),              # image moments -> centroid, flux, size, cosi, PA (+pi)
)

# 2. MAP finder: every endpoint kept, clustered into basins
r = find_map(task, starts, seed=seed)                       # bounded, polish 8 x 3 by default
r.theta_map, r.neg_logpost, r.n_basins, r.basin_margin, r.winning_label
r.map_grad_norm, r.map_min_eigenvalue                       # stationarity of the final MAP

# 3. Laplace metric with an explicit floor rule
pre = build_preconditioner(task, r, floor=EigenFloor('prior', 0.5))
pre.n_floored_eigenvalues, pre.condition_number

# 4. chain initial points (sampling coordinates)
inits = chain_inits(pre, inv_mass, n_chains=4, mode='map_basins', seed=seed, transform=transform)
```

### Start proposals

| family | what | randomness |
|---|---|---|
| `prior` | independent prior draws | `PRNGKey(seed + 1)` |
| `pa_stratified` | prior draws with `theta_int` on a grid: `2 n_pa` points over the circle (periodic prior) or `n_pa` over the bounds | `PRNGKey(seed + 2)` |
| `moments` | adaptive elliptical-Gaussian moments of each broadband stamp, PSF-deconvolved (PSF second moments measured through the galaxy's own weight) and pixel-corrected: centroid for every component, band pixel sums for fluxes, size for every `rscale` (matched-Gaussian sigma / 1.164 for an exponential), inclination from the axis ratio with the sech^2 thickness correction `cosi^2 = (q^2 - q0^2) / (1 - q0^2)`, `q0 = 1.15 h_over_r`, position angle for `theta_int` at PA and PA + pi; shear 0; everything else at the prior median | none |

Moment accuracy on the 32 noisy cosmos25 bank stamps (SNR 40-50 per band):
size within 10-25% (Roman PSF wings bias it high), cosi within 0.1, PA within
0.15 rad, centroid within 1-1.5 pixels, flux within 20%. A stamp with no
usable object raises `MomentsError`; `Initializer` warns, goes on with the
other families and records `map_moment_starts_ok = no`.

### MAP finder

`find_map` runs L-BFGS-B in prior-scaled coordinates (`theta = loc + scale
u`, `loc`/`scale` from 512 prior draws) from every start, keeps every finite
endpoint (converged or not; the lowest objective wins), and clusters the
endpoints into basins by single linkage at 0.25 prior sigma (periodic
parameters wrapped). `bounded=True` hands the prior support bounds to
L-BFGS-B so the projected gradient slides along a wall instead of stepping
through it. `polish_steps=n` runs regularized Newton descent (|eigenvalue| of
the scaled Hessian, floored at 1e-6 of the largest; backtracking line search)
from the best endpoint of each of the `polish_basins` best basins; it stops
when the scaled gradient norm is below 1e-3 and the Hessian is positive
definite. `map_grad_norm` and `map_min_eigenvalue` report the final MAP's
stationarity: a large gradient or a negative eigenvalue means the point is
not a local maximum and the metric built on it is not a Laplace
approximation.

### Eigenvalue floor

`EigenFloor('prior', 0.5)` (default): eigenvalues of the prior-scaled Hessian
below 0.5 are raised to it, in units where 1 means the posterior is as wide as
the prior along that direction. `EigenFloor('relative', 1e-4)`: eigenvalues
below `1e-4 x max` are raised to it (condition number capped at 1e4).

### Chain initial points

`'map_jitter'`: every chain at the MAP plus 1% of the metric scale times a
standard normal. `'map_basins'`: chain 0 at the MAP, the next chains at the
best endpoint of each competing basin within `chain_init_max_margin` nats
(default 20), the rest jittered. With every chain started in the MAP basin
r-hat cannot see a basin disagreement.

## Ensemble spec knobs

Same names as `InitConfig`; each can be A/B'd alone on the benchmark bank
(`docs/benchmarks/README.md`).

```yaml
fit:
  n_map_starts: 4            # prior-draw starts (4 PA-stratified starts per direction are always added)
  map_moment_starts: false   # add image-moment starts
  map_bounded: true          # projected L-BFGS-B on the prior support
  map_polish_steps: 8        # Newton polish steps (0 = off)
  map_polish_basins: 3       # basins polished
  eig_floor_mode: prior      # prior | relative
  eig_floor: null            # mode default (0.5 prior, 1e-4 relative)
  chain_init: map_jitter     # map_jitter | map_basins
  chain_init_max_margin: 20.0
```

A run directory stores the resolved values of every knob
(`provenance/ensemble_spec_resolved.yaml`), so a rebuild reads what the fits
ran with rather than the defaults of the rebuilding code.

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
from kl_pipe.sampling.initialization import InitConfig, Initializer

task, map_theta, chains, truth, names = rebuild_fit_posterior(run_dir, fit_id)
result = Initializer(task, InitConfig.from_spec(spec), seed=seed).run()
print(result.map.format_summary())
```
