# Sampler failure ledger

Permanent record of per-fit sampler failures in the ensemble pipeline: what
the diagnostic columns mean, the failure classes seen so far with their
signatures and evidence, and a per-campaign tally. Append to it whenever a
production-like run is read; never rewrite history. Handoffs in
`docs/sessions/` hold the narrative, this file holds the facts that must
survive sessions.

## Per-fit diagnostics (summary parquet rows, `results/<fit_id>.parquet`)

| column | meaning |
|---|---|
| `max_rhat`, `min_ess` | escalation-gate quantities over all sampled parameters |
| `max_rhat_param`, `min_ess_param` | which parameter sets each (failure correlation) |
| `ess_g1`, `ess_g2` | shear ESS, the quantities the science uses |
| `n_attempts`, `escalated`, `first_attempt_max_rhat`, `first_attempt_min_ess` | escalation history |
| `divergence_rate`, `mean_accept_prob` | NUTS health |
| `num_steps_total` | leapfrog steps summed over chains and draws (per-draw array in `chains/<fit_id>.npz['num_steps']`) |
| `precond_condition_number` | condition of the floored Laplace metric (`1/eig_floor` when the floor is active) |
| `precond_n_negative_eigenvalues` | negative eigenvalues of the unregularized scale-normalized MAP Hessian; > 0 means the optimizer stopped on a saddle or in a lower basin |
| `precond_min_eigenvalue_ratio` | min/max eigenvalue before flooring (negative when the above is > 0) |
| `n_map_starts_converged` | L-BFGS starts that met the convergence test |
| `map.<param>`, `map_minus_postmean_over_sigma.<param>` | MAP vs posterior mean, per parameter |

Gate (production specs): `rhat_max` 1.05, `ess_min` 50, one escalation retry
(800/1000 warmup/samples, donated adapted metric). A fit that fails the gate
after the retry is kept and flagged in `status`/`collate` as catastrophic.

## Failure classes

| class | signature | evidence | status | candidate fix |
|---|---|---|---|---|
| A. Multimodal chains | first-attempt `max_rhat` >> 1.1 (2.6-2.9), `min_ess` 2-4, no divergences; chains sit in different basins | fit dacee1cd (noise_seed 1780535355) in every fp64 arm of the cosmos25_noise_ab_matched bank; escalation reaches only rhat 1.07 at 460k-670k steps; sets the 8-worker run wall | OPEN, the slow tail | mode enumeration: PA-flip counterpart descent + basin margin at the preconditioner; chain init spread across basins; report both modes |
| B. Saddle / lower-basin MAP | `precond_n_negative_eigenvalues` > 0; first attempt `min_ess` ~25, `max_rhat` ~1.08 | fit 95e70579 (n_neg 5, escape +1167 logpost -> n_neg 0, min_ess 24 -> 99, rhat 1.085 -> 1.048); fit 12eab959 (n_neg 4, coarse line scan finds no improving direction, sampling unaffected) | 2/16 at the production MAP (PA-stratified starts); the 2026-08-28 "7/16" count predates them | iterated escape (line scan along the most negative eigenvector + re-descent), opt-in stage in `laplace_preconditioner`; finer or 2-D scan for the residual class |
| C. Ridge geometry | steps/draw pinned at 63-127 (tree depth 6-7) on clean fits; directional curvature changes 30-400x within +/-1 sigma along the softest eigenvectors | all 16 bank fits (curvature_swing study 2026-09-07); metric changes (fd vs ad, escaped MAP) leave steps/draw unchanged | OPEN, the per-draw cost floor | sampler-layer bijective reparam informed by the MAP Hessian (spin-2 disk-frame shear, vcirc sin i); position-dependent metric; MAMS |
| D. Prior-wall regularization loss | flat shear prior: escalations 4 -> 7, steps +34%, median fit wall 17 -> 43 min, posteriors unchanged | flatprior arm 974729 vs 968810 | PARKED (Gaussian prior stays) | reparam first |
| E. Tree-depth cap | `max_tree_depth` 8: steps -3%, one extra catastrophic fit (rhat 1.29 / ess 12 after escalation, converged uncapped) | depth8 arm 976034 vs 968810 | REFUTED as a lever, do not adopt | none |

## Infrastructure failures (not sampler pathologies)

| failure | signature | evidence | fix |
|---|---|---|---|
| fp32 escalation retry rejected the donor metric | `ValueError: init_inverse_mass_matrix must be symmetric` at `_run_fit_escalated` | fp32-ad arm 974798, all 4 escalating fits | eccfc62: donor symmetrized at working precision |
| AD Hessian OOM at pack 8 | `RESOURCE_EXHAUSTED ... 5.7 GiB ... jit__log_posterior_jittable` | fp64-ad arm 974797, 4/16 fits | eccfc62: one HVP per parameter |
| cuFFT single-precision plan failure | `Failed to make cuFFT batched plan: 5` at the first fp32 FFT, every fit, within a minute | container nightly on all nodes tried (2026-09-03); release jax 0.11.1 on c642-032 (976033) but not on c642-002, c642-042, c642-072 | OPEN: node-dependent; fp32 production needs a startup probe + requeue |

## Campaign tally

| date | run | code | N | escalated | catastrophic | notes |
|---|---|---|---|---|---|---|
| 2026-08-06 | census-era baseline, 4 workers | Aug-6 | 16 | 6 | 0 | 11.93 h sum wall |
| 2026-09-05 | 968810 fp64 fd (reference) | d2fcdd6 | 16 | 4 | 0 | 7.45 h; one class-A fit sets the run wall |
| 2026-09-05 | 968857 fp64 local window | 803757c | 16 | 4 | 0 | posteriors identical to reference |
| 2026-09-06 | 974729 fp64 flat shear prior | 803757c | 16 | 7 | 0 | class D |
| 2026-09-06 | 974797 fp64 AD Hessian | 232f1cf | 12/16 | 2 | 0 | 4 OOM (infra) |
| 2026-09-06 | 974798 fp32 AD Hessian | 232f1cf | 12/16 | 0 | 0 | 4 died at escalation (infra) |
| 2026-09-06 | 976032 fp64 AD Hessian | eccfc62 | 16 | 2 | 0 | agrees with fd on all 16 |
| 2026-09-06 | 976033 fp32 AD Hessian | eccfc62 | 0/16 | - | - | cuFFT plan failure, node c642-032 |
| 2026-09-06 | 976034 fp64 depth cap 8 | eccfc62 | 16 | 3 | 1 | class E |
| 2026-09-07 | 976765 fp32 AD Hessian | eccfc62 | 16 | 6 | 1 | node c642-042; shear means agree with fp64 (median 0.11 sigma), widths +3%; ms/step 0.79x but sum wall equal because of the extra escalations |
| 2026-09-07 | 976766 saddle-escape A/B, 7 fits x 2 arms | eccfc62 | 7 | - | - | class B prevalence 2/7; class A fit unchanged; steps/draw unchanged |
