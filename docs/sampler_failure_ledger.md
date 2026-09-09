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
| `n_attempts`, `escalated`, `escalation_mode`, `escalation_n_blocks`, `first_attempt_max_rhat`, `first_attempt_min_ess` | escalation history; `escalation_mode` is 'restart' (fresh warmup, donated metric), 'continue' (more draws from the warm chains in `escalation_n_blocks` blocks, first-attempt draws kept) or '' (no escalation) |
| `restart_reason`, `restart_recommended` | why more draws would not (or did not) rescue the first attempt: '' (marginal), 'rhat' (> `continue_rhat_max`), 'divergences' (> `continue_divergence_max`), 'blocks_exhausted' (continued to the block cap, still below the gate); `restart_recommended` marks fits to re-run fresh |
| `divergence_rate`, `mean_accept_prob` | NUTS health |
| `num_steps_total` | leapfrog steps summed over chains and draws (per-draw array in `chains/<fit_id>.npz['num_steps']`) |
| `precond_condition_number` | condition of the floored Laplace metric (`1/eig_floor` when the floor is active) |
| `precond_n_negative_eigenvalues` | negative eigenvalues of the unregularized scale-normalized MAP Hessian; > 0 means the optimizer stopped on a saddle or in a lower basin |
| `precond_min_eigenvalue_ratio` | min/max eigenvalue before flooring (negative when the above is > 0) |
| `n_map_starts_converged` | L-BFGS starts that met the convergence test |
| `map_pa_flip_margin` | negative-log-posterior margin of the MAP over the best optimization start that settled in the counter-rotating PA basin (`inf`: no start settled there; `nan`: half-turn PA prior, no such basin) |
| `map.<param>`, `map_minus_postmean_over_sigma.<param>` | MAP vs posterior mean, per parameter |
| `map_postmean_max_dev`, `map_postmean_max_dev_param` | largest \|MAP - posterior mean\| / sigma over the sampled parameters, and which one; healthy fits sit at 0.5-2.7 (64 fits, banks 983442 + 985872), a MAP the optimizer left in a wrong basin at 11-25 |
| `map_chi2`, `postmean_chi2`, `n_data` | -2 log L at the MAP and at the posterior mean (no data constant, so plain chi-squares) against the number of masked data pixels; a posterior stuck in a wrong basin shows a chi-square excess of hundreds to thousands (fit fcfb5651 in 985872: 1586 nats = 3172 in chi-square above the truth basin) |

Gate (production specs): `rhat_max` 1.05, `ess_min` 50, one escalation retry
(800/1000 warmup/samples, donated adapted metric; with `escalation.mode: auto`
a marginal first attempt is instead continued from its warm chains in blocks
of `continue_block` = 300 draws per chain, gate re-checked per block, at most
`continue_max_blocks` = 4 blocks). A fit that fails the gate after the retry
is kept and flagged in `status`/`collate` as catastrophic.

## Failure classes

| class | signature | evidence | status | candidate fix |
|---|---|---|---|---|
| A. Multimodal chains | first-attempt `max_rhat` >> 1.1 (2.6-2.9), `min_ess` 2-4, no divergences; chains sit in different basins | fit dacee1cd (noise_seed 1780535355) in every fp64 arm of the cosmos25_noise_ab_matched bank; escalation reaches only rhat 1.07 at 460k-670k steps; sets the 8-worker run wall | OPEN, the slow tail | mode enumeration: PA-flip counterpart descent + basin margin at the preconditioner; chain init spread across basins; report both modes |
| B. Saddle / lower-basin MAP | `precond_n_negative_eigenvalues` > 0; first attempt `min_ess` ~25, `max_rhat` ~1.08 | fit 95e70579 (n_neg 5, escape +1167 logpost -> n_neg 0, min_ess 24 -> 99, rhat 1.085 -> 1.048); fit 12eab959 (n_neg 4, coarse line scan finds no improving direction, sampling unaffected) | 2/16 at the production MAP (PA-stratified starts); the 2026-08-28 "7/16" count predates them | iterated escape (line scan along the most negative eigenvector + re-descent), opt-in stage in `laplace_preconditioner`; finer or 2-D scan for the residual class |
| C. Ridge geometry | steps/draw pinned at 63-127 (tree depth 6-7) on clean fits; directional curvature changes 30-400x within +/-1 sigma along the softest eigenvectors | all 16 bank fits (curvature_swing study 2026-09-07); metric changes (fd vs ad, escaped MAP) leave steps/draw unchanged | OPEN, the per-draw cost floor | sampler-layer bijective reparam informed by the MAP Hessian (spin-2 disk-frame shear, vcirc sin i); position-dependent metric; MAMS |
| D. Prior-wall regularization loss | flat shear prior: escalations 4 -> 7, steps +34%, median fit wall 17 -> 43 min, posteriors unchanged | flatprior arm 974729 vs 968810 | PARKED (Gaussian prior stays) | reparam first |
| E. Tree-depth cap | `max_tree_depth` 8: steps -3%, one extra catastrophic fit (rhat 1.29 / ess 12 after escalation, converged uncapped) | depth8 arm 976034 vs 968810 | REFUTED as a lever, do not adopt | none |
| F. Position-angle prior wall | truth `theta_int` within ~0.3 rad of the `Uniform(0, pi)` fit-prior walls; first-attempt `max_rhat` 1.07-2.9, `min_ess` 2-60; the counter-rotating solution (theta + pi) is outside the prior, so it appears as theta near the opposite wall with a compensating shear (fit dacee1cd: modes at theta 0.15-0.25 with g2 -0.19 / -0.24 and at 2.9 with g2 0.0, truth 3.04; log-posterior gap ~8 nats in favour of the truth mode) | census v1 (200 fits, July): fail rate 33% / 53% for wall distance < 0.15 / 0.15-0.3 rad vs 15% beyond 0.8 rad, median steps 2x, median theta pull 1.14 sigma vs ~0.5; 16-fit bank 968810: all 4 escalations among the 6 fits within 0.4 rad of a wall, 0 of 10 beyond 0.5 rad | FIXED: `fit.pa_prior: full_circle` is the default since 4a4a317 (`CircularUniform` prior, periodic sampling coordinate, PA starts over both rotation directions, `map_pa_flip_margin` column); A/B 983939 vs 968810: steps -30%, near-wall steps x0.38, tail fit 460k -> 104k first try | none; fits with `map_pa_flip_margin` < ~3 nats have a genuinely ambiguous rotation direction and the circle samples both modes |

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
| 2026-09-08 | 983442 fp64 fd, 32-fit bank cosmos25_bank32 | a3af13c | 32 | 4 | 0 | first-pass 12.5%; theta_int sets min_ess in 16/32; near-wall median steps 107-132k vs 61k far; no negative MAP eigenvalues among the escalations |
| 2026-09-08 | 983939 fp64 fd, full-circle PA prior (class F fix) | 1a6f914 | 16 | 4 | 0 | vs 968810: steps 2.65M -> 1.85M, sum wall 7.45 -> 5.09 h; near-wall (6 fits) steps x0.38, escalations 4 -> 1, the tail fit 460k -> 104k steps first try; far-wall (10 fits) steps x1.11, escalations 0 -> 3 at first-attempt rhat 1.06-1.07; shear means agree (g1 max 0.16 sigma; g2 max 0.72 sigma on a near-wall fit, 0.12 far); flip margins 0.6-1396 nats, 2/16 below 2 nats |
| 2026-09-09 | 985582 fp64 fd, escalation mode auto (continue), same 16 fits as 983939 | 4a4a317 | 16 | 4 | 0 | first attempts bit-identical to 983939; all 4 escalations marginal (rhat 1.06-1.07, min_ess 63-130) -> continued 1000 draws/chain, all passed (rhat 1.004-1.009, min_ess 367-703); escalated fits wall 113 vs 140 min (x0.81) despite 1.6x more sampling steps (no 800-draw re-warmup); run wall 2662 vs 3175 s; posteriors agree with the restart arm (shear |dmean|/sigma <= 0.12, widths 0.91-1.05); the 1000-draw block was ~3x more than the gate needed, hence the block-wise continuation (continue_block 300, max 4) |
| 2026-09-09 | 985872 fp64 fd, cosmos25_bank32_w250 (n_warmup 250; also the first bank32 run at the full-circle PA prior default, so NOT a single-variable twin of 983442; matched reference w200 = job 986080) | 3c80b5a | 32 | 4 | 1 | first-pass 4/32 again but a different set: 3 of the 4 983442 failures (incl. 967ce510, tau 102) pass cleanly, 3 new fits fail (one at rhat 1.158; one still 1.080 after the restart); steps/draw on clean fits 63 -> 34, clean fit wall 10.9 -> 9.2 min, sum wall 9.45 -> 8.37 h; median first-attempt rhat 1.020 -> 1.023, worst tau 6.9 -> 6.3 (unchanged); per-chain adapted-metric spread measured directly from the saved matrices (window 100): x1.47 median along the softest direction, generalized-eigenvalue range 0.21-2.36 over the full matrix, and NO correlation with first-attempt rhat (rho -0.01); g2 widths x1.09 (circle prior, as in 983939) |
