# Sampler benchmark records

Permanent, git-tracked record of sampler A/B runs on the canonical benchmark
bank. One JSON record per run under `runs/` plus a copy of that run's 32-row
summary parquet beside it; the table below is generated from the records.

## The benchmark

`configs/ensembles/cosmos25_bank32.yaml` and its twins
`cosmos25_bank32_*.yaml`: 16 COSMOS25 galaxies x a ring pair (theta and
theta + pi/2) = 32 fits, same population seed, noise streams and sampler
seeds in every twin, so a twin differs from the bank in exactly the knob
under test and fits match one-to-one on (galaxy_id, ring_member,
noise_seed). At the ~12% first-pass failure rate 32 fits give ~4 expected
failures per arm for ~9 h of summed fit wall, enough to see the failure set
move; a 16-fit bank does not.

Rule: every sampler-side change (prior, reparameterization, warmup, mass
matrix, escalation policy, precision) gets a row here before it becomes a
default. The 983442 baseline predates the full-circle position-angle prior
default (4a4a317); later rows include it, so single-variable comparisons
against 983442 are confounded by the prior change.

## Adding a record

Two commands, run from the repo root after the run's `results.parquet` is
collated (`python -m kl_pipe.ensemble collate --run-dir R`):

```
python -m kl_pipe.ensemble.bench record --run-dir runs/<run> --job-id <slurm id> \
    --precision fp64 --node <node> --run-wall <s> \
    --reference docs/benchmarks/runs/<reference>.json \
    --hypothesis "..." --reason "..." [--conclusion "..."]
python -m kl_pipe.ensemble.bench index
```

`record` reads the arm name, commit and spec from `runs/<run>/provenance/`;
pass `--commit` when the fits ran at a later commit than the expansion, and
`--results`/`--manifest`/`--spec-path` (plus `--commit`) for a run dir
without provenance. Fill the conclusion later with
`python -m kl_pipe.ensemble.bench annotate docs/benchmarks/runs/<record>.json --conclusion "..."`
and rerun `index`.

## Comparing

`python -m kl_pipe.ensemble.bench compare --run-dir A --run-dir-ref B` prints
both metric columns and the posterior agreement (|dmean|/sigma_ref and width
ratios for g1, g2, cosi, theta_int, vel.vcirc; fits that fail in only one
arm) without writing anything. `--reference` on `record` stores the same
agreement block in the record under `metrics.agreement_vs_reference`.
`kl_pipe.ensemble.bench.load_records()` returns the records as a DataFrame.

Metric definitions: first-pass fail = escalated; final fail = gate
(rhat > 1.05 or min ESS < 50) still failed after escalation; catastrophic =
fit status not succeeded; steps/draw clean = leapfrog steps per first-attempt
draw (n_chains x n_samples from the spec) over non-escalated fits; medians
over succeeded fits. theta_int differences are wrapped modulo 2 pi, so a
counter-rotating mode (theta + pi) shows up as a large pull.

Chains and mocks are not copied: they stay in the run directory on vista or
in the local `runs/` copies (gitignored).

## Index

<!-- index:start -->
| date | arm | job | commit | first-pass fails | final fails | steps/draw clean | clean fit wall median [min] | sum wall [h] | conclusion |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-08 | cosmos25_bank32 | 983442 | a3af13c | 4/32 | 0/32 | 58.5 | 12.8 | 9.45 | 4/32 first-pass fail, 0 final; theta_int limits ESS in 16/32 |
| 2026-09-09 | cosmos25_bank32_w250 | 985872 | 3c80b5a | 4/32 | 1/32 | 36.1 | 9.4 | 8.37 | 4/32 first-pass fails again but a different set; steps/draw on clean fits 63... |
| 2026-09-10 | cosmos25_bank32_w400 | 985873 | 3c80b5a | 3/32 | 0/32 | 32.0 | 11.2 | 9.67 | 31/32 fits (one lost to a JAX autotune-cache race between the remainder job a... |
| 2026-09-10 | cosmos25_bank32_w200 | 986080 | cbf2d0d | 2/32 | 0/32 | 41.9 | 9.1 | 7.14 | vs 983442: steps/draw on clean fits 58 -> 42, clean fit wall 12.8 -> 8.8 min,... |
| 2026-09-10 | cosmos25_bank32_mapfix | 988356 | e73c019 | 3/32 | 0/32 | 46.7 | 9.0 | 8.18 | SUPPORTED on the MAP: max dev 0.5-1.2 on all 32 (ref 24.7 on g5_r90); grad no... |
| 2026-09-10 | cosmos25_bank32_priorfloor | 988824 | e73c019 | 0/32 | 0/32 | 48.3 | 9.3 | 6.48 | SUPPORTED. 14/32 fits (truth cosi >= 0.32, no clipping either way) reproduce... |
| 2026-09-11 | cosmos25_bank32_robust | 990891 | ae63ae0 | 2/32 | 0/32 | 45.1 | 8.8 | 7.81 | Composes as expected. 32/32, 2 first-pass fails (g15_r90 rhat 1.20 / ess 9, g... |
<!-- index:end -->
