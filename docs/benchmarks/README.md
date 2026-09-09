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
<!-- index:end -->
