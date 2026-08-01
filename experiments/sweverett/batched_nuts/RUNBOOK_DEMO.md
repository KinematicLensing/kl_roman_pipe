# Batched-NUTS vista demo — self-serve runbook

You (user) submit and run; the agent reads results afterward. Budget: one
gh-dev node, <= 2 h (<= 2 SU). Everything is idempotent; nothing touches the
se/ensemble checkout.

## 0. Local: push the branch (agent does this)

`cc/batched-nuts` on origin carries the demo. Already pushed if you are
reading this on vista.

## 1. Vista login node: separate worktree (one-time)

```bash
cd $STOCKYARD/repos/kl_roman_pipe
git fetch origin
git worktree add $STOCKYARD/repos/kl_batched_nuts origin/cc/batched-nuts
cd $STOCKYARD/repos/kl_batched_nuts
ln -s $STOCKYARD/repos/kl_roman_pipe/data/cosmos2025/private data/cosmos2025/private
```

## 2. Expand the run dir (login node, ~1 min, deterministic seed)

```bash
cd $STOCKYARD/repos/kl_batched_nuts
source experiments/sweverett/vista_kit/env_vista.sh   # apptainer alias, sets JAX cache
klpython -m kl_pipe.ensemble expand configs/ensembles/cosmos25_ab_bb32gr32.yaml
# -> runs/cosmos25_ab_bb32gr32 (same fit_ids as the A/B arm: same spec seed)
```

Sanity: `ls runs/cosmos25_ab_bb32gr32/manifest.parquet`.

## 3. idev demo (GPU node)

```bash
idev -p gh-dev -N 1 -n 1 -t 02:00:00
cd $STOCKYARD/repos/kl_batched_nuts
source experiments/sweverett/vista_kit/env_vista.sh
export PYTHONPATH=$PWD
klpython -u experiments/sweverett/batched_nuts/batched_nuts_demo.py \
  2>&1 | tee runs/batched_nuts_demo/demo.log
```

Defaults: 16 fits x 4 chains = 64 lanes, warmup 400 + samples 300, dense
per-lane window adaptation, max_num_doublings 7, fp64. Expected wall from
the pessimistic estimate: ~90 min + ~15 min compile. If the warmup stage is
still running past ~75 min, ctrl-C and rerun with
`KLPIPE_NUTS_WARMUP=200 KLPIPE_NUTS_SAMPLES=200` (still enough draws for
the parity gate at min_ess ~50 scale).

Outputs land in `runs/batched_nuts_demo/`:
`demo_results.npz` (positions, acceptance, per-iter integration steps,
divergences, step sizes) + `demo_meta.json` (timings, config) + `demo.log`.

## 4. Analysis (login node or agent-side; needs solo baselines)

```bash
klpython experiments/sweverett/batched_nuts/analyze_demo.py \
  runs/batched_nuts_demo \
  $STOCKYARD/repos/kl_roman_pipe/runs/vista_stamp_ab/cosmos25_ab_bb32gr32/results
```

Prints: min/fit-equivalent + fits/node-hr, straggler overhead
(lockstep max/mean integration steps), divergence rate, per-fit R-hat/ESS,
and parity z-scores + width ratios vs the solo results; writes `parity.csv`.

The agent pulls `runs/batched_nuts_demo/` (npz ~ a few MB) over ssh for the
full writeup once your multiplexed session is up.

## Interpretation guide (agreed gates)

- parity: |z| mostly < 2-3 (MC-error scale), width ratios ~1 within ~20-30%.
  Systematic width inflation or mean shifts = investigate before any adopt
  decision. Note the demo inits at truth+1% jitter (solo used MAP+1%) and
  adapts dense mass per lane via blackjax window adaptation (solo used
  Laplace + adapt-mass numpyro) -- same target, different samplers, so
  parity within MC error is the correct expectation, not equality.
- cost: fits/node-hr vs the ~6 effective today (16-18 min solo, 1.7x
  aggregate at 4-wide packing). The stack beyond this demo: larger B, fp32,
  warmup shortening (Laplace-metric init), retry re-sizing.
- straggler: if lockstep overhead >> 1.5x, the tree-depth cap and/or
  MAMS-style fixed-length trajectories move up the priority list.
