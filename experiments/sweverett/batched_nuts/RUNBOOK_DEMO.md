# Batched-NUTS vista runbook — production-honest campaign (v4)

You (user) submit and run; the agent prepares everything and reads results
afterward. Nothing touches the se/ensemble checkout.

## v4b (2026-08-03): batch queue replaces idev

Three idev sessions were lost or clipped by the 2 h gh-dev wall (the demo
saves one npz at the very end). Everything now goes through `sbatch` on
the `gh` partition: SLURM charges actual runtime, not the requested wall,
so generous walltimes cost nothing extra and the wall cannot kill a run
mid-sampling. All jobs set JAX_LOG_COMPILES=1 so cold/warm cache state is
readable from the logs (the $SCRATCH/jax_cache persistent cache spans
sessions; per-compile walls let timing numbers be decomposed honestly).

```bash
cd $STOCKYARD/repos/kl_roman_pipe
git fetch origin && git reset --hard origin/cc/batched-nuts
sbatch experiments/sweverett/batched_nuts/run_t1b_mapwindow.slurm
sbatch experiments/sweverett/batched_nuts/run_t2_b32.slurm
sbatch experiments/sweverett/batched_nuts/run_t3b_solos.slurm
squeue --me
```

- `run_t1b_mapwindow.slurm` (3 h wall): T1b, B=16 map_window w200 s600 —
  decision input 1. Auto-runs the gate/parity analysis at the end.
- `run_t2_b32.slurm` (6 h wall): expand (if needed) + T2 B=32 at 300
  draws, then an in-job retry of sub-gate fits at 1000 draws (skipped if
  more than half the bank fails) — decision input 2 plus the escalation
  analog, no second queue round-trip. NOTE: runs `map_window`, not the
  originally specced `map_laplace` — T1 refuted the fixed-Laplace metric
  (3/16), so scaling is measured on the live candidate recipe.
- `run_t3b_solos.slurm` (2 h wall): remaining 2 solo production baselines;
  the compile log settles whether solo 9.5 min/fit rode a warm cache.

Pessimistic charged cost: T1b ~2 + T2 ~4 + T3b ~1 = ~7 node-hr worst
case; ~4-5 expected. The idev instructions below are kept for provenance;
do not use them for these trials.

## Superseded idev campaign (v4, 2026-08-02)

The v1-v3 demos initialized at truth with no MAP phase — their timing is
NOT production-achievable. Everything below uses the production recipe:
truth-free multi-start MAP through the one shared compiled program, fixed
Laplace metric, step-size-only warmup (`KLPIPE_INIT=map_laplace`).

## 0. One-time setup (login node)

The vista checkout $STOCKYARD/repos/kl_roman_pipe is already ON
cc/batched-nuts; everything runs from there. env_vista.sh binds that
checkout and $WORK/klpipe_pipdeps (blackjax/arviz) -- no PYTHONPATH
override needed; the demo script adds its own experiments dir to sys.path.

```bash
cd $STOCKYARD/repos/kl_roman_pipe
git fetch origin && git reset --hard origin/cc/batched-nuts
source experiments/sweverett/vista_kit/env_vista.sh
```

NOTE: the LogNormal window-sizing fix (7f40616) grows line windows
(halfwidth 23->29 / 31->37), so per-eval cost rises somewhat vs v3 —
expected, not a regression. `runs/cosmos25_ab_bb32gr32` (16 fits) already
exists from the v3 session and is REUSED as-is: its mocks/manifest predate
nothing (expansion does not depend on the window fix; windows are sized at
fit time).

## SESSION A (idev gh-dev, 2 h): honest B=16 + first solo parity fits

```bash
idev -p gh-dev -N 1 -n 1 -t 02:00:00
cd $STOCKYARD/repos/kl_roman_pipe
source experiments/sweverett/vista_kit/env_vista.sh
$KLPIPE_PYTHON -c "import blackjax, arviz; print('deps ok')"
```

### T1 — batched B=16, production init (target ~75-90 min)

```bash
mkdir -p runs/demo_v4_maplaplace
KLPIPE_INIT=map_laplace KLPIPE_NUTS_WARMUP=100 KLPIPE_NUTS_SAMPLES=600 \
KLPIPE_DEMO_OUT=runs/demo_v4_maplaplace \
  $KLPIPE_PYTHON -u experiments/sweverett/batched_nuts/batched_nuts_demo.py \
  2>&1 | tee runs/demo_v4_maplaplace/demo.log
```

Phases + pessimistic walls: build ~1 min; MAP+Laplace ~10-20 min for 16
fits (multi-start L-BFGS-B, ~4-6 ms/eval solo through the shared program;
printed per fit — if the first fit takes > 2 min, something is wrong,
ctrl-C and ping the agent); warmup ~2-6 min (100 steps, fixed metric,
step-size-only; v3 paid 37 min here); sampling ~55-70 min (600 draws,
target accept 0.9). Abort criterion: any phase at 2x its pessimistic wall.

Quick look while it runs / after:

```bash
$KLPIPE_PYTHON experiments/sweverett/batched_nuts/analyze_demo.py \
  runs/demo_v4_maplaplace runs/no_solo_yet
```

### T3a — solo production baselines, first 2 fits (fills the remainder)

```bash
$KLPIPE_PYTHON -m kl_pipe.ensemble worker \
  --run-dir runs/cosmos25_ab_bb32gr32 --label solo_a --max-fits 2 \
  2>&1 | tee runs/cosmos25_ab_bb32gr32/solo_a.log
```

Production numpyro path exactly as shipped (MAP+Laplace, warmup 200,
300 draws x 4 chains, escalation 800/1000). Pessimistic ~20-25 min/fit
including compile; 2 fits ~45 min. If < 40 min remain in the session,
run with `--max-fits 1`.

## SESSION B (idev gh-dev, 2 h): B=32 scaling + retry + remaining solos

### Expand the 32-fit run dir (once, in the idev shell)

```bash
$KLPIPE_PYTHON -m kl_pipe.ensemble expand configs/ensembles/cosmos25_batched_b32.yaml
ls runs/cosmos25_batched_b32/manifest.parquet
```

### T2 — batched B=32 (128 lanes), production-matched first-pass draws

```bash
mkdir -p runs/demo_v5_b32
KLPIPE_RUN_DIR=runs/cosmos25_batched_b32 KLPIPE_N_FITS=32 \
KLPIPE_INIT=map_laplace KLPIPE_NUTS_WARMUP=100 KLPIPE_NUTS_SAMPLES=300 \
KLPIPE_DEMO_OUT=runs/demo_v5_b32 \
  $KLPIPE_PYTHON -u experiments/sweverett/batched_nuts/batched_nuts_demo.py \
  2>&1 | tee runs/demo_v5_b32/demo.log
```

Pessimistic walls: build ~2 min; MAP ~20-40 min (32 fits, sequential);
warmup ~3-8 min; sampling ~30-70 min (300 draws, 128 lanes — the scaling
number this trial exists to measure). HBM watch: if it OOMs at 128 lanes,
rerun with `KLPIPE_N_FITS=24`; that is itself a useful ceiling measurement.

### T2r — retry pass for sub-gate fits (production escalation analog)

```bash
$KLPIPE_PYTHON experiments/sweverett/batched_nuts/analyze_demo.py \
  runs/demo_v5_b32 runs/no_solo_yet          # read the per-fit gate table
mkdir -p runs/demo_v5_b32_retry
KLPIPE_RUN_DIR=runs/cosmos25_batched_b32 \
KLPIPE_FIT_IDS=<comma-list of FAIL fit_ids from the gate table> \
KLPIPE_INIT=map_laplace KLPIPE_NUTS_WARMUP=100 KLPIPE_NUTS_SAMPLES=1000 \
KLPIPE_DEMO_SEED=20260803 KLPIPE_DEMO_OUT=runs/demo_v5_b32_retry \
  $KLPIPE_PYTHON -u experiments/sweverett/batched_nuts/batched_nuts_demo.py \
  2>&1 | tee runs/demo_v5_b32_retry/demo.log
```

### T3b — remaining solo baselines (if time; also fine on a later session)

```bash
$KLPIPE_PYTHON -m kl_pipe.ensemble worker \
  --run-dir runs/cosmos25_ab_bb32gr32 --label solo_b --max-fits 2 \
  2>&1 | tee runs/cosmos25_ab_bb32gr32/solo_b.log
```

## Afterward

Agent pulls `runs/demo_v4_maplaplace/`, `runs/demo_v5_b32*/` (npz, a few
MB each) and `runs/cosmos25_ab_bb32gr32/{results,chains,status}` over your
multiplexed ssh and writes up parity + honest-throughput numbers.

Parity gate (T1/T2 vs T3 solos): per-fit per-param |z| mostly < 2-3 at
MC-error scale, width ratios ~1 within ~20-30%. Systematic width inflation
or mean shifts = STOP before any adopt decision. Cost gate: fits/node-hr
at matched ESS (not matched draws), MAP phase included on both sides.
