# Vista GH200 benchmark kit -- runbook

Turnkey benchmark of the kl_pipe grism posterior on a TACC Vista GH200
node. Everything lives in this directory; nothing in the repo is modified.
Local smoke-tested on CPU (klpipe env) 2026-07-04, including a
galsim-blocked "Vista simulation" pass.

## What the kit contains

| file | purpose |
|---|---|
| `bench_matrix.py` | the runner; one JSON per invocation, sections a-f |
| `tasks_vista.py` | galsim-free Q (flagship) + P (production) task builders |
| `psf_numpy.py` | analytic Gaussian PSF (galsim-free) + runtime shims |
| `ckpt_port.py` | jax.checkpoint monkeypatch variants (vendored from ckpt_probe.py) |
| `stream_fusion_port.py` | BCOO/gather/scan dispersion variants (ported from stream_fusion_probe.py) |
| `check_parity.py` | local-only shim-vs-galsim validation (already PASSED) |
| `run_vista.slurm` | SLURM batch template (fp64 pass + fp32 pass) |
| `provision_vista.sh` | idempotent one-shot: pull container + pip sidecar to persistent $WORK |
| `smoke_full.json` | local CPU reference run (M3 Max, all sections, galsim blocked, --nreps 3) |

## 0. Quick path (recommended)

Steps 1 + 3 are automated. After cloning the repo (step 2), run once on a
login node:

```bash
cd $STOCKYARD/repos/kl_roman_pipe/experiments/sweverett/vista_kit
bash provision_vista.sh          # pulls the container + pip deps to $WORK
```

It is idempotent: re-run any time (e.g. after a $SCRATCH purge or in a new
session) and it skips whatever is already in place. Then jump to step 4
(GPU sanity on a gh-dev node) or step 5 (sbatch). Steps 1 and 3 below
document what the script does, by hand.

Persistence note: durable artifacts (container, pip deps) live on `$WORK`
(= `$STOCKYARD/vista`), NOT `$SCRATCH`. TACC purges `$SCRATCH` files not
accessed for ~10 days; an 8.5 GB image on scratch would need re-pulling.
`$WORK` persists (quota only). The repo (step 2) and results already live
on persistent `$STOCKYARD`.

Why galsim-free: NGC JAX containers do not ship galsim and building it on
aarch64 is exactly the kind of yak-shave this kit avoids. The kit replaces
the galsim Gaussian PSF with an exact analytic construction
(`psf_numpy.py`). Verified locally (`check_parity.py`): identical kernel
grid sizes (so identical timings), grism render agreement 7e-9 of peak,
broadband 2e-16, log-posterior agreement 4e-5 absolute.

## 1. Get the container on Vista

```bash
# login node
module load tacc-apptainer
mkdir -p $WORK/containers && cd $WORK/containers
apptainer pull jax_26.06-py3.sif docker://nvcr.io/nvidia/jax:26.06-py3
```

`26.06-py3` is "recent as of this writing": check
https://catalog.ngc.nvidia.com/orgs/nvidia/containers/jax/tags for the
latest tag before pulling (newer = newer jax/CUDA; any 24.10+ tag is fine
for this kit). The image is multi-arch (~8.5 GB); apptainer resolves the
arm64 variant on Vista automatically. Pull once; keep it on `$WORK` so a
`$SCRATCH` purge never forces a re-pull.

NEVER build jaxlib from source on the GH200 (known failure history --
see docs/plans/PRODUCTION_SPEEDUPS.md Sec 3.1). If the container route
fails, the fallback is a venv with `pip install "jax[cuda12]"` (official
aarch64 wheels exist), but the container is primary.

## 2. Get the repo on Vista

```bash
mkdir -p $STOCKYARD/repos && cd $STOCKYARD/repos
git clone git@github.com:KinematicLensing/kl_roman_pipe.git
cd kl_roman_pipe && git checkout se/speedups   # or the commit you want benchmarked
```

(sweveret layout: `$STOCKYARD=/work/09102/sweveret`, machine-independent;
`run_vista.slurm` is pre-filled with these paths, queue `gh-dev`, and
charge code `JPL-SPHEREx`.)

The kit records the git commit in the output JSON -- keep the tree clean.

## 3. Install the missing python deps

The NGC JAX container ships jax/jaxlib/CUDA + numpy/scipy. kl_pipe's
inference path additionally needs: **numpyro, astropy, pyyaml**
(scipy usually present -- verify). NOT needed: galsim (kit avoids it),
matplotlib/emcee/nautilus/blackjax/arviz (lazy imports, unused here).

Install into a bind-mounted target dir (containers are read-only;
`--target` keeps it explicit and image-independent):

```bash
mkdir -p $WORK/klpipe_pipdeps
apptainer exec $WORK/containers/jax_26.06-py3.sif \
  python -m pip install --target $WORK/klpipe_pipdeps \
  numpyro astropy pyyaml
```

If scipy turns out to be missing from the image, add it to that list.
Do NOT let pip touch jax/jaxlib (numpyro may try to upgrade jax -- if pip
reports installing jax into the target dir, redo with
`--no-deps numpyro` plus explicit `multipledispatch tqdm` and re-check).

kl_pipe itself is used straight from the repo via
`PYTHONPATH=$REPO:$PIPDIR` (no `pip install -e .` needed -- avoids
pulling the full dependency tree; the kit only imports the inference
modules).

## 4. Sanity checks (dev node or first job step)

```bash
apptainer exec --nv --bind $STOCKYARD/repos/kl_roman_pipe:$STOCKYARD/repos/kl_roman_pipe \
  --env PYTHONPATH=$STOCKYARD/repos/kl_roman_pipe:$WORK/klpipe_pipdeps \
  $WORK/containers/jax_26.06-py3.sif python - <<'EOF'
import jax
print(jax.devices())              # expect [CudaDevice(id=0)]
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
print(jnp.zeros(1).dtype)         # expect float64
import numpyro, astropy, yaml     # deps present
import kl_pipe.source             # repo importable
print('sanity OK')
EOF
```

Then a 2-minute micro-run before burning queue time:

```bash
cd $STOCKYARD/repos/kl_roman_pipe/experiments/sweverett/vista_kit
apptainer exec --nv ... python bench_matrix.py --sections a --configs Q --nreps 5
```

## 5. Run the full matrix

```bash
cd $STOCKYARD/repos/kl_roman_pipe/experiments/sweverett/vista_kit
sbatch run_vista.slurm   # pre-filled: gh-dev, -A JPL-SPHEREx, $WORK container/pipdeps, $STOCKYARD paths
```

If you pulled a container tag other than `26.06-py3`, update `CONTAINER=`
in `run_vista.slurm` (and re-run `provision_vista.sh <tag>`) to match.

Two passes: fp64 (GH200 fp64 is strong; this is the apples-to-apples vs
the M3/TACC-CPU numbers) then fp32 (the GPU-throughput headline mode).
~30-60 min total expected; the 2 h walltime is margin for slow compiles.

## 6. What each section means / what numbers to bring home

Output: one JSON per pass (`results_vista_<stamp>_<host>_fp{64,32}.json`).
Every section is `{status: ok|error|skipped, result|error|reason, wallclock_s}`;
errors carry the full traceback -- a failed section never kills the run.

- **env**: jax/jaxlib/numpyro versions, devices, hostname, git commit.
  Confirm `devices` shows the GH200 and `x64_active` matches the pass.
- **a (posterior timings)** -- the headline. `min_ms` of
  primal/grad/value_and_grad for Q and P. CPU reference points (M3 Max
  fp64, min-of-30): Q grad ~12.9 ms, P grad ~25.6 ms. **Bring home: the
  GPU/CPU ratio per config per precision.** This directly scales the
  30M-galaxy compute plan.
- **b (chain_method)** -- wallclock for vectorized / sequential / parallel
  (identical chains by construction; check `worst_posterior_mean_shift_sigma`
  ~0 and equal `sum_num_steps`). On CPU, parallel won (1.44x vs
  sequential); on 1 GPU, parallel is auto-skipped (needs 1 device/chain)
  and the question is **vectorized vs sequential on-GPU** -- expectation
  from the plan is that batching wins on GPU; this measures it.
  ALSO verify the shipped auto-dispatch: run one short fit with
  `chain_method=None` (the new NumpyroSamplerConfig default) and confirm
  the result `metadata['chain_method']` resolves to `'vectorized'` on the
  GPU backend (it resolves to `'parallel'`/`'sequential'` on CPU depending
  on KLPIPE_CPU_DEVICES). If the section-b timings say vectorized is NOT
  the GPU winner, the dispatch table in
  kl_pipe/sampling/numpyro.py::_resolve_chain_method must be updated.
- **c (jax.checkpoint on build_cube)** -- `vg_min_ms` per variant vs base,
  `grad_matches_base` must be true. CPU verdict: cube_dots 1.20-1.25x.
  **Bring home: does the remat win grow, shrink, or invert on GH200?**
  (Decides whether the ~3-line adoption is worth it before the GPU port.)
- **d (dispersion layouts)** -- BCOO vs gather vs scan-streaming,
  `vg_min_ms` at 1 and 4 rolls. CPU verdict: BCOO wins by ~4-12x. GPU may
  reverse this (cuSPARSE vs dense gather). **Bring home: the fastest
  layout per roll-count** -- feeds the A4 operator design.
- **e (fp32 vs fp64)** -- pixel deltas (`grism_max_abs_diff_over_peak`,
  CPU reference 4.9e-5; broadband ~1e-7) + `vg_speedup_fp32_over_fp64`.
  On CPU the microbench speedup was ~2-2.7x; GH200 fp32 should be larger.
  **Bring home: the fp32 speedup and confirmation the pixel deltas match
  the CPU-measured ones** (they should -- same math; large deviations
  would flag a CUDA-math surprise). Accuracy ACCEPTANCE (posterior-level,
  <0.1 sigma shear) is a separate Tier-C campaign, not this kit.
- Also record from the .out file: compile times (`compile_ms`) -- GPU
  compile latency matters for the per-galaxy-recompile problem (plan
  Sec 3.2 item 3).

## 7. Known limitations / notes

- Section b `parallel` on a single GPU is skipped by design (recorded in
  JSON). For multi-GPU chain parallelism use a multi-node/array setup later.
- The kit intercepts kl_pipe's import-time x64 forcing in fp32 mode and
  monkeypatches the PSF precompute for the shim PSF only -- both are
  loud (printed) and kit-local; repo behavior elsewhere is untouched.
- P-config task construction renders 4 rolls of synthetic data first;
  a few minutes of compile-heavy setup before timings start is normal.
- Numbers to trust: `min_ms` (min-of-N). `mean_ms` is reported for
  thermals/contention context.
