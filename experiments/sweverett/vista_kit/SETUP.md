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
| `stage_vista.sh` | mirror the $WORK master to fast per-machine $SCRATCH (run before big campaigns) |
| `klrun.sh` | one-command wrapper: run any `python ...` inside the container (compute node only) |
| `sanity_check.py` | GPU device + import check (run via `bash klrun.sh python sanity_check.py`) |
| `results/<machine>/<run>/` | all result JSONs, organized per machine per run (vista_gh200, stampede3_spr, local_m3); `results/local_m3/smoke_full.json` = the M3 reference run. The SLURM scripts write new runs into `results/<machine>/<stamp>_<host>*/` automatically |

## 0. Quick path (recommended)

Steps 1 + 3 are automated. After cloning the repo (step 2), grab a compute
node and run once THERE:

```bash
idev -p gh-dev -N 1 -n 1 -t 01:00:00     # interactive compute node
cd $STOCKYARD/repos/kl_roman_pipe/experiments/sweverett/vista_kit
bash provision_vista.sh                   # pull container + pip deps to $WORK
bash klrun.sh python sanity_check.py      # GPU device + import check
```

`provision_vista.sh` is idempotent: re-run any time (e.g. after a $SCRATCH
purge or new session) and it skips whatever is already in place. Steps 1 + 3
below document what it does, by hand.

**CRITICAL -- apptainer is a NO-OP on Vista login nodes.** It prints a "do not
run Apptainer on the login nodes" banner and exits WITHOUT executing, so a
login-node `provision_vista.sh` silently pulls/installs nothing yet reports
success. Everything apptainer -- pull, pip install, and all execution -- must
run on a compute node via `idev` or `sbatch`. (`git clone`/`pull` in step 2
is fine on the login node; that's not apptainer.)

Storage model (two tiers):
- **Master on `$WORK`** (persistent, = `$STOCKYARD/vista`): `provision_vista.sh`
  pulls the container + pip deps here once. `$WORK` survives `$SCRATCH`
  purges (~10 days no-access), so you never re-pull the 8.5 GB image.
- **Hot copy on `$SCRATCH`** (fast, per-machine): jobs read from here.
  `stage_vista.sh` mirrors master -> `$SCRATCH`; `run_vista.slurm` auto-stages
  for a single job.

Why the hot copy: the `.sif` is read-only, so concurrent reads never clobber
-- but `$WORK`/Stockyard is a single filesystem shared across all TACC
machines and is not built for job I/O. At hundreds/thousands of concurrent
fits, read from `$SCRATCH`, not `$WORK`. **Before a large job array, run
`bash stage_vista.sh` ONCE** (don't let each array task copy the image).
The repo (step 2) and results live on persistent `$STOCKYARD`.

Why galsim-free: NGC JAX containers do not ship galsim and building it on
aarch64 is exactly the kind of yak-shave this kit avoids. The kit replaces
the galsim Gaussian PSF with an exact analytic construction
(`psf_numpy.py`). Verified locally (`check_parity.py`): identical kernel
grid sizes (so identical timings), grism render agreement 7e-9 of peak,
broadband 2e-16, log-posterior agreement 4e-5 absolute.

## 1. Get the container on Vista

```bash
# on a COMPUTE node (idev), NOT the login node -- apptainer no-ops on login
module load tacc-apptainer
export APPTAINER_CACHEDIR=$SCRATCH/apptainer_cache   # default $HOME cache blows home quota
mkdir -p $WORK/containers $APPTAINER_CACHEDIR
# redirect the pull: apptainer's progress bar panics on this image; no tty = no panic
apptainer pull $WORK/containers/jax_26.06-py3.sif docker://nvcr.io/nvidia/jax:26.06-py3 > $HOME/pull.log 2>&1
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

GitHub auth: **do NOT run `ssh-keygen` on Vista** -- TACC auto-generates
the `~/.ssh` key pair the batch system uses for intra-node SSH; making your
own breaks job launch (recovery: `mv .ssh dot.ssh.old`, log out, log back
in). Instead forward your laptop's agent (`ForwardAgent yes` in the laptop's
`~/.ssh/config` for the Vista host), or clone over HTTPS with a PAT.

```bash
mkdir -p $STOCKYARD/repos && cd $STOCKYARD/repos
git clone git@github.com:KinematicLensing/kl_roman_pipe.git   # via forwarded agent
cd kl_roman_pipe && git checkout se/speedups   # or the commit you want benchmarked
```

(sweveret layout: `$STOCKYARD=/work/09102/sweveret`, machine-independent;
`run_vista.slurm` is pre-filled with these paths, queue `gh-dev`, and
charge code `JPL-SPHEREx`.)

The kit records the git commit in the output JSON -- keep the tree clean.

## 3. Install the missing python deps

The NGC JAX container ships jax/jaxlib/CUDA + numpy/scipy. kl_pipe's
inference path additionally needs: **numpyro, astropy, pyyaml, matplotlib**
(scipy usually present -- verify; matplotlib IS needed -- bench section a
imports it at module load). NOT needed: galsim (kit avoids it),
emcee/nautilus/blackjax/arviz (lazy imports, unused here).

Install into a bind-mounted target dir (containers are read-only; `-B` binds
it in so pip --target can write from inside the container):

```bash
mkdir -p $WORK/klpipe_pipdeps
apptainer exec -B $WORK/klpipe_pipdeps $WORK/containers/jax_26.06-py3.sif \
  python -m pip install --no-cache-dir --target $WORK/klpipe_pipdeps \
  numpyro astropy pyyaml matplotlib
# numpyro drags in a jax/jaxlib/CUDA-plugin set that mismatches + shadows the
# container's GPU jax; strip it so only the container's stack is used:
rm -rf $WORK/klpipe_pipdeps/jax* $WORK/klpipe_pipdeps/nvidia*
```

If scipy turns out to be missing from the image, add it to that list.
`provision_vista.sh` does all of the above (incl. the jax/CUDA strip)
automatically.

kl_pipe itself is used straight from the repo via
`PYTHONPATH=$REPO:$PIPDIR` (no `pip install -e .` needed -- avoids
pulling the full dependency tree; the kit only imports the inference
modules).

## 4. Sanity checks (compute node -- idev, NOT the login node)

Grab an interactive GPU node first (container execution is banned on login
nodes):

```bash
idev -p gh-dev -N 1 -n 1 -t 01:00:00
```

Then, on the compute node (idev does NOT inherit the module -- load it here):

```bash
module load tacc-apptainer
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

## 7. Node-packing benchmark (section j) -- GPU vs CPU platform decision

Section j answers the paper-platform question: a single fit underutilizes
the GH200 ~5-7x (section g) and uses one core of a CPU node, so the number
that matters is **fits/node-hour at pack width N**, not single-fit speed.
It launches N simultaneous child processes (each a real section-i
precond+NUTS fit) and reports the makespan-aggregated throughput per sweep
point, plus per-child wallclock spread, peak RSS, and returncodes.

Three run modes:

| where | script | pinning |
|---|---|---|
| Stampede3 SPR (112-core CPU) | `run_stampede3_pack.slurm` | `taskset -c k` per child, BLAS/OMP=1 |
| Vista GH200 (GPU packing) | `run_pack_vista.slurm` | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85/N` (+MPS if available) |
| local smoke (macOS) | see below | none (recorded in JSON) |

Stampede3 setup (no container -- x86 wheels install directly):

```bash
# login node, once:
bash provision_stampede3.sh        # venv at $WORK/klpipe_cpu_venv
# then from vista_kit/:
sbatch run_stampede3_pack.slurm    # Q sweep 1,56,112 + P sweep 1,112
```

Local smoke (2 workers, tiny fit, ~2 min):

```bash
python bench_matrix.py --sections j --pack-configs Q --pack-workers 1,2 \
  --fit-samples 50 --fit-warmup 25 --fit-chains 2 --nreps 3
```

Numbers to bring home: `fits_per_node_hour` at each N vs the GH200 serial
reference (P = 20.2 fits/node-hr warm-cache, Q = 71), the N=112 vs N=1
per-child slowdown (memory-bandwidth contention), per-child `peak_rss_mb`
at full pack (128 GB node / 112 procs ~ 1.1 GB each is the ceiling), and
on the GPU side whether N=4-8 packing recovers the section-g headroom.
Divide by the machine's SU rate per node-hour for the final fits/SU
comparison -- that decides the production platform.

Notes:
- Always pass `--jax-cache-dir` (the SLURM scripts do): children share
  compiles; the first sweep point doubles as the cache warmer.
- Children default to the SAME NUTS seed (clean contention measurement:
  identical work per child). `--pack-vary-seeds` gives per-child seeds for
  a realistic trajectory-length spread instead.
- `--pack-vary-data` gives each child a DISTINCT mock-noise realization.
  Data arrays are baked into the jitted posterior as constants, so same-data
  children share compiles via the cache while distinct-data children each
  recompile -- exactly the per-galaxy recompile cost a production ensemble
  pays. Same-data = clean contention curve; vary-data = honest production
  throughput. The SLURM scripts use vary-data for the P headline sweeps.
- A dead child (OOM at N=112) is recorded with its returncode + log path
  and excluded from `fits_completed`; throughput stays honest.

## 8. Known limitations / notes

- Section b `parallel` on a single GPU is skipped by design (recorded in
  JSON). For multi-GPU chain parallelism use a multi-node/array setup later.
- The kit intercepts kl_pipe's import-time x64 forcing in fp32 mode and
  monkeypatches the PSF precompute for the shim PSF only -- both are
  loud (printed) and kit-local; repo behavior elsewhere is untouched.
- P-config task construction renders 4 rolls of synthetic data first;
  a few minutes of compile-heavy setup before timings start is normal.
- Numbers to trust: `min_ms` (min-of-N). `mean_ms` is reported for
  thermals/contention context.
