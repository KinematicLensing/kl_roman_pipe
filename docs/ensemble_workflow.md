# Ensemble fitting workflow

How to declare, run, and collate an MCMC fit campaign with
`kl_pipe.ensemble`. Design rationale lives in
`docs/plans/ensemble_pipeline_plan.md`; this is the operator's guide.

## Concepts

One *campaign* = one ensemble spec YAML = one `run_name` = one observing
config. The spec declares a galaxy bank (stratified cos i grid + per-galaxy
drawn truths + fixed constants), a shear scheme, SNR knobs, fit settings, and
dispatch/output settings. The expander resolves it into a flat
`manifest.parquet` (one row per fit, fully-resolved truth, deterministic
`fit_id`), which workers consume via a lock-free filesystem ledger.

```
ensemble_spec.yaml ──expand──► runs/<run_name>/
                                 manifest.parquet      (immutable)
                                 provenance/           (spec + observing-config
                                                        snapshot + hashes + git)
                                 status/{claims,done,failed}/
                                 results/<fit_id>.parquet   (one per fit)
                                 chains/ mocks/         (retention subsets)
                    ──run────► workers: claim -> mock -> NUTS -> result
                    ──collate► results.parquet + tally
```

## Commands

```bash
# 1. expand the spec into a run directory
python -m kl_pipe.ensemble expand configs/ensembles/sigma_eps_cosi_dev.yaml \
    --runs-dir runs

# 2a. run locally (serial, or --workers N concurrent worker processes)
python -m kl_pipe.ensemble run --run-dir runs/sigma_eps_cosi_dev --workers 2

# 2b. or emit + submit a SLURM job array (Vista/Stampede)
python -m kl_pipe.ensemble slurm --run-dir runs/sigma_eps_cosi_dev
sbatch runs/sigma_eps_cosi_dev/submit.slurm

# 3. check status between rounds
python -m kl_pipe.ensemble status --run-dir runs/sigma_eps_cosi_dev

# 4. recover: release stale claims (and optionally failed markers), re-run
python -m kl_pipe.ensemble reclaim --run-dir runs/sigma_eps_cosi_dev --clear-failed
python -m kl_pipe.ensemble run --run-dir runs/sigma_eps_cosi_dev

# 5. collate per-fit results into the catalog
python -m kl_pipe.ensemble collate --run-dir runs/sigma_eps_cosi_dev
```

Re-running a campaign only picks up `never_run ∪ failed(after reclaim) ∪
stale(after reclaim)` fits: done markers make completed fits idempotent.
The worker entrypoint is identical on a laptop and on a cluster node; the
SLURM script is only a launcher.

## Spec anatomy

See `configs/ensembles/sigma_eps_cosi_dev.yaml` for a complete example.
Key blocks:

- `bank.stratify`: the plot axis. Two kinds:
  - truth stratification (`cosi: {n_bins, range}`, measurement
    `sigma_eps_vs_cosi`): uniform bins, truth = bin centers, an independent
    galaxy bank per bin (the property IS the galaxy). Uniform-in-cos-i IS
    the random-orientation population, so the collapsed average needs no
    reweighting.
  - config sweep (`line_snr: {values: [30, 50, 100]}`, measurement
    `sigma_eps_vs_line_snr`): ONE galaxy bank shared across every value
    with the same noise seed (common random numbers -- the knob is external
    to the galaxy, so shared noise cancels in the trend). cosi moves to
    `bank.draw`; the top-level `line_snr` scalar must be omitted. `line_snr`
    is the emission-LINE matched-filter SNR (continuum is marginalized and
    does not enter the SNR; see kl_pipe/noise.py grism_line_noise).
- `bank.draw`: per-galaxy truths, seeded deterministically from the global
  `seed`. Supported dists: `uniform`, `lognormal_tf` (Tully-Fisher:
  `center_kms`, `sigma_tf_dex`; 0.08 dex fiducial per Xu+2022/Pranjal+2022).
  Keys may be shared params (`theta_int`, `z`), the `vcirc` alias
  (-> `vel.vcirc`), or dotted source-model names.
- `bank.fixed`: constants. `h_over_r`/`x0`/`y0` broadcast to every scene
  component; anything else must be a dotted name (`flux` is rejected as
  ambiguous). Everything not drawn/stratified/fixed comes from the flagship
  scene defaults in `kl_pipe/ensemble/scene.py`.
- `shear`: `fixed` (sigma_eps campaigns) or `grid` + `component` (bias m, c
  campaigns; one component at a time).
- `ring.enabled`: 2-member 90-degree pairs for bias campaigns (identical
  intrinsics + shear, theta rotated 90 degrees, independent noise).
- `observed_config`: id into `configs/observing/` (structural instrument
  setup). SNR knobs stay in the spec.
- `fit`: numpyro NUTS settings; `precondition: laplace` recommended;
  `n_map_starts` random MAP starts (plus 4 automatic PA-stratified starts --
  the position-angle basins are the known multimodality, and random draws
  alone can land every start in the wrong basin, whose shape-shear-
  compensated mode traps the sampler); `pin_z_to_truth: true` (v1 -- sampled
  narrow-spec-z planned). A fit whose chains come back broken (max_rhat >
  1.1 or divergence rate > 0.9) is retried once with a fresh sampler seed;
  `n_attempts` is recorded in the summary row.
- `output.save_chains/save_mocks`: `none | subset | all`. `subset` = the
  first galaxy of each cos-i bin; chains -> `chains/<fit_id>.npz`, mock
  datavectors + truth/MAP renders -> `mocks/<fit_id>.npz`.

## Fit priors (self-consistency)

The fit prior for every drawn parameter IS its generating distribution
(uniform theta_int, population TF LogNormal on vcirc -- centered on the
population median, NOT the per-galaxy truth, because the population is what
an analyst actually knows); stratified cos i gets the population prior over
the stratify range; injected shear gets the wide uninformative
Gaussian(0, 0.1) (flagship convention -- posterior widths must reflect the
data's shear constraint, not the prior); nuisance parameters keep flagship
prior widths centered on the scene truth; z is pinned to the per-fit truth.
Fit == truth by construction, so recovered biases are pipeline biases, not
prior mismatch.

## Architecture: which layers are generic, which are paper-specific

```
                          GENERIC (datavector-agnostic)
  ┌─────────────────────────────────────────────────────────────────────┐
  │ spec.py      EnsembleSpec: bank (stratify/draw/fixed roles), shear, │
  │              ring, fit, dispatch, output                            │
  │ expander.py  roles -> manifest rows; seed streams (galaxy/noise/    │
  │              sampler); CRN rule; fit_id; provenance snapshot        │
  │ ledger.py    atomic claims + done/failed markers + derived status   │
  │ dispatch.py  local N-worker backend; submit.slurm emission          │
  │ worker.py    claim -> build -> fit -> per-fit parquet loop; summary │
  │              row (per-sampled-param stats + quality columns)        │
  │ collate.py   results.parquet + manifest join + tally                │
  │ calibration  pure numpy post-processing                             │
  └───────────────────────────┬─────────────────────────────────────────┘
                              │ the extension seam:
                              │ (truth, noise_seed, spec, config)
                              │        -> FitInputs(source, priors,
                              │                     image_obs, grism_obs)
  ┌───────────────────────────┴─────────────────────────────────────────┐
  │           PAPER-SPECIFIC "datavector recipe" (v1: Roman Q/P)        │
  │ ObservingConfig  schema: bands / grism rolls / lines / psf / stamp  │
  │ scene.py         truth defaults + prior rules + SourceModel builder │
  │ mocks.py         model-rendered mock + matched-filter noise +       │
  │                  gaussian PSF -> ImageObs/GrismObs per channel      │
  └─────────────────────────────────────────────────────────────────────┘
```

Everything above the seam manipulates only (a) manifest rows -- fit_id,
integer indices, `truth.*` columns with arbitrary dotted names, seeds -- and
(b) parquet/npz files keyed by fit_id. Nothing generic knows what a band or
a grism is; `truth.*` columns are opaque dotted parameter names.

The seam itself is `mocks.build_fit_inputs(truth, noise_seed, spec, config)`
-> `FitInputs`, consumed by the worker as
`InferenceTask.from_obs(source, priors, image_obs=..., grism_obs=...)`.
`from_obs` already accepts any combination of image/grism/velocity obs, so a
different datavector type (e.g. velocity-map ensembles, IFU cubes,
different instrument) needs exactly three things: an observing-config schema
for it, a scene (truth defaults + priors + SourceModel), and a mock builder
returning FitInputs. Dispatch, ledger, manifest, collation, and calibration
are untouched.

Known v1 leaks across the seam (deliberate, small, all loud):
- the spec's `broadband_snr`/`line_snr` knobs and the ObservingConfig
  schema are imaging+grism-shaped (a generic version would move SNR knobs
  into a per-recipe block);
- the worker imports `build_fit_inputs` directly instead of resolving a
  recipe from config (a one-line indirection when a second recipe exists);
- `ess_g1`/`ess_g2` quality columns assume shear params exist (NaN
  otherwise).
Generalizing = introducing a recipe registry keyed from the observing
config; deferred until a second datavector type actually exists.

## Noise model (why seeds are assigned the way they are)

The expander draws one noise seed per `(galaxy, ring_member, noise_rep)` and
holds it constant across the shear grid:

- **Common random numbers (CRN) for contrasts.** When the plot axis is an
  external knob applied to the same galaxy (shear grid -> bias slope m;
  config sweeps -> delta sigma_eps), reusing one noise realization per galaxy
  across the axis cancels correlated noise in the slope/difference. This is
  standard variance reduction, not cheating: the honest error bar on m comes
  from galaxy-to-galaxy scatter, and noise is fully sampled across the
  population [Pujol+2020, A&A 641 A164; Euclid Collab. 2024,
  arXiv:2401.08239].
- **Independent noise for absolutes and ring pairs.** sigma_eps is an
  absolute ensemble average over independently-drawn galaxies; ring pairs
  cancel intrinsic shape (a galaxy property, not noise), so shared noise
  would only correlate the two fits and corrupt error propagation
  [Nakajima & Bernstein 2007; Voigt & Bridle 2010, arXiv:0905.4801;
  Mandelbaum+2015 (GREAT3), MNRAS 450 2963].
- **Intrinsic-property axes (cos i, theta_int, v_circ) cannot use CRN** --
  the property is the galaxy, so different bins hold different galaxies.

Verify citations against the papers before quoting them in the manuscript.

## Running on Vista (TACC)

The worker is backend-agnostic; on Vista it runs inside the NGC JAX
container with the pip sidecar (galsim, pandas, pyarrow, corner included by
`experiments/sweverett/vista_kit/provision_vista.sh`).

One-time setup (inside `idev -p gh-dev -N 1 -n 1 -t 01:00:00`; idempotent,
also repairs after a $SCRATCH purge):

```bash
cd $STOCKYARD/repos/kl_roman_pipe && git pull
bash experiments/sweverett/vista_kit/provision_vista.sh
```

Define the container launcher (both idev and batch use it). Shortcut:

```bash
source experiments/sweverett/vista_kit/env_vista.sh
```

which is equivalent to:

```bash
module load tacc-apptainer
export KLPIPE_PYTHON="apptainer exec --nv \
  -B $STOCKYARD/repos/kl_roman_pipe -B $WORK/klpipe_pipdeps -B $SCRATCH \
  --env PYTHONPATH=$STOCKYARD/repos/kl_roman_pipe:$WORK/klpipe_pipdeps \
  --env LD_PRELOAD=$WORK/klpipe_pipdeps/galsim/libfftw3.so.3 \
  --env JAX_COMPILATION_CACHE_DIR=$SCRATCH/jax_cache \
  $WORK/containers/jax_26.06-py3.sif python"
```

(the from-source galsim finds fftw via `LD_PRELOAD` of the copy provision
embeds in the sidecar -- no fftw bind or `LD_LIBRARY_PATH` needed.)

**idev micro-run** (an idev node is just a local machine -- use the local
backend; no SLURM machinery involved):

```bash
idev -p gh-dev -N 1 -n 1 -t 01:00:00
cd $STOCKYARD/repos/kl_roman_pipe
# (re-export KLPIPE_PYTHON as above -- idev starts a fresh shell)
$KLPIPE_PYTHON -m kl_pipe.ensemble expand \
    configs/ensembles/sigma_eps_cosi_dev.yaml --runs-dir $SCRATCH/kl_runs
$KLPIPE_PYTHON -m kl_pipe.ensemble run \
    --run-dir $SCRATCH/kl_runs/sigma_eps_cosi_dev --max-fits 2
$KLPIPE_PYTHON -m kl_pipe.ensemble status \
    --run-dir $SCRATCH/kl_runs/sigma_eps_cosi_dev
```

**Node job** (the 150-fit statistics run; spec sets gh-dev queue, account,
8-worker GPU packing -- the emitted script slices HBM per worker and
staggers jax inits). Apptainer NO-OPS on Vista login nodes, so run the
expand + script emission inside the same idev session as the micro-run:

```bash
# inside idev (same session as the micro-run is fine)
$KLPIPE_PYTHON -m kl_pipe.ensemble expand \
    configs/ensembles/sigma_eps_cosi_dev_vista.yaml --runs-dir $SCRATCH/kl_runs
$KLPIPE_PYTHON -m kl_pipe.ensemble slurm \
    --run-dir $SCRATCH/kl_runs/sigma_eps_cosi_dev_vista
# edit submit.slurm: uncomment the KLPIPE_PYTHON block (paths as above), then
# from the LOGIN node:
sbatch $SCRATCH/kl_runs/sigma_eps_cosi_dev_vista/submit.slurm
squeue -u $USER
# progress without the container (plain filesystem check, login-node safe):
ls $SCRATCH/kl_runs/sigma_eps_cosi_dev_vista/status/done | wc -l   # /150
# when done (inside idev; resubmit/run again picks up any remainder):
$KLPIPE_PYTHON -m kl_pipe.ensemble collate \
    --run-dir $SCRATCH/kl_runs/sigma_eps_cosi_dev_vista
$KLPIPE_PYTHON -c "from kl_pipe.ensemble.diagnostics import run_report; \
    run_report('$SCRATCH/kl_runs/sigma_eps_cosi_dev_vista')"
# pull the report + catalog back for local inspection
scp -r vista:'$SCRATCH/kl_runs/sigma_eps_cosi_dev_vista/{diagnostics,results.parquet,manifest.parquet}' .
```

Caveats: run dirs live on `$SCRATCH` (purged after ~10 idle days -- collate
and copy off promptly); any command through `$KLPIPE_PYTHON` must run on a
compute node (idev or batch), never a login node.

## Outputs

`results.parquet` (via `collate`) holds one row per fit: recovered posterior
(mean/std/median per sampled param), the Laplace MAP, and inclusive quality
columns (`max_rhat`, `min_ess`, `ess_g1/g2`, `n_divergences`,
`divergence_rate`, `mean_accept_prob`, `num_steps_total`,
`n_map_starts_converged`, `precond_condition_number`,
`map_minus_postmean_over_sigma.*`, wallclocks). Gate policy (e.g. mask
`max_rhat > 1.01 OR min_ess < 400 OR divergence-rate outlier`) is applied
post hoc in analysis -- nothing is filtered at write time.

Join truth with recovery via `kl_pipe.ensemble.collate.analysis_table(run_dir)`
(manifest ⨝ results on `fit_id`), then feed `kl_pipe/calibration.py`:
`rotate_to_galaxy_frame`, `measure_shear_bias`, `compute_shape_noise`.

## Catalog-mode runs

Catalog-backed specs (`population.type: catalog`, e.g.
`configs/ensembles/flagship2_shear_dev.yaml`) reuse the same
expand -> run/slurm -> collate flow. Everything catalog-specific (raw
schema, unique row key, preprocess to standardized columns, catalog-fitted
prior constants) lives in a `kl_pipe/ensemble/catalogs/` adapter selected by
`population.catalog.kind` (default `flagship2`); the selection, paint, and
expansion machinery is catalog-agnostic. The only operational differences
from sampled mode:

1. **Catalog data must be present** at the spec's `population.catalog.data_dir`
   (default `data/cosmohub/<download>.parquet`). Download it with the
   idempotent, content-hashed helper -- it writes next to its query-spec YAML
   (i.e. into `data/cosmohub/`) and authenticates via `~/.netrc` for
   `api.cosmohub.pic.es`:
   ```bash
   python scripts/download_cosmohub.py data/cosmohub/flagship2_dev.yaml
   # resume an already-ready CosmoHub query instead of resubmitting:
   python scripts/download_cosmohub.py data/cosmohub/flagship2_v1.yaml --query-id <ID>
   ```
   The Q1 validation anchor is `python scripts/download_q1.py` (IRSA TAP).

2. **`expand` auto-materializes the population.** For a catalog spec, `expand`
   builds `population.parquet` (adapter catalog rows + isotropic cos i redraw
   + Tully-Fisher / sigma0 paint + bulge morphology paint + ring-pair shear)
   *and* `manifest.parquet` in one step -- no separate command. Broadband
   bands render as bulge+disk; the grism line + continuum stay single-disk.

3. **Bulge morphology is painted, not taken from the catalog.** Flagship2
   assigns the bulge Sersic index and bulge size as uncorrelated random draws
   (its own Sect. 7.1 caveat), so the population step replaces them with
   literature-anchored distributions (`population.py` `BULGE_*` constants:
   pseudobulge/classical n mixture + a capped lognormal bulge-to-disk size
   ratio); only the calibrated catalog `bulge_fraction` is kept, and the
   science selection is disk-dominated, `selection.bulge_fraction_max: 0.3`.
   The fit samples bulge fraction + size under priors equal to these
   generating distributions. None of this changes the download: the paint
   consumes existing catalog columns, so previously downloaded parquets stay
   valid (catalog values are retained in `catalog_*` columns for validation).

Everything else (`run`, `slurm`, `status`, `collate`, diagnostics) is
identical to the sampled-mode commands above.

### Vista quick-start (catalog mode)

In dependency order; steps 1-2 need no code update and can start immediately.

1. **Login node -- catalog downloads** (compute nodes have no external
   network; needs `~/.netrc` for `api.cosmohub.pic.es`). With the repo on
   `$STOCKYARD`, the helper writes into the repo's `data/cosmohub/`:
   ```bash
   cd $STOCKYARD/repos/kl_roman_pipe
   python scripts/download_cosmohub.py data/cosmohub/flagship2_dev.yaml   # ~40 MB, minutes
   python scripts/download_cosmohub.py data/cosmohub/flagship2_v1.yaml    # ~1 GB; async Hive job
   ```
   The v1 job is asynchronous on the CosmoHub side (tens of minutes) --
   start it early; if the terminal drops, resume with `--query-id <ID>`
   (printed at submission). Only `flagship2_dev` is needed for the dev spec;
   `flagship2_v1` is the production row bank.
2. **Login node -- code**: `git pull` the target branch on
   `$STOCKYARD/repos/kl_roman_pipe` (catalog runs: `se/ensemble`).
3. **idev node -- provision + micro-run**: provision (idempotent), source the
   launcher, then the dev catalog spec end to end:
   ```bash
   idev -p gh-dev -N 1 -n 1 -t 01:00:00
   cd $STOCKYARD/repos/kl_roman_pipe
   bash experiments/sweverett/vista_kit/provision_vista.sh
   source experiments/sweverett/vista_kit/env_vista.sh
   $KLPIPE_PYTHON -m kl_pipe.ensemble expand \
       configs/ensembles/flagship2_shear_dev.yaml --runs-dir $SCRATCH/kl_runs
   $KLPIPE_PYTHON -m kl_pipe.ensemble run \
       --run-dir $SCRATCH/kl_runs/flagship2_shear_dev --max-fits 2
   $KLPIPE_PYTHON -m kl_pipe.ensemble collate \
       --run-dir $SCRATCH/kl_runs/flagship2_shear_dev
   ```
   Check the collated `max_rhat` / `divergence_rate` before scaling up.
4. **Node job**: as in the sampled-mode Vista section above (`slurm` emit from
   idev, `sbatch` from login; spec dispatch block already carries queue
   `gh-dev` + account `JPL-PUB`).

### Local end-to-end dry-run (faithful pre-TACC sanity check)

The unit/smoke tests cover the forward model and a log-prob+grad smoke, but
**not** the full dispatch -> worker(NUTS) -> collate loop. To exercise that
exact machinery locally before submitting on TACC -- same code path, just few
galaxies and a tiny sampler -- run the catalog dev spec end to end in the
`klpipe` conda env:

```bash
python -m kl_pipe.ensemble expand configs/ensembles/flagship2_shear_dev.yaml \
    --runs-dir runs                              # builds population + manifest
python -m kl_pipe.ensemble run    --run-dir runs/flagship2_shear_dev --max-fits 2
python -m kl_pipe.ensemble status --run-dir runs/flagship2_shear_dev
python -m kl_pipe.ensemble collate --run-dir runs/flagship2_shear_dev
python -c "from kl_pipe.ensemble.diagnostics import run_report; \
    run_report('runs/flagship2_shear_dev')"
```

This is the recommended first move: it flushes out per-fit cost with
BulgeDisk, NUTS convergence with the bulge nuisances, and any collate/report
wiring before any TACC time is spent. On Vista/Stampede the same commands run
under `$KLPIPE_PYTHON` (see the container launcher above); the catalog parquet
must be in `$STOCKYARD/repos/kl_roman_pipe/data/cosmohub/`, which is where
`download_cosmohub.py` writes when the repo is checked out on `$STOCKYARD`
(run it from a **login node** -- compute nodes have no external network).
