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

- `bank.stratify`: the plot axis (v1: cos i only). Uniform bins over the
  range; truth = bin centers. Uniform-in-cos-i IS the random-orientation
  population, so the collapsed average needs no reweighting.
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
  `pin_z_to_truth: true` (v1 -- sampled narrow-spec-z planned).
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
- the spec's `broadband_snr`/`grism_snr` knobs and the ObservingConfig
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
