# kl_pipe.ensemble

Config-driven MCMC fit campaigns: declare an ensemble in one spec YAML,
expand it into a per-fit manifest, dispatch it (locally or as a SLURM array),
collate results, and diagnose the run.

Full operator guide (spec anatomy, noise model rationale, architecture,
TACC runbooks): **`docs/ensemble_workflow.md`**.

## Quickstart

```bash
python -m kl_pipe.ensemble expand configs/ensembles/sigma_eps_cosi_dev.yaml --runs-dir runs
python -m kl_pipe.ensemble run     --run-dir runs/sigma_eps_cosi_dev --workers 2
python -m kl_pipe.ensemble status  --run-dir runs/sigma_eps_cosi_dev
python -m kl_pipe.ensemble collate --run-dir runs/sigma_eps_cosi_dev
python -c "from kl_pipe.ensemble.diagnostics import run_report; run_report('runs/sigma_eps_cosi_dev')"
```

Re-running a campaign is idempotent: done fits are skipped, so `run` after a
crash/timeout picks up only the remainder (`reclaim` first if claims went
stale or fits failed).

## Vista (TACC) end-to-end worked example

The complete session for a statistics run (here `sigma_eps_gsnr_P_vista`;
swap the run name for any spec in `configs/ensembles/`). Prerequisite (once,
or after a `$SCRATCH` purge): `bash experiments/sweverett/vista_kit/provision_vista.sh`
inside idev.

```bash
# ---- login node -------------------------------------------------------
cd $STOCKYARD/repos/kl_roman_pipe && git pull
idev -p gh-dev -N 1 -n 1 -t 00:30:00

# ---- inside idev ------------------------------------------------------
cd $STOCKYARD/repos/kl_roman_pipe
source experiments/sweverett/vista_kit/env_vista.sh   # defines KLPIPE_PYTHON
# (equivalently: eval "$(make -s env-vista)")
export RUN=sigma_eps_gsnr_P_vista

# expand -> manifest; slurm -> submit.slurm (BOTH steps are required;
# expand alone writes no submit script)
$KLPIPE_PYTHON -m kl_pipe.ensemble expand configs/ensembles/$RUN.yaml --runs-dir $SCRATCH/kl_runs
$KLPIPE_PYTHON -m kl_pipe.ensemble slurm  --run-dir $SCRATCH/kl_runs/$RUN
# must print "baked KLPIPE_PYTHON launcher into script: ..." -- if it warns
# the var was unset, re-source env_vista.sh and re-emit

# optional micro-run (1 fit, single in-process worker, full GPU):
$KLPIPE_PYTHON -m kl_pipe.ensemble run --run-dir $SCRATCH/kl_runs/$RUN --max-fits 1 --workers 1

# ---- login node -------------------------------------------------------
sbatch $SCRATCH/kl_runs/$RUN/submit.slurm
squeue -u $USER
ls $SCRATCH/kl_runs/$RUN/status/done | wc -l          # progress, container-free

# stragglers (failed fits / stale claims from killed tasks): inside idev
#   $KLPIPE_PYTHON -m kl_pipe.ensemble reclaim --run-dir $SCRATCH/kl_runs/$RUN --clear-failed
# then sbatch the SAME submit.slurm again -- it sweeps only the remainder.

# ---- when done: collate + report (inside idev) ------------------------
$KLPIPE_PYTHON -m kl_pipe.ensemble collate --run-dir $SCRATCH/kl_runs/$RUN
$KLPIPE_PYTHON -c "from kl_pipe.ensemble.diagnostics import run_report; run_report('$SCRATCH/kl_runs/$RUN')"
# prints headline sigma_eps; writes diagnostics/ (sigma_eps.png = the
# measurement plot, recovery/pulls/quality plots + CSVs, per-fit corner and
# datavector deep dives for flagged + representative fits)

# ---- from your laptop: pull products (SCRATCH purges after ~10 idle days)
rsync -av "vista.tacc.utexas.edu:/scratch/09102/sweveret/kl_runs/$RUN/{diagnostics,results.parquet,manifest.parquet}" $RUN/
# add {chains,mocks} to re-plot corners/datavectors locally via
# diagnostics.plot_corner_fit / plot_datavector_fit
```

Gotchas that have bitten before:
- `expand` does NOT write `submit.slurm`; the `slurm` subcommand does.
- sbatch does not inherit the idev shell's exports; the emitted script works
  because emission bakes `KLPIPE_PYTHON` in (script preflights the import
  and fails loud otherwise).
- Never run `$KLPIPE_PYTHON` on a login node (apptainer no-ops there).
- gh-dev QOS rejects job arrays while an idev session is running
  (`QOSMaxSubmitJobPerUserLimit`) -- statistics runs use `queue: gh`;
  gh-dev is for idev shakedowns only.
- `observed_config: canonical_Q` (1 band, 1 roll) is the shakedown config
  only -- it cannot constrain vcirc or break g2/theta_int; measurement runs
  need `canonical_P` (2 bands, 4 rolls).

## Module map

```
spec.py         EnsembleSpec + ObservingConfig registry (strict YAML validation)
scene.py        canonical galaxy scene: truth defaults + fit prior rules
expander.py     spec -> manifest.parquet (deterministic seeds, CRN rule,
                fit_id, provenance snapshot)
mocks.py        truth row -> noisy mock observations (model-rendered,
                matched-filter noise; PSF per config: galsim Gaussian or
                Roman WFI via galsim.roman.getPSF, z-dependent grism kernel)
worker.py       claim -> mock -> NUTS (Laplace precond, PA-stratified MAP
                starts, one retry on broken chains) -> per-fit parquet
ledger.py       lock-free filesystem status (atomic-mkdir claims,
                done/failed markers, stale detection)
dispatch.py     local N-worker backend + submit.slurm emission
collate.py      per-fit results -> results.parquet + manifest join + tally
diagnostics.py  quality gates, recovery pulls, sigma_eps tables + plots
__main__.py     CLI: expand | run | worker | status | collate | slurm | reclaim
```

Generic vs paper-specific: everything except `scene.py`, `mocks.py`, and the
`ObservingConfig` schema is datavector-agnostic; those three form the
"datavector recipe" for the Roman imaging+grism paper (see the architecture
section of `docs/ensemble_workflow.md` for the extension seam).
