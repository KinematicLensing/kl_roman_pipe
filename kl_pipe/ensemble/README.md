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

## Module map

```
spec.py         EnsembleSpec + ObservingConfig registry (strict YAML validation)
scene.py        canonical galaxy scene: truth defaults + fit prior rules
expander.py     spec -> manifest.parquet (deterministic seeds, CRN rule,
                fit_id, provenance snapshot)
mocks.py        truth row -> noisy mock observations (model-rendered,
                matched-filter noise, galsim Gaussian PSFs)
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
