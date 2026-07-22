"""
Ensemble fitting pipeline: declare a fit campaign in one spec file, expand it
into a per-fit manifest, dispatch it (locally or as a SLURM array), and
collate per-fit results into a single catalog.

Modules
-------
spec       Ensemble spec + observation-config registry (YAML, validated)
scene      Canonical galaxy scene: truth defaults + fit prior rules
population Catalog-backed population (Flagship2 rows + kinematic paint)
expander   Deterministic spec -> manifest.parquet expansion
mocks      Per-fit on-the-fly mock observation construction
worker     Claim -> mock -> fit -> per-fit-result loop
ledger     Filesystem-derived status markers (done/claims/failed)
dispatch   Local multi-worker backend + SLURM array emission
collate    Merge per-fit results -> results.parquet + analysis table

Usage
-----
python -m kl_pipe.ensemble expand path/to/ensemble_spec.yaml
python -m kl_pipe.ensemble run --run-dir runs/<run_name>
python -m kl_pipe.ensemble status --run-dir runs/<run_name>
python -m kl_pipe.ensemble collate --run-dir runs/<run_name>
"""

from kl_pipe.ensemble.spec import (
    CatalogPopulationSpec,
    EnsembleSpec,
    ObservationConfig,
    PSFSpec,
)

__all__ = ['CatalogPopulationSpec', 'EnsembleSpec', 'ObservationConfig', 'PSFSpec']
