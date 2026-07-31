"""
Catalog adapter interface: everything catalog-specific behind one seam.

A ``CatalogAdapter`` owns the raw file schema, the unique row key, the
preprocess step that maps raw columns to the standardized contract, and the
catalog-fitted prior constants. The generic population machinery
(``population.build_population``) consumes only the contract columns and the
adapter's declared capabilities, so a new input catalog is one new adapter
module plus a spec ``catalog.kind`` value -- no changes to the selection,
paint, or expansion chain.

Contract columns (post-preprocess)
----------------------------------
Required for every catalog:

- ``z``                  true redshift
- ``logm``               physical log10 stellar mass
- ``log_sfr``            log10 star formation rate (diagnostics)
- ``f_line_cgs``         integrated Halpha line flux [erg/s/cm2]
- ``lambda_obs_a``       observed Halpha wavelength [Angstrom]
- ``f_lambda_cont_cgs``  continuum density at lambda_obs [erg/cm2/s/A]
- ``ew_rest_a``          rest-frame Halpha equivalent width [A]
- ``rscale_arcsec``      disk exponential scale length [arcsec]
- ``disk_r50``           disk half-light radius [arcsec]

Plus the adapter's ``id_columns`` (integer, jointly unique; their order is
the seed-key order and is part of the determinism contract), ``z_obs`` when
``has_observed_redshift``, the bulge trio (``bulge_fraction``, ``bulge_r50``,
``bulge_nsersic``) when ``has_bulge``, and the source columns named by
``validation_columns``. ``validate_contract`` enforces exact set equality --
a column the adapter forgot, or one it leaked through, is a loud error.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, TYPE_CHECKING, Union

import pandas as pd

if TYPE_CHECKING:
    from kl_pipe.ensemble.spec import CatalogPopulationSpec

# contract columns every adapter's preprocess must produce (see module doc)
REQUIRED_CONTRACT_COLUMNS = (
    'z',
    'logm',
    'log_sfr',
    'f_line_cgs',
    'lambda_obs_a',
    'f_lambda_cont_cgs',
    'ew_rest_a',
    'rscale_arcsec',
    'disk_r50',
    # per-band Roman imaging fluxes [uJy], line-inclusive (what a broadband
    # image of the source actually contains), interpolated from the catalog's
    # own photometry to the Roman effective wavelengths
    'flux_f106_ujy',
    'flux_f129_ujy',
    'flux_f158_ujy',
)

# contract columns present only when the adapter declares bulge support
BULGE_CONTRACT_COLUMNS = ('bulge_fraction', 'bulge_r50', 'bulge_nsersic')


@dataclass(frozen=True)
class CatalogPriorConstants:
    """Catalog-fitted scene-prior constants (see each adapter for provenance).

    These are the numbers that must be refit when the input catalog changes:
    population distributions of the selected sample (size, continuum
    amplitude, bulge fraction) and the prior support bounds sized against
    the catalog's reachable truths. Bulge fields are None for a catalog
    without bulge columns.
    """

    # disk scale length: log10 median/scatter of the selected sample, and
    # support bounds clearing the painted size products
    rscale_log10_mu: float
    rscale_log10_sigma: float
    rscale_low: float
    rscale_high: float
    # continuum amplitude [1e-17 erg/s/cm2 per nm]
    cont_flux_log10_mu: float
    cont_flux_log10_sigma: float
    cont_flux_low: float
    cont_flux_high: float
    # measurement-prior support bounds: line flux [1e-17 erg/s/cm2] and
    # band flux [uJy], generous margins around the selected sample's reach
    # (the priors are simulated measurements at SNR >~ 20, so the bounds
    # only guard support, never shape the posterior)
    line_flux_low: float
    line_flux_high: float
    band_flux_low: float
    band_flux_high: float
    # bulge fraction + bulge half-light-radius support (None = no bulge)
    bulge_frac_loc: Optional[float] = None
    bulge_frac_scale: Optional[float] = None
    bulge_hlr_low: Optional[float] = None
    bulge_hlr_high: Optional[float] = None


class CatalogAdapter(ABC):
    """One input catalog: schema, key, preprocess, and fitted constants.

    Class attributes (each subclass must define all of them):

    - ``kind``: registry name, the spec's ``population.catalog.kind`` value
    - ``columns``: exact raw file schema; loads validate set equality
    - ``id_columns``: integer columns forming the unique row key; their
      order is the per-galaxy seed-key order (determinism contract)
    - ``flux_variants``: legal ``preprocess.flux_variant`` spec values
    - ``download_hint``: how to obtain the file (used in error messages)
    - ``has_bulge``: catalog carries the bulge contract trio
    - ``has_observed_redshift``: catalog carries ``z_obs``
    - ``validation_columns``: population output name -> contract column name
      for catalog values carried through purely for validation/diagnostics
    - ``prior_constants``: the catalog-fitted ``CatalogPriorConstants``
    - ``citation_bibkeys``: bibkeys of the catalog's source paper(s), used
      by the prior-provenance registry for catalog-sourced entries
    """

    kind: str
    columns: Tuple[str, ...]
    id_columns: Tuple[str, ...]
    flux_variants: Tuple[str, ...]
    download_hint: str
    has_bulge: bool
    has_observed_redshift: bool
    validation_columns: Dict[str, str]
    prior_constants: CatalogPriorConstants
    citation_bibkeys: Tuple[str, ...]

    _REQUIRED_ATTRS = (
        'kind',
        'columns',
        'id_columns',
        'flux_variants',
        'download_hint',
        'has_bulge',
        'has_observed_redshift',
        'validation_columns',
        'prior_constants',
        'citation_bibkeys',
    )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        missing = [a for a in cls._REQUIRED_ATTRS if not hasattr(cls, a)]
        if missing:
            raise TypeError(f"{cls.__name__} is missing adapter attributes: {missing}")
        if cls.has_bulge and cls.prior_constants.bulge_frac_loc is None:
            raise TypeError(f"{cls.__name__}: has_bulge requires bulge prior constants")

    @abstractmethod
    def preprocess(
        self, df: pd.DataFrame, spec: 'CatalogPopulationSpec'
    ) -> pd.DataFrame:
        """Map raw catalog rows to the contract columns.

        Must return exactly the contract set ``contract_columns()`` and set
        ``attrs['kills']`` to a dict of per-stage dropped-row counts (empty
        dict if the preprocess drops nothing).
        """

    def contract_columns(self) -> Tuple[str, ...]:
        """The exact column set ``preprocess`` must return."""
        cols = list(self.id_columns) + list(REQUIRED_CONTRACT_COLUMNS)
        if self.has_observed_redshift:
            cols.append('z_obs')
        if self.has_bulge:
            cols.extend(BULGE_CONTRACT_COLUMNS)
        for src in self.validation_columns.values():
            if src not in cols:
                cols.append(src)
        return tuple(cols)


def validate_contract(adapter: CatalogAdapter, df: pd.DataFrame) -> None:
    """Enforce exact contract-column set equality on a preprocess output."""
    expected = set(adapter.contract_columns())
    actual = set(df.columns)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f"{adapter.kind} preprocess violates the contract: missing "
            f"columns {missing}, unexpected columns {extra}"
        )
    if 'kills' not in df.attrs or not isinstance(df.attrs['kills'], dict):
        raise ValueError(
            f"{adapter.kind} preprocess must set attrs['kills'] to a dict "
            f"of per-stage dropped-row counts"
        )


def load_catalog(
    adapter: CatalogAdapter, name: str, data_dir: Union[str, Path]
) -> pd.DataFrame:
    """Load and verify a named catalog download for an adapter.

    Reads ``<data_dir>/<name>.parquet``, verifies its sha256 against the
    mandatory ``<name>.provenance.json`` sidecar, and validates the
    adapter's exact column schema, non-emptiness, and ``id_columns``
    uniqueness.
    """
    data_dir = Path(data_dir)
    parquet_path = data_dir / f'{name}.parquet'
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"catalog parquet not found: {parquet_path}; {adapter.download_hint}"
        )
    sidecar_path = data_dir / f'{name}.provenance.json'
    if not sidecar_path.exists():
        raise RuntimeError(
            f"provenance sidecar not found: {sidecar_path}; provenance is "
            f"mandatory ({adapter.download_hint})"
        )
    provenance = json.loads(sidecar_path.read_text())
    if 'sha256' not in provenance:
        raise RuntimeError(f"{sidecar_path}: sidecar carries no 'sha256' key")
    actual_sha = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    if actual_sha != provenance['sha256']:
        raise RuntimeError(
            f"catalog sha256 mismatch for {parquet_path}: file has "
            f"{actual_sha}, provenance sidecar records "
            f"{provenance['sha256']}; the download is corrupt or was "
            f"modified -- re-download it"
        )

    df = pd.read_parquet(parquet_path)
    missing = [c for c in adapter.columns if c not in df.columns]
    extra = [c for c in df.columns if c not in adapter.columns]
    if missing or extra:
        raise ValueError(
            f"{parquet_path}: schema mismatch vs the {adapter.kind} query "
            f"spec; missing columns {missing}, unexpected columns {extra}"
        )
    if len(df) == 0:
        raise ValueError(f"{parquet_path}: catalog is empty")
    if df.duplicated(subset=list(adapter.id_columns)).any():
        raise ValueError(
            f"{parquet_path}: duplicate {adapter.id_columns} rows; the "
            f"per-galaxy seed keys require a unique compound id"
        )
    return df


def catalog_provenance(name: str, data_dir: Union[str, Path]) -> dict:
    """Load the provenance sidecar for a named catalog download."""
    sidecar_path = Path(data_dir) / f'{name}.provenance.json'
    if not sidecar_path.exists():
        raise RuntimeError(f"provenance sidecar not found: {sidecar_path}")
    return json.loads(sidecar_path.read_text())
