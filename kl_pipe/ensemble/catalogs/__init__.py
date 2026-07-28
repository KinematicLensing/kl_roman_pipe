"""
Per-catalog adapters for the catalog-backed ensemble population.

The spec's ``population.catalog.kind`` selects an adapter; everything
catalog-specific (raw schema, unique row key, preprocess, fitted prior
constants) lives in that adapter module. See ``base.py`` for the interface
and the contract-column definition.
"""

from kl_pipe.ensemble.catalogs.base import (
    BULGE_CONTRACT_COLUMNS,
    CatalogAdapter,
    CatalogPriorConstants,
    REQUIRED_CONTRACT_COLUMNS,
    catalog_provenance,
    load_catalog,
    validate_contract,
)
from kl_pipe.ensemble.catalogs.cosmos25 import Cosmos25Adapter
from kl_pipe.ensemble.catalogs.flagship2 import Flagship2Adapter

_ADAPTERS = {
    adapter.kind: adapter for adapter in (Flagship2Adapter(), Cosmos25Adapter())
}


def get_catalog_adapter(kind: str) -> CatalogAdapter:
    """Look up a catalog adapter by its registry kind (case-insensitive)."""
    key = kind.lower()
    if key not in _ADAPTERS:
        raise ValueError(
            f"unknown catalog kind '{kind}'; registered kinds: " f"{sorted(_ADAPTERS)}"
        )
    return _ADAPTERS[key]


__all__ = [
    'BULGE_CONTRACT_COLUMNS',
    'CatalogAdapter',
    'CatalogPriorConstants',
    'REQUIRED_CONTRACT_COLUMNS',
    'catalog_provenance',
    'get_catalog_adapter',
    'load_catalog',
    'validate_contract',
    'Cosmos25Adapter',
    'Flagship2Adapter',
]
