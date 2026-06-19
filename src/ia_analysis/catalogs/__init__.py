"""Catalog and particle data loading namespace.

Purpose
-------
The catalogs package contains readers for ClusterSims and IllustrisTNG-style
FoF, subhalo, and particle data products.

Provides
--------
- HDF5 chunk discovery and numeric sorting utilities.
- ClusterSims catalog access helpers.
- TNG catalog wrappers that keep API/cache details outside science modules.

Notes
-----
This package is intentionally low level.  It should not import shapes, spectra,
or pipeline orchestration code.
"""

from __future__ import annotations

from typing import Any

from ia_analysis._lazy_imports import ExportMap, load_export

_EXPORTS: ExportMap = {
    "CSCatalog": ("ia_analysis.catalogs.api", "CSCatalog"),
    "ClusterSimsCatalog": ("ia_analysis.catalogs.api", "ClusterSimsCatalog"),
    "TNGCatalog": ("ia_analysis.catalogs.api", "TNGCatalog"),
    "IllustrisTNGCatalog": ("ia_analysis.catalogs.api", "IllustrisTNGCatalog"),
    "LegacyTNGCatalog": ("ia_analysis.catalogs.api", "LegacyTNGCatalog"),
    "open_catalog": ("ia_analysis.catalogs.api", "open_catalog"),
    "open_cluster_catalog": ("ia_analysis.catalogs.api", "open_cluster_catalog"),
    "open_tng_catalog": ("ia_analysis.catalogs.api", "open_tng_catalog"),
    "sort_hdf5_chunks": ("ia_analysis.catalogs.api", "sort_hdf5_chunks"),
}

__all__ = [*list(_EXPORTS), "api", "catalog_loader", "TNGCatLoader"]


def __getattr__(name: str) -> Any:
    """Resolve public catalog helpers lazily from the structured API facade."""
    return load_export(_EXPORTS, name)
