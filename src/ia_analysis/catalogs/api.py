"""Structured catalog-loading API facade.

Purpose
-------
This module gives new scripts a concise import path for ClusterSims and
IllustrisTNG catalog readers while preserving the historical implementation
files underneath.  The reader classes are lazy-loaded so importing this facade
does not touch HDF5, network, or Illustris-specific dependencies.

Provides
--------
- Stable aliases for ClusterSims and TNG catalog classes.
- Factory helpers for opening catalogs with explicit implementation choices.
- Numeric HDF5 chunk sorting helpers used by tests and data-discovery code.
"""

from __future__ import annotations

from typing import Any, Iterable

from ia_analysis._lazy_imports import ExportMap, call_export, export_names, load_export

_EXPORTS: ExportMap = {
    "ClusterSimsCatalog": ("ia_analysis.catalogs.catalog_loader", "CSCatalog"),
    "CSCatalog": ("ia_analysis.catalogs.catalog_loader", "CSCatalog"),
    "IllustrisTNGCatalog": ("ia_analysis.catalogs.TNGCatLoader", "TNGCatalog"),
    "TNGCatalog": ("ia_analysis.catalogs.TNGCatLoader", "TNGCatalog"),
    "LegacyTNGCatalog": ("ia_analysis.catalogs.catalog_loader", "TNGCatalog"),
    "hdf5_chunk_sort_key": ("ia_analysis.catalogs.catalog_loader", "_hdf5_chunk_sort_key"),
    "tng_hdf5_chunk_sort_key": ("ia_analysis.catalogs.TNGCatLoader", "_hdf5_chunk_sort_key"),
}

__all__ = [
    *export_names(_EXPORTS),
    "open_catalog",
    "open_cluster_catalog",
    "open_tng_catalog",
    "sort_hdf5_chunks",
]


def __getattr__(name: str) -> Any:
    """Resolve class aliases lazily from their implementation modules."""
    return load_export(_EXPORTS, name)


def hdf5_chunk_sort_key(filepath: str) -> tuple[Any, ...]:
    """Return the ClusterSims/TNG-compatible numeric HDF5 chunk sort key."""
    return call_export(_EXPORTS, "hdf5_chunk_sort_key", filepath)


def tng_hdf5_chunk_sort_key(filepath: str) -> tuple[Any, ...]:
    """Return the standalone TNG loader numeric HDF5 chunk sort key."""
    return call_export(_EXPORTS, "tng_hdf5_chunk_sort_key", filepath)


def sort_hdf5_chunks(paths: Iterable[str], *, implementation: str = "cluster") -> list[str]:
    """Sort split HDF5 files by numeric chunk id instead of lexicographic order."""
    key_name = (
        "tng_hdf5_chunk_sort_key"
        if str(implementation).lower() in {"tng", "standalone"}
        else "hdf5_chunk_sort_key"
    )
    key = load_export(_EXPORTS, key_name)
    return sorted(list(paths), key=key)


def open_cluster_catalog(*args: Any, **kwargs: Any) -> Any:
    """Instantiate the ClusterSims catalog loader."""
    cls = load_export(_EXPORTS, "ClusterSimsCatalog")
    return cls(*args, **kwargs)


def open_tng_catalog(*args: Any, implementation: str = "standalone", **kwargs: Any) -> Any:
    """Instantiate a TNG catalog loader using the requested implementation."""
    key = "LegacyTNGCatalog" if str(implementation).lower() in {"legacy", "catalog_loader"} else "IllustrisTNGCatalog"
    cls = load_export(_EXPORTS, key)
    return cls(*args, **kwargs)


def open_catalog(kind: str, *args: Any, **kwargs: Any) -> Any:
    """Instantiate a catalog loader by short dataset family name."""
    key = str(kind).strip().lower()
    if key in {"cs", "clustersims", "cluster"}:
        return open_cluster_catalog(*args, **kwargs)
    if key in {"tng", "illustris", "illustristng"}:
        return open_tng_catalog(*args, **kwargs)
    raise ValueError("Catalog kind must be one of: 'cs', 'clustersims', 'tng', or 'illustris'.")
