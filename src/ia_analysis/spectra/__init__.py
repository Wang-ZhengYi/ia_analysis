"""Spectra and mesh-construction namespace.

Purpose
-------
The spectra package contains mesh builders, folded power-spectrum drivers,
correlation functions, NLA theory helpers, and velocity/momentum field tools.

Provides
--------
- Catalog and snapshot mesh construction modules.
- IA, matter, galaxy, and velocity power-spectrum measurement entrypoints.
- Momentum-divergence cross-spectrum utilities for self-folded analyses.

Notes
-----
Heavy Pylians dependencies are imported inside the modules that need them, not
at package import time.
"""

from __future__ import annotations

from typing import Any

from ia_analysis._lazy_imports import ExportMap, load_export

_EXPORTS: ExportMap = {
    "discover_spectrum_files": ("ia_analysis.spectra.api", "discover_spectrum_files"),
    "read_spectrum": ("ia_analysis.spectra.api", "read_spectrum"),
    "load_spectrum_collection": ("ia_analysis.spectra.api", "load_spectrum_collection"),
    "relative_to_reference": ("ia_analysis.spectra.api", "relative_to_reference"),
}


__all__ = [
    *list(_EXPORTS),
    "api",
    "analysis",
    "CatMesh",
    "SnapMesh",
    "catalog_mesh",
    "snapshot_mesh",
    "velocity_momentum",
    "powers",
    "ia_pk_cs",
    "ia_pk_folded",
    "ia_corr",
    "theory_nla_pk",
]


def __getattr__(name: str) -> Any:
    """Resolve public spectrum-analysis helpers lazily."""
    return load_export(_EXPORTS, name)
