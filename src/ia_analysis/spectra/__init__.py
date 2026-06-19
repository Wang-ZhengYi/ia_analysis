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


__all__ = [
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
