# Architecture

The package is split by scientific responsibility rather than by historical
script order.

## Dependency Direction

Low-level modules should remain reusable and should not import pipeline code:

```text
catalogs / shapes / tides
        -> dynamics
        -> spectra / covariance
        -> pipelines
```

Mesh construction now lives under `spectra` because the mesh objects are the
direct inputs to power-spectrum estimation.  The `ia_analysis.meshes` namespace
is retained as a compatibility layer only.

`visualization` and `orbits` are side modules used by notebooks and experiments.

## Public Imports

The repository does not ship root-level Python wrappers.  New code should import
from `ia_analysis.*` after installing the package or setting `PYTHONPATH=src`.
Command-line execution should use `python -m ia_analysis.<subpackage>.<module>`.

For mesh construction, prefer `ia_analysis.spectra.CatMesh`,
`ia_analysis.spectra.SnapMesh`, `ia_analysis.spectra.catalog_mesh`, or
`ia_analysis.spectra.snapshot_mesh`.  The `ia_analysis.meshes` namespace remains
inside `src` as a deprecation layer for older notebooks, but no root-level mesh
wrappers are published.

## Optional Heavy Dependencies

Modules that require `pyccl`, Pylians (`MAS_library`, `Pk_library`),
`illustris_python`, or `halotools` are kept in their functional subpackages.
The top-level `ia_analysis` package deliberately avoids importing those modules
at import time.
