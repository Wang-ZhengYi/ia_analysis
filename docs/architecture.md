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

## Compatibility

Root-level files are compatibility wrappers. They make the `src` package
importable in-place and re-export the new package modules. Existing code such as
`from shape import ShapeKin` and `python run_cs.py ...` is expected to continue
working from the repository root.

New code should import from `ia_analysis.*`.

For mesh construction, prefer `ia_analysis.spectra.CatMesh`,
`ia_analysis.spectra.SnapMesh`, `ia_analysis.spectra.catalog_mesh`, or
`ia_analysis.spectra.snapshot_mesh`.  Historical imports through
`ia_analysis.meshes` still work.

## Optional Heavy Dependencies

Modules that require `pyccl`, Pylians (`MAS_library`, `Pk_library`),
`illustris_python`, or `halotools` are kept in their functional subpackages.
The top-level `ia_analysis` package deliberately avoids importing those modules
at import time.
