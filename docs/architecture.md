# Architecture

The package is split by scientific responsibility rather than by historical
script order.

## Dependency Direction

Low-level modules should remain reusable and should not import pipeline code:

```text
catalogs / shapes / tides
        -> dynamics / MergerTree
        -> correlations / spectra / covariance
        -> pipelines
```

Mesh construction now lives under `spectra` because the mesh objects are the
direct inputs to power-spectrum estimation.  The `ia_analysis.meshes` namespace
is retained as a compatibility layer only.

`MergerTree` is an orchestration layer for cross-time TNG reading.  It follows
merger-tree tracks and calls `catalogs`, `shapes`, and `dynamics` for the work
done at each snapshot.

`visualization` and `orbits` are side modules used by notebooks and experiments.
The `orbits` package now also contains a planned 2LPT-template layer for
Pinocchio-like merger trees, group-internal subhalo phase-space templates, and
ellipsoidal approximations for HOD one-halo augmentation.

`correlations` owns real-space object-pair statistics for density, IA shape,
velocity, and figure-rotation fields.  It consumes catalog arrays and does not
load simulation products directly.

The `dynamics` package owns particle-level and component-level halo dynamics.
Besides shell-wise tensor diagnostics, it now provides component binding-energy
profiles: array-level helpers live in `halo_dynamics.py`, while TNG catalog
loading and unit conversion live in `hd_tng.py`.

## Public Imports

The repository does not ship root-level Python wrappers.  New code should import
from `ia_analysis.*` after installing the package or setting `PYTHONPATH=src`.
Command-line execution should use `python -m ia_analysis.<subpackage>.<module>`.

For non-spectra domains, prefer the structured API facade in each package:
`ia_analysis.catalogs.api`, `ia_analysis.shapes.api`, `ia_analysis.tides.api`,
`ia_analysis.dynamics.api`, `ia_analysis.MergerTree.api`,
`ia_analysis.correlations.api`,
`ia_analysis.covariance.api`,
`ia_analysis.pipelines.api`, `ia_analysis.orbits.api`, and
`ia_analysis.visualization.api`.  The top-level `ia_analysis.api` registry can
discover these domains without importing heavy implementation modules.

For mesh construction, prefer `ia_analysis.spectra.CatMesh`,
`ia_analysis.spectra.SnapMesh`, `ia_analysis.spectra.catalog_mesh`, or
`ia_analysis.spectra.snapshot_mesh`.  The `ia_analysis.meshes` namespace remains
inside `src` as a deprecation layer for older notebooks, but no root-level mesh
wrappers are published.

## Dependencies

`h5py` is a required baseline dependency because the project reads and writes
HDF5 catalogues, snapshots, spectra, and covariance products throughout the core
workflow.

Heavier domain-specific dependencies such as `pyccl`, Pylians (`MAS_library`,
`Pk_library`), `illustris_python`, or `halotools` are kept in their functional
subpackages.  The top-level `ia_analysis` package deliberately avoids importing
those modules at import time.

## Dynamics Consolidation

Historical implementation filenames remain available inside `src` for old
notebooks and scripts when they still carry unique behavior.  Duplicate TNG
dynamics implementations have been removed: `hd_tng.py` is now the only
maintained TNG halo-dynamics driver and includes the richer cross-time
diagnostics from the former patched/enriched files.
