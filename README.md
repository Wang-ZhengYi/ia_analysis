# MSTA / IA Analysis

Structured Python package for intrinsic-alignment, shape-tide, halo dynamics,
and power-spectrum analysis on ClusterSims and IllustrisTNG-style data.

The importable package is `ia_analysis`. Legacy root-level modules such as
`shape.py`, `run_cs.py`, and `CatMesh.py` are retained as compatibility wrappers.

## Layout

- `src/ia_analysis/catalogs`: CS/TNG catalog and particle loaders.
- `src/ia_analysis/shapes`: inertia tensors, principal axes, kinematics, IA projections.
- `src/ia_analysis/tides`: potential gridding and tidal tensor interpolation.
- `src/ia_analysis/dynamics`: shell-wise halo/subhalo dynamics and TNG wrappers.
- `src/ia_analysis/spectra`: mesh construction, IA/matter/velocity power spectra, and NLA helpers.
- `src/ia_analysis/meshes`: compatibility namespace for old mesh imports.
- `src/ia_analysis/covariance`: Gaussian, cNG, and SSC covariance tools.
- `src/ia_analysis/pipelines`: end-to-end CS/TNG global catalog drivers.
- `src/ia_analysis/orbits`: NFW orbit and synthetic halo utilities.
- `src/ia_analysis/visualization`: plotting and animation helpers.
- `src/ia_analysis/notebook_pipelines`: generated exports from raw notebooks.

## Usage

Legacy commands still work from the repository root:

```bash
python run_cs.py --help
python run_tng.py --help
```

New package-style commands are preferred after installing the package or setting
`PYTHONPATH=src`:

```bash
python -m ia_analysis.pipelines.run_cs --help
python -m ia_analysis.pipelines.run_tng --help
```

## Dependencies

Install the baseline stack with:

```bash
pip install -r requirements/core.txt
pip install -r requirements/dev.txt
```

HPC/COSMA-specific dependencies are documented in `requirements/cosma.txt`;
some are not always available from standard PyPI indexes.

## Documentation

- `docs/architecture.md`: module boundaries and dependency direction.
- `docs/running.md`: CLI entrypoints, configuration, and environment variables.
- `docs/visualization.md`: structured visualization module map.
- `docs/velocity_divergence_self_folding.md`: velocity-divergence folding decision and `tm` proxy.
- `docs/notebook_exports.md`: generated map from raw notebooks to exported Python scripts.
- `docs/notebook_pipeline_map.md`: raw notebooks, curated notebooks, and module mapping.
- `configs/example_paths.json`: example local/COSMA path configuration.
- `notebooks/README.md`: archived notebook inventory.

## Notebooks

Raw notebooks are archived under `notebooks/raw_20260618`. Their code cells are
exported to `src/ia_analysis/notebook_pipelines/exports` for review and future
refactoring. Maintained workflow notebooks live under `notebooks/pipelines`.
