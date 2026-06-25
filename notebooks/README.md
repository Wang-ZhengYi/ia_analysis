# Notebook Management

The project now keeps notebooks in three layers.

## Raw Notebooks

`raw_20260618/` contains the latest notebooks exactly as provided. These files
are preserved for provenance and should not be edited directly during routine
analysis cleanup.

## Archived Notebooks

`archive_20260616/` contains the older notebooks from the previous project
layout pass.

## Curated Pipeline Notebooks

`pipelines/` contains the new maintained notebooks. Each notebook covers one
workflow and points to the Python modules that now hold reusable logic. They
use the installed `Python 3.12 (py312)` kernel and can be run from either the
repository root or the `notebooks/pipelines` directory.

Recommended order:

1. `00_project_overview_and_paths.ipynb`
2. `01_global_catalog_generation.ipynb`
3. `02_alignment_figure_suite.ipynb`
4. `03_hod_population_pipeline.ipynb`
5. `04_power_spectrum_pipeline.ipynb`
6. `05_correlation_pipeline.ipynb`
7. `06_tng_dynamics_layered_pipeline.ipynb`
8. `07_orbits_and_shell_visualization.ipynb`
9. `08_orbit_template_tidal_stripping_demo.ipynb`

## Python Exports

Code cells from raw notebooks are exported to
`src/ia_analysis/notebook_pipelines/exports/`. The export index is documented in
`docs/notebook_exports.md`.

When adding new notebook code, first move reusable functions into
`src/ia_analysis`, then keep notebooks as orchestration and visualization
records.

Each maintained workflow notebook also includes a generated legacy-function
manifest. It lists the pipeline and plotting functions from the former
notebooks mapped to that workflow. Use `show_legacy_source(...)` in a notebook
to inspect a preserved definition without importing and executing the former
notebook's top-level data-loading code.

## WSL JupyterLab

For the orbit notebooks, open WSL and run:

```bash
cd /mnt/c/Users/hydro/Workspace/MSTA
bash tools/wsl_jupyter_orbits.sh
```

Then select the `Python (ia-orbits)` kernel in JupyterLab.  The full setup and
the optional `pyccl` environment are documented in
`docs/wsl_jupyter_orbits.md`.
