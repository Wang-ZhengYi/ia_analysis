# MSTA / IA Analysis

`ia_analysis` is a structured Python package for intrinsic-alignment, shape,
tidal-field, halo-dynamics, real-space correlation, and power-spectrum analysis
on ClusterSims and IllustrisTNG-style simulation data.

The repository is organized as a source-first Python project.  Importable code
lives under `src/ia_analysis`; root-level historical Python wrappers are not
published.  New scripts and notebooks should import from `ia_analysis.*` or run
entrypoints with `python -m ia_analysis.<subpackage>.<module>`.

## What This Project Provides

- Catalog readers for ClusterSims and IllustrisTNG FoF, subhalo, snapshot, and
  particle products.
- Shape and IA tools for inertia tensors, principal axes, angular momentum,
  rotational support, shape errors, and projected ellipticity fields.
- Tidal-field tools for potential grids, CIC gridding, tidal tensors, and
  interpolation at object positions.
- Halo-dynamics tools for shell-wise affine flow, figure rotation, pattern-speed
  diagnostics, and component mass distributions over binding energy.
- Merger-tree workflow helpers for cross-snapshot TNG reading and orchestration.
- Real-space correlation tools for `ee`, `ed`, `dd`, velocity correlations,
  figure-rotation correlations, halo/sample categories, covariance, and a
  compact `v-e-d-omega` four-point diagnostic.
- Mesh and spectrum tools for galaxy/particle meshes, IA power spectra, folded
  spectra, velocity momentum-divergence proxies, and NLA theory helpers.
- HOD tools for ordinary and assembly-biased occupation, satellite phase space,
  component-level IA measurements, conditional IA strength, and lightweight
  omega/eta validation.
- Covariance tools for Gaussian, connected non-Gaussian, and super-sample
  covariance products used by the power-spectrum workflow.
- Visualization helpers migrated from notebooks into reusable modules for
  alignment atlases, spectra, correlations, dynamics, merger-tree figures, and
  orbit animations.
- Maintained workflow notebooks under `notebooks/pipelines` with raw-notebook
  exports archived under `src/ia_analysis/notebook_pipelines`.

## Repository Layout

```text
configs/                 Example path and environment configuration.
docs/                    Architecture notes, user guides, and PDF manual.
notebooks/pipelines/     Maintained workflow notebooks.
requirements/            Baseline, development, and COSMA dependency lists.
src/ia_analysis/         Importable package source.
tests/                   Lightweight smoke and unit tests.
tools/                   Notebook export and maintenance helpers.
```

Important subpackages:

- `ia_analysis.catalogs`: ClusterSims and TNG catalog/particle loading.
- `ia_analysis.shapes`: shape tensors, axes, spin, kappa rotation, and IA
  projection helpers.
- `ia_analysis.tides`: potential grids, CIC mass assignment, tidal tensors, and
  tidal interpolation.
- `ia_analysis.dynamics`: shell-wise halo dynamics, TNG halo wrappers, affine
  flow, figure rotation, and binding-energy mass profiles.
- `ia_analysis.MergerTree`: cross-time merger-tree reading and workflow
  orchestration.
- `ia_analysis.correlations`: real-space IA, density, velocity, figure-rotation,
  covariance, and compressed four-point correlations.
- `ia_analysis.spectra`: mesh construction, power spectra, folded spectra,
  velocity momentum-divergence proxy fields, and NLA theory helpers.
- `ia_analysis.meshes`: compatibility namespace for older mesh imports.
- `ia_analysis.covariance`: Gaussian, cNG, and SSC covariance helpers for
  measured spectra.
- `ia_analysis.pipelines`: end-to-end ClusterSims and TNG command entrypoints.
- `ia_analysis.orbits`: NFW orbit experiments, mock halo generation, 2LPT
  orbit-template libraries, ellipsoidal group approximations, and lightweight
  tidal-stripping post-processing.
- `ia_analysis.hod`: ordinary, decorated, environment/concentration,
  phase-space, and component-based assembly-biased IA-HOD models.
- `ia_analysis.visualization`: plotting, figure-saving, atlas, diagnostic, and
  animation helpers.

## Dependency Direction

The package is split by scientific responsibility.  Low-level modules should
remain reusable and should not import high-level orchestration code:

```text
catalogs / shapes / tides
        -> hod / dynamics / MergerTree
        -> correlations / spectra / covariance
        -> pipelines
```

`visualization` and `orbits` are side modules used by notebooks and experiments.
Heavy optional dependencies are imported only by the submodules that need them.

## Installation

From a fresh checkout:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements/dev.txt
```

For a quick local run without installation, set `PYTHONPATH=src` before running
Python commands:

```bash
set PYTHONPATH=src
python -c "import ia_analysis; print(ia_analysis.__version__)"
```

Baseline dependencies are listed in `requirements/core.txt` and include
`numpy`, `scipy`, `h5py`, `numba`, `matplotlib`, `seaborn`, `pandas`, `tqdm`,
and `psutil`.  `h5py` is required because catalog, snapshot, covariance, and
analysis products are HDF5-based throughout the project.

COSMA/HPC-specific dependencies are listed in `requirements/cosma.txt`.  Some
of those packages, such as Pylians, `illustris_python`, `halotools`, or `pyccl`,
may need local cluster modules or non-standard installation steps.

For WSL Anaconda/JupyterLab orbit notebooks, use the maintained conda files and
launcher:

```bash
cd /mnt/c/Users/hydro/Workspace/MSTA
bash tools/wsl_jupyter_orbits.sh
```

This creates the `ia-orbits` environment, registers the `Python (ia-orbits)`
kernel, and starts JupyterLab at the project root.  Use
`requirements/orbits-full-conda.yml` when full `pyccl` orbit integration is
needed.

## Configuration And Data Paths

Use CLI flags, environment variables, or `configs/example_paths.json` rather
than editing source files.  TNG-related code commonly uses:

- `TNG_API_KEY`
- `TNG_SIM_NAME`
- `TNG_CACHE_DIR`

ClusterSims paths, TNG base paths, output roots, and COSMA paths should be kept
outside the package source and passed into pipeline functions or command-line
entrypoints.

## Public API Discovery

Each major domain exposes a lightweight `api.py` facade.  The top-level registry
lets notebooks discover the available domains without importing heavy backends:

```python
from ia_analysis import api

print(api.available_domains())
catalog_api = api.load_domain_api("catalogs")
shape_api = api.load_domain_api("shapes")
```

Current domains include `catalogs`, `shapes`, `tides`, `dynamics`,
`merger_tree`, `hod`, `correlations`, `covariance`, `pipelines`, `orbits`, and
`visualization`.

## Common Usage Examples

Shape measurements:

```python
from ia_analysis.shapes import measure_iterative_shape, project_shape_ellipticity

shape = measure_iterative_shape(positions, masses=masses)
ellipticity = project_shape_ellipticity(shape["tensor"], los=(0.0, 0.0, 1.0))
```

Tidal-field sampling:

```python
from ia_analysis.tides import build_tidal_grid, sample_tidal_grid

tidal = build_tidal_grid(particle_positions, particle_masses, boxsize=205.0)
samples = sample_tidal_grid(tidal, object_positions)
```

TNG component binding-energy profiles:

```python
from ia_analysis.dynamics import compute_tng_component_binding_profiles

out = compute_tng_component_binding_profiles(
    base_path="/path/to/TNG300-1",
    snap=99,
    subhalo_id=12345,
)
summary = out["summary"]
binding_distribution = out["binding_distribution"]
```

Merger-tree cross-time workflow:

```python
from ia_analysis.MergerTree import run_cross_time_workflow, save_cross_time_products

products = run_cross_time_workflow(
    base_path="/path/to/TNG300-1",
    snap0=99,
    subhalo_id0=12345,
    snap_track=[99, 91, 84, 67],
    sim_name="TNG300-1",
    components=("dm", "stars"),
)
save_cross_time_products(products, "outputs/subhalo_12345_cross_time.pkl")
```

Real-space correlations:

```python
import numpy as np
from ia_analysis.correlations import CorrelationCatalog, measure_default_correlations

catalog = CorrelationCatalog(
    positions=positions,
    fields={"e": e, "v": velocity, "omega": omega},
    host_id=host_id,
    sample_type=sample_type,
    boxsize=205.0,
)

rbins = np.logspace(-1, 1.5, 16)
suite = measure_default_correlations(
    catalog,
    rbins,
    covariance="jackknife",
    nsub=3,
)

ee_total = suite.results["ee"].values["total"]
ev_2h_cs_cov = suite.covariance["ev"]["2h_cs"]["cov"]
```

The correlation suite keeps summary categories `total`, `1h`, and `2h`, plus
five detailed central/satellite categories: `1h_cs`, `1h_ss`, `2h_cc`,
`2h_cs`, and `2h_ss`.  It also estimates `vedomega4` and
`vedomega4_connected` as compact pair-binned four-point diagnostics.

Visualization:

```python
from ia_analysis.visualization import set_project_style, save_figure
from ia_analysis.visualization.alignment_atlas import plot_metric_atlas

set_project_style()
fig, axes = plot_metric_atlas(frame, x_col="radius", metric_cols=["ed", "ee"], hue_col="flag")
save_figure(fig, "alignment_metric_atlas", root="plots")
```

## Command-Line Entrypoints

Package-style commands are preferred:

```bash
python -m ia_analysis.pipelines.run_cs --help
python -m ia_analysis.pipelines.run_tng --help
python -m ia_analysis.spectra.ia_pk_cs --help
python -m ia_analysis.spectra.ia_pk_folded --help
python -m ia_analysis.covariance.Cov --help
```

The old root-level script layout is intentionally not published.  If an older
notebook imports a historical module name, update it to the corresponding
`ia_analysis.<subpackage>` path or use the compatibility namespaces inside
`src/ia_analysis`.

## Notebook Workflow

Maintained notebooks live in `notebooks/pipelines`:

- `00_project_overview_and_paths.ipynb`
- `01_global_catalog_generation.ipynb`
- `02_alignment_figure_suite.ipynb`
- `03_hod_population_pipeline.ipynb`
- `04_power_spectrum_pipeline.ipynb`
- `05_correlation_pipeline.ipynb`
- `06_tng_dynamics_layered_pipeline.ipynb`
- `07_orbits_and_shell_visualization.ipynb`
- `08_orbit_template_tidal_stripping_demo.ipynb`

Raw notebook exports are kept under `src/ia_analysis/notebook_pipelines/exports`
for review and gradual migration into reusable modules.  New reusable plotting
code should go into `ia_analysis.visualization`, not into notebooks.

## Testing And Validation

Run the lightweight test suite after structural changes:

```bash
python -m pytest tests
```

When the local environment does not have every optional HPC dependency, the
most useful smoke checks are:

```bash
python -m compileall -q src tests
python -c "import sys; sys.path.insert(0, 'src'); import ia_analysis; print(ia_analysis.__version__)"
```

Tests use synthetic arrays where possible.  Real COSMA/TNG data checks should be
run separately in the data environment.

## Documentation

- `docs/architecture.md`: module boundaries and dependency direction.
- `docs/api_facades.md`: recommended structured import paths.
- `docs/merger_tree.md`: merger-tree and cross-time workflow steps.
- `docs/correlations.md`: real-space correlation categories, covariance, and
  four-point products.
- `docs/orbit_template_plan.md`: 2LPT/Pinocchio orbit-template plan for HOD
  one-halo augmentation and ellipsoidal shape modeling.
- `docs/tidal_stripping_literature.md`: tidal-stripping literature notes,
  lightweight algorithm choices, and calibration roadmap.
- `docs/running.md`: CLI entrypoints, configuration, and environment variables.
- `docs/wsl_jupyter_orbits.md`: WSL Anaconda and JupyterLab setup for orbit
  notebooks.
- `docs/visualization.md`: structured visualization module map.
- `docs/velocity_divergence_self_folding.md`: velocity-divergence folding
  decision and the `tm` momentum-divergence proxy.
- `docs/notebook_exports.md`: map from raw notebooks to exported Python scripts.
- `docs/notebook_pipeline_map.md`: raw notebooks, curated notebooks, and module
  mapping.
- `docs/ia_analysis_user_manual.pdf`: printable project usage manual.
- `configs/example_paths.json`: example local/COSMA path configuration.

## Development Notes

- Keep comments and docstrings in English.
- Prefer structured APIs and local helper modules over ad hoc notebook code.
- Keep bottom-layer modules independent from `pipelines`.
- Add new plotting helpers under `ia_analysis.visualization` by function area.
- Add tests with synthetic arrays before wiring code into real data workflows.
- Avoid committing generated data products, cache directories, raw media outputs,
  or machine-specific path files.
