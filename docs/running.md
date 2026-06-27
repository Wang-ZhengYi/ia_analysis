# Running

## Package Entrypoints

After installing the package, or when `PYTHONPATH=src` is set:

```bash
python -m ia_analysis.pipelines.run_cs --help
python -m ia_analysis.pipelines.run_tng --help
python -m ia_analysis.spectra.ia_pk_cs --help
python -m ia_analysis.covariance.Cov --help
```

## WSL JupyterLab For Orbit Notebooks

The maintained orbit notebooks can be run from WSL with the Anaconda launcher:

```bash
cd /mnt/c/Users/<your-user>/Workspace/ia_analysis
bash tools/wsl_jupyter_orbits.sh
```

The launcher creates or reuses the `ia-orbits` conda environment, installs the
package in editable mode, registers the `Python (ia-orbits)` kernel, and starts
JupyterLab at the project root.  See `docs/wsl_jupyter_orbits.md` for the full
workflow and the optional `ia-orbits-full` environment with `pyccl`.
When running from outside the project root, give the correct relative or
absolute path to `tools/wsl_jupyter_orbits.sh`.

## Configuration

Current CLIs keep their historical arguments. `configs/example_paths.json`
documents the path conventions used on COSMA and the environment variables used
by the TNG loader:

- `TNG_API_KEY`
- `TNG_SIM_NAME`
- `TNG_CACHE_DIR`

Prefer passing data paths through CLI flags or environment variables rather than
editing source files.

## Lightweight Pinocchio-Like Tables

CSV, TSV, and whitespace-delimited ASCII orbit tables can be loaded without
Pinocchio-specific dependencies:

```python
from ia_analysis.orbits import PinocchioColumnMap, read_pinocchio_table
from ia_analysis.orbits.pinocchio import tracks_from_table

table = read_pinocchio_table("group_tracks.csv")
columns = PinocchioColumnMap(object_id="group_id", snapshot="snap")
tracks = tracks_from_table(table, columns=columns)
```

The reader does not assign scientific meaning to columns. Use
`PinocchioColumnMap` to adapt each catalog schema.

## Synthetic HOD and IA-HOD Analysis

The HOD layer accepts in-memory mappings, structured NumPy arrays, or pandas
DataFrames and does not require simulation-specific readers:

```python
from ia_analysis.hod import standardize_hod_catalog, measure_hod

catalog = standardize_hod_catalog(halo_table, galaxy_table)
measurement = measure_hod(catalog, mass_bins=mass_edges, sample_label="LRG")
```

See `docs/hod.md` for assembly, phase-space, IA-component, fitting, plotting,
and serialization examples.
