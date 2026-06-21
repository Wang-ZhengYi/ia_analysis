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
cd /mnt/c/Users/hydro/Workspace/MSTA
bash tools/wsl_jupyter_orbits.sh
```

The launcher creates or reuses the `ia-orbits` conda environment, installs the
package in editable mode, registers the `Python (ia-orbits)` kernel, and starts
JupyterLab at the project root.  See `docs/wsl_jupyter_orbits.md` for the full
workflow and the optional `ia-orbits-full` environment with `pyccl`.

## Configuration

Current CLIs keep their historical arguments. `configs/example_paths.json`
documents the path conventions used on COSMA and the environment variables used
by the TNG loader:

- `TNG_API_KEY`
- `TNG_SIM_NAME`
- `TNG_CACHE_DIR`

Prefer passing data paths through CLI flags or environment variables rather than
editing source files.
