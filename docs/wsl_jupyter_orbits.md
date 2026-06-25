# WSL JupyterLab Orbit Notebook Setup

This guide sets up a WSL Anaconda environment that can open the project
workspace in JupyterLab and run the maintained orbit notebooks.

## Notebook Targets

- `notebooks/pipelines/07_orbits_and_shell_visualization.ipynb`
- `notebooks/pipelines/08_orbit_template_tidal_stripping_demo.ipynb`

The default environment is enough for these curated notebooks and the reusable
orbit-template and tidal-stripping utilities.  Use the full environment when you
also need `ia_analysis.orbits.orbit_nfw.OrbitSimulator`, because that module
imports `pyccl`.

## Quick Start

Open a WSL terminal and move to the Windows workspace through the WSL mount:

```bash
cd /mnt/c/Users/<your-user>/Workspace/ia_analysis
bash tools/wsl_jupyter_orbits.sh
```

The command above assumes the current directory is the project root. From
another directory, invoke the launcher with the correct relative or absolute
path, for example:

```bash
bash /path/to/ia_analysis/tools/wsl_jupyter_orbits.sh
```

The script will:

- find the project root,
- create the `ia-orbits` conda environment if needed,
- install `ia-analysis` in editable mode,
- register the Jupyter kernel `Python (ia-orbits)`,
- start JupyterLab with the notebook root set to the project workspace.

Open the JupyterLab URL printed by the terminal, then choose the kernel
`Python (ia-orbits)` inside the orbit notebooks.

## Full NFW Orbit Environment

Use this variant when you need the full NFW orbit integrator with `pyccl`:

```bash
cd /mnt/c/Users/<your-user>/Workspace/ia_analysis
IA_ORBITS_ENV=ia-orbits-full \
IA_ORBITS_ENV_FILE=requirements/orbits-full-conda.yml \
bash tools/wsl_jupyter_orbits.sh
```

This registers the kernel `Python (ia-orbits-full)`.

## Updating An Existing Environment

If dependencies changed, request an environment update before JupyterLab starts:

```bash
cd /mnt/c/Users/<your-user>/Workspace/ia_analysis
IA_ORBITS_UPDATE=1 bash tools/wsl_jupyter_orbits.sh
```

For the full environment:

```bash
cd /mnt/c/Users/<your-user>/Workspace/ia_analysis
IA_ORBITS_ENV=ia-orbits-full \
IA_ORBITS_ENV_FILE=requirements/orbits-full-conda.yml \
IA_ORBITS_UPDATE=1 \
bash tools/wsl_jupyter_orbits.sh
```

## Manual Commands

The script is only a convenience wrapper.  The equivalent manual setup is:

```bash
cd /mnt/c/Users/<your-user>/Workspace/ia_analysis
conda env create -f requirements/orbits-conda.yml
conda activate ia-orbits
python -m pip install -e .
python -m ipykernel install --user --name ia-orbits --display-name "Python (ia-orbits)"
python -m jupyter lab --notebook-dir . --ip 0.0.0.0 --port 8888 --no-browser
```

## Troubleshooting

- If `conda` is not found, activate Anaconda first or make sure WSL can see one
  of the common install paths such as `$HOME/anaconda3`.
- If the full environment cannot solve `pyccl`, use the default
  `requirements/orbits-conda.yml` environment for notebooks `07` and `08`, then
  install `pyccl` later from conda-forge when the solver issue is resolved.
- If JupyterLab opens but imports fail, confirm the notebook kernel is
  `Python (ia-orbits)` or `Python (ia-orbits-full)`, not the base Python.
- If the browser on Windows cannot open the WSL URL automatically, copy the full
  `http://127.0.0.1:8888/lab?...` URL printed by JupyterLab into the Windows
  browser.
