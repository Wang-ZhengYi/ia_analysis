#!/usr/bin/env bash
# Start JupyterLab for the orbit notebooks from WSL.
#
# Usage from the project root or any subdirectory:
#   bash tools/wsl_jupyter_orbits.sh
#
# Optional environment variables:
#   IA_ORBITS_ENV=ia-orbits-full
#   IA_ORBITS_ENV_FILE=requirements/orbits-full-conda.yml
#   IA_ORBITS_UPDATE=1
#   JUPYTER_PORT=8888

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="${IA_ORBITS_ENV:-ia-orbits}"
ENV_FILE_REL="${IA_ORBITS_ENV_FILE:-requirements/orbits-conda.yml}"
ENV_FILE="${PROJECT_ROOT}/${ENV_FILE_REL}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Conda environment file not found: ${ENV_FILE}" >&2
  exit 1
fi

load_conda() {
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    eval "$(conda shell.bash hook)"
    return
  fi

  local candidate
  for candidate in \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/mambaforge/etc/profile.d/conda.sh" \
    "${HOME}/miniforge3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "${candidate}" ]]; then
      # shellcheck disable=SC1090
      source "${candidate}"
      return
    fi
  done

  echo "Could not find conda. Activate Anaconda first or install conda in WSL." >&2
  exit 1
}

load_conda
cd "${PROJECT_ROOT}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  if [[ "${IA_ORBITS_UPDATE:-0}" == "1" ]]; then
    echo "Updating conda environment: ${ENV_NAME}"
    conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
  fi
else
  echo "Creating conda environment: ${ENV_NAME}"
  conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
fi

conda activate "${ENV_NAME}"

# Keep the editable install fresh even when the environment already exists.
python -m pip install -e "${PROJECT_ROOT}"
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"

echo
echo "Project root: ${PROJECT_ROOT}"
echo "Conda env:    ${ENV_NAME}"
echo "Kernel name:  Python (${ENV_NAME})"
echo "Notebook dir: ${PROJECT_ROOT}"
echo
echo "Open notebooks/pipelines/07_orbits_and_shell_visualization.ipynb"
echo "or notebooks/pipelines/08_orbit_template_tidal_stripping_demo.ipynb"
echo

python -m jupyter lab \
  --notebook-dir="${PROJECT_ROOT}" \
  --ip=0.0.0.0 \
  --port="${JUPYTER_PORT}" \
  --no-browser
