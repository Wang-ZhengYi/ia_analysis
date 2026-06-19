"""Create curated workflow notebooks for the structured IA analysis project.

Purpose
-------
This maintenance script writes the small, human-facing notebooks that live in
``notebooks/pipelines``.  The notebooks are intentionally lightweight: they
explain the project workflow, show import and command templates, and point users
to the reusable package modules under ``src/ia_analysis``.

Provides
--------
- A project overview notebook for paths and execution order.
- Pipeline notebooks for catalog generation, alignment figures, population
  analysis, spectra, correlations, TNG dynamics, and orbit visualization.
- English Markdown cells that use dollar-delimited math notation.

Notes
-----
The script does not execute science pipelines.  It only regenerates notebook
JSON files from reproducible templates so the published repository has clean,
consistent, reviewable documentation notebooks.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "pipelines"


def md(source: str) -> dict:
    """Return a Markdown notebook cell with normalized trailing newline."""
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(source: str) -> dict:
    """Return an unexecuted code notebook cell with empty output state."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


def write_notebook(name: str, cells: list[dict]) -> None:
    """Write one notebook file using a minimal Python 3 kernelspec."""
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


COMMON_SETUP = """
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd()
if (PROJECT_ROOT / "src").exists():
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

print("Project root:", PROJECT_ROOT)
"""


def main() -> None:
    """Regenerate all curated pipeline notebooks."""
    write_notebook(
        "00_project_overview_and_paths.ipynb",
        [
            md(
                """
# 00 Project Overview And Paths

Use this notebook to confirm the repository layout, data locations, and output
directories before running any analysis workflow.

Core directories:

- `src/ia_analysis`: importable Python package.
- `src/ia_analysis/notebook_pipelines/exports`: code exported from raw analysis notebooks for review.
- `notebooks/pipelines`: maintained workflow notebooks.
- `configs/example_paths.json`: example path configuration.
"""
            ),
            code(COMMON_SETUP),
            code(
                """
import json

CONFIG_PATH = PROJECT_ROOT / "configs" / "example_paths.json"
with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = json.load(f)

config
"""
            ),
            md(
                """
## Recommended Pipeline Order

1. Run a global catalog pipeline to produce a subhalo-level HDF5 table.
2. Run the alignment figure pipeline to inspect shape and tidal alignments.
3. Run the HOD and population pipeline to define LRG, ELG, and fixed-density samples.
4. Run the power-spectrum pipeline to measure $P(k)$ products.
5. Run the correlation pipeline to measure real-space IA correlation functions.
6. Run the TNG dynamics pipeline for shell-wise and layered shape-tide diagnostics.
"""
            ),
        ],
    )

    write_notebook(
        "01_global_catalog_generation.ipynb",
        [
            md(
                """
# 01 Global Catalog Generation

This notebook documents the ClusterSims and TNG subhalo-level catalog
generation workflow.  The products are columnar HDF5 files used downstream by
alignment, HOD, $P(k)$, and correlation measurements.
"""
            ),
            code(COMMON_SETUP),
            code(
                """
cs_command = (
    "python -m ia_analysis.pipelines.run_cs "
    "--basepath /path/to/ClusterSims/L302_N1136_GR "
    "--snap 21 --nworker 16 --out outputs/global_cs_GR_s021.hdf5"
)

tng_command = (
    "python -m ia_analysis.pipelines.run_tng "
    "--nworker 16 --out outputs/global_tng_s099.hdf5 "
    "--api-key $TNG_API_KEY"
)

print(cs_command)
print(tng_command)
"""
            ),
            md(
                """
## Output Checks

After generation, verify that:

- The HDF5 root contains keys such as `SubhaloID`, `GroupID`, `DM`, `Star`, `Tidal_grp`, and `Tidal_tot`.
- `DM/I` and `Star/I` have shape $N \\times 3 \\times 3$.
- `Star/cos_err` is finite and compatible with later selection thresholds.
"""
            ),
        ],
    )

    write_notebook(
        "02_alignment_figure_suite.ipynb",
        [
            md(
                """
# 02 Alignment Figure Suite

This notebook loads MA or MArenew alignment catalogs and calls the structured
visualization API to generate shape, tidal, velocity, and radial-alignment
figures.
"""
            ),
            code(COMMON_SETUP),
            code(
                """
from ia_analysis.visualization import alignment_catalogs, alignment_plots

alignment_plots.configure_paper_style()

MA_PATH = "MArenew.pkl"

# Example:
# MAset, flags, snap_list = alignment_catalogs.load_legacy_alignment_pickle(MA_PATH)
# alignment_catalogs.configure_alignment_context(MAset, flags, snap_list=snap_list)
# alignment_plots.list_alignment_chapters()
"""
            ),
            code(
                """
# Example: plot one alignment specification.
# alignment_plots.plot_alignment_pair("star_shape_dm_shape_major", save=True, show=True)

# Example: batch a full chapter.
# alignment_plots.plot_alignment_chapter("shape_shape", save=True, show=False)
"""
            ),
            md(
                """
## Source Mapping

Notebook plotting and analysis code exported from the raw notebooks is stored
under `src/ia_analysis/notebook_pipelines/exports`.  Prefer the structured
`ia_analysis.visualization` modules for new work, and use exports as historical
references when a workflow has not yet been fully refactored.
"""
            ),
        ],
    )

    write_notebook(
        "03_hod_population_pipeline.ipynb",
        [
            md(
                """
# 03 HOD And Population Pipeline

This notebook organizes HOD, LRG/ELG, satellite radial-distribution, and merger
population analyses.
"""
            ),
            code(COMMON_SETUP),
            code(
                """
exports = PROJECT_ROOT / "src" / "ia_analysis" / "notebook_pipelines" / "exports"
for name in [
    "hod_lrg_elg_nb.py",
    "hod_data_nb.py",
    "hod_measure_lrg_elg_nb.py",
    "maset_satellite_radial_distribution_nb.py",
    "maset_satellite_radial_distribution_compare_nb.py",
    "merger_align_nb.py",
    "merger_stripping_nb.py",
]:
    print(exports / name)
"""
            ),
            md(
                """
## Suggested Workflow

1. Load a sample from a global HDF5 file or from the original FoF/Subhalo catalogs.
2. Split by stellar mass, SFR, central/satellite status, and host mass.
3. Produce HOD tables, satellite radial profiles, and merger diagnostics.
4. Write tables to `outputs/tables` and figures to `outputs/figures/hod_population`.
"""
            ),
        ],
    )

    write_notebook(
        "04_power_spectrum_pipeline.ipynb",
        [
            md(
                """
# 04 Power Spectrum Pipeline

This notebook documents folded mesh construction, IA field generation, matter
density fields, and $P(k)$ measurement.
"""
            ),
            code(COMMON_SETUP),
            code(
                """
from ia_analysis.spectra.ia_pk_cs import parse_pk_types, spec_keys_from_pk_types

pk_types = parse_pk_types("core")
spec_keys = spec_keys_from_pk_types(pk_types)

pk_types, spec_keys
"""
            ),
            code(
                """
command = (
    "python -m ia_analysis.spectra.ia_pk_cs --flag GR --snap 21 "
    "--threads 8 --nmesh 512 --folds 1,2,4,8,16,32 "
    "--pk-types full --outdir outputs/pks"
)
print(command)
"""
            ),
            md(
                """
## Outputs

The main product is `pks_FLAG_SNAP.hdf5`.  Each sample contains folded spectra,
stitched native spectra, stitched target-$k$ spectra, and noise-corrected
spectra where available.
"""
            ),
        ],
    )

    write_notebook(
        "05_correlation_pipeline.ipynb",
        [
            md(
                """
# 05 Correlation Pipeline

This notebook documents real-space IA correlation measurements, including
jackknife covariance and mass-bin samples.
"""
            ),
            code(COMMON_SETUP),
            code(
                """
command = (
    "python -m ia_analysis.spectra.ia_corr --flag GR --snap 21 "
    "--boxsize 205.0 --nsub 3 --nthreads 8 "
    "--out outputs/cfs/cfs_GR_s021.hdf5"
)
print(command)
"""
            ),
            md(
                """
## Notes

This pipeline depends on `halotools`.  If you only need to review the workflow,
read `src/ia_analysis/notebook_pipelines/exports/ia_corr_nb.py` and
`ia_corr_abundance_nb.py` before running large jobs.
"""
            ),
        ],
    )

    write_notebook(
        "06_tng_dynamics_layered_pipeline.ipynb",
        [
            md(
                """
# 06 TNG Dynamics And Layered Shape-Tide Pipeline

This notebook covers TNG halo dynamics, cross-redshift tracking, layered
ellipsoidal shells, and tidal comparisons.
"""
            ),
            code(COMMON_SETUP),
            code(
                """
import numpy as np
from ia_analysis.dynamics import halo_dynamics

M = np.eye(3)
halo_dynamics.eigh_sorted_desc(M)
"""
            ),
            code(
                """
command = (
    "python -m ia_analysis.pipelines.tng_layered_shape_tide "
    "--snap 99 --base /path/to/tng_data "
    "--out outputs/tng_layered_s099.hdf5"
)
print(command)
"""
            ),
            md(
                """
## Related Exports

Historical workflows include `hd_tng_crossZ.ipynb`, `hd_tng_plot.ipynb`,
`crossz.ipynb`, and `TNGCatLoader.ipynb`.  Their code exports are available in
the notebook exports package for reference.
"""
            ),
        ],
    )

    write_notebook(
        "07_orbits_and_shell_visualization.ipynb",
        [
            md(
                """
# 07 Orbits And Shell Visualization

This notebook covers NFW orbit mocks, radial shell plots, binding shell plots,
and 3D visual diagnostics.
"""
            ),
            code(COMMON_SETUP),
            code(
                """
from ia_analysis.orbits.halo_maker import gen_nfw, transform_points_to_ellipsoid

pts = gen_nfw(size=128)
unrotated, rotated = transform_points_to_ellipsoid(
    pts,
    a=2.0,
    b=1.0,
    c=0.5,
    principal_axis=[1.0, 1.0, 0.2],
)

pts.shape, rotated.shape
"""
            ),
            md(
                """
## Visualization Modules

Use `ia_analysis.visualization.shell_plots` for radial and binding shell panels,
`ia_analysis.visualization.orbit_animation` for movies and frame previews, and
`ia_analysis.visualization.scene3d` for 3D scene helpers.
"""
            ),
        ],
    )


if __name__ == "__main__":
    main()

