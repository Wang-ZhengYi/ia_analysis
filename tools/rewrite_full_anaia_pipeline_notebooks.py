"""Rewrite pipeline notebooks from the complete anaIA notebook sources."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PIPELINE = REPO.parent / "pipeline"
ANAIA = Path("/cosma/home/dp203/dc-wang17/IA_analysis/anaIA")


def source_notebook(name: str) -> dict:
    return json.loads((ANAIA / name).read_text(encoding="utf-8"))


def clean_source(source: str) -> str:
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%%"):
            lines.append(f"# Original notebook magic: {stripped}")
        elif stripped.startswith("%"):
            lines.append(f"# Original notebook magic: {stripped}")
        elif stripped.startswith("!pwd"):
            lines.append('print("Working directory:", Path.cwd())')
        elif stripped.startswith("!"):
            lines.append(f"# Original shell command disabled: {stripped}")
        else:
            lines.append(line)
    return "\n".join(lines).strip() + "\n"


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": clean_source(source),
    }


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip() + "\n",
    }


def extracted_cells(name: str, indices: list[int] | None = None) -> list[dict]:
    notebook = source_notebook(name)
    selected = range(len(notebook["cells"])) if indices is None else indices
    cells = []
    for index in selected:
        cell = notebook["cells"][index]
        if cell["cell_type"] == "code":
            cells.append(code("".join(cell.get("source", []))))
        elif cell["cell_type"] == "markdown":
            cells.append(markdown("".join(cell.get("source", []))))
    return cells


def _safe_config_value(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_safe_config_value(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            key is None or _safe_config_value(key)
            for key in node.keys
        ) and all(_safe_config_value(value) for value in node.values)
    if isinstance(node, ast.UnaryOp):
        return _safe_config_value(node.operand)
    if isinstance(node, ast.BinOp):
        return _safe_config_value(node.left) and _safe_config_value(node.right)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in {"Path", "dict", "tuple", "list"}:
            return all(_safe_config_value(arg) for arg in node.args) and all(
                keyword.arg is not None and _safe_config_value(keyword.value)
                for keyword in node.keywords
            )
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"
            and node.func.attr in {"array", "asarray", "linspace", "logspace", "geomspace", "arange"}
        ):
            return all(_safe_config_value(arg) for arg in node.args) and all(
                keyword.arg is not None and _safe_config_value(keyword.value)
                for keyword in node.keywords
            )
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.value.id == "np" and node.attr in {"nan", "inf", "pi"}
    return False


def source_function_library(names: tuple[str, ...], section_title: str) -> list[dict]:
    """Extract reusable definitions without executing source notebook runners."""
    cells = [
        markdown(
            f"""
## {section_title}

The following cells preserve reusable functions, classes, and literal
configuration constants from the mapped `anaIA` notebooks. They intentionally
exclude top-level compute loops and plotting calls, so the API remains
available without repeating expensive production work.

Use `load_preserved_api("source.ipynb")` to obtain an isolated namespace, or
`load_preserved_api("source.ipynb", export=True)` to place the preserved names
in the notebook's global namespace. `extra_globals` can provide source-specific
paths such as `OUTDIR` when needed.
"""
        )
    ]
    cells.append(
        code(
            """
from types import SimpleNamespace

PRESERVED_API_SOURCES = globals().get("PRESERVED_API_SOURCES", {})
PRESERVED_API_DEFINITIONS = globals().get("PRESERVED_API_DEFINITIONS", {})

def load_preserved_api(source_notebook, *, extra_globals=None, export=False):
    if source_notebook not in PRESERVED_API_SOURCES:
        raise KeyError(f"No preserved API registered for {source_notebook!r}")
    namespace = dict(globals())
    namespace.update({
        "OUTPUT_DIR": OUTPUT_DIR,
        "OUTDIR": OUTPUT_DIR,
        "HOD_DATA_DIR": Path("/cosma/home/dp203/dc-wang17/IA_analysis/anaIA/curve_data_hdf5"),
        "BASE_CF_DIR": globals().get("BASE_CF_DIR", Path("/cosma/home/dp203/dc-wang17/IA_analysis/cfs")),
        "DATA_DIR": globals().get("DATA_DIR", Path("/cosma/home/dp203/dc-wang17/IA_analysis/pks")),
        "MODULE_DIR": globals().get("MODULE_DIR", ANAIA_ROOT),
        "zmap": globals().get("zmap", getattr(globals().get("arts_IA", None), "ZMAP_ALL", {})),
        "flags": globals().get("flags", globals().get("FLAGS", [])),
        "samples": globals().get("samples", globals().get("SAMPLES", [])),
        "spectra_by_sample": globals().get("spectra_by_sample", {}),
    })
    if extra_globals:
        namespace.update(extra_globals)
    exec(PRESERVED_API_SOURCES[source_notebook], namespace)
    names = PRESERVED_API_DEFINITIONS[source_notebook]
    exported = {name: namespace[name] for name in names if name in namespace}
    if export:
        globals().update(exported)
    return SimpleNamespace(**exported)
"""
        )
    )
    index_rows = []
    for name in names:
        chunks = []
        definitions = []
        notebook = source_notebook(name)
        for source_cell in notebook["cells"]:
            if source_cell["cell_type"] != "code":
                continue
            source = clean_source("".join(source_cell.get("source", [])))
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            selected = []
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    selected.append(node)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        definitions.append(node.name)
                elif isinstance(node, ast.Assign) and _safe_config_value(node.value):
                    target_names = {
                        target.id for target in node.targets
                        if isinstance(target, ast.Name)
                    }
                    reserved = {
                        "OUTPUT_DIR", "OUTDIR", "DATA_DIR", "MODULE_DIR",
                        "HOD_DATA_DIR", "BASE_DIR", "BASE_CF_DIR",
                    }
                    if not target_names & reserved:
                        selected.append(node)
                elif isinstance(node, ast.Assign):
                    target_names = {
                        target.id for target in node.targets
                        if isinstance(target, ast.Name)
                    }
                    if target_names & {"ALL_SNAP_LIST", "RADIAL_BINS", "files_df"}:
                        selected.append(node)
                elif isinstance(node, ast.AnnAssign) and node.value is not None and _safe_config_value(node.value):
                    selected.append(node)
            if selected:
                module = ast.Module(body=selected, type_ignores=[])
                chunks.append(ast.unparse(ast.fix_missing_locations(module)))
        if chunks:
            cells.append(markdown(f"### Preserved API from `{name}`"))
            source_text = "\n\n".join(chunks)
            cells.append(
                code(
                    f"""
PRESERVED_API_SOURCES[{name!r}] = {source_text!r}
PRESERVED_API_DEFINITIONS[{name!r}] = {tuple(dict.fromkeys(definitions))!r}
print("Registered preserved API:", {name!r}, "definitions:", len(PRESERVED_API_DEFINITIONS[{name!r}]))
"""
                )
            )
        index_rows.append((name, len(definitions), ", ".join(definitions)))

    table = [
        "| Source notebook | Definitions preserved | Names |",
        "|---|---:|---|",
    ]
    for name, count, names_text in index_rows:
        table.append(f"| `{name}` | {count} | {names_text or 'No top-level definitions'} |")
    cells.insert(1, markdown("\n".join(table)))
    return cells


def replace(cells: list[dict], replacements: list[tuple[str, str]]) -> list[dict]:
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        for old, new in replacements:
            source = source.replace(old, new)
        cell["source"] = source
    return cells


def bootstrap(folder: str, sources: tuple[str, ...]) -> dict:
    pipeline_dir = PIPELINE / folder
    return code(
        f"""
from pathlib import Path
import os
import sys
from IPython.display import display

PIPELINE_DIR = Path({str(pipeline_dir)!r})
OUTPUT_DIR = PIPELINE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANAIA_ROOT = Path({str(ANAIA)!r})
if str(ANAIA_ROOT) not in sys.path:
    sys.path.insert(0, str(ANAIA_ROOT))

os.chdir(PIPELINE_DIR)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
os.environ.setdefault("IPYTHONDIR", str(OUTPUT_DIR / ".ipython"))

print("Pipeline directory:", PIPELINE_DIR)
print("Output directory:", OUTPUT_DIR)
print("Source notebooks:", {sources!r})
print("Python:", sys.executable)
"""
    )


def header(title: str, sources: tuple[str, ...]) -> dict:
    joined = ", ".join(f"`{name}`" for name in sources)
    return markdown(
        f"""
# {title}

This notebook is a direct, editable migration of the complete analysis and
plotting code from {joined} in
`/cosma/home/dp203/dc-wang17/IA_analysis/anaIA`.

Existing simulation products are read in place. New figures, tables, caches,
and movies are written under this folder's `outputs/` directory.
"""
    )


def write(folder: str, filename: str, cells: list[dict]) -> None:
    target = PIPELINE / folder / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    cells.append(
        code(
            """
import csv

output_files = sorted(path for path in OUTPUT_DIR.rglob("*") if path.is_file())
with (OUTPUT_DIR / "output_manifest.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["relative_path", "bytes"])
    for path in output_files:
        writer.writerow([path.relative_to(OUTPUT_DIR), path.stat().st_size])
print(f"Output manifest contains {len(output_files)} files:", OUTPUT_DIR / "output_manifest.csv")
"""
        )
    )
    for index, cell in enumerate(cells):
        cell["id"] = hashlib.sha1(f"{folder}:{index}".encode()).hexdigest()[:12]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3.12 (py312)",
                "language": "python",
                "name": "py312",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def global_catalog() -> None:
    sources = ("global_test.ipynb", "CL_test-Copy1.ipynb", "TNGCatLoader.ipynb", "TNGCatLoader_test.ipynb")
    cells = [header("Global catalogue validation and result analysis", sources), bootstrap("00_global_catalog", sources)]
    cells.append(
        markdown(
            """
## Workflow

1. Read every completed ClusterSims and TNG global catalogue in place.
2. Validate object counts, central/satellite classification, shape convergence,
   effective particle numbers, mass ranges, and required HDF5 datasets.
3. Produce catalogue-size, convergence, and model/snapshot evolution figures.
4. Export machine-readable summary and validation tables.

The original single-object catalogue tests remain available in the preserved
API section, while the default run analyses the completed production files.
"""
        )
    )
    # Keep the original imports, but do not rerun the single-halo test. The
    # production catalogues already exist and this notebook analyses all of them.
    cells += extracted_cells("global_test.ipynb", [0])
    cells += [
        code(
            """
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

RESULT_ROOT = Path("/cosma8/data/dp203/dc-wang17/MG_global")

def summarize_global_file(path):
    match = re.search(r"(?:L302_N1136_([^_]+)_s|global_([^_]+)_s)(\\d+)", path.stem)
    flag = (match.group(1) or match.group(2)) if match else "unknown"
    snap = int(match.group(3)) if match else -1
    with h5py.File(path, "r") as h5:
        n = len(h5["SubhaloID"])
        central = np.asarray(h5["SubhaloID"]) == np.asarray(h5["CenID"])
        star_neff = np.asarray(h5["Star/Neff"], float)
        dm_neff = np.asarray(h5["DM/Neff"], float)
        required = ("SubhaloID", "CenID", "DM/I", "Star/I", "pos_abs", "vel_abs")
        return {
            "file": path.name,
            "flag": flag,
            "snap": snap,
            "objects": n,
            "central_fraction": float(np.mean(central)),
            "star_converged_fraction": float(np.mean(np.asarray(h5["Star/converged"], bool))),
            "dm_converged_fraction": float(np.mean(np.asarray(h5["DM/converged"], bool))),
            "median_star_neff": float(np.nanmedian(star_neff)),
            "median_dm_neff": float(np.nanmedian(dm_neff)),
            "required_datasets_present": all(key in h5 for key in required),
        }

GLOBAL_SUMMARY = pd.DataFrame(
    summarize_global_file(path)
    for path in sorted(RESULT_ROOT.glob("*.hdf5"))
    if not path.name.startswith("test_")
)
GLOBAL_SUMMARY.to_csv(OUTPUT_DIR / "global_catalog_summary.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
GLOBAL_SUMMARY["objects"].plot.hist(bins=20, ax=axes[0])
axes[0].set(xlabel="Objects per catalogue", title="Catalogue sizes")
GLOBAL_SUMMARY[["star_converged_fraction", "dm_converged_fraction"]].plot.box(ax=axes[1])
axes[1].set(ylabel="Fraction", title="Shape convergence")
for flag, panel in GLOBAL_SUMMARY[GLOBAL_SUMMARY.snap >= 0].groupby("flag"):
    axes[2].plot(panel["snap"], panel["objects"], marker="o", label=flag)
axes[2].set(xlabel="Snapshot", ylabel="Objects", title="Catalogue evolution")
axes[2].legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "global_catalog_summary.png", dpi=200, bbox_inches="tight")
plt.show()
GLOBAL_SUMMARY
"""
        )
    ]
    cells += source_function_library(sources, "Complete source function library")
    write("00_global_catalog", "global_catalog_analysis.ipynb", cells)


def alignment() -> None:
    sources = (
        "full_alignments.ipynb",
        "full.ipynb",
        "cluster_ia_paper_figure_suite_errorband_smooth.ipynb",
        "MAset_satellite_radial_distribution_compare.ipynb",
        "MAset_satellite_radial_distribution.ipynb",
        "tri3D.ipynb",
    )
    cells = [header("Complete alignment figure pipeline", sources), bootstrap("01_alignment", sources)]
    cells.append(
        markdown(
            """
## Scientific scope and controls

This pipeline contains galaxy--halo, subhalo--radial, tidal-field, velocity,
angular-momentum, shape-cluster, satellite radial-distribution, and 3D
triad/alignment diagnostics. The configuration cell controls models,
snapshots, shape axes, chapters, error bands, redshift colors, saving, and
display.

The default execution is intentionally a finite production run (GR, snapshot
21, galaxy major axis). Expand `REQUESTED_FLAGS`, `SNAP_LIST`, `SHAPE_AXES`,
and the `RUN_*` switches to reproduce the full publication suite. All source
functions are retained below even when their chapter is disabled by default.
"""
        )
    )
    cells += extracted_cells("full_alignments.ipynb")
    replace(
        cells,
        [
            (
                'OUTPUT_ROOT = Path("/cosma/home/dp203/dc-wang17/IA_analysis/anaIA/plots/full_alignments_publication")',
                'OUTPUT_ROOT = OUTPUT_DIR',
            ),
            ("SHOW_FIGURES = True", "SHOW_FIGURES = False"),
            ("SAVE_FIGURES = False", "SAVE_FIGURES = True"),
            ('SHAPE_AXES = ["major", "medium", "minor"]', 'SHAPE_AXES = ["major"]'),
            ("SNAP_LIST = [1, 3, 6, 8, 10, 12, 15, 18, 21]", "SNAP_LIST = [21]"),
            (
                'REQUESTED_FLAGS = ["GR", "F40", "F45", "F50", "F55", "F60"]',
                'REQUESTED_FLAGS = ["GR"]',
            ),
            ("RUN_HALO   = True", "RUN_HALO   = False"),
            ("RUN_TIDAL  = True", "RUN_TIDAL  = False"),
            ("RUN_RADIAL = True", "RUN_RADIAL = False"),
            ('mode="both"', 'mode="snapshot"'),
        ],
    )
    cells += [
        markdown(
            """
## Additional alignment diagnostics

The following summary records the loaded sample size and central fraction for
each active model/snapshot. It is useful for checking whether changes in an
alignment curve could be driven by sample composition.
"""
        ),
        code(
            """
import pandas as pd

alignment_rows = []
for flag in flags:
    for snap in SNAP_LIST:
        key = f"{int(snap):03d}"
        if key not in MAset.get(flag, {}):
            continue
        catalogue = MAset[flag][key]
        n = len(catalogue["SubhaloID"])
        central = np.asarray(catalogue["SubhaloID"]) == np.asarray(catalogue["CenID"])
        alignment_rows.append({
            "flag": flag, "snapshot": snap, "objects": n,
            "central_fraction": float(np.mean(central)),
            "satellite_fraction": float(np.mean(~central)),
        })
ALIGNMENT_SAMPLE_SUMMARY = pd.DataFrame(alignment_rows)
ALIGNMENT_SAMPLE_SUMMARY.to_csv(OUTPUT_DIR / "alignment_sample_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
for flag, panel in ALIGNMENT_SAMPLE_SUMMARY.groupby("flag"):
    ax.plot(panel["snapshot"], panel["satellite_fraction"], marker="o", label=arts_IA.flag_label(flag))
ax.set(xlabel="Snapshot", ylabel="Satellite fraction", title="Alignment sample composition")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "alignment_sample_composition.png", dpi=220, bbox_inches="tight")
plt.close(fig)
ALIGNMENT_SAMPLE_SUMMARY
"""
        ),
    ]
    cells += source_function_library(sources[1:], "Additional preserved alignment APIs")
    write("01_alignment", "alignment_pipeline.ipynb", cells)


def hod() -> None:
    sources = ("hod_measure_lrg_elg.ipynb", "HOD_LRG_ELG.ipynb", "HOD_data.ipynb")
    cells = [header("Complete LRG/ELG HOD and radial-profile pipeline", sources), bootstrap("02_hod_lrg_elg", sources)]
    cells.append(
        markdown(
            """
## Workflow and products

The notebook retains both catalogue-level HOD measurement and saved-curve
analysis. It supports LRG/ELG selections, central/satellite/total occupation,
FoF versus R200c membership, redshift-dependent cuts, environmental
quantiles, smooth interpolation, ClusterSims/TNG comparison, and halo-centric
radial profiles.

For reproducibility, the default run reads the completed HDF5 curve products.
Set `CURVE_DATA_SOURCE="catalog_first"` to recompute curves from raw
catalogues. New figures remain local to `outputs/`.
"""
        )
    )
    cells += extracted_cells("hod_measure_lrg_elg.ipynb")
    cells += extracted_cells("HOD_LRG_ELG.ipynb")
    replace(
        cells,
        [
            ('OUTPUT_DIR = Path("./hod_outputs")', 'MEASUREMENT_OUTPUT_DIR = OUTPUT_DIR / "measurements"'),
            ('OUTPUT_DIR = Path("hod_outputs")', 'MEASUREMENT_OUTPUT_DIR = OUTPUT_DIR / "measurements"'),
            ('CURVE_DATA_ROOT = Path("./curve_data_hdf5")', 'CURVE_DATA_ROOT = Path("/cosma/home/dp203/dc-wang17/IA_analysis/anaIA/curve_data_hdf5")'),
            ('CURVE_DATA_SOURCE = "catalog_first"', 'CURVE_DATA_SOURCE = "hdf5_only"'),
            (
                "zmap = {s: arts.ZMAP_ALL[s] for s in snaps}",
                "zmap = {1: 2.00, 3: 1.48, 6: 0.97, 8: 0.78, 10: 0.64, 12: 0.51, 15: 0.33, 18: 0.16, 21: 0.00}",
            ),
            ('outdir = Path("./plots")', 'outdir = OUTPUT_DIR / "cluster_hod"'),
            ('tng_outdir = Path("./plots_tng")', 'tng_outdir = OUTPUT_DIR / "tng_hod"'),
            ('radial_outdir = Path("./plots_radial_profiles")', 'radial_outdir = OUTPUT_DIR / "cluster_radial_profiles"'),
            ('tng_radial_outdir = Path("./plots_tng_radial_profiles")', 'tng_radial_outdir = OUTPUT_DIR / "tng_radial_profiles"'),
        ],
    )
    cells += [
        markdown(
            """
## Additional HOD evolution summary

This section reads the same saved curves and measures the characteristic mass
where the total occupation first exceeds unity. The diagnostic exposes
redshift and modified-gravity trends in a compact form.
"""
        ),
        code(
            """
import re
import h5py
import pandas as pd

characteristic_rows = []
curve_pattern = re.compile(r"ClusterSims_([^_]+)_snap(\\d+)\\.hdf5")
for path in sorted(HDF5_CURVE_DATA_DIR.glob("ClusterSims_*_snap*.hdf5")):
    match = curve_pattern.match(path.name)
    if not match:
        continue
    flag, snap = match.group(1), int(match.group(2))
    with h5py.File(path, "r") as h5:
        for sample in ("LRG", "ELG"):
            for region in ("fof", "r200c"):
                group = h5[f"hod/{sample}/{region}/Ntot/smooth"]
                x = np.asarray(group["x"], float)
                y = np.asarray(group["y"], float)
                valid = np.isfinite(x) & np.isfinite(y) & (y >= 1)
                characteristic_rows.append({
                    "flag": flag, "snapshot": snap, "redshift": zmap.get(snap, np.nan),
                    "sample": sample, "region": region,
                    "mass_at_Ntot_ge_1": float(x[valid][0]) if valid.any() else np.nan,
                })
HOD_CHARACTERISTIC_MASS = pd.DataFrame(characteristic_rows)
HOD_CHARACTERISTIC_MASS.to_csv(OUTPUT_DIR / "hod_characteristic_mass.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, sample in zip(axes, ("LRG", "ELG")):
    panel = HOD_CHARACTERISTIC_MASS.query("sample == @sample and region == 'r200c'")
    for flag, values in panel.groupby("flag"):
        ax.plot(values["redshift"], values["mass_at_Ntot_ge_1"], marker="o", label=flag)
    ax.set(xlabel="Redshift", title=sample, yscale="log")
    ax.invert_xaxis()
axes[0].set_ylabel(r"$M_{200c}$ where $\\langle N_{tot}\\rangle\\geq1$")
axes[0].legend(ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "hod_characteristic_mass_evolution.png", dpi=220, bbox_inches="tight")
plt.close(fig)
HOD_CHARACTERISTIC_MASS.head()
"""
        ),
    ]
    cells += source_function_library(("HOD_data.ipynb",), "Additional preserved HOD APIs")
    write("02_hod_lrg_elg", "hod_lrg_elg_pipeline.ipynb", cells)


def spectra() -> None:
    sources = ("plot_all_pks.ipynb", "plot_pks.ipynb", "pks_PK_AIA.ipynb")
    cells = [header("Complete power-spectrum and AIA publication pipeline", sources), bootstrap("03_power_spectrum_aia", sources)]
    cells.append(
        markdown(
            """
## Workflow and plotting controls

The migrated workflow discovers samples and spectra directly from the HDF5
files and retains:

- model grids at fixed redshift;
- redshift evolution by gravity model;
- enhancement relative to GR;
- signed IA cross-spectrum handling;
- AIA estimators from delta--E and galaxy--E ratios;
- smoothing, interpolation, uncertainty bands, covariance hooks, and
  publication styling.

The default run uses existing `pks_*.hdf5` products and never updates their
covariance groups. Set `USE_THEORY_COVARIANCE=True` only when an intentional
write-back/recomputation is desired.
"""
        )
    )
    cells += extracted_cells("pks_PK_AIA.ipynb")
    replace(
        cells,
        [
            ("MODULE_DIR = Path.cwd()", "MODULE_DIR = ANAIA_ROOT"),
            ('DATA_DIR = Path("../pks")', 'DATA_DIR = Path("/cosma/home/dp203/dc-wang17/IA_analysis/pks")'),
            ('OUTPUT_DIR = Path("figures_PK_AIA")', 'OUTPUT_DIR = PIPELINE_DIR / "outputs"'),
            ("SAMPLES = None  # set to None to auto-discover", 'SAMPLES = ["all", "LRG", "ELG"]  # set to None to auto-discover all samples'),
            ("USE_THEORY_COVARIANCE = True", "USE_THEORY_COVARIANCE = False"),
            ("SHOW_FIGURES = True", "SHOW_FIGURES = False"),
        ],
    )
    cells += source_function_library(
        ("plot_all_pks.ipynb", "plot_pks.ipynb"),
        "Preserved enhancement, covariance, and legacy plotting APIs",
    )
    cells += [
        markdown(
            """
## Additional GR-relative spectrum diagnostic

This compact diagnostic measures the median fractional difference from GR over
the configured k range for every model, sample, spectrum, and snapshot. It
provides a numerical companion to the publication figures.
"""
        ),
        code(
            """
enhancement_rows = []
for sample in samples:
    for spectrum in spectra_by_sample.get(sample, []):
        if spectrum not in SPECTRA:
            continue
        for snap in SNAPS:
            ref = read_spectrum_or_none(pks_path("GR", snap), sample, spectrum)
            if ref is None:
                continue
            k_ref, p_ref, _, _ = ref
            for flag in FLAGS:
                if flag == "GR":
                    continue
                got = read_spectrum_or_none(pks_path(flag, snap), sample, spectrum)
                if got is None:
                    continue
                k, power, z, _ = got
                ref_interp = np.interp(k, k_ref, p_ref, left=np.nan, right=np.nan)
                use = np.isfinite(power) & np.isfinite(ref_interp) & (ref_interp != 0)
                use &= (k >= KMIN_PLOT) & (k <= KMAX_PLOT)
                ratio = power[use] / ref_interp[use] - 1
                enhancement_rows.append({
                    "sample": sample, "spectrum": spectrum, "flag": flag,
                    "snapshot": snap, "redshift": z,
                    "median_fractional_difference": float(np.nanmedian(ratio)) if ratio.size else np.nan,
                    "median_absolute_fractional_difference": float(np.nanmedian(np.abs(ratio))) if ratio.size else np.nan,
                })
ENHANCEMENT_SUMMARY = pd.DataFrame(enhancement_rows)
ENHANCEMENT_SUMMARY.to_csv(OUTPUT_DIR / "gr_relative_spectrum_summary.csv", index=False)

panel = ENHANCEMENT_SUMMARY.query("sample == 'all' and spectrum == 'P_dd'")
fig, ax = plt.subplots(figsize=(8, 5))
for flag, values in panel.groupby("flag"):
    ax.plot(values["redshift"], values["median_fractional_difference"], marker="o",
            color=FLAG_COLOR.get(flag), label=FLAG_LABEL.get(flag, flag))
ax.axhline(0, color="k", lw=0.8)
ax.set(xlabel="Redshift", ylabel="Median P/P_GR - 1",
       title="Matter-power enhancement relative to GR")
ax.invert_xaxis()
ax.legend(ncol=2)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "all_P_dd_GR_relative_evolution.png", dpi=220, bbox_inches="tight")
plt.close(fig)
ENHANCEMENT_SUMMARY.head()
"""
        ),
    ]
    write("03_power_spectrum_aia", "power_spectrum_aia_pipeline.ipynb", cells)


def merger() -> None:
    sources = (
        "hd_tng_plot.ipynb",
        "crossz.ipynb",
        "hd_tng_crossZ.ipynb",
        "merger_align.ipynb",
        "merger_stripping.ipynb",
        "TNGCatLoader.ipynb",
    )
    cells = [header("Complete TNG merger and shell-dynamics analysis", sources), bootstrap("04_tng_merger_dynamics", sources)]
    cells.append(
        markdown(
            """
## Workflow and scientific products

The pipeline retains instantaneous and cross-time halo dynamics, main
progenitor tracking, orbit-plane visualization, radial and binding-energy
shells, affine Pi decomposition, direct/finite-difference closure tests,
Dimroth--Watson alignment fits, density maps, component fractions, and
particle binding/radial profiles.

Completed CSV products are copied into local `outputs/` before analysis.
Network/API recomputation and particle-profile recomputation are disabled by
default; enable the documented switches when those expensive stages are
required.
"""
        )
    )
    # Preserve all compute and plotting function definitions from the most complete source.
    cells += extracted_cells("merger_stripping.ipynb", [1, 3, 5, 7, 13, 16, 17, 18, 19, 20, 22])
    cells += [
        code(
            """
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SOURCE_RESULTS = Path("/cosma/home/dp203/dc-wang17/IA_analysis/anaIA/hd_tng_outputs")
OUTDIR = OUTPUT_DIR
OUTDIR.mkdir(parents=True, exist_ok=True)

def copy_analysis_tables():
    copied = []
    patterns = (
        "hd_tng_instantaneous_pi_closure.csv",
        "pi_alignment_fraction_diagnostics.csv",
        "pi_omega_H_component_fractions.csv",
        "pi_vector_balance_diagnostics.csv",
        "subhalo_*_orbit_table.csv",
        "subhalo_*_pi_closure_cross_time.csv",
    )
    for pattern in patterns:
        for source in SOURCE_RESULTS.glob(pattern):
            target = OUTDIR / source.name
            shutil.copy2(source, target)
            copied.append(target)
    return copied

COPIED_TABLES = copy_analysis_tables()
print(f"Copied {len(COPIED_TABLES)} analysis tables to {OUTDIR}")

closure = pd.read_csv(OUTDIR / "hd_tng_instantaneous_pi_closure.csv")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, component in zip(axes, ("01", "02", "12")):
    x = closure[f"Pi_direct_{component}"].to_numpy(float)
    y = closure[f"Pi_aff_{component}"].to_numpy(float)
    good = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[good], y[good], s=5, alpha=0.25)
    if good.any():
        lo = min(x[good].min(), y[good].min())
        hi = max(x[good].max(), y[good].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set(xlabel=rf"$\\Pi^{{direct}}_{{{component}}}$",
           ylabel=rf"$\\Pi^{{affine}}_{{{component}}}$", title=component)
fig.tight_layout()
fig.savefig(OUTDIR / "pi_closure_scatter_recomputed.png", dpi=220, bbox_inches="tight")
plt.show()

cross_time = []
for path in sorted(OUTDIR.glob("subhalo_*_pi_closure_cross_time.csv")):
    frame = pd.read_csv(path)
    frame["source_file"] = path.name
    cross_time.append(frame)
CROSS_TIME = pd.concat(cross_time, ignore_index=True)
CROSS_TIME.to_csv(OUTDIR / "all_subhaloes_pi_closure_cross_time.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, (method, panel) in zip(axes, CROSS_TIME.groupby("shell_method")):
    for component in ("01", "02", "12"):
        ax.scatter(panel["Redshift"], panel[f"rel_resid_{component}"], s=12, alpha=0.55, label=component)
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xlabel="Redshift", title=method)
    ax.invert_xaxis()
axes[0].set_ylabel("Relative Pi-closure residual")
axes[0].legend()
fig.tight_layout()
fig.savefig(OUTDIR / "pi_closure_cross_time_recomputed.png", dpi=220, bbox_inches="tight")
plt.show()
"""
        )
    ]
    replace(
        cells,
        [
            ("OUTDIR = Path('hd_tng_outputs')", "OUTDIR = OUTPUT_DIR"),
            ('OUTDIR = Path("hd_tng_outputs")', "OUTDIR = OUTPUT_DIR"),
            ("RUN_CROSS_TIME = True", "RUN_CROSS_TIME = False"),
            (
                "RUN_PARTICLE_PROFILE_COMPUTE = globals().get('RUN_PARTICLE_PROFILE_COMPUTE', True)",
                "RUN_PARTICLE_PROFILE_COMPUTE = False",
            ),
        ],
    )
    cells += source_function_library(
        ("hd_tng_plot.ipynb", "crossz.ipynb", "hd_tng_crossZ.ipynb", "merger_align.ipynb", "TNGCatLoader.ipynb"),
        "Preserved TNG dynamics, catalogue, and plotting APIs",
    )
    cells += [
        markdown(
            """
## Additional Pi component-balance analysis

The following plots quantify how rotational and strain terms contribute to
the affine Pi tensor and how their vector alignment changes with shell.
"""
        ),
        code(
            """
fractions = pd.read_csv(OUTDIR / "pi_omega_H_component_fractions.csv")
balance = pd.read_csv(OUTDIR / "pi_vector_balance_diagnostics.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for source, color in (("f_Omega_abs", "#35618f"), ("f_H_abs", "#b44a4a")):
    grouped = fractions.groupby("component")[source]
    axes[0].boxplot(
        [grouped.get_group(component).dropna() for component in sorted(fractions.component.unique())],
        positions=np.arange(len(sorted(fractions.component.unique()))) + (0.0 if source == "f_Omega_abs" else 0.2),
        widths=0.18, patch_artist=True,
        boxprops={"facecolor": color, "alpha": 0.55},
        medianprops={"color": "black"},
    )
axes[0].set_xticks(np.arange(len(sorted(fractions.component.unique()))) + 0.1)
axes[0].set_xticklabels(sorted(fractions.component.unique()))
axes[0].set(xlabel="Pi component", ylabel="Absolute contribution fraction",
            title="Rotation and strain contributions")

axes[1].hist(balance["cos_Omega_H"].dropna(), bins=35, density=True, alpha=0.75)
axes[1].axvline(balance["cos_Omega_H"].median(), color="k", ls="--",
                label=f"median={balance['cos_Omega_H'].median():.2f}")
axes[1].set(xlabel=r"$\\cos(\\Pi^\\Omega,\\Pi^H)$", ylabel="Density",
            title="Pi component-vector alignment")
axes[1].legend()
fig.tight_layout()
fig.savefig(OUTDIR / "pi_component_balance_summary.png", dpi=220, bbox_inches="tight")
plt.close(fig)

PI_COMPONENT_SUMMARY = fractions.groupby("component")[["f_Omega_abs", "f_H_abs"]].agg(["median", "mean", "std"])
PI_COMPONENT_SUMMARY.to_csv(OUTDIR / "pi_component_fraction_summary.csv")
PI_COMPONENT_SUMMARY
"""
        ),
    ]
    write("04_tng_merger_dynamics", "tng_merger_dynamics_pipeline.ipynb", cells)


def orbit() -> None:
    sources = ("orbit.ipynb", "merger_stripping.ipynb")
    cells = [header("Complete orbit integration and tidal-stripping pipeline", sources), bootstrap("05_orbit_stripping", sources)]
    cells.append(
        markdown(
            """
## Workflow

This notebook retains NFW host construction, orbit integration with and
without dynamical friction, mass and launch-angle sweeps, tidal-field
diagnostics, static figures, and three-/six-panel movies. Movie paths and
static outputs are redirected to this pipeline's `outputs/` directory.

The merger-stripping source API is preserved below so orbit templates can be
connected to particle and shell stripping analyses without changing
notebooks.
"""
        )
    )
    cells += extracted_cells("orbit.ipynb")
    replace(
        cells,
        [
            ('outfile="orbit6_tri.mp4"', 'outfile=str(OUTPUT_DIR / "orbit6_tri.mp4")'),
            ('outfile="orbit3_tri.mp4"', 'outfile=str(OUTPUT_DIR / "orbit3_tri.mp4")'),
            ("fig.savefig('DF.png'", 'fig.savefig(OUTPUT_DIR / "DF.png"'),
            ("fig.savefig('ang.png'", 'fig.savefig(OUTPUT_DIR / "ang.png"'),
            ('Video("/cosma/home/dp203/dc-wang17/IA_analysis/anaIA/orbit3_tri.mp4"', 'Video(str(OUTPUT_DIR / "orbit3_tri.mp4")'),
        ],
    )
    cells += source_function_library(("merger_stripping.ipynb",), "Preserved tidal-stripping and particle-profile APIs")
    cells += [
        markdown(
            """
## Additional orbit output inventory

The static summary below records the generated media and provides a quick
visual check of file sizes. Large or zero-byte movie files usually indicate an
encoder or filesystem problem.
"""
        ),
        code(
            """
import pandas as pd

ORBIT_OUTPUTS = pd.DataFrame(
    [{"file": path.name, "suffix": path.suffix, "bytes": path.stat().st_size}
     for path in sorted(OUTPUT_DIR.glob("*")) if path.is_file()]
)
ORBIT_OUTPUTS.to_csv(OUTPUT_DIR / "orbit_output_inventory.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4.5))
if len(ORBIT_OUTPUTS):
    ax.barh(ORBIT_OUTPUTS["file"], ORBIT_OUTPUTS["bytes"] / 1024**2)
ax.set(xlabel="File size [MiB]", title="Orbit pipeline output inventory")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "orbit_output_inventory.png", dpi=200, bbox_inches="tight")
plt.close(fig)
ORBIT_OUTPUTS
"""
        ),
    ]
    write("05_orbit_stripping", "orbit_stripping_pipeline.ipynb", cells)


def correlations() -> None:
    sources = ("ia_corr.ipynb", "ia_corr_abundance.ipynb", "plot_tcfs_3x5.ipynb")
    cells = [header("Complete IA correlation-function analysis", sources), bootstrap("06_correlations", sources)]
    cells.append(
        markdown(
            """
## Workflow and statistics

The notebook retains mass-bin and abundance-matched catalogue selections,
jackknife region assignment, mean/covariance estimation, one-/two-halo
decomposition, ED/EE aliases, projected w_g+ and w_++, 2x5/3x5/4x5 comparison
grids, and GR-relative model comparisons.

The default run analyses completed HDF5 correlation products. Measurement
functions from `ia_corr.ipynb` and `ia_corr_abundance.ipynb` are preserved
below for deliberate recomputation.
"""
        )
    )
    cells += extracted_cells("plot_tcfs_3x5.ipynb")
    replace(
        cells,
        [
            ("fig.savefig('./plots/etas.png')", 'fig.savefig(OUTPUT_DIR / "etas.png", dpi=220, bbox_inches="tight")'),
            ("fig.savefig('./plots/eta_1h.png')", 'fig.savefig(OUTPUT_DIR / "eta_1h.png", dpi=220, bbox_inches="tight")'),
            ("fig.savefig('./plots/eta_2h.png')", 'fig.savefig(OUTPUT_DIR / "eta_2h.png", dpi=220, bbox_inches="tight")'),
            ("fig.savefig('./plots/xis.png')", 'fig.savefig(OUTPUT_DIR / "xis.png", dpi=220, bbox_inches="tight")'),
            ("fig.savefig('./plots/wpps.png')", 'fig.savefig(OUTPUT_DIR / "wpps.png", dpi=220, bbox_inches="tight")'),
        ],
    )
    cells += [
        code(
            """
import pandas as pd

open_figures = list(plt.get_fignums())
for sequence, figure_number in enumerate(open_figures, start=1):
    figure = plt.figure(figure_number)
    target = OUTPUT_DIR / f"correlation_figure_{sequence:02d}.png"
    if not target.exists():
        figure.savefig(target, dpi=220, bbox_inches="tight")
plt.close("all")

manifest = pd.DataFrame(
    {
        "file": [path.name for path in sorted(OUTPUT_DIR.glob("*"))],
        "bytes": [path.stat().st_size for path in sorted(OUTPUT_DIR.glob("*"))],
    }
) if "pd" in globals() else None
if manifest is not None:
    manifest.to_csv(OUTPUT_DIR / "output_manifest.csv", index=False)
manifest
"""
        )
    ]
    cells += source_function_library(
        ("ia_corr.ipynb", "ia_corr_abundance.ipynb"),
        "Preserved correlation measurement and jackknife APIs",
    )
    cells += [
        markdown(
            """
## Additional covariance-quality analysis

This section summarizes covariance conditioning and signal-to-noise for every
available correlation file. It can identify noisy jackknife estimates before
publication plotting.
"""
        ),
        code(
            """
quality_rows = []
for path in sorted(BASE_CF_DIR.glob("*.hdf5")):
    with h5py.File(path, "r") as h5:
        for statistic in ("xi_tot", "ed_tot", "ee_tot", "wgp", "wpp"):
            if statistic not in h5 or "mean" not in h5[statistic]:
                continue
            mean = np.asarray(h5[statistic]["mean"], float)
            cov = np.asarray(h5[statistic]["cov"], float)
            eig = np.linalg.eigvalsh(np.nan_to_num(cov, nan=0.0))
            variance = np.clip(np.diag(cov), 0, None)
            sn2 = np.nansum(np.divide(mean**2, variance, out=np.zeros_like(mean), where=variance > 0))
            quality_rows.append({
                "file": path.name, "statistic": statistic,
                "signal_to_noise": float(np.sqrt(sn2)),
                "minimum_eigenvalue": float(eig.min()),
                "maximum_eigenvalue": float(eig.max()),
                "condition_number": float(np.inf if eig.min() <= 0 else eig.max() / eig.min()),
            })
CORRELATION_QUALITY = pd.DataFrame(quality_rows)
CORRELATION_QUALITY.to_csv(OUTPUT_DIR / "correlation_covariance_quality.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
for statistic, panel in CORRELATION_QUALITY.groupby("statistic"):
    ax.hist(panel["signal_to_noise"], bins=20, alpha=0.45, label=statistic)
ax.set(xlabel="Approximate total signal-to-noise", ylabel="Files",
       title="Correlation-product quality")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "correlation_signal_to_noise_distribution.png", dpi=220, bbox_inches="tight")
plt.close(fig)
CORRELATION_QUALITY.head()
"""
        ),
    ]
    write("06_correlations", "correlation_analysis.ipynb", cells)


def main() -> None:
    global_catalog()
    alignment()
    hod()
    spectra()
    merger()
    orbit()
    correlations()
    print(f"Rewrote full anaIA pipelines under {PIPELINE}")


if __name__ == "__main__":
    main()
