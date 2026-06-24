"""Build curated pipelines from anaIA notebooks modified in the last two weeks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT = REPO.parent / "pipeline"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip() + "\n",
    }


COMMON = r"""
from pathlib import Path
import os
import sys

def find_repo(start=None):
    start = Path.cwd() if start is None else Path(start).resolve()
    candidates = [start, *start.parents]
    candidates += [p / "ia_analysis" for p in candidates]
    for path in candidates:
        if (path / "pyproject.toml").is_file() and (path / "src" / "ia_analysis").is_dir():
            return path
    raise FileNotFoundError("Cannot locate the ia_analysis repository.")

REPO_ROOT = find_repo()
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RUNTIME_DIR = REPO_ROOT.parent / ".notebook_runtime"
RUNTIME_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_DIR / "matplotlib"))
os.environ.setdefault("IPYTHONDIR", str(RUNTIME_DIR / "ipython"))

print("Repository:", REPO_ROOT)
print("Python:", sys.executable)
"""


def write(folder: str, filename: str, cells: list[dict]) -> None:
    target = OUT / folder
    target.mkdir(parents=True, exist_ok=True)
    for old in target.glob("*.ipynb"):
        old.unlink()
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
    (target / filename).write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def alignment() -> None:
    write(
        "01_alignment",
        "alignment_pipeline.ipynb",
        [
            md(
                """
# Alignment pipeline

Source workflow: `full_alignments.ipynb` (June 15, 2026).

This pipeline covers the galaxy, halo, tidal, radial, shape-cluster, and
physical sanity-check figure groups. Computation and plotting use
`ia_analysis.visualization.alignment_catalogs` and `alignment_plots` wherever
possible. Edit the configuration cell below to control data, samples, figure
groups, size, DPI, saving, and display behavior.
"""
            ),
            code(COMMON),
            code(
                r"""
from dataclasses import dataclass, field

@dataclass
class AlignmentConfig:
    data_path: Path = Path("/cosma8/data/dp203/dc-wang17/MG_global")
    pickle_path: Path | None = None
    flags: tuple[str, ...] = ("GR",)
    snapshots: tuple[int, ...] = (21,)
    chapters: tuple[str, ...] = ("galaxy",)
    shape_axes: tuple[str, ...] = ("major",)
    selected_specs: tuple[str, ...] = ("CGHA_Mstar_major", "SGHA_Mstar_major")
    plot_mode: str = "snapshot"
    output_dir: Path = REPO_ROOT.parent / "pipeline_outputs" / "alignment"
    figsize: tuple[float, float] = (12.0, 7.0)
    dpi: int = 180
    bins: int = 12
    min_count: int = 5
    save: bool = False
    show: bool = False
    run_pipeline: bool = True

CFG = AlignmentConfig()
CFG
"""
            ),
            code(
                r"""
import matplotlib.pyplot as plt
from ia_analysis.visualization import alignment_catalogs, alignment_plots
from ia_analysis.visualization.plot_styles import set_project_style

set_project_style(rc={"figure.figsize": CFG.figsize, "savefig.dpi": CFG.dpi})

def load_alignment_data(cfg=CFG):
    if cfg.pickle_path is not None:
        maset, flags, snaps = alignment_catalogs.load_legacy_alignment_pickle(
            cfg.pickle_path,
            requested_flags=cfg.flags,
            requested_snap_list=cfg.snapshots,
        )
    else:
        maset, flags, missing = alignment_catalogs.load_alignment_catalogs(
            cfg.data_path,
            requested_flags=cfg.flags,
            snap_list=cfg.snapshots,
        )
    snaps = tuple(cfg.snapshots)
    alignment_catalogs.configure_alignment_context(maset, flags, snap_list=snaps)
    return maset, flags, snaps, missing

def available_workflow():
    return {
        chapter: alignment_plots.list_alignment_specs(chapter)
        for chapter in alignment_plots.list_alignment_chapters()
    }

print("Use available_workflow() to list every chapter and alignment specification.")
"""
            ),
            code(
                r"""
def run_alignment_chapters(cfg=CFG):
    maset, flags, snaps, missing = load_alignment_data(cfg)
    if cfg.save:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
    figures = {}
    for chapter in cfg.chapters:
        figures[chapter] = alignment_plots.plot_alignment_chapter(
            chapter,
            shape_axes=cfg.shape_axes,
            snap_list=snaps,
            flags_to_use=flags,
            output_root=cfg.output_dir,
            save=cfg.save,
            show=cfg.show,
        )
    return {
        "MAset": maset, "flags": flags, "snapshots": snaps,
        "missing": missing, "figures": figures,
    }

def run_alignment_pipeline(cfg=CFG):
    maset, flags, snaps, missing = load_alignment_data(cfg)
    if cfg.save:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        name: alignment_plots.plot_alignment(
            name, mode=cfg.plot_mode, snap_list=snaps, flags_to_use=flags,
            bins=cfg.bins, min_count=cfg.min_count,
            output_root=cfg.output_dir, save=cfg.save, show=cfg.show,
        )
        for name in cfg.selected_specs
    }
    return {
        "MAset": maset, "flags": flags, "snapshots": snaps,
        "missing": missing, "figures": figures,
    }

RESULT = run_alignment_pipeline() if CFG.run_pipeline else None
print("Generated alignment analyses:", [] if RESULT is None else sorted(RESULT["figures"]))
print("Call run_alignment_chapters() to execute the configured full chapters.")
"""
            ),
            md(
                """
## Single-figure customization

Call `alignment_plots.plot_alignment_pair(name, ...)` for one figure. Use
`available_workflow()` to list all figures mapped from each chapter of the
source notebook. Matplotlib settings can also be changed through the returned
`fig, axes` objects.
"""
            ),
        ],
    )


def global_catalog() -> None:
    write(
        "00_global_catalog",
        "global_catalog_analysis.ipynb",
        [
            md(
                """
# Global catalog result analysis

Computation entrypoints in this directory:

- `submit_global.sh`
- `run_global.slurm`
- `run_cs.py`
- `run_tng.py`
- `global_tng.py`

This notebook does not produce catalogs. It reads the HDF5 products generated
by the Slurm jobs and checks fields, object counts, shape-tensor completeness,
and stellar-particle selections.
"""
            ),
            code(COMMON),
            code(
                r"""
from dataclasses import dataclass

@dataclass
class GlobalAnalysisConfig:
    result_dir: Path = Path("/cosma8/data/dp203/dc-wang17/MG_global")
    pattern: str = "*.hdf5"
    include_test_files: bool = False
    flags: tuple[str, ...] = ("GR", "F40", "F45", "F50", "F55", "F60", "TNG")
    snapshots: tuple[int, ...] = (1, 3, 6, 8, 10, 12, 15, 18, 21, 99)
    figsize: tuple[float, float] = (11.0, 5.0)
    dpi: int = 160

CFG = GlobalAnalysisConfig()
CFG
"""
            ),
            code(
                r"""
import re
import h5py
import numpy as np
import pandas as pd

def parse_catalog_name(path):
    cs = re.match(r"L302_N1136_([^_]+)_s(\d+)\.hdf5$", path.name)
    if cs:
        return "run_cs", cs.group(1), int(cs.group(2))
    tng = re.match(r"global_TNG_s(\d+)\.hdf5$", path.name)
    if tng:
        return "run_tng", "TNG", int(tng.group(1))
    return "other", "unknown", -1

def summarize_catalog(path):
    pipeline, flag, snap = parse_catalog_name(path)
    with h5py.File(path, "r") as h5:
        n = int(h5["SubhaloID"].shape[0])
        central = np.asarray(h5["SubhaloID"]) == np.asarray(h5["CenID"])
        star_conv = np.asarray(h5["Star/converged"], bool)
        dm_conv = np.asarray(h5["DM/converged"], bool)
        star_neff = np.asarray(h5["Star/Neff"], float)
        dm_neff = np.asarray(h5["DM/Neff"], float)
        star_mass = np.asarray(h5["Star/mass"], float)
        dm_mass = np.asarray(h5["DM/mass"], float)
        tidal_complete = ("Tidal_tot" in h5) or ("Tidal/Tidal_tot" in h5)
        required_complete = all(key in h5 for key in ("DM/I", "Star/I", "pos_abs", "vel_abs"))
    return {
        "pipeline": pipeline,
        "flag": flag,
        "snap": snap,
        "file": path.name,
        "objects": n,
        "central_fraction": float(np.mean(central)),
        "star_converged": float(np.mean(star_conv)),
        "dm_converged": float(np.mean(dm_conv)),
        "median_star_neff": float(np.nanmedian(star_neff)),
        "median_dm_neff": float(np.nanmedian(dm_neff)),
        "median_star_mass": float(np.nanmedian(star_mass[star_mass > 0])),
        "median_dm_mass": float(np.nanmedian(dm_mass[dm_mass > 0])),
        "complete": required_complete and tidal_complete,
    }

FILES = sorted(CFG.result_dir.glob(CFG.pattern))
if not CFG.include_test_files:
    FILES = [p for p in FILES if not p.name.startswith("test_")]
SUMMARY = pd.DataFrame([summarize_catalog(path) for path in FILES])
if len(SUMMARY):
    SUMMARY = SUMMARY[SUMMARY["flag"].isin(CFG.flags) & SUMMARY["snap"].isin(CFG.snapshots)]
SUMMARY.sort_values(["pipeline", "snap", "flag"]).reset_index(drop=True)
"""
            ),
            code(
                r"""
import matplotlib.pyplot as plt
import seaborn as sns
from ia_analysis.visualization.plot_styles import set_project_style

set_project_style(rc={"figure.figsize": CFG.figsize, "savefig.dpi": CFG.dpi})

def plot_catalog_summary(summary=SUMMARY):
    fig, axes = plt.subplots(1, 2, figsize=CFG.figsize)
    if len(summary):
        cs = summary[summary.pipeline == "run_cs"]
        pivot = cs.pivot(index="flag", columns="snap", values="objects")
        sns.heatmap(pivot, annot=False, cmap="viridis", ax=axes[0], cbar_kws={"label": "objects"})
        axes[0].set_title("run_cs output object counts")
        quality = summary.groupby("pipeline")[["star_converged", "dm_converged"]].mean()
        quality.plot.bar(ax=axes[1])
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("converged fraction")
        axes[1].tick_params(axis="x", rotation=0)
    else:
        axes[0].text(0.5, 0.5, "No files found", ha="center")
        axes[1].set_axis_off()
    fig.tight_layout()
    return fig, axes

FIG, AXES = plot_catalog_summary()
plt.close(FIG)
print(f"Found {len(FILES)} result files in {CFG.result_dir}")
SUMMARY
"""
            ),
        ],
    )


def hod() -> None:
    write(
        "02_hod_lrg_elg",
        "hod_lrg_elg_pipeline.ipynb",
        [
            md(
                """
# LRG/ELG HOD and radial-profile pipeline

Source workflows: `hod_measure_lrg_elg.ipynb` and `HOD_LRG_ELG.ipynb`
(June 16, 2026).

The pipeline supports HOD measurements from existing galaxy catalogs and
central/satellite occupation and halo-centric radial profiles derived from
ClusterSims or TNG catalogs. Plotting parameters are centralized in the
configuration cell.
"""
            ),
            code(COMMON),
            code(
                r"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

@dataclass
class HODConfig:
    curve_dir: Path = Path("/cosma/home/dp203/dc-wang17/IA_analysis/anaIA/curve_data_hdf5")
    plot_snapshot: int = 21
    models: tuple[str, ...] = ("GR", "F6", "F5", "F4")
    snapshots: tuple[int, ...] = (8, 10, 12, 15, 18, 21)
    mass_bins: np.ndarray = field(default_factory=lambda: np.logspace(11.0, 15.5, 25))
    radial_bins: np.ndarray = field(default_factory=lambda: np.linspace(0.0, 2.0, 25))
    min_halos_per_bin: int = 3
    output_dir: Path = REPO_ROOT.parent / "pipeline_outputs" / "hod_lrg_elg"
    figsize: tuple[float, float] = (10.0, 7.0)
    dpi: int = 180
    colors: tuple[str, str, str] = ("#222222", "#1f77b4", "#d62728")
    save: bool = False
    show: bool = True
    run_pipeline: bool = True

CFG = HODConfig()
CFG
"""
            ),
            code(
                r"""
from ia_analysis.catalogs import CSCatalog, TNGCatalog
from ia_analysis.visualization.plot_styles import set_project_style

set_project_style(rc={"figure.figsize": CFG.figsize, "savefig.dpi": CFG.dpi})

def safe_ssfr(sfr, stellar_mass_msun):
    sfr = np.asarray(sfr, float)
    mass = np.asarray(stellar_mass_msun, float)
    out = np.full_like(sfr, np.nan)
    good = np.isfinite(sfr) & np.isfinite(mass) & (sfr > 0) & (mass > 0)
    out[good] = np.log10(sfr[good] / mass[good])
    return out

def select_lrg_elg(stellar_mass_msun, sfr, *, lrg_mass_cut=8.2e10, elg_logssfr_cut=-9.6):
    mass = np.asarray(stellar_mass_msun, float)
    logssfr = safe_ssfr(sfr, mass)
    return {
        "LRG": np.isfinite(mass) & (mass >= lrg_mass_cut),
        "ELG": np.isfinite(logssfr) & (logssfr >= elg_logssfr_cut),
    }

def occupation_counts(group_mass, group_id, central_mask, selected):
    group_mass = np.asarray(group_mass, float)
    group_id = np.asarray(group_id, int)
    central_mask = np.asarray(central_mask, bool)
    selected = np.asarray(selected, bool)
    nhalo = len(group_mass)
    valid = selected & (group_id >= 0) & (group_id < nhalo)
    total = np.bincount(group_id[valid], minlength=nhalo)
    cen = np.bincount(group_id[valid & central_mask], minlength=nhalo)
    sat = total - cen
    return pd.DataFrame({"halo_mass": group_mass, "Ntotal": total, "Ncen": cen, "Nsat": sat})

def binned_hod(counts, *, bins=CFG.mass_bins, min_halos=CFG.min_halos_per_bin):
    mass = counts["halo_mass"].to_numpy(float)
    index = np.digitize(mass, bins) - 1
    rows = []
    for i in range(len(bins) - 1):
        take = index == i
        if take.sum() < min_halos:
            continue
        row = {"mass": np.sqrt(bins[i] * bins[i + 1]), "Nhalo": int(take.sum())}
        for key in ("Ntotal", "Ncen", "Nsat"):
            values = counts.loc[take, key].to_numpy(float)
            row[key] = float(np.mean(values))
            row[key + "_err"] = float(np.std(values) / np.sqrt(values.size))
        rows.append(row)
    return pd.DataFrame(rows)

def radial_profile(radius_over_r200, selected, *, bins=CFG.radial_bins):
    radius = np.asarray(radius_over_r200, float)
    selected = np.asarray(selected, bool)
    values = radius[selected & np.isfinite(radius) & (radius >= bins[0]) & (radius <= bins[-1])]
    count, edges = np.histogram(values, bins=bins)
    width = np.diff(edges)
    return pd.DataFrame({"radius": 0.5 * (edges[:-1] + edges[1:]), "count": count, "density": count / width})
"""
            ),
            code(
                r"""
import matplotlib.pyplot as plt
from ia_analysis.visualization.profile_plots import draw_series_with_band

def plot_hod(hod, *, title="", cfg=CFG):
    fig, ax = plt.subplots(figsize=cfg.figsize)
    styles = {"Ntotal": "-", "Ncen": "--", "Nsat": ":"}
    colors = dict(zip(styles, cfg.colors))
    for key, linestyle in styles.items():
        draw_series_with_band(
            ax, hod["mass"], hod[key], hod.get(key + "_err"),
            color=colors[key], label=key, linestyle=linestyle, marker="o",
        )
    ax.set(xscale="log", yscale="log", xlabel=r"$M_h\,[M_\odot/h]$", ylabel=r"$\langle N|M_h\rangle$", title=title)
    ax.legend()
    fig.tight_layout()
    return fig, ax

def plot_radial_profile(profile, *, title="", cfg=CFG):
    fig, ax = plt.subplots(figsize=cfg.figsize)
    ax.plot(profile["radius"], profile["density"], marker="o", color=cfg.colors[1])
    ax.set(xlabel=r"$r/R_{200}$", ylabel="count density", title=title)
    fig.tight_layout()
    return fig, ax
"""
            ),
            code(
                r"""
import re
import h5py
from ia_analysis.visualization.plot_styles import model_color, model_label

def discover_curve_files(cfg=CFG):
    pattern = re.compile(r"(?:ClusterSims_([^_]+)_snap|TNG_([^_]+)_snap)(\d+)\.hdf5$")
    rows = []
    for path in sorted(cfg.curve_dir.glob("*.hdf5")):
        match = pattern.match(path.name)
        if not match:
            continue
        label = match.group(1) or match.group(2) or "TNG"
        with h5py.File(path, "r") as h5:
            rows.append({
                "label": label, "snap": int(match.group(3)),
                "z": float(h5["meta"].attrs.get("z", np.nan)), "path": path,
            })
    return pd.DataFrame(rows)

def read_saved_curve(path, group_path, kind="smooth"):
    with h5py.File(path, "r") as h5:
        group = h5[f"{group_path}/{kind}"]
        return np.asarray(group["x"], float), np.asarray(group["y"], float)

def plot_saved_hod(sample="LRG", region="r200c", component="Ntot", snap=21, cfg=CFG):
    fig, ax = plt.subplots(figsize=cfg.figsize)
    panel = CURVE_FILES[CURVE_FILES.snap == int(snap)]
    for row in panel.itertuples():
        x, y = read_saved_curve(row.path, f"hod/{sample}/{region}/{component}")
        if x.size:
            ax.plot(x, y, color=model_color(row.label), label=model_label(row.label))
    ax.set(xscale="log", yscale="log", xlabel=r"$M_{200c}\,[M_\odot/h]$",
           ylabel=rf"$\langle {component}\rangle$", title=f"{sample}, {region}, snap={snap}")
    ax.legend(ncol=2)
    fig.tight_layout()
    return fig, ax

def plot_saved_radial_profiles(sample="LRG", snap=21, cfg=CFG):
    panel = CURVE_FILES[CURVE_FILES.snap == int(snap)]
    if panel.empty:
        raise FileNotFoundError(f"No saved radial profiles for snap={snap}")
    with h5py.File(panel.iloc[0].path, "r") as h5:
        mass_bins = sorted(h5[f"radial_profiles/{sample}"].keys())
    fig, axes = plt.subplots(2, 2, figsize=cfg.figsize, sharex=True)
    for ax, mass_bin in zip(axes.flat, mass_bins):
        for row in panel.itertuples():
            x, y = read_saved_curve(row.path, f"radial_profiles/{sample}/{mass_bin}")
            if x.size:
                ax.plot(x, y, color=model_color(row.label), label=model_label(row.label))
        ax.set(yscale="log", title=mass_bin.replace("_", " "))
        ax.set_xlabel(r"$r/R_{200c}$")
        ax.set_ylabel("mean shell count")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        axes.flat[0].legend(ncol=2, fontsize=8)
    fig.suptitle(f"{sample} radial profiles, snap={snap}")
    fig.tight_layout()
    return fig, axes

CURVE_FILES = discover_curve_files()
print(f"Found {len(CURVE_FILES)} saved HOD/radial-profile files in {CFG.curve_dir}")
HOD_FIGURE = plot_saved_hod(snap=CFG.plot_snapshot) if CFG.run_pipeline and len(CURVE_FILES) else None
RADIAL_FIGURE = plot_saved_radial_profiles(snap=CFG.plot_snapshot) if CFG.run_pipeline and len(CURVE_FILES) else None
"""
            ),
            code(
                r"""
def run_from_arrays(group_mass, group_id, central_mask, stellar_mass_msun, sfr, radius_over_r200=None, cfg=CFG):
    selections = select_lrg_elg(stellar_mass_msun, sfr)
    products = {}
    for sample, selected in selections.items():
        counts = occupation_counts(group_mass, group_id, central_mask, selected)
        products[sample] = {"counts": counts, "hod": binned_hod(counts)}
        if radius_over_r200 is not None:
            products[sample]["radial_profile"] = radial_profile(radius_over_r200, selected)
    return products

print("Saved curves are plotted above. Use run_from_arrays(...) to recompute from catalog columns.")
"""
            ),
        ],
    )


def spectra() -> None:
    write(
        "03_power_spectrum_aia",
        "power_spectrum_aia_pipeline.ipynb",
        [
            md(
                """
# Power-spectrum and AIA publication pipeline

Source workflows: `plot_all_pks.ipynb`, `plot_pks.ipynb`, and
`pks_PK_AIA.ipynb` (June 14-16, 2026).

The notebook discovers `pks_FLAG_SNAP.hdf5` files, samples, and all `P_*_Pk`
datasets automatically. It uses `ia_analysis.visualization.spectrum_plots` and
retains covariance and AIA extension points. Plot ranges, $kP(k)$ scaling,
signed-spectrum handling, and output formats are configurable.
"""
            ),
            code(COMMON),
            code(
                r"""
from dataclasses import dataclass

@dataclass
class SpectrumConfig:
    data_dir: Path = Path("/cosma/home/dp203/dc-wang17/IA_analysis/pks")
    output_dir: Path = Path("/cosma/home/dp203/dc-wang17/IA_analysis/figures_PK_AIA")
    flags: tuple[str, ...] = ("GR", "F40", "F45", "F50", "F55", "F60")
    snapshots: tuple[int, ...] = (1, 3, 6, 8, 10, 12, 15, 18, 21)
    samples: tuple[str, ...] | None = ("all",)
    spectra: tuple[str, ...] | None = ("P_dd", "P_dE", "P_gE", "P_EE")
    plot_snapshot: int = 21
    plot_sample: str = "all"
    plot_spectrum: str = "P_dd"
    source_group: str = "stitched_native_corr"
    xlim: tuple[float, float] = (0.2, 20.0)
    figsize: tuple[float, float] = (10.0, 7.0)
    dpi: int = 220
    plot_k_times_pk: bool = True
    absolute_signed_spectra: bool = True
    save: bool = False
    show: bool = True
    run_pipeline: bool = True

CFG = SpectrumConfig()
CFG
"""
            ),
            code(
                r"""
import re
import h5py
import numpy as np
import pandas as pd

def discover_files(cfg=CFG):
    rows = []
    pattern = re.compile(r"pks_([^_]+)_(\d+)\.hdf5$")
    for path in sorted(cfg.data_dir.glob("pks_*.hdf5")):
        match = pattern.match(path.name)
        if match:
            rows.append({"flag": match.group(1), "snap": int(match.group(2)), "path": path})
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame[frame.flag.isin(cfg.flags) & frame.snap.isin(cfg.snapshots)]
    return frame.reset_index(drop=True)

def list_samples(path, source_group=CFG.source_group):
    with h5py.File(path, "r") as h5:
        return sorted(
            key for key in h5
            if isinstance(h5[key], h5py.Group) and source_group in h5[key]
        )

def list_spectra(path, sample, source_group=CFG.source_group):
    with h5py.File(path, "r") as h5:
        group = h5[sample][source_group]
        return sorted(key[:-3] for key in group if key.startswith("P_") and key.endswith("_Pk"))

def read_spectrum(path, sample, spectrum, source_group=CFG.source_group):
    with h5py.File(path, "r") as h5:
        group = h5[sample][source_group]
        k_key = f"{spectrum}_k" if f"{spectrum}_k" in group else "k"
        return {
            "k": np.asarray(group[k_key], float),
            "P": np.asarray(group[f"{spectrum}_Pk"], float),
            "z": float(h5.attrs.get("z", np.nan)),
        }

def inventory(cfg=CFG):
    files = discover_files(cfg)
    rows = []
    for item in files.itertuples():
        for sample in list_samples(item.path, cfg.source_group):
            if cfg.samples is not None and sample not in cfg.samples:
                continue
            for spectrum in list_spectra(item.path, sample, cfg.source_group):
                if cfg.spectra is None or spectrum in cfg.spectra:
                    rows.append({**item._asdict(), "sample": sample, "spectrum": spectrum})
    return pd.DataFrame(rows)

FILES = discover_files()
print(f"Found {len(FILES)} spectrum files in {CFG.data_dir}")
"""
            ),
            code(
                r"""
import matplotlib.pyplot as plt
from ia_analysis.visualization.plot_styles import model_color, model_label, set_project_style
from ia_analysis.visualization.spectrum_plots import draw_power_spectrum_series, draw_ratio_series, draw_aia_series
from ia_analysis.spectra.theory_nla_pk import build_cosmo, growth_factor

set_project_style(rc={"figure.figsize": CFG.figsize, "savefig.dpi": CFG.dpi})

def plot_model_comparison(sample, spectrum, snap, cfg=CFG):
    files = discover_files(cfg)
    panel = files[files.snap == int(snap)]
    fig, ax = plt.subplots(figsize=cfg.figsize)
    loaded = {}
    for row in panel.itertuples():
        if sample not in list_samples(row.path, cfg.source_group):
            continue
        if spectrum not in list_spectra(row.path, sample, cfg.source_group):
            continue
        data = read_spectrum(row.path, sample, spectrum, cfg.source_group)
        power = data["k"] * data["P"] if cfg.plot_k_times_pk else data["P"]
        if cfg.absolute_signed_spectra:
            power = np.abs(power)
        draw_power_spectrum_series(
            ax, data["k"], power, color=model_color(row.flag),
            label=model_label(row.flag), log_axes=True,
        )
        loaded[row.flag] = data
    ax.set_xlim(*cfg.xlim)
    ax.set_title(f"{sample}: {spectrum}, snap={snap}")
    ax.legend()
    fig.tight_layout()
    return fig, ax, loaded

def plot_enhancement(sample, spectrum, snap, reference="GR", cfg=CFG):
    _, _, loaded = plot_model_comparison(sample, spectrum, snap, cfg)
    ref = loaded[reference]
    fig, ax = plt.subplots(figsize=cfg.figsize)
    for flag, data in loaded.items():
        if flag == reference or not np.array_equal(data["k"], ref["k"]):
            continue
        draw_ratio_series(ax, data["k"], data["P"], ref["P"], color=model_color(flag), label=model_label(flag))
    ax.set_xlim(*cfg.xlim)
    ax.legend()
    fig.tight_layout()
    return fig, ax

def read_aia(path, sample, method="deltaE", cfg=CFG):
    pairs = {"deltaE": ("P_dE", "P_dd"), "gE": ("P_gE", "P_dg")}
    numerator, denominator = pairs[method]
    num = read_spectrum(path, sample, numerator, cfg.source_group)
    den = read_spectrum(path, sample, denominator, cfg.source_group)
    den_power = np.interp(num["k"], den["k"], den["P"], left=np.nan, right=np.nan)
    with h5py.File(path, "r") as h5:
        omega_c = float(h5.attrs["Omega_c"])
        omega_b = float(h5.attrs["Omega_b"])
        cosmo_dict = {
            "Omega0": omega_c + omega_b,
            "OmegaBaryon": omega_b,
            "HubbleParam": float(h5.attrs["h"]),
            "sigma8": float(h5.attrs["sigma8"]),
            "n_s": float(h5.attrs["n_s"]),
        }
        z = float(h5.attrs["z"])
    D = float(growth_factor(build_cosmo(cosmo_dict), z))
    omega_m = cosmo_dict["Omega0"]
    prefactor = -3.0 * D / (2.0 * 0.0134 * omega_m)
    with np.errstate(divide="ignore", invalid="ignore"):
        aia = prefactor * num["P"] / den_power
    return {"k": num["k"], "AIA": aia, "z": z, "prefactor": prefactor}

def plot_aia_comparison(sample, method, snap, cfg=CFG):
    fig, ax = plt.subplots(figsize=cfg.figsize)
    for row in discover_files(cfg).query("snap == @snap").itertuples():
        try:
            data = read_aia(row.path, sample, method, cfg)
        except (KeyError, ValueError):
            continue
        draw_aia_series(ax, data["k"], data["AIA"], color=model_color(row.flag), label=model_label(row.flag))
    ax.set_xlim(*cfg.xlim)
    ax.set_title(f"{sample}: AIA({method}), snap={snap}")
    ax.legend()
    fig.tight_layout()
    return fig, ax

print("Use inventory(), plot_model_comparison(...), plot_enhancement(...), and plot_aia_comparison(...).")
"""
            ),
            code(
                r"""
from ia_analysis.covariance import write_covariance_hdf5_group

def ensure_covariance(path, sample, **kwargs):
    # Write/update the ia_analysis covariance group for one sample.
    return write_covariance_hdf5_group(path, sample=sample, **kwargs)

if CFG.run_pipeline:
    table = inventory(CFG)
    PK_MODEL_FIGURE = plot_model_comparison(CFG.plot_sample, CFG.plot_spectrum, CFG.plot_snapshot, CFG)
    PK_ENHANCEMENT_FIGURE = plot_enhancement(CFG.plot_sample, CFG.plot_spectrum, CFG.plot_snapshot, cfg=CFG)
    AIA_DELTA_E_FIGURE = plot_aia_comparison(CFG.plot_sample, "deltaE", CFG.plot_snapshot, CFG)
    AIA_G_E_FIGURE = plot_aia_comparison(CFG.plot_sample, "gE", CFG.plot_snapshot, CFG)
    print(table.head())
else:
    print("Set CFG.run_pipeline=True to execute the saved-result analysis.")
"""
            ),
        ],
    )


def merger() -> None:
    write(
        "04_tng_merger_dynamics",
        "tng_merger_dynamics_pipeline.ipynb",
        [
            md(
                r"""
# TNG merger-tree alignment and shell-dynamics pipeline

Source workflows: `crossz.ipynb`, `hd_tng_plot.ipynb`, and
`merger_align.ipynb` (June 10-16, 2026).

The workflow is separated into compute/save and load/plot phases. By default,
it only validates configuration and APIs and does not download TNG data.
Orbit-plane, $\Pi$-closure, shell-density, and particle-profile plots use
`ia_analysis.visualization.merger_tree_plots`.
"""
            ),
            code(COMMON),
            code(
                r"""
from dataclasses import dataclass

@dataclass
class MergerConfig:
    base_path: Path = Path("/cosma8/data/dp203/dc-wang17/TNG/tng_data")
    sim_name: str = "TNG300-1"
    snap0: int = 99
    subhalo_id0: int = 0
    snapshots: tuple[int, ...] = (99, 91, 84, 72, 67, 59, 50, 40, 33)
    components: tuple[str, ...] = ("dm", "stars", "gas")
    shell_methods: tuple[str, ...] = ("radial", "binding_energy")
    existing_dir: Path = Path("/cosma/home/dp203/dc-wang17/IA_analysis/anaIA/hd_tng_outputs")
    output_file: Path = REPO_ROOT.parent / "pipeline_outputs" / "tng_merger" / "cross_time.pkl"
    figsize: tuple[float, float] = (10.0, 7.0)
    dpi: int = 180
    cmap: str = "viridis"
    save_figures: bool = False
    show: bool = True
    compute: bool = False
    load_existing: bool = False

CFG = MergerConfig()
CFG
"""
            ),
            code(
                r"""
from ia_analysis.MergerTree import (
    run_cross_time_workflow,
    save_cross_time_products,
    load_cross_time_products,
)
from ia_analysis.visualization.plot_styles import set_project_style

set_project_style(rc={"figure.figsize": CFG.figsize, "savefig.dpi": CFG.dpi})

def compute_products(cfg=CFG):
    products = run_cross_time_workflow(
        cfg.base_path,
        cfg.snap0,
        cfg.subhalo_id0,
        cfg.snapshots,
        sim_name=cfg.sim_name,
        api_key=os.environ.get("TNG_API_KEY"),
        components=cfg.components,
        shape_components=("dm", "stars"),
        shell_methods=cfg.shell_methods,
        reference="host_central",
        compute_closure=True,
    )
    cfg.output_file.parent.mkdir(parents=True, exist_ok=True)
    save_cross_time_products(products, cfg.output_file)
    return products

PRODUCTS = compute_products() if CFG.compute else (
    load_cross_time_products(CFG.output_file) if CFG.load_existing and CFG.output_file.exists() else None
)
print("Products loaded:", PRODUCTS is not None)
"""
            ),
            code(
                r"""
from ia_analysis.visualization import merger_tree_plots

def plot_available_products(products, cfg=CFG):
    figures = {}
    if products is None:
        return figures
    track = products.get("track_table")
    closure = products.get("closure_table", products.get("closure_all"))
    shell_table = products.get("shell_table")
    profile_table = products.get("particle_profile_table")

    if track is not None and {"x", "y", "SnapNum"}.issubset(track.columns):
        figures["orbit"] = merger_tree_plots.plot_orbit_plane_evolution(
            track, cmap=cfg.cmap, figsize=cfg.figsize
        )
    if closure is not None and len(closure):
        figures["closure"] = merger_tree_plots.plot_pi_closure_evolution(closure)
    if shell_table is not None and len(shell_table):
        figures["shells"] = merger_tree_plots.plot_shell_density_summary(shell_table, cmap=cfg.cmap)
    if profile_table is not None and len(profile_table):
        figures["profiles"] = merger_tree_plots.plot_particle_profile_comparison(profile_table)
    return figures

FIGURES = plot_available_products(PRODUCTS)
print("Generated figure groups:", sorted(FIGURES))
"""
            ),
            code(
                r"""
import pandas as pd
import numpy as np
from ia_analysis.visualization import tng_dynamics_plots

def load_existing_diagnostics(cfg=CFG):
    paths = {
        "closure": cfg.existing_dir / "hd_tng_instantaneous_pi_closure.csv",
        "alignment": cfg.existing_dir / "pi_vector_balance_diagnostics.csv",
        "fractions": cfg.existing_dir / "pi_omega_H_component_fractions.csv",
    }
    return {name: pd.read_csv(path) for name, path in paths.items() if path.exists()}

def plot_existing_diagnostics(tables):
    figures = {}
    closure = tables.get("closure")
    if closure is not None and len(closure):
        figures["pi_closure"] = tng_dynamics_plots.plot_pi_closure_table(closure)
        figures["pi_residuals"] = tng_dynamics_plots.plot_pi_residual_histogram(closure)
    alignment = tables.get("alignment")
    if alignment is not None and "cos_Omega_H" in alignment:
        figures["omega_h_alignment"] = tng_dynamics_plots.plot_dw_alignment_distribution(
            np.abs(alignment["cos_Omega_H"].to_numpy(float)), fit=True,
            title=r"$|\cos(\Pi^\Omega,\Pi^H)|$",
        )
    fractions = tables.get("fractions")
    if fractions is not None and len(fractions):
        long = fractions.melt(
            id_vars=["component"], value_vars=["f_Omega_abs", "f_H_abs"],
            var_name="source", value_name="fraction",
        )
        long["component"] = long["component"].astype(str) + " " + long["source"]
        figures["component_fractions"] = tng_dynamics_plots.plot_component_fraction_panel(
            long, value_col="fraction", title="Affine Pi component fractions",
        )
    return figures

EXISTING_TABLES = load_existing_diagnostics()
EXISTING_FIGURES = plot_existing_diagnostics(EXISTING_TABLES)
print("Loaded existing diagnostics:", sorted(EXISTING_TABLES))
print("Generated diagnostic figures:", sorted(EXISTING_FIGURES))
"""
            ),
            code(
                r"""
def load_cross_time_closure(cfg=CFG):
    frames = []
    for path in sorted(cfg.existing_dir.glob("subhalo_*_pi_closure_cross_time.csv")):
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def plot_cross_time_closure(frame):
    figures = {}
    if frame.empty:
        return figures
    for method, panel in frame.groupby("shell_method"):
        long = pd.concat(
            [
                panel.assign(component=component, relative_residual=panel[f"rel_resid_{component}"])
                for component in ("01", "02", "12")
                if f"rel_resid_{component}" in panel
            ],
            ignore_index=True,
        )
        figures[method] = merger_tree_plots.plot_pi_closure_evolution(
            long, x_col="Redshift", component_col="component",
            value_col="relative_residual", invert_xaxis=True,
            ylabel=r"$(\Pi_{\rm aff}-\Pi_{\rm direct})/|\Pi_{\rm direct}|$",
        )
    return figures

CROSS_TIME_CLOSURE = load_cross_time_closure()
CROSS_TIME_FIGURES = plot_cross_time_closure(CROSS_TIME_CLOSURE)
print(f"Loaded {len(CROSS_TIME_CLOSURE)} cross-time closure rows.")
print("Generated cross-time figures:", sorted(CROSS_TIME_FIGURES))
"""
            ),
        ],
    )


def correlations() -> None:
    write(
        "06_correlations",
        "correlation_analysis.ipynb",
        [
            md(
                """
# Correlation-function result analysis

Computation entrypoints in this directory are `submit_cf.sh`, `run_cf.slurm`,
and `run_cf.py`. Slurm jobs produce `tcfs_FLAG_SNAP.hdf5`; this notebook only
reads the means and jackknife covariance and performs model comparisons and
redshift-evolution analysis.
"""
            ),
            code(COMMON),
            code(
                r"""
from dataclasses import dataclass

@dataclass
class CorrelationConfig:
    result_dir: Path = Path("/cosma/home/dp203/dc-wang17/IA_analysis/cfs")
    flags: tuple[str, ...] = ("GR", "F40", "F45", "F50", "F55", "F60")
    snapshots: tuple[int, ...] = (1, 6, 8, 12, 15, 18, 21)
    mass_selection: str = "Mstar_103"
    statistic: str = "ed_tot"
    figsize: tuple[float, float] = (9.0, 6.0)
    dpi: int = 180

CFG = CorrelationConfig()
CFG
"""
            ),
            code(
                r"""
import re
import h5py
import numpy as np
import pandas as pd

def discover_results(cfg=CFG):
    patterns = (
        re.compile(r"tcfs_([^_]+)_(\d+)\.hdf5$"),
        re.compile(r"(Mstar_\d+)_([^_]+)_(\d+)\.hdf5$"),
    )
    rows = []
    for path in sorted(cfg.result_dir.glob("*.hdf5")):
        match = patterns[0].match(path.name)
        if match:
            rows.append({"selection": "all", "flag": match.group(1), "snap": int(match.group(2)), "path": path})
            continue
        match = patterns[1].match(path.name)
        if match:
            rows.append({"selection": match.group(1), "flag": match.group(2), "snap": int(match.group(3)), "path": path})
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame[
            frame.flag.isin(cfg.flags) & frame.snap.isin(cfg.snapshots)
            & (frame.selection == cfg.mass_selection)
        ]
    return frame.reset_index(drop=True)

def find_dataset(group, suffixes):
    found = {}
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            for suffix in suffixes:
                if name.endswith(suffix):
                    found.setdefault(suffix, name)
    group.visititems(visitor)
    return found

def read_statistic(path, statistic=CFG.statistic):
    with h5py.File(path, "r") as h5:
        root = h5
        if statistic in root and isinstance(root[statistic], h5py.Group):
            group = root[statistic]
            mean = np.asarray(group["mean"], float)
            cov = np.asarray(group["cov"], float) if "cov" in group else None
            radius = np.asarray(root["rbins"], float) if "rbins" in root else np.arange(mean.size)
            if radius.size == mean.size + 1:
                radius = np.sqrt(radius[:-1] * radius[1:])
            return radius, mean, cov
        names = find_dataset(root, (statistic, statistic + "_mean", statistic + "_cov", "rbins", "r"))
        mean_key = names.get(statistic + "_mean", names.get(statistic))
        if mean_key is None:
            raise KeyError(f"{statistic} not found in {path}")
        mean = np.asarray(root[mean_key], float)
        cov_key = names.get(statistic + "_cov")
        cov = None if cov_key is None else np.asarray(root[cov_key], float)
        radius_key = names.get("rbins", names.get("r"))
        radius = np.arange(mean.size) if radius_key is None else np.asarray(root[radius_key], float)
        if radius.size == mean.size + 1:
            radius = np.sqrt(radius[:-1] * radius[1:])
    return radius, mean, cov

FILES = discover_results()
FILES
"""
            ),
            code(
                r"""
import matplotlib.pyplot as plt
from ia_analysis.visualization.plot_styles import model_color, model_label, set_project_style
from ia_analysis.visualization.profile_plots import draw_series_with_band

set_project_style(rc={"figure.figsize": CFG.figsize, "savefig.dpi": CFG.dpi})

def plot_snapshot(snap, cfg=CFG):
    fig, ax = plt.subplots(figsize=cfg.figsize)
    for row in FILES[FILES.snap == int(snap)].itertuples():
        radius, mean, cov = read_statistic(row.path, cfg.statistic)
        sigma = None if cov is None else np.sqrt(np.clip(np.diag(cov), 0.0, None))
        draw_series_with_band(ax, radius, mean, sigma, color=model_color(row.flag), label=model_label(row.flag), marker="o")
    ax.set_xscale("log")
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set(xlabel=r"$r\,[h^{-1}\mathrm{Mpc}]$", ylabel=cfg.statistic, title=f"{cfg.mass_selection}, snap={snap}")
    ax.legend()
    fig.tight_layout()
    return fig, ax

print(f"Found {len(FILES)} correlation result files.")
CORRELATION_FIGURE = plot_snapshot(max(FILES.snap)) if len(FILES) else None
"""
            ),
        ],
    )


def orbit() -> None:
    write(
        "05_orbit_stripping",
        "orbit_stripping_pipeline.ipynb",
        [
            md(
                r"""
# Orbit integration and tidal-stripping pipeline

Source workflows: `orbit.ipynb` and `merger_stripping.ipynb`
(June 16, 2026).

The pipeline includes NFW orbits, orbit templates, tidal stripping,
$V_\max/r_\max$ tidal tracks, and shell/profile visualization. It runs a
lightweight synthetic template by default so the notebook can be validated
end to end. Full NFW integration can be enabled in the configuration.
"""
            ),
            code(COMMON),
            code(
                r"""
from dataclasses import dataclass

@dataclass
class OrbitConfig:
    n_samples: int = 240
    initial_bound_mass: float = 1.0e10
    density_slope: float = 2.0
    tau_orbits: float = 0.75
    minimum_bound_fraction: float = 1.0e-4
    output_dir: Path = REPO_ROOT.parent / "pipeline_outputs" / "orbit_stripping"
    figsize: tuple[float, float] = (9.0, 6.5)
    dpi: int = 180
    cmap: str = "viridis"
    save: bool = False
    show: bool = False
    run_full_nfw: bool = False

CFG = OrbitConfig()
CFG
"""
            ),
            code(
                r"""
import numpy as np
from ia_analysis.orbits import TreeTrack, build_orbit_template, TidalStrippingOptions, stripping_history_from_template
from ia_analysis.orbits.tidal_stripping import template_host_curvature_powerlaw, stripping_summary
from ia_analysis.visualization.plot_styles import set_project_style

set_project_style(rc={"figure.figsize": CFG.figsize, "savefig.dpi": CFG.dpi})

def synthetic_template(cfg=CFG):
    time = np.linspace(0.0, 6.0, cfg.n_samples)
    phase = 3.4 * np.pi * time / time[-1]
    radius = 1.08 - 0.42 * np.cos(phase) + 0.08 * np.cos(2.0 * phase)
    position = np.column_stack([radius * np.cos(phase), radius * np.sin(phase), 0.15 * np.sin(0.5 * phase)])
    velocity = np.gradient(position, time, axis=0)
    host = TreeTrack(object_id="host", snapshots=time, positions=np.zeros_like(position), velocities=np.zeros_like(velocity))
    subhalo = TreeTrack(object_id="subhalo", snapshots=time, positions=position, velocities=velocity)
    return build_orbit_template(host, subhalo)

def stripping_from_template(template, cfg=CFG):
    radius = np.linalg.norm(template.relative_position, axis=1)
    curvature = template_host_curvature_powerlaw(radius, amplitude=0.95, scale_radius=0.32, exponent=3.0)
    options = TidalStrippingOptions(
        mode="delayed_powerlaw",
        density_slope=cfg.density_slope,
        tau_orbits=cfg.tau_orbits,
        minimum_bound_fraction=cfg.minimum_bound_fraction,
    )
    history = stripping_history_from_template(
        template,
        time=template.snapshots,
        mass0=cfg.initial_bound_mass,
        host_curvature=curvature,
        options=options,
    )
    return history

TEMPLATE = synthetic_template()
HISTORY = stripping_from_template(TEMPLATE)
stripping_summary(HISTORY)
"""
            ),
            code(
                r"""
import matplotlib.pyplot as plt
from ia_analysis.visualization.spectrum_plots import draw_ratio_series

def plot_orbit_and_stripping(template=TEMPLATE, history=HISTORY, cfg=CFG):
    fig, axes = plt.subplots(2, 2, figsize=cfg.figsize)
    pos = template.relative_position
    radius = np.linalg.norm(pos, axis=1)
    axes[0, 0].scatter(pos[:, 0], pos[:, 1], c=template.snapshots, s=8, cmap=cfg.cmap)
    axes[0, 0].set(xlabel="x", ylabel="y", title="Orbit template", aspect="equal")
    axes[0, 1].plot(template.snapshots, radius)
    axes[0, 1].set(xlabel="time", ylabel="radius", title="Radial motion")
    axes[1, 0].plot(history.time, history.mass_fraction, label="bound mass")
    axes[1, 0].plot(history.time, history.target_mass / cfg.initial_bound_mass, "--", label="tidal target")
    axes[1, 0].set(xlabel="time", ylabel=r"$M/M_0$", title="Tidal stripping")
    axes[1, 0].legend()
    axes[1, 1].plot(history.mass_fraction, history.vmax_ratio, label=r"$V_{\max}$")
    axes[1, 1].plot(history.mass_fraction, history.rmax_ratio, label=r"$r_{\max}$")
    axes[1, 1].set(xlabel=r"$M/M_0$", ylabel="normalized track", title="Tidal tracks")
    axes[1, 1].legend()
    fig.tight_layout()
    return fig, axes

FIG, AXES = plot_orbit_and_stripping()
if CFG.save:
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    FIG.savefig(CFG.output_dir / "orbit_stripping_summary.png", dpi=CFG.dpi)
if not CFG.show:
    plt.close(FIG)
print("Synthetic orbit/stripping workflow completed.")
"""
            ),
            code(
                r"""
from ia_analysis.orbits import NFWHost, OrbitSimulator

def run_full_nfw_orbit(host_kwargs, simulator_kwargs, initial_state, cfg=CFG):
    # Optional full pyccl-backed NFW orbit integration.
    host = NFWHost(**host_kwargs)
    simulator = OrbitSimulator(host=host, **simulator_kwargs)
    return simulator.integrate(initial_state)

print("pyccl-backed NFW classes imported. Set CFG.run_full_nfw=True and provide physical parameters when needed.")
"""
            ),
        ],
    )


def main() -> None:
    global_catalog()
    alignment()
    hod()
    spectra()
    merger()
    orbit()
    correlations()
    print(f"Wrote pipeline notebooks under {OUT}")


if __name__ == "__main__":
    main()
