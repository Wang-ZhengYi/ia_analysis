"""Exported code from notebooks/raw_20260618/HOD_data.ipynb.

This file is generated for project management and refactoring. Review before running end to end.
"""


# %% [markdown] cell 1
# # HOD and radial-profile curve-data comparison from `HOD_data` This notebook reads saved HDF5 curve data in `./HOD_data/` and produces publication-style comparison figures. For HOD curves, for each galaxy sample (`LRG`, `ELG`) and radial selection (`FoF`, `R200c`), it makes: 1. **Same-redshift model comparison**: 9 snapshots in a 3×3 grid; gravity models are colors and `Ncen/Nsat/Ntot` are line styles. 2. **Same-model redshift evolution**: six gravity-model panels in a 3×2 layout; redshift is sh

# %% code cell 2

# ============================================================
# Imports and user configuration
# ============================================================

from pathlib import Path
import re
import json
import warnings

import h5py
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ------------------------------------------------------------
# Input / output directories
# ------------------------------------------------------------
# The notebook is intended to sit in the same directory as HOD_data/.
HOD_DATA_DIR = Path("./HOD_data")
OUTPUT_DIR = Path("./figures_HOD_data_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Which dataset and curves to plot
# ------------------------------------------------------------
DATASET_LABEL = "ClusterSims"
SAMPLES = ["LRG", "ELG"]
REGIONS = ["fof", "r200c"]
COMPONENTS = ["Ncen", "Nsat", "Ntot"]

# 9 snapshots: 3x3 grid, matched to the P(k) plotting notebook.
SNAPS = [1, 3, 6, 8, 10, 12, 15, 18, 21]

# Gravity models and colors, matched to the P(k) plotting notebook.
FLAGS = ["GR", "F40", "F45", "F50", "F55", "F60"]
FR_COLORS = {
    "F4":   "#ff0000",
    "F4.5": "#ff8c00",
    "F5":   "#008000",
    "F5.5": "#ff00ff",
    "F6":   "#0000ff",
    "F6.5": "#9c9c9c",
    "GR":   "#000000",
}
FLAG_LABEL = {
    "GR": "GR",
    "F40": "F4",
    "F45": "F4.5",
    "F50": "F5",
    "F55": "F5.5",
    "F60": "F6",
    "F65": "F6.5",
}
FLAG_COLOR = {flag: FR_COLORS.get(FLAG_LABEL.get(flag, flag), "0.4") for flag in FLAGS + ["F65"]}

# Snapshot-redshift mapping, same convention as the P(k) plotting notebook.
ZMAP = {
    0: 3.00, 1: 2.00, 2: 1.36, 3: 1.26, 4: 1.15, 5: 1.06,
    6: 0.97, 7: 0.88, 8: 0.80, 9: 0.73, 10: 0.65, 11: 0.58,
    12: 0.51, 13: 0.45, 14: 0.39, 15: 0.33, 16: 0.27, 17: 0.21,
    18: 0.16, 19: 0.10, 20: 0.05, 21: 0.00,
}

# Plot style.
SAVE_FIGURES = True
SHOW_FIGURES = True
SAVE_FORMATS = ("png", "pdf")
DPI = 300

# HOD axis settings.
# Set to None for automatic limits.
XLIM_M200C = None      # e.g. (1e10, 3e15)
YLIM_HOD = None        # e.g. (1e-3, 3e2)
HOD_YLOG = True
PREFER_SMOOTH = True   # Use /smooth curves if present; otherwise fall back to /binned.

# Redshift-evolution colormap, matched to the P(k) plotting notebook.
REDSHIFT_CMAP = "turbo"

# Component line styles: same colour for one model/redshift, different line style for Ncen/Nsat/Ntot.
COMPONENT_STYLE = {
    "Ncen": dict(ls="--", lw=1.65, alpha=0.95),
    "Nsat": dict(ls=":",  lw=1.95, alpha=0.95),
    "Ntot": dict(ls="-",  lw=2.05, alpha=0.98),
}
COMPONENT_LABEL = {
    "Ncen": r"$\langle N_{\rm cen}\rangle$",
    "Nsat": r"$\langle N_{\rm sat}\rangle$",
    "Ntot": r"$\langle N_{\rm tot}\rangle$",
}
REGION_LABEL = {
    "fof": "FoF members",
    "r200c": r"within $R_{200c}$",
}
SAMPLE_LABEL = {
    "LRG": "LRG",
    "ELG": "ELG",
}


# ------------------------------------------------------------
# Radial-profile settings
# ------------------------------------------------------------
# The saved radial-profile groups have no FoF/R200c split. They describe FoF
# member profiles as a function of r/R200c, usually separated by host-mass bins.
PROFILE_YKEY = "mean_shell_count"      # "mean_shell_count", "number_density", "cumulative", or "counts"
PROFILE_PREFER_SMOOTH = True
PROFILE_XLIM = (1.0e-2, 5.0)
PROFILE_YLIM = None
PROFILE_XLOG = True
PROFILE_YLOG = True

# %% code cell 3

# ============================================================
# Publication-style Matplotlib settings
# ============================================================

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": DPI,
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.05,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "mathtext.fontset": "cm",
    "mathtext.default": "regular",
    "font.family": "serif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def flag_display(flag):
    return FLAG_LABEL.get(str(flag), str(flag))


def flag_color(flag):
    return FLAG_COLOR.get(str(flag), "0.4")


def format_z_title(z):
    if not np.isfinite(z):
        return r"$\mathrm{z}$ unknown"
    if abs(float(z)) < 5e-3:
        return r"$\mathrm{z}=0$"
    return rf"$\mathrm{{z}}={float(z):.2f}$"


def output_safe_name(x):
    x = str(x)
    x = x.replace("/", "_").replace(" ", "_").replace("$", "")
    x = x.replace("\\", "").replace("{", "").replace("}", "")
    x = re.sub(r"[^A-Za-z0-9_.+-]+", "_", x)
    return re.sub(r"_+", "_", x).strip("_")


def save_figure(fig, stem, subdir=None):
    """Save one figure in all requested formats."""
    if not SAVE_FIGURES:
        return
    outdir = OUTPUT_DIR if subdir is None else OUTPUT_DIR / subdir
    outdir.mkdir(exist_ok=True, parents=True)
    for ext in SAVE_FORMATS:
        path = outdir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
    print(f"[saved] {outdir / (stem + '.' + SAVE_FORMATS[0])}")


def maybe_close(fig):
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def add_axis_colorbar(fig, ax, cmap, norm):
    """Place a colorbar immediately to the right of one subplot with identical height."""
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.08)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r"$\mathrm{z}$", labelpad=4)
    cb.ax.tick_params(labelsize=8)
    return cb

# %% code cell 4

# ============================================================
# HDF5 discovery and robust curve readers
# ============================================================

def _decode_attr(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return x


def _safe_group_name(x):
    """The HOD writer used this naming convention for HDF5 groups."""
    s = str(x)
    s = s.replace("/", "_").replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_.+\-]+", "_", s)
    return s.strip("_") or "unknown"


def canonical_flag(label):
    """Convert display labels such as F4.5 back to internal labels such as F45."""
    s = str(label)
    aliases = {
        "GR": "GR",
        "F4": "F40",
        "F4.0": "F40",
        "F4.5": "F45",
        "F5": "F50",
        "F5.0": "F50",
        "F5.5": "F55",
        "F6": "F60",
        "F6.0": "F60",
        "F6.5": "F65",
    }
    return aliases.get(s, s)


def _parse_file_metadata_from_name(path):
    """Fallback parser for files like ClusterSims_GR_snap021.hdf5."""
    stem = Path(path).stem
    m = re.match(r"(?P<dataset>.+)_(?P<label>[^_]+)_snap(?P<snap>\d+)$", stem)
    if m is None:
        return dict(dataset="unknown", label=stem, snap=np.nan, z=np.nan)
    return dict(
        dataset=m.group("dataset"),
        label=m.group("label"),
        snap=int(m.group("snap")),
        z=ZMAP.get(int(m.group("snap")), np.nan),
    )


def read_file_metadata(path):
    """Read /meta attrs from one HDF5 file, with filename fallback."""
    fallback = _parse_file_metadata_from_name(path)
    meta = dict(fallback)
    try:
        with h5py.File(path, "r") as f:
            if "meta" in f:
                attrs = f["meta"].attrs
                for key in ["dataset", "label", "snap", "z"]:
                    if key in attrs:
                        meta[key] = _decode_attr(attrs[key])
                if "snap" in meta and meta["snap"] is not None:
                    try:
                        meta["snap"] = int(meta["snap"])
                    except Exception:
                        pass
                if "z" in meta and meta["z"] is not None:
                    try:
                        meta["z"] = float(meta["z"])
                    except Exception:
                        pass
    except OSError:
        pass
    meta["path"] = Path(path)
    meta["flag"] = canonical_flag(meta.get("label", ""))
    return meta


def discover_hod_files(data_dir=HOD_DATA_DIR, dataset_label=DATASET_LABEL):
    """Scan HOD_data/ and build a compact manifest table."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.hdf5")) + sorted(data_dir.glob("*.h5"))
    rows = []
    for path in files:
        meta = read_file_metadata(path)
        if dataset_label is not None and str(meta.get("dataset")) != str(dataset_label):
            continue
        try:
            with h5py.File(path, "r") as f:
                if ("hod" not in f) and ("radial_profiles" not in f):
                    continue
        except OSError:
            continue
        rows.append(meta)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return pd.DataFrame(columns=["dataset", "label", "flag", "snap", "z", "path"])

    df["snap"] = df["snap"].astype(int)
    df["z"] = pd.to_numeric(df["z"], errors="coerce")
    flag_order = {f: i for i, f in enumerate(FLAGS)}
    df["flag_order"] = df["flag"].map(lambda x: flag_order.get(x, 999))
    df = df.sort_values(["snap", "flag_order", "flag"]).reset_index(drop=True)
    return df


files_df = discover_hod_files()
print(f"Discovered {len(files_df)} HDF5 curve-data files in {HOD_DATA_DIR.resolve()}")
files_df.head(20)

# %% code cell 5

# ============================================================
# Inspect available samples / regions / components
# ============================================================

def _find_key(group, desired, *, contains_ok=True):
    """Find a key in an HDF5 group using exact, case-insensitive, and contains matching."""
    if group is None:
        return None
    desired = str(desired)
    keys = list(group.keys())
    if desired in keys:
        return desired
    safe = _safe_group_name(desired)
    if safe in keys:
        return safe

    low = desired.lower()
    safe_low = safe.lower()
    for key in keys:
        if key.lower() == low or key.lower() == safe_low:
            return key

    if contains_ok:
        for key in keys:
            kl = key.lower()
            if low in kl or safe_low in kl:
                return key

    return None


def _find_region_key(sample_grp, region):
    """Find region group, accepting fof/FoF and r200c/R200c variants."""
    region = str(region).lower()
    if region in ["fof", "fof_members", "fofmember"]:
        candidates = ["fof", "FoF", "FoF members", "fof_members"]
        contains = "fof"
    elif region in ["r200c", "r200", "within_r200c"]:
        candidates = ["r200c", "R200c", "within_R200c", "within R200c"]
        contains = "r200"
    else:
        candidates = [region]
        contains = region

    for cand in candidates:
        key = _find_key(sample_grp, cand, contains_ok=False)
        if key is not None:
            return key
    for key in sample_grp.keys():
        if contains in key.lower():
            return key
    return None


def list_hod_structure(path):
    """Return nested structure: sample -> region -> components."""
    out = {}
    with h5py.File(path, "r") as f:
        if "hod" not in f:
            return out
        hod = f["hod"]
        for sample_key in hod.keys():
            if sample_key == "summary_json":
                continue
            sample_grp = hod[sample_key]
            if not isinstance(sample_grp, h5py.Group):
                continue
            out[sample_key] = {}
            for region_key in sample_grp.keys():
                region_grp = sample_grp[region_key]
                if not isinstance(region_grp, h5py.Group):
                    continue
                comps = [k for k in region_grp.keys() if isinstance(region_grp[k], h5py.Group)]
                out[sample_key][region_key] = comps
    return out


if len(files_df) > 0:
    example_path = files_df.iloc[0]["path"]
    print("Example file:", example_path)
    print(json.dumps(list_hod_structure(example_path), indent=2))
else:
    print("No HOD HDF5 files found. Check HOD_DATA_DIR.")

# %% code cell 6

# ============================================================
# HOD curve loading
# ============================================================

def get_hod_path(flag, snap, files_df=files_df, dataset_label=DATASET_LABEL):
    """Return HDF5 path for one gravity model and snapshot, or None if missing."""
    if files_df is None or len(files_df) == 0:
        return None
    flag = canonical_flag(flag)
    m = (files_df["flag"].astype(str) == str(flag)) & (files_df["snap"].astype(int) == int(snap))
    if dataset_label is not None and "dataset" in files_df:
        m &= (files_df["dataset"].astype(str) == str(dataset_label))
    rows = files_df.loc[m]
    if len(rows) == 0:
        return None
    return Path(rows.iloc[0]["path"])


def read_hod_curve(path, sample, region, component, prefer_smooth=PREFER_SMOOTH):
    """Read one HOD curve from a saved HDF5 file."""
    path = Path(path)
    if not path.exists():
        return None

    with h5py.File(path, "r") as f:
        if "hod" not in f:
            return None
        hod = f["hod"]
        sample_key = _find_key(hod, sample)
        if sample_key is None:
            return None
        sample_grp = hod[sample_key]

        region_key = _find_region_key(sample_grp, region)
        if region_key is None:
            return None
        region_grp = sample_grp[region_key]

        comp_key = _find_key(region_grp, component, contains_ok=False)
        if comp_key is None:
            return None
        comp_grp = region_grp[comp_key]

        curve_grp = None
        source = None
        if prefer_smooth and "smooth" in comp_grp and "x" in comp_grp["smooth"] and "y" in comp_grp["smooth"]:
            curve_grp = comp_grp["smooth"]
            source = "smooth"
        elif "binned" in comp_grp and "x" in comp_grp["binned"] and "y" in comp_grp["binned"]:
            curve_grp = comp_grp["binned"]
            source = "binned"
        elif "smooth" in comp_grp and "x" in comp_grp["smooth"] and "y" in comp_grp["smooth"]:
            curve_grp = comp_grp["smooth"]
            source = "smooth"

        if curve_grp is None:
            return None

        x = np.asarray(curve_grp["x"][()], dtype=float)
        y = np.asarray(curve_grp["y"][()], dtype=float)
        good = np.isfinite(x) & np.isfinite(y)
        if HOD_YLOG:
            good &= y > 0
        if np.count_nonzero(good) == 0:
            return None
        return dict(x=x[good], y=y[good], source=source, path=path)

# %% code cell 7

# ============================================================
# Axis and legend helpers
# ============================================================

def apply_hod_axis_style(ax, *, show_xlabel=True, show_ylabel=True):
    ax.set_xscale("log")
    if HOD_YLOG:
        ax.set_yscale("log")
    if show_xlabel:
        ax.set_xlabel(r"$M_{200c}\,[M_\odot/h]$")
    if show_ylabel:
        ax.set_ylabel(r"$\langle N\,|\,M_{200c}\rangle$")
    if XLIM_M200C is not None:
        ax.set_xlim(*XLIM_M200C)
    if YLIM_HOD is not None:
        ax.set_ylim(*YLIM_HOD)
    ax.grid(True, which="major", alpha=0.22, lw=0.6)
    ax.grid(True, which="minor", alpha=0.08, lw=0.4)


def model_legend_handles(flags=FLAGS):
    return [Line2D([0], [0], color=flag_color(flag), lw=2.2, label=flag_display(flag)) for flag in flags]


def component_legend_handles(color="0.15"):
    return [
        Line2D([0], [0], color=color, lw=COMPONENT_STYLE[comp]["lw"], ls=COMPONENT_STYLE[comp]["ls"], label=COMPONENT_LABEL.get(comp, comp))
        for comp in COMPONENTS
    ]


def add_two_row_legend(fig, flags=FLAGS, y_model=1.030, y_comp=0.990):
    """Two-row legend: first row models, second row Ncen/Nsat/Ntot line styles."""
    leg1 = fig.legend(handles=model_legend_handles(flags), loc="upper center", ncol=min(len(flags), 6), frameon=False,
                      bbox_to_anchor=(0.5, y_model), columnspacing=1.5, handlelength=2.4)
    fig.add_artist(leg1)
    leg2 = fig.legend(handles=component_legend_handles(), loc="upper center", ncol=3, frameon=False,
                      bbox_to_anchor=(0.5, y_comp), columnspacing=1.8, handlelength=3.0)
    fig.add_artist(leg2)
    return leg1, leg2


def add_component_legend(fig, y=1.015):
    """Legend for Ncen/Nsat/Ntot only, used in redshift-evolution figures."""
    leg = fig.legend(handles=component_legend_handles(), loc="upper center", ncol=3, frameon=False,
                     bbox_to_anchor=(0.5, y), columnspacing=1.8, handlelength=3.0)
    fig.add_artist(leg)
    return leg


def plot_hod_curve(ax, x, y, *, color, component, label=None, zorder=3):
    style = COMPONENT_STYLE.get(component, {})
    ax.plot(x, y, color=color, ls=style.get("ls", "-"), lw=style.get("lw", 1.8),
            alpha=style.get("alpha", 1.0), label=label, zorder=zorder)

# %% code cell 8

# ============================================================
# Figure 1: same-redshift model comparison, 3x3 snapshot grid
# ============================================================

def plot_model_comparison_grid(sample, region, *, flags=FLAGS, snaps=SNAPS, save=True, show=True):
    """
    Same-redshift comparison across gravity models.

    Layout: 3x3 snapshots. In each panel:
        - model = colour;
        - Ncen/Nsat/Ntot = line style.
    """
    ncols = 3
    nrows = int(np.ceil(len(snaps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.4, 12.2), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, snap in zip(axes, snaps):
        plotted = False
        z = ZMAP.get(int(snap), np.nan)
        for flag in flags:
            path = get_hod_path(flag, snap)
            if path is None:
                continue
            for component in COMPONENTS:
                curve = read_hod_curve(path, sample, region, component, prefer_smooth=PREFER_SMOOTH)
                if curve is None:
                    continue
                plot_hod_curve(ax, curve["x"], curve["y"], color=flag_color(flag), component=component)
                plotted = True
        apply_hod_axis_style(ax)
        ax.set_title(format_z_title(z), pad=6)
        if not plotted:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", color="0.45")

    for ax in axes[len(snaps):]:
        ax.axis("off")

    sample_label = SAMPLE_LABEL.get(sample, sample)
    region_label = REGION_LABEL.get(region, region)
    fig.suptitle(rf"{sample_label}: HOD model comparison ({region_label})", y=1.060, fontsize=16)
    add_two_row_legend(fig, flags=flags, y_model=1.030, y_comp=0.992)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.930))

    if save:
        stem = f"{output_safe_name(sample)}_{output_safe_name(region)}_HOD_model_grid"
        save_figure(fig, stem, subdir="model_comparison")
    if show:
        maybe_close(fig)
    else:
        plt.close(fig)
    return fig

# %% code cell 9

# ============================================================
# Figure 2: same-model redshift evolution, 3 columns x 2 rows
# ============================================================

def plot_redshift_evolution_by_model(sample, region, *, flags=FLAGS, snaps=SNAPS, save=True, show=True):
    """
    Same-model redshift evolution.

    Layout: 3 columns x 2 rows for six gravity models. In each panel:
        - redshift = colour, with a P(k)-style colourbar beside the panel;
        - Ncen/Nsat/Ntot = line style.
    """
    ncols = 3
    nrows = int(np.ceil(len(flags) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(17.2, 9.2), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    zvals = np.array([ZMAP.get(int(s), np.nan) for s in snaps], dtype=float)
    finite_z = np.isfinite(zvals)
    zmin, zmax = np.nanmin(zvals[finite_z]), np.nanmax(zvals[finite_z])
    cmap = plt.get_cmap(REDSHIFT_CMAP)
    norm = plt.Normalize(vmin=zmin, vmax=zmax)

    for ax, flag in zip(axes, flags):
        plotted = False
        for snap in snaps:
            path = get_hod_path(flag, snap)
            if path is None:
                continue
            z = ZMAP.get(int(snap), np.nan)
            if not np.isfinite(z):
                continue
            color = cmap(norm(z))
            for component in COMPONENTS:
                curve = read_hod_curve(path, sample, region, component, prefer_smooth=PREFER_SMOOTH)
                if curve is None:
                    continue
                plot_hod_curve(ax, curve["x"], curve["y"], color=color, component=component)
                plotted = True
        apply_hod_axis_style(ax)
        ax.set_title(flag_display(flag), color=flag_color(flag), pad=6, fontweight="bold")
        if plotted:
            add_axis_colorbar(fig, ax, cmap, norm)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", color="0.45")

    for ax in axes[len(flags):]:
        ax.axis("off")

    sample_label = SAMPLE_LABEL.get(sample, sample)
    region_label = REGION_LABEL.get(region, region)
    fig.suptitle(rf"{sample_label}: HOD redshift evolution ({region_label})", y=1.035, fontsize=16)
    add_component_legend(fig, y=0.995)
    fig.subplots_adjust(left=0.075, right=0.965, bottom=0.085, top=0.890, wspace=0.42, hspace=0.35)

    if save:
        stem = f"{output_safe_name(sample)}_{output_safe_name(region)}_HOD_redshift_evolution"
        save_figure(fig, stem, subdir="redshift_evolution")
    if show:
        maybe_close(fig)
    else:
        plt.close(fig)
    return fig

# %% code cell 10
# ============================================================
# Radial-profile curve loading and plotting
# ============================================================

PROFILE_YKEY = globals().get("PROFILE_YKEY", "mean_shell_count")
PROFILE_PREFER_SMOOTH = globals().get("PROFILE_PREFER_SMOOTH", True)
PROFILE_XLIM = globals().get("PROFILE_XLIM", (1.0e-2, 5.0))
PROFILE_YLIM = globals().get("PROFILE_YLIM", None)
PROFILE_XLOG = globals().get("PROFILE_XLOG", True)
PROFILE_YLOG = globals().get("PROFILE_YLOG", True)

PROFILE_YLABEL = {
    "mean_shell_count": r"$\langle N_{\rm shell}\rangle$",
    "number_density": r"$\langle N_{\rm shell}\rangle / \Delta V$",
    "cumulative": r"$\langle N(<r)\rangle$",
    "counts": r"$N_{\rm shell}$",
}

MASS_BIN_STYLE_CYCLE = [
    dict(ls="-",  lw=2.05, alpha=0.98),
    dict(ls="--", lw=1.85, alpha=0.95),
    dict(ls=":",  lw=2.20, alpha=0.95),
    dict(ls="-.", lw=1.85, alpha=0.95),
    dict(ls=(0, (5, 1, 1, 1)), lw=1.75, alpha=0.95),
]


def _profile_label_from_attrs(mb_grp, fallback_key):
    """Return a readable host-mass-bin label for a radial-profile group."""
    label = mb_grp.attrs.get("mass_bin_label", None)
    label = _decode_attr(label)
    if label not in [None, ""]:
        return str(label)

    lo = mb_grp.attrs.get("mass_bin_lo", np.nan)
    hi = mb_grp.attrs.get("mass_bin_hi", np.nan)
    try:
        lo = float(lo)
        hi = float(hi)
        if np.isfinite(lo) and np.isfinite(hi) and lo > 0 and hi > 0:
            return rf"$10^{{{np.log10(lo):.0f}}}$--$10^{{{np.log10(hi):.0f}}}$"
    except Exception:
        pass
    return str(fallback_key)


def read_profile_curves(path, sample, *, ykey=PROFILE_YKEY, prefer_smooth=PROFILE_PREFER_SMOOTH):
    """
    Read all radial-profile host-mass-bin curves for one sample from one HDF5 file.

    Expected HDF5 structure:
        /radial_profiles/<sample>/<mass_bin>/binned/{x,y or ykey}
        /radial_profiles/<sample>/<mass_bin>/smooth/{x,y}
    """
    path = Path(path)
    if not path.exists():
        return {}

    curves = {}
    with h5py.File(path, "r") as f:
        if "radial_profiles" not in f:
            return {}
        rad = f["radial_profiles"]
        sample_key = _find_key(rad, sample)
        if sample_key is None:
            return {}
        sample_grp = rad[sample_key]

        for mb_key in sample_grp.keys():
            mb_grp = sample_grp[mb_key]
            if not isinstance(mb_grp, h5py.Group):
                continue

            source = None
            curve_grp = None
            if prefer_smooth and "smooth" in mb_grp and "x" in mb_grp["smooth"] and "y" in mb_grp["smooth"]:
                curve_grp = mb_grp["smooth"]
                source = "smooth"
            elif "binned" in mb_grp and "x" in mb_grp["binned"]:
                bgrp = mb_grp["binned"]
                if "y" in bgrp or ykey in bgrp:
                    curve_grp = bgrp
                    source = "binned"
            elif "smooth" in mb_grp and "x" in mb_grp["smooth"] and "y" in mb_grp["smooth"]:
                curve_grp = mb_grp["smooth"]
                source = "smooth"

            if curve_grp is None:
                continue

            x = np.asarray(curve_grp["x"][()], dtype=float)
            if "y" in curve_grp:
                y = np.asarray(curve_grp["y"][()], dtype=float)
            elif ykey in curve_grp:
                y = np.asarray(curve_grp[ykey][()], dtype=float)
            else:
                continue

            good = np.isfinite(x) & np.isfinite(y)
            if PROFILE_XLOG:
                good &= x > 0
            if PROFILE_YLOG:
                good &= y > 0
            if np.count_nonzero(good) == 0:
                continue

            label = _profile_label_from_attrs(mb_grp, mb_key)
            try:
                lo = float(mb_grp.attrs.get("mass_bin_lo", np.nan))
                hi = float(mb_grp.attrs.get("mass_bin_hi", np.nan))
            except Exception:
                lo, hi = np.nan, np.nan
            sort_key = lo if np.isfinite(lo) else len(curves)
            curves[str(mb_key)] = dict(
                x=x[good],
                y=y[good],
                label=label,
                source=source,
                path=path,
                mass_bin_key=str(mb_key),
                mass_bin_lo=lo,
                mass_bin_hi=hi,
                sort_key=sort_key,
            )
    return dict(sorted(curves.items(), key=lambda kv: kv[1].get("sort_key", 999)))


def profile_mass_bin_order(sample=SAMPLES[0], flags=FLAGS, snaps=SNAPS):
    """Infer a stable host-mass-bin order from the first available radial-profile HDF5 file."""
    for snap in snaps:
        for flag in flags:
            path = get_hod_path(flag, snap)
            if path is None:
                continue
            curves = read_profile_curves(path, sample)
            if curves:
                return list(curves.keys()), {k: v.get("label", k) for k, v in curves.items()}
    return [], {}


def mass_bin_style_map(sample=SAMPLES[0], flags=FLAGS, snaps=SNAPS):
    keys, labels = profile_mass_bin_order(sample=sample, flags=flags, snaps=snaps)
    style = {}
    for i, key in enumerate(keys):
        st = MASS_BIN_STYLE_CYCLE[i % len(MASS_BIN_STYLE_CYCLE)].copy()
        st["label"] = labels.get(key, key)
        style[key] = st
    return style


def apply_profile_axis_style(ax, *, show_xlabel=True, show_ylabel=True):
    if PROFILE_XLOG:
        ax.set_xscale("log")
    if PROFILE_YLOG:
        ax.set_yscale("log")
    if show_xlabel:
        ax.set_xlabel(r"$r/R_{200c}$")
    if show_ylabel:
        ax.set_ylabel(PROFILE_YLABEL.get(PROFILE_YKEY, PROFILE_YKEY))
    if PROFILE_XLIM is not None:
        ax.set_xlim(*PROFILE_XLIM)
    if PROFILE_YLIM is not None:
        ax.set_ylim(*PROFILE_YLIM)
    ax.grid(True, which="major", alpha=0.22, lw=0.6)
    ax.grid(True, which="minor", alpha=0.08, lw=0.4)


def profile_mass_bin_legend_handles(sample=SAMPLES[0], flags=FLAGS, snaps=SNAPS, color="0.15"):
    style = mass_bin_style_map(sample=sample, flags=flags, snaps=snaps)
    handles = []
    for key, st in style.items():
        handles.append(Line2D(
            [0], [0],
            color=color,
            lw=st.get("lw", 1.8),
            ls=st.get("ls", "-"),
            label=st.get("label", key),
        ))
    if len(handles) == 0:
        handles.append(Line2D([0], [0], color=color, lw=2, ls="-", label="host-mass bin"))
    return handles


def add_profile_two_row_legend(fig, sample, flags=FLAGS, y_model=1.030, y_mass=0.990):
    """Two-row legend for radial profiles: first row models, second row host-mass bins."""
    leg1 = fig.legend(
        handles=model_legend_handles(flags),
        loc="upper center",
        ncol=min(len(flags), 6),
        frameon=False,
        bbox_to_anchor=(0.5, y_model),
        columnspacing=1.5,
        handlelength=2.4,
    )
    fig.add_artist(leg1)
    mass_handles = profile_mass_bin_legend_handles(sample=sample, flags=flags)
    leg2 = fig.legend(
        handles=mass_handles,
        loc="upper center",
        ncol=min(len(mass_handles), 4),
        frameon=False,
        bbox_to_anchor=(0.5, y_mass),
        columnspacing=1.5,
        handlelength=3.0,
        title=r"Host mass bin",
        title_fontsize=10,
    )
    fig.add_artist(leg2)
    return leg1, leg2


def add_profile_mass_legend(fig, sample, y=1.015):
    handles = profile_mass_bin_legend_handles(sample=sample)
    leg = fig.legend(
        handles=handles,
        loc="upper center",
        ncol=min(len(handles), 4),
        frameon=False,
        bbox_to_anchor=(0.5, y),
        columnspacing=1.5,
        handlelength=3.0,
        title=r"Host mass bin",
        title_fontsize=10,
    )
    fig.add_artist(leg)
    return leg


def plot_profile_curve(ax, x, y, *, color, mass_bin_key, style_map=None, zorder=3):
    style_map = {} if style_map is None else style_map
    st = style_map.get(mass_bin_key, MASS_BIN_STYLE_CYCLE[0])
    ax.plot(
        x,
        y,
        color=color,
        ls=st.get("ls", "-"),
        lw=st.get("lw", 1.8),
        alpha=st.get("alpha", 0.95),
        zorder=zorder,
    )


def plot_profile_model_comparison_grid(sample, *, flags=FLAGS, snaps=SNAPS, save=True, show=True):
    """
    Same-redshift radial-profile comparison across gravity models.

    Layout: 3x3 snapshots. In each panel:
        - model = colour;
        - host-mass bin = line style.
    """
    ncols = 3
    nrows = int(np.ceil(len(snaps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.4, 12.2), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)
    style_map = mass_bin_style_map(sample=sample, flags=flags, snaps=snaps)

    for ax, snap in zip(axes, snaps):
        plotted = False
        z = ZMAP.get(int(snap), np.nan)
        for flag in flags:
            path = get_hod_path(flag, snap)
            if path is None:
                continue
            curves = read_profile_curves(path, sample, ykey=PROFILE_YKEY, prefer_smooth=PROFILE_PREFER_SMOOTH)
            for mb_key, curve in curves.items():
                plot_profile_curve(ax, curve["x"], curve["y"], color=flag_color(flag), mass_bin_key=mb_key, style_map=style_map)
                plotted = True
        apply_profile_axis_style(ax)
        ax.set_title(format_z_title(z), pad=6)
        if not plotted:
            ax.text(0.5, 0.5, "No profile data", transform=ax.transAxes, ha="center", va="center", color="0.45")

    for ax in axes[len(snaps):]:
        ax.axis("off")

    sample_label = SAMPLE_LABEL.get(sample, sample)
    fig.suptitle(rf"{sample_label}: radial-profile model comparison", y=1.060, fontsize=16)
    add_profile_two_row_legend(fig, sample=sample, flags=flags, y_model=1.030, y_mass=0.992)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.930))

    if save:
        stem = f"{output_safe_name(sample)}_RadialProfile_model_grid_{output_safe_name(PROFILE_YKEY)}"
        save_figure(fig, stem, subdir="profile_model_comparison")
    if show:
        maybe_close(fig)
    else:
        plt.close(fig)
    return fig


def plot_profile_redshift_evolution_by_model(sample, *, flags=FLAGS, snaps=SNAPS, save=True, show=True):
    """
    Same-model radial-profile redshift evolution.

    Layout: 3 columns x 2 rows for six gravity models. In each panel:
        - redshift = colour, with a P(k)-style colourbar beside the panel;
        - host-mass bin = line style.
    """
    ncols = 3
    nrows = int(np.ceil(len(flags) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(17.2, 9.2), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)
    style_map = mass_bin_style_map(sample=sample, flags=flags, snaps=snaps)

    zvals = np.array([ZMAP.get(int(s), np.nan) for s in snaps], dtype=float)
    finite_z = np.isfinite(zvals)
    zmin, zmax = np.nanmin(zvals[finite_z]), np.nanmax(zvals[finite_z])
    cmap = plt.get_cmap(REDSHIFT_CMAP)
    norm = plt.Normalize(vmin=zmin, vmax=zmax)

    for ax, flag in zip(axes, flags):
        plotted = False
        for snap in snaps:
            path = get_hod_path(flag, snap)
            if path is None:
                continue
            z = ZMAP.get(int(snap), np.nan)
            if not np.isfinite(z):
                continue
            color = cmap(norm(z))
            curves = read_profile_curves(path, sample, ykey=PROFILE_YKEY, prefer_smooth=PROFILE_PREFER_SMOOTH)
            for mb_key, curve in curves.items():
                plot_profile_curve(ax, curve["x"], curve["y"], color=color, mass_bin_key=mb_key, style_map=style_map)
                plotted = True
        apply_profile_axis_style(ax)
        ax.set_title(flag_display(flag), color=flag_color(flag), pad=6, fontweight="bold")
        if plotted:
            add_axis_colorbar(fig, ax, cmap, norm)
        else:
            ax.text(0.5, 0.5, "No profile data", transform=ax.transAxes, ha="center", va="center", color="0.45")

    for ax in axes[len(flags):]:
        ax.axis("off")

    sample_label = SAMPLE_LABEL.get(sample, sample)
    fig.suptitle(rf"{sample_label}: radial-profile redshift evolution", y=1.035, fontsize=16)
    add_profile_mass_legend(fig, sample=sample, y=0.995)
    fig.subplots_adjust(left=0.075, right=0.965, bottom=0.085, top=0.890, wspace=0.42, hspace=0.35)

    if save:
        stem = f"{output_safe_name(sample)}_RadialProfile_redshift_evolution_{output_safe_name(PROFILE_YKEY)}"
        save_figure(fig, stem, subdir="profile_redshift_evolution")
    if show:
        maybe_close(fig)
    else:
        plt.close(fig)
    return fig


def plot_one_profile_set(sample, *, flags=FLAGS, snaps=SNAPS):
    """Make both radial-profile figures for one sample."""
    print(f"\n=== radial profile | {sample} ===")
    fig1 = plot_profile_model_comparison_grid(sample, flags=flags, snaps=snaps)
    fig2 = plot_profile_redshift_evolution_by_model(sample, flags=flags, snaps=snaps)
    return fig1, fig2


def plot_all_profile_comparisons(samples=SAMPLES, *, flags=FLAGS, snaps=SNAPS):
    """Plot all LRG/ELG radial-profile comparison figures."""
    manifest = []
    for sample in samples:
        try:
            plot_profile_model_comparison_grid(sample, flags=flags, snaps=snaps)
            manifest.append((sample, "radial_profile", "model_comparison"))
        except Exception as exc:
            print(f"[warn] profile model comparison failed for {sample}: {exc}")
        try:
            plot_profile_redshift_evolution_by_model(sample, flags=flags, snaps=snaps)
            manifest.append((sample, "radial_profile", "redshift_evolution"))
        except Exception as exc:
            print(f"[warn] profile redshift evolution failed for {sample}: {exc}")
    return pd.DataFrame(manifest, columns=["sample", "profile", "figure_type"])

# %% code cell 11

# ============================================================
# Batch plotting interface
# ============================================================

def plot_one_hod_set(sample, region, *, flags=FLAGS, snaps=SNAPS):
    """Make both required HOD figures for one sample and one region."""
    print(f"\n=== HOD | {sample} | {region} ===")
    fig1 = plot_model_comparison_grid(sample, region, flags=flags, snaps=snaps)
    fig2 = plot_redshift_evolution_by_model(sample, region, flags=flags, snaps=snaps)
    return fig1, fig2


def plot_all_hod_comparisons(samples=SAMPLES, regions=REGIONS, *, flags=FLAGS, snaps=SNAPS):
    """Plot all LRG/ELG and FoF/R200c HOD comparison figures."""
    manifest = []
    for sample in samples:
        for region in regions:
            try:
                plot_model_comparison_grid(sample, region, flags=flags, snaps=snaps)
                manifest.append((sample, region, "HOD", "model_comparison"))
            except Exception as exc:
                print(f"[warn] HOD model comparison failed for {sample}/{region}: {exc}")
            try:
                plot_redshift_evolution_by_model(sample, region, flags=flags, snaps=snaps)
                manifest.append((sample, region, "HOD", "redshift_evolution"))
            except Exception as exc:
                print(f"[warn] HOD redshift evolution failed for {sample}/{region}: {exc}")
    return pd.DataFrame(manifest, columns=["sample", "region", "quantity", "figure_type"])


def plot_all_comparisons(samples=SAMPLES, regions=REGIONS, *, flags=FLAGS, snaps=SNAPS, include_hod=True, include_profiles=True):
    """Plot all HOD and radial-profile comparison figures."""
    tables = []
    if include_hod:
        tables.append(plot_all_hod_comparisons(samples=samples, regions=regions, flags=flags, snaps=snaps))
    if include_profiles:
        tables.append(plot_all_profile_comparisons(samples=samples, flags=flags, snaps=snaps))
    if not tables:
        return pd.DataFrame(columns=["sample", "region", "quantity", "figure_type"])
    return pd.concat(tables, ignore_index=True)

# %% code cell 12

# ============================================================
# Run all plots
# ============================================================
# This cell will read HOD_data/*.hdf5 and save figures to OUTPUT_DIR.
# Change RUN_ALL_PLOTS to False if you only want to import the functions.

RUN_ALL_PLOTS = True
RUN_HOD_PLOTS = True
RUN_PROFILE_PLOTS = True

if RUN_ALL_PLOTS:
    if len(files_df) == 0:
        raise FileNotFoundError(
            f"No HDF5 curve-data files found in {HOD_DATA_DIR.resolve()}. "
            "Move this notebook next to the HOD_data/ folder or edit HOD_DATA_DIR."
        )
    manifest = plot_all_comparisons(include_hod=RUN_HOD_PLOTS, include_profiles=RUN_PROFILE_PLOTS)
    display(manifest)
else:
    print("RUN_ALL_PLOTS = False. Examples:")
    print("  plot_one_hod_set('LRG', 'fof')")
    print("  plot_one_hod_set('LRG', 'r200c')")
    print("  plot_one_profile_set('LRG')")
    print("  plot_all_comparisons(include_hod=True, include_profiles=True)")

# %% [markdown] cell 13
# ## Useful manual calls HOD figures: ```python plot_one_hod_set("LRG", "fof") plot_one_hod_set("LRG", "r200c") plot_one_hod_set("ELG", "fof") plot_one_hod_set("ELG", "r200c") ``` Radial-profile figures: ```python plot_one_profile_set("LRG") plot_one_profile_set("ELG") ``` Run everything: ```python manifest = plot_all_comparisons(include_hod=True, include_profiles=True) display(manifest) ``` You can adjust global settings at the top of the notebook, for example: ```python XLIM_M200C = (1e10, 3e15)
