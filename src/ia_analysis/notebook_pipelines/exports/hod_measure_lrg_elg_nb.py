"""Exported code from notebooks/raw_20260618/hod_measure_lrg_elg.ipynb.

This file is generated for project management and refactoring. Review before running end to end.
"""


# %% [markdown] cell 1
# # HOD measurement for LRG/ELG catalogues This notebook measures halo occupation distributions for galaxy catalogues: \[ \langle N_{\rm cen}\mid M_h\rangle,\quad \langle N_{\rm sat}\mid M_h\rangle,\quad \langle N_{\rm tot}\mid M_h\rangle \] It is designed for the MG/GR catalogues described by Michael's email: - halo catalogue: `Combined_Catalogs/Haloes/{model}_combined_halo_cat_{snap:03d}.txt` - halo mass default: `M200m`, i.e. Julia column 13, Python column 12 - galaxy catalogue columns are conf

# %% code cell 2
# ------------------------------------------------------------
# Imports and plotting defaults
# ------------------------------------------------------------
from __future__ import annotations

import os
import glob
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import h5py
except ImportError:
    h5py = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 220,
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
    "axes.linewidth": 1.1,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})

# %% [markdown] cell 3
# ## Configuration Edit this cell first. The halo catalogue part follows the email directly. The LRG/ELG part must be adjusted to the actual file layout. ### Important indexing convention Michael noted that some catalogues produced in Julia use **1-based indexing**. For any `host_col` or match index coming from such a file, set: ```python "host_index_base": 1 ``` If the file is already Python-style, use: ```python "host_index_base": 0 ``` The notebook also supports `"auto"`, but explicit is safer.

# %% code cell 4
# ------------------------------------------------------------
# Main configuration
# ------------------------------------------------------------
HUBBLE_H = 0.6774

# Edit as needed.
MODELS = ["GR", "F6", "F5", "F4"]
SNAPSHOTS = [21]

# Michael's corrected halo catalogue path.
COMBINED_CATALOG_DIR = "/cosma8/data/dp004/dc-coll5/MG_generic/Combined_Catalogs"
HALO_CATALOG_PATTERN = (
    COMBINED_CATALOG_DIR
    + "/Haloes/{model}_combined_halo_cat_{snap:03d}.txt"
)

# Halo mass column from Michael's email.
# Julia column 13 = Python column 12: M200m.
# Julia column 4  = Python column 3 : dark-matter M200c.
HALO_MASS_COL = 12
HALO_MASS_NAME = "M200m"
HALO_MASS_UNIT = "Msun"

# Halo row index convention. A row in the halo catalogue is treated as the FoF halo index.
# The row number itself is Python 0-based in this notebook.
HALO_INDEX_BASE = 0

# Mass bins for HOD curves.
# Adjust if your catalogue uses Msun/h or 1e10 Msun units.
LOGM_BINS = np.arange(10.5, 15.6, 0.25)

# Output directory.
OUTPUT_DIR = Path("hod_outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Optional model colors. Add/remove models freely.
MODEL_COLORS = {
    "GR": "black",
    "F6": "tab:blue",
    "F5.5": "tab:purple",
    "F5": "tab:green",
    "F4.5": "tab:orange",
    "F4": "tab:red",
}

# ------------------------------------------------------------
# Galaxy catalogue configuration
# ------------------------------------------------------------
# The path and columns below are templates. You must edit them to match the actual LRG/ELG files.
# Supported formats: auto, txt, csv, npy, npz, hdf5, h5.
# Supported column selectors:
#   - int:    column number for txt/csv/numpy arrays
#   - str:    column/dataset name for pandas/HDF5 tables
#
# central_mode options:
#   - "column": use central_col and central_values
#   - "rank": central if rank_col == central_rank_value
#   - "first_by_score": choose one central per host using max(score_col), useful only as a fallback
#   - "none": no central/satellite split; only N_tot is reliable
#
# host_index_base:
#   - 0, 1, or "auto"

GALAXY_CATALOGS = {
    "LRG": {
        # FIXME: replace this with the real LRG catalogue pattern.
        # Available placeholders: {model}, {snap}, {sample}
        "pattern": "/cosma8/data/dp004/dc-coll5/MG_generic/LRG_ELG_Catalogs/{model}_LRG_{snap:03d}.txt",
        "format": "auto",
        "has_header": False,
        "delimiter": None,
        "hdf_group": None,

        # FIXME: set these after inspecting the catalogue.
        "host_col": 0,
        "host_index_base": "auto",
        "central_mode": "column",
        "central_col": 1,
        "central_values": [1, True, "1", "True", "true", "central", "Central"],

        # Usually None for plain HOD counts.
        "weight_col": None,

        # If True, enforce N_cen <= 1 per halo after counting.
        "clip_centrals_to_one": True,
    },
    "ELG": {
        # FIXME: replace this with the real ELG catalogue pattern.
        "pattern": "/cosma8/data/dp004/dc-coll5/MG_generic/LRG_ELG_Catalogs/{model}_ELG_{snap:03d}.txt",
        "format": "auto",
        "has_header": False,
        "delimiter": None,
        "hdf_group": None,

        # FIXME: set these after inspecting the catalogue.
        "host_col": 0,
        "host_index_base": "auto",
        "central_mode": "column",
        "central_col": 1,
        "central_values": [1, True, "1", "True", "true", "central", "Central"],

        "weight_col": None,
        "clip_centrals_to_one": True,
    },
}

# %% [markdown] cell 5
# ## Helper functions

# %% code cell 6
# ------------------------------------------------------------
# Path and file inspection helpers
# ------------------------------------------------------------
def format_catalog_path(pattern: str, model: str, snap: int, sample: Optional[str] = None) -> str:
    """Format a catalogue path pattern."""
    return pattern.format(model=model, snap=int(snap), sample=sample if sample is not None else "")


def find_candidate_files(root: str, keywords: Iterable[str] = (), max_results: int = 50) -> list[str]:
    """Recursively search for candidate catalogue files.

    This is useful when the LRG/ELG path is unknown. Run this on COSMA, not locally.
    """
    root_path = Path(root)
    if not root_path.exists():
        print(f"Root does not exist: {root}")
        return []

    keywords_lower = [k.lower() for k in keywords]
    out = []
    for path in root_path.rglob("*"):
        if not path.is_file():
            continue
        name = str(path).lower()
        if all(k in name for k in keywords_lower):
            out.append(str(path))
            if len(out) >= max_results:
                break
    return out


def preview_configured_paths(models=MODELS, snapshots=SNAPSHOTS, samples=None):
    """Print whether configured halo and galaxy paths exist."""
    samples = list(GALAXY_CATALOGS) if samples is None else samples

    rows = []
    for model in models:
        for snap in snapshots:
            halo_path = format_catalog_path(HALO_CATALOG_PATTERN, model, snap)
            rows.append({
                "kind": "halo",
                "sample": "halo",
                "model": model,
                "snap": snap,
                "exists": Path(halo_path).exists(),
                "path": halo_path,
            })
            for sample in samples:
                cfg = GALAXY_CATALOGS[sample]
                gal_path = format_catalog_path(cfg["pattern"], model, snap, sample)
                rows.append({
                    "kind": "galaxy",
                    "sample": sample,
                    "model": model,
                    "snap": snap,
                    "exists": Path(gal_path).exists(),
                    "path": gal_path,
                })
    return pd.DataFrame(rows)

# %% code cell 7
# ------------------------------------------------------------
# Generic table readers
# ------------------------------------------------------------
def infer_format(path: Union[str, Path], requested: str = "auto") -> str:
    """Infer catalogue format from extension."""
    requested = (requested or "auto").lower()
    if requested != "auto":
        return requested
    suffix = Path(path).suffix.lower()
    if suffix in [".hdf5", ".h5"]:
        return "hdf5"
    if suffix == ".npy":
        return "npy"
    if suffix == ".npz":
        return "npz"
    if suffix == ".csv":
        return "csv"
    return "txt"


def read_hdf5_as_dataframe(path: Union[str, Path], group: Optional[str] = None) -> pd.DataFrame:
    """Read 1D HDF5 datasets in a group into a DataFrame.

    Multi-dimensional datasets are skipped, except arrays with shape (N, 1).
    """
    if h5py is None:
        raise ImportError("h5py is required for reading HDF5 files.")

    path = Path(path)
    data = {}
    with h5py.File(path, "r") as f:
        g = f[group] if group is not None else f

        def collect(prefix, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            arr = obj[()]
            if arr.ndim == 1:
                key = prefix.strip("/")
                data[key] = arr
            elif arr.ndim == 2 and arr.shape[1] == 1:
                key = prefix.strip("/")
                data[key] = arr[:, 0]

        g.visititems(collect)

    if not data:
        raise ValueError(f"No 1D datasets found in {path} group={group!r}.")

    lengths = {k: len(v) for k, v in data.items()}
    n_mode = max(set(lengths.values()), key=list(lengths.values()).count)
    data = {k: v for k, v in data.items() if len(v) == n_mode}
    return pd.DataFrame(data)


def read_table(path: Union[str, Path], cfg: Optional[Dict[str, Any]] = None) -> Union[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]:
    """Read a text/CSV/NumPy/HDF5 table."""
    cfg = {} if cfg is None else dict(cfg)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    fmt = infer_format(path, cfg.get("format", "auto"))

    if fmt in ["txt", "dat"]:
        has_header = bool(cfg.get("has_header", False))
        delimiter = cfg.get("delimiter", None)
        if has_header:
            return pd.read_csv(path, sep=delimiter if delimiter is not None else r"\s+", engine="python", comment="#")
        return np.loadtxt(path)

    if fmt == "csv":
        has_header = bool(cfg.get("has_header", True))
        if has_header:
            return pd.read_csv(path, comment="#")
        return pd.read_csv(path, header=None, comment="#")

    if fmt == "npy":
        return np.load(path, allow_pickle=False)

    if fmt == "npz":
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}

    if fmt in ["hdf5", "h5"]:
        return read_hdf5_as_dataframe(path, group=cfg.get("hdf_group", None))

    raise ValueError(f"Unsupported format: {fmt}")


def get_column(table: Union[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]], col: Union[int, str, None], name: str = "column") -> np.ndarray:
    """Extract a column from a DataFrame, array, or dictionary."""
    if col is None:
        raise ValueError(f"{name} is None.")

    if isinstance(table, pd.DataFrame):
        if isinstance(col, int):
            return table.iloc[:, col].to_numpy()
        return table[col].to_numpy()

    if isinstance(table, dict):
        if not isinstance(col, str):
            raise TypeError(f"For dict/npz tables, {name} must be a string key.")
        return np.asarray(table[col])

    arr = np.asarray(table)
    if arr.ndim == 1:
        if col not in [0, "0"]:
            raise ValueError(f"1D array only has column 0, but requested {col!r}.")
        return arr

    if isinstance(col, str):
        raise TypeError(f"For plain arrays, {name} must be an integer column index.")
    return arr[:, col]


def table_shape(table: Union[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]) -> Tuple[int, Optional[int]]:
    """Return table shape as (N, K)."""
    if isinstance(table, pd.DataFrame):
        return table.shape
    if isinstance(table, dict):
        n = len(next(iter(table.values()))) if table else 0
        return n, len(table)
    arr = np.asarray(table)
    if arr.ndim == 1:
        return len(arr), 1
    return arr.shape

# %% code cell 8
# ------------------------------------------------------------
# Halo and galaxy catalogue loading
# ------------------------------------------------------------
def read_halo_catalog(model: str, snap: int) -> pd.DataFrame:
    """Read Michael-style halo catalogue and return a halo-level DataFrame."""
    path = format_catalog_path(HALO_CATALOG_PATTERN, model, snap)
    arr = read_table(path, {"format": "txt", "has_header": False})
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Halo catalogue must be a 2D array. Got shape={arr.shape} from {path}")
    if arr.shape[1] <= HALO_MASS_COL:
        raise ValueError(f"Halo catalogue has only {arr.shape[1]} columns; HALO_MASS_COL={HALO_MASS_COL} is invalid.")

    n_halo = arr.shape[0]
    df = pd.DataFrame({
        "halo_index": np.arange(n_halo, dtype=np.int64),
        "model": model,
        "snap": int(snap),
        "M_halo": arr[:, HALO_MASS_COL].astype(float),
    })

    # Add useful Michael columns if present.
    if arr.shape[1] >= 15:
        df["x"] = arr[:, 0]
        df["y"] = arr[:, 1]
        df["z"] = arr[:, 2]
        df["M200c_dm"] = arr[:, 3]
        df["R200c"] = arr[:, 4]
        df["logrho0_nfw"] = arr[:, 5]
        df["c200c"] = arr[:, 6]
        df["SFR_central_2Rhalf"] = arr[:, 10]
        df["R200m"] = arr[:, 11]
        df["M200m"] = arr[:, 12]
        df["Rvir"] = arr[:, 13]
        df["Mvir"] = arr[:, 14]

    if np.any(~np.isfinite(df["M_halo"])) or np.any(df["M_halo"] <= 0):
        bad = np.sum((~np.isfinite(df["M_halo"])) | (df["M_halo"] <= 0))
        warnings.warn(f"{bad} haloes have non-positive or non-finite M_halo in {path}.")

    return df


def read_galaxy_catalog(sample: str, model: str, snap: int):
    """Read one configured galaxy catalogue."""
    cfg = GALAXY_CATALOGS[sample]
    path = format_catalog_path(cfg["pattern"], model, snap, sample=sample)
    table = read_table(path, cfg)
    return table, path, cfg


def inspect_catalogue(path: str, cfg: Optional[Dict[str, Any]] = None, nrows: int = 5):
    """Inspect a catalogue quickly."""
    table = read_table(path, cfg or {"format": "auto"})
    print("Path:", path)
    print("Type:", type(table))
    print("Shape:", table_shape(table))
    if isinstance(table, pd.DataFrame):
        display(table.head(nrows))
        print("Columns:", list(table.columns))
    elif isinstance(table, dict):
        print("Keys:", list(table.keys()))
        for k, v in list(table.items())[:10]:
            print(k, np.shape(v), np.asarray(v)[:nrows])
    else:
        arr = np.asarray(table)
        print(arr[:nrows])
    return table

# %% [markdown] cell 9
# ## Preview configured paths Run this before the full measurement. If the LRG/ELG paths are missing, use `find_candidate_files` to locate them and then edit `GALAXY_CATALOGS`.

# %% code cell 10
preview_df = preview_configured_paths()
display(preview_df)

# %% code cell 11
# Optional: search for candidate files on COSMA.
# This can be slow if the root is broad. Narrow the root if possible.
#
# candidates = find_candidate_files(
#     "/cosma8/data/dp004/dc-coll5/MG_generic",
#     keywords=["LRG", "099"],
#     max_results=30,
# )
# for p in candidates:
#     print(p)

# %% [markdown] cell 12
# ## Occupation counting functions

# %% code cell 13
# ------------------------------------------------------------
# Index conversion and central/satellite classification
# ------------------------------------------------------------
def convert_host_indices(raw_host: np.ndarray, n_halo: int, index_base: Union[int, str] = 0) -> np.ndarray:
    """Convert host halo indices to Python 0-based indexing.

    Parameters
    ----------
    raw_host:
        Host halo indices from the galaxy catalogue.
    n_halo:
        Number of haloes in the corresponding halo catalogue.
    index_base:
        0, 1, or "auto".
    """
    host = np.asarray(raw_host)
    if np.any(~np.isfinite(host.astype(float))):
        raise ValueError("Host index contains non-finite values.")
    host = host.astype(np.int64)

    if index_base == "auto":
        valid_like = host[host >= 0]
        if len(valid_like) == 0:
            return host
        minv = valid_like.min()
        maxv = valid_like.max()
        if minv == 0:
            detected = 0
        elif minv >= 1 and maxv <= n_halo:
            detected = 1
        elif minv >= 0 and maxv < n_halo:
            detected = 0
        else:
            raise ValueError(
                f"Cannot auto-detect host_index_base: min={minv}, max={maxv}, n_halo={n_halo}. "
                "Set host_index_base explicitly to 0 or 1."
            )
        print(f"Auto-detected host_index_base={detected} for host range [{minv}, {maxv}], n_halo={n_halo}")
        index_base = detected

    if index_base not in [0, 1]:
        raise ValueError("index_base must be 0, 1, or 'auto'.")

    out = host.copy()
    positive_or_valid = out >= 0
    out[positive_or_valid] = out[positive_or_valid] - int(index_base)
    return out


def central_mask_from_config(table, host_index: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """Return a boolean mask selecting central galaxies."""
    mode = cfg.get("central_mode", "column")
    n_gal = len(host_index)

    if mode == "none":
        warnings.warn("central_mode='none': N_cen and N_sat cannot be measured reliably. N_cen will be zero.")
        return np.zeros(n_gal, dtype=bool)

    if mode == "column":
        raw = get_column(table, cfg.get("central_col"), name="central_col")
        values = cfg.get("central_values", [1, True])
        return np.isin(raw, values)

    if mode == "rank":
        raw = get_column(table, cfg.get("rank_col"), name="rank_col")
        value = cfg.get("central_rank_value", 0)
        return raw == value

    if mode == "first_by_score":
        # Fallback: choose one object per host with maximum score as central.
        # This is not as clean as a true central flag. Use only if no central flag exists.
        score = get_column(table, cfg.get("score_col"), name="score_col").astype(float)
        central = np.zeros(n_gal, dtype=bool)
        valid = host_index >= 0
        df = pd.DataFrame({"i": np.arange(n_gal), "host": host_index, "score": score})
        df = df[valid]
        if len(df) > 0:
            idx = df.sort_values("score").groupby("host").tail(1)["i"].to_numpy()
            central[idx] = True
        warnings.warn("central_mode='first_by_score' is a fallback, not a physical central flag.")
        return central

    raise ValueError(f"Unknown central_mode={mode!r}")


def count_occupations(halo_df: pd.DataFrame, gal_table, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Count N_cen, N_sat, and N_tot for every halo."""
    n_halo = len(halo_df)
    raw_host = get_column(gal_table, cfg.get("host_col"), name="host_col")
    host = convert_host_indices(raw_host, n_halo=n_halo, index_base=cfg.get("host_index_base", 0))

    valid = (host >= 0) & (host < n_halo)
    n_invalid = np.count_nonzero(~valid)
    if n_invalid > 0:
        warnings.warn(f"Dropping {n_invalid} galaxies with invalid host indices outside [0, {n_halo - 1}].")

    host_valid = host[valid]

    weight_col = cfg.get("weight_col", None)
    if weight_col is None:
        weight = np.ones(len(host), dtype=float)
    else:
        weight = get_column(gal_table, weight_col, name="weight_col").astype(float)
    weight_valid = weight[valid]

    central_all = central_mask_from_config(gal_table, host, cfg)
    central_valid = central_all[valid]

    N_tot = np.bincount(host_valid, weights=weight_valid, minlength=n_halo).astype(float)
    N_cen = np.bincount(host_valid[central_valid], weights=weight_valid[central_valid], minlength=n_halo).astype(float)

    if cfg.get("clip_centrals_to_one", False):
        too_many = np.count_nonzero(N_cen > 1)
        if too_many > 0:
            warnings.warn(f"{too_many} haloes have N_cen > 1. Clipping N_cen to 1.")
        N_cen = np.minimum(N_cen, 1.0)

    N_sat = N_tot - N_cen
    if np.any(N_sat < -1e-8):
        warnings.warn("Some haloes have N_sat < 0 after central clipping; check central flags/weights.")
    N_sat = np.maximum(N_sat, 0.0)

    out = halo_df.copy()
    out["N_cen"] = N_cen
    out["N_sat"] = N_sat
    out["N_tot"] = N_tot

    print(
        f"Counted galaxies: input={len(host)}, valid={len(host_valid)}, "
        f"sum(N_tot)={N_tot.sum():.0f}, sum(N_cen)={N_cen.sum():.0f}, sum(N_sat)={N_sat.sum():.0f}"
    )
    return out

# %% code cell 14
# ------------------------------------------------------------
# HOD binning
# ------------------------------------------------------------
def hod_from_counts(
    count_df: pd.DataFrame,
    logm_bins: np.ndarray = LOGM_BINS,
    sample: str = "sample",
    model: str = "model",
    snap: int = 0,
) -> pd.DataFrame:
    """Compute <N_cen>, <N_sat>, and <N_tot> in halo mass bins."""
    mass = count_df["M_halo"].to_numpy(dtype=float)
    logm = np.log10(mass)

    rows = []
    components = ["N_cen", "N_sat", "N_tot"]
    for lo, hi in zip(logm_bins[:-1], logm_bins[1:]):
        msel = (logm >= lo) & (logm < hi) & np.isfinite(logm)
        n_halo_bin = int(np.count_nonzero(msel))
        if n_halo_bin == 0:
            for comp in components:
                rows.append({
                    "sample": sample,
                    "model": model,
                    "snap": int(snap),
                    "component": comp,
                    "logM_lo": lo,
                    "logM_hi": hi,
                    "logM_mid": 0.5 * (lo + hi),
                    "logM_mean": np.nan,
                    "N_mean": np.nan,
                    "N_std": np.nan,
                    "N_sem": np.nan,
                    "N_halo": 0,
                    "N_gal_sum": np.nan,
                })
            continue

        logm_mean = float(np.mean(logm[msel]))
        for comp in components:
            vals = count_df.loc[msel, comp].to_numpy(dtype=float)
            rows.append({
                "sample": sample,
                "model": model,
                "snap": int(snap),
                "component": comp,
                "logM_lo": lo,
                "logM_hi": hi,
                "logM_mid": 0.5 * (lo + hi),
                "logM_mean": logm_mean,
                "N_mean": float(np.mean(vals)),
                "N_std": float(np.std(vals, ddof=1)) if n_halo_bin > 1 else 0.0,
                "N_sem": float(np.std(vals, ddof=1) / np.sqrt(n_halo_bin)) if n_halo_bin > 1 else 0.0,
                "N_halo": n_halo_bin,
                "N_gal_sum": float(np.sum(vals)),
            })
    return pd.DataFrame(rows)


def measure_one(sample: str, model: str, snap: int, save_counts: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Measure per-halo counts and binned HOD for one sample/model/snapshot."""
    print(f"\n=== Measuring {sample}, {model}, snap={snap:03d} ===")
    halo_df = read_halo_catalog(model, snap)
    gal_table, gal_path, cfg = read_galaxy_catalog(sample, model, snap)
    print("Galaxy path:", gal_path)
    print("Galaxy table shape:", table_shape(gal_table))

    count_df = count_occupations(halo_df, gal_table, cfg)
    count_df["sample"] = sample

    hod_df = hod_from_counts(count_df, LOGM_BINS, sample=sample, model=model, snap=snap)

    if save_counts:
        count_path = OUTPUT_DIR / f"halo_occupation_{sample}_{model}_snap{snap:03d}.csv"
        hod_path = OUTPUT_DIR / f"hod_{sample}_{model}_snap{snap:03d}.csv"
        count_df.to_csv(count_path, index=False)
        hod_df.to_csv(hod_path, index=False)
        print("Saved:", count_path)
        print("Saved:", hod_path)

    return count_df, hod_df


def measure_all(samples=None, models=MODELS, snapshots=SNAPSHOTS, save_counts: bool = True):
    """Measure all configured samples/models/snapshots."""
    samples = list(GALAXY_CATALOGS) if samples is None else list(samples)
    count_tables = []
    hod_tables = []

    tasks = [(sample, model, snap) for sample in samples for model in models for snap in snapshots]
    for sample, model, snap in tqdm(tasks, desc="HOD tasks"):
        try:
            count_df, hod_df = measure_one(sample, model, snap, save_counts=save_counts)
            count_tables.append(count_df)
            hod_tables.append(hod_df)
        except FileNotFoundError as e:
            warnings.warn(f"Missing file for {sample}, {model}, snap={snap:03d}: {e}")
        except Exception as e:
            warnings.warn(f"Failed {sample}, {model}, snap={snap:03d}: {type(e).__name__}: {e}")
            raise

    all_counts = pd.concat(count_tables, ignore_index=True) if count_tables else pd.DataFrame()
    all_hod = pd.concat(hod_tables, ignore_index=True) if hod_tables else pd.DataFrame()

    if len(all_hod) > 0:
        all_hod_path = OUTPUT_DIR / "hod_all.csv"
        all_hod.to_csv(all_hod_path, index=False)
        print("Saved combined HOD:", all_hod_path)

    if len(all_counts) > 0:
        all_counts_path = OUTPUT_DIR / "halo_occupation_all.csv"
        all_counts.to_csv(all_counts_path, index=False)
        print("Saved combined per-halo occupations:", all_counts_path)

    return all_counts, all_hod

# %% [markdown] cell 15
# ## Run measurement After editing the configuration, run the following cell.

# %% code cell 16
# Set this to True after checking the paths and column configuration.
RUN_MEASUREMENT = False

if RUN_MEASUREMENT:
    all_counts, all_hod = measure_all()
    display(all_hod.head())
else:
    print("RUN_MEASUREMENT is False. Edit the configuration, preview paths, then set it to True.")

# %% [markdown] cell 17
# ## Plotting functions

# %% code cell 18
# ------------------------------------------------------------
# HOD plotting utilities
# ------------------------------------------------------------
COMPONENT_LABELS = {
    "N_cen": r"$\langle N_{\rm cen}\rangle$",
    "N_sat": r"$\langle N_{\rm sat}\rangle$",
    "N_tot": r"$\langle N_{\rm tot}\rangle$",
}

COMPONENT_LINESTYLES = {
    "N_tot": "-",
    "N_cen": "--",
    "N_sat": ":",
}


def _safe_hod_xy(df: pd.DataFrame, component: str, min_halo: int = 3):
    sub = df[(df["component"] == component) & (df["N_halo"] >= min_halo)].copy()
    sub = sub[np.isfinite(sub["N_mean"]) & np.isfinite(sub["logM_mid"])]
    return sub


def plot_components_for_model(
    hod_df: pd.DataFrame,
    sample: str,
    model: str,
    snap: int,
    yscale: str = "log",
    min_halo: int = 3,
    save: bool = True,
):
    """Plot N_cen, N_sat, and N_tot for one sample/model/snapshot."""
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    for comp in ["N_tot", "N_cen", "N_sat"]:
        sub = hod_df[
            (hod_df["sample"] == sample)
            & (hod_df["model"] == model)
            & (hod_df["snap"] == int(snap))
        ]
        sub = _safe_hod_xy(sub, comp, min_halo=min_halo)
        if yscale == "log":
            sub = sub[sub["N_mean"] > 0]
        if len(sub) == 0:
            continue
        ax.plot(
            sub["logM_mid"],
            sub["N_mean"],
            linestyle=COMPONENT_LINESTYLES[comp],
            linewidth=2.0,
            label=COMPONENT_LABELS[comp],
        )
        # Optional SEM band; uncomment if desired.
        # lo = np.maximum(sub["N_mean"] - sub["N_sem"], 1e-8)
        # hi = sub["N_mean"] + sub["N_sem"]
        # ax.fill_between(sub["logM_mid"], lo, hi, alpha=0.18)

    ax.set_xlabel(rf"$\log_{{10}}(M_{{\rm halo}}/[{HALO_MASS_UNIT}])$  ({HALO_MASS_NAME})")
    ax.set_ylabel(r"Mean occupation")
    ax.set_title(f"{sample} HOD: {model}, snap {snap:03d}")
    if yscale == "log":
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-3)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    if save:
        out = OUTPUT_DIR / f"hod_components_{sample}_{model}_snap{snap:03d}.png"
        fig.savefig(out, bbox_inches="tight")
        print("Saved:", out)
    return fig, ax


def plot_models_for_component(
    hod_df: pd.DataFrame,
    sample: str,
    snap: int,
    component: str = "N_tot",
    models=MODELS,
    yscale: str = "log",
    min_halo: int = 3,
    save: bool = True,
):
    """Plot one component for all models at one snapshot."""
    fig, ax = plt.subplots(figsize=(6.6, 4.8))

    for model in models:
        sub = hod_df[
            (hod_df["sample"] == sample)
            & (hod_df["model"] == model)
            & (hod_df["snap"] == int(snap))
        ]
        sub = _safe_hod_xy(sub, component, min_halo=min_halo)
        if yscale == "log":
            sub = sub[sub["N_mean"] > 0]
        if len(sub) == 0:
            continue
        ax.plot(
            sub["logM_mid"],
            sub["N_mean"],
            linewidth=2.0,
            color=MODEL_COLORS.get(model, None),
            label=model,
        )

    ax.set_xlabel(rf"$\log_{{10}}(M_{{\rm halo}}/[{HALO_MASS_UNIT}])$  ({HALO_MASS_NAME})")
    ax.set_ylabel(COMPONENT_LABELS.get(component, component))
    ax.set_title(f"{sample}: {COMPONENT_LABELS.get(component, component)}, snap {snap:03d}")
    if yscale == "log":
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-3)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    if save:
        out = OUTPUT_DIR / f"hod_models_{sample}_{component}_snap{snap:03d}.png"
        fig.savefig(out, bbox_inches="tight")
        print("Saved:", out)
    return fig, ax


def plot_all_hod(hod_df: pd.DataFrame, samples=None, snapshots=None, models=MODELS, save: bool = True):
    """Produce a standard set of HOD plots."""
    samples = sorted(hod_df["sample"].unique()) if samples is None else list(samples)
    snapshots = sorted(hod_df["snap"].unique()) if snapshots is None else list(snapshots)

    for sample in samples:
        for snap in snapshots:
            for comp in ["N_tot", "N_cen", "N_sat"]:
                plot_models_for_component(hod_df, sample, snap, component=comp, models=models, save=save)
            for model in models:
                plot_components_for_model(hod_df, sample, model, snap, save=save)

# %% [markdown] cell 19
# ## Make plots Run after `all_hod` has been created. If you restarted the kernel, load `hod_outputs/hod_all.csv` first.

# %% code cell 20
# Load previous combined HOD table if needed.
hod_all_path = OUTPUT_DIR / "hod_all.csv"
if "all_hod" not in globals() or len(globals().get("all_hod", [])) == 0:
    if hod_all_path.exists():
        all_hod = pd.read_csv(hod_all_path)
        print("Loaded:", hod_all_path)
    else:
        all_hod = pd.DataFrame()
        print("No HOD table found yet. Run the measurement first.")

display(all_hod.head() if len(all_hod) else all_hod)

# %% code cell 21
RUN_PLOTS = False

if RUN_PLOTS and len(all_hod) > 0:
    plot_all_hod(all_hod, save=True)
else:
    print("RUN_PLOTS is False or all_hod is empty. Set RUN_PLOTS=True after measuring HOD.")

# %% [markdown] cell 22
# ## Optional: environment-split HOD If you later compute an environment value per halo, such as `rho_env` or `delta_env`, merge it into `all_counts` and use this function to split the HOD by environment quantile. Expected columns in `count_df`: - `M_halo` - `N_cen`, `N_sat`, `N_tot` - an environment column, e.g. `rho_env`

# %% code cell 23
def add_environment_quantiles(
    count_df: pd.DataFrame,
    env_col: str = "rho_env",
    q: int = 4,
    label_col: str = "env_bin",
    within_mass_bins: bool = False,
    logm_bins: np.ndarray = LOGM_BINS,
) -> pd.DataFrame:
    """Add environment quantile labels to a per-halo occupation table.

    If within_mass_bins=True, environment quantiles are computed separately in each mass bin.
    This is usually better for assembly-bias studies because environment correlates with mass.
    """
    df = count_df.copy()
    df[label_col] = pd.NA

    if env_col not in df.columns:
        raise KeyError(f"Missing env_col={env_col!r} in count_df.")

    if not within_mass_bins:
        good = np.isfinite(df[env_col].to_numpy(dtype=float))
        df.loc[good, label_col] = pd.qcut(df.loc[good, env_col], q=q, labels=False, duplicates="drop")
        return df

    logm = np.log10(df["M_halo"].to_numpy(dtype=float))
    for lo, hi in zip(logm_bins[:-1], logm_bins[1:]):
        msel = (logm >= lo) & (logm < hi) & np.isfinite(logm)
        good = msel & np.isfinite(df[env_col].to_numpy(dtype=float))
        if np.count_nonzero(good) < q:
            continue
        df.loc[good, label_col] = pd.qcut(df.loc[good, env_col], q=q, labels=False, duplicates="drop")
    return df


def hod_from_counts_by_environment(
    count_df: pd.DataFrame,
    env_label_col: str = "env_bin",
    logm_bins: np.ndarray = LOGM_BINS,
    sample: str = "sample",
    model: str = "model",
    snap: int = 0,
) -> pd.DataFrame:
    """Compute HOD separately for each environment bin."""
    tables = []
    for env_bin, sub in count_df.dropna(subset=[env_label_col]).groupby(env_label_col):
        h = hod_from_counts(sub, logm_bins=logm_bins, sample=sample, model=model, snap=snap)
        h["env_bin"] = int(env_bin)
        tables.append(h)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def plot_environment_hod(
    env_hod_df: pd.DataFrame,
    sample: str,
    model: str,
    snap: int,
    component: str = "N_tot",
    yscale: str = "log",
    min_halo: int = 3,
):
    """Plot environment-split HOD for one sample/model/snapshot/component."""
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    base = env_hod_df[
        (env_hod_df["sample"] == sample)
        & (env_hod_df["model"] == model)
        & (env_hod_df["snap"] == int(snap))
        & (env_hod_df["component"] == component)
        & (env_hod_df["N_halo"] >= min_halo)
    ]
    for env_bin, sub in base.groupby("env_bin"):
        sub = sub[np.isfinite(sub["N_mean"])]
        if yscale == "log":
            sub = sub[sub["N_mean"] > 0]
        if len(sub) == 0:
            continue
        ax.plot(sub["logM_mid"], sub["N_mean"], linewidth=2.0, label=f"env bin {env_bin}")

    ax.set_xlabel(rf"$\log_{{10}}(M_{{\rm halo}}/[{HALO_MASS_UNIT}])$  ({HALO_MASS_NAME})")
    ax.set_ylabel(COMPONENT_LABELS.get(component, component))
    ax.set_title(f"Environment-split {sample} HOD: {model}, snap {snap:03d}")
    if yscale == "log":
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-3)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax

# %% [markdown] cell 24
# ## Quick sanity checks Use these checks after measuring HOD. They are deliberately strict because HOD errors are often caused by a wrong host-index convention.

# %% code cell 25
def sanity_check_counts(count_df: pd.DataFrame, sample: Optional[str] = None):
    """Print useful sanity checks for a per-halo occupation table."""
    df = count_df.copy()
    if sample is not None and "sample" in df.columns:
        df = df[df["sample"] == sample]
    if len(df) == 0:
        print("Empty count table.")
        return

    print("Rows / haloes:", len(df))
    print("Total N_tot:", df["N_tot"].sum())
    print("Total N_cen:", df["N_cen"].sum())
    print("Total N_sat:", df["N_sat"].sum())
    print("Haloes with N_tot > 0:", np.count_nonzero(df["N_tot"] > 0))
    print("Haloes with N_cen > 1:", np.count_nonzero(df["N_cen"] > 1))
    print("Haloes with N_sat < 0:", np.count_nonzero(df["N_sat"] < 0))
    print("Mass range:", np.nanmin(df["M_halo"]), np.nanmax(df["M_halo"]))

    # If nearly all galaxies fall into one or a few halo indices, the host column is probably wrong.
    top = df.sort_values("N_tot", ascending=False).head(10)
    display(top[["sample", "model", "snap", "halo_index", "M_halo", "N_cen", "N_sat", "N_tot"]] if "sample" in top.columns else top.head())

# Example:
# sanity_check_counts(all_counts, sample="LRG")
