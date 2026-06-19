"""
arts.py

Publication-style plotting helpers for MA/MArenew intrinsic-alignment catalogues.

This module is designed to be used by `full_alignments_MArenew.ipynb`.
It keeps the original schematic helper `visualize_galaxy_system` idea, but adds a
complete plotting API:

    set_plot_output_root(...)
    set_paper_style()
    set_alignment_context(MAset, flags, zmap, snap_list=None)
    list_alignment_chapters()
    plot_alignment_chapter(...)
    plot_alignment(...)
    plot_physical(...)

The code is intentionally defensive: it accepts several MA dictionary layouts and
skips a curve gracefully if a required field is missing.
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.lines import Line2D

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

# -----------------------------------------------------------------------------
# Global plotting context
# -----------------------------------------------------------------------------

PLOT_ROOT = Path("./plots/full_alignments")

MASET: Dict[str, Dict[str, dict]] = {}
FLAGS: List[str] = []
ZMAP: Dict[int, float] = {}
SNAP_LIST: List[int] = []

# Common ClusterSims snapshot redshifts.  If your MArenew.pkl stores a different
# z-map, pass it through set_alignment_context(...).
ZMAP_ALL = {
    1: 2.00,
    3: 1.50,
    6: 1.00,
    8: 0.75,
    10: 0.60,
    12: 0.50,
    15: 0.30,
    18: 0.15,
    21: 0.00,
    33: 2.00,
    40: 1.50,
    50: 1.00,
    59: 0.70,
    67: 0.50,
    72: 0.40,
    78: 0.30,
    84: 0.20,
    91: 0.10,
    99: 0.00,
}

FR_COLORS = {
    "F4":   "#ff0000",
    "F4.5": "#ff8c00",
    "F5":   "#008000",
    "F5.5": "#ff00ff",
    "F6":   "#0000ff",
    "F6.5": "#9c9c9c",
    "F40":  "#ff0000",
    "F45":  "#ff8c00",
    "F50":  "#008000",
    "F55":  "#ff00ff",
    "F60":  "#0000ff",
    "F65":  "#9c9c9c",
    "GR":   "#000000",
    "TNG":  "#000000",
    "TNG300-1": "#000000",
}

MODEL_LABELS = {
    "F40": "F4",
    "F45": "F4.5",
    "F50": "F5",
    "F55": "F5.5",
    "F60": "F6",
    "F65": "F6.5",
    "GR": "GR",
    "TNG": "TNG",
    "TNG300-1": "TNG",
}

# Small legacy palettes used by the schematic helper.
clist = [
    "#c02c38", "#c2c116", "#3c9566", "#1177b0",
    "#ff7c38", "#bec936", "#e03e36", "#b80d57",
    "#700961", "#11659a", "#abcdef", "#fedcba",
]
DH = [
    "#A73D30", "#C16355", "#D77E73", "#F0D0C6",
    "#0C52B5", "#387CBC", "#5F81C2", "#79B9DC",
    "#81521D", "#C1823E", "#DAB25B", "#E9D077",
    "#305937", "#718A70", "#68A270", "#8FC198",
]


def set_plot_output_root(root: os.PathLike | str) -> None:
    """Set the global output folder for all figures."""
    global PLOT_ROOT
    PLOT_ROOT = Path(root)
    _ensure_plot_directories(PLOT_ROOT)


def _ensure_plot_directories(root: os.PathLike | str | None = None) -> None:
    """Create output subdirectories used by this module."""
    root = Path(PLOT_ROOT if root is None else root)
    for sub in [
        "alignment_snapshot_grids",
        "alignment_redshift_evolution",
        "physical_distributions",
        "shape_cluster_alignment",
    ]:
        (root / sub).mkdir(parents=True, exist_ok=True)


def set_paper_style() -> None:
    """Apply a clean serif publication plotting style."""
    if sns is not None:
        sns.set_theme(style="ticks")
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 220,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "axes.linewidth": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })


def set_alignment_context(
    MAset: Dict[str, Dict[str, dict]],
    flags: Sequence[str],
    zmap: Dict[int, float] | None = None,
    snap_list: Sequence[int] | None = None,
) -> None:
    """Register the MA catalogues that plotting functions should use."""
    global MASET, FLAGS, ZMAP, SNAP_LIST
    MASET = MAset
    FLAGS = list(flags)
    SNAP_LIST = [int(s) for s in (snap_list if snap_list is not None else _infer_snap_list(MAset, flags))]
    if zmap is None:
        ZMAP = {int(s): ZMAP_ALL.get(int(s), np.nan) for s in SNAP_LIST}
    else:
        ZMAP = {int(k): float(v) for k, v in zmap.items()}


def _infer_snap_list(MAset: Dict[str, Dict[str, dict]], flags: Sequence[str]) -> List[int]:
    snaps = set()
    for f in flags:
        for skey in MAset.get(f, {}):
            try:
                snaps.add(int(str(skey).replace("snap", "")))
            except Exception:
                pass
    return sorted(snaps)


def flag_label(flag: str) -> str:
    return MODEL_LABELS.get(flag, str(flag))


def flag_color(flag: str) -> str:
    label = flag_label(flag)
    return FR_COLORS.get(str(flag), FR_COLORS.get(label, "k"))


def _snap_key(snap: int | str) -> str:
    if isinstance(snap, bytes):
        snap = snap.decode()
    if isinstance(snap, str):
        s = snap.strip().replace("snap", "").replace("_", "")
        if s.isdigit():
            return f"{int(s):03d}"
        return s
    return f"{int(snap):03d}"


def available_flags_for_snap(snap: int) -> List[str]:
    skey = _snap_key(snap)
    return [f for f in FLAGS if skey in MASET.get(f, {})]


# -----------------------------------------------------------------------------
# MA/global catalogue loaders
# -----------------------------------------------------------------------------

def _ordered_model_flags(found_flags, requested_flags=None):
    """Return model flags in a stable, publication-friendly order."""
    found = [str(f) for f in found_flags]
    if requested_flags is not None:
        return [str(f) for f in requested_flags if str(f) in found]
    order = ["GR", "F40", "F45", "F50", "F55", "F60", "F65", "F4", "F4.5", "F5", "F5.5", "F6", "F6.5", "TNG", "TNG300-1"]
    out = [f for f in order if f in found]
    out += sorted([f for f in found if f not in out])
    return out


def _ordered_snapshot_list(found_snaps, requested_snap_list=None):
    """Return snapshot integers in a stable order."""
    vals = sorted({int(s) for s in found_snaps})
    if requested_snap_list is None:
        return vals
    have = set(vals)
    return [int(s) for s in requested_snap_list if int(s) in have]


def _looks_like_ma_catalog(obj) -> bool:
    """Heuristic test for one MA catalogue dictionary."""
    if not isinstance(obj, dict):
        return False
    keys = set(obj.keys())
    common = {
        "SubhaloID", "GroupID", "CenID", "I_Star", "I_DM", "R", "V",
        "SubhaloMassInRadType", "Group_M_Crit200", "T_GR", "T_grp", "T_MG",
        "Tidal_grp", "Tidal_tot", "Star/I", "DM/I",
    }
    return len(keys & common) >= 2


def _standardize_ma_keys(MA: dict) -> dict:
    """
    Add compatibility aliases to a loaded MA dictionary.

    The plotting code historically expects keys such as I_Star, I_DM, R, V,
    T_GR, T_grp and T_MG. Newer HDF5 outputs may instead store quantities under
    groups such as Star/I, DM/I, Tidal_grp, Tidal_tot, pos_rel and vel_rel.
    This function keeps all original keys and adds the legacy aliases.
    """
    out = dict(MA)

    def alias(dst, *srcs):
        if dst in out:
            return
        for src in srcs:
            if src in out:
                out[dst] = out[src]
                return

    # Position/velocity aliases.
    alias("R", "pos_rel", "SubhaloPosRel", "r_rel")
    alias("V", "vel_rel", "SubhaloVelRel", "v_rel")
    alias("pos_abs", "SubhaloPos", "Coordinates")
    alias("vel_abs", "SubhaloVel", "Velocities")

    # Shape/inertia aliases from grouped HDF5 layouts.
    alias("I_Star", "Star/I", "stars/I", "Stellar/I", "I_stars")
    alias("I_DM", "DM/I", "dm/I", "DarkMatter/I")
    alias("dI_Star", "Star/dI", "stars/dI", "Stellar/dI")
    alias("dI_DM", "DM/dI", "dm/dI", "DarkMatter/dI")
    alias("L_Star", "Star/L", "stars/L", "Stellar/L")
    alias("L_DM", "DM/L", "dm/L", "DarkMatter/L")
    alias("omega_Star", "Star/omega", "stars/omega", "omega_star")
    alias("omega_DM", "DM/omega", "dm/omega", "omega_dm")
    alias("kappa_rot_Star", "Star/kappa_rot", "stars/kappa_rot")
    alias("kappa_rot_DM", "DM/kappa_rot", "dm/kappa_rot")
    alias("Neff_Star", "Star/Neff", "stars/Neff")
    alias("Neff_DM", "DM/Neff", "dm/Neff")
    alias("axis_relerr_Star", "Star/axis_relerr", "stars/axis_relerr")
    alias("axis_relerr_DM", "DM/axis_relerr", "dm/axis_relerr")
    alias("cos_err_Star", "Star/cos_err", "stars/cos_err")
    alias("cos_err_DM", "DM/cos_err", "dm/cos_err")
    alias("converged_Star", "Star/converged", "stars/converged")
    alias("converged_DM", "DM/converged", "dm/converged")

    # Historical scalar quality cuts often use max cos_err per object.
    if "cos_err_max_Star" not in out and "cos_err_Star" in out:
        ce = np.asarray(out["cos_err_Star"], dtype=float)
        out["cos_err_max_Star"] = np.nanmax(np.abs(ce), axis=1) if ce.ndim == 2 else np.abs(ce)
    if "cos_err_max_DM" not in out and "cos_err_DM" in out:
        ce = np.asarray(out["cos_err_DM"], dtype=float)
        out["cos_err_max_DM"] = np.nanmax(np.abs(ce), axis=1) if ce.ndim == 2 else np.abs(ce)

    # Tidal tensor aliases.  The three physical classes are:
    #   T_GR   : standard GR/Newtonian total tidal tensor
    #   T_grp  : group/intra-halo matter tidal tensor
    #   T_MG   : GR+MG total tidal tensor, if available
    alias("T_grp", "Tidal_grp", "T_group", "Tidal_group")
    alias("T_GR", "Tidal_GR", "T_GR", "Tidal_tot", "T_tot")
    alias("T_MG", "Tidal_GRMG", "Tidal_tot_mg", "Tidal_tot_MG", "T_GRMG", "Tidal_MG", "T_MG")
    # If this is a GR/TNG-only file with no explicit GR+MG tensor, use T_GR as a harmless fallback.
    if "T_MG" not in out and "T_GR" in out:
        out["T_MG"] = out["T_GR"]

    # Basic mass aliases.
    alias("SubhaloMass_1e10Msun_h", "SubhaloMass")
    alias("Group_M_Crit200_1e10Msun_h", "Group_M_Crit200")
    alias("Group_R_Crit200_ckpc_h", "Group_R_Crit200")

    return out


def _read_hdf5_flat(path) -> dict:
    """Read an HDF5 file into a flat dictionary with full-path keys and basename keys."""
    try:
        import h5py
    except Exception as exc:  # pragma: no cover
        raise ImportError("h5py is required to read HDF5 MA catalogues.") from exc

    out = {}
    with h5py.File(path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                try:
                    arr = obj[()]
                except Exception:
                    return
                out[name] = arr
                base = name.split("/")[-1]
                if base not in out:
                    out[base] = arr
        f.visititems(visitor)
    return _standardize_ma_keys(out)


def _candidate_alignment_files(root, flag, snap):
    """Candidate global catalogue filenames for one model and snapshot."""
    root = Path(root)
    flag = str(flag)
    snap = int(snap)
    s3 = f"{snap:03d}"
    patterns = [
        f"global_{flag}_s{s3}.hdf5",
        f"global_{flag}_s{s3}.h5",
        f"{flag}_s{s3}.hdf5",
        f"{flag}_s{s3}.h5",
        f"L302_N1136_{flag}_s{s3}.hdf5",
        f"L302_N1136_{flag}_s{s3}.h5",
        f"global_{flag}_{s3}.hdf5",
        f"global_{flag}_{s3}.h5",
    ]
    candidates = []
    for p in patterns:
        candidates.append(root / p)
        candidates.append(root / flag / p)
    # TNG naming convention.
    if flag.upper().startswith("TNG") or flag == "TNG300-1":
        candidates.extend([
            root / f"global_TNG_s{s3}.hdf5",
            root / f"global_TNG_s{s3}.h5",
            root / "TNG" / f"global_TNG_s{s3}.hdf5",
            root / "TNG300-1" / f"global_TNG_s{s3}.hdf5",
        ])
    # Keep order but remove duplicates.
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def _discover_alignment_files(root):
    """
    Discover global alignment HDF5 files under root.

    Returns
    -------
    found : dict
        Mapping (flag, snap_int) -> path.
    """
    root = Path(root)
    found = {}
    if not root.exists():
        return found

    files = list(root.glob("*.hdf5")) + list(root.glob("*.h5"))
    files += list(root.glob("*/*.hdf5")) + list(root.glob("*/*.h5"))

    import re
    patterns = [
        re.compile(r"global_(?P<flag>.+?)_s(?P<snap>\d{1,3})\.hdf5$"),
        re.compile(r"global_(?P<flag>.+?)_s(?P<snap>\d{1,3})\.h5$"),
        re.compile(r"L302_N1136_(?P<flag>.+?)_s(?P<snap>\d{1,3})\.hdf5$"),
        re.compile(r"(?P<flag>GR|F\d+|F\d+\.\d+|TNG|TNG300-1)_s(?P<snap>\d{1,3})\.hdf5$"),
    ]
    for fp in files:
        name = fp.name
        for pat in patterns:
            m = pat.match(name)
            if m:
                flag = m.group("flag")
                snap = int(m.group("snap"))
                found[(flag, snap)] = fp
                break
    return found


def load_alignment_maset(
    root_dir,
    requested_flags=None,
    snap_list=None,
    verbose=True,
    strict=False,
):
    """
    Load global alignment catalogues from HDF5 files.

    Parameters
    ----------
    root_dir : str or Path
        Directory containing global HDF5 files, e.g.
        global_GR_s021.hdf5 or L302_N1136_GR_s021.hdf5.
    requested_flags : sequence[str] or None
        Models to load.  If None, all discovered models are used.
    snap_list : sequence[int] or None
        Snapshots to load.  If None, all discovered snapshots are used.
    verbose : bool
        Print loaded and missing files.
    strict : bool
        If True, raise FileNotFoundError when any requested file is missing.

    Returns
    -------
    MAset : dict
        MAset[flag][snap_key] = MA dictionary.
    flags : list[str]
        Loaded model flags in stable order.
    missing_files : list[tuple]
        Tuples of (flag, snap, tried_paths) for missing files.
    """
    root = Path(root_dir)
    discovered = _discover_alignment_files(root)

    if requested_flags is None:
        flags0 = sorted({k[0] for k in discovered})
    else:
        flags0 = [str(f) for f in requested_flags]
    flags0 = _ordered_model_flags(flags0, requested_flags=requested_flags)

    if snap_list is None:
        snaps0 = sorted({k[1] for k in discovered})
    else:
        snaps0 = [int(s) for s in snap_list]

    MAset = {}
    missing = []

    for flag in flags0:
        for snap in snaps0:
            path = discovered.get((flag, int(snap)))
            tried = []
            if path is None:
                for cand in _candidate_alignment_files(root, flag, snap):
                    tried.append(str(cand))
                    if cand.exists():
                        path = cand
                        break
            if path is None:
                missing.append((flag, int(snap), tried))
                continue
            try:
                MA = _read_hdf5_flat(path)
            except Exception as exc:
                if strict:
                    raise
                missing.append((flag, int(snap), [str(path), f"READ_ERROR: {exc}"]))
                if verbose:
                    print(f"[skip] {flag} snap={int(snap):03d}: {exc}")
                continue
            MAset.setdefault(flag, {})[_snap_key(snap)] = MA
            if verbose:
                n = len(MA.get("SubhaloID", [])) if "SubhaloID" in MA else "?"
                print(f"[load] {flag:>6s} snap={int(snap):03d}: {path}  N={n}")

    loaded_flags = _ordered_model_flags([f for f, d in MAset.items() if d], requested_flags=requested_flags)
    if strict and missing:
        msg = "Missing requested alignment catalogues:\n" + "\n".join(f"  {f} snap={s:03d}" for f, s, _ in missing)
        raise FileNotFoundError(msg)
    if verbose and missing:
        print(f"[load_alignment_maset] Missing/failed files: {len(missing)}")
    return MAset, loaded_flags, missing


def load_marenew_pickle(path="MArenew.pkl", requested_flags=None, requested_snap_list=None, verbose=True):
    """
    Load MArenew.pkl and normalize it to MAset[flag][snap_key] = MA.

    This helper mirrors the notebook-side loader, so notebooks can either use
    arts.load_marenew_pickle(...) or keep their local normalizer.
    """
    import pickle
    path = Path(path)
    with open(path, "rb") as f:
        raw = pickle.load(f)

    def snap_to_key(s):
        return _snap_key(s)

    def snap_to_int(s):
        return int(_snap_key(s))

    def looks_like_snap_dict(obj):
        return isinstance(obj, dict) and any(_looks_like_ma_catalog(v) for v in obj.values())

    if isinstance(raw, dict) and "MAset" in raw:
        raw = raw["MAset"]

    MAset = {}
    if isinstance(raw, dict) and raw:
        # flag -> snap -> MA
        if any(looks_like_snap_dict(v) for v in raw.values()):
            for flag, sd in raw.items():
                if not isinstance(sd, dict):
                    continue
                for snap, ma in sd.items():
                    if _looks_like_ma_catalog(ma):
                        MAset.setdefault(str(flag), {})[snap_to_key(snap)] = _standardize_ma_keys(ma)
        # snap -> flag -> MA
        elif all(isinstance(v, dict) for v in raw.values()):
            for snap, fd in raw.items():
                if not isinstance(fd, dict):
                    continue
                for flag, ma in fd.items():
                    if _looks_like_ma_catalog(ma):
                        MAset.setdefault(str(flag), {})[snap_to_key(snap)] = _standardize_ma_keys(ma)
        # flat keys such as (flag, snap) or "GR_021"
        if not MAset:
            import re
            for key, ma in raw.items():
                if not _looks_like_ma_catalog(ma):
                    continue
                flag = None
                snap = None
                if isinstance(key, tuple) and len(key) >= 2:
                    flag, snap = key[0], key[1]
                elif isinstance(key, str):
                    m = re.search(r"(?P<flag>GR|F\d+\.?\d*|TNG|TNG300-1).*?(?P<snap>\d{1,3})", key)
                    if m:
                        flag, snap = m.group("flag"), m.group("snap")
                if flag is not None and snap is not None:
                    MAset.setdefault(str(flag), {})[snap_to_key(snap)] = _standardize_ma_keys(ma)

    flags = _ordered_model_flags([f for f, d in MAset.items() if d], requested_flags=requested_flags)
    all_snaps = []
    for f in flags:
        all_snaps.extend([snap_to_int(s) for s in MAset.get(f, {})])
    snaps = _ordered_snapshot_list(all_snaps, requested_snap_list=requested_snap_list)
    if verbose:
        print(f"Loaded MArenew from {path}")
        print("Models:", flags)
        print("Snapshots:", snaps)
    return MAset, flags, snaps

# -----------------------------------------------------------------------------
# Robust data access
# -----------------------------------------------------------------------------


def _as_array(x: Any, dtype=float) -> np.ndarray:
    """Convert HDF5-like or array-like objects to a NumPy array."""
    try:
        if hasattr(x, "__array__"):
            return np.asarray(x, dtype=dtype)
        return np.asarray(x[()], dtype=dtype)
    except Exception:
        return np.asarray(x, dtype=dtype)


def has_field(MA: dict, *candidates: str) -> bool:
    try:
        get_field(MA, candidates)
        return True
    except Exception:
        return False


def get_field(MA: dict, candidates: str | Sequence[str]) -> Any:
    """
    Get a field from a flat or lightly nested MA dictionary.

    Examples of supported aliases:
        "I_Star", "Star/I", ("omega_Star", "Star/omega", "Star/L")
    """
    if isinstance(candidates, str):
        candidates = [candidates]
    for key in candidates:
        if key in MA:
            return MA[key]
        if "/" in key:
            head, tail = key.split("/", 1)
            if head in MA and isinstance(MA[head], dict) and tail in MA[head]:
                return MA[head][tail]
    # Common nested fallbacks.
    for key in candidates:
        if key == "I_Star" and "Star" in MA and isinstance(MA["Star"], dict) and "I" in MA["Star"]:
            return MA["Star"]["I"]
        if key == "I_DM" and "DM" in MA and isinstance(MA["DM"], dict) and "I" in MA["DM"]:
            return MA["DM"]["I"]
    raise KeyError(f"None of the candidate fields exist: {candidates}")


def maybe_field(MA: dict, candidates: str | Sequence[str], default=None) -> Any:
    try:
        return get_field(MA, candidates)
    except Exception:
        return default


def safe_log10(x: Any, min_positive: float | None = None) -> np.ndarray:
    x = _as_array(x, dtype=float)
    y = np.full(x.shape, np.nan, dtype=float)
    m = np.isfinite(x) & (x > 0)
    if min_positive is not None:
        m &= x > min_positive
    y[m] = np.log10(x[m])
    return y


def mask_population(
    MA: dict,
    population: str = "all",
    err_field: str | None = None,
    err_max: float | None = None,
    converged_field: str | None = None,
) -> np.ndarray:
    """Return a boolean population mask."""
    n = _catalog_length(MA)
    sid = maybe_field(MA, "SubhaloID")
    cenid = maybe_field(MA, "CenID")
    if sid is not None and cenid is not None:
        sid = _as_array(sid, dtype=np.int64)
        cenid = _as_array(cenid, dtype=np.int64)
        if population == "central":
            mask = sid == cenid
        elif population == "satellite":
            mask = sid != cenid
        elif population == "all":
            mask = np.ones(n, dtype=bool)
        else:
            raise ValueError(f"Unknown population: {population}")
    else:
        mask = np.ones(n, dtype=bool)

    if err_field is not None and err_max is not None:
        err = maybe_field(MA, err_field)
        if err is not None:
            err = _as_array(err, dtype=float)
            if err.ndim > 1:
                err = np.nanmax(err, axis=tuple(range(1, err.ndim)))
            mask &= np.isfinite(err) & (err <= err_max)

    if converged_field is not None:
        conv = maybe_field(MA, converged_field)
        if conv is not None:
            mask &= _as_array(conv, dtype=float).astype(bool)

    return mask


def _catalog_length(MA: dict) -> int:
    for key in ["SubhaloID", "GroupID", "CenID", "pos_rel", "R", "SubhaloMassInRadType"]:
        val = maybe_field(MA, key)
        if val is not None:
            return len(val)
    for v in MA.values():
        if isinstance(v, dict):
            continue
        try:
            a = np.asarray(v)
            if a.ndim >= 1:
                return len(a)
        except Exception:
            pass
    raise ValueError("Cannot infer catalogue length.")


def apply_range_mask(x: np.ndarray, mask: np.ndarray, xrange: Tuple[float, float] | None) -> np.ndarray:
    if xrange is None:
        return mask
    lo, hi = xrange
    if lo is not None:
        mask &= x >= lo
    if hi is not None:
        mask &= x <= hi
    return mask

# -----------------------------------------------------------------------------
# Linear algebra and alignment estimators
# -----------------------------------------------------------------------------

_AXIS_INDEX = {"major": 0, "medium": 1, "intermediate": 1, "minor": 2}
_AXIS_LABEL = {"major": "major", "medium": "intermediate", "intermediate": "intermediate", "minor": "minor"}


def _normalize_vectors(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    out = np.full_like(v, np.nan, dtype=float)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    m = np.isfinite(norm[..., 0]) & (norm[..., 0] > 0)
    out[m] = v[m] / norm[m]
    return out


def tensor_axes(T: Any) -> np.ndarray:
    """
    Return eigenvectors sorted by descending eigenvalue.

    Output shape is (N, 3, 3), where output[:, 0, :] is the major-axis vector,
    output[:, 1, :] the intermediate-axis vector, and output[:, 2, :] the minor-axis vector.
    """
    T = _as_array(T, dtype=float)
    if T.ndim == 2:
        T = T[None, ...]
    n = T.shape[0]
    axes = np.full((n, 3, 3), np.nan, dtype=float)
    good = np.isfinite(T).all(axis=(1, 2))
    for i in np.where(good)[0]:
        try:
            w, v = np.linalg.eigh(0.5 * (T[i] + T[i].T))
            order = np.argsort(w)[::-1]
            axes[i] = v[:, order].T
        except Exception:
            continue
    return axes


def axis_vector(T: Any, axis: str = "major") -> np.ndarray:
    return tensor_axes(T)[:, _AXIS_INDEX[axis], :]


def abs_cos_vectors(a: Any, b: Any) -> np.ndarray:
    a = _normalize_vectors(_as_array(a, dtype=float))
    b = _normalize_vectors(_as_array(b, dtype=float))
    return np.abs(np.sum(a * b, axis=-1))


def tensor_tensor_alignment(A: Any, B: Any, axis_a: str = "major", axis_b: str = "major") -> np.ndarray:
    return abs_cos_vectors(axis_vector(A, axis_a), axis_vector(B, axis_b))


def vector_tensor_alignment(v: Any, T: Any, axis: str = "major") -> np.ndarray:
    return abs_cos_vectors(v, axis_vector(T, axis))


def vector_vector_alignment(v: Any, w: Any) -> np.ndarray:
    return abs_cos_vectors(v, w)

# -----------------------------------------------------------------------------
# MA field wrappers
# -----------------------------------------------------------------------------


def I_star(MA: dict) -> np.ndarray:
    return _as_array(get_field(MA, ["I_Star", "Star/I"]), dtype=float)


def I_dm(MA: dict) -> np.ndarray:
    return _as_array(get_field(MA, ["I_DM", "DM/I"]), dtype=float)


def tidal_tensor(MA: dict, kind: str) -> np.ndarray:
    """Return one of the three allowed tidal tensor families."""
    if kind == "GR":
        return _as_array(get_field(MA, ["T_GR", "Tidal_GR", "Tidal_tot"]), dtype=float)
    if kind == "group":
        return _as_array(get_field(MA, ["T_grp", "Tidal_grp", "T_group", "Tidal_group"]), dtype=float)
    if kind in {"GRMG", "GR+MG", "total"}:
        return _as_array(get_field(MA, ["T_MG", "Tidal_tot", "T_total", "T_tot"]), dtype=float)
    raise ValueError(f"Unknown tidal tensor kind: {kind}")


def R_vec(MA: dict) -> np.ndarray:
    return _as_array(get_field(MA, ["R", "pos_rel"]), dtype=float)


def V_vec(MA: dict) -> np.ndarray:
    return _as_array(get_field(MA, ["V", "vel_rel"]), dtype=float)


def omega_star_vec(MA: dict) -> np.ndarray:
    # Directional fallback to angular momentum if an angular-velocity field is unavailable.
    return _as_array(get_field(MA, ["omega_Star", "Star/omega", "omega_star", "Star/L", "L_Star"]), dtype=float)


def omega_dm_vec(MA: dict) -> np.ndarray:
    return _as_array(get_field(MA, ["omega_DM", "DM/omega", "omega_dm", "DM/L", "L_DM"]), dtype=float)


def log_stellar_mass(MA: dict) -> np.ndarray:
    if has_field(MA, "SubhaloMassInRadType"):
        m = _as_array(get_field(MA, "SubhaloMassInRadType"), dtype=float)
        if m.ndim == 2 and m.shape[1] > 4:
            return safe_log10(m[:, 4]) + 10.0
    if has_field(MA, ["Star/mass", "Mstar", "stellar_mass"]):
        return safe_log10(get_field(MA, ["Star/mass", "Mstar", "stellar_mass"]))
    raise KeyError("Cannot find stellar mass field.")


def log_subhalo_mass(MA: dict) -> np.ndarray:
    if has_field(MA, "SubhaloMass"):
        return safe_log10(get_field(MA, "SubhaloMass")) + 10.0
    if has_field(MA, "SubhaloMassInRadType"):
        m = _as_array(get_field(MA, "SubhaloMassInRadType"), dtype=float)
        return safe_log10(np.nansum(m, axis=1)) + 10.0
    raise KeyError("Cannot find subhalo mass field.")


def log_halo_mass(MA: dict) -> np.ndarray:
    return safe_log10(get_field(MA, ["Group_M_Crit200", "Group_M_Crit200_Msun", "M200c"])) + (
        10.0 if np.nanmedian(_as_array(get_field(MA, ["Group_M_Crit200", "Group_M_Crit200_Msun", "M200c"]), dtype=float)) < 1e8 else 0.0
    )


def r_over_r200c(MA: dict) -> np.ndarray:
    for key in ["r_over_r200c", "r_over_R200c", "R_over_R200c", "R_R200c"]:
        val = maybe_field(MA, key)
        if val is not None:
            return _as_array(val, dtype=float)
    R = R_vec(MA)
    rr = np.linalg.norm(R, axis=1)
    R200 = _as_array(get_field(MA, ["Group_R_Crit200", "Group_R_Crit200_kpc", "R200c"]), dtype=float)
    return rr / R200


def baryon_dm_ratio(MA: dict, log: bool = True, include_bh: bool = False) -> np.ndarray:
    m = _as_array(get_field(MA, "SubhaloMassInRadType"), dtype=float)
    gas = m[:, 0] if m.ndim == 2 and m.shape[1] > 0 else 0.0
    dm = m[:, 1] if m.ndim == 2 and m.shape[1] > 1 else np.nan
    star = m[:, 4] if m.ndim == 2 and m.shape[1] > 4 else 0.0
    bh = m[:, 5] if include_bh and m.ndim == 2 and m.shape[1] > 5 else 0.0
    ratio = (gas + star + bh) / dm
    ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, np.nan)
    return safe_log10(ratio) if log else ratio

# -----------------------------------------------------------------------------
# Schematic panel
# -----------------------------------------------------------------------------


def visualize_galaxy_system(
    ax=None,
    components=("central", "satellite"),
    sat_sub_vec="",
    cen_vec="",
    misalignment_angle=60,
    size_factor=0.8,
    galaxy_color=None,
    show_dashed_axis=True,
    show_central_black_axis=False,
    title="Galaxy System Misalignment",
):
    """Draw a compact schematic used as the first panel in alignment grids."""
    if galaxy_color is None:
        galaxy_color = clist[6]

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    def draw_axis(pos, angle, length=1.1, color="k", alpha=1.0, zorder=5, dashed=True):
        dx = length * np.cos(np.radians(angle))
        dy = length * np.sin(np.radians(angle))
        arr = FancyArrowPatch(
            posA=(pos[0] - dx * 0.55, pos[1] - dy * 0.55),
            posB=(pos[0] + dx * 0.55, pos[1] + dy * 0.55),
            arrowstyle="-|>",
            mutation_scale=10 * size_factor,
            lw=1.4 * size_factor,
            color=color,
            alpha=alpha,
            zorder=zorder,
        )
        ax.add_patch(arr)
        if dashed and show_dashed_axis:
            ax.plot([pos[0] - dx, pos[0] + dx], [pos[1] - dy, pos[1] + dy], ls="--", lw=0.9, color=color, alpha=0.35 * alpha, zorder=zorder - 1)

    ax.clear()
    ax.set_aspect("equal")
    ax.set_xlim(-4.8, 5.0)
    ax.set_ylim(-3.5, 3.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Host halo.
    halo = Ellipse((0, 0), width=8.4 * size_factor, height=5.8 * size_factor, angle=0,
                   fc="#DDDDDD", ec="0.25", alpha=0.65, lw=1.3, zorder=1)
    ax.add_patch(halo)
    ax.plot([-4.2, 4.2], [0, 0], color="0.3", lw=1.0, alpha=0.35, zorder=2)

    central_pos = (0.0, 0.0)
    central_alpha = 0.92 if "central" in components or "cenvec" in components else 0.20

    # Central host halo and central galaxy.
    ch = Ellipse(central_pos, width=1.85 * size_factor, height=1.10 * size_factor, angle=5,
                 fc="lightgray", ec="gray", alpha=central_alpha, hatch="////", zorder=3)
    ax.add_patch(ch)
    cg = Ellipse(central_pos, width=1.55 * size_factor, height=0.82 * size_factor, angle=20,
                 fc=clist[4], ec=galaxy_color, alpha=central_alpha, hatch="////", zorder=4)
    ax.add_patch(cg)
    if "central" in components or "cenvec" in components:
        draw_axis(central_pos, 20, color=galaxy_color, alpha=0.75, zorder=7)
        draw_axis(central_pos, 5, color="dimgray", alpha=0.65, zorder=6)
        if show_central_black_axis:
            draw_axis(central_pos, 0, color="k", alpha=0.5, zorder=6)
    if "cenvec" in components:
        draw_axis(central_pos, 170, color=clist[4], alpha=0.9, zorder=8, dashed=False)
        ax.text(-1.55 * size_factor, 0.20, cen_vec, fontsize=13, color=clist[4], weight="bold")

    sat_pos = (2.9 * size_factor, 1.9 * size_factor)
    sub_alpha = 0.95 if "subhalo" in components or "subhalo_axis" in components else 0.20
    sat_alpha = 0.92 if "satellite" in components or "satellite_axis" in components else 0.20

    sub = Ellipse(sat_pos, width=1.55 * size_factor, height=0.98 * size_factor, angle=50,
                  fc="lightgray", ec="gray", alpha=sub_alpha, hatch="////", zorder=4)
    ax.add_patch(sub)
    sat_angle = 50 + misalignment_angle
    sat = Ellipse(sat_pos, width=1.05 * size_factor, height=0.45 * size_factor, angle=sat_angle,
                  fc=galaxy_color, ec=galaxy_color, alpha=sat_alpha, zorder=5)
    ax.add_patch(sat)
    if "subhalo_axis" in components or "subhalo" in components:
        draw_axis(sat_pos, 50, length=0.95, color="dimgray", alpha=sub_alpha, zorder=7)
    if "satellite_axis" in components or "satellite" in components:
        draw_axis(sat_pos, sat_angle, length=0.95, color=galaxy_color, alpha=sat_alpha, zorder=8)
    if "position_vector" in components:
        ax.arrow(central_pos[0], central_pos[1], sat_pos[0], sat_pos[1], width=0.018,
                 head_width=0.13, color="#1f77b4", alpha=0.9, length_includes_head=True, zorder=6)
        ax.text(0.48 * sat_pos[0], 0.48 * sat_pos[1] + 0.10, r"$\vec r$", color="#1f77b4", fontsize=15, weight="bold")
    if "subvec" in components:
        draw_axis(sat_pos, 170, color=clist[4], alpha=0.9, zorder=8, dashed=False)
        ax.text(sat_pos[0] - 1.45 * size_factor, sat_pos[1] + 0.05, sat_sub_vec, fontsize=13, color=clist[4], weight="bold")

    ax.text(0.0, -1.05 * size_factor, "Central", ha="center", fontsize=9, color=clist[4], alpha=central_alpha)
    ax.text(sat_pos[0], sat_pos[1] - 1.05 * size_factor, "Satellite", ha="center", fontsize=9, color=galaxy_color, alpha=sat_alpha)
    ax.set_title(title, fontsize=11, pad=4, weight="bold")
    return ax


def _draw_schematic_for_spec(ax, spec: "AlignmentSpec") -> None:
    comps = spec.schematic_components or ("central",)
    visualize_galaxy_system(ax=ax, components=comps, misalignment_angle=35, size_factor=0.75,
                            show_dashed_axis=True, title=spec.schematic_title or spec.title)

# -----------------------------------------------------------------------------
# Alignment specifications
# -----------------------------------------------------------------------------

@dataclass
class AlignmentSpec:
    name: str
    title: str
    chapter: str
    population: str
    mufunc: Callable[[dict], np.ndarray]
    xfunc: Callable[[dict], np.ndarray]
    xlabel: str
    xlim: Tuple[float, float] | None = None
    ylim: Tuple[float, float] | None = (0.0, 1.0)
    bins: int | Sequence[float] = 10
    sample_xrange: Tuple[float, float] | None = None
    err_field: str | None = None
    err_max: float | None = None
    schematic_components: Tuple[str, ...] = ("central",)
    schematic_title: str | None = None
    logx: bool = False


def _x_specs() -> Dict[str, Tuple[Callable[[dict], np.ndarray], str, Tuple[float, float], int, Tuple[float, float] | None]]:
    return {
        "Mstar": (log_stellar_mass, r"$\log_{10} M_\star\,[M_\odot/h]$", (9.5, 12.5), 10, None),
        "Msub": (log_subhalo_mass, r"$\log_{10} M_{\rm sub}\,[M_\odot/h]$", (9.5, 14.0), 10, None),
        "M200c": (log_halo_mass, r"$\log_{10} M_{200c}\,[M_\odot/h]$", (11.0, 15.0), 10, None),
        "R": (r_over_r200c, r"$r/R_{200c}$", (0.01, 3.0), np.logspace(-2, np.log10(3.0), 11), (0.01, 3.0)),
        "BaryonDM": (lambda MA: baryon_dm_ratio(MA, log=True), r"$\log_{10}[(M_{\rm gas}+M_\star)/M_{\rm DM}]$", (-2.0, 1.0), 10, (-2.0, 1.0)),
    }


def _shape_tensor(which: str) -> Callable[[dict], np.ndarray]:
    return I_star if which == "Star" else I_dm


def _shape_label(which: str) -> str:
    return "stellar" if which == "Star" else "DM"


def _make_shape_shape_func(which_a: str, which_b: str, axis: str) -> Callable[[dict], np.ndarray]:
    Ta = _shape_tensor(which_a)
    Tb = _shape_tensor(which_b)
    return lambda MA: tensor_tensor_alignment(Ta(MA), Tb(MA), axis, axis)


def _make_vector_shape_func(vector_name: str, shape: str, axis: str) -> Callable[[dict], np.ndarray]:
    vfunc = {"V": V_vec, "omega_Star": omega_star_vec, "omega_DM": omega_dm_vec}[vector_name]
    Tfunc = _shape_tensor(shape)
    return lambda MA: vector_tensor_alignment(vfunc(MA), Tfunc(MA), axis)


def _make_vector_radial_func(vector_name: str) -> Callable[[dict], np.ndarray]:
    vfunc = {"V": V_vec, "omega_Star": omega_star_vec, "omega_DM": omega_dm_vec}[vector_name]
    return lambda MA: vector_vector_alignment(vfunc(MA), R_vec(MA))


def _make_shape_radial_func(shape: str, axis: str) -> Callable[[dict], np.ndarray]:
    Tfunc = _shape_tensor(shape)
    return lambda MA: vector_tensor_alignment(R_vec(MA), Tfunc(MA), axis)


def _make_tidal_shape_func(tidal_kind: str, shape: str, axis: str) -> Callable[[dict], np.ndarray]:
    Tfunc = _shape_tensor(shape)
    return lambda MA: tensor_tensor_alignment(-tidal_tensor(MA, tidal_kind), Tfunc(MA), "major", axis)


def _make_tidal_radial_func(tidal_kind: str) -> Callable[[dict], np.ndarray]:
    return lambda MA: vector_tensor_alignment(R_vec(MA), -tidal_tensor(MA, tidal_kind), "major")


def build_alignment_specs() -> List[AlignmentSpec]:
    specs: List[AlignmentSpec] = []
    xs = _x_specs()
    axes = ["major", "medium", "minor"]

    # Galaxy--halo / galaxy--subhalo alignment.
    for xname, (xfunc, xlabel, xlim, bins, sample_xrange) in xs.items():
        for axis in axes:
            axlab = _AXIS_LABEL[axis]
            specs.append(AlignmentSpec(
                name=f"CGHA_{xname}_{axis}",
                title=fr"Central galaxy--halo alignment ({axlab}) vs {xname}",
                chapter="galaxy",
                population="central",
                mufunc=_make_shape_shape_func("Star", "DM", axis),
                xfunc=xfunc,
                xlabel=xlabel,
                xlim=xlim,
                bins=bins,
                sample_xrange=sample_xrange,
                err_field=None,
                err_max=None,
                schematic_components=("central",),
                schematic_title="Central galaxy--halo",
            ))
            specs.append(AlignmentSpec(
                name=f"SGHA_{xname}_{axis}",
                title=fr"Satellite galaxy--subhalo alignment ({axlab}) vs {xname}",
                chapter="galaxy",
                population="satellite",
                mufunc=_make_shape_shape_func("Star", "DM", axis),
                xfunc=xfunc,
                xlabel=xlabel,
                xlim=xlim,
                bins=bins,
                sample_xrange=sample_xrange,
                err_field=None,
                err_max=None,
                schematic_components=("satellite", "subhalo", "satellite_axis", "subhalo_axis"),
                schematic_title="Satellite galaxy--subhalo",
            ))

    # Radial alignments.
    for xname, (xfunc, xlabel, xlim, bins, sample_xrange) in xs.items():
        for axis in axes:
            axlab = _AXIS_LABEL[axis]
            specs.append(AlignmentSpec(
                name=f"SatStarRadial_{xname}_{axis}",
                title=fr"Satellite stellar {axlab} axis--radial alignment vs {xname}",
                chapter="radial",
                population="satellite",
                mufunc=_make_shape_radial_func("Star", axis),
                xfunc=xfunc,
                xlabel=xlabel,
                xlim=xlim,
                bins=bins,
                sample_xrange=sample_xrange,
                schematic_components=("position_vector", "satellite", "satellite_axis"),
                schematic_title="Satellite radial alignment",
            ))
            specs.append(AlignmentSpec(
                name=f"SubDMRadial_{xname}_{axis}",
                title=fr"Subhalo DM {axlab} axis--radial alignment vs {xname}",
                chapter="halo",
                population="satellite",
                mufunc=_make_shape_radial_func("DM", axis),
                xfunc=xfunc,
                xlabel=xlabel,
                xlim=xlim,
                bins=bins,
                sample_xrange=sample_xrange,
                schematic_components=("position_vector", "subhalo", "subhalo_axis"),
                schematic_title="Subhalo radial alignment",
            ))
            specs.append(AlignmentSpec(
                name=f"Vel_StarAxis_{xname}_{axis}",
                title=fr"Satellite velocity--stellar {axlab} axis alignment vs {xname}",
                chapter="radial",
                population="satellite",
                mufunc=_make_vector_shape_func("V", "Star", axis),
                xfunc=xfunc,
                xlabel=xlabel,
                xlim=xlim,
                bins=bins,
                sample_xrange=sample_xrange,
                schematic_components=("position_vector", "satellite", "satellite_axis"),
                schematic_title="Velocity--shape alignment",
            ))
            specs.append(AlignmentSpec(
                name=f"OmegaStar_StarAxis_{xname}_{axis}",
                title=fr"Satellite stellar angular-velocity--stellar {axlab} axis alignment vs {xname}",
                chapter="radial",
                population="satellite",
                mufunc=_make_vector_shape_func("omega_Star", "Star", axis),
                xfunc=xfunc,
                xlabel=xlabel,
                xlim=xlim,
                bins=bins,
                sample_xrange=sample_xrange,
                schematic_components=("position_vector", "satellite", "satellite_axis"),
                schematic_title="Angular velocity--shape alignment",
            ))
        specs.append(AlignmentSpec(
            name=f"VelRadial_{xname}",
            title=fr"Satellite velocity--radial alignment vs {xname}",
            chapter="radial",
            population="satellite",
            mufunc=_make_vector_radial_func("V"),
            xfunc=xfunc,
            xlabel=xlabel,
            xlim=xlim,
            bins=bins,
            sample_xrange=sample_xrange,
            schematic_components=("position_vector", "satellite"),
            schematic_title="Velocity radial alignment",
        ))
        specs.append(AlignmentSpec(
            name=f"OmegaStarRadial_{xname}",
            title=fr"Satellite stellar angular-velocity--radial alignment vs {xname}",
            chapter="radial",
            population="satellite",
            mufunc=_make_vector_radial_func("omega_Star"),
            xfunc=xfunc,
            xlabel=xlabel,
            xlim=xlim,
            bins=bins,
            sample_xrange=sample_xrange,
            schematic_components=("position_vector", "satellite"),
            schematic_title="Angular velocity radial alignment",
        ))

    # Tidal alignments.  Only three tensor families are allowed.
    tidal_defs = [
        ("GR", r"$T_{\rm GR}$"),
        ("group", r"$T_{\rm group}$"),
        ("GRMG", r"$T_{\rm GR+MG}$"),
    ]
    for tidal_kind, tidal_label in tidal_defs:
        for xname, (xfunc, xlabel, xlim, bins, sample_xrange) in xs.items():
            specs.append(AlignmentSpec(
                name=f"TidalMajorRadial_{tidal_kind}_{xname}",
                title=fr"{tidal_label} major-axis radial alignment vs {xname}",
                chapter="tidal",
                population="satellite",
                mufunc=_make_tidal_radial_func(tidal_kind),
                xfunc=xfunc,
                xlabel=xlabel,
                xlim=xlim,
                bins=bins,
                sample_xrange=sample_xrange,
                schematic_components=("position_vector", "satellite", "subhalo"),
                schematic_title="Tidal major-axis radial alignment",
            ))
            for shape in ["Star", "DM"]:
                for axis in axes:
                    axlab = _AXIS_LABEL[axis]
                    specs.append(AlignmentSpec(
                        name=f"{shape}Shape_{tidal_kind}Tidal_{xname}_{axis}",
                        title=fr"{_shape_label(shape).capitalize()} {axlab} axis--{tidal_label} alignment vs {xname}",
                        chapter="tidal",
                        population="satellite",
                        mufunc=_make_tidal_shape_func(tidal_kind, shape, axis),
                        xfunc=xfunc,
                        xlabel=xlabel,
                        xlim=xlim,
                        bins=bins,
                        sample_xrange=sample_xrange,
                        schematic_components=("satellite", "subhalo", "satellite_axis", "subhalo_axis"),
                        schematic_title="Shape--tidal alignment",
                    ))
    return specs

ALIGNMENT_SPECS: List[AlignmentSpec] = build_alignment_specs()
ALIGNMENT_SPEC_BY_NAME = {s.name: s for s in ALIGNMENT_SPECS}

# Backward-compatible aliases commonly used in older notebooks.
_ALIAS = {
    "CGHA_Mstar": "CGHA_Mstar_major",
    "CGHA_M200c": "CGHA_M200c_major",
    "CGHA_baryon_dm_ratio": "CGHA_BaryonDM_major",
    "SGHA_Mstar": "SGHA_Mstar_major",
    "SGHA_baryon_dm_ratio": "SGHA_BaryonDM_major",
    "SatRadial_R": "SatStarRadial_R_major",
    "SubRadial_R": "SubDMRadial_R_major",
    "SatStar_GR_tidal_Mstar": "StarShape_GRTidal_Mstar_major",
    "SatStar_Total_tidal_Mstar": "StarShape_GRMGTidal_Mstar_major",
    "CenStar_GR_tidal_M200c": "StarShape_GRTidal_M200c_major",
    "CenStar_Total_tidal_M200c": "StarShape_GRMGTidal_M200c_major",
    "SatStar_GR_tidal_R": "StarShape_GRTidal_R_major",
    "SatStar_Group_tidal_R": "StarShape_groupTidal_R_major",
    "SatStar_Total_tidal_R": "StarShape_GRMGTidal_R_major",
}


def get_alignment_spec_by_name(name: str) -> AlignmentSpec:
    name = _ALIAS.get(name, name)
    if name not in ALIGNMENT_SPEC_BY_NAME:
        raise KeyError(f"Unknown alignment spec {name!r}. Use list_alignment_specs() to inspect names.")
    return ALIGNMENT_SPEC_BY_NAME[name]


def list_alignment_specs(chapter: str | None = None) -> List[str]:
    specs = [s for s in ALIGNMENT_SPECS if chapter is None or s.chapter == chapter]
    for s in specs:
        print(f"{s.name:42s} | {s.chapter:8s} | {s.title}")
    return [s.name for s in specs]


def list_alignment_chapters() -> Dict[str, int]:
    chapters: Dict[str, int] = {}
    for s in ALIGNMENT_SPECS:
        chapters[s.chapter] = chapters.get(s.chapter, 0) + 1
    print("Available alignment chapters:")
    for k in sorted(chapters):
        print(f"  {k:8s}: {chapters[k]} specs")
    return chapters

# -----------------------------------------------------------------------------
# Binning and plotting core
# -----------------------------------------------------------------------------


def _copy_spec_with_overrides(
    spec: AlignmentSpec,
    xlim=None,
    ylim=None,
    sample_xrange=None,
    bins=None,
    err_max=None,
    min_count=None,
) -> AlignmentSpec:
    d = spec.__dict__.copy()
    if xlim is not None:
        d["xlim"] = xlim
    if ylim is not None:
        d["ylim"] = ylim
    if sample_xrange is not None:
        d["sample_xrange"] = sample_xrange
    if bins is not None:
        d["bins"] = bins
    if err_max is not None:
        d["err_max"] = err_max
    if min_count is not None:
        d["min_count"] = min_count
    return AlignmentSpec(**{k: v for k, v in d.items() if k in AlignmentSpec.__dataclass_fields__})


def _get_MA(flag: str, snap: int) -> dict:
    return MASET[flag][_snap_key(snap)]


def get_alignment_arrays(spec: AlignmentSpec, flag: str, snap: int):
    MA = _get_MA(flag, snap)
    x = np.asarray(spec.xfunc(MA), dtype=float)
    mu = np.asarray(spec.mufunc(MA), dtype=float)
    mask = mask_population(MA, population=spec.population, err_field=spec.err_field, err_max=spec.err_max)
    mask &= np.isfinite(x) & np.isfinite(mu)
    mask = apply_range_mask(x, mask, spec.sample_xrange)
    return MA, x, mu, mask


def _bin_edges(x: np.ndarray, bins: int | Sequence[float], xlim: Tuple[float, float] | None = None, logx: bool = False) -> np.ndarray:
    if not np.isscalar(bins):
        return np.asarray(bins, dtype=float)
    nb = int(bins)
    if xlim is not None and xlim[0] is not None and xlim[1] is not None:
        lo, hi = xlim
    else:
        lo, hi = np.nanpercentile(x, [1, 99])
    if logx and lo > 0:
        return np.logspace(np.log10(lo), np.log10(hi), nb + 1)
    return np.linspace(lo, hi, nb + 1)


def binned_profile(x, y, mask=None, bins=10, xlim=None, min_count=3, statistic="mean"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if mask is None:
        mask = np.ones(len(x), dtype=bool)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < min_count:
        return None
    edges = _bin_edges(x, bins, xlim=xlim)
    centers = 0.5 * (edges[:-1] + edges[1:])
    yy = np.full(len(centers), np.nan)
    ee = np.full(len(centers), np.nan)
    nn = np.zeros(len(centers), dtype=int)
    for i in range(len(centers)):
        m = (x >= edges[i]) & (x < edges[i + 1])
        n = int(np.count_nonzero(m))
        nn[i] = n
        if n < min_count:
            continue
        vals = y[m]
        if statistic == "median":
            yy[i] = np.nanmedian(vals)
            ee[i] = 1.253 * np.nanstd(vals) / np.sqrt(n)
        else:
            yy[i] = np.nanmean(vals)
            ee[i] = np.nanstd(vals) / np.sqrt(n)
    return centers, yy, ee, nn


def plot_alignment_on_axis(
    spec: AlignmentSpec,
    ax,
    flag: str,
    snap: int,
    label: str | None = None,
    color: str | None = None,
    alpha: float = 1.0,
    lw: float = 1.8,
    error_style: str = "shade",
    error_alpha: float = 0.30,
    min_count: int = 3,
    verbose: bool = False,
):
    if _snap_key(snap) not in MASET.get(flag, {}):
        return None
    try:
        MA, x, mu, mask = get_alignment_arrays(spec, flag, snap)
        xlim_for_bins = spec.sample_xrange if spec.sample_xrange is not None else spec.xlim
        out = binned_profile(x, mu, mask=mask, bins=spec.bins, xlim=xlim_for_bins, min_count=min_count)
    except Exception as exc:
        if verbose:
            print(f"[skip] {spec.name} {flag} snap={snap:03d}: {exc}")
        return None
    if out is None:
        if verbose:
            print(f"[skip] {spec.name} {flag} snap={snap:03d}: too few valid objects")
        return None
    xc, yy, ee, nn = out
    good = np.isfinite(yy)
    if not np.any(good):
        return None
    c = color if color is not None else flag_color(flag)
    line, = ax.plot(xc[good], yy[good], color=c, lw=lw, alpha=alpha, label=label)
    if error_style == "shade":
        egood = good & np.isfinite(ee)
        ax.fill_between(xc[egood], yy[egood] - ee[egood], yy[egood] + ee[egood], color=c, alpha=error_alpha, lw=0)
    elif error_style == "errorbar":
        egood = good & np.isfinite(ee)
        ax.errorbar(xc[egood], yy[egood], yerr=ee[egood], fmt="none", ecolor=c, alpha=0.55, lw=0.8, capsize=1.5)
    return line


def apply_alignment_axis_format(ax, spec: AlignmentSpec) -> None:
    if spec.xlim is not None:
        ax.set_xlim(*spec.xlim)
    if spec.ylim is not None:
        ax.set_ylim(*spec.ylim)
    if spec.logx:
        ax.set_xscale("log")
    ax.set_xlabel(spec.xlabel)
    ax.set_ylabel(r"$\langle |\cos\theta| \rangle$")
    ax.grid(alpha=0.18)


def _model_legend_handles(flags_to_use):
    return [Line2D([0], [0], color=flag_color(f), lw=2.2, label=flag_label(f)) for f in flags_to_use]


def _draw_model_legend_in_axis(ax, flags_to_use):
    ax.axis("off")
    ax.legend(handles=_model_legend_handles(flags_to_use), loc="center", frameon=False, title="Model",
              fontsize=10, title_fontsize=11, handlelength=2.6)


def _save_figure(fig, folder: str, filename: str, save: bool = True):
    if not save:
        return None
    outdir = Path(PLOT_ROOT) / folder
    outdir.mkdir(parents=True, exist_ok=True)
    fout = outdir / filename
    fig.savefig(fout, dpi=220, bbox_inches="tight")
    print("Saved:", fout)
    return fout


def plot_alignment_snapshot_grid(
    spec: AlignmentSpec | str,
    snap_list: Sequence[int] | None = None,
    flags_to_use: Sequence[str] | None = None,
    save: bool = True,
    show: bool = True,
    output_root: str | os.PathLike | None = None,
    xlim=None,
    ylim=None,
    sample_xrange=None,
    bins=None,
    err_max=None,
    error_style: str = "shade",
    error_alpha: float = 0.30,
    min_count: int = 3,
):
    if output_root is not None:
        set_plot_output_root(output_root)
    if isinstance(spec, str):
        spec = get_alignment_spec_by_name(spec)
    spec = _copy_spec_with_overrides(spec, xlim=xlim, ylim=ylim, sample_xrange=sample_xrange, bins=bins, err_max=err_max)

    set_paper_style()
    snap_list = list(SNAP_LIST if snap_list is None else snap_list)
    flags_to_use = list(FLAGS if flags_to_use is None else flags_to_use)
    flags_to_use = [f for f in flags_to_use if f in FLAGS]

    # If there are <= 8 snapshots, reserve the upper-left panel for the schematic.
    use_schematic = len(snap_list) <= 8
    fig, axes = plt.subplots(3, 3, figsize=(18, 13), sharey=False)
    axes = axes.ravel()
    if use_schematic:
        _draw_schematic_for_spec(axes[0], spec)
        data_axes = axes[1:1 + len(snap_list)]
    else:
        data_axes = axes[:len(snap_list)]

    for ax, snap in zip(data_axes, snap_list):
        active = [f for f in flags_to_use if _snap_key(snap) in MASET.get(f, {})]
        for flag in active:
            plot_alignment_on_axis(
                spec, ax, flag, snap, label=flag_label(flag), color=flag_color(flag),
                alpha=0.95, lw=1.9, error_style=error_style, error_alpha=error_alpha,
                min_count=min_count,
            )
        z = ZMAP.get(int(snap), ZMAP_ALL.get(int(snap), np.nan))
        ax.set_title(rf"$z={z:.2f}$  (snap={int(snap):03d})")
        apply_alignment_axis_format(ax, spec)

    first_empty = (1 + len(snap_list)) if use_schematic else len(snap_list)
    if first_empty < len(axes):
        _draw_model_legend_in_axis(axes[first_empty], flags_to_use)
        for ax in axes[first_empty + 1:]:
            ax.axis("off")
    else:
        fig.legend(handles=_model_legend_handles(flags_to_use), loc="upper center", ncol=min(len(flags_to_use), 7), frameon=False,
                   bbox_to_anchor=(0.5, 0.985))

    fig.suptitle(spec.title, fontsize=17, weight="bold", y=0.985)
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.94), w_pad=2.0, h_pad=2.1)
    _save_figure(fig, "alignment_snapshot_grids", f"{spec.name}_snapshot_grid.png", save=save)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_alignment_redshift_evolution(
    spec: AlignmentSpec | str,
    snap_list: Sequence[int] | None = None,
    flags_to_use: Sequence[str] | None = None,
    save: bool = True,
    show: bool = True,
    output_root: str | os.PathLike | None = None,
    xlim=None,
    ylim=None,
    sample_xrange=None,
    bins=None,
    err_max=None,
    error_style: str = "shade",
    error_alpha: float = 0.30,
    cmap_name: str = "turbo",
    min_count: int = 3,
):
    if output_root is not None:
        set_plot_output_root(output_root)
    if isinstance(spec, str):
        spec = get_alignment_spec_by_name(spec)
    spec = _copy_spec_with_overrides(spec, xlim=xlim, ylim=ylim, sample_xrange=sample_xrange, bins=bins, err_max=err_max)

    set_paper_style()
    snap_list = list(SNAP_LIST if snap_list is None else snap_list)
    flags_to_use = [f for f in (FLAGS if flags_to_use is None else flags_to_use) if f in FLAGS]
    n_model = len(flags_to_use)
    n_panel = 1 + n_model + 1
    ncols = 3 if n_panel <= 9 else 4
    nrows = int(np.ceil(n_panel / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.1 * nrows), sharex=False, sharey=False)
    axes = np.atleast_1d(axes).ravel()
    _draw_schematic_for_spec(axes[0], spec)

    zvals = np.array([ZMAP.get(int(s), ZMAP_ALL.get(int(s), np.nan)) for s in snap_list], dtype=float)
    finite_z = zvals[np.isfinite(zvals)]
    if len(finite_z) == 0:
        finite_z = np.arange(len(snap_list))
        zvals = finite_z
    norm = plt.Normalize(vmin=np.nanmin(finite_z), vmax=np.nanmax(finite_z))
    cmap = plt.get_cmap(cmap_name)

    for ax, flag in zip(axes[1:1 + n_model], flags_to_use):
        for snap, z in zip(snap_list, zvals):
            if _snap_key(snap) not in MASET.get(flag, {}):
                continue
            plot_alignment_on_axis(
                spec, ax, flag, snap, label=rf"$z={z:.2f}$", color=cmap(norm(z)),
                alpha=0.92, lw=1.7, error_style=error_style, error_alpha=error_alpha,
                min_count=min_count,
            )
        ax.set_title(flag_label(flag), color=flag_color(flag), weight="bold")
        apply_alignment_axis_format(ax, spec)

    cbar_index = 1 + n_model
    if cbar_index < len(axes):
        cax_panel = axes[cbar_index]
        cax_panel.axis("off")
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cax = cax_panel.inset_axes([0.47, 0.14, 0.065, 0.72])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Redshift", labelpad=8)
        cbar.ax.tick_params(labelsize=10)
        for ax in axes[cbar_index + 1:]:
            ax.axis("off")

    line_handles = [Line2D([0], [0], color="0.2", lw=2.0, label=r"redshift curves")]
    fig.legend(handles=line_handles, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 0.982))
    fig.suptitle(spec.title + " — redshift evolution", fontsize=17, weight="bold", y=0.955)
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.91), w_pad=2.0, h_pad=2.1)
    _save_figure(fig, "alignment_redshift_evolution", f"{spec.name}_redshift_evolution.png", save=save)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_alignment(
    name: str,
    mode: str = "both",
    snap_list: Sequence[int] | None = None,
    flags_to_use: Sequence[str] | None = None,
    save: bool = True,
    show: bool = True,
    output_root: str | os.PathLike | None = None,
    xlim=None,
    ylim=None,
    sample_xrange=None,
    bins=None,
    err_max=None,
    error_style: str = "shade",
    error_alpha: float = 0.30,
    cmap_name: str = "turbo",
    min_count: int = 3,
):
    spec = get_alignment_spec_by_name(name)
    out = []
    if mode in {"snapshot", "both"}:
        out.append(plot_alignment_snapshot_grid(spec, snap_list=snap_list, flags_to_use=flags_to_use, save=save, show=show,
                                                output_root=output_root, xlim=xlim, ylim=ylim, sample_xrange=sample_xrange,
                                                bins=bins, err_max=err_max, error_style=error_style, error_alpha=error_alpha,
                                                min_count=min_count))
    if mode in {"redshift", "both"}:
        out.append(plot_alignment_redshift_evolution(spec, snap_list=snap_list, flags_to_use=flags_to_use, save=save, show=show,
                                                     output_root=output_root, xlim=xlim, ylim=ylim, sample_xrange=sample_xrange,
                                                     bins=bins, err_max=err_max, error_style=error_style, error_alpha=error_alpha,
                                                     cmap_name=cmap_name, min_count=min_count))
    if mode not in {"snapshot", "redshift", "both"}:
        raise ValueError("mode must be 'snapshot', 'redshift', or 'both'.")
    return out


def plot_alignment_pair(name: str, **kwargs):
    return plot_alignment(name, mode="both", **kwargs)


def plot_alignment_chapter(
    chapter: str,
    shape_axes: Sequence[str] = ("major", "medium", "minor"),
    mode: str = "both",
    snap_list: Sequence[int] | None = None,
    flags_to_use: Sequence[str] | None = None,
    save: bool = True,
    show: bool = True,
    output_root: str | os.PathLike | None = None,
    error_style: str = "shade",
    error_alpha: float = 0.30,
    cmap_name: str = "turbo",
    continue_on_error: bool = True,
):
    """Plot all specs in one chapter, optionally limiting to selected shape axes."""
    allowed_axes = set(shape_axes)
    specs = []
    for spec in ALIGNMENT_SPECS:
        if spec.chapter != chapter:
            continue
        # Most names end with _major/_medium/_minor.  Keep non-axis specs too.
        parts = spec.name.split("_")
        if parts[-1] in _AXIS_INDEX and parts[-1] not in allowed_axes:
            continue
        specs.append(spec)

    failed = []
    for i, spec in enumerate(specs, 1):
        print(f"[{i}/{len(specs)}] {spec.name}")
        try:
            plot_alignment(
                spec.name,
                mode=mode,
                snap_list=snap_list,
                flags_to_use=flags_to_use,
                save=save,
                show=show,
                output_root=output_root,
                error_style=error_style,
                error_alpha=error_alpha,
                cmap_name=cmap_name,
            )
        except Exception as exc:
            failed.append((spec.name, str(exc)))
            print(f"[FAILED] {spec.name}: {exc}")
            if not continue_on_error:
                raise
    return failed

# -----------------------------------------------------------------------------
# Physical distributions: minimal compatible implementation
# -----------------------------------------------------------------------------

@dataclass
class PhysicalSpec:
    name: str
    title: str
    value_func: Callable[[dict], np.ndarray]
    xlabel: str
    xlim: Tuple[float, float] | None = None
    bins: int = 50

PHYSICAL_DISTRIBUTION_SPECS = [
    PhysicalSpec("Mstar_distribution", "Stellar mass distribution", log_stellar_mass, r"$\log_{10}M_\star$", (9.5, 12.5), 50),
    PhysicalSpec("Msub_distribution", "Subhalo mass distribution", log_subhalo_mass, r"$\log_{10}M_{\rm sub}$", (9.5, 14.0), 50),
    PhysicalSpec("M200c_distribution", "Host halo mass distribution", log_halo_mass, r"$\log_{10}M_{200c}$", (11.0, 15.0), 50),
    PhysicalSpec("baryon_dm_ratio_distribution", "Baryon-to-DM ratio distribution", lambda MA: baryon_dm_ratio(MA, log=True), r"$\log_{10}[(M_{\rm gas}+M_\star)/M_{\rm DM}]$", (-2, 1), 50),
]
PHYSICAL_SPEC_BY_NAME = {s.name: s for s in PHYSICAL_DISTRIBUTION_SPECS}


def plot_physical(name: str, snap_list=None, flags_to_use=None, save=True, show=True, output_root=None, xlim=None, ylim=None, bins=None, **kwargs):
    if output_root is not None:
        set_plot_output_root(output_root)
    spec = PHYSICAL_SPEC_BY_NAME[name]
    set_paper_style()
    snap_list = list(SNAP_LIST if snap_list is None else snap_list)
    flags_to_use = [f for f in (FLAGS if flags_to_use is None else flags_to_use) if f in FLAGS]
    fig, axes = plt.subplots(3, 3, figsize=(18, 13), sharey=False)
    axes = axes.ravel()
    axes[0].axis("off")
    axes[0].text(0.05, 0.82, spec.title, fontsize=16, weight="bold", transform=axes[0].transAxes)
    for ax, snap in zip(axes[1:], snap_list):
        for flag in flags_to_use:
            if _snap_key(snap) not in MASET.get(flag, {}):
                continue
            try:
                x = np.asarray(spec.value_func(_get_MA(flag, snap)), dtype=float)
                x = x[np.isfinite(x)]
                if len(x) == 0:
                    continue
                ax.hist(x, bins=bins or spec.bins, histtype="step", density=True, color=flag_color(flag), lw=1.5, label=flag_label(flag))
            except Exception:
                continue
        z = ZMAP.get(int(snap), ZMAP_ALL.get(int(snap), np.nan))
        ax.set_title(rf"$z={z:.2f}$  (snap={int(snap):03d})")
        ax.set_xlabel(spec.xlabel)
        ax.set_ylabel("PDF")
        ax.grid(alpha=0.18)
        ax.set_xlim(*(xlim or spec.xlim)) if (xlim or spec.xlim) is not None else None
        if ylim is not None:
            ax.set_ylim(*ylim)
    if len(snap_list) < 8:
        for ax in axes[1 + len(snap_list):]:
            ax.axis("off")
    fig.legend(handles=_model_legend_handles(flags_to_use), loc="upper center", ncol=min(len(flags_to_use), 7), frameon=False)
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.94))
    _save_figure(fig, "physical_distributions", f"{name}.png", save=save)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig

# -----------------------------------------------------------------------------
# Optional shape-cluster placeholder
# -----------------------------------------------------------------------------


def plot_shape_cluster_alignment_suite(
    MA: dict,
    mask=None,
    n_clusters: int = 30,
    cmap: str = "bwr",
    output_root=None,
    prefix: str = "shape_cluster",
    show: bool = True,
    save: bool = True,
):
    """
    Minimal compatible helper for notebooks that optionally call this function.

    It plots the stellar q-s plane for a selected sample.  This is intentionally
    lightweight; the full clustering analysis can be added downstream if needed.
    """
    if output_root is not None:
        set_plot_output_root(output_root)
    set_paper_style()
    try:
        I = I_star(MA)
        w = np.linalg.eigvalsh(I)
        w = np.sort(w, axis=1)[:, ::-1]
        q = np.sqrt(np.clip(w[:, 1] / w[:, 0], 0, np.inf))
        s = np.sqrt(np.clip(w[:, 2] / w[:, 0], 0, np.inf))
    except Exception:
        raise RuntimeError("Cannot compute q-s plane from I_Star/Star/I.")
    if mask is None:
        mask = np.isfinite(q) & np.isfinite(s)
    else:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(q) & np.isfinite(s)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(q[mask], s[mask], s=4, alpha=0.35, color="0.2")
    ax.set_xlabel(r"$q_\star$")
    ax.set_ylabel(r"$s_\star$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.18)
    ax.set_title("Stellar shape axis-ratio plane")
    _save_figure(fig, "shape_cluster_alignment", f"{prefix}_qs_plane.png", save=save)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return {"q": q, "s": s, "mask": mask}

# Ensure default directories exist on import.
_ensure_plot_directories(PLOT_ROOT)
