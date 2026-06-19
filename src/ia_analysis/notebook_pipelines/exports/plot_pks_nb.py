"""Exported code from notebooks/raw_20260618/plot_pks.ipynb.

This file is generated for project management and refactoring. Review before running end to end.
"""


# %% [markdown] cell 1
# # Plot all available $P(k)$ spectra This notebook scans all `pks_{FLAG}_{SNAP}.hdf5` files, discovers every available `P_*` spectrum in each selected sample, reads existing covariance products if they are already present, and produces publication-style figures. Main outputs: 1. **For each galaxy-dependent spectrum:** a 3×3 redshift grid comparing gravity models at fixed redshift, grouped by galaxy sample. 2. **For snapshot-particle spectra:** one set of figures only, without galaxy-sample labels

# %% code cell 2

from pathlib import Path
import re
import sys
import warnings
import importlib
from arts import *
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -------------------------------------------------
# User configuration
# -------------------------------------------------
DATA_DIR = Path("../pks")
MODULE_DIR = Path("./")     # directory containing Cov.py
OUTPUT_DIR = Path("figures_all_pk")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# If None, scan all matching files in DATA_DIR.
# Otherwise use explicit lists below.
FLAGS = ["GR", "F40", "F45", "F50", "F55", "F60"]
SNAPS = [1, 3, 6, 8, 10, 12, 15, 18, 21]

# Which samples to plot.
# None means all HDF5 sample groups that contain SOURCE_GROUP.
# For a compact first run, set SAMPLES = ["all"].
SAMPLES = ["LRG"]

# Which spectra to plot.
# None means every P_* spectrum discovered in the selected sample(s).
SPECTRA = ["P_gg", "P_dg", "P_dE", "P_gE", "P_EE", "P_dd", "P_dtp", "P_tptp", "P_tpE"]

SOURCE_GROUP = "stitched_corr"   # "stitched_corr", "stitched_raw", or "stitched_noise"
NOISE_GROUP = "stitched_noise"
PLOT_K_TIMES_PK = True

# Spectra built only from these fields are snapshot-particle spectra.
# They are stored under sample groups in the HDF5 file for convenience, but
# physically they do not depend on the galaxy sample, so they are plotted once.
SAMPLE_INDEPENDENT_FIELDS = {"d", "tp"}
SNAPSHOT_SPECTRA_LABEL = "Snapshot particles"
SNAPSHOT_SPECTRA_OUTPUT = "snapshot_particles"

# -------------------------------------------------
# Theoretical covariance / uncertainty
# -------------------------------------------------
# This notebook reads existing sample/covariance groups if present.
# If covariance is missing, it can compute covariance in memory from Cov.py.
# It never opens HDF5 files in write mode and never writes covariance back.
USE_THEORY_COVARIANCE = True
READ_EXISTING_COVARIANCE_ONLY = False
COMPUTE_COVARIANCE_IN_MEMORY_IF_MISSING = True
FORCE_RECOMPUTE_COVARIANCE = False
COV_GROUP = "covariance"
COV_KIND = "gaussian"               # "total", "gaussian", "cNG", or "SSC"
COV_SOURCE = "measured_or_ccl_nla"  # retained for compatibility; not used by this read-only notebook
COV_AIA_DEFAULT = 1.0
COV_C1_RHOCRIT = 0.0134
COV_KMIN_FIT = 0.05
COV_KMAX_FIT = 0.30
COV_INCLUDE_CNG = False
COV_INCLUDE_SSC = False
COV_SSC_MODE = "periodic"        # "periodic" is appropriate for a full fixed-background box
COV_SSC_BACKEND = "ccl_halomodel"
COV_HM_OPTIONS = None

# Fallback cosmology and box metadata used if they are absent from the HDF5 attrs.
COSMO_DEFAULTS = dict(
    Omega_c=0.2589,
    Omega_b=0.0486,
    h=0.6774,
    sigma8=0.8159,
    n_s=0.9667,
)
BOXSIZE_DEFAULT = 205          # Mpc/h for TNG300-like boxes
NMESH_DEFAULT = 512

# How to draw uncertainty.
SHOW_UNCERTAINTY = True
UNCERTAINTY_STYLE = "shade"        # "errorbar", "shade", or "both"
UNCERTAINTY_ALPHA = 0.18
ERRORBAR_EVERY = 1

# Display and saving controls.
SHOW_FIGURES = True
SAVE_FIGURES = False
SAVE_FORMATS = ( "pdf")#"png",
DPI = 280

# Log plotting convention.
# If True, signed spectra are shown as abs(kP); negative points use open markers.
USE_LOG_ABS_FOR_SIGNED = True

# Figure layout.
N_COLS_GRID = 3
FIGSIZE_GRID = (15.4, 12.2)
FIGSIZE_MODEL_EVOLUTION = (17.2, 9.2)

# Redshift color map for same-model redshift evolution.
REDSHIFT_CMAP = "turbo"

# Optional: resample the plotted curves to a fixed target-k grid.
# This affects plotting only and never changes the HDF5 measurements.
USE_TARGET_K_FOR_PLOTTING = True
TARGET_K_MIN = 0.3
TARGET_K_MAX = 20.0
TARGET_NK = 15
TARGET_K = np.logspace(np.log10(TARGET_K_MIN), np.log10(TARGET_K_MAX), TARGET_NK)

# Optional: restrict k plotting range. Use None for automatic.
XLIM = (TARGET_K_MIN, TARGET_K_MAX)
YLIM = None

# Optional enhancement plots relative to GR.
MAKE_ENHANCEMENT_RELATIVE_TO_GR = True

# Optional switches.
RUN_PK_PLOTS = True
RUN_AIA_PLOTS = True

# A_IA estimator settings.
AIA_C1_RHOCRIT = 0.0134
AIA_USE_CCL_GROWTH = True
AIA_FIT_KMIN = 0.2
AIA_FIT_KMAX = 0.35
AIA_METHODS_TO_PLOT = ("deltaE", "gE")

# f(R) color convention.
# FR_COLORS = {
#     "F4":   "#ff0000",
#     "F4.5": "#ff8c00",
#     "F5":   "#008000",
#     "F5.5": "#ff00ff",
#     "F6":   "#0000ff",
#     "GR":   "#000000",
# }

FR_COLORS = {
    "F4":   '#b84c4c', 
    "F4.5": '#c08a4a', 
    "F5":   '#a5ba9e', 
    "F5.5": '#c5aade', 
    "F6":   '#5f75a8', 
    "GR":   '#2b2b2b'
}

FLAG_LABEL = {
    "GR": "GR",
    "F40": "F4",
    "F45": "F4.5",
    "F50": "F5",
    "F55": "F5.5",
    "F60": "F6",
}

FLAG_COLOR = {flag: FR_COLORS.get(label, "0.4") for flag, label in FLAG_LABEL.items()}

# Snapshot-redshift mapping from ia_pk_cs.py.
ZMAP = {
    0: 3.00, 1: 2.00, 2: 1.36, 3: 1.26, 4: 1.15, 5: 1.06,
    6: 0.97, 7: 0.88, 8: 0.80, 9: 0.73, 10: 0.65, 11: 0.58,
    12: 0.51, 13: 0.45, 14: 0.39, 15: 0.33, 16: 0.27, 17: 0.21,
    18: 0.16, 19: 0.10, 20: 0.05, 21: 0.00,
}

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

# Import Cov.py from MODULE_DIR.
sys.path.insert(0, str(MODULE_DIR))
try:
    import Cov
    importlib.reload(Cov)
except Exception as exc:
    Cov = None
    warnings.warn(f"Could not import Cov.py from {MODULE_DIR}: {exc}")

# %% code cell 3
# Target k grid used for plotting.
# Units: h Mpc^{-1}
print("USE_TARGET_K_FOR_PLOTTING =", USE_TARGET_K_FOR_PLOTTING)
print("TARGET_K =", np.array2string(TARGET_K, precision=4, separator=", "))

# %% code cell 4
FR_COLORS.values()

# %% code cell 5
FLAG_COLOR

# %% code cell 6

# %% code cell 7
# get_colors(["#ff0000",
# "#ff8c00",
# "#008000",
# "#ff00ff",
# "#0000ff",
# "#000000"])

# %% code cell 8
get_colors(list(FR_COLORS.values()))

# %% code cell 9
FR_COLORS

# %% code cell 10
FR_COLORS.values

# %% code cell 11

# -------------------------------------------------
# Discovery and HDF5 reading helpers
# -------------------------------------------------
def snap3(snap):
    return f"{int(snap):03d}"


def pks_path(flag, snap):
    return DATA_DIR / f"pks_{flag}_{snap3(snap)}.hdf5"


def parse_flag_snap(path):
    m = re.match(r"pks_(.+)_(\d{3})\.hdf5$", Path(path).name)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def discover_files(flags=FLAGS, snaps=SNAPS):
    rows = []
    if flags is None or snaps is None:
        for path in sorted(DATA_DIR.glob("pks_*.hdf5")):
            flag, snap = parse_flag_snap(path)
            if flag is None:
                continue
            rows.append({"flag": flag, "snap": snap, "path": path})
    else:
        for flag in flags:
            for snap in snaps:
                path = pks_path(flag, snap)
                if path.exists():
                    rows.append({"flag": flag, "snap": int(snap), "path": path})
                else:
                    print(f"[missing] {path}")
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise FileNotFoundError(f"No pks_*.hdf5 files found in {DATA_DIR}")
    df["z"] = df["snap"].map(ZMAP).astype(float)
    df["flag_label"] = df["flag"].map(lambda x: FLAG_LABEL.get(x, x))
    return df.sort_values(["snap", "flag"]).reset_index(drop=True)


def is_sample_group(h5, key, source_group=SOURCE_GROUP):
    if key.startswith("stitched") or key in {"summary", "target_k"}:
        return False
    obj = h5.get(key, None)
    return isinstance(obj, h5py.Group) and source_group in obj


def list_samples(path, source_group=SOURCE_GROUP):
    with h5py.File(path, "r") as f:
        out = []
        for key in f.keys():
            if is_sample_group(f, key, source_group=source_group):
                out.append(key)
        return sorted(out)


def list_spectra(path, sample, source_group=SOURCE_GROUP):
    with h5py.File(path, "r") as f:
        if sample not in f or source_group not in f[sample]:
            return []
        g = f[sample][source_group]
        out = []
        for key in g.keys():
            if key.startswith("P_") and key.endswith("_Pk"):
                out.append(key[:-3])  # remove "_Pk"
        return sorted(set(out))


def discover_samples_and_spectra(files_df, requested_samples=SAMPLES, requested_spectra=SPECTRA):
    all_samples = set()
    all_spectra_by_sample = {}
    for path in files_df["path"]:
        for sample in list_samples(path):
            all_samples.add(sample)

    if requested_samples is None:
        samples = sorted(all_samples)
    else:
        samples = list(requested_samples)

    for sample in samples:
        specs = set()
        for path in files_df["path"]:
            for sp in list_spectra(path, sample):
                specs.add(sp)
        if requested_spectra is None:
            all_spectra_by_sample[sample] = sorted(specs)
        else:
            req = list(requested_spectra)
            all_spectra_by_sample[sample] = [s for s in req if s in specs]

    return samples, all_spectra_by_sample


def read_spectrum(path, sample, spectrum, source_group=SOURCE_GROUP):
    """Return k and P array for one spectrum key such as P_dE."""
    with h5py.File(path, "r") as f:
        g = f[sample][source_group]
        k = np.asarray(g["k"], dtype=float)
        pkey = spectrum + "_Pk"
        if pkey not in g:
            raise KeyError(f"{pkey} not in {path}:{sample}/{source_group}")
        P = np.asarray(g[pkey], dtype=float)
        z = float(f.attrs.get("z", np.nan))
    return k, P, z


files_df = discover_files()
samples, spectra_by_sample = discover_samples_and_spectra(files_df)

print(f"Discovered {len(files_df)} files.")
print("Samples:")
for s in samples:
    print(f"  {s}: {len(spectra_by_sample.get(s, []))} spectra")
files_df.head()

# %% code cell 12

# -------------------------------------------------
# Label, covariance and plotting helpers
# -------------------------------------------------
_FIELD_LABELS = {
    "d": r"\delta",
    "g": r"g",
    "E": r"E",
    "B": r"B",
    "t": r"\theta_{\mathrm{g}}",
    "tp": r"\theta_{\mathrm{p}}",
}

_KNOWN_FIELDS = ["tp", "d", "g", "E", "B", "t"]
_COV_FAILURE_CACHE = set()


def split_spectrum_fields(spectrum):
    """Convert P_dE -> ("d", "E"), P_tpt -> ("tp", "t"), etc."""
    s = str(spectrum)
    if s.startswith("P_"):
        s = s[2:]

    # Greedy parse known field names.
    for a in sorted(_KNOWN_FIELDS, key=len, reverse=True):
        if s.startswith(a):
            b = s[len(a):]
            if b in _KNOWN_FIELDS:
                return a, b

    # Fallback for simple two-character labels.
    if len(s) == 2:
        return s[0], s[1]
    mid = len(s) // 2
    return s[:mid], s[mid:]


def pk_key_for_cov(spectrum):
    """Convert plot key P_dE -> Cov.py key dE."""
    s = str(spectrum)
    return s[2:] if s.startswith("P_") else s


def is_sample_independent_spectrum(spectrum):
    """True for spectra that are built only from snapshot-particle fields.

    Examples: P_dd, P_dtp, P_tptp.  These should not be labelled by LRG/ELG/etc.
    """
    a, b = split_spectrum_fields(spectrum)
    return (a in SAMPLE_INDEPENDENT_FIELDS) and (b in SAMPLE_INDEPENDENT_FIELDS)


def display_sample_label(sample, spectrum):
    """Label used in figure titles."""
    if is_sample_independent_spectrum(spectrum):
        return SNAPSHOT_SPECTRA_LABEL
    return str(sample)


def output_sample_label(sample, spectrum):
    """Directory/file token used for saved figures."""
    if is_sample_independent_spectrum(spectrum):
        return SNAPSHOT_SPECTRA_OUTPUT
    return output_safe_name(sample)


def spectrum_math_label(spectrum, include_k=True):
    r"""Return a mathtext-safe spectrum label, e.g. P_{\delta\,E}(k).

    The explicit thin space prevents mathtext from reading "\deltaE" or
    "\deltaB" as one unknown command.
    """
    a, b = split_spectrum_fields(spectrum)
    la = _FIELD_LABELS.get(a, a)
    lb = _FIELD_LABELS.get(b, b)
    base = rf"P_{{{la}\,{lb}}}(k)"
    if include_k and PLOT_K_TIMES_PK:
        return rf"k{base}"
    return base

def format_z_title(z):
    if not np.isfinite(z):
        return r"$\mathrm{z}$ unknown"
    if abs(z) < 5e-3:
        return r"$\mathrm{z}=0$"
    return rf"$\mathrm{{z}}={z:.2f}$"


def output_safe_name(x):
    x = str(x)
    x = x.replace("/", "_").replace(" ", "_").replace("$", "")
    x = x.replace("\\", "").replace("{", "").replace("}", "")
    x = re.sub(r"[^A-Za-z0-9_.+-]+", "_", x)
    return re.sub(r"_+", "_", x).strip("_")


def prepare_y(k, P, sigma=None):
    y = k * P if PLOT_K_TIMES_PK else P.copy()
    if sigma is None:
        sy = None
    else:
        sy = k * sigma if PLOT_K_TIMES_PK else sigma.copy()

    neg = y < 0
    if USE_LOG_ABS_FOR_SIGNED:
        yplot = np.abs(y)
    else:
        yplot = y

    # Only k and P decide whether the curve itself is plotted.
    # Missing covariance/errorbar values should not remove valid P(k) points.
    good = np.isfinite(k) & np.isfinite(yplot) & (yplot > 0)
    return y, yplot, sy, neg, good


def resample_to_target_k(k, P, sigma=None, target_k=None):
    """Interpolate a spectrum onto the fixed plotting grid.

    Interpolation is linear in log(k) but linear in P.  This is safer than
    log-log interpolation for signed spectra such as P_dE and P_gE.
    """
    if target_k is None:
        target_k = TARGET_K

    k = np.asarray(k, dtype=float)
    P = np.asarray(P, dtype=float)
    kt = np.asarray(target_k, dtype=float)

    good = np.isfinite(k) & np.isfinite(P) & (k > 0)
    if np.count_nonzero(good) < 2:
        if sigma is None:
            return k, P, None
        return k, P, np.asarray(sigma, dtype=float)

    kk = k[good]
    pp = P[good]
    order = np.argsort(kk)
    kk = kk[order]
    pp = pp[order]

    # Keep only target-k values covered by this measured spectrum.
    kt = kt[(kt >= kk.min()) & (kt <= kk.max())]
    if kt.size == 0:
        return np.array([]), np.array([]), None if sigma is None else np.array([])

    Pt = np.interp(np.log(kt), np.log(kk), pp)

    if sigma is None:
        return kt, Pt, None

    sigma = np.asarray(sigma, dtype=float)
    good_s = np.isfinite(k) & np.isfinite(sigma) & (k > 0)
    if np.count_nonzero(good_s) < 2:
        return kt, Pt, None

    ks = k[good_s]
    ss = sigma[good_s]
    order_s = np.argsort(ks)
    ks = ks[order_s]
    ss = ss[order_s]
    sigt = np.interp(np.log(kt), np.log(ks), ss, left=np.nan, right=np.nan)
    return kt, Pt, sigt

def apply_pk_axis_style(ax, spectrum, sample=None):
    ax.set_xscale("log")
    if USE_LOG_ABS_FOR_SIGNED:
        ax.set_yscale("log")
    ax.set_xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$")
    label_core = spectrum_math_label(spectrum, include_k=PLOT_K_TIMES_PK)
    unit = r"\,[(h^{-1}\,\mathrm{Mpc})^2]" if PLOT_K_TIMES_PK else r"\,[(h^{-1}\,\mathrm{Mpc})^3]"
    ax.set_ylabel("$" + label_core + unit + "$")
    if XLIM is not None:
        ax.set_xlim(*XLIM)
    if YLIM is not None:
        ax.set_ylim(*YLIM)
    ax.grid(True, which="major", alpha=0.22, lw=0.6)
    ax.grid(True, which="minor", alpha=0.08, lw=0.4)


def save_figure(fig, stem, subdir=None):
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


def _attr_float(attrs, names, default=None):
    for name in names:
        if name in attrs:
            try:
                return float(attrs[name])
            except Exception:
                pass
    return default


def infer_cosmo_box_metadata(path, snap=None):
    """Read cosmology/box metadata from an HDF5 file, with safe TNG-like defaults."""
    cosmo = dict(COSMO_DEFAULTS)
    boxsize = BOXSIZE_DEFAULT
    nmesh = NMESH_DEFAULT
    z = np.nan

    with h5py.File(path, "r") as f:
        attrs = f.attrs
        # Cosmology.
        for key in ("Omega_c", "Omega_b", "h", "sigma8", "n_s"):
            val = _attr_float(attrs, [key, key.upper(), key.lower()], None)
            if val is not None:
                cosmo[key] = val
        omega_m = _attr_float(attrs, ["Omega_m", "Omega0", "Om0"], None)
        if omega_m is not None:
            if "Omega_b" in cosmo:
                cosmo["Omega_c"] = float(omega_m) - float(cosmo["Omega_b"])
        # Box / grid.
        box_val = _attr_float(attrs, ["boxsize", "BoxSize", "Lbox", "box_size"], None)
        if box_val is not None:
            # If the file stores ckpc/h, convert to Mpc/h.
            boxsize = box_val / 1000.0 if box_val > 1e4 else box_val
        nmesh_val = _attr_float(attrs, ["nmesh", "Nmesh", "grid", "GridSize"], None)
        if nmesh_val is not None:
            nmesh = int(nmesh_val)
        z_val = _attr_float(attrs, ["z", "redshift", "Redshift"], None)
        if z_val is not None:
            z = z_val

    if not np.isfinite(z) and snap is not None:
        z = float(ZMAP.get(int(snap), np.nan))
    return cosmo, float(boxsize), int(nmesh), float(z)


def available_cov_pk_types(path, sample):
    """Return Cov.py pk-type names present in sample/source group, e.g. dd, dg, dE."""
    specs = list_spectra(path, sample, source_group=SOURCE_GROUP)
    keys = [pk_key_for_cov(s) for s in specs]
    if Cov is not None and hasattr(Cov, "PK_TYPE_MAP"):
        allowed = set(Cov.PK_TYPE_MAP)
        keys = [k for k in keys if k in allowed]
    return sorted(set(keys))


def covariance_group_has_spectrum(h5_sample_group, spectrum):
    if COV_GROUP not in h5_sample_group:
        return False
    g = h5_sample_group[COV_GROUP]
    if f"cov_{COV_KIND}" not in g:
        return False
    if "spec_labels" not in g:
        return False
    labels = [x.decode() if isinstance(x, bytes) else str(x) for x in g["spec_labels"][()]]
    key = pk_key_for_cov(spectrum)
    return key in labels or ("P_" + key) in labels


def _cov_kind_dataset_name():
    kind = str(COV_KIND)
    if kind == "cNG":
        return "cov_cNG"
    if kind == "SSC":
        return "cov_SSC"
    return f"cov_{kind}"


_COV_IN_MEMORY_CACHE = {}


def _extract_sigma_from_cov_payload(payload, spectrum):
    """Extract sqrt(diag(cov)) for one spectrum from an existing or in-memory covariance payload."""
    if payload is None:
        return None, None

    kcov = np.asarray(payload.get("k", []), dtype=float)
    labels = payload.get("labels", [])
    cov = payload.get(_cov_kind_dataset_name(), None)
    if cov is None:
        return None, None

    labels = [x.decode() if isinstance(x, bytes) else str(x) for x in labels]
    key = pk_key_for_cov(spectrum)
    if key in labels:
        ispec = labels.index(key)
    elif ("P_" + key) in labels:
        ispec = labels.index("P_" + key)
    else:
        return None, None

    cov = np.asarray(cov, dtype=float)
    nk = len(kcov)
    block = cov[ispec*nk:(ispec+1)*nk, ispec*nk:(ispec+1)*nk]
    diag = np.diag(block)
    sigma = np.sqrt(np.where(diag >= 0, diag, np.nan))
    return kcov, sigma


def read_existing_covariance_payload(path, sample):
    """Read an existing covariance group without modifying the HDF5 file."""
    try:
        with h5py.File(path, "r") as f:
            if sample not in f:
                return None
            if COV_GROUP not in f[sample]:
                return None
            g = f[sample][COV_GROUP]
            cov_name = _cov_kind_dataset_name()
            if cov_name not in g or "spec_labels" not in g or "k" not in g:
                return None
            payload = {
                "k": np.asarray(g["k"], dtype=float),
                "labels": [x.decode() if isinstance(x, bytes) else str(x) for x in g["spec_labels"][()]],
                cov_name: np.asarray(g[cov_name], dtype=float),
            }
            return payload
    except Exception:
        return None


def compute_covariance_payload_in_memory(path, sample, snap=None):
    """Compute covariance in memory from Cov.py without writing to HDF5.

    This is intentionally read-only for the input HDF5 file.  It calls Cov.py's
    internal construction routines but never creates or overwrites any HDF5
    group/dataset.
    """
    if (not USE_THEORY_COVARIANCE) or (not COMPUTE_COVARIANCE_IN_MEMORY_IF_MISSING):
        return None
    if Cov is None:
        return None

    cache_key = (
        str(path), str(sample), str(COV_KIND), str(SOURCE_GROUP), str(NOISE_GROUP),
        bool(COV_INCLUDE_CNG), bool(COV_INCLUDE_SSC), str(COV_SSC_MODE), str(COV_SSC_BACKEND)
    )
    if cache_key in _COV_IN_MEMORY_CACHE:
        return _COV_IN_MEMORY_CACHE[cache_key]
    if cache_key in _COV_FAILURE_CACHE:
        return None

    try:
        pk_types = available_cov_pk_types(path, sample)
        if len(pk_types) == 0:
            _COV_FAILURE_CACHE.add(cache_key)
            return None

        cosmo, boxsize, nmesh, z = infer_cosmo_box_metadata(path, snap=snap)
        with h5py.File(path, "r") as f:
            if sample not in f:
                return None
            h5_group = f[sample]
            if SOURCE_GROUP not in h5_group:
                return None

            k, measured, noise = Cov._read_group_spectra(
                h5_group, SOURCE_GROUP, NOISE_GROUP, pk_types
            )

        volume = float(boxsize) ** 3
        nmodes = Cov.estimate_nmodes_from_k(k, volume)
        spectra, meta = Cov.merge_spectra_for_covariance(
            k, measured,
            source=COV_SOURCE,
            z=z,
            cosmo_params=cosmo,
            aia_default=COV_AIA_DEFAULT,
            c1_rhocrit=COV_C1_RHOCRIT,
            kmin_fit=COV_KMIN_FIT,
            kmax_fit=COV_KMAX_FIT,
        )
        field_coeff = Cov.field_coefficients_from_meta(meta)

        sigma_b2 = 0.0
        if COV_INCLUDE_SSC:
            try:
                sigma_b2 = Cov.get_sigma_b2(
                    mode=COV_SSC_MODE,
                    boxsize=float(boxsize),
                    volume=volume,
                    z=float(z),
                    cosmo_params=cosmo,
                )
            except Exception as exc:
                print(f"[warn] sigma_b2 failed for {path.name}/{sample}; SSC disabled. Reason: {exc}")
                sigma_b2 = 0.0

        model = Cov.CovarianceModel(
            k=k,
            pk_types=pk_types,
            spectra=spectra,
            noise=noise,
            nmodes=nmodes,
            volume=volume,
            sigma_b2=sigma_b2,
            responses=Cov.build_ssc_responses_tree(k, pk_types, spectra),
            field_coefficients=field_coeff,
        )

        cov_g = Cov.gaussian_covariance(model)

        if COV_INCLUDE_CNG:
            try:
                cov_c = Cov.cng_covariance_halomodel(
                    model,
                    cosmo_params=cosmo,
                    z=float(z),
                    hm_options=COV_HM_OPTIONS,
                    include=True,
                )
            except Exception as exc:
                print(f"[warn] cNG covariance failed for {path.name}/{sample}; using zero cNG. Reason: {exc}")
                cov_c = np.zeros_like(cov_g)
        else:
            cov_c = np.zeros_like(cov_g)

        if COV_INCLUDE_SSC:
            try:
                cov_s = Cov.ssc_covariance(
                    model,
                    cosmo_params=cosmo,
                    z=float(z),
                    hm_options=COV_HM_OPTIONS,
                    backend=COV_SSC_BACKEND,
                )
            except Exception as exc:
                print(f"[warn] SSC covariance failed for {path.name}/{sample}; using zero SSC. Reason: {exc}")
                cov_s = np.zeros_like(cov_g)
        else:
            cov_s = np.zeros_like(cov_g)

        payload = {
            "k": np.asarray(k, dtype=float),
            "labels": list(pk_types),
            "cov_gaussian": cov_g,
            "cov_cNG": cov_c,
            "cov_SSC": cov_s,
            "cov_total": cov_g + cov_c + cov_s,
            "Nmodes": nmodes,
        }
        _COV_IN_MEMORY_CACHE[cache_key] = payload
        return payload

    except Exception as exc:
        _COV_FAILURE_CACHE.add(cache_key)
        print(f"[warn] in-memory covariance failed for {path.name}/{sample}: {exc}")
        return None


def ensure_covariance(path, sample, spectrum, snap=None):
    """Return True if uncertainty is available from existing covariance or in-memory Cov.py."""
    if not USE_THEORY_COVARIANCE:
        return False

    payload = read_existing_covariance_payload(path, sample)
    kcov, sig = _extract_sigma_from_cov_payload(payload, spectrum)
    if kcov is not None and sig is not None:
        return True

    if READ_EXISTING_COVARIANCE_ONLY:
        return False

    payload = compute_covariance_payload_in_memory(path, sample, snap=snap)
    kcov, sig = _extract_sigma_from_cov_payload(payload, spectrum)
    return (kcov is not None) and (sig is not None)


def read_covariance_sigma(path, sample, spectrum, snap=None):
    """Read or compute sqrt(diag(cov)) for a single spectrum. Returns (k_cov, sigma_P).

    The input HDF5 file is never modified.  Missing stored covariance triggers
    an in-memory Cov.py calculation only when COMPUTE_COVARIANCE_IN_MEMORY_IF_MISSING=True.
    """
    if not SHOW_UNCERTAINTY or not USE_THEORY_COVARIANCE:
        return None, None

    payload = read_existing_covariance_payload(path, sample)
    kcov, sigma = _extract_sigma_from_cov_payload(payload, spectrum)
    if kcov is not None and sigma is not None:
        return kcov, sigma

    if READ_EXISTING_COVARIANCE_ONLY:
        return None, None

    payload = compute_covariance_payload_in_memory(path, sample, snap=snap)
    return _extract_sigma_from_cov_payload(payload, spectrum)


def sigma_for_plot_grid(path, sample, spectrum, k, snap=None):
    kcov, sigma = read_covariance_sigma(path, sample, spectrum, snap=snap)
    if kcov is None or sigma is None:
        return None
    if len(kcov) == len(k) and np.allclose(kcov, k, rtol=1e-5, atol=1e-10):
        return sigma
    return np.interp(k, kcov, sigma, left=np.nan, right=np.nan)


def draw_pk_series(ax, k, P, sigma, spectrum, color, label=None, lw=1.35, ms=3.3):
    if USE_TARGET_K_FOR_PLOTTING:
        k, P, sigma = resample_to_target_k(k, P, sigma=sigma, target_k=TARGET_K)

    y_signed, y_plot, yerr, neg, good = prepare_y(k, P, sigma=sigma)
    if not np.any(good):
        return False

    style = str(UNCERTAINTY_STYLE).lower()
    draw_band = SHOW_UNCERTAINTY and yerr is not None and np.any(np.isfinite(yerr[good])) and style in ("shade", "band", "fill", "fill_between", "both")
    draw_err = SHOW_UNCERTAINTY and yerr is not None and np.any(np.isfinite(yerr[good])) and style in ("errorbar", "errorbars", "err", "both")

    ax.plot(k[good], y_plot[good],  ms=ms, lw=lw, color=color, label=label)

    if draw_band:
        low = y_plot - yerr
        high = y_plot + yerr
        band = good & np.isfinite(low) & np.isfinite(high)
        if np.any(band):
            positive_y = y_plot[band]
            floor = np.nanmin(positive_y[positive_y > 0]) * 0.2 if np.any(positive_y > 0) else 1e-300
            low = np.maximum(low, floor)
            ax.fill_between(k[band], low[band], high[band],
                            color=color, alpha=UNCERTAINTY_ALPHA, linewidth=0, zorder=1)

    if draw_err:
        idx = np.where(good & np.isfinite(yerr))[0]
        if ERRORBAR_EVERY and int(ERRORBAR_EVERY) > 1:
            idx = idx[::int(ERRORBAR_EVERY)]
        if idx.size:
            ax.errorbar(k[idx], y_plot[idx], yerr=yerr[idx],
                        fmt="none", ecolor=color, elinewidth=1.05,
                        capsize=2.2, capthick=1.05, alpha=0.85, zorder=3)

    if USE_LOG_ABS_FOR_SIGNED and np.any(good & neg):
        mm = good & neg
        ax.plot(k[mm], y_plot[mm], linestyle="none",  ms=ms+0.9,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.05, zorder=4)
    return True


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

# %% code cell 13

# -------------------------------------------------
# Plot 1: 3x3 redshift grids, comparing gravity models at fixed redshift
# -------------------------------------------------
def plot_model_comparison_grid(sample, spectrum, files_df=files_df, flags=FLAGS, snaps=SNAPS):
    ncols = int(N_COLS_GRID)
    nrows = int(np.ceil(len(snaps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE_GRID, sharex=True, sharey=False)
    axes = np.asarray(axes).reshape(-1)

    for ax, snap in zip(axes, snaps):
        z = ZMAP.get(int(snap), np.nan)
        plotted = False
        for flag in flags:
            path = pks_path(flag, snap)
            if not path.exists():
                continue
            try:
                k, P, _ = read_spectrum(path, sample, spectrum)
            except Exception:
                continue

            sigma = sigma_for_plot_grid(path, sample, spectrum, k, snap=snap)
            label = FLAG_LABEL.get(flag, flag)
            color = FLAG_COLOR.get(flag, "0.4")
            plotted |= draw_pk_series(ax, k, P, sigma, spectrum, color=color, label=label, lw=1.35, ms=3.3)

        apply_pk_axis_style(ax, spectrum, sample=sample)
        ax.set_title(format_z_title(z), pad=6)

        if not plotted:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", color="0.45")

    for ax in axes[len(snaps):]:
        ax.axis("off")

    handles = [
        Line2D([0], [0], color=FLAG_COLOR.get(flag, "0.4"), lw=1.8,  ms=4,
               label=FLAG_LABEL.get(flag, flag))
        for flag in flags
    ]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 6),
               frameon=False, bbox_to_anchor=(0.5, 1.015))
    sample_label = display_sample_label(sample, spectrum)
    if is_sample_independent_spectrum(spectrum):
        title = rf"Gravity-model comparison for ${spectrum_math_label(spectrum, include_k=False)}$"
    else:
        title = rf"{sample_label}: gravity-model comparison for ${spectrum_math_label(spectrum, include_k=False)}$"
    fig.suptitle(title, y=1.045, fontsize=16)
    fig.tight_layout()
    out_sample = output_sample_label(sample, spectrum)
    stem = f"{out_sample}_{output_safe_name(spectrum)}_grid_models"
    save_figure(fig, stem, subdir=f"{out_sample}/model_grids")
    maybe_close(fig)
    return fig

# %% code cell 14

# -------------------------------------------------
# Plot 2: same model, different redshifts
# -------------------------------------------------
def plot_redshift_evolution_by_model(sample, spectrum, files_df=files_df, flags=FLAGS, snaps=SNAPS):
    ncols = min(3, len(flags))
    nrows = int(np.ceil(len(flags) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE_MODEL_EVOLUTION, sharex=True, sharey=False)
    axes = np.asarray(axes).reshape(-1)

    zvals = np.array([ZMAP.get(int(s), np.nan) for s in snaps], dtype=float)
    finite_z = np.isfinite(zvals)
    zmin, zmax = np.nanmin(zvals[finite_z]), np.nanmax(zvals[finite_z])
    cmap = plt.get_cmap(REDSHIFT_CMAP)
    norm = plt.Normalize(vmin=zmin, vmax=zmax)

    for ax, flag in zip(axes, flags):
        plotted = False
        for snap in snaps:
            path = pks_path(flag, snap)
            if not path.exists():
                continue
            try:
                k, P, z = read_spectrum(path, sample, spectrum)
            except Exception:
                continue
            if not np.isfinite(z):
                z = ZMAP.get(int(snap), np.nan)
            color = cmap(norm(z))
            sigma = sigma_for_plot_grid(path, sample, spectrum, k, snap=snap)
            plotted |= draw_pk_series(ax, k, P, sigma, spectrum, color=color, label=None, lw=1.20, ms=3.0)

        apply_pk_axis_style(ax, spectrum, sample=sample)
        ax.set_title(FLAG_LABEL.get(flag, flag), pad=6)
        if plotted:
            # One colorbar per subplot, placed to the right of that subplot.
            # It has the same height as the corresponding subplot and does not overlap.
            add_axis_colorbar(fig, ax, cmap, norm)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", color="0.45")

    for ax in axes[len(flags):]:
        ax.axis("off")

    sample_label = display_sample_label(sample, spectrum)
    if is_sample_independent_spectrum(spectrum):
        title = rf"Redshift evolution for ${spectrum_math_label(spectrum, include_k=False)}$"
    else:
        title = rf"{sample_label}: redshift evolution for ${spectrum_math_label(spectrum, include_k=False)}$"
    fig.suptitle(title, y=1.015, fontsize=16)
    fig.subplots_adjust(left=0.075, right=0.965, bottom=0.085, top=0.900, wspace=0.42, hspace=0.35)
    out_sample = output_sample_label(sample, spectrum)
    stem = f"{out_sample}_{output_safe_name(spectrum)}_redshift_evolution"
    save_figure(fig, stem, subdir=f"{out_sample}/redshift_evolution")
    maybe_close(fig)
    return fig

# %% code cell 15

# -------------------------------------------------
# Plot 3: enhancement relative to GR
# -------------------------------------------------
def plot_enhancement_relative_to_gr(sample, spectrum, flags=FLAGS, snaps=SNAPS):
    if "GR" not in flags:
        print("[skip] GR is not in FLAGS, cannot compute enhancement.")
        return None

    compare_flags = [f for f in flags if f != "GR"]
    if not compare_flags:
        print("[skip] no non-GR flags.")
        return None

    ncols = int(N_COLS_GRID)
    nrows = int(np.ceil(len(snaps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE_GRID, sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, snap in zip(axes, snaps):
        gr_path = pks_path("GR", snap)
        z = ZMAP.get(int(snap), np.nan)
        if not gr_path.exists():
            ax.text(0.5, 0.5, "No GR", transform=ax.transAxes, ha="center", va="center", color="0.45")
            ax.set_title(format_z_title(z), pad=6)
            continue

        try:
            k_gr, P_gr, _ = read_spectrum(gr_path, sample, spectrum)
        except Exception:
            ax.text(0.5, 0.5, "No GR spectrum", transform=ax.transAxes, ha="center", va="center", color="0.45")
            ax.set_title(format_z_title(z), pad=6)
            continue
        sig_gr = sigma_for_plot_grid(gr_path, sample, spectrum, k_gr, snap=snap)

        plotted = False
        for flag in compare_flags:
            path = pks_path(flag, snap)
            if not path.exists():
                continue
            try:
                k, P, _ = read_spectrum(path, sample, spectrum)
            except Exception:
                continue

            if k.shape != k_gr.shape or not np.allclose(k, k_gr, rtol=1e-4, atol=1e-8):
                Pgr_i = np.interp(k, k_gr, P_gr, left=np.nan, right=np.nan)
                sig_gr_i = None if sig_gr is None else np.interp(k, k_gr, sig_gr, left=np.nan, right=np.nan)
            else:
                Pgr_i = P_gr
                sig_gr_i = sig_gr

            ratio = P / Pgr_i - 1.0
            good = np.isfinite(k) & np.isfinite(ratio)
            if not np.any(good):
                continue

            sigma = sigma_for_plot_grid(path, sample, spectrum, k, snap=snap)
            sigma_ratio = None
            if SHOW_UNCERTAINTY and sigma is not None and sig_gr_i is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    sigma_ratio = np.sqrt((sigma / Pgr_i)**2 + (P * sig_gr_i / Pgr_i**2)**2)

            color = FLAG_COLOR.get(flag, "0.4")
            ax.axhline(0.0, color="0.25", lw=0.8, alpha=0.7)
            ax.plot(k[good], ratio[good],  ms=3.3, lw=1.35,
                    color=color, label=FLAG_LABEL.get(flag, flag))
            if sigma_ratio is not None:
                band = good & np.isfinite(sigma_ratio)
                if np.any(band):
                    style = str(UNCERTAINTY_STYLE).lower()
                    if style in ("shade", "band", "fill", "fill_between", "both"):
                        ax.fill_between(k[band],
                                        ratio[band] - sigma_ratio[band],
                                        ratio[band] + sigma_ratio[band],
                                        color=color, alpha=UNCERTAINTY_ALPHA, linewidth=0, zorder=1)
                    if style in ("errorbar", "errorbars", "err", "both"):
                        idx = np.where(band)[0]
                        if ERRORBAR_EVERY and int(ERRORBAR_EVERY) > 1:
                            idx = idx[::int(ERRORBAR_EVERY)]
                        ax.errorbar(k[idx], ratio[idx], yerr=sigma_ratio[idx],
                                    fmt="none", ecolor=color, elinewidth=0.95,
                                    capsize=2.0, capthick=0.95, alpha=0.80, zorder=3)
            plotted = True

        ax.set_xscale("log")
        ax.set_xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$")
        ax.set_ylabel(r"$P/P_{\mathrm{GR}}-1$")
        ax.set_title(format_z_title(z), pad=6)
        ax.grid(True, which="major", alpha=0.22, lw=0.6)
        ax.grid(True, which="minor", alpha=0.08, lw=0.4)
        ax.set_ylim(-1,1)
        if XLIM is not None:
            ax.set_xlim(*XLIM)
        if not plotted:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", color="0.45")

    for ax in axes[len(snaps):]:
        ax.axis("off")

    handles = [
        Line2D([0], [0], color=FLAG_COLOR.get(flag, "0.4"), lw=1.8,  ms=4,
               label=FLAG_LABEL.get(flag, flag))
        for flag in compare_flags
    ]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 5),
               frameon=False, bbox_to_anchor=(0.5, 1.015))
    sample_label = display_sample_label(sample, spectrum)
    if is_sample_independent_spectrum(spectrum):
        title = rf"Enhancement of ${spectrum_math_label(spectrum, include_k=False)}$ relative to GR"
    else:
        title = rf"{sample_label}: enhancement of ${spectrum_math_label(spectrum, include_k=False)}$ relative to GR"
    fig.suptitle(title, y=1.045, fontsize=16)
    fig.tight_layout()
    out_sample = output_sample_label(sample, spectrum)
    stem = f"{out_sample}_{output_safe_name(spectrum)}_enhancement_vs_GR"
    save_figure(fig, stem, subdir=f"{out_sample}/enhancement")
    maybe_close(fig)
    return fig

# %% code cell 16

# -------------------------------------------------
# A_IA estimators and plotting
# -------------------------------------------------
AIA_METHODS = {
    "deltaE": {
        "numerator": "P_dE",
        "denominator": "P_dd",
        "label": r"$A_{\mathrm{IA}}^{\delta E}$",
        "long_label": r"$A_{\mathrm{IA}}^{\delta E}\propto -P_{\delta\,E}/P_{\delta\delta}$",
        "directory": "aia_deltaE",
    },
    "gE": {
        "numerator": "P_gE",
        "denominator": "P_dg",
        "label": r"$A_{\mathrm{IA}}^{gE}$",
        "long_label": r"$A_{\mathrm{IA}}^{gE}\propto -P_{gE}/P_{\delta g}$",
        "directory": "aia_gE",
    },
}


def omega_m_from_cosmo(cosmo):
    return float(cosmo.get("Omega_c", COSMO_DEFAULTS["Omega_c"])) + float(cosmo.get("Omega_b", COSMO_DEFAULTS["Omega_b"]))


def growth_factor_fallback(z, omega_m0):
    """Approximate growth factor D(z), normalized to D(0)=1.

    This is used only if pyccl is unavailable.  It uses the Carroll, Press &
    Turner-style growth suppression approximation for a flat LCDM cosmology.
    """
    z = float(z)
    a = 1.0 / (1.0 + z)
    om0 = float(omega_m0)
    ol0 = 1.0 - om0

    def g_of_a(a_):
        ez2 = om0 / a_**3 + ol0
        om = (om0 / a_**3) / ez2
        ol = ol0 / ez2
        return 5.0 * om / (2.0 * (om**(4.0/7.0) - ol + (1.0 + 0.5*om) * (1.0 + ol/70.0)))

    return a * g_of_a(a) / g_of_a(1.0)


def growth_factor_D(z, cosmo):
    omega_m0 = omega_m_from_cosmo(cosmo)
    if AIA_USE_CCL_GROWTH:
        try:
            import pyccl as ccl
            ccl_cosmo = ccl.Cosmology(
                Omega_c=float(cosmo.get("Omega_c", COSMO_DEFAULTS["Omega_c"])),
                Omega_b=float(cosmo.get("Omega_b", COSMO_DEFAULTS["Omega_b"])),
                h=float(cosmo.get("h", COSMO_DEFAULTS["h"])),
                sigma8=float(cosmo.get("sigma8", COSMO_DEFAULTS["sigma8"])),
                n_s=float(cosmo.get("n_s", COSMO_DEFAULTS["n_s"])),
                transfer_function="bbks",
            )
            return float(ccl.growth_factor(ccl_cosmo, 1.0/(1.0 + float(z))))
        except Exception:
            pass
    return float(growth_factor_fallback(z, omega_m0))


def aia_prefactor(z, cosmo):
    """NLA-style prefactor multiplying P_XE/P_Xdelta.

    A_IA = prefactor * P_XE/P_Xdelta,
    where prefactor = -3 D(z) / [2 C1 rho_crit Omega_m].
    """
    om = omega_m_from_cosmo(cosmo)
    D = growth_factor_D(z, cosmo)
    return -3.0 * D / (2.0 * float(AIA_C1_RHOCRIT) * om)


def read_spectrum_or_none(path, sample, spectrum):
    try:
        return read_spectrum(path, sample, spectrum)
    except Exception:
        return None


def read_aia_estimator(path, sample, method, snap=None):
    """Read and compute one k-dependent A_IA estimator from one HDF5 file.

    This function never writes to the HDF5 file.
    """
    if method not in AIA_METHODS:
        raise KeyError(f"Unknown A_IA method: {method}")
    cfg = AIA_METHODS[method]
    num_spec = cfg["numerator"]
    den_spec = cfg["denominator"]

    got_num = read_spectrum_or_none(path, sample, num_spec)
    got_den = read_spectrum_or_none(path, sample, den_spec)
    if got_num is None or got_den is None:
        return None

    k_num, P_num, z = got_num
    k_den, P_den, zden = got_den
    if not np.isfinite(z):
        z = zden
    if not np.isfinite(z) and snap is not None:
        z = ZMAP.get(int(snap), np.nan)

    if k_den.shape != k_num.shape or not np.allclose(k_den, k_num, rtol=1e-4, atol=1e-8):
        P_den_i = np.interp(k_num, k_den, P_den, left=np.nan, right=np.nan)
    else:
        P_den_i = P_den

    cosmo, _, _, z_meta = infer_cosmo_box_metadata(path, snap=snap)
    if not np.isfinite(z):
        z = z_meta
    pref = aia_prefactor(z, cosmo)

    with np.errstate(divide="ignore", invalid="ignore"):
        A = pref * P_num / P_den_i

    # Diagonal uncertainty propagation, if the covariance already exists.
    sig_num = sigma_for_plot_grid(path, sample, num_spec, k_num, snap=snap)
    sig_den = sigma_for_plot_grid(path, sample, den_spec, k_num, snap=snap)
    sig_A = None
    if sig_num is not None and sig_den is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            sig_A = abs(pref) * np.sqrt((sig_num / P_den_i)**2 + (P_num * sig_den / P_den_i**2)**2)

    good = np.isfinite(k_num) & np.isfinite(A)
    return dict(k=k_num, A=A, sigma=sig_A, good=good, z=float(z), prefactor=float(pref))


def aia_fit_value(k, A):
    k = np.asarray(k, dtype=float)
    A = np.asarray(A, dtype=float)
    m = np.isfinite(k) & np.isfinite(A) & (k >= AIA_FIT_KMIN) & (k <= AIA_FIT_KMAX)
    if np.count_nonzero(m) == 0:
        return np.nan
    return float(np.nanmedian(A[m]))


def draw_aia_series(ax, k, A, sigma, color, label=None, lw=1.4, ms=3.2):
    if USE_TARGET_K_FOR_PLOTTING:
        k, A, sigma = resample_to_target_k(k, A, sigma=sigma, target_k=TARGET_K)

    good = np.isfinite(k) & np.isfinite(A)
    if not np.any(good):
        return False

    style = str(UNCERTAINTY_STYLE).lower()
    draw_band = SHOW_UNCERTAINTY and sigma is not None and style in ("shade", "band", "fill", "fill_between", "both")
    draw_err = SHOW_UNCERTAINTY and sigma is not None and style in ("errorbar", "errorbars", "err", "both")

    ax.plot(k[good], A[good],  ms=ms, lw=lw, color=color, label=label)

    if draw_band:
        band = good & np.isfinite(sigma)
        if np.any(band):
            ax.fill_between(k[band], A[band] - sigma[band], A[band] + sigma[band],
                            color=color, alpha=UNCERTAINTY_ALPHA, linewidth=0, zorder=1)

    if draw_err:
        idx = np.where(good & np.isfinite(sigma))[0]
        if ERRORBAR_EVERY and int(ERRORBAR_EVERY) > 1:
            idx = idx[::int(ERRORBAR_EVERY)]
        if idx.size:
            ax.errorbar(k[idx], A[idx], yerr=sigma[idx], fmt="none",
                        ecolor=color, elinewidth=1.05, capsize=2.2,
                        capthick=1.05, alpha=0.85, zorder=3)
    
    return True


def apply_aia_axis_style(ax, method):
    ax.axhline(0.0, color="0.25", lw=0.8, alpha=0.75)
    ax.axvspan(AIA_FIT_KMIN, AIA_FIT_KMAX, color="0.8", alpha=0.13, linewidth=0)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_ylim(1e-3,1e2)
    ax.set_xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(AIA_METHODS[method]["label"] + r"$(k)$")
    ax.grid(True, which="major", alpha=0.22, lw=0.6)
    ax.grid(True, which="minor", alpha=0.08, lw=0.4)
    if XLIM is not None:
        ax.set_xlim(*XLIM)


def plot_aia_model_comparison_grid(sample, method, flags=FLAGS, snaps=SNAPS):
    ncols = int(N_COLS_GRID)
    nrows = int(np.ceil(len(snaps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE_GRID, sharex=True, sharey=False)
    axes = np.asarray(axes).reshape(-1)

    for ax, snap in zip(axes, snaps):
        z = ZMAP.get(int(snap), np.nan)
        plotted = False
        for flag in flags:
            path = pks_path(flag, snap)
            if not path.exists():
                continue
            res = read_aia_estimator(path, sample, method, snap=snap)
            if res is None:
                continue
            Abar = aia_fit_value(res["k"], res["A"])
            label = FLAG_LABEL.get(flag, flag)
            if np.isfinite(Abar):
                label = f"{label} ({Abar:.2g})"
            color = FLAG_COLOR.get(flag, "0.4")
            plotted |= draw_aia_series(ax, res["k"], res["A"], res["sigma"],
                                       color=color, label=label, lw=1.35, ms=3.2)

        apply_aia_axis_style(ax, method)
        ax.set_title(format_z_title(z), pad=6)
        if not plotted:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", color="0.45")

    for ax in axes[len(snaps):]:
        ax.axis("off")

    handles = [
        Line2D([0], [0], color=FLAG_COLOR.get(flag, "0.4"), lw=1.8,  ms=4,
               label=FLAG_LABEL.get(flag, flag))
        for flag in flags
    ]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 6),
               frameon=False, bbox_to_anchor=(0.5, 1.015))
    fig.suptitle(rf"{sample}: gravity-model comparison for {AIA_METHODS[method]['long_label']}",
                 y=1.045, fontsize=16)
    fig.tight_layout()
    out_sample = output_safe_name(sample)
    stem = f"{out_sample}_{AIA_METHODS[method]['directory']}_grid_models"
    save_figure(fig, stem, subdir=f"{out_sample}/aia/{AIA_METHODS[method]['directory']}/model_grids")
    maybe_close(fig)
    return fig


def plot_aia_redshift_evolution_by_model(sample, method, flags=FLAGS, snaps=SNAPS):
    ncols = min(3, len(flags))
    nrows = int(np.ceil(len(flags) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE_MODEL_EVOLUTION, sharex=True, sharey=False)
    axes = np.asarray(axes).reshape(-1)

    zvals = np.array([ZMAP.get(int(s), np.nan) for s in snaps], dtype=float)
    finite_z = np.isfinite(zvals)
    zmin, zmax = np.nanmin(zvals[finite_z]), np.nanmax(zvals[finite_z])
    cmap = plt.get_cmap(REDSHIFT_CMAP)
    norm = plt.Normalize(vmin=zmin, vmax=zmax)

    for ax, flag in zip(axes, flags):
        plotted = False
        for snap in snaps:
            path = pks_path(flag, snap)
            if not path.exists():
                continue
            res = read_aia_estimator(path, sample, method, snap=snap)
            if res is None:
                continue
            z = res["z"]
            if not np.isfinite(z):
                z = ZMAP.get(int(snap), np.nan)
            color = cmap(norm(z))
            plotted |= draw_aia_series(ax, res["k"], res["A"], res["sigma"],
                                       color=color, label=None, lw=1.15, ms=3.0)

        apply_aia_axis_style(ax, method)
        ax.set_title(FLAG_LABEL.get(flag, flag), pad=6)
        if plotted:
            add_axis_colorbar(fig, ax, cmap, norm)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", color="0.45")

    for ax in axes[len(flags):]:
        ax.axis("off")

    fig.suptitle(rf"{sample}: redshift evolution for {AIA_METHODS[method]['long_label']}",
                 y=1.015, fontsize=16)
    fig.subplots_adjust(left=0.075, right=0.965, bottom=0.085, top=0.900, wspace=0.42, hspace=0.35)
    out_sample = output_safe_name(sample)
    stem = f"{out_sample}_{AIA_METHODS[method]['directory']}_redshift_evolution"
    save_figure(fig, stem, subdir=f"{out_sample}/aia/{AIA_METHODS[method]['directory']}/redshift_evolution")
    maybe_close(fig)
    return fig


def plot_aia_enhancement_relative_to_gr(sample, method, flags=FLAGS, snaps=SNAPS):
    if "GR" not in flags:
        print("[skip] GR is not in FLAGS, cannot compute A_IA enhancement.")
        return None

    compare_flags = [f for f in flags if f != "GR"]
    ncols = int(N_COLS_GRID)
    nrows = int(np.ceil(len(snaps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE_GRID, sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, snap in zip(axes, snaps):
        z = ZMAP.get(int(snap), np.nan)
        gr_path = pks_path("GR", snap)
        if not gr_path.exists():
            ax.text(0.5, 0.5, "No GR", transform=ax.transAxes, ha="center", va="center", color="0.45")
            ax.set_title(format_z_title(z), pad=6)
            continue

        gr = read_aia_estimator(gr_path, sample, method, snap=snap)
        if gr is None:
            ax.text(0.5, 0.5, "No GR estimator", transform=ax.transAxes, ha="center", va="center", color="0.45")
            ax.set_title(format_z_title(z), pad=6)
            continue

        plotted = False
        for flag in compare_flags:
            path = pks_path(flag, snap)
            if not path.exists():
                continue
            res = read_aia_estimator(path, sample, method, snap=snap)
            if res is None:
                continue

            k = res["k"]
            if gr["k"].shape != k.shape or not np.allclose(gr["k"], k, rtol=1e-4, atol=1e-8):
                Agr = np.interp(k, gr["k"], gr["A"], left=np.nan, right=np.nan)
                sig_gr = None if gr["sigma"] is None else np.interp(k, gr["k"], gr["sigma"], left=np.nan, right=np.nan)
            else:
                Agr = gr["A"]
                sig_gr = gr["sigma"]

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = res["A"] / Agr - 1.0

            sigma_ratio = None
            if SHOW_UNCERTAINTY and res["sigma"] is not None and sig_gr is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    sigma_ratio = np.sqrt((res["sigma"] / Agr)**2 + (res["A"] * sig_gr / Agr**2)**2)

            if USE_TARGET_K_FOR_PLOTTING:
                k, ratio, sigma_ratio = resample_to_target_k(k, ratio, sigma=sigma_ratio, target_k=TARGET_K)

            good = np.isfinite(k) & np.isfinite(ratio)
            if not np.any(good):
                continue

            color = FLAG_COLOR.get(flag, "0.4")
            ax.axhline(0.0, color="0.25", lw=0.8, alpha=0.7)
            ax.plot(k[good], ratio[good],  ms=3.3, lw=1.35,
                    color=color, label=FLAG_LABEL.get(flag, flag))

            if sigma_ratio is not None:
                band = good & np.isfinite(sigma_ratio)
                if np.any(band):
                    style = str(UNCERTAINTY_STYLE).lower()
                    if style in ("shade", "band", "fill", "fill_between", "both"):
                        ax.fill_between(k[band],
                                        ratio[band] - sigma_ratio[band],
                                        ratio[band] + sigma_ratio[band],
                                        color=color, alpha=UNCERTAINTY_ALPHA, linewidth=0, zorder=1)
                    if style in ("errorbar", "errorbars", "err", "both"):
                        idx = np.where(band)[0]
                        if ERRORBAR_EVERY and int(ERRORBAR_EVERY) > 1:
                            idx = idx[::int(ERRORBAR_EVERY)]
                        ax.errorbar(k[idx], ratio[idx], yerr=sigma_ratio[idx],
                                    fmt="none", ecolor=color, elinewidth=0.95,
                                    capsize=2.0, capthick=0.95, alpha=0.80, zorder=3)
            plotted = True

        ax.set_xscale("log")
        ax.set_xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$")
        ax.set_ylabel(r"$A_{\mathrm{IA}}/A_{\mathrm{IA,GR}}-1$")
        ax.set_title(format_z_title(z), pad=6)
        ax.grid(True, which="major", alpha=0.22, lw=0.6)
        ax.grid(True, which="minor", alpha=0.08, lw=0.4)
        if XLIM is not None:
            ax.set_xlim(*XLIM)
        if not plotted:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", color="0.45")

    for ax in axes[len(snaps):]:
        ax.axis("off")

    handles = [
        Line2D([0], [0], color=FLAG_COLOR.get(flag, "0.4"), lw=1.8,  ms=4,
               label=FLAG_LABEL.get(flag, flag))
        for flag in compare_flags
    ]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 5),
               frameon=False, bbox_to_anchor=(0.5, 1.015))
    fig.suptitle(rf"{sample}: enhancement of {AIA_METHODS[method]['label']} relative to GR",
                 y=1.045, fontsize=16)
    fig.tight_layout()
    out_sample = output_safe_name(sample)
    stem = f"{out_sample}_{AIA_METHODS[method]['directory']}_enhancement_vs_GR"
    save_figure(fig, stem, subdir=f"{out_sample}/aia/{AIA_METHODS[method]['directory']}/enhancement")
    maybe_close(fig)
    return fig


def sample_has_aia_method(sample, method, spectra_by_sample=spectra_by_sample):
    cfg = AIA_METHODS[method]
    spectra = set(spectra_by_sample.get(sample, []))
    return (cfg["numerator"] in spectra) and (cfg["denominator"] in spectra)


def plot_all_aia(samples=samples, spectra_by_sample=spectra_by_sample):
    manifest = []
    for sample in samples:
        # A_IA is defined for galaxy/shape samples only.  Skip purely synthetic
        # snapshot output directories.
        methods = [m for m in AIA_METHODS_TO_PLOT if m in AIA_METHODS and sample_has_aia_method(sample, m)]
        if len(methods) == 0:
            print(f"[skip] {sample}: no complete A_IA estimator inputs")
            continue

        print(f"\n=== A_IA estimators: {sample} | {methods} ===")
        for method in methods:
            print(f"  plotting A_IA method: {method}")
            try:
                plot_aia_model_comparison_grid(sample, method)
                manifest.append((sample, method, "aia_model_grid"))
            except Exception as exc:
                print(f"    [warn] A_IA model grid failed for {sample}/{method}: {exc}")

            try:
                plot_aia_redshift_evolution_by_model(sample, method)
                manifest.append((sample, method, "aia_redshift_evolution"))
            except Exception as exc:
                print(f"    [warn] A_IA redshift evolution failed for {sample}/{method}: {exc}")

            if MAKE_ENHANCEMENT_RELATIVE_TO_GR:
                try:
                    plot_aia_enhancement_relative_to_gr(sample, method)
                    manifest.append((sample, method, "aia_enhancement"))
                except Exception as exc:
                    print(f"    [warn] A_IA enhancement failed for {sample}/{method}: {exc}")

    return pd.DataFrame(manifest, columns=["sample", "method", "figure_type"])

# %% code cell 17
# -------------------------------------------------
# Run all plots
# -------------------------------------------------
def choose_representative_sample_for_spectrum(spectrum, samples=samples, spectra_by_sample=spectra_by_sample):
    """Choose one HDF5 sample group from which to read sample-independent spectra."""
    # Prefer "all" if it exists, since it is the least misleading representative.
    if "all" in samples and spectrum in spectra_by_sample.get("all", []):
        return "all"
    for sample in samples:
        if spectrum in spectra_by_sample.get(sample, []):
            return sample
    return None


def plot_one_task(sample, spectrum, manifest, label_for_print=None):
    label_for_print = label_for_print or sample
    print(f"  plotting {spectrum} [{label_for_print}]")

    try:
        plot_model_comparison_grid(sample, spectrum)
        manifest.append((display_sample_label(sample, spectrum), spectrum, "model_grid"))
    except Exception as exc:
        print(f"    [warn] model grid failed for {label_for_print}/{spectrum}: {exc}")

    try:
        plot_redshift_evolution_by_model(sample, spectrum)
        manifest.append((display_sample_label(sample, spectrum), spectrum, "redshift_evolution"))
    except Exception as exc:
        print(f"    [warn] redshift evolution failed for {label_for_print}/{spectrum}: {exc}")

    if MAKE_ENHANCEMENT_RELATIVE_TO_GR:
        try:
            plot_enhancement_relative_to_gr(sample, spectrum)
            manifest.append((display_sample_label(sample, spectrum), spectrum, "enhancement"))
        except Exception as exc:
            print(f"    [warn] enhancement failed for {label_for_print}/{spectrum}: {exc}")


def plot_everything(samples=samples, spectra_by_sample=spectra_by_sample):
    manifest = []

    # First plot sample-dependent spectra under each galaxy sample.
    for sample in samples:
        spectra = spectra_by_sample.get(sample, [])
        spectra_sample_dependent = [sp for sp in spectra if not is_sample_independent_spectrum(sp)]

        if len(spectra_sample_dependent) == 0:
            print(f"[skip] {sample}: no galaxy-sample-dependent spectra discovered")
            continue

        print(f"\n=== Galaxy sample: {sample} | {len(spectra_sample_dependent)} spectra ===")
        for spectrum in spectra_sample_dependent:
            plot_one_task(sample, spectrum, manifest, label_for_print=sample)

    # Then plot snapshot-particle spectra only once.
    all_snapshot_spectra = sorted({
        sp
        for sample in samples
        for sp in spectra_by_sample.get(sample, [])
        if is_sample_independent_spectrum(sp)
    })

    if all_snapshot_spectra:
        print(f"\n=== {SNAPSHOT_SPECTRA_LABEL} | {len(all_snapshot_spectra)} spectra ===")
    for spectrum in all_snapshot_spectra:
        rep_sample = choose_representative_sample_for_spectrum(spectrum, samples, spectra_by_sample)
        if rep_sample is None:
            print(f"[skip] no representative sample found for {spectrum}")
            continue
        # The representative sample is only an HDF5 storage location.  It is not
        # written into the figure title for P_dd, P_dtp, P_tptp, etc.
        plot_one_task(rep_sample, spectrum, manifest, label_for_print=SNAPSHOT_SPECTRA_LABEL)

    return pd.DataFrame(manifest, columns=["sample", "spectrum", "figure_type"])

# %% code cell 18
manifest_pk = plot_everything() if RUN_PK_PLOTS else pd.DataFrame(
    columns=["sample", "spectrum", "figure_type"]
)
manifest_aia = plot_all_aia() if RUN_AIA_PLOTS else pd.DataFrame(
    columns=["sample", "method", "figure_type"]
)

manifest = {
    "pk": manifest_pk,
    "aia": manifest_aia,
}
manifest

# %% [markdown] cell 19
# ## Notes - Galaxy-dependent spectra such as `P_gg`, `P_dg`, `P_dE`, `P_gE`, `P_EE`, and `P_BB` are plotted separately for each galaxy sample. - Snapshot-particle spectra such as `P_dd`, `P_dtp`, and `P_tptp` are plotted only once and are saved under `figures_all_pk/snapshot_particles/`. - These snapshot-particle spectra may be stored under several sample groups in the HDF5 file, but the sample name is only a storage location and is not a physical label. - The two IA-amplitude estimators are: ```

# %% code cell 20
path = pks_path("GR", 18)

with h5py.File(path, "r") as f:
    def visit(name, obj):
        if "k" in name or "P_gg" in name:
            print(name, getattr(obj, "shape", None))
    f["LRG"].visititems(visit)

# %% code cell 21
# IPython-only: !pwd

# %% code cell 22

# %% code cell 23

# %% code cell 24

# %% code cell 25
