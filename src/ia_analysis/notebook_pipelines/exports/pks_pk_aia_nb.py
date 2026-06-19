"""Exported code from notebooks/raw_20260618/pks_PK_AIA.ipynb.

This file is generated for project management and refactoring. Review before running end to end.
"""


# %% [markdown] cell 1
# # Publication-ready $P(k)$ and $A_{\rm IA}$ plots This notebook focuses only on $P(k)$ and $A_{\rm IA}(k)$ plotting. Main plotting choices: - $0.2 \le k \le 20\,h\,\mathrm{Mpc}^{-1}$. - Curves are interpolated onto a logarithmic $k$-grid and smoothed with a conservative moving average. - $P(k)$ and $A_{\rm IA}$ uncertainties are drawn as shaded bands. - Uncertainties are obtained by calling `Cov.py`, following the workflow in `plot_all_pks.ipynb`. - No absolute values are taken for $A_{\rm IA}$.

# %% code cell 2

# ============================================================
# Imports and user configuration
# ============================================================
from pathlib import Path
import re
import sys
import warnings
import importlib

try:
    from arts import *
except Exception as exc:
    warnings.warn(f"Could not import `arts`; continuing with local plotting helpers. Reason: {exc}")

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
MODULE_DIR = Path.cwd()      # directory containing Cov.py
DATA_DIR = Path("../pks")
OUTPUT_DIR = Path("figures_PK_AIA")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Data selection
# ------------------------------------------------------------
FLAGS = ["GR", "F40", "F45", "F50", "F55", "F60"]
SNAPS = [ 8, 10, 12, 15, 18, 21]#1, 3, 6,

SAMPLES = None  # set to None to auto-discover
SPECTRA = ["P_gg", "P_dg", "P_dE", "P_gE", "P_EE", "P_dd"]

SOURCE_GROUP = "stitched_corr"
NOISE_GROUP = "stitched_noise"

# ------------------------------------------------------------
# Theoretical covariance / uncertainty from Cov.py
# ------------------------------------------------------------
# This follows the plot_all_pks.ipynb workflow:
#   1. import Cov.py from MODULE_DIR;
#   2. if the requested covariance is absent, call
#      Cov.write_covariance_hdf5_group(...);
#   3. read sqrt(diag(cov)) from sample/covariance/cov_<COV_KIND>.
USE_THEORY_COVARIANCE = True
FORCE_RECOMPUTE_COVARIANCE = False
COV_GROUP = "covariance"
COV_KIND = "gaussian"               # "total", "gaussian", "cNG", or "SSC"
COV_SOURCE = "measured_or_ccl_nla"
COV_AIA_DEFAULT = 1.0
COV_C1_RHOCRIT = 0.0134
COV_KMIN_FIT = 0.05
COV_KMAX_FIT = 0.30
COV_INCLUDE_CNG = False
COV_INCLUDE_SSC = False
COV_SSC_MODE = "periodic"
COV_SSC_BACKEND = "ccl_halomodel"
COV_HM_OPTIONS = None


# ------------------------------------------------------------
# Plotting range and smoothing
# ------------------------------------------------------------
KMIN_PLOT = 0.2
KMAX_PLOT = 20.0
NK_SMOOTH = 512
SMOOTH_K = np.logspace(np.log10(KMIN_PLOT), np.log10(KMAX_PLOT), NK_SMOOTH)

PLOT_K_TIMES_PK = True
# Display convention: do not take absolute values.  For IA cross spectra,
# plot -P_deltaE and -P_gE directly, matching the usual IA sign convention.
SIGN_FLIP_SPECTRA_FOR_DISPLAY = {"P_dE", "P_gE"}
USE_LOG_ABS_FOR_SIGNED = False
SHOW_UNCERTAINTY = True
UNCERTAINTY_STYLE = "shade"     # draw P(k) and A_IA errors as shaded bands
UNCERTAINTY_ALPHA = 0.20
# Use measured/covariance-based errors only. If no error dataset/covariance exists,
# the notebook will draw the central curve without inventing an artificial band.
USE_FALLBACK_ERROR_BAND = False
ERRORBAR_EVERY = 8

APPLY_MOVING_AVERAGE = True
MOVING_AVERAGE_WINDOW = 51      # odd integer; larger values suppress more sample/cosmic-variance wiggles
MOVING_AVERAGE_PASSES = 1
MOVING_AVERAGE_KERNEL = "hanning" # "boxcar" or "hanning"
SMOOTH_POSITIVE_IN_LOGY = True    # smooth positive spectra in log amplitude
SAFE_SMOOTH_CLIP_TO_LOCAL_RANGE = True  # prevent smoothing from creating larger local wiggles

# ------------------------------------------------------------
# Figure layout
# ------------------------------------------------------------
N_COLS_GRID = 3
FIGSIZE_GRID = (15.0, 11.4)
FIGSIZE_MODEL_EVOLUTION = (15.0, 8.8)
SAVE_FIGURES = True
SAVE_FORMATS = ("png",)
DPI = 300
SHOW_FIGURES = True
PER_SUBPLOT_REDSHIFT_COLORBAR = True

# ------------------------------------------------------------
# IA estimators and fitting ranges
# ------------------------------------------------------------
AIA_C1_RHOCRIT = 0.0134
AIA_USE_CCL_GROWTH = True
AIA_PLOT_ABS = False
AIA_USE_LOGLOG = True  # signed A_IA; linear y-axis avoids hiding negative points

FIT_KMIN = 0.2
FIT_KMAX = 20.0

AIA_METHODS_TO_USE = ("deltaE", "gE")
RUN_PK_PLOTS = True
RUN_AIA_PLOTS = True

# ------------------------------------------------------------
# Cosmology defaults if HDF5 metadata is absent
# ------------------------------------------------------------
COSMO_DEFAULTS = dict(
    Omega_c=0.2589,
    Omega_b=0.0486,
    h=0.6774,
    sigma8=0.8159,
    n_s=0.9667,
)
BOXSIZE_DEFAULT = 205.0
NMESH_DEFAULT = 512

# ------------------------------------------------------------
# Colours, labels, redshifts
# ------------------------------------------------------------
FR_COLORS = {
    "F4":   "#b84c4c",
    "F4.5": "#c08a4a",
    "F5":   "#a5ba9e",
    "F5.5": "#c5aade",
    "F6":   "#5f75a8",
    "GR":   "#2b2b2b",
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

METHOD_LINESTYLE = {
    "deltaE": "-",
    "gE": "--",
}
METHOD_LABEL = {
    "deltaE": r"$\delta E$",
    "gE": r"$gE$",
}

ZMAP = {
    0: 3.00, 1: 2.00, 2: 1.36, 3: 1.26, 4: 1.15, 5: 1.06,
    6: 0.97, 7: 0.88, 8: 0.80, 9: 0.73, 10: 0.65, 11: 0.58,
    12: 0.51, 13: 0.45, 14: 0.39, 15: 0.33, 16: 0.27, 17: 0.21,
    18: 0.16, 19: 0.10, 20: 0.05, 21: 0.00,
}

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": DPI,
    "font.size": 12,
    "axes.titlesize": 12.5,
    "axes.labelsize": 12.5,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10.2,
    "ytick.labelsize": 10.2,
    "axes.linewidth": 1.05,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "mathtext.fontset": "cm",
    "mathtext.default": "regular",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "cmr10", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ------------------------------------------------------------
# Import Cov.py from MODULE_DIR
# ------------------------------------------------------------
sys.path.insert(0, str(MODULE_DIR))
try:
    import Cov
    importlib.reload(Cov)
except Exception as exc:
    Cov = None
    warnings.warn(f"Could not import Cov.py from {MODULE_DIR}: {exc}")

# %% code cell 3

# ============================================================
# General utilities
# ============================================================
def output_safe_name(x):
    return str(x).replace("/", "_").replace(" ", "_").replace("$", "").replace("\\", "")


def format_z_title(z):
    if np.isfinite(z):
        return rf"$z={z:.2f}$"
    return r"$z=\,?$"



def compact_spectrum_label(spectrum):
    return rf"${spectrum_math_label(spectrum, include_k=False)}$" if "spectrum_math_label" in globals() else str(spectrum)


def compact_method_label(method):
    return METHOD_LABEL.get(method, str(method)) if "METHOD_LABEL" in globals() else str(method)


def subplot_title(*parts):
    """Compact subplot title assembled from non-empty components."""
    clean = [str(p) for p in parts if p is not None and str(p) != ""]
    return "  |  ".join(clean)


def flag_sort_key(flag):
    order = {f: i for i, f in enumerate(FLAGS)}
    return order.get(flag, 999)


def pks_path(flag, snap):
    snap = int(snap)
    candidates = [
        DATA_DIR / f"pks_{flag}_{snap:03d}.hdf5",
        DATA_DIR / f"pks_{flag}_{snap}.hdf5",
        DATA_DIR / f"{flag}_{snap:03d}.hdf5",
        DATA_DIR / f"{flag}_{snap}.hdf5",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def parse_flag_snap_from_path(path):
    name = Path(path).name
    m = re.search(r"(GR|F40|F45|F50|F55|F60).*?(\d{1,3})", name)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, stem, subdir=""):
    outdir = ensure_dir(OUTPUT_DIR / subdir)
    if SAVE_FIGURES:
        for fmt in SAVE_FORMATS:
            fig.savefig(outdir / f"{stem}.{fmt}", bbox_inches="tight")
    if not SHOW_FIGURES:
        plt.close(fig)


def make_model_legend(flags=FLAGS):
    return [
        Line2D([0], [0], color=FLAG_COLOR.get(flag, "0.4"), lw=2.0, label=FLAG_LABEL.get(flag, flag))
        for flag in flags
    ]


def apply_paper_grid(ax):
    ax.grid(True, which="major", alpha=0.18, lw=0.55)
    ax.grid(True, which="minor", alpha=0.055, lw=0.35)


def set_shared_labels(fig, xlabel, ylabel):
    fig.supxlabel(xlabel, y=0.035, fontsize=13)
    fig.supylabel(ylabel, x=0.035, fontsize=13)


def finite_median(x, fallback=np.nan):
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x)
    if np.count_nonzero(m) == 0:
        return fallback
    return float(np.nanmedian(x[m]))


def infer_cosmo_box_metadata(path, snap=None):
    cosmo = dict(COSMO_DEFAULTS)
    boxsize = BOXSIZE_DEFAULT
    nmesh = NMESH_DEFAULT
    z = np.nan

    with h5py.File(path, "r") as f:
        attrs_to_scan = [f.attrs]
        for key in f.keys():
            if isinstance(f[key], h5py.Group):
                attrs_to_scan.append(f[key].attrs)
        for attrs in attrs_to_scan:
            for key in list(cosmo.keys()):
                if key in attrs:
                    cosmo[key] = float(attrs[key])
            for key in ("boxsize", "BoxSize", "Lbox", "box_size"):
                if key in attrs:
                    boxsize = float(attrs[key])
            for key in ("nmesh", "Nmesh", "NmeshDefault"):
                if key in attrs:
                    nmesh = int(attrs[key])
            for key in ("z", "redshift", "Redshift"):
                if key in attrs:
                    z = float(attrs[key])
    if not np.isfinite(z) and snap is not None:
        z = ZMAP.get(int(snap), np.nan)
    return cosmo, boxsize, nmesh, z


def omega_m_from_cosmo(cosmo):
    return float(cosmo.get("Omega_c", COSMO_DEFAULTS["Omega_c"])) + float(cosmo.get("Omega_b", COSMO_DEFAULTS["Omega_b"]))


def growth_factor_fallback(z, omega_m0):
    """Approximate GR growth factor D(z), normalized to D(0)=1."""
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


def ia_F_of_z(z, cosmo):
    om = omega_m_from_cosmo(cosmo)
    D = growth_factor_D(z, cosmo)
    return 2.0 * float(AIA_C1_RHOCRIT) * om / (3.0 * D)


def aia_prefactor(z, cosmo):
    """A_IA(k)=prefactor*P_XE/P_Xdelta, with prefactor=-1/F(z)."""
    return -1.0 / ia_F_of_z(z, cosmo)

# %% code cell 4

# ============================================================
# HDF5 discovery and reading
# ============================================================
def _get_group(f, sample, source_group=SOURCE_GROUP):
    if sample in f and isinstance(f[sample], h5py.Group):
        if source_group in f[sample]:
            return f[sample][source_group]
        return f[sample]
    if source_group in f:
        return f[source_group]
    raise KeyError(f"Cannot find sample={sample!r}, source_group={source_group!r}")


def _dataset_candidates(spectrum):
    return [
        f"{spectrum}_Pk",
        f"{spectrum}_pk",
        f"{spectrum}_P",
        f"{spectrum}",
    ]


def read_spectrum(path, sample, spectrum, source_group=SOURCE_GROUP):
    """Read one raw P(k) spectrum from one HDF5 file."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        g = _get_group(f, sample, source_group=source_group)

        if "k" in g:
            k = np.asarray(g["k"][:], dtype=float)
        elif "k_center" in g:
            k = np.asarray(g["k_center"][:], dtype=float)
        else:
            raise KeyError(f"No k grid found in {path}:{g.name}")

        P = None
        used_name = None
        for name in _dataset_candidates(spectrum):
            if name in g:
                P = np.asarray(g[name][:], dtype=float)
                used_name = name
                break
        if P is None:
            raise KeyError(f"No dataset for {spectrum} in {path}:{g.name}")

        z = np.nan
        for attrs in (f.attrs, g.attrs):
            for key in ("z", "redshift", "Redshift"):
                if key in attrs:
                    z = float(attrs[key])
                    break
            if np.isfinite(z):
                break

    flag, snap = parse_flag_snap_from_path(path)
    if not np.isfinite(z) and snap is not None:
        z = ZMAP.get(int(snap), np.nan)

    return k, P, z, used_name


def read_spectrum_or_none(path, sample, spectrum, source_group=SOURCE_GROUP):
    try:
        return read_spectrum(path, sample, spectrum, source_group=source_group)
    except Exception:
        return None



def list_spectra(path, sample, source_group=SOURCE_GROUP):
    """List P_* spectra available under sample/source_group."""
    spectra = []
    try:
        with h5py.File(path, "r") as f:
            g = _get_group(f, sample, source_group=source_group)
            for key in g.keys():
                if key.startswith("P_") and key.endswith("_Pk"):
                    spectra.append(key[:-3])
                elif key.startswith("P_") and key not in ("P",):
                    if not key.endswith("_fold"):
                        spectra.append(key)
    except Exception:
        return []
    return sorted(set(spectra))

def discover_samples_and_spectra():
    """Discover available samples and spectra from existing HDF5 files."""
    existing = [pks_path(flag, snap) for flag in FLAGS for snap in SNAPS if pks_path(flag, snap).exists()]
    if len(existing) == 0:
        print(f"[warn] no HDF5 files found in {DATA_DIR.resolve()}")
        return [], {}

    samples_found = set()
    spectra_by_sample = {}

    for path in existing:
        try:
            with h5py.File(path, "r") as f:
                for sample in f.keys():
                    if not isinstance(f[sample], h5py.Group):
                        continue
                    if SOURCE_GROUP not in f[sample]:
                        continue
                    g = f[sample][SOURCE_GROUP]
                    spectra = []
                    for key in g.keys():
                        if key.startswith("P_") and key.endswith("_Pk"):
                            spectra.append(key[:-3])
                        elif key.startswith("P_") and key not in ("P",):
                            if not key.endswith("_fold"):
                                spectra.append(key)
                    if spectra:
                        samples_found.add(sample)
                        spectra_by_sample.setdefault(sample, set()).update(spectra)
        except Exception:
            continue

    samples = sorted(samples_found)
    spectra_by_sample = {s: sorted(v) for s, v in spectra_by_sample.items()}
    return samples, spectra_by_sample


def initialize_data_selection():
    discovered_samples, discovered_spectra = discover_samples_and_spectra()

    if SAMPLES is None:
        samples = discovered_samples
    else:
        samples = [s for s in SAMPLES if s in discovered_samples] or list(SAMPLES)

    spectra_by_sample = {}
    for sample in samples:
        if SPECTRA is None:
            spectra_by_sample[sample] = discovered_spectra.get(sample, [])
        else:
            spectra_by_sample[sample] = list(SPECTRA)

    return samples, spectra_by_sample


samples, spectra_by_sample = initialize_data_selection()
print("Samples:", samples)
print("Spectra by sample:", spectra_by_sample)


_COV_FAILURE_CACHE = set()


def pk_key_for_cov(spectrum):
    """Convert plot key P_dE -> Cov.py key dE."""
    s = str(spectrum)
    return s[2:] if s.startswith("P_") else s


def available_cov_pk_types(path, sample):
    """Return Cov.py pk-type names present in sample/source group."""
    specs = list_spectra(path, sample, source_group=SOURCE_GROUP)
    keys = [pk_key_for_cov(s) for s in specs]
    if Cov is not None and hasattr(Cov, "PK_TYPE_MAP"):
        allowed = set(Cov.PK_TYPE_MAP)
        keys = [k for k in keys if k in allowed]
    return sorted(set(keys))


def covariance_group_has_spectrum(h5_sample_group, spectrum):
    """Check whether sample/covariance already contains the requested spectrum."""
    if COV_GROUP not in h5_sample_group:
        return False
    g = h5_sample_group[COV_GROUP]
    if f"cov_{COV_KIND}" not in g:
        return False
    if "spec_labels" not in g:
        return False
    labels = [x.decode() if isinstance(x, bytes) else str(x) for x in g["spec_labels"][()]]
    key = pk_key_for_cov(spectrum)
    return (key in labels) or (("P_" + key) in labels)


def ensure_covariance(path, sample, spectrum, snap=None):
    """Make sure path/sample/covariance exists and contains the requested spectrum.

    This intentionally mirrors the older plot_all_pks.ipynb workflow and calls
    Cov.write_covariance_hdf5_group when the covariance block is missing.
    """
    if not USE_THEORY_COVARIANCE:
        return False
    if Cov is None:
        return False

    cache_key = (str(path), sample)
    if cache_key in _COV_FAILURE_CACHE:
        return False

    try:
        with h5py.File(path, "r") as f:
            if sample not in f:
                return False
            if (not FORCE_RECOMPUTE_COVARIANCE) and covariance_group_has_spectrum(f[sample], spectrum):
                return True
    except Exception:
        return False

    pk_types = available_cov_pk_types(path, sample)
    if len(pk_types) == 0:
        return False

    cosmo, boxsize, nmesh, z = infer_cosmo_box_metadata(path, snap=snap)

    try:
        with h5py.File(path, "r+") as f:
            Cov.write_covariance_hdf5_group(
                f[sample],
                pk_types=pk_types,
                cosmo_params=cosmo,
                z=z,
                boxsize=boxsize,
                nmesh=nmesh,
                source=COV_SOURCE,
                aia_default=COV_AIA_DEFAULT,
                c1_rhocrit=COV_C1_RHOCRIT,
                kmin_fit=COV_KMIN_FIT,
                kmax_fit=COV_KMAX_FIT,
                include_cng=COV_INCLUDE_CNG,
                include_ssc=COV_INCLUDE_SSC,
                ssc_mode=COV_SSC_MODE,
                ssc_backend=COV_SSC_BACKEND,
                hm_options=COV_HM_OPTIONS,
                input_group=SOURCE_GROUP,
                noise_group=NOISE_GROUP,
                output_group=COV_GROUP,
            )
        return True
    except Exception as exc:
        _COV_FAILURE_CACHE.add(cache_key)
        print(f"[warn] failed to compute covariance for {Path(path).name}/{sample}: {exc}")
        return False


def read_covariance_sigma(path, sample, spectrum, snap=None):
    """Read sqrt(diag(cov)) for a single spectrum. Returns (k_cov, sigma_P)."""
    if not SHOW_UNCERTAINTY or not USE_THEORY_COVARIANCE:
        return None, None
    if not ensure_covariance(path, sample, spectrum, snap=snap):
        return None, None

    try:
        with h5py.File(path, "r") as f:
            g = f[sample][COV_GROUP]
            cov_name = f"cov_{COV_KIND}"
            if cov_name not in g:
                return None, None
            labels = [x.decode() if isinstance(x, bytes) else str(x) for x in g["spec_labels"][()]]
            key = pk_key_for_cov(spectrum)
            if key in labels:
                ispec = labels.index(key)
            elif ("P_" + key) in labels:
                ispec = labels.index("P_" + key)
            else:
                return None, None
            kcov = np.asarray(g["k"], dtype=float)
            cov = np.asarray(g[cov_name], dtype=float)
            nk = len(kcov)
            block = cov[ispec*nk:(ispec+1)*nk, ispec*nk:(ispec+1)*nk]
            diag = np.diag(block)
            sigma = np.sqrt(np.where(diag >= 0, diag, np.nan))
            return kcov, sigma
    except Exception as exc:
        print(f"[warn] failed to read covariance for {Path(path).name}/{sample}/{spectrum}: {exc}")
        return None, None


def read_existing_sigma_or_covariance(path, sample, spectrum, k_ref, snap=None):
    """Fallback reader for already-stored sigma/covariance layouts."""
    path = Path(path)
    k_ref = np.asarray(k_ref, dtype=float)
    try:
        with h5py.File(path, "r") as f:
            possible_groups = []
            try:
                possible_groups.append(_get_group(f, sample, SOURCE_GROUP))
            except Exception:
                pass
            if sample in f and COV_GROUP in f[sample]:
                possible_groups.append(f[sample][COV_GROUP])
            if COV_GROUP in f:
                possible_groups.append(f[COV_GROUP])

            for g in possible_groups:
                for name in (f"{spectrum}_sigma", f"sigma_{spectrum}", f"{spectrum}_err", f"err_{spectrum}"):
                    if name in g:
                        arr = np.asarray(g[name][:], dtype=float)
                        if arr.shape == k_ref.shape:
                            return arr

                if spectrum in g and isinstance(g[spectrum], h5py.Group):
                    sg = g[spectrum]
                    for name in ("sigma", "err", "error"):
                        if name in sg:
                            arr = np.asarray(sg[name][:], dtype=float)
                            if arr.shape == k_ref.shape:
                                return arr
                    for name in (COV_KIND, "cov", "covariance", "total", "gaussian"):
                        if name in sg:
                            mat = np.asarray(sg[name][:], dtype=float)
                            if mat.ndim == 2:
                                return np.sqrt(np.clip(np.diag(mat), 0, np.inf))
                            if mat.ndim == 1 and mat.shape == k_ref.shape:
                                return np.sqrt(np.clip(mat, 0, np.inf))
    except Exception:
        pass
    return None


def sigma_for_spectrum(path, sample, spectrum, k_ref, snap=None):
    """Return sigma_P(k) for a spectrum using Cov.py first.

    Priority:
        1. Cov.py theoretical covariance written/read from sample/covariance.
        2. Existing sigma/covariance datasets already stored in the file.
        3. None, unless USE_FALLBACK_ERROR_BAND=True is used by the caller.
    """
    k_ref = np.asarray(k_ref, dtype=float)

    kcov, sigma = read_covariance_sigma(path, sample, spectrum, snap=snap)
    if kcov is not None and sigma is not None:
        if len(kcov) == len(k_ref) and np.allclose(kcov, k_ref, rtol=1e-5, atol=1e-10):
            return sigma
        return np.interp(k_ref, kcov, sigma, left=np.nan, right=np.nan)

    return read_existing_sigma_or_covariance(path, sample, spectrum, k_ref, snap=snap)


def fallback_sigma(y):
    """Conservative diagonal uncertainty fallback if no covariance exists."""
    y = np.asarray(y, dtype=float)
    amp = finite_median(np.abs(y), fallback=1.0)
    return np.full_like(y, max(0.10 * amp, 1e-30), dtype=float)

# %% code cell 5

# ============================================================
# Smoothing and plotting helpers
# ============================================================
def interpolate_logk(k_src, y_src, k_tgt=SMOOTH_K):
    k_src = np.asarray(k_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    k_tgt = np.asarray(k_tgt, dtype=float)

    good = np.isfinite(k_src) & np.isfinite(y_src) & (k_src > 0)
    if np.count_nonzero(good) < 2:
        return np.full_like(k_tgt, np.nan, dtype=float)

    x = np.log(k_src[good])
    y = y_src[good]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    xu, idx = np.unique(x, return_index=True)
    yu = y[idx]

    if len(xu) < 2:
        return np.full_like(k_tgt, np.nan, dtype=float)

    return np.interp(np.log(k_tgt), xu, yu, left=np.nan, right=np.nan)


def _local_window_bounds(y, window):
    """Return local min/max arrays on a finite sequence for safe clipping."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    half = int(window) // 2
    lo = np.empty(n, dtype=float)
    hi = np.empty(n, dtype=float)
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        yy = y[a:b]
        lo[i] = np.nanmin(yy)
        hi[i] = np.nanmax(yy)
    return lo, hi


def moving_average_nan(y, window=MOVING_AVERAGE_WINDOW, kernel=MOVING_AVERAGE_KERNEL):
    """NaN-aware moving average with optional local-range clipping.

    The clipping step prevents the smoother from creating a new larger local
    oscillation than the original curve in the same window.
    """
    y = np.asarray(y, dtype=float)
    out = y.copy()
    good = np.isfinite(y)
    n = int(np.count_nonzero(good))
    if n < 3:
        return out

    win = int(window)
    if win % 2 == 0:
        win += 1
    win = max(3, min(win, n if n % 2 == 1 else n - 1))
    if win < 3:
        return out

    if str(kernel).lower() == "hanning" and win >= 5:
        ker = np.hanning(win)
    else:
        ker = np.ones(win, dtype=float)
    ker = ker / np.sum(ker)

    yy = y[good]
    pad = win // 2
    yy_pad = np.pad(yy, pad_width=pad, mode="edge")
    sm = np.convolve(yy_pad, ker, mode="valid")

    if SAFE_SMOOTH_CLIP_TO_LOCAL_RANGE:
        lo, hi = _local_window_bounds(yy, win)
        sm = np.clip(sm, lo, hi)

    out[good] = sm
    return out


def smooth_curve(y):
    """Smooth a plotted curve without overfitting.

    Positive curves are smoothed in log-amplitude, which damps fractional
    wiggles and avoids creating negative values.  Signed curves are smoothed in
    linear space.  In both cases the result is clipped to the local input range.
    """
    y = np.asarray(y, dtype=float)
    if not APPLY_MOVING_AVERAGE:
        return y

    good = np.isfinite(y)
    if np.count_nonzero(good) < 3:
        return y

    use_log = bool(SMOOTH_POSITIVE_IN_LOGY) and np.all(y[good] > 0)
    if use_log:
        out = y.copy()
        work = np.log(y[good])
        for _ in range(max(1, int(MOVING_AVERAGE_PASSES))):
            work = moving_average_nan(work)
        out[good] = np.exp(work)
        return out

    out = y.copy()
    for _ in range(max(1, int(MOVING_AVERAGE_PASSES))):
        out = moving_average_nan(out)
    return out



def resample_smooth(k, y, sigma=None, target_k=SMOOTH_K, smooth=True):
    yt = interpolate_logk(k, y, target_k)
    if smooth:
        yt = smooth_curve(yt)

    if sigma is None:
        return np.asarray(target_k, dtype=float), yt, None

    st = interpolate_logk(k, sigma, target_k)
    return np.asarray(target_k, dtype=float), yt, st


def prepare_signed_y(k, P, sigma=None, multiply_by_k=PLOT_K_TIMES_PK):
    k = np.asarray(k, dtype=float)
    P = np.asarray(P, dtype=float)
    y = k * P if multiply_by_k else P
    sig = None if sigma is None else (k * np.asarray(sigma, dtype=float) if multiply_by_k else np.asarray(sigma, dtype=float))

    if USE_LOG_ABS_FOR_SIGNED:
        y_plot = np.abs(y)
    else:
        y_plot = y

    good = np.isfinite(k) & np.isfinite(y_plot)
    if USE_LOG_ABS_FOR_SIGNED:
        good &= (k > 0) & (y_plot > 0)

    return y, y_plot, sig, good


def spectrum_math_label(spectrum, include_k=True):
    mapping = {
        "P_gg": r"P_{gg}",
        "P_dg": r"P_{\delta g}",
        "P_dE": r"P_{\delta E}",
        "P_gE": r"P_{gE}",
        "P_EE": r"P_{EE}",
        "P_dd": r"P_{\delta\delta}",
        "P_dtp": r"P_{\delta\theta}",
        "P_tptp": r"P_{\theta\theta}",
        "P_tpE": r"P_{\theta E}",
    }
    base = mapping.get(spectrum, spectrum.replace("_", r"\_"))
    if include_k:
        return rf"{base}(k)"
    return base


def pk_ylabel(spectrum):
    label = spectrum_math_label(spectrum, include_k=True)
    if PLOT_K_TIMES_PK:
        return rf"$k\,{label}\;[(h^{{-1}}\,\mathrm{{Mpc}})^2]$"
    return rf"${label}\;[(h^{{-1}}\,\mathrm{{Mpc}})^3]$"


def draw_series_with_band(ax, k, y, sigma=None, color="0.3", label=None, ls="-", lw=1.55, logy=True):
    k = np.asarray(k, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(k) & np.isfinite(y)
    if logy:
        good &= (k > 0) & (y > 0)
    if not np.any(good):
        return False

    ax.plot(k[good], y[good], color=color, ls=ls, lw=lw, label=label, zorder=2)

    if SHOW_UNCERTAINTY and sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        band = good & np.isfinite(sigma)
        if np.any(band):
            style = str(UNCERTAINTY_STYLE).lower()
            if style in ("shade", "band", "fill", "fill_between", "both"):
                if logy:
                    positive = y[band][y[band] > 0]
                    floor = np.nanmin(positive) * 0.2 if positive.size else 1e-300
                    low = np.maximum(y[band] - sigma[band], floor)
                    high = y[band] + sigma[band]
                    ok = np.isfinite(low) & np.isfinite(high) & (low > 0)
                    if np.any(ok):
                        ax.fill_between(k[band][ok], low[ok], high[ok], color=color,
                                        alpha=UNCERTAINTY_ALPHA, linewidth=0, zorder=1)
                else:
                    ax.fill_between(k[band], y[band] - sigma[band], y[band] + sigma[band],
                                    color=color, alpha=UNCERTAINTY_ALPHA, linewidth=0, zorder=1)

            if style in ("errorbar", "errorbars", "err", "both"):
                idx = np.where(band)[0]
                if ERRORBAR_EVERY and int(ERRORBAR_EVERY) > 1:
                    idx = idx[::int(ERRORBAR_EVERY)]
                if idx.size:
                    ax.errorbar(k[idx], y[idx], yerr=sigma[idx], fmt="none",
                                ecolor=color, elinewidth=0.8, capsize=1.5,
                                capthick=0.8, alpha=0.65, zorder=3)
    return True

# %% code cell 6

# ============================================================
# P(k) plotting
# ============================================================
def load_plot_spectrum(path, sample, spectrum):
    got = read_spectrum_or_none(path, sample, spectrum)
    if got is None:
        return None
    k, P, z, _ = got

    # Do not take absolute values.  IA cross spectra are shown with an
    # explicit minus sign: -P_deltaE and -P_gE.
    if spectrum in SIGN_FLIP_SPECTRA_FOR_DISPLAY:
        P = -np.asarray(P, dtype=float)

    sigma = sigma_for_spectrum(path, sample, spectrum, k)
    if sigma is None and USE_FALLBACK_ERROR_BAND:
        sigma = fallback_sigma(P)
    kt, Pt, sigt = resample_smooth(k, P, sigma=sigma, target_k=SMOOTH_K, smooth=True)
    _, y_plot, yerr, good = prepare_signed_y(kt, Pt, sigma=sigt, multiply_by_k=PLOT_K_TIMES_PK)
    return dict(k=kt, P=Pt, y=y_plot, sigma=yerr, good=good, z=z)


def plot_pk_model_grid(sample, spectrum, flags=FLAGS, snaps=SNAPS):
    ncols = int(N_COLS_GRID)
    nrows = int(np.ceil(len(snaps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE_GRID, sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, snap in zip(axes, snaps):
        z = ZMAP.get(int(snap), np.nan)
        plotted = False
        for flag in flags:
            path = pks_path(flag, snap)
            if not path.exists():
                continue
            res = load_plot_spectrum(path, sample, spectrum)
            if res is None:
                continue
            good = res["good"]
            if np.any(good):
                draw_series_with_band(
                    ax, res["k"], res["y"], sigma=res["sigma"],
                    color=FLAG_COLOR.get(flag, "0.4"),
                    label=FLAG_LABEL.get(flag, flag),
                    logy=True,
                )
                plotted = True

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(subplot_title(format_z_title(z), compact_spectrum_label(spectrum)), pad=5)
        apply_paper_grid(ax)
        ax.set_xlim(KMIN_PLOT, KMAX_PLOT)
        if not plotted:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="0.5")

    for ax in axes[len(snaps):]:
        ax.axis("off")

    fig.legend(handles=make_model_legend(flags), loc="upper center",
               ncol=min(len(flags), 6), frameon=False, bbox_to_anchor=(0.5, 1.01))
    set_shared_labels(fig, r"$k\,[h\,\mathrm{Mpc}^{-1}]$", pk_ylabel(spectrum))
    fig.tight_layout(rect=(0.04, 0.04, 1.0, 0.97))

    stem = f"{output_safe_name(sample)}_{output_safe_name(spectrum)}_pk_model_grid"
    save_figure(fig, stem, subdir=f"{output_safe_name(sample)}/pk")
    return fig


def add_redshift_colorbar_to_axis(fig, ax, sm):
    """Attach a redshift colourbar to the right side of one subplot."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.8%", pad=0.045)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$z$", fontsize=9.5, labelpad=2)
    cbar.ax.tick_params(labelsize=8.0, direction="in", length=2.5)
    return cbar


def plot_pk_redshift_evolution_by_model(sample, spectrum, flags=FLAGS, snaps=SNAPS):
    ncols = 3
    nrows = int(np.ceil(len(flags) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE_MODEL_EVOLUTION, sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    zvals = np.array([ZMAP.get(int(s), np.nan) for s in snaps], dtype=float)
    norm = Normalize(vmin=np.nanmin(zvals), vmax=np.nanmax(zvals))
    cmap = plt.get_cmap("turbo")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    for ax, flag in zip(axes, flags):
        plotted = False
        for snap in snaps:
            path = pks_path(flag, snap)
            if not path.exists():
                continue
            res = load_plot_spectrum(path, sample, spectrum)
            if res is None:
                continue
            z = ZMAP.get(int(snap), res["z"])
            color = cmap(norm(z))
            if np.any(res["good"]):
                ax.plot(res["k"][res["good"]], res["y"][res["good"]],
                        color=color, lw=1.45)
                plotted = True

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(KMIN_PLOT, KMAX_PLOT)
        ax.set_title(subplot_title(FLAG_LABEL.get(flag, flag), compact_spectrum_label(spectrum)), pad=5)
        apply_paper_grid(ax)
        if PER_SUBPLOT_REDSHIFT_COLORBAR:
            add_redshift_colorbar_to_axis(fig, ax, sm)
        if not plotted:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="0.5")

    for ax in axes[len(flags):]:
        ax.axis("off")

    set_shared_labels(fig, r"$k\,[h\,\mathrm{Mpc}^{-1}]$", pk_ylabel(spectrum))
    fig.tight_layout(rect=(0.04, 0.04, 1.0, 0.98))

    stem = f"{output_safe_name(sample)}_{output_safe_name(spectrum)}_pk_redshift_evolution"
    save_figure(fig, stem, subdir=f"{output_safe_name(sample)}/pk")
    return fig



def plot_all_pk(samples=samples, spectra_by_sample=spectra_by_sample):
    manifest = []
    for sample in samples:
        for spectrum in spectra_by_sample.get(sample, []):
            print(f"[P(k)] {sample}/{spectrum}")
            try:
                plot_pk_model_grid(sample, spectrum)
                manifest.append((sample, spectrum, "model_grid"))
            except Exception as exc:
                print(f"  [warn] model grid failed: {exc}")
            try:
                plot_pk_redshift_evolution_by_model(sample, spectrum)
                manifest.append((sample, spectrum, "redshift_evolution"))
            except Exception as exc:
                print(f"  [warn] redshift evolution failed: {exc}")
    return pd.DataFrame(manifest, columns=["sample", "spectrum", "figure_type"])

# %% code cell 7

# ============================================================
# A_IA estimators and plotting
# ============================================================
AIA_METHODS = {
    "deltaE": {
        "numerator": "P_dE",
        "denominator": "P_dd",
        "label": r"$\delta E$",
        "aia_label": r"$A_{\rm IA}^{\delta E}$",
        "directory": "deltaE",
    },
    "gE": {
        "numerator": "P_gE",
        "denominator": "P_dg",
        "label": r"$gE$",
        "aia_label": r"$A_{\rm IA}^{gE}$",
        "directory": "gE",
    },
}


def sample_has_method(sample, method):
    spectra = set(spectra_by_sample.get(sample, []))
    cfg = AIA_METHODS[method]
    return cfg["numerator"] in spectra and cfg["denominator"] in spectra


def read_aia_estimator(path, sample, method, snap=None):
    cfg = AIA_METHODS[method]
    got_num = read_spectrum_or_none(path, sample, cfg["numerator"])
    got_den = read_spectrum_or_none(path, sample, cfg["denominator"])
    if got_num is None or got_den is None:
        return None

    k, Pnum, z, _ = got_num
    kd, Pden, zd, _ = got_den
    if kd.shape != k.shape or not np.allclose(kd, k, rtol=1e-4, atol=1e-8):
        Pden = np.interp(k, kd, Pden, left=np.nan, right=np.nan)

    if not np.isfinite(z):
        z = zd
    if not np.isfinite(z) and snap is not None:
        z = ZMAP.get(int(snap), np.nan)

    cosmo, _, _, zmeta = infer_cosmo_box_metadata(path, snap=snap)
    if not np.isfinite(z):
        z = zmeta

    pref = aia_prefactor(z, cosmo)
    # pref = -1/F(z), so this is the direct signed IA convention:
    # A_IA(k) = - P_XE(k) / [F(z) P_Xdelta(k)].
    # No absolute value is taken later in the plotting stage.
    with np.errstate(divide="ignore", invalid="ignore"):
        A = pref * Pnum / Pden

    sig_num = sigma_for_spectrum(path, sample, cfg["numerator"], k, snap=snap)
    sig_den = sigma_for_spectrum(path, sample, cfg["denominator"], k, snap=snap)
    sig_A = None
    if sig_num is not None and sig_den is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            sig_A = abs(pref) * np.sqrt((sig_num / Pden)**2 + (Pnum * sig_den / Pden**2)**2)
    elif USE_FALLBACK_ERROR_BAND:
        sig_A = fallback_sigma(A)

    return dict(k=k, A=A, sigma=sig_A, z=z)


def load_plot_aia(path, sample, method, snap=None):
    res = read_aia_estimator(path, sample, method, snap=snap)
    if res is None:
        return None
    k, A, sigma = res["k"], res["A"], res["sigma"]
    kt, At, sigt = resample_smooth(k, A, sigma=sigma, target_k=SMOOTH_K, smooth=True)
    y = At
    good = np.isfinite(kt) & np.isfinite(y)
    if AIA_USE_LOGLOG:
        good &= (kt > 0) & (y > 0)
    return dict(k=kt, y=y, sigma=sigt, good=good, z=res["z"])


def apply_aia_axis_style(ax):
    ax.set_xscale("log")
    if AIA_USE_LOGLOG:
        ax.set_yscale("log")
    ax.set_xlim(KMIN_PLOT, KMAX_PLOT)
    apply_paper_grid(ax)


def plot_aia_model_grid(sample, method, flags=FLAGS, snaps=SNAPS):
    ncols = int(N_COLS_GRID)
    nrows = int(np.ceil(len(snaps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE_GRID, sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, snap in zip(axes, snaps):
        z = ZMAP.get(int(snap), np.nan)
        plotted = False
        for flag in flags:
            path = pks_path(flag, snap)
            if not path.exists():
                continue
            res = load_plot_aia(path, sample, method, snap=snap)
            if res is None:
                continue
            if np.any(res["good"]):
                draw_series_with_band(ax, res["k"], res["y"], sigma=res["sigma"],
                                      color=FLAG_COLOR.get(flag, "0.4"),
                                      label=FLAG_LABEL.get(flag, flag),
                                      logy=AIA_USE_LOGLOG)
                plotted = True
        apply_aia_axis_style(ax)
        ax.set_title(subplot_title(format_z_title(z), METHOD_LABEL.get(method, method)), pad=5)
        if not plotted:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="0.5")

    for ax in axes[len(snaps):]:
        ax.axis("off")

    fig.legend(handles=make_model_legend(flags), loc="upper center",
               ncol=min(len(flags), 6), frameon=False, bbox_to_anchor=(0.5, 1.01))
    ylabel = AIA_METHODS[method]["aia_label"]
    set_shared_labels(fig, r"$k\,[h\,\mathrm{Mpc}^{-1}]$", ylabel)
    fig.tight_layout(rect=(0.04, 0.04, 1.0, 0.97))

    stem = f"{output_safe_name(sample)}_{method}_AIA_model_grid"
    save_figure(fig, stem, subdir=f"{output_safe_name(sample)}/aia/{method}")
    return fig


def plot_aia_method_comparison(sample, flag, snaps=SNAPS):
    methods = [m for m in AIA_METHODS_TO_USE if sample_has_method(sample, m)]
    if len(methods) < 2:
        return None

    ncols = int(N_COLS_GRID)
    nrows = int(np.ceil(len(snaps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE_GRID, sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, snap in zip(axes, snaps):
        z = ZMAP.get(int(snap), np.nan)
        path = pks_path(flag, snap)
        plotted = False
        if path.exists():
            for method in methods:
                res = load_plot_aia(path, sample, method, snap=snap)
                if res is None:
                    continue
                draw_series_with_band(ax, res["k"], res["y"], sigma=res["sigma"],
                                      color=FLAG_COLOR.get(flag, "0.4"),
                                      label=METHOD_LABEL.get(method, method),
                                      ls=METHOD_LINESTYLE.get(method, "-"),
                                      logy=AIA_USE_LOGLOG)
                plotted = True
        apply_aia_axis_style(ax)
        ax.set_title(subplot_title(format_z_title(z), FLAG_LABEL.get(flag, flag)), pad=5)
        if not plotted:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="0.5")

    for ax in axes[len(snaps):]:
        ax.axis("off")

    handles = [Line2D([0], [0], color="0.25", ls=METHOD_LINESTYLE[m], lw=2.0,
                      label=METHOD_LABEL.get(m, m)) for m in methods]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 1.01))
    ylabel = r"$A_{\rm IA}(k)$"
    set_shared_labels(fig, r"$k\,[h\,\mathrm{Mpc}^{-1}]$", ylabel)
    fig.tight_layout(rect=(0.04, 0.04, 1.0, 0.97))

    stem = f"{output_safe_name(sample)}_{FLAG_LABEL.get(flag, flag)}_AIA_deltaE_gE"
    save_figure(fig, stem, subdir=f"{output_safe_name(sample)}/aia/method_comparison")
    return fig

# %% code cell 8

# ============================================================
# Run only P(k) and A_IA plotting products
# ============================================================
def run_pk_products():
    if not RUN_PK_PLOTS:
        return pd.DataFrame(columns=["sample", "spectrum", "figure_type"])

    rows = []
    for sample in samples:
        for spectrum in spectra_by_sample.get(sample, []):
            if spectrum not in SPECTRA:
                continue
            print(f"[P(k)] {sample}/{spectrum}")
            try:
                plot_pk_model_grid(sample, spectrum)
                rows.append((sample, spectrum, "model_grid"))
            except Exception as exc:
                print(f"  [warn] P(k) model grid failed: {exc}")
            try:
                plot_pk_redshift_evolution_by_model(sample, spectrum)
                rows.append((sample, spectrum, "redshift_evolution"))
            except Exception as exc:
                print(f"  [warn] P(k) redshift evolution failed: {exc}")
    return pd.DataFrame(rows, columns=["sample", "spectrum", "figure_type"])


def run_aia_products():
    if not RUN_AIA_PLOTS:
        return pd.DataFrame(columns=["sample", "method_or_flag", "figure_type"])

    rows = []
    for sample in samples:
        methods = [m for m in AIA_METHODS_TO_USE if sample_has_method(sample, m)]
        if not methods:
            print(f"[skip] {sample}: no complete A_IA method")
            continue

        for method in methods:
            print(f"[A_IA] {sample}/{method}")
            try:
                plot_aia_model_grid(sample, method)
                rows.append((sample, method, "aia_model_grid"))
            except Exception as exc:
                print(f"  [warn] A_IA model grid failed: {exc}")

        for flag in FLAGS:
            try:
                plot_aia_method_comparison(sample, flag)
                rows.append((sample, flag, "aia_deltaE_gE_comparison"))
            except Exception as exc:
                print(f"  [warn] A_IA method comparison failed for {sample}/{flag}: {exc}")

    return pd.DataFrame(rows, columns=["sample", "method_or_flag", "figure_type"])


manifest = {
    "pk": run_pk_products(),
    "aia": run_aia_products(),
}
manifest

# %% [markdown] cell 9
# ## Notes - The uncertainty bands now follow the `plot_all_pks.ipynb` method: ```python import Cov Cov.write_covariance_hdf5_group(...) ``` If the covariance block is absent, the notebook writes `sample/covariance/cov_<COV_KIND>` into the HDF5 file. - $P_{\delta E}$ and $P_{gE}$ are displayed as $-P_{\delta E}$ and $-P_{gE}$; no `abs()` is used. - $A_{\rm IA}$ is plotted as the signed estimator: \[ A_{\rm IA}(k)=-\frac{P_{XE}(k)}{F(z)P_{X\delta}(k)}. \] - Error bands are drawn with: ```python SHO

# %% code cell 10
