"""Exported code from notebooks/raw_20260618/plot_all_pks.ipynb.

This file is generated for project management and refactoring. Review before running end to end.
"""


# %% [markdown] cell 1
# # Plot all available $P(k)$ spectra This notebook scans all `pks_{FLAG}_{SNAP}.hdf5` files, discovers every available `P_*` spectrum in each selected sample, reads `Cov.py` from the notebook directory, computes/reads theoretical covariance inside the notebook, and produces publication-style figures. Main outputs: 1. **For each spectrum:** a 3×3 redshift grid comparing gravity models at fixed redshift. 2. **For each spectrum:** same-model redshift evolution. 3. Optional **enhancement relative to 

# %% code cell 2

from pathlib import Path
import re
import sys
import warnings
import importlib

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -------------------------------------------------
# User configuration
# -------------------------------------------------
DATA_DIR = Path("/cosma/home/dp203/dc-wang17/IA_analysis/pks")
MODULE_DIR = Path.cwd()      # directory containing Cov.py
OUTPUT_DIR = Path("figures_all_pk")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# If None, scan all matching files in DATA_DIR.
# Otherwise use explicit lists below.
FLAGS = ["GR", "F40", "F45", "F50", "F55", "F60"]
SNAPS = [1, 3, 6, 8, 10, 12, 15, 18, 21]

# Which samples to plot.
# None means all HDF5 sample groups that contain SOURCE_GROUP.
# For a compact first run, set SAMPLES = ["all"].
SAMPLES = ['all']#None

# Which spectra to plot.
# None means every P_* spectrum discovered in the selected sample(s).
SPECTRA = ['P_dd','P_dE','P_EE','P_gg','P_dg','P_gE']

SOURCE_GROUP = "stitched_corr"   # "stitched_corr", "stitched_raw", or "stitched_noise"
NOISE_GROUP = "stitched_noise"
PLOT_K_TIMES_PK = True

# -------------------------------------------------
# Theoretical covariance / uncertainty
# -------------------------------------------------
# The notebook imports Cov.py and writes/reads sample/covariance inside each HDF5 file.
# ia_pk_cs.py still only measures P(k).
USE_THEORY_COVARIANCE = True
FORCE_RECOMPUTE_COVARIANCE = False
COV_GROUP = "covariance"
COV_KIND = "gaussian"               # "total", "gaussian", "cNG", or "SSC"
COV_SOURCE = "measured_or_ccl_nla"
COV_AIA_DEFAULT = 1.0
COV_C1_RHOCRIT = 0.0134
COV_KMIN_FIT = 0.05
COV_KMAX_FIT = 0.30
COV_INCLUDE_CNG = False #True
COV_INCLUDE_SSC = False #True
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
BOXSIZE_DEFAULT = 302.6          # Mpc/h for TNG300-like boxes
NMESH_DEFAULT = 512

# How to draw uncertainty.
SHOW_UNCERTAINTY = True
UNCERTAINTY_STYLE = "shade"      # default: shaded band. Use "errorbar" only if needed.
UNCERTAINTY_ALPHA = 0.20

# Display and saving controls.
SHOW_FIGURES = True
SAVE_FIGURES = False #True
SAVE_FORMATS = ("png", "pdf")
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

# Optional: restrict k plotting range. Use None for automatic.
XLIM = None
YLIM = None

# Optional enhancement plots relative to GR.
MAKE_ENHANCEMENT_RELATIVE_TO_GR = True

# f(R) color convention.
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

# %% code cell 4

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
    good = np.isfinite(k) & np.isfinite(yplot) & (yplot > 0)
    if sy is not None:
        good &= np.isfinite(sy)
    return y, yplot, sy, neg, good


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


def ensure_covariance(path, sample, spectrum, snap=None):
    """Make sure path/sample/covariance exists and contains the requested spectrum."""
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
        print(f"[warn] failed to compute covariance for {path.name}/{sample}: {exc}")
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
        print(f"[warn] failed to read covariance for {path.name}/{sample}/{spectrum}: {exc}")
        return None, None


def sigma_for_plot_grid(path, sample, spectrum, k, snap=None):
    kcov, sigma = read_covariance_sigma(path, sample, spectrum, snap=snap)
    if kcov is None or sigma is None:
        return None
    if len(kcov) == len(k) and np.allclose(kcov, k, rtol=1e-5, atol=1e-10):
        return sigma
    return np.interp(k, kcov, sigma, left=np.nan, right=np.nan)


def draw_pk_series(ax, k, P, sigma, spectrum, color, label=None, lw=1.35, ms=3.3):
    y_signed, y_plot, yerr, neg, good = prepare_y(k, P, sigma=sigma)
    if not np.any(good):
        return False

    if SHOW_UNCERTAINTY and yerr is not None and np.any(np.isfinite(yerr[good])):
        if str(UNCERTAINTY_STYLE).lower().startswith("err"):
            ax.errorbar(k[good], y_plot[good], yerr=yerr[good],
                        fmt="-", ms=ms, lw=lw, capsize=2.2,
                        color=color, label=label)
        else:
            ax.plot(k[good], y_plot[good], ms=ms, lw=lw, color=color, label=label)
            low = y_plot - yerr
            high = y_plot + yerr
            band = good & np.isfinite(low) & np.isfinite(high)
            if np.any(band):
                positive_y = y_plot[band]
                floor = np.nanmin(positive_y[positive_y > 0]) * 0.2 if np.any(positive_y > 0) else 1e-300
                low = np.maximum(low, floor)
                ax.fill_between(k[band], low[band], high[band],
                                color=color, alpha=UNCERTAINTY_ALPHA, linewidth=0)
    else:
        ax.plot(k[good], y_plot[good], ms=ms, lw=lw, color=color, label=label)

    if USE_LOG_ABS_FOR_SIGNED and np.any(good & neg):
        mm = good & neg
        ax.plot(k[mm], y_plot[mm], linestyle="none", ms=ms+0.9,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.05)
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

# %% code cell 5

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
        Line2D([0], [0], color=FLAG_COLOR.get(flag, "0.4"), lw=1.8, ms=4,
               label=FLAG_LABEL.get(flag, flag))
        for flag in flags
    ]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 6),
               frameon=False, bbox_to_anchor=(0.5, 1.015))
    fig.suptitle(rf"{sample}: gravity-model comparison for ${spectrum_math_label(spectrum, include_k=False)}$",
                 y=1.045, fontsize=16)
    fig.tight_layout()
    stem = f"{output_safe_name(sample)}_{output_safe_name(spectrum)}_grid_models"
    save_figure(fig, stem, subdir=f"{output_safe_name(sample)}/model_grids")
    maybe_close(fig)
    return fig

# %% code cell 6

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

    fig.suptitle(rf"{sample}: redshift evolution for ${spectrum_math_label(spectrum, include_k=False)}$",
                 y=1.015, fontsize=16)
    fig.subplots_adjust(left=0.075, right=0.965, bottom=0.085, top=0.900, wspace=0.42, hspace=0.35)
    stem = f"{output_safe_name(sample)}_{output_safe_name(spectrum)}_redshift_evolution"
    save_figure(fig, stem, subdir=f"{output_safe_name(sample)}/redshift_evolution")
    maybe_close(fig)
    return fig

# %% code cell 7

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
            ax.plot(k[good], ratio[good], ms=3.3, lw=1.35,
                    color=color, label=FLAG_LABEL.get(flag, flag))
            if sigma_ratio is not None:
                band = good & np.isfinite(sigma_ratio)
                if np.any(band):
                    if str(UNCERTAINTY_STYLE).lower().startswith("err"):
                        ax.errorbar(k[band], ratio[band], yerr=sigma_ratio[band],
                                    fmt="none", ecolor=color, elinewidth=0.9, capsize=2.0, alpha=0.65)
                    else:
                        ax.fill_between(k[band],
                                        ratio[band] - sigma_ratio[band],
                                        ratio[band] + sigma_ratio[band],
                                        color=color, alpha=UNCERTAINTY_ALPHA, linewidth=0)
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
        Line2D([0], [0], color=FLAG_COLOR.get(flag, "0.4"), lw=1.8, ms=4,
               label=FLAG_LABEL.get(flag, flag))
        for flag in compare_flags
    ]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 5),
               frameon=False, bbox_to_anchor=(0.5, 1.015))
    fig.suptitle(rf"{sample}: enhancement of ${spectrum_math_label(spectrum, include_k=False)}$ relative to GR",
                 y=1.045, fontsize=16)
    fig.tight_layout()
    stem = f"{output_safe_name(sample)}_{output_safe_name(spectrum)}_enhancement_vs_GR"
    save_figure(fig, stem, subdir=f"{output_safe_name(sample)}/enhancement")
    maybe_close(fig)
    return fig

# %% code cell 8

# -------------------------------------------------
# Run all plots
# -------------------------------------------------
def plot_everything(samples=samples, spectra_by_sample=spectra_by_sample):
    manifest = []
    for sample in samples:
        spectra = spectra_by_sample.get(sample, [])
        if len(spectra) == 0:
            print(f"[skip] {sample}: no spectra discovered")
            continue

        print(f"\n=== Sample: {sample} | {len(spectra)} spectra ===")
        for spectrum in spectra:
            print(f"  plotting {spectrum}")
            try:
                plot_model_comparison_grid(sample, spectrum)
                manifest.append((sample, spectrum, "model_grid"))
            except Exception as exc:
                print(f"    [warn] model grid failed for {sample}/{spectrum}: {exc}")

            try:
                plot_redshift_evolution_by_model(sample, spectrum)
                manifest.append((sample, spectrum, "redshift_evolution"))
            except Exception as exc:
                print(f"    [warn] redshift evolution failed for {sample}/{spectrum}: {exc}")

            if MAKE_ENHANCEMENT_RELATIVE_TO_GR:
                try:
                    plot_enhancement_relative_to_gr(sample, spectrum)
                    manifest.append((sample, spectrum, "enhancement"))
                except Exception as exc:
                    print(f"    [warn] enhancement failed for {sample}/{spectrum}: {exc}")

    return pd.DataFrame(manifest, columns=["sample", "spectrum", "figure_type"])


manifest = plot_everything()
manifest

# %% [markdown] cell 9
# ## Notes - This notebook deliberately does **not** filter spectra by a predefined list such as `core` or `full`; it discovers all `P_*_Pk` datasets that actually exist in the HDF5 files. - If the output is too large, set: ```python SAMPLES = ["all"] SPECTRA = ["P_gg", "P_dg", "P_dE", "P_EE"] SHOW_FIGURES = False ``` and rerun from the discovery cell onward. - Figures are saved under `figures_all_pk/<sample>/...`.
