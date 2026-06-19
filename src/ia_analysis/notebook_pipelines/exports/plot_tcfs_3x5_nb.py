"""Exported code from notebooks/raw_20260618/plot_tcfs_3x5.ipynb.

This file is generated for project management and refactoring. Review before running end to end.
"""


# %% [markdown] cell 1
# # Plot tcfs correlation functions across snaps and MG flags Reads files named `tcfs_{flag}_{snap:03d}.hdf5` and makes: - a **3×5 grid**: 3 statistics (rows) × 5 snapshots (columns) - each panel overlays **6 simulations**: GR + F40/F45/F50/F55/F60 - optional diagonal error bars from JK covariance - colors: GR=black, MG uses first 5 colors of provided palette You control which three statistics to plot via `stat_keys3` using aliases like `xi`, `xi_1h`, `eta`, `omega_2h`, etc.

# %% code cell 2
import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown] cell 3
# ## Configuration + labels + aliases

# %% code cell 4
BASE_CF_DIR = Path("/cosma/home/dp203/dc-wang17/IA_analysis/cfs")

# Palette (user-provided)
clist=['#c02c38','#c2c116','#3c9566','#1177b0','#ff7c38',
       '#be8936','#e03e36','#b80d57','#700961','#11659a',
       '#abcdef','#fedcba']

FLAGS_ORDER = [ "F40", "F45", "F50", "F55", "F60","GR"]

# Colors: GR=black; MG=first 5 colors
FLAG_COLOR = {
    "F40": '#abcdef',
    "F45": '#79B9DC',
    "F50": '#5F81C2',
    "F55": '#687CBC',
    "F60": '#0C52B5',
    "GR": "k",
}

# Optional line styles / fmt per simulation (override when calling plot_3x5)
DEFAULT_FMT = {k: "-" for k in FLAGS_ORDER}

# snap -> z map
zmap = {
    0: 3.00, 1: 2.00, 2: 1.36, 3: 1.26, 4: 1.15, 5: 1.06,
    6: 0.97, 7: 0.88, 8: 0.80, 9: 0.73, 10: 0.65, 11: 0.58,
    12: 0.51, 13: 0.45, 14: 0.39, 15: 0.33, 16: 0.27, 17: 0.21,
    18: 0.16, 19: 0.10, 20: 0.05, 21: 0.00,
}


MASS_LABEL = {
    "M10_11": r"$10<\lg(M_*)<11$",
    "M11_12": r"$11<\lg(M_*)<12$",
}


fmt = {"F40":"--","F45":"--","F50":"--","F55":"--","F60":"--","GR":"-",}
def normalize_stat_key(stat_key_user):
    return STAT_ALIASES.get(stat_key_user, stat_key_user)

def stat_ylabel(stat_key_user):
    k = normalize_stat_key(stat_key_user)
    return STAT_LABEL.get(k, k)

def tcfs_file(flag, snap, base_dir=BASE_CF_DIR):
    return Path(base_dir) / f"tcfs_{flag}_{int(snap):03d}.hdf5"

# %% code cell 5

FLAGS_ORDER = ["GR", "F40", "F45", "F50", "F55", "F60"]
FLAG_COLOR = {"GR":"k", "F40":clist[0], "F45":clist[1], "F50":clist[2], "F55":clist[3], "F60":clist[4]}
DEFAULT_FMT = {k: "-" for k in FLAGS_ORDER}



# your stat label map (internal keys)
STAT_LABEL = {
    "xi_tot": r"$\xi(r)$",
    "xi_1h":  r"$\xi^{1h}(r)$",
    "xi_2h":  r"$\xi^{2h}(r)$",
    "ee_tot": r"$\eta(r)$",
    "ee_1h":  r"$\eta^{1h}(r)$",
    "ee_2h":  r"$\eta^{2h}(r)$",
    "ed_tot": r"$\omega(r)$",
    "ed_1h":  r"$\omega^{1h}(r)$",
    "ed_2h":  r"$\omega^{2h}(r)$",
    "wpp":    r"$w_{++}(r_p)$",
    "wgp":    r"$w_{g+}(r_p)$",
}

# user aliases -> internal keys
STAT_ALIASES = {
    "xi": "xi_tot",
    "xi_1h": "xi_1h",
    "xi_2h": "xi_2h",
    "eta": "ee_tot",
    "eta_1h": "ee_1h",
    "eta_2h": "ee_2h",
    "omega": "ed_tot",
    "omega_1h": "ed_1h",
    "omega_2h": "ed_2h",
    "wpp": "wpp",
    "wgp": "wgp",
}

# density code -> nice label
DENS_LABEL = {"103": r"$\bar n=10^{-3}$", "104": r"$\bar n=10^{-4}$"}

def normalize_stat_key(stat_key_user):
    return STAT_ALIASES.get(stat_key_user, stat_key_user)

def stat_ylabel(stat_key_user):
    k = normalize_stat_key(stat_key_user)
    return STAT_LABEL.get(k, k)

def abundance_file(mode, code, flag, snap, base_dir=BASE_CF_DIR):
    # e.g. Mstar_103_GR_006.hdf5
    return Path(base_dir) / f"{mode}_{code}_{flag}_{int(snap):03d}.hdf5"

def load_abundance_mean_covdiag(mode, code, flag, snap, stat_key_user, base_dir=BASE_CF_DIR):
    """
    This file layout is ROOT-level groups:
      /xi_tot/mean, /xi_tot/cov, ...
      /rbins, /rp_bins
    For wpp/wgp use rp_bins; otherwise use rbins.
    Returns (r_cent, y, yerr_diag) or (None,None,None).
    """
    stat_key = normalize_stat_key(stat_key_user)
    fpath = abundance_file(mode, code, flag, snap, base_dir)
    if not fpath.exists():
        return None, None, None

    try:
        with h5py.File(fpath, "r") as f:
            if stat_key not in f:
                return None, None, None

            # choose bins
            use_rp = stat_key in ("wpp", "wgp")
            bins_name = "rp_bins" if use_rp else "rbins"
            if bins_name not in f:
                return None, None, None

            bins = f[bins_name][:]
            r = 0.5 * (bins[:-1] + bins[1:])

            y = f[f"{stat_key}/mean"][:]

            yerr = None
            cov_path = f"{stat_key}/cov"
            if cov_path in f:
                cov = f[cov_path][:]
                if cov.ndim == 2 and cov.shape[0] == cov.shape[1] == y.shape[0]:
                    yerr = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))

        return r, y, yerr
    except Exception:
        return None, None, None

# %% [markdown] cell 6
# ## I/O: load mean + diagonal JK error

# %% code cell 7
def load_mean_covdiag(flag, snap, mass_bin, stat_key_user, base_dir=BASE_CF_DIR):
    """Return (r_cent, mean, diag_err) or (None,None,None)."""
    internal_key = normalize_stat_key(stat_key_user)
    fpath = tcfs_file(flag, snap, base_dir)
    if not fpath.exists():
        return None, None, None

    try:
        with h5py.File(fpath, "r") as f:
            if "rbins" not in f:
                return None, None, None
            rb = f["rbins"][:]
            r = 0.5 * (rb[:-1] + rb[1:])

            g = f"{mass_bin}/{internal_key}"
            if g not in f:
                return None, None, None

            y = f[f"{g}/mean"][:]

            yerr = None
            cov_path = f"{g}/cov"
            if cov_path in f:
                cov = f[cov_path][:]
                if cov.ndim == 2 and cov.shape[0] == cov.shape[1] == y.shape[0]:
                    yerr = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))

        return r, y, yerr
    except Exception:
        return None, None, None

# %% [markdown] cell 8
# ## Plot: 3×5 grid (3 stats × 5 snaps), with fmt + optional errors - `stat_keys3` length must be 3 (user keys) - `ylog_rows` and `ylims` are keyed by the **same user keys** used in `stat_keys3` - `fmt` controls line style (and marker if you include it)

# %% code cell 9
def plot_3x5(mass_bin, stat_keys3, *,
             snaps=(6,12,15,18,21),
             flags=FLAGS_ORDER,
             fmt=None,
             show_err=False,
             xlim=(0.1, 20.0),
             ylims=None,
             ylog_rows=None,
             r_times_index=None,
             legend=True):

    if len(stat_keys3) != 3:
        raise ValueError("stat_keys3 must have length 3")

    if fmt is None:
        fmt = DEFAULT_FMT
    if ylims is None:
        ylims = {}
    if ylog_rows is None:
        ylog_rows = {}
    if r_times_index is None:
        r_times_index = {}

    ylabels = [stat_ylabel(k) for k in stat_keys3]
    sns.set(style='ticks')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['mathtext.fontset'] = 'cm'
    fig, axes = plt.subplots(3, len(snaps),
                             figsize=(4.2*len(snaps), 3.6*3),
                             sharex=True)
    if len(snaps) == 1:
        axes = axes.reshape(3, 1)

    fig.suptitle(f"{MASS_LABEL.get(mass_bin, mass_bin)}", fontsize=16, y=0.98)

    def _rpref(p):
        # p=0 -> nothing; p=1 -> r; integer -> r^{int}; non-integer -> 2 sig figs
        if p == 0 or p == 0.0:
            return ""
        if abs(p - round(p)) < 1e-12:
            p_int = int(round(p))
            if p_int == 0:
                return ""
            if p_int == 1:
                return r"$r\,$"
            return rf"$r^{p_int}\,$"
        p2 = f"{p:.2g}"
        return rf"$r^{{{p2}}}\,$"

    for c, snap in enumerate(snaps):
        z = zmap.get(int(snap), None)
        col_title = f"z = {z:.2f}" if z is not None else f"s{int(snap):03d}"

        for r_i, user_stat in enumerate(stat_keys3):
            ax = axes[r_i, c]
            if r_i == 0:
                ax.set_title(col_title, fontsize=12)

            any_plotted = False
            p = float(r_times_index.get(user_stat, 0.0))  # exponent for this row

            for flag in flags:
                rr, yy, yerr = load_mean_covdiag(flag, snap, mass_bin, user_stat)
                if rr is None:
                    continue
                any_plotted = True

                color = FLAG_COLOR.get(flag, "0.5")
                f = fmt.get(flag, "-")

                fac = rr**p if p != 0.0 else 1.0
                yplot = fac * yy
                eplot = (fac * yerr) if (yerr is not None) else None

                if show_err and (eplot is not None):
                    ax.errorbar(rr, yplot, yerr=eplot,
                                fmt=f, linewidth=2.0, capsize=2,
                                color=color,
                                label=flag if (r_i==0 and c==0) else None)
                else:
                    ax.plot(rr, yplot, f, linewidth=2.0, color=color,
                            label=flag if (r_i==0 and c==0) else None)

            ax.set_xscale("log")
            ax.set_xlim(*xlim)
            ax.grid(True, alpha=0.22)

            if ylog_rows.get(user_stat, False):
                ax.set_yscale("log")

            if user_stat in ylims and ylims[user_stat] is not None:
                ax.set_ylim(*ylims[user_stat])

            if c == 0:
                ax.set_ylabel(_rpref(p) + ylabels[r_i],fontsize=20)
            else:
                ax.set_ylabel("")

            if r_i == 2:
                ax.set_xlabel(r"$r\ [\mathrm{Mpc}/h]$",fontsize=20)
            else:
                ax.set_xlabel("")

            if not any_plotted:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

    if legend:
        handles, labels = axes[0,0].get_legend_handles_labels()
        if len(handles) > 0:
            fig.legend(handles, labels, loc="upper right",
                       frameon=False, bbox_to_anchor=(0.995, 0.995),ncol=2)

    plt.tight_layout(rect=[0, 0, 0.98, 0.95])
    plt.show()
    return fig, axes

# %% code cell 10
def plot_3x5_abundance(mode, code, stat_keys3, *,
                       snaps=(6,12,15,18,21),
                       flags=FLAGS_ORDER,
                       fmt=None,
                       show_err=False,
                       xlim=(0.1, 20.0),
                       ylims=None,
                       ylog_rows=None,
                       r_times_index=None,
                       legend=True):

    if len(stat_keys3) != 3:
        raise ValueError("stat_keys3 must have length 3")

    if fmt is None:
        fmt = DEFAULT_FMT
    if ylims is None:
        ylims = {}
    if ylog_rows is None:
        ylog_rows = {}
    if r_times_index is None:
        r_times_index = {}

    ylabels = [stat_ylabel(k) for k in stat_keys3]

    fig, axes = plt.subplots(3, len(snaps),
                             figsize=(4.2*len(snaps), 3.6*3),
                             sharex=True)
    if len(snaps) == 1:
        axes = axes.reshape(3, 1)

    fig.suptitle(f"{mode}  |  {DENS_LABEL.get(str(code), str(code))}", fontsize=16, y=0.98)

    def _rpref(p):
        if p == 0 or p == 0.0:
            return ""
        if abs(p - round(p)) < 1e-12:
            p_int = int(round(p))
            if p_int == 0:
                return ""
            if p_int == 1:
                return r"$r\,$"
            return rf"$r^{p_int}\,$"
        p2 = f"{p:.2g}"
        return rf"$r^{{{p2}}}\,$"

    for c, snap in enumerate(snaps):
        z = zmap.get(int(snap), None)
        col_title = f"z = {z:.2f}" if z is not None else f"s{int(snap):03d}"

        for r_i, user_stat in enumerate(stat_keys3):
            ax = axes[r_i, c]
            if r_i == 0:
                ax.set_title(col_title, fontsize=12)

            any_plotted = False
            p = float(r_times_index.get(user_stat, 0.0))

            for flag in flags:
                rr, yy, yerr = load_abundance_mean_covdiag(mode, code, flag, snap, user_stat)
                if rr is None:
                    continue
                any_plotted = True

                color = FLAG_COLOR.get(flag, "0.5")
                f = fmt.get(flag, "-")

                fac = rr**p if p != 0.0 else 1.0
                yplot = fac * yy
                eplot = (fac * yerr) if (yerr is not None) else None

                if show_err and (eplot is not None):
                    ax.errorbar(rr, yplot, yerr=eplot,
                                fmt=f, linewidth=2.0, capsize=2,
                                color=color,
                                label=flag if (r_i==0 and c==0) else None)
                else:
                    ax.plot(rr, yplot, f, linewidth=2.0, color=color,
                            label=flag if (r_i==0 and c==0) else None)

            ax.set_xscale("log")
            ax.set_xlim(*xlim)
            # ax.set_xlim(0.2,5)
            ax.grid(True, alpha=0.22)

            if ylog_rows.get(user_stat, False):
                ax.set_yscale("log")

            if user_stat in ylims and ylims[user_stat] is not None:
                ax.set_ylim(*ylims[user_stat])

            if c == 0:
                ax.set_ylabel(_rpref(p) + ylabels[r_i],fontsize=20)
            else:
                ax.set_ylabel("")

            if r_i == 2:
                ax.set_xlabel(r"$r\ [\mathrm{Mpc}/h]$",fontsize=20)
            else:
                ax.set_xlabel("")

            if not any_plotted:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

    if legend:
        handles, labels = axes[0,0].get_legend_handles_labels()
        if len(handles) > 0:
            fig.legend(handles, labels, loc="upper right",
                       frameon=False, bbox_to_anchor=(0.995, 0.995),ncol=2)

    plt.tight_layout(rect=[0, 0, 0.98, 0.95])
    plt.show()
    return fig, axes

# %% code cell 11
def plot_2x5_abundance(mode, code, *,
                       snaps=(6,12,15,18,21),
                       flags=FLAGS_ORDER,
                       fmt=None,
                       show_err=False,
                       xlim=(0.1, 20.0),
                       ylims=None,
                       ylog_rows=None,
                       r_times_index=None,
                       legend=True):
    """
    Fixed two rows: wgp and wpp.
    """
    stat_keys2 = ["wgp", "wpp"]

    if fmt is None:
        fmt = DEFAULT_FMT
    if ylims is None:
        ylims = {}
    if ylog_rows is None:
        ylog_rows = {}
    if r_times_index is None:
        r_times_index = {}

    ylabels = [stat_ylabel(k) for k in stat_keys2]

    fig, axes = plt.subplots(2, len(snaps),
                             figsize=(4.2*len(snaps), 3.6*2),
                             sharex=True)
    if len(snaps) == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(f"{mode}  |  {DENS_LABEL.get(str(code), str(code))}", fontsize=16, y=0.98)

    def _rpref(p):
        if p == 0 or p == 0.0:
            return ""
        if abs(p - round(p)) < 1e-12:
            p_int = int(round(p))
            if p_int == 0:
                return ""
            if p_int == 1:
                return r"$r_p\,$"
            return rf"$r_p^{p_int}\,$"
        p2 = f"{p:.2g}"
        return rf"$r_p^{{{p2}}}\,$"

    for c, snap in enumerate(snaps):
        z = zmap.get(int(snap), None)
        col_title = f"z = {z:.2f}" if z is not None else f"s{int(snap):03d}"

        for r_i, user_stat in enumerate(stat_keys2):
            ax = axes[r_i, c]
            if r_i == 0:
                ax.set_title(col_title, fontsize=12)

            any_plotted = False
            p = float(r_times_index.get(user_stat, 0.0))

            for flag in flags:
                rr, yy, yerr = load_abundance_mean_covdiag(mode, code, flag, snap, user_stat)
                if rr is None:
                    continue
                any_plotted = True

                color = FLAG_COLOR.get(flag, "0.5")
                f = fmt.get(flag, "-")

                fac = rr**p if p != 0.0 else 1.0
                yplot = fac * yy
                eplot = (fac * yerr) if (yerr is not None) else None

                if show_err and (eplot is not None):
                    ax.errorbar(rr, yplot, yerr=eplot,
                                fmt=f, linewidth=2.0, capsize=2,
                                color=color,
                                label=flag if (r_i==0 and c==0) else None)
                else:
                    ax.plot(rr, yplot, f, linewidth=2.0, color=color,
                            label=flag if (r_i==0 and c==0) else None)

            ax.set_xscale("log")
            ax.set_xlim(*xlim)
            ax.grid(True, alpha=0.22)

            if ylog_rows.get(user_stat, False):
                ax.set_yscale("log")

            if user_stat in ylims and ylims[user_stat] is not None:
                ax.set_ylim(*ylims[user_stat])

            if c == 0:
                ax.set_ylabel(_rpref(p) + ylabels[r_i],fontsize=20)
            else:
                ax.set_ylabel("")

            if r_i == 1:
                ax.set_xlabel(r"$r_p\ [\mathrm{Mpc}/h]$",fontsize=20)
            else:
                ax.set_xlabel("")

            if not any_plotted:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

    if legend:
        handles, labels = axes[0,0].get_legend_handles_labels()
        if len(handles) > 0:
            fig.legend(handles, labels, loc="upper right",
                       frameon=False, bbox_to_anchor=(0.995, 0.995),ncol=2)

    plt.tight_layout(rect=[0, 0, 0.98, 0.95])
    plt.show()
    return fig, axes

# %% code cell 12

stat_keys3 = ["xi", "eta", "omega"] 
ylog_rows = {stat_keys3[0]: True, stat_keys3[1]: True, stat_keys3[2]: False}
r_times_index = {stat_keys3[0]: 2.0, stat_keys3[1]: 0., stat_keys3[2]: 0.0}
# xlim=(0.2,)
plot_3x5(
    mass_bin="M10_11" ,
    stat_keys3=stat_keys3,
    snaps=(6,12,15,18,21),
    flags=FLAGS_ORDER,
    fmt=fmt,
    r_times_index =r_times_index ,
    show_err=True,
    ylims={stat_keys3[0]: (1,20), stat_keys3[1]: None, stat_keys3[2]:None},
    ylog_rows=ylog_rows,
    legend=True
)

plot_3x5(
    mass_bin="M11_12" ,
    stat_keys3=stat_keys3,
    snaps=(6,12,15,18,21),
    flags=FLAGS_ORDER,
    fmt=fmt,
    r_times_index =r_times_index ,
    show_err=True,
    ylims={stat_keys3[0]: (3,50), stat_keys3[1]: None, stat_keys3[2]:None},
    ylog_rows=ylog_rows,
    legend=True
)

# %% code cell 13

stat_keys3 = ["xi", "xi_1h", "xi_2h"] 
ylog_rows = {stat_keys3[0]: True, stat_keys3[1]: True, stat_keys3[2]: True}
r_times_index = {stat_keys3[0]: 2.0, stat_keys3[1]:2, stat_keys3[2]: 2}

plot_3x5(
    mass_bin="M10_11" ,
    stat_keys3=stat_keys3,
    snaps=(6,12,15,18,21),
    flags=FLAGS_ORDER,
    fmt=fmt,
    r_times_index =r_times_index ,
    show_err=True,
    ylog_rows=ylog_rows,
    ylims={},
    legend=True
)

plot_3x5(
    mass_bin="M11_12" ,
    stat_keys3=stat_keys3,
    snaps=(6,12,15,18,21),
    flags=FLAGS_ORDER,
    fmt=fmt,
    r_times_index =r_times_index ,
    show_err=True,
    ylog_rows=ylog_rows,
    ylims={},
    legend=True
)

# %% code cell 14

stat_keys3 = ["eta", "eta_1h", "eta_2h"] 
ylog_rows = {stat_keys3[0]: True, stat_keys3[1]: True, stat_keys3[2]: True}
r_times_index = {stat_keys3[0]: 0.5, stat_keys3[1]:0.5, stat_keys3[2]: 0.5}

plot_3x5(
    mass_bin="M10_11" ,
    stat_keys3=stat_keys3,
    snaps=(6,12,15,18,21),
    flags=FLAGS_ORDER,
    fmt=fmt,
    r_times_index =r_times_index ,
    show_err=True,
    ylog_rows=ylog_rows,
    ylims={},
    legend=True
)

plot_3x5(
    mass_bin="M11_12" ,
    stat_keys3=stat_keys3,
    snaps=(6,12,15,18,21),
    flags=FLAGS_ORDER,
    fmt=fmt,
    r_times_index =r_times_index ,
    show_err=True,
    ylog_rows=ylog_rows,
    ylims={},
    legend=True
)

# %% code cell 15

stat_keys3 = ["omega", "omega_1h", "omega_2h"] 
ylog_rows = {stat_keys3[0]: False, stat_keys3[1]: False, stat_keys3[2]: False}
r_times_index = {stat_keys3[0]: 0., stat_keys3[1]:0., stat_keys3[2]: 0.}

plot_3x5(
    mass_bin="M10_11" ,
    stat_keys3=stat_keys3,
    snaps=(6,12,15,18,21),
    flags=FLAGS_ORDER,
    fmt=fmt,
    r_times_index =r_times_index ,
    show_err=True,
    ylog_rows=ylog_rows,
    ylims={},
    legend=True
)

plot_3x5(
    mass_bin="M11_12" ,
    stat_keys3=stat_keys3,
    snaps=(6,12,15,18,21),
    flags=FLAGS_ORDER,
    fmt=fmt,
    r_times_index =r_times_index ,
    show_err=True,
    ylog_rows=ylog_rows,
    ylims={},
    legend=True
)

# %% code cell 16

stat_keys3 = ["xi", "xi_1h", "xi_2h"] 
ylog_rows = {stat_keys3[0]: True, stat_keys3[1]: True, stat_keys3[2]: True}
r_times_index = {stat_keys3[0]: 2.0, stat_keys3[1]:2, stat_keys3[2]: 2}
ylims={stat_keys3[0]: (1,50), stat_keys3[1]: (1,30), stat_keys3[2]: (1,30)}
plot_3x5_abundance(
    mode="Mstar",
    code="103",
    stat_keys3=stat_keys3,
    show_err=True,
    ylog_rows=ylog_rows,
    fmt=fmt,
    xlim=(0.2,5),
    ylims=ylims,
    r_times_index=r_times_index,
)
plot_3x5_abundance(
    mode="SFR",
    code="103",
    stat_keys3=stat_keys3,
    show_err=True,
    ylog_rows=ylog_rows,
        xlim=(0.2,5),
    ylims=ylims,
    fmt=fmt,
    r_times_index=r_times_index,
)

# %% code cell 17

stat_keys3 = ["eta", "eta_1h", "eta_2h"] 
ylog_rows = {stat_keys3[0]: True, stat_keys3[1]: True, stat_keys3[2]: True}
r_times_index = {stat_keys3[0]: 0., stat_keys3[1]:0., stat_keys3[2]: 0.}

plot_3x5_abundance(
    mode="Mstar",
    code="103",
    stat_keys3=stat_keys3,
    show_err=True,
    ylog_rows=ylog_rows,
    fmt=fmt,
    r_times_index=r_times_index,
)
plot_3x5_abundance(
    mode="SFR",
    code="103",
    stat_keys3=stat_keys3,
    show_err=True,
    ylog_rows=ylog_rows,
    fmt=fmt,
    r_times_index=r_times_index,
)

# %% code cell 18
stat_keys3 = ["omega", "omega_1h", "omega_2h"] 
ylog_rows = {stat_keys3[0]: False, stat_keys3[1]: False, stat_keys3[2]: False}
r_times_index = {stat_keys3[0]: 0, stat_keys3[1]:0, stat_keys3[2]: 0}

plot_3x5_abundance(
    mode="Mstar",
    code="103",
    stat_keys3=stat_keys3,
    show_err=True,
    ylog_rows=ylog_rows,
    fmt=fmt,
    r_times_index=r_times_index,
)
plot_3x5_abundance(
    mode="SFR",
    code="103",
    stat_keys3=stat_keys3,
    show_err=True,
    ylog_rows=ylog_rows,
    fmt=fmt,
    r_times_index=r_times_index,
)

# %% code cell 19
ylog_rows={"wgp": False, "wpp": False}
plot_2x5_abundance(
    mode="Mstar",
    code="103",
    show_err=True,
    fmt=fmt,
    ylog_rows=ylog_rows,
    r_times_index={"wgp": 1.0, "wpp": 1.0},  # e.g. plot r_p * w
    
)
plot_2x5_abundance(
    mode="SFR",
    code="103",
    show_err=True,
    fmt=fmt,
    ylog_rows=ylog_rows,
    r_times_index={"wgp": 1.0, "wpp": 1.0},  # e.g. plot r_p * w
)

# %% code cell 20
def plot_4x5_compare(stat_keys, *,
                     snaps=(6,12,15,18,21),
                     flags=FLAGS_ORDER,
                     fmt=None,
                     show_err=False,
                     xlim=(0.1, 20.0),
                     ylims=None,          # dict keyed by user stat key -> (ymin,ymax) or None
                     ylog_rows=None,      # dict keyed by user stat key -> bool
                     r_times_index=None,  # dict keyed by user stat key -> float
                     row_specs=None,
                     # NEW:
                     unify_y=True,         # force same y-limits across all panels (works best when len(stat_keys)==1)
                     include_err_in_ylim=True,  # include ±err when computing global y-range
                     legend_top=True):

    if fmt is None:
        fmt = DEFAULT_FMT
    if ylims is None:
        ylims = {}
    if ylog_rows is None:
        ylog_rows = {}
    if r_times_index is None:
        r_times_index = {}

    if row_specs is None:
        row_specs = [
            {"kind":"mass",  "label": r"$10<\lg(M_*)<11$", "mass_bin":"M10_11"},
            {"kind":"mass",  "label": r"$11<\lg(M_*)<12$", "mass_bin":"M11_12"},
            {"kind":"abund", "label": r"$M_*\,\,\bar n=10^{-3}$", "mode":"Mstar", "code":"103"},
            {"kind":"abund", "label": r"$\mathrm{SFR}\,\,\bar n=10^{-3}$", "mode":"SFR",   "code":"103"},
        ]

    if len(row_specs) != 4:
        raise ValueError("row_specs must have length 4 (for a 4×5 grid).")

    if not (len(stat_keys) == 1):
        raise ValueError("For y-aligned 4×5, please pass a single stat: stat_keys=['xi'] / ['xi_1h'] / ['eta'] ...")

    user_stat = stat_keys[0]
    p = float(r_times_index.get(user_stat, 0.0))
    y_is_log = bool(ylog_rows.get(user_stat, False))

    def _rpref(p):
        if p == 0 or p == 0.0:
            return ""
        if abs(p - round(p)) < 1e-12:
            p_int = int(round(p))
            if p_int == 0:
                return ""
            if p_int == 1:
                return r"$r\,$"
            return rf"$r^{p_int}\,$"
        p2 = f"{p:.2g}"
        return rf"$r^{{{p2}}}\,$"

    ylab = _rpref(p) + stat_ylabel(user_stat)

    # choose loader per row
    def _load(spec, flag, snap):
        if spec["kind"] == "mass":
            return load_mean_covdiag(flag, snap, spec["mass_bin"], user_stat)
        else:
            return load_abundance_mean_covdiag(spec["mode"], spec["code"], flag, snap, user_stat)

    # --------- compute global y-limits if needed ----------
    global_ylim = None
    if unify_y and (user_stat not in ylims or ylims[user_stat] is None):
        ymin = np.inf
        ymax = -np.inf

        for spec in row_specs:
            for snap in snaps:
                for flag in flags:
                    rr, yy, yerr = _load(spec, flag, snap)
                    if rr is None:
                        continue

                    fac = rr**p if p != 0.0 else 1.0
                    yplot = fac * yy

                    if show_err and include_err_in_ylim and (yerr is not None):
                        eplot = fac * yerr
                        lo = yplot - eplot
                        hi = yplot + eplot
                    else:
                        lo = yplot
                        hi = yplot

                    if y_is_log:
                        # for log scale we need positive values
                        lo = lo[lo > 0]
                        hi = hi[hi > 0]
                        if lo.size == 0 or hi.size == 0:
                            continue

                    ymin = min(ymin, np.nanmin(lo))
                    ymax = max(ymax, np.nanmax(hi))

        if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
            # small padding
            if y_is_log:
                global_ylim = (ymin * 0.9, ymax * 1.1)
            else:
                pad = 0.05 * (ymax - ymin)
                global_ylim = (ymin - pad, ymax + pad)

    # if user provided ylims, that wins
    if user_stat in ylims and ylims[user_stat] is not None:
        global_ylim = ylims[user_stat]

    # --------- plotting ----------
    fig, axes = plt.subplots(4, len(snaps),
                             figsize=(4.2*len(snaps), 3.0*4),
                             sharex=True, sharey=True)

    if len(snaps) == 1:
        axes = axes.reshape(4, 1)

    # column titles
    for c, snap in enumerate(snaps):
        z = zmap.get(int(snap), None)
        col_title = f"z = {z:.2f}" if z is not None else f"s{int(snap):03d}"
        axes[0, c].set_title(col_title, fontsize=15)

    for r_i, spec in enumerate(row_specs):
        for c, snap in enumerate(snaps):
            ax = axes[r_i, c]
            ax.set_xscale("log")
            ax.set_xlim(*xlim)
            ax.grid(True, alpha=0.22)

            if y_is_log:
                ax.set_yscale("log")

            if global_ylim is not None:
                ax.set_ylim(*global_ylim)

            any_plotted = False
            for flag in flags:
                rr, yy, yerr = _load(spec, flag, snap)
                if rr is None:
                    continue
                any_plotted = True

                color = FLAG_COLOR.get(flag, "0.5")
                fflag = fmt.get(flag, "-")

                fac = rr**p if p != 0.0 else 1.0
                yplot = fac * yy
                eplot = (fac * yerr) if (show_err and yerr is not None) else None

                if show_err and (eplot is not None):
                    ax.errorbar(rr, yplot, yerr=eplot,
                                fmt=fflag, linewidth=2.0, capsize=2,
                                color=color)
                else:
                    ax.plot(rr, yplot, fflag, linewidth=2.0, color=color)

            # leftmost y-label includes row label + stat label
            if c == 0:
                ax.set_ylabel(spec["label"] + "\n" + ylab,fontsize=15)

            # bottom x-label only
            if r_i == 3:
                ax.set_xlabel(r"$r\ [\mathrm{Mpc}/h]$",fontsize=15)

            if not any_plotted:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

    # --------- top legend for gravity types ----------
    if legend_top:
        handles = []
        labels = []
        for flag in flags:
            h, = plt.plot([], [], fmt.get(flag, "-"),
                          color=FLAG_COLOR.get(flag, "0.5"),
                          lw=2.5, label=flag)
            handles.append(h); labels.append(flag)

        fig.legend(handles, labels,
                   loc="upper center",
                   ncol=len(flags),
                   frameon=False,
                   bbox_to_anchor=(0.5, 0.995),
                   handlelength=3.0,
                   columnspacing=1.4,fontsize=15)

    # make room for top legend
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    return fig,axes

# %% code cell 21
stat_key = ["eta"]  # or ["xi_1h"], ["eta"], ["omega"], ...
fig,axes=plot_4x5_compare(
    stat_keys=stat_key,
    snaps=(6,12,15,18,21),
    fmt=fmt,
    show_err=True,
    ylog_rows={"eta": True},
    r_times_index={"eta": 0.0},
    ylims={"eta": (0.02, 0.7)},
    xlim=(0.1, 20.0),
)
fig.savefig('./plots/etas.png')

# %% code cell 22
stat_key = ["eta_1h"]  # or ["xi_1h"], ["eta"], ["omega"], ...
fig,axes=plot_4x5_compare(
    stat_keys=stat_key,
    snaps=(6,12,15,18,21),
    fmt=fmt,
    show_err=True,
    ylog_rows={"eta_1h": True},
    r_times_index={"eta_1h": 0.0},
    ylims={"eta_1h": (0.002, 0.7)},
    xlim=(0.1, 20.0),
)
fig.savefig('./plots/eta_1h.png')

# %% code cell 23
stat_key = ["eta_2h"]  # or ["xi_1h"], ["eta"], ["omega"], ...
fig,axes=plot_4x5_compare(
    stat_keys=stat_key,
    snaps=(6,12,15,18,21),
    fmt=fmt,
    show_err=True,
    ylog_rows={"eta_2h": True},
    r_times_index={"eta_2h": 0.0},
    ylims={"eta_2h": (0.02, 0.7)},
    xlim=(0.1, 20.0),
)
fig.savefig('./plots/eta_2h.png')

# %% code cell 24
stat_key = ["xi"]  # or ["xi_1h"], ["eta"], ["omega"], ...
fig,axes=plot_4x5_compare(
    stat_keys=stat_key,
    snaps=(6,12,15,18,21),
    fmt=fmt,
    show_err=True,
    ylog_rows={"xi": True},
    r_times_index={"xi": 2.0},
    ylims={"xi": (1, 70)},
    xlim=(0.1, 5),
)
fig.savefig('./plots/xis.png')

# %% code cell 25
def plot_2x5_abundance_wpp_only(*,
                                code,
                                snaps=(6,12,15,18,21),
                                flags=FLAGS_ORDER,
                                fmt=None,
                                show_err=False,
                                xlim=(0.1, 20.0),
                                ylims=None,
                                ylog_rows=None,
                                r_times_index=None,
                                legend=True):
    """
    Two rows, both for wpp only.

    Top row    : Mstar
    Bottom row : SFR
    """
    row_modes = ["Mstar", "SFR"]
    user_stat = "wpp"

    if fmt is None:
        fmt = DEFAULT_FMT
    if ylims is None:
        ylims = {}
    if ylog_rows is None:
        ylog_rows = {}
    if r_times_index is None:
        r_times_index = {}

    fig, axes = plt.subplots(2, len(snaps),
                             figsize=(4.2*len(snaps), 3.6*2),
                             sharex=True)
    if len(snaps) == 1:
        axes = axes.reshape(2, 1)

    # fig.suptitle(f"wpp  |  {DENS_LABEL.get(str(code), str(code))}", fontsize=16, y=0.98)

    def _rpref(p):
        if p == 0 or p == 0.0:
            return ""
        if abs(p - round(p)) < 1e-12:
            p_int = int(round(p))
            if p_int == 0:
                return ""
            if p_int == 1:
                return r"$r_p\,$"
            return rf"$r_p^{p_int}\,$"
        p2 = f"{p:.2g}"
        return rf"$r_p^{{{p2}}}\,$"

    for r_i, mode in enumerate(row_modes):
        for c, snap in enumerate(snaps):
            ax = axes[r_i, c]

            z = zmap.get(int(snap), None)
            col_title = f"z = {z:.2f}" if z is not None else f"s{int(snap):03d}"
            if r_i == 0:
                ax.set_title(col_title, fontsize=12)

            any_plotted = False
            p = float(r_times_index.get(user_stat, 0.0))

            for flag in flags:
                rr, yy, yerr = load_abundance_mean_covdiag(mode, code, flag, snap, user_stat)
                if rr is None:
                    continue
                any_plotted = True

                color = FLAG_COLOR.get(flag, "0.5")
                f = fmt.get(flag, "-")

                fac = rr**p if p != 0.0 else 1.0
                yplot = fac * yy
                eplot = (fac * yerr) if (yerr is not None) else None

                if show_err and (eplot is not None):
                    ax.errorbar(rr, yplot, yerr=eplot,
                                fmt=f, linewidth=2.0, capsize=2,
                                color=color,
                                label=flag if (r_i == 0 and c == 0) else None)
                else:
                    ax.plot(rr, yplot, f, linewidth=2.0, color=color,
                            label=flag if (r_i == 0 and c == 0) else None)

            ax.set_xscale("log")
            ax.set_xlim(*xlim)
            ax.grid(True, alpha=0.22)

            if ylog_rows.get(user_stat, False):
                ax.set_yscale("log")

            if user_stat in ylims and ylims[user_stat] is not None:
                ax.set_ylim(*ylims[user_stat])

            if c == 0:
                ax.set_ylabel(_rpref(p) + stat_ylabel(user_stat), fontsize=20)
                ax.text(-0.30, 0.5, mode, transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=18)
            else:
                ax.set_ylabel("")

            if r_i == 1:
                ax.set_xlabel(r"$r_p\ [\mathrm{Mpc}/h]$", fontsize=20)
            else:
                ax.set_xlabel("")

            if not any_plotted:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

    if legend:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if len(handles) > 0:
            fig.legend(handles, labels, loc="upper right",
                       frameon=False, bbox_to_anchor=(0.995, 0.995), ncol=2)

    plt.tight_layout(rect=[0, 0, 0.98, 0.95])
    plt.show()
    return fig, axes

# %% code cell 26
ylog_rows = {"wpp": False}

fig, axes=plot_2x5_abundance_wpp_only(
    code="103",
    show_err=True,
    fmt=fmt,
    ylog_rows=ylog_rows,
    r_times_index={"wpp": 1.0},
)
fig.savefig('./plots/wpps.png')

# %% code cell 27
