"""Exported code from notebooks/raw_20260618/MAset_satellite_radial_distribution_compare.ipynb.

This file is generated for project management and refactoring. Review before running end to end.
"""


# %% [markdown] cell 1
# # Satellite radial distribution: comparison plots This notebook extends the previous radial-distribution notebook and adds two new comparison modes: 1. **Redshift comparison**: for a fixed gravity model, each subplot overlays different snapshots / redshifts. 2. **Gravity-model comparison**: for a fixed snapshot, each subplot overlays different gravity models. It also keeps the original plotting mode: - columns = gravity models - rows = galaxy classifications - one figure per snapshot The radial 

# %% code cell 2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

with open('/cosma8/data/dp203/dc-wang17/MG_global/MA.pkl', 'rb') as f:
    MAset = pickle.load(f)

zmap = {
    6: 0.97,
    12: 0.51,
    15: 0.33,
    18: 0.16,
    21: 0.00,
}

flags = ['F40', 'F45', 'F50', 'F55', 'F60', 'GR']
snaps = [6, 12, 15, 18, 21]

clist = ['#c02c38','#c2c116','#3c9566','#1177b0','#ff7c38','#be8936',
         '#e03e36','#b80d57','#700961','#11659a','#abcdef','#fedcba']

flag_colors = {flag: clist[i] for i, flag in enumerate(flags)}
snap_colors = {
    6:  clist[0],
    12: clist[1],
    15: clist[2],
    18: clist[3],
    21: clist[5],
}

print("Loaded MAset keys:", list(MAset.keys()))

# %% code cell 3
# Configuration

SUBHALO_ID_KEY = "SubhaloID"
CENID_KEY = "CenID"
R_KEY = "R_over_R_200c"
STELLAR_MASS_KEY = "SubhaloMassInRadType"
SFR_KEY = "SubhaloSFR"
KAPPA_KEY = "kappa_rot_Star"

STELLAR_COMPONENT_INDEX = 4
STELLAR_MASS_SCALE = 1e10

R_RANGE = (0.0, 2.5)
NBINS = 15
RADIAL_BINS = np.linspace(R_RANGE[0], R_RANGE[1], NBINS + 1)

DENSITY = True
MIN_COUNT = 5
YLOG = False
LINEWIDTH = 2.0
SHOW_COUNTS_IN_PANEL = True
COMPARE_NCOLS = 3

# %% code cell 4
# Helper functions

def get_snap_dict(MAset, flag, snap):
    d = MAset[flag]
    key = f"{int(snap):03d}"
    if key in d:
        return d[key]
    if int(snap) in d:
        return d[int(snap)]
    if str(int(snap)) in d:
        return d[str(int(snap))]
    raise KeyError(f"Snapshot {snap} not found in MAset['{flag}'].")


def get_arr(data, key):
    if key == STELLAR_MASS_KEY:
        arr = np.asarray(data[key])
        return np.asarray(arr[:, STELLAR_COMPONENT_INDEX], dtype=float) * STELLAR_MASS_SCALE

    arr = np.asarray(data[key])
    if arr.ndim != 1:
        raise ValueError(f"data['{key}'] must be 1D, got shape={arr.shape}")
    return arr


def build_mask_from_limits(data, limits=None, base_mask=None):
    if base_mask is None:
        n = len(np.asarray(data[SUBHALO_ID_KEY]))
        mask = np.ones(n, dtype=bool)
    else:
        mask = np.asarray(base_mask, dtype=bool).copy()

    if limits is None:
        return mask

    for key, rule in limits.items():
        arr = get_arr(data, key)
        local = np.isfinite(arr)

        if callable(rule):
            local &= np.asarray(rule(arr), dtype=bool)
        else:
            low, high = rule
            if low is not None:
                local &= (arr >= low)
            if high is not None:
                local &= (arr < high)

        mask &= local

    return mask


def make_analysis_specs():
    return [
        {"name": "all",         "title": "All satellites",                             "galaxy_limits": None},
        {"name": "mstar_10_11", "title": r"$10^{10}-10^{11}\ M_\odot/h$",           "galaxy_limits": {"SubhaloMassInRadType": (1e10, 1e11)}},
        {"name": "mstar_gt_11", "title": r"$>10^{11}\ M_\odot/h$",                  "galaxy_limits": {"SubhaloMassInRadType": (1e11, None)}},
        {"name": "sfr_low",     "title": r"${\rm SFR} < 1$",                         "galaxy_limits": {"SubhaloSFR": (None, 1.0)}},
        {"name": "sfr_high",    "title": r"${\rm SFR} \geq 1$",                     "galaxy_limits": {"SubhaloSFR": (1.0, None)}},
        {"name": "spheroid",    "title": r"Spheroid: $\kappa_{\rm rot,Star}<0.45$", "galaxy_limits": {"kappa_rot_Star": (None, 0.45)}},
        {"name": "disk",        "title": r"Disk: $\kappa_{\rm rot,Star}>0.55$",     "galaxy_limits": {"kappa_rot_Star": (0.55, None)}},
    ]


def compute_radial_profile(data, galaxy_limits=None, radial_bins=RADIAL_BINS, density=True, min_count=5):
    sid = np.asarray(get_arr(data, SUBHALO_ID_KEY), dtype=np.int64)
    cenid = np.asarray(get_arr(data, CENID_KEY), dtype=np.int64)
    rr = np.asarray(get_arr(data, R_KEY), dtype=float)

    base_mask = (
        np.isfinite(sid)
        & np.isfinite(cenid)
        & np.isfinite(rr)
        & (sid != cenid)
        & (rr >= radial_bins[0])
        & (rr < radial_bins[-1])
    )

    sel = build_mask_from_limits(data, galaxy_limits, base_mask=base_mask)
    rr_sel = rr[sel]

    counts, edges = np.histogram(rr_sel, bins=radial_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    if rr_sel.size < min_count:
        y = np.full_like(centers, np.nan, dtype=float)
    else:
        if density:
            total = counts.sum()
            y = counts / (total * widths) if total > 0 else np.full_like(centers, np.nan, dtype=float)
        else:
            y = counts.astype(float)

    return {
        "x": centers,
        "y": y,
        "counts": counts,
        "bin_edges": edges,
        "bin_width": widths,
        "N_selected": int(rr_sel.size),
    }


def build_profiles_for_snapshot(MAset, flags, snap, analysis_specs):
    profiles = {}
    for spec in analysis_specs:
        sname = spec["name"]
        profiles[sname] = {}
        for flag in flags:
            data = get_snap_dict(MAset, flag, snap)
            profiles[sname][flag] = compute_radial_profile(
                data,
                galaxy_limits=spec.get("galaxy_limits", None),
                radial_bins=RADIAL_BINS,
                density=DENSITY,
                min_count=MIN_COUNT,
            )
    return profiles


def build_profile_cube(MAset, flags, snaps, analysis_specs):
    cube = {}
    for spec in analysis_specs:
        sname = spec["name"]
        cube[sname] = {}
        for flag in flags:
            cube[sname][flag] = {}
            for snap in snaps:
                data = get_snap_dict(MAset, flag, snap)
                cube[sname][flag][snap] = compute_radial_profile(
                    data,
                    galaxy_limits=spec.get("galaxy_limits", None),
                    radial_bins=RADIAL_BINS,
                    density=DENSITY,
                    min_count=MIN_COUNT,
                )
    return cube

# %% code cell 5
# Plotting functions

def plot_original_grid(profiles, analysis_specs, flags=flags, snap=None, zmap=zmap, figsize=(21, 18), ylog=False):
    nrows = len(analysis_specs)
    ncols = len(flags)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for i, spec in enumerate(analysis_specs):
        sname = spec["name"]
        for j, flag in enumerate(flags):
            ax = axes[i, j]
            prof = profiles[sname][flag]
            x = prof["x"]
            y = prof["y"]
            nsel = prof["N_selected"]

            good = np.isfinite(x) & np.isfinite(y)
            if np.any(good):
                ax.plot(x[good], y[good], color=flag_colors[flag], lw=LINEWIDTH)

            if i == 0:
                ax.set_title(flag, fontsize=13)

            if j == 0:
                ylabel = spec["title"] + "\n"
                ylabel += r"$p(r/R_{200\mathrm{c}})$" if DENSITY else r"${\rm counts}$"
                ax.set_ylabel(ylabel, fontsize=11)

            if SHOW_COUNTS_IN_PANEL:
                ax.text(0.96, 0.92, f"N={nsel}", ha='right', va='top', transform=ax.transAxes, fontsize=9)

            if ylog:
                ax.set_yscale("log")

            ax.set_xlim(R_RANGE)
            ax.grid(True, alpha=0.3)

    for j in range(ncols):
        axes[-1, j].set_xlabel(r"$r/R_{200\mathrm{c}}$", fontsize=11)

    if snap is not None:
        title = f"Satellite radial distribution at snap={snap:03d}"
        if snap in zmap:
            title += f" (z={zmap[snap]:.2f})"
        fig.suptitle(title, y=0.995, fontsize=17)

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    return fig, axes


def _setup_compare_axes(n_panels, ncols=3, base_height=4.0, base_width=6.0):
    nrows = int(ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(base_width * ncols, base_height * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    return fig, axes, nrows, ncols


def plot_redshift_comparison_grid(cube, analysis_specs, fixed_flag="GR", snaps=snaps, zmap=zmap, ncols=3, ylog=False):
    n_panels = len(analysis_specs)
    fig, axes, nrows, ncols_real = _setup_compare_axes(n_panels, ncols=ncols)

    for i, spec in enumerate(analysis_specs):
        ax = axes[i]
        sname = spec["name"]
        total_n = 0

        for snap in snaps:
            prof = cube[sname][fixed_flag][snap]
            x = prof["x"]
            y = prof["y"]
            total_n += prof["N_selected"]
            good = np.isfinite(x) & np.isfinite(y)
            if not np.any(good):
                continue

            label = f"snap={snap:03d}"
            if snap in zmap:
                label += f" (z={zmap[snap]:.2f})"

            ax.plot(x[good], y[good], color=snap_colors.get(snap, None), lw=LINEWIDTH, label=label)

        ax.set_title(spec["title"], fontsize=12)
        ax.set_xlim(R_RANGE)
        ax.grid(True, alpha=0.3)
        if ylog:
            ax.set_yscale("log")
        if SHOW_COUNTS_IN_PANEL:
            ax.text(0.96, 0.92, f"N={total_n}", ha='right', va='top', transform=ax.transAxes, fontsize=9)

    for k in range(len(analysis_specs), len(axes)):
        axes[k].axis("off")

    for idx, ax in enumerate(axes[:len(analysis_specs)]):
        row = idx // ncols_real
        col = idx % ncols_real
        if row == nrows - 1:
            ax.set_xlabel(r"$r/R_{200\mathrm{c}}$", fontsize=11)
        if col == 0:
            ax.set_ylabel(r"$p(r/R_{200\mathrm{c}})$" if DENSITY else r"${\rm counts}$", fontsize=11)

    fig.suptitle(f"Satellite radial distribution: redshift comparison ({fixed_flag})", y=0.995, fontsize=17)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.97),
                   ncol=min(len(handles), 3), frameon=False, title="Snapshots")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig, axes


def plot_gravity_comparison_grid(cube, analysis_specs, fixed_snap=21, flags=flags, zmap=zmap, ncols=3, ylog=False):
    n_panels = len(analysis_specs)
    fig, axes, nrows, ncols_real = _setup_compare_axes(n_panels, ncols=ncols)

    for i, spec in enumerate(analysis_specs):
        ax = axes[i]
        sname = spec["name"]
        total_n = 0

        for flag in flags:
            prof = cube[sname][flag][fixed_snap]
            x = prof["x"]
            y = prof["y"]
            total_n += prof["N_selected"]
            good = np.isfinite(x) & np.isfinite(y)
            if not np.any(good):
                continue

            ax.plot(x[good], y[good], color=flag_colors[flag], lw=LINEWIDTH, label=flag)

        ax.set_title(spec["title"], fontsize=12)
        ax.set_xlim(R_RANGE)
        ax.grid(True, alpha=0.3)
        if ylog:
            ax.set_yscale("log")
        if SHOW_COUNTS_IN_PANEL:
            ax.text(0.96, 0.92, f"N={total_n}", ha='right', va='top', transform=ax.transAxes, fontsize=9)

    for k in range(len(analysis_specs), len(axes)):
        axes[k].axis("off")

    for idx, ax in enumerate(axes[:len(analysis_specs)]):
        row = idx // ncols_real
        col = idx % ncols_real
        if row == nrows - 1:
            ax.set_xlabel(r"$r/R_{200\mathrm{c}}$", fontsize=11)
        if col == 0:
            ax.set_ylabel(r"$p(r/R_{200\mathrm{c}})$" if DENSITY else r"${\rm counts}$", fontsize=11)

    title = f"Satellite radial distribution: gravity-model comparison (snap={fixed_snap:03d}"
    if fixed_snap in zmap:
        title += f", z={zmap[fixed_snap]:.2f}"
    title += ")"
    fig.suptitle(title, y=0.995, fontsize=17)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.97),
                   ncol=min(len(handles), 6), frameon=False, title="Gravity models")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig, axes

# %% [markdown] cell 6
# ## 1. Original layout

# %% code cell 7
analysis_specs = make_analysis_specs()

snap = 21
profiles = build_profiles_for_snapshot(MAset, flags, snap, analysis_specs)

fig, axes = plot_original_grid(
    profiles,
    analysis_specs,
    flags=flags,
    snap=snap,
    zmap=zmap,
    figsize=(21, 18),
    ylog=YLOG,
)
plt.show()

# %% [markdown] cell 8
# ## 2. Different redshifts in the same subplot for each classification

# %% code cell 9
analysis_specs = make_analysis_specs()
cube = build_profile_cube(MAset, flags, snaps, analysis_specs)

fixed_flag = "GR"

fig, axes = plot_redshift_comparison_grid(
    cube,
    analysis_specs,
    fixed_flag=fixed_flag,
    snaps=snaps,
    zmap=zmap,
    ncols=3,
    ylog=YLOG,
)
plt.show()

# %% [markdown] cell 10
# ## 3. Different gravity models in the same subplot for each classification

# %% code cell 11
analysis_specs = make_analysis_specs()
cube = build_profile_cube(MAset, flags, snaps, analysis_specs)

fixed_snap = 21

fig, axes = plot_gravity_comparison_grid(
    cube,
    analysis_specs,
    fixed_snap=fixed_snap,
    flags=flags,
    zmap=zmap,
    ncols=3,
    ylog=YLOG,
)
plt.show()

# %% [markdown] cell 12
# ## 4. Optional loop: redshift-comparison figures for all gravity models ```python analysis_specs = make_analysis_specs() cube = build_profile_cube(MAset, flags, snaps, analysis_specs) for fixed_flag in flags: fig, axes = plot_redshift_comparison_grid( cube, analysis_specs, fixed_flag=fixed_flag, snaps=snaps, zmap=zmap, ncols=3, ylog=YLOG, ) plt.show() ```

# %% [markdown] cell 13
# ## 5. Optional loop: gravity-comparison figures for all snapshots ```python analysis_specs = make_analysis_specs() cube = build_profile_cube(MAset, flags, snaps, analysis_specs) for fixed_snap in snaps: fig, axes = plot_gravity_comparison_grid( cube, analysis_specs, fixed_snap=fixed_snap, flags=flags, zmap=zmap, ncols=3, ylog=YLOG, ) plt.show() ```

# %% [markdown] cell 14
# ## 6. Optional saving block ```python import os analysis_specs = make_analysis_specs() cube = build_profile_cube(MAset, flags, snaps, analysis_specs) outdir = "./satellite_radial_profiles_compare" os.makedirs(outdir, exist_ok=True) for fixed_flag in flags: fig, axes = plot_redshift_comparison_grid( cube, analysis_specs, fixed_flag=fixed_flag, snaps=snaps, zmap=zmap, ncols=3, ylog=YLOG, ) fig.savefig(f"{outdir}/radial_compare_redshift_{fixed_flag}.png", dpi=200, bbox_inches="tight") plt.close(fig

# %% code cell 15
