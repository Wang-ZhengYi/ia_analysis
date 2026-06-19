"""Exported code from notebooks/raw_20260618/MAset_satellite_radial_distribution.ipynb.

This file is generated for project management and refactoring. Review before running end to end.
"""


# %% [markdown] cell 1
# # Radial distribution of satellite galaxies from `MAset` This notebook computes the **radial distribution function of satellite galaxies** using the precomputed quantity `R_over_R_200c` in `MAset`. ## Definition used here For each selected satellite sample, this notebook measures the normalized 1D distribution \[ p(x), \qquad x \equiv r / R_{200\mathrm{c}}, \] such that \[ \int p(x)\, dx = 1. \] So the y-axis is a **normalized probability density**, not the raw number count. ## Classification sc

# %% code cell 2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

with open('/cosma8/data/dp203/dc-wang17/MG_global/bak1/MA.pkl', 'rb') as f:
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

print("MAset keys:", list(MAset.keys()))
print("Example snapshot keys:", list(MAset['GR'].keys()))
print("Model colors:", flag_colors)

# %% code cell 3
# ============================================================
# Configuration
# ============================================================

GROUP_ID_KEY = "GroupID"
SUBHALO_ID_KEY = "SubhaloID"
CENID_KEY = "CenID"
R_KEY = "R_over_R_200c"
MSTAR_KEY = "SubhaloMassInRadType"
SFR_KEY = "SubhaloSFR"
KAPPA_KEY = "kappa_rot_Star"

STELLAR_MASS_SCALE = 1e10
STELLAR_COMPONENT_INDEX = 4

R_RANGE = (0.0, 2.5)
NBINS = 25
RADIAL_BINS = np.linspace(R_RANGE[0], R_RANGE[1], NBINS + 1)

DENSITY = True
MIN_COUNT = 5

YLOG = False
LINEWIDTH = 2.0
SHOW_COUNTS_IN_PANEL = True

# %% code cell 4
# ============================================================
# Helper functions
# ============================================================

def get_snap_dict(MAset, flag, snap):
    x = MAset[flag]
    snap_key = f"{int(snap):03d}"
    if snap_key in x:
        return x[snap_key]
    if int(snap) in x:
        return x[int(snap)]
    if str(int(snap)) in x:
        return x[str(int(snap))]
    raise KeyError(
        f"Cannot find snapshot={snap} inside MAset['{flag}']. "
        f"Available keys: {list(x.keys())[:10]}"
    )


def get_arr(data, key):
    if key == "SubhaloMassInRadType":
        arr = np.asarray(data[key])
        if arr.ndim != 2:
            raise ValueError(f"Expected {key} to be 2D, got shape={arr.shape}")
        if arr.shape[1] <= STELLAR_COMPONENT_INDEX:
            raise ValueError(
                f"{key} has shape {arr.shape}; cannot access column {STELLAR_COMPONENT_INDEX}"
            )
        return np.asarray(arr[:, STELLAR_COMPONENT_INDEX], dtype=float) * STELLAR_MASS_SCALE

    if key not in data:
        raise KeyError(f"Key '{key}' not found in data.")

    arr = np.asarray(data[key])
    if arr.ndim != 1:
        raise ValueError(f"data['{key}'] must be 1D, got shape={arr.shape}")
    return np.asarray(arr)


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
            if not (isinstance(rule, (tuple, list)) and len(rule) == 2):
                raise ValueError(f"Limit for '{key}' must be a callable or a (low, high) tuple.")
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
            if total > 0:
                y = counts / (total * widths)
            else:
                y = np.full_like(centers, np.nan, dtype=float)
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


def plot_model_by_sample_grid(
    profiles,
    analysis_specs,
    *,
    flags=flags,
    snap=None,
    zmap=None,
    figsize=(21, 18),
    ylog=False,
):
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
                ax.plot(
                    x[good], y[good],
                    color=flag_colors[flag],
                    lw=LINEWIDTH,
                )

            if i == 0:
                ax.set_title(flag, fontsize=13)

            if j == 0:
                ylabel = spec["title"] + "\n"
                ylabel += r"$p(r/R_{200\mathrm{c}})$" if DENSITY else r"${\rm counts}$"
                ax.set_ylabel(ylabel, fontsize=11)

            if SHOW_COUNTS_IN_PANEL:
                ax.text(
                    0.96, 0.92,
                    f"N={nsel}",
                    ha='right', va='top',
                    transform=ax.transAxes,
                    fontsize=9,
                )

            if ylog:
                ax.set_yscale("log")

            ax.set_xlim(R_RANGE)
            ax.grid(True, alpha=0.3)

    for j in range(ncols):
        axes[-1, j].set_xlabel(r"$r/R_{200\mathrm{c}}$", fontsize=11)

    if snap is not None:
        stitle = "Satellite radial distribution"
        if zmap is not None and snap in zmap:
            stitle += f" at snap={snap:03d} (z={zmap[snap]:.2f})"
        else:
            stitle += f" at snap={snap:03d}"
        fig.suptitle(stitle, y=0.995, fontsize=17)

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    return fig, axes

# %% [markdown] cell 5
# ## Single-snapshot example This cell generates the requested layout for one chosen snapshot.

# %% code cell 6
snap = 21

analysis_specs = make_analysis_specs()
profiles = build_profiles_for_snapshot(MAset, flags, snap, analysis_specs)

fig, axes = plot_model_by_sample_grid(
    profiles,
    analysis_specs,
    flags=flags,
    snap=snap,
    zmap=zmap,
    figsize=(21, 18),
    ylog=YLOG,
)

plt.show()

# %% [markdown] cell 7
# ## Loop over all snapshots Run this cell to generate one figure for each snapshot.

# %% code cell 8
for snap in snaps:
    analysis_specs = make_analysis_specs()
    profiles = build_profiles_for_snapshot(MAset, flags, snap, analysis_specs)

    fig, axes = plot_model_by_sample_grid(
        profiles,
        analysis_specs,
        flags=flags,
        snap=snap,
        zmap=zmap,
        figsize=(21, 18),
        ylog=YLOG,
    )

    plt.show()

# %% [markdown] cell 9
# ## Optional: save figures to disk Uncomment and run if you want PNG outputs. ```python import os outdir = "./satellite_radial_profiles" os.makedirs(outdir, exist_ok=True) for snap in snaps: analysis_specs = make_analysis_specs() profiles = build_profiles_for_snapshot(MAset, flags, snap, analysis_specs) fig, axes = plot_model_by_sample_grid( profiles, analysis_specs, flags=flags, snap=snap, zmap=zmap, figsize=(21, 18), ylog=YLOG, ) fig.savefig( f"{outdir}/satellite_radial_profiles_snap{snap:03d}.
