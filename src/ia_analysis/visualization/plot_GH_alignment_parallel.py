#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# Prevent BLAS/OpenMP oversubscription before importing numpy/scipy/sklearn
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import multiprocessing as mp
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from ia_analysis.visualization.DWE import DimrothWatson
from ia_analysis.shapes.Iana import II
from ia_analysis.visualization.arts import KMeans2DClusterer


# ============================================================
# Defaults
# ============================================================

DEFAULT_MA_PKL = "/cosma8/data/dp203/dc-wang17/MG_global/bak1/MA.pkl"
DEFAULT_OUTDIR = "./plots"

DEFAULT_ZMAP = {
    6: 0.97,
    12: 0.51,
    15: 0.33,
    18: 0.16,
    21: 0.00,
}

DEFAULT_FLAGS = ["F40", "F45", "F50", "F55", "F60", "GR"]
DEFAULT_SNAPS = [6, 12, 15, 18, 21]
DEFAULT_AXIS_LABELS = ["major", "medium", "minor"]


# ============================================================
# Globals shared by forked workers
# ============================================================

G_MASET = None
G_DW = None


# ============================================================
# Worker helpers
# ============================================================

def _init_worker():
    """Per-worker lightweight init."""
    global G_DW
    G_DW = DimrothWatson()


def _fit_mu(sample, min_cluster_size=8, symmetrize_mu=True,
            fit_bounds=(-500, 500), fit_max_iter=5000):
    """Fit cluster-level mu from a 1D cos(theta) sample."""
    global G_DW

    a = np.asarray(sample, dtype=float).ravel()
    a = a[np.isfinite(a) & (np.abs(a) <= 1.0)]

    if a.size < min_cluster_size:
        return np.nan

    if symmetrize_mu:
        a = np.r_[a, -a]

    try:
        out = G_DW.fit(a, bounds=fit_bounds, max_iter=fit_max_iter)
        return out.get("mu", np.nan)
    except Exception:
        return np.nan


def _prepare_panel_task(task):
    """
    Compute one panel in parallel.

    Returns a dict containing:
      axis_label, flag, snap, x, y, point_values
    where point_values already store cluster-level mu for each point.
    """
    global G_MASET

    (
        axis_label,
        flag,
        snap,
        n_clusters,
        random_state,
        min_cluster_size,
        symmetrize_mu,
    ) = task

    MA = G_MASET[flag][f"{snap:03d}"]

    Mstar = MA["SubhaloMassInRadType"][:, 4]
    M_DM = MA["SubhaloMassInRadType"][:, 1]
    axe_err = MA["cos_err_max_Star"][:]

    # Original logic was log10(M) > 0, i.e. M > 1
    valid_mass = (
        np.isfinite(Mstar) &
        np.isfinite(M_DM) &
        (Mstar > 1.0) &
        (M_DM > 1.0)
    )

    sat_ind_MA = valid_mass & np.isfinite(axe_err) & (axe_err < 0.01)

    if np.count_nonzero(sat_ind_MA) == 0:
        return {
            "axis_label": axis_label,
            "flag": flag,
            "snap": snap,
            "x": None,
            "y": None,
            "point_values": None,
        }

    x = np.log10(Mstar[sat_ind_MA]) + 10.0
    y = np.log10(M_DM[sat_ind_MA]) + 10.0

    mu_all = II(MA["I_Star"][sat_ind_MA], MA["I_DM"][sat_ind_MA])[axis_label]

    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(mu_all)
    if np.count_nonzero(good) == 0:
        return {
            "axis_label": axis_label,
            "flag": flag,
            "snap": snap,
            "x": None,
            "y": None,
            "point_values": None,
        }

    x = x[good]
    y = y[good]
    mu_all = mu_all[good]

    if x.size == 0:
        return {
            "axis_label": axis_label,
            "flag": flag,
            "snap": snap,
            "x": None,
            "y": None,
            "point_values": None,
        }

    K_eff = min(int(n_clusters), x.size)
    km = KMeans2DClusterer(n_clusters=K_eff, random_state=random_state).fit(x, y)

    labels = km.get_numeric_labels()
    masks = km.get_boolean_index()

    cluster_mu = np.full(K_eff, np.nan, dtype=float)
    for k in range(K_eff):
        cluster_mu[k] = _fit_mu(
            mu_all[masks[k]],
            min_cluster_size=min_cluster_size,
            symmetrize_mu=symmetrize_mu,
        )

    point_values = cluster_mu[labels]

    return {
        "axis_label": axis_label,
        "flag": flag,
        "snap": snap,
        "x": x,
        "y": y,
        "point_values": point_values,
    }


# ============================================================
# Plotting
# ============================================================

def _plot_one_axis_grid(
    panel_cache,
    zmap,
    flags,
    snaps,
    axis_label,
    *,
    outname,
    n_clusters=30,
    cmap="Blues",
    vmin=0.2,
    vmax=1.0,
    random_state=42,
    figsize=(16, 18),
    dpi=300,
    show_boundaries=False,
    boundary_mode="hull",
    boundary_kwargs=None,
):
    """
    Plot one 6x5 grid for a given axis_label from already computed panel cache.
    """
    if boundary_kwargs is None:
        boundary_kwargs = dict(
            linestyle="-",
            linewidth=0.8,
            color="k",
            alpha=0.7,
        )

    norm = Normalize(vmin=vmin, vmax=vmax)
    mapper = ScalarMappable(norm=norm, cmap=cmap)
    mapper.set_array([])

    # Global limits
    xmin, xmax = np.inf, -np.inf
    ymin, ymax = np.inf, -np.inf

    for flag in flags:
        for snap in snaps:
            pdata = panel_cache.get((flag, snap), None)
            if pdata is None or pdata["x"] is None:
                continue
            xmin = min(xmin, np.min(pdata["x"]))
            xmax = max(xmax, np.max(pdata["x"]))
            ymin = min(ymin, np.min(pdata["y"]))
            ymax = max(ymax, np.max(pdata["y"]))

    if not np.isfinite(xmin):
        raise ValueError(f"No valid panels found for axis_label={axis_label}")

    dx = xmax - xmin
    dy = ymax - ymin
    xpad = 0.03 * dx if dx > 0 else 0.1
    ypad = 0.05 * dy if dy > 0 else 0.1

    xlim = (xmin - xpad, xmax + xpad)
    ylim = (ymin - ypad, ymax + ypad)

    nrows = len(flags)
    ncols = len(snaps)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.90,
        bottom=0.07,
        top=0.94,
        wspace=0.03,
        hspace=0.03,
    )

    fig.suptitle(f"Galaxy-Halo Alignment ({axis_label.capitalize()})", fontsize=22)

    # Column titles
    for c, snap in enumerate(snaps):
        axes[0, c].set_title(rf"$z={zmap[snap]:.2f}$", fontsize=16, pad=8)

    # Draw panels
    for r, flag in enumerate(flags):
        for c, snap in enumerate(snaps):
            ax = axes[r, c]
            pdata = panel_cache.get((flag, snap), None)

            if pdata is not None and pdata["x"] is not None:
                x = pdata["x"]
                y = pdata["y"]
                point_values = pdata["point_values"]

                ax.scatter(
                    x,
                    y,
                    c=point_values,
                    cmap=cmap,
                    norm=norm,
                    s=6,
                    alpha=0.6,
                    edgecolors="none",
                    rasterized=True,
                )

                if show_boundaries:
                    K_eff = min(int(n_clusters), x.size)
                    km = KMeans2DClusterer(
                        n_clusters=K_eff,
                        random_state=random_state,
                    ).fit(x, y)
                    km.plot_boundaries(ax, mode=boundary_mode, **boundary_kwargs)

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.25)

            if c == 0:
                ax.tick_params(axis="y", labelleft=True)
                ax.set_ylabel(r"$M_{\rm DM}$", fontsize=14)
            else:
                ax.tick_params(axis="y", labelleft=False)

            if r == nrows - 1:
                ax.tick_params(axis="x", labelbottom=True)
                ax.set_xlabel(r"$M_\ast$", fontsize=14)
            else:
                ax.tick_params(axis="x", labelbottom=False)

            # Row label outside the leftmost ylabel
            if c == 0:
                ax.text(
                    -0.58, 0.5, flag,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                    clip_on=False,
                )

    # One colorbar per row, outside the rightmost panel
    for r in range(nrows):
        ax_right = axes[r, -1]
        bbox = ax_right.get_position()

        cax = fig.add_axes([
            bbox.x1 + 0.006,
            bbox.y0,
            0.010,
            bbox.height,
        ])
        cb = fig.colorbar(mapper, cax=cax)
        cb.set_label(r"$\mu$", fontsize=12)
        cax.yaxis.set_ticks_position("right")
        cax.yaxis.set_label_position("right")

    fig.savefig(outname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Driver
# ============================================================

def build_alignment_grids_parallel(
    *,
    ma_pkl=DEFAULT_MA_PKL,
    outdir=DEFAULT_OUTDIR,
    zmap=None,
    flags=None,
    snaps=None,
    axis_labels=None,
    n_clusters=30,
    cmap="Blues",
    vmin=0.2,
    vmax=1.0,
    random_state=42,
    min_cluster_size=8,
    symmetrize_mu=True,
    figsize=(16, 18),
    dpi=300,
    show_boundaries=False,
    boundary_mode="hull",
    boundary_kwargs=None,
    max_workers=None,
):
    """
    Parallel build of all GH alignment grids.
    """
    global G_MASET

    if zmap is None:
        zmap = DEFAULT_ZMAP
    if flags is None:
        flags = DEFAULT_FLAGS
    if snaps is None:
        snaps = DEFAULT_SNAPS
    if axis_labels is None:
        axis_labels = DEFAULT_AXIS_LABELS

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(ma_pkl, "rb") as f:
        G_MASET = pickle.load(f)

    if max_workers is None:
        max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

    tasks = []
    for axis_label in axis_labels:
        for flag in flags:
            for snap in snaps:
                tasks.append((
                    axis_label,
                    flag,
                    snap,
                    n_clusters,
                    random_state,
                    min_cluster_size,
                    symmetrize_mu,
                ))

    panel_cache_all = {axlb: {} for axlb in axis_labels}

    # COSMA/Linux: use fork so the loaded MAset is shared copy-on-write
    ctx = mp.get_context("fork")

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_init_worker,
    ) as ex:
        fut_to_task = {ex.submit(_prepare_panel_task, task): task for task in tasks}

        for fut in tqdm(as_completed(fut_to_task), total=len(fut_to_task), desc="Computing panels"):
            task = fut_to_task[fut]
            axis_label, flag, snap = task[0], task[1], task[2]
            try:
                res = fut.result()
            except Exception as e:
                raise RuntimeError(
                    f"Task failed for axis_label={axis_label}, flag={flag}, snap={snap:03d}"
                ) from e

            if res["x"] is None:
                panel_cache_all[axis_label][(flag, snap)] = None
            else:
                panel_cache_all[axis_label][(flag, snap)] = {
                    "x": res["x"],
                    "y": res["y"],
                    "point_values": res["point_values"],
                }

    for axis_label in axis_labels:
        outname = outdir / f"GH2D_{axis_label}_grid.png"
        _plot_one_axis_grid(
            panel_cache_all[axis_label],
            zmap,
            flags,
            snaps,
            axis_label,
            outname=str(outname),
            n_clusters=n_clusters,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            random_state=random_state,
            figsize=figsize,
            dpi=dpi,
            show_boundaries=show_boundaries,
            boundary_mode=boundary_mode,
            boundary_kwargs=boundary_kwargs,
        )


# ============================================================
# CLI
# ============================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel galaxy-halo alignment 6x5 grid plotter."
    )
    parser.add_argument("--ma-pkl", type=str, default=DEFAULT_MA_PKL,
                        help="Path to MA.pkl")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR,
                        help="Output directory for figures")
    parser.add_argument("--cpus", type=int, default=None,
                        help="Number of worker processes (default: SLURM_CPUS_PER_TASK or os.cpu_count)")
    parser.add_argument("--n-clusters", type=int, default=30,
                        help="Number of KMeans clusters per panel")
    parser.add_argument("--vmin", type=float, default=0.2,
                        help="Color scale minimum")
    parser.add_argument("--vmax", type=float, default=1.0,
                        help="Color scale maximum")
    parser.add_argument("--figsize", nargs=2, type=float, default=(16.0, 18.0),
                        metavar=("W", "H"), help="Figure size in inches")
    parser.add_argument("--show-boundaries", action="store_true",
                        help="Overlay KMeans boundaries on each panel")
    parser.add_argument("--axis-labels", nargs="+",
                        default=DEFAULT_AXIS_LABELS,
                        choices=["major", "medium", "minor"],
                        help="Axis labels to plot")
    return parser.parse_args()


def main():
    args = _parse_args()

    sns.set(style="ticks")
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["mathtext.fontset"] = "cm"

    build_alignment_grids_parallel(
        ma_pkl=args.ma_pkl,
        outdir=args.outdir,
        zmap=DEFAULT_ZMAP,
        flags=DEFAULT_FLAGS,
        snaps=DEFAULT_SNAPS,
        axis_labels=args.axis_labels,
        n_clusters=args.n_clusters,
        cmap="Blues",
        vmin=args.vmin,
        vmax=args.vmax,
        random_state=42,
        min_cluster_size=8,
        symmetrize_mu=True,
        figsize=tuple(args.figsize),
        dpi=300,
        show_boundaries=args.show_boundaries,
        boundary_mode="hull",
        boundary_kwargs=None,
        max_workers=args.cpus,
    )


if __name__ == "__main__":
    main()
