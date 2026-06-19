"""Exported code from notebooks/raw_20260618/merger_align.ipynb.

This file is generated for project management and refactoring. Review before running end to end.
"""


# %% [markdown] cell 1
# # TNG merger-tree alignment, shell dynamics, and density visualization (`merger_align`) This notebook computes and plots the cross-redshift evolution of a tracked TNG subhalo along its SubLink main progenitor branch. The workflow is intentionally split into a compute/save stage and a read/plot stage: 1. Compute/download TNG products and save them to `PRODUCTS_PATH`. 2. Reload the saved products for all plots. 3. Optionally enrich the products with target gas and FoF/environment particles. 4. Plo

# %% code cell 2
from pathlib import Path
import os
import sys
import pickle
import importlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import patheffects
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

try:
    from DWE import DimrothWatson
    dw = DimrothWatson()
except Exception:
    dw = None

from TNGCatLoader import TNGCatalog
import hd_tng
import halo_dynamics
import shape
import Iana

importlib.reload(hd_tng)
importlib.reload(halo_dynamics)
importlib.reload(shape)
importlib.reload(Iana)

print('hd_tng loaded from:', hd_tng.__file__)

# %% [markdown] cell 3
# ## Configuration

# %% code cell 4

SIM_NAME = os.environ.get('TNG_SIM_NAME', 'TNG300-1')
BASE_PATH = os.environ.get('TNG_BASE_PATH', '/cosma7/data/dp203/dc-wang17/TNG/tng_data')
SNAP = int(os.environ.get('TNG_SNAP', '99'))
API_KEY = os.environ.get('TNG_API_KEY', 'ec7a0419719cacfd0a27d964d8993b9d')

OUTDIR = Path('hd_tng_outputs')
OUTDIR.mkdir(exist_ok=True)

# Persistent API/cutout cache.  This is important: using system temp causes
# repeated downloads between cells/kernels.  Override with env TNG_API_CACHE.
_default_cache = str(Path(BASE_PATH).parent / 'tng_api_cache')
TNG_API_CACHE = Path(os.environ.get('TNG_API_CACHE', _default_cache))
try:
    TNG_API_CACHE.mkdir(parents=True, exist_ok=True)
except Exception:
    TNG_API_CACHE = OUTDIR / 'tng_api_cache'
    TNG_API_CACHE.mkdir(parents=True, exist_ok=True)

CFG = dict(
    sim_name=SIM_NAME,
    snap=SNAP,
    api_key=API_KEY,
    download_if_missing=True,
    cache_dir=str(TNG_API_CACHE),
    delete_cache=False,
    top_n_groups=1,
    max_subhaloes_per_group=5,
    min_dm_particles=200,
    include_central=True,
    n_radial_shells=6,
    n_binding_shells=6,
    min_particles_per_shell=100,
    shell_methods=('radial', 'binding_energy'),
    keep_particles=True,
    compute_binding_potential_if_missing=True,
    api_max_retries=6,
    api_retry_base_sleep=5.0,
    api_retry_max_sleep=90.0,
    verbose=True,
    auto_select_targets=True,
)

TNG_CATALOG_KWARGS = dict(
    sim_name=SIM_NAME,
    api_key=API_KEY,
    download_if_missing=True,
    cache_dir=str(TNG_API_CACHE),
    delete_cache=False,
    verbose=True,
    timeout=int(CFG.get('timeout', 180)),
    max_retries=int(CFG.get('api_max_retries', 6)),
    retry_base_sleep=float(CFG.get('api_retry_base_sleep', 5.0)),
    retry_max_sleep=float(CFG.get('api_retry_max_sleep', 90.0)),
)

print('SIM_NAME =', SIM_NAME)
print('BASE_PATH =', BASE_PATH)
print('SNAP =', SNAP)
print('TNG_API_CACHE =', TNG_API_CACHE)
print('delete_cache =', CFG['delete_cache'])

# Output product file.  The first half of the notebook writes this file;
# the plotting half only reads this file.
PRODUCTS_PATH = OUTDIR / f'subhalo_{SID0 if "SID0" in globals() else 3}_crossz_products.pkl'

# %% [markdown] cell 5
# ## User controls

# %% code cell 6

# ============================================================
# User controls for cross-time analysis
# ============================================================
RUN_CROSS_TIME = True
SID0 = 3
SNAP0 = SNAP
SNAP_TRACK = [99, 84, 72, 67, 59, 50, 40, 33]
SHELL_METHODS = ('radial', 'binding_energy')

print('SID0 =', SID0)
print('SNAP0 =', SNAP0)
print('SNAP_TRACK =', SNAP_TRACK)
print('SHELL_METHODS =', SHELL_METHODS)

PRODUCTS_PATH = OUTDIR / f'subhalo_{SID0}_crossz_products.pkl'
FORCE_RECOMPUTE = False
print('PRODUCTS_PATH =', PRODUCTS_PATH)


# Optional density-particle enrichment used by the FoF/environment density plots.
# This loads extra particle cutouts after PRODUCTS is available.
RUN_DENSITY_PARTICLE_ENRICHMENT = True
SAVE_PRODUCTS_AFTER_DENSITY_ENRICHMENT = True
DENSITY_ENRICH_METHODS = ('radial',)      # density plots use radial records by default
FOF_GID_MODE = 'host_central'             # use FoF group of the z=0 host-central MPB
MAX_FOF_PARTICLES_PER_TYPE = 1_000_000    # set to None for full FoF particles, but memory can be high
MAX_SUBHALO_GAS_PARTICLES = None

# %% [markdown] cell 7
# ## Compute helpers

# %% code cell 8
# Private compute helper. Do not call this for plotting; use compute_cross_time_products first, then plot_* functions.
# ============================================================
# _compute_orbitplane_products
# ============================================================
# Function:
#   1. Input SID0 and SNAP_TRACK.
#   2. Find the FoF group containing SID0 at SNAP0.
#   3. Use that z=0 FoF group's central subhalo as the host-central reference.
#   4. Track both:
#        - target SID0 along its MPB;
#        - z=0 host central along its MPB.
#   5. At each snapshot, plot target subhalo particles relative to the
#      back-traced z=0 host-central progenitor.
#   6. Fit an orbit-plane projection from these relative positions.
#   7. Draw density, density contours, shell ellipses, redshift labels,
#      FoF-centre marker, and galaxy major-axis direction.
#   8. Build and return a detailed per-snapshot/per-shell diagnostic table.
#
# Return:
#   fig, ax, records, table_df, save_path
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator
from pathlib import Path


def _compute_orbitplane_products(
    SID0,
    SNAP_TRACK,
    *,
    SNAP0=99,
    BASE_PATH=None,
    SIM_NAME=None,
    API_KEY=None,
    CFG=None,
    OUTDIR=None,
    TNG_CATALOG_KWARGS=None,

    SHELL_METHOD="radial",
    N_SHELLS=None,
    MIN_PARTICLES_PER_SHELL=None,

    COMMON_LIM_KPC=None,
    PAD_FACTOR=1.15,
    PLOT_IN_TIME_ORDER=True,

    DENSITY_BINS_PER_SUBHALO=400,
    DENSITY_ALPHA=0.72,
    DENSITY_PERCENTILE_LIMIT=98.0,
    DENSITY_CMAP="magma",
    DRAW_PARTICLE_DENSITY=True,
    DRAW_DENSITY_CONTOURS=True,
    DENSITY_CONTOUR_LEVELS=6,
    DENSITY_CONTOUR_COLOR="#9eeaff",
    DENSITY_CONTOUR_LW=0.60,

    DRAW_SHELL_ELLIPSES=True,
    ELLIPSE_NSIGMA=2.0,
    SHELL_COLOR="#a5ba9e",
    SHELL_BASE_LW=0.95,
    SHELL_STROKE_LW=1.45,

    DRAW_FOF_CENTRAL_MARKER=True,
    DRAW_GALAXY_MAJOR_AXIS=True,
    GALAXY_MAJOR_AXIS_COLOR="#ff2020",
    GALAXY_MAJOR_AXIS_LW=1.65,
    GALAXY_MAJOR_AXIS_NSIGMA=2.2,

    FIG_FACE="#d9d9d9",
    AX_FACE="black",
    TICK_COLOR="white",
    TICKLABEL_COLOR="black",
    SPINE_COLOR="white",
    TEXT_COLOR="black",

    make_figure=True,
    display_table=True,
    save=True,
    show=True,
    dpi=220,
):
    """
    Compute one tracked subhalo at multiple snapshots in one fitted orbit plane.

    Reference frame
    ---------------
    The origin at each snapshot is the MPB progenitor of the z=0 host central,
    where "host central" means the central subhalo of the FoF group containing
    SID0 at SNAP0.

    Table
    -----
    One row per snapshot per shell. The table includes:
      - target SID/GID and z=0 host-central MPB SID/GID;
      - dark matter mass, baryon mass, baryon/DM mass ratio;
      - galaxy/global stellar axes and global DM halo axes;
      - cosines between galaxy and DM axes;
      - cosines between axes and the host-centric radial direction;
      - shell DM axes;
      - cosines between global galaxy axes and shell axes;
      - cosines between shell axes and the radial direction.
    """

    # ------------------------------------------------------------
    # Resolve globals
    # ------------------------------------------------------------
    if BASE_PATH is None:
        BASE_PATH = globals()["BASE_PATH"]
    if SIM_NAME is None:
        SIM_NAME = globals()["SIM_NAME"]
    if API_KEY is None:
        API_KEY = globals().get("API_KEY", None)
    if CFG is None:
        CFG = globals()["CFG"]
    if OUTDIR is None:
        OUTDIR = Path(globals().get("OUTDIR", "hd_tng_outputs"))
    else:
        OUTDIR = Path(OUTDIR)

    OUTDIR.mkdir(parents=True, exist_ok=True)

    SID0 = int(SID0)
    SNAP0 = int(SNAP0)
    SNAP_TRACK = [int(s) for s in SNAP_TRACK]

    bad_snaps = [s for s in SNAP_TRACK if s > SNAP0]
    if bad_snaps:
        raise ValueError(
            f"SNAP_TRACK contains snapshots greater than SNAP0={SNAP0}: {bad_snaps}. "
            "This MPB-based plot tracks backward in time, so use snap <= SNAP0."
        )

    if N_SHELLS is None:
        if str(SHELL_METHOD).startswith("binding"):
            N_SHELLS = int(CFG.get("n_binding_shells", CFG.get("n_radial_shells", 6)))
        else:
            N_SHELLS = int(CFG.get("n_radial_shells", 6))

    if MIN_PARTICLES_PER_SHELL is None:
        MIN_PARTICLES_PER_SHELL = int(CFG.get("min_particles_per_shell", 100))

    TNG_API_CACHE = globals().get(
        "TNG_API_CACHE",
        "/cosma8/data/dp203/dc-wang17/TNG/tng_api_cache",
    )
    Path(TNG_API_CACHE).mkdir(parents=True, exist_ok=True)

    if TNG_CATALOG_KWARGS is None:
        TNG_CATALOG_KWARGS = globals().get(
            "TNG_CATALOG_KWARGS",
            dict(
                sim_name=SIM_NAME,
                api_key=API_KEY,
                download_if_missing=True,
                cache_dir=TNG_API_CACHE,
                delete_cache=False,
                verbose=True,
                timeout=int(CFG.get("timeout", 180)),
                max_retries=int(CFG.get("api_max_retries", 6)),
                retry_base_sleep=float(CFG.get("api_retry_base_sleep", 5.0)),
                retry_max_sleep=float(CFG.get("api_retry_max_sleep", 90.0)),
            ),
        )

    RETRY_CFG = dict(
        max_retries=int(CFG.get("api_max_retries", 6)),
        base_sleep=float(CFG.get("api_retry_base_sleep", 5.0)),
        max_sleep=float(CFG.get("api_retry_max_sleep", 90.0)),
        verbose=bool(CFG.get("verbose", True)),
    )

    # ------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------
    def _open_catalog_compat(base_path, snap, *, group_fields=None, subhalo_fields=None):
        try:
            return hd_tng.open_catalog(
                base_path,
                int(snap),
                group_fields=group_fields,
                subhalo_fields=subhalo_fields,
                tng_catalog_kwargs=TNG_CATALOG_KWARGS,
                retry_cfg=RETRY_CFG,
            )
        except TypeError:
            return hd_tng.open_catalog(
                base_path,
                int(snap),
                tng_catalog_kwargs=TNG_CATALOG_KWARGS,
                retry_cfg=RETRY_CFG,
            )

    def _load_sublink_mpb_compat(base_path, snap, sid):
        try:
            return hd_tng.load_sublink_mpb(
                base_path,
                int(snap),
                int(sid),
                tng_catalog_kwargs=TNG_CATALOG_KWARGS,
            )
        except TypeError:
            return hd_tng.load_sublink_mpb(
                base_path,
                int(snap),
                int(sid),
            )

    def _tree_track_map(tree, snap_track):
        df = hd_tng.tree_to_dataframe(tree)
        sub = hd_tng.select_tree_rows_for_snaps(df, snap_track)
        if len(sub) == 0:
            return {}, sub

        out = {
            int(row["SnapNum"]): int(row["SubfindID"])
            for _, row in sub.iterrows()
        }
        return out, sub

    def _ckpc_h_to_physical_kpc(x, header):
        if hasattr(hd_tng, "ckpc_h_to_physical_kpc"):
            return hd_tng.ckpc_h_to_physical_kpc(x, header)
        return np.asarray(x, dtype=float) * hd_tng.scale_factor_from_header(header) / hd_tng.hubble_from_header(header)

    def _mass_1e10_msun_h_to_msun(x, header):
        h = hd_tng.hubble_from_header(header)
        return np.asarray(x, dtype=float) * 1.0e10 / h

    # ------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------
    def _shape_axes_from_points(X, masses=None, min_particles=20):
        """
        Return shape tensor eigenvalues and axes.

        Axes are columns of R:
            R[:, 0] = major axis
            R[:, 1] = intermediate axis
            R[:, 2] = minor axis

        Since these are unoriented axes, later cosine comparisons use abs(dot).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 3:
            return dict(valid=False, evals=np.full(3, np.nan), R=np.full((3, 3), np.nan), N=0)

        good = np.all(np.isfinite(X), axis=1)
        X = X[good]

        if masses is None:
            w = np.ones(X.shape[0], dtype=float)
        else:
            w = np.asarray(masses, dtype=float)[good]
            w = np.where(np.isfinite(w) & (w > 0), w, 0.0)

        good_w = w > 0
        X = X[good_w]
        w = w[good_w]

        if X.shape[0] < int(min_particles):
            return dict(valid=False, evals=np.full(3, np.nan), R=np.full((3, 3), np.nan), N=int(X.shape[0]))

        cen = np.sum(X * w[:, None], axis=0) / np.sum(w)
        Y = X - cen[None, :]
        I = np.einsum("n,ni,nj->ij", w, Y, Y) / np.sum(w)
        I = 0.5 * (I + I.T)

        evals, R = np.linalg.eigh(I)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        R = R[:, idx]

        # Enforce right-handed frame. Axis signs are still arbitrary.
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1.0

        return dict(valid=True, evals=evals, R=R, N=int(X.shape[0]), center=cen, I=I)

    def _abs_cos(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.shape != (3,) or b.shape != (3,):
            return np.nan
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if not (np.isfinite(na) and np.isfinite(nb)) or na <= 0 or nb <= 0:
            return np.nan
        return float(abs(np.dot(a, b) / (na * nb)))

    def _axis_cols(prefix, R):
        out = {}
        names = ["major", "intermediate", "minor"]
        R = np.asarray(R, dtype=float)
        for iax, name in enumerate(names):
            if R.shape == (3, 3) and np.all(np.isfinite(R[:, iax])):
                out[f"{prefix}_{name}_x"] = float(R[0, iax])
                out[f"{prefix}_{name}_y"] = float(R[1, iax])
                out[f"{prefix}_{name}_z"] = float(R[2, iax])
            else:
                out[f"{prefix}_{name}_x"] = np.nan
                out[f"{prefix}_{name}_y"] = np.nan
                out[f"{prefix}_{name}_z"] = np.nan
        return out

    def _axis_evals_cols(prefix, evals):
        evals = np.asarray(evals, dtype=float)
        names = ["major", "intermediate", "minor"]
        out = {}
        for i, name in enumerate(names):
            out[f"{prefix}_{name}_lambda"] = float(evals[i]) if evals.size > i and np.isfinite(evals[i]) else np.nan
            out[f"{prefix}_{name}_rms_kpc"] = float(np.sqrt(evals[i])) if evals.size > i and np.isfinite(evals[i]) and evals[i] >= 0 else np.nan
        return out

    # ------------------------------------------------------------
    # Particle loading helpers
    # ------------------------------------------------------------
    def _load_subhalo_star_particles(cat, subs, sid, *, snap, base_path, header, retry_cfg):
        """
        Load stellar particles. If unavailable or too few, return an empty set.
        """
        sid = int(sid)
        fields = ["Coordinates", "Masses", "ParticleIDs"]

        try:
            pdata = hd_tng.retry_call(
                cat.loadSubhalos,
                sid,
                ptypes=[4],
                fields=fields,
                **retry_cfg,
            )
            p4 = pdata.get("PartType4", {})
        except Exception as exc:
            print(f"[stars] snap={snap}, sid={sid}: star cutout unavailable: {exc}")
            p4 = {}

        if "Coordinates" not in p4:
            return dict(
                sid=sid,
                snap=int(snap),
                coords_ckpc_h=np.empty((0, 3), dtype=float),
                X_kpc=np.empty((0, 3), dtype=float),
                X_hostref_kpc=np.empty((0, 3), dtype=float),
                masses=np.empty(0, dtype=float),
                ids=np.empty(0, dtype=np.int64),
            )

        coords = np.asarray(p4["Coordinates"], dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
            return dict(
                sid=sid,
                snap=int(snap),
                coords_ckpc_h=np.empty((0, 3), dtype=float),
                X_kpc=np.empty((0, 3), dtype=float),
                X_hostref_kpc=np.empty((0, 3), dtype=float),
                masses=np.empty(0, dtype=float),
                ids=np.empty(0, dtype=np.int64),
            )

        center_ckpc_h = np.asarray(subs["SubhaloPos"][sid], dtype=float)

        X_kpc = hd_tng.tng_relative_positions_to_physical_kpc(
            coords,
            center_ckpc_h,
            header,
        )

        if "Masses" in p4:
            masses = _mass_1e10_msun_h_to_msun(np.asarray(p4["Masses"], dtype=float), header)
        else:
            masses = np.ones(coords.shape[0], dtype=float)

        ids = np.asarray(p4.get("ParticleIDs", np.arange(coords.shape[0])), dtype=np.int64)

        return dict(
            sid=sid,
            snap=int(snap),
            coords_ckpc_h=coords,
            X_kpc=X_kpc,
            masses=masses,
            ids=ids,
        )

    # ------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------
    def _safe_ellipse_from_points(xy, weights=None, nsigma=2.0):
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 5:
            return None

        if weights is None:
            w = np.ones(xy.shape[0], dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if w.shape != (xy.shape[0],):
                return None

        good = np.isfinite(xy).all(axis=1) & np.isfinite(w) & (w > 0)
        if np.count_nonzero(good) < 5:
            return None

        xy = xy[good]
        w = w[good]

        cen = np.sum(xy * w[:, None], axis=0) / np.sum(w)
        Y = xy - cen[None, :]
        C = np.einsum("n,ni,nj->ij", w, Y, Y) / np.sum(w)
        C = 0.5 * (C + C.T)

        vals, vecs = np.linalg.eigh(C)
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]

        if np.any(vals <= 0) or not np.all(np.isfinite(vals)):
            return None

        width = 2.0 * nsigma * np.sqrt(vals[0])
        height = 2.0 * nsigma * np.sqrt(vals[1])
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

        return cen, width, height, angle

    def _add_scalebar_with_endticks(ax, xlim, ylim, length_kpc=None, label=None, loc=(0.08, 0.085)):
        x0, x1 = map(float, xlim)
        y0, y1 = map(float, ylim)
        xr = abs(x1 - x0)
        yr = abs(y1 - y0)

        if length_kpc is None:
            raw = 0.18 * xr
            pow10 = 10 ** np.floor(np.log10(max(raw, 1e-6)))
            candidates = np.array([1, 2, 5, 10], dtype=float) * pow10
            length_kpc = float(candidates[np.argmin(np.abs(candidates - raw))])

        if label is None:
            label = f"{int(length_kpc):d} kpc" if abs(length_kpc - int(length_kpc)) < 1e-8 else f"{length_kpc:g} kpc"

        xb = x0 + loc[0] * xr
        yb = y0 + loc[1] * yr
        tick_h = 0.026 * yr

        ax.plot([xb, xb + length_kpc], [yb, yb], lw=0.85, color="white", solid_capstyle="butt", zorder=1000)
        ax.plot([xb, xb], [yb - tick_h / 2.0, yb + tick_h / 2.0], lw=0.85, color="white", zorder=1001)
        ax.plot([xb + length_kpc, xb + length_kpc], [yb - tick_h / 2.0, yb + tick_h / 2.0], lw=0.85, color="white", zorder=1001)

        ax.text(
            xb + 0.5 * length_kpc,
            yb + 0.045 * yr,
            label,
            ha="center",
            va="bottom",
            fontsize=13,
            color="white",
            fontweight="bold",
            zorder=1002,
            path_effects=[patheffects.withStroke(linewidth=2.2, foreground="black", alpha=0.95)],
        )

    def _draw_density_map_with_contours(ax, xy, xlim, ylim):
        xy = np.asarray(xy, dtype=float)
        good = (
            np.isfinite(xy).all(axis=1)
            & (xy[:, 0] >= xlim[0]) & (xy[:, 0] <= xlim[1])
            & (xy[:, 1] >= ylim[0]) & (xy[:, 1] <= ylim[1])
        )
        xy = xy[good]

        if xy.shape[0] < 12:
            return None

        H, xe, ye = np.histogram2d(
            xy[:, 0],
            xy[:, 1],
            bins=int(DENSITY_BINS_PER_SUBHALO),
            range=[xlim, ylim],
        )
        H = H.T

        pos = H > 0
        if np.count_nonzero(pos) < 5:
            return None

        img = np.full_like(H, np.nan, dtype=float)
        img[pos] = np.log10(H[pos])

        im = ax.imshow(
            img,
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
            origin="lower",
            interpolation="nearest",
            cmap=DENSITY_CMAP,
            alpha=DENSITY_ALPHA,
            aspect="equal",
            zorder=2,
        )

        if not DRAW_DENSITY_CONTOURS:
            return im

        valid = np.isfinite(img)
        if np.count_nonzero(valid) < 5:
            return im

        vmin = np.nanmin(img[valid])
        vmax = np.nanmax(img[valid])

        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return im

        levels = np.linspace(vmin + 0.20 * (vmax - vmin), vmax, int(DENSITY_CONTOUR_LEVELS))
        xc = 0.5 * (xe[:-1] + xe[1:])
        yc = 0.5 * (ye[:-1] + ye[1:])

        ax.contour(
            xc,
            yc,
            img,
            levels=levels,
            colors=DENSITY_CONTOUR_COLOR,
            linewidths=DENSITY_CONTOUR_LW,
            alpha=0.72,
            zorder=6,
        )

        return im

    def _draw_shell_ellipses_only(ax, xy_global, masks, masses):
        if not DRAW_SHELL_ELLIPSES:
            return

        for ish, mi in enumerate(masks):
            mi = np.asarray(mi, dtype=bool)

            if mi.shape[0] != xy_global.shape[0]:
                continue
            if np.count_nonzero(mi) < MIN_PARTICLES_PER_SHELL:
                continue

            pts = xy_global[mi]
            wm = masses[mi] if masses is not None else None

            ell = _safe_ellipse_from_points(
                pts,
                weights=wm,
                nsigma=ELLIPSE_NSIGMA,
            )

            if ell is None:
                continue

            cen, width, height, angle = ell
            alpha = 0.42 + 0.45 * (ish + 1) / max(len(masks), 1)

            e = Ellipse(
                cen,
                width=width,
                height=height,
                angle=angle,
                fill=False,
                lw=SHELL_BASE_LW,
                color=SHELL_COLOR,
                alpha=alpha,
                zorder=30 + ish,
                path_effects=[
                    patheffects.withStroke(
                        linewidth=SHELL_STROKE_LW,
                        foreground="white",
                        alpha=0.30,
                    )
                ],
            )
            ax.add_patch(e)

    def _label_position_outside_subhalo(xy_global, center_xy, xlim, ylim):
        xy = np.asarray(xy_global, dtype=float)
        center_xy = np.asarray(center_xy, dtype=float)

        good = np.isfinite(xy).all(axis=1)
        xy = xy[good]

        if xy.shape[0] < 5 or not np.all(np.isfinite(center_xy)):
            return center_xy

        if np.linalg.norm(center_xy) > 1e-10:
            direction = center_xy / np.linalg.norm(center_xy)
        else:
            Y = xy - np.nanmedian(xy, axis=0, keepdims=True)
            C = np.cov(Y.T)
            vals, vecs = np.linalg.eigh(C)
            direction = vecs[:, np.argmax(vals)]
            direction = direction / max(np.linalg.norm(direction), 1e-12)

        rel = xy - center_xy[None, :]
        proj = rel @ direction
        finite = np.isfinite(proj)

        if np.count_nonzero(finite) == 0:
            return center_xy

        outer = np.nanpercentile(proj[finite], 98.0)
        outer = max(float(outer), 0.0)

        xrange = abs(xlim[1] - xlim[0])
        yrange = abs(ylim[1] - ylim[0])
        pad = 0.018 * max(xrange, yrange)

        label_xy = center_xy + direction * (outer + pad)

        margin_x = 0.03 * xrange
        margin_y = 0.03 * yrange
        label_xy[0] = np.clip(label_xy[0], xlim[0] + margin_x, xlim[1] - margin_x)
        label_xy[1] = np.clip(label_xy[1], ylim[0] + margin_y, ylim[1] - margin_y)

        return label_xy

    # ------------------------------------------------------------
    # Orbit-plane helpers
    # ------------------------------------------------------------
    def _fit_orbit_plane_basis_from_centers(centers, eps=1e-12):
        C = np.asarray(centers, dtype=float)
        good = np.all(np.isfinite(C), axis=1)
        Cg = C[good]

        if Cg.shape[0] >= 3:
            C0 = Cg - np.mean(Cg, axis=0, keepdims=True)
            _, svals, Vt = np.linalg.svd(C0, full_matrices=True)

            e1 = Vt[0].astype(float)
            e2 = Vt[1].astype(float)

            d = Cg[-1] - Cg[0]
            if np.linalg.norm(d) > eps and np.dot(e1, d) < 0:
                e1 *= -1.0

            n = np.cross(e1, e2)
            n_norm = np.linalg.norm(n)
            if n_norm < eps:
                raise RuntimeError("Degenerate orbit plane from PCA.")

            n /= n_norm
            e2 = np.cross(n, e1)
            e2 /= np.linalg.norm(e2)

        elif Cg.shape[0] == 2:
            e1 = Cg[1] - Cg[0]
            e1_norm = np.linalg.norm(e1)

            if e1_norm < eps:
                e1 = np.array([1.0, 0.0, 0.0])
            else:
                e1 /= e1_norm

            ref = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(e1, ref)) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])

            e2 = ref - np.dot(ref, e1) * e1
            if np.linalg.norm(e2) < eps:
                e2 = np.array([0.0, 1.0, 0.0])
            e2 /= np.linalg.norm(e2)

            n = np.cross(e1, e2)
            n /= np.linalg.norm(n)
            svals = np.array([np.linalg.norm(Cg[1] - Cg[0]), 0.0, 0.0])

        else:
            e1 = np.array([1.0, 0.0, 0.0])
            e2 = np.array([0.0, 1.0, 0.0])
            n = np.array([0.0, 0.0, 1.0])
            svals = np.zeros(3)

        return dict(e1=e1, e2=e2, n=n, singular_values=svals)

    def _project_to_orbit_plane(X3, basis):
        X3 = np.asarray(X3, dtype=float)
        return np.column_stack([
            X3 @ basis["e1"],
            X3 @ basis["e2"],
        ])

    # ------------------------------------------------------------
    # 1. Identify z=0 host FoF central
    # ------------------------------------------------------------
    header0 = hd_tng.read_header_for_snap(
        BASE_PATH,
        SNAP0,
        sim_name=SIM_NAME,
        api_key=API_KEY,
    )

    cat0, halos0, subs0 = _open_catalog_compat(
        BASE_PATH,
        SNAP0,
        group_fields=[
            "GroupFirstSub",
            "GroupLenType",
            "Group_R_Crit200",
            "GroupPos",
        ],
        subhalo_fields=[
            "SubhaloGrNr",
            "SubhaloLenType",
            "SubhaloPos",
            "SubhaloVel",
            "SubhaloMassType",
        ],
    )

    try:
        gid0 = int(np.asarray(subs0["SubhaloGrNr"], dtype=np.int64)[SID0])
        cen_sid0 = int(np.asarray(halos0["GroupFirstSub"], dtype=np.int64)[gid0])

        if cen_sid0 < 0:
            raise RuntimeError(
                f"SNAP0={SNAP0}, GID0={gid0} has invalid GroupFirstSub={cen_sid0}."
            )

        print(f"[reference] SNAP0={SNAP0}, target SID0={SID0}")
        print(f"[reference] target's z=0 FoF GroupID={gid0}")
        print(f"[reference] z=0 host central CenID0={cen_sid0}")

    finally:
        try:
            cat0.cleanup()
        except Exception:
            pass
        if cat0 in hd_tng._OPEN_CATALOGS:
            hd_tng._OPEN_CATALOGS.remove(cat0)

    # ------------------------------------------------------------
    # 2. Load target MPB and z=0 host-central MPB
    # ------------------------------------------------------------
    target_tree = _load_sublink_mpb_compat(BASE_PATH, SNAP0, SID0)
    central_tree = _load_sublink_mpb_compat(BASE_PATH, SNAP0, cen_sid0)

    target_map, target_track = _tree_track_map(target_tree, SNAP_TRACK)
    central_map, central_track = _tree_track_map(central_tree, SNAP_TRACK)

    common_snaps = [s for s in SNAP_TRACK if (s in target_map and s in central_map)]

    if len(common_snaps) == 0:
        raise RuntimeError("No common snapshots between target MPB and z=0 host-central MPB.")

    if len(common_snaps) < len(SNAP_TRACK):
        missing_target = [s for s in SNAP_TRACK if s not in target_map]
        missing_central = [s for s in SNAP_TRACK if s not in central_map]
        print("[warning] target MPB missing snapshots:", missing_target)
        print("[warning] z=0 host-central MPB missing snapshots:", missing_central)
        print("[warning] using only common snapshots:", common_snaps)

    common_snaps = sorted(common_snaps) if PLOT_IN_TIME_ORDER else sorted(common_snaps, reverse=True)

    track_df = pd.DataFrame(
        [
            dict(
                SnapNum=s,
                TargetSubfindID=int(target_map[s]),
                Z0HostCentralSubfindID=int(central_map[s]),
            )
            for s in common_snaps
        ]
    )

    print("Tracked target and z=0 host-central progenitors:")
    try:
        display(track_df)
    except Exception:
        print(track_df)

    # ------------------------------------------------------------
    # 3. Load particle data and build shape diagnostics
    # ------------------------------------------------------------
    records = []
    table_rows = []

    for snap_i in common_snaps:
        sid_i = int(target_map[snap_i])
        cen_sid_i = int(central_map[snap_i])

        print(
            f"[overlay] snap={snap_i}: "
            f"target SubfindID={sid_i}, z0-host-central MPB SubfindID={cen_sid_i}"
        )

        header = hd_tng.read_header_for_snap(
            BASE_PATH,
            snap_i,
            sim_name=SIM_NAME,
            api_key=API_KEY,
        )

        cat, halos, subs = _open_catalog_compat(
            BASE_PATH,
            snap_i,
            group_fields=[
                "GroupFirstSub",
                "GroupLenType",
                "Group_R_Crit200",
                "GroupPos",
            ],
            subhalo_fields=[
                "SubhaloGrNr",
                "SubhaloLenType",
                "SubhaloPos",
                "SubhaloVel",
                "SubhaloMassType",
            ],
        )

        try:
            target_gid_i = int(np.asarray(subs["SubhaloGrNr"], dtype=np.int64)[sid_i])
            central_gid_i = int(np.asarray(subs["SubhaloGrNr"], dtype=np.int64)[cen_sid_i])

            target_center_ckpc_h = np.asarray(subs["SubhaloPos"][sid_i], dtype=np.float64)
            host_central_center_ckpc_h = np.asarray(subs["SubhaloPos"][cen_sid_i], dtype=np.float64)

            pdata_dm = hd_tng.load_subhalo_dm_particles(
                cat,
                subs,
                sid_i,
                snap=snap_i,
                base_path=BASE_PATH,
                header=header,
                retry_cfg=RETRY_CFG,
            )

            pdata_star = _load_subhalo_star_particles(
                cat,
                subs,
                sid_i,
                snap=snap_i,
                base_path=BASE_PATH,
                header=header,
                retry_cfg=RETRY_CFG,
            )

            # Coordinates relative to z=0 host-central MPB progenitor at the same snapshot.
            X_dm_hostref_kpc = hd_tng.tng_relative_positions_to_physical_kpc(
                pdata_dm["coords_ckpc_h"],
                host_central_center_ckpc_h,
                header,
            )

            X_star_hostref_kpc = hd_tng.tng_relative_positions_to_physical_kpc(
                pdata_star["coords_ckpc_h"],
                host_central_center_ckpc_h,
                header,
            ) if pdata_star["coords_ckpc_h"].shape[0] else np.empty((0, 3), dtype=float)

            target_center_hostref_kpc = hd_tng.tng_relative_positions_to_physical_kpc(
                target_center_ckpc_h[None, :],
                host_central_center_ckpc_h,
                header,
            )[0]

            radial_vec = target_center_hostref_kpc
            radial_norm = np.linalg.norm(radial_vec)
            radial_hat = radial_vec / radial_norm if np.isfinite(radial_norm) and radial_norm > 0 else np.full(3, np.nan)

            masks, shell_info = hd_tng.build_shell_masks_for_particles(
                pdata_dm,
                method=SHELL_METHOD,
                n_shells=N_SHELLS,
                equal_number=bool(CFG.get("equal_number_radial_shells", True)),
                compute_binding_potential_if_missing=bool(CFG.get("compute_binding_potential_if_missing", True)),
                min_particles=MIN_PARTICLES_PER_SHELL,
            )

            # Masses from SubhaloMassType: TNG units are 1e10 Msun/h.
            if "SubhaloMassType" in subs:
                smt = np.asarray(subs["SubhaloMassType"], dtype=float)[sid_i]
                Mdm_msun = float(_mass_1e10_msun_h_to_msun(smt[1], header)) if smt.size > 1 else np.nan
                Mgas_msun = float(_mass_1e10_msun_h_to_msun(smt[0], header)) if smt.size > 0 else np.nan
                Mstar_msun = float(_mass_1e10_msun_h_to_msun(smt[4], header)) if smt.size > 4 else np.nan
                Mbh_msun = float(_mass_1e10_msun_h_to_msun(smt[5], header)) if smt.size > 5 else np.nan
                Mbaryon_msun = np.nansum([Mgas_msun, Mstar_msun, Mbh_msun])
            else:
                Mdm_msun = float(np.nansum(pdata_dm["masses"]))
                Mstar_msun = float(np.nansum(pdata_star["masses"]))
                Mgas_msun = np.nan
                Mbh_msun = np.nan
                Mbaryon_msun = Mstar_msun

            Mbaryon_over_Mdm = float(Mbaryon_msun / Mdm_msun) if np.isfinite(Mdm_msun) and Mdm_msun > 0 else np.nan

            # Global shapes: galaxy/stars are not layered; DM halo global uses all DM particles.
            gal_shape = _shape_axes_from_points(
                pdata_star["X_kpc"],
                pdata_star["masses"],
                min_particles=max(20, MIN_PARTICLES_PER_SHELL // 2),
            )

            dm_shape = _shape_axes_from_points(
                pdata_dm["X_kpc"],
                pdata_dm["masses"],
                min_particles=MIN_PARTICLES_PER_SHELL,
            )

            Rg = gal_shape["R"]
            Rd = dm_shape["R"]

            r200_ckpc_h = float(np.asarray(halos["Group_R_Crit200"], dtype=float)[central_gid_i])
            r200_kpc = float(_ckpc_h_to_physical_kpc(r200_ckpc_h, header))

            records.append(
                dict(
                    snap=int(snap_i),
                    sid=int(sid_i),
                    gid=int(target_gid_i),
                    z0_host_central_sid=int(cen_sid_i),
                    z0_host_central_gid=int(central_gid_i),
                    z=float(header.get("Redshift", np.nan)),
                    r200_kpc=r200_kpc,
                    X_global_kpc=X_dm_hostref_kpc,
                    X_star_global_kpc=X_star_hostref_kpc,
                    center_global_kpc=target_center_hostref_kpc,
                    masses=np.asarray(pdata_dm["masses"], dtype=float),
                    star_masses=np.asarray(pdata_star["masses"], dtype=float),
                    masks=masks,
                    gal_shape=gal_shape,
                    dm_shape=dm_shape,
                )
            )

            # One row per shell.
            for ish, mi in enumerate(masks):
                mi = np.asarray(mi, dtype=bool)
                shell_X = pdata_dm["X_kpc"][mi]
                shell_m = pdata_dm["masses"][mi]
                shell_shape = _shape_axes_from_points(
                    shell_X,
                    shell_m,
                    min_particles=MIN_PARTICLES_PER_SHELL,
                )
                Rs = shell_shape["R"]

                row = dict(
                    SnapNum=int(snap_i),
                    Redshift=float(header.get("Redshift", np.nan)),
                    TargetSubfindID=int(sid_i),
                    TargetGroupID=int(target_gid_i),
                    Z0HostCentralSubfindID=int(cen_sid_i),
                    Z0HostCentralGroupID=int(central_gid_i),
                    ShellMethod=str(SHELL_METHOD),
                    Shell=int(ish),
                    Ndm=int(np.asarray(pdata_dm["X_kpc"]).shape[0]),
                    Nstar=int(np.asarray(pdata_star["X_kpc"]).shape[0]),
                    Nshell=int(np.count_nonzero(mi)),
                    Mdm_msun=Mdm_msun,
                    Mgas_msun=Mgas_msun,
                    Mstar_msun=Mstar_msun,
                    Mbh_msun=Mbh_msun,
                    Mbaryon_msun=Mbaryon_msun,
                    Mbaryon_over_Mdm=Mbaryon_over_Mdm,
                    R_hostref_kpc=float(radial_norm) if np.isfinite(radial_norm) else np.nan,
                    R_over_R200c=float(radial_norm / r200_kpc) if np.isfinite(radial_norm) and np.isfinite(r200_kpc) and r200_kpc > 0 else np.nan,
                    GalaxyShapeValid=bool(gal_shape["valid"]),
                    DMShapeValid=bool(dm_shape["valid"]),
                    ShellShapeValid=bool(shell_shape["valid"]),
                )

                row.update(_axis_cols("galaxy", Rg))
                row.update(_axis_cols("dmhalo", Rd))
                row.update(_axis_cols("shell", Rs))

                row.update(_axis_evals_cols("galaxy", gal_shape["evals"]))
                row.update(_axis_evals_cols("dmhalo", dm_shape["evals"]))
                row.update(_axis_evals_cols("shell", shell_shape["evals"]))

                axis_names = ["major", "intermediate", "minor"]

                for iax, name in enumerate(axis_names):
                    # Galaxy vs global DM halo, corresponding axis type.
                    row[f"cos_galaxy_dmhalo_{name}"] = _abs_cos(Rg[:, iax], Rd[:, iax])

                    # Global axes vs host-centric radial line.
                    row[f"cos_galaxy_{name}_radial"] = _abs_cos(Rg[:, iax], radial_hat)
                    row[f"cos_dmhalo_{name}_radial"] = _abs_cos(Rd[:, iax], radial_hat)

                    # Galaxy global axis vs DM shell axis, corresponding type.
                    row[f"cos_galaxy_shell_{name}"] = _abs_cos(Rg[:, iax], Rs[:, iax])

                    # Shell axis vs host-centric radial line.
                    row[f"cos_shell_{name}_radial"] = _abs_cos(Rs[:, iax], radial_hat)

                    # Optional useful diagnostic: global DM halo vs shell axis.
                    row[f"cos_dmhalo_shell_{name}"] = _abs_cos(Rd[:, iax], Rs[:, iax])

                table_rows.append(row)

        finally:
            try:
                cat.cleanup()
            except Exception:
                pass
            if cat in hd_tng._OPEN_CATALOGS:
                hd_tng._OPEN_CATALOGS.remove(cat)

    if len(records) == 0:
        raise RuntimeError("No records were loaded.")

    table_df = pd.DataFrame(table_rows)

    # ------------------------------------------------------------
    # 4. Fit orbit plane using target positions relative to z=0 host-central MPB
    # ------------------------------------------------------------
    centers_3d = np.asarray([rec["center_global_kpc"] for rec in records], dtype=float)
    orbit_basis = _fit_orbit_plane_basis_from_centers(centers_3d)

    print("Orbit-plane basis in simulation coordinates:")
    print("e1 =", orbit_basis["e1"])
    print("e2 =", orbit_basis["e2"])
    print("n  =", orbit_basis["n"])
    print("PCA singular values =", orbit_basis["singular_values"])

    for rec in records:
        rec["xy_global"] = _project_to_orbit_plane(rec["X_global_kpc"], orbit_basis)
        rec["xy_star_global"] = _project_to_orbit_plane(rec["X_star_global_kpc"], orbit_basis) if rec["X_star_global_kpc"].shape[0] else np.empty((0, 2))
        rec["center_xy"] = _project_to_orbit_plane(
            np.asarray(rec["center_global_kpc"], dtype=float)[None, :],
            orbit_basis,
        )[0]

        if rec["gal_shape"]["valid"]:
            gmajor = rec["gal_shape"]["R"][:, 0]
            rec["galaxy_major_axis_xy"] = np.array([
                np.dot(gmajor, orbit_basis["e1"]),
                np.dot(gmajor, orbit_basis["e2"]),
            ], dtype=float)
        else:
            rec["galaxy_major_axis_xy"] = np.full(2, np.nan)

    # ------------------------------------------------------------
    # 5. Common plotting limits.
    #    The view must include FoF centre and all snapshot subhalo centres.
    #    The plotting window is centred at the mean of these points.
    # ------------------------------------------------------------
    points_for_center = [np.array([0.0, 0.0], dtype=float)]
    points_for_center.extend([np.asarray(rec["center_xy"], dtype=float) for rec in records])
    points_for_center = np.asarray(points_for_center, dtype=float)
    good_center = np.all(np.isfinite(points_for_center), axis=1)

    if np.count_nonzero(good_center):
        view_center = np.nanmean(points_for_center[good_center], axis=0)
    else:
        view_center = np.array([0.0, 0.0], dtype=float)

    if COMMON_LIM_KPC is None:
        vals = []

        # Ensure FoF centre and subhalo centres are inside.
        core_points = points_for_center[good_center]
        if core_points.size:
            vals.append(np.nanmax(np.abs(core_points - view_center[None, :])))

        # Also include high-percentile particle extents so the subhalo bodies are visible.
        for rec in records:
            xy = rec["xy_global"]
            good = np.isfinite(xy).all(axis=1)
            if np.count_nonzero(good) > 0:
                dxy = xy[good] - view_center[None, :]
                vals.append(np.nanpercentile(np.abs(dxy), DENSITY_PERCENTILE_LIMIT))

        half_width = float(np.nanmax(vals) * PAD_FACTOR) if vals else 100.0
        if not np.isfinite(half_width) or half_width <= 0:
            half_width = 100.0
    else:
        half_width = float(COMMON_LIM_KPC)

    xlim = (view_center[0] - half_width, view_center[0] + half_width)
    ylim = (view_center[1] - half_width, view_center[1] + half_width)

    # ------------------------------------------------------------
    # 6. Plot
    # ------------------------------------------------------------
    fig = None
    ax = None
    save_path = None

    if make_figure:
        plt.close("all")

        fig, ax = plt.subplots(
            figsize=(9.2, 8.8),
            facecolor=FIG_FACE,
            constrained_layout=True,
        )

        ax.set_facecolor(AX_FACE)

        if DRAW_FOF_CENTRAL_MARKER:
            ax.scatter(
                [0.0],
                [0.0],
                marker="x",
                s=95,
                linewidths=1.6,
                color="white",
                zorder=250,
            )

        for rec in records:
            xy = rec["xy_global"]
            masses = rec["masses"]
            masks = rec["masks"]
            center_xy = rec["center_xy"]

            if DRAW_PARTICLE_DENSITY:
                _draw_density_map_with_contours(ax, xy, xlim, ylim)

            _draw_shell_ellipses_only(ax, xy, masks, masses)

            if DRAW_GALAXY_MAJOR_AXIS and np.all(np.isfinite(rec["galaxy_major_axis_xy"])):
                v = np.asarray(rec["galaxy_major_axis_xy"], dtype=float)
                nv = np.linalg.norm(v)
                if np.isfinite(nv) and nv > 0:
                    v = v / nv

                    # Use stellar major-axis RMS as line length when available.
                    lam0 = rec["gal_shape"]["evals"][0] if rec["gal_shape"]["evals"].size else np.nan
                    if np.isfinite(lam0) and lam0 > 0:
                        length = GALAXY_MAJOR_AXIS_NSIGMA * np.sqrt(lam0)
                    else:
                        length = 0.035 * (xlim[1] - xlim[0])

                    p0 = center_xy - length * v
                    p1 = center_xy + length * v

                    ax.plot(
                        [p0[0], p1[0]],
                        [p0[1], p1[1]],
                        color=GALAXY_MAJOR_AXIS_COLOR,
                        lw=GALAXY_MAJOR_AXIS_LW,
                        alpha=0.98,
                        zorder=120,
                        solid_capstyle="round",
                        path_effects=[
                            patheffects.withStroke(
                                linewidth=GALAXY_MAJOR_AXIS_LW + 1.4,
                                foreground="black",
                                alpha=0.65,
                            )
                        ],
                    )

            label_xy = _label_position_outside_subhalo(xy, center_xy, xlim, ylim)

            ha = "left" if label_xy[0] >= center_xy[0] else "right"
            va = "bottom" if label_xy[1] >= center_xy[1] else "top"

            ax.text(
                label_xy[0],
                label_xy[1],
                f"z={rec['z']:.2f}",
                fontsize=11.5,
                color=TEXT_COLOR,
                ha=ha,
                va=va,
                zorder=350,
                path_effects=[
                    patheffects.withStroke(
                        linewidth=3.0,
                        foreground="white",
                        alpha=0.95,
                    )
                ],
            )

        _add_scalebar_with_endticks(ax, xlim, ylim)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")

        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))

        ax.tick_params(
            direction="in",
            top=True,
            right=True,
            color=TICK_COLOR,
            labelcolor=TICKLABEL_COLOR,
            labelsize=12,
            length=5,
            width=0.9,
        )

        for spine in ax.spines.values():
            spine.set_color(SPINE_COLOR)
            spine.set_linewidth(1.1)

        ax.set_xlabel(
            "orbit-plane coordinate u relative to z=0 host central MPB [physical kpc]",
            fontsize=14,
            color="black",
        )

        ax.set_ylabel(
            "orbit-plane coordinate v relative to z=0 host central MPB [physical kpc]",
            fontsize=14,
            color="black",
        )

        method_label = "radial shells" if SHELL_METHOD == "radial" else "binding-energy shells"

        ax.set_title(
            f"{SIM_NAME}: tracked subhalo {SID0} in fitted orbit plane\n"
            f"Origin: MPB of z=0 host central; density, {method_label}, galaxy major axis",
            fontsize=16,
            color="black",
        )

        if save:
            save_path = OUTDIR / (
                f"subhalo_{SID0}_z0hostcentralMPB_orbitplane_{SHELL_METHOD}_with_table.png"
            )
            fig.savefig(
                save_path,
                dpi=dpi,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )
            print("Saved:", save_path)

        if show:
            plt.show()

    # ------------------------------------------------------------
    # 7. Print and return table
    # ------------------------------------------------------------
    if display_table:
        print("Subhalo time/shell diagnostic table:")
        try:
            display(table_df)
        except Exception:
            print(table_df)

    return fig, ax, records, table_df, save_path


# ============================================================
# Shared cross-time computation products
# ============================================================
# Compute once, plot many times.
#
# Heavy work:
#   - MPB tracking
#   - group catalog reading
#   - target-subhalo DM/star cutouts
#   - shell masks / shape table
#   - cross-time Pi closure table
#
# After this cell, all plot functions should consume the returned products.
# ============================================================

import os
import traceback
import numpy as np
import pandas as pd
from pathlib import Path

_SUBHALO_CROSSTIME_CACHE = {}


def make_shared_tng_catalog_kwargs(
    *,
    SIM_NAME=None,
    API_KEY=None,
    CFG=None,
    cache_dir=None,
    verbose=True,
):
    """
    Build a stable TNG catalog/API/cache configuration.

    This is the important fix for repeated downloads and for API fallback:
    every heavy function receives the same download_if_missing=True, API key,
    and persistent cache_dir.
    """
    if SIM_NAME is None:
        SIM_NAME = globals().get("SIM_NAME", "TNG300-1")
    if API_KEY is None:
        API_KEY = globals().get("API_KEY", None)
    if CFG is None:
        CFG = globals().get("CFG", {})

    if cache_dir is None:
        cache_dir = globals().get("TNG_API_CACHE", CFG.get("cache_dir", None))

    if cache_dir is None:
        base_path = globals().get("BASE_PATH", ".")
        cache_dir = Path(base_path).parent / "tng_api_cache"

    cache_dir = Path(cache_dir)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        cache_dir = Path(globals().get("OUTDIR", ".")) / "tng_api_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

    return dict(
        sim_name=SIM_NAME,
        api_key=API_KEY,
        download_if_missing=True,
        cache_dir=str(cache_dir),
        delete_cache=False,
        verbose=bool(verbose),
        timeout=int(CFG.get("timeout", 180)),
        max_retries=int(CFG.get("api_max_retries", 6)),
        retry_base_sleep=float(CFG.get("api_retry_base_sleep", 5.0)),
        retry_max_sleep=float(CFG.get("api_retry_max_sleep", 90.0)),
    )


def _call_cross_time_pattern_speed_once(
    SID0,
    SNAP_TRACK,
    *,
    SNAP0=99,
    BASE_PATH=None,
    CFG=None,
    TNG_CATALOG_KWARGS=None,
    shell_method="radial",
    verbose=True,
):
    """
    Robust wrapper around hd_tng.cross_time_pattern_speed_for_subhalo.
    """
    if BASE_PATH is None:
        BASE_PATH = globals()["BASE_PATH"]
    if CFG is None:
        CFG = globals()["CFG"]

    if not hasattr(hd_tng, "cross_time_pattern_speed_for_subhalo"):
        raise AttributeError(
            "hd_tng.cross_time_pattern_speed_for_subhalo is missing. "
            "Reload the hd_tng version with cross-time Pi calculation."
        )

    attempts = []

    base_kw = dict(
        snap_track=[int(s) for s in SNAP_TRACK],
        cfg=CFG,
        shell_method=str(shell_method),
    )
    if TNG_CATALOG_KWARGS is not None:
        base_kw["tng_catalog_kwargs"] = TNG_CATALOG_KWARGS

    attempts.append((
        "keyword snap_track/cfg/shell_method/kwargs",
        (BASE_PATH, int(SNAP0), int(SID0)),
        dict(base_kw),
    ))

    kw_no_api = dict(base_kw)
    kw_no_api.pop("tng_catalog_kwargs", None)
    attempts.append((
        "keyword without tng_catalog_kwargs",
        (BASE_PATH, int(SNAP0), int(SID0)),
        kw_no_api,
    ))

    errors = []

    for label, args, kwargs in attempts:
        try:
            if verbose:
                print(f"[Pi closure] trying {label}: method={shell_method}")

            cross_df, track_df = hd_tng.cross_time_pattern_speed_for_subhalo(
                *args,
                **kwargs,
            )

            if not isinstance(cross_df, pd.DataFrame):
                cross_df = pd.DataFrame(cross_df)
            if not isinstance(track_df, pd.DataFrame):
                track_df = pd.DataFrame(track_df)

            cross_df = cross_df.copy()
            track_df = track_df.copy()

            if "shell_method" not in cross_df.columns:
                cross_df["shell_method"] = str(shell_method)

            if verbose:
                print(
                    f"[Pi closure] success: method={shell_method}, "
                    f"rows={len(cross_df)}"
                )

            return cross_df, track_df

        except Exception as exc:
            errors.append((label, repr(exc), traceback.format_exc(limit=4)))

    print(f"\n[Pi closure] failed for method={shell_method}")
    for label, err, tb in errors:
        print(f"\n--- attempt: {label} ---")
        print(err)
        print(tb)

    raise RuntimeError(
        f"cross_time_pattern_speed_for_subhalo failed for method={shell_method}."
    )


def standardize_pi_closure_table(
    df,
    *,
    SID0=None,
    SNAP_TRACK=None,
    SNAP0=99,
    BASE_PATH=None,
    SIM_NAME=None,
    API_KEY=None,
):
    """
    Standardize cross-time Pi closure table columns.

    Canonical columns:
        shell_method, shell, shell_plot
        SnapA, SnapB, Redshift
        Pi_direct_01, Pi_aff_01, Pi_Omega_01, Pi_H_01, ...
        Pi_resid_01, rel_resid_01, fOmega_abs_01, fH_abs_01
    """
    if BASE_PATH is None:
        BASE_PATH = globals().get("BASE_PATH", None)
    if SIM_NAME is None:
        SIM_NAME = globals().get("SIM_NAME", "TNG300-1")
    if API_KEY is None:
        API_KEY = globals().get("API_KEY", None)

    df = df.copy()
    n = len(df)
    comps = ["01", "02", "12"]

    def choose_col(names):
        for name in names:
            if name in df.columns:
                return name
        return None

    def redshift_for_snap(snap):
        try:
            header = hd_tng.read_header_for_snap(
                BASE_PATH,
                int(snap),
                sim_name=SIM_NAME,
                api_key=API_KEY,
            )
            return float(header.get("Redshift", np.nan))
        except Exception:
            return np.nan

    zmap = {}
    if SNAP_TRACK is not None:
        for s in set([int(SNAP0)] + [int(x) for x in SNAP_TRACK]):
            zmap[int(s)] = redshift_for_snap(int(s))

    # method
    c = choose_col(["shell_method", "ShellMethod", "method"])
    if c is None:
        df["shell_method"] = "unknown"
    elif c != "shell_method":
        df["shell_method"] = df[c].astype(str)
    else:
        df["shell_method"] = df["shell_method"].astype(str)

    # shell
    c = choose_col(["shell", "Shell", "shell_id", "ShellID", "ish"])
    if c is None:
        df["shell"] = 0
    elif c != "shell":
        df["shell"] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    else:
        df["shell"] = pd.to_numeric(df["shell"], errors="coerce").fillna(0).astype(int)

    shell_raw = pd.to_numeric(df["shell"], errors="coerce").to_numpy(dtype=float)
    if np.isfinite(shell_raw).any() and int(np.nanmin(shell_raw)) == 0:
        df["shell_plot"] = pd.to_numeric(df["shell"], errors="coerce").fillna(0).astype(int) + 1
    else:
        df["shell_plot"] = pd.to_numeric(df["shell"], errors="coerce").fillna(1).astype(int)

    # snapshots
    ca = choose_col([
        "SnapA", "snapA", "snap_a", "Snap0", "snap0",
        "snap_i", "snap_prev", "snap_start", "snap_lo",
    ])
    cb = choose_col([
        "SnapB", "snapB", "snap_b", "Snap1", "snap1",
        "snap_j", "snap_next", "snap_end", "snap_hi",
    ])
    cs = choose_col(["SnapNum", "snap", "Snapshot"])

    if ca is not None:
        df["SnapA"] = pd.to_numeric(df[ca], errors="coerce").astype("Int64")
    elif cs is not None:
        df["SnapA"] = pd.to_numeric(df[cs], errors="coerce").astype("Int64")
    else:
        df["SnapA"] = pd.Series([pd.NA] * n, dtype="Int64")

    if cb is not None:
        df["SnapB"] = pd.to_numeric(df[cb], errors="coerce").astype("Int64")
    else:
        df["SnapB"] = pd.Series([pd.NA] * n, dtype="Int64")

    # redshift
    cz = choose_col(["Redshift", "redshift", "z", "z_mid", "Redshift_mid", "z_eff"])
    if cz is not None:
        df["Redshift"] = pd.to_numeric(df[cz], errors="coerce")
    else:
        zvals = np.full(n, np.nan, dtype=float)
        for i, row in df.iterrows():
            sa = row.get("SnapA", pd.NA)
            sb = row.get("SnapB", pd.NA)

            za = zmap.get(int(sa), np.nan) if pd.notna(sa) else np.nan
            zb = zmap.get(int(sb), np.nan) if pd.notna(sb) else np.nan

            if np.isfinite(za) and np.isfinite(zb):
                zvals[i] = 0.5 * (za + zb)
            elif np.isfinite(za):
                zvals[i] = za
            elif np.isfinite(zb):
                zvals[i] = zb

        df["Redshift"] = zvals

    # subhalo id
    if "SubhaloID" not in df.columns:
        c = choose_col(["TargetSubfindID", "SubfindID", "sid", "SID"])
        if c is not None:
            df["SubhaloID"] = pd.to_numeric(df[c], errors="coerce")
        else:
            df["SubhaloID"] = SID0

    # Pi aliases
    aliases = {}
    for comp in comps:
        aliases[f"Pi_direct_{comp}"] = [
            f"Pi_direct_{comp}",
            f"Pi_dI_{comp}",
            f"Pi_measured_{comp}",
            f"Pi_mea_{comp}",
            f"Pi_fd_{comp}",
            f"Pi_cross_{comp}",
        ]
        aliases[f"Pi_aff_{comp}"] = [
            f"Pi_aff_{comp}",
            f"Pi_est_{comp}",
            f"Pi_affine_{comp}",
            f"Pi_fig_{comp}",
            f"Pi_model_{comp}",
            f"Pi_OmegaH_{comp}",
            f"Pi_Omega_plus_H_{comp}",
            f"Pi_Omega_H_{comp}",
        ]
        aliases[f"Pi_Omega_{comp}"] = [
            f"Pi_Omega_{comp}",
            f"Pi_omega_{comp}",
            f"Omega_{comp}",
            f"Omega_hat_{comp}",
        ]
        aliases[f"Pi_H_{comp}"] = [
            f"Pi_H_{comp}",
            f"Pi_h_{comp}",
            f"Pi_mathcalH_{comp}",
            f"H_{comp}",
            f"H_hat_{comp}",
        ]

    for canon, names in aliases.items():
        if canon in df.columns:
            df[canon] = pd.to_numeric(df[canon], errors="coerce")
            continue

        c = choose_col(names)
        if c is not None:
            df[canon] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[canon] = np.nan

    # derived closure quantities
    for comp in comps:
        direct = pd.to_numeric(df[f"Pi_direct_{comp}"], errors="coerce").to_numpy(dtype=float)
        aff = pd.to_numeric(df[f"Pi_aff_{comp}"], errors="coerce").to_numpy(dtype=float)
        Om = pd.to_numeric(df[f"Pi_Omega_{comp}"], errors="coerce").to_numpy(dtype=float)
        H = pd.to_numeric(df[f"Pi_H_{comp}"], errors="coerce").to_numpy(dtype=float)

        missing_aff = ~np.isfinite(aff)
        can_make_aff = np.isfinite(Om) & np.isfinite(H)
        aff[missing_aff & can_make_aff] = Om[missing_aff & can_make_aff] + H[missing_aff & can_make_aff]
        df[f"Pi_aff_{comp}"] = aff

        resid = direct - aff
        df[f"Pi_resid_{comp}"] = resid

        rel = np.full(n, np.nan, dtype=float)
        good = np.isfinite(direct) & np.isfinite(aff) & (np.abs(direct) > 0)
        rel[good] = resid[good] / np.abs(direct[good])
        df[f"rel_resid_{comp}"] = rel

        denom = np.abs(Om) + np.abs(H)
        fO = np.full(n, np.nan, dtype=float)
        fH = np.full(n, np.nan, dtype=float)
        good = np.isfinite(denom) & (denom > 0)
        fO[good] = np.abs(Om[good]) / denom[good]
        fH[good] = np.abs(H[good]) / denom[good]
        df[f"fOmega_abs_{comp}"] = fO
        df[f"fH_abs_{comp}"] = fH

    return df


def compute_cross_time_products(
    SID0,
    SNAP_TRACK,
    *,
    SNAP0=99,
    BASE_PATH=None,
    SIM_NAME=None,
    API_KEY=None,
    CFG=None,
    OUTDIR=None,
    methods=("radial", "binding_energy"),
    force=False,
    save_tables=True,
    verbose=True,
):
    """
    Main compute-once function.

    Returns
    -------
    products : dict
        products["methods"][method]["records"]
        products["methods"][method]["table_df"]
        products["methods"][method]["closure_df"]
        products["methods"][method]["track_df"]
    """
    if BASE_PATH is None:
        BASE_PATH = globals()["BASE_PATH"]
    if SIM_NAME is None:
        SIM_NAME = globals().get("SIM_NAME", "TNG300-1")
    if API_KEY is None:
        API_KEY = globals().get("API_KEY", None)
    if CFG is None:
        CFG = globals()["CFG"]
    if OUTDIR is None:
        OUTDIR = Path(globals().get("OUTDIR", "hd_tng_outputs"))
    else:
        OUTDIR = Path(OUTDIR)

    OUTDIR.mkdir(parents=True, exist_ok=True)

    SID0 = int(SID0)
    SNAP0 = int(SNAP0)
    SNAP_TRACK = tuple(int(s) for s in SNAP_TRACK)
    methods = tuple(str(m) for m in methods)

    tng_kwargs = make_shared_tng_catalog_kwargs(
        SIM_NAME=SIM_NAME,
        API_KEY=API_KEY,
        CFG=CFG,
        verbose=bool(verbose),
    )

    cache_key = (
        SID0,
        SNAP0,
        SNAP_TRACK,
        methods,
        int(CFG.get("n_radial_shells", 6)),
        int(CFG.get("n_binding_shells", 6)),
        int(CFG.get("min_particles_per_shell", 100)),
        str(tng_kwargs.get("cache_dir")),
    )

    if (not force) and cache_key in _SUBHALO_CROSSTIME_CACHE:
        if verbose:
            print("[products] using in-memory cached products")
        return _SUBHALO_CROSSTIME_CACHE[cache_key]

    products = dict(
        SID0=SID0,
        SNAP0=SNAP0,
        SNAP_TRACK=list(SNAP_TRACK),
        methods={},
        TNG_CATALOG_KWARGS=tng_kwargs,
        BASE_PATH=BASE_PATH,
        SIM_NAME=SIM_NAME,
    )

    for method in methods:
        print(f"\n[products] computing method={method}")

        # --------------------------------------------------------
        # A. Snapshot-wise orbit-plane / shape / mass table
        # --------------------------------------------------------
        print("[products] building records/table via _compute_orbitplane_products(..., make_figure=False)")
        fig, ax, records, table_df, save_path = _compute_orbitplane_products(
            SID0,
            list(SNAP_TRACK),
            SNAP0=SNAP0,
            BASE_PATH=BASE_PATH,
            SIM_NAME=SIM_NAME,
            API_KEY=API_KEY,
            CFG=CFG,
            TNG_CATALOG_KWARGS=tng_kwargs,
            SHELL_METHOD=method,
            make_figure=False,
            display_table=False,
            save=False,
            show=False,
        )

        # --------------------------------------------------------
        # B. Cross-time Pi closure table
        # --------------------------------------------------------
        print("[products] building cross-time Pi closure table")
        try:
            raw_closure, track_df = _call_cross_time_pattern_speed_once(
                SID0,
                list(SNAP_TRACK),
                SNAP0=SNAP0,
                BASE_PATH=BASE_PATH,
                CFG=CFG,
                TNG_CATALOG_KWARGS=tng_kwargs,
                shell_method=method,
                verbose=verbose,
            )
            closure_df = standardize_pi_closure_table(
                raw_closure,
                SID0=SID0,
                SNAP_TRACK=list(SNAP_TRACK),
                SNAP0=SNAP0,
                BASE_PATH=BASE_PATH,
                SIM_NAME=SIM_NAME,
                API_KEY=API_KEY,
            )
        except Exception as exc:
            print(f"[products] WARNING: Pi closure failed for method={method}")
            print(repr(exc))
            raw_closure = pd.DataFrame()
            closure_df = pd.DataFrame()
            track_df = pd.DataFrame()

        products["methods"][method] = dict(
            records=records,
            table_df=table_df,
            raw_closure_df=raw_closure,
            closure_df=closure_df,
            track_df=track_df,
        )

        if save_tables:
            table_path = OUTDIR / f"subhalo_{SID0}_{method}_orbit_table.csv"
            table_df.to_csv(table_path, index=False)
            print("[products] saved:", table_path)

            if len(closure_df):
                closure_path = OUTDIR / f"subhalo_{SID0}_{method}_pi_closure_cross_time.csv"
                closure_df.to_csv(closure_path, index=False)
                print("[products] saved:", closure_path)

    _SUBHALO_CROSSTIME_CACHE[cache_key] = products
    return products



def save_cross_time_products(products, path=None):
    """Save computed products to a pickle file."""
    if path is None:
        path = globals().get('PRODUCTS_PATH', Path('hd_tng_outputs') / 'cross_time_products.pkl')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(products, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('[products] saved:', path)
    return path


def load_cross_time_products(path=None):
    """Load previously computed products from a pickle file."""
    if path is None:
        path = globals().get('PRODUCTS_PATH', Path('hd_tng_outputs') / 'cross_time_products.pkl')
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Computed products file does not exist: {path}')
    with open(path, 'rb') as f:
        products = pickle.load(f)
    print('[products] loaded:', path)
    return products


def _resolve_products(products=None, path=None):
    """Resolve plotting input: explicit products, global PRODUCTS, or pickle file."""
    if products is not None:
        return products
    if 'PRODUCTS' in globals() and isinstance(globals()['PRODUCTS'], dict):
        return globals()['PRODUCTS']
    return load_cross_time_products(path)

# %% [markdown] cell 9
# ## Compute / save products

# %% code cell 10

# ============================================================
# Compute / download phase
# ============================================================
# This cell is the only place that should read TNG data, download API cutouts,
# follow SubLink MPBs, compute shell products, and compute cross-time Pi closure.
# The results are written to PRODUCTS_PATH.  Later plotting cells only reload
# this file.
# ============================================================

if RUN_CROSS_TIME:
    if PRODUCTS_PATH.exists() and not FORCE_RECOMPUTE:
        print('[products] existing file found; set FORCE_RECOMPUTE=True to recompute:')
        print(' ', PRODUCTS_PATH)
        PRODUCTS = load_cross_time_products(PRODUCTS_PATH)
    else:
        PRODUCTS = compute_cross_time_products(
            SID0,
            SNAP_TRACK,
            SNAP0=SNAP0,
            BASE_PATH=BASE_PATH,
            SIM_NAME=SIM_NAME,
            API_KEY=API_KEY,
            CFG=CFG,
            OUTDIR=OUTDIR,
            methods=SHELL_METHODS,
            force=FORCE_RECOMPUTE,
            save_tables=True,
            verbose=True,
        )
        save_cross_time_products(PRODUCTS, PRODUCTS_PATH)
else:
    print('RUN_CROSS_TIME=False; compute cell skipped.')

# %% [markdown] cell 11
# ## Load products for plotting

# %% code cell 12

# ============================================================
# Read computed products for plotting
# ============================================================
# Everything below should use PRODUCTS loaded here, not recompute/download.
# ============================================================

PRODUCTS = load_cross_time_products(PRODUCTS_PATH)
print('Available shell methods:', list(PRODUCTS['methods'].keys()))
for method, block in PRODUCTS['methods'].items():
    print(method, 'table rows =', len(block.get('table_df', [])), 'closure rows =', len(block.get('closure_df', [])))

# %% [markdown] cell 13
# ## Density helper utilities and optional FoF/gas particle enrichment

# %% code cell 14

# ============================================================
# Shared density/orbit-plane plotting helpers
# ============================================================
# These helpers are defined once and reused by the density enrichment and
# density plotting functions below. They do not download data.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib import patheffects
from matplotlib.ticker import MaxNLocator
from pathlib import Path


def _az_resolve_outdir(OUTDIR=None, path=None):
    """Resolve and create the output directory used by align_z/merger_align plots."""
    if OUTDIR is None:
        OUTDIR = path
    if OUTDIR is None:
        OUTDIR = Path(globals().get("OUTDIR", "hd_tng_outputs"))
    else:
        OUTDIR = Path(OUTDIR)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    return OUTDIR


def _az_as_xy(x):
    """Return a clean (N,2) float array, or an empty (0,2) array."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 0:
        return arr
    return np.empty((0, 2), dtype=float)


def _az_as_w(w, n):
    """Return a clean positive 1-D weight array matching length n, or None."""
    if w is None:
        return None
    arr = np.asarray(w, dtype=float)
    if arr.ndim == 1 and arr.size == int(n):
        arr = np.where(np.isfinite(arr) & (arr > 0), arr, 0.0)
        if np.sum(arr) > 0:
            return arr
    return None


def _az_get_first_existing(rec, keys):
    for key in keys:
        if key in rec:
            return rec[key]
    return None


def _az_combine_xy_mass(parts):
    """Combine several (xy, mass) pairs, filling missing masses with unit weights."""
    xy_list = []
    m_list = []
    for xy, m in parts:
        xy = _az_as_xy(xy)
        m = np.asarray(m, dtype=float)
        if len(xy) == 0:
            continue
        xy_list.append(xy)
        if m.ndim == 1 and m.size == len(xy):
            m_list.append(m)
        else:
            m_list.append(np.ones(len(xy), dtype=float))
    if len(xy_list) == 0:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)
    return np.vstack(xy_list), np.concatenate(m_list)


def _az_get_component_xy_w(rec, kind):
    """Get tracked-subhalo projected component particles: total, dm, star, or gas."""
    kind = str(kind).lower()

    if kind == "dm":
        xy = _az_as_xy(_az_get_first_existing(rec, [
            "xy_dm_global", "dm_xy_global", "xy_global",
        ]))
        w = _az_as_w(_az_get_first_existing(rec, ["dm_masses", "masses"]), len(xy))
        return xy, w

    if kind == "star":
        xy = _az_as_xy(_az_get_first_existing(rec, [
            "xy_star_global", "star_xy_global", "xy_stars_global", "stellar_xy_global",
        ]))
        w = _az_as_w(_az_get_first_existing(rec, ["star_masses", "stellar_masses"]), len(xy))
        return xy, w

    if kind == "gas":
        xy = _az_as_xy(_az_get_first_existing(rec, ["xy_gas_global", "gas_xy_global"]))
        w = _az_as_w(_az_get_first_existing(rec, ["gas_masses"]), len(xy))
        return xy, w

    if kind == "total":
        xy = _az_as_xy(_az_get_first_existing(rec, [
            "xy_all_global", "all_xy_global", "xy_total_global", "total_xy_global",
        ]))
        w = _az_as_w(_az_get_first_existing(rec, ["all_masses", "total_masses"]), len(xy))
        if len(xy):
            return xy, w
        xy_dm, w_dm = _az_get_component_xy_w(rec, "dm")
        xy_st, w_st = _az_get_component_xy_w(rec, "star")
        xy_gs, w_gs = _az_get_component_xy_w(rec, "gas")
        return _az_combine_xy_mass([(xy_dm, w_dm), (xy_st, w_st), (xy_gs, w_gs)])

    raise ValueError(f"Unknown component kind: {kind!r}")


def _az_get_env_xy_w(rec, kind):
    """Get FoF/environment projected component particles: total, dm, star, or gas."""
    kind = str(kind).lower()

    if kind == "dm":
        xy = _az_as_xy(_az_get_first_existing(rec, [
            "fof_xy_dm_global", "fof_dm_xy_global", "env_xy_dm_global", "env_dm_xy_global",
            "fof_xy_global", "env_xy_global",
        ]))
        w = _az_as_w(_az_get_first_existing(rec, [
            "fof_dm_masses", "env_dm_masses", "fof_masses", "env_masses",
        ]), len(xy))
        return xy, w

    if kind == "star":
        xy = _az_as_xy(_az_get_first_existing(rec, [
            "fof_xy_star_global", "fof_star_xy_global", "env_xy_star_global", "env_star_xy_global",
        ]))
        w = _az_as_w(_az_get_first_existing(rec, ["fof_star_masses", "env_star_masses"]), len(xy))
        return xy, w

    if kind == "gas":
        xy = _az_as_xy(_az_get_first_existing(rec, [
            "fof_xy_gas_global", "fof_gas_xy_global", "env_xy_gas_global", "env_gas_xy_global",
        ]))
        w = _az_as_w(_az_get_first_existing(rec, ["fof_gas_masses", "env_gas_masses"]), len(xy))
        return xy, w

    if kind == "total":
        xy = _az_as_xy(_az_get_first_existing(rec, [
            "fof_xy_all_global", "fof_all_xy_global", "env_xy_all_global", "env_all_xy_global",
            "fof_xy_total_global", "env_xy_total_global",
        ]))
        w = _az_as_w(_az_get_first_existing(rec, [
            "fof_all_masses", "env_all_masses", "fof_total_masses", "env_total_masses",
        ]), len(xy))
        if len(xy):
            return xy, w
        xy_dm, w_dm = _az_get_env_xy_w(rec, "dm")
        xy_st, w_st = _az_get_env_xy_w(rec, "star")
        xy_gs, w_gs = _az_get_env_xy_w(rec, "gas")
        return _az_combine_xy_mass([(xy_dm, w_dm), (xy_st, w_st), (xy_gs, w_gs)])

    raise ValueError(f"Unknown environment kind: {kind!r}")


def _az_record_redshift(rec):
    for key in ["z", "Redshift", "redshift", "z_mid", "Redshift_mid"]:
        if key in rec:
            try:
                return float(rec[key])
            except Exception:
                pass
    return np.nan


def _az_sort_records(records, *, time_order="early_to_late"):
    records = list(records)
    if time_order == "snap_order":
        return records
    z = np.array([_az_record_redshift(r) for r in records], dtype=float)
    if not np.isfinite(z).any():
        return records
    if time_order == "early_to_late":
        order = np.argsort(z)[::-1]
    elif time_order == "late_to_early":
        order = np.argsort(z)
    else:
        raise ValueError("time_order must be 'early_to_late', 'late_to_early', or 'snap_order'.")
    return [records[i] for i in order]


def _az_mix_with_white(color, amount):
    rgb = np.array(to_rgb(color), dtype=float)
    return tuple((1.0 - float(amount)) * rgb + float(amount) * np.ones(3))


def _az_shell_palette(nshell, base_color):
    """Same-family shell colours: inner/most-bound darker, outer/least-bound lighter."""
    nshell = int(max(nshell, 1))
    if nshell == 1:
        return [base_color]
    return [_az_mix_with_white(base_color, 0.08 + 0.62 * i / (nshell - 1)) for i in range(nshell)]


def _az_ellipse_from_xy(xy, w=None, nsigma=2.0):
    xy = _az_as_xy(xy)
    if len(xy) < 5:
        return None
    ww = _az_as_w(w, len(xy))
    if ww is None:
        ww = np.ones(len(xy), dtype=float)
    good = np.isfinite(xy).all(axis=1) & np.isfinite(ww) & (ww > 0)
    if np.count_nonzero(good) < 5:
        return None
    xy = xy[good]
    ww = ww[good]
    cen = np.sum(xy * ww[:, None], axis=0) / np.sum(ww)
    y = xy - cen[None, :]
    cov = np.einsum("n,ni,nj->ij", ww, y, y) / np.sum(ww)
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    if np.any(vals <= 0) or not np.all(np.isfinite(vals)):
        return None
    width = 2.0 * float(nsigma) * np.sqrt(vals[0])
    height = 2.0 * float(nsigma) * np.sqrt(vals[1])
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return cen, width, height, angle


def _az_draw_major_axis(ax, rec, xlim, *, color=None, colour=None, lw=1.45, nsigma=2.2):
    """Draw the galaxy major axis. Accepts both color= and colour= for compatibility."""
    if color is None:
        color = colour
    if color is None:
        color = "#ff2020"

    v = np.asarray(rec.get("galaxy_major_axis_xy", np.full(2, np.nan)), dtype=float)
    cen = np.asarray(rec.get("center_xy", np.full(2, np.nan)), dtype=float)
    nv = np.linalg.norm(v)
    if not (np.isfinite(nv) and nv > 0 and np.all(np.isfinite(cen))):
        return
    v = v / nv
    gal_shape = rec.get("gal_shape", {})
    evals = np.asarray(gal_shape.get("evals", []), dtype=float) if isinstance(gal_shape, dict) else np.array([])
    if evals.size and np.isfinite(evals[0]) and evals[0] > 0:
        length = float(nsigma) * np.sqrt(evals[0])
    else:
        length = 0.030 * abs(float(xlim[1]) - float(xlim[0]))
    p0 = cen - length * v
    p1 = cen + length * v
    ax.plot(
        [p0[0], p1[0]], [p0[1], p1[1]],
        color=color, lw=lw, alpha=0.95, solid_capstyle="round", zorder=60,
        path_effects=[patheffects.withStroke(linewidth=lw + 1.0, foreground="white", alpha=0.5)],
    )


def _az_hist_positive_log_values(xy, w, xlim, ylim, bins):
    xy = _az_as_xy(xy)
    if len(xy) < 10:
        return None
    good = (
        np.isfinite(xy).all(axis=1)
        & (xy[:, 0] >= xlim[0]) & (xy[:, 0] <= xlim[1])
        & (xy[:, 1] >= ylim[0]) & (xy[:, 1] <= ylim[1])
    )
    if np.count_nonzero(good) < 10:
        return None
    xy2 = xy[good]
    ww = _az_as_w(w, len(xy))
    if ww is not None:
        ww = ww[good]
    H, _, _ = np.histogram2d(xy2[:, 0], xy2[:, 1], bins=int(bins), range=[xlim, ylim], weights=ww)
    pos = H > 0
    if np.count_nonzero(pos) < 5:
        return None
    vals = np.log10(H[pos])
    vals = vals[np.isfinite(vals)]
    return vals if vals.size else None


def _az_hist_log_range_same_limits(xy_w_list, xlim, ylim, bins, *, qlo=5.0, qhi=99.5):
    """Common log-density range for multiple sets sharing one xlim/ylim."""
    vals = []
    for xy, w in xy_w_list:
        v = _az_hist_positive_log_values(xy, w, xlim, ylim, bins)
        if v is not None and len(v):
            vals.append(v)
    if len(vals) == 0:
        return None, None
    vals = np.concatenate(vals)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None
    vmin = float(np.nanpercentile(vals, float(qlo)))
    vmax = float(np.nanpercentile(vals, float(qhi)))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None, None
    if vmax <= vmin:
        vmax = vmin + 1.0e-6
    return vmin, vmax


def _az_hist_log_range_multi_limits(xy_w_list, xlim_list, ylim_list, bins, *, qlo=5.0, qhi=99.5):
    """Common log-density range for multiple sets with independent xlim/ylim."""
    vals = []
    for (xy, w), xlim, ylim in zip(xy_w_list, xlim_list, ylim_list):
        v = _az_hist_positive_log_values(xy, w, xlim, ylim, bins)
        if v is not None and len(v):
            vals.append(v)
    if len(vals) == 0:
        return None, None
    vals = np.concatenate(vals)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None
    vmin = float(np.nanpercentile(vals, float(qlo)))
    vmax = float(np.nanpercentile(vals, float(qhi)))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None, None
    if vmax <= vmin:
        vmax = vmin + 1.0e-6
    return vmin, vmax


def _az_draw_hist(ax, xy, w, xlim, ylim, *, bins=400, cmap="magma", alpha=0.75, vmin=None, vmax=None, zorder=2):
    xy = _az_as_xy(xy)
    if len(xy) < 10:
        return None
    good = (
        np.isfinite(xy).all(axis=1)
        & (xy[:, 0] >= xlim[0]) & (xy[:, 0] <= xlim[1])
        & (xy[:, 1] >= ylim[0]) & (xy[:, 1] <= ylim[1])
    )
    if np.count_nonzero(good) < 10:
        return None
    xy2 = xy[good]
    ww = _az_as_w(w, len(xy))
    if ww is not None:
        ww = ww[good]
    H, _, _ = np.histogram2d(xy2[:, 0], xy2[:, 1], bins=int(bins), range=[xlim, ylim], weights=ww)
    H = H.T
    pos = H > 0
    if np.count_nonzero(pos) < 5:
        return None
    img = np.full_like(H, np.nan, dtype=float)
    img[pos] = np.log10(H[pos])
    return ax.imshow(
        img, origin="lower", extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        cmap=cmap, alpha=alpha, interpolation="nearest", vmin=vmin, vmax=vmax,
        aspect="equal", zorder=zorder,
    )


def _az_fit_orbit_basis_from_records(records):
    """Fit an orbit-plane basis from tracked 3-D centres saved in records."""
    centers = []
    for rec in records:
        c = np.asarray(rec.get("center_global_kpc", np.full(3, np.nan)), dtype=float)
        if c.shape == (3,) and np.all(np.isfinite(c)):
            centers.append(c)
    if len(centers) >= 2:
        X = np.asarray(centers, dtype=float)
        X0 = X - np.mean(X, axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(X0, full_matrices=False)
        e1 = vt[0]
        if vt.shape[0] >= 2:
            e2 = vt[1]
        else:
            tmp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(tmp, e1)) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0])
            e2 = tmp - np.dot(tmp, e1) * e1
            e2 = e2 / np.linalg.norm(e2)
        n = np.cross(e1, e2)
        n = n / np.linalg.norm(n)
        e2 = np.cross(n, e1)
        e2 = e2 / np.linalg.norm(e2)
    else:
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
        n = np.array([0.0, 0.0, 1.0])
    return {"e1": e1, "e2": e2, "n": n}


def _az_project_to_basis(X, basis):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3 or X.shape[0] == 0:
        return np.empty((0, 2), dtype=float)
    return np.column_stack([X @ basis["e1"], X @ basis["e2"]])


def _az_half_width_for_record(rec, *, zoom_factor=2.8, min_half_width_kpc=25.0, density_percentile=99.0):
    """Estimate a zoom half-width around the tracked subhalo."""
    center = np.asarray(rec.get("center_xy", np.full(2, np.nan)), dtype=float)
    if not np.all(np.isfinite(center)):
        center = np.array([0.0, 0.0], dtype=float)
    xy_total, _ = _az_get_component_xy_w(rec, "total")
    if len(xy_total) >= 10:
        rel = xy_total - center[None, :]
        size = np.nanpercentile(np.abs(rel), float(density_percentile))
        half = max(float(min_half_width_kpc), float(zoom_factor) * float(size))
    else:
        half = float(min_half_width_kpc)
    if not np.isfinite(half) or half <= 0:
        half = float(min_half_width_kpc)
    return half


def _az_limits_for_record(rec, common_half_width):
    """Use one half-width but centre each panel on its own subhalo position."""
    center = np.asarray(rec.get("center_xy", np.full(2, np.nan)), dtype=float)
    if not np.all(np.isfinite(center)):
        center = np.array([0.0, 0.0], dtype=float)
    half = float(common_half_width)
    return (center[0] - half, center[0] + half), (center[1] - half, center[1] + half)


def _az_nice_scalebar_length(width_kpc):
    width_kpc = float(width_kpc)
    if not np.isfinite(width_kpc) or width_kpc <= 0:
        return 10.0
    raw = 0.22 * width_kpc
    pow10 = 10 ** np.floor(np.log10(max(raw, 1.0e-8)))
    candidates = np.array([1.0, 2.0, 5.0, 10.0]) * pow10
    return float(candidates[np.argmin(np.abs(candidates - raw))])


def _az_add_scalebar(ax, xlim, ylim, *, color="black", fixed_length=None, fontsize=9.0, linewidth=0.85):
    x0, x1 = map(float, xlim)
    y0, y1 = map(float, ylim)
    xr = abs(x1 - x0)
    yr = abs(y1 - y0)
    length = _az_nice_scalebar_length(xr) if fixed_length is None else float(fixed_length)
    if not np.isfinite(length) or length <= 0:
        return
    xb = x0 + 0.075 * xr
    yb = y0 + 0.085 * yr
    tick_h = 0.022 * yr
    label = f"{int(length):d} kpc" if abs(length - int(length)) < 1.0e-8 else f"{length:g} kpc"
    ax.plot([xb, xb + length], [yb, yb], color=color, lw=linewidth, solid_capstyle="butt", zorder=1000)
    ax.plot([xb, xb], [yb - tick_h / 2.0, yb + tick_h / 2.0], color=color, lw=linewidth, zorder=1001)
    ax.plot([xb + length, xb + length], [yb - tick_h / 2.0, yb + tick_h / 2.0], color=color, lw=linewidth, zorder=1001)
    ax.text(xb + 0.5 * length, yb + 0.04 * yr, label, ha="center", va="bottom", color=color, fontsize=fontsize, zorder=1002)


def _az_apply_image_axis_style(
    ax, xlim, ylim, *, scalebar=True, scalebar_color="black",
    fixed_scalebar_length=None, scalebar_fontsize=9.0, max_ticks=4,
    title=None, ylabel=None, tick_labels=True,
):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MaxNLocator(3 if not tick_labels else max_ticks))
    ax.yaxis.set_major_locator(MaxNLocator(3 if not tick_labels else max_ticks))
    ax.tick_params(direction="in", top=True, right=True, colors="black", labelsize=8.5, length=4.0, width=0.75)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)
    if title is not None:
        ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if scalebar:
        _az_add_scalebar(
            ax, xlim, ylim, color=scalebar_color,
            fixed_length=fixed_scalebar_length, fontsize=scalebar_fontsize,
        )


# Backward-compatible alias used by earlier intermediate cells.
def _az_style_density_axis(ax, xlim, ylim, *, title=None, ylabel=None, scalebar=True, fixed_scalebar_length=None, tick_labels=True):
    return _az_apply_image_axis_style(
        ax, xlim, ylim,
        scalebar=scalebar,
        scalebar_color="black",
        fixed_scalebar_length=fixed_scalebar_length,
        scalebar_fontsize=8.5,
        title=title,
        ylabel=ylabel,
        tick_labels=tick_labels,
    )

# %% code cell 15
# ============================================================
# Load target gas and FoF/environment particles for density plots
# ============================================================
# Run this after PRODUCTS has been loaded.  It modifies PRODUCTS in place by
# adding gas, FoF DM, FoF gas, FoF stellar, and FoF total projected density
# arrays into PRODUCTS["methods"][method]["records"].
# ============================================================

import pickle
import numpy as np
from TNGCatLoader import TNGCatalog


RUN_DENSITY_PARTICLE_ENRICHMENT = globals().get("RUN_DENSITY_PARTICLE_ENRICHMENT", True)
SAVE_PRODUCTS_AFTER_DENSITY_ENRICHMENT = globals().get("SAVE_PRODUCTS_AFTER_DENSITY_ENRICHMENT", True)
DENSITY_ENRICH_METHODS = tuple(globals().get("DENSITY_ENRICH_METHODS", ("radial",)))
MAX_FOF_PARTICLES_PER_TYPE = globals().get("MAX_FOF_PARTICLES_PER_TYPE", 350_000)
MAX_SUBHALO_GAS_PARTICLES = globals().get("MAX_SUBHALO_GAS_PARTICLES", None)
FOF_GID_MODE = globals().get("FOF_GID_MODE", "host_central")


def enrich_products_with_fof_and_gas_particles(
    products,
    *,
    methods=("radial",),
    BASE_PATH=None,
    SIM_NAME=None,
    API_KEY=None,
    CFG=None,
    TNG_CATALOG_KWARGS=None,
    fof_gid_mode="host_central",
    max_fof_particles_per_type=350_000,
    max_subhalo_gas_particles=None,
    random_seed=20260610,
    force=False,
    verbose=True,
):
    """
    Enrich PRODUCTS with gas and FoF/environment particles for density plots.

    Added per-record fields include:
        xy_gas_global, gas_masses,
        xy_dm_global, dm_masses,
        xy_all_global, all_masses,
        fof_xy_dm_global, fof_dm_masses,
        fof_xy_gas_global, fof_gas_masses,
        fof_xy_star_global, fof_star_masses,
        fof_xy_all_global, fof_all_masses.

    The projection basis is refitted from the same tracked 3-D centres used by
    the orbit-plane products, so the added density fields are consistent with
    the existing xy_global and center_xy fields.
    """
    if not isinstance(products, dict) or "methods" not in products:
        raise TypeError("products must be the existing PRODUCTS dictionary.")

    if BASE_PATH is None:
        BASE_PATH = products.get("BASE_PATH", globals().get("BASE_PATH"))
    if SIM_NAME is None:
        SIM_NAME = products.get("SIM_NAME", globals().get("SIM_NAME", "TNG300-1"))
    if API_KEY is None:
        API_KEY = globals().get("API_KEY", None)
    if CFG is None:
        CFG = globals().get("CFG", {})
    if TNG_CATALOG_KWARGS is None:
        TNG_CATALOG_KWARGS = products.get("TNG_CATALOG_KWARGS", globals().get("TNG_CATALOG_KWARGS", {}))

    if BASE_PATH is None:
        raise ValueError("BASE_PATH is missing.")

    methods = tuple(str(m) for m in methods)
    rng = np.random.default_rng(int(random_seed))

    retry_cfg = {
        "max_retries": int(CFG.get("api_max_retries", 6)),
        "base_sleep": float(CFG.get("api_retry_base_sleep", 5.0)),
        "max_sleep": float(CFG.get("api_retry_max_sleep", 90.0)),
        "verbose": bool(verbose),
    }

    def _mass_1e10_msun_h_to_msun(m, header):
        h = float(header.get("HubbleParam", 0.6774))
        return np.asarray(m, dtype=float) * 1.0e10 / h

    def _dm_mass_array(header, n):
        try:
            return hd_tng.dm_mass_msun_from_header(header, int(n))
        except Exception:
            h = float(header.get("HubbleParam", 0.6774))
            mt = np.asarray(header.get("MassTable", np.zeros(6)), dtype=float)
            if mt.size > 1 and mt[1] > 0:
                return np.full(int(n), mt[1] * 1.0e10 / h, dtype=float)
            return np.ones(int(n), dtype=float)

    def _open_catalog_for_snap(snap):
        kwargs = dict(TNG_CATALOG_KWARGS or {})
        cat = TNGCatalog(str(BASE_PATH), int(snap), **kwargs)
        halos, subs = hd_tng.retry_call(
            cat.loadFoF,
            group_fields=["GroupFirstSub", "GroupLenType", "GroupPos", "Group_R_Crit200"],
            subhalo_fields=["SubhaloGrNr", "SubhaloLenType", "SubhaloPos", "SubhaloVel", "SubhaloMassType"],
            **retry_cfg,
        )
        return cat, halos, subs

    def _load_particle_block(cat, *, kind, obj_id, ptype, fields):
        try:
            if kind == "subhalo":
                out = hd_tng.retry_call(
                    cat.loadSubhalos,
                    int(obj_id),
                    ptypes=[int(ptype)],
                    fields=list(fields),
                    **retry_cfg,
                )
            elif kind == "halo":
                out = hd_tng.retry_call(
                    cat.loadHalos,
                    ptypes=[int(ptype)],
                    fields=list(fields),
                    gid=int(obj_id),
                    **retry_cfg,
                )
            else:
                raise ValueError("kind must be 'subhalo' or 'halo'.")
            return out.get(f"PartType{int(ptype)}", {})
        except Exception as exc:
            if verbose:
                print(f"[density enrich] missing {kind}={obj_id}, ptype={ptype}: {exc}")
            return {}

    def _coords_masses_ids_from_block(block, *, ptype, header):
        if "Coordinates" not in block:
            return np.empty((0, 3), dtype=float), np.empty(0, dtype=float), np.empty(0, dtype=np.int64)

        coords = np.asarray(block["Coordinates"], dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
            return np.empty((0, 3), dtype=float), np.empty(0, dtype=float), np.empty(0, dtype=np.int64)

        if int(ptype) == 1:
            masses = _dm_mass_array(header, coords.shape[0])
        elif "Masses" in block:
            masses = _mass_1e10_msun_h_to_msun(np.asarray(block["Masses"], dtype=float), header)
        else:
            masses = np.ones(coords.shape[0], dtype=float)

        ids = np.asarray(block.get("ParticleIDs", np.arange(coords.shape[0])), dtype=np.int64)
        return coords, masses, ids

    def _downsample(coords, masses, ids, max_n):
        coords = np.asarray(coords, dtype=float)
        masses = np.asarray(masses, dtype=float)
        ids = np.asarray(ids, dtype=np.int64)
        n = coords.shape[0]
        if max_n is None or n <= int(max_n):
            return coords, masses, ids

        max_n = int(max_n)
        idx = rng.choice(n, size=max_n, replace=False)
        factor = float(n) / float(max_n)
        return coords[idx], masses[idx] * factor, ids[idx]

    for method in methods:
        if method not in products["methods"]:
            if verbose:
                print(f"[density enrich] method={method!r} missing; skipped.")
            continue

        records = products["methods"][method].get("records", [])
        if len(records) == 0:
            if verbose:
                print(f"[density enrich] method={method!r} has no records; skipped.")
            continue

        basis = _az_fit_orbit_basis_from_records(records)

        if verbose:
            print(f"[density enrich] method={method}: records={len(records)}")

        for rec in records:
            if (not force) and rec.get("_fof_gas_enriched", False):
                continue

            snap = int(rec.get("snap", rec.get("SnapNum")))
            sid = int(rec.get("sid", rec.get("TargetSubfindID", -1)))

            if fof_gid_mode == "host_central":
                gid_env = int(rec.get("z0_host_central_gid", rec.get("Z0HostCentralGroupID", rec.get("gid", -1))))
            elif fof_gid_mode == "target":
                gid_env = int(rec.get("gid", rec.get("TargetGroupID", -1)))
            else:
                raise ValueError("fof_gid_mode must be 'host_central' or 'target'.")

            cen_sid = int(rec.get("z0_host_central_sid", rec.get("Z0HostCentralSubfindID", -1)))

            if verbose:
                print(f"[density enrich] snap={snap}, sid={sid}, env_gid={gid_env}, host_cen_sid={cen_sid}")

            header = hd_tng.read_header_for_snap(BASE_PATH, snap, sim_name=SIM_NAME, api_key=API_KEY)
            cat, halos, subs = _open_catalog_for_snap(snap)

            try:
                if cen_sid >= 0 and cen_sid < len(subs["SubhaloPos"]):
                    host_center_ckpc_h = np.asarray(subs["SubhaloPos"][cen_sid], dtype=float)
                else:
                    host_center_ckpc_h = np.asarray(halos["GroupPos"][gid_env], dtype=float)

                # Existing DM/star aliases from the original products.
                if "xy_dm_global" not in rec:
                    rec["xy_dm_global"] = np.asarray(rec.get("xy_global", np.empty((0, 2))), dtype=float)
                if "dm_masses" not in rec:
                    rec["dm_masses"] = np.asarray(rec.get("masses", np.empty(0)), dtype=float)
                if "xy_star_global" not in rec and "X_star_global_kpc" in rec:
                    rec["xy_star_global"] = _az_project_to_basis(rec["X_star_global_kpc"], basis)
                if "star_masses" not in rec:
                    rec["star_masses"] = np.asarray(rec.get("stellar_masses", np.empty(0)), dtype=float)

                # Target gas particles.
                gas_block = _load_particle_block(
                    cat,
                    kind="subhalo",
                    obj_id=sid,
                    ptype=0,
                    fields=["Coordinates", "Masses", "ParticleIDs"],
                )
                gas_coords, gas_m, gas_ids = _coords_masses_ids_from_block(gas_block, ptype=0, header=header)
                gas_coords, gas_m, gas_ids = _downsample(gas_coords, gas_m, gas_ids, max_subhalo_gas_particles)
                X_gas_host = (
                    hd_tng.tng_relative_positions_to_physical_kpc(gas_coords, host_center_ckpc_h, header)
                    if gas_coords.shape[0]
                    else np.empty((0, 3), dtype=float)
                )
                rec["X_gas_global_kpc"] = X_gas_host
                rec["xy_gas_global"] = _az_project_to_basis(X_gas_host, basis)
                rec["gas_masses"] = gas_m
                rec["gas_ids"] = gas_ids

                rec["xy_all_global"], rec["all_masses"] = _az_combine_xy_mass([
                    (rec.get("xy_dm_global", np.empty((0, 2))), rec.get("dm_masses", np.empty(0))),
                    (rec.get("xy_star_global", np.empty((0, 2))), rec.get("star_masses", np.empty(0))),
                    (rec.get("xy_gas_global", np.empty((0, 2))), rec.get("gas_masses", np.empty(0))),
                ])

                # FoF/environment particles.
                fof_dm_block = _load_particle_block(cat, kind="halo", obj_id=gid_env, ptype=1, fields=["Coordinates", "ParticleIDs"])
                fof_dm_coords, fof_dm_m, fof_dm_ids = _coords_masses_ids_from_block(fof_dm_block, ptype=1, header=header)
                fof_dm_coords, fof_dm_m, fof_dm_ids = _downsample(fof_dm_coords, fof_dm_m, fof_dm_ids, max_fof_particles_per_type)

                fof_gas_block = _load_particle_block(cat, kind="halo", obj_id=gid_env, ptype=0, fields=["Coordinates", "Masses", "ParticleIDs"])
                fof_gas_coords, fof_gas_m, fof_gas_ids = _coords_masses_ids_from_block(fof_gas_block, ptype=0, header=header)
                fof_gas_coords, fof_gas_m, fof_gas_ids = _downsample(fof_gas_coords, fof_gas_m, fof_gas_ids, max_fof_particles_per_type)

                fof_star_block = _load_particle_block(cat, kind="halo", obj_id=gid_env, ptype=4, fields=["Coordinates", "Masses", "ParticleIDs"])
                fof_star_coords, fof_star_m, fof_star_ids = _coords_masses_ids_from_block(fof_star_block, ptype=4, header=header)
                fof_star_coords, fof_star_m, fof_star_ids = _downsample(fof_star_coords, fof_star_m, fof_star_ids, max_fof_particles_per_type)

                def _host_project(coords):
                    X = (
                        hd_tng.tng_relative_positions_to_physical_kpc(coords, host_center_ckpc_h, header)
                        if coords.shape[0]
                        else np.empty((0, 3), dtype=float)
                    )
                    return X, _az_project_to_basis(X, basis)

                fof_X_dm, fof_xy_dm = _host_project(fof_dm_coords)
                fof_X_gas, fof_xy_gas = _host_project(fof_gas_coords)
                fof_X_star, fof_xy_star = _host_project(fof_star_coords)

                rec["fof_gid"] = gid_env
                rec["fof_X_dm_global_kpc"] = fof_X_dm
                rec["fof_xy_dm_global"] = fof_xy_dm
                rec["fof_dm_masses"] = fof_dm_m
                rec["fof_dm_ids"] = fof_dm_ids

                rec["fof_X_gas_global_kpc"] = fof_X_gas
                rec["fof_xy_gas_global"] = fof_xy_gas
                rec["fof_gas_masses"] = fof_gas_m
                rec["fof_gas_ids"] = fof_gas_ids

                rec["fof_X_star_global_kpc"] = fof_X_star
                rec["fof_xy_star_global"] = fof_xy_star
                rec["fof_star_masses"] = fof_star_m
                rec["fof_star_ids"] = fof_star_ids

                rec["fof_xy_all_global"], rec["fof_all_masses"] = _az_combine_xy_mass([
                    (fof_xy_dm, fof_dm_m),
                    (fof_xy_star, fof_star_m),
                    (fof_xy_gas, fof_gas_m),
                ])

                rec["_fof_gas_enriched"] = True

                if verbose:
                    print(
                        "    loaded:",
                        f"target gas={len(gas_m):d},",
                        f"FoF DM={len(fof_dm_m):d},",
                        f"FoF stars={len(fof_star_m):d},",
                        f"FoF gas={len(fof_gas_m):d}",
                    )

            finally:
                try:
                    cat.cleanup()
                except Exception:
                    pass

    return products


if RUN_DENSITY_PARTICLE_ENRICHMENT:
    PRODUCTS = enrich_products_with_fof_and_gas_particles(
        PRODUCTS,
        methods=DENSITY_ENRICH_METHODS,
        fof_gid_mode=FOF_GID_MODE,
        max_fof_particles_per_type=MAX_FOF_PARTICLES_PER_TYPE,
        max_subhalo_gas_particles=MAX_SUBHALO_GAS_PARTICLES,
        force=False,
        verbose=True,
    )

    if SAVE_PRODUCTS_AFTER_DENSITY_ENRICHMENT:
        try:
            save_cross_time_products(PRODUCTS, PRODUCTS_PATH)
        except NameError:
            with open(PRODUCTS_PATH, "wb") as f:
                pickle.dump(PRODUCTS, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("[density enrich] saved enriched PRODUCTS to", PRODUCTS_PATH)
else:
    print("[density enrich] skipped; set RUN_DENSITY_PARTICLE_ENRICHMENT=True to load FoF/gas particles.")

# %% [markdown] cell 16
# ## Plot functions

# %% code cell 17
# ============================================================
# plot_subhalo_orbitplane_overlay
# ============================================================
# Plot only from computed products. No download, no recomputation.
#
# Redshift label:
#   placed outside each subhalo along the outward radial direction from
#   the FoF centre, so it does not cover the density map / shell contour.
#
# Usage:
#   fig, ax, records, table_df, save_path = plot_subhalo_orbitplane_overlay(
#       PRODUCTS,
#       method="radial",
#   )
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator
from pathlib import Path


def plot_subhalo_orbitplane_overlay(
    products,
    *,
    method="radial",
    path=None,
    OUTDIR=None,
    COMMON_LIM_KPC=None,
    PAD_FACTOR=1.15,
    DENSITY_BINS_PER_SUBHALO=400,
    DENSITY_ALPHA=0.72,
    DENSITY_PERCENTILE_LIMIT=98.0,
    DENSITY_CMAP="magma",
    DRAW_DENSITY_CONTOURS=True,
    DENSITY_CONTOUR_LEVELS=6,
    DENSITY_CONTOUR_COLOR="#9eeaff",
    DENSITY_CONTOUR_LW=0.55,
    DRAW_SHELL_ELLIPSES=True,
    ELLIPSE_NSIGMA=2.0,
    SHELL_COLOR="#a5ba9e",
    SHELL_BASE_LW=0.85,
    SHELL_STROKE_LW=1.20,
    DRAW_GALAXY_MAJOR_AXIS=True,
    GALAXY_MAJOR_AXIS_COLOR="#ff2020",
    GALAXY_MAJOR_AXIS_LW=1.55,
    GALAXY_MAJOR_AXIS_NSIGMA=2.2,
    FIG_FACE="#d9d9d9",
    AX_FACE="black",
    TICK_COLOR="white",
    TICKLABEL_COLOR="black",
    SPINE_COLOR="white",
    TEXT_COLOR="black",
    save=True,
    show=True,
    dpi=220,
):
    """
    Draw the orbit-plane overlay from precomputed products.

    Expected product structure:
        products["methods"][method]["records"]
        products["methods"][method]["table_df"]

    Returns
    -------
    fig, ax, records, table_df, save_path
    """

    if not isinstance(products, dict):
        raise TypeError("products must be a dict produced by the compute-products cell.")

    if OUTDIR is None:
        OUTDIR = path
    if OUTDIR is None:
        OUTDIR = Path(globals().get("OUTDIR", "hd_tng_outputs"))
    else:
        OUTDIR = Path(OUTDIR)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    method = str(method)
    SID0 = int(products.get("SID0", products.get("sid", -1)))
    SIM_NAME = products.get("SIM_NAME", globals().get("SIM_NAME", "TNG"))

    if "methods" not in products or method not in products["methods"]:
        raise KeyError(f"method={method!r} not found in products['methods'].")

    block = products["methods"][method]

    records = block.get("records", None)
    if records is None or len(records) == 0:
        raise RuntimeError(f"No precomputed records found for method={method!r}.")

    table_df = block.get("table_df", pd.DataFrame())

    # ------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------
    def _safe_ellipse_from_points(xy, weights=None, nsigma=2.0):
        xy = np.asarray(xy, dtype=float)

        if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 5:
            return None

        if weights is None:
            w = np.ones(xy.shape[0], dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if w.shape != (xy.shape[0],):
                return None

        good = np.isfinite(xy).all(axis=1) & np.isfinite(w) & (w > 0)
        if np.count_nonzero(good) < 5:
            return None

        xy = xy[good]
        w = w[good]

        cen = np.sum(xy * w[:, None], axis=0) / np.sum(w)
        Y = xy - cen[None, :]
        C = np.einsum("n,ni,nj->ij", w, Y, Y) / np.sum(w)
        C = 0.5 * (C + C.T)

        vals, vecs = np.linalg.eigh(C)
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]

        if np.any(vals <= 0) or not np.all(np.isfinite(vals)):
            return None

        width = 2.0 * nsigma * np.sqrt(vals[0])
        height = 2.0 * nsigma * np.sqrt(vals[1])
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

        return cen, width, height, angle

    def _draw_density_map_with_contours(ax, xy, xlim, ylim):
        xy = np.asarray(xy, dtype=float)

        good = (
            np.isfinite(xy).all(axis=1)
            & (xy[:, 0] >= xlim[0]) & (xy[:, 0] <= xlim[1])
            & (xy[:, 1] >= ylim[0]) & (xy[:, 1] <= ylim[1])
        )
        xy = xy[good]

        if xy.shape[0] < 12:
            return None

        H, xe, ye = np.histogram2d(
            xy[:, 0],
            xy[:, 1],
            bins=int(DENSITY_BINS_PER_SUBHALO),
            range=[xlim, ylim],
        )
        H = H.T

        pos = H > 0
        if np.count_nonzero(pos) < 5:
            return None

        img = np.full_like(H, np.nan, dtype=float)
        img[pos] = np.log10(H[pos])

        im = ax.imshow(
            img,
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
            origin="lower",
            interpolation="nearest",
            cmap=DENSITY_CMAP,
            alpha=DENSITY_ALPHA,
            aspect="equal",
            zorder=2,
        )

        if not DRAW_DENSITY_CONTOURS:
            return im

        valid = np.isfinite(img)
        if np.count_nonzero(valid) < 5:
            return im

        vmin = np.nanmin(img[valid])
        vmax = np.nanmax(img[valid])

        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return im

        levels = np.linspace(
            vmin + 0.20 * (vmax - vmin),
            vmax,
            int(DENSITY_CONTOUR_LEVELS),
        )

        xc = 0.5 * (xe[:-1] + xe[1:])
        yc = 0.5 * (ye[:-1] + ye[1:])

        ax.contour(
            xc,
            yc,
            img,
            levels=levels,
            colors=DENSITY_CONTOUR_COLOR,
            linewidths=DENSITY_CONTOUR_LW,
            alpha=0.72,
            zorder=6,
        )

        return im

    def _draw_shell_ellipses(ax, xy, masks, masses):
        if not DRAW_SHELL_ELLIPSES:
            return

        xy = np.asarray(xy, dtype=float)
        masses = np.asarray(masses, dtype=float)

        for ish, mi in enumerate(masks):
            mi = np.asarray(mi, dtype=bool)

            if mi.shape[0] != xy.shape[0]:
                continue
            if np.count_nonzero(mi) < 5:
                continue

            wm = masses[mi] if masses.shape[0] == xy.shape[0] else None
            ell = _safe_ellipse_from_points(
                xy[mi],
                weights=wm,
                nsigma=ELLIPSE_NSIGMA,
            )

            if ell is None:
                continue

            cen, width, height, angle = ell
            alpha = 0.42 + 0.45 * (ish + 1) / max(len(masks), 1)

            e = Ellipse(
                cen,
                width=width,
                height=height,
                angle=angle,
                fill=False,
                lw=SHELL_BASE_LW,
                color=SHELL_COLOR,
                alpha=alpha,
                zorder=30 + ish,
                path_effects=[
                    patheffects.withStroke(
                        linewidth=SHELL_STROKE_LW,
                        foreground="white",
                        alpha=0.28,
                    )
                ],
            )
            ax.add_patch(e)

    def _add_scalebar_with_endticks(ax, xlim, ylim):
        x0, x1 = map(float, xlim)
        y0, y1 = map(float, ylim)

        xr = abs(x1 - x0)
        yr = abs(y1 - y0)

        raw = 0.18 * xr
        pow10 = 10 ** np.floor(np.log10(max(raw, 1e-6)))
        candidates = np.array([1, 2, 5, 10], dtype=float) * pow10
        length_kpc = float(candidates[np.argmin(np.abs(candidates - raw))])

        label = f"{int(length_kpc):d} kpc" if abs(length_kpc - int(length_kpc)) < 1e-8 else f"{length_kpc:g} kpc"

        xb = x0 + 0.08 * xr
        yb = y0 + 0.085 * yr
        tick_h = 0.026 * yr

        ax.plot(
            [xb, xb + length_kpc],
            [yb, yb],
            lw=0.80,
            color="white",
            solid_capstyle="butt",
            zorder=1000,
        )
        ax.plot(
            [xb, xb],
            [yb - tick_h / 2.0, yb + tick_h / 2.0],
            lw=0.80,
            color="white",
            zorder=1001,
        )
        ax.plot(
            [xb + length_kpc, xb + length_kpc],
            [yb - tick_h / 2.0, yb + tick_h / 2.0],
            lw=0.80,
            color="white",
            zorder=1001,
        )

        ax.text(
            xb + 0.5 * length_kpc,
            yb + 0.045 * yr,
            label,
            ha="center",
            va="bottom",
            fontsize=13,
            color="white",
            fontweight="bold",
            zorder=1002,
            path_effects=[
                patheffects.withStroke(
                    linewidth=2.2,
                    foreground="black",
                    alpha=0.95,
                )
            ],
        )

    def _outside_redshift_label_position(
        xy,
        center_xy,
        xlim,
        ylim,
        *,
        percentile=98.5,
        pad_frac=0.035,
    ):
        """
        Place redshift label outside the subhalo body.

        Direction:
            primarily away from the FoF centre at (0, 0).
            If the subhalo is near the centre, use the local major-axis direction.
        """
        xy = np.asarray(xy, dtype=float)
        center_xy = np.asarray(center_xy, dtype=float)

        good = np.isfinite(xy).all(axis=1)
        xy = xy[good]

        if xy.shape[0] < 5 or not np.all(np.isfinite(center_xy)):
            return center_xy + np.array([0.0, 0.0])

        direction = center_xy.copy()
        nd = np.linalg.norm(direction)

        if not np.isfinite(nd) or nd < 1e-8:
            # fallback: use projected particle PCA major axis
            Y = xy - np.nanmedian(xy, axis=0, keepdims=True)
            C = np.cov(Y.T)
            vals, vecs = np.linalg.eigh(C)
            direction = vecs[:, np.argmax(vals)]
            nd = np.linalg.norm(direction)

        if not np.isfinite(nd) or nd < 1e-8:
            direction = np.array([1.0, 0.0], dtype=float)
        else:
            direction = direction / nd

        rel = xy - center_xy[None, :]
        proj = rel @ direction
        finite = np.isfinite(proj)

        if np.count_nonzero(finite) == 0:
            outer = 0.0
        else:
            outer = float(np.nanpercentile(proj[finite], percentile))
            outer = max(outer, 0.0)

        xrange = abs(xlim[1] - xlim[0])
        yrange = abs(ylim[1] - ylim[0])
        pad = pad_frac * max(xrange, yrange)

        label_xy = center_xy + direction * (outer + pad)

        # Slight tangential offset to avoid sitting directly on contours.
        tangent = np.array([-direction[1], direction[0]], dtype=float)
        label_xy = label_xy + 0.008 * max(xrange, yrange) * tangent

        # Clip inside plotting region.
        margin_x = 0.035 * xrange
        margin_y = 0.035 * yrange

        label_xy[0] = np.clip(label_xy[0], xlim[0] + margin_x, xlim[1] - margin_x)
        label_xy[1] = np.clip(label_xy[1], ylim[0] + margin_y, ylim[1] - margin_y)

        return label_xy

    # ------------------------------------------------------------
    # Common view limits
    # ------------------------------------------------------------
    points = [np.array([0.0, 0.0], dtype=float)]

    for rec in records:
        if "center_xy" in rec:
            points.append(np.asarray(rec["center_xy"], dtype=float))

    points = np.asarray(points, dtype=float)
    good_points = np.isfinite(points).all(axis=1)

    if np.count_nonzero(good_points):
        view_center = np.nanmean(points[good_points], axis=0)
    else:
        view_center = np.array([0.0, 0.0], dtype=float)

    if COMMON_LIM_KPC is None:
        vals = []

        if np.count_nonzero(good_points):
            vals.append(np.nanmax(np.abs(points[good_points] - view_center[None, :])))

        for rec in records:
            xy = np.asarray(rec.get("xy_global", []), dtype=float)
            if xy.ndim == 2 and xy.shape[1] == 2:
                good = np.isfinite(xy).all(axis=1)
                if np.count_nonzero(good) > 0:
                    vals.append(
                        np.nanpercentile(
                            np.abs(xy[good] - view_center[None, :]),
                            DENSITY_PERCENTILE_LIMIT,
                        )
                    )

        half_width = float(np.nanmax(vals) * PAD_FACTOR) if len(vals) else 100.0
        if not np.isfinite(half_width) or half_width <= 0:
            half_width = 100.0
    else:
        half_width = float(COMMON_LIM_KPC)

    xlim = (view_center[0] - half_width, view_center[0] + half_width)
    ylim = (view_center[1] - half_width, view_center[1] + half_width)

    # ------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------
    plt.close("all")

    fig, ax = plt.subplots(
        figsize=(9.2, 8.8),
        facecolor=FIG_FACE,
        constrained_layout=True,
    )

    ax.set_facecolor(AX_FACE)

    # FoF / z=0 host central MPB origin.
    ax.scatter(
        [0.0],
        [0.0],
        marker="x",
        s=95,
        linewidths=1.6,
        color="white",
        zorder=250,
    )

    for rec in records:
        xy = np.asarray(rec.get("xy_global", []), dtype=float)
        center_xy = np.asarray(rec.get("center_xy", np.full(2, np.nan)), dtype=float)
        masses = np.asarray(rec.get("masses", np.ones(len(xy))), dtype=float)
        masks = rec.get("masks", [])

        if xy.ndim != 2 or xy.shape[1] != 2:
            continue

        _draw_density_map_with_contours(ax, xy, xlim, ylim)
        _draw_shell_ellipses(ax, xy, masks, masses)

        # Bright red galaxy major axis.
        if DRAW_GALAXY_MAJOR_AXIS:
            v = np.asarray(rec.get("galaxy_major_axis_xy", np.full(2, np.nan)), dtype=float)
            nv = np.linalg.norm(v)

            if np.isfinite(nv) and nv > 0 and np.all(np.isfinite(center_xy)):
                v = v / nv

                gal_shape = rec.get("gal_shape", {})
                evals = np.asarray(gal_shape.get("evals", []), dtype=float) if isinstance(gal_shape, dict) else np.array([])

                if evals.size > 0 and np.isfinite(evals[0]) and evals[0] > 0:
                    length = GALAXY_MAJOR_AXIS_NSIGMA * np.sqrt(evals[0])
                else:
                    length = 0.035 * (xlim[1] - xlim[0])

                p0 = center_xy - length * v
                p1 = center_xy + length * v

                ax.plot(
                    [p0[0], p1[0]],
                    [p0[1], p1[1]],
                    color=GALAXY_MAJOR_AXIS_COLOR,
                    lw=GALAXY_MAJOR_AXIS_LW,
                    alpha=0.98,
                    zorder=120,
                    solid_capstyle="round",
                    path_effects=[
                        patheffects.withStroke(
                            linewidth=GALAXY_MAJOR_AXIS_LW + 1.3,
                            foreground="black",
                            alpha=0.65,
                        )
                    ],
                )

        # Redshift label outside subhalo.
        label_xy = _outside_redshift_label_position(
            xy,
            center_xy,
            xlim,
            ylim,
            percentile=98.5,
            pad_frac=0.035,
        )

        # Align text away from the subhalo centre.
        dx = label_xy[0] - center_xy[0]
        dy = label_xy[1] - center_xy[1]
        ha = "left" if dx >= 0 else "right"
        va = "bottom" if dy >= 0 else "top"

        zval = rec.get("z", rec.get("Redshift", np.nan))
        try:
            ztxt = f"z={float(zval):.2f}"
        except Exception:
            ztxt = "z=?"

        ax.text(
            label_xy[0],
            label_xy[1],
            ztxt,
            fontsize=11.5,
            color=TEXT_COLOR,
            ha=ha,
            va=va,
            zorder=350,
            path_effects=[
                patheffects.withStroke(
                    linewidth=3.0,
                    foreground="white",
                    alpha=0.95,
                )
            ],
        )

    _add_scalebar_with_endticks(ax, xlim, ylim)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))

    ax.tick_params(
        direction="in",
        top=True,
        right=True,
        color=TICK_COLOR,
        labelcolor=TICKLABEL_COLOR,
        labelsize=12,
        length=5,
        width=0.9,
    )

    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(1.1)

    ax.set_xlabel(
        "orbit-plane coordinate u [kpc]",
        fontsize=14,
        color="black",
    )

    ax.set_ylabel(
        "orbit-plane coordinate v [kpc]",
        fontsize=14,
        color="black",
    )

    ax.set_title(
        f"{SIM_NAME}: tracked subhalo ID: {SID0} in orbit plane\n"
        f"{method} shells",
        fontsize=16,
        color="black",
    )

    save_path = None
    if save:
        save_path = OUTDIR / f"subhalo_{SID0}_{method}_orbitplane_overlay.png"
        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        print("Saved:", save_path)

    if show:
        plt.show()

    return fig, ax, records, table_df, save_path

# %% code cell 18
# ============================================================
# plot_subhalo_orbitplane_table_evolution
# ============================================================
# Changes relative to previous version
# -----------------------------------
# 1. Publication-style layout retained.
# 2. Shell labels are displayed from 1.
# 3. The host-centric distance panel now has the SAME x-axis direction as
#    the other panels: high redshift on the left, low redshift on the right.
#    This is enforced both on the main axis and on the twin y-axis.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


def plot_subhalo_orbitplane_table_evolution(
    products,
    *,
    method="radial",
    table_df=None,
    path=None,
    OUTDIR=None,
    x_key="Redshift",
    invert_redshift_axis=True,
    save=True,
    show=True,
    dpi=300,
):
    """
    Plot mass/shape/alignment time evolution from precomputed products.

    This function does not read TNG catalogs, download cutouts, or recompute
    shell quantities.  It only uses the table saved by
    compute_cross_time_products(...).

    Parameters
    ----------
    products : dict
        Product dictionary loaded from PRODUCTS_PATH.
    method : {"radial", "binding_energy"}
        Shell method to draw.
    table_df : pandas.DataFrame or None
        Optional table override.  If None, uses products["methods"][method]["table_df"].

    Returns
    -------
    figs : list
        [fig_align, fig_phys]
    table_df : pandas.DataFrame
    save_paths : list[pathlib.Path]
    """

    if OUTDIR is None:
        OUTDIR = path
    if OUTDIR is None:
        OUTDIR = Path(globals().get("OUTDIR", "hd_tng_outputs"))
    else:
        OUTDIR = Path(OUTDIR)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if not isinstance(products, dict):
        raise TypeError("products must be a dict produced by compute_cross_time_products or load_cross_time_products.")

    SID0 = int(products.get("SID0", products.get("sid", -1)))
    SNAP0 = int(products.get("SNAP0", products.get("snap0", 99)))
    SNAP_TRACK = [int(s) for s in products.get("SNAP_TRACK", [])]
    method = str(method)

    if table_df is None:
        if "methods" not in products or method not in products["methods"]:
            raise KeyError(f"method={method!r} not found in products['methods'].")
        table_df = products["methods"][method].get("table_df", pd.DataFrame()).copy()
    else:
        table_df = table_df.copy()

    if not isinstance(table_df, pd.DataFrame) or len(table_df) == 0:
        raise RuntimeError("The diagnostic table is empty or invalid.")

    if x_key not in table_df.columns:
        raise KeyError(f"x_key={x_key!r} not found in table_df columns.")

    # 2. Publication style
    # ------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.0,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "figure.dpi": 150,
        "savefig.dpi": dpi,
    })

    AXIS_COLORS = {
        "major": "#0072B2",         # blue
        "intermediate": "#D55E00",  # vermillion
        "minor": "#009E73",         # green
    }

    AXIS_MARKERS = {
        "major": "o",
        "intermediate": "s",
        "minor": "^",
    }

    AXIS_LABELS = {
        "major": "major",
        "intermediate": "intermediate",
        "minor": "minor",
    }

    SHELL_COLORS = [
        "#332288", "#88CCEE", "#44AA99", "#117733",
        "#999933", "#DDCC77", "#CC6677", "#882255",
        "#AA4499", "#DDDDDD",
    ]
    SHELL_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]

    GRID_COLOR = "0.86"
    ZERO_COLOR = "0.35"

    axis_names = ["major", "intermediate", "minor"]

    # ------------------------------------------------------------
    # 3. Internal helpers
    # ------------------------------------------------------------
    def _sort_df_for_plot(df):
        df = df.copy()
        x = pd.to_numeric(df[x_key], errors="coerce").to_numpy(dtype=float)

        if x_key == "Redshift":
            order = np.argsort(x)[::-1] if invert_redshift_axis else np.argsort(x)
        else:
            order = np.argsort(x)

        return df.iloc[order].reset_index(drop=True)

    def _snapshot_table(df):
        out = (
            df.sort_values(["SnapNum", "Shell"])
            .drop_duplicates("SnapNum")
            .reset_index(drop=True)
        )
        return _sort_df_for_plot(out)

    def _add_shell_plot_index(df):
        df = df.copy()
        if "Shell" not in df.columns:
            df["Shell"] = 0

        shells = pd.to_numeric(df["Shell"], errors="coerce")
        if shells.notna().any() and int(np.nanmin(shells.to_numpy(dtype=float))) == 0:
            df["ShellPlot"] = shells.astype(int) + 1
        else:
            df["ShellPlot"] = shells.astype(int)

        return df

    def _apply_axis_style(ax, *, ylim=None):
        ax.tick_params(direction="in", top=True, right=True, length=4.0, width=0.8)
        ax.grid(True, color=GRID_COLOR, lw=0.6, alpha=0.75)
        ax.set_axisbelow(True)

        if ylim is not None:
            ax.set_ylim(*ylim)

        if x_key == "Redshift" and invert_redshift_axis:
            ax.invert_xaxis()

    def _plot_if_column(ax, df, col, label=None, color="k", marker="o", ls="-", lw=1.5, alpha=1.0):
        if col not in df.columns:
            return False

        x = pd.to_numeric(df[x_key], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

        good = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(good) == 0:
            return False

        ax.plot(
            x[good],
            y[good],
            marker=marker,
            ms=4.2,
            ls=ls,
            lw=lw,
            alpha=alpha,
            color=color,
            label=label if label is not None else col,
        )
        return True

    def _plot_global_axis_set(ax, df, columns, title, ylabel=r"$|\cos\theta|$"):
        has_any = False
        for name in axis_names:
            col = columns[name]
            if col not in df.columns:
                continue

            x = pd.to_numeric(df[x_key], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            good = np.isfinite(x) & np.isfinite(y)

            if np.count_nonzero(good) == 0:
                continue

            ax.plot(
                x[good],
                y[good],
                color=AXIS_COLORS[name],
                marker=AXIS_MARKERS[name],
                ms=4.2,
                lw=1.55,
                label=AXIS_LABELS[name],
            )
            has_any = True

        if not has_any:
            ax.text(
                0.5, 0.5,
                "No valid data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(x_key)
        _apply_axis_style(ax, ylim=(-0.03, 1.03))

    def _plot_shell_family(ax, df, col, title, ylabel=r"$|\cos\theta|$", ylim=(-0.03, 1.03)):
        if col not in df.columns:
            ax.text(
                0.5, 0.5,
                f"Missing column:\n{col}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
            )
            ax.set_title(title)
            ax.set_xlabel(x_key)
            ax.set_ylabel(ylabel)
            _apply_axis_style(ax, ylim=ylim)
            return False

        has_any = False

        for k, (shell_plot, sub) in enumerate(df.groupby("ShellPlot")):
            sub = _sort_df_for_plot(sub)
            x = pd.to_numeric(sub[x_key], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)

            good = np.isfinite(x) & np.isfinite(y)
            if np.count_nonzero(good) == 0:
                continue

            color = SHELL_COLORS[k % len(SHELL_COLORS)]
            marker = SHELL_MARKERS[k % len(SHELL_MARKERS)]

            ax.plot(
                x[good],
                y[good],
                marker=marker,
                ms=3.8,
                lw=1.25,
                alpha=0.92,
                color=color,
                label=f"Shell {int(shell_plot)}",
            )
            has_any = True

        if not has_any:
            ax.text(
                0.5, 0.5,
                "No valid data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(x_key)
        _apply_axis_style(ax, ylim=ylim)
        return has_any

    snap_df = _snapshot_table(table_df)
    shell_df = _add_shell_plot_index(table_df)

    figs = []
    save_paths = []

    # ============================================================
    # Figure 1: all alignment quantities
    # ============================================================
    fig_align, axes = plt.subplots(
        3, 3,
        figsize=(14.8, 11.2),
        constrained_layout=True,
    )

    _plot_global_axis_set(
        axes[0, 0],
        snap_df,
        {
            "major": "cos_galaxy_dmhalo_major",
            "intermediate": "cos_galaxy_dmhalo_intermediate",
            "minor": "cos_galaxy_dmhalo_minor",
        },
        r"(a) Galaxy axes vs global DM halo axes",
    )

    _plot_global_axis_set(
        axes[0, 1],
        snap_df,
        {
            "major": "cos_galaxy_major_radial",
            "intermediate": "cos_galaxy_intermediate_radial",
            "minor": "cos_galaxy_minor_radial",
        },
        r"(b) Galaxy axes vs host-centric radial direction",
    )

    _plot_global_axis_set(
        axes[0, 2],
        snap_df,
        {
            "major": "cos_dmhalo_major_radial",
            "intermediate": "cos_dmhalo_intermediate_radial",
            "minor": "cos_dmhalo_minor_radial",
        },
        r"(c) Global DM halo axes vs host-centric radial direction",
    )

    for j, name in enumerate(axis_names):
        _plot_shell_family(
            axes[1, j],
            shell_df,
            f"cos_galaxy_shell_{name}",
            title=rf"({chr(ord('d') + j)}) Galaxy {AXIS_LABELS[name]} axis vs shell {AXIS_LABELS[name]} axis",
            ylabel=r"$|\cos\theta|$",
            ylim=(-0.03, 1.03),
        )

    for j, name in enumerate(axis_names):
        _plot_shell_family(
            axes[2, j],
            shell_df,
            f"cos_shell_{name}_radial",
            title=rf"({chr(ord('g') + j)}) Shell {AXIS_LABELS[name]} axis vs radial direction",
            ylabel=r"$|\cos\theta|$",
            ylim=(-0.03, 1.03),
        )

    axis_handles = [
        Line2D([0], [0], color=AXIS_COLORS[name], marker=AXIS_MARKERS[name], lw=1.7, label=AXIS_LABELS[name])
        for name in axis_names
    ]

    shell_values = sorted(shell_df["ShellPlot"].dropna().unique().astype(int).tolist())
    shell_handles = [
        Line2D(
            [0], [0],
            color=SHELL_COLORS[k % len(SHELL_COLORS)],
            marker=SHELL_MARKERS[k % len(SHELL_MARKERS)],
            lw=1.4,
            label=f"Shell {shell}",
        )
        for k, shell in enumerate(shell_values)
    ]

    leg1 = fig_align.legend(
        handles=axis_handles,
        loc="upper left",
        bbox_to_anchor=(0.075, 1.018),
        frameon=False,
        ncol=3,
        title="Axis type",
        handlelength=1.8,
        columnspacing=1.2,
    )

    fig_align.legend(
        handles=shell_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 1.018),
        frameon=False,
        ncol=min(6, max(1, len(shell_handles))),
        title="DM shells",
        handlelength=1.7,
        columnspacing=1.0,
    )

    fig_align.add_artist(leg1)

    fig_align.suptitle(
        f"Subhalo {SID0}: alignment evolution in the z=0 host-central frame",
        fontsize=15,
        y=1.045,
    )

    figs.append(fig_align)

    if save:
        p = OUTDIR / f"subhalo_{SID0}_alignment_evolution_publication.png"
        fig_align.savefig(p, dpi=dpi, bbox_inches="tight")
        save_paths.append(p)
        print("Saved:", p)

    # ============================================================
    # Figure 2: all non-alignment physical quantities
    # ============================================================
    fig_phys, axes = plt.subplots(
        2, 3,
        figsize=(14.8, 7.6),
        constrained_layout=True,
    )

    # (a) Mass components
    ax = axes[0, 0]
    _plot_if_column(ax, snap_df, "Mdm_msun",      label=r"$M_{\rm DM}$",      color="#0072B2", marker="o")
    _plot_if_column(ax, snap_df, "Mbaryon_msun",  label=r"$M_{\rm baryon}$",  color="#009E73", marker="s")
    _plot_if_column(ax, snap_df, "Mstar_msun",    label=r"$M_\star$",         color="#D55E00", marker="^")
    _plot_if_column(ax, snap_df, "Mgas_msun",     label=r"$M_{\rm gas}$",     color="#CC79A7", marker="D")
    ax.set_yscale("log")
    ax.set_title(r"(a) Mass components")
    ax.set_ylabel(r"Mass [$M_\odot$]")
    ax.set_xlabel(x_key)
    ax.legend(frameon=False, fontsize=8.8)
    _apply_axis_style(ax)

    # (b) Mass ratio
    ax = axes[0, 1]
    _plot_if_column(
        ax,
        snap_df,
        "Mbaryon_over_Mdm",
        label=r"$M_{\rm baryon}/M_{\rm DM}$",
        color="#E69F00",
        marker="o",
    )
    ax.axhline(0.0, color=ZERO_COLOR, lw=0.8, alpha=0.55)
    ax.set_title(r"(b) Baryon-to-DM mass ratio")
    ax.set_ylabel(r"$M_{\rm baryon}/M_{\rm DM}$")
    ax.set_xlabel(x_key)
    ax.legend(frameon=False, fontsize=8.8)
    _apply_axis_style(ax)

    # (c) Host-centric distance  -- explicitly force same x-direction as others
    ax = axes[0, 2]
    _plot_if_column(ax, snap_df, "R_hostref_kpc", label=r"$R$", color="#56B4E9", marker="o")
    ax.set_title(r"(c) Host-centric distance")
    ax.set_ylabel(r"$R$ [kpc]")
    ax.set_xlabel(x_key)
    _apply_axis_style(ax)

    ax_r = ax.twinx()
    if "R_over_R200c" in snap_df.columns:
        x = pd.to_numeric(snap_df[x_key], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(snap_df["R_over_R200c"], errors="coerce").to_numpy(dtype=float)
        good = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(good):
            ax_r.plot(
                x[good],
                y[good],
                color="#882255",
                marker="s",
                ms=4.0,
                lw=1.25,
                ls="--",
                label=r"$R/R_{200c}$",
            )
            ax_r.set_ylabel(r"$R/R_{200c}$")
            ax_r.tick_params(direction="in", right=True)

    # Crucial fix: synchronize x-limits/direction explicitly
    ax_r.set_xlim(ax.get_xlim())
    if x_key == "Redshift" and invert_redshift_axis:
        # Re-assert after plotting; this guarantees left=high z, right=low z
        ax.set_xlim(ax.get_xlim())
        ax_r.set_xlim(ax.get_xlim())

    # (d) Particle counts
    ax = axes[1, 0]
    _plot_if_column(ax, snap_df, "Ndm",   label=r"$N_{\rm DM}$",   color="#0072B2", marker="o")
    _plot_if_column(ax, snap_df, "Nstar", label=r"$N_\star$",      color="#D55E00", marker="s")
    ax.set_yscale("log")
    ax.set_title(r"(d) Particle counts")
    ax.set_ylabel("Particle count")
    ax.set_xlabel(x_key)
    ax.legend(frameon=False, fontsize=8.8)
    _apply_axis_style(ax)

    # (e) Galaxy stellar axis lengths
    ax = axes[1, 1]
    for name in axis_names:
        _plot_if_column(
            ax,
            snap_df,
            f"galaxy_{name}_rms_kpc",
            label=AXIS_LABELS[name],
            color=AXIS_COLORS[name],
            marker=AXIS_MARKERS[name],
        )
    ax.set_title(r"(e) Galaxy stellar RMS axis lengths")
    ax.set_ylabel("RMS axis length [kpc]")
    ax.set_xlabel(x_key)
    ax.legend(frameon=False, fontsize=8.8)
    _apply_axis_style(ax)

    # (f) Global DM and median shell RMS axis lengths
    ax = axes[1, 2]

    for name in axis_names:
        _plot_if_column(
            ax,
            snap_df,
            f"dmhalo_{name}_rms_kpc",
            label=rf"global DM {AXIS_LABELS[name]}",
            color=AXIS_COLORS[name],
            marker=AXIS_MARKERS[name],
            ls="-",
            lw=1.45,
        )

        shell_col = f"shell_{name}_rms_kpc"
        if shell_col in shell_df.columns:
            med = (
                shell_df
                .groupby("SnapNum", as_index=False)
                .agg({
                    x_key: "first",
                    shell_col: "median",
                })
            )
            med = _sort_df_for_plot(med)

            _plot_if_column(
                ax,
                med,
                shell_col,
                label=rf"median shell {AXIS_LABELS[name]}",
                color=AXIS_COLORS[name],
                marker=AXIS_MARKERS[name],
                ls="--",
                lw=1.15,
            )

    ax.set_title(r"(f) Global DM and median shell RMS axis lengths")
    ax.set_ylabel("RMS axis length [kpc]")
    ax.set_xlabel(x_key)
    ax.legend(frameon=False, fontsize=7.6, ncol=1)
    _apply_axis_style(ax)

    fig_phys.suptitle(
        f"Subhalo {SID0}: physical evolution in the z=0 host-central frame",
        fontsize=15,
        y=1.035,
    )

    figs.append(fig_phys)

    if save:
        p = OUTDIR / f"subhalo_{SID0}_physical_evolution_publication.png"
        fig_phys.savefig(p, dpi=dpi, bbox_inches="tight")
        save_paths.append(p)
        print("Saved:", p)

    if show:
        plt.show()
    else:
        for fig in figs:
            plt.close(fig)

    return figs, table_df, save_paths

# %% code cell 19
# ============================================================
# plot_pi_closure_table_evolution
# ============================================================
# Plot cross-time Pi-closure evolution from precomputed PRODUCTS.
#
# New behavior:
#   - x-axis is Redshift directly
#   - no residual panels
#   - plot:
#       (1) ||Pi^dI|| and ||Pi^{Omega+H}||
#       (2) |Omega|/(|Omega|+|H|) and |H|/(|Omega|+|H|)
#       (3) angle between Omega and H vectors
#   - show both overall evolution and each shell evolution
#   - shell curves: different colors, lower alpha, line only
#   - no colorbar, no scatter points
#
# Usage:
#   figs_pi, closure_used, track_tables, pi_paths = plot_pi_closure_table_evolution(
#       PRODUCTS,
#       methods=SHELL_METHODS,
#   )
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


def plot_pi_closure_table_evolution(
    products,
    *,
    methods=None,
    closure_df=None,
    path=None,
    OUTDIR=None,
    save=True,
    show=True,
    dpi=300,
    verbose=True,
):
    """
    Plot Pi-closure redshift evolution from precomputed products.

    Parameters
    ----------
    products : dict
        Output from your compute-products stage.
    methods : sequence[str] or None
        Shell methods to plot. If None, use all methods present in products["methods"].
    closure_df : pandas.DataFrame or None
        If given, use this table directly.
    path, OUTDIR : str/path or None
        Output directory.
    save, show : bool
        Save and/or show figures.

    Returns
    -------
    figs_by_method : dict
        One figure per shell method.
    closure_used : pandas.DataFrame
        Standardized closure table used in plotting.
    track_tables : dict
        track_df collected from products.
    save_paths : list[pathlib.Path]
    """

    # ------------------------------------------------------------
    # Resolve output directory
    # ------------------------------------------------------------
    if OUTDIR is None:
        OUTDIR = path
    if OUTDIR is None:
        OUTDIR = Path(globals().get("OUTDIR", "hd_tng_outputs"))
    else:
        OUTDIR = Path(OUTDIR)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if not isinstance(products, dict):
        raise TypeError("products must be a dict produced by the compute-products stage.")

    SID0 = int(products.get("SID0", products.get("sid", -1)))

    if methods is None:
        if "methods" in products and isinstance(products["methods"], dict):
            methods = tuple(products["methods"].keys())
        else:
            methods = ("radial", "binding_energy")
    methods = tuple(str(m) for m in methods)

    # ------------------------------------------------------------
    # Publication style
    # ------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9.6,
        "ytick.labelsize": 9.6,
        "legend.fontsize": 8.8,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "figure.dpi": 150,
        "savefig.dpi": dpi,
    })

    # Okabe-Ito-like publication-friendly palette
    SHELL_COLORS = [
        "#0072B2", "#D55E00", "#009E73", "#CC79A7",
        "#E69F00", "#56B4E9", "#999999", "#882255",
        "#44AA99", "#332288"
    ]

    COLOR_OVERALL_DIRECT = "#1f3b73"
    COLOR_OVERALL_AFF    = "#b24a00"
    COLOR_OVERALL_OMEGA  = "#2b6cb0"
    COLOR_OVERALL_H      = "#c05621"
    COLOR_OVERALL_ANGLE  = "#111111"

    GRID_COLOR = "0.86"

    COMPONENTS = ["01", "02", "12"]

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _drop_duplicate_columns(df):
        return df.loc[:, ~df.columns.duplicated(keep="first")].copy()

    def _choose_col(df, names):
        for name in names:
            if name in df.columns:
                return name
        return None

    def _series(df, col, default=np.nan):
        if col not in df.columns:
            return pd.Series(np.full(len(df), default), index=df.index)
        obj = df[col]
        if isinstance(obj, pd.DataFrame):
            obj = obj.iloc[:, 0]
        return pd.Series(obj, index=df.index)

    def _numeric_series(df, col, default=np.nan):
        return pd.to_numeric(_series(df, col, default=default), errors="coerce")

    def _set_from_alias(df, canon, aliases, default=np.nan):
        if canon in df.columns:
            df[canon] = _numeric_series(df, canon, default=default)
            return df
        c = _choose_col(df, aliases)
        if c is None:
            df[canon] = default
        else:
            df[canon] = _numeric_series(df, c, default=default)
        return df

    def _sort_by_redshift(df):
        z = _numeric_series(df, "Redshift").to_numpy(dtype=float)
        order = np.argsort(z)[::-1]  # high z on left, low z on right
        return df.iloc[order].reset_index(drop=True)

    def _median_by_redshift(df, ycol):
        if ycol not in df.columns:
            return pd.DataFrame({"Redshift": [], ycol: []})

        z = _numeric_series(df, "Redshift").to_numpy(dtype=float)
        y = _numeric_series(df, ycol).to_numpy(dtype=float)

        sub = pd.DataFrame({
            "Redshift": z,
            ycol: y,
        })

        good = np.isfinite(sub["Redshift"].to_numpy(dtype=float)) & np.isfinite(sub[ycol].to_numpy(dtype=float))
        sub = sub.loc[good].copy()

        if len(sub) == 0:
            return pd.DataFrame({"Redshift": [], ycol: []})

        med = sub.groupby("Redshift", as_index=False).agg({ycol: "median"})
        return _sort_by_redshift(med)

    def _plot_shell_lines(ax, df, ycol, *, ls="-", lw=1.35, alpha=0.45):
        shell_vals = sorted(
            pd.to_numeric(df["shell_plot"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        shell_handles = []
        for i, sh in enumerate(shell_vals):
            ds = df.loc[pd.to_numeric(df["shell_plot"], errors="coerce") == int(sh)].copy()
            ds = _sort_by_redshift(ds)

            z = _numeric_series(ds, "Redshift").to_numpy(dtype=float)
            y = _numeric_series(ds, ycol).to_numpy(dtype=float)
            good = np.isfinite(z) & np.isfinite(y)
            if np.count_nonzero(good) < 2:
                continue

            color = SHELL_COLORS[i % len(SHELL_COLORS)]
            ax.plot(
                z[good],
                y[good],
                color=color,
                lw=lw,
                ls=ls,
                alpha=alpha,
                zorder=2,
            )

            shell_handles.append(
                Line2D([0], [0], color=color, lw=1.8, alpha=0.8, label=f"Shell {sh}")
            )

        return shell_handles

    def _plot_overall_line(ax, df, ycol, *, color, label, ls="-", lw=2.6):
        dmed = _median_by_redshift(df, ycol)
        if len(dmed) == 0:
            return None

        z = _numeric_series(dmed, "Redshift").to_numpy(dtype=float)
        y = _numeric_series(dmed, ycol).to_numpy(dtype=float)
        good = np.isfinite(z) & np.isfinite(y)
        if np.count_nonzero(good) < 2:
            return None

        ax.plot(
            z[good],
            y[good],
            color=color,
            lw=lw,
            ls=ls,
            alpha=0.98,
            label=label,
            zorder=5,
        )
        return Line2D([0], [0], color=color, lw=lw, ls=ls, label=label)

    def _style_axis(ax, *, xlabel="Redshift", ylabel=None, ylim=None, zero_line=False):
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if zero_line:
            ax.axhline(0.0, color="0.45", lw=0.8, ls="--", alpha=0.7)
        ax.grid(True, color=GRID_COLOR, lw=0.6, alpha=0.72)
        ax.set_axisbelow(True)
        ax.tick_params(direction="in", top=True, right=True, length=4, width=0.8)
        ax.invert_xaxis()  # high z on the left

    # ------------------------------------------------------------
    # Collect closure tables
    # ------------------------------------------------------------
    track_tables = {}

    if closure_df is None:
        tabs = []

        # top-level
        for key in ["closure_df", "pi_closure_df", "closure_used", "raw_closure_df"]:
            if key in products and isinstance(products[key], pd.DataFrame) and len(products[key]):
                tabs.append(products[key].copy())

        # method-level
        method_blocks = products.get("methods", {})
        if isinstance(method_blocks, dict):
            for method in methods:
                block = method_blocks.get(method, {})
                if not isinstance(block, dict):
                    continue

                tdf = block.get("track_df", None)
                if isinstance(tdf, pd.DataFrame):
                    track_tables[str(method)] = tdf.copy()

                found = None
                for key in ["closure_df", "pi_closure_df", "cross_time_closure", "raw_closure_df", "closure"]:
                    obj = block.get(key, None)
                    if isinstance(obj, pd.DataFrame) and len(obj):
                        found = obj.copy()
                        break

                if found is not None:
                    if "shell_method" not in found.columns and "ShellMethod" not in found.columns:
                        found["shell_method"] = str(method)
                    tabs.append(found)

        if len(tabs) == 0:
            raise RuntimeError("No non-empty Pi-closure table is available in computed products.")

        closure_df = pd.concat(tabs, ignore_index=True)
    else:
        closure_df = closure_df.copy()

    closure_df = _drop_duplicate_columns(closure_df)

    # ------------------------------------------------------------
    # Standardize closure table
    # ------------------------------------------------------------
    def _standardize_closure_df(df):
        df = _drop_duplicate_columns(df)
        n = len(df)

        # shell method
        c = _choose_col(df, ["shell_method", "ShellMethod", "method"])
        if c is None:
            df["shell_method"] = "unknown"
        else:
            df["shell_method"] = _series(df, c).astype(str)

        # shell index
        c = _choose_col(df, ["shell", "Shell", "shell_id", "ShellID", "ish"])
        if c is None:
            df["shell"] = 0
        else:
            df["shell"] = pd.to_numeric(_series(df, c), errors="coerce").fillna(0).astype(int)

        shell_raw = pd.to_numeric(df["shell"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(shell_raw).any() and int(np.nanmin(shell_raw)) == 0:
            df["shell_plot"] = df["shell"].astype(int) + 1
        else:
            df["shell_plot"] = df["shell"].astype(int)

        # redshift
        cz = _choose_col(df, ["Redshift", "redshift", "z", "z_mid", "Redshift_mid", "z_eff"])
        if cz is not None:
            df["Redshift"] = pd.to_numeric(_series(df, cz), errors="coerce")
        else:
            df["Redshift"] = np.nan

        # aliases for Pi components
        for comp in COMPONENTS:
            _set_from_alias(
                df,
                f"Pi_direct_{comp}",
                [
                    f"Pi_direct_{comp}",
                    f"Pi_direct_late_{comp}",
                    f"Pi_dI_{comp}",
                    f"Pi_measured_{comp}",
                    f"Pi_mea_{comp}",
                    f"Pi_fd_{comp}",
                ],
            )

            _set_from_alias(
                df,
                f"Pi_aff_{comp}",
                [
                    f"Pi_aff_{comp}",
                    f"Pi_aff_late_{comp}",
                    f"Pi_est_{comp}",
                    f"Pi_affine_{comp}",
                    f"Pi_fig_{comp}",
                    f"Pi_model_{comp}",
                    f"Pi_OmegaH_{comp}",
                    f"Pi_Omega_plus_H_{comp}",
                    f"Pi_Omega_H_{comp}",
                ],
            )

            _set_from_alias(
                df,
                f"Pi_Omega_{comp}",
                [
                    f"Pi_Omega_{comp}",
                    f"Pi_omega_{comp}",
                    f"Omega_{comp}",
                    f"Omega_hat_{comp}",
                ],
            )

            _set_from_alias(
                df,
                f"Pi_H_{comp}",
                [
                    f"Pi_H_{comp}",
                    f"Pi_h_{comp}",
                    f"Pi_mathcalH_{comp}",
                    f"H_{comp}",
                    f"H_hat_{comp}",
                ],
            )

            # If Pi_aff is missing but Omega and H are present, construct it
            aff = _numeric_series(df, f"Pi_aff_{comp}").to_numpy(dtype=float)
            Om = _numeric_series(df, f"Pi_Omega_{comp}").to_numpy(dtype=float)
            H  = _numeric_series(df, f"Pi_H_{comp}").to_numpy(dtype=float)
            missing_aff = ~np.isfinite(aff)
            can_make = np.isfinite(Om) & np.isfinite(H)
            aff[missing_aff & can_make] = Om[missing_aff & can_make] + H[missing_aff & can_make]
            df[f"Pi_aff_{comp}"] = aff

        # Derived vector quantities
        Pi_dir = np.vstack([
            _numeric_series(df, "Pi_direct_01").to_numpy(dtype=float),
            _numeric_series(df, "Pi_direct_02").to_numpy(dtype=float),
            _numeric_series(df, "Pi_direct_12").to_numpy(dtype=float),
        ]).T

        Pi_aff = np.vstack([
            _numeric_series(df, "Pi_aff_01").to_numpy(dtype=float),
            _numeric_series(df, "Pi_aff_02").to_numpy(dtype=float),
            _numeric_series(df, "Pi_aff_12").to_numpy(dtype=float),
        ]).T

        Pi_Om = np.vstack([
            _numeric_series(df, "Pi_Omega_01").to_numpy(dtype=float),
            _numeric_series(df, "Pi_Omega_02").to_numpy(dtype=float),
            _numeric_series(df, "Pi_Omega_12").to_numpy(dtype=float),
        ]).T

        Pi_H = np.vstack([
            _numeric_series(df, "Pi_H_01").to_numpy(dtype=float),
            _numeric_series(df, "Pi_H_02").to_numpy(dtype=float),
            _numeric_series(df, "Pi_H_12").to_numpy(dtype=float),
        ]).T

        dir_norm = np.sqrt(np.nansum(Pi_dir**2, axis=1))
        aff_norm = np.sqrt(np.nansum(Pi_aff**2, axis=1))
        Om_norm  = np.sqrt(np.nansum(Pi_Om**2, axis=1))
        H_norm   = np.sqrt(np.nansum(Pi_H**2, axis=1))

        df["Pi_direct_norm"] = dir_norm
        df["Pi_aff_norm"] = aff_norm
        df["Pi_Omega_norm"] = Om_norm
        df["Pi_H_norm"] = H_norm

        denom = Om_norm + H_norm
        fOm = np.full(n, np.nan, dtype=float)
        fH = np.full(n, np.nan, dtype=float)
        good = np.isfinite(denom) & (denom > 0)
        fOm[good] = Om_norm[good] / denom[good]
        fH[good] = H_norm[good] / denom[good]

        df["fOmega_norm"] = fOm
        df["fH_norm"] = fH

        # angle between Omega and H vectors, in degrees
        dot = np.nansum(Pi_Om * Pi_H, axis=1)
        denom_ang = Om_norm * H_norm
        mu = np.full(n, np.nan, dtype=float)
        good = np.isfinite(dot) & np.isfinite(denom_ang) & (denom_ang > 0)
        mu[good] = dot[good] / denom_ang[good]
        mu[good] = np.clip(mu[good], -1.0, 1.0)
        ang = np.full(n, np.nan, dtype=float)
        ang[good] = np.degrees(np.arccos(mu[good]))

        df["angle_Omega_H_deg"] = ang

        return _drop_duplicate_columns(df)

    closure_used = _standardize_closure_df(closure_df)

    if verbose:
        print("[Pi closure] standardized table summary")
        print("  rows:", len(closure_used))
        print("  methods:", sorted(closure_used["shell_method"].astype(str).unique().tolist()))
        print("  shell labels:", sorted(pd.to_numeric(closure_used["shell_plot"], errors="coerce").dropna().astype(int).unique().tolist()))

    # ------------------------------------------------------------
    # Draw one figure for each method
    # ------------------------------------------------------------
    figs_by_method = {}
    save_paths = []

    for method in methods:
        dmethod = closure_used.loc[
            closure_used["shell_method"].astype(str) == str(method)
        ].copy()

        if len(dmethod) == 0:
            print(f"[Pi closure] no rows for method={method}; skipped.")
            continue

        dmethod = _sort_by_redshift(dmethod)

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(16.2, 4.8),
            constrained_layout=True,
        )

        # ---------- Panel 1: ||Pi^dI|| and ||Pi^{Omega+H}|| ----------
        ax = axes[0]

        shell_handles_1 = _plot_shell_lines(ax, dmethod, "Pi_direct_norm", ls="-",  lw=1.35, alpha=0.42)
        _plot_shell_lines(ax, dmethod, "Pi_aff_norm",   ls="--", lw=1.25, alpha=0.36)

        h1 = _plot_overall_line(
            ax, dmethod, "Pi_direct_norm",
            color=COLOR_OVERALL_DIRECT,
            label=r"overall $||\Pi^{\rm dI}||$",
            ls="-", lw=2.6,
        )
        h2 = _plot_overall_line(
            ax, dmethod, "Pi_aff_norm",
            color=COLOR_OVERALL_AFF,
            label=r"overall $||\Pi^{\Omega+\mathcal{H}}||$",
            ls="--", lw=2.6,
        )

        ax.set_title(r"(a) $||\Pi^{\rm dI}||$ and $||\Pi^{\Omega+\mathcal{H}}||$")
        _style_axis(ax, ylabel=r"[Gyr$^{-1}$]")

        quantity_handles_1 = [
            Line2D([0], [0], color="0.40", lw=1.5, ls="-",  alpha=0.55, label=r"shell $||\Pi^{\rm dI}||$"),
            Line2D([0], [0], color="0.40", lw=1.5, ls="--", alpha=0.55, label=r"shell $||\Pi^{\Omega+\mathcal{H}}||$"),
        ]
        if h1 is not None:
            quantity_handles_1.append(h1)
        if h2 is not None:
            quantity_handles_1.append(h2)

        leg_q1 = ax.legend(
            handles=quantity_handles_1,
            loc="upper right",
            frameon=False,
        )
        ax.add_artist(leg_q1)

        if len(shell_handles_1):
            ax.legend(
                handles=shell_handles_1,
                loc="upper left",
                frameon=False,
                ncol=1 if len(shell_handles_1) <= 5 else 2,
            )

        # ---------- Panel 2: fraction evolution ----------
        ax = axes[1]

        shell_handles_2 = _plot_shell_lines(ax, dmethod, "fOmega_norm", ls="-",  lw=1.35, alpha=0.42)
        _plot_shell_lines(ax, dmethod, "fH_norm",     ls="--", lw=1.25, alpha=0.36)

        h3 = _plot_overall_line(
            ax, dmethod, "fOmega_norm",
            color=COLOR_OVERALL_OMEGA,
            label=r"overall $|\Omega|/(|\Omega|+|\mathcal{H}|)$",
            ls="-", lw=2.6,
        )
        h4 = _plot_overall_line(
            ax, dmethod, "fH_norm",
            color=COLOR_OVERALL_H,
            label=r"overall $|\mathcal{H}|/(|\Omega|+|\mathcal{H}|)$",
            ls="--", lw=2.6,
        )

        ax.set_title(r"(b) fractional contribution of $|\Omega|$ and $|\mathcal{H}|$")
        _style_axis(ax, ylabel="fraction", ylim=(-0.03, 1.03))

        quantity_handles_2 = [
            Line2D([0], [0], color="0.40", lw=1.5, ls="-",  alpha=0.55, label=r"shell $f_\Omega$"),
            Line2D([0], [0], color="0.40", lw=1.5, ls="--", alpha=0.55, label=r"shell $f_{\mathcal{H}}$"),
        ]
        if h3 is not None:
            quantity_handles_2.append(h3)
        if h4 is not None:
            quantity_handles_2.append(h4)

        leg_q2 = ax.legend(
            handles=quantity_handles_2,
            loc="upper right",
            frameon=False,
        )
        ax.add_artist(leg_q2)

        if len(shell_handles_2):
            ax.legend(
                handles=shell_handles_2,
                loc="upper left",
                frameon=False,
                ncol=1 if len(shell_handles_2) <= 5 else 2,
            )

        # ---------- Panel 3: angle evolution ----------
        ax = axes[2]

        shell_handles_3 = _plot_shell_lines(ax, dmethod, "angle_Omega_H_deg", ls="-", lw=1.35, alpha=0.45)

        h5 = _plot_overall_line(
            ax, dmethod, "angle_Omega_H_deg",
            color=COLOR_OVERALL_ANGLE,
            label=r"overall angle$(\Omega,\mathcal{H})$",
            ls="-", lw=2.6,
        )

        ax.set_title(r"(c) angle between $\Omega$ and $\mathcal{H}$")
        _style_axis(ax, ylabel="angle [deg]", ylim=(0, 180))

        quantity_handles_3 = [
            Line2D([0], [0], color="0.40", lw=1.5, ls="-", alpha=0.55, label=r"shell angle$(\Omega,\mathcal{H})$"),
        ]
        if h5 is not None:
            quantity_handles_3.append(h5)

        leg_q3 = ax.legend(
            handles=quantity_handles_3,
            loc="upper right",
            frameon=False,
        )
        ax.add_artist(leg_q3)

        if len(shell_handles_3):
            ax.legend(
                handles=shell_handles_3,
                loc="upper left",
                frameon=False,
                ncol=1 if len(shell_handles_3) <= 5 else 2,
            )

        fig.suptitle(
            f"Cross-time $\\Pi$ evolution: {method} shells, tracked subhalo {SID0}",
            y=1.03,
            fontsize=14.5,
        )

        figs_by_method[str(method)] = fig

        if save:
            save_path = OUTDIR / f"pi_closure_cross_time_{method}_evolution.png"
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            save_paths.append(save_path)
            print("Saved:", save_path)

    if show:
        plt.show()
    else:
        for fig in figs_by_method.values():
            plt.close(fig)

    return figs_by_method, closure_used, track_tables, save_paths

# %% code cell 20

# ============================================================
# plot_subhalo_shell_density_summary
# ============================================================
# Two shell-contour panels plus four overlaid density panels.
#
# Requirements:
#   - run the complete _az_* helper cell before this cell.
#   - FoF/environment background uses an independent grey normalisation.
#   - Lower env_norm_percentiles[1] makes FoF structures darker.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_subhalo_shell_density_summary(
    products,
    *,
    shell_methods=("radial", "binding_energy"),
    density_method="radial",
    OUTDIR=None,
    path=None,
    COMMON_LIM_KPC=None,
    PAD_FACTOR=1.18,
    density_bins=400,
    density_percentile=99.0,
    env_alpha=1.0,
    env_cmap="Greys",
    env_norm_percentiles=(0.01, 55.0),
    subhalo_alpha=0.55,
    subhalo_norm_percentiles=(5.0, 99.5),
    density_cmap="magma",
    total_cmap="inferno",
    shell_nsigma=2.0,
    galaxy_major_axis_color="#ff2020",
    galaxy_major_axis_lw=1.45,
    galaxy_major_axis_nsigma=2.2,
    fig_face="#d9d9d9",
    ax_face="#d9d9d9",
    save=True,
    show=True,
    dpi=300,
):
    """
    Plot shell contours and overlaid density panels.

    Figure layout
    -------------
    Top-left:
        radial shell ellipses + galaxy major axis.
    Top-middle:
        binding-energy shell ellipses + galaxy major axis.
    Remaining four panels:
        total, dark matter, stellar, and gas density.

    Density convention
    ------------------
    FoF/environment:
        grey-scale background with independent normalisation.
    Tracked subhalo:
        coloured foreground with independent normalisation.
    """
    OUTDIR = _az_resolve_outdir(OUTDIR=OUTDIR, path=path)

    SID0 = int(products.get("SID0", products.get("sid", -1)))
    SIM_NAME = products.get("SIM_NAME", globals().get("SIM_NAME", "TNG"))

    method_blocks = products.get("methods", None)
    if not isinstance(method_blocks, dict):
        raise KeyError("products['methods'] is missing or invalid.")

    shell_methods = tuple(str(m) for m in shell_methods)
    density_method = str(density_method)

    for method in shell_methods:
        if method not in method_blocks:
            raise KeyError(f"shell method {method!r} not found in products['methods'].")

    if density_method not in method_blocks:
        density_method = shell_methods[0]

    density_records = list(method_blocks[density_method].get("records", []))
    if len(density_records) == 0:
        raise RuntimeError(f"No records found for density_method={density_method!r}.")

    # ------------------------------------------------------------
    # Common view limits
    # ------------------------------------------------------------
    all_records_for_limits = []
    for method in shell_methods:
        all_records_for_limits.extend(method_blocks[method].get("records", []))

    points = [np.array([0.0, 0.0], dtype=float)]
    for rec in all_records_for_limits:
        if "center_xy" in rec:
            points.append(np.asarray(rec["center_xy"], dtype=float))

    points = np.asarray(points, dtype=float)
    good_points = np.isfinite(points).all(axis=1)

    if np.count_nonzero(good_points):
        view_center = np.nanmean(points[good_points], axis=0)
    else:
        view_center = np.array([0.0, 0.0], dtype=float)

    if COMMON_LIM_KPC is None:
        vals = []
        if np.count_nonzero(good_points):
            vals.append(np.nanmax(np.abs(points[good_points] - view_center[None, :])))
        for rec in density_records:
            for kind in ["total", "dm", "star", "gas"]:
                xy, _ = _az_get_component_xy_w(rec, kind)
                if len(xy):
                    vals.append(np.nanpercentile(np.abs(xy - view_center[None, :]), density_percentile))
        half_width = float(np.nanmax(vals) * PAD_FACTOR) if len(vals) else 100.0
        if not np.isfinite(half_width) or half_width <= 0:
            half_width = 100.0
    else:
        half_width = float(COMMON_LIM_KPC)

    xlim = (view_center[0] - half_width, view_center[0] + half_width)
    ylim = (view_center[1] - half_width, view_center[1] + half_width)

    # Use the lowest-redshift record as the z=0-like FoF/environment background.
    zvals = np.array([_az_record_redshift(rec) for rec in density_records], dtype=float)
    if np.isfinite(zvals).any():
        rec_env = density_records[int(np.nanargmin(zvals))]
    else:
        rec_env = density_records[0]

    env_qlo, env_qhi = env_norm_percentiles
    sub_qlo, sub_qhi = subhalo_norm_percentiles

    # ------------------------------------------------------------
    # Draw figure
    # ------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 9.5), facecolor=fig_face, constrained_layout=True)
    axes = np.asarray(axes)

    shell_base_colours = {"radial": "#1b7837", "binding_energy": "#762a83"}

    # ------------------------------------------------------------
    # Shell panels
    # ------------------------------------------------------------
    for iax, method in enumerate(shell_methods[:2]):
        ax = axes[0, iax]
        ax.set_facecolor(ax_face)

        records = list(method_blocks[method].get("records", []))
        base_colour = shell_base_colours.get(method, "#2166ac")

        nshell = max([len(rec.get("masks", [])) for rec in records] + [1])
        shell_colours = _az_shell_palette(nshell, base_colour)

        for rec in records:
            xy = _az_as_xy(rec.get("xy_global", []))
            masses = np.asarray(rec.get("masses", np.ones(len(xy))), dtype=float)
            masks = rec.get("masks", [])

            for ish, mask in enumerate(masks):
                mask = np.asarray(mask, dtype=bool)
                if len(mask) != len(xy) or np.count_nonzero(mask) < 5:
                    continue
                weights = masses[mask] if len(masses) == len(xy) else None
                ell = _az_ellipse_from_xy(xy[mask], w=weights, nsigma=shell_nsigma)
                if ell is None:
                    continue
                cen, width, height, angle = ell
                ax.add_patch(
                    Ellipse(
                        cen, width, height, angle=angle, fill=False, lw=1.0,
                        color=shell_colours[min(ish, len(shell_colours) - 1)], alpha=0.82, zorder=20 + ish,
                    )
                )

            _az_draw_major_axis(
                ax, rec, xlim,
                color=galaxy_major_axis_color,
                lw=galaxy_major_axis_lw,
                nsigma=galaxy_major_axis_nsigma,
            )

        ax.set_title(f"{method} shell contours + galaxy major axis")
        _az_apply_image_axis_style(ax, xlim, ylim, scalebar=True, scalebar_color="black", scalebar_fontsize=10.5)

    # ------------------------------------------------------------
    # Density panels
    # ------------------------------------------------------------
    density_specs = [
        ("total", "total density", total_cmap),
        ("dm", "dark matter density", density_cmap),
        ("star", "stellar density", density_cmap),
        ("gas", "gas density", density_cmap),
    ]
    density_axes = [axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]

    for ax, (kind, title, cmap) in zip(density_axes, density_specs):
        ax.set_facecolor(ax_face)

        # FoF/environment background: independent grey standard.
        env_xy, env_w = _az_get_env_xy_w(rec_env, kind)
        if len(env_xy):
            env_vmin, env_vmax = _az_hist_log_range_same_limits(
                [(env_xy, env_w)], xlim, ylim, density_bins, qlo=env_qlo, qhi=env_qhi,
            )
            _az_draw_hist(
                ax, env_xy, env_w, xlim, ylim,
                bins=density_bins, cmap=env_cmap, alpha=env_alpha,
                vmin=env_vmin, vmax=env_vmax, zorder=1,
            )

        # Tracked-subhalo foreground: separate coloured standard.
        sub_xy_w = [_az_get_component_xy_w(rec, kind) for rec in density_records]
        sub_vmin, sub_vmax = _az_hist_log_range_same_limits(
            sub_xy_w, xlim, ylim, density_bins, qlo=sub_qlo, qhi=sub_qhi,
        )

        has_any = False
        for xy, w in sub_xy_w:
            if len(xy):
                _az_draw_hist(
                    ax, xy, w, xlim, ylim,
                    bins=density_bins, cmap=cmap, alpha=subhalo_alpha,
                    vmin=sub_vmin, vmax=sub_vmax, zorder=5,
                )
                has_any = True

        if not has_any:
            ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center", fontsize=12, color="black")

        ax.set_title(title)
        _az_apply_image_axis_style(ax, xlim, ylim, scalebar=True, scalebar_color="black", scalebar_fontsize=10.5)

    fig.suptitle(
        f"{SIM_NAME}: tracked subhalo {SID0}, separated shell and density views",
        fontsize=15, y=1.02,
    )

    save_path = None
    if save:
        save_path = OUTDIR / f"subhalo_{SID0}_shell_density_summary.png"
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        print("Saved:", save_path)

    if show:
        plt.show()

    return fig, axes, save_path

# %% code cell 21

# ============================================================
# plot_subhalo_density_time_sequence
# ============================================================
# Four rows:
#   total, dark matter, stars, gas
#
# Columns:
#   snapshots ordered by redshift.
#
# Requirements:
#   - run the complete _az_* helper cell before this cell.
#   - FoF/environment background uses an independent grey normalisation.
#   - all panels have the same physical size and same scale-bar length.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt


def plot_subhalo_density_time_sequence(
    products,
    *,
    method="radial",
    OUTDIR=None,
    path=None,
    time_order="early_to_late",
    COMMON_HALF_WIDTH_KPC=None,
    zoom_factor=2.8,
    min_half_width_kpc=25.0,
    density_bins=400,
    density_percentile=99.0,
    env_alpha=1.0,
    env_cmap="Greys",
    env_norm_percentiles=(0.01, 55.0),
    subhalo_alpha=0.55,
    subhalo_norm_percentiles=(5.0, 99.5),
    density_cmap="magma",
    total_cmap="inferno",
    fig_face="#d9d9d9",
    ax_face="#d9d9d9",
    save=True,
    show=True,
    dpi=300,
):
    """
    Plot zoom-in density sequence.

    Rows:
        total, dark matter, stars, gas.

    Columns:
        snapshots in redshift order.

    The FoF/environment background and tracked-subhalo foreground use
    independent colour normalisations.
    """
    OUTDIR = _az_resolve_outdir(OUTDIR=OUTDIR, path=path)

    method = str(method)
    SID0 = int(products.get("SID0", products.get("sid", -1)))
    SIM_NAME = products.get("SIM_NAME", globals().get("SIM_NAME", "TNG"))

    if "methods" not in products or method not in products["methods"]:
        raise KeyError(f"method={method!r} not found in products['methods'].")

    records = list(products["methods"][method].get("records", []))
    if len(records) == 0:
        raise RuntimeError(f"No records found for method={method!r}.")

    records = _az_sort_records(records, time_order=time_order)
    n_snap = len(records)

    density_rows = [
        ("total", "total", total_cmap),
        ("dm", "dark matter", density_cmap),
        ("star", "stars", density_cmap),
        ("gas", "gas", density_cmap),
    ]

    env_qlo, env_qhi = env_norm_percentiles
    sub_qlo, sub_qhi = subhalo_norm_percentiles

    # ------------------------------------------------------------
    # Use one physical half-width for all panels
    # ------------------------------------------------------------
    if COMMON_HALF_WIDTH_KPC is None:
        half_values = [
            _az_half_width_for_record(
                rec,
                zoom_factor=zoom_factor,
                min_half_width_kpc=min_half_width_kpc,
                density_percentile=density_percentile,
            )
            for rec in records
        ]
        common_half_width = max(half_values + [min_half_width_kpc])
    else:
        common_half_width = float(COMMON_HALF_WIDTH_KPC)

    if not np.isfinite(common_half_width) or common_half_width <= 0:
        common_half_width = float(min_half_width_kpc)

    xlim_list = []
    ylim_list = []
    for rec in records:
        xlim, ylim = _az_limits_for_record(rec, common_half_width)
        xlim_list.append(xlim)
        ylim_list.append(ylim)

    fixed_scalebar_length = _az_nice_scalebar_length(2.0 * common_half_width)

    # ------------------------------------------------------------
    # Common normalisation per row
    # ------------------------------------------------------------
    env_ranges = {}
    sub_ranges = {}
    for kind, _, _ in density_rows:
        env_xy_w = [_az_get_env_xy_w(rec, kind) for rec in records]
        sub_xy_w = [_az_get_component_xy_w(rec, kind) for rec in records]
        env_ranges[kind] = _az_hist_log_range_multi_limits(
            env_xy_w, xlim_list, ylim_list, density_bins, qlo=env_qlo, qhi=env_qhi,
        )
        sub_ranges[kind] = _az_hist_log_range_multi_limits(
            sub_xy_w, xlim_list, ylim_list, density_bins, qlo=sub_qlo, qhi=sub_qhi,
        )

    # ------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------
    fig_width = max(12.0, 2.7 * n_snap)
    fig, axes = plt.subplots(
        4, n_snap,
        figsize=(fig_width, 12.0),
        facecolor=fig_face,
        constrained_layout=True,
        squeeze=False,
    )

    for irow, (kind, row_label, cmap) in enumerate(density_rows):
        env_vmin, env_vmax = env_ranges[kind]
        sub_vmin, sub_vmax = sub_ranges[kind]

        for icol, rec in enumerate(records):
            ax = axes[irow, icol]
            ax.set_facecolor(ax_face)

            xlim = xlim_list[icol]
            ylim = ylim_list[icol]

            # FoF/environment background.
            env_xy, env_w = _az_get_env_xy_w(rec, kind)
            if len(env_xy):
                _az_draw_hist(
                    ax, env_xy, env_w, xlim, ylim,
                    bins=density_bins, cmap=env_cmap, alpha=env_alpha,
                    vmin=env_vmin, vmax=env_vmax, zorder=1,
                )

            # Tracked-subhalo foreground.
            sub_xy, sub_w = _az_get_component_xy_w(rec, kind)
            if len(sub_xy):
                _az_draw_hist(
                    ax, sub_xy, sub_w, xlim, ylim,
                    bins=density_bins, cmap=cmap, alpha=subhalo_alpha,
                    vmin=sub_vmin, vmax=sub_vmax, zorder=5,
                )
            else:
                ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center", fontsize=9, color="black")

            # Column title: redshift only.
            if irow == 0:
                z = _az_record_redshift(rec)
                ztxt = f"z={z:.2f}" if np.isfinite(z) else "z=?"
                ax.set_title(ztxt, fontsize=10.5)

            if icol == 0:
                ax.set_ylabel(row_label, fontsize=11)

            _az_apply_image_axis_style(
                ax, xlim, ylim,
                scalebar=True,
                scalebar_color="black",
                fixed_scalebar_length=fixed_scalebar_length,
                scalebar_fontsize=8.5,
            )

            if irow < 3:
                ax.set_xticklabels([])
            if icol > 0:
                ax.set_yticklabels([])

    fig.suptitle(
        f"{SIM_NAME}: tracked subhalo {SID0}, density sequence ({method} records)",
        fontsize=15, y=1.01,
    )

    save_path = None
    if save:
        save_path = OUTDIR / f"subhalo_{SID0}_{method}_density_time_sequence.png"
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        print("Saved:", save_path)

    if show:
        plt.show()

    return fig, axes, save_path

# %% [markdown] cell 22
# ## Figures

# %% code cell 23
# ============================================================
# Orbit-plane overlay plots
# ============================================================

fig_rad, ax_rad, records_rad, table_rad, path_rad = plot_subhalo_orbitplane_overlay(
    PRODUCTS,
    method='radial',
)

fig_be, ax_be, records_be, table_be, path_be = plot_subhalo_orbitplane_overlay(
    PRODUCTS,
    method='binding_energy',
)

# %% code cell 24

# ============================================================
# Shell/density summary and density time-sequence plots
# ============================================================
# FoF/environment contrast is controlled by env_norm_percentiles.
# Lowering the upper percentile makes the FoF high-density structure darker.
# ============================================================

fig_shell_density, axes_shell_density, path_shell_density = plot_subhalo_shell_density_summary(
    PRODUCTS,
    shell_methods=("radial", "binding_energy"),
    density_method="radial",
    density_bins=400,
    env_alpha=1.0,
    env_norm_percentiles=(0.01, 15.0),
    subhalo_alpha=1,
)

fig_density_seq, axes_density_seq, path_density_seq = plot_subhalo_density_time_sequence(
    PRODUCTS,
    method="radial",
    time_order="early_to_late",
    density_bins=400,
    env_alpha=1.0,
    env_norm_percentiles=(0.01, 15.0),
    subhalo_alpha=1,
)

# %% code cell 25

# ============================================================
# Time-evolution plots for mass / axes / alignments
# ============================================================

figs_rad, table_rad, paths_rad = plot_subhalo_orbitplane_table_evolution(
    PRODUCTS,
    method='radial',
)

figs_be, table_be, paths_be = plot_subhalo_orbitplane_table_evolution(
    PRODUCTS,
    method='binding_energy',
)

# %% code cell 26

# ============================================================
# Cross-time Pi-closure plots
# ============================================================

figs_pi, closure_used, track_tables, pi_paths = plot_pi_closure_table_evolution(
    PRODUCTS,
    methods=SHELL_METHODS,
)

# %% code cell 27

# ============================================================
# Tables available after loading products
# ============================================================

table_radial = PRODUCTS['methods'].get('radial', {}).get('table_df', pd.DataFrame())
table_binding = PRODUCTS['methods'].get('binding_energy', {}).get('table_df', pd.DataFrame())
closure_radial = PRODUCTS['methods'].get('radial', {}).get('closure_df', pd.DataFrame())
closure_binding = PRODUCTS['methods'].get('binding_energy', {}).get('closure_df', pd.DataFrame())

print('table_radial:', table_radial.shape)
print('table_binding:', table_binding.shape)
print('closure_radial:', closure_radial.shape)
print('closure_binding:', closure_binding.shape)

display(table_radial.head())
display(closure_radial.head())

# %% code cell 28

# ============================================================
# Explicit cleanup if needed
# ============================================================
# Normally not required when delete_cache=False and cache files should persist.
# Use only if you intentionally want to close open catalog handles.
# ============================================================

# hd_tng.cleanup_open_catalogs()
