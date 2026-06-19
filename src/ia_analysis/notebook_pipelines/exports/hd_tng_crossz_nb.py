"""Exported code from notebooks/raw_20260618/hd_tng_crossZ.ipynb.

This file is generated for project management and refactoring. Review before running end to end.
"""


# %% code cell 1

from pathlib import Path
import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DWE import DimrothWatson


from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

dw = DimrothWatson()


import importlib


from TNGCatLoader import TNGCatalog
import hd_tng
import arts
print('hd_tng loaded from:', hd_tng.__file__)
print('arts loaded from:', arts.__file__)

# %% code cell 2
import halo_dynamics
import hd_tng
import shape
import Iana
importlib.reload(halo_dynamics)
importlib.reload(hd_tng)
importlib.reload(shape)
importlib.reload(Iana)

# %% [markdown] cell 3
# ## Configuration

# %% code cell 4

SIM_NAME = os.environ.get('TNG_SIM_NAME', 'TNG300-1')
BASE_PATH = os.environ.get('TNG_BASE_PATH', '/cosma7/data/dp203/dc-wang17/TNG/tng_data')
SNAP = int(os.environ.get('TNG_SNAP', '99'))
API_KEY = os.environ.get('TNG_API_KEY','ec7a0419719cacfd0a27d964d8993b9d')

OUTDIR = Path('hd_tng_outputs')
OUTDIR.mkdir(exist_ok=True)

CFG = dict(
    sim_name=SIM_NAME,
    snap=SNAP,
    api_key=API_KEY,
    download_if_missing=True,
    # Important: cache_dir=None means a private system-temp directory, normally /tmp.
    # delete_cache=True deletes downloaded API files after compute_haloes returns.
    cache_dir=None,
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
    auto_select_targets= True
)

print('SIM_NAME =', SIM_NAME)
print('BASE_PATH =', BASE_PATH)
print('SNAP =', SNAP)
print('Temporary API cache: system temp directory; delete after use =', CFG['delete_cache'])

# %% code cell 5

# run = hd_tng.compute_haloes(
#     BASE_PATH,
#     SNAP,
#     cfg=CFG,
# )

# %% code cell 6
# print('Number of successful subhaloes:', len(run['results']))
# print('Number of failures:', len(run['failures']))
# display(run['target_table'])
# if len(run['failures']):
#     display(run['failures'])

# closure_all = run['closure_all']
# closure_all.to_csv(OUTDIR / 'hd_tng_instantaneous_pi_closure.csv', index=False)
# print('closure rows:', len(closure_all))
# display(closure_all.head())

# %% [markdown] cell 7
# ## Run the packaged calculation

# %% [markdown] cell 8
# ## Optional cross-time pattern-speed test with SubLink

# %% code cell 9
RUN_CROSS_TIME = True
SNAP_TRACK = [99,  84,  72, 67,59,50,40,33]

# %% code cell 10
# if RUN_CROSS_TIME:
#     cross_df, track_df = hd_tng.cross_time_pattern_speed_for_subhalo(
#         BASE_PATH,
#         SNAP,
#         SID,
#         snap_track=SNAP_TRACK,
#         cfg=CFG,
#         shell_method='radial',
#     )

# %% code cell 11
# cross_df.to_csv(OUTDIR / f'subhalo_{SID}_cross_time_pattern_speed.csv', index=False)
# display(track_df)
# display(cross_df.head())

# %% code cell 12
# ============================================================
# plot_subhalo_orbitplane_overlay
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


def plot_subhalo_orbitplane_overlay(
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

    DENSITY_BINS_PER_SUBHALO=150,
    DENSITY_ALPHA=0.72,
    DENSITY_PERCENTILE_LIMIT=99.0,
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
    Plot one tracked subhalo at multiple snapshots in one fitted orbit plane.

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

# %% code cell 13
# ============================================================
# plot_subhalo_orbitplane_table_evolution (redefined)
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
    SID0,
    SNAP_TRACK,
    *,
    SNAP0=99,
    table_df=None,
    OUTDIR=None,
    x_key="Redshift",
    invert_redshift_axis=True,
    save=True,
    show=True,
    dpi=300,
    **overlay_kwargs,
):
    """
    Publication-style time-evolution plots from the diagnostic table returned by
    plot_subhalo_orbitplane_overlay.

    Produces two figures:

    1. Alignment figure:
       - global galaxy-DM-halo axis alignment;
       - galaxy axes vs host-centric radial direction;
       - global DM halo axes vs host-centric radial direction;
       - galaxy global axes vs shell axes;
       - shell axes vs host-centric radial direction.

    2. Physical quantity figure:
       - mass components;
       - baryon-to-DM ratio;
       - host-centric distance;
       - particle counts;
       - galaxy stellar axis lengths;
       - global DM and shell axis lengths.

    Notes
    -----
    - Shell labels in the plots are displayed from 1.
    - In the host-centric distance panel, the x-axis is explicitly synchronized
      so that it runs in the same direction as all other panels
      (high redshift on the left, low redshift on the right).

    Returns
    -------
    figs : list
        [fig_align, fig_phys]
    table_df : pandas.DataFrame
    save_paths : list[pathlib.Path]
    """

    if OUTDIR is None:
        OUTDIR = Path(globals().get("OUTDIR", "hd_tng_outputs"))
    else:
        OUTDIR = Path(OUTDIR)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    SID0 = int(SID0)
    SNAP_TRACK = [int(s) for s in SNAP_TRACK]

    # ------------------------------------------------------------
    # 1. Build or reuse table
    # ------------------------------------------------------------
    if table_df is None:
        if "plot_subhalo_orbitplane_overlay" not in globals():
            raise NameError(
                "plot_subhalo_orbitplane_overlay is not defined. "
                "Run the overlay-function cell first."
            )

        compute_kwargs = dict(overlay_kwargs)
        compute_kwargs.update(
            dict(
                SNAP0=SNAP0,
                make_figure=False,
                display_table=False,
                save=False,
                show=False,
            )
        )

        _, _, _, table_df, _ = plot_subhalo_orbitplane_overlay(
            SID0,
            SNAP_TRACK,
            **compute_kwargs,
        )
    else:
        table_df = table_df.copy()

    if not isinstance(table_df, pd.DataFrame) or len(table_df) == 0:
        raise RuntimeError("The diagnostic table is empty or invalid.")

    if x_key not in table_df.columns:
        raise KeyError(f"x_key={x_key!r} not found in table_df columns.")

    # ------------------------------------------------------------
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

# %% code cell 14
# ============================================================
# Robust plot_pi_closure_table_evolution
# ============================================================
# Supports:
#   plot_pi_closure_table_evolution(SID0, SNAP_TRACK, ...)
#   plot_pi_closure_table_evolution(SID0, SNAP0, SNAP_TRACK, ...)
#
# It internally calls hd_tng.cross_time_pattern_speed_for_subhalo unless
# closure_df is provided.
# ============================================================

import inspect
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path


def plot_pi_closure_table_evolution(
    SID0,
    *args,
    SNAP0=99,
    BASE_PATH=None,
    SIM_NAME=None,
    API_KEY=None,
    CFG=None,
    OUTDIR=None,
    TNG_CATALOG_KWARGS=None,
    methods=("radial", "binding_energy"),
    closure_df=None,
    x_key="Redshift",
    invert_redshift_axis=True,
    save=True,
    show=True,
    dpi=300,
    verbose=True,
):
    """
    Cross-time Pi-closure calculation and plotting.

    Accepted calls
    --------------
    plot_pi_closure_table_evolution(SID0, SNAP_TRACK, ...)
    plot_pi_closure_table_evolution(SID0, SNAP0, SNAP_TRACK, ...)

    Returns
    -------
    figs_by_method : dict
    closure_used : pandas.DataFrame
    track_tables : dict
    save_paths : list[pathlib.Path]
    """

    # ------------------------------------------------------------
    # Parse flexible positional arguments
    # ------------------------------------------------------------
    if len(args) == 1:
        SNAP_TRACK = args[0]
    elif len(args) == 2:
        SNAP0 = int(args[0])
        SNAP_TRACK = args[1]
    else:
        raise TypeError(
            "Use either plot_pi_closure_table_evolution(SID0, SNAP_TRACK, ...) "
            "or plot_pi_closure_table_evolution(SID0, SNAP0, SNAP_TRACK, ...)."
        )

    # ------------------------------------------------------------
    # Resolve globals
    # ------------------------------------------------------------
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
    SNAP_TRACK = [int(s) for s in SNAP_TRACK]

    if TNG_CATALOG_KWARGS is None:
        TNG_CATALOG_KWARGS = globals().get("TNG_CATALOG_KWARGS", None)

    if "hd_tng" not in globals():
        raise NameError("hd_tng is not imported.")

    if not hasattr(hd_tng, "cross_time_pattern_speed_for_subhalo"):
        raise AttributeError(
            "hd_tng.cross_time_pattern_speed_for_subhalo is missing. "
            "Reload the hd_tng version that contains the cross-time Pi calculation."
        )

    # ------------------------------------------------------------
    # Style
    # ------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11.2,
        "xtick.labelsize": 9.4,
        "ytick.labelsize": 9.4,
        "legend.fontsize": 8.8,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "figure.dpi": 150,
        "savefig.dpi": dpi,
    })

    COMPONENTS = ["01", "02", "12"]
    COMP_LABEL = {"01": "0,1", "02": "0,2", "12": "1,2"}

    COLOR_DIRECT = "#0072B2"
    COLOR_AFF    = "#D55E00"
    COLOR_RESID  = "#CC6677"
    COLOR_OMEGA  = "#332288"
    COLOR_H      = "#E69F00"
    GRID_COLOR   = "0.86"
    CMAP_NAME    = "viridis_r"

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------
    def _choose_col(df, names):
        for name in names:
            if name in df.columns:
                return name
        return None

    def _read_redshift_map():
        zmap = {}
        for snap in sorted(set(SNAP_TRACK + [SNAP0])):
            try:
                header = hd_tng.read_header_for_snap(
                    BASE_PATH,
                    int(snap),
                    sim_name=SIM_NAME,
                    api_key=API_KEY,
                )
                zmap[int(snap)] = float(header.get("Redshift", np.nan))
            except Exception:
                zmap[int(snap)] = np.nan
        return zmap

    zmap = _read_redshift_map()

    def _call_cross_time_for_method(method):
        """
        Try several calling conventions, and expose the real failure.
        """
        func = hd_tng.cross_time_pattern_speed_for_subhalo

        attempts = []

        base_kwargs = dict(
            snap_track=SNAP_TRACK,
            cfg=CFG,
            shell_method=str(method),
        )
        if TNG_CATALOG_KWARGS is not None:
            base_kwargs["tng_catalog_kwargs"] = TNG_CATALOG_KWARGS

        # New-style keyword call.
        attempts.append((
            "keyword snap_track/cfg/shell_method",
            (BASE_PATH, SNAP0, SID0),
            dict(base_kwargs),
        ))

        # Same but without tng_catalog_kwargs.
        kw2 = dict(base_kwargs)
        kw2.pop("tng_catalog_kwargs", None)
        attempts.append((
            "keyword without tng_catalog_kwargs",
            (BASE_PATH, SNAP0, SID0),
            kw2,
        ))

        # Positional snap_track, cfg.
        kw3 = dict(shell_method=str(method))
        if TNG_CATALOG_KWARGS is not None:
            kw3["tng_catalog_kwargs"] = TNG_CATALOG_KWARGS
        attempts.append((
            "positional snap_track/cfg + shell_method",
            (BASE_PATH, SNAP0, SID0, SNAP_TRACK, CFG),
            kw3,
        ))

        # Positional without tng_catalog_kwargs.
        attempts.append((
            "positional snap_track/cfg no tng_catalog_kwargs",
            (BASE_PATH, SNAP0, SID0, SNAP_TRACK, CFG),
            dict(shell_method=str(method)),
        ))

        # Some older versions may only accept radial implicitly.
        if str(method) == "radial":
            attempts.append((
                "minimal old-style radial fallback",
                (BASE_PATH, SNAP0, SID0, SNAP_TRACK, CFG),
                {},
            ))

        errors = []

        for label, a, kw in attempts:
            try:
                if verbose:
                    print(f"[Pi closure] trying {label}: method={method}")
                out = func(*a, **kw)

                if not isinstance(out, tuple) or len(out) < 2:
                    raise RuntimeError(
                        f"Unexpected return from cross_time_pattern_speed_for_subhalo: {type(out)}"
                    )

                cross_df, track_df = out[0], out[1]

                if not isinstance(cross_df, pd.DataFrame):
                    cross_df = pd.DataFrame(cross_df)
                if not isinstance(track_df, pd.DataFrame):
                    track_df = pd.DataFrame(track_df)

                cross_df = cross_df.copy()
                track_df = track_df.copy()

                if "shell_method" not in cross_df.columns:
                    cross_df["shell_method"] = str(method)

                if verbose:
                    print(
                        f"[Pi closure] success: method={method}, "
                        f"rows={len(cross_df)}, columns={list(cross_df.columns)[:12]}"
                    )

                return cross_df, track_df

            except Exception as exc:
                tb = traceback.format_exc(limit=3)
                errors.append((label, repr(exc), tb))

        print("\n[Pi closure] all call attempts failed for method =", method)
        for label, err, tb in errors:
            print(f"\n--- attempt: {label} ---")
            print(err)
            print(tb)

        raise RuntimeError(
            f"cross_time_pattern_speed_for_subhalo failed for method={method}. "
            "See printed attempts above for the real error."
        )

    def _compute_cross_time_closure():
        all_tabs = []
        track_tables = {}

        for method in methods:
            print(f"\n[Pi closure] computing cross-time table: method={method}")

            try:
                tab, track = _call_cross_time_for_method(method)
            except Exception as exc:
                print(f"[Pi closure] method={method} failed and will be skipped.")
                print("Reason:", repr(exc))
                continue

            if len(tab) == 0:
                print(f"[Pi closure] method={method} returned an EMPTY table.")
                if len(track):
                    print("[Pi closure] track table preview:")
                    try:
                        display(track.head())
                    except Exception:
                        print(track.head())
                continue

            track_tables[str(method)] = track
            all_tabs.append(tab)

        if len(all_tabs) == 0:
            raise RuntimeError(
                "No cross-time Pi-closure table was produced. "
                "The printed messages above should now show the real reason."
            )

        return pd.concat(all_tabs, ignore_index=True), track_tables

    def _standardize_closure_df(df):
        df = df.copy()
        n = len(df)

        # Method
        c = _choose_col(df, ["shell_method", "ShellMethod", "method"])
        if c is None:
            df["shell_method"] = "unknown"
        elif c != "shell_method":
            df["shell_method"] = df[c].astype(str)
        else:
            df["shell_method"] = df["shell_method"].astype(str)

        # Shell
        c = _choose_col(df, ["shell", "Shell", "shell_id", "ShellID", "ish"])
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

        # Snapshots
        snap_a_col = _choose_col(df, [
            "SnapA", "snapA", "snap_a", "snap0", "Snap0",
            "snap_i", "snap_prev", "snap_start", "snap_lo"
        ])
        snap_b_col = _choose_col(df, [
            "SnapB", "snapB", "snap_b", "snap1", "Snap1",
            "snap_j", "snap_next", "snap_end", "snap_hi"
        ])
        snap_single_col = _choose_col(df, ["SnapNum", "snap", "Snapshot"])

        if snap_a_col is not None:
            df["SnapA"] = pd.to_numeric(df[snap_a_col], errors="coerce").astype("Int64")
        elif snap_single_col is not None:
            df["SnapA"] = pd.to_numeric(df[snap_single_col], errors="coerce").astype("Int64")
        else:
            df["SnapA"] = pd.Series([pd.NA] * n, dtype="Int64")

        if snap_b_col is not None:
            df["SnapB"] = pd.to_numeric(df[snap_b_col], errors="coerce").astype("Int64")
        else:
            df["SnapB"] = pd.Series([pd.NA] * n, dtype="Int64")

        # Redshift
        red_col = _choose_col(df, [
            "Redshift", "redshift", "z",
            "z_mid", "Redshift_mid", "z_eff", "z_pair"
        ])

        if red_col is not None:
            df["Redshift"] = pd.to_numeric(df[red_col], errors="coerce")
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

        if "SubhaloID" not in df.columns:
            c = _choose_col(df, ["TargetSubfindID", "SubfindID", "sid", "SID"])
            if c is not None:
                df["SubhaloID"] = pd.to_numeric(df[c], errors="coerce")
            else:
                df["SubhaloID"] = SID0

        # Pi aliases
        aliases = {}
        for comp in COMPONENTS:
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

            c = _choose_col(df, names)
            if c is not None:
                df[canon] = pd.to_numeric(df[c], errors="coerce")
            else:
                df[canon] = np.nan

        # Derived quantities
        for comp in COMPONENTS:
            direct = pd.to_numeric(df[f"Pi_direct_{comp}"], errors="coerce").to_numpy(dtype=float)
            aff    = pd.to_numeric(df[f"Pi_aff_{comp}"], errors="coerce").to_numpy(dtype=float)
            Om     = pd.to_numeric(df[f"Pi_Omega_{comp}"], errors="coerce").to_numpy(dtype=float)
            H      = pd.to_numeric(df[f"Pi_H_{comp}"], errors="coerce").to_numpy(dtype=float)

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

        if x_key not in df.columns:
            raise KeyError(f"x_key={x_key!r} is not available after standardization.")

        return df

    def _sort_df(df):
        df = df.copy()
        x = pd.to_numeric(df[x_key], errors="coerce").to_numpy(dtype=float)
        if x_key == "Redshift":
            order = np.argsort(x)[::-1] if invert_redshift_axis else np.argsort(x)
        else:
            order = np.argsort(x)
        return df.iloc[order].reset_index(drop=True)

    def _median_by_x(df, ycol):
        if ycol not in df.columns:
            return pd.DataFrame(columns=[x_key, ycol, "Redshift"])

        sub = df[[x_key, "Redshift", ycol]].copy()
        sub[x_key] = pd.to_numeric(sub[x_key], errors="coerce")
        sub["Redshift"] = pd.to_numeric(sub["Redshift"], errors="coerce")
        sub[ycol] = pd.to_numeric(sub[ycol], errors="coerce")
        sub = sub[np.isfinite(sub[x_key]) & np.isfinite(sub[ycol])]
        if len(sub) == 0:
            return pd.DataFrame(columns=[x_key, ycol, "Redshift"])

        med = sub.groupby(x_key, as_index=False).agg({
            ycol: "median",
            "Redshift": "median",
        })
        return _sort_df(med)

    def _apply_axis_style(ax, *, ylabel=None, xlabel=None, ylim=None, zero_line=False):
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylim is not None:
            ax.set_ylim(*ylim)

        if zero_line:
            ax.axhline(0.0, color="0.35", lw=0.85, ls="--", alpha=0.68)

        ax.tick_params(direction="in", top=True, right=True, length=4.0, width=0.8)
        ax.grid(True, color=GRID_COLOR, lw=0.6, alpha=0.72)
        ax.set_axisbelow(True)

        if x_key == "Redshift" and invert_redshift_axis:
            ax.invert_xaxis()

    def _scatter_only(ax, df, ycol, *, marker, norm, alpha=0.50, size=22):
        if ycol not in df.columns:
            return False
        x = pd.to_numeric(df[x_key], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
        z = pd.to_numeric(df["Redshift"], errors="coerce").to_numpy(dtype=float)
        good = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if np.count_nonzero(good) == 0:
            return False

        ax.scatter(
            x[good],
            y[good],
            c=z[good],
            cmap=CMAP_NAME,
            norm=norm,
            s=size,
            marker=marker,
            edgecolors="white",
            linewidths=0.35,
            alpha=alpha,
            rasterized=True,
            zorder=3,
        )
        return True

    def _plot_line_with_redshift_points(ax, df, ycol, *, color, marker, label, norm, lw=1.75):
        if ycol not in df.columns or len(df) == 0:
            return False

        sub = df[[x_key, "Redshift", ycol]].copy()
        sub[x_key] = pd.to_numeric(sub[x_key], errors="coerce")
        sub["Redshift"] = pd.to_numeric(sub["Redshift"], errors="coerce")
        sub[ycol] = pd.to_numeric(sub[ycol], errors="coerce")
        sub = sub[np.isfinite(sub[x_key]) & np.isfinite(sub[ycol]) & np.isfinite(sub["Redshift"])]
        if len(sub) == 0:
            return False

        sub = _sort_df(sub)
        ax.plot(
            sub[x_key].to_numpy(dtype=float),
            sub[ycol].to_numpy(dtype=float),
            color=color,
            lw=lw,
            label=label,
            zorder=2,
        )
        ax.scatter(
            sub[x_key].to_numpy(dtype=float),
            sub[ycol].to_numpy(dtype=float),
            c=sub["Redshift"].to_numpy(dtype=float),
            cmap=CMAP_NAME,
            norm=norm,
            s=30,
            marker=marker,
            edgecolors="white",
            linewidths=0.45,
            alpha=0.96,
            zorder=4,
        )
        return True

    def _draw_one_figure(df_in, *, method_label, shell_label=None, norm=None):
        df = _sort_df(df_in)
        aggregate = shell_label is None

        fig, axes = plt.subplots(
            3, 3,
            figsize=(14.8, 10.8),
            constrained_layout=True,
        )

        for j, comp in enumerate(COMPONENTS):
            cdir = f"Pi_direct_{comp}"
            caff = f"Pi_aff_{comp}"
            cres = f"Pi_resid_{comp}"
            cfO  = f"fOmega_abs_{comp}"
            cfH  = f"fH_abs_{comp}"

            # Row 1
            ax = axes[0, j]
            if aggregate:
                _scatter_only(ax, df, cdir, marker="o", norm=norm, alpha=0.42)
                _scatter_only(ax, df, caff, marker="s", norm=norm, alpha=0.42)
                _plot_line_with_redshift_points(
                    ax, _median_by_x(df, cdir), cdir,
                    color=COLOR_DIRECT, marker="o", label=r"$\Pi^{\rm dI}$ median", norm=norm, lw=1.9
                )
                _plot_line_with_redshift_points(
                    ax, _median_by_x(df, caff), caff,
                    color=COLOR_AFF, marker="s", label=r"$\Pi^{\Omega+\mathcal{H}}$ median", norm=norm, lw=1.9
                )
            else:
                _plot_line_with_redshift_points(
                    ax, df, cdir,
                    color=COLOR_DIRECT, marker="o", label=r"$\Pi^{\rm dI}$", norm=norm
                )
                _plot_line_with_redshift_points(
                    ax, df, caff,
                    color=COLOR_AFF, marker="s", label=r"$\Pi^{\Omega+\mathcal{H}}$", norm=norm
                )

            ax.set_title(rf"({chr(ord('a') + j)}) component $({COMP_LABEL[comp]})$")
            _apply_axis_style(
                ax,
                ylabel=rf"$\Pi_{{{COMP_LABEL[comp]}}}$ [Gyr$^{{-1}}$]",
                xlabel=x_key,
                zero_line=True,
            )

            # Row 2
            ax = axes[1, j]
            if aggregate:
                _scatter_only(ax, df, cres, marker="D", norm=norm, alpha=0.45)
                _plot_line_with_redshift_points(
                    ax, _median_by_x(df, cres), cres,
                    color=COLOR_RESID, marker="D", label="median residual", norm=norm, lw=1.9
                )
            else:
                _plot_line_with_redshift_points(
                    ax, df, cres,
                    color=COLOR_RESID, marker="D",
                    label=r"$\Pi^{\rm dI}-\Pi^{\Omega+\mathcal{H}}$",
                    norm=norm,
                )

            ax.set_title(
                rf"({chr(ord('d') + j)}) residual "
                rf"$\Pi^{{\rm dI}}_{{{COMP_LABEL[comp]}}}"
                rf"-\Pi^{{\Omega+\mathcal{{H}}}}_{{{COMP_LABEL[comp]}}}$"
            )
            _apply_axis_style(
                ax,
                ylabel=rf"residual$_{{{COMP_LABEL[comp]}}}$ [Gyr$^{{-1}}$]",
                xlabel=x_key,
                zero_line=True,
            )

            # Row 3
            ax = axes[2, j]
            if aggregate:
                _scatter_only(ax, df, cfO, marker="o", norm=norm, alpha=0.40)
                _scatter_only(ax, df, cfH, marker="s", norm=norm, alpha=0.40)
                _plot_line_with_redshift_points(
                    ax, _median_by_x(df, cfO), cfO,
                    color=COLOR_OMEGA, marker="o", label=r"$f_\Omega$ median", norm=norm, lw=1.9
                )
                _plot_line_with_redshift_points(
                    ax, _median_by_x(df, cfH), cfH,
                    color=COLOR_H, marker="s", label=r"$f_\mathcal{H}$ median", norm=norm, lw=1.9
                )
            else:
                _plot_line_with_redshift_points(
                    ax, df, cfO,
                    color=COLOR_OMEGA, marker="o", label=r"$f_\Omega$", norm=norm
                )
                _plot_line_with_redshift_points(
                    ax, df, cfH,
                    color=COLOR_H, marker="s", label=r"$f_\mathcal{H}$", norm=norm
                )

            ax.set_title(rf"({chr(ord('g') + j)}) fractional contribution $({COMP_LABEL[comp]})$")
            _apply_axis_style(
                ax,
                ylabel="fraction",
                xlabel=x_key,
                ylim=(-0.03, 1.03),
            )

        handles = [
            Line2D([0], [0], color=COLOR_DIRECT, marker="o", lw=1.8, label=r"$\Pi^{\rm dI}$"),
            Line2D([0], [0], color=COLOR_AFF, marker="s", lw=1.8, label=r"$\Pi^{\Omega+\mathcal{H}}$"),
            Line2D([0], [0], color=COLOR_RESID, marker="D", lw=1.8, label="residual"),
            Line2D([0], [0], color=COLOR_OMEGA, marker="o", lw=1.8, label=r"$f_\Omega$"),
            Line2D([0], [0], color=COLOR_H, marker="s", lw=1.8, label=r"$f_\mathcal{H}$"),
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.50, 1.018),
            ncol=5,
            frameon=False,
            columnspacing=1.15,
            handlelength=1.8,
        )

        sm = ScalarMappable(norm=norm, cmap=CMAP_NAME)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, fraction=0.022, pad=0.015, aspect=35)
        cbar.set_label("Redshift")

        if shell_label is None:
            fig.suptitle(
                rf"Cross-time $\Pi$-closure evolution: {method_label} shells, all shells combined",
                y=1.045,
                fontsize=15,
            )
        else:
            fig.suptitle(
                rf"Cross-time $\Pi$-closure evolution: {method_label} shells, Shell {shell_label}",
                y=1.045,
                fontsize=15,
            )

        return fig

    # ------------------------------------------------------------
    # Compute or use provided closure table
    # ------------------------------------------------------------
    track_tables = {}

    if closure_df is None:
        raw_closure, track_tables = _compute_cross_time_closure()
    else:
        raw_closure = closure_df.copy()

    closure_used = _standardize_closure_df(raw_closure)

    print("\n[Pi closure] standardized table summary")
    print("  rows:", len(closure_used))
    print("  methods:", sorted(closure_used["shell_method"].astype(str).unique().tolist()))
    print("  shell labels:", sorted(closure_used["shell_plot"].dropna().unique().astype(int).tolist()))
    print("  columns:", list(closure_used.columns))

    for comp in COMPONENTS:
        print(f"\n  component {comp}:")
        for c in [
            f"Pi_direct_{comp}", f"Pi_aff_{comp}",
            f"Pi_Omega_{comp}", f"Pi_H_{comp}",
            f"fOmega_abs_{comp}", f"fH_abs_{comp}",
        ]:
            arr = pd.to_numeric(closure_used[c], errors="coerce").to_numpy(dtype=float)
            print(f"    {c:18s}: {np.count_nonzero(np.isfinite(arr))}/{len(arr)} finite")

    z = pd.to_numeric(closure_used["Redshift"], errors="coerce").to_numpy(dtype=float)
    zgood = z[np.isfinite(z)]
    if zgood.size == 0:
        zmin, zmax = 0.0, 1.0
    else:
        zmin, zmax = float(np.nanmin(zgood)), float(np.nanmax(zgood))
        if zmax <= zmin:
            zmax = zmin + 1e-6
    norm = Normalize(vmin=zmin, vmax=zmax)

    figs_by_method = {}
    save_paths = []

    for method in methods:
        dmethod = closure_used.loc[
            closure_used["shell_method"].astype(str) == str(method)
        ].copy()

        if len(dmethod) == 0:
            print(f"[Pi closure] no rows for method={method}; skipped.")
            continue

        figs_by_method[str(method)] = {}

        fig_all = _draw_one_figure(
            dmethod,
            method_label=str(method),
            shell_label=None,
            norm=norm,
        )
        figs_by_method[str(method)]["all"] = fig_all

        if save:
            p = OUTDIR / f"pi_closure_cross_time_{method}_all_shells.png"
            fig_all.savefig(p, dpi=dpi, bbox_inches="tight")
            save_paths.append(p)
            print("Saved:", p)

        for shell_plot in sorted(dmethod["shell_plot"].dropna().unique().astype(int).tolist()):
            ds = dmethod.loc[dmethod["shell_plot"].astype(int) == int(shell_plot)].copy()
            if len(ds) == 0:
                continue

            fig_shell = _draw_one_figure(
                ds,
                method_label=str(method),
                shell_label=int(shell_plot),
                norm=norm,
            )
            figs_by_method[str(method)][f"shell_{int(shell_plot)}"] = fig_shell

            if save:
                p = OUTDIR / f"pi_closure_cross_time_{method}_shell_{int(shell_plot)}.png"
                fig_shell.savefig(p, dpi=dpi, bbox_inches="tight")
                save_paths.append(p)
                print("Saved:", p)

    if show:
        plt.show()
    else:
        for method, figdict in figs_by_method.items():
            for _, fig in figdict.items():
                plt.close(fig)

    return figs_by_method, closure_used, track_tables, save_paths

# %% code cell 15
SNAP_TRACK

# %% code cell 16
fig, ax, records, table_df, save_path = plot_subhalo_orbitplane_overlay(
    3,
    SNAP_TRACK ,
)

table_df, save_paths = plot_subhalo_orbitplane_table_evolution(
    3,
    SNAP_TRACK , table_df= table_df,
)
figs_pi, closure_used, track_tables, pi_paths = plot_pi_closure_table_evolution(
    3,
    SNAP_TRACK ,
    methods=("radial",),
)

# %% code cell 17
fig, ax, records, table_df, save_path = plot_subhalo_orbitplane_overlay(
    3,
    SNAP_TRACK ,SHELL_METHOD="binding_energy",
)
figs_pi, closure_used, track_tables, pi_paths = plot_pi_closure_table_evolution(
    3,
    SNAP_TRACK ,
    methods=("binding_energy",),
) 
figs, table_df, save_paths = plot_subhalo_orbitplane_table_evolution(
    3,
    SNAP_TRACK , table_df= table_df,
)

# %% code cell 18

# %% [markdown] cell 19
# ## Explicit cleanup

# %% code cell 20

# hd_tng.cleanup_open_catalogs()
print('Temporary TNG API files opened through hd_tng have been deleted.')

# %% code cell 21

# %% code cell 22

# %% code cell 23

# %% code cell 24
