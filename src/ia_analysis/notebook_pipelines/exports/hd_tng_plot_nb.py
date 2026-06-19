"""Exported code from notebooks/raw_20260618/hd_tng_plot.ipynb.

This file is generated for project management and refactoring. Review before running end to end.
"""


# %% [markdown] cell 1
# # TNG300-1 FoF/Subhalo shell dynamics with `hd_tng.py` This notebook calls the high-level driver `hd_tng.py` to run the full workflow: 1. read or download TNG300-1 group catalogues and subhalo cutouts; 2. select subhaloes in the five largest FoF groups; 3. build radial and binding-energy shells; 4. measure the affine velocity field $$ A=\mathcal H+\Omega, $$ and the figure-rotation estimate $$ \Pi_{ij}=\widehat\Omega_{ij}+\dfrac{\lambda_i+\lambda_j}{\lambda_i-\lambda_j}\widehat{\mathcal H}_{ij};

# %% code cell 2
# IPython-only: !pwd

# %% code cell 3

from pathlib import Path
import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DWE import DimrothWatson


dw = DimrothWatson()


import importlib


from TNGCatLoader import TNGCatalog
import hd_tng
import arts
print('hd_tng loaded from:', hd_tng.__file__)
print('arts loaded from:', arts.__file__)

# %% code cell 4
import halo_dynamics
import hd_tng
import shape
import Iana
importlib.reload(halo_dynamics)
importlib.reload(hd_tng)
importlib.reload(shape)
importlib.reload(Iana)

# %% [markdown] cell 5
# ## Configuration

# %% code cell 6

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
    top_n_groups=5,
    max_subhaloes_per_group=20,
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

# %% [markdown] cell 7
# ## Run the packaged calculation

# %% code cell 8

run = hd_tng.compute_haloes(
    BASE_PATH,
    SNAP,
    cfg=CFG,
)

# %% code cell 9
print('Number of successful subhaloes:', len(run['results']))
print('Number of failures:', len(run['failures']))
display(run['target_table'])
if len(run['failures']):
    display(run['failures'])

closure_all = run['closure_all']
closure_all.to_csv(OUTDIR / 'hd_tng_instantaneous_pi_closure.csv', index=False)
print('closure rows:', len(closure_all))
display(closure_all.head())

# %% code cell 10
# ------------------------------------------------------------
# Force-patch FoF-centric metadata into run and closure_all
# ------------------------------------------------------------
# This cell does NOT recompute particles or shell dynamics.
# It only reloads FoF/subhalo catalogue fields and patches:
#
#   run["results"][i]["Sub_info"]
#   run["subhalo_metadata"]
#   run["closure_all"]
#   local closure_all
#
# Required for colour-coding points by r/R200c.
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
from pathlib import Path

# Prefer hd_tng_mea if it exists; otherwise use hd_tng.
HDMOD = globals().get("hd_tng_mea", None)
if HDMOD is None:
    HDMOD = globals().get("hd_tng", None)

if HDMOD is None:
    raise RuntimeError("Neither hd_tng_mea nor hd_tng is defined in this notebook.")

print("Using module:", getattr(HDMOD, "__name__", "<unknown>"))
print("Using module file:", getattr(HDMOD, "__file__", "<unknown>"))


GROUP_FIELDS_META = [
    "GroupPos",
    "Group_R_Crit200",
    "Group_M_Crit200",
    "GroupFirstSub",
    "GroupLenType",
    "GroupNsubs",
]

SUBHALO_FIELDS_META = [
    "SubhaloPos",
    "SubhaloGrNr",
    "SubhaloMass",
    "SubhaloMassType",
    "SubhaloLenType",
]


def _get_header_safe(base_path, snap):
    """Get TNG header from HDMOD if possible."""
    try:
        sim_name = (
            run.get("metadata", {})
            .get("sim_name", globals().get("SIM_NAME", "TNG50-1"))
        )
        cfg = run.get("metadata", {}).get("cfg", {})
        api_key = cfg.get("api_key", os.environ.get("TNG_API_KEY", None))

        return HDMOD.read_header_for_snap(
            base_path,
            int(snap),
            sim_name=sim_name,
            api_key=api_key,
        )
    except Exception as exc:
        print("Header read via HDMOD failed:", repr(exc))
        return run.get("metadata", {}).get("header", {})


def _header_value(header, key, default=np.nan):
    try:
        return float(header.get(key, default))
    except Exception:
        return float(default)


def _ckpc_h_to_kpc(x_ckpc_h, header):
    """physical kpc = ckpc/h * a / h."""
    a = _header_value(header, "Time", 1.0)
    h = _header_value(header, "HubbleParam", 0.6774)
    return np.asarray(x_ckpc_h, dtype=float) * a / h


def _mass_1e10msun_h_to_msun(m, header):
    """Msun = value * 1e10 / h."""
    h = _header_value(header, "HubbleParam", 0.6774)
    return np.asarray(m, dtype=float) * 1.0e10 / h


def _periodic_delta_ckpc_h(x, x0, header):
    """Periodic displacement x - x0 in ckpc/h."""
    dx = np.asarray(x, dtype=float) - np.asarray(x0, dtype=float)
    box = _header_value(header, "BoxSize", np.nan)

    if np.isfinite(box) and box > 0:
        dx = dx - box * np.rint(dx / box)

    return dx


def _load_catalog_metadata(base_path, snap):
    """
    Load only the group/subhalo fields needed for metadata.
    Prefer HDMOD.open_catalog, then fall back to illustris_python.groupcat.
    """
    # Try HDMOD.open_catalog first.
    if hasattr(HDMOD, "open_catalog"):
        cat = None
        try:
            cfg = run.get("metadata", {}).get("cfg", {})
            sim_name = run.get("metadata", {}).get("sim_name", globals().get("SIM_NAME", "TNG50-1"))

            tng_catalog_kwargs = globals().get("TNG_CATALOG_KWARGS", None)
            if tng_catalog_kwargs is None and hasattr(HDMOD, "default_tng_catalog_kwargs"):
                tng_catalog_kwargs = HDMOD.default_tng_catalog_kwargs(
                    sim_name=sim_name,
                    api_key=cfg.get("api_key", os.environ.get("TNG_API_KEY", None)),
                    download_if_missing=bool(cfg.get("download_if_missing", True)),
                    delete_cache=bool(cfg.get("delete_cache", True)),
                    cache_dir=cfg.get("cache_dir", None),
                    verbose=bool(cfg.get("verbose", True)),
                    timeout=int(cfg.get("timeout", 180)),
                )

            retry_cfg = dict(
                max_retries=int(cfg.get("api_max_retries", 6)),
                base_sleep=float(cfg.get("api_retry_base_sleep", 5.0)),
                max_sleep=float(cfg.get("api_retry_max_sleep", 90.0)),
                verbose=bool(cfg.get("verbose", True)),
            )

            cat, halos, subs = HDMOD.open_catalog(
                base_path,
                int(snap),
                group_fields=GROUP_FIELDS_META,
                subhalo_fields=SUBHALO_FIELDS_META,
                tng_catalog_kwargs=tng_catalog_kwargs,
                retry_cfg=retry_cfg,
            )

            print("[metadata] catalogue loaded through HDMOD.open_catalog")
            return halos, subs, cat

        except Exception as exc:
            print("[metadata] HDMOD.open_catalog failed; fallback to illustris_python:", repr(exc))
            if cat is not None:
                try:
                    cat.cleanup()
                except Exception:
                    pass

    # Fallback: illustris_python.groupcat.
    try:
        from illustris_python import groupcat

        halos = groupcat.loadHalos(
            str(base_path),
            int(snap),
            fields=GROUP_FIELDS_META,
        )
        subs = groupcat.loadSubhalos(
            str(base_path),
            int(snap),
            fields=SUBHALO_FIELDS_META,
        )

        print("[metadata] catalogue loaded through illustris_python.groupcat")
        return halos, subs, None

    except Exception as exc:
        raise RuntimeError(f"Could not load metadata catalogue fields: {exc}") from exc


def _safe_sub_array(subs, key, sid, default):
    if key not in subs:
        return np.array(default, dtype=float)
    try:
        return np.asarray(subs[key][int(sid)])
    except Exception:
        return np.array(default, dtype=float)


def _safe_group_array(halos, key, gid, default):
    if key not in halos:
        return np.array(default, dtype=float)
    try:
        return np.asarray(halos[key][int(gid)])
    except Exception:
        return np.array(default, dtype=float)


def force_patch_fof_metadata_into_run(run, base_path, snap):
    """
    Force-build FoF metadata from catalog fields and merge it into run.
    """
    header = _get_header_safe(base_path, snap)
    halos, subs, cat = _load_catalog_metadata(base_path, snap)

    try:
        if "SubhaloGrNr" not in subs:
            raise RuntimeError("SubhaloGrNr is missing; cannot determine FoF host group.")

        sub_grnr = np.asarray(subs["SubhaloGrNr"], dtype=int)

        meta_rows = []

        for res in run.get("results", []):
            sub_info = res.setdefault("Sub_info", {})
            sid = int(sub_info.get("SubhaloID", -1))

            if sid < 0 or sid >= len(sub_grnr):
                continue

            gid_catalog = int(sub_grnr[sid])
            gid_input = int(sub_info.get("GroupID", gid_catalog))

            # Subhalo fields
            sub_pos_ckpc_h = _safe_sub_array(subs, "SubhaloPos", sid, [np.nan, np.nan, np.nan]).astype(float)
            sub_mass_1e10 = float(_safe_sub_array(subs, "SubhaloMass", sid, np.nan))
            sub_mass_type_1e10 = _safe_sub_array(subs, "SubhaloMassType", sid, np.full(6, np.nan)).astype(float)

            # Group fields
            group_pos_ckpc_h = _safe_group_array(halos, "GroupPos", gid_catalog, [np.nan, np.nan, np.nan]).astype(float)
            group_r200c_ckpc_h = float(_safe_group_array(halos, "Group_R_Crit200", gid_catalog, np.nan))
            group_m200c_1e10 = float(_safe_group_array(halos, "Group_M_Crit200", gid_catalog, np.nan))
            group_first_sub = int(_safe_group_array(halos, "GroupFirstSub", gid_catalog, -1))

            # Group-centric radius
            if np.all(np.isfinite(sub_pos_ckpc_h)) and np.all(np.isfinite(group_pos_ckpc_h)):
                d_ckpc_h = _periodic_delta_ckpc_h(sub_pos_ckpc_h, group_pos_ckpc_h, header)
                rgrp_ckpc_h = float(np.linalg.norm(d_ckpc_h))
            else:
                d_ckpc_h = np.full(3, np.nan)
                rgrp_ckpc_h = np.nan

            if np.isfinite(group_r200c_ckpc_h) and group_r200c_ckpc_h > 0:
                r_over_r200c = float(rgrp_ckpc_h / group_r200c_ckpc_h)
            else:
                r_over_r200c = np.nan

            # Unit conversions
            group_r200c_kpc = float(_ckpc_h_to_kpc(group_r200c_ckpc_h, header))
            rgrp_kpc = float(_ckpc_h_to_kpc(rgrp_ckpc_h, header))
            group_m200c_msun = float(_mass_1e10msun_h_to_msun(group_m200c_1e10, header))
            sub_mass_msun = float(_mass_1e10msun_h_to_msun(sub_mass_1e10, header))
            sub_mass_type_msun = _mass_1e10msun_h_to_msun(sub_mass_type_1e10, header)

            dm_mass_1e10 = float(sub_mass_type_1e10[1]) if sub_mass_type_1e10.size > 1 else np.nan
            dm_mass_msun = float(sub_mass_type_msun[1]) if sub_mass_type_msun.size > 1 else np.nan
            star_mass_1e10 = float(sub_mass_type_1e10[4]) if sub_mass_type_1e10.size > 4 else np.nan
            star_mass_msun = float(sub_mass_type_msun[4]) if sub_mass_type_msun.size > 4 else np.nan

            is_central = bool(sid == group_first_sub)

            # Patch Sub_info
            sub_info.update({
                "SubhaloID": sid,
                "GroupID": gid_input,
                "GroupID_catalog": gid_catalog,
                "IsCentral": is_central,

                "SubhaloPos_ckpc_h": sub_pos_ckpc_h,
                "SubhaloPos_kpc": _ckpc_h_to_kpc(sub_pos_ckpc_h, header),

                "GroupPos_ckpc_h": group_pos_ckpc_h,
                "GroupPos_kpc": _ckpc_h_to_kpc(group_pos_ckpc_h, header),

                "GroupCentricDelta_ckpc_h": d_ckpc_h,
                "GroupCentricDelta_kpc": _ckpc_h_to_kpc(d_ckpc_h, header),

                "Group_R_Crit200_ckpc_h": group_r200c_ckpc_h,
                "Group_R_Crit200_kpc": group_r200c_kpc,
                "Group_M_Crit200_1e10Msun_h": group_m200c_1e10,
                "Group_M_Crit200_Msun": group_m200c_msun,

                "GroupCentricRadius_ckpc_h": rgrp_ckpc_h,
                "GroupCentricRadius_kpc": rgrp_kpc,
                "r_over_r200c": r_over_r200c,

                "SubhaloMass_1e10Msun_h": sub_mass_1e10,
                "SubhaloMass_Msun": sub_mass_msun,
                "SubhaloMassType_1e10Msun_h": sub_mass_type_1e10,
                "SubhaloMassType_Msun": sub_mass_type_msun,

                "SubhaloMassType_dm_1e10Msun_h": dm_mass_1e10,
                "SubhaloMassType_dm_Msun": dm_mass_msun,
                "SubhaloMassType_stars_1e10Msun_h": star_mass_1e10,
                "SubhaloMassType_stars_Msun": star_mass_msun,
            })

            meta_rows.append({
                "SubhaloID": sid,
                "GroupID": gid_input,
                "GroupID_catalog": gid_catalog,
                "IsCentral": is_central,

                "Group_R_Crit200_ckpc_h": group_r200c_ckpc_h,
                "Group_R_Crit200_kpc": group_r200c_kpc,
                "Group_M_Crit200_1e10Msun_h": group_m200c_1e10,
                "Group_M_Crit200_Msun": group_m200c_msun,

                "GroupCentricRadius_ckpc_h": rgrp_ckpc_h,
                "GroupCentricRadius_kpc": rgrp_kpc,
                "r_over_r200c": r_over_r200c,

                "SubhaloMass_1e10Msun_h": sub_mass_1e10,
                "SubhaloMass_Msun": sub_mass_msun,

                "SubhaloMassType_dm_1e10Msun_h": dm_mass_1e10,
                "SubhaloMassType_dm_Msun": dm_mass_msun,
                "SubhaloMassType_stars_1e10Msun_h": star_mass_1e10,
                "SubhaloMassType_stars_Msun": star_mass_msun,
            })

        meta = pd.DataFrame(meta_rows)

        if len(meta):
            meta = meta.drop_duplicates(subset=["SubhaloID"], keep="last").reset_index(drop=True)

        run["subhalo_metadata"] = meta

        # Force-merge into closure_all
        closure = run.get("closure_all", pd.DataFrame()).copy()

        if isinstance(closure, pd.DataFrame) and len(closure) and len(meta):
            closure["SubhaloID"] = closure["SubhaloID"].astype(int)
            meta["SubhaloID"] = meta["SubhaloID"].astype(int)

            drop_cols = [
                c for c in meta.columns
                if c != "SubhaloID" and c in closure.columns
            ]
            closure = closure.drop(columns=drop_cols, errors="ignore")

            closure = closure.merge(
                meta,
                on="SubhaloID",
                how="left",
                validate="many_to_one",
            )

            run["closure_all"] = closure

        return meta

    finally:
        if cat is not None:
            try:
                cat.cleanup()
            except Exception:
                pass


# ------------------------------------------------------------
# Run force patch
# ------------------------------------------------------------
subhalo_metadata = force_patch_fof_metadata_into_run(
    run,
    BASE_PATH,
    SNAP,
)

closure_all = run.get("closure_all", pd.DataFrame())

# Save patched closure table
if isinstance(closure_all, pd.DataFrame) and len(closure_all):
    closure_all.to_csv(OUTDIR / "hd_tng_instantaneous_pi_closure.csv", index=False)

# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------
print("\nsubhalo metadata rows:", len(subhalo_metadata))

display_cols = [
    "SubhaloID",
    "GroupID",
    "GroupID_catalog",
    "IsCentral",
    "r_over_r200c",
    "Group_R_Crit200_kpc",
    "Group_M_Crit200_Msun",
    "SubhaloMass_Msun",
]
display(subhalo_metadata[[c for c in display_cols if c in subhalo_metadata.columns]].head(20))

if isinstance(closure_all, pd.DataFrame) and len(closure_all):
    print("\nclosure_all columns patched:")
    for col in ["r_over_r200c", "Group_M_Crit200_Msun", "SubhaloMass_Msun"]:
        if col in closure_all.columns:
            vals = closure_all[col].to_numpy(dtype=float)
            print(
                f"  finite {col}: {np.isfinite(vals).sum()} / {len(vals)}",
                end="",
            )
            if np.isfinite(vals).any():
                print(
                    f" ; min={np.nanmin(vals):.4g}, max={np.nanmax(vals):.4g}"
                )
            else:
                print()
        else:
            print(f"  Column {col} is not present in closure_all.")

    if "r_over_r200c" in closure_all.columns:
        rr = closure_all["r_over_r200c"].to_numpy(dtype=float)
        finite_rr = rr[np.isfinite(rr)]
        print("\nunique finite r/R200c values:")
        print(np.unique(np.round(finite_rr, 5)))

print("\nImportant: rebuild pi_diag_df / pi_vec_df / dyn_* tables after this patch.")

# %% code cell 11

# Optional: save the in-memory result dictionary.  This can be large if keep_particles=True.
# with open(OUTDIR / 'hd_tng_run.pkl', 'wb') as f:
#     pickle.dump(run, f, protocol=pickle.HIGHEST_PROTOCOL)
# print('Saved:', OUTDIR / 'hd_tng_run.pkl')

# %% [markdown] cell 12
# ## Plot instantaneous closure: $\Pi_{\Omega+\mathcal H}$ vs $\Pi_{\dot I}$

# %% code cell 13
# Close any previously broken figures left by failed mathtext rendering
plt.close("all")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.lines import Line2D
from pathlib import Path

# ------------------------------------------------------------
# Publication-style settings
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.0,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.color": "k",
    "ytick.color": "k",
    "savefig.dpi": 240,
    "figure.dpi": 140,
})

CENTRAL_COLOR = "black"
SAT_CMAP = plt.cm.Blues_r      # smaller r/R200c -> darker blue
SAT_FALLBACK = "#6BAED6"


def _component_limits(x, y, pad_frac=0.07):
    """Return equal x/y limits for the closure panel."""
    vals = np.concatenate([np.asarray(x), np.asarray(y)])
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return -1.0, 1.0

    lo = np.nanmin(vals)
    hi = np.nanmax(vals)

    if not np.isfinite(lo) or not np.isfinite(hi):
        return -1.0, 1.0

    if hi <= lo:
        pad = max(abs(lo), 1.0) * 0.1
        return lo - pad, hi + pad

    pad = pad_frac * (hi - lo)
    return lo - pad, hi + pad


def _get_bool_column(df, col, default=False):
    """Safely read a boolean column from a dataframe."""
    if col not in df.columns:
        return np.full(len(df), bool(default), dtype=bool)
    return np.asarray(df[col]).astype(bool)


def _get_float_column(df, col, fill=np.nan):
    """Safely read a float column from a dataframe."""
    if col not in df.columns:
        return np.full(len(df), fill, dtype=float)
    return df[col].to_numpy(dtype=float)


def _get_rel_residual(df, lab, x, y):
    """
    Dimensionless relative residual:
        delta_Pi = (Pi_aff - Pi_direct) / |Pi_direct|

    If the dataframe only has rel_residual_*_pct, convert it back to
    dimensionless form by dividing by 100.
    """
    pct_col = f"rel_residual_{lab}_pct"
    dim_col = f"rel_residual_{lab}"

    if dim_col in df.columns:
        return df[dim_col].to_numpy(dtype=float)

    if pct_col in df.columns:
        return df[pct_col].to_numpy(dtype=float) / 100.0

    denom = np.abs(x)
    out = np.full_like(x, np.nan, dtype=float)
    good = np.isfinite(x) & np.isfinite(y) & (denom > 0.0)
    out[good] = (y[good] - x[good]) / denom[good]
    return out


def _get_rel_residual_err(df, lab):
    """
    Optional propagated uncertainty of dimensionless relative residual.

    If the dataframe only has *_pct, convert it back to dimensionless form.
    """
    candidates_dimless = [
        f"rel_residual_err_{lab}",
        f"rel_residual_{lab}_err",
    ]
    for col in candidates_dimless:
        if col in df.columns:
            return df[col].to_numpy(dtype=float)

    candidates_pct = [
        f"rel_residual_err_{lab}_pct",
        f"rel_residual_{lab}_err_pct",
    ]
    for col in candidates_pct:
        if col in df.columns:
            return df[col].to_numpy(dtype=float) / 100.0

    return None


def _make_satellite_norm(df, satellite_rmax=None):
    """
    Build color normalization for satellite r/R200c.
    central objects are not included in the normalization.
    """
    is_central = _get_bool_column(df, "IsCentral", default=False)
    rr = _get_float_column(df, "r_over_r200c", fill=np.nan)

    sat_rr = rr[(~is_central) & np.isfinite(rr)]

    if satellite_rmax is None:
        if sat_rr.size:
            satellite_rmax = np.nanpercentile(sat_rr, 95.0)
            satellite_rmax = max(float(satellite_rmax), 0.2)
        else:
            satellite_rmax = 1.0

    norm = colors.Normalize(vmin=0.0, vmax=float(satellite_rmax))
    return norm, is_central, rr


def _point_colors(is_central, rr, norm, cmap):
    """
    central: black
    satellite: color by r/R200c
    """
    cols = []

    for cen, val in zip(is_central, rr):
        if cen:
            cols.append(CENTRAL_COLOR)
        else:
            if np.isfinite(val):
                cols.append(cmap(norm(val)))
            else:
                cols.append(SAT_FALLBACK)

    return np.asarray(cols, dtype=object)


def _rotate_residual_yticklabels(ax, angle=45):
    """Rotate residual-panel y tick labels counter-clockwise."""
    for tick in ax.get_yticklabels():
        tick.set_rotation(angle)
        tick.set_ha("right")
        tick.set_va("center")


def _draw_errorbar_subset(
    ax,
    x,
    y,
    *,
    xerr=None,
    yerr=None,
    mask=None,
    ecolor="0.65",
    zorder=1,
):
    """
    Draw errorbars only for points where the relevant error values are finite.
    This avoids killing the main scatter when error columns are NaN.
    """
    if mask is None:
        mask = np.ones_like(x, dtype=bool)

    good = np.asarray(mask, dtype=bool).copy()
    good &= np.isfinite(x) & np.isfinite(y)

    if xerr is not None:
        good &= np.isfinite(xerr)
    if yerr is not None:
        good &= np.isfinite(yerr)

    if not np.any(good):
        return

    ax.errorbar(
        x[good],
        y[good],
        xerr=xerr[good] if xerr is not None else None,
        yerr=yerr[good] if yerr is not None else None,
        fmt="none",
        ecolor=ecolor,
        elinewidth=0.55,
        alpha=0.30,
        capsize=0,
        zorder=zorder,
    )


def plot_pi_closure_table(
    df,
    *,
    title=None,
    savepath=None,
    show_errorbar=False,
    residual_ylim=1e-12,
    satellite_rmax=None,
):
    """
    Three component columns. Each component contains:

    upper panel:
        Pi_direct versus Pi_aff.

    lower panel:
        dimensionless relative residual,

            delta_Pi = (Pi_aff - Pi_direct) / |Pi_direct|.

    color:
        central   -> black
        satellite -> Blues_r by r/R200c, darker means closer to group centre.
    """

    comps = [("01", "0,1"), ("02", "0,2"), ("12", "1,2")]

    norm, is_central_all, rr_all = _make_satellite_norm(
        df,
        satellite_rmax=satellite_rmax,
    )
    point_colors_all = _point_colors(is_central_all, rr_all, norm, SAT_CMAP)

    fig = plt.figure(figsize=(15.6, 6.9))

    gs = fig.add_gridspec(
        2, 8,
        width_ratios=[1.0, 0.040, 0.070, 1.0, 0.040, 0.070, 1.0, 0.040],
        height_ratios=[3.0, 1.05],
        hspace=0.055,
        wspace=0.10,
    )

    axes_top = []
    axes_bot = []

    plot_cols = [0, 3, 6]
    cbar_cols = [1, 4, 7]

    for i, (lab, pretty) in enumerate(comps):
        ax = fig.add_subplot(gs[0, plot_cols[i]])
        axr = fig.add_subplot(gs[1, plot_cols[i]], sharex=ax)
        cax = fig.add_subplot(gs[:, cbar_cols[i]])

        axes_top.append(ax)
        axes_bot.append(axr)

        xcol = f"Pi_direct_{lab}"
        ycol = f"Pi_aff_{lab}"

        if xcol not in df.columns or ycol not in df.columns:
            ax.text(
                0.5,
                0.5,
                f"Missing columns:\n{xcol}\n{ycol}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            axr.set_axis_off()
            cax.set_axis_off()
            continue

        x = df[xcol].to_numpy(dtype=float)
        y = df[ycol].to_numpy(dtype=float)
        rel = _get_rel_residual(df, lab, x, y)

        xerr = (
            df[f"Pi_direct_err_{lab}"].to_numpy(dtype=float)
            if f"Pi_direct_err_{lab}" in df.columns else None
        )
        yerr = (
            df[f"Pi_aff_err_{lab}"].to_numpy(dtype=float)
            if f"Pi_aff_err_{lab}" in df.columns else None
        )
        relerr = _get_rel_residual_err(df, lab)

        # Crucial fix:
        # Top closure panel should not require finite residual or finite errors.
        good_top = np.isfinite(x) & np.isfinite(y)

        # Residual panel needs finite relative residual.
        good_res = np.isfinite(x) & np.isfinite(y) & np.isfinite(rel)

        xg = x[good_top]
        yg = y[good_top]
        xerrg = xerr[good_top] if xerr is not None else None
        yerrg = yerr[good_top] if yerr is not None else None

        cen = is_central_all[good_top]
        sat = ~cen
        cg = point_colors_all[good_top]

        xr = x[good_res]
        relr = rel[good_res]
        relerrr = relerr[good_res] if relerr is not None else None

        cen_r = is_central_all[good_res]
        sat_r = ~cen_r
        cr = point_colors_all[good_res]

        lo, hi = _component_limits(xg, yg)

        # ----------------------------------------------------
        # Upper panel: closure scatter
        # ----------------------------------------------------
        ax.plot(
            [lo, hi],
            [lo, hi],
            lw=1.15,
            color="0.12",
            alpha=0.85,
            zorder=0,
        )

        if np.any(sat):
            if show_errorbar:
                _draw_errorbar_subset(
                    ax,
                    xg,
                    yg,
                    xerr=xerrg,
                    yerr=yerrg,
                    mask=sat,
                    ecolor="0.65",
                    zorder=1,
                )

            ax.scatter(
                xg[sat],
                yg[sat],
                s=28,
                c=list(cg[sat]),
                alpha=0.82,
                edgecolor="white",
                linewidth=0.25,
                rasterized=True,
                zorder=2,
            )

        if np.any(cen):
            if show_errorbar:
                _draw_errorbar_subset(
                    ax,
                    xg,
                    yg,
                    xerr=xerrg,
                    yerr=yerrg,
                    mask=cen,
                    ecolor="0.20",
                    zorder=3,
                )

            ax.scatter(
                xg[cen],
                yg[cen],
                s=36,
                c=CENTRAL_COLOR,
                alpha=0.93,
                edgecolor="white",
                linewidth=0.28,
                rasterized=True,
                zorder=4,
            )

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(rf"component $({pretty})$")
        ax.grid(alpha=0.16, lw=0.65)
        ax.tick_params(labelbottom=False)

        if i == 0:
            ax.set_ylabel(r"$\Pi_{\Omega+\mathcal{H}}$ [Gyr$^{-1}$]")
        else:
            ax.set_ylabel("")

        if xg.size:
            abs_res = yg - xg
            rms = np.sqrt(np.nanmean(abs_res**2))
            med_rel = np.nanmedian(relr) if relr.size else np.nan

            ax.text(
                0.055,
                0.945,
                rf"$N_{{\rm shell}}={xg.size}$" + "\n" +
                rf"$\mathrm{{RMS}}={rms:.2g}$" + "\n" +
                rf"$\mathrm{{med}}(\delta_\Pi)={med_rel:.2g}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9.2,
                bbox=dict(
                    facecolor="white",
                    edgecolor="0.86",
                    alpha=0.88,
                    boxstyle="round,pad=0.25",
                ),
            )

        if i == 0:
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor="black",
                    markeredgecolor="white",
                    markeredgewidth=0.35,
                    markersize=6.8,
                    label="central",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=SAT_CMAP(norm(0.6 * norm.vmax)),
                    markeredgecolor="white",
                    markeredgewidth=0.35,
                    markersize=6.8,
                    label="satellite",
                ),
            ]
            ax.legend(
                handles=legend_handles,
                loc="lower right",
                frameon=True,
                framealpha=0.88,
                borderpad=0.32,
                handletextpad=0.45,
            )

        # ----------------------------------------------------
        # Lower panel: relative residual
        # ----------------------------------------------------
        axr.axhline(0.0, color="0.12", lw=1.05, alpha=0.85, zorder=0)

        if np.any(sat_r):
            if show_errorbar and relerrr is not None:
                _draw_errorbar_subset(
                    axr,
                    xr,
                    relr,
                    yerr=relerrr,
                    mask=sat_r,
                    ecolor="0.65",
                    zorder=1,
                )

            axr.scatter(
                xr[sat_r],
                relr[sat_r],
                s=24,
                c=list(cr[sat_r]),
                alpha=0.82,
                edgecolor="white",
                linewidth=0.22,
                rasterized=True,
                zorder=2,
            )

        if np.any(cen_r):
            if show_errorbar and relerrr is not None:
                _draw_errorbar_subset(
                    axr,
                    xr,
                    relr,
                    yerr=relerrr,
                    mask=cen_r,
                    ecolor="0.20",
                    zorder=3,
                )

            axr.scatter(
                xr[cen_r],
                relr[cen_r],
                s=32,
                c=CENTRAL_COLOR,
                alpha=0.93,
                edgecolor="white",
                linewidth=0.24,
                rasterized=True,
                zorder=4,
            )

        axr.set_xlim(lo, hi)
        axr.set_ylim(-residual_ylim, residual_ylim)
        axr.set_xlabel(r"$\Pi_{\dot{I}}$ [Gyr$^{-1}$]")

        if i == 0:
            axr.set_ylabel(r"$\delta_\Pi$")
        else:
            axr.set_ylabel("")

        _rotate_residual_yticklabels(axr, angle=45)
        axr.grid(alpha=0.16, lw=0.65)

        # ----------------------------------------------------
        # Per-component colorbar spanning both rows
        # ----------------------------------------------------
        sm = cm.ScalarMappable(norm=norm, cmap=SAT_CMAP)
        sm.set_array([])

        cbar = fig.colorbar(sm, cax=cax)
        cbar.ax.tick_params(direction="in", labelsize=9.0)

        if i == 2:
            cbar.set_label(r"satellite $r/R_{200\mathrm{c}}$", fontsize=11)
        else:
            cbar.set_label("")

    if title:
        fig.suptitle(title, y=0.975, fontsize=14.5)

    fig.align_ylabels([axes_top[0], axes_bot[0]])

    fig.subplots_adjust(
        left=0.070,
        right=0.985,
        bottom=0.108,
        top=0.895,
    )

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=240, bbox_inches="tight")

    return fig, np.array([axes_top, axes_bot], dtype=object)


def plot_pi_residual_histogram(df, *, savepath=None):
    """
    Summary histogram of dimensionless relative residuals.
    """
    comps = [("01", "0,1"), ("02", "0,2"), ("12", "1,2")]

    colors_comp = {
        "01": "#3E5C76",
        "02": "#5A9367",
        "12": "#A23E48",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    all_rr = []

    for lab, pretty in comps:
        xcol = f"Pi_direct_{lab}"
        ycol = f"Pi_aff_{lab}"

        if xcol not in df.columns or ycol not in df.columns:
            continue

        x = df[xcol].to_numpy(dtype=float)
        y = df[ycol].to_numpy(dtype=float)

        rr = _get_rel_residual(df, lab, x, y)
        rr = rr[np.isfinite(rr)]

        if rr.size:
            all_rr.append(rr)
            ax.hist(
                rr[abs(rr)<3e-14],
                bins=50,
                histtype="step",
                lw=1.8,
                color=colors_comp.get(lab, "0.25"),
                label=rf"$({pretty})$",
            )

    if all_rr:
        all_rr = np.concatenate(all_rr)
        all_rr = all_rr[np.isfinite(all_rr)]

        if all_rr.size:
            lim = np.nanpercentile(np.abs(all_rr), 99.0)
            if not np.isfinite(lim) or lim <= 0:
                lim = 1e-12
            ax.set_xlim(-1.1 * lim, 1.1 * lim)

    ax.axvline(0.0, color="0.12", lw=1.0, alpha=0.8)
    ax.set_xlabel(
        r"$\delta_\Pi = (\Pi_{\Omega+\mathcal{H}}-\Pi_{\dot{I}})/|\Pi_{\dot{I}}|$"
    )
    ax.set_ylabel("Number of shells")
    ax.legend(frameon=False, ncol=3)
    ax.grid(alpha=0.16, lw=0.65)

    fig.tight_layout()

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=240, bbox_inches="tight")

    return fig, ax


# ------------------------------------------------------------
# Run plotting
# ------------------------------------------------------------
SHOW_ERRORBAR = True
RESIDUAL_YLIM = 1e-12

if len(closure_all):
    fig, axes = plot_pi_closure_table(
        closure_all,
        title="Instantaneous shell-wise figure-rotation closure",
        savepath=OUTDIR / "pi_closure_scatter_with_residuals.png",
        show_errorbar=SHOW_ERRORBAR,
        residual_ylim=RESIDUAL_YLIM,
    )
    plt.show()

    fig, ax = plot_pi_residual_histogram(
        closure_all,
        savepath=OUTDIR / "pi_closure_residual_hist.png",
    )
    plt.show()

else:
    print("No closure rows to plot.")

# %% code cell 14
closure_all["r_over_r200c"]

# %% [markdown] cell 15
# ## Shell visualisation for one example subhalo

# %% code cell 16
examples = run['results']#.sort(key=lambda r: len(r['particles']['dm']['X_kpc']))

# %% code cell 17
examples.sort(key=lambda r: len(r['particles']['dm']['X_kpc']))

# %% code cell 18
example=examples[3]

# %% code cell 19
example

# %% code cell 20


plt.close("all")

# -----------------------------
# plotting style: TNG-like look
# -----------------------------
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 180,
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "axes.linewidth": 1.1,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
     "xtick.color": "white",
    "ytick.color": "white",
})

# ---------------------------------------------------------
# helper functions
# ---------------------------------------------------------





# ---------------------------------------------------------
# make the plots
# ---------------------------------------------------------
if not run['results']:
    raise RuntimeError('No successful subhalo result. Lower MIN_DM_PARTICLES or check data/API access.')


SID = example['Sub_info']['SubhaloID']
pdata = example['particles']['dm']
print('Example SubhaloID:', SID, 'Ndm =', len(pdata['X_kpc']))

radial_block = example['shells']['radial']
binding_block = example['shells']['binding_energy']

X = pdata['X_kpc']
m = pdata['masses']
radial_masks = radial_block['masks']
binding_masks = binding_block['masks']

basis, evals = arts.principal_plane_basis_from_points(X, masses=m)

SIM_LABEL = globals().get("SIM_NAME", "TNG50-1")

fig, ax = arts.plot_radial_shells_pretty(
    X, m, radial_masks,
    basis=basis,
    bins=180,
    ellipse_nsigma=2.0,
    title=f'{SIM_LABEL} snap {SNAP}, Subhalo {SID}: radial shells',
    output_path=OUTDIR / f'subhalo_{SID}_radial_shell_ellipses.png',
)
for ax in fig.axes:
    ax.tick_params(
        axis="both",
        which="both",
        colors="white",       # tick + ticklabel
        labelcolor="black"    # ticklabel
    )
plt.show()

fig, axes = arts.plot_binding_shell_panels_pretty(
    X, m, binding_masks,
    basis=basis,
    bins=120,
    ellipse_nsigma=2.0,
    title=f'{SIM_LABEL} snap {SNAP}, Subhalo {SID}: binding-energy shells',
    output_path=OUTDIR / f'subhalo_{SID}_binding_shell_panels.png',
)
for ax in fig.axes:
    ax.tick_params(
        axis="both",
        which="both",
        colors="white",       # tick + ticklabel
        labelcolor="black"    # ticklabel
    )
plt.show()

# %% code cell 21
# ------------------------------------------------------------
# DW alignment distribution (reported by mu, not kappa)
# + component-wise violin plots for Omega/H fractions
# Robust version for hd_tng using direct dI Pi + affine decomposition
# ------------------------------------------------------------

plt.close("all")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
import importlib.util
import sys

# ------------------------------------------------------------
# Make sure the DW fitter is available
# ------------------------------------------------------------
if "dw" not in globals():
    try:
        from DWE import DimrothWatson
    except Exception:
        dwe_path = Path("DWE.py")
        if not dwe_path.exists():
            dwe_path = Path("/mnt/data/DWE.py")
        if not dwe_path.exists():
            raise FileNotFoundError(
                "Could not find DWE.py. Put it in the current directory or /mnt/data."
            )
        spec = importlib.util.spec_from_file_location("DWE", str(dwe_path))
        dwe_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dwe_mod)
        DimrothWatson = dwe_mod.DimrothWatson

    dw = DimrothWatson(name="dimroth_watson")


# -----------------------------
# User controls
# -----------------------------
SHELL_METHOD_FILTER = None   # None, "radial", or "binding_energy"
SAVE_FIG = True
SAVE_TABLE = True
MIN_SHELL_PARTICLES = 0      # e.g. 100 if you want to cut low-N shells

OUTDIR = Path(globals().get("OUTDIR", "hd_tng_outputs"))
OUTDIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Publication-style settings
# -----------------------------
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.2,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "savefig.dpi": 260,
    "figure.dpi": 140,
})

# publication-like palette
COLOR_OMEGA = "#355C7D"   # muted blue
COLOR_H     = "#E09F3E"   # muted orange
COLOR_DW    = "#2F4B7C"
COLOR_BAND  = "#C44E52"
COLOR_HIST  = "0.20"
COLOR_GRID  = "0.86"
COLOR_TEXT  = "0.15"

PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _safe_matrix_from_dict(dct, key, default=np.nan):
    """
    Safely read a 3x3 matrix from a dictionary.
    """
    if not isinstance(dct, dict):
        return np.full((3, 3), default, dtype=float)

    arr = np.asarray(dct.get(key, np.full((3, 3), default)), dtype=float)

    if arr.shape != (3, 3):
        return np.full((3, 3), default, dtype=float)

    return arr


def _get_affine_figure_block(sh):
    """
    In the current hd_tng / halo_dynamics_mea logic:

      sh["figure"]        : direct dI figure-rotation information
      sh["figure_affine"] : affine Omega + H decomposition

    This function prefers figure_affine, but falls back to figure for older
    runs where Omega_hat/H_hat/eta may have been stored there.
    """
    fig_aff = sh.get("figure_affine", {})
    fig_dir = sh.get("figure", {})

    if isinstance(fig_aff, dict):
        has_aff = any(k in fig_aff for k in ["Omega_hat", "H_hat", "eta", "Pi"])
        if has_aff:
            return fig_aff, fig_dir

    if isinstance(fig_dir, dict):
        return fig_dir, fig_dir

    return {}, {}


def build_pi_alignment_fraction_table(
    run,
    *,
    unit_factor=None,
    shell_method_filter=None,
    min_shell_particles=0,
    verbose=True,
):
    """
    Build a shell-wise table containing:

      1) vector alignment between Pi^Omega and Pi^H
      2) component-wise absolute contribution fractions
      3) optional direct dI and affine total Pi values, when available

    Vector representation uses the three independent off-diagonal components:
        [01, 02, 12].

    Important:
      Omega_hat, H_hat and eta are read from sh["figure_affine"] first.
      This is required for the current dI-based hd_tng version.
    """
    if unit_factor is None:
        try:
            unit_factor = float(hd_tng.KM_S_PER_KPC_TO_GYR_INV)
        except Exception:
            unit_factor = 1.0227121650537077  # (km/s)/kpc -> Gyr^-1

    rows = []
    comps = [("01", 0, 1), ("02", 0, 2), ("12", 1, 2)]

    results = run.get("results", []) if isinstance(run, dict) else []
    if not isinstance(results, list):
        raise TypeError("Expected run['results'] to be a list of subhalo result dictionaries.")

    n_shell_total = 0
    n_shell_valid = 0
    n_affinite = 0

    for result in results:
        if not isinstance(result, dict):
            continue

        sub_info = result.get("Sub_info", {})
        sid = int(sub_info.get("SubhaloID", -1))
        gid = int(sub_info.get("GroupID", -1))
        cen_id = int(sub_info.get("CenID", sub_info.get("CenID_catalog", -1))) if isinstance(sub_info, dict) else -1
        is_central = bool(sub_info.get("IsCentral", False)) if isinstance(sub_info, dict) else False

        r_over_r200c = np.nan
        for key in ["r_over_r200c", "RRelCen_over_R200c", "RRelCen_over_R200"]:
            if key in sub_info:
                try:
                    r_over_r200c = float(sub_info[key])
                    break
                except Exception:
                    pass

        shells_dict = result.get("shells", {})

        for shell_method, block in shells_dict.items():
            if shell_method_filter is not None and str(shell_method) != str(shell_method_filter):
                continue

            if not isinstance(block, dict):
                continue

            analysis = block.get("analysis", {})
            shells = analysis.get("shells", []) if isinstance(analysis, dict) else []

            for ish, sh in enumerate(shells):
                n_shell_total += 1

                if not isinstance(sh, dict):
                    continue

                if not sh.get("valid", False):
                    continue

                n_used = int(sh.get("N_used", sh.get("N", 0)))
                if n_used < int(min_shell_particles):
                    continue

                n_shell_valid += 1

                fig_aff, fig_dir = _get_affine_figure_block(sh)

                Omega_hat = _safe_matrix_from_dict(fig_aff, "Omega_hat")
                H_hat = _safe_matrix_from_dict(fig_aff, "H_hat")
                eta = _safe_matrix_from_dict(fig_aff, "eta")

                Pi_aff = _safe_matrix_from_dict(fig_aff, "Pi")
                Pi_dir = _safe_matrix_from_dict(fig_dir, "Pi")

                has_aff_decomp = (
                    np.any(np.isfinite(Omega_hat))
                    and np.any(np.isfinite(H_hat))
                    and np.any(np.isfinite(eta))
                )
                if has_aff_decomp:
                    n_affinite += 1

                # Vector form using the three independent off-diagonal components
                pO = np.array([
                    Omega_hat[0, 1],
                    Omega_hat[0, 2],
                    Omega_hat[1, 2],
                ], dtype=float) * unit_factor

                pH = np.array([
                    eta[0, 1] * H_hat[0, 1],
                    eta[0, 2] * H_hat[0, 2],
                    eta[1, 2] * H_hat[1, 2],
                ], dtype=float) * unit_factor

                nO = np.linalg.norm(pO)
                nH = np.linalg.norm(pH)

                if (
                    np.all(np.isfinite(pO))
                    and np.all(np.isfinite(pH))
                    and np.isfinite(nO)
                    and np.isfinite(nH)
                    and nO > 0
                    and nH > 0
                ):
                    cos_OH = float(np.dot(pO, pH) / (nO * nH))
                    cos_OH = float(np.clip(cos_OH, -1.0, 1.0))
                else:
                    cos_OH = np.nan

                base = {
                    "SubhaloID": sid,
                    "GroupID": gid,
                    "CenID": cen_id,
                    "IsCentral": is_central,
                    "r_over_r200c": r_over_r200c,
                    "shell_method": shell_method,
                    "shell": int(ish),
                    "N_shell_particles": n_used,
                    "cos_OmegaH": cos_OH,
                    "PiOmega_norm": float(nO) if np.isfinite(nO) else np.nan,
                    "PiH_norm": float(nH) if np.isfinite(nH) else np.nan,
                    "has_affine_decomposition": bool(has_aff_decomp),
                }

                for comp, i, j in comps:
                    Pi_Omega = Omega_hat[i, j] * unit_factor
                    Pi_H = eta[i, j] * H_hat[i, j] * unit_factor
                    Pi_aff_comp = Pi_aff[i, j] * unit_factor
                    Pi_dir_comp = Pi_dir[i, j] * unit_factor

                    denom = abs(Pi_Omega) + abs(Pi_H)
                    if np.isfinite(denom) and denom > 0:
                        fO = abs(Pi_Omega) / denom
                        fH = abs(Pi_H) / denom
                    else:
                        fO = np.nan
                        fH = np.nan

                    row = dict(base)
                    row.update({
                        "component": comp,
                        "i": int(i),
                        "j": int(j),
                        "PiOmega_component": float(Pi_Omega) if np.isfinite(Pi_Omega) else np.nan,
                        "PiH_component": float(Pi_H) if np.isfinite(Pi_H) else np.nan,
                        "Pi_aff_component": float(Pi_aff_comp) if np.isfinite(Pi_aff_comp) else np.nan,
                        "Pi_direct_dI_component": float(Pi_dir_comp) if np.isfinite(Pi_dir_comp) else np.nan,
                        "fOmega_abs": float(fO) if np.isfinite(fO) else np.nan,
                        "fH_abs": float(fH) if np.isfinite(fH) else np.nan,
                    })
                    rows.append(row)

    out = pd.DataFrame(rows)

    if verbose:
        print("[Pi diagnostic table]")
        print(f"  total shell entries scanned      : {n_shell_total}")
        print(f"  valid shell entries kept         : {n_shell_valid}")
        print(f"  shell entries with affine blocks : {n_affinite}")
        print(f"  diagnostic table rows            : {len(out)}")
        if len(out):
            print("\nFinite counts:")
            for c in [
                "PiOmega_component",
                "PiH_component",
                "Pi_aff_component",
                "Pi_direct_dI_component",
                "fOmega_abs",
                "fH_abs",
                "cos_OmegaH",
            ]:
                print(f"  {c:24s}: {np.isfinite(pd.to_numeric(out[c], errors='coerce')).sum()}/{len(out)}")

    return out


def plot_dw_alignment_distribution(data_cos, ax=None, title=""):
    """
    Draw DW fit for the alignment distribution of

        cos(theta) = cos(Pi^Omega, Pi^H),

    but report the fitted alignment parameter

        mu = -2 arctan(kappa) / pi,

    with
        mu = 1   : perfect alignment,
        mu = 0   : random,
        mu = -1  : perfect perpendicularity.

    If no valid data are available, annotate the panel instead of raising.
    """
    data_cos = np.asarray(data_cos, dtype=float)
    data_cos = data_cos[np.isfinite(data_cos)]
    data_cos = np.clip(data_cos, -1.0, 1.0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 4.8))
    else:
        fig = ax.figure

    if data_cos.size == 0:
        ax.text(
            0.5,
            0.58,
            "No valid alignment data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12.0,
            color=COLOR_TEXT,
        )
        ax.text(
            0.5,
            0.38,
            r"Check whether $\Omega_{\rm hat}$, $H_{\rm hat}$, or $\eta$"
            "\nare missing from shell['figure_affine'].",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.5,
            color=COLOR_TEXT,
        )
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(
            r"$\cos\theta_{\Omega\mathcal{H}}"
            r"=\cos(\boldsymbol{\Pi}^{\Omega},\boldsymbol{\Pi}^{\mathcal{H}})$"
        )
        ax.set_ylabel("Probability density")
        ax.set_title(title, pad=10)
        ax.grid(alpha=0.20, lw=0.7, color=COLOR_GRID)
        return fig, ax, None

    # Symmetrized sample, following the original plotting convention
    cos_sym = np.concatenate([data_cos, -data_cos])

    fit_res = dw.fit(cos_sym)

    xs = np.linspace(-1.0, 1.0, 400)
    pdfs = dw._pdf(xs, fit_res["kappa"])
    pdf_p = dw._pdf(xs, fit_res["kappa"] + fit_res["kappa_error"])
    pdf_m = dw._pdf(xs, fit_res["kappa"] - fit_res["kappa_error"])

    mu_fit = float(fit_res["mu"])
    mu_err = float(fit_res["mu_error"])

    ax.hist(
        cos_sym,
        bins=50,
        density=True,
        histtype="step",
        color=COLOR_HIST,
        lw=1.4,
        label="symmetrized shells",
    )

    ax.plot(xs, pdfs, color=COLOR_DW, lw=2.0, label="DW fit")
    ax.fill_between(
        xs, pdf_m, pdf_p,
        color=COLOR_BAND,
        alpha=0.28,
        label=r"fit uncertainty",
    )

    ax.axvline(0.0, color="0.35", lw=0.9, ls="--", alpha=0.75)

    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel(
        r"$\cos\theta_{\Omega\mathcal{H}}"
        r"=\cos(\boldsymbol{\Pi}^{\Omega},\boldsymbol{\Pi}^{\mathcal{H}})$"
    )
    ax.set_ylabel("Probability density")
    ax.set_title(title, pad=10)
    ax.grid(alpha=0.20, lw=0.7, color=COLOR_GRID)

    txt = (
        rf"$N_{{\rm shell}}={data_cos.size}$" + "\n" +
        rf"$\mu={mu_fit:.3f}\pm{mu_err:.3f}$"
    )

    ax.text(
        0.04, 0.96, txt,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9.4,
        color=COLOR_TEXT,
        bbox=dict(
            facecolor="white",
            edgecolor="0.85",
            alpha=0.90,
            boxstyle="round,pad=0.28",
        ),
    )

    ax.legend(frameon=False, loc="upper right")

    return fig, ax, fit_res


def _style_violin_parts(parts, color):
    """
    Apply publication-style colouring to matplotlib violin parts.
    """
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("0.15")
        body.set_linewidth(0.8)
        body.set_alpha(0.82)

    for key in ["cbars", "cmins", "cmaxes", "cmedians"]:
        if key in parts:
            parts[key].set_color("0.15")
            parts[key].set_linewidth(1.0 if key == "cmedians" else 0.8)


def plot_component_fraction_violin_panel(df_comp, ax=None, title=""):
    """
    Violin plot for one component: Omega fraction vs H fraction.

    If no valid fraction data are available, annotate the panel instead of raising.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.6, 4.8))
    else:
        fig = ax.figure

    fO = pd.to_numeric(df_comp.get("fOmega_abs", pd.Series([], dtype=float)), errors="coerce").to_numpy(dtype=float)
    fH = pd.to_numeric(df_comp.get("fH_abs", pd.Series([], dtype=float)), errors="coerce").to_numpy(dtype=float)

    fO = fO[np.isfinite(fO)]
    fH = fH[np.isfinite(fH)]

    if (fO.size == 0) or (fH.size == 0):
        ax.text(
            0.5,
            0.55,
            "No valid fraction data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color=COLOR_TEXT,
        )
        ax.text(
            0.5,
            0.36,
            "This component has no finite\n"
            r"$f^\Omega_{ij}$ or $f^{\mathcal{H}}_{ij}$.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color=COLOR_TEXT,
        )
        ax.set_xlim(0.4, 2.6)
        ax.set_ylim(-0.02, 1.08)
        ax.set_xticks([1.0, 2.0])
        ax.set_xticklabels([r"$\Omega$", r"$\mathcal{H}$"])
        ax.set_title(title, pad=10)
        ax.grid(axis="y", alpha=0.20, lw=0.7, color=COLOR_GRID)
        ax.set_axisbelow(True)
        return fig, ax

    pos_O = 1.0
    pos_H = 2.0

    parts_O = ax.violinplot(
        [fO],
        positions=[pos_O],
        widths=0.72,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    _style_violin_parts(parts_O, COLOR_OMEGA)

    parts_H = ax.violinplot(
        [fH],
        positions=[pos_H],
        widths=0.72,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    _style_violin_parts(parts_H, COLOR_H)

    # Add 16--84 percentile bars + median points
    for xpos, vals in [(pos_O, fO), (pos_H, fH)]:
        q16, q50, q84 = np.nanpercentile(vals, [16, 50, 84])
        ax.vlines(xpos, q16, q84, color="0.10", lw=2.0, zorder=4)
        ax.scatter([xpos], [q50], s=22, color="white", edgecolor="0.10", zorder=5)

        ax.text(
            xpos, 1.025,
            rf"$N_{{\rm shell}}={len(vals)}$",
            ha="center", va="bottom",
            fontsize=8.8,
            color=COLOR_TEXT,
        )

    ax.set_xlim(0.4, 2.6)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xticks([pos_O, pos_H])
    ax.set_xticklabels([r"$\Omega$", r"$\mathcal{H}$"])
    ax.set_title(title, pad=10)
    ax.grid(axis="y", alpha=0.20, lw=0.7, color=COLOR_GRID)
    ax.set_axisbelow(True)

    return fig, ax


# ------------------------------------------------------------
# Build diagnostics table
# ------------------------------------------------------------
pi_diag_df = build_pi_alignment_fraction_table(
    run,
    shell_method_filter=SHELL_METHOD_FILTER,
    min_shell_particles=MIN_SHELL_PARTICLES,
    verbose=True,
)

if SAVE_TABLE:
    pi_diag_df.to_csv(OUTDIR / "pi_alignment_fraction_diagnostics.csv", index=False)
    print("Saved table:", OUTDIR / "pi_alignment_fraction_diagnostics.csv")

if len(pi_diag_df) == 0:
    raise RuntimeError("No shell diagnostics were built from run['results'].")

# One row per shell for the DW alignment panel
mu_df = (
    pi_diag_df[
        ["SubhaloID", "shell_method", "shell", "cos_OmegaH"]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

cos_vals = pd.to_numeric(mu_df["cos_OmegaH"], errors="coerce").to_numpy(dtype=float)
cos_vals = cos_vals[np.isfinite(cos_vals)]

print(f"Valid cos_OmegaH values for DW fit: {cos_vals.size}")

# %% code cell 22
# ------------------------------------------------------------
# Plot: one DW panel + three violin panels
# ------------------------------------------------------------
fig = plt.figure(figsize=(15.6, 5.1))
gs = fig.add_gridspec(
    1, 4,
    width_ratios=[1.45, 1.0, 1.0, 1.0],
    wspace=0.28,
)

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2], sharey=ax1)
ax3 = fig.add_subplot(gs[0, 3], sharey=ax1)

# (a) DW alignment distribution
_, _, fit_res = plot_dw_alignment_distribution(
    cos_vals,
    ax=ax0,
    title=r"(a) DW alignment of $\boldsymbol{\Pi}^{\Omega}$ and $\boldsymbol{\Pi}^{\mathcal{H}}$",
)

# (b)(c)(d) violin panels
for ax, comp, pretty, tag in [
    (ax1, "01", "0,1", "(b)"),
    (ax2, "02", "0,2", "(c)"),
    (ax3, "12", "1,2", "(d)"),
]:
    sub = pi_diag_df[pi_diag_df["component"].eq(comp)].copy()
    plot_component_fraction_violin_panel(
        sub,
        ax=ax,
        title=rf"{tag} component $({pretty})$",
    )

# Y label only on first violin panel
ax1.set_ylabel("Absolute contribution fraction")
ax2.set_ylabel("")
ax3.set_ylabel("")

# Shared explanatory legend for violins
legend_handles = [
    Line2D([0], [0], color=COLOR_OMEGA, lw=8, alpha=0.82, label=r"$f^\Omega_{ij}$"),
    Line2D([0], [0], color=COLOR_H, lw=8, alpha=0.82, label=r"$f^{\mathcal{H}}_{ij}$"),
]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.71, 1.02),
    frameon=False,
    ncol=2,
    handlelength=1.8,
    columnspacing=1.8,
)

fig.suptitle(
    r"Alignment and component-wise contribution distributions of "
    r"$\boldsymbol{\Pi}^{\Omega}$ and $\boldsymbol{\Pi}^{\mathcal{H}}$",
    y=1.08,
    fontsize=14.5,
)

fig.subplots_adjust(left=0.060, right=0.995, bottom=0.16, top=0.84)

if SAVE_FIG:
    fig_path = OUTDIR / "pi_dw_alignment_and_component_violins.png"
    fig.savefig(fig_path, dpi=260, bbox_inches="tight")
    print("Saved figure:", fig_path)

plt.show()

display(pi_diag_df.head())

# %% [markdown] cell 23
# ## Shell-wise matrices for the example

# %% code cell 24

# Inspect H, Omega and Pi for radial shells.
radial_analysis = radial_block['analysis']
H = hd_tng.hd.stack_shell_quantity(radial_analysis, ['H'])
Omega = hd_tng.hd.stack_shell_quantity(radial_analysis, ['Omega'])
Pi = hd_tng.hd.stack_shell_quantity(radial_analysis, ['figure', 'Pi']) * hd_tng.KM_S_PER_KPC_TO_GYR_INV

print('H shape:', H.shape, 'Omega shape:', Omega.shape, 'Pi shape:', Pi.shape)
for i in range(len(radial_analysis['shells'])):
    print('\nShell', i)
    print('H [km/s/kpc] =\n', H[i])
    print('Omega [km/s/kpc] =\n', Omega[i])
    print('Pi [Gyr^-1] =\n', Pi[i])

# %% code cell 25
SNAP_TRACK

# %% code cell 26
# fig, ax, records, save_path = plot_subhalo_orbitplane_overlay(
#     1,
#     [99, 84, 72, 67, 59, 50, 40, 33],
# )

# %% [markdown] cell 27
# ## Explicit cleanup

# %% code cell 28

# hd_tng.cleanup_open_catalogs()
print('Temporary TNG API files opened through hd_tng have been deleted.')

# %% code cell 29

# %% code cell 30

# %% code cell 31

# %% code cell 32
