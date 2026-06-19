#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tng_layered_shape_tide.py

TNG-only *layered* analysis (ellipsoidal shells), separated from the unified
global analysis in global_shape_tide.py.

This script performs:
  - Tidal tensors at subhalo centers (FoF mass-based DM Poisson solve)
  - Global DM/stars inertia + kinematics + variances (same as global script)
  - DM layered (ellipsoidal fraction shells) inertia tensors I_shell, dI_shell
    built from *the subhalo's own DM particles*, using the converged global DM
    ellipsoid as the reference geometry, following the CGA logic.

CLI
---
python tng_layered_shape_tide.py --snap 99 --base /path/to/tng_data --out tng_layered_s099.hdf5

Knobs (selection, grids, shell setup) are grouped at the top.
"""

from __future__ import annotations

import os
import time
import math
import gc
import logging
import argparse
import multiprocessing as mp
import sys
from functools import partial

import numpy as np

if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print(
        """usage: tng_layered_shape_tide.py --snap SNAP --base BASE --out OUT [options]

TNG layered shape-tide pipeline with ellipsoidal shells.

core options:
  -h, --help
  --snap SNAP
  --base BASE
  --out OUT
"""
    )
    raise SystemExit(0)

import h5py
from tqdm import tqdm
import psutil

# --------------------------
# User configuration (EDIT)
# --------------------------
DEFAULT_TNG_ROOT = "/cosma8/data/dp203/dc-wang17/TNG/tng_data/"
HUBBLE_h = 0.6774

PTYPE_DM = 1
PTYPE_STAR = 4

# Selection (same as global script; edit here if desired)
TNG_MAG_LIM = 20.0
TNG_MIN_SUB_MASS_Msunh = 10.0
TNG_MIN_STELLAR_INRAD_1e10Msunh = 1.0 / 1000.0
TNG_MIN_NSTAR = 150

# Shape knobs
DM_PERCENTILE = 98.0
STAR_PERCENTILE = 100.0
MAX_ITER = 100
TOL = 5e-2

# Tidal grid knobs
GRID_SIZE = 128
PADDING   = 0.20
SOFTENING = 1.0
TNG_DM_FIXED_MASS_MSUN = 5.9e7

# Layering: r_fracs in [RFRAC_MIN, RFRAC_MAX]
NBINS      = 31
RFRAC_MIN  = 0.01
RFRAC_MAX  = 1.0
SPACING    = "linear"   # "linear" or "log"

# Chunking / checkpointing
PART_NSUB = int(os.environ.get("TNG_LAYER_PART_NSUB", 20000))
PARTS_PER_RUN = int(os.environ.get("TNG_LAYER_PARTS_PER_RUN", 0))
MEM_PER_WORKER_GB = float(os.environ.get("TNG_LAYER_MEM_PER_WORKER_GB", 3.0))

# --------------------------
# Logging
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tng_layered_shape_tide")

# --------------------------
# Imports
# --------------------------
try:
    from illustris_python import groupcat, snapshot
except Exception as e:
    raise RuntimeError("illustris_python is required for this script.") from e

try:
    from ia_analysis.tides.tidal_field import compute_gravitational_potential, PotentialInterpolator
except Exception as e:
    raise RuntimeError("tidal_field.compute_gravitational_potential + PotentialInterpolator required.") from e

try:
    from ia_analysis.shapes.shape import ShapeKin, compute_axis
except Exception as e:
    raise RuntimeError("Need ShapeKin + compute_axis from your new shape.py.") from e

try:
    from ia_analysis.shapes.Iana import chiSO
except Exception as e:
    raise RuntimeError("Need chiSO from Iana.py.") from e


# ============================================================
# Helpers
# ============================================================

def ensure_dir(p: str) -> None:
    d = os.path.dirname(os.path.abspath(p))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def h5_is_complete(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with h5py.File(path, "r") as f:
            return int(f.attrs.get("complete", 0)) == 1
    except Exception:
        return False


def part_path(out_base: str, ip: int) -> str:
    return f"{out_base}.part{ip:03d}.hdf5"


def auto_max_workers() -> int:
    cpu = os.cpu_count() or 1
    avail_gb = psutil.virtual_memory().available / 1024**3
    max_by_mem = max(1, int(avail_gb // MEM_PER_WORKER_GB))
    max_auto = min(cpu, max_by_mem)
    return int(os.environ.get("TNG_LAYER_MAX_WORKERS", max_auto))


def build_r_fracs() -> np.ndarray:
    if SPACING.lower() == "log":
        return np.logspace(np.log10(RFRAC_MIN), np.log10(RFRAC_MAX), NBINS).astype(np.float64)
    return np.linspace(RFRAC_MIN, RFRAC_MAX, NBINS).astype(np.float64)


def compute_group_tidal(base_path: str, snap: int, gid: int, centers: np.ndarray) -> np.ndarray:
    dm = snapshot.loadHalo(base_path, snap, gid, partType="dm", fields=["Coordinates"])
    dm_pos = dm["Coordinates"]
    if dm_pos.shape[0] == 0:
        return np.zeros((centers.shape[0], 3, 3), dtype=np.float32)

    dm_mass = np.full(dm_pos.shape[0], TNG_DM_FIXED_MASS_MSUN, dtype=float)
    res = compute_gravitational_potential(
        positions=dm_pos,
        masses=dm_mass,
        grid_size=GRID_SIZE,
        boundary_padding=PADDING,
        softening=SOFTENING,
        G=1.0,
    )
    interp = PotentialInterpolator(res)
    tidal = np.array([interp.tidal_tensor(c) for c in centers], dtype=np.float32)
    return tidal


def safe_global_shape(pos, vel, mass, center, percentile, sid):
    """
    Global shape + kinematics + variances for one component.
    """
    out = {
        "I": np.full((3,3), np.nan, np.float64),
        "dI": np.full((3,3), np.nan, np.float64),
        "abc": np.full(3, np.nan, np.float64),
        "axes_major": np.full(3, np.nan, np.float64),
        "axes_medium": np.full(3, np.nan, np.float64),
        "axes_minor": np.full(3, np.nan, np.float64),
        "chi": np.nan, "q": np.nan, "s": np.nan,
        "L": np.full(3, np.nan, np.float64),
        "Lhat": np.full(3, np.nan, np.float64),
        "kappa_rot": np.nan, "K_tot": np.nan, "K_rot": np.nan, "N_used_kin": 0,
        "Neff": np.nan,
        "var_evals": np.full(3, np.nan, np.float64),
        "mean_theta2": np.full(3, np.nan, np.float64),
        "var_cos": np.full(3, np.nan, np.float64),
        "mask": None,
    }
    if pos is None or pos.shape[0] < 3:
        return out

    sk = ShapeKin(particles=pos, masses=mass, velocities=vel, accelerations=None, Pos=center)
    sk.run_shape(percentile=percentile, max_iter=MAX_ITER, tol=TOL, return_dI=(vel is not None),
                 return_ddI=False, r_ell=None, TheSubHaloID=sid)
    out["I"] = sk.I
    if sk.dI is not None:
        out["dI"] = sk.dI
    out["mask"] = sk.mask

    abc, evecs = compute_axis(sk.I)
    out["abc"] = abc
    out["axes_major"], out["axes_medium"], out["axes_minor"] = evecs[:,0], evecs[:,1], evecs[:,2]

    try:
        ch, q, s = chiSO(sk.I)
        out["chi"], out["q"], out["s"] = float(ch), float(q), float(s)
    except Exception:
        pass

    # variances on SAME subset
    stats = sk.var_eig(normalize=True)
    out["Neff"] = float(stats.get("Neff", np.nan))
    out["var_evals"] = stats.get("var_evals", out["var_evals"])
    out["mean_theta2"] = stats.get("mean_theta2", out["mean_theta2"])
    out["var_cos"] = stats.get("var_cos", out["var_cos"])

    if vel is not None:
        L, Lhat = sk.L(v_ref=None)
        out["L"], out["Lhat"] = L, Lhat
        kap = sk.kappa(v_ref=None, Lhat=Lhat)
        out["kappa_rot"] = float(kap.get("kappa_rot", np.nan))
        out["K_tot"] = float(kap.get("K_tot", np.nan))
        out["K_rot"] = float(kap.get("K_rot", np.nan))
        out["N_used_kin"] = int(kap.get("N_used", 0))

    return out


def dm_shells_from_global(dm_pos: np.ndarray, dm_vel: np.ndarray | None, center: np.ndarray, abc_dm: np.ndarray, axes_dm: np.ndarray, r_fracs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build ellipsoidal shells using the *global DM ellipsoid* as the reference.

    Geometry:
      rho^2 = (x_b/a)^2 + (y_b/b)^2 + (z_b/c)^2   in body frame of global ellipsoid.

    For each r_frac = s in r_fracs:
      select rho^2 <= s^2  (inner ellipsoid scaled by s),
      then compute raw I and dI directly on that subset (no further I_iters),
      returning NaNs if subset too small.

    Outputs:
      I_bins  : (Nv,3,3)
      dI_bins : (Nv,3,3)
      n_bins  : (Nv,)
    """
    Nv = r_fracs.size
    I_bins = np.full((Nv, 3, 3), np.nan, dtype=np.float32)
    dI_bins = np.full((Nv, 3, 3), np.nan, dtype=np.float32)
    n_bins = np.zeros((Nv,), dtype=np.int32)

    if dm_pos is None or dm_pos.shape[0] < 3:
        return I_bins, dI_bins, n_bins

    # Rotate into body frame (columns of axes_dm are eigenvectors)
    X = dm_pos - center[None, :]
    Xb = X @ axes_dm  # (N,3) where axes_dm columns are basis vectors
    a, b, c = float(abc_dm[0]), float(abc_dm[1]), float(abc_dm[2])
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c) and a > 0 and b > 0 and c > 0):
        return I_bins, dI_bins, n_bins

    rho2 = (Xb[:,0]/a)**2 + (Xb[:,1]/b)**2 + (Xb[:,2]/c)**2

    for j, s in enumerate(r_fracs):
        if not np.isfinite(s) or s <= 0.0:
            continue
        sel = rho2 <= (s*s)
        n = int(np.count_nonzero(sel))
        n_bins[j] = n
        if n < 3:
            continue

        # Raw inertia with equal masses
        P = X[sel]
        I = np.einsum("ij,ik->jk", P, P).astype(np.float32)
        I_bins[j] = I

        if dm_vel is not None:
            V = dm_vel[sel]
            dI = (np.einsum("ij,ik->jk", P, V) + np.einsum("ij,ik->jk", V, P)).astype(np.float32)
            dI_bins[j] = dI

    return I_bins, dI_bins, n_bins


# ============================================================
# Main pipeline
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TNG-only layered (ellipsoidal shell) analysis.")
    p.add_argument("--snap", required=True, type=int, help="Snapshot number.")
    p.add_argument("--base", default=None, type=str, help="TNG base path (tng_data).")
    p.add_argument("--out", default=None, type=str, help="Output HDF5.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    snap = int(args.snap)
    base_path = DEFAULT_TNG_ROOT if args.base is None else args.base
    out_final = args.out or f"tng_layered_s{snap:03d}.hdf5"

    max_workers = auto_max_workers()
    avail_gb = psutil.virtual_memory().available / 1024**3
    logger.info(f"[env] CPU={os.cpu_count() or 1} avail_mem={avail_gb:.1f} GB max_workers={max_workers}")

    # r_fracs grid
    r_fracs = build_r_fracs()
    logger.info(f"[cfg] NBINS={NBINS} r_fracs in [{r_fracs.min():.3g}, {r_fracs.max():.3g}] ({SPACING})")

    halos = groupcat.loadHalos(base_path, snap, fields=["GroupFirstSub", "GroupNsubs"])
    subhalos = groupcat.loadSubhalos(
        base_path, snap,
        fields=[
            "SubhaloGrNr",
            "SubhaloPos",
            "SubhaloMass",
            "SubhaloLenType",
            "SubhaloMassInRadType",
            "SubhaloStellarPhotometrics",
        ],
    )

    # selection
    msub = np.asarray(subhalos["SubhaloMass"], dtype=float) / float(HUBBLE_h)
    nstar = subhalos["SubhaloLenType"][:, PTYPE_STAR].astype(np.int64)
    mstar_inrad = np.asarray(subhalos["SubhaloMassInRadType"][:, PTYPE_STAR], dtype=float) / float(HUBBLE_h)
    BAND_R_INDEX = int(os.environ.get("TNG_RBAND_INDEX", 5))
    phot = np.asarray(subhalos["SubhaloStellarPhotometrics"], dtype=float)
    Mr = phot[:, BAND_R_INDEX] if phot.ndim == 2 and phot.shape[1] > BAND_R_INDEX else np.full(msub.shape, np.nan)

    sel = (
        np.isfinite(msub) & (msub > TNG_MIN_SUB_MASS_Msunh) &
        np.isfinite(Mr) & (Mr < -TNG_MAG_LIM) &
        np.isfinite(mstar_inrad) & (mstar_inrad > TNG_MIN_STELLAR_INRAD_1e10Msunh) &
        (nstar >= TNG_MIN_NSTAR)
    )

    SubIDs_sel = np.nonzero(sel)[0].astype(np.int32)
    if SubIDs_sel.size == 0:
        raise RuntimeError("No subhalos selected with current TNG cuts.")
    GroupID_sel = subhalos["SubhaloGrNr"][sel].astype(np.int32)
    centers_all = subhalos["SubhaloPos"][sel].astype(np.float32)
    CenSubhaloID_sel = halos["GroupFirstSub"][GroupID_sel].astype(np.int32)

    N_sel = SubIDs_sel.size
    logger.info(f"[TNG layered] selected subhalos: {N_sel}")

    out_base = out_final.replace(".hdf5", "")
    n_parts = int(math.ceil(N_sel / PART_NSUB))
    logger.info(f"[TNG layered] PART_NSUB={PART_NSUB} => n_parts={n_parts}")

    missing = [ip for ip in range(n_parts) if not h5_is_complete(part_path(out_base, ip))]
    if not missing:
        logger.info("[TNG layered] all parts complete; proceed to merge.")
    else:
        logger.info(f"[TNG layered] missing parts: {len(missing)}/{n_parts}")

    parts_done = 0
    for ip in missing:
        if PARTS_PER_RUN > 0 and parts_done >= PARTS_PER_RUN:
            logger.info(f"[TNG layered] PARTS_PER_RUN={PARTS_PER_RUN}: stopping early.")
            break

        pfile = part_path(out_base, ip)
        i0 = ip * PART_NSUB
        i1 = min((ip + 1) * PART_NSUB, N_sel)
        n_local = i1 - i0

        logger.info(f"[part {ip+1}/{n_parts}] slice selected[{i0}:{i1}] (N={n_local})")
        SubIDs_part = SubIDs_sel[i0:i1]
        GroupID_part = GroupID_sel[i0:i1]
        centers_part = centers_all[i0:i1]
        CenSub_part = CenSubhaloID_sel[i0:i1]

        # Allocate part outputs
        Z33 = np.zeros((n_local, 3, 3), dtype=np.float32)
        Z3 = np.zeros((n_local, 3), dtype=np.float32)
        Z1 = np.zeros((n_local,), dtype=np.float32)
        Zi = np.zeros((n_local,), dtype=np.int32)

        tidal_grp = Z33.copy()
        tidal_tot = Z33.copy()

        # Global shapes (dm + star)
        I_dm = Z33.copy(); dI_dm = Z33.copy()
        abc_dm = Z3.copy()
        axes_dm_major = Z3.copy(); axes_dm_medium = Z3.copy(); axes_dm_minor = Z3.copy()
        chi_dm = Z1.copy(); q_dm = Z1.copy(); s_dm = Z1.copy()
        L_dm = Z3.copy(); Lhat_dm = Z3.copy()
        kappa_dm = Z1.copy(); Ktot_dm = Z1.copy(); Krot_dm = Z1.copy(); Nkin_dm = Zi.copy()
        Neff_dm = Z1.copy(); var_evals_dm = Z3.copy(); mean_theta2_dm = Z3.copy(); var_cos_dm = Z3.copy()

        I_st = Z33.copy(); dI_st = Z33.copy()
        abc_st = Z3.copy()
        axes_st_major = Z3.copy(); axes_st_medium = Z3.copy(); axes_st_minor = Z3.copy()
        chi_st = Z1.copy(); q_st = Z1.copy(); s_st = Z1.copy()
        L_st = Z3.copy(); Lhat_st = Z3.copy()
        kappa_st = Z1.copy(); Ktot_st = Z1.copy(); Krot_st = Z1.copy(); Nkin_st = Zi.copy()
        Neff_st = Z1.copy(); var_evals_st = Z3.copy(); mean_theta2_st = Z3.copy(); var_cos_st = Z3.copy()

        # Layered DM shells
        Nv = r_fracs.size
        I_dm_bins = np.full((n_local, Nv, 3, 3), np.nan, dtype=np.float32)
        dI_dm_bins = np.full((n_local, Nv, 3, 3), np.nan, dtype=np.float32)
        n_dm_bins = np.zeros((n_local, Nv), dtype=np.int32)

        # (1) Tidal tensors per FoF group (groups in this part only)
        groups_in_part = np.unique(GroupID_part).astype(int).tolist()
        group_to_local = {}
        for j, gid in enumerate(GroupID_part.tolist()):
            group_to_local.setdefault(int(gid), []).append(j)
        for g in list(group_to_local.keys()):
            group_to_local[g] = np.asarray(group_to_local[g], dtype=np.int64)

        logger.info(f"[part {ip:03d}] tidal groups: {len(groups_in_part)}")
        with mp.get_context("fork").Pool(processes=max_workers) as pool:
            jobs = []
            for gid in groups_in_part:
                idx = group_to_local[gid]
                centers = centers_part[idx].astype(float)
                jobs.append(pool.apply_async(compute_group_tidal, args=(base_path, snap, int(gid), centers)))
            for gid, job in zip(groups_in_part, tqdm(jobs, desc=f"[part {ip:03d}] tidal", leave=False)):
                idx = group_to_local[int(gid)]
                tidal_grp[idx] = job.get()
        tidal_tot[:] = tidal_grp  # mass-based only

        # (2) Per-subhalo global shapes + shells
        logger.info(f"[part {ip:03d}] shapes + shells")
        for j, sid in enumerate(tqdm(SubIDs_part.tolist(), desc=f"[part {ip:03d}] subhalo", leave=False)):
            center = centers_part[j].astype(float)

            # DM + stars particles (SUBHALO member particles)
            dm = snapshot.loadSubhalo(base_path, snap, int(sid), partType="dm", fields=["Coordinates", "Velocities"])
            st = snapshot.loadSubhalo(base_path, snap, int(sid), partType="stars", fields=["Coordinates", "Velocities", "Masses"])

            dm_pos = dm.get("Coordinates", np.empty((0,3), float))
            dm_vel = dm.get("Velocities", None)

            st_pos = st.get("Coordinates", np.empty((0,3), float))
            st_vel = st.get("Velocities", None)
            st_mass = st.get("Masses", None)

            dm_g = safe_global_shape(dm_pos, dm_vel, None, center, DM_PERCENTILE, int(sid))
            st_g = safe_global_shape(st_pos, st_vel, st_mass, center, STAR_PERCENTILE, int(sid))

            # Write globals
            I_dm[j] = dm_g["I"].astype(np.float32)
            dI_dm[j] = dm_g["dI"].astype(np.float32)
            abc_dm[j] = dm_g["abc"].astype(np.float32)

            # axes matrix for shells: columns are eigenvectors
            axes_mat = np.column_stack([dm_g["axes_major"], dm_g["axes_medium"], dm_g["axes_minor"]]).astype(np.float64)
            axes_dm_major[j] = dm_g["axes_major"].astype(np.float32)
            axes_dm_medium[j] = dm_g["axes_medium"].astype(np.float32)
            axes_dm_minor[j] = dm_g["axes_minor"].astype(np.float32)

            chi_dm[j] = np.float32(dm_g["chi"]); q_dm[j] = np.float32(dm_g["q"]); s_dm[j] = np.float32(dm_g["s"])
            L_dm[j] = dm_g["L"].astype(np.float32); Lhat_dm[j] = dm_g["Lhat"].astype(np.float32)
            kappa_dm[j] = np.float32(dm_g["kappa_rot"]); Ktot_dm[j] = np.float32(dm_g["K_tot"]); Krot_dm[j] = np.float32(dm_g["K_rot"]); Nkin_dm[j] = np.int32(dm_g["N_used_kin"])
            Neff_dm[j] = np.float32(dm_g["Neff"])
            var_evals_dm[j] = dm_g["var_evals"].astype(np.float32)
            mean_theta2_dm[j] = dm_g["mean_theta2"].astype(np.float32)
            var_cos_dm[j] = dm_g["var_cos"].astype(np.float32)

            I_st[j] = st_g["I"].astype(np.float32)
            dI_st[j] = st_g["dI"].astype(np.float32)
            abc_st[j] = st_g["abc"].astype(np.float32)
            axes_st_major[j] = st_g["axes_major"].astype(np.float32)
            axes_st_medium[j] = st_g["axes_medium"].astype(np.float32)
            axes_st_minor[j] = st_g["axes_minor"].astype(np.float32)
            chi_st[j] = np.float32(st_g["chi"]); q_st[j] = np.float32(st_g["q"]); s_st[j] = np.float32(st_g["s"])
            L_st[j] = st_g["L"].astype(np.float32); Lhat_st[j] = st_g["Lhat"].astype(np.float32)
            kappa_st[j] = np.float32(st_g["kappa_rot"]); Ktot_st[j] = np.float32(st_g["K_tot"]); Krot_st[j] = np.float32(st_g["K_rot"]); Nkin_st[j] = np.int32(st_g["N_used_kin"])
            Neff_st[j] = np.float32(st_g["Neff"])
            var_evals_st[j] = st_g["var_evals"].astype(np.float32)
            mean_theta2_st[j] = st_g["mean_theta2"].astype(np.float32)
            var_cos_st[j] = st_g["var_cos"].astype(np.float32)

            # DM shells using global ellipsoid
            I_bins, dI_bins, n_bins = dm_shells_from_global(dm_pos, dm_vel, center, dm_g["abc"], axes_mat, r_fracs)
            I_dm_bins[j] = I_bins
            dI_dm_bins[j] = dI_bins
            n_dm_bins[j] = n_bins

        # Write part file
        ensure_dir(pfile)
        with h5py.File(pfile, "w") as f:
            f.attrs["creation_time"] = time.ctime()
            f.attrs["complete"] = 1
            f.attrs["sim_kind"] = "TNG"
            f.attrs["base_path"] = base_path
            f.attrs["snapshot"] = int(snap)
            f.attrs["part_index"] = int(ip)
            f.attrs["part_i0"] = int(i0)
            f.attrs["part_i1"] = int(i1)
            f.attrs["N_sel_total"] = int(N_sel)

            # config echo
            f.attrs["GRID_SIZE"] = GRID_SIZE
            f.attrs["PADDING"] = PADDING
            f.attrs["SOFTENING"] = SOFTENING
            f.attrs["DM_PERCENTILE"] = DM_PERCENTILE
            f.attrs["STAR_PERCENTILE"] = STAR_PERCENTILE
            f.attrs["TNG_RBAND_INDEX"] = BAND_R_INDEX

            f.create_dataset("r_fracs", data=r_fracs.astype(np.float32))

            f.create_dataset("SubhaloID", data=SubIDs_part.astype(np.int32))
            f.create_dataset("GroupID", data=GroupID_part.astype(np.int32))
            f.create_dataset("CenSubhaloID", data=CenSub_part.astype(np.int32))
            f.create_dataset("CenterPos", data=centers_part.astype(np.float32))

            f.create_dataset("tidal_grp", data=tidal_grp)
            f.create_dataset("tidal_tot", data=tidal_tot)

            # global dm/star
            gdm = f.create_group("dm")
            gdm.create_dataset("I", data=I_dm)
            gdm.create_dataset("dI", data=dI_dm)
            gdm.create_dataset("abc", data=abc_dm)
            gdm.create_dataset("axes_major", data=axes_dm_major)
            gdm.create_dataset("axes_medium", data=axes_dm_medium)
            gdm.create_dataset("axes_minor", data=axes_dm_minor)
            gdm.create_dataset("chi", data=chi_dm)
            gdm.create_dataset("q", data=q_dm)
            gdm.create_dataset("s", data=s_dm)
            gdm.create_dataset("L", data=L_dm)
            gdm.create_dataset("Lhat", data=Lhat_dm)
            gdm.create_dataset("kappa_rot", data=kappa_dm)
            gdm.create_dataset("K_tot", data=Ktot_dm)
            gdm.create_dataset("K_rot", data=Krot_dm)
            gdm.create_dataset("N_used_kin", data=Nkin_dm)
            gdm.create_dataset("Neff", data=Neff_dm)
            gdm.create_dataset("var_evals", data=var_evals_dm)
            gdm.create_dataset("mean_theta2", data=mean_theta2_dm)
            gdm.create_dataset("var_cos", data=var_cos_dm)

            gs = f.create_group("star")
            gs.create_dataset("I", data=I_st)
            gs.create_dataset("dI", data=dI_st)
            gs.create_dataset("abc", data=abc_st)
            gs.create_dataset("axes_major", data=axes_st_major)
            gs.create_dataset("axes_medium", data=axes_st_medium)
            gs.create_dataset("axes_minor", data=axes_st_minor)
            gs.create_dataset("chi", data=chi_st)
            gs.create_dataset("q", data=q_st)
            gs.create_dataset("s", data=s_st)
            gs.create_dataset("L", data=L_st)
            gs.create_dataset("Lhat", data=Lhat_st)
            gs.create_dataset("kappa_rot", data=kappa_st)
            gs.create_dataset("K_tot", data=Ktot_st)
            gs.create_dataset("K_rot", data=Krot_st)
            gs.create_dataset("N_used_kin", data=Nkin_st)
            gs.create_dataset("Neff", data=Neff_st)
            gs.create_dataset("var_evals", data=var_evals_st)
            gs.create_dataset("mean_theta2", data=mean_theta2_st)
            gs.create_dataset("var_cos", data=var_cos_st)

            # layered dm
            gl = f.create_group("dm_layers")
            gl.create_dataset("I_bins", data=I_dm_bins)
            gl.create_dataset("dI_bins", data=dI_dm_bins)
            gl.create_dataset("n_bins", data=n_dm_bins)

        parts_done += 1
        gc.collect()

    # Merge if complete
    if not all(h5_is_complete(part_path(out_base, ip)) for ip in range(n_parts)):
        logger.info("[merge] not all parts complete; exit (rerun continues).")
        return

    logger.info("[merge] all parts complete; merging final HDF5.")
    if os.path.exists(out_final):
        logger.warning(f"[merge] removing existing: {out_final}")
        os.remove(out_final)

    ensure_dir(out_final)
    with h5py.File(out_final, "w") as f:
        f.attrs["creation_time"] = time.ctime()
        f.attrs["complete"] = 1
        f.attrs["sim_kind"] = "TNG"
        f.attrs["base_path"] = base_path
        f.attrs["snapshot"] = int(snap)
        f.attrs["N_sel"] = int(N_sel)

        f.attrs["GRID_SIZE"] = GRID_SIZE
        f.attrs["PADDING"] = PADDING
        f.attrs["SOFTENING"] = SOFTENING
        f.attrs["DM_PERCENTILE"] = DM_PERCENTILE
        f.attrs["STAR_PERCENTILE"] = STAR_PERCENTILE
        f.attrs["TNG_MAG_LIM"] = TNG_MAG_LIM
        f.attrs["TNG_MIN_SUB_MASS_Msunh"] = TNG_MIN_SUB_MASS_Msunh
        f.attrs["TNG_MIN_STELLAR_INRAD_1e10Msunh"] = TNG_MIN_STELLAR_INRAD_1e10Msunh
        f.attrs["TNG_MIN_NSTAR"] = TNG_MIN_NSTAR
        f.attrs["TNG_RBAND_INDEX"] = BAND_R_INDEX

        f.create_dataset("r_fracs", data=r_fracs.astype(np.float32))
        f.create_dataset("SubhaloID", data=SubIDs_sel.astype(np.int32))
        f.create_dataset("GroupID", data=GroupID_sel.astype(np.int32))
        f.create_dataset("CenSubhaloID", data=CenSubhaloID_sel.astype(np.int32))
        f.create_dataset("CenterPos", data=centers_all.astype(np.float32))

        # preallocate and fill from parts
        chunks_33 = (min(1024, N_sel), 3, 3)
        f.create_dataset("tidal_grp", shape=(N_sel,3,3), dtype=np.float32, chunks=chunks_33)
        f.create_dataset("tidal_tot", shape=(N_sel,3,3), dtype=np.float32, chunks=chunks_33)

        def alloc_global(g):
            g.create_dataset("I", shape=(N_sel,3,3), dtype=np.float32, chunks=chunks_33)
            g.create_dataset("dI", shape=(N_sel,3,3), dtype=np.float32, chunks=chunks_33)
            chunks_3 = (min(1024, N_sel), 3)
            chunks_1 = (min(4096, N_sel),)
            g.create_dataset("abc", shape=(N_sel,3), dtype=np.float32, chunks=chunks_3)
            g.create_dataset("axes_major", shape=(N_sel,3), dtype=np.float32, chunks=chunks_3)
            g.create_dataset("axes_medium", shape=(N_sel,3), dtype=np.float32, chunks=chunks_3)
            g.create_dataset("axes_minor", shape=(N_sel,3), dtype=np.float32, chunks=chunks_3)
            g.create_dataset("chi", shape=(N_sel,), dtype=np.float32, chunks=chunks_1)
            g.create_dataset("q", shape=(N_sel,), dtype=np.float32, chunks=chunks_1)
            g.create_dataset("s", shape=(N_sel,), dtype=np.float32, chunks=chunks_1)
            g.create_dataset("L", shape=(N_sel,3), dtype=np.float32, chunks=chunks_3)
            g.create_dataset("Lhat", shape=(N_sel,3), dtype=np.float32, chunks=chunks_3)
            g.create_dataset("kappa_rot", shape=(N_sel,), dtype=np.float32, chunks=chunks_1)
            g.create_dataset("K_tot", shape=(N_sel,), dtype=np.float32, chunks=chunks_1)
            g.create_dataset("K_rot", shape=(N_sel,), dtype=np.float32, chunks=chunks_1)
            g.create_dataset("N_used_kin", shape=(N_sel,), dtype=np.int32, chunks=chunks_1)
            g.create_dataset("Neff", shape=(N_sel,), dtype=np.float32, chunks=chunks_1)
            g.create_dataset("var_evals", shape=(N_sel,3), dtype=np.float32, chunks=chunks_3)
            g.create_dataset("mean_theta2", shape=(N_sel,3), dtype=np.float32, chunks=chunks_3)
            g.create_dataset("var_cos", shape=(N_sel,3), dtype=np.float32, chunks=chunks_3)

        gdm = f.create_group("dm"); alloc_global(gdm)
        gs  = f.create_group("star"); alloc_global(gs)

        Nv = r_fracs.size
        chunks_bins = (min(256, N_sel), Nv, 3, 3)
        gl = f.create_group("dm_layers")
        gl.create_dataset("I_bins", shape=(N_sel,Nv,3,3), dtype=np.float32, chunks=chunks_bins)
        gl.create_dataset("dI_bins", shape=(N_sel,Nv,3,3), dtype=np.float32, chunks=chunks_bins)
        gl.create_dataset("n_bins", shape=(N_sel,Nv), dtype=np.int32, chunks=(min(512, N_sel), Nv))

        for ip in range(n_parts):
            pfile = part_path(out_base, ip)
            with h5py.File(pfile, "r") as pf:
                i0 = int(pf.attrs["part_i0"]); i1 = int(pf.attrs["part_i1"])
                f["tidal_grp"][i0:i1] = pf["tidal_grp"][...]
                f["tidal_tot"][i0:i1] = pf["tidal_tot"][...]
                for comp in ["dm", "star"]:
                    for key in pf[comp].keys():
                        f[f"{comp}/{key}"][i0:i1] = pf[f"{comp}/{key}"][...]
                f["dm_layers/I_bins"][i0:i1] = pf["dm_layers/I_bins"][...]
                f["dm_layers/dI_bins"][i0:i1] = pf["dm_layers/dI_bins"][...]
                f["dm_layers/n_bins"][i0:i1] = pf["dm_layers/n_bins"][...]

    logger.info(f"[done] {out_final}")


if __name__ == "__main__":
    main()
