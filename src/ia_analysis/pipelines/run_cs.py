#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_cs.py

ClusterSims driver script:
- Select subhalos by stellar particle number >= threshold (default: 30).
- Build tasks grouped by FoF GroupID (gid).
- Parallelize over gid-shards (ProcessPoolExecutor), where each worker calls
  global_cs.compute_many(...) on its shard.

Typical usage (COSMA):
    python run_cs.py --basepath /path/to/CS --snap 21 --nworker 16 --out /path/to/out_cs_s021.hdf5

Notes
-----
- The heavy work is performed inside global_cs_updated.py (tidal grids cached per gid
  within each worker call).
- This script only shards the (gid -> list[sid]) mapping across workers.
- If some optional catalog fields do not exist in a run, global_cs will record NaN.

Data structure:
-----
HDF5 catalog layout (run_cs.py output; columnar arrays, N = number of selected subhalos, ordered by SubhaloID ascending)

/
├── SubhaloID                 (N,)        int64
├── GroupID                   (N,)        int64
├── CenID                     (N,)        int64
│
├── pos_abs                   (N,3)       float64
├── vel_abs                   (N,3)       float64
├── pos_rel                   (N,3)       float64
├── vel_rel                   (N,3)       float64
│
├── SubhaloSFR                (N,)        float64
├── SubhaloGasMetallicity     (N,)        float64
├── SubhaloMass               (N,)        float64
├── SubhaloMassInRadType      (N,6)       float64
├── SubhaloVmax               (N,)        float64
├── SubhaloWindMass           (N,)        float64
├── SubhaloBHMass             (N,)        float64
├── SubhaloBHMdot             (N,)        float64
│
├── Group_M_Crit500           (N,)        float64
├── Group_M_Crit200           (N,)        float64
├── Group_R_Crit500           (N,)        float64
├── Group_R_Crit200           (N,)        float64
│
├── Tidal_self                (N,3,3)     float64   # target object's own matter, GR/Newtonian mass-potential branch
├── Tidal_tot                 (N,3,3)     float64   # inclusive GR acceleration branch
├── Tidal_grp                 (N,3,3)     float64   # host/group matter branch, target self particles removed
├── Tidal_tot_mg              (N,3,3)     float64   # inclusive MG/fifth-force acceleration branch
│
├── DM/                       (group)
│   ├── I                     (N,3,3)     float64
│   ├── dI                    (N,3,3)     float64
│   ├── ddI                   (N,3,3)     float64
│   ├── mass                  (N,)        float64
│   ├── L                     (N,3)       float64
│   ├── K_tot                 (N,)        float64
│   ├── kappa_rot             (N,)        float64
│   ├── Neff                  (N,)        float64
│   ├── axis_relerr           (N,3)       float64
│   ├── cos_err               (N,3)       float64
│   └── converged             (N,)        int8   (0/1)
│
├── Star/                     (group)
│   ├── I                     (N,3,3)     float64
│   ├── dI                    (N,3,3)     float64
│   ├── ddI                   (N,3,3)     float64
│   ├── mass                  (N,)        float64
│   ├── L                     (N,3)       float64
│   ├── K_tot                 (N,)        float64
│   ├── kappa_rot             (N,)        float64
│   ├── Neff                  (N,)        float64
│   ├── axis_relerr           (N,3)       float64
│   ├── cos_err               (N,3)       float64
│   └── converged             (N,)        int8   (0/1)
│
└── meta/                     (group)
    └── cfg_json              (scalar bytes; JSON-serialized cfg)

run_cs.py

ClusterSims driver script:
- Select subhalos by stellar particle number >= threshold (default: 30).
- Build tasks grouped by FoF GroupID (gid).
- Parallelize over gid-shards (ProcessPoolExecutor), where each worker calls
  global_cs.compute_many(...) on its shard.

NEW (minimal change)
--------------------
- Add Neff to the columnar HDF5 output under DM/Neff and Star/Neff.
  Neff is produced by ShapeKin variance stage (effective sample size):
      Neff = 1 / sum(w_i^2)
- Add Tidal_self to the columnar HDF5 output.
- Tidal_grp is target-exclusive, while Tidal_tot and Tidal_tot_mg remain
  inclusive. Tidal_self is computed from the target object's own all-type
  particles after the self-shape selection.
- Final tidal outputs are the four analysis branches:
    Tidal_self    : target object's own selected matter under GR/Newtonian gravity
    Tidal_tot     : inclusive GR acceleration-derived tensor
    Tidal_grp     : group matter tensor with the target object's own matter removed
    Tidal_tot_mg  : inclusive MG/fifth-force acceleration-derived tensor
- All tidal tensors use cfg["legacy_tidal_sign"].  With the default
  legacy_tidal_sign=True, files store the Hessian convention
  T_ij = +d_i d_j Phi.  The physical largest-stretching direction is the major
  eigenvector of -T.
"""


import argparse
import json
import os
import logging
import time
import uuid
import sys
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np

if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print(
        """usage: run_cs.py --basepath BASEPATH --snap SNAP --nworker NWORKER --out OUT [options]

ClusterSims global run (gid-sharded parallel).

core options:
  -h, --help
  --basepath BASEPATH
  --snap SNAP
  --nworker NWORKER
  --out OUT
  --min-star MIN_STAR
  --max-sub MAX_SUB
"""
    )
    raise SystemExit(0)

import h5py

from ia_analysis.catalogs.catalog_loader import CSCatalog
from ia_analysis.pipelines.global_cs import compute_many


TIDAL_SIGN_CONVENTION = "legacy_hessian:+d_i d_j Phi"
TIDAL_STRETCHING_CONVENTION = "largest_stretching_axis=eigvec_max(-dataset)"
TIDAL_UNITS = "km^2/s^2/(ckpc/h)^2"


def _create_tidal_dataset(parent, name, data, *, source, input_branch, target_exclusive):
    ds = parent.create_dataset(name, data=data)
    ds.attrs["source"] = source
    ds.attrs["input_branch"] = input_branch
    ds.attrs["target_exclusive"] = bool(target_exclusive)
    ds.attrs["sign_convention"] = TIDAL_SIGN_CONVENTION
    ds.attrs["stretching_convention"] = TIDAL_STRETCHING_CONVENTION
    ds.attrs["units"] = TIDAL_UNITS
    return ds


# -----------------------
# Default configuration
# -----------------------

DM_PERCENTILE = 98.0
STAR_PERCENTILE = 100.0
STAR_APERTURE_FACTOR = 2.0
MAX_ITER = 100
TOL = 1e-2

GRID_SIZE = 64
PADDING = 0.20

REPORT_EVERY = 120  # seconds

CS_DM_FIXED_MASS = 0.135401  # 1e10Msun/h

BOX_SIZE = 205.0  # Mpc/h

SUB_FIELDS_EXTRA = [
    "SubhaloSFR",
    "SubhaloGasMetallicity",
    "SubhaloMass",
    "SubhaloMassInRadType",
    "SubhaloVmax",
    "SubhaloWindMass",
    "SubhaloBHMass",
    "SubhaloBHMdot",
]

GROUP_FIELDS_EXTRA = [
    "Group_M_Crit500",
    "Group_M_Crit200",
    "Group_R_Crit500",
    "Group_R_Crit200",
]


def _default_cfg() -> dict:
    return dict(
        dm_shape_percentile=float(DM_PERCENTILE),
        star_shape_percentile=float(STAR_PERCENTILE),
        star_aperture_factor=float(STAR_APERTURE_FACTOR),
        shape_max_iter=int(MAX_ITER),
        shape_tol=float(TOL),
        shape_tensor_mode="reduced",
        tidal_grid_size=int(GRID_SIZE),
        tidal_padding=float(PADDING),
        box_size_mpc_h=float(BOX_SIZE),
        box_size_ckpc_h=float(BOX_SIZE) * 1000.0,
        dm_particle_mass=float(CS_DM_FIXED_MASS),
        sub_fields_extra=list(SUB_FIELDS_EXTRA),
        group_fields_extra=list(GROUP_FIELDS_EXTRA),
    )


def _group_to_shards(sids: np.ndarray, gids: np.ndarray, nworker: int):
    """Split gids across workers (round-robin over sorted gid blocks)."""
    sids = np.asarray(sids, dtype=np.int64)
    gids = np.asarray(gids, dtype=np.int64)

    order = np.argsort(gids, kind="mergesort")
    sids_s = sids[order]
    gids_s = gids[order]

    uniq, start = np.unique(gids_s, return_index=True)
    gid_slices = []
    for i, g in enumerate(uniq):
        s = start[i]
        e = start[i + 1] if i + 1 < len(start) else gids_s.size
        gid_slices.append((int(g), slice(s, e)))

    buckets = [[] for _ in range(max(1, int(nworker)))]
    for i, (_, sl) in enumerate(gid_slices):
        buckets[i % len(buckets)].append(sl)

    shards = []
    for sl_list in buckets:
        if not sl_list:
            continue
        sid_parts = [sids_s[sl] for sl in sl_list]
        gid_parts = [gids_s[sl] for sl in sl_list]
        shards.append((np.concatenate(sid_parts), np.concatenate(gid_parts)))
    return shards


def _nan_item(sid: int, gid: int) -> dict:
    """Fallback record used when a worker shard fails."""
    nan3 = np.full((3,), np.nan, dtype=np.float64)
    nan33 = np.full((3, 3), np.nan, dtype=np.float64)
    nan6 = np.full((6,), np.nan, dtype=np.float64)

    return {
        "Sub_info": {
            "SubhaloID": int(sid),
            "GroupID": int(gid),
            "CenID": -1,
            "pos_abs": nan3.copy(),
            "vel_abs": nan3.copy(),
            "pos_rel": nan3.copy(),
            "vel_rel": nan3.copy(),
            "SubhaloSFR": np.nan,
            "SubhaloGasMetallicity": np.nan,
            "SubhaloMass": np.nan,
            "SubhaloHalfmassRadType": nan6.copy(),
            "Group_M_Crit500": np.nan,
        },
        "Shape": {
            "dm": {
                "I": nan33.copy(),
                "dI": nan33.copy(),
                "ddI": nan33.copy(),
                "mass": np.nan,
                "L": nan3.copy(),
                "K_tot": np.nan,
                "kappa_rot": np.nan,
                "Neff": np.nan,  # NEW
                "axis_relerr": nan3.copy(),
                "cos_err": nan3.copy(),
                "converged": False,
            },
            "stars": {
                "I": nan33.copy(),
                "dI": nan33.copy(),
                "ddI": nan33.copy(),
                "mass": np.nan,
                "L": nan3.copy(),
                "K_tot": np.nan,
                "kappa_rot": np.nan,
                "Neff": np.nan,  # NEW
                "axis_relerr": nan3.copy(),
                "cos_err": nan3.copy(),
                "converged": False,
            },
        },
        "Tidal": {
            "tidal_grp": nan33.copy(),
            "tidal_tot": nan33.copy(),
            "tidal_tot_mg": nan33.copy(),
            "tidal_self": nan33.copy(),
        },
    }


def _acquire_output_lock(lock_path: str, retries: int = 12, sleep: float = 10.0):
    """Acquire a simple exclusive lock file for one output HDF5 path.

    This prevents two Slurm jobs from writing the same final output at the
    same time.  The lock is independent of HDF5's own file locking and is
    deliberately implemented with ``O_EXCL`` so it works on shared filesystems.
    """
    lock_path = os.path.abspath(lock_path)
    last_err = None

    for attempt in range(int(retries) + 1):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            msg = f"pid={os.getpid()} host={os.uname().nodename} time={time.time():.3f}\n"
            os.write(fd, msg.encode("utf-8"))
            os.close(fd)
            return lock_path
        except FileExistsError as exc:
            last_err = exc
            if attempt >= int(retries):
                break
            time.sleep(float(sleep))

    raise BlockingIOError(
        f"Could not acquire output lock after {retries} retries: {lock_path}. "
        "Another job may be writing the same output file. "
        "Check that every FLAG/SNAP job has a unique --out path, or remove a stale .lock file."
    ) from last_err


def _release_output_lock(lock_path: str):
    """Release an output lock file, ignoring stale-cleanup failures."""
    try:
        if lock_path and os.path.exists(lock_path):
            os.unlink(lock_path)
    except OSError:
        pass


def _write_results_hdf5(
    out_path: str,
    results: list,
    cfg: dict,
    *,
    overwrite: bool = False,
    write_retries: int = 12,
    write_retry_sleep: float = 10.0,
):
    """Write columnar arrays to HDF5, sorted by SubhaloID.

    The file is written to a unique temporary HDF5 file in the same directory
    and then moved into place.  A lightweight ``.lock`` file guards the final
    output path so repeated or duplicate Slurm jobs do not fight over the same
    HDF5 file.  This avoids the common COSMA/Lustre error:

        BlockingIOError: unable to lock file, errno = 11

    Parameters
    ----------
    overwrite : bool
        If False, an existing output file is left untouched and the write is
        skipped.  If True, the final file is replaced atomically.
    """
    out_path = os.path.abspath(out_path)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    items = []
    for it in results:
        if not it:
            continue
        info = it.get("Sub_info", {}) or {}
        if "SubhaloID" not in info:
            continue
        sid = int(np.asarray(info["SubhaloID"]).reshape(-1)[0])
        if sid < 0:
            continue
        items.append(it)

    items.sort(key=lambda it: int(np.asarray(it["Sub_info"]["SubhaloID"]).reshape(-1)[0]))
    N = len(items)

    SubhaloID = np.full((N,), -1, dtype=np.int64)
    CenID = np.full((N,), -1, dtype=np.int64)
    GroupID = np.full((N,), -1, dtype=np.int64)

    pos_abs = np.full((N, 3), np.nan, dtype=np.float64)
    vel_abs = np.full((N, 3), np.nan, dtype=np.float64)
    pos_rel = np.full((N, 3), np.nan, dtype=np.float64)
    vel_rel = np.full((N, 3), np.nan, dtype=np.float64)

    sub_extra = list(cfg.get("sub_fields_extra", []))
    grp_extra = list(cfg.get("group_fields_extra", []))

    sub_scalar_fields = [k for k in sub_extra if k != "SubhaloMassInRadType"]
    sub_vec6_fields = ["SubhaloMassInRadType"] if "SubhaloMassInRadType" in sub_extra else []
    grp_scalar_fields = list(grp_extra)

    sub_scalars = {k: np.full((N,), np.nan, dtype=np.float64) for k in sub_scalar_fields}
    sub_vec6 = {k: np.full((N, 6), np.nan, dtype=np.float64) for k in sub_vec6_fields}
    grp_scalars = {k: np.full((N,), np.nan, dtype=np.float64) for k in grp_scalar_fields}

    def _alloc_shape():
        return dict(
            I=np.full((N, 3, 3), np.nan, dtype=np.float64),
            dI=np.full((N, 3, 3), np.nan, dtype=np.float64),
            ddI=np.full((N, 3, 3), np.nan, dtype=np.float64),
            mass=np.full((N,), np.nan, dtype=np.float64),
            L=np.full((N, 3), np.nan, dtype=np.float64),
            K_tot=np.full((N,), np.nan, dtype=np.float64),
            kappa_rot=np.full((N,), np.nan, dtype=np.float64),
            Neff=np.full((N,), np.nan, dtype=np.float64),  # NEW
            axis_relerr=np.full((N, 3), np.nan, dtype=np.float64),
            cos_err=np.full((N, 3), np.nan, dtype=np.float64),
            converged=np.zeros((N,), dtype=np.int8),
        )

    DM = _alloc_shape()
    Star = _alloc_shape()

    Tidal_grp = np.full((N, 3, 3), np.nan, dtype=np.float64)
    Tidal_tot = np.full((N, 3, 3), np.nan, dtype=np.float64)
    Tidal_tot_mg = np.full((N, 3, 3), np.nan, dtype=np.float64)
    Tidal_self = np.full((N, 3, 3), np.nan, dtype=np.float64)

    for i, it in enumerate(items):
        info = it.get("Sub_info", {}) or {}
        SubhaloID[i] = int(np.asarray(info.get("SubhaloID", -1)).reshape(-1)[0])
        CenID[i] = int(np.asarray(info.get("CenID", -1)).reshape(-1)[0])
        GroupID[i] = int(np.asarray(info.get("GroupID", -1)).reshape(-1)[0])

        if "pos_abs" in info:
            pos_abs[i, :] = np.asarray(info["pos_abs"], dtype=np.float64).reshape(3)
        if "vel_abs" in info:
            vel_abs[i, :] = np.asarray(info["vel_abs"], dtype=np.float64).reshape(3)
        if "pos_rel" in info:
            pos_rel[i, :] = np.asarray(info["pos_rel"], dtype=np.float64).reshape(3)
        if "vel_rel" in info:
            vel_rel[i, :] = np.asarray(info["vel_rel"], dtype=np.float64).reshape(3)

        for k in sub_scalar_fields:
            if k in info:
                sub_scalars[k][i] = float(np.asarray(info[k]).reshape(-1)[0])
        for k in sub_vec6_fields:
            if k in info:
                sub_vec6[k][i, :] = np.asarray(info[k], dtype=np.float64).reshape(6)
        for k in grp_scalar_fields:
            if k in info:
                grp_scalars[k][i] = float(np.asarray(info[k]).reshape(-1)[0])

        sh = it.get("Shape", {}) or {}

        def _fill_comp(comp: str, out: dict):
            d = (sh.get(comp, {}) or {})
            if "I" in d:
                out["I"][i, :, :] = np.asarray(d["I"], dtype=np.float64).reshape(3, 3)
            if "dI" in d:
                out["dI"][i, :, :] = np.asarray(d["dI"], dtype=np.float64).reshape(3, 3)
            if "ddI" in d:
                out["ddI"][i, :, :] = np.asarray(d["ddI"], dtype=np.float64).reshape(3, 3)
            if "mass" in d:
                out["mass"][i] = float(np.asarray(d["mass"]).reshape(-1)[0])
            if "L" in d:
                out["L"][i, :] = np.asarray(d["L"], dtype=np.float64).reshape(3)
            if "K_tot" in d:
                out["K_tot"][i] = float(np.asarray(d["K_tot"]).reshape(-1)[0])
            if "kappa_rot" in d:
                out["kappa_rot"][i] = float(np.asarray(d["kappa_rot"]).reshape(-1)[0])
            if "Neff" in d:
                out["Neff"][i] = float(np.asarray(d["Neff"]).reshape(-1)[0])
            if "axis_relerr" in d:
                out["axis_relerr"][i, :] = np.asarray(d["axis_relerr"], dtype=np.float64).reshape(3)
            if "cos_err" in d:
                out["cos_err"][i, :] = np.asarray(d["cos_err"], dtype=np.float64).reshape(3)
            if "converged" in d:
                out["converged"][i] = 1 if bool(d["converged"]) else 0

        _fill_comp("dm", DM)
        _fill_comp("stars", Star)

        td = it.get("Tidal", {}) or {}
        if "tidal_grp" in td:
            Tidal_grp[i, :, :] = np.asarray(td["tidal_grp"], dtype=np.float64).reshape(3, 3)
        if "tidal_tot" in td:
            Tidal_tot[i, :, :] = np.asarray(td["tidal_tot"], dtype=np.float64).reshape(3, 3)
        if "tidal_tot_mg" in td:
            Tidal_tot_mg[i, :, :] = np.asarray(td["tidal_tot_mg"], dtype=np.float64).reshape(3, 3)
        if "tidal_self" in td:
            Tidal_self[i, :, :] = np.asarray(td["tidal_self"], dtype=np.float64).reshape(3, 3)

    if os.path.exists(out_path) and not overwrite:
        logging.getLogger("run_cs").info(
            "[skip] output exists and --overwrite is not set: %s", out_path
        )
        return False

    lock_path = out_path + ".lock"
    tmp_path = os.path.join(
        out_dir,
        f".{os.path.basename(out_path)}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}",
    )

    acquired_lock = None
    try:
        acquired_lock = _acquire_output_lock(
            lock_path,
            retries=int(write_retries),
            sleep=float(write_retry_sleep),
        )

        # Another job may have completed while this job waited for the lock.
        if os.path.exists(out_path) and not overwrite:
            logging.getLogger("run_cs").info(
                "[skip] output was created by another job while waiting: %s", out_path
            )
            return False

        with h5py.File(tmp_path, "w") as f:
            cfg_json = json.dumps(cfg, sort_keys=True)

            meta = f.create_group("meta")
            meta.create_dataset("cfg_json", data=np.bytes_(cfg_json))
            meta.attrs["creator"] = "run_cs.py"
            meta.attrs["format"] = "columnar arrays"
            meta.attrs["n_subhalos"] = int(N)

            f.create_dataset("SubhaloID", data=SubhaloID)
            f.create_dataset("CenID", data=CenID)
            f.create_dataset("GroupID", data=GroupID)
            f.create_dataset("pos_abs", data=pos_abs)
            f.create_dataset("vel_abs", data=vel_abs)
            f.create_dataset("pos_rel", data=pos_rel)
            f.create_dataset("vel_rel", data=vel_rel)

            for k, arr in sub_scalars.items():
                f.create_dataset(k, data=arr)
            for k, arr in sub_vec6.items():
                f.create_dataset(k, data=arr)
            for k, arr in grp_scalars.items():
                f.create_dataset(k, data=arr)

            gdm = f.create_group("DM")
            for k, v in DM.items():
                gdm.create_dataset(k, data=v)

            gst = f.create_group("Star")
            for k, v in Star.items():
                gst.create_dataset(k, data=v)

            _create_tidal_dataset(
                f,
                "Tidal_self",
                Tidal_self,
                source="target object's own selected matter under GR/Newtonian gravity",
                input_branch="mass_to_potential",
                target_exclusive=False,
            )
            _create_tidal_dataset(
                f,
                "Tidal_tot",
                Tidal_tot,
                source="inclusive GR acceleration field",
                input_branch="acceleration_to_tidal",
                target_exclusive=False,
            )
            _create_tidal_dataset(
                f,
                "Tidal_grp",
                Tidal_grp,
                source="host/group matter with target object's own matter removed",
                input_branch="mass_to_potential",
                target_exclusive=True,
            )
            _create_tidal_dataset(
                f,
                "Tidal_tot_mg",
                Tidal_tot_mg,
                source="inclusive MG/fifth-force acceleration field",
                input_branch="acceleration_to_tidal",
                target_exclusive=False,
            )

        # Move the completed temporary file into place only after h5py closed it.
        if overwrite:
            os.replace(tmp_path, out_path)
        else:
            # No-clobber finalization.  ``os.link`` fails if out_path appeared
            # between the last check and now, protecting against duplicate jobs.
            os.link(tmp_path, out_path)
            os.unlink(tmp_path)

        return True

    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        _release_output_lock(acquired_lock)


def main():
    ap = argparse.ArgumentParser(description="ClusterSims global run (gid-sharded parallel).")
    ap.add_argument("--basepath", required=True, help="ClusterSims base path.")
    ap.add_argument("--snap", required=True, type=int, help="Snapshot number.")
    ap.add_argument("--nworker", required=True, type=int, help="Number of worker processes.")
    ap.add_argument("--out", required=True, help="Output HDF5 path.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file. Without this flag, existing outputs are skipped.")
    ap.add_argument("--write-retries", type=int, default=12, help="Number of retries when waiting for the output .lock file.")
    ap.add_argument("--write-retry-sleep", type=float, default=10.0, help="Seconds between output-lock retries.")
    ap.add_argument("--min-star", type=int, default=50, help="Min stellar particle number.")
    ap.add_argument("--max-sub", type=int, default=None, help="Optional cap on selected subhalos.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | run_cs | %(message)s")
    logger = logging.getLogger("run_cs")

    if os.path.exists(args.out) and not args.overwrite:
        logger.info("[skip] output exists and --overwrite is not set: %s", args.out)
        return

    cfg = _default_cfg()

    cat = CSCatalog(args.basepath, args.snap)
    halos, subs = cat.loadFoF(
        group_fields=["GroupFirstSub", "GroupNsubs", "GroupLenType"] + cfg["group_fields_extra"],
        subhalo_fields=["SubhaloPos", "SubhaloVel", "SubhaloLenType"] + cfg["sub_fields_extra"],
    )

    slt = np.asarray(subs["SubhaloLenType"], dtype=np.int64)
    nstar = slt[:, 4]
    mask = nstar >= int(args.min_star)

    if "GroupID" in subs:
        gid_all = np.asarray(subs["GroupID"], dtype=np.int64)
    else:
        gid_all = np.full(slt.shape[0], -1, dtype=np.int64)
        gfs = np.asarray(halos["GroupFirstSub"], dtype=np.int64)
        gns = np.asarray(halos["GroupNsubs"], dtype=np.int64)
        for gid, (fs, ns) in enumerate(zip(gfs, gns)):
            if ns <= 0:
                continue
            gid_all[fs:fs + ns] = gid

    sid_sel = np.where(mask)[0].astype(np.int64)
    gid_sel = gid_all[sid_sel].astype(np.int64)
    good = gid_sel >= 0
    sid_sel = sid_sel[good]
    gid_sel = gid_sel[good]

    if args.max_sub is not None and sid_sel.size > int(args.max_sub):
        sid_sel = sid_sel[: int(args.max_sub)]
        gid_sel = gid_sel[: int(args.max_sub)]

    if sid_sel.size == 0:
        raise RuntimeError("No subhalos passed selection; check --min-star.")

    shards = _group_to_shards(sid_sel, gid_sel, args.nworker)

    all_outputs = [None] * sid_sel.size
    sid_to_global_idx = {int(s): i for i, s in enumerate(sid_sel.tolist())}

    futures = []
    fut_gid_n = {}
    fut_inputs = {}

    with ProcessPoolExecutor(max_workers=int(args.nworker)) as ex:
        for sid_chunk, gid_chunk in shards:
            fut = ex.submit(compute_many, args.basepath, int(args.snap), sid_chunk, gid_chunk, cfg, None, None)
            futures.append(fut)
            fut_gid_n[fut] = int(np.unique(gid_chunk).size)
            fut_inputs[fut] = (sid_chunk.copy(), gid_chunk.copy())

        n_gid_total = int(np.unique(gid_sel).size)
        done_gids = 0
        last_report = time.monotonic()

        logger.info("[select] selected_subhalos=%d total_gids=%d n_shards=%d", int(sid_sel.size), int(n_gid_total), int(len(futures)))

        def _pct(x: int) -> float:
            return 100.0 * float(x) / float(n_gid_total) if n_gid_total > 0 else 0.0

        def _log_progress(force: bool = False) -> None:
            nonlocal last_report
            now = time.monotonic()
            if (not force) and (now - last_report < float(REPORT_EVERY)):
                return
            remaining_gids = n_gid_total - done_gids
            logger.info("[progress] done=%d (%.1f%%) remaining=%d (%.1f%%)", int(done_gids), _pct(int(done_gids)), int(remaining_gids), _pct(int(remaining_gids)))
            last_report = now

        _log_progress(force=True)

        while futures:
            done_set, _ = wait(futures, timeout=float(REPORT_EVERY), return_when=FIRST_COMPLETED)
            if not done_set:
                _log_progress(force=True)
                continue

            for fut in list(done_set):
                futures.remove(fut)
                try:
                    outs = fut.result()
                except Exception as e:
                    sid_chunk, gid_chunk = fut_inputs.get(fut, (np.array([], dtype=np.int64), np.array([], dtype=np.int64)))
                    logger.warning("[worker-failed] shard failed; filling NaNs for %d subhalos. Reason: %s", int(sid_chunk.size), repr(e))
                    outs = [_nan_item(int(s), int(g)) for s, g in zip(sid_chunk.tolist(), gid_chunk.tolist())]

                done_gids += int(fut_gid_n.get(fut, 0))
                for out in outs:
                    if not out:
                        continue
                    sid = int(out.get("Sub_info", {}).get("SubhaloID", -1))
                    if sid < 0:
                        continue
                    if ("Shape" not in out) or (out["Shape"] is None):
                        out["Shape"] = _nan_item(sid, int(out.get("Sub_info", {}).get("GroupID", -1)))["Shape"]
                    if ("Tidal" not in out) or (out["Tidal"] is None):
                        out["Tidal"] = _nan_item(sid, int(out.get("Sub_info", {}).get("GroupID", -1)))["Tidal"]
                    all_outputs[sid_to_global_idx[sid]] = out

            _log_progress(force=True)

        _log_progress(force=True)

    wrote = _write_results_hdf5(
        args.out,
        all_outputs,
        cfg,
        overwrite=bool(args.overwrite),
        write_retries=int(args.write_retries),
        write_retry_sleep=float(args.write_retry_sleep),
    )
    if wrote:
        logger.info("[done] wrote: %s  (N=%d)", args.out, int(len(all_outputs)))
    else:
        logger.info("[done] output already existed; no file was modified: %s", args.out)


if __name__ == "__main__":
    main()
