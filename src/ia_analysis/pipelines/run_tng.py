#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_tng_updated.py

TNG driver (gid-sharded parallel), aligned to the existing columnar HDF5 output.

Selection (unchanged)
---------------------
- Select subhalos by stellar particle number >= threshold (default: 150).

NEW (minimal change)
--------------------
- Write Neff to the HDF5 output (DM/Neff and Star/Neff).
  Neff comes from ShapeKin variance stage:
      Neff = 1 / sum(w_i^2)
- Add Tidal_self to the HDF5 output.
- Tidal_grp is target-exclusive, Tidal_tot is inclusive, and Tidal_self is
  computed from the target object's own all-type particles after the self-shape
  selection. TNG has no MG/fifth-force branch, so Tidal_tot_mg is not written.
- Add optional IllustrisTNG API-backed downloading through TNGCatalog.
  Existing local-file usage is unchanged.  If local files are missing, pass
  --api-key or set the TNG_API_KEY environment variable.

No other functionality is changed.
"""
"""
Data structure:
-----
HDF5 catalog layout (run_tng.py output; columnar arrays, N = number of selected subhalos, ordered by SubhaloID ascending)

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
├── Tidal/                    (group)
│   ├── Tidal_grp             (N,3,3)     float64   # host/group mass tidal with target self removed
│   ├── Tidal_tot             (N,3,3)     float64   # inclusive potential-derived tidal
│   └── Tidal_self            (N,3,3)     float64   # target self tidal from selected all-type particles
│
├── DM/                       (group)
│   ├── I                     (N,3,3)     float64
│   ├── dI                    (N,3,3)     float64
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
"""
import argparse
import json
import os
import logging
import time
import sys
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np

if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print(
        """usage: run_tng.py --nworker NWORKER --out OUT [options]

TNG global run (gid-sharded parallel).

core options:
  -h, --help
  --nworker NWORKER
  --out OUT
  --min-star MIN_STAR
  --max-sub MAX_SUB
  --api-key API_KEY
  --sim-name SIM_NAME
  --cache-dir CACHE_DIR
"""
    )
    raise SystemExit(0)

import h5py

from ia_analysis.catalogs.TNGCatLoader import TNGCatalog
from ia_analysis.pipelines.global_tng import compute_many

TNG_BASE_PATH = "/cosma8/data/dp203/dc-wang17/TNG/tng_data"
TNG_SNAP = 99

# Optional API-backed downloader settings.
# Keep the key out of the source file when possible:
#     export TNG_API_KEY="..."
# or pass it at runtime:
#     python run_tng.py ... --api-key "..."
DEFAULT_TNG_SIM_NAME = os.environ.get("TNG_SIM_NAME", "TNG300-1")
DEFAULT_TNG_API_KEY = os.environ.get("TNG_API_KEY", None)
DEFAULT_TNG_CACHE_DIR = os.environ.get("TNG_CACHE_DIR", None)

DM_PERCENTILE = 98.0
STAR_PERCENTILE = 100.0
STAR_APERTURE_FACTOR = 2.0
MAX_ITER = 100
TOL = 1e-2

GRID_SIZE = 64
PADDING = 0.20

REPORT_EVERY = 120

TNG_DM_FIXED_MASS = 5.9e7

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


def _default_cfg():
    return dict(
        dm_shape_percentile=float(DM_PERCENTILE),
        star_shape_percentile=float(STAR_PERCENTILE),
        star_aperture_factor=float(STAR_APERTURE_FACTOR),
        shape_max_iter=int(MAX_ITER),
        shape_tol=float(TOL),
        shape_tensor_mode="reduced",
        tidal_grid_size=int(GRID_SIZE),
        tidal_padding=float(PADDING),
        tidal_softening=0.01,
        legacy_tidal_sign=True,
        dm_particle_mass=float(TNG_DM_FIXED_MASS),
        sub_fields_extra=list(SUB_FIELDS_EXTRA),
        group_fields_extra=list(GROUP_FIELDS_EXTRA),
    )


def _tng_catalog_kwargs_from_args(args):
    """Build keyword arguments forwarded to ``TNGCatalog``.

    The API key is intentionally not written into the output file metadata.
    ``delete_cache=True`` by default means that API-downloaded temporary files
    are removed when the Python process exits.  Use ``--keep-cache`` if you
    want to preserve them for later runs.
    """
    return dict(
        api_key=args.api_key,
        download_if_missing=args.download_if_missing,
        cache_dir=args.cache_dir,
        delete_cache=(not bool(args.keep_cache)),
        sim_name=args.sim_name,
        verbose=(not bool(args.quiet_tng_download)),
        prefer_cutout=True,
    )


def _safe_catalog_metadata(tng_catalog_kwargs):
    """Return non-secret downloader metadata for HDF5 output attributes."""
    kw = dict(tng_catalog_kwargs or {})
    kw.pop("api_key", None)
    return kw


def _group_to_shards(sids, gids, nworker):
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


def _nan_item(sid, gid):
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
            "SubhaloMassInRadType": nan6.copy(),
            "SubhaloVmax": np.nan,
            "SubhaloWindMass": np.nan,
            "SubhaloBHMass": np.nan,
            "SubhaloBHMdot": np.nan,
            "Group_M_Crit500": np.nan,
            "Group_M_Crit200": np.nan,
            "Group_R_Crit500": np.nan,
            "Group_R_Crit200": np.nan,
        },
        "Shape": {
            "dm": {
                "I": nan33.copy(),
                "dI": nan33.copy(),
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
            "tidal_self": nan33.copy(),
        },
    }


def _write_results_hdf5_columnar(out_path, results, cfg, tng_catalog_kwargs=None):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    items = [r for r in results if r is not None and "Sub_info" in r and "SubhaloID" in r["Sub_info"]]
    if not items:
        raise RuntimeError("No valid results to write.")

    sids = np.array([int(r["Sub_info"]["SubhaloID"]) for r in items], dtype=np.int64)
    order = np.argsort(sids)
    items = [items[i] for i in order]
    sids = sids[order]
    N = int(len(items))

    def _stack_vec(key, default=np.nan):
        arr = np.full((N, 3), default, dtype=np.float64)
        for i, r in enumerate(items):
            v = r["Sub_info"].get(key, None)
            if v is None:
                continue
            arr[i] = np.asarray(v, dtype=np.float64).reshape(3)
        return arr

    def _stack_scalar(key, default=np.nan):
        arr = np.full(N, default, dtype=np.float64)
        for i, r in enumerate(items):
            v = r["Sub_info"].get(key, None)
            if v is None:
                continue
            arr[i] = float(np.asarray(v))
        return arr

    def _stack_int(key, default=-1):
        arr = np.full(N, default, dtype=np.int64)
        for i, r in enumerate(items):
            v = r["Sub_info"].get(key, None)
            if v is None:
                continue
            arr[i] = int(np.asarray(v))
        return arr

    SubhaloID = sids
    GroupID = _stack_int("GroupID", default=-1)
    CenID = _stack_int("CenID", default=-1)

    pos_abs = _stack_vec("pos_abs")
    vel_abs = _stack_vec("vel_abs")
    pos_rel = _stack_vec("pos_rel")
    vel_rel = _stack_vec("vel_rel")

    sub_extra = cfg.get("sub_fields_extra", []) or []
    grp_extra = cfg.get("group_fields_extra", []) or []

    sub_extra_data = {}
    for k in sub_extra:
        v0 = items[0]["Sub_info"].get(k, None)
        if v0 is None:
            sub_extra_data[k] = np.full(N, np.nan, dtype=np.float64)
            continue
        a0 = np.asarray(v0)
        if a0.ndim == 0:
            sub_extra_data[k] = _stack_scalar(k)
        else:
            shp = a0.shape
            arr = np.full((N,) + shp, np.nan, dtype=np.float64)
            for i, r in enumerate(items):
                v = r["Sub_info"].get(k, None)
                if v is None:
                    continue
                arr[i] = np.asarray(v, dtype=np.float64).reshape(shp)
            sub_extra_data[k] = arr

    grp_extra_data = {}
    for k in grp_extra:
        v0 = items[0]["Sub_info"].get(k, None)
        if v0 is None:
            grp_extra_data[k] = np.full(N, np.nan, dtype=np.float64)
            continue
        a0 = np.asarray(v0)
        if a0.ndim == 0:
            arr = np.full(N, np.nan, dtype=np.float64)
            for i, r in enumerate(items):
                v = r["Sub_info"].get(k, None)
                if v is None:
                    continue
                arr[i] = float(np.asarray(v))
            grp_extra_data[k] = arr
        else:
            shp = a0.shape
            arr = np.full((N,) + shp, np.nan, dtype=np.float64)
            for i, r in enumerate(items):
                v = r["Sub_info"].get(k, None)
                if v is None:
                    continue
                arr[i] = np.asarray(v, dtype=np.float64).reshape(shp)
            grp_extra_data[k] = arr

    def _stack_shape(comp, key, shape):
        arr = np.full((N,) + shape, np.nan, dtype=np.float64)
        for i, r in enumerate(items):
            v = r.get("Shape", {}).get(comp, {}).get(key, None)
            if v is None:
                continue
            arr[i] = np.asarray(v, dtype=np.float64).reshape(shape)
        return arr

    def _stack_shape_scalar(comp, key):
        arr = np.full(N, np.nan, dtype=np.float64)
        for i, r in enumerate(items):
            v = r.get("Shape", {}).get(comp, {}).get(key, None)
            if v is None:
                continue
            arr[i] = float(np.asarray(v))
        return arr

    def _stack_shape_int(comp, key):
        arr = np.zeros(N, dtype=np.int8)
        for i, r in enumerate(items):
            v = r.get("Shape", {}).get(comp, {}).get(key, None)
            if v is None:
                continue
            arr[i] = np.int8(bool(v))
        return arr

    DM_I = _stack_shape("dm", "I", (3, 3))
    DM_dI = _stack_shape("dm", "dI", (3, 3))
    DM_mass = _stack_shape_scalar("dm", "mass")
    DM_L = _stack_shape("dm", "L", (3,))
    DM_Ktot = _stack_shape_scalar("dm", "K_tot")
    DM_kappa = _stack_shape_scalar("dm", "kappa_rot")
    DM_Neff = _stack_shape_scalar("dm", "Neff")  # NEW
    DM_axis_relerr = _stack_shape("dm", "axis_relerr", (3,))
    DM_cos_err = _stack_shape("dm", "cos_err", (3,))
    DM_conv = _stack_shape_int("dm", "converged")

    ST_I = _stack_shape("stars", "I", (3, 3))
    ST_dI = _stack_shape("stars", "dI", (3, 3))
    ST_mass = _stack_shape_scalar("stars", "mass")
    ST_L = _stack_shape("stars", "L", (3,))
    ST_Ktot = _stack_shape_scalar("stars", "K_tot")
    ST_kappa = _stack_shape_scalar("stars", "kappa_rot")
    ST_Neff = _stack_shape_scalar("stars", "Neff")  # NEW
    ST_axis_relerr = _stack_shape("stars", "axis_relerr", (3,))
    ST_cos_err = _stack_shape("stars", "cos_err", (3,))
    ST_conv = _stack_shape_int("stars", "converged")

    T_grp = np.full((N, 3, 3), np.nan, dtype=np.float64)
    T_tot = np.full((N, 3, 3), np.nan, dtype=np.float64)
    T_self = np.full((N, 3, 3), np.nan, dtype=np.float64)
    for i, r in enumerate(items):
        td = r.get("Tidal", {})
        if "tidal_grp" in td:
            T_grp[i] = np.asarray(td["tidal_grp"], dtype=np.float64).reshape(3, 3)
        if "tidal_tot" in td:
            T_tot[i] = np.asarray(td["tidal_tot"], dtype=np.float64).reshape(3, 3)
        if "tidal_self" in td:
            T_self[i] = np.asarray(td["tidal_self"], dtype=np.float64).reshape(3, 3)

    with h5py.File(out_path, "w") as f:
        meta = f.create_group("meta")
        meta.create_dataset("cfg_json", data=np.string_(json.dumps(cfg, sort_keys=True)))
        meta.attrs["creator"] = "run_tng_updated.py"
        meta.attrs["format"] = "columnar"
        meta.attrs["n_subhalos"] = int(N)
        meta.attrs["tng_base_path"] = TNG_BASE_PATH
        meta.attrs["tng_snap"] = int(TNG_SNAP)
        meta.attrs["tng_catalog_kwargs_json"] = np.string_(json.dumps(_safe_catalog_metadata(tng_catalog_kwargs), sort_keys=True))

        f.create_dataset("SubhaloID", data=SubhaloID)
        f.create_dataset("GroupID", data=GroupID)
        f.create_dataset("CenID", data=CenID)
        f.create_dataset("pos_abs", data=pos_abs)
        f.create_dataset("vel_abs", data=vel_abs)
        f.create_dataset("pos_rel", data=pos_rel)
        f.create_dataset("vel_rel", data=vel_rel)

        for k, arr in sub_extra_data.items():
            f.create_dataset(k, data=arr)
        for k, arr in grp_extra_data.items():
            f.create_dataset(k, data=arr)

        gDM = f.create_group("DM")
        gDM.create_dataset("I", data=DM_I)
        gDM.create_dataset("dI", data=DM_dI)
        gDM.create_dataset("mass", data=DM_mass)
        gDM.create_dataset("L", data=DM_L)
        gDM.create_dataset("K_tot", data=DM_Ktot)
        gDM.create_dataset("kappa_rot", data=DM_kappa)
        gDM.create_dataset("Neff", data=DM_Neff)  # NEW
        gDM.create_dataset("axis_relerr", data=DM_axis_relerr)
        gDM.create_dataset("cos_err", data=DM_cos_err)
        gDM.create_dataset("converged", data=DM_conv)

        gST = f.create_group("Star")
        gST.create_dataset("I", data=ST_I)
        gST.create_dataset("dI", data=ST_dI)
        gST.create_dataset("mass", data=ST_mass)
        gST.create_dataset("L", data=ST_L)
        gST.create_dataset("K_tot", data=ST_Ktot)
        gST.create_dataset("kappa_rot", data=ST_kappa)
        gST.create_dataset("Neff", data=ST_Neff)  # NEW
        gST.create_dataset("axis_relerr", data=ST_axis_relerr)
        gST.create_dataset("cos_err", data=ST_cos_err)
        gST.create_dataset("converged", data=ST_conv)

        gT = f.create_group("Tidal")
        gT.create_dataset("Tidal_grp", data=T_grp)
        gT.create_dataset("Tidal_tot", data=T_tot)
        gT.create_dataset("Tidal_self", data=T_self)


def main():
    ap = argparse.ArgumentParser(description="TNG global run (gid-sharded parallel).")
    ap.add_argument("--nworker", required=True, type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-star", type=int, default=150)
    ap.add_argument("--max-sub", type=int, default=None)

    # Optional TNG API-backed downloader.  The original local-file workflow is
    # unchanged if no API key is given and all local files are present.
    ap.add_argument("--api-key", default=DEFAULT_TNG_API_KEY, help="IllustrisTNG API key. Can also be set through TNG_API_KEY.")
    ap.add_argument("--sim-name", default=DEFAULT_TNG_SIM_NAME, help="TNG simulation name used by the API, e.g. TNG300-1.")
    ap.add_argument("--cache-dir", default=DEFAULT_TNG_CACHE_DIR, help="Temporary/persistent cache directory for API downloads.")
    ap.add_argument("--keep-cache", action="store_true", help="Keep API-downloaded cache files after the program exits.")
    ap.add_argument("--quiet-tng-download", action="store_true", help="Suppress TNGCatalog API download status messages.")
    ap.add_argument("--download-if-missing", dest="download_if_missing", action="store_true", help="Force API download attempts when required files are missing.")
    ap.add_argument("--no-download-if-missing", dest="download_if_missing", action="store_false", help="Disable API download attempts even if an API key is provided.")
    ap.set_defaults(download_if_missing=None)

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | run_tng | %(message)s")
    logger = logging.getLogger("run_tng")

    cfg = _default_cfg()
    tng_catalog_kwargs = _tng_catalog_kwargs_from_args(args)

    if args.api_key:
        logger.info(
            "[TNG API] enabled: sim_name=%s cache_dir=%s delete_cache=%s download_if_missing=%s",
            args.sim_name,
            str(args.cache_dir),
            str(not bool(args.keep_cache)),
            str(args.download_if_missing),
        )
    else:
        logger.info("[TNG API] disabled: no API key provided; using local files only.")

    cat = TNGCatalog(TNG_BASE_PATH, TNG_SNAP, **tng_catalog_kwargs)
    try:
        halos, subs = cat.loadFoF(
            group_fields=["GroupFirstSub", "GroupNsubs", "GroupLenType"] + cfg["group_fields_extra"],
            subhalo_fields=["SubhaloPos", "SubhaloVel", "SubhaloLenType", "SubhaloGrNr", "SubhaloHalfmassRadType"] + cfg["sub_fields_extra"],
        )
    finally:
        # If groupcat fields were downloaded through the API subset endpoint,
        # they have already been loaded into memory and can be deleted.
        cat.cleanup()

    slt = np.asarray(subs["SubhaloLenType"], dtype=np.int64)
    nstar = slt[:, 4]
    mask = nstar >= int(args.min_star)

    sid_sel = np.where(mask)[0].astype(np.int64)
    gid_all = np.asarray(subs["GroupID"], dtype=np.int64) if "GroupID" in subs else np.asarray(subs["SubhaloGrNr"], dtype=np.int64)
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
            fut = ex.submit(
                compute_many,
                TNG_BASE_PATH,
                int(TNG_SNAP),
                sid_chunk,
                gid_chunk,
                cfg,
                None,
                None,
                tng_catalog_kwargs,
            )
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

    _write_results_hdf5_columnar(args.out, all_outputs, cfg, tng_catalog_kwargs=tng_catalog_kwargs)
    logger.info("[done] wrote: %s (N=%d)", args.out, int(len(all_outputs)))


if __name__ == "__main__":
    main()
