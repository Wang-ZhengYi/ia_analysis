#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Real-space intrinsic-alignment correlation pipeline.

Purpose
-------
This module measures galaxy clustering and IA correlation functions from
ClusterSims-style catalogs, including jackknife covariance estimates.

Provides
--------
- Command-line parsing for correlation-function runs.
- Sample selection and catalog field preparation.
- Halotools-based pair counting for density-shape and shape-shape statistics.
- HDF5 output for correlation measurements and covariance products.

Notes
-----
The heavy correlation backend is imported only after the fast ``--help`` path so
basic command discovery works in lightweight environments.
"""

import argparse
import time
import sys
from pathlib import Path

if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print(
        """usage: ia_corr.py --flag FLAG --snap SNAP [options]

Measure IA correlation functions with jackknife covariance.

core options:
  -h, --help
  --flag FLAG
  --snap SNAP
  --out OUT
  --base-dir BASE_DIR
  --boxsize BOXSIZE
  --cos-err-max COS_ERR_MAX
  --nsub NSUB
  --nthreads NTHREADS
"""
    )
    raise SystemExit(0)

import h5py
import numpy as np

from halotools.mock_observables import (
    tpcf, ee_3d, ed_3d,
    tpcf_one_two_halo_decomp,
    ee_3d_one_two_halo_decomp,
    ed_3d_one_two_halo_decomp,
)

# -----------------------------
# Fixed defaults
# -----------------------------
BASE_DIR = Path("/cosma8/data/dp203/dc-wang17/MG_global")
OUT_DIR  = Path("/cosma/home/dp203/dc-wang17/IA_analysis/cfs")

LOS = np.array([0.0, 0.0, 1.0], dtype=float)

RMIN, RMAX = 0.1, 20.0
NBIN = 20
RBINS = np.logspace(np.log10(RMIN), np.log10(RMAX), NBIN + 1)

# mstar bins in ia_pk_cs convention (mstar ~ 1 means 1e10 Msun/h)
MASS_BINS = [
    ("M10_11", 1.0, 10.0),      # 1e10 - 1e11
    ("M11_12", 10.0, 100.0),    # 1e11 - 1e12
]


def catalog_path(flag, snap, base_dir):
    return Path(base_dir) / f"L302_N1136_{flag}_s{int(snap):03d}.hdf5"


def default_out_path(flag, snap):
    return OUT_DIR / f"tcfs_{flag}_{int(snap):03d}.hdf5"


def read_catalog(mg_file):
    with h5py.File(mg_file, "r") as f:
        pos = f["pos_abs"][:] / 1000.0  # ckpc/h -> Mpc/h
        I = f["Star"]["I"][:]
        cos_err = f["Star"]["cos_err"][:]
        mstar = f["SubhaloMassInRadType"][:, 4]
        host = f["CenID"][:]  # host halo/group ID for 1h/2h

    cos_err = np.abs(cos_err)
    if cos_err.ndim > 1:
        cos_err = np.max(cos_err, axis=tuple(range(1, cos_err.ndim)))
    return pos, I, cos_err, mstar, host


def select_mass_bin(pos, I, cos_err, mstar, host, *, cos_err_max, mlo, mhi, los):
    ok = (
        np.isfinite(pos).all(axis=1)
        & np.isfinite(I).all(axis=(1, 2))
        & np.isfinite(cos_err)
        & np.isfinite(mstar)
        & np.isfinite(host)
        & (cos_err < float(cos_err_max))
        & (mstar >= float(mlo)) & (mstar < float(mhi))
    )

    pos = pos[ok]
    I = I[ok]
    host = host[ok]

    # 3D major-axis from shape tensor
    w, v = np.linalg.eigh(I)  # ascending eigenvalues
    ori = v[:, :, 2]          # largest-eigenvalue eigenvector

    nrm = np.linalg.norm(ori, axis=1)
    good = np.isfinite(nrm) & (nrm > 0.0)

    pos = pos[good]
    ori = ori[good] / nrm[good, None]
    host = host[good]

    # optional sign convention: ori·LOS >= 0
    sgn = np.sign(ori @ los)
    sgn[sgn == 0.0] = 1.0
    ori = ori * sgn[:, None]

    return pos, ori, host


def jk_tags(pos, Lbox, nsub):
    p = np.mod(pos, Lbox) / Lbox
    ix = np.minimum((p[:, 0] * nsub).astype(int), nsub - 1)
    iy = np.minimum((p[:, 1] * nsub).astype(int), nsub - 1)
    iz = np.minimum((p[:, 2] * nsub).astype(int), nsub - 1)
    return ix + nsub * iy + (nsub * nsub) * iz


def jk_mean_cov(X):
    X = np.asarray(X, float)
    mu = np.mean(X, axis=0)
    d = X - mu[None, :]
    nj = X.shape[0]
    cov = (nj - 1.0) / nj * (d.T @ d)
    return mu, cov


def measure_one(pos, ori, host, *, rbins, Lbox, nthreads):
    # totals
    xi_tot = tpcf(pos, rbins, period=Lbox, num_threads=nthreads)
    ee_tot = ee_3d(pos, ori, pos, ori, rbins, period=Lbox, num_threads=nthreads)
    ed_tot = ed_3d(pos, ori, pos, rbins, period=Lbox, num_threads=nthreads)

    # 1h/2h (keyword args: robust to halotools signature differences)
    xi_1h, xi_2h = tpcf_one_two_halo_decomp(
        sample1=pos,
        rbins=rbins,
        sample1_host_halo_id=host,
        period=Lbox,
        num_threads=nthreads,
    )

    ee_1h, ee_2h = ee_3d_one_two_halo_decomp(
        sample1=pos, orientations1=ori,
        sample2=pos, orientations2=ori,
        rbins=rbins,
        sample1_host_halo_id=host,
        sample2_host_halo_id=host,
        period=Lbox,
        num_threads=nthreads,
    )

    ed_1h, ed_2h = ed_3d_one_two_halo_decomp(
        sample1=pos, orientations1=ori,
        sample2=pos,
        rbins=rbins,
        sample1_host_halo_id=host,
        sample2_host_halo_id=host,
        period=Lbox,
        num_threads=nthreads,
    )

    return dict(
        xi_tot=xi_tot, xi_1h=xi_1h, xi_2h=xi_2h,
        ee_tot=ee_tot, ee_1h=ee_1h, ee_2h=ee_2h,
        ed_tot=ed_tot, ed_1h=ed_1h, ed_2h=ed_2h,
    )


def measure_with_jackknife(pos, ori, host, *, rbins, Lbox, nsub, nthreads, label=""):
    tags = jk_tags(pos, Lbox, nsub)
    njk = nsub ** 3

    keys = [
        "xi_tot", "xi_1h", "xi_2h",
        "ee_tot", "ee_1h", "ee_2h",
        "ed_tot", "ed_1h", "ed_2h",
    ]
    stacks = {k: [] for k in keys}

    t0 = time.time()
    for j in range(njk):
        keep = (tags != j)
        out = measure_one(
            pos[keep], ori[keep], host[keep],
            rbins=rbins, Lbox=Lbox, nthreads=nthreads
        )
        for k in keys:
            stacks[k].append(out[k])

        if (j % 3) == 0:
            dt = (time.time() - t0) / 60.0
            print(f"[{label}] JK {j+1:02d}/{njk}  elapsed={dt:.1f} min", flush=True)

    res = {"meta": dict(rbins=rbins, Lbox=Lbox, nsub=nsub, Njk=njk, nthreads=nthreads)}
    for k in keys:
        mu, cov = jk_mean_cov(np.asarray(stacks[k]))
        res[k] = dict(mean=mu, cov=cov)

    print(f"[{label}] done  walltime={(time.time()-t0)/60:.1f} min", flush=True)
    return res


def save_job_hdf5(out_file, job_meta, sample_results):
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_file, "w") as f:
        gmeta = f.create_group("meta")
        for k, v in job_meta.items():
            if isinstance(v, str):
                gmeta.attrs[k] = v
            elif np.isscalar(v):
                gmeta.attrs[k] = v
            else:
                gmeta.create_dataset(k, data=np.asarray(v))

        f.create_dataset("rbins", data=np.asarray(RBINS))

        for sname, res in sample_results.items():
            g = f.create_group(sname)

            gsm = g.create_group("meta")
            for k, v in res["meta"].items():
                if np.isscalar(v):
                    gsm.attrs[k] = v
                else:
                    gsm.create_dataset(k, data=np.asarray(v))

            for key, val in res.items():
                if key == "meta":
                    continue
                gg = g.create_group(key)
                gg.create_dataset("mean", data=np.asarray(val["mean"]))
                gg.create_dataset("cov", data=np.asarray(val["cov"]))

    return out_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flag", required=True, help="Simulation flag, e.g. GR, F40, F45, F50, F55, F60")
    ap.add_argument("--snap", required=True, type=int, help="Snapshot number, e.g. 6, 12, 15")
    ap.add_argument("--out", default=None, help="Output HDF5 path. If omitted, save to IA_analysis/cfs.")
    ap.add_argument("--base-dir", default=str(BASE_DIR), help="Catalog base directory")
    ap.add_argument("--boxsize", type=float, default=205.0)
    ap.add_argument("--cos-err-max", type=float, default=0.01)
    ap.add_argument("--nsub", type=int, default=3, help="Jackknife subdivisions per axis; total JK regions = nsub^3")
    ap.add_argument("--nthreads", type=int, default=8)
    args = ap.parse_args()

    mg_file = catalog_path(args.flag, args.snap, args.base_dir)
    if not Path(mg_file).exists():
        raise FileNotFoundError(f"Missing catalog: {mg_file}")

    out_file = Path(args.out) if args.out is not None else default_out_path(args.flag, args.snap)

    print(f"[input]  {mg_file}")
    print(f"[output] {out_file}")

    pos_all, I_all, cos_err_all, mstar_all, host_all = read_catalog(mg_file)

    sample_results = {}
    for sname, mlo, mhi in MASS_BINS:
        pos, ori, host = select_mass_bin(
            pos_all, I_all, cos_err_all, mstar_all, host_all,
            cos_err_max=args.cos_err_max, mlo=mlo, mhi=mhi, los=LOS
        )
        print(f"[sample] {sname}  N={len(pos)}  mstar[{mlo},{mhi})  cos_err<{args.cos_err_max}")

        if len(pos) < 50:
            print("  [skip] too few objects")
            continue

        res = measure_with_jackknife(
            pos, ori, host,
            rbins=RBINS, Lbox=float(args.boxsize),
            nsub=int(args.nsub), nthreads=int(args.nthreads),
            label=f"{args.flag}_s{args.snap:03d}_{sname}"
        )

        res["meta"]["sample"] = sname
        res["meta"]["mlo"] = float(mlo)
        res["meta"]["mhi"] = float(mhi)
        res["meta"]["N"] = int(len(pos))

        sample_results[sname] = res

    job_meta = dict(
        flag=str(args.flag),
        snap=int(args.snap),
        catalog=str(mg_file),
        base_dir=str(args.base_dir),
        Lbox=float(args.boxsize),
        cos_err_max=float(args.cos_err_max),
        nsub=int(args.nsub),
        Njk=int(args.nsub**3),
        nthreads=int(args.nthreads),
        rmin=float(RMIN),
        rmax=float(RMAX),
        nbin=int(NBIN),
        los=np.array([0.0, 0.0, 1.0], dtype=float),
    )

    saved = save_job_hdf5(out_file, job_meta, sample_results)
    print("[saved]", saved)


if __name__ == "__main__":
    main()
