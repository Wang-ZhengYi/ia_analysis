"""Exported code from notebooks/raw_20260618/ia_corr.ipynb.

This file is generated for project management and refactoring. Review before running end to end.
"""


# %% [markdown] cell 1
# # IA correlation functions: $\xi(r)$, EE$(r)$, ED$(r)$ with 1h/2h + jackknife This notebook measures 3D correlation functions from the MG catalogs: - $\xi(r)$ (galaxy clustering) - $\eta(r)$ (3D alignment / shape-shape) - $\omega(r)$ (3D alignment / density-shape) For each statistic we compute: - total - 1-halo and 2-halo pieces (Halotools built-ins) - spatial jackknife covariance (periodic box split into `nsub^3` regions) **Inputs:** `flag` (GR/F40/...), `snap` (int). **Catalog path:** `/cosma8

# %% code cell 2
# --- Imports ---
from pathlib import Path
import numpy as np
import h5py
import time
import seaborn as sns
from tqdm.auto import tqdm

from halotools.mock_observables import (
    tpcf, ee_3d, ed_3d,
    tpcf_one_two_halo_decomp,
    ee_3d_one_two_halo_decomp,
    ed_3d_one_two_halo_decomp,
)

print("Imports OK.")

# %% [markdown] cell 3
# ## 1) Configuration

# %% code cell 4
BASE_DIR = Path("/cosma8/data/dp203/dc-wang17/MG_global")
OUT_DIR  = Path("/cosma/home/dp203/dc-wang17/IA_analysis/cfs")

# Run multiple catalogs:
# - Either set flags/snaps to form the Cartesian product (flags x snaps),
# - Or define an explicit list of (flag, snap) pairs in `jobs` below.
flags = ["GR","F40","F45","F50","F55","F60"]     # e.g. ["GR","F40","F45","F50","F55","F60"]
snaps = [18]          # e.g. [6,12,15,21]

# Optional: explicit job list overrides flags x snaps if not None
jobs = None   # e.g. [("GR",6), ("F40",6), ("F55",21)]

boxsize = 205.0
cos_err_max = 0.01
nsub = 3          # 27 jackknife regions
nthreads = 16

LOS = np.array([0.0, 0.0, 1.0], dtype=float)

# rbins: logspace 0.1-20 Mpc/h, 20 bins
RMIN, RMAX = 0.1, 20.0
NBIN = 20
RBINS = np.logspace(np.log10(RMIN), np.log10(RMAX), NBIN + 1)

# mass bins in ia_pk_cs convention (mstar ~ 1 means 1e10 Msun/h)
MASS_BINS = [
    ("M10_11", 1.0, 10.0),      # 1e10 - 1e11
    ("M11_12", 10.0, 100.0),    # 1e11 - 1e12
]

def catalog_path(flag, snap, base=BASE_DIR):
    return Path(base) / f"L302_N1136_{flag}_s{int(snap):03d}.hdf5"

def output_path(flag, snap, out_dir=OUT_DIR):
    return Path(out_dir) / f"cf_{flag}_{int(snap):03d}.hdf5"

def iter_jobs():
    if jobs is not None:
        for fl, sn in jobs:
            yield str(fl), int(sn)
    else:
        for fl in flags:
            for sn in snaps:
                yield str(fl), int(sn)

print("Planned jobs:")
for fl, sn in list(iter_jobs())[:10]:
    print(" ", fl, f"s{sn:03d}")

# %% [markdown] cell 5
# ## 2) Functions (read → select → jackknife → measure) Details: - **3D orientations** are the **major-axis eigenvectors** of the 3×3 shape tensor `I`. - **1h/2h** uses Halotools `*_one_two_halo_decomp` with `CenID` as `host_halo_id`. - Decomp calls use **all keyword arguments** to avoid Halotools version signature differences.

# %% code cell 6
def read_catalog(mg_file):
    with h5py.File(mg_file, "r") as f:
        pos = f["pos_abs"][:] / 1000.0
        I = f["Star"]["I"][:]
        cos_err = f["Star"]["cos_err"][:]
        mstar = f["SubhaloMassInRadType"][:, 4]
        host = f["CenID"][:]

    cos_err = np.abs(cos_err)
    if cos_err.ndim > 1:
        cos_err = np.max(cos_err, axis=tuple(range(1, cos_err.ndim)))
    return pos, I, cos_err, mstar, host


def select_mass_bin(pos, I, cos_err, mstar, host, *, cos_err_max, mlo, mhi):
    ok = (
        np.isfinite(pos).all(axis=1)
        & np.isfinite(I).all(axis=(1,2))
        & np.isfinite(cos_err)
        & np.isfinite(mstar)
        & np.isfinite(host)
        & (cos_err < cos_err_max)
        & (mstar >= mlo) & (mstar < mhi)
    )

    pos = pos[ok]
    I = I[ok]
    host = host[ok]

    # 3D major-axis orientations from shape tensor: (N,3)
    w, v = np.linalg.eigh(I)     # eigenvalues ascending
    ori = v[:, :, 2]             # largest-eigenvalue eigenvector

    nrm = np.linalg.norm(ori, axis=1)
    good = np.isfinite(nrm) & (nrm > 0.0)
    pos, ori, host = pos[good], ori[good] / nrm[good, None], host[good]

    # optional sign convention: ori · LOS >= 0
    sgn = np.sign(ori @ LOS)
    sgn[sgn == 0.0] = 1.0
    ori = ori * sgn[:, None]

    return pos, ori, host


def jk_tags(pos, Lbox, nsub):
    p = np.mod(pos, Lbox) / Lbox
    ix = np.minimum((p[:,0] * nsub).astype(int), nsub-1)
    iy = np.minimum((p[:,1] * nsub).astype(int), nsub-1)
    iz = np.minimum((p[:,2] * nsub).astype(int), nsub-1)
    return ix + nsub*iy + (nsub*nsub)*iz


def jk_mean_cov(X):
    X = np.asarray(X, float)
    mu = np.mean(X, axis=0)
    d = X - mu[None,:]
    nj = X.shape[0]
    cov = (nj - 1.0)/nj * (d.T @ d)
    return mu, cov


def measure_one(pos, ori, host, *, rbins, Lbox, nthreads):
    xi_tot = tpcf(pos, rbins, period=Lbox, num_threads=nthreads)
    ee_tot = ee_3d(pos, ori, pos, ori, rbins, period=Lbox, num_threads=nthreads)
    ed_tot = ed_3d(pos, ori, pos, rbins, period=Lbox, num_threads=nthreads)

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


def measure_with_jackknife(pos, ori, host, *, rbins, Lbox, nsub, nthreads, desc="JK"):
    tags = jk_tags(pos, Lbox, nsub)
    njk = nsub**3

    keys = ["xi_tot","xi_1h","xi_2h", "ee_tot","ee_1h","ee_2h", "ed_tot","ed_1h","ed_2h"]
    stacks = {k: [] for k in keys}

    t0 = time.time()
    for j in tqdm(range(njk), desc=desc, total=njk, leave=True):
        keep = (tags != j)
        out = measure_one(pos[keep], ori[keep], host[keep], rbins=rbins, Lbox=Lbox, nthreads=nthreads)
        for k in keys:
            stacks[k].append(out[k])

    res = {"meta": dict(rbins=rbins, Lbox=Lbox, nsub=nsub, Njk=njk, nthreads=nthreads)}
    for k in keys:
        mu, cov = jk_mean_cov(np.asarray(stacks[k]))
        res[k] = dict(mean=mu, cov=cov)

    print(f"[done] {desc}  walltime={(time.time()-t0)/60:.1f} min")
    return res


def save_hdf5(out_file, results, *, flag, snap, rbins, Lbox, cos_err_max, nsub, nthreads):
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_file, "w") as f:
        for sname, res in results.items():
            g = f.create_group(sname)

            gmeta = g.create_group("meta")
            gmeta.attrs["flag"] = str(flag)
            gmeta.attrs["snap"] = int(snap)
            gmeta.attrs["Lbox"] = float(Lbox)
            gmeta.attrs["cos_err_max"] = float(cos_err_max)
            gmeta.attrs["nsub"] = int(nsub)
            gmeta.attrs["Njk"] = int(nsub**3)
            gmeta.attrs["nthreads"] = int(nthreads)
            gmeta.create_dataset("rbins", data=np.asarray(rbins))

            for k, v in res.items():
                if k == "meta":
                    continue
                gg = g.create_group(k)
                gg.create_dataset("mean", data=np.asarray(v["mean"]))
                gg.create_dataset("cov", data=np.asarray(v["cov"]))

    return out_file

# %% [markdown] cell 7
# ## 3) Run (two mass bins) and save HDF5

# %% code cell 8
# --- Run over all (flag, snap) jobs ---
for flag, snap in iter_jobs():
    mg_file = catalog_path(flag, snap)
    out_file = output_path(flag, snap)

    if not mg_file.exists():
        print(f"[skip] missing catalog: {mg_file}")
        continue

    print(f"\n=== RUN flag={flag} snap={snap:03d} ===")
    print("Catalog:", mg_file)
    print("Output :", out_file)

    pos_all, I_all, cos_err_all, mstar_all, host_all = read_catalog(mg_file)
    print("Loaded N_all =", len(pos_all))

    all_results = {}
    for sample_name, mlo, mhi in MASS_BINS:
        pos, ori, host = select_mass_bin(
            pos_all, I_all, cos_err_all, mstar_all, host_all,
            cos_err_max=cos_err_max, mlo=mlo, mhi=mhi
        )
        print(f"[sample] {sample_name}  N={len(pos)}  mstar[{mlo},{mhi})")

        all_results[sample_name] = measure_with_jackknife(
            pos, ori, host,
            rbins=RBINS, Lbox=boxsize,
            nsub=nsub, nthreads=nthreads,
            desc=f"{flag} s{snap:03d} {sample_name}"
        )

    saved = save_hdf5(
        out_file, all_results,
        flag=flag, snap=snap, rbins=RBINS, Lbox=boxsize,
        cos_err_max=cos_err_max, nsub=nsub, nthreads=nthreads
    )
    print("[saved]", saved)

# %% [markdown] cell 9
# ## 4) Quick sanity plots (optional)

# %% code cell 10
import matplotlib.pyplot as plt

def diag_err(cov):
    return np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))

r_cent = 0.5*(RBINS[:-1] + RBINS[1:])


ylabels=dict(
xi_tot=r"$\xi(r)$",
xi_1h=r"$\xi^{1h}(r)$",
xi_2h=r"$\xi(r)$",
ee_tot=r"$\eta(r)$",
ee_1h=r"$\eta^{1h}(r)$",
ee_2h=r"$\eta^{2h}(r)$",
ed_tot=r"$\omega(r)$",
ed_1h=r"$\omega^{1h}(r)$",
ed_2h=r"$\omega^{2h}(r)$",
)
def plot_key(key, title):
    plt.figure()
    sns.set(style='ticks')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['mathtext.fontset'] = 'cm'
    for sname in all_results:
        mu = all_results[sname][key]["mean"]
        err = diag_err(all_results[sname][key]["cov"])
        plt.errorbar(r_cent, mu, yerr=err, label=sname, fmt='o', ms=3, capsize=2)
    plt.xscale("log")
    plt.xlabel(r"$r [Mpc/h]$")
    plt.ylabel(ylabels[key])
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if key[:2]=='xi':
        plt.loglog()
    plt.show()

plot_key("xi_tot", r"$\xi(r)$"+" total")
plot_key("xi_1h",  r"$\xi(r)$"+" 1-halo")
plot_key("xi_2h",  r"$\xi(r)$"+" 2-halo")
plot_key("ee_tot", r"$\eta(r)$"+" total")
plot_key("ee_1h",  r"$\eta(r)$"+" 1-halo")
plot_key("ee_2h",  r"$\eta(r)$"+" 2-halo")
plot_key("ed_tot", r"$\omega(r)$"+" total")
plot_key("ed_1h",  r"$\omega(r)$"+" 1-halo")
plot_key("ed_2h",  r"$\omega(r)$"+" 2-halo")

# %% code cell 11

# %% code cell 12
