"""Exported code from notebooks/raw_20260618/ia_corr_abundance.ipynb.

This file is generated for project management and refactoring. Review before running end to end.
"""


# %% [markdown] cell 1
# # Abundance-selected CFs (xi, EE, ED, w++, wg+) with jackknife This notebook computes correlation functions for **fixed number-density selections** (top-N by `Mstar` or `SFR`). Statistics: - $\xi(r)$ (3D clustering) - EE$(r)$ and ED$(r)$ (3D; includes 1h/2h decomposition via Halotools) - $w_{++}(r_p)$ and $w_{g+}(r_p)$ (projected; **no 1h/2h decomposition**) Selections: - `1e-3` density → **8610** galaxies (code `103`) - `1e-4` density → **861** galaxies (code `104`) **Outputs:** one HDF5 per (m

# %% code cell 2
# --- Imports ---
from pathlib import Path
import numpy as np
import h5py
import time

from tqdm.auto import tqdm

from halotools.mock_observables import (
    tpcf, ee_3d, ed_3d,
    tpcf_one_two_halo_decomp,
    ee_3d_one_two_halo_decomp,
    ed_3d_one_two_halo_decomp,
    ii_plus_projected,
    gi_plus_projected,
)

from Iana import epsilon_from_shape_matrix

print("Imports OK.")

# %% [markdown] cell 3
# ## 1) Configuration

# %% code cell 4
BASE_DIR = Path("/cosma8/data/dp203/dc-wang17/MG_global")
OUT_DIR  = Path("/cosma/home/dp203/dc-wang17/IA_analysis/cfs")

# Run multiple catalogs (choose one):
flags = ["GR","F40","F45","F50","F55","F60"]          # e.g. ["GR","F40","F45","F50","F55","F60"]
snaps =  [6,12,15,16,17,18,19,20,21]           # e.g. [6,12,15,21]
jobs = None              # e.g. [("GR",6), ("F40",6), ("F55",21)]

boxsize = 205.0
cos_err_max = 0.01
nsub = 3
nthreads = 16

LOS = np.array([0.0, 0.0, 1.0], dtype=float)

# 3D r-bins for xi/EE/ED: logspace 0.1-20 Mpc/h, 20 bins
RMIN, RMAX = 0.1, 20.0
NBIN = 20
RBINS = np.logspace(np.log10(RMIN), np.log10(RMAX), NBIN + 1)

# Projected bins for w++ and wg+ (use same edges in rp)
RP_BINS = RBINS.copy()
PI_MAX = RMAX   # adjust if desired (e.g. 60.0)

# Abundance selections: exact target counts + naming code
ABUND = [
    ("103", 8610),  # ~1e-3 for Lbox=205
    ("104", 861),   # ~1e-4 for Lbox=205
]
SELECTION_MODES = ["Mstar", "SFR"]

def catalog_path(flag, snap, base=BASE_DIR):
    return Path(base) / f"L302_N1136_{flag}_s{int(snap):03d}.hdf5"

def output_path(mode, code, flag, snap, out_dir=OUT_DIR):
    return Path(out_dir) / f"{mode}_{code}_{flag}_{int(snap):03d}.hdf5"

def iter_jobs():
    if jobs is not None:
        for fl, sn in jobs:
            yield str(fl), int(sn)
    else:
        for fl in flags:
            for sn in snaps:
                yield str(fl), int(sn)

print("Planned jobs:")
for fl, sn in list(iter_jobs())[:]:
    print(" ", fl, f"s{sn:03d}")

# %% [markdown] cell 5
# ## 2) Core functions

# %% code cell 6
def read_catalog(mg_file):
    with h5py.File(mg_file, "r") as f:
        pos = f["pos_abs"][:] / 1000.0
        I = f["Star"]["I"][:]
        cos_err = f["Star"]["cos_err"][:]
        mstar = f["SubhaloMassInRadType"][:, 4]
        sfr = f["SubhaloSFR"][:]
        host = f["CenID"][:]

    cos_err = np.abs(cos_err)
    if cos_err.ndim > 1:
        cos_err = np.max(cos_err, axis=tuple(range(1, cos_err.ndim)))
    return pos, I, cos_err, mstar, sfr, host


def abundance_mask(prop, base_mask, Ntarget):
    idx = np.where(base_mask & np.isfinite(prop))[0]
    if idx.size == 0:
        return np.zeros_like(base_mask, dtype=bool), np.nan
    if Ntarget > idx.size:
        Ntarget = idx.size

    vals = prop[idx]
    k = vals.size - Ntarget
    thr = np.partition(vals, k)[k]
    sel = base_mask & np.isfinite(prop) & (prop >= thr)

    sel_idx = np.where(sel)[0]
    if sel_idx.size > Ntarget:
        order = np.argsort(prop[sel_idx])[::-1]
        keep_idx = sel_idx[order[:Ntarget]]
        sel2 = np.zeros_like(base_mask, dtype=bool)
        sel2[keep_idx] = True
        sel = sel2

    return sel, float(thr)


def build_vectors(pos, I, *, los):
    # 3D major axis from I
    w, v = np.linalg.eigh(I)
    ori3 = v[:, :, 2]
    nrm = np.linalg.norm(ori3, axis=1)
    good = np.isfinite(nrm) & (nrm > 0.0)

    ori3 = ori3[good] / nrm[good, None]
    pos3 = pos[good]

    # sign convention
    sgn = np.sign(ori3 @ los)
    sgn[sgn == 0.0] = 1.0
    ori3 = ori3 * sgn[:, None]

    # 2D ellipticity + 2D orientation for projected stats
    e1, e2 = epsilon_from_shape_matrix(I[good], los=los, apply_responsivity=False)
    e1 = np.asarray(e1, float)
    e2 = np.asarray(e2, float)
    ell = e1 + 1j * e2  # complex ellipticity, shape (N,)

    phi = 0.5 * np.arctan2(e2, e1)
    ori2 = np.column_stack([np.cos(phi), np.sin(phi)])

    ok2 = (
        np.isfinite(e1) & np.isfinite(e2)
        & np.isfinite(ori2).all(axis=1)
        & ((e1*e1 + e2*e2) > 0.0)
    )

    # idx_map maps back to the selected-array indexing
    idx_map = np.where(good)[0][ok2]
    return pos3[ok2], ori3[ok2], ori2[ok2], ell[ok2], idx_map


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


def measure_one(pos, host, ori3, ori2, ell, *, rbins, rp_bins, pi_max, Lbox, nthreads):
    # totals
    xi_tot = tpcf(pos, rbins, period=Lbox, num_threads=nthreads)
    ee_tot = ee_3d(pos, ori3, pos, ori3, rbins, period=Lbox, num_threads=nthreads)
    ed_tot = ed_3d(pos, ori3, pos, rbins, period=Lbox, num_threads=nthreads)

    # 1h/2h for xi, EE, ED
    xi_1h, xi_2h = tpcf_one_two_halo_decomp(
        sample1=pos, rbins=rbins, sample1_host_halo_id=host,
        period=Lbox, num_threads=nthreads
    )
    ee_1h, ee_2h = ee_3d_one_two_halo_decomp(
        sample1=pos, orientations1=ori3,
        sample2=pos, orientations2=ori3,
        rbins=rbins,
        sample1_host_halo_id=host, sample2_host_halo_id=host,
        period=Lbox, num_threads=nthreads
    )
    ed_1h, ed_2h = ed_3d_one_two_halo_decomp(
        sample1=pos, orientations1=ori3,
        sample2=pos,
        rbins=rbins,
        sample1_host_halo_id=host, sample2_host_halo_id=host,
        period=Lbox, num_threads=nthreads
    )

    # projected w++ and wg+ (no 1h/2h)
    wpp = ii_plus_projected(
        pos, ori2, ell,
        pos, ori2, ell,
        rp_bins,
        pi_max=pi_max,
        period=Lbox,
        num_threads=nthreads
    )
    wgp = gi_plus_projected(
        pos, ori2, ell,
        pos,
        rp_bins,
        pi_max=pi_max,
        period=Lbox,
        num_threads=nthreads
    )

    return dict(
        xi_tot=xi_tot, xi_1h=xi_1h, xi_2h=xi_2h,
        ee_tot=ee_tot, ee_1h=ee_1h, ee_2h=ee_2h,
        ed_tot=ed_tot, ed_1h=ed_1h, ed_2h=ed_2h,
        wpp=wpp, wgp=wgp
    )


def measure_with_jackknife(pos, host, ori3, ori2, ell, *, rbins, rp_bins, pi_max, Lbox, nsub, nthreads, desc="JK"):
    tags = jk_tags(pos, Lbox, nsub)
    njk = nsub**3

    keys = ["xi_tot","xi_1h","xi_2h","ee_tot","ee_1h","ee_2h","ed_tot","ed_1h","ed_2h","wpp","wgp"]
    stacks = {k: [] for k in keys}

    t0 = time.time()
    for j in tqdm(range(njk), desc=desc, total=njk, leave=True):
        keep = (tags != j)
        out = measure_one(
            pos[keep], host[keep],
            ori3[keep], ori2[keep], ell[keep],
            rbins=rbins, rp_bins=rp_bins, pi_max=pi_max,
            Lbox=Lbox, nthreads=nthreads
        )
        for k in keys:
            stacks[k].append(out[k])

    res = {"meta": dict(rbins=rbins, rp_bins=rp_bins, pi_max=pi_max, Lbox=Lbox, nsub=nsub, Njk=njk, nthreads=nthreads)}
    for k in keys:
        mu, cov = jk_mean_cov(np.asarray(stacks[k]))
        res[k] = dict(mean=mu, cov=cov)

    print(f"[done] {desc}  walltime={(time.time()-t0)/60:.1f} min")
    return res


def save_hdf5(out_file, res, *, meta_attrs):
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_file, "w") as f:
        gmeta = f.create_group("meta")
        for k, v in meta_attrs.items():
            if isinstance(v, str):
                gmeta.attrs[k] = v
            elif np.isscalar(v):
                gmeta.attrs[k] = v
            else:
                gmeta.create_dataset(k, data=np.asarray(v))

        # save stats
        for k, v in res.items():
            if k == "meta":
                continue
            g = f.create_group(k)
            g.create_dataset("mean", data=np.asarray(v["mean"]))
            g.create_dataset("cov", data=np.asarray(v["cov"]))

        # save bins at root
        f.create_dataset("rbins", data=np.asarray(res["meta"]["rbins"]))
        f.create_dataset("rp_bins", data=np.asarray(res["meta"]["rp_bins"]))

    return out_file

# %% [markdown] cell 7
# ## 3) Run: one selection → one file For each `(flag,snap)` and each selection `(mode, code)`, the notebook computes CFs and **immediately saves** the HDF5.

# %% code cell 8
for flag, snap in iter_jobs():
    mg_file = catalog_path(flag, snap)
    if not mg_file.exists():
        print(f"[skip] missing catalog: {mg_file}")
        continue

    print(f"\n=== CATALOG flag={flag} snap={snap:03d} ===")
    print("Catalog:", mg_file)

    pos_all, I_all, cos_err_all, mstar_all, sfr_all, host_all = read_catalog(mg_file)

    base = (
        np.isfinite(pos_all).all(axis=1)
        & np.isfinite(I_all).all(axis=(1,2))
        & np.isfinite(cos_err_all)
        & np.isfinite(mstar_all)
        & np.isfinite(sfr_all)
        & np.isfinite(host_all)
        & (cos_err_all < cos_err_max)
    )

    for mode in SELECTION_MODES:
        prop = mstar_all if mode == "Mstar" else sfr_all

        for code, Ntarget in ABUND:
            # output file for this selection
            out_file = output_path(mode, code, flag, snap)

            # --- NEW: if file exists, skip this task ---
            if out_file.exists():
                print(f"[skip] exists: {out_file}")
                continue

            sel, thr = abundance_mask(prop, base, Ntarget)

            pos_sel = pos_all[sel]
            I_sel = I_all[sel]
            host_sel = host_all[sel]

            pos, ori3, ori2, ell, idx_map = build_vectors(pos_sel, I_sel, los=LOS)
            host = host_sel[idx_map]

            print(f"[sel] {mode} {code}  N={len(pos)}  thr={thr:.6g}")

            if len(pos) < 50:
                print("  [skip] too few objects after cuts")
                continue

            res = measure_with_jackknife(
                pos, host, ori3, ori2, ell,
                rbins=RBINS, rp_bins=RP_BINS, pi_max=PI_MAX,
                Lbox=boxsize, nsub=nsub, nthreads=nthreads,
                desc=f"{mode} {code} {flag} s{snap:03d}"
            )

            meta_attrs = dict(
                mode=mode, code=code, Ntarget=int(Ntarget), threshold=float(thr),
                flag=flag, snap=int(snap), Lbox=float(boxsize),
                cos_err_max=float(cos_err_max),
                nsub=int(nsub), Njk=int(nsub**3), nthreads=int(nthreads),
                pi_max=float(PI_MAX)
            )
            saved = save_hdf5(out_file, res, meta_attrs=meta_attrs)
            print("[saved]", saved)

# %% [markdown] cell 9
# ## Notes - EE/ED are 3D and include 1h/2h decomp. - w++/wg+ are projected and do not include 1h/2h decomp. - All statistics include jackknife mean/cov.

# %% code cell 10
print('ALL DONE!')

# %% code cell 11
test_cf= h5py.File('/cosma/home/dp203/dc-wang17/IA_analysis/cfs/Mstar_103_GR_006.hdf5','r')

# %% code cell 12
test_cf.keys()

# %% code cell 13
