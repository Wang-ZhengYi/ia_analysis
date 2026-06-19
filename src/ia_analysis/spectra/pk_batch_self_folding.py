#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Batch driver for self-folded IA power-spectrum production.

Purpose
-------
This module coordinates repeated folded-mesh and power-spectrum measurements
across sample selections, fold factors, and output products.

Provides
--------
- Command-line batching around the catalog mesh and power-spectrum modules.
- Reusable fold/sample loops for ClusterSims power-spectrum jobs.
- Output naming and bookkeeping for large production runs.

Notes
-----
The module is an orchestration layer inside the spectra package.  It should not
own low-level mesh painting or power-spectrum math.
"""

from pathlib import Path
import argparse
import sys

if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print(
        """usage: pk_batch_self_folding.py --flag FLAG --snap SNAP [options]

Batch self-folding power-spectrum pipeline.

core options:
  -h, --help
  --flag FLAG
  --snap SNAP
  --threads THREADS
  --outdir OUTDIR
"""
    )
    raise SystemExit(0)

import h5py
import numpy as np
import pyccl as ccl

from ia_analysis.spectra import CatMesh
from ia_analysis.shapes.Iana import epsilon_from_shape_matrix
from ia_analysis.spectra.powers import PowerConfig, PowerSpectrumEstimator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flag', required=True)
    parser.add_argument('--snap', required=True, type=int)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--outdir', default='/cosma/home/dp203/dc-wang17/IA_analysis/pks')
    args = parser.parse_args()

    data_root = Path('/cosma8/data/dp203/dc-wang17/MG_global/conv')
    file_path = data_root / f'L302_N1136_{args.flag}_s{args.snap:03d}.hdf5'
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f'pks_{args.flag}_{args.snap:03d}.hdf5'

    snap_list = [6, 12, 15, 20]
    zlist = [0.97, 0.51, 0.33, 0.05]
    z = dict(zip(snap_list, zlist))[args.snap]

    boxsize = 302.0
    nmesh = 1024
    fold_list = [1, 2, 4, 8, 16, 32]
    alpha = 0.5
    overlap = 0.999
    los = (0, 0, 1)
    assign = 'CIC'
    cos_err_max = 0.01
    selection_mode_list = ['Mstar', 'SFR']
    number_density_list = [1e-4, 1e-3]

    Omega_c = 0.3089 - 0.0486
    Omega_b = 0.0486
    h = 0.6774
    sigma8 = 0.8159
    n_s = 0.9667
    cosmo = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=h, sigma8=sigma8, n_s=n_s)

    with h5py.File(file_path, 'r') as f:
        pos_all = f['pos_abs'][:]
        vel_all = f['vel_abs'][:]
        I_all = f['Star']['I'][:]
        cos_err = f['Star']['cos_err'][:]
        mstar_all = f['SubhaloMassInRadType'][:, 4]
        sfr_all = f['SubhaloSFR'][:]

    if cos_err.ndim > 1:
        cos_err = np.max(np.abs(cos_err), axis=tuple(range(1, cos_err.ndim)))
    else:
        cos_err = np.abs(cos_err)

    base_mask = np.isfinite(pos_all).all(axis=1)
    base_mask &= np.isfinite(vel_all).all(axis=1)
    base_mask &= np.isfinite(I_all).all(axis=(1, 2))
    base_mask &= np.isfinite(cos_err)
    base_mask &= (cos_err < cos_err_max)

    print(f'#####flag={args.flag} snap={args.snap} z={z:.6f}#####')

    with h5py.File(outfile, 'w') as fout:
        fout.attrs['flag'] = args.flag
        fout.attrs['snap'] = args.snap
        fout.attrs['z'] = z
        fout.attrs['boxsize'] = boxsize
        fout.attrs['nmesh'] = nmesh
        fout.attrs['alpha'] = alpha
        fout.attrs['overlap'] = overlap
        fout.attrs['los'] = los
        fout.attrs['assign'] = assign
        fout.attrs['cos_err_max'] = cos_err_max
        fout.attrs['Omega_c'] = Omega_c
        fout.attrs['Omega_b'] = Omega_b
        fout.attrs['h'] = h
        fout.attrs['sigma8'] = sigma8
        fout.attrs['n_s'] = n_s

        summary = []

        for selection_mode in selection_mode_list:
            prop = mstar_all if selection_mode == 'Mstar' else sfr_all

            for nbar in number_density_list:
                nbar_label = f'{nbar:.0e}'.replace('e-0', 'e-')
                print(f'---sample={selection_mode} nbar={nbar_label}---')

                target_count = int(boxsize**3 * nbar)
                threshold = np.percentile(prop, 100 * (1 - target_count / len(prop)))
                sample_mask = base_mask & (prop >= threshold)

                pos = pos_all[sample_mask]
                vel = vel_all[sample_mask]
                I = I_all[sample_mask]
                prop_sel = prop[sample_mask]
                cos_err_sel = cos_err[sample_mask]

                eps1, eps2 = epsilon_from_shape_matrix(I, los=los, apply_responsivity=False)
                R = 1.0 - 0.5 * np.mean(eps1**2 + eps2**2)
                g1 = eps1 / (2.0 * R)
                g2 = eps2 / (2.0 * R)

                case = fout.create_group(f'{selection_mode}/{nbar_label}')
                case.attrs['threshold'] = threshold
                case.attrs['target_count'] = target_count
                case.attrs['selected_count'] = len(pos)
                case.attrs['R'] = R
                case.attrs['mean_cos_err'] = cos_err_sel.mean() if len(cos_err_sel) else np.nan
                case.attrs['max_cos_err'] = cos_err_sel.max() if len(cos_err_sel) else np.nan
                case.attrs['mean_prop'] = prop_sel.mean() if len(prop_sel) else np.nan

                results = {}

                for fold in fold_list:
                    print(f'========================fold={fold}========================')

                    cfg = CatMesh.CatalogMeshConfig(
                        boxsize=boxsize,
                        nmesh=nmesh,
                        mas_gal=assign,
                        mas_shape=assign,
                        pos_unit='Mpc/h',
                    )
                    builder = CatMesh.CatalogMeshBuilder(cfg, cosmo=cosmo)

                    out = builder.build(
                        pos=pos,
                        vel=vel,
                        vmesh='theta',
                        z=z,
                        e1=g1,
                        e2=g2,
                        e_are_gamma=True,
                        los=los,
                        space='real',
                        folding_factor=fold,
                        verbose=False,
                    )

                    meshes = {
                        'g': out['g_mesh'],
                        'E': out['E_mesh'],
                        'B': out['B_mesh'],
                        't': out['theta_mesh'],
                    }

                    est = PowerSpectrumEstimator(PowerConfig(boxsize=boxsize / fold, los=los, ells=(0,), threads=args.threads))
                    pk_native = est.compute(
                        meshes=meshes,
                        pairs=[('g','g'),('g','E'),('E','E'),('B','B'),('t','t'),('g','t'),('t','E')],
                        mas={'g': assign, 'E': assign, 'B': assign, 't': 'None'},
                        verbose=False,
                    )

                    pk = {
                        'k': pk_native['k'],
                        'P_gg': pk_native['P_gg'],
                        'P_gE': pk_native['P_gE'],
                        'P_EE': pk_native['P_EE'],
                        'P_BB': pk_native['P_BB'],
                        'P_tt': pk_native['P_tt'],
                        'P_gt': pk_native['P_gt'],
                        'P_tE': pk_native['P_tE'],
                    }

                    Lf = boxsize / fold
                    n_eff = len(pos) / Lf**3
                    gg_noise = np.full_like(pk['k'], 1.0 / n_eff)
                    shape_var = 0.5 * np.mean(g1**2 + g2**2)
                    EE_noise = np.full_like(pk['k'], shape_var / n_eff)
                    BB_noise = np.full_like(pk['k'], shape_var / n_eff)
                    dv = vel - vel.mean(axis=0, keepdims=True)
                    sigma1d = np.sqrt(np.mean(dv**2) / 3.0)
                    a = 1.0 / (1.0 + z)
                    Hz = 100.0 * h * ccl.h_over_h0(cosmo, a)
                    dx = Lf / nmesh
                    ke = np.sin(pk['k'] * dx) / dx
                    tt_noise = sigma1d**2 * ke**2 / ((a * Hz)**2 * n_eff)

                    noise = {
                        'P_gg': gg_noise,
                        'P_gE': np.zeros_like(pk['k']),
                        'P_EE': EE_noise,
                        'P_BB': BB_noise,
                        'P_tt': tt_noise,
                        'P_gt': np.zeros_like(pk['k']),
                        'P_tE': np.zeros_like(pk['k']),
                    }
                    pk_corr = {}
                    for key in ['P_gg','P_gE','P_EE','P_BB','P_tt','P_gt','P_tE']:
                        pk_corr[key] = pk[key] - noise[key]

                    results[fold] = {'pk': pk, 'noise': noise, 'pk_corr': pk_corr}

                    gf = case.create_group(f'fold_{fold}')
                    gf.create_dataset('k', data=pk['k'])
                    for key in ['P_gg','P_gE','P_EE','P_BB','P_tt','P_gt','P_tE']:
                        gf.create_dataset(key, data=pk[key])
                        gf.create_dataset(key + '_noise', data=noise[key])
                        gf.create_dataset(key + '_corr', data=pk_corr[key])

                gs_raw = case.create_group('stitched_raw')
                gs_corr = case.create_group('stitched_corr')
                for key in ['P_gg','P_gE','P_EE','P_BB','P_tt','P_gt','P_tE']:
                    k_all = []
                    p_all = []
                    f_all = []
                    prev_kmax = None
                    for fold in fold_list:
                        k = results[fold]['pk']['k']
                        p = results[fold]['pk'][key]
                        kNy = np.pi * nmesh / (boxsize / fold)
                        kmax = alpha * kNy
                        if prev_kmax is None:
                            m = k <= kmax
                        else:
                            m = (k > overlap * prev_kmax) & (k <= kmax)
                        k_all.append(k[m])
                        p_all.append(p[m])
                        f_all.append(np.full(np.sum(m), fold))
                        prev_kmax = kmax
                    gs_raw.create_dataset(key + '_k', data=np.concatenate(k_all))
                    gs_raw.create_dataset(key + '_Pk', data=np.concatenate(p_all))
                    gs_raw.create_dataset(key + '_fold', data=np.concatenate(f_all))

                    k_all = []
                    p_all = []
                    f_all = []
                    prev_kmax = None
                    for fold in fold_list:
                        k = results[fold]['pk']['k']
                        p = results[fold]['pk_corr'][key]
                        kNy = np.pi * nmesh / (boxsize / fold)
                        kmax = alpha * kNy
                        if prev_kmax is None:
                            m = k <= kmax
                        else:
                            m = (k > overlap * prev_kmax) & (k <= kmax)
                        k_all.append(k[m])
                        p_all.append(p[m])
                        f_all.append(np.full(np.sum(m), fold))
                        prev_kmax = kmax
                    gs_corr.create_dataset(key + '_k', data=np.concatenate(k_all))
                    gs_corr.create_dataset(key + '_Pk', data=np.concatenate(p_all))
                    gs_corr.create_dataset(key + '_fold', data=np.concatenate(f_all))

                summary.append([
                    args.flag,
                    args.snap,
                    z,
                    selection_mode.encode(),
                    nbar_label.encode(),
                    target_count,
                    len(pos),
                    threshold,
                    R,
                    cos_err_sel.mean() if len(cos_err_sel) else np.nan,
                    cos_err_sel.max() if len(cos_err_sel) else np.nan,
                    prop_sel.mean() if len(prop_sel) else np.nan,
                ])

        dt = np.dtype([
            ('flag', 'S8'), ('snap', 'i4'), ('z', 'f8'), ('selection_mode', 'S16'), ('nbar', 'S8'),
            ('target_count', 'i8'), ('selected_count', 'i8'), ('threshold', 'f8'), ('R', 'f8'),
            ('mean_cos_err', 'f8'), ('max_cos_err', 'f8'), ('mean_prop', 'f8')
        ])
        arr = np.empty(len(summary), dtype=dt)
        for i, row in enumerate(summary):
            arr[i] = (row[0].encode(), row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11])
        fout.create_dataset('summary', data=arr)

    print(f'Saved {outfile}')


if __name__ == '__main__':
    main()
