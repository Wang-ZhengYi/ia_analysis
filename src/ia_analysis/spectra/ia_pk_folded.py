#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ia_pk_folded.py
===================================================

Measure self-folded power spectra for galaxy and particle fields in ClusterSims.

This script combines two data sources:
1. The precomputed MG/shape catalog, used for galaxy positions, galaxy velocities,
   and galaxy shape tensors.
2. The raw particle snapshot, used to build the total matter overdensity field and
   the particle-based velocity-divergence field.

Fields measured in this script
------------------------------
Galaxy-based fields:
- g   : galaxy overdensity field
- E   : E-mode intrinsic shape field
- B   : B-mode intrinsic shape field
- t   : galaxy-based velocity divergence theta, built from galaxy velocities

Particle-based fields:
- d   : total matter overdensity field, built from gas + DM + stars
- tp  : particle-based velocity divergence theta_p

Power spectra measured
----------------------
- P_gg, P_gE, P_EE, P_BB
- P_tt, P_gt, P_tE
- P_dd, P_dt, P_dE, P_dg
- P_tptp, P_dtp, P_tpt, P_gtp, P_Etp

Notes
-----
- The galaxy theta field (t) and particle theta field (tp) are kept strictly separate.
- Shape ellipticity is computed with Iana.epsilon_from_shape_matrix, then converted
  to gamma using gamma = epsilon / (2R), with R = 1 - <epsilon_i^2>.
- Self-folding uses CIC assignment, nmesh = 1024, and alpha = 0.5.
"""

from pathlib import Path
import argparse
import logging
import gc
import gc
import sys

if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print(
        """usage: ia_pk_folded.py --flag FLAG --snap SNAP [options]

Measure folded IA and matter power spectra for ClusterSims products.

core options:
  -h, --help
  --flag FLAG
  --snap SNAP
  --threads THREADS
  --outdir OUTDIR
  --nmesh NMESH
  --folds FOLDS
"""
    )
    raise SystemExit(0)

import h5py
import numpy as np
import pyccl as ccl

from ia_analysis.spectra import CatMesh
from ia_analysis.spectra import SnapMesh
from ia_analysis.shapes.Iana import epsilon_from_shape_matrix
from ia_analysis.spectra.powers import PowerConfig, PowerSpectrumEstimator
from ia_analysis.catalogs.catalog_loader import CSCatalog


def setup_logging(flag, snap):
    """Configure a simple logger and print the job header."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )
    logging.info('start flag=%s snap=%03d', flag, snap)


def get_cosmo():
    """
    Build the fiducial cosmology used throughout the script.

    Returns
    -------
    cosmo : ccl.Cosmology
        CCL cosmology object.
    pars : dict
        Plain dictionary version of the cosmological parameters, useful for
        storing metadata in the output HDF5 file.
    """
    Omega_c = 0.3089 - 0.0486
    Omega_b = 0.0486
    h = 0.6774
    sigma8 = 0.8159
    n_s = 0.9667

    cosmo = ccl.Cosmology(
        Omega_c=Omega_c,
        Omega_b=Omega_b,
        h=h,
        sigma8=sigma8,
        n_s=n_s,
    )

    pars = {
        'Omega_c': Omega_c,
        'Omega_b': Omega_b,
        'h': h,
        'sigma8': sigma8,
        'n_s': n_s,
    }
    return cosmo, pars


def load_mg_sample(mg_file, selection_mode, nbar, boxsize, cos_err_max, los):
    """
    Load a galaxy sample from the precomputed MG catalog and apply the selection.

    Selection rules
    ---------------
    - Keep only objects with finite position, velocity, and shape tensor.
    - Require Star/cos_err < cos_err_max.
    - Select the top objects in either Mstar or SFR to match the target number density.

    Parameters
    ----------
    mg_file : str or Path
        HDF5 file containing the galaxy catalog.
    selection_mode : {'Mstar', 'SFR'}
        Which galaxy property is used for the abundance cut.
    nbar : float
        Target number density in (Mpc/h)^-3.
    boxsize : float
        Simulation box size in Mpc/h.
    cos_err_max : float
        Maximum allowed shape error.
    los : tuple
        Line-of-sight direction passed to Iana.

    Returns
    -------
    dict
        Dictionary containing the selected sample and basic sample statistics.
    """
    with h5py.File(mg_file, 'r') as f:
        pos_all = f['pos_abs'][:] / 1000.0
        vel_all = f['vel_abs'][:]
        I_all = f['Star']['I'][:]
        cos_err = f['Star']['cos_err'][:]
        mstar_all = f['SubhaloMassInRadType'][:, 4]
        sfr_all = f['SubhaloSFR'][:]

    # Some catalogs store cos_err as a vector-like quantity. Reduce it to a single
    # conservative scalar per object by taking the maximum absolute component.
    if cos_err.ndim > 1:
        cos_err = np.max(np.abs(cos_err), axis=tuple(range(1, cos_err.ndim)))
    else:
        cos_err = np.abs(cos_err)

    # Build the base quality mask.
    base_mask = np.isfinite(pos_all).all(axis=1)
    base_mask &= np.isfinite(vel_all).all(axis=1)
    base_mask &= np.isfinite(I_all).all(axis=(1, 2))
    base_mask &= np.isfinite(cos_err)
    base_mask &= (cos_err < cos_err_max)

    # Choose the property used for abundance matching.
    prop = mstar_all if selection_mode == 'Mstar' else sfr_all

    # Convert target number density into a target object count.
    target_count = int(boxsize**3 * nbar)
    threshold = np.percentile(prop, 100 * (1 - target_count / len(prop)))
    mask = base_mask & (prop >= threshold)

    pos = pos_all[mask]
    vel = vel_all[mask]
    I = I_all[mask]
    prop_sel = prop[mask]
    cos_err_sel = cos_err[mask]

    # Convert shape tensors into ellipticity components using Iana.
    eps1, eps2 = epsilon_from_shape_matrix(I, los=los, apply_responsivity=False)

    # Explicitly follow the paper-style responsivity conversion:
    #   R = 1 - <epsilon_i^2>
    #   gamma = epsilon / (2R)
    R = 1.0 - 0.5 * np.mean(eps1**2 + eps2**2)
    g1 = eps1 / (2.0 * R)
    g2 = eps2 / (2.0 * R)

    return {
        'pos': pos,
        'vel': vel,
        'I': I,
        'g1': g1,
        'g2': g2,
        'threshold': threshold,
        'target_count': target_count,
        'selected_count': len(pos),
        'R': R,
        'mean_cos_err': cos_err_sel.mean() if len(cos_err_sel) else np.nan,
        'max_cos_err': cos_err_sel.max() if len(cos_err_sel) else np.nan,
        'mean_prop': prop_sel.mean() if len(prop_sel) else np.nan,
    }


def build_galaxy_meshes(pos, vel, g1, g2, boxsize, nmesh, assign, fold, z, cosmo, los):
    """
    Build galaxy-based meshes with CatMesh.

    Returned fields include:
    - g_mesh     : galaxy overdensity
    - E_mesh/B_mesh : shape E/B modes
    - theta_mesh : galaxy velocity-divergence field
    """
    cfg = CatMesh.CatalogMeshConfig(
        boxsize=boxsize,
        nmesh=nmesh,
        mas_gal=assign,
        mas_shape=assign,
        pos_unit='Mpc/h',
    )
    builder = CatMesh.CatalogMeshBuilder(cfg, cosmo=cosmo)
    return builder.build(
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


def build_particle_meshes(cs_cat, boxsize, nmesh, assign, fold, z, cosmo, threads, dm_fixed_mass):
    """
    Build particle-based meshes with SnapMesh.

    This uses gas + DM + stars to construct:
    - delta_mesh : total matter overdensity field
    - theta_mesh : particle-based velocity divergence theta_p

    The DM particle mass is fixed externally because ClusterSims DM particles do
    not carry an explicit Masses field in the same way as baryons.
    """
    cfg = SnapMesh.SnapshotMeshConfig(
        boxsize=boxsize,
        nmesh=nmesh,
        mas=assign,
        pos_unit='kpc/h',
    )
    builder = SnapMesh.SnapshotMeshBuilder(cfg, cosmo=cosmo)
    return builder.build_stream_cs(
        cs_cat,
        z=z,
        ptypes=(0, 1, 4),
        dm_fixed_mass=dm_fixed_mass,
        dm_ptype=1,
        folding_factor=fold,
        mas=assign,
        nworker=1,
        two_pass=True,
        want_theta=True,
        verbose=False,
    )


def measure_power(meshes, boxsize, fold, los, threads, assign):
    """
    Measure all required monopole power spectra for one folding factor.

    Field naming convention
    -----------------------
    g  : galaxy overdensity
    E  : E-mode shape
    B  : B-mode shape
    t  : galaxy theta
    d  : particle overdensity
    tp : particle theta
    """
    est = PowerSpectrumEstimator(
        PowerConfig(boxsize=boxsize / fold, los=los, ells=(0,), threads=threads)
    )

    pk_native = est.compute(
        meshes=meshes,
        pairs=[
            ('g', 'g'), ('g', 'E'), ('E', 'E'), ('B', 'B'),
            ('t', 't'), ('g', 't'), ('t', 'E'),
            ('d', 'd'), ('d', 't'), ('d', 'E'), ('d', 'g'),
            ('tp', 'tp'), ('d', 'tp'), ('tp', 't'), ('g', 'tp'), ('E', 'tp'),
        ],
        mas={
            'g': assign,
            'E': assign,
            'B': assign,
            't': 'None',
            'd': assign,
            'tp': 'None',
        },
        verbose=False,
    )

    return {
        'k': pk_native['k'],
        'P_gg': pk_native['P_gg'],
        'P_gE': pk_native['P_gE'],
        'P_EE': pk_native['P_EE'],
        'P_BB': pk_native['P_BB'],
        'P_tt': pk_native['P_tt'],
        'P_gt': pk_native['P_gt'],
        'P_tE': pk_native['P_tE'],
        'P_dd': pk_native['P_dd'],
        'P_dt': pk_native['P_dt'],
        'P_dE': pk_native['P_dE'],
        'P_dg': pk_native['P_dg'],
        'P_tptp': pk_native['P_tptp'],
        'P_dtp': pk_native['P_dtp'],
        'P_tpt': pk_native['P_tpt'],
        'P_gtp': pk_native['P_gtp'],
        'P_Etp': pk_native['P_Etp'],
    }


def estimate_noise(pk, gal_vel, g1, g2, selected_count, boxsize, nmesh, fold, z, h, cosmo, snap_meta):
    """
    Estimate simple noise terms for the measured spectra.

    Assumptions
    -----------
    - galaxy density g has shot noise 1/n
    - galaxy IA E/B have white shape noise
    - galaxy theta t has a k-dependent velocity-divergence noise model
    - particle delta d and particle theta_p tp are treated as effectively noise-free
      for the current use case
    """
    Lf = boxsize / fold
    n_eff = selected_count / Lf**3

    gg_noise = np.full_like(pk['k'], 1.0 / n_eff)
    shape_var = 0.5 * np.mean(g1**2 + g2**2)
    EE_noise = np.full_like(pk['k'], shape_var / n_eff)
    BB_noise = np.full_like(pk['k'], shape_var / n_eff)

    dv = gal_vel - gal_vel.mean(axis=0, keepdims=True)
    sigma1d_gal = np.sqrt(np.mean(dv**2) / 3.0)
    a = 1.0 / (1.0 + z)
    Hz = 100.0 * h * ccl.h_over_h0(cosmo, a)
    dx = Lf / nmesh
    ke = np.sin(pk['k'] * dx) / dx
    tt_noise = sigma1d_gal**2 * ke**2 / ((a * Hz)**2 * n_eff)

    z0 = np.zeros_like(pk['k'])
    return {
        'P_gg': gg_noise,
        'P_gE': z0.copy(),
        'P_EE': EE_noise,
        'P_BB': BB_noise,
        'P_tt': tt_noise,
        'P_gt': z0.copy(),
        'P_tE': z0.copy(),
        'P_dd': z0.copy(),
        'P_dt': z0.copy(),
        'P_dE': z0.copy(),
        'P_dg': z0.copy(),
        'P_tptp': z0.copy(),
        'P_dtp': z0.copy(),
        'P_tpt': z0.copy(),
        'P_gtp': z0.copy(),
        'P_Etp': z0.copy(),
    }

def write_fold_group(case, fold, pk, noise, pk_corr, delta_mesh_mean):
    """Write one folding block into the output HDF5 file."""
    gf = case.create_group(f'fold_{fold}')
    gf.create_dataset('k', data=pk['k'])
    gf.create_dataset('delta_mesh_mean', data=np.array([delta_mesh_mean]))

    keys = [
        'P_gg', 'P_gE', 'P_EE', 'P_BB', 'P_tt', 'P_gt', 'P_tE',
        'P_dd', 'P_dt', 'P_dE', 'P_dg', 'P_tptp', 'P_dtp', 'P_tpt', 'P_gtp', 'P_Etp',
    ]
    for key in keys:
        gf.create_dataset(key, data=pk[key])
        gf.create_dataset(key + '_noise', data=noise[key])
        gf.create_dataset(key + '_corr', data=pk_corr[key])


def write_stitched(case, results, fold_list, boxsize, nmesh, alpha, overlap):
    """
    Build stitched spectra from the individual folded measurements.

    For each fold we keep modes up to k_max = alpha * k_Ny and optionally remove
    a small overlap region with the previous fold.
    """
    gs_raw = case.create_group('stitched_raw')
    gs_corr = case.create_group('stitched_corr')

    keys = [
        'P_gg', 'P_gE', 'P_EE', 'P_BB', 'P_tt', 'P_gt', 'P_tE',
        'P_dd', 'P_dt', 'P_dE', 'P_dg', 'P_tptp', 'P_dtp', 'P_tpt', 'P_gtp', 'P_Etp',
    ]

    # First write the raw stitched spectra.
    for key in keys:
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

    # Then write the noise-subtracted stitched spectra.
    for key in keys:
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


def main():
    """Main entry point for one (flag, snap) job."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--flag', required=True, help='Cosmology flag, e.g. GR, F40, F45, F50, F55, F60.')
    parser.add_argument('--snap', required=True, type=int, help='Snapshot number.')
    parser.add_argument('--threads', type=int, default=8, help='Thread / worker count for mesh construction.')
    parser.add_argument('--outdir', default='/cosma/home/dp203/dc-wang17/IA_analysis/pks', help='Output directory for HDF5 files.')
    args = parser.parse_args()

    setup_logging(args.flag, args.snap)

    # Input and output paths.
    mg_root = Path('/cosma8/data/dp203/dc-wang17/MG_global/')
    mg_file = mg_root / f'L302_N1136_{args.flag}_s{args.snap:03d}.hdf5'
    cs_root = Path(f'/cosma8/data/dp203/bl267/Data/ClusterSims/L302_N1136_{args.flag}')
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f'pks_{args.flag}_{args.snap:03d}.hdf5'

    # Redshifts are specified explicitly for the snapshots we actually use.
    zmap = {6: 0.97, 12: 0.51, 15: 0.33, 21: 0.0}
    z = zmap[args.snap]

    # Main run configuration.
    boxsize = 205.0
    nmesh = 1024
    fold_list = [1, 2, 4, 8, 16, 32]
    alpha = 0.5
    overlap = 0.999
    los = (0, 0, 1)
    assign = 'CIC'
    cos_err_max = 0.01
    selection_mode_list = ['Mstar', 'SFR']
    number_density_list = [1e-4, 1e-3]
    CS_DM_FIXED_MASS = 1.35401e9

    cosmo, pars = get_cosmo()
    cs_cat = CSCatalog(cs_root, args.snap)

    with h5py.File(outfile, 'w') as fout:
        # Global metadata for the whole job.
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
        fout.attrs['CS_DM_FIXED_MASS'] = CS_DM_FIXED_MASS
        fout.attrs['cs_root'] = str(cs_root)
        fout.attrs['mg_file'] = str(mg_file)
        for k, v in pars.items():
            fout.attrs[k] = v

        summary = []
        total_cases = len(selection_mode_list) * len(number_density_list)
        icase = 0

        for selection_mode in selection_mode_list:
            for nbar in number_density_list:
                icase += 1
                nbar_label = f'{nbar:.0e}'.replace('e-0', 'e-')
                logging.info('case %d/%d | sample=%s nbar=%s | loading sample', icase, total_cases, selection_mode, nbar_label)

                sample = load_mg_sample(mg_file, selection_mode, nbar, boxsize, cos_err_max, los)

                case = fout.create_group(f'{selection_mode}/{nbar_label}')
                case.attrs['threshold'] = sample['threshold']
                case.attrs['target_count'] = sample['target_count']
                case.attrs['selected_count'] = sample['selected_count']
                case.attrs['R'] = sample['R']
                case.attrs['mean_cos_err'] = sample['mean_cos_err']
                case.attrs['max_cos_err'] = sample['max_cos_err']
                case.attrs['mean_prop'] = sample['mean_prop']

                results = {}
                for ifold, fold in enumerate(fold_list, start=1):
                    logging.info('case %d/%d | sample=%s nbar=%s | fold %d/%d start', icase, total_cases, selection_mode, nbar_label, ifold, len(fold_list))

                    # Build galaxy-based meshes.
                    gal_out = build_galaxy_meshes(
                        sample['pos'], sample['vel'], sample['g1'], sample['g2'],
                        boxsize, nmesh, assign, fold, z, cosmo, los,
                    )
                    logging.info('case %d/%d | sample=%s nbar=%s | fold %d/%d galaxy meshes done', icase, total_cases, selection_mode, nbar_label, ifold, len(fold_list))

                    # Build particle-based meshes.
                    snap_out = build_particle_meshes(
                        cs_cat, boxsize, nmesh, assign, fold, z, cosmo,
                        args.threads, CS_DM_FIXED_MASS,
                    )
                    logging.info('case %d/%d | sample=%s nbar=%s | fold %d/%d particle meshes done', icase, total_cases, selection_mode, nbar_label, ifold, len(fold_list))

                    # Keep theta from galaxies (t) and theta_p from particles (tp)
                    # strictly separate.
                    meshes = {
                        'g': gal_out['g_mesh'],
                        'E': gal_out['E_mesh'],
                        'B': gal_out['B_mesh'],
                        't': gal_out['theta_mesh'],
                        'd': snap_out['delta_mesh'],
                        'tp': snap_out['theta_mesh'],
                    }

                    pk = measure_power(meshes, boxsize, fold, los, args.threads, assign)
                    noise = estimate_noise(
                        pk,
                        sample['vel'],
                        sample['g1'],
                        sample['g2'],
                        sample['selected_count'],
                        boxsize,
                        nmesh,
                        fold,
                        z,
                        pars['h'],
                        cosmo,
                        snap_out['meta'],
                    )
                    pk_corr = {key: pk[key] - noise[key] for key in noise.keys()}

                    delta_mesh_mean = float(np.mean(snap_out['delta_mesh'], dtype=np.float64))
                    results[fold] = {
                        'pk': pk,
                        'noise': noise,
                        'pk_corr': pk_corr,
                    }

                    write_fold_group(case, fold, pk, noise, pk_corr, delta_mesh_mean)
                    case.file.flush()

                    del meshes
                    del gal_out
                    del snap_out
                    gc.collect()

                    logging.info('case %d/%d | sample=%s nbar=%s | fold %d/%d finished and memory cleaned', icase, total_cases, selection_mode, nbar_label, ifold, len(fold_list))

                write_stitched(case, results, fold_list, boxsize, nmesh, alpha, overlap)
                del results
                gc.collect()
                logging.info('case %d/%d | sample=%s nbar=%s | stitching done', icase, total_cases, selection_mode, nbar_label)

                summary.append([
                    args.flag.encode(),
                    args.snap,
                    z,
                    selection_mode.encode(),
                    nbar_label.encode(),
                    sample['target_count'],
                    sample['selected_count'],
                    sample['threshold'],
                    sample['R'],
                    sample['mean_cos_err'],
                    sample['max_cos_err'],
                    sample['mean_prop'],
                ])

        # Save a compact summary table.
        dt = np.dtype([
            ('flag', 'S8'),
            ('snap', 'i4'),
            ('z', 'f8'),
            ('selection_mode', 'S16'),
            ('nbar', 'S8'),
            ('target_count', 'i8'),
            ('selected_count', 'i8'),
            ('threshold', 'f8'),
            ('R', 'f8'),
            ('mean_cos_err', 'f8'),
            ('max_cos_err', 'f8'),
            ('mean_prop', 'f8'),
        ])
        arr = np.empty(len(summary), dtype=dt)
        for i, row in enumerate(summary):
            arr[i] = tuple(row)
        fout.create_dataset('summary', data=arr)

    logging.info('saved %s', outfile)


if __name__ == '__main__':
    main()
