#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SnapMesh.py

Streaming mesh construction for ClusterSims particle snapshots.

This module builds, for one folding factor at a time:
- delta_mesh : total matter overdensity field from gas + DM + stars
- theta_mesh : particle velocity-divergence field theta_p = -(∇·v)/(aH)

Implementation notes
--------------------
- Particle chunks are read from HDF5, painted with MAS_library, and discarded
  immediately after use.
- delta and theta_p are built in the same streaming pass, so the snapshot is
  only read once per fold.
- The implementation is intentionally single-process. For 1024^3 meshes, a
  process pool is memory-inefficient because each worker would need its own set
  of large mesh arrays.
- After delta/theta meshes are built, large temporary arrays are explicitly
  deleted before returning.
"""

from dataclasses import dataclass
import gc

import numpy as np
try:
    import h5py
except Exception:  # pragma: no cover - optional HDF5 dependency
    h5py = None
try:
    import MAS_library as MASL
except Exception:  # pragma: no cover - optional HPC dependency
    MASL = None
try:
    import pyccl as ccl
except Exception:  # pragma: no cover - optional cosmology dependency
    ccl = None


def _iter_slices(n, chunk_size):
    if chunk_size is None or chunk_size <= 0 or chunk_size >= n:
        yield slice(0, n)
    else:
        a = 0
        while a < n:
            b = min(a + chunk_size, n)
            yield slice(a, b)
            a = b


def _pos_unit_factor_to_mpc_h(pos_unit, cosmo_h):
    u = str(pos_unit).strip().lower()
    if u in ('mpc/h', 'mpch', 'mpcph', 'mpc_per_h'):
        return 1.0
    if u in ('kpc/h', 'kpch', 'kpcph', 'kpc_per_h'):
        return 1.0 / 1000.0
    if u == 'mpc':
        if cosmo_h is None:
            raise ValueError("pos_unit='Mpc' requires cosmo_h")
        return float(cosmo_h)
    if u == 'kpc':
        if cosmo_h is None:
            raise ValueError("pos_unit='kpc' requires cosmo_h")
        return float(cosmo_h) / 1000.0
    raise ValueError("Unsupported pos_unit")


@dataclass
class SnapshotMeshConfig:
    boxsize: float
    nmesh: int
    mas: str = 'CIC'
    pos_unit: str = 'Mpc/h'


class SnapshotMeshBuilder:
    def __init__(self, config, cosmo=None):
        self.cfg = config
        self.cosmo = cosmo

    @staticmethod
    def _Hz_kms_per_Mpc(cosmo, z):
        if ccl is None:
            raise ImportError("Snapshot theta meshes require pyccl.")
        a = 1.0 / (1.0 + float(z))
        Ez = float(ccl.h_over_h0(cosmo, a))
        H0 = 100.0 * float(cosmo['h'])
        return H0 * Ez

    @staticmethod
    def _require_hdf5():
        if h5py is None:
            raise ImportError("SnapshotMeshBuilder requires h5py to read snapshot files.")

    @staticmethod
    def _require_mas_library():
        if MASL is None:
            raise ImportError("SnapshotMeshBuilder requires MAS_library from Pylians3.")

    def build_stream_cs(self, cat, z=None, ptypes=(0, 1, 4), dm_fixed_mass=1.0,
                        dm_ptype=1, folding_factor=1, boxsize_override=None,
                        mas=None, chunk_size=2_000_000,
                        require_masses_for_baryons=True, nworker=1,
                        two_pass=False, want_theta=None, verbose=True):
        """
        Build delta and optionally theta_p from ClusterSims particle snapshots.

        Parameters
        ----------
        cat : CSCatalog-like object
            Must provide _list_hdf5_files(prefix='snapdir').
        z : float, optional
            Required when want_theta=True.
        ptypes : tuple
            Particle types to include, usually gas(0), DM(1), stars(4).
        dm_fixed_mass : float
            Fixed DM particle mass used when Masses is absent for DM.
        folding_factor : int
            Self-folding factor. Effective box size is L/fold.

        Returns
        -------
        dict
            Always contains delta_mesh. Contains theta_mesh when want_theta=True.
        """
        self._require_hdf5()
        self._require_mas_library()

        mas_use = str(self.cfg.mas) if mas is None else str(mas)
        base_L = float(self.cfg.boxsize) if boxsize_override is None else float(boxsize_override)
        fold = int(folding_factor)
        if fold < 1:
            raise ValueError('folding_factor must be >= 1')
        L = base_L / float(fold)
        V = L ** 3
        N = int(self.cfg.nmesh)

        files = cat._list_hdf5_files(prefix='snapdir')
        if not files:
            raise FileNotFoundError('No snapdir_*.hdf5 files found')

        if want_theta is None:
            want_theta = True
        compute_theta = bool(want_theta)
        if compute_theta:
            if z is None:
                raise ValueError('z is required when want_theta=True')
            if self.cosmo is None:
                raise ValueError('cosmo is required when want_theta=True')

        cosmo_h = None if self.cosmo is None else float(self.cosmo['h'])
        fac = _pos_unit_factor_to_mpc_h(self.cfg.pos_unit, cosmo_h)

        if verbose:
            print(f'[SnapMesh] Nmesh={N} L={L:.6g} fold={fold} mas={mas_use} workers=1 theta={compute_theta}')

        Ntot = 0
        per_counts = {}
        total_weight = 0.0
        total_weight2 = 0.0
        # For the approximate particle-theta shot-noise model we need the
        # mass-weighted bulk velocity and the w^2-weighted velocity variance.
        # This is not a full velocity-field covariance; it is the natural
        # Poisson term associated with a mass-weighted sparse velocity sample.
        total_wv = np.zeros(3, dtype=np.float64)
        total_w2v = np.zeros(3, dtype=np.float64)
        total_w2v2 = 0.0

        rho_sum = np.zeros((N, N, N), dtype=np.float32)
        if compute_theta:
            px_sum = np.zeros_like(rho_sum)
            py_sum = np.zeros_like(rho_sum)
            pz_sum = np.zeros_like(rho_sum)

        for ifile, fp in enumerate(files, start=1):
            with h5py.File(fp, 'r') as f:
                for pt in ptypes:
                    gname = f'PartType{int(pt)}'
                    g = f.get(gname, None)
                    if g is None or 'Coordinates' not in g:
                        continue
                    if compute_theta and ('Velocities' not in g):
                        raise KeyError(f'{fp}:{gname} missing Velocities')

                    n = int(g['Coordinates'].shape[0])
                    if n <= 0:
                        continue

                    have_mass = 'Masses' in g
                    if (int(pt) != int(dm_ptype)) and require_masses_for_baryons and (not have_mass):
                        raise KeyError(f'{fp}:{gname} missing Masses')

                    for slc in _iter_slices(n, chunk_size):
                        pos = np.asarray(g['Coordinates'][slc], dtype=np.float64)
                        pos *= fac
                        pos = np.mod(pos, L)

                        if have_mass:
                            w = np.asarray(g['Masses'][slc], dtype=np.float64)
                        elif int(pt) == int(dm_ptype):
                            w = np.full(pos.shape[0], float(dm_fixed_mass), dtype=np.float64)
                        else:
                            w = np.ones(pos.shape[0], dtype=np.float64)

                        # Accumulate the exact mass-weighted particle shot-noise
                        # coefficient for delta_m = rho/<rho> - 1.  For a weighted
                        # particle field the white-noise level is
                        #   P_shot = V * sum_i w_i^2 / (sum_i w_i)^2.
                        # This replaces the equal-particle-mass approximation 1/nbar.
                        total_weight += float(np.sum(w, dtype=np.float64))
                        total_weight2 += float(np.sum(w * w, dtype=np.float64))

                        pos32 = pos.astype(np.float32, copy=False)
                        w32 = w.astype(np.float32, copy=False)
                        MASL.MA(pos32, rho_sum, L, mas_use, W=w32)

                        if compute_theta:
                            vel = np.asarray(g['Velocities'][slc], dtype=np.float64)
                            wv = w[:, None] * vel
                            w2 = w * w
                            total_wv += np.sum(wv, axis=0, dtype=np.float64)
                            total_w2v += np.sum(w2[:, None] * vel, axis=0, dtype=np.float64)
                            total_w2v2 += float(np.sum(w2 * np.sum(vel * vel, axis=1), dtype=np.float64))
                            MASL.MA(pos32, px_sum, L, mas_use, W=(w * vel[:, 0]).astype(np.float32))
                            MASL.MA(pos32, py_sum, L, mas_use, W=(w * vel[:, 1]).astype(np.float32))
                            MASL.MA(pos32, pz_sum, L, mas_use, W=(w * vel[:, 2]).astype(np.float32))
                            del vel, wv, w2

                        m = int(pos.shape[0])
                        Ntot += m
                        per_counts[gname] = per_counts.get(gname, 0) + m

                        del pos, w, pos32, w32

            if verbose:
                print(f'[SnapMesh] file progress: {ifile}/{len(files)}')

        if not compute_theta:
            mean_rho = float(np.mean(rho_sum, dtype=np.float64))
            if mean_rho > 0.0:
                rho_sum /= np.float32(mean_rho)
                rho_sum -= np.float32(1.0)
            else:
                rho_sum.fill(0.0)

            nbar = (Ntot / V) if Ntot > 0 else 0.0
            shot_number = (1.0 / nbar) if nbar > 0 else None
            shot_mass = (V * total_weight2 / (total_weight * total_weight)) if total_weight > 0.0 else None
            out = {
                'delta_mesh': rho_sum.astype(np.float32, copy=False),
                'meta': {
                    'mode': 'periodic_stream_serial',
                    'boxsize': float(L),
                    'volume': float(V),
                    'nmesh': int(N),
                    'mas': str(mas_use),
                    'folding_factor': int(fold),
                    'weighting': 'mass',
                    'N_p': int(Ntot),
                    'nbar_p': float(nbar),
                    # Mass-weighted shot noise appropriate for the constructed
                    # total-matter overdensity field.  The number-weighted value is
                    # kept for diagnostics only.
                    'shotnoise_dd': None if shot_mass is None else float(shot_mass),
                    'shotnoise_dd_number_weighted': None if shot_number is None else float(shot_number),
                    'total_weight': float(total_weight),
                    'total_weight2': float(total_weight2),
                    'stack_info': dict(per_ptype_counts=per_counts, N_total=int(Ntot)),
                }
            }
            gc.collect()
            return out

        px_sum = np.divide(px_sum, rho_sum, out=px_sum, where=(rho_sum != 0))
        vx_k = np.fft.fftn(px_sum)
        del px_sum
        gc.collect()

        py_sum = np.divide(py_sum, rho_sum, out=py_sum, where=(rho_sum != 0))
        vy_k = np.fft.fftn(py_sum)
        del py_sum
        gc.collect()

        pz_sum = np.divide(pz_sum, rho_sum, out=pz_sum, where=(rho_sum != 0))
        vz_k = np.fft.fftn(pz_sum)
        del pz_sum
        gc.collect()

        dx = float(L) / float(N)
        k1 = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
        div_k = 1j * (
            k1[:, None, None] * vx_k +
            k1[None, :, None] * vy_k +
            k1[None, None, :] * vz_k
        )
        del vx_k, vy_k, vz_k
        gc.collect()

        div_x = np.fft.ifftn(div_k).real.astype(np.float32)
        del div_k
        gc.collect()

        a = 1.0 / (1.0 + float(z))
        Hz = self._Hz_kms_per_Mpc(self.cosmo, float(z))
        h = float(self.cosmo['h'])
        theta = -(div_x * h) / (a * Hz)
        del div_x
        gc.collect()

        mean_rho = float(np.mean(rho_sum, dtype=np.float64))
        if mean_rho > 0.0:
            rho_sum /= np.float32(mean_rho)
            rho_sum -= np.float32(1.0)
        else:
            rho_sum.fill(0.0)

        nbar = (Ntot / V) if Ntot > 0 else 0.0
        shot_number = (1.0 / nbar) if nbar > 0 else None
        shot_mass = (V * total_weight2 / (total_weight * total_weight)) if total_weight > 0.0 else None

        velocity_shotnoise_1d = None
        velocity_bulk = np.full(3, np.nan, dtype=np.float64)
        if total_weight > 0.0:
            velocity_bulk = total_wv / total_weight
            # S_v = V * sum_i w_i^2 |v_i - v_bulk|^2 / [3 (sum_i w_i)^2].
            centered_w2v2 = (
                total_w2v2
                - 2.0 * float(np.dot(velocity_bulk, total_w2v))
                + float(np.dot(velocity_bulk, velocity_bulk)) * total_weight2
            )
            centered_w2v2 = max(0.0, float(centered_w2v2))
            velocity_shotnoise_1d = float(V * centered_w2v2 / (3.0 * total_weight * total_weight))

        out = {
            'delta_mesh': rho_sum.astype(np.float32, copy=False),
            'theta_mesh': theta.astype(np.float32, copy=False),
            'meta': {
                'mode': 'periodic_stream_serial',
                'boxsize': float(L),
                'volume': float(V),
                'nmesh': int(N),
                'mas': str(mas_use),
                'folding_factor': int(fold),
                'weighting': 'mass',
                'z': float(z),
                'a': float(a),
                'Hz_kms_per_Mpc': float(Hz),
                'N_p': int(Ntot),
                'nbar_p': float(nbar),
                # Mass-weighted shot noise appropriate for the constructed
                # total-matter overdensity field.  The number-weighted value is
                # kept for diagnostics only.
                'shotnoise_dd': None if shot_mass is None else float(shot_mass),
                'shotnoise_dd_number_weighted': None if shot_number is None else float(shot_number),
                'total_weight': float(total_weight),
                'total_weight2': float(total_weight2),
                'velocity_bulk_kms': velocity_bulk.astype(float),
                'velocity_shotnoise_1d': None if velocity_shotnoise_1d is None else float(velocity_shotnoise_1d),
                'stack_info': dict(per_ptype_counts=per_counts, N_total=int(Ntot)),
            }
        }
        gc.collect()
        return out
