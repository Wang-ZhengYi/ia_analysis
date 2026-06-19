#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ia_pk_cs.py
============

Measure self-folded power spectra for ClusterSims galaxy, intrinsic-alignment,
matter-density, and velocity fields.

This script is designed as the science driver on top of three fixed utility
modules:

- CatMesh.py   : builds galaxy overdensity, IA E/B, and galaxy theta meshes;
- SnapMesh.py  : builds total-matter overdensity and optional particle-theta meshes;
- powers.py    : measures P(k) multipoles and applies the self-folding volume
                 convention through ``power_norm_boxsize``.

Fields
------
The following mesh fields can be measured:

- g   : galaxy overdensity, delta_g = n_g/<n_g> - 1
- E   : E-mode intrinsic-alignment field
- B   : B-mode intrinsic-alignment field
- d   : total matter overdensity from gas + DM + stars
- t   : galaxy velocity-divergence field, theta_g = -div(v_g)/(aH)
- tp  : particle velocity-divergence field, theta_p = -div(v_p)/(aH)
- tm  : self-foldable momentum-divergence proxy from additive momentum density

Default spectra
---------------
The default ``--pk-types full`` measures a compact but broad set of auto- and
cross-power spectra relevant for IA, density fields, galaxy clustering, matter
velocity fields, and galaxy velocity fields:

    gg, gE, gB, EE, BB, EB,
    dd, dg, dE, dB,
    tt, gt, tE, tB, dt,
    tptp, dtp, gtp, Etp, Btp, tpt

Here ``t`` is the galaxy theta field, ``tp`` is the particle theta field, and
``tm`` is the momentum-divergence proxy.  ``tm`` is self-foldable, but it is not
the same estimator as the local velocity-divergence fields.  Request ``tm``
spectra explicitly with names such as ``dtm``, ``gtm``, or ``Etm`` after a
pipeline has supplied ``theta_momentum_mesh``.
For a smaller IA-only run use:

    --pk-types core

which expands to:

    gg,gE,EE,BB,dd,dg,dE

Sample definition
-----------------
This script constructs a full parent sample, fixed LRG/ELG-threshold samples,
and fixed-number-density samples.  The full parent sample is:

- all : all galaxies passing the common finite-value, cos_err, and Mstar>0 cuts,
        with shapes measured from the stellar shape tensor I.

The redshift-dependent LRG/ELG table used here is:

+------+-------------------------------+--------------------------+
| z    | ELG: log10(sSFR [yr^-1]) cut | LRG: Mstar [1e10 Msun]  |
+------+-------------------------------+--------------------------+
| 1.16 | -9.3                          | 7.5                      |
| 0.97 | -9.4                          | 7.4                      |
| 0.73 | -9.5                          | 8.2                      |
| 0.51 | -9.6                          | 8.2                      |
| 0.27 | -9.9                          | 8.6                      |
+------+-------------------------------+--------------------------+

Definitions:

- ELG sample: log10(SFR/Mstar) >= the tabulated log10(sSFR) threshold.
- LRG sample: Mstar >= the tabulated stellar-mass threshold.

By default the threshold at the nearest table redshift is used.  Linear
interpolation can be selected with ``--threshold-mode linear``.

Mass-unit convention
--------------------
ClusterSims/TNG-like catalog fields such as ``SubhaloMassInRadType[:,4]`` are
usually stored in units of 1e10 Msun/h.  The default
``--catalog-mass-unit 1e10Msun_h`` therefore converts stellar mass to physical
Msun for sSFR and converts the LRG threshold back to catalog units for selection.
Use ``--catalog-mass-unit 1e10Msun`` if your catalog is already in physical
1e10 Msun units.

Fixed-number-density samples
----------------------------
In addition to LRG and ELG, the script constructs top-ranked samples with
number densities

    1e-2, 1e-3, 1e-4  (h/Mpc)^3

using either stellar mass or SFR as the ranking variable:

- Mstar/1e-2, Mstar/1e-3, Mstar/1e-4
- SFR/1e-2,   SFR/1e-3,   SFR/1e-4

The target count is N = nbar * L_box^3, with L_box in Mpc/h.

Kinematic samples
-----------------
The previous global kappa_rot-based split is preserved:

- kappa_lt_045 : kappa_rot < 0.45, shape measured from stellar shape tensor I;
- kappa_gt_055 : kappa_rot > 0.55, shape measured from stellar spin L.

LRG and ELG are kept as standalone tracer samples; no LRG/ELG kappa, central,
or satellite subclasses are generated.


HDF5 output structure
---------------------
The output file is written to:

    {outdir}/pks_{FLAG}_{SNAP3}.hdf5

Root-level attributes store the run configuration, including ``flag``, ``snap``,
``z``, ``boxsize``, ``nmesh``, ``folds``, ``alpha``, ``overlap``, ``assign``,
``theta_mas``, ``cos_err_max``, ``pk_types``, ``spec_keys``, input paths, mass
unit convention, and sample-construction switches.

Root-level datasets are:

- target_k
    User-facing target k-array for interpolated stitched spectra.
- lrg_elg_table_z
- lrg_elg_table_log_ssfr_cut
- lrg_elg_table_mstar_cut_1e10_msun
    The threshold table used for LRG/ELG construction.
- summary
    Compact structured table with one row per sample.  It records the sample
    name, sample type, tracer label, shape mode, selected count, responsivity,
    LRG/ELG thresholds, and fixed-number-density thresholds where applicable.

Each sample is written as one HDF5 group.  Default sample groups are:

- all
- LRG
- ELG
- Mstar/1e-2, Mstar/1e-3, Mstar/1e-4
- SFR/1e-2, SFR/1e-3, SFR/1e-4
- kappa_lt_045, kappa_gt_055
- centrals, satellites   unless ``--no-central-satellite`` is used

For each sample group, attributes store the sample metadata.  For every fold f,
the group ``fold_f`` contains:

- k
    Native Pylians k-bin centers for this fold.
- delta_mesh_mean
    Mean of the matter overdensity mesh; should be close to zero.
- P_<type>
    Raw power spectrum in the original-box volume convention.
- P_<type>_noise
    Analytic additive noise estimate in the same convention.
- P_<type>_corr
    Noise-subtracted spectrum, ``P_<type> - P_<type>_noise``.

The stitched products are stored in six subgroups:

- stitched_native_raw
- stitched_native_corr
- stitched_native_noise
    Native k-bin stitched products.  For each spectrum ``P_<type>``, datasets are
    ``P_<type>_k``, ``P_<type>_Pk``, and ``P_<type>_fold``.
- stitched_raw
- stitched_corr
- stitched_noise
    Products interpolated to ``target_k``.  Each group contains ``k`` plus, for
    each spectrum, ``P_<type>_Pk`` and ``P_<type>_fold``.

Self-folding convention
-----------------------
For a fold factor f, CatMesh/SnapMesh build meshes in a folded box L/f.
The power estimator is called with:

    PowerConfig(boxsize=L/f, power_norm_boxsize=L)

so the returned spectra use the original simulation-volume convention and are
multiplied internally by f^3.  Do not multiply the output spectra by f^3 again.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import gc
import logging
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print(
        """usage: ia_pk_cs.py --flag FLAG --snap SNAP [options]

Measure self-folded power spectra for ClusterSims IA, density, and velocity fields.

core options:
  -h, --help
  --flag FLAG
  --snap SNAP
  --threads THREADS
  --outdir OUTDIR
  --nmesh NMESH
  --folds FOLDS
  --pk-types {core,full,...}
  --write-cov
"""
    )
    raise SystemExit(0)

import h5py
import numpy as np
import pyccl as ccl

from ia_analysis.spectra import CatMesh
from ia_analysis.spectra import SnapMesh
from ia_analysis.shapes.Iana import epsilon_from_shape_matrix, epsilon_from_spin
from ia_analysis.spectra.powers import PowerConfig, PowerSpectrumEstimator
from ia_analysis.catalogs.catalog_loader import CSCatalog



# -----------------------------------------------------------------------------
# Redshifts and LRG/ELG thresholds
# -----------------------------------------------------------------------------

ZMAP: Dict[int, float] = {
    0: 3.00, 1: 2.00, 2: 1.36, 3: 1.26, 4: 1.15, 5: 1.06,
    6: 0.97, 7: 0.88, 8: 0.80, 9: 0.73, 10: 0.65, 11: 0.58,
    12: 0.51, 13: 0.45, 14: 0.39, 15: 0.33, 16: 0.27, 17: 0.21,
    18: 0.16, 19: 0.10, 20: 0.05, 21: 0.00,
}

# Columns: z, ELG log10(sSFR/yr^-1) cut, LRG Mstar cut in 1e10 Msun.
LRG_ELG_TABLE = np.array([
    (1.16, -9.3, 7.5),
    (0.97, -9.4, 7.4),
    (0.73, -9.5, 8.2),
    (0.51, -9.6, 8.2),
    (0.27, -9.9, 8.6),
], dtype=[('z', 'f8'), ('log_ssfr_cut', 'f8'), ('mstar_cut_1e10_msun', 'f8')])


# -----------------------------------------------------------------------------
# Spectrum registry
# -----------------------------------------------------------------------------

CORE_PK_TYPES = ['gg', 'gE', 'EE', 'BB', 'dd', 'dg', 'dE']
FULL_PK_TYPES = [
    'gg', 'gE', 'gB', 'EE', 'BB', 'EB',
    'dd', 'dg', 'dE', 'dB',
    'tt', 'gt', 'tE', 'tB', 'dt',
    'tptp', 'dtp', 'gtp', 'Etp', 'Btp', 'tpt',
]

# shorthand -> (pair, required_fields)
PK_TYPE_MAP = {
    'gg':   (('g', 'g'), {'g'}),
    'gE':   (('g', 'E'), {'g', 'E'}),
    'gB':   (('g', 'B'), {'g', 'B'}),
    'EE':   (('E', 'E'), {'E'}),
    'BB':   (('B', 'B'), {'B'}),
    'EB':   (('E', 'B'), {'E', 'B'}),

    'dd':   (('d', 'd'), {'d'}),
    'dg':   (('d', 'g'), {'d', 'g'}),
    'dE':   (('d', 'E'), {'d', 'E'}),
    'dB':   (('d', 'B'), {'d', 'B'}),

    'tt':   (('t', 't'), {'t'}),
    'gt':   (('g', 't'), {'g', 't'}),
    'tE':   (('t', 'E'), {'t', 'E'}),
    'tB':   (('t', 'B'), {'t', 'B'}),
    'dt':   (('d', 't'), {'d', 't'}),

    'tptp': (('tp', 'tp'), {'tp'}),
    'dtp':  (('d', 'tp'), {'d', 'tp'}),
    'gtp':  (('g', 'tp'), {'g', 'tp'}),
    'Etp':  (('E', 'tp'), {'E', 'tp'}),
    'Btp':  (('B', 'tp'), {'B', 'tp'}),
    'tpt':  (('tp', 't'), {'tp', 't'}),

    # Momentum-divergence proxy.  This field is self-foldable because it is
    # built from additive momentum density, but it is not identical to the
    # local velocity-divergence theta estimator.
    'tmtm': (('tm', 'tm'), {'tm'}),
    'dtm':  (('d', 'tm'), {'d', 'tm'}),
    'gtm':  (('g', 'tm'), {'g', 'tm'}),
    'Etm':  (('E', 'tm'), {'E', 'tm'}),
    'Btm':  (('B', 'tm'), {'B', 'tm'}),
}


# -----------------------------------------------------------------------------
# Logging and cosmology
# -----------------------------------------------------------------------------

def setup_logging(flag: str, snap: int) -> None:
    """Configure a simple process-level logger."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )
    logging.info('start flag=%s snap=%03d', flag, snap)


def get_cosmo() -> Tuple[ccl.Cosmology, Dict[str, float]]:
    """Return the fiducial cosmology and a metadata dictionary."""
    omega_c = 0.3089 - 0.0486
    omega_b = 0.0486
    h = 0.6774
    sigma8 = 0.8159
    n_s = 0.9667

    cosmo = ccl.Cosmology(
        Omega_c=omega_c,
        Omega_b=omega_b,
        h=h,
        sigma8=sigma8,
        n_s=n_s,
    )
    pars = dict(Omega_c=omega_c, Omega_b=omega_b, h=h, sigma8=sigma8, n_s=n_s)
    return cosmo, pars


def parse_pk_types(text: str) -> List[str]:
    """Parse --pk-types, including the aliases 'core' and 'full'."""
    s = str(text).strip()
    if not s or s.lower() == 'full':
        return list(FULL_PK_TYPES)
    if s.lower() == 'core':
        return list(CORE_PK_TYPES)
    out = [x.strip() for x in s.split(',') if x.strip()]
    bad = [x for x in out if x not in PK_TYPE_MAP]
    if bad:
        raise ValueError(f'Unknown pk-types {bad}. Supported: {sorted(PK_TYPE_MAP)} plus aliases core/full.')
    return out


def default_target_k() -> np.ndarray:
    """Default target k-grid for the resampled stitched products."""
    return np.logspace(np.log10(0.2), np.log10(20.0), 30)


def parse_k_array(text: Optional[str]) -> np.ndarray:
    """Parse a user-provided target k-array."""
    if text is None or not str(text).strip():
        return default_target_k()

    s = str(text).strip()
    if s.startswith('logspace:'):
        _, a, b, n = s.split(':')
        return np.logspace(np.log10(float(a)), np.log10(float(b)), int(n))

    if ',' in s:
        arr = np.array([float(x) for x in s.split(',') if x.strip()], dtype=float)
    else:
        arr = np.array([float(x) for x in s.split() if x.strip()], dtype=float)

    if arr.ndim != 1 or arr.size == 0:
        raise ValueError('Invalid --k-array input.')
    return np.sort(arr)


def thresholds_at_z(z: float, mode: str = 'nearest') -> Tuple[float, float, float]:
    """Return (z_ref, log_ssfr_cut, mstar_cut_1e10_msun) at the requested redshift."""
    zs = np.asarray(LRG_ELG_TABLE['z'], dtype=float)
    logcuts = np.asarray(LRG_ELG_TABLE['log_ssfr_cut'], dtype=float)
    mcuts = np.asarray(LRG_ELG_TABLE['mstar_cut_1e10_msun'], dtype=float)

    mode = str(mode).strip().lower()
    if mode == 'nearest':
        idx = int(np.argmin(np.abs(zs - float(z))))
        return float(zs[idx]), float(logcuts[idx]), float(mcuts[idx])

    if mode == 'linear':
        # np.interp expects an ascending x-array.
        order = np.argsort(zs)
        zasc = zs[order]
        logasc = logcuts[order]
        masc = mcuts[order]
        z_clip = float(np.clip(float(z), zasc[0], zasc[-1]))
        return float(z_clip), float(np.interp(z_clip, zasc, logasc)), float(np.interp(z_clip, zasc, masc))

    raise ValueError("threshold-mode must be 'nearest' or 'linear'.")


def mstar_code_to_msun(mstar_code: np.ndarray, h: float, unit: str) -> np.ndarray:
    """Convert catalog stellar mass to physical Msun."""
    unit = str(unit).strip()
    if unit == '1e10Msun_h':
        return np.asarray(mstar_code, dtype=float) * 1.0e10 / float(h)
    if unit == '1e10Msun':
        return np.asarray(mstar_code, dtype=float) * 1.0e10
    raise ValueError("catalog-mass-unit must be '1e10Msun_h' or '1e10Msun'.")


def mstar_threshold_to_code(mcut_1e10_msun: float, h: float, unit: str) -> float:
    """Convert an LRG threshold in 1e10 Msun to catalog mass units."""
    unit = str(unit).strip()
    if unit == '1e10Msun_h':
        return float(mcut_1e10_msun) * float(h)
    if unit == '1e10Msun':
        return float(mcut_1e10_msun)
    raise ValueError("catalog-mass-unit must be '1e10Msun_h' or '1e10Msun'.")



def parse_float_list(text: str, default: Sequence[float]) -> List[float]:
    """Parse a comma-separated float list."""
    if text is None or not str(text).strip():
        return [float(x) for x in default]
    return [float(x.strip()) for x in str(text).split(',') if x.strip()]


def nbar_label(nbar: float) -> str:
    """Return a compact HDF5-friendly label for a number density."""
    return f'{float(nbar):.0e}'.replace('e-0', 'e-').replace('e+0', 'e+')


def top_n_mask_from_property(valid_mask: np.ndarray, prop: np.ndarray, target_count: int) -> Tuple[np.ndarray, float, int]:
    """Select exactly the top-N finite objects from a property array.

    Parameters
    ----------
    valid_mask : ndarray of bool
        Objects eligible before ranking.
    prop : ndarray
        Ranking variable. Larger values are selected first.
    target_count : int
        Requested number of objects. If fewer objects are eligible, all eligible
        objects are returned.

    Returns
    -------
    mask : ndarray of bool
        Final selected mask with at most target_count objects.
    threshold : float
        Minimum property value among the selected objects.
    n_eligible : int
        Number of eligible objects before applying the top-N cut.
    """
    prop = np.asarray(prop, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(prop)
    idx = np.flatnonzero(valid)
    n_eligible = int(idx.size)

    out = np.zeros_like(valid, dtype=bool)
    if n_eligible == 0 or target_count <= 0:
        return out, np.nan, n_eligible

    n_take = min(int(target_count), n_eligible)
    # Use argpartition for speed, then sort only the retained subset for a clean threshold.
    prop_valid = prop[idx]
    if n_take == n_eligible:
        chosen = idx
    else:
        part = np.argpartition(prop_valid, -n_take)[-n_take:]
        chosen = idx[part]
    out[chosen] = True
    threshold = float(np.nanmin(prop[chosen])) if chosen.size else np.nan
    return out, threshold, n_eligible


# -----------------------------------------------------------------------------
# Shape conversion and sample loading
# -----------------------------------------------------------------------------

def _finite_mask_from_arrays(*arrays: np.ndarray) -> np.ndarray:
    """Return a row-wise finite mask for arrays sharing their first dimension."""
    if not arrays:
        raise ValueError('Need at least one array.')
    nobj = len(arrays[0])
    mask = np.ones(nobj, dtype=bool)
    for arr in arrays:
        if len(arr) != nobj:
            raise ValueError('All arrays must share the same leading dimension.')
        a = np.asarray(arr)
        if a.ndim == 1:
            mask &= np.isfinite(a)
        else:
            mask &= np.isfinite(a).all(axis=tuple(range(1, a.ndim)))
    return mask


def _reduce_cos_err(cos_err: np.ndarray) -> np.ndarray:
    """Reduce scalar/vector/tensor cos_err into one conservative scalar per object."""
    c = np.asarray(cos_err, dtype=float)
    if c.ndim <= 1:
        return np.abs(c)
    return np.max(np.abs(c), axis=tuple(range(1, c.ndim)))


def gamma_from_I(I: np.ndarray, los: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, float]:
    """Convert stellar shape tensors into finite shear-like gamma components."""
    eps1, eps2 = epsilon_from_shape_matrix(I, los=los, apply_responsivity=False)
    eps1 = np.asarray(eps1, dtype=float)
    eps2 = np.asarray(eps2, dtype=float)
    good = np.isfinite(eps1) & np.isfinite(eps2)

    g1 = np.full_like(eps1, np.nan, dtype=float)
    g2 = np.full_like(eps2, np.nan, dtype=float)
    R = np.nan
    if np.any(good):
        R = 1.0 - 0.5 * float(np.mean(eps1[good] ** 2 + eps2[good] ** 2))
        if np.isfinite(R) and R > 0.0:
            g1[good] = eps1[good] / (2.0 * R)
            g2[good] = eps2[good] / (2.0 * R)
    return g1, g2, float(R)


def gamma_from_L(L: np.ndarray, los: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, float]:
    """Convert spin vectors into finite shear-like gamma components."""
    eps1, eps2 = epsilon_from_spin(L, np.asarray(los, dtype=float))
    eps1 = np.asarray(eps1, dtype=float)
    eps2 = np.asarray(eps2, dtype=float)
    good = np.isfinite(eps1) & np.isfinite(eps2)

    g1 = np.full_like(eps1, np.nan, dtype=float)
    g2 = np.full_like(eps2, np.nan, dtype=float)
    R = np.nan
    if np.any(good):
        R = 1.0 - 0.5 * float(np.mean(eps1[good] ** 2 + eps2[good] ** 2))
        if np.isfinite(R) and R > 0.0:
            g1[good] = eps1[good] / (2.0 * R)
            g2[good] = eps2[good] / (2.0 * R)
    return g1, g2, float(R)


def build_sample_dict(
    name: str,
    pos: np.ndarray,
    vel: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    R: float,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Create a compact sample dictionary after a final finite-value cleanup."""
    mask = np.isfinite(pos).all(axis=1)
    mask &= np.isfinite(vel).all(axis=1)
    mask &= np.isfinite(g1)
    mask &= np.isfinite(g2)

    out: Dict[str, object] = {
        'name': name,
        'pos': pos[mask],
        'vel': vel[mask],
        'g1': g1[mask],
        'g2': g2[mask],
        'R': R if np.isfinite(R) else np.nan,
        'selected_count': int(np.count_nonzero(mask)),
        'pre_finite_count': int(len(pos)),
    }
    if extra is not None:
        out.update(extra)
    return out


def load_science_samples(
    mg_file: Path,
    *,
    z: float,
    h: float,
    boxsize: float,
    cos_err_max: float,
    los: Sequence[float],
    catalog_mass_unit: str = '1e10Msun_h',
    threshold_mode: str = 'nearest',
    exclusive_lrg_elg: bool = False,
    include_central_satellite: bool = True,
    include_density_samples: bool = True,
    density_nbars: Sequence[float] = (1e-2, 1e-3, 1e-4),
    density_selection_modes: Sequence[str] = ('Mstar', 'SFR'),
) -> List[Dict[str, object]]:
    """Load all science samples from the MG catalog.

    The returned sample list contains, in order:

    1. The full parent sample, named ``all``.
    2. Standalone LRG and ELG samples from the redshift-dependent threshold table.
    3. Fixed-number-density samples selected by top Mstar and/or top SFR.
    4. Global kappa_rot samples, preserving the previous kinematic split.
    5. Optional global central/satellite diagnostic samples.

    LRG and ELG are intentionally not split into kappa, central, or satellite
    subclasses. This keeps the tracer definitions clean for LRG/ELG comparisons.
    """
    with h5py.File(mg_file, 'r') as f:
        pos_all = f['pos_abs'][:] / 1000.0  # ckpc/h -> cMpc/h
        vel_all = f['vel_abs'][:]
        I_all = f['Star']['I'][:]
        L_all = f['Star']['L'][:]
        kappa_rot_all = f['Star']['kappa_rot'][:]
        cos_err = _reduce_cos_err(f['Star']['cos_err'][:])
        mstar_code = f['SubhaloMassInRadType'][:, 4]
        sfr_all = f['SubhaloSFR'][:]
        subhalo_id = f['SubhaloID'][:] if 'SubhaloID' in f else np.arange(len(pos_all))
        cen_id = f['CenID'][:] if 'CenID' in f else np.arange(len(pos_all))

    z_ref, log_ssfr_cut, mstar_cut_1e10_msun = thresholds_at_z(z, threshold_mode)
    mstar_msun = mstar_code_to_msun(mstar_code, h, catalog_mass_unit)
    mstar_cut_code = mstar_threshold_to_code(mstar_cut_1e10_msun, h, catalog_mass_unit)

    log_ssfr = np.full_like(np.asarray(sfr_all, dtype=float), np.nan, dtype=float)
    good_ssfr = np.isfinite(sfr_all) & np.isfinite(mstar_msun) & (sfr_all > 0.0) & (mstar_msun > 0.0)
    log_ssfr[good_ssfr] = np.log10(np.asarray(sfr_all, dtype=float)[good_ssfr] / mstar_msun[good_ssfr])

    lrg_mask = np.isfinite(mstar_code) & (mstar_code >= mstar_cut_code)
    elg_mask = np.isfinite(log_ssfr) & (log_ssfr >= log_ssfr_cut)
    if exclusive_lrg_elg:
        # Keep the physically massive LRG sample unchanged and remove overlaps from ELG.
        elg_mask &= ~lrg_mask

    valid_common = _finite_mask_from_arrays(
        pos_all, vel_all, kappa_rot_all, cos_err, mstar_code, sfr_all, subhalo_id, cen_id,
    )
    valid_common &= (cos_err < cos_err_max)
    valid_common &= (mstar_code > 0.0)

    valid_I = valid_common & _finite_mask_from_arrays(I_all)
    valid_L = valid_common & _finite_mask_from_arrays(L_all)
    is_central = (subhalo_id == cen_id)

    base_extra = {
        'z_table_ref': z_ref,
        'log_ssfr_cut': log_ssfr_cut,
        'mstar_cut_1e10_msun': mstar_cut_1e10_msun,
        'mstar_cut_code_units': mstar_cut_code,
        'catalog_mass_unit': catalog_mass_unit,
        'threshold_mode': threshold_mode,
        'exclusive_lrg_elg': bool(exclusive_lrg_elg),
        'cos_err_max': cos_err_max,
    }

    samples: List[Dict[str, object]] = []

    def append_I_sample(name: str, mask: np.ndarray, extra: Optional[Dict[str, object]] = None) -> None:
        """Append a sample whose shape field is measured from the stellar tensor I."""
        m = valid_I & mask
        pos = pos_all[m]
        vel = vel_all[m]
        if len(pos) > 0:
            g1, g2, R = gamma_from_I(I_all[m], los)
        else:
            g1 = np.array([], dtype=float)
            g2 = np.array([], dtype=float)
            R = np.nan
        meta = dict(base_extra)
        meta.update({'shape_mode': 'I', 'sample_pre_count': int(np.count_nonzero(m))})
        if extra:
            meta.update(extra)
        samples.append(build_sample_dict(name, pos, vel, g1, g2, R, meta))

    def append_L_sample(name: str, mask: np.ndarray, extra: Optional[Dict[str, object]] = None) -> None:
        """Append a sample whose shape field is measured from the stellar spin L."""
        m = valid_L & mask
        pos = pos_all[m]
        vel = vel_all[m]
        if len(pos) > 0:
            g1, g2, R = gamma_from_L(L_all[m], los)
        else:
            g1 = np.array([], dtype=float)
            g2 = np.array([], dtype=float)
            R = np.nan
        meta = dict(base_extra)
        meta.update({'shape_mode': 'L', 'sample_pre_count': int(np.count_nonzero(m))})
        if extra:
            meta.update(extra)
        samples.append(build_sample_dict(name, pos, vel, g1, g2, R, meta))

    # ------------------------------------------------------------------
    # 1. Full parent sample. This is useful as a baseline and for sanity checks.
    # ------------------------------------------------------------------
    append_I_sample('all', np.ones_like(valid_I, dtype=bool), {'sample_type': 'full', 'tracer': 'all'})

    # ------------------------------------------------------------------
    # 2. Standalone LRG/ELG samples. Do not create LRG/ELG subclasses.
    # ------------------------------------------------------------------
    append_I_sample('LRG', lrg_mask, {'sample_type': 'LRG_ELG', 'tracer': 'LRG'})
    append_I_sample('ELG', elg_mask, {'sample_type': 'LRG_ELG', 'tracer': 'ELG'})

    # ------------------------------------------------------------------
    # 3. Fixed-number-density samples selected by top Mstar or top SFR.
    # ------------------------------------------------------------------
    if include_density_samples:
        prop_map = {
            'Mstar': np.asarray(mstar_code, dtype=float),
            'SFR': np.asarray(sfr_all, dtype=float),
        }
        unit_map = {
            'Mstar': str(catalog_mass_unit),
            'SFR': 'catalog_SubhaloSFR_units',
        }
        for mode in density_selection_modes:
            mode = str(mode).strip()
            if mode not in prop_map:
                raise ValueError(f"Unknown density-selection mode '{mode}'. Supported: {sorted(prop_map)}")
            prop = prop_map[mode]
            for nbar in density_nbars:
                target_count = int(round(float(nbar) * float(boxsize) ** 3))
                mask, threshold, n_eligible = top_n_mask_from_property(valid_I, prop, target_count)
                label = nbar_label(float(nbar))
                append_I_sample(
                    f'{mode}/{label}',
                    mask,
                    {
                        'sample_type': 'fixed_nbar',
                        'tracer': mode,
                        'selection_property': mode,
                        'selection_property_unit': unit_map[mode],
                        'target_nbar_h3_Mpc3': float(nbar),
                        'target_count': int(target_count),
                        'eligible_count': int(n_eligible),
                        'property_threshold': float(threshold),
                    },
                )

    # ------------------------------------------------------------------
    # 4. Global kappa_rot split. This is not split further by LRG or ELG.
    # ------------------------------------------------------------------
    sph = kappa_rot_all < 0.45
    disk = kappa_rot_all > 0.55
    append_I_sample('kappa_lt_045', sph, {'sample_type': 'kappa', 'tracer': 'all', 'kappa_class': 'lt_045'})
    append_L_sample('kappa_gt_055', disk, {'sample_type': 'kappa', 'tracer': 'all', 'kappa_class': 'gt_055'})

    # ------------------------------------------------------------------
    # 5. Optional global central/satellite diagnostic samples.
    # ------------------------------------------------------------------
    if include_central_satellite:
        append_I_sample('centrals', is_central, {'sample_type': 'central_satellite', 'tracer': 'all', 'central_satellite': 'central'})
        append_I_sample('satellites', ~is_central, {'sample_type': 'central_satellite', 'tracer': 'all', 'central_satellite': 'satellite'})

    return samples


# -----------------------------------------------------------------------------
# Mesh construction and P(k) measurement
# -----------------------------------------------------------------------------

def build_galaxy_meshes(
    sample: Dict[str, object],
    *,
    boxsize: float,
    nmesh: int,
    assign: str,
    fold: int,
    z: float,
    cosmo: ccl.Cosmology,
    los: Sequence[float],
    want_theta: bool,
) -> Dict[str, np.ndarray]:
    """Build galaxy overdensity, IA E/B, and optionally galaxy theta meshes."""
    cfg = CatMesh.CatalogMeshConfig(
        boxsize=boxsize,
        nmesh=nmesh,
        mas_gal=assign,
        mas_shape=assign,
        pos_unit='Mpc/h',
    )
    builder = CatMesh.CatalogMeshBuilder(cfg, cosmo=cosmo)
    out = builder.build(
        pos=np.asarray(sample['pos']),
        vel=np.asarray(sample['vel']),
        vmesh='theta' if want_theta else None,
        z=z,
        e1=np.asarray(sample['g1']),
        e2=np.asarray(sample['g2']),
        e_are_gamma=True,
        los=los,
        space='real',
        folding_factor=fold,
        verbose=False,
    )
    ret: Dict[str, np.ndarray] = {
        'g_mesh': out['g_mesh'],
        'E_mesh': out['E_mesh'],
        'B_mesh': out['B_mesh'],
    }
    if 'theta_mesh' in out:
        ret['t_mesh'] = out['theta_mesh']
    elif 't_mesh' in out:
        ret['t_mesh'] = out['t_mesh']
    return ret


def build_particle_meshes(
    cs_cat: CSCatalog,
    *,
    boxsize: float,
    nmesh: int,
    assign: str,
    fold: int,
    z: float,
    cosmo: ccl.Cosmology,
    dm_fixed_mass: float,
    want_theta_p: bool,
) -> Dict[str, object]:
    """Build total matter overdensity and optionally particle theta meshes."""
    cfg = SnapMesh.SnapshotMeshConfig(
        boxsize=boxsize,
        nmesh=nmesh,
        mas=assign,
        pos_unit='kpc/h',
    )
    builder = SnapMesh.SnapshotMeshBuilder(cfg, cosmo=cosmo)
    out = builder.build_stream_cs(
        cs_cat,
        z=z,
        ptypes=(0, 1, 4),
        dm_fixed_mass=dm_fixed_mass,
        dm_ptype=1,
        folding_factor=fold,
        mas=assign,
        nworker=1,
        two_pass=True,
        want_theta=bool(want_theta_p),
        verbose=False,
    )
    ret: Dict[str, object] = {
        'delta_mesh': out['delta_mesh'],
        'meta': out.get('meta', {}),
    }
    if 'theta_mesh' in out:
        ret['tp_mesh'] = out['theta_mesh']
    elif 'tp_mesh' in out:
        ret['tp_mesh'] = out['tp_mesh']
    return ret


def spec_keys_from_pk_types(pk_types: Sequence[str]) -> List[str]:
    """Convert shorthand names such as gg,gE,tptp into HDF5 dataset keys."""
    return ['P_' + p for p in pk_types]


def pairs_and_required_fields(pk_types: Sequence[str]) -> Tuple[List[Tuple[str, str]], set]:
    """Return estimator pairs and the union of required field names."""
    pairs: List[Tuple[str, str]] = []
    required = set()
    for p in pk_types:
        pair, req = PK_TYPE_MAP[p]
        pairs.append(pair)
        required |= set(req)
    return pairs, required


def build_estimator_meshes(meshes: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Map internal mesh names to the short field names expected by powers.py."""
    out: Dict[str, np.ndarray] = {}
    if 'g_mesh' in meshes:
        out['g'] = meshes['g_mesh']
    if 'E_mesh' in meshes:
        out['E'] = meshes['E_mesh']
    if 'B_mesh' in meshes:
        out['B'] = meshes['B_mesh']
    if 'delta_mesh' in meshes:
        out['d'] = meshes['delta_mesh']
    if 't_mesh' in meshes:
        out['t'] = meshes['t_mesh']
    if 'tp_mesh' in meshes:
        out['tp'] = meshes['tp_mesh']
    if 'theta_momentum_mesh' in meshes:
        out['tm'] = meshes['theta_momentum_mesh']
    if 'tm_mesh' in meshes:
        out['tm'] = meshes['tm_mesh']
    return out


def mas_map_for_fields(required_fields: Iterable[str], assign: str, theta_mas: str = 'None') -> Dict[str, str]:
    """Return MAS-deconvolution settings for each field.

    The density and IA meshes are CIC-assigned fields and use the requested MAS
    deconvolution.  The theta fields are derived mesh fields; by default they are
    not passed through an additional MAS deconvolution.
    """
    out = {}
    for name in required_fields:
        if name in {'t', 'tp', 'tm'}:
            out[name] = str(theta_mas)
        else:
            out[name] = str(assign)
    return out


def measure_power(
    meshes: Dict[str, np.ndarray],
    *,
    boxsize: float,
    fold: int,
    los: Sequence[float],
    threads: int,
    assign: str,
    theta_mas: str,
    pk_types: Sequence[str],
) -> Dict[str, object]:
    """Measure requested spectra for one folded mesh set."""
    pairs, required_fields = pairs_and_required_fields(pk_types)
    est_meshes = build_estimator_meshes(meshes)
    missing = sorted([f for f in required_fields if f not in est_meshes])
    if missing:
        raise RuntimeError(f'Missing required fields {missing}; available fields are {sorted(est_meshes)}')

    est = PowerSpectrumEstimator(
        PowerConfig(
            boxsize=boxsize / float(fold),
            power_norm_boxsize=boxsize,
            los=tuple(los),
            ells=(0,),
            threads=threads,
        )
    )
    pk = est.compute(
        meshes=est_meshes,
        pairs=pairs,
        mas=mas_map_for_fields(required_fields, assign, theta_mas=theta_mas),
        verbose=False,
    )
    return pk


def estimate_noise(
    pk: Dict[str, object],
    *,
    sample: Dict[str, object],
    boxsize: float,
    nmesh: int,
    fold: int,
    z: float,
    h: float,
    cosmo: ccl.Cosmology,
    snap_meta: Optional[Dict[str, object]],
    spec_keys: Sequence[str],
    subtract_matter_shot_noise: bool = False,
) -> Dict[str, np.ndarray]:
    """Estimate additive white-noise terms in the original-box convention.

    This function returns noise biases to subtract from auto spectra.  It is
    not a covariance or an error bar.  The theta noise is matched to the
    spectral-derivative definition used in CatMesh/SnapMesh: k is in h/Mpc,
    while H(z) is in km/s/Mpc, so the derivative factor is (h k).
    """
    k = np.asarray(pk['k'], dtype=float)
    z0 = np.zeros_like(k, dtype=float)
    noise = {sk: z0.copy() for sk in spec_keys}

    Ngal = int(sample.get('selected_count', 0))
    Vbox = float(boxsize) ** 3
    nbar = Ngal / Vbox if Ngal > 0 else 0.0
    a = 1.0 / (1.0 + float(z))
    Hz = 100.0 * float(h) * float(ccl.h_over_h0(cosmo, a))  # km/s/Mpc
    k_mpc = float(h) * k  # convert h/Mpc -> 1/Mpc for consistency with Hz

    if nbar > 0.0:
        if 'P_gg' in noise:
            noise['P_gg'] = np.full_like(k, 1.0 / nbar, dtype=float)

        g1 = np.asarray(sample['g1'], dtype=float)
        g2 = np.asarray(sample['g2'], dtype=float)
        e2 = np.nanmean(g1 * g1 + g2 * g2)
        if np.isfinite(e2):
            shape_noise = 0.5 * float(e2) / nbar
            if 'P_EE' in noise:
                noise['P_EE'] = np.full_like(k, shape_noise, dtype=float)
            if 'P_BB' in noise:
                noise['P_BB'] = np.full_like(k, shape_noise, dtype=float)

        if 'P_tt' in noise:
            vel = np.asarray(sample['vel'], dtype=float)
            dv = vel - np.mean(vel, axis=0, keepdims=True)
            sigma1d = float(np.sqrt(np.mean(dv * dv) / 3.0)) if vel.size else np.nan
            if np.isfinite(sigma1d) and Hz > 0.0:
                noise['P_tt'] = sigma1d ** 2 * k_mpc ** 2 / ((a * Hz) ** 2 * nbar)

    if subtract_matter_shot_noise and snap_meta is not None and 'P_dd' in noise:
        shot = snap_meta.get('shotnoise_dd', None)
        if shot is not None and np.isfinite(float(shot)):
            # SnapMesh stores folded-volume shot noise. powers.py returns P(k)
            # in the original simulation-volume convention, so rescale by fold^3.
            noise['P_dd'] = np.full_like(k, float(shot) * float(fold) ** 3, dtype=float)

    if snap_meta is not None and 'P_tptp' in noise:
        vshot = snap_meta.get('velocity_shotnoise_1d', None)
        if vshot is not None and np.isfinite(float(vshot)) and Hz > 0.0:
            # vshot is in (km/s)^2 (Mpc/h)^3 for the folded box; convert to
            # the original-volume convention consistently with powers.py.
            vshot_orig = float(vshot) * float(fold) ** 3
            noise['P_tptp'] = vshot_orig * k_mpc ** 2 / ((a * Hz) ** 2)

    return noise


# -----------------------------------------------------------------------------
# HDF5 output and stitching
# -----------------------------------------------------------------------------

def sanitize_group_name(name: str) -> str:
    """Return an HDF5-safe group path fragment."""
    return str(name).replace('//', '/').strip('/')


def add_sample_attrs(case: h5py.Group, sample: Dict[str, object]) -> None:
    """Write sample metadata as HDF5 attributes."""
    for key, val in sample.items():
        if key in {'pos', 'vel', 'g1', 'g2'}:
            continue
        if isinstance(val, (str, bytes, int, float, np.integer, np.floating, bool, np.bool_)):
            case.attrs[key] = val
        elif val is None:
            continue
        else:
            case.attrs[key] = str(val)


def write_fold_group(
    case: h5py.Group,
    fold: int,
    pk: Dict[str, object],
    noise: Dict[str, np.ndarray],
    pk_corr: Dict[str, np.ndarray],
    delta_mesh_mean: float,
    spec_keys: Sequence[str],
    snap_meta: Optional[Dict[str, object]] = None,
) -> None:
    """Write one folded measurement block."""
    gf = case.create_group(f'fold_{fold}')
    gf.create_dataset('k', data=np.asarray(pk['k'], dtype=float))
    if 'Nmodes' in pk:
        gf.create_dataset('Nmodes', data=np.asarray(pk['Nmodes'], dtype=float))
    if 'Nmodes_native' in pk:
        gf.create_dataset('Nmodes_native', data=np.asarray(pk['Nmodes_native'], dtype=float))
    gf.create_dataset('delta_mesh_mean', data=np.array([delta_mesh_mean], dtype=float))
    gf.attrs['boxsize_mesh'] = float(pk.get('boxsize', np.nan))
    gf.attrs['power_norm_boxsize'] = float(pk.get('power_norm_boxsize', np.nan))
    gf.attrs['volume_factor'] = float(pk.get('volume_factor', np.nan))

    if snap_meta:
        for key in (
            'shotnoise_dd', 'shotnoise_dd_number_weighted', 'total_weight',
            'total_weight2', 'velocity_shotnoise_1d'
        ):
            if key in snap_meta and snap_meta[key] is not None:
                gf.attrs[key] = float(snap_meta[key])

    for key in spec_keys:
        gf.create_dataset(key, data=np.asarray(pk[key], dtype=float))
        gf.create_dataset(key + '_noise', data=np.asarray(noise[key], dtype=float))
        gf.create_dataset(key + '_corr', data=np.asarray(pk_corr[key], dtype=float))


def get_fold_selection_mask(
    k: np.ndarray,
    fold: int,
    *,
    boxsize: float,
    nmesh: int,
    alpha: float,
    overlap: float,
    prev_kmax: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Return the trusted k-mask for one folded spectrum."""
    k_ny = np.pi * float(nmesh) / (float(boxsize) / float(fold))
    kmax = float(alpha) * k_ny
    if prev_kmax is None:
        mask = k <= kmax
    else:
        mask = (k > float(overlap) * prev_kmax) & (k <= kmax)
    return mask, kmax


def build_native_stitched(
    results: Dict[int, Dict[str, Dict[str, np.ndarray]]],
    fold_list: Sequence[int],
    *,
    boxsize: float,
    nmesh: int,
    alpha: float,
    overlap: float,
    source_key: str,
    spec_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate trusted native k-bins from all folds."""
    k_all: List[np.ndarray] = []
    p_all: List[np.ndarray] = []
    f_all: List[np.ndarray] = []
    prev_kmax = None

    for fold in fold_list:
        if fold not in results:
            continue
        k = np.asarray(results[fold][source_key]['k'], dtype=float)
        p = np.asarray(results[fold][source_key][spec_key], dtype=float)
        mask, prev_kmax = get_fold_selection_mask(
            k, fold, boxsize=boxsize, nmesh=nmesh, alpha=alpha, overlap=overlap, prev_kmax=prev_kmax,
        )
        if np.any(mask):
            k_all.append(k[mask])
            p_all.append(p[mask])
            f_all.append(np.full(np.count_nonzero(mask), fold, dtype=int))

    if not k_all:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=int)

    k_cat = np.concatenate(k_all)
    p_cat = np.concatenate(p_all)
    f_cat = np.concatenate(f_all)
    order = np.argsort(k_cat)
    return k_cat[order], p_cat[order], f_cat[order]


def resample_stitched_to_target_k(
    results: Dict[int, Dict[str, Dict[str, np.ndarray]]],
    fold_list: Sequence[int],
    *,
    boxsize: float,
    nmesh: int,
    alpha: float,
    overlap: float,
    source_key: str,
    spec_key: str,
    target_k: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate stitched native spectra to the requested target-k array."""
    target_k = np.asarray(target_k, dtype=float)
    out = np.full(target_k.shape, np.nan, dtype=float)
    out_fold = np.full(target_k.shape, -1, dtype=int)
    prev_kmax = None

    for fold in fold_list:
        if fold not in results:
            continue
        k = np.asarray(results[fold][source_key]['k'], dtype=float)
        p = np.asarray(results[fold][source_key][spec_key], dtype=float)
        mask, prev_kmax = get_fold_selection_mask(
            k, fold, boxsize=boxsize, nmesh=nmesh, alpha=alpha, overlap=overlap, prev_kmax=prev_kmax,
        )
        if np.count_nonzero(mask) < 2:
            continue
        k_sel = k[mask]
        p_sel = p[mask]
        take = (target_k >= k_sel[0]) & (target_k <= k_sel[-1]) & (out_fold < 0)
        if np.any(take):
            out[take] = np.interp(target_k[take], k_sel, p_sel)
            out_fold[take] = fold

    return out, out_fold


def write_stitched_groups(
    case: h5py.Group,
    results: Dict[int, Dict[str, Dict[str, np.ndarray]]],
    *,
    fold_list: Sequence[int],
    boxsize: float,
    nmesh: int,
    alpha: float,
    overlap: float,
    target_k: np.ndarray,
    spec_keys: Sequence[str],
) -> None:
    """Write native stitched and target-k stitched spectra."""
    groups = {
        'stitched_native_raw': case.create_group('stitched_native_raw'),
        'stitched_native_corr': case.create_group('stitched_native_corr'),
        'stitched_native_noise': case.create_group('stitched_native_noise'),
        'stitched_raw': case.create_group('stitched_raw'),
        'stitched_corr': case.create_group('stitched_corr'),
        'stitched_noise': case.create_group('stitched_noise'),
    }
    target_k = np.asarray(target_k, dtype=float)
    for gname in ('stitched_raw', 'stitched_corr', 'stitched_noise'):
        groups[gname].create_dataset('k', data=target_k)

    for spec_key in spec_keys:
        for source_key, group_name in [
            ('pk', 'stitched_native_raw'),
            ('pk_corr', 'stitched_native_corr'),
            ('noise', 'stitched_native_noise'),
        ]:
            kk, pp, ff = build_native_stitched(
                results, fold_list, boxsize=boxsize, nmesh=nmesh, alpha=alpha,
                overlap=overlap, source_key=source_key, spec_key=spec_key,
            )
            groups[group_name].create_dataset(spec_key + '_k', data=kk)
            groups[group_name].create_dataset(spec_key + '_Pk', data=pp)
            groups[group_name].create_dataset(spec_key + '_fold', data=ff)

        for source_key, group_name in [
            ('pk', 'stitched_raw'),
            ('pk_corr', 'stitched_corr'),
            ('noise', 'stitched_noise'),
        ]:
            pp, ff = resample_stitched_to_target_k(
                results, fold_list, boxsize=boxsize, nmesh=nmesh, alpha=alpha,
                overlap=overlap, source_key=source_key, spec_key=spec_key, target_k=target_k,
            )
            groups[group_name].create_dataset(spec_key + '_Pk', data=pp)
            groups[group_name].create_dataset(spec_key + '_fold', data=ff)


def build_summary_array(samples: Sequence[Dict[str, object]], flag: str, snap: int, z: float) -> np.ndarray:
    """Build a compact sample summary table."""
    dt = np.dtype([
        ('flag', 'S8'),
        ('snap', 'i4'),
        ('z', 'f8'),
        ('sample', 'S96'),
        ('sample_type', 'S24'),
        ('tracer', 'S16'),
        ('shape_mode', 'S8'),
        ('selection_property', 'S16'),
        ('selected_count', 'i8'),
        ('pre_finite_count', 'i8'),
        ('R', 'f8'),
        ('z_table_ref', 'f8'),
        ('log_ssfr_cut', 'f8'),
        ('mstar_cut_1e10_msun', 'f8'),
        ('target_nbar_h3_Mpc3', 'f8'),
        ('target_count', 'i8'),
        ('eligible_count', 'i8'),
        ('property_threshold', 'f8'),
    ])
    arr = np.empty(len(samples), dtype=dt)
    for i, sample in enumerate(samples):
        arr[i] = (
            flag.encode(),
            int(snap),
            float(z),
            str(sample.get('name', '')).encode(),
            str(sample.get('sample_type', '')).encode(),
            str(sample.get('tracer', '')).encode(),
            str(sample.get('shape_mode', '')).encode(),
            str(sample.get('selection_property', '')).encode(),
            int(sample.get('selected_count', 0)),
            int(sample.get('pre_finite_count', 0)),
            float(sample.get('R', np.nan)),
            float(sample.get('z_table_ref', np.nan)),
            float(sample.get('log_ssfr_cut', np.nan)),
            float(sample.get('mstar_cut_1e10_msun', np.nan)),
            float(sample.get('target_nbar_h3_Mpc3', np.nan)),
            int(sample.get('target_count', -1)),
            int(sample.get('eligible_count', -1)),
            float(sample.get('property_threshold', np.nan)),
        )
    return arr


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--flag', required=True, help='Cosmology flag, e.g. GR, F40, F45, F50, F55, F60.')
    parser.add_argument('--snap', required=True, type=int, help='Snapshot number.')
    parser.add_argument('--threads', type=int, default=8, help='Thread count for Pylians.')
    parser.add_argument('--outdir', default='/cosma/home/dp203/dc-wang17/IA_analysis/pks', help='Output directory.')
    parser.add_argument('--nmesh', type=int, default=512, help='Mesh size per dimension.')
    parser.add_argument('--folds', default='1,2,4,8,16,32', help='Comma-separated folding factors.')
    parser.add_argument('--alpha', type=float, default=0.5, help='Trusted high-k fraction of each folded Nyquist frequency.')
    parser.add_argument('--overlap', type=float, default=0.999, help='Overlap-removal factor between adjacent folds.')
    parser.add_argument('--assign', default='CIC', help='MAS scheme for density/IA meshes.')
    parser.add_argument('--theta-mas', default='None', help="MAS deconvolution for theta fields; default is 'None'.")
    parser.add_argument('--cos-err-max', type=float, default=0.01, help='Maximum allowed shape cos_err.')
    parser.add_argument('--k-array', default=None, help='Target stitched k array, e.g. logspace:0.2:20:30.')
    parser.add_argument('--pk-types', default='full', help="Comma-separated spectra or aliases 'core'/'full'.")
    parser.add_argument('--threshold-mode', default='nearest', choices=['nearest', 'linear'], help='How to use the LRG/ELG threshold table.')
    parser.add_argument('--catalog-mass-unit', default='1e10Msun_h', choices=['1e10Msun_h', '1e10Msun'], help='Unit of SubhaloMassInRadType[:,4].')
    parser.add_argument('--exclusive-lrg-elg', action='store_true', help='Remove LRG objects from the ELG sample.')
    parser.add_argument('--density-nbars', default='1e-2,1e-3,1e-4', help='Comma-separated fixed number densities in h^3 Mpc^-3 for top-ranked samples.')
    parser.add_argument('--density-selection-modes', default='Mstar,SFR', help='Comma-separated ranking variables for fixed-nbar samples: Mstar,SFR.')
    parser.add_argument('--no-density-samples', action='store_true', help='Do not construct fixed-number-density Mstar/SFR samples.')
    parser.add_argument('--no-central-satellite', action='store_true', help='Do not write global central/satellite diagnostic samples.')
    parser.add_argument('--subtract-matter-shot-noise', action='store_true', help='Subtract mass-weighted particle shot noise from P_dd_corr.')
    parser.add_argument('--mg-root', default='/cosma8/data/dp203/dc-wang17/MG_global/', help='Directory containing MG catalogs.')
    parser.add_argument('--cs-root-base', default='/cosma8/data/dp203/bl267/Data/ClusterSims', help='Base ClusterSims snapshot directory.')
    parser.add_argument('--dm-fixed-mass', type=float, default=1.35401e9, help='Fixed DM particle mass used when Masses is absent.')
    args = parser.parse_args()

    setup_logging(args.flag, args.snap)

    if args.snap not in ZMAP:
        raise ValueError(f'snapshot {args.snap} is not in ZMAP: {sorted(ZMAP)}')
    z = ZMAP[args.snap]

    pk_types = parse_pk_types(args.pk_types)
    spec_keys = spec_keys_from_pk_types(pk_types)
    _, required_fields = pairs_and_required_fields(pk_types)
    need_t = 't' in required_fields
    need_tp = 'tp' in required_fields

    fold_list = [int(x) for x in str(args.folds).split(',') if x.strip()]
    target_k = parse_k_array(args.k_array)
    density_nbars = parse_float_list(args.density_nbars, default=(1e-2, 1e-3, 1e-4))
    density_selection_modes = [x.strip() for x in str(args.density_selection_modes).split(',') if x.strip()]
    los = (0.0, 0.0, 1.0)
    boxsize = 205.0

    mg_root = Path(args.mg_root)
    mg_file = mg_root / f'L302_N1136_{args.flag}_s{args.snap:03d}.hdf5'
    cs_root = Path(args.cs_root_base) / f'L302_N1136_{args.flag}'
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f'pks_{args.flag}_{args.snap:03d}.hdf5'

    cosmo, pars = get_cosmo()
    samples = load_science_samples(
        mg_file,
        z=z,
        h=pars['h'],
        boxsize=boxsize,
        cos_err_max=args.cos_err_max,
        los=los,
        catalog_mass_unit=args.catalog_mass_unit,
        threshold_mode=args.threshold_mode,
        exclusive_lrg_elg=bool(args.exclusive_lrg_elg),
        include_central_satellite=not bool(args.no_central_satellite),
        include_density_samples=not bool(args.no_density_samples),
        density_nbars=density_nbars,
        density_selection_modes=density_selection_modes,
    )
    cs_cat = CSCatalog(cs_root, args.snap)

    logging.info('loaded %d samples', len(samples))
    logging.info('pk_types=%s', ','.join(pk_types))
    logging.info('target_k=[%g,%g], N=%d', target_k[0], target_k[-1], len(target_k))

    # Per-sample cache used for final stitching.  Empty samples are kept in the
    # summary table but skipped in the expensive mesh loop.
    results_cache: Dict[str, Dict[int, Dict[str, Dict[str, np.ndarray]]]] = {
        sanitize_group_name(str(sample['name'])): {}
        for sample in samples
        if int(sample.get('selected_count', 0)) > 0
    }

    with h5py.File(outfile, 'w') as fout:
        fout.attrs['flag'] = args.flag
        fout.attrs['snap'] = int(args.snap)
        fout.attrs['z'] = float(z)
        fout.attrs['boxsize'] = float(boxsize)
        fout.attrs['nmesh'] = int(args.nmesh)
        fout.attrs['folds'] = ','.join(str(f) for f in fold_list)
        fout.attrs['alpha'] = float(args.alpha)
        fout.attrs['overlap'] = float(args.overlap)
        fout.attrs['assign'] = str(args.assign)
        fout.attrs['theta_mas'] = str(args.theta_mas)
        fout.attrs['cos_err_max'] = float(args.cos_err_max)
        fout.attrs['pk_types'] = ','.join(pk_types)
        fout.attrs['spec_keys'] = ','.join(spec_keys)
        fout.attrs['mg_file'] = str(mg_file)
        fout.attrs['cs_root'] = str(cs_root)
        fout.attrs['dm_fixed_mass'] = float(args.dm_fixed_mass)
        fout.attrs['catalog_mass_unit'] = str(args.catalog_mass_unit)
        fout.attrs['threshold_mode'] = str(args.threshold_mode)
        fout.attrs['exclusive_lrg_elg'] = bool(args.exclusive_lrg_elg)
        fout.attrs['include_density_samples'] = not bool(args.no_density_samples)
        fout.attrs['density_nbars'] = ','.join(nbar_label(x) for x in density_nbars)
        fout.attrs['density_selection_modes'] = ','.join(density_selection_modes)
        fout.attrs['subtract_matter_shot_noise'] = bool(args.subtract_matter_shot_noise)
        fout.create_dataset('target_k', data=target_k)
        fout.create_dataset('lrg_elg_table_z', data=np.asarray(LRG_ELG_TABLE['z'], dtype=float))
        fout.create_dataset('lrg_elg_table_log_ssfr_cut', data=np.asarray(LRG_ELG_TABLE['log_ssfr_cut'], dtype=float))
        fout.create_dataset('lrg_elg_table_mstar_cut_1e10_msun', data=np.asarray(LRG_ELG_TABLE['mstar_cut_1e10_msun'], dtype=float))
        for k, v in pars.items():
            fout.attrs[k] = float(v)

        for sample in samples:
            case = fout.require_group(sanitize_group_name(str(sample['name'])))
            add_sample_attrs(case, sample)

        for ifold, fold in enumerate(fold_list, start=1):
            logging.info('fold %d/%d | building particle mesh', ifold, len(fold_list))
            snap_out = build_particle_meshes(
                cs_cat,
                boxsize=boxsize,
                nmesh=args.nmesh,
                assign=args.assign,
                fold=fold,
                z=z,
                cosmo=cosmo,
                dm_fixed_mass=args.dm_fixed_mass,
                want_theta_p=need_tp,
            )
            delta_mesh_mean = float(np.mean(snap_out['delta_mesh'], dtype=np.float64))
            snap_meta = snap_out.get('meta', {}) if isinstance(snap_out, dict) else {}
            logging.info('fold %d/%d | particle mesh done | <delta>=%.4e', ifold, len(fold_list), delta_mesh_mean)

            for isample, sample in enumerate(samples, start=1):
                sample_name = sanitize_group_name(str(sample['name']))
                N = int(sample.get('selected_count', 0))
                logging.info('fold %d/%d | sample %d/%d | %s | N=%d', ifold, len(fold_list), isample, len(samples), sample_name, N)
                if N <= 0:
                    continue

                gal_out = build_galaxy_meshes(
                    sample,
                    boxsize=boxsize,
                    nmesh=args.nmesh,
                    assign=args.assign,
                    fold=fold,
                    z=z,
                    cosmo=cosmo,
                    los=los,
                    want_theta=need_t,
                )

                meshes: Dict[str, np.ndarray] = {
                    'g_mesh': gal_out['g_mesh'],
                    'E_mesh': gal_out['E_mesh'],
                    'B_mesh': gal_out['B_mesh'],
                    'delta_mesh': snap_out['delta_mesh'],
                }
                if 't_mesh' in gal_out:
                    meshes['t_mesh'] = gal_out['t_mesh']
                if 'tp_mesh' in snap_out:
                    meshes['tp_mesh'] = snap_out['tp_mesh']  # type: ignore[index]

                pk = measure_power(
                    meshes,
                    boxsize=boxsize,
                    fold=fold,
                    los=los,
                    threads=args.threads,
                    assign=args.assign,
                    theta_mas=args.theta_mas,
                    pk_types=pk_types,
                )
                noise = estimate_noise(
                    pk,
                    sample=sample,
                    boxsize=boxsize,
                    nmesh=args.nmesh,
                    fold=fold,
                    z=z,
                    h=pars['h'],
                    cosmo=cosmo,
                    snap_meta=snap_meta,
                    spec_keys=spec_keys,
                    subtract_matter_shot_noise=bool(args.subtract_matter_shot_noise),
                )
                pk_corr = {key: np.asarray(pk[key], dtype=float) - noise[key] for key in spec_keys}
                pk_corr['k'] = np.asarray(pk['k'], dtype=float)
                noise['k'] = np.asarray(pk['k'], dtype=float)

                case = fout[sample_name]
                write_fold_group(case, fold, pk, noise, pk_corr, delta_mesh_mean, spec_keys, snap_meta=snap_meta)
                results_cache[sample_name][fold] = {'pk': pk, 'pk_corr': pk_corr, 'noise': noise}
                fout.flush()

                del gal_out, meshes, pk, noise, pk_corr
                gc.collect()

            del snap_out
            gc.collect()

        for sample in samples:
            sample_name = sanitize_group_name(str(sample['name']))
            if sample_name not in results_cache or not results_cache[sample_name]:
                continue
            write_stitched_groups(
                fout[sample_name],
                results_cache[sample_name],
                fold_list=fold_list,
                boxsize=boxsize,
                nmesh=args.nmesh,
                alpha=args.alpha,
                overlap=args.overlap,
                target_k=target_k,
                spec_keys=spec_keys,
            )
            fout.flush()

        fout.create_dataset('summary', data=build_summary_array(samples, args.flag, args.snap, z))

    logging.info('saved %s', outfile)


if __name__ == '__main__':
    main()
