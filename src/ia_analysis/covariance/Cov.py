#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cov.py
======

Covariance utilities for the shape-tide IA P(k) pipeline.

This version is designed for a periodic simulation box.  It computes:

1. Gaussian covariance
   Cov[P_XY, P_UV] = [(P_XU+N_XU)(P_YV+N_YV)
                     +(P_XV+N_XV)(P_YU+N_YU)] / N_modes.

2. Connected non-Gaussian covariance, cNG
   Uses the CCL halo-model isotropized matter trispectrum
   ``pyccl.halos.pk_4pt.halomod_Tk3D_cNG``.  For IA fields the current closure is
   linear-bias + NLA:
        d = 1 * m,
        g = b_g * m,
        E = F_IA * m,
        B = 0,
   so T_XYUV = c_X c_Y c_U c_V T_mmmm.

3. Super-sample covariance, SSC
   Can use either a CCL halo-model SSC response product or the tree-level
   isotropic response.  For a full periodic simulation box, ``ssc_mode='periodic'``
   sets sigma_b^2=0, which is often the correct fixed-background convention.  If
   you want a finite-window estimate, use ``ssc_mode='cubic'`` or ``'spherical'``.

Units
-----
- k is h/Mpc.
- P(k) is (Mpc/h)^3.
- V is (Mpc/h)^3.
- CCL halo-model trispectra are evaluated internally at k in 1/Mpc and converted
  back to h-units.  For cNG this means T_h = h^9 T_Mpc and Cov_cNG = T_h / V_h.

Limitations
-----------
- Halo-model cNG is implemented for density-like fields d,g,E via the NLA closure.
  Velocity fields t/tp and B-mode IA receive zero cNG/SSC unless you extend the
  field-coefficient model.
- This is a covariance model, not a mock-calibrated covariance.  Treat small-scale
  cNG/SSC as a model-dependent term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import h5py

try:
    import pyccl as ccl
except Exception:  # pragma: no cover
    ccl = None


PK_TYPE_MAP: Dict[str, Tuple[str, str]] = {
    'gg': ('g', 'g'), 'gE': ('g', 'E'), 'gB': ('g', 'B'),
    'EE': ('E', 'E'), 'BB': ('B', 'B'), 'EB': ('E', 'B'),
    'dd': ('d', 'd'), 'dg': ('d', 'g'), 'dE': ('d', 'E'), 'dB': ('d', 'B'),
    'tt': ('t', 't'), 'gt': ('g', 't'), 'tE': ('t', 'E'), 'tB': ('t', 'B'), 'dt': ('d', 't'),
    'tptp': ('tp', 'tp'), 'dtp': ('d', 'tp'), 'gtp': ('g', 'tp'),
    'Etp': ('E', 'tp'), 'Btp': ('B', 'tp'), 'tpt': ('tp', 't'),
}


@dataclass
class HaloModelOptions:
    mass_def_delta: str = 'vir'
    mass_def_type: str = 'matter'
    mass_function: str = 'MassFuncSheth99'
    halo_bias: str = 'HaloBiasSheth99'
    concentration: str = 'constant'
    concentration_value: float = 5.0
    log10M_min: float = 8.0
    log10M_max: float = 16.0
    nM: int = 128
    nlk: int = 96
    na: int = 5
    use_log_trispectrum: bool = False
    separable_growth: bool = False


@dataclass
class CovarianceModel:
    k: np.ndarray
    pk_types: List[str]
    spectra: Dict[Tuple[str, str], np.ndarray]
    noise: Dict[Tuple[str, str], np.ndarray]
    nmodes: np.ndarray
    volume: float
    sigma_b2: float = 0.0
    responses: Optional[Dict[Tuple[str, str], np.ndarray]] = None
    field_coefficients: Optional[Dict[str, float]] = None


def canonical_pair(a: str, b: str) -> Tuple[str, str]:
    """Return an order-independent pair key for symmetric spectra."""
    return tuple(sorted((str(a), str(b))))  # type: ignore[return-value]


def pk_type_to_pair(pk_type: str) -> Tuple[str, str]:
    if pk_type not in PK_TYPE_MAP:
        raise ValueError(f"Unknown pk_type '{pk_type}'. Known keys: {sorted(PK_TYPE_MAP)}")
    return PK_TYPE_MAP[pk_type]


def spec_key(pk_type: str) -> str:
    return 'P_' + str(pk_type)


def finite_median(x: np.ndarray, default: float = np.nan) -> float:
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x)
    if not np.any(m):
        return float(default)
    return float(np.nanmedian(x[m]))


def make_ccl_cosmology(cosmo_params: Mapping[str, float]):
    if ccl is None:
        raise ImportError('pyccl is required for CCL/NLA/halo-model covariance spectra.')
    return ccl.Cosmology(
        Omega_c=float(cosmo_params['Omega_c']),
        Omega_b=float(cosmo_params['Omega_b']),
        h=float(cosmo_params['h']),
        sigma8=float(cosmo_params['sigma8']),
        n_s=float(cosmo_params['n_s']),
    )


def ccl_matter_pk_hmpc(
    k_hmpc: np.ndarray,
    z: float,
    cosmo_params: Mapping[str, float],
    *,
    nonlinear: bool = True,
) -> np.ndarray:
    """Return matter P(k) in (Mpc/h)^3 for k in h/Mpc."""
    cosmo = make_ccl_cosmology(cosmo_params)
    h = float(cosmo_params['h'])
    a = 1.0 / (1.0 + float(z))
    k_mpc = np.asarray(k_hmpc, dtype=float) * h
    if nonlinear:
        try:
            cosmo.compute_nonlin_power()
        except Exception:
            pass
        p_mpc3 = ccl.nonlin_matter_power(cosmo, k_mpc, a)
    else:
        p_mpc3 = ccl.linear_matter_power(cosmo, k_mpc, a)
    return np.asarray(p_mpc3, dtype=float) * h**3


def growth_factor(cosmo_params: Mapping[str, float], z: float) -> float:
    cosmo = make_ccl_cosmology(cosmo_params)
    a = 1.0 / (1.0 + float(z))
    return float(ccl.growth_factor(cosmo, a))


def estimate_nmodes_from_k(k: np.ndarray, volume: float, *, edges: Optional[np.ndarray] = None) -> np.ndarray:
    """Estimate spherical-shell mode counts for a periodic cubic box."""
    k = np.asarray(k, dtype=float)
    if edges is None:
        if k.size == 1:
            dk = max(float(k[0]), 1e-6)
            edges = np.array([max(0.0, k[0] - 0.5 * dk), k[0] + 0.5 * dk], dtype=float)
        else:
            edges = np.empty(k.size + 1, dtype=float)
            edges[1:-1] = 0.5 * (k[1:] + k[:-1])
            edges[0] = max(0.0, k[0] - 0.5 * (k[1] - k[0]))
            edges[-1] = k[-1] + 0.5 * (k[-1] - k[-2])
    edges = np.asarray(edges, dtype=float)
    shell_vol = 4.0 * np.pi / 3.0 * (edges[1:]**3 - edges[:-1]**3)
    fundamental_cell = (2.0 * np.pi) ** 3 / float(volume)
    nm = shell_vol / fundamental_cell
    return np.maximum(nm, 1.0)


def _read_group_spectra(h5_group, input_group: str, noise_group: str, pk_types: Sequence[str]):
    g = h5_group[input_group]
    gn = h5_group[noise_group] if noise_group in h5_group else None
    k = np.asarray(g['k'], dtype=float)

    spectra: Dict[Tuple[str, str], np.ndarray] = {}
    noise: Dict[Tuple[str, str], np.ndarray] = {}
    for pt in pk_types:
        a, b = pk_type_to_pair(pt)
        pair = canonical_pair(a, b)
        key = spec_key(pt) + '_Pk'
        if key in g:
            spectra[pair] = np.asarray(g[key], dtype=float)
        if gn is not None and key in gn:
            noise[pair] = np.asarray(gn[key], dtype=float)
        else:
            noise[pair] = np.zeros_like(k)
    return k, spectra, noise


def _maybe_get(spectra: Mapping[Tuple[str, str], np.ndarray], a: str, b: str) -> Optional[np.ndarray]:
    return spectra.get(canonical_pair(a, b), None)


def _nla_Fia_from_AIA(
    cosmo_params: Mapping[str, float],
    z: float,
    A_IA: float,
    c1_rhocrit: float,
) -> float:
    D = growth_factor(cosmo_params, z)
    omega_m = float(cosmo_params['Omega_c']) + float(cosmo_params['Omega_b'])
    return -float(A_IA) * float(c1_rhocrit) * omega_m / max(D, 1e-30)


def infer_bias_and_aia(
    k: np.ndarray,
    spectra: Mapping[Tuple[str, str], np.ndarray],
    *,
    z: float,
    cosmo_params: Mapping[str, float],
    aia_default: float = 1.0,
    c1_rhocrit: float = 0.0134,
    kmin_fit: float = 0.05,
    kmax_fit: float = 0.30,
) -> Tuple[float, float, Dict[str, float]]:
    """Infer b_g from P_dg/P_dd and A_IA from P_dE/P_dd when possible."""
    k = np.asarray(k, dtype=float)
    fit = np.isfinite(k) & (k >= float(kmin_fit)) & (k <= float(kmax_fit))
    Pdd = _maybe_get(spectra, 'd', 'd')
    Pdg = _maybe_get(spectra, 'd', 'g')
    PdE = _maybe_get(spectra, 'd', 'E')

    b = 1.0
    if Pdd is not None and Pdg is not None:
        denom = np.asarray(Pdd, dtype=float)
        ratio = np.asarray(Pdg, dtype=float) / np.where(denom != 0.0, denom, np.nan)
        b = finite_median(ratio[fit], default=1.0)
        if not np.isfinite(b) or b <= 0.0:
            b = 1.0

    aia = float(aia_default)
    Fia = np.nan
    if Pdd is not None and PdE is not None:
        denom = np.asarray(Pdd, dtype=float)
        ratio = np.asarray(PdE, dtype=float) / np.where(denom != 0.0, denom, np.nan)
        Fia = finite_median(ratio[fit], default=np.nan)
        if np.isfinite(Fia):
            D = growth_factor(cosmo_params, z)
            omega_m = float(cosmo_params['Omega_c']) + float(cosmo_params['Omega_b'])
            if c1_rhocrit * omega_m != 0.0:
                aia = -Fia * D / (float(c1_rhocrit) * omega_m)

    if not np.isfinite(Fia):
        Fia = _nla_Fia_from_AIA(cosmo_params, z, aia, c1_rhocrit)

    meta = {'bias_g': float(b), 'A_IA': float(aia), 'F_IA': float(Fia)}
    return float(b), float(aia), meta


def field_coefficients_from_meta(meta: Mapping[str, float]) -> Dict[str, float]:
    """Coefficients c_X in X = c_X delta_m for the NLA closure."""
    return {
        'd': 1.0,
        'g': float(meta.get('bias_g', 1.0)),
        'E': float(meta.get('F_IA', 0.0)),
        'B': 0.0,
        't': 0.0,
        'tp': 0.0,
    }


def nla_spectra(
    k: np.ndarray,
    z: float,
    cosmo_params: Mapping[str, float],
    *,
    b_g: float = 1.0,
    A_IA: float = 1.0,
    c1_rhocrit: float = 0.0134,
    pmm: Optional[np.ndarray] = None,
    nonlinear: bool = True,
) -> Dict[Tuple[str, str], np.ndarray]:
    """Build d/g/E/B spectra from CCL Pmm plus linear bias and NLA IA."""
    k = np.asarray(k, dtype=float)
    if pmm is None:
        pmm = ccl_matter_pk_hmpc(k, z, cosmo_params, nonlinear=nonlinear)
    else:
        pmm = np.asarray(pmm, dtype=float)

    Fia = _nla_Fia_from_AIA(cosmo_params, z, A_IA, c1_rhocrit)
    b = float(b_g)
    zero = np.zeros_like(k)
    return {
        canonical_pair('d', 'd'): pmm,
        canonical_pair('d', 'g'): b * pmm,
        canonical_pair('g', 'g'): b * b * pmm,
        canonical_pair('d', 'E'): Fia * pmm,
        canonical_pair('g', 'E'): b * Fia * pmm,
        canonical_pair('E', 'E'): Fia * Fia * pmm,
        canonical_pair('B', 'B'): zero,
        canonical_pair('E', 'B'): zero,
        canonical_pair('d', 'B'): zero,
        canonical_pair('g', 'B'): zero,
    }


def merge_spectra_for_covariance(
    k: np.ndarray,
    measured: Mapping[Tuple[str, str], np.ndarray],
    *,
    source: str,
    z: float,
    cosmo_params: Mapping[str, float],
    aia_default: float,
    c1_rhocrit: float,
    kmin_fit: float,
    kmax_fit: float,
) -> Tuple[Dict[Tuple[str, str], np.ndarray], Dict[str, float]]:
    """Return spectra dictionary used inside covariance."""
    b, aia, meta = infer_bias_and_aia(
        k, measured, z=z, cosmo_params=cosmo_params, aia_default=aia_default,
        c1_rhocrit=c1_rhocrit, kmin_fit=kmin_fit, kmax_fit=kmax_fit,
    )

    pmm_measured = _maybe_get(measured, 'd', 'd')
    if source == 'ccl_nla' or (source == 'measured_or_ccl_nla' and pmm_measured is None):
        base = nla_spectra(k, z, cosmo_params, b_g=b, A_IA=aia, c1_rhocrit=c1_rhocrit)
        meta['cov_source_effective'] = 'ccl_nla'
    else:
        pmm = pmm_measured
        base = nla_spectra(k, z, cosmo_params, b_g=b, A_IA=aia, c1_rhocrit=c1_rhocrit, pmm=pmm)
        meta['cov_source_effective'] = 'measured_dd_plus_nla_closure'

    if source in ('measured', 'measured_or_ccl_nla'):
        for pair, arr in measured.items():
            base[pair] = np.asarray(arr, dtype=float)
        if source == 'measured':
            meta['cov_source_effective'] = 'measured_with_nla_fill_for_missing_pairs'

    return base, meta


def total_power(
    spectra: Mapping[Tuple[str, str], np.ndarray],
    noise: Mapping[Tuple[str, str], np.ndarray],
    a: str,
    b: str,
    k: np.ndarray,
) -> np.ndarray:
    pair = canonical_pair(a, b)
    P = spectra.get(pair, np.zeros_like(k))
    N = noise.get(pair, np.zeros_like(k))
    return np.asarray(P, dtype=float) + np.asarray(N, dtype=float)


def gaussian_covariance(model: CovarianceModel) -> np.ndarray:
    """Return flattened Gaussian covariance matrix."""
    k = np.asarray(model.k, dtype=float)
    nk = k.size
    ns = len(model.pk_types)
    cov = np.zeros((ns * nk, ns * nk), dtype=float)
    nm = np.maximum(np.asarray(model.nmodes, dtype=float), 1.0)

    pairs = [pk_type_to_pair(pt) for pt in model.pk_types]
    for i, (x, y) in enumerate(pairs):
        for j, (u, v) in enumerate(pairs):
            term = (
                total_power(model.spectra, model.noise, x, u, k)
                * total_power(model.spectra, model.noise, y, v, k)
                + total_power(model.spectra, model.noise, x, v, k)
                * total_power(model.spectra, model.noise, y, u, k)
            ) / nm
            sl_i = slice(i * nk, (i + 1) * nk)
            sl_j = slice(j * nk, (j + 1) * nk)
            cov[sl_i, sl_j] = np.diag(term)
    return cov


def _get_ccl_halos_attr(name: str):
    if ccl is None:
        raise ImportError('pyccl is required for halo-model covariance.')
    if hasattr(ccl.halos, name):
        return getattr(ccl.halos, name)
    for sub in ('hmfunc', 'hbias', 'halo_model', 'massdef'):
        obj = getattr(ccl.halos, sub, None)
        if obj is not None and hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f'Could not find pyccl.halos.{name}')


def build_halo_model_components(cosmo, opts: HaloModelOptions):
    """Build CCL HMCalculator and NFW matter profile with version-tolerant fallbacks."""
    MassDef = _get_ccl_halos_attr('MassDef')
    try:
        mass_def = MassDef(opts.mass_def_delta, opts.mass_def_type)
    except Exception:
        # Newer CCL also accepts canonical names such as '200m'; vir is the target fallback.
        try:
            mass_def = MassDef.from_specs(opts.mass_def_delta, opts.mass_def_type)
        except Exception:
            mass_def = MassDef('vir', 'matter')

    MF = _get_ccl_halos_attr(opts.mass_function)
    HB = _get_ccl_halos_attr(opts.halo_bias)
    try:
        hmf = MF(mass_def=mass_def, mass_def_strict=False)
    except TypeError:
        hmf = MF(mass_def=mass_def)
    try:
        hbf = HB(mass_def=mass_def, mass_def_strict=False)
    except TypeError:
        hbf = HB(mass_def=mass_def)

    HMCalculator = _get_ccl_halos_attr('HMCalculator')
    hmc_kwargs = dict(
        mass_function=hmf,
        halo_bias=hbf,
        mass_def=mass_def,
        log10M_min=float(opts.log10M_min),
        log10M_max=float(opts.log10M_max),
        nM=int(opts.nM),
    )
    try:
        hmc = HMCalculator(**hmc_kwargs)
    except TypeError:
        # Older versions may not accept mass grid controls.
        hmc = HMCalculator(mass_function=hmf, halo_bias=hbf, mass_def=mass_def)

    # Concentration and NFW profile.
    prof_cls = ccl.halos.profiles.nfw.HaloProfileNFW
    concentration = None
    if str(opts.concentration).lower() == 'constant':
        try:
            conc_cls = ccl.halos.concentration.ConcentrationConstant
            concentration = conc_cls(c=float(opts.concentration_value), mass_def=mass_def)
        except Exception:
            concentration = 'Constant'
    elif hasattr(ccl.halos.concentration, str(opts.concentration)):
        conc_cls = getattr(ccl.halos.concentration, str(opts.concentration))
        try:
            concentration = conc_cls(mass_def=mass_def)
        except Exception:
            concentration = conc_cls()
    else:
        concentration = str(opts.concentration)

    try:
        prof = prof_cls(mass_def=mass_def, concentration=concentration)
    except TypeError:
        prof = prof_cls(mass_def=mass_def)
    return hmc, prof


def _a_arr_around_z(z: float, na: int = 5) -> np.ndarray:
    a = 1.0 / (1.0 + float(z))
    if int(na) < 2:
        return np.array([a], dtype=float)
    lo = max(0.03, min(a * 0.70, a - 0.05))
    hi = min(1.0, max(a * 1.30, a + 0.05))
    arr = np.linspace(lo, hi, int(na))
    arr = np.unique(np.sort(np.concatenate([arr, [a]])))
    return arr


def _lk_arr_for_k_hmpc(k_hmpc: np.ndarray, h: float, nlk: int = 96) -> np.ndarray:
    k_mpc = np.asarray(k_hmpc, dtype=float) * float(h)
    kpos = k_mpc[np.isfinite(k_mpc) & (k_mpc > 0)]
    if kpos.size == 0:
        raise ValueError('k array has no positive finite entries.')
    kmin = max(float(np.min(kpos)) * 0.5, 1e-5)
    kmax = max(float(np.max(kpos)) * 2.0, kmin * 1.01)
    nk = max(int(nlk), int(kpos.size), 8)
    return np.log(np.geomspace(kmin, kmax, nk))


def halo_model_matter_cng_trispectrum_hunits(
    k_hmpc: np.ndarray,
    z: float,
    cosmo_params: Mapping[str, float],
    *,
    hm_options: Optional[HaloModelOptions] = None,
) -> np.ndarray:
    """Return CCL halo-model matter cNG trispectrum T_mmmm in (Mpc/h)^9."""
    if ccl is None:
        raise ImportError('pyccl is required for halo-model cNG covariance.')
    opts = hm_options or HaloModelOptions()
    cosmo = make_ccl_cosmology(cosmo_params)
    h = float(cosmo_params['h'])
    k_hmpc = np.asarray(k_hmpc, dtype=float)
    k_mpc = k_hmpc * h
    a = 1.0 / (1.0 + float(z))
    hmc, prof = build_halo_model_components(cosmo, opts)
    lk_arr = _lk_arr_for_k_hmpc(k_hmpc, h, opts.nlk)
    a_arr = _a_arr_around_z(z, opts.na)

    tk = ccl.halos.pk_4pt.halomod_Tk3D_cNG(
        cosmo,
        hmc,
        prof,
        lk_arr=lk_arr,
        a_arr=a_arr,
        use_log=bool(opts.use_log_trispectrum),
        separable_growth=bool(opts.separable_growth),
    )
    T_mpc9 = np.asarray(tk(k_mpc, a), dtype=float)
    T_h9 = T_mpc9 * h**9
    return np.nan_to_num(T_h9, nan=0.0, posinf=0.0, neginf=0.0)


def halo_model_matter_ssc_response_product_hunits(
    k_hmpc: np.ndarray,
    z: float,
    cosmo_params: Mapping[str, float],
    *,
    hm_options: Optional[HaloModelOptions] = None,
) -> np.ndarray:
    """Return CCL halo-model matter SSC response product in ((Mpc/h)^3)^2."""
    if ccl is None:
        raise ImportError('pyccl is required for halo-model SSC covariance.')
    opts = hm_options or HaloModelOptions()
    cosmo = make_ccl_cosmology(cosmo_params)
    h = float(cosmo_params['h'])
    k_hmpc = np.asarray(k_hmpc, dtype=float)
    k_mpc = k_hmpc * h
    a = 1.0 / (1.0 + float(z))
    hmc, prof = build_halo_model_components(cosmo, opts)
    lk_arr = _lk_arr_for_k_hmpc(k_hmpc, h, opts.nlk)
    a_arr = _a_arr_around_z(z, opts.na)

    tk = ccl.halos.pk_4pt.halomod_Tk3D_SSC(
        cosmo,
        hmc,
        prof,
        lk_arr=lk_arr,
        a_arr=a_arr,
        use_log=bool(opts.use_log_trispectrum),
    )
    T_mpc6 = np.asarray(tk(k_mpc, a), dtype=float)
    T_h6 = T_mpc6 * h**6
    return np.nan_to_num(T_h6, nan=0.0, posinf=0.0, neginf=0.0)


def _pair_coeff(pair: Tuple[str, str], coeff: Mapping[str, float]) -> float:
    a, b = pair
    return float(coeff.get(a, 0.0)) * float(coeff.get(b, 0.0))


def cng_covariance_halomodel(
    model: CovarianceModel,
    *,
    cosmo_params: Mapping[str, float],
    z: float,
    hm_options: Optional[HaloModelOptions] = None,
    include: bool = True,
) -> np.ndarray:
    """Connected non-Gaussian covariance using CCL halo-model T_mmmm/V."""
    k = np.asarray(model.k, dtype=float)
    nk = k.size
    ns = len(model.pk_types)
    cov = np.zeros((ns * nk, ns * nk), dtype=float)
    if not include:
        return cov

    coeff = model.field_coefficients or {'d': 1.0, 'g': 1.0, 'E': 0.0, 'B': 0.0, 't': 0.0, 'tp': 0.0}
    T = halo_model_matter_cng_trispectrum_hunits(k, z, cosmo_params, hm_options=hm_options)
    T_over_V = T / float(model.volume)

    pairs = [pk_type_to_pair(pt) for pt in model.pk_types]
    for i, pi_raw in enumerate(pairs):
        pi = canonical_pair(*pi_raw)
        ci = _pair_coeff(pi, coeff)
        if ci == 0.0:
            continue
        for j, pj_raw in enumerate(pairs):
            pj = canonical_pair(*pj_raw)
            cj = _pair_coeff(pj, coeff)
            if cj == 0.0:
                continue
            cov[i*nk:(i+1)*nk, j*nk:(j+1)*nk] = ci * cj * T_over_V
    return cov


def dln_k3p_dlnk(k: np.ndarray, p: np.ndarray) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    p = np.asarray(p, dtype=float)
    y = np.log(np.maximum(k**3 * np.abs(p), 1e-300))
    x = np.log(np.maximum(k, 1e-300))
    if k.size < 3:
        return np.zeros_like(k)
    return np.gradient(y, x, edge_order=1)


def matter_ssc_response(k: np.ndarray, pmm: np.ndarray) -> np.ndarray:
    """Tree-level isotropic matter power response to background density."""
    return 68.0 / 21.0 - (1.0 / 3.0) * dln_k3p_dlnk(k, pmm)


def build_ssc_responses_tree(
    k: np.ndarray,
    pk_types: Sequence[str],
    spectra: Mapping[Tuple[str, str], np.ndarray],
) -> Dict[Tuple[str, str], np.ndarray]:
    pmm = spectra.get(canonical_pair('d', 'd'), None)
    if pmm is None:
        pmm = np.ones_like(k)
    Rm = matter_ssc_response(k, np.asarray(pmm, dtype=float))
    responses: Dict[Tuple[str, str], np.ndarray] = {}
    for pt in pk_types:
        pair = canonical_pair(*pk_type_to_pair(pt))
        P = np.asarray(spectra.get(pair, np.zeros_like(k)), dtype=float)
        if pair == canonical_pair('B', 'B') or pair == canonical_pair('E', 'B'):
            responses[pair] = np.zeros_like(k)
        else:
            responses[pair] = Rm * P
    return responses


def sigma_b2_spherical_tophat(
    volume: float,
    z: float,
    cosmo_params: Mapping[str, float],
    *,
    qmin: float = 1e-4,
    qmax: float = 10.0,
    nq: int = 512,
) -> float:
    """Approximate background variance using an equal-volume spherical window."""
    q = np.logspace(np.log10(qmin), np.log10(qmax), int(nq))  # h/Mpc
    R = (3.0 * float(volume) / (4.0 * np.pi)) ** (1.0 / 3.0)  # Mpc/h
    p = ccl_matter_pk_hmpc(q, z, cosmo_params, nonlinear=False)
    x = q * R
    W = np.ones_like(x)
    m = np.abs(x) > 1e-4
    xm = x[m]
    W[m] = 3.0 * (np.sin(xm) - xm * np.cos(xm)) / (xm**3)
    integrand = q**2 * p * W**2 / (2.0 * np.pi**2)
    return float(np.trapz(integrand, q))


def sigma_b2_cubic_window(
    boxsize: float,
    z: float,
    cosmo_params: Mapping[str, float],
    *,
    ngrid: int = 64,
    qmax: Optional[float] = None,
) -> float:
    """Numerically integrate sigma_b^2 for a cubic top-hat window of side L.

    sigma_b^2 = ∫ d^3q/(2π)^3 P_L(q) Π_i sinc^2(q_i L/2).
    q and L are in h/Mpc and Mpc/h respectively.
    """
    L = float(boxsize)
    if qmax is None:
        # Large enough for the sinc tails of a ~200 Mpc/h box without being too slow.
        qmax = max(1.0, 80.0 * 2.0 * np.pi / L)
    q1 = np.linspace(-float(qmax), float(qmax), int(ngrid))
    dq = float(q1[1] - q1[0]) if q1.size > 1 else 2.0 * float(qmax)
    qx, qy, qz = np.meshgrid(q1, q1, q1, indexing='ij')
    q = np.sqrt(qx*qx + qy*qy + qz*qz)
    W = (
        np.sinc(qx * L / (2.0 * np.pi))
        * np.sinc(qy * L / (2.0 * np.pi))
        * np.sinc(qz * L / (2.0 * np.pi))
    )
    p = np.zeros_like(q)
    m = q > 0.0
    p[m] = ccl_matter_pk_hmpc(q[m], z, cosmo_params, nonlinear=False)
    integral = np.sum(p * W*W) * dq**3 / (2.0 * np.pi)**3
    return float(max(integral, 0.0))


def get_sigma_b2(
    *,
    mode: str,
    boxsize: float,
    volume: float,
    z: float,
    cosmo_params: Mapping[str, float],
    cubic_ngrid: int = 64,
) -> float:
    mode = str(mode).strip().lower()
    if mode in ('none', 'periodic', 'periodic_fixed_mean', 'fixed_mean'):
        return 0.0
    if mode in ('spherical', 'spherical_tophat', 'sphere'):
        return sigma_b2_spherical_tophat(volume, z, cosmo_params)
    if mode in ('cubic', 'cube', 'cubic_window', 'box'):
        return sigma_b2_cubic_window(boxsize, z, cosmo_params, ngrid=cubic_ngrid)
    raise ValueError("ssc_mode must be one of {'periodic','cubic','spherical','none'}")


def ssc_covariance(
    model: CovarianceModel,
    *,
    cosmo_params: Optional[Mapping[str, float]] = None,
    z: Optional[float] = None,
    hm_options: Optional[HaloModelOptions] = None,
    backend: str = 'ccl_halomodel',
) -> np.ndarray:
    k = np.asarray(model.k, dtype=float)
    nk = k.size
    ns = len(model.pk_types)
    cov = np.zeros((ns * nk, ns * nk), dtype=float)
    if model.sigma_b2 <= 0.0:
        return cov

    coeff = model.field_coefficients or {'d': 1.0, 'g': 1.0, 'E': 0.0, 'B': 0.0, 't': 0.0, 'tp': 0.0}
    backend = str(backend).strip().lower()

    if backend in ('ccl', 'ccl_halomodel', 'halo_model'):
        if cosmo_params is None or z is None:
            raise ValueError('cosmo_params and z are required for ccl_halomodel SSC.')
        Tresp_mm = halo_model_matter_ssc_response_product_hunits(k, float(z), cosmo_params, hm_options=hm_options)
        pairs = [canonical_pair(*pk_type_to_pair(pt)) for pt in model.pk_types]
        for i, pi in enumerate(pairs):
            ci = _pair_coeff(pi, coeff)
            if ci == 0.0:
                continue
            for j, pj in enumerate(pairs):
                cj = _pair_coeff(pj, coeff)
                if cj == 0.0:
                    continue
                cov[i*nk:(i+1)*nk, j*nk:(j+1)*nk] = model.sigma_b2 * ci * cj * Tresp_mm
        return cov

    # Tree-level response fallback.
    responses = model.responses or build_ssc_responses_tree(k, model.pk_types, model.spectra)
    pairs = [canonical_pair(*pk_type_to_pair(pt)) for pt in model.pk_types]
    for i, pi in enumerate(pairs):
        ri = np.asarray(responses.get(pi, np.zeros_like(k)), dtype=float)
        for j, pj in enumerate(pairs):
            rj = np.asarray(responses.get(pj, np.zeros_like(k)), dtype=float)
            cov[i*nk:(i+1)*nk, j*nk:(j+1)*nk] = model.sigma_b2 * np.outer(ri, rj)
    return cov


def write_covariance_hdf5_group(
    h5_group,
    *,
    pk_types: Sequence[str],
    cosmo_params: Mapping[str, float],
    z: float,
    boxsize: float,
    nmesh: int,
    source: str = 'measured_or_ccl_nla',
    aia_default: float = 1.0,
    c1_rhocrit: float = 0.0134,
    kmin_fit: float = 0.05,
    kmax_fit: float = 0.30,
    include_cng: bool = True,
    include_ssc: bool = True,
    ssc_mode: str = 'cubic',
    ssc_backend: str = 'ccl_halomodel',
    hm_options: Optional[HaloModelOptions] = None,
    # Deprecated compatibility parameter. Ignored unless include_cng=False and you
    # choose to add your own diagonal model later.
    cng_fraction: float = 0.0,
    input_group: str = 'stitched_corr',
    noise_group: str = 'stitched_noise',
    output_group: str = 'covariance',
) -> None:
    """Build and write covariance matrices under one sample HDF5 group."""
    if input_group not in h5_group:
        raise KeyError(f"Missing input group '{input_group}' in sample group {h5_group.name}")

    pk_types = list(pk_types)
    k, measured, noise = _read_group_spectra(h5_group, input_group, noise_group, pk_types)
    volume = float(boxsize) ** 3
    nmodes = estimate_nmodes_from_k(k, volume)
    spectra, meta = merge_spectra_for_covariance(
        k, measured, source=source, z=z, cosmo_params=cosmo_params,
        aia_default=aia_default, c1_rhocrit=c1_rhocrit,
        kmin_fit=kmin_fit, kmax_fit=kmax_fit,
    )
    field_coeff = field_coefficients_from_meta(meta)

    sigma_b2 = 0.0
    if include_ssc:
        sigma_b2 = get_sigma_b2(
            mode=ssc_mode,
            boxsize=float(boxsize),
            volume=volume,
            z=float(z),
            cosmo_params=cosmo_params,
        )

    model = CovarianceModel(
        k=k,
        pk_types=pk_types,
        spectra=spectra,
        noise=noise,
        nmodes=nmodes,
        volume=volume,
        sigma_b2=sigma_b2,
        responses=build_ssc_responses_tree(k, pk_types, spectra),
        field_coefficients=field_coeff,
    )

    cov_g = gaussian_covariance(model)
    cov_c = cng_covariance_halomodel(
        model,
        cosmo_params=cosmo_params,
        z=float(z),
        hm_options=hm_options,
        include=bool(include_cng),
    )
    cov_s = ssc_covariance(
        model,
        cosmo_params=cosmo_params,
        z=float(z),
        hm_options=hm_options,
        backend=ssc_backend,
    ) if include_ssc else np.zeros_like(cov_g)
    cov_t = cov_g + cov_c + cov_s

    if output_group in h5_group:
        del h5_group[output_group]
    out = h5_group.create_group(output_group)
    out.attrs['description'] = 'Flattened covariance matrices for stitched spectra; index = ispec*Nk + ik.'
    out.attrs['input_group'] = input_group
    out.attrs['noise_group'] = noise_group
    out.attrs['source_requested'] = str(source)
    out.attrs['source_effective'] = str(meta.get('cov_source_effective', 'unknown'))
    out.attrs['boxsize'] = float(boxsize)
    out.attrs['volume'] = volume
    out.attrs['nmesh'] = int(nmesh)
    out.attrs['z'] = float(z)
    out.attrs['bias_g'] = float(meta.get('bias_g', np.nan))
    out.attrs['A_IA'] = float(meta.get('A_IA', np.nan))
    out.attrs['F_IA'] = float(meta.get('F_IA', np.nan))
    out.attrs['c1_rhocrit'] = float(c1_rhocrit)
    out.attrs['sigma_b2'] = float(sigma_b2)
    out.attrs['ssc_mode'] = str(ssc_mode)
    out.attrs['ssc_backend'] = str(ssc_backend)
    out.attrs['ssc_model'] = 'CCL halo-model response product' if str(ssc_backend).lower().startswith(('ccl','halo')) else 'tree-level isotropic response'
    out.attrs['include_cNG'] = bool(include_cng)
    out.attrs['cng_model'] = 'CCL halo-model halomod_Tk3D_cNG mapped through d,g,E NLA coefficients' if include_cng else 'disabled'
    out.attrs['deprecated_cng_fraction_ignored'] = float(cng_fraction)
    out.attrs['pk_types'] = ','.join(pk_types)
    out.attrs['hm_mass_def_delta'] = (hm_options or HaloModelOptions()).mass_def_delta
    out.attrs['hm_mass_def_type'] = (hm_options or HaloModelOptions()).mass_def_type
    out.attrs['hm_mass_function'] = (hm_options or HaloModelOptions()).mass_function
    out.attrs['hm_halo_bias'] = (hm_options or HaloModelOptions()).halo_bias
    out.attrs['hm_concentration'] = (hm_options or HaloModelOptions()).concentration
    out.attrs['hm_log10M_min'] = float((hm_options or HaloModelOptions()).log10M_min)
    out.attrs['hm_log10M_max'] = float((hm_options or HaloModelOptions()).log10M_max)
    out.attrs['hm_nM'] = int((hm_options or HaloModelOptions()).nM)

    out.create_dataset('k', data=k)
    out.create_dataset('Nmodes', data=nmodes)
    out.create_dataset('cov_gaussian', data=cov_g)
    out.create_dataset('cov_cNG', data=cov_c)
    out.create_dataset('cov_SSC', data=cov_s)
    out.create_dataset('cov_total', data=cov_t)
    out.create_dataset('spec_labels', data=np.asarray([p.encode() for p in pk_types]))

    gcg = out.create_group('field_coefficients')
    for name, val in field_coeff.items():
        gcg.attrs[name] = float(val)

    gs = out.create_group('model_spectra')
    for pt in pk_types:
        pair = canonical_pair(*pk_type_to_pair(pt))
        gs.create_dataset(spec_key(pt), data=np.asarray(spectra.get(pair, np.zeros_like(k)), dtype=float))
        gs.create_dataset(spec_key(pt) + '_noise', data=np.asarray(noise.get(pair, np.zeros_like(k)), dtype=float))
        resp = model.responses.get(pair, np.zeros_like(k)) if model.responses is not None else np.zeros_like(k)
        gs.create_dataset(spec_key(pt) + '_tree_response_dP_ddelta_b', data=np.asarray(resp, dtype=float))


def build_covariance_from_arrays(
    k: np.ndarray,
    pk_types: Sequence[str],
    measured_spectra: Mapping[Tuple[str, str], np.ndarray],
    noise: Optional[Mapping[Tuple[str, str], np.ndarray]],
    *,
    cosmo_params: Mapping[str, float],
    z: float,
    volume: float,
    source: str = 'measured_or_ccl_nla',
    aia_default: float = 1.0,
    c1_rhocrit: float = 0.0134,
    include_cng: bool = True,
    include_ssc: bool = True,
    ssc_mode: str = 'cubic',
    ssc_backend: str = 'ccl_halomodel',
    hm_options: Optional[HaloModelOptions] = None,
) -> Dict[str, np.ndarray]:
    """Standalone array API returning Gaussian, cNG, SSC, and total covariance."""
    k = np.asarray(k, dtype=float)
    spectra, meta = merge_spectra_for_covariance(
        k, measured_spectra, source=source, z=z, cosmo_params=cosmo_params,
        aia_default=aia_default, c1_rhocrit=c1_rhocrit, kmin_fit=0.05, kmax_fit=0.30,
    )
    boxsize = float(volume) ** (1.0 / 3.0)
    sigma_b2 = get_sigma_b2(mode=ssc_mode, boxsize=boxsize, volume=volume, z=z, cosmo_params=cosmo_params) if include_ssc else 0.0
    model = CovarianceModel(
        k=k,
        pk_types=list(pk_types),
        spectra=spectra,
        noise=dict(noise or {}),
        nmodes=estimate_nmodes_from_k(k, volume),
        volume=float(volume),
        sigma_b2=sigma_b2,
        responses=build_ssc_responses_tree(k, pk_types, spectra),
        field_coefficients=field_coefficients_from_meta(meta),
    )
    cg = gaussian_covariance(model)
    cc = cng_covariance_halomodel(model, cosmo_params=cosmo_params, z=z, hm_options=hm_options, include=include_cng)
    cs = ssc_covariance(model, cosmo_params=cosmo_params, z=z, hm_options=hm_options, backend=ssc_backend) if include_ssc else np.zeros_like(cg)
    return {'gaussian': cg, 'cNG': cc, 'SSC': cs, 'total': cg + cc + cs, 'k': k, 'Nmodes': model.nmodes}


__all__ = [
    'PK_TYPE_MAP', 'HaloModelOptions', 'CovarianceModel', 'canonical_pair', 'pk_type_to_pair',
    'ccl_matter_pk_hmpc', 'nla_spectra', 'infer_bias_and_aia',
    'estimate_nmodes_from_k', 'gaussian_covariance', 'cng_covariance_halomodel',
    'ssc_covariance', 'write_covariance_hdf5_group', 'build_covariance_from_arrays',
    'sigma_b2_cubic_window', 'sigma_b2_spherical_tophat',
]


def _parse_pk_types_cli(text: Optional[str], h5root=None) -> List[str]:
    if text is None or not str(text).strip():
        if h5root is not None and 'pk_types' in h5root.attrs:
            raw = h5root.attrs['pk_types']
            if isinstance(raw, bytes):
                raw = raw.decode()
            return [x.strip() for x in str(raw).split(',') if x.strip()]
        return ['gg', 'gE', 'EE', 'BB', 'dd', 'dg', 'dE']
    s = str(text).strip()
    if s.lower() == 'core':
        return ['gg', 'gE', 'EE', 'BB', 'dd', 'dg', 'dE']
    if s.lower() == 'full':
        return [
            'gg', 'gE', 'gB', 'EE', 'BB', 'EB', 'dd', 'dg', 'dE', 'dB',
            'tt', 'gt', 'tE', 'tB', 'dt', 'tptp', 'dtp', 'gtp', 'Etp', 'Btp', 'tpt'
        ]
    return [x.strip() for x in s.split(',') if x.strip()]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description='Write covariance matrices into an existing pks_*.hdf5 file.')
    ap.add_argument('--input', required=True, help='Input/output HDF5 file produced by ia_pk_cs.py.')
    ap.add_argument('--sample', default='all', help='Sample group name, e.g. all, LRG, ELG, Mstar/1e-3.')
    ap.add_argument('--pk-types', default=None, help="Comma list or aliases 'core'/'full'. Default reads file attrs.")
    ap.add_argument('--source', default='measured_or_ccl_nla', choices=['measured_or_ccl_nla', 'ccl_nla', 'measured'])
    ap.add_argument('--input-group', default='stitched_corr')
    ap.add_argument('--noise-group', default='stitched_noise')
    ap.add_argument('--output-group', default='covariance')
    ap.add_argument('--a-ia', type=float, default=1.0)
    ap.add_argument('--c1-rhocrit', type=float, default=0.0134)
    ap.add_argument('--kmin-fit', type=float, default=0.05)
    ap.add_argument('--kmax-fit', type=float, default=0.30)
    ap.add_argument('--no-cng', action='store_true', help='Disable halo-model connected non-Gaussian covariance.')
    ap.add_argument('--no-ssc', action='store_true', help='Disable SSC covariance term.')
    ap.add_argument('--ssc-mode', default='cubic', choices=['cubic', 'spherical', 'periodic', 'none'], help='Background-mode variance model.')
    ap.add_argument('--ssc-backend', default='ccl_halomodel', choices=['ccl_halomodel', 'tree'], help='SSC response backend.')
    ap.add_argument('--hm-log10m-min', type=float, default=8.0)
    ap.add_argument('--hm-log10m-max', type=float, default=16.0)
    ap.add_argument('--hm-nm', type=int, default=128)
    ap.add_argument('--hm-nlk', type=int, default=96)
    ap.add_argument('--hm-na', type=int, default=5)
    ap.add_argument('--hm-concentration-value', type=float, default=5.0)
    args = ap.parse_args()

    hmopts = HaloModelOptions(
        log10M_min=args.hm_log10m_min,
        log10M_max=args.hm_log10m_max,
        nM=args.hm_nm,
        nlk=args.hm_nlk,
        na=args.hm_na,
        concentration_value=args.hm_concentration_value,
    )

    with h5py.File(args.input, 'r+') as f:
        if args.sample not in f:
            raise KeyError(f"Sample group '{args.sample}' not found in {args.input}")
        pars = {k: float(f.attrs[k]) for k in ['Omega_c', 'Omega_b', 'h', 'sigma8', 'n_s']}
        pk_types = _parse_pk_types_cli(args.pk_types, f)
        write_covariance_hdf5_group(
            f[args.sample],
            pk_types=pk_types,
            cosmo_params=pars,
            z=float(f.attrs['z']),
            boxsize=float(f.attrs['boxsize']),
            nmesh=int(f.attrs['nmesh']),
            source=args.source,
            aia_default=args.a_ia,
            c1_rhocrit=args.c1_rhocrit,
            kmin_fit=args.kmin_fit,
            kmax_fit=args.kmax_fit,
            include_cng=not bool(args.no_cng),
            include_ssc=not bool(args.no_ssc),
            ssc_mode=args.ssc_mode,
            ssc_backend=args.ssc_backend,
            hm_options=hmopts,
            input_group=args.input_group,
            noise_group=args.noise_group,
            output_group=args.output_group,
        )
        print(f"[Cov.py] wrote {args.sample}/{args.output_group} in {args.input}")


if __name__ == '__main__':
    main()
