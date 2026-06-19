#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
theory_nla_pk.py

Compute simple large-scale theory spectra and Gaussian errors for
    Pgg, PgE, PEE, Ptt, Pgt, PtE
under the assumptions:
- NLA intrinsic alignment model
- theta = f delta on all scales
- nonlinear matter power from CCL
- growth rate from CCL

This script is intended as a callable helper rather than a full fitting code.
"""

import numpy as np
import pyccl as ccl

C1_RHOCRIT = 0.0134


def build_cosmo(cosmo_dict):
    """Build a CCL cosmology from a plain parameter dictionary."""
    Om = cosmo_dict['Omega0']
    Ob = cosmo_dict['OmegaBaryon']
    Oc = Om - Ob
    h = cosmo_dict['HubbleParam']
    sigma8 = cosmo_dict['sigma8']
    n_s = cosmo_dict['n_s']
    return ccl.Cosmology(Omega_c=Oc, Omega_b=Ob, h=h, sigma8=sigma8, n_s=n_s)


def h_over_h0(cosmo, z):
    a = 1.0 / (1.0 + z)
    return ccl.background.h_over_h0(cosmo, a)


def growth_factor(cosmo, z):
    a = 1.0 / (1.0 + z)
    return ccl.background.growth_factor(cosmo, a)


def growth_rate(cosmo, z):
    a = 1.0 / (1.0 + z)
    return ccl.background.growth_rate(cosmo, a)


def matter_power_hunits(cosmo, k_hmpc, z):
    """Return nonlinear matter power in (Mpc/h)^3 for k in h/Mpc."""
    a = 1.0 / (1.0 + z)
    h = cosmo['h']
    k_mpc = np.asarray(k_hmpc, dtype=float) * h
    p_mpc3 = ccl.nonlin_matter_power(cosmo, k_mpc, a)
    return p_mpc3 * h**3


def ia_prefactor_nla(cosmo_dict, D, A_IA=1.0, eta_IA=0.0, z=0.0, z0=0.0):
    """NLA amplitude prefactor F_IA(z)."""
    Om = cosmo_dict['Omega0']
    return -A_IA * C1_RHOCRIT * Om / D * ((1.0 + z) / (1.0 + z0))**eta_IA


def k_eff_1d(k, dx):
    return np.sin(np.asarray(k) * dx) / dx


def nmodes_box(k, dk, boxsize):
    """Approximate shell mode count in a periodic box."""
    V = boxsize**3
    return V * np.asarray(k) ** 2 * np.asarray(dk) / (2.0 * np.pi**2)


def gaussian_sigma_auto(Ptot, Nmodes):
    return np.sqrt(2.0 / Nmodes) * Ptot


def gaussian_sigma_cross(Paa_tot, Pbb_tot, Pab_tot, Nmodes):
    return np.sqrt((Paa_tot * Pbb_tot + Pab_tot**2) / Nmodes)


def compute_nla_spectra_and_errors(k, z, cosmo_dict, bias, n_g, boxsize, dk,
                                   A_IA=1.0, eta_IA=0.0, sigma_gamma=0.2,
                                   sigma_v1d=300.0, use_keff=True, z0_IA=0.0,
                                   cosmo=None):
    """
    Compute theory spectra and Gaussian errors for galaxy density, IA E-mode,
    and velocity-divergence spectra.

    Parameters
    ----------
    k : array
        Wavenumber in h/Mpc.
    z : float
        Redshift.
    cosmo_dict : dict
        Must contain Omega0, OmegaBaryon, HubbleParam, sigma8, n_s.
    bias : float or array
        Linear galaxy bias b1.
    n_g : float
        Galaxy number density in (h/Mpc)^3.
    boxsize : float
        Periodic box size in Mpc/h.
    dk : float or array
        Bin width in h/Mpc.
    A_IA : float
        NLA IA amplitude. Default 1.
    eta_IA : float
        Redshift power-law index for A_IA.
    sigma_gamma : float
        RMS per-component IA/shear field for white shape noise.
    sigma_v1d : float
        One-dimensional galaxy velocity dispersion in km/s.
    use_keff : bool
        If True, use k_eff for Ptt noise.
    z0_IA : float
        Pivot redshift for the IA redshift scaling.

    Returns
    -------
    dict
        Spectra, noise terms, and Gaussian 1-sigma errors.
    """
    k = np.asarray(k, dtype=float)
    dk = np.asarray(dk, dtype=float)
    b1 = np.asarray(bias, dtype=float)
    if b1.ndim == 0:
        b1 = np.full_like(k, float(b1))

    if cosmo is None:
        cosmo = build_cosmo(cosmo_dict)

    a = 1.0 / (1.0 + z)
    h = cosmo_dict['HubbleParam']
    Hz = 100.0 * h * ccl.background.h_over_h0(cosmo, a)
    D = ccl.background.growth_factor(cosmo, a)
    f = ccl.background.growth_rate(cosmo, a)
    Pmm = matter_power_hunits(cosmo, k, z)
    Fia = ia_prefactor_nla(cosmo_dict, D, A_IA=A_IA, eta_IA=eta_IA, z=z, z0=z0_IA)

    Pgg = b1**2 * Pmm
    PgE = b1 * Fia * Pmm
    PEE = Fia**2 * Pmm
    Ptt = f**2 * Pmm
    Pgt = b1 * f * Pmm
    PtE = f * Fia * Pmm

    Ngg = np.full_like(k, 1.0 / n_g)
    Nee = np.full_like(k, sigma_gamma**2 / n_g)
    if use_keff:
        dx = boxsize / 1024.0
        ke = k_eff_1d(k, dx)
    else:
        ke = k
    Ntt = sigma_v1d**2 * ke**2 / ((a * Hz)**2 * n_g)

    Nmodes = nmodes_box(k, dk, boxsize)

    sig_Pgg = gaussian_sigma_auto(Pgg + Ngg, Nmodes)
    sig_PEE = gaussian_sigma_auto(PEE + Nee, Nmodes)
    sig_Ptt = gaussian_sigma_auto(Ptt + Ntt, Nmodes)
    sig_PgE = gaussian_sigma_cross(Pgg + Ngg, PEE + Nee, PgE, Nmodes)
    sig_Pgt = gaussian_sigma_cross(Pgg + Ngg, Ptt + Ntt, Pgt, Nmodes)
    sig_PtE = gaussian_sigma_cross(Ptt + Ntt, PEE + Nee, PtE, Nmodes)

    return {
        'k': k,
        'z': float(z),
        'a': float(a),
        'D': np.asarray(D),
        'f': np.asarray(f),
        'F_IA': np.asarray(Fia),
        'Pmm': Pmm,
        'Pgg': Pgg,
        'PgE': PgE,
        'PEE': PEE,
        'Ptt': Ptt,
        'Pgt': Pgt,
        'PtE': PtE,
        'Ngg': Ngg,
        'Nee': Nee,
        'Ntt': Ntt,
        'Nmodes': Nmodes,
        'sigma_Pgg': sig_Pgg,
        'sigma_PgE': sig_PgE,
        'sigma_PEE': sig_PEE,
        'sigma_Ptt': sig_Ptt,
        'sigma_Pgt': sig_Pgt,
        'sigma_PtE': sig_PtE,
    }


if __name__ == '__main__':
    cosmo_dict = {
        'Omega0': 0.3089,
        'OmegaBaryon': 0.0486,
        'HubbleParam': 0.6774,
        'sigma8': 0.8159,
        'n_s': 0.9667,
    }
    k = np.logspace(np.log10(0.03), np.log10(3.0), 30)
    dk = np.empty_like(k)
    dk[1:-1] = 0.5 * (k[2:] - k[:-2])
    dk[0] = k[1] - k[0]
    dk[-1] = k[-1] - k[-2]
    out = compute_nla_spectra_and_errors(
        k=k,
        z=0.5,
        cosmo_dict=cosmo_dict,
        bias=2.0,
        n_g=1.0e-4,
        boxsize=205.0,
        dk=dk,
        A_IA=1.0,
    )
    print('f(z)=', out['f'])
    print('Pgg[:3]=', out['Pgg'][:3])
