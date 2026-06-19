#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orbit_nfw.py

Planar orbit integration for a test subhalo in a spherical NFW host.

Returns
-------
- r(t), phi(t) in ckpc/h
- velocity v(t) in km/s
- host potential Phi(t) in (km/s)^2 with Phi(infty)=0
- tidal tensor T_ij(t) in inertial Cartesian frame (3x3)
- tidal tensor T_ab(t) in the natural co-rotating basis (e_r, e_phi, e_z) (3x3)
- optional Jacobi stripping diagnostics: r_t(t), M(t), dM/dt
- optional subhalo boundary size: r_sub(t)

User-facing units
-----------------
- Length:  ckpc/h
- Mass:    1e10 Msun/h
- Time:    Gyr
- Speed:   km/s

Implementation note
-------------------
The orbit state is integrated directly in user-facing units:
    y = [x, y, vx, vy]
with x,y in ckpc/h and vx,vy in km/s.

Only when evaluating the host NFW potential / force / density do we convert:
    r_phys[kpc] = r[ckpc/h] * (a/h)
and
    M[Msun] = M[1e10 Msun/h] * 1e10 / h

Backward compatibility
----------------------
The main calling pattern is preserved:
    OrbitSimulator(...).run(E=..., L=..., r0=..., ...)

A new optional keyword is added:
    gamma_p : float or None
If None, gamma_p defaults to alpha_sub.

Meaning of r_sub
----------------
Assume the subhalo density follows a power law
    rho_sub(r) = A r^{-gamma_p},   0 < gamma_p < 3

We define the subhalo boundary by local density matching:
    rho_sub(r_sub) = rho_host(R)

Using the analytic mass integral for a power law,
    M(<r_sub) = 4 pi A r_sub^{3-gamma_p} / (3-gamma_p),
one obtains the algebraic boundary radius
    r_sub = [ (3-gamma_p) M / (4 pi rho_host) ]^{1/3}

This is evaluated at t=0 from the input initial position and mass, and at later
times using the current bound mass M(t) and the host density at the current orbit
radius R(t).

Dependencies
------------
- numpy
- pyccl
- tqdm
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import math
import pyccl as ccl

try:
    from tqdm import tqdm
except Exception as exc:
    raise RuntimeError("This module requires tqdm. Install with: pip install tqdm") from exc


# ======================================================================
# Constants
# ======================================================================

# Newton's constant in: kpc * (km/s)^2 / Msun
G_KPC_KMS2_PER_MSUN = 4.30091727003628e-6

KPC_PER_MPC = 1.0e3
SEC_PER_GYR = 1.0e9 * 365.25 * 24.0 * 3600.0
KM_PER_KPC = 3.085677581491367e16  # km

# 1 (km/s) = KPC_PER_GYR_PER_KMS (kpc/Gyr)
KPC_PER_GYR_PER_KMS = SEC_PER_GYR / KM_PER_KPC


# ======================================================================
# Small conversion helpers (kept for compatibility / convenience)
# ======================================================================

def ckpc_h_to_kpc_phys(x_ckpc_h, h: float, a: float):
    """Convert ckpc/h -> physical kpc."""
    return np.asarray(x_ckpc_h, dtype=float) * (a / h)


def kpc_phys_to_ckpc_h(x_kpc_phys, h: float, a: float):
    """Convert physical kpc -> ckpc/h."""
    return np.asarray(x_kpc_phys, dtype=float) * (h / a)


def soften_ckpc_h_to_kpc_phys(soften: float, h: float, a: float) -> float:
    """Convert softening from ckpc/h to physical kpc."""
    return float(soften) * (a / h)


def mass_1e10_msun_h_to_msun(m_1e10_msun_h: float, h: float) -> float:
    """Convert 1e10 Msun/h -> Msun."""
    return float(m_1e10_msun_h) * 1.0e10 / h


def mass_msun_to_1e10_msun_h(m_msun: float, h: float) -> float:
    """Convert Msun -> 1e10 Msun/h."""
    return float(m_msun) * h / 1.0e10


# ======================================================================
# NFW host
# ======================================================================

class NFWHost:
    """
    Spherical NFW host with Phi(infty)=0.

    Parameters
    ----------
    cosmo : ccl.Cosmology
    M200c : float
        Host mass in units of 1e10 Msun/h.
    c : float
        NFW concentration.
    z : float
        Redshift.
    """

    def __init__(self, cosmo: ccl.Cosmology, M200c: float, c: float, z: float = 0.0):
        self.cosmo = cosmo
        self.z = float(z)
        self.a = 1.0 / (1.0 + self.z)
        self.h = float(cosmo["h"])

        self.M200_msun = mass_1e10_msun_h_to_msun(M200c, self.h)
        self.c = float(c)

        # rho_crit(z): Msun/Mpc^3 -> Msun/kpc^3
        rho_crit_msun_mpc3 = float(ccl.rho_x(self.cosmo, self.a, "critical"))
        rho_crit_msun_kpc3 = rho_crit_msun_mpc3 / (KPC_PER_MPC ** 3)

        self.r200_kpc = (3.0 * self.M200_msun / (4.0 * np.pi * 200.0 * rho_crit_msun_kpc3)) ** (1.0 / 3.0)
        self.rs_kpc = self.r200_kpc / self.c

        self.f_c = np.log(1.0 + self.c) - self.c / (1.0 + self.c)
        self.rho_s_msun_kpc3 = self.M200_msun / (4.0 * np.pi * self.rs_kpc ** 3 * self.f_c)

    def rho(self, r_kpc: float) -> float:
        """NFW density in Msun/kpc^3."""
        x = max(float(r_kpc) / self.rs_kpc, 1e-30)
        return float(self.rho_s_msun_kpc3 / (x * (1.0 + x) ** 2))

    def menc(self, r_kpc: float) -> float:
        """Enclosed mass in Msun."""
        x = max(float(r_kpc) / self.rs_kpc, 0.0)
        f = np.log(1.0 + x) - x / (1.0 + x)
        return float(4.0 * np.pi * self.rho_s_msun_kpc3 * self.rs_kpc ** 3 * f)

    @staticmethod
    def _u_soft(r_kpc: float, soften_kpc: float) -> float:
        """Softened radius u = sqrt(r^2 + eps^2)."""
        r = max(float(r_kpc), 0.0)
        eps = float(soften_kpc)
        if eps <= 0.0:
            return max(r, 1e-30)
        return float(np.sqrt(r * r + eps * eps))

    def phi(self, r_kpc: float, soften_kpc: float = 0.0) -> float:
        """Potential Phi(r) in (km/s)^2."""
        u = self._u_soft(r_kpc, soften_kpc)
        x = max(u / self.rs_kpc, 1e-30)
        pref = 4.0 * np.pi * G_KPC_KMS2_PER_MSUN * self.rho_s_msun_kpc3 * self.rs_kpc ** 3
        return float(-pref * np.log(1.0 + x) / max(u, 1e-30))

    def dphi_dr(self, r_kpc: float, soften_kpc: float = 0.0) -> float:
        """First radial derivative dPhi/dr in (km/s)^2/kpc."""
        r = float(r_kpc)
        u = self._u_soft(r, soften_kpc)

        rs = self.rs_kpc
        A = 4.0 * np.pi * G_KPC_KMS2_PER_MSUN * self.rho_s_msun_kpc3 * rs ** 3

        f = np.log(1.0 + u / rs)
        term1 = 1.0 / (max(u, 1e-30) * max(u + rs, 1e-30))
        term2 = f / max(u, 1e-30) ** 2
        dphi_du = -A * (term1 - term2)

        du_dr = (r / u) if u > 0.0 else 0.0
        return float(dphi_du * du_dr)

    def d2phi_dr2(self, r_kpc: float, soften_kpc: float = 0.0) -> float:
        """Second radial derivative d^2Phi/dr^2 in (km/s)^2/kpc^2."""
        r = float(r_kpc)
        eps = float(soften_kpc)
        u = self._u_soft(r, eps)

        rs = self.rs_kpc
        A = 4.0 * np.pi * G_KPC_KMS2_PER_MSUN * self.rho_s_msun_kpc3 * rs ** 3
        f = np.log(1.0 + u / rs)

        term1 = 1.0 / (max(u, 1e-30) * max(u + rs, 1e-30))
        term2 = f / max(u, 1e-30) ** 2
        dphi_du = -A * (term1 - term2)

        denom1 = max(u, 1e-30) ** 2 * max(u + rs, 1e-30) ** 2
        term1_p = -(2.0 * u + rs) / denom1
        term2_p = 1.0 / (max(u + rs, 1e-30) * max(u, 1e-30) ** 2) - 2.0 * f / max(u, 1e-30) ** 3
        d2phi_du2 = -A * (term1_p - term2_p)

        gprime = (eps ** 2) / max(u, 1e-30) ** 3 if eps > 0.0 else 0.0
        r2_over_u2 = (r ** 2) / max(u, 1e-30) ** 2
        return float(gprime * dphi_du + r2_over_u2 * d2phi_du2)

    def accel_cartesian(self, x_kpc: float, y_kpc: float, soften_kpc: float = 0.0) -> Tuple[float, float]:
        """
        Cartesian host acceleration in the orbit plane.

        Returns
        -------
        ax, ay : float
            Units: (km/s)^2 / kpc.
        """
        r = float(np.hypot(x_kpc, y_kpc))
        if r <= 0.0:
            return 0.0, 0.0
        d1 = self.dphi_dr(r, soften_kpc=soften_kpc)
        amag = -d1
        return float(amag * x_kpc / r), float(amag * y_kpc / r)

    def tidal_tensor_inertial_kpc(self, x_kpc: float, y_kpc: float, soften_kpc: float = 0.0) -> np.ndarray:
        """
        Inertial Cartesian tidal tensor T_ij = d_i d_j Phi.

        Units: (km/s)^2 / kpc^2
        """
        r = float(np.hypot(x_kpc, y_kpc))
        if r <= 0.0:
            return np.zeros((3, 3), dtype=float)

        d1 = self.dphi_dr(r, soften_kpc=soften_kpc)
        d2 = self.d2phi_dr2(r, soften_kpc=soften_kpc)

        rx = x_kpc / r
        ry = y_kpc / r
        rhat = np.array([rx, ry, 0.0], dtype=float)
        outer = np.outer(rhat, rhat)

        T = (d2 - d1 / r) * outer + (d1 / r) * np.eye(3, dtype=float)
        return T

    @staticmethod
    def basis_natural(phi: float) -> np.ndarray:
        """Natural basis E = [e_r, e_phi, e_z] in inertial Cartesian coordinates."""
        c = float(np.cos(phi))
        s = float(np.sin(phi))
        e_r = np.array([c, s, 0.0], dtype=float)
        e_phi = np.array([-s, c, 0.0], dtype=float)
        e_z = np.array([0.0, 0.0, 1.0], dtype=float)
        return np.column_stack([e_r, e_phi, e_z])

    def tidal_tensor_natural_kpc(self, x_kpc: float, y_kpc: float, phi: float, soften_kpc: float = 0.0) -> np.ndarray:
        """Tidal tensor in the natural basis (e_r, e_phi, e_z)."""
        Tin = self.tidal_tensor_inertial_kpc(x_kpc, y_kpc, soften_kpc=soften_kpc)
        E = self.basis_natural(phi)
        return E.T @ Tin @ E


# ======================================================================
# Dynamical friction
# ======================================================================

def chandrasekhar_df_accel_kpc(
    host: NFWHost,
    x_kpc: float,
    y_kpc: float,
    vx: float,
    vy: float,
    m_sub_1e10_msun_h: float,
    lnLambda: float = 3.0,
    soften_kpc: float = 0.0,
) -> Tuple[float, float]:
    """
    Approximate Chandrasekhar dynamical friction in an NFW background.

    Returns
    -------
    ax_df, ay_df : float
        Units: (km/s)^2 / kpc.
    """
    r = float(np.hypot(x_kpc, y_kpc))
    v = float(np.hypot(vx, vy))
    if r <= 0.0 or v <= 0.0 or m_sub_1e10_msun_h <= 0.0:
        return 0.0, 0.0

    r_eff = float(np.sqrt(r * r + soften_kpc * soften_kpc)) if soften_kpc > 0.0 else r

    rho = host.rho(r_eff)               # Msun/kpc^3
    Menc = host.menc(r_eff)             # Msun
    Vc2 = G_KPC_KMS2_PER_MSUN * Menc / max(r_eff, 1e-30)
    sigma = np.sqrt(max(Vc2, 1e-30)) / np.sqrt(2.0)

    X = v / (np.sqrt(2.0) * sigma)
    F = float(math.erf(X) - (2.0 * X / np.sqrt(np.pi)) * np.exp(-X * X))

    m_sub_msun = mass_1e10_msun_h_to_msun(m_sub_1e10_msun_h, host.h)

    amag = 4.0 * np.pi * (G_KPC_KMS2_PER_MSUN ** 2) * m_sub_msun * rho * lnLambda * F / max(v * v, 1e-30)
    return float(-amag * vx / v), float(-amag * vy / v)


# ======================================================================
# Result container
# ======================================================================

@dataclass
class OrbitResult:
    """
    Orbit time series and derived diagnostics.

    Core arrays have length N.

    Optional outputs
    ----------------
    r_t : ndarray or None
        Jacobi / Roche tidal radius in ckpc/h.
    M : ndarray or None
        Bound mass in 1e10 Msun/h.
    dMdt : ndarray or None
        Time derivative dM/dt in (1e10 Msun/h)/Gyr.
    r_sub : ndarray or None
        Subhalo boundary radius estimated from density matching in ckpc/h.

    Merge flag
    ----------
    merged : bool
        True if the integration stopped early due to merge / guard conditions.
    t_merge : float or None
        Time of early stop in Gyr.
    r_merge : float or None
        Radius threshold used, in ckpc/h.
    """
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    r: np.ndarray
    phi: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    Phi: np.ndarray
    T_in: np.ndarray
    T_na: np.ndarray

    r_t: Optional[np.ndarray] = None
    M: Optional[np.ndarray] = None
    dMdt: Optional[np.ndarray] = None
    r_sub: Optional[np.ndarray] = None

    merged: bool = False
    t_merge: Optional[float] = None
    r_merge: Optional[float] = None

    @property
    def v(self) -> np.ndarray:
        """Velocity vectors as (N,3) in km/s."""
        return np.column_stack([self.vx, self.vy, np.zeros_like(self.vx)])

    @property
    def pos(self) -> np.ndarray:
        """Position vectors as (N,3) in ckpc/h."""
        return np.column_stack([self.x, self.y, np.zeros_like(self.x)])

    @property
    def Trr(self) -> np.ndarray:
        return self.T_na[:, 0, 0]

    @property
    def Trphi(self) -> np.ndarray:
        return self.T_na[:, 0, 1]

    @property
    def Tphiphi(self) -> np.ndarray:
        return self.T_na[:, 1, 1]

    @property
    def Tzz(self) -> np.ndarray:
        return self.T_na[:, 2, 2]


# ======================================================================
# RK4
# ======================================================================

def _rk4_step(rhs, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """One classical RK4 step."""
    k1 = rhs(t, y)
    k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = rhs(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ======================================================================
# Public API
# ======================================================================

def default_tng_cosmology() -> ccl.Cosmology:
    """Default TNG-like cosmology."""
    Omega_m = 0.3089
    Omega_b = 0.0486
    Omega_c = Omega_m - Omega_b
    return ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=0.6774, n_s=0.9667, sigma8=0.8159)


class OrbitSimulator:
    """
    Integrate a planar orbit in a spherical NFW host and compute tidal tensors.

    Parameters
    ----------
    M200c : float
        Host M200c in units of 1e10 Msun/h.
    c : float
        NFW concentration.
    z : float
        Redshift for rho_crit(z).
    cosmo : ccl.Cosmology, optional
        If None, uses default_tng_cosmology().
    """

    def __init__(
        self,
        M200c: float,
        c: float,
        z: float = 0.0,
        cosmo: Optional[ccl.Cosmology] = None,
    ):
        self.cosmo = default_tng_cosmology() if cosmo is None else cosmo
        self.host = NFWHost(self.cosmo, M200c=M200c, c=c, z=z)

    @property
    def h(self) -> float:
        return self.host.h

    @property
    def a(self) -> float:
        return self.host.a

    def run(
        self,
        *,
        E: float,
        L: float,
        r0: float,
        phi0: float,
        vr_sign: int,
        t_end: float,
        dt: float,
        soften: float = 0.0,
        # merge safety
        r_merge: Optional[float] = None,
        v_merge: Optional[float] = None,
        # dynamical friction
        with_df: bool = False,
        m_sub: Optional[float] = None,
        lnLambda: float = 3.0,
        # tidal stripping
        with_strip: bool = False,
        alpha_sub: float = 2.0,
        r_trunc0: Optional[float] = None,
        A_floor: float = 1e-12,
        # new subhalo density slope for r_sub
        gamma_p: Optional[float] = None,
        # verbosity
        show_progress: bool = True,
        progress_desc: str = "Integrating orbit",
        progress_update_every: int = 1,
    ) -> OrbitResult:
        """
        Integrate the orbit.

        Parameters
        ----------
        E : float
            Specific orbital energy in (km/s)^2.
        L : float
            Specific angular momentum in (ckpc/h) * (km/s).
        r0 : float
            Initial radius in ckpc/h.
        phi0 : float
            Initial azimuth.
        vr_sign : int
            +1 for outgoing, -1 for infalling.
        t_end : float
            End time in Gyr.
        dt : float
            Time step in Gyr.
        soften : float
            Plummer-like softening length in ckpc/h.

        with_df : bool
            If True, include Chandrasekhar dynamical friction.
        m_sub : float or None
            Initial subhalo mass in 1e10 Msun/h. Required if with_df=True or with_strip=True.
        lnLambda : float
            Coulomb logarithm.

        with_strip : bool
            If True, apply the simple Jacobi stripping model.
        alpha_sub : float
            Power-law density slope used in the stripping closure,
            rho_sub ~ r^{-alpha_sub}, must satisfy 0 < alpha_sub < 3.
        r_trunc0 : float or None
            Reference truncation radius in ckpc/h. If None, it is set from the
            local density-matching estimate at t=0.
        A_floor : float
            Floor for D = Omega^2 - Phi'' in the Jacobi formula, in (km/s)^2/kpc^2.

        gamma_p : float or None
            Power-law slope used to estimate the subhalo boundary radius r_sub
            from local density matching. If None, gamma_p = alpha_sub.

        Notes
        -----
        - The calling pattern is intentionally kept compatible with the older code.
        - When with_df=True and with_strip=True, the DF term uses the instantaneous
          bound mass M(t), not the initial mass.
        - r_sub(t) is derived algebraically from M(t) and the host density at the
          current orbital radius.
        """
        if vr_sign not in (-1, +1):
            raise ValueError("vr_sign must be +1 or -1.")
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if t_end <= 0.0:
            raise ValueError("t_end must be positive.")
        if progress_update_every < 1:
            raise ValueError("progress_update_every must be >= 1.")
        if float(soften) < 0.0:
            raise ValueError("soften must be >= 0.")
        if with_df and (m_sub is None):
            raise ValueError("with_df=True requires m_sub.")
        if with_strip and (m_sub is None):
            raise ValueError("with_strip=True requires m_sub.")
        if with_strip and not (0.0 < float(alpha_sub) < 3.0):
            raise ValueError("with_strip=True requires 0 < alpha_sub < 3.")
        if float(A_floor) <= 0.0:
            raise ValueError("A_floor must be > 0.")
        gp = float(alpha_sub if gamma_p is None else gamma_p)
        if not (0.0 < gp < 3.0):
            raise ValueError("gamma_p must satisfy 0 < gamma_p < 3.")

        # Unit conversion scalars used only when calling the host model.
        len_fac = self.a / self.h      # kpc_phys per ckpc/h
        inv_len_fac = self.h / self.a  # ckpc/h per kpc_phys
        soften_kpc = float(soften) * len_fac

        # Default merge radius: 2*soften (or 0 if soften=0).
        if r_merge is None:
            r_merge = max(2.0 * float(soften), 0.0)
        r_merge = float(r_merge)
        if r_merge < 0.0:
            raise ValueError("r_merge must be >= 0 or None.")

        # Initial position in user units.
        phi0 = float(phi0)
        x0 = float(r0) * np.cos(phi0)
        y0 = float(r0) * np.sin(phi0)

        # Initial speed from (E, L, r0). Since both L and r are in the same length
        # unit family, vt0 = L/r0 is directly in km/s.
        r0_kpc = float(r0) * len_fac
        Phi0 = self.host.phi(r0_kpc, soften_kpc=soften_kpc)
        vt0 = float(L) / max(float(r0), 1e-30)

        vr0_sq = 2.0 * (float(E) - Phi0) - vt0 * vt0
        if vr0_sq < 0.0:
            raise ValueError(
                f"Inconsistent (E,L,r0): v_r^2 < 0. Got {vr0_sq:.6e} (km/s)^2.\n"
                "Adjust E and/or L and/or r0."
            )
        vr0 = float(vr_sign) * np.sqrt(vr0_sq)

        c0 = np.cos(phi0)
        s0 = np.sin(phi0)
        vx0 = vr0 * c0 - vt0 * s0
        vy0 = vr0 * s0 + vt0 * c0

        # State vector: x,y in ckpc/h; vx,vy in km/s.
        y = np.array([x0, y0, vx0, vy0], dtype=float)

        nstep = int(np.floor(t_end / dt)) + 1
        t = np.linspace(0.0, dt * (nstep - 1), nstep, dtype=float)

        # Outputs
        x = np.empty(nstep, dtype=float)
        y_out = np.empty(nstep, dtype=float)
        vx = np.empty(nstep, dtype=float)
        vy = np.empty(nstep, dtype=float)
        r = np.empty(nstep, dtype=float)
        phi = np.empty(nstep, dtype=float)
        Phi = np.empty(nstep, dtype=float)
        T_in = np.empty((nstep, 3, 3), dtype=float)
        T_na = np.empty((nstep, 3, 3), dtype=float)

        M_out = np.full(nstep, np.nan, dtype=float) if (m_sub is not None and (with_df or with_strip or gamma_p is not None)) else None
        r_t_out = np.full(nstep, np.nan, dtype=float) if with_strip else None
        r_sub_out = np.full(nstep, np.nan, dtype=float) if (m_sub is not None) else None

        m0 = float(m_sub) if m_sub is not None else np.nan

        def rsub_from_mass_pos(m_1e10: float, r_ckpc_h: float) -> float:
            """
            Estimate the subhalo boundary from local density matching:
                rho_sub(r_sub) = rho_host(R)

            For rho_sub ~ r^{-gamma_p}, this gives
                r_sub = [ (3-gamma_p) M / (4 pi rho_host) ]^{1/3}

            Returns r_sub in ckpc/h.
            """
            if not np.isfinite(m_1e10) or m_1e10 <= 0.0:
                return 0.0

            r_kpc = max(float(r_ckpc_h) * len_fac, 1e-30)
            r_eff_kpc = np.sqrt(r_kpc * r_kpc + soften_kpc * soften_kpc) if soften_kpc > 0.0 else r_kpc

            rho_bg = self.host.rho(r_eff_kpc)                     # Msun/kpc^3
            m_msun = m_1e10 * 1.0e10 / self.h                    # Msun
            rsub_kpc = ((3.0 - gp) * m_msun / (4.0 * np.pi * max(rho_bg, 1e-300))) ** (1.0 / 3.0)
            return rsub_kpc * inv_len_fac

        # Initial subhalo boundary estimate from the initial location and mass.
        if m_sub is not None:
            r_sub0 = float(r_trunc0) if (r_trunc0 is not None) else rsub_from_mass_pos(m0, float(r0))
        else:
            r_sub0 = np.nan

        def mass_rt_rsub_from_state(x_now: float, y_now: float) -> Tuple[float, float, float]:
            """
            Return (m_now, r_t_now, r_sub_now) in user-facing units:
                m_now  : 1e10 Msun/h
                r_t_now: ckpc/h
                r_sub_now: ckpc/h

            - m_now comes from the stripping model if enabled, otherwise m0.
            - r_t_now is the Jacobi / Roche radius if stripping is enabled, else NaN.
            - r_sub_now is always the density-matching boundary if m_sub is provided.
            """
            rr = float(np.hypot(x_now, y_now))
            rr_kpc = max(rr * len_fac, 1e-30)
            r_eff_kpc = np.sqrt(rr_kpc * rr_kpc + soften_kpc * soften_kpc) if soften_kpc > 0.0 else rr_kpc

            m_now = float(m0)
            r_t_now = float("nan")

            if with_strip:
                # Omega = L_phys / r_phys^2
                #       = (L*len_fac) / (r^2 * len_fac^2)
                #       = L / (r^2 * len_fac)
                Omega = float(L) / (max(rr, 1e-30) ** 2 * len_fac)  # km/s/kpc
                phi2 = self.host.d2phi_dr2(r_eff_kpc, soften_kpc=soften_kpc)
                D = max(Omega * Omega - phi2, float(A_floor))

                m0_msun = m0 * 1.0e10 / self.h
                r_ref_kpc = max(r_sub0 * len_fac, 1e-30)

                # Fixed-mass Jacobi radius
                r_t_fix_kpc = (G_KPC_KMS2_PER_MSUN * m0_msun / D) ** (1.0 / 3.0)

                if r_t_fix_kpc >= r_ref_kpc:
                    m_now = float(m0)
                    r_t_now = r_t_fix_kpc * inv_len_fac
                else:
                    alpha = float(alpha_sub)
                    gamma = 3.0 - alpha
                    r_t_kpc = (G_KPC_KMS2_PER_MSUN * m0_msun / D) ** (1.0 / alpha) * (r_ref_kpc ** (-gamma / alpha))
                    r_t_now = r_t_kpc * inv_len_fac
                    m_now = float(m0) * (r_t_now / max(r_sub0, 1e-30)) ** gamma

            r_sub_now = rsub_from_mass_pos(m_now, rr) if np.isfinite(m_now) and m_now > 0.0 else 0.0
            return float(m_now), float(r_t_now), float(r_sub_now)

        def rhs(t_now: float, yy: np.ndarray) -> np.ndarray:
            xx, yy_pos, vvx, vvy = float(yy[0]), float(yy[1]), float(yy[2]), float(yy[3])

            x_kpc = xx * len_fac
            y_kpc = yy_pos * len_fac

            ax_g, ay_g = self.host.accel_cartesian(x_kpc, y_kpc, soften_kpc=soften_kpc)

            ax_df, ay_df = 0.0, 0.0
            if with_df:
                m_now, _, _ = mass_rt_rsub_from_state(xx, yy_pos)
                ax_df, ay_df = chandrasekhar_df_accel_kpc(
                    host=self.host,
                    x_kpc=x_kpc,
                    y_kpc=y_kpc,
                    vx=vvx,
                    vy=vvy,
                    m_sub_1e10_msun_h=m_now,
                    lnLambda=lnLambda,
                    soften_kpc=soften_kpc,
                )

            ax = ax_g + ax_df
            ay = ay_g + ay_df

            # x,y are in ckpc/h
            dxdt = vvx * KPC_PER_GYR_PER_KMS * inv_len_fac
            dydt = vvy * KPC_PER_GYR_PER_KMS * inv_len_fac

            # vx,vy are in km/s
            dvxdt = ax * KPC_PER_GYR_PER_KMS
            dvydt = ay * KPC_PER_GYR_PER_KMS

            return np.array([dxdt, dydt, dvxdt, dvydt], dtype=float)

        # Store i=0
        xx, yy_pos, vvx, vvy = y
        x[0], y_out[0], vx[0], vy[0] = xx, yy_pos, vvx, vvy
        rr0 = float(np.hypot(xx, yy_pos))
        r[0] = rr0
        ph0 = float(np.arctan2(yy_pos, xx))
        phi[0] = ph0

        rr0_kpc = rr0 * len_fac
        Phi[0] = self.host.phi(rr0_kpc, soften_kpc=soften_kpc)
        Tin0_kpc = self.host.tidal_tensor_inertial_kpc(xx * len_fac, yy_pos * len_fac, soften_kpc=soften_kpc)
        Tna0_kpc = self.host.tidal_tensor_natural_kpc(xx * len_fac, yy_pos * len_fac, ph0, soften_kpc=soften_kpc)

        conv_T = len_fac * len_fac  # d^2/d(ckpc/h)^2 = (a/h)^2 d^2/d(kpc)^2
        T_in[0] = Tin0_kpc * conv_T
        T_na[0] = Tna0_kpc * conv_T

        if M_out is not None or r_t_out is not None or r_sub_out is not None:
            m_now, r_t_now, r_sub_now = mass_rt_rsub_from_state(xx, yy_pos)
            if M_out is not None:
                M_out[0] = m_now
            if r_t_out is not None:
                r_t_out[0] = r_t_now
            if r_sub_out is not None:
                r_sub_out[0] = r_sub_now

        merged = False
        t_merge = None
        last_index = 0

        # Integrate
        pbar = tqdm(
            range(nstep - 1),
            desc=progress_desc,
            disable=(not show_progress),
            mininterval=0.2,
            miniters=progress_update_every,
        )

        for i in pbar:
            rr = float(np.hypot(y[0], y[1]))
            vv = float(np.hypot(y[2], y[3]))

            if r_merge > 0.0 and rr <= r_merge:
                merged = True
                t_merge = float(t[i])
                last_index = i
                break
            if (v_merge is not None) and (vv >= float(v_merge)):
                merged = True
                t_merge = float(t[i])
                last_index = i
                break

            y = _rk4_step(rhs, t[i], y, dt)
            j = i + 1
            last_index = j

            xx, yy_pos, vvx, vvy = y
            x[j], y_out[j], vx[j], vy[j] = xx, yy_pos, vvx, vvy

            rr = float(np.hypot(xx, yy_pos))
            r[j] = rr
            ph = float(np.arctan2(yy_pos, xx))
            phi[j] = ph

            rr_kpc = rr * len_fac
            Phi[j] = self.host.phi(rr_kpc, soften_kpc=soften_kpc)
            Tin_kpc = self.host.tidal_tensor_inertial_kpc(xx * len_fac, yy_pos * len_fac, soften_kpc=soften_kpc)
            Tna_kpc = self.host.tidal_tensor_natural_kpc(xx * len_fac, yy_pos * len_fac, ph, soften_kpc=soften_kpc)
            T_in[j] = Tin_kpc * conv_T
            T_na[j] = Tna_kpc * conv_T

            if M_out is not None or r_t_out is not None or r_sub_out is not None:
                m_now, r_t_now, r_sub_now = mass_rt_rsub_from_state(xx, yy_pos)

                # Enforce irreversible stripping / shrinking when applicable.
                if M_out is not None and j > 0 and np.isfinite(M_out[j - 1]):
                    m_now = min(m_now, M_out[j - 1])
                if r_t_out is not None and j > 0 and np.isfinite(r_t_out[j - 1]):
                    r_t_now = min(r_t_now, r_t_out[j - 1])
                if r_sub_out is not None:
                    # Recompute r_sub using the post-limited bound mass for better consistency.
                    r_sub_now = rsub_from_mass_pos(m_now, rr)
                    if j > 0 and np.isfinite(r_sub_out[j - 1]):
                        r_sub_now = min(r_sub_now, r_sub_out[j - 1])

                if M_out is not None:
                    M_out[j] = m_now
                if r_t_out is not None:
                    r_t_out[j] = r_t_now
                if r_sub_out is not None:
                    r_sub_out[j] = r_sub_now

                if M_out is not None and m_now <= 0.0:
                    merged = True
                    t_merge = float(t[j])
                    last_index = j
                    break

        # Truncate to actual length
        N = last_index + 1
        t = t[:N]
        x = x[:N]
        y_out = y_out[:N]
        vx = vx[:N]
        vy = vy[:N]
        r = r[:N]
        phi = phi[:N]
        Phi = Phi[:N]
        T_in = T_in[:N]
        T_na = T_na[:N]

        if M_out is not None:
            M_out = M_out[:N]
        if r_t_out is not None:
            r_t_out = r_t_out[:N]
        if r_sub_out is not None:
            r_sub_out = r_sub_out[:N]

        dMdt_out = None
        if M_out is not None:
            if N >= 3:
                dMdt_out = np.gradient(M_out, t)
            elif N == 2:
                slope = (M_out[1] - M_out[0]) / (t[1] - t[0])
                dMdt_out = np.array([slope, slope], dtype=float)
            else:
                dMdt_out = np.zeros_like(M_out)

        return OrbitResult(
            t=t,
            x=x,
            y=y_out,
            r=r,
            phi=phi,
            vx=vx,
            vy=vy,
            Phi=Phi,
            T_in=T_in,
            T_na=T_na,
            r_t=r_t_out,
            M=M_out,
            dMdt=dMdt_out,
            r_sub=r_sub_out,
            merged=bool(merged),
            t_merge=float(t_merge) if t_merge is not None else None,
            r_merge=float(r_merge) if r_merge is not None else None,
        )
