"""
Iana.py
------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
import h5py
import os
from tqdm import tqdm
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
from numba import njit,set_num_threads
from illustris_python import groupcat, snapshot
from typing import Optional, Tuple, Dict, Any, Union, Sequence





ArrayLike = Union[np.ndarray, Sequence[float]]

try:
    from .shape import *
except:
    from ia_analysis.shapes.shape import *

import pyccl as ccl




def Gfac(q: float, s: float, eps: float = 1e-15):
    """
    Compute shape factors G_i (i in {a,b,c}) for a triaxial ellipsoid using
    axis ratios:
        q = b/a,  s = c/a    (NOTE: s is c/a)
    with the convention a = 1, b = q, c = s.

    G_i = sum_{j≠i} (a_i^2 a_j^2) / (a_i^2 - a_j^2)^2

    Returns
    -------
    dict: {"a": Ga, "b": Gb, "c": Gc}

    Caveat
    ------
    Diverges / becomes ill-conditioned when axes are nearly degenerate:
      q≈1 (b≈a), s≈q (c≈b), or s≈1 (c≈a).
    """
    if q <= 0 or s <= 0:
        raise ValueError("q and s must be positive.")
    if q > 1.0 + 1e-12 or s > 1.0 + 1e-12:
        raise ValueError("Typically expect 0 < s <= q <= 1 (a>=b>=c convention).")

    a = 1.0
    b = float(q)
    c = float(s)

    axes = {"a": a, "b": b, "c": c}
    a2 = {k: v * v for k, v in axes.items()}

    G = {}
    for i in ("a", "b", "c"):
        acc = 0.0
        for j in ("a", "b", "c"):
            if j == i:
                continue
            den = (a2[i] - a2[j]) ** 2
            if den < eps:
                raise ValueError(
                    f"Axis degeneracy: (a_{i}^2-a_{j}^2)^2 too small "
                    f"(q={q:g}, s={s:g}). G diverges."
                )
            acc += (a2[i] * a2[j]) / den
        G[i] = acc
    return G


def relerr(gamma: float, N: int) -> float:
    """
    Relative error on each semi-axis (same for i=a,b,c):
        sigma(a_i)/a_i = (5-gamma)/sqrt(5(7-gamma)(3-gamma)) * 1/sqrt(N)
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if (7.0 - gamma) <= 0.0 or (3.0 - gamma) <= 0.0:
        raise ValueError("Need gamma < 3 (and < 7) for this expression.")
    denom = 5.0 * (7.0 - gamma) * (3.0 - gamma)
    return (5.0 - gamma) / np.sqrt(denom) / np.sqrt(float(N))


def varcos(gamma: float, N: int, q: float, s: float, eps: float = 1e-15):
    """
    Variance of cosine misalignment for each principal axis i:
        Var(cos θ_i) ≃ (9/50) * (5-gamma)^4 / [(7-gamma)^2 (3-gamma)^2] * G_i^2 / N^2

    Inputs
    ------
    gamma : float
    N : int
    q : float
        q = b/a
    s : float
        s = c/a   (NOTE corrected)
    eps : float
        degeneracy guard for G_i denominators

    Returns
    -------
    dict: {"a": Var(cosθ_a), "b": ..., "c": ...}
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if (7.0 - gamma) <= 0.0 or (3.0 - gamma) <= 0.0:
        raise ValueError("Need gamma < 3 (and < 7) for this expression.")

    G = Gfac(q=q, s=s, eps=eps)

    pref = (9.0 / 50.0) * (5.0 - gamma) ** 4 / ((7.0 - gamma) ** 2 * (3.0 - gamma) ** 2)
    N2 = float(N) ** 2
    return {k: pref * (G[k] ** 2) / N2 for k in ("a", "b", "c")}






def MatAssembly(components):
    """
    components: (N,6) with columns [Txx, Txy, Txz, Tyy, Tyz, Tzz]
    returns: (N,3,3) symmetric tensors
    """
    c = np.asarray(components, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 6:
        raise ValueError(f"Expected components shape (N,6), got {c.shape}")

    T = np.empty((c.shape[0], 3, 3), dtype=c.dtype)

    Txx, Txy, Txz, Tyy, Tyz, Tzz = (c[:, 0], c[:, 1], c[:, 2], c[:, 3], c[:, 4], c[:, 5])

    T[:, 0, 0] = Txx
    T[:, 0, 1] = Txy
    T[:, 0, 2] = Txz
    T[:, 1, 0] = Txy
    T[:, 1, 1] = Tyy
    T[:, 1, 2] = Tyz
    T[:, 2, 0] = Txz
    T[:, 2, 1] = Tyz
    T[:, 2, 2] = Tzz

    return T



def estimate_kappa_rot_from_subhalo(
    subhalo_spin,
    subhalo_mass_type,
    subhalo_halfmassrad_type,
    subhalo_veldisp,
    k_shape: float = 1.0,
    star_type: int = 4,
    min_radius: float = 1e-3,
) -> np.ndarray:
    """
    Estimate a proxy for the rotational support parameter kappa_rot for
    each subhalo using ONLY group catalog fields.

    This is NOT the particle-level definition used in the literature
    (which requires stellar particle phase-space information), but an
    approximate, calibration-friendly proxy based on:

        K_rot  ~ L^2 / (2 I_star)
        I_star ~ k_shape * M_star * R_star^2
        K_rand ~ (3/2) * M_star * sigma_1D^2
        kappa_rot = K_rot / (K_rot + K_rand)

    Parameters
    ----------
    subhalo_spin : array_like, shape (N, 3)
        `SubhaloSpin` array from the group catalog. Note that in TNG this
        is the total subhalo spin (all particle types), not stellar-only.
        We assume it is dominated by the baryonic component for galaxies.
    subhalo_mass_type : array_like, shape (N, 6)
        `SubhaloMassType` or equivalent; stellar mass is taken from the
        column `star_type` (default: 4).
    subhalo_halfmassrad_type : array_like, shape (N, 6)
        `SubhaloHalfmassRadType` or equivalent; the stellar half-mass
        radius is taken from column `star_type`.
    subhalo_veldisp : array_like, shape (N,)
        `SubhaloVelDisp` (1D velocity dispersion, e.g. from TNG), assumed
        to be representative of the stellar component.
    k_shape : float, optional
        Geometric factor in the moment of inertia approximation:
            I_star ~ k_shape * M_star * R_star^2
        Default is 1.0. You can tune this against a subsample with
        particle-level kappa_rot measurements.
    star_type : int, optional
        Index of the stellar component in the *_Type arrays. For TNG,
        stars are type 4.
    min_radius : float, optional
        Minimum stellar half-mass radius to consider [same units as input].
        Subhalos below this threshold are treated as R_star ~ min_radius
        to avoid division by zero.

    Returns
    -------
    kappa_rot : ndarray, shape (N,)
        Approximate kappa_rot proxy for each subhalo, in the range [0, 1].
        Values where the inputs are pathological (e.g. zero mass) are set
        to 0 by construction.

    Notes
    -----
    * This proxy should be calibrated/validated against true particle-level
      kappa_rot on a smaller sample before using hard thresholds in science
      analyses (e.g. disk vs spheroid cuts).
    * All input quantities must be expressed in mutually consistent units
      (e.g. Msun/h, kpc/h, km/s). The result is dimensionless.
    """

    # Convert inputs to arrays
    subhalo_spin = np.asarray(subhalo_spin)
    mass_type = np.asarray(subhalo_mass_type)
    halfmassrad_type = np.asarray(subhalo_halfmassrad_type)
    veldisp = np.asarray(subhalo_veldisp)

    # Stellar mass and half-mass radius
    M_star = mass_type[:, star_type]          # stellar mass
    R_star = halfmassrad_type[:, star_type]   # stellar half-mass radius

    # Enforce a minimum radius to avoid division by zero
    R_star_eff = np.where(R_star > min_radius, R_star, min_radius)

    # Total spin amplitude
    L = np.linalg.norm(subhalo_spin, axis=-1)

    # Approximate stellar moment of inertia: I_star ~ k_shape * M_star * R_star^2
    I_star = k_shape * M_star * R_star_eff**2

    # Rotational kinetic energy proxy: K_rot ~ L^2 / (2 I_star)
    # Where M_star or I_star are zero/very small, K_rot will be ~0.
    K_rot = np.zeros_like(M_star, dtype=float)
    valid_I = I_star > 0
    K_rot[valid_I] = 0.5 * (L[valid_I]**2) / I_star[valid_I]

    # Random kinetic energy from 1D velocity dispersion:
    # <v^2> = 3 sigma_1D^2, so K_rand = (1/2) M_star <v^2> = (3/2) M_star sigma^2
    sigma_1d = veldisp
    K_rand = 1.5 * M_star * sigma_1d**2

    # Total kinetic energy proxy
    K_tot = K_rot + K_rand

    # kappa_rot = K_rot / K_tot
    kappa_rot = np.zeros_like(M_star, dtype=float)
    valid_tot = K_tot > 0
    kappa_rot[valid_tot] = K_rot[valid_tot] / K_tot[valid_tot]

    # Clip to [0, 1] just in case of tiny numerical overshoots
    kappa_rot = np.clip(kappa_rot, 0.0, 1.0)

    return kappa_rot







def _normalize(v: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Return v / ||v|| with basic safety checks (supports last-axis vectors)."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(n <= eps):
        raise ValueError("Cannot normalize a zero (or near-zero) vector.")
    return v / n


def _is_axis_aligned(nhat: np.ndarray, atol: float = 1e-12) -> bool:
    """True if nhat is (approximately) one of ±x, ±y, ±z."""
    ax = int(np.argmax(np.abs(nhat)))
    return (np.isclose(abs(nhat[ax]), 1.0, atol=atol) and (np.sum(np.abs(nhat) > atol) == 1))


def _axis_aligned_basis(nhat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic sky basis for axis-aligned LOS.

    We choose e1,e2 to be aligned with the remaining Cartesian axes, with a sign
    choice that keeps (e1, e2, nhat) right-handed.
    """
    ax = int(np.argmax(np.abs(nhat)))
    sgn = 1.0 if nhat[ax] >= 0 else -1.0

    if ax == 2:  # ±z
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0]) if sgn > 0 else np.array([0.0, -1.0, 0.0])
    elif ax == 0:  # ±x
        e1 = np.array([0.0, 1.0, 0.0])
        e2 = np.array([0.0, 0.0, 1.0]) if sgn > 0 else np.array([0.0, 0.0, -1.0])
    else:  # ax == 1, ±y
        e1 = np.array([0.0, 0.0, 1.0])
        e2 = np.array([1.0, 0.0, 0.0]) if sgn > 0 else np.array([-1.0, 0.0, 0.0])

    return e1, e2


def _make_sky_basis(los: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an orthonormal basis (e1, e2) spanning the plane perpendicular to LOS.

    Parameters
    ----------
    los : ndarray, shape (3,) or (N,3)
        Line-of-sight direction(s). Need not be normalized.

    Returns
    -------
    e1, e2 : ndarrays
        Orthonormal basis vectors perpendicular to LOS.
        Shapes match los: (3,) for a single LOS, (N,3) for per-object LOS.
    """
    los = np.asarray(los, dtype=float)

    if los.ndim == 1:
        if los.shape != (3,):
            raise ValueError("los must have shape (3,) or (N,3)")
        nhat = _normalize(los)

        if _is_axis_aligned(nhat):
            return _axis_aligned_basis(nhat)

        # Generic LOS: pick a fixed reference vector not too parallel to nhat
        a = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(nhat, a)) > 0.9:
            a = np.array([0.0, 1.0, 0.0])

        e1 = np.cross(nhat, a)
        e1 = _normalize(e1)
        e2 = np.cross(nhat, e1)
        return e1, e2

    if los.ndim == 2:
        if los.shape[1] != 3:
            raise ValueError("los must have shape (3,) or (N,3)")
        nhat = _normalize(los)

        if np.all([_is_axis_aligned(nhat[i]) for i in range(nhat.shape[0])]):
            e1 = np.zeros_like(nhat)
            e2 = np.zeros_like(nhat)
            for i in range(nhat.shape[0]):
                e1[i], e2[i] = _axis_aligned_basis(nhat[i])
            return e1, e2

        # Generic per-row fallback
        a = np.tile(np.array([1.0, 0.0, 0.0]), (nhat.shape[0], 1))
        parallel = np.abs(np.sum(nhat * a, axis=1)) > 0.9
        a[parallel] = np.array([0.0, 1.0, 0.0])

        e1 = np.cross(nhat, a)
        e1 = _normalize(e1)
        e2 = np.cross(nhat, e1)
        return e1, e2

    raise ValueError("los must have shape (3,) or (N,3)")


def _coerce_I3D(I3D: np.ndarray) -> np.ndarray:
    """
    Coerce input shape tensor into an (N,3,3) float array.

    Accepted forms
    --------------
    - (3,3) : single tensor
    - (N,3,3) : batch tensors
    - (N,6) : symmetric packed ordering [Ixx,Ixy,Ixz,Iyy,Iyz,Izz]

    Notes
    -----
    The packed ordering is deliberately the same as ``MatAssembly`` in this
    file and as the common upper-triangular storage convention used in the
    shape-tide analysis scripts.  Older versions of this helper documented a
    different ordering; do not mix those files.
    """
    I3D = np.asarray(I3D, dtype=float)

    if I3D.ndim == 2 and I3D.shape == (3, 3):
        return I3D[None, :, :]

    if I3D.ndim == 3 and I3D.shape[1:] == (3, 3):
        return I3D

    if I3D.ndim == 2 and I3D.shape[1] == 6:
        out = np.zeros((I3D.shape[0], 3, 3), dtype=float)
        out[:, 0, 0] = I3D[:, 0]
        out[:, 0, 1] = out[:, 1, 0] = I3D[:, 1]
        out[:, 0, 2] = out[:, 2, 0] = I3D[:, 2]
        out[:, 1, 1] = I3D[:, 3]
        out[:, 1, 2] = out[:, 2, 1] = I3D[:, 4]
        out[:, 2, 2] = I3D[:, 5]
        return out

    raise ValueError("I3D must have shape (3,3), (N,3,3), or (N,6)")





# ============================================================
# 0) Low-level helpers (safe, vectorized, per-object NaN fallback)
# ============================================================

def _as_batch_vec(V):
    """Coerce V into (N,3). Accept (3,) or (N,3)."""
    V = np.asarray(V, dtype=np.float64)
    if V.ndim == 1 and V.shape == (3,):
        return V[None, :]
    if V.ndim == 2 and V.shape[1] == 3:
        return V
    raise ValueError(f"Expected vector shape (3,) or (N,3), got {V.shape}")

def _as_batch_mat(I):
    """Coerce I into (N,3,3). Accept (3,3) or (N,3,3)."""
    I = np.asarray(I, dtype=np.float64)
    if I.ndim == 2 and I.shape == (3, 3):
        return I[None, :, :]
    if I.ndim == 3 and I.shape[1:] == (3, 3):
        return I
    raise ValueError(f"Expected tensor shape (3,3) or (N,3,3), got {I.shape}")

def _cos_safe(V1, V2, eps=1e-30):
    """
    Safe cosine for (N,3) arrays. Returns (N,) with NaN where invalid.
    """
    V1 = _as_batch_vec(V1)
    V2 = _as_batch_vec(V2)
    if V1.shape[0] != V2.shape[0]:
        # broadcast not allowed here; per your usage it's a strict pairwise match
        return np.full((max(V1.shape[0], V2.shape[0]),), np.nan, dtype=np.float64)

    dot = np.sum(V1 * V2, axis=1)
    n1 = np.linalg.norm(V1, axis=1)
    n2 = np.linalg.norm(V2, axis=1)
    denom = n1 * n2

    out = np.full((V1.shape[0],), np.nan, dtype=np.float64)
    good = np.isfinite(dot) & np.isfinite(denom) & (denom > eps)
    out[good] = dot[good] / denom[good]
    return out

def _safe_eigh_batch(A, symmetrize=True, scale=True,
                     jitter_seq=(0.0, 1e-14, 1e-12, 1e-10)):
    """
    Safe batched eigh:
      - returns evals (N,3), evecs (N,3,3), ok (N,)
      - never raises; failures keep NaN and ok=False
    """
    A = np.asarray(A, dtype=np.float64)
    if symmetrize:
        A = 0.5 * (A + np.swapaxes(A, -1, -2))

    batch_shape = A.shape[:-2]
    evals = np.full(batch_shape + (3,), np.nan, dtype=np.float64)
    evecs = np.full(batch_shape + (3, 3), np.nan, dtype=np.float64)
    ok = np.zeros(batch_shape, dtype=bool)

    finite = np.isfinite(A).all(axis=(-1, -2))

    if scale:
        s = np.max(np.abs(A), axis=(-1, -2))
        s = np.where((s > 0) & np.isfinite(s), s, 1.0)
        Awork = A / s[..., None, None]
    else:
        s = np.ones(batch_shape, dtype=np.float64)
        Awork = A

    Aflat = Awork.reshape((-1, 3, 3))
    finite_flat = finite.reshape(-1)
    ok_flat = ok.reshape(-1)

    idx = np.flatnonzero(finite_flat)
    if idx.size > 0:
        try:
            w, v = np.linalg.eigh(Aflat[idx])           # w:(M,3) v:(M,3,3)
            sf = s.reshape(-1)[idx][:, None]            # (M,1) critical for broadcasting
            evals.reshape((-1, 3))[idx] = w * sf
            evecs.reshape((-1, 3, 3))[idx] = v
            ok_flat[idx] = True
        except np.linalg.LinAlgError:
            pass

    need = np.flatnonzero(finite_flat & (~ok_flat))
    if need.size > 0:
        I3 = np.eye(3, dtype=np.float64)
        sflat = s.reshape(-1)
        evals_flat = evals.reshape((-1, 3))
        evecs_flat = evecs.reshape((-1, 3, 3))
        for k in need:
            Ak = Aflat[k]
            sk = sflat[k]
            for jit in jitter_seq:
                try:
                    w, v = np.linalg.eigh(Ak + jit * I3)
                    evals_flat[k] = w * sk
                    evecs_flat[k] = v
                    ok_flat[k] = True
                    break
                except np.linalg.LinAlgError:
                    continue

    return evals, evecs, ok


def _proper_rotation_frames_batch(R, ok=None, eps=1e-12):
    """
    Force a batch of principal-axis frames into SO(3).

    Parameters
    ----------
    R : ndarray, shape (..., 3, 3)
        Frame matrices with basis vectors stored in columns.
    ok : ndarray or None
        Optional validity mask. Invalid frames are left as NaN.

    Returns
    -------
    Q : ndarray, shape (..., 3, 3)
        Right-handed frames with det(Q)=+1 where possible.
    ok_out : ndarray
        Updated validity mask.
    """
    Q = np.asarray(R, dtype=np.float64).copy()
    batch_shape = Q.shape[:-2]
    Qf = Q.reshape((-1, 3, 3))

    if ok is None:
        okf = np.ones(Qf.shape[0], dtype=bool)
    else:
        okf = np.asarray(ok, dtype=bool).reshape(-1).copy()

    for k in range(Qf.shape[0]):
        if not okf[k] or not np.all(np.isfinite(Qf[k])):
            Qf[k] = np.nan
            okf[k] = False
            continue

        try:
            for j in range(3):
                n = float(np.linalg.norm(Qf[k, :, j]))
                if (not np.isfinite(n)) or n < eps:
                    raise ValueError("degenerate frame")
                Qf[k, :, j] /= n

            if np.linalg.det(Qf[k]) < 0.0:
                Qf[k, :, 2] *= -1.0

            U, _, Vt = np.linalg.svd(Qf[k])
            Qk = U @ Vt
            if np.linalg.det(Qk) < 0.0:
                U[:, -1] *= -1.0
                Qk = U @ Vt

            Qf[k] = Qk
            okf[k] = bool(np.linalg.det(Qk) > 0.0)

        except Exception:
            Qf[k] = np.nan
            okf[k] = False

    return Qf.reshape(batch_shape + (3, 3)), okf.reshape(batch_shape)


def _eigh_sort_desc_safe(I, take_abs_evals=False):
    """
    I: (N,3,3)
    returns evals_sorted (N,3), evecs_sorted (N,3,3), ok (N,)
      - sorted by |eval| desc (default)
      - never raises; ok marks successful eigh
    """
    evals, evecs, ok = _safe_eigh_batch(I, symmetrize=True, scale=True)
    key = np.abs(evals) if take_abs_evals else evals
    idx = np.argsort(key, axis=-1)[:, ::-1]  # (N,3)
    evals_s = np.take_along_axis(evals, idx, axis=-1)
    evecs_s = np.take_along_axis(evecs, idx[:, None, :], axis=-1)  # reorder columns
    evecs_s, ok = _proper_rotation_frames_batch(evecs_s, ok=ok)
    return evals_s, evecs_s, ok

def _nan_dict(N):
    nan = np.full((N,), np.nan, dtype=np.float64)
    return {"major": nan.copy(), "medium": nan.copy(), "minor": nan.copy()}

# ============================================================
# 1) VV: cosine between N pairs of vectors
# ============================================================

def VV(v1, v2):
    """
    v1, v2: (N,3) or (3,)
    returns: (N,) cosine values; NaN for invalid rows.
    """
    try:
        return _cos_safe(v1, v2)
    except Exception:
        # total failure -> all NaN of inferred length
        try:
            N = _as_batch_vec(v1).shape[0]
        except Exception:
            N = _as_batch_vec(v2).shape[0] if np.asarray(v2).ndim else 1
        return np.full((N,), np.nan, dtype=np.float64)

# ============================================================
# 2) II: cosine between corresponding principal axes of two tensor sets
# ============================================================

def II(I1, I2):
    """
    I1, I2: (N,3,3) or (3,3)
    returns dict: {"major": (N,), "medium": (N,), "minor": (N,)}
    Any failure per-object -> NaN for that object.
    """
    try:
        A = _as_batch_mat(I1)
        B = _as_batch_mat(I2)
        if A.shape[0] != B.shape[0]:
            N = max(A.shape[0], B.shape[0])
            return _nan_dict(N)

        _, e1, ok1 = _eigh_sort_desc_safe(A)
        _, e2, ok2 = _eigh_sort_desc_safe(B)
        ok = ok1 & ok2

        cmaj = _cos_safe(e1[:, :, 0], e2[:, :, 0])
        cmed = _cos_safe(e1[:, :, 1], e2[:, :, 1])
        cmin = _cos_safe(e1[:, :, 2], e2[:, :, 2])

        # any failed eigens -> NaN
        bad = ~ok
        cmaj = cmaj.astype(np.float64); cmed = cmed.astype(np.float64); cmin = cmin.astype(np.float64)
        cmaj[bad] = np.nan; cmed[bad] = np.nan; cmin[bad] = np.nan

        return {"major": cmaj, "medium": cmed, "minor": cmin}
    except Exception:
        try:
            N = _as_batch_mat(I1).shape[0]
        except Exception:
            N = _as_batch_mat(I2).shape[0]
        return _nan_dict(N)

# ============================================================
# 3) VI: cosine between vectors and each principal axis of tensors
# ============================================================

def VI(V, I):
    """
    V: (N,3) or (3,)
    I: (N,3,3) or (3,3)
    returns dict: {"major": (N,), "medium": (N,), "minor": (N,)}
    Any failure per-object -> NaN for that object.
    """
    try:
        Vb = _as_batch_vec(V)
        Ib = _as_batch_mat(I)
        if Vb.shape[0] != Ib.shape[0]:
            N = max(Vb.shape[0], Ib.shape[0])
            return _nan_dict(N)

        _, e, ok = _eigh_sort_desc_safe(Ib)

        cmaj = _cos_safe(Vb, e[:, :, 0])
        cmed = _cos_safe(Vb, e[:, :, 1])
        cmin = _cos_safe(Vb, e[:, :, 2])

        bad = ~ok
        cmaj[bad] = np.nan; cmed[bad] = np.nan; cmin[bad] = np.nan
        return {"major": cmaj, "medium": cmed, "minor": cmin}
    except Exception:
        try:
            N = _as_batch_vec(V).shape[0]
        except Exception:
            N = _as_batch_mat(I).shape[0]
        return _nan_dict(N)

# ============================================================
# 4) omega_fig: figure rotation ω (and α if ddI given) in body frame
# ============================================================

def omega_fig(I, dI, ddI=None, dt=None,
              rel_gap_min=1e-10, abs_gap_min=0.0):
    """
    Estimate figure-rotation angular velocity ω_body (and angular acceleration α_body)
    from I, dI, ddI in the instantaneous principal-axis (body) frame.

    Inputs
    ------
    I, dI, ddI : (N,3,3) or (3,3)
        Tensor and its 1st/2nd time derivatives.
        If dt is provided, interpret dI and ddI as finite differences:
            dI -> dI/dt ; ddI -> ddI/dt^2
    Returns
    -------
    if ddI is None:
        omega_body : (N,3)
    else:
        omega_body : (N,3)
        alpha_body : (N,3)

    Policy
    ------
    Any per-object failure (eigh nonconvergence, NaNs, near-degenerate eigenvalues)
    -> corresponding row(s) set to NaN.
    """
    try:
        Ib  = _as_batch_mat(I)
        dIb = _as_batch_mat(dI)
        if Ib.shape[0] != dIb.shape[0]:
            N = max(Ib.shape[0], dIb.shape[0])
            out = np.full((N, 3), np.nan, dtype=np.float64)
            return (out, out.copy()) if ddI is not None else out

        if dt is not None:
            dt = float(dt)
            dIb = dIb / dt

        lam, R, ok_eigh = _eigh_sort_desc_safe(Ib)  # lam:(N,3), R:(N,3,3)

        Rt = np.swapaxes(R, -1, -2)
        K = Rt @ dIb @ R                       # K = R^T dI R

        denom = lam[:, None, :] - lam[:, :, None]  # (N,3,3) with denom[i,j]=λ_j-λ_i
        gap = np.abs(denom)
        g01, g02, g12 = gap[:, 0, 1], gap[:, 0, 2], gap[:, 1, 2]
        scale = np.maximum(1.0, np.max(np.abs(lam), axis=1))
        min_gap = np.minimum(np.minimum(g01, g02), g12)
        ok_gap = (min_gap >= abs_gap_min) & (min_gap >= rel_gap_min * scale)
        ok = ok_eigh & ok_gap & np.isfinite(K).all(axis=(1, 2)) & np.isfinite(lam).all(axis=1)

        # Solve Ω_ij = K_ij / (λ_j - λ_i) for i!=j
        Omega = np.zeros_like(K)
        mask = np.abs(denom) > (rel_gap_min * scale[:, None, None] + abs_gap_min)
        diag = np.eye(3, dtype=bool)
        mask[:, diag] = False
        Omega[mask] = K[mask] / denom[mask]
        Omega = 0.5 * (Omega - np.swapaxes(Omega, -1, -2))  # enforce antisym

        omega_body = np.stack([Omega[:, 2, 1], Omega[:, 0, 2], Omega[:, 1, 0]], axis=1)  # (N,3)

        # If only ω requested
        if ddI is None:
            omega_body[~ok] = np.nan
            return omega_body

        # With ddI: compute α via off-diagonal constraint on ddot(I)_body
        ddIb = _as_batch_mat(ddI)
        if ddIb.shape[0] != Ib.shape[0]:
            N = max(ddIb.shape[0], Ib.shape[0])
            out = np.full((N, 3), np.nan, dtype=np.float64)
            return out, out.copy()
        if dt is not None:
            ddIb = ddIb / (dt * dt)

        J = Rt @ ddIb @ R  # J = R^T ddI R

        # commutator [Ω, K] = ΩK - KΩ
        comm_OK = Omega @ K - K @ Omega

        # dotλ_i = K_ii
        dotlam = np.stack([K[:, 0, 0], K[:, 1, 1], K[:, 2, 2]], axis=1)
        dotdiff = dotlam[:, None, :] - dotlam[:, :, None]  # (N,3,3) with dotλ_j - dotλ_i

        # Solve dotΩ_ij:
        # 0 = (J - [Ω,K])_ij - (λ_j-λ_i) dotΩ_ij - Ω_ij (dotλ_j-dotλ_i)
        # => dotΩ_ij = ((J - [Ω,K])_ij - Ω_ij(dotλ_j-dotλ_i)) / (λ_j-λ_i)
        dotOmega = np.zeros_like(Omega)
        num = (J - comm_OK) - Omega * dotdiff
        dotOmega[mask] = num[mask] / denom[mask]
        dotOmega = 0.5 * (dotOmega - np.swapaxes(dotOmega, -1, -2))

        alpha_body = np.stack([dotOmega[:, 2, 1], dotOmega[:, 0, 2], dotOmega[:, 1, 0]], axis=1)

        bad = ~ok | (~np.isfinite(J).all(axis=(1, 2)))
        omega_body[bad] = np.nan
        alpha_body[bad] = np.nan
        return omega_body, alpha_body

    except Exception:
        # catastrophic failure: infer N as best as possible, return NaNs
        try:
            N = _as_batch_mat(I).shape[0]
        except Exception:
            N = _as_batch_mat(dI).shape[0]
        out = np.full((N, 3), np.nan, dtype=np.float64)
        return (out, out.copy()) if ddI is not None else out






def chiSO(I, rel_gap_min=1e-14):
    """
    Compute signed smooth shape parameter chi and axis ratios q=b/a, s=c/a
    from a symmetric 3x3 shape/inertia-like tensor.

    This strict version is physically consistent for shape/inertia-like tensors:
    eigenvalues are sorted by descending algebraic value (not absolute value),
    and any non-positive eigenvalue is treated as invalid.

    Parameters
    ----------
    I : (N,3,3) or (3,3) array_like
        Symmetric (or nearly symmetric) shape / inertia-like tensor.
    rel_gap_min : float, optional
        Relative eigenvalue-gap threshold used to reject nearly degenerate
        tensors. Such cases are treated as undefined and returned as NaN.

    Returns
    -------
    out : dict
        Dictionary with keys:
            "chi" : (N,) ndarray
                Smooth signed shape parameter:
                    chi = (1 - s) * (1 - 2T)
                where positive ~ oblate, negative ~ prolate.
            "q"   : (N,) ndarray
                Intermediate-to-major axis ratio, q = b/a.
            "s"   : (N,) ndarray
                Minor-to-major axis ratio, s = c/a.

        Any invalid object is assigned NaN.

    Notes
    -----
    - Eigenvalues are computed from the symmetrized tensor.
    - Eigenvalues are sorted by descending algebraic value:
          lambda1 >= lambda2 >= lambda3
    - Axis lengths are defined as:
          a = sqrt(lambda1), b = sqrt(lambda2), c = sqrt(lambda3)
      so all three eigenvalues must be strictly positive.
    - A valid object must satisfy:
        1. all tensor entries finite,
        2. eigendecomposition succeeds,
        3. eigenvalues sufficiently non-degenerate,
        4. lambda1 > 0, lambda2 > 0, lambda3 > 0,
        5. resulting axis ratios q and s are finite,
        6. triaxiality denominator (1 - s^2) != 0.
    - Near-spherical or degenerate cases are not regularized; they are NaN.
    """

    import numpy as np

    # ------------------------------------------------------------
    # Normalize input shape
    # ------------------------------------------------------------
    try:
        I = np.asarray(I, dtype=np.float64)
        if I.ndim == 2 and I.shape == (3, 3):
            I = I[None, :, :]
        if I.ndim != 3 or I.shape[1:] != (3, 3):
            raise ValueError
        N = I.shape[0]
    except Exception:
        nan = np.full((1,), np.nan, dtype=np.float64)
        return {"chi": nan.copy(), "q": nan.copy(), "s": nan.copy()}

    out = {
        "chi": np.full(N, np.nan, dtype=np.float64),
        "q":   np.full(N, np.nan, dtype=np.float64),
        "s":   np.full(N, np.nan, dtype=np.float64),
    }

    # ------------------------------------------------------------
    # Symmetrize and reject non-finite tensors
    # ------------------------------------------------------------
    A = 0.5 * (I + np.swapaxes(I, -1, -2))
    finite = np.isfinite(A).all(axis=(1, 2))
    if not np.any(finite):
        return out

    # ------------------------------------------------------------
    # Eigenvalues per object; failure of one object does not kill batch
    # ------------------------------------------------------------
    evals = np.full((N, 3), np.nan, dtype=np.float64)
    ok = np.zeros(N, dtype=bool)

    for i in np.flatnonzero(finite):
        try:
            w, _ = np.linalg.eigh(A[i])
            evals[i] = w
            ok[i] = True
        except Exception:
            ok[i] = False

    if not np.any(ok):
        return out

    # ------------------------------------------------------------
    # Sort by descending algebraic eigenvalue
    # ------------------------------------------------------------
    idx = np.argsort(evals, axis=1)[:, ::-1]
    w_sorted = np.take_along_axis(evals, idx, axis=1)

    # ------------------------------------------------------------
    # Reject non-finite and non-positive eigenvalues
    # ------------------------------------------------------------
    ok &= np.isfinite(w_sorted).all(axis=1)
    ok &= np.all(w_sorted > 0.0, axis=1)

    if not np.any(ok):
        return out

    # ------------------------------------------------------------
    # Reject nearly degenerate tensors
    # ------------------------------------------------------------
    scale = np.maximum(1.0, w_sorted[:, 0])
    g01 = np.abs(w_sorted[:, 0] - w_sorted[:, 1])
    g12 = np.abs(w_sorted[:, 1] - w_sorted[:, 2])
    g02 = np.abs(w_sorted[:, 0] - w_sorted[:, 2])
    min_gap = np.minimum(np.minimum(g01, g12), g02)

    ok &= (min_gap >= rel_gap_min * scale)

    if not np.any(ok):
        return out

    # ------------------------------------------------------------
    # Axis lengths
    # ------------------------------------------------------------
    a = np.full(N, np.nan, dtype=np.float64)
    b = np.full(N, np.nan, dtype=np.float64)
    c = np.full(N, np.nan, dtype=np.float64)

    a[ok] = np.sqrt(w_sorted[ok, 0])
    b[ok] = np.sqrt(w_sorted[ok, 1])
    c[ok] = np.sqrt(w_sorted[ok, 2])

    ok &= np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    ok &= (a > 0.0)

    if not np.any(ok):
        return out

    # ------------------------------------------------------------
    # Axis ratios
    # ------------------------------------------------------------
    q = np.full(N, np.nan, dtype=np.float64)
    s = np.full(N, np.nan, dtype=np.float64)

    q[ok] = b[ok] / a[ok]
    s[ok] = c[ok] / a[ok]

    valid_ratio = ok & np.isfinite(q) & np.isfinite(s)
    valid_ratio &= (q >= 0.0) & (q <= 1.0)
    valid_ratio &= (s >= 0.0) & (s <= 1.0)

    if not np.any(valid_ratio):
        return out

    out["q"] = q
    out["s"] = s

    # ------------------------------------------------------------
    # Triaxiality:
    #     T = (1 - q^2) / (1 - s^2)
    # Undefined if denominator == 0 -> NaN directly
    # ------------------------------------------------------------
    numer = 1.0 - q * q
    denom = 1.0 - s * s

    valid_T = valid_ratio & np.isfinite(numer) & np.isfinite(denom) & (denom != 0.0)
    if not np.any(valid_T):
        return out

    T = np.full(N, np.nan, dtype=np.float64)
    T[valid_T] = numer[valid_T] / denom[valid_T]

    valid_T &= np.isfinite(T)
    valid_T &= (T >= 0.0) & (T <= 1.0)

    if not np.any(valid_T):
        return out

    # ------------------------------------------------------------
    # Smooth signed shape parameter
    # ------------------------------------------------------------
    chi = np.full(N, np.nan, dtype=np.float64)
    chi[valid_T] = (1.0 - s[valid_T]) * (1.0 - 2.0 * T[valid_T])

    out["chi"] = chi
    return out







def epsilon_from_shape_matrix(
    I3D: np.ndarray,
    los: ArrayLike = (0.0, 0.0, 1.0),
    *,
    apply_responsivity: bool = True,
    responsivity: Optional[float] = None,
    e2_mean_per_component: Optional[float] = None,
    clip_R_min: float = 0.05,
    return_responsivity: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, float]]:
    """
    Project a 3D shape/inertia tensor into 2D ellipticity and (optionally) IA shear.

    Parameters
    ----------
    I3D : ndarray
        Shape/inertia tensor(s). Accepted shapes:
          - (3,3) or (N,3,3)
          - (N,6) packed symmetric: [Ixx,Iyy,Izz,Ixy,Ixz,Iyz]
        Only relative values matter: overall scaling cancels out.
    los : array-like
        Line of sight direction. Either:
          - (3,) for a global plane-parallel LOS (recommended for FFT-in-a-box), or
          - (N,3) for per-object LOS (use with care if you later do FFT E/B).
    apply_responsivity : bool
        If True (default), return gamma_{+,x} = epsilon_{+,x}/(2R).
        If False, return raw epsilon_{+,x}.
    responsivity : float, optional
        If provided, use this fixed R.
    e2_mean_per_component : float, optional
        Provide <epsilon_i^2>_i directly; then R = 1 - e2_mean_per_component.
    clip_R_min : float
        Safety floor to avoid blow-ups if R is tiny/negative due to pathological inputs.
    return_responsivity : bool
        If True, also return the scalar R used.

    Returns
    -------
    comp1, comp2 : (N,) arrays
        Either (epsilon_plus, epsilon_cross) or (gamma_plus, gamma_cross).
    R : float, optional
        Returned only if return_responsivity=True.
    """
    I = _coerce_I3D(I3D)  # (N,3,3)
    N = I.shape[0]

    los = np.asarray(los, dtype=float)
    if los.ndim == 1:
        e1, e2 = _make_sky_basis(los)
        e1 = np.tile(e1, (N, 1))
        e2 = np.tile(e2, (N, 1))
    elif los.ndim == 2:
        if los.shape[0] != N:
            raise ValueError("If los is (N,3), it must have the same N as I3D.")
        e1, e2 = _make_sky_basis(los)
    else:
        raise ValueError("los must have shape (3,) or (N,3)")

    # Project: Q_ab = e_a^T I e_b
    Q11 = np.einsum("ni,nij,nj->n", e1, I, e1)
    Q22 = np.einsum("ni,nij,nj->n", e2, I, e2)
    Q12 = np.einsum("ni,nij,nj->n", e1, I, e2)

    tr2 = Q11 + Q22
    bad = tr2 == 0.0
    if np.any(bad):
        raise ValueError("Projected 2D trace (Q11+Q22) contains zeros; cannot define ellipticity.")

    eps_plus = (Q11 - Q22) / tr2
    eps_cross = (2.0 * Q12) / tr2

    if not apply_responsivity:
        if return_responsivity:
            return eps_plus, eps_cross, np.nan
        return eps_plus, eps_cross

    # --- Responsivity calibration (Shi+2020-like) ---
    if responsivity is not None:
        R = float(responsivity)
    else:
        if e2_mean_per_component is not None:
            e2_pc = float(e2_mean_per_component)
        else:
            if N < 2:
                raise ValueError(
                    "Cannot estimate responsivity from a single object. "
                    "Provide responsivity=... or e2_mean_per_component=..."
                )
            good = np.isfinite(eps_plus) & np.isfinite(eps_cross)
            if np.sum(good) < 2:
                raise ValueError(
                    "Cannot estimate responsivity: not enough finite ellipticities. "
                    "Provide responsivity=... or e2_mean_per_component=..."
                )
            e2_pc = 0.5 * (np.mean(eps_plus[good] ** 2) + np.mean(eps_cross[good] ** 2))

        R = 1.0 - e2_pc

    # Safety
    if not np.isfinite(R):
        raise ValueError("Computed responsivity R is not finite.")
    if R < clip_R_min:
        R = clip_R_min

    gamma_plus = eps_plus / (2.0 * R)
    gamma_cross = eps_cross / (2.0 * R)

    if return_responsivity:
        return gamma_plus, gamma_cross, float(R)
    return gamma_plus, gamma_cross

    
def epsilon_from_spin(
    spin: np.ndarray,
    los: np.ndarray,
    tiny: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute projected ellipticity (epsilon_plus, epsilon_cross) from
    the total angular momentum (spin) vector of each galaxy, following
    a thin-disk approximation (as used e.g. in Shi+2020).

    Convention
    ----------
    Treat the spin direction as the disk normal. In a sky-plane basis (e1,e2)
    perpendicular to the line of sight n_hat, define components of L_hat as:
        L1 = L_hat ⋅ e1
        L2 = L_hat ⋅ e2
        L3 = L_hat ⋅ n_hat

    Then
        epsilon_+ = -(L1^2 - L2^2) / (1 + L3^2)
        epsilon_x = -2 L1 L2 / (1 + L3^2)

    Parameters
    ----------
    spin : ndarray, shape (N,3)
        Spin vectors. Only the direction matters.
    los : ndarray, shape (3,) or (N,3)
        Line-of-sight direction(s).
    tiny : float
        Small floor to treat nearly-zero spin safely.

    Returns
    -------
    eps_plus, eps_cross : ndarrays, shape (N,)
    """
    spin = np.asarray(spin, dtype=float)
    if spin.ndim != 2 or spin.shape[1] != 3:
        raise ValueError("spin must have shape (N, 3)")
    N = spin.shape[0]

    los = np.asarray(los, dtype=float)
    if los.shape == (3,):
        los = np.broadcast_to(los, (N, 3))
    elif los.shape == (N, 3):
        pass
    else:
        raise ValueError("los must have shape (3,) or (N, 3)")

    # Build sky-plane basis using the existing helper (returns e1,e2),
    # and normalize LOS to get n_hat.
    n_norm = np.linalg.norm(los, axis=1, keepdims=True)
    if np.any(n_norm <= 0.0):
        raise ValueError("los contains zero-norm vectors.")
    n_hat = los / n_norm
    e1, e2 = _make_sky_basis(n_hat)  # uses your previously-defined helper

    # Normalize spin to get L_hat
    L_norm = np.linalg.norm(spin, axis=1, keepdims=True)
    # Where spin is ~zero, set L_hat to 0 and return eps=0
    safe = (L_norm[:, 0] > tiny)
    L_hat = np.zeros_like(spin)
    L_hat[safe] = spin[safe] / L_norm[safe]

    # Components in the {e1, e2, n_hat} basis
    L1 = np.einsum("ni,ni->n", L_hat, e1)
    L2 = np.einsum("ni,ni->n", L_hat, e2)
    L3 = np.einsum("ni,ni->n", L_hat, n_hat)

    eps_plus = np.zeros(N, dtype=float)
    eps_cross = np.zeros(N, dtype=float)

    denom = 1.0 + L3[safe] ** 2
    eps_plus[safe] = -(L1[safe] ** 2 - L2[safe] ** 2) / denom
    eps_cross[safe] = -2.0 * L1[safe] * L2[safe] / denom

    # Clip |epsilon| <= 1 (thin-disk idealization)
    eps_abs = np.sqrt(eps_plus**2 + eps_cross**2)
    too_big = eps_abs > 1.0 + 1e-6
    if np.any(too_big):
        eps_plus[too_big] /= eps_abs[too_big]
        eps_cross[too_big] /= eps_abs[too_big]

    return eps_plus, eps_cross
    
def r2s(
    cosmo: Union[ccl.Cosmology, dict],
    z: float,
    positions: np.ndarray,
    velocities: np.ndarray,
    boxsize: Union[float, Sequence[float]],
    los: Sequence[float] = (0.0, 0.0, 1.0),
    Hz_override: float | None = None,
    positions_unit: str = "Mpc/h",
) -> np.ndarray:
    """
    Map real-space comoving positions x -> redshift-space positions s
    in a periodic box, using linear RSD:
        s = x + (v_parallel / (a * H_unit(z))) * n_hat

    Parameters
    ----------
    cosmo : ccl.Cosmology or dict
        Cosmology used to compute H(z). If a dict is given, a
        `ccl.Cosmology(**cosmo)` object is constructed internally.
        Ignored if Hz_override is provided.
    z : float
        Redshift of the snapshot.
    positions : ndarray, shape (N, 3)
        Real-space comoving positions x of galaxies.
        Default expected unit is Mpc/h (see positions_unit).
    velocities : ndarray, shape (N, 3)
        Peculiar velocities v of galaxies, typically in km/s.
    boxsize : float or array-like of length 3
        Periodic box size along each axis, in the same length units as `positions`.
    los : array-like of length 3, optional
        Line-of-sight unit vector n_hat. Default = (0, 0, 1) (z-axis).
    Hz_override : float, optional
        If provided, use this value as H(z) directly.
        IMPORTANT: Hz_override must be in km/s per *positions_unit*.
        - If positions_unit="Mpc/h", then Hz_override must be km/s/(Mpc/h).
        - If positions_unit="Mpc",   then Hz_override must be km/s/Mpc.
    positions_unit : str, optional
        Unit of `positions` and `boxsize`. Supported: {"Mpc/h","Mpc"}.
        Default is "Mpc/h".

    Returns
    -------
    s_positions : ndarray, shape (N, 3)
        Redshift-space comoving positions s, wrapped into [0, boxsize)
        along each axis (periodic boundary conditions).

    Notes
    -----
    - pyccl's H(a) can be obtained via h_over_h0(a) * H0, with H0=100*h km/s/Mpc,
      giving H(z) in km/s/Mpc (no h in the distance unit).
    - If your coordinates are in Mpc/h, we convert H(z) to km/s/(Mpc/h) by:
          H_{km/s/(Mpc/h)} = H_{km/s/Mpc} / h
      so that v/H has units of Mpc/h.
    """
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)

    if positions.shape != velocities.shape:
        raise ValueError("positions and velocities must have the same shape (N, 3)")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if velocities.ndim != 2 or velocities.shape[1] != 3:
        raise ValueError("velocities must have shape (N, 3)")

    # Box size handling (periodic cube or rectangular box)
    boxsize = np.asarray(boxsize, dtype=float)
    if boxsize.size == 1:
        boxsize = np.repeat(boxsize, 3)
    if boxsize.shape != (3,):
        raise ValueError("boxsize must be scalar or length-3")

    # Normalize LOS
    los = np.asarray(los, dtype=float)
    if los.shape != (3,):
        raise ValueError("los must be length-3")
    los_norm = np.linalg.norm(los)
    if los_norm == 0.0:
        raise ValueError("los vector must be non-zero")
    los = los / los_norm

    # Scale factor
    a = 1.0 / (1.0 + float(z))

    # Hubble parameter H(z)
    unit = str(positions_unit).strip().lower()
    if unit not in ("mpc/h", "mpc"):
        raise ValueError("positions_unit must be 'Mpc/h' or 'Mpc'.")

    if Hz_override is not None:
        # Caller guarantees Hz_override matches positions_unit.
        Hz = float(Hz_override)
    else:
        # Build Cosmology if needed
        if isinstance(cosmo, dict):
            cosmo = ccl.Cosmology(**cosmo)

        # Get h from cosmology (avoid hard-coding)
        try:
            h = float(cosmo["h"])
        except Exception:
            # Fallback: if cosmo doesn't support __getitem__, try attribute
            # (this path may rarely be needed depending on pyccl version)
            h = float(getattr(cosmo, "h", None))
            if not np.isfinite(h):
                raise ValueError("Could not read 'h' from cosmo; please pass Hz_override.")

        # Compute H(z) in km/s/Mpc:
        # H(a) = (H/H0)(a) * H0, with H0 = 100*h km/s/Mpc
        Hz_mpc = float(ccl.background.h_over_h0(cosmo, a) * (100.0 * h))

        if unit == "mpc":
            Hz = Hz_mpc  # km/s/Mpc
        else:
            # Convert to km/s/(Mpc/h) so that v/H is in Mpc/h.
            Hz = Hz_mpc / h

    # v_parallel = v ⋅ n̂   (km/s)
    v_par = np.dot(velocities, los)  # shape (N,)

    # Displacement along LOS in the same length unit as `positions` (Mpc/h by default):
    factor = 1.0 / (a * Hz)
    disp = factor * v_par[:, None] * los[None, :]  # (N,3)

    s_positions = positions + disp

    # Periodic wrapping into [0, L)
    s_positions = np.mod(s_positions, boxsize)

    return s_positions
# ============================================================
# Simple NFW fit (very lightweight)
# ============================================================

def _nfw_Menc(r, rs, rho_s):
    x = r / rs
    return 4.0 * np.pi * rho_s * rs**3 * (np.log(1.0 + x) - x / (1.0 + x))

def fit_enfw_profile(
    positions: np.ndarray,
    masses: np.ndarray | None = None,
    center: np.ndarray | None = None,
    nbins: int = 40,
) -> dict:
    """
    Crude enclosed-mass NFW fit using cumulative mass profile.

    Parameters
    ----------
    positions : (N,3)
    masses : (N,) or None (equal masses)
    center : (3,) or None (use mean position)
    nbins : number of radial bins

    Returns
    -------
    out : dict
        rs, rho_s, c (based on Rmax/rs), Rmax, fit_ok (bool)
    """
    if curve_fit is None:
        raise RuntimeError("scipy not available: cannot fit NFW profile")

    X = np.asarray(positions, float)
    N = X.shape[0]
    if N < 10:
        return {"fit_ok": False, "rs": np.nan, "rho_s": np.nan, "c": np.nan, "Rmax": np.nan}

    if masses is None:
        m = np.ones(N, float)
    else:
        m = np.asarray(masses, float)

    if center is None:
        cen = np.mean(X, axis=0)
    else:
        cen = np.asarray(center, float)

    r = np.linalg.norm(X - cen[None, :], axis=1)
    rmax = float(np.nanmax(r))
    if not np.isfinite(rmax) or rmax <= 0:
        return {"fit_ok": False, "rs": np.nan, "rho_s": np.nan, "c": np.nan, "Rmax": np.nan}

    # radial bins
    edges = np.linspace(0.0, rmax, nbins + 1)
    rc = 0.5 * (edges[1:] + edges[:-1])

    # cumulative mass
    order = np.argsort(r)
    r_sorted = r[order]
    m_sorted = m[order]
    m_cum = np.cumsum(m_sorted)

    # interpolate M(<r) at rc
    Mrc = np.interp(rc, r_sorted, m_cum, left=0.0, right=float(m_cum[-1]))

    # Fit only outside the very center (avoid r~0)
    use = rc > (0.02 * rmax)
    rc_fit = rc[use]
    M_fit = Mrc[use]

    if rc_fit.size < 10:
        return {"fit_ok": False, "rs": np.nan, "rho_s": np.nan, "c": np.nan, "Rmax": rmax}

    # initial guesses
    rs0 = 0.2 * rmax
    rho0 = (M_fit[-1]) / (4.0 * np.pi * rs0**3 * (np.log(1 + rmax/rs0) - (rmax/rs0)/(1 + rmax/rs0) + 1e-30))
    try:
        popt, pcov = curve_fit(_nfw_Menc, rc_fit, M_fit, p0=[rs0, rho0], maxfev=20000)
        rs, rho_s = popt
        c = rmax / rs if (np.isfinite(rs) and rs > 0) else np.nan
        return {"fit_ok": True, "rs": float(rs), "rho_s": float(rho_s), "c": float(c), "Rmax": rmax}
    except Exception:
        return {"fit_ok": False, "rs": np.nan, "rho_s": np.nan, "c": np.nan, "Rmax": rmax}
