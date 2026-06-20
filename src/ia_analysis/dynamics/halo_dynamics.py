# -*- coding: utf-8 -*-
"""
halo_dynamics.py
----------------

Shell-wise halo/subhalo dynamical analysis utilities.

This module is intended to sit on top of particle-level data from simulations
such as IllustrisTNG, Aquarius, or idealized N-body subhalo experiments.  It is
focused on the dynamical quantities discussed in the shell-wise tensor response
model:

    * shell masks for radial, binding-energy, or user-supplied subsets;
    * unnormalised shape tensor I_ij = sum m x_i x_j;
    * affine mean-flow tensor A_ij = d u_i / d x_j in a least-squares sense;
    * symmetric velocity-flow Hessian H_ij = (A_ij + A_ji)/2;
    * material-flow rotation tensor Omega_ij = (A_ij - A_ji)/2;
    * residual velocity-dispersion tensor S_ij;
    * figure-rotation estimate Pi_ij from the kinematic relation
          Pi_ij = Omega_hat_ij + eta_ij H_hat_ij;
    * gravitational/tidal Hessian utilities and tidal torque diagnostics;
    * shell-wise wrappers for later TNG/Aquarius analysis pipelines.

Conventions
-----------
Positions are assumed to be in a Cartesian coordinate system.  The user is
responsible for providing mutually consistent units for positions, velocities,
masses, the gravitational constant G, and any tidal Hessian.  The unnormalised
shape tensor is used throughout by default.

For an affine velocity field

    U = A X + c,

we use

    A = P^T I^{-1},     P_ij = sum m X_i U_j.

The velocity-flow Hessian H is the symmetric part of A, and Omega is the
antisymmetric material-flow rotation.  This H is a velocity-gradient Hessian,
not a gravitational Hessian.  Gravitational Hessians are named T, G, or
`tidal_hessian` explicitly.

The module is deliberately lightweight.  It optionally calls existing `shape.py`
utilities when available, but all core measurements below are implemented here
so that the file can be reused in different projects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]

try:
    # Optional: use existing helpers if this file is placed in the same directory.
    from ia_analysis.shapes.shape import compute_axis as _shape_compute_axis
    from ia_analysis.shapes.shape import ang_mom as _shape_ang_mom
    from ia_analysis.shapes.shape import kappa_rot as _shape_kappa_rot
    from ia_analysis.shapes.shape import beta_fig as _shape_beta_fig
except Exception:  # pragma: no cover - optional dependency
    _shape_compute_axis = None
    _shape_ang_mom = None
    _shape_kappa_rot = None
    _shape_beta_fig = None

try:
    # Optional import.  This is not required by the core routines.
    from ia_analysis.tides.tidal_field import PotentialInterpolator as _PotentialInterpolator
except Exception:  # pragma: no cover - optional dependency
    _PotentialInterpolator = None


# -----------------------------------------------------------------------------
# Basic linear-algebra helpers
# -----------------------------------------------------------------------------

_EPS3 = np.zeros((3, 3, 3), dtype=np.float64)
_EPS3[0, 1, 2] = _EPS3[1, 2, 0] = _EPS3[2, 0, 1] = 1.0
_EPS3[0, 2, 1] = _EPS3[2, 1, 0] = _EPS3[1, 0, 2] = -1.0


def _as_2d_positions(x: ArrayLike, name: str = "positions") -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"`{name}` must have shape (N, 3); got {arr.shape}")
    return arr


def _as_1d_masses(masses: Optional[ArrayLike], n: int) -> np.ndarray:
    if masses is None:
        return np.ones(n, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64)
    if m.shape != (n,):
        raise ValueError(f"`masses` must have shape ({n},); got {m.shape}")
    return m


def _as_vector(v: Optional[ArrayLike], name: str, default: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    if v is None:
        return default
    arr = np.asarray(v, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"`{name}` must have shape (3,); got {arr.shape}")
    return arr


def symmetrize(a: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a square matrix."""
    a = np.asarray(a, dtype=np.float64)
    return 0.5 * (a + a.T)


def skew(a: np.ndarray) -> np.ndarray:
    """Return the antisymmetric part of a square matrix."""
    a = np.asarray(a, dtype=np.float64)
    return 0.5 * (a - a.T)


def eigh_sorted_desc(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric eigendecomposition sorted by descending eigenvalue."""
    w, v = np.linalg.eigh(symmetrize(a))
    idx = np.argsort(w)[::-1]
    return w[idx].astype(np.float64), v[:, idx].astype(np.float64)


def skew_to_vector(omega_mat: np.ndarray) -> np.ndarray:
    """
    Convert an antisymmetric matrix M_ij = -epsilon_ijk omega_k to omega.
    """
    m = np.asarray(omega_mat, dtype=np.float64)
    return np.array([m[2, 1], m[0, 2], m[1, 0]], dtype=np.float64)


def vector_to_skew(omega: ArrayLike) -> np.ndarray:
    """
    Convert omega to an antisymmetric matrix M_ij = -epsilon_ijk omega_k.
    """
    w = np.asarray(omega, dtype=np.float64)
    if w.shape != (3,):
        raise ValueError("`omega` must have shape (3,)")
    return np.array(
        [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]],
        dtype=np.float64,
    )


def safe_inverse(a: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """Return inv(a), falling back to a pseudo-inverse if needed."""
    try:
        return np.linalg.inv(a)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(a, rcond=rcond)


def finite_particle_mask(
    positions: np.ndarray,
    velocities: Optional[np.ndarray] = None,
    masses: Optional[np.ndarray] = None,
    extra_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a finite, positive-mass particle mask."""
    good = np.all(np.isfinite(positions), axis=1)
    if velocities is not None:
        good &= np.all(np.isfinite(velocities), axis=1)
    if masses is not None:
        good &= np.isfinite(masses) & (masses > 0.0)
    if extra_mask is not None:
        extra = np.asarray(extra_mask, dtype=bool)
        if extra.shape != good.shape:
            raise ValueError("`extra_mask` has incompatible shape")
        good &= extra
    return good


# -----------------------------------------------------------------------------
# Centres and basic moments
# -----------------------------------------------------------------------------


def mass_weighted_mean(values: np.ndarray, masses: Optional[np.ndarray] = None) -> np.ndarray:
    """Mass-weighted mean of an (N, d) array."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("`values` must be a 2D array")
    if masses is None:
        return np.mean(x, axis=0)
    m = np.asarray(masses, dtype=np.float64)
    if m.shape != (x.shape[0],):
        raise ValueError("`masses` length mismatch")
    mtot = np.sum(m)
    if not np.isfinite(mtot) or mtot <= 0.0:
        raise ValueError("Non-positive total mass")
    return np.sum(x * m[:, None], axis=0) / mtot


def prepare_relative_phase_space(
    positions: ArrayLike,
    velocities: Optional[ArrayLike] = None,
    masses: Optional[ArrayLike] = None,
    center: Optional[ArrayLike] = None,
    v_ref: Optional[ArrayLike] = None,
    mask: Optional[np.ndarray] = None,
    center_mode: str = "given_or_mass_mean",
    velocity_mode: str = "given_or_mass_mean",
) -> Dict[str, np.ndarray]:
    """
    Build relative coordinates X and U for a selected particle subset.

    Parameters
    ----------
    center_mode : {'given_or_mass_mean', 'zero'}
        If center is None, either use the mass-weighted mean position or zero.
    velocity_mode : {'given_or_mass_mean', 'zero'}
        If v_ref is None, either use the mass-weighted mean velocity or zero.
    """
    pos = _as_2d_positions(positions)
    n = pos.shape[0]
    m = _as_1d_masses(masses, n)

    vel = None
    if velocities is not None:
        vel = _as_2d_positions(velocities, name="velocities")
        if vel.shape != pos.shape:
            raise ValueError("`velocities` must match `positions`")

    good = finite_particle_mask(pos, vel, m, extra_mask=mask)
    pos_s = pos[good]
    m_s = m[good]
    vel_s = None if vel is None else vel[good]

    if pos_s.shape[0] == 0:
        raise ValueError("No finite particles in the selected subset")

    if center is None:
        if center_mode == "given_or_mass_mean":
            cen = mass_weighted_mean(pos_s, m_s)
        elif center_mode == "zero":
            cen = np.zeros(3, dtype=np.float64)
        else:
            raise ValueError("Unknown center_mode")
    else:
        cen = _as_vector(center, "center")

    X = pos_s - cen[None, :]

    if vel_s is None:
        U = None
        v0 = None
    else:
        if v_ref is None:
            if velocity_mode == "given_or_mass_mean":
                v0 = mass_weighted_mean(vel_s, m_s)
            elif velocity_mode == "zero":
                v0 = np.zeros(3, dtype=np.float64)
            else:
                raise ValueError("Unknown velocity_mode")
        else:
            v0 = _as_vector(v_ref, "v_ref")
        U = vel_s - v0[None, :]

    return {
        "X": X,
        "U": U,
        "masses": m_s,
        "center": cen,
        "v_ref": v0,
        "selected_mask": good,
    }


def shape_tensor(X: np.ndarray, masses: Optional[np.ndarray] = None, normalize_mass: bool = False) -> np.ndarray:
    """Compute I_ij = sum m X_i X_j, optionally mass-normalized."""
    X = _as_2d_positions(X, name="X")
    m = _as_1d_masses(masses, X.shape[0])
    if normalize_mass:
        mtot = np.sum(m)
        if not np.isfinite(mtot) or mtot <= 0.0:
            raise ValueError("Non-positive total mass")
        m = m / mtot
    I = np.einsum("n,ni,nj->ij", m, X, X)
    return symmetrize(I)


def mixed_moment(X: np.ndarray, U: np.ndarray, masses: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute P_ij = sum m X_i U_j."""
    X = _as_2d_positions(X, name="X")
    U = _as_2d_positions(U, name="U")
    if U.shape != X.shape:
        raise ValueError("`U` must match `X`")
    m = _as_1d_masses(masses, X.shape[0])
    return np.einsum("n,ni,nj->ij", m, X, U)


def moment_derivative_tensor(
    X: np.ndarray,
    U: np.ndarray,
    masses: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Directly measured first derivative of the unnormalised shape tensor.

    Definition
    ----------
        I_ij      = sum_n m_n X_{n,i} X_{n,j},
        dI_ij/dt = sum_n m_n (U_{n,i} X_{n,j} + X_{n,i} U_{n,j}).

    This is the primitive particle-level dI measurement.  It does not assume
    an affine velocity field and does not use A = P.T I^{-1}.
    """
    X = _as_2d_positions(X, name="X")
    U = _as_2d_positions(U, name="U")
    if U.shape != X.shape:
        raise ValueError("`U` must match `X`")

    m = _as_1d_masses(masses, X.shape[0])

    dI = (
        np.einsum("n,ni,nj->ij", m, U, X)
        + np.einsum("n,ni,nj->ij", m, X, U)
    )
    return symmetrize(dI)


def angular_momentum_from_relative(X: np.ndarray, U: np.ndarray, masses: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute L = sum m X cross U."""
    X = _as_2d_positions(X, name="X")
    U = _as_2d_positions(U, name="U")
    if U.shape != X.shape:
        raise ValueError("`U` must match `X`")
    m = _as_1d_masses(masses, X.shape[0])
    return np.sum(np.cross(X, U) * m[:, None], axis=0)


# -----------------------------------------------------------------------------
# Shell masks
# -----------------------------------------------------------------------------


def radial_shell_masks(
    positions: ArrayLike,
    center: Optional[ArrayLike] = None,
    base_mask: Optional[np.ndarray] = None,
    n_shells: int = 5,
    edges: Optional[ArrayLike] = None,
    log: bool = False,
    equal_number: bool = False,
    r_min: Optional[float] = None,
    r_max: Optional[float] = None,
    include_outer_edge: bool = True,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Build radial shell masks.

    If `edges` is supplied, it is used directly.  Otherwise edges are computed
    from either equal-number quantiles or linearly/logarithmically spaced radii.
    """
    pos = _as_2d_positions(positions)
    n = pos.shape[0]
    if center is None:
        cen = np.mean(pos, axis=0)
    else:
        cen = _as_vector(center, "center")

    base = np.ones(n, dtype=bool) if base_mask is None else np.asarray(base_mask, dtype=bool).copy()
    if base.shape != (n,):
        raise ValueError("`base_mask` length mismatch")
    base &= np.all(np.isfinite(pos), axis=1)

    r = np.linalg.norm(pos - cen[None, :], axis=1)
    r_use = r[base]
    if r_use.size == 0:
        return [np.zeros(n, dtype=bool) for _ in range(n_shells)], {"edges": np.array([]), "r": r}

    if edges is None:
        if equal_number:
            q = np.linspace(0.0, 1.0, int(n_shells) + 1)
            edges_arr = np.quantile(r_use, q)
            edges_arr[0] = np.min(r_use)
            edges_arr[-1] = np.max(r_use)
        else:
            r0 = np.min(r_use) if r_min is None else float(r_min)
            r1 = np.max(r_use) if r_max is None else float(r_max)
            if log:
                positive = r_use[r_use > 0.0]
                if r_min is None:
                    r0 = np.min(positive) if positive.size else 1e-30
                edges_arr = np.logspace(np.log10(max(r0, 1e-30)), np.log10(r1), int(n_shells) + 1)
                edges_arr[0] = 0.0 if r_min is None else edges_arr[0]
            else:
                edges_arr = np.linspace(r0, r1, int(n_shells) + 1)
    else:
        edges_arr = np.asarray(edges, dtype=np.float64)
        if edges_arr.ndim != 1 or edges_arr.size < 2:
            raise ValueError("`edges` must be a 1D array with at least two values")
        n_shells = edges_arr.size - 1

    masks: List[np.ndarray] = []
    for i in range(int(n_shells)):
        lo, hi = edges_arr[i], edges_arr[i + 1]
        if i == int(n_shells) - 1 and include_outer_edge:
            mi = base & (r >= lo) & (r <= hi)
        else:
            mi = base & (r >= lo) & (r < hi)
        masks.append(mi)

    info = {
        "method": "radial",
        "center": cen,
        "r": r,
        "edges": edges_arr,
        "labels": [f"r[{edges_arr[i]:.4g},{edges_arr[i+1]:.4g})" for i in range(int(n_shells))],
    }
    return masks, info


def compute_particle_potential_direct(
    positions: ArrayLike,
    masses: Optional[ArrayLike] = None,
    source_positions: Optional[ArrayLike] = None,
    source_masses: Optional[ArrayLike] = None,
    G: float = 4.302e-6,
    softening: float = 0.0,
    exclude_self: bool = True,
    chunk_size: int = 4096,
) -> np.ndarray:
    """
    Direct softened potential Phi = -G sum m / sqrt(r^2 + eps^2).

    This is intended for moderate particle numbers or validation.  For large
    cosmological samples, pass precomputed particle potentials instead.
    """
    xq = _as_2d_positions(positions)
    if source_positions is None:
        xs = xq
        same_array = True
    else:
        xs = _as_2d_positions(source_positions, name="source_positions")
        same_array = False

    ms = _as_1d_masses(source_masses if source_masses is not None else masses, xs.shape[0])
    eps2 = float(softening) ** 2
    phi = np.zeros(xq.shape[0], dtype=np.float64)

    for i0 in range(0, xq.shape[0], int(chunk_size)):
        i1 = min(xq.shape[0], i0 + int(chunk_size))
        dx = xq[i0:i1, None, :] - xs[None, :, :]
        r2 = np.sum(dx * dx, axis=2) + eps2
        if exclude_self and same_array:
            rows = np.arange(i0, i1)
            r2[np.arange(i1 - i0), rows] = np.inf
        invr = 1.0 / np.sqrt(r2)
        phi[i0:i1] = -float(G) * np.sum(invr * ms[None, :], axis=1)
    return phi


def binding_energy_shell_masks(
    positions: ArrayLike,
    velocities: ArrayLike,
    masses: Optional[ArrayLike] = None,
    center: Optional[ArrayLike] = None,
    v_ref: Optional[ArrayLike] = None,
    potentials: Optional[ArrayLike] = None,
    base_mask: Optional[np.ndarray] = None,
    n_shells: int = 5,
    quantiles: Optional[ArrayLike] = None,
    energy_edges: Optional[ArrayLike] = None,
    most_bound_first: bool = True,
    compute_potential_if_missing: bool = False,
    G: float = 4.302e-6,
    softening: float = 0.0,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Build shell masks by specific binding energy.

    The specific energy is

        E = 0.5 |U|^2 + Phi.

    Lower E is more bound.  If `potentials` is None, direct potential
    computation is only performed when `compute_potential_if_missing=True`.
    """
    pos = _as_2d_positions(positions)
    vel = _as_2d_positions(velocities, name="velocities")
    if vel.shape != pos.shape:
        raise ValueError("`velocities` must match `positions`")
    n = pos.shape[0]
    m = _as_1d_masses(masses, n)

    base = np.ones(n, dtype=bool) if base_mask is None else np.asarray(base_mask, dtype=bool).copy()
    if base.shape != (n,):
        raise ValueError("`base_mask` length mismatch")
    base &= finite_particle_mask(pos, vel, m)

    if center is None:
        cen = mass_weighted_mean(pos[base], m[base]) if np.any(base) else np.zeros(3)
    else:
        cen = _as_vector(center, "center")

    if v_ref is None:
        v0 = mass_weighted_mean(vel[base], m[base]) if np.any(base) else np.zeros(3)
    else:
        v0 = _as_vector(v_ref, "v_ref")

    U = vel - v0[None, :]

    if potentials is None:
        if not compute_potential_if_missing:
            raise ValueError(
                "`potentials` is required for binding-energy shells unless "
                "compute_potential_if_missing=True."
            )
        phi = compute_particle_potential_direct(
            pos,
            masses=m,
            G=G,
            softening=softening,
            exclude_self=True,
        )
    else:
        phi = np.asarray(potentials, dtype=np.float64)
        if phi.shape != (n,):
            raise ValueError("`potentials` must have shape (N,)")

    E = 0.5 * np.sum(U * U, axis=1) + phi
    valid = base & np.isfinite(E)
    if np.count_nonzero(valid) == 0:
        return [np.zeros(n, dtype=bool) for _ in range(n_shells)], {"energy": E, "edges": np.array([])}

    if energy_edges is not None:
        edges = np.asarray(energy_edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("`energy_edges` must be a 1D array with at least two values")
        n_shells = edges.size - 1
    else:
        if quantiles is None:
            q = np.linspace(0.0, 1.0, int(n_shells) + 1)
        else:
            q = np.asarray(quantiles, dtype=np.float64)
            if q.ndim != 1 or q.size < 2:
                raise ValueError("`quantiles` must be a 1D array with at least two values")
            n_shells = q.size - 1
        edges = np.quantile(E[valid], q)
        edges[0] = np.min(E[valid])
        edges[-1] = np.max(E[valid])

    masks: List[np.ndarray] = []
    for i in range(int(n_shells)):
        lo, hi = edges[i], edges[i + 1]
        if i == int(n_shells) - 1:
            mi = valid & (E >= lo) & (E <= hi)
        else:
            mi = valid & (E >= lo) & (E < hi)
        masks.append(mi)

    if not most_bound_first:
        masks = masks[::-1]
        label_edges = edges[::-1]
    else:
        label_edges = edges

    info = {
        "method": "binding_energy",
        "center": cen,
        "v_ref": v0,
        "energy": E,
        "potential": phi,
        "edges": edges,
        "most_bound_first": most_bound_first,
        "labels": [f"E[{label_edges[i]:.4g},{label_edges[i+1]:.4g})" for i in range(int(n_shells))],
    }
    return masks, info


def spherical_potential_from_radial_mass(
    radii: ArrayLike,
    source_radii: ArrayLike,
    source_masses: ArrayLike,
    *,
    G: float = 4.30091727003628e-6,
    softening: float = 0.0,
) -> np.ndarray:
    """
    Approximate the gravitational potential from a spherical mass profile.

    Parameters
    ----------
    radii
        Query radii in the same length unit used by ``G``.
    source_radii, source_masses
        Radii and masses that define the spherical source profile.
    G
        Gravitational constant in units compatible with the input arrays.  The
        default is kpc (km/s)^2 / Msun.
    softening
        Plummer-like minimum radius used only to avoid singular denominators.

    Notes
    -----
    The returned potential is

        Phi(r) = -G [M(<r) / r + integral_r^inf dM(r') / r']

    with softened radii in both terms.  This is much cheaper than the direct
    pairwise potential and is the default for multi-component halo profiles.
    """
    rq = np.asarray(radii, dtype=np.float64).reshape(-1)
    rs = np.asarray(source_radii, dtype=np.float64).reshape(-1)
    ms = np.asarray(source_masses, dtype=np.float64).reshape(-1)
    if rs.shape != ms.shape:
        raise ValueError("`source_radii` and `source_masses` must have matching shapes")

    good = np.isfinite(rs) & np.isfinite(ms) & (rs >= 0.0) & (ms > 0.0)
    phi = np.full(rq.shape, np.nan, dtype=np.float64)
    qgood = np.isfinite(rq) & (rq >= 0.0)
    if not np.any(good) or not np.any(qgood):
        return phi

    eps = max(float(softening), 0.0)
    order = np.argsort(rs[good])
    r_sorted = rs[good][order]
    m_sorted = ms[good][order]
    r_soft = np.sqrt(r_sorted * r_sorted + eps * eps)
    r_soft = np.maximum(r_soft, np.finfo(np.float64).tiny)

    cumulative_mass = np.cumsum(m_sorted)
    cumulative_m_over_r = np.cumsum(m_sorted / r_soft)
    total_m_over_r = cumulative_m_over_r[-1]

    idx = np.searchsorted(r_sorted, rq[qgood], side="right") - 1
    inner_mass = np.where(idx >= 0, cumulative_mass[np.clip(idx, 0, cumulative_mass.size - 1)], 0.0)
    inner_m_over_r = np.where(idx >= 0, cumulative_m_over_r[np.clip(idx, 0, cumulative_m_over_r.size - 1)], 0.0)
    outer_m_over_r = total_m_over_r - inner_m_over_r

    rq_soft = np.sqrt(rq[qgood] * rq[qgood] + eps * eps)
    rq_soft = np.maximum(rq_soft, np.finfo(np.float64).tiny)
    phi[qgood] = -float(G) * (inner_mass / rq_soft + outer_m_over_r)
    return phi


def _optional_specific_array(values: Optional[ArrayLike], n: int, name: str) -> Optional[np.ndarray]:
    """Return an optional one-dimensional specific-energy-like array."""
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape != (n,):
        raise ValueError(f"`{name}` must have shape ({n},); got {arr.shape}")
    return arr


def _gas_specific_support_term(
    n: int,
    *,
    internal_energy: Optional[ArrayLike] = None,
    pressure: Optional[ArrayLike] = None,
    density: Optional[ArrayLike] = None,
    pressure_over_density: Optional[ArrayLike] = None,
    gas_energy_mode: str = "enthalpy",
    gas_gamma: float = 5.0 / 3.0,
) -> Tuple[np.ndarray, str]:
    """
    Build the gas thermal/pressure support term in velocity-squared units.

    ``InternalEnergy`` in Gadget/TNG data is a specific thermal energy.  For
    binding-energy work the most useful pressure-supported choice is often the
    specific enthalpy, ``u + P/rho = gamma * u`` for an ideal gas.  If pressure
    and density are supplied directly, the same term is computed as
    ``gamma / (gamma - 1) * P/rho``.
    """
    mode = str(gas_energy_mode or "none").lower().strip()
    if mode in ("none", "off", "false", "no"):
        return np.zeros(n, dtype=np.float64), "none"

    gamma = float(gas_gamma)
    if not np.isfinite(gamma) or gamma <= 1.0:
        raise ValueError("`gas_gamma` must be finite and greater than one")

    u = _optional_specific_array(internal_energy, n, "internal_energy")
    p_over_rho = _optional_specific_array(pressure_over_density, n, "pressure_over_density")
    if p_over_rho is None and pressure is not None and density is not None:
        p = _optional_specific_array(pressure, n, "pressure")
        rho = _optional_specific_array(density, n, "density")
        assert p is not None and rho is not None
        good = np.isfinite(p) & np.isfinite(rho) & (rho > 0.0)
        p_over_rho = np.zeros(n, dtype=np.float64)
        p_over_rho[good] = p[good] / rho[good]

    if mode in ("auto", "enthalpy", "pressure", "pressure_support"):
        if u is not None:
            term = gamma * u
            return np.where(np.isfinite(term), term, 0.0), "enthalpy_from_internal_energy"
        if p_over_rho is not None:
            term = gamma / (gamma - 1.0) * p_over_rho
            return np.where(np.isfinite(term), term, 0.0), "enthalpy_from_pressure_density"
        return np.zeros(n, dtype=np.float64), "missing_pressure_data"

    if mode in ("thermal", "internal", "internal_energy"):
        if u is not None:
            term = u
            return np.where(np.isfinite(term), term, 0.0), "internal_energy"
        if p_over_rho is not None:
            term = p_over_rho / (gamma - 1.0)
            return np.where(np.isfinite(term), term, 0.0), "thermal_from_pressure_density"
        return np.zeros(n, dtype=np.float64), "missing_pressure_data"

    if mode in ("p_over_rho", "pressure_over_density"):
        if p_over_rho is not None:
            return np.where(np.isfinite(p_over_rho), p_over_rho, 0.0), "pressure_over_density"
        return np.zeros(n, dtype=np.float64), "missing_pressure_data"

    raise ValueError(
        "`gas_energy_mode` must be one of 'auto', 'enthalpy', 'thermal', "
        "'p_over_rho', or 'none'"
    )


def component_binding_energy(
    positions: ArrayLike,
    velocities: ArrayLike,
    masses: Optional[ArrayLike] = None,
    *,
    center: Optional[ArrayLike] = None,
    v_ref: Optional[ArrayLike] = None,
    potentials: Optional[ArrayLike] = None,
    source_positions: Optional[ArrayLike] = None,
    source_masses: Optional[ArrayLike] = None,
    internal_energy: Optional[ArrayLike] = None,
    pressure: Optional[ArrayLike] = None,
    density: Optional[ArrayLike] = None,
    pressure_over_density: Optional[ArrayLike] = None,
    component: str = "matter",
    gas_energy_mode: str = "auto",
    gas_gamma: float = 5.0 / 3.0,
    potential_method: str = "spherical",
    compute_potential_if_missing: bool = True,
    G: float = 4.30091727003628e-6,
    softening: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute per-particle specific binding energy for one halo component.

    The specific total energy is

        e = 0.5 |v - v_ref|^2 + Phi + q_gas,

    where ``q_gas`` is zero for collisionless components.  For gas, ``q_gas``
    can include internal energy or pressure support through ``gas_energy_mode``.
    The returned ``specific_binding_energy`` is ``-e``; particles with positive
    values are energetically bound under this convention.

    By default positions and velocities are interpreted as already relative to
    the subhalo.  Pass ``center`` and ``v_ref`` when using absolute coordinates.
    """
    pos = _as_2d_positions(positions)
    vel = _as_2d_positions(velocities, name="velocities")
    if vel.shape != pos.shape:
        raise ValueError("`velocities` must match `positions`")
    n = pos.shape[0]
    m = _as_1d_masses(masses, n)

    cen = np.zeros(3, dtype=np.float64) if center is None else _as_vector(center, "center")
    v0 = np.zeros(3, dtype=np.float64) if v_ref is None else _as_vector(v_ref, "v_ref")
    X = pos - cen[None, :]
    U = vel - v0[None, :]
    radius = np.sqrt(np.sum(X * X, axis=1))
    kinetic = 0.5 * np.sum(U * U, axis=1)

    if potentials is not None:
        phi = np.asarray(potentials, dtype=np.float64).reshape(-1)
        if phi.shape != (n,):
            raise ValueError(f"`potentials` must have shape ({n},); got {phi.shape}")
        potential_source = "input"
    else:
        if not compute_potential_if_missing:
            raise ValueError("`potentials` is required unless compute_potential_if_missing=True")
        method = str(potential_method or "spherical").lower().strip()
        xs = X if source_positions is None else _as_2d_positions(source_positions, name="source_positions")
        ms = _as_1d_masses(source_masses, xs.shape[0])
        if method in ("spherical", "shell", "radial"):
            rs = np.sqrt(np.sum(xs * xs, axis=1))
            phi = spherical_potential_from_radial_mass(radius, rs, ms, G=G, softening=softening)
            potential_source = "spherical"
        elif method in ("direct", "pairwise"):
            phi = compute_particle_potential_direct(
                X,
                masses=m,
                source_positions=xs,
                source_masses=ms,
                G=G,
                softening=softening,
                exclude_self=source_positions is None,
            )
            potential_source = "direct"
        else:
            raise ValueError("`potential_method` must be 'spherical' or 'direct'")

    comp_name = str(component or "matter").lower()
    is_gas = comp_name in ("gas", "parttype0", "0")
    if is_gas:
        gas_term, gas_term_source = _gas_specific_support_term(
            n,
            internal_energy=internal_energy,
            pressure=pressure,
            density=density,
            pressure_over_density=pressure_over_density,
            gas_energy_mode=gas_energy_mode,
            gas_gamma=gas_gamma,
        )
    else:
        gas_term = np.zeros(n, dtype=np.float64)
        gas_term_source = "not_gas"

    specific_total_energy = kinetic + phi + gas_term
    specific_binding_energy = -specific_total_energy
    valid = finite_particle_mask(X, U, m) & np.isfinite(phi) & np.isfinite(specific_total_energy)
    bound = valid & (specific_binding_energy > 0.0)

    return {
        "component": component,
        "X": X,
        "U": U,
        "masses": m,
        "radius": radius,
        "kinetic": kinetic,
        "potential": phi,
        "gas_term": gas_term,
        "specific_total_energy": specific_total_energy,
        "specific_binding_energy": specific_binding_energy,
        "valid_mask": valid,
        "bound_mask": bound,
        "center": cen,
        "v_ref": v0,
        "potential_source": potential_source,
        "gas_term_source": gas_term_source,
    }


def binding_energy_mass_distribution(
    binding_energy: ArrayLike,
    masses: ArrayLike,
    *,
    bins: Union[int, ArrayLike] = 64,
    edges: Optional[ArrayLike] = None,
    log_bins: bool = True,
    bound_only: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Histogram mass as a function of positive specific binding energy.

    Parameters
    ----------
    binding_energy
        Specific binding energy, usually ``-specific_total_energy``.
    masses
        Particle masses used as histogram weights.
    bins, edges
        Either an integer bin count or explicit edges.  ``edges`` takes
        precedence when provided.
    log_bins
        Use logarithmic binning when explicit edges are not supplied.
    bound_only
        If True, keep only particles with positive binding energy.
    """
    e = np.asarray(binding_energy, dtype=np.float64).reshape(-1)
    m = np.asarray(masses, dtype=np.float64).reshape(-1)
    if e.shape != m.shape:
        raise ValueError("`binding_energy` and `masses` must have matching shapes")

    valid = np.isfinite(e) & np.isfinite(m) & (m > 0.0)
    if bound_only:
        valid &= e > 0.0

    if edges is not None:
        bin_edges = np.asarray(edges, dtype=np.float64).reshape(-1)
    elif np.ndim(bins) > 0 and not np.isscalar(bins):
        bin_edges = np.asarray(bins, dtype=np.float64).reshape(-1)
    else:
        n_bins = int(bins)
        if n_bins <= 0:
            raise ValueError("`bins` must be positive")
        values = e[valid]
        if values.size == 0:
            bin_edges = np.array([], dtype=np.float64)
        elif log_bins:
            values = values[values > 0.0]
            if values.size == 0:
                bin_edges = np.array([], dtype=np.float64)
            else:
                lo = float(np.nanmin(values))
                hi = float(np.nanmax(values))
                if not hi > lo:
                    lo = max(lo * 0.5, np.finfo(np.float64).tiny)
                    hi = hi * 1.5 if hi > 0.0 else 1.0
                bin_edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
        else:
            lo = float(np.nanmin(values))
            hi = float(np.nanmax(values))
            if not hi > lo:
                pad = max(abs(lo) * 0.5, 1.0)
                lo -= pad
                hi += pad
            bin_edges = np.linspace(lo, hi, n_bins + 1)

    if bin_edges.size < 2:
        return {
            "edges": bin_edges,
            "centers": np.array([], dtype=np.float64),
            "mass": np.array([], dtype=np.float64),
            "count": np.array([], dtype=np.int64),
            "mass_density": np.array([], dtype=np.float64),
        }

    hist_mass, _ = np.histogram(e[valid], bins=bin_edges, weights=m[valid])
    hist_count, _ = np.histogram(e[valid], bins=bin_edges)
    if np.all(bin_edges > 0.0):
        centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        widths = np.diff(np.log10(bin_edges))
    else:
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        widths = np.diff(bin_edges)
    widths = np.where(widths > 0.0, widths, np.nan)
    density = hist_mass / widths

    return {
        "edges": bin_edges,
        "centers": centers,
        "mass": hist_mass.astype(np.float64),
        "count": hist_count.astype(np.int64),
        "mass_density": density.astype(np.float64),
    }


def _component_value(component: Mapping[str, Any], *names: str) -> Any:
    """Return the first present component field from a list of aliases."""
    for name in names:
        if name in component:
            return component[name]
    return None


def component_binding_energy_profiles(
    components: Mapping[str, Mapping[str, Any]],
    *,
    source_components: Optional[Sequence[str]] = None,
    bins: Union[int, ArrayLike] = 64,
    energy_edges: Optional[ArrayLike] = None,
    log_bins: bool = True,
    bound_only: bool = True,
    gas_energy_mode: str = "enthalpy",
    gas_gamma: float = 5.0 / 3.0,
    potential_method: str = "spherical",
    G: float = 4.30091727003628e-6,
    softening: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute mass distributions over binding energy for multiple components.

    ``components`` is a mapping from component name to arrays.  Each component
    may use either generic keys (``positions``, ``velocities``) or the TNG
    wrapper keys (``X_kpc``, ``U_kms``).  Masses are read from ``masses``.  Gas
    pressure support is read from ``internal_energy`` and, when present,
    ``pressure`` plus ``density``.

    The source potential is built from the union of ``source_components``.  By
    default every supplied component contributes to the potential.
    """
    if not components:
        raise ValueError("`components` must contain at least one component")

    normalized: Dict[str, Dict[str, Any]] = {}
    for name, comp in components.items():
        pos = _component_value(comp, "positions", "X_kpc", "x")
        vel = _component_value(comp, "velocities", "U_kms", "v")
        if pos is None or vel is None:
            raise KeyError(f"Component '{name}' must provide positions/X_kpc and velocities/U_kms")
        pos_arr = _as_2d_positions(pos, name=f"{name}.positions")
        vel_arr = _as_2d_positions(vel, name=f"{name}.velocities")
        if vel_arr.shape != pos_arr.shape:
            raise ValueError(f"Component '{name}' velocity array does not match positions")
        mass_arr = _as_1d_masses(_component_value(comp, "masses", "mass"), pos_arr.shape[0])
        normalized[str(name)] = {
            "positions": pos_arr,
            "velocities": vel_arr,
            "masses": mass_arr,
            "potential": _component_value(comp, "potential", "potentials"),
            "internal_energy": _component_value(comp, "internal_energy", "InternalEnergy"),
            "pressure": _component_value(comp, "pressure", "Pressure"),
            "density": _component_value(comp, "density", "Density"),
            "pressure_over_density": _component_value(comp, "pressure_over_density", "p_over_rho"),
        }

    source_names = list(source_components) if source_components is not None else list(normalized)
    source_positions_list: List[np.ndarray] = []
    source_masses_list: List[np.ndarray] = []
    for name in source_names:
        if name not in normalized:
            raise KeyError(f"Source component '{name}' is not present in components")
        comp = normalized[name]
        good = finite_particle_mask(comp["positions"], comp["velocities"], comp["masses"])
        if np.any(good):
            source_positions_list.append(comp["positions"][good])
            source_masses_list.append(comp["masses"][good])
    if not source_positions_list:
        raise ValueError("No finite positive-mass particles are available for the potential source")
    source_positions_all = np.vstack(source_positions_list)
    source_masses_all = np.concatenate(source_masses_list)

    energies: Dict[str, Dict[str, Any]] = {}
    all_binding: List[np.ndarray] = []
    for name, comp in normalized.items():
        pot = comp["potential"] if str(potential_method).lower().strip() in ("input", "precomputed") else None
        out = component_binding_energy(
            comp["positions"],
            comp["velocities"],
            comp["masses"],
            potentials=pot,
            source_positions=source_positions_all,
            source_masses=source_masses_all,
            internal_energy=comp["internal_energy"],
            pressure=comp["pressure"],
            density=comp["density"],
            pressure_over_density=comp["pressure_over_density"],
            component=name,
            gas_energy_mode=gas_energy_mode,
            gas_gamma=gas_gamma,
            potential_method="spherical" if str(potential_method).lower().strip() in ("input", "precomputed") else potential_method,
            G=G,
            softening=softening,
        )
        energies[name] = out
        good = out["valid_mask"]
        if bound_only:
            good = good & out["bound_mask"]
        all_binding.append(out["specific_binding_energy"][good])

    common_edges = None if energy_edges is None else np.asarray(energy_edges, dtype=np.float64)
    if common_edges is None:
        values = np.concatenate([x[np.isfinite(x)] for x in all_binding if x.size]) if any(x.size for x in all_binding) else np.array([])
        if log_bins:
            values = values[values > 0.0]
        if values.size:
            common_edges = binding_energy_mass_distribution(
                values,
                np.ones_like(values),
                bins=bins,
                log_bins=log_bins,
                bound_only=bound_only,
            )["edges"]

    distributions: Dict[str, Dict[str, np.ndarray]] = {}
    summary: List[Dict[str, Any]] = []
    for name, out in energies.items():
        hist = binding_energy_mass_distribution(
            out["specific_binding_energy"],
            out["masses"],
            bins=bins,
            edges=common_edges,
            log_bins=log_bins,
            bound_only=bound_only,
        )
        distributions[name] = hist
        valid = out["valid_mask"]
        bound = out["bound_mask"]
        total_mass = float(np.sum(out["masses"][valid])) if np.any(valid) else 0.0
        bound_mass = float(np.sum(out["masses"][bound])) if np.any(bound) else 0.0
        be = out["specific_binding_energy"][bound]
        summary.append(
            {
                "component": name,
                "n_particles": int(out["masses"].size),
                "n_valid": int(np.count_nonzero(valid)),
                "n_bound": int(np.count_nonzero(bound)),
                "mass_total": total_mass,
                "mass_bound": bound_mass,
                "bound_mass_fraction": bound_mass / total_mass if total_mass > 0.0 else np.nan,
                "median_binding_energy": float(np.nanmedian(be)) if be.size else np.nan,
                "potential_source": out["potential_source"],
                "gas_term_source": out["gas_term_source"],
            }
        )

    return {
        "components": energies,
        "binding_distribution": distributions,
        "summary": summary,
        "energy_edges": common_edges if common_edges is not None else np.array([], dtype=np.float64),
        "source_components": source_names,
    }


def normalize_subset_masks(subsets: Sequence[np.ndarray], n_particles: int) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Validate user-supplied subset masks for shell-wise analysis."""
    masks: List[np.ndarray] = []
    for i, s in enumerate(subsets):
        mi = np.asarray(s, dtype=bool)
        if mi.shape != (n_particles,):
            raise ValueError(f"subset mask {i} has shape {mi.shape}, expected ({n_particles},)")
        masks.append(mi.copy())
    info = {"method": "subsets", "labels": [f"subset_{i}" for i in range(len(masks))]}
    return masks, info


def make_shell_masks(
    positions: ArrayLike,
    velocities: Optional[ArrayLike] = None,
    masses: Optional[ArrayLike] = None,
    center: Optional[ArrayLike] = None,
    v_ref: Optional[ArrayLike] = None,
    method: str = "radial",
    subsets: Optional[Sequence[np.ndarray]] = None,
    base_mask: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Unified shell-mask builder.

    Supported methods
    -----------------
    'radial'
        Radial shells around `center`.
    'binding_energy'
        Shells ordered by specific energy.  Requires velocities and potentials,
        unless direct potential computation is explicitly requested.
    'subsets'
        Use user-supplied boolean masks directly.

    This is the main entry point when a later analysis function needs the
    particle subset corresponding to each shell/layer.
    """
    pos = _as_2d_positions(positions)
    mode = str(method).lower().strip()

    if mode in ("radial", "radius", "r"):
        return radial_shell_masks(pos, center=center, base_mask=base_mask, **kwargs)

    if mode in ("binding", "binding_energy", "energy", "e"):
        if velocities is None:
            raise ValueError("`velocities` is required for binding-energy shell masks")
        return binding_energy_shell_masks(
            pos,
            velocities,
            masses=masses,
            center=center,
            v_ref=v_ref,
            base_mask=base_mask,
            **kwargs,
        )

    if mode in ("subsets", "subset", "mask", "masks"):
        if subsets is None:
            raise ValueError("`subsets` must be provided for method='subsets'")
        masks, info = normalize_subset_masks(subsets, pos.shape[0])
        if base_mask is not None:
            base = np.asarray(base_mask, dtype=bool)
            if base.shape != (pos.shape[0],):
                raise ValueError("`base_mask` length mismatch")
            masks = [mi & base for mi in masks]
        return masks, info

    raise ValueError("Unknown shell-mask method. Use 'radial', 'binding_energy', or 'subsets'.")


# -----------------------------------------------------------------------------
# Kinematic analysis: A, H, Omega, velocity dispersion
# -----------------------------------------------------------------------------


def compute_affine_kinematics(
    positions: ArrayLike,
    velocities: ArrayLike,
    masses: Optional[ArrayLike] = None,
    center: Optional[ArrayLike] = None,
    v_ref: Optional[ArrayLike] = None,
    mask: Optional[np.ndarray] = None,
    normalize_mass: bool = False,
    min_particles: int = 10,
    rcond: float = 1e-12,
) -> Dict[str, Any]:
    """
    Measure shell kinematics for one subset.

    Returned quantities
    -------------------
    I
        Shape tensor, I_ij = sum m X_i X_j unless normalize_mass=True.

    dI
        Direct particle-level tensor derivative,

            dI_ij = sum m (U_i X_j + X_i U_j).

        This is the primitive measured quantity used to define the measured
        figure rotation Pi.

    A, H, Omega
        Best-fitting affine mean-flow quantities.  These are retained for
        the Omega + eta H model decomposition, but are not used to define
        the measured Pi.
    """
    data = prepare_relative_phase_space(
        positions,
        velocities=velocities,
        masses=masses,
        center=center,
        v_ref=v_ref,
        mask=mask,
    )
    X = data["X"]
    U = data["U"]
    m = data["masses"]
    n_used = X.shape[0]

    if n_used < int(min_particles):
        nan33 = np.full((3, 3), np.nan)
        nan3 = np.full(3, np.nan)
        return {
            "valid": False,
            "N_used": int(n_used),
            "center": data["center"],
            "v_ref": data["v_ref"],
            "selected_mask": data["selected_mask"],
            "X": X,
            "U": U,
            "masses": m,
            "I": nan33.copy(),
            "dI": nan33.copy(),
            "P": nan33.copy(),
            "A": nan33.copy(),
            "H": nan33.copy(),
            "Omega": nan33.copy(),
            "omega": nan3.copy(),
            "S": nan33.copy(),
            "L": nan3.copy(),
        }

    I = shape_tensor(X, m, normalize_mass=normalize_mass)

    # Primitive measured dI.  This is the definition of measured Pi.
    dI = moment_derivative_tensor(X, U, m)

    # Affine quantities retained for the Omega + H decomposition.
    P = mixed_moment(X, U, m)
    A = P.T @ safe_inverse(I, rcond=rcond)
    H = symmetrize(A)
    Om = skew(A)
    omega = skew_to_vector(Om)

    # Residual velocity after the best-fitting affine flow.
    U_model = X @ A.T
    c = U - U_model
    S = np.einsum("n,ni,nj->ij", m, c, c)
    S = symmetrize(S)
    L = angular_momentum_from_relative(X, U, m)

    return {
        "valid": True,
        "N_used": int(n_used),
        "center": data["center"],
        "v_ref": data["v_ref"],
        "selected_mask": data["selected_mask"],
        "X": X,
        "U": U,
        "masses": m,
        "I": I,
        "dI": dI,
        "P": P,
        "A": A,
        "H": H,
        "Omega": Om,
        "omega": omega,
        "S": S,
        "L": L,
    }


def figure_rotation_from_dI(
    I: np.ndarray,
    dI: np.ndarray,
    rel_gap_min: float = 1e-8,
) -> Dict[str, Any]:
    """
    Directly measured body-frame figure rotation from I and dI.

    This is the measurement definition.  Let R diagonalise I and define

        K = R.T @ dI @ R.

    For i != j,

        Pi_ij = K_ij / (lambda_j - lambda_i).

    This Pi is inferred directly from the measured shape-tensor derivative dI,
    not from the affine Omega + eta H decomposition.
    """
    nan33 = np.full((3, 3), np.nan, dtype=np.float64)
    nan3 = np.full(3, np.nan, dtype=np.float64)

    try:
        I = symmetrize(np.asarray(I, dtype=np.float64))
        dI = symmetrize(np.asarray(dI, dtype=np.float64))

        evals, evecs = eigh_sorted_desc(I)
        scale = max(np.max(np.abs(evals)), 1.0)
        gaps = np.array(
            [
                abs(evals[1] - evals[0]),
                abs(evals[2] - evals[0]),
                abs(evals[2] - evals[1]),
            ],
            dtype=np.float64,
        )

        valid = bool(
            np.all(np.isfinite(evals))
            and np.all(np.isfinite(dI))
            and np.min(gaps) >= rel_gap_min * scale
        )

        if not valid:
            return {
                "valid": False,
                "source": "direct_dI",
                "evals": evals,
                "evecs": evecs,
                "dI_hat": nan33.copy(),
                "lambda_dot": nan3.copy(),
                "Pi": nan33.copy(),
                "varpi": nan3.copy(),
                "A_hat": nan33.copy(),
                "H_hat": nan33.copy(),
                "Omega_hat": nan33.copy(),
                "eta": nan33.copy(),
            }

        R = evecs
        dI_hat = R.T @ dI @ R
        lambda_dot = np.diag(dI_hat).astype(np.float64)

        Pi = np.zeros((3, 3), dtype=np.float64)
        eta = np.full((3, 3), np.nan, dtype=np.float64)

        for i in range(3):
            for j in range(i + 1, 3):
                delta = evals[j] - evals[i]

                Pi_ij = dI_hat[i, j] / delta
                Pi[i, j] = Pi_ij
                Pi[j, i] = -Pi_ij

                eta_ij = (evals[i] + evals[j]) / delta
                eta[i, j] = eta_ij
                eta[j, i] = -eta_ij

        return {
            "valid": True,
            "source": "direct_dI",
            "evals": evals,
            "evecs": R,
            "dI_hat": dI_hat,
            "lambda_dot": lambda_dot,
            "Pi": Pi,
            "varpi": skew_to_vector(Pi),
            "A_hat": nan33.copy(),
            "H_hat": nan33.copy(),
            "Omega_hat": nan33.copy(),
            "eta": eta,
        }

    except Exception as exc:
        return {
            "valid": False,
            "source": "direct_dI",
            "error": str(exc),
            "evals": nan3.copy(),
            "evecs": nan33.copy(),
            "dI_hat": nan33.copy(),
            "lambda_dot": nan3.copy(),
            "Pi": nan33.copy(),
            "varpi": nan3.copy(),
            "A_hat": nan33.copy(),
            "H_hat": nan33.copy(),
            "Omega_hat": nan33.copy(),
            "eta": nan33.copy(),
        }


def figure_rotation_from_affine(
    I: np.ndarray,
    A: np.ndarray,
    rel_gap_min: float = 1e-8,
) -> Dict[str, Any]:
    """
    Compute body-frame figure rotation Pi from I and affine velocity gradient A.

    Pi_ij = Omega_hat_ij + eta_ij H_hat_ij,   i != j.
    """
    nan33 = np.full((3, 3), np.nan, dtype=np.float64)
    nan3 = np.full(3, np.nan, dtype=np.float64)
    try:
        evals, evecs = eigh_sorted_desc(I)
        scale = max(np.max(np.abs(evals)), 1.0)
        gaps = np.array(
            [abs(evals[1] - evals[0]), abs(evals[2] - evals[0]), abs(evals[2] - evals[1])]
        )
        valid = bool(np.all(np.isfinite(evals)) and np.min(gaps) >= rel_gap_min * scale)
        if not valid:
            return {
                "valid": False,
                "evals": evals,
                "evecs": evecs,
                "A_hat": nan33.copy(),
                "H_hat": nan33.copy(),
                "Omega_hat": nan33.copy(),
                "Pi": nan33.copy(),
                "varpi": nan3.copy(),
                "eta": nan33.copy(),
            }

        R = evecs
        Ahat = R.T @ A @ R
        Hhat = symmetrize(Ahat)
        Ohat = skew(Ahat)
        Pi = np.zeros((3, 3), dtype=np.float64)
        eta = np.full((3, 3), np.nan, dtype=np.float64)
        for i in range(3):
            for j in range(i + 1, 3):
                delta = evals[j] - evals[i]
                eta_ij = (evals[i] + evals[j]) / delta
                eta[i, j] = eta_ij
                eta[j, i] = -eta_ij
                Pi[i, j] = Ohat[i, j] + eta_ij * Hhat[i, j]
                Pi[j, i] = -Pi[i, j]
        return {
            "valid": True,
            "evals": evals,
            "evecs": evecs,
            "A_hat": Ahat,
            "H_hat": Hhat,
            "Omega_hat": Ohat,
            "Pi": Pi,
            "varpi": skew_to_vector(Pi),
            "eta": eta,
        }
    except Exception:
        return {
            "valid": False,
            "evals": nan3.copy(),
            "evecs": nan33.copy(),
            "A_hat": nan33.copy(),
            "H_hat": nan33.copy(),
            "Omega_hat": nan33.copy(),
            "Pi": nan33.copy(),
            "varpi": nan3.copy(),
            "eta": nan33.copy(),
        }


def measure_hessian_and_omega(
    positions: ArrayLike,
    velocities: ArrayLike,
    masses: Optional[ArrayLike] = None,
    center: Optional[ArrayLike] = None,
    v_ref: Optional[ArrayLike] = None,
    mask: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Convenience wrapper to measure H, Omega, and figure rotation.

    The ``fig_*`` keys now refer to the directly measured dI-based Pi.
    The affine Omega+eta H model is also returned with ``fig_affine_*`` keys.
    """
    kin = compute_affine_kinematics(
        positions,
        velocities,
        masses=masses,
        center=center,
        v_ref=v_ref,
        mask=mask,
        **kwargs,
    )

    if kin["valid"]:
        fig_direct = figure_rotation_from_dI(kin["I"], kin["dI"])
        fig_affine = figure_rotation_from_affine(kin["I"], kin["A"])
    else:
        fig_direct = {}
        fig_affine = {}

    kin.update({f"fig_{k}": v for k, v in fig_direct.items()})
    kin.update({f"fig_affine_{k}": v for k, v in fig_affine.items()})
    return kin


# -----------------------------------------------------------------------------
# Gravitational/tidal Hessian and torque utilities
# -----------------------------------------------------------------------------


def potential_hessian_direct(
    query_positions: ArrayLike,
    source_positions: ArrayLike,
    source_masses: Optional[ArrayLike] = None,
    G: float = 4.302e-6,
    softening: float = 0.0,
    exclude_self: bool = False,
    chunk_size: int = 1024,
) -> np.ndarray:
    """
    Direct softened gravitational Hessian of Phi at query positions.

    For Phi = -G m / sqrt(r^2 + eps^2),

        d_i d_j Phi = G m [delta_ij / s^3 - 3 r_i r_j / s^5],

    where r = x_query - x_source and s^2 = r^2 + eps^2.

    Parameters
    ----------
    exclude_self : bool
        If True and query/source arrays have the same length, the diagonal
        self-contribution is removed.  Use this for all-particle self queries.
    """
    xq = _as_2d_positions(query_positions, name="query_positions")
    xs = _as_2d_positions(source_positions, name="source_positions")
    ms = _as_1d_masses(source_masses, xs.shape[0])
    eps2 = float(softening) ** 2
    out = np.zeros((xq.shape[0], 3, 3), dtype=np.float64)
    same_length = xq.shape[0] == xs.shape[0]

    eye = np.eye(3, dtype=np.float64)
    for i0 in range(0, xq.shape[0], int(chunk_size)):
        i1 = min(xq.shape[0], i0 + int(chunk_size))
        r = xq[i0:i1, None, :] - xs[None, :, :]
        r2 = np.sum(r * r, axis=2) + eps2
        if exclude_self and same_length:
            rows = np.arange(i0, i1)
            r2[np.arange(i1 - i0), rows] = np.inf
        inv3 = r2 ** (-1.5)
        inv5 = r2 ** (-2.5)
        term1 = inv3[:, :, None, None] * eye[None, None, :, :]
        term2 = 3.0 * inv5[:, :, None, None] * r[:, :, :, None] * r[:, :, None, :]
        h = float(G) * np.sum(ms[None, :, None, None] * (term1 - term2), axis=1)
        out[i0:i1] = symmetrize_batch(h)
    return out


def symmetrize_batch(a: np.ndarray) -> np.ndarray:
    """Symmetrize an array of matrices with shape (..., 3, 3)."""
    return 0.5 * (a + np.swapaxes(a, -1, -2))


def average_hessian_over_subset(
    hessians: np.ndarray,
    masses: Optional[ArrayLike] = None,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mass-weighted average of per-particle Hessians over a subset."""
    H = np.asarray(hessians, dtype=np.float64)
    if H.ndim != 3 or H.shape[1:] != (3, 3):
        raise ValueError("`hessians` must have shape (N, 3, 3)")
    n = H.shape[0]
    m = _as_1d_masses(masses, n)
    good = np.all(np.isfinite(H), axis=(1, 2)) & np.isfinite(m) & (m > 0.0)
    if mask is not None:
        mi = np.asarray(mask, dtype=bool)
        if mi.shape != (n,):
            raise ValueError("`mask` length mismatch")
        good &= mi
    if np.count_nonzero(good) == 0:
        return np.full((3, 3), np.nan)
    w = m[good]
    return symmetrize(np.sum(H[good] * w[:, None, None], axis=0) / np.sum(w))


def tidal_tensor_from_components(components: ArrayLike) -> np.ndarray:
    """
    Convert [Txx, Txy, Txz, Tyy, Tyz, Tzz] to a symmetric 3x3 matrix.
    """
    c = np.asarray(components, dtype=np.float64)
    if c.shape != (6,):
        raise ValueError("`components` must have shape (6,)")
    return np.array(
        [[c[0], c[1], c[2]], [c[1], c[3], c[4]], [c[2], c[4], c[5]]],
        dtype=np.float64,
    )


def sample_tidal_hessian(
    position: ArrayLike,
    tidal_source: Union[np.ndarray, Callable[[np.ndarray], Any]],
) -> np.ndarray:
    """
    Sample or coerce a tidal Hessian at one position.

    `tidal_source` may be:
      * a fixed (3,3) tensor;
      * a 6-component vector [Txx,Txy,Txz,Tyy,Tyz,Tzz];
      * a callable, e.g. PotentialInterpolator, returning either T or (Phi,T).
    """
    pos = np.asarray(position, dtype=np.float64)
    if pos.shape != (3,):
        raise ValueError("`position` must have shape (3,)")

    if callable(tidal_source):
        val = tidal_source(pos)
        if isinstance(val, tuple) and len(val) == 2:
            T = val[1]
        else:
            T = val
        T = np.asarray(T, dtype=np.float64)
    else:
        T = np.asarray(tidal_source, dtype=np.float64)

    if T.shape == (6,):
        return tidal_tensor_from_components(T)
    if T.shape == (3, 3):
        return symmetrize(T)
    raise ValueError("Could not interpret tidal_source as a 3x3 Hessian")


def torque_from_hessian(I: np.ndarray, hessian: np.ndarray) -> np.ndarray:
    """
    Compute tidal torque tau_i = -epsilon_ijk I_jl T_kl.
    """
    I = symmetrize(np.asarray(I, dtype=np.float64))
    T = symmetrize(np.asarray(hessian, dtype=np.float64))
    return -np.einsum("ijk,jl,kl->i", _EPS3, I, T).astype(np.float64)


def tidal_response_terms(
    I: np.ndarray,
    A: np.ndarray,
    S: Optional[np.ndarray] = None,
    G_int: Optional[np.ndarray] = None,
    T_ext: Optional[np.ndarray] = None,
    rel_gap_min: float = 1e-8,
) -> Dict[str, Any]:
    """
    Compute body-frame source terms in the slow-shape Pi-dot budget.

        dot Pi_ij = D_flow_ij + D_disp_ij + D_int_ij + D_tide_ij.

    The returned matrices are antisymmetric and live in the shell body frame.
    """
    nan33 = np.full((3, 3), np.nan, dtype=np.float64)
    try:
        evals, R = eigh_sorted_desc(I)
        scale = max(np.max(np.abs(evals)), 1.0)
        gaps = np.array(
            [abs(evals[1] - evals[0]), abs(evals[2] - evals[0]), abs(evals[2] - evals[1])]
        )
        if np.min(gaps) < rel_gap_min * scale:
            raise ValueError("near-degenerate shape tensor")

        Ahat = R.T @ np.asarray(A, dtype=np.float64) @ R
        Shat = np.zeros((3, 3), dtype=np.float64) if S is None else R.T @ symmetrize(S) @ R
        Ghat = np.zeros((3, 3), dtype=np.float64) if G_int is None else R.T @ symmetrize(G_int) @ R
        That = np.zeros((3, 3), dtype=np.float64) if T_ext is None else R.T @ symmetrize(T_ext) @ R

        D_flow = np.zeros((3, 3), dtype=np.float64)
        D_disp = np.zeros((3, 3), dtype=np.float64)
        D_int = np.zeros((3, 3), dtype=np.float64)
        D_tide = np.zeros((3, 3), dtype=np.float64)
        eta = np.full((3, 3), np.nan, dtype=np.float64)

        for i in range(3):
            for j in range(i + 1, 3):
                delta = evals[j] - evals[i]
                eta_ij = (evals[i] + evals[j]) / delta
                eta[i, j] = eta_ij
                eta[j, i] = -eta_ij
                flow_ij = 2.0 * np.sum(evals * Ahat[i, :] * Ahat[j, :]) / delta
                disp_ij = 2.0 * Shat[i, j] / delta
                int_ij = -eta_ij * Ghat[i, j]
                tide_ij = -eta_ij * That[i, j]
                D_flow[i, j] = flow_ij
                D_disp[i, j] = disp_ij
                D_int[i, j] = int_ij
                D_tide[i, j] = tide_ij
                D_flow[j, i] = -flow_ij
                D_disp[j, i] = -disp_ij
                D_int[j, i] = -int_ij
                D_tide[j, i] = -tide_ij

        D_total = D_flow + D_disp + D_int + D_tide
        return {
            "valid": True,
            "evals": evals,
            "evecs": R,
            "A_hat": Ahat,
            "S_hat": Shat,
            "G_hat": Ghat,
            "T_hat": That,
            "eta": eta,
            "D_flow": D_flow,
            "D_disp": D_disp,
            "D_int": D_int,
            "D_tide": D_tide,
            "D_total": D_total,
            "D_total_vec": skew_to_vector(D_total),
        }
    except Exception as exc:
        return {
            "valid": False,
            "error": str(exc),
            "evals": np.full(3, np.nan),
            "evecs": nan33.copy(),
            "A_hat": nan33.copy(),
            "S_hat": nan33.copy(),
            "G_hat": nan33.copy(),
            "T_hat": nan33.copy(),
            "eta": nan33.copy(),
            "D_flow": nan33.copy(),
            "D_disp": nan33.copy(),
            "D_int": nan33.copy(),
            "D_tide": nan33.copy(),
            "D_total": nan33.copy(),
            "D_total_vec": np.full(3, np.nan),
        }


# -----------------------------------------------------------------------------
# Shell-wise analysis wrappers
# -----------------------------------------------------------------------------


def analyze_shell(
    positions: ArrayLike,
    velocities: ArrayLike,
    masses: Optional[ArrayLike] = None,
    center: Optional[ArrayLike] = None,
    v_ref: Optional[ArrayLike] = None,
    mask: Optional[np.ndarray] = None,
    host_tidal_tensor: Optional[np.ndarray] = None,
    internal_hessian: Optional[np.ndarray] = None,
    min_particles: int = 10,
) -> Dict[str, Any]:
    """
    Analyze one particle subset/shell.

    Convention
    ----------
    kin["figure"] is the measured figure rotation from direct dI:

        Pi_measured = Pi(dI).

    kin["figure_affine"] stores the affine model decomposition:

        Pi_affine = Omega_hat + eta H_hat.
    """
    kin = compute_affine_kinematics(
        positions,
        velocities,
        masses=masses,
        center=center,
        v_ref=v_ref,
        mask=mask,
        min_particles=min_particles,
    )
    if not kin["valid"]:
        return kin

    fig_direct = figure_rotation_from_dI(kin["I"], kin["dI"])
    fig_affine = figure_rotation_from_affine(kin["I"], kin["A"])

    kin["figure"] = fig_direct
    kin["figure_affine"] = fig_affine

    T = None if host_tidal_tensor is None else symmetrize(host_tidal_tensor)
    Gint = None if internal_hessian is None else symmetrize(internal_hessian)
    terms = tidal_response_terms(kin["I"], kin["A"], S=kin["S"], G_int=Gint, T_ext=T)
    kin["response_terms"] = terms

    if T is not None:
        kin["torque_host"] = torque_from_hessian(kin["I"], T)
    if Gint is not None:
        kin["torque_internal"] = torque_from_hessian(kin["I"], Gint)

    return kin

def analyze_halo_shells(
    positions: ArrayLike,
    velocities: ArrayLike,
    masses: Optional[ArrayLike] = None,
    center: Optional[ArrayLike] = None,
    v_ref: Optional[ArrayLike] = None,
    shell_method: str = "radial",
    shell_masks: Optional[Sequence[np.ndarray]] = None,
    shell_kwargs: Optional[Dict[str, Any]] = None,
    host_tidal_tensor: Optional[Union[np.ndarray, Callable[[np.ndarray], Any]]] = None,
    internal_hessians: Optional[Sequence[np.ndarray]] = None,
    min_particles: int = 10,
) -> Dict[str, Any]:
    """
    Full shell-wise analysis for one halo/subhalo.

    This is the main high-level function for later TNG/Aquarius usage.
    It can use radial shells, binding-energy shells, or user-supplied masks.
    """
    pos = _as_2d_positions(positions)
    vel = _as_2d_positions(velocities, name="velocities")
    if vel.shape != pos.shape:
        raise ValueError("`velocities` must match `positions`")
    n = pos.shape[0]
    m = _as_1d_masses(masses, n)

    if center is None:
        cen = mass_weighted_mean(pos, m)
    else:
        cen = _as_vector(center, "center")

    if v_ref is None:
        v0 = mass_weighted_mean(vel, m)
    else:
        v0 = _as_vector(v_ref, "v_ref")

    if shell_masks is None:
        shell_kwargs = {} if shell_kwargs is None else dict(shell_kwargs)
        if shell_method.lower().strip() in ("subsets", "subset", "mask", "masks"):
            if shell_masks is None:
                raise ValueError("Provide `shell_masks` for shell_method='subsets'")
        masks, shell_info = make_shell_masks(
            pos,
            velocities=vel,
            masses=m,
            center=cen,
            v_ref=v0,
            method=shell_method,
            **shell_kwargs,
        )
    else:
        masks, shell_info = make_shell_masks(pos, method="subsets", subsets=shell_masks)

    T_host = None
    if host_tidal_tensor is not None:
        T_host = sample_tidal_hessian(cen, host_tidal_tensor)

    shells: List[Dict[str, Any]] = []
    for i, mi in enumerate(masks):
        Gint = None
        if internal_hessians is not None:
            Gint = np.asarray(internal_hessians[i], dtype=np.float64)
        out_i = analyze_shell(
            pos,
            vel,
            masses=m,
            center=cen,
            v_ref=v0,
            mask=mi,
            host_tidal_tensor=T_host,
            internal_hessian=Gint,
            min_particles=min_particles,
        )
        out_i["shell_index"] = i
        out_i["shell_mask"] = mi
        if "labels" in shell_info and i < len(shell_info["labels"]):
            out_i["label"] = shell_info["labels"][i]
        shells.append(out_i)

    return {
        "center": cen,
        "v_ref": v0,
        "shell_info": shell_info,
        "host_tidal_tensor": T_host,
        "shells": shells,
    }


def stack_shell_quantity(result: Dict[str, Any], key_path: Sequence[str], fill_value: float = np.nan) -> np.ndarray:
    """
    Extract and stack a nested quantity from `analyze_halo_shells` output.

    Example
    -------
    stack_shell_quantity(out, ["H"])                       -> (Nshell,3,3)
    stack_shell_quantity(out, ["figure", "varpi"])        -> (Nshell,3)
    stack_shell_quantity(out, ["response_terms", "D_tide"]) -> (Nshell,3,3)
    """
    vals: List[np.ndarray] = []
    for sh in result["shells"]:
        obj: Any = sh
        ok = True
        for k in key_path:
            if isinstance(obj, dict) and k in obj:
                obj = obj[k]
            else:
                ok = False
                break
        if ok:
            vals.append(np.asarray(obj, dtype=np.float64))
        else:
            vals.append(np.asarray(fill_value, dtype=np.float64))
    try:
        return np.stack(vals, axis=0)
    except Exception:
        return np.asarray(vals, dtype=object)


# -----------------------------------------------------------------------------
# Optional direct wrappers around existing shape.py, for compatibility
# -----------------------------------------------------------------------------


def shape_beta_fig_from_moments(I: np.ndarray, dI: np.ndarray, ddI: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
    """
    Use shape.beta_fig if available; otherwise raise a clear error.
    """
    if _shape_beta_fig is None:
        raise ImportError("shape.beta_fig is not available in the current Python path")
    return _shape_beta_fig(I, dI, ddI, **kwargs)


__all__ = [
    "radial_shell_masks",
    "binding_energy_shell_masks",
    "spherical_potential_from_radial_mass",
    "component_binding_energy",
    "binding_energy_mass_distribution",
    "component_binding_energy_profiles",
    "make_shell_masks",
    "moment_derivative_tensor",
    "compute_affine_kinematics",
    "measure_hessian_and_omega",
    "figure_rotation_from_dI",
    "figure_rotation_from_affine",
    "potential_hessian_direct",
    "compute_particle_potential_direct",
    "average_hessian_over_subset",
    "tidal_tensor_from_components",
    "sample_tidal_hessian",
    "torque_from_hessian",
    "tidal_response_terms",
    "analyze_shell",
    "analyze_halo_shells",
    "stack_shell_quantity",
    "shape_beta_fig_from_moments",
]
