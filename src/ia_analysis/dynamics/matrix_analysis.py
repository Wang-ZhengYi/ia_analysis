"""Matrix-level helpers for snapshot target analysis.

This module keeps pure tensor algebra separate from particle selection,
catalog I/O, and plotting.  It wraps the historical ``halo_dynamics`` matrix
primitives under names that make their role explicit in snapshot workflows.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ia_analysis.dynamics import halo_dynamics as hd


def symmetrize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric part of one 3x3 matrix."""
    return hd.symmetrize(np.asarray(matrix, dtype=np.float64))


def sorted_eigensystem(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Return descending eigenvalues/eigenvectors for a symmetric matrix."""
    evals, evecs = hd.eigh_sorted_desc(matrix)
    return {"evals": evals, "evecs": evecs}


def measure_shape_tensor(
    positions: np.ndarray,
    masses: Optional[np.ndarray] = None,
    *,
    normalize_mass: bool = False,
) -> np.ndarray:
    """Compute ``I_ij = sum m x_i x_j`` for already-relative positions."""
    return hd.shape_tensor(positions, masses=masses, normalize_mass=normalize_mass)


def measure_mixed_moment(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute ``P_ij = sum m x_i u_j`` for relative phase-space coordinates."""
    return hd.mixed_moment(positions, velocities, masses=masses)


def measure_moment_derivative(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the direct particle-level first derivative of the shape tensor."""
    return hd.moment_derivative_tensor(positions, velocities, masses=masses)


def coerce_tidal_hessian(position: np.ndarray, tidal_source: Any) -> np.ndarray:
    """Sample or coerce a tidal Hessian at one position."""
    return hd.sample_tidal_hessian(position, tidal_source)


def tidal_hessian_from_components(components: np.ndarray) -> np.ndarray:
    """Convert ``[Txx, Txy, Txz, Tyy, Tyz, Tzz]`` to a symmetric Hessian."""
    return hd.tidal_tensor_from_components(components)


def tidal_stretch_eigensystem(hessian: np.ndarray) -> dict[str, np.ndarray]:
    """
    Return eigenvectors ordered by stretching strength.

    ``halo_dynamics`` Hessians use the gravitational-potential convention
    ``H = d_i d_j Phi``.  Tidal acceleration is ``a = -grad Phi``, so the local
    differential acceleration tensor is ``-H``.  The major stretching direction
    is therefore the largest eigenvalue of ``-H``.
    """
    stretch = -symmetrize_matrix(hessian)
    evals, evecs = hd.eigh_sorted_desc(stretch)
    return {"stretch_tensor": stretch, "stretch_evals": evals, "stretch_evecs": evecs}


def torque_from_hessian(shape_tensor: np.ndarray, hessian: np.ndarray) -> np.ndarray:
    """Compute tidal torque from a shape tensor and potential Hessian."""
    return hd.torque_from_hessian(shape_tensor, hessian)


def stack_shell_quantity(result: dict[str, Any], key_path: list[str] | tuple[str, ...], fill_value: float = np.nan) -> np.ndarray:
    """Stack a nested shell quantity from ``analyze_halo_shells`` output."""
    return hd.stack_shell_quantity(result, key_path, fill_value=fill_value)


__all__ = [
    "symmetrize_matrix",
    "sorted_eigensystem",
    "measure_shape_tensor",
    "measure_mixed_moment",
    "measure_moment_derivative",
    "coerce_tidal_hessian",
    "tidal_hessian_from_components",
    "tidal_stretch_eigensystem",
    "torque_from_hessian",
    "stack_shell_quantity",
]
