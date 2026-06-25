"""Time-dependent shape-tensor response to a tidal field.

The model evolves a symmetric positive-definite shape tensor toward a
tidal-eigenframe equilibrium.  It is the reusable implementation of the
orbit-shape notebook model.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

KPC_PER_GYR_PER_KMS = 1.022712165045695


def sorted_eigh_symmetric(matrix: np.ndarray, *, descending: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decompose a symmetrized matrix with deterministic ordering."""
    values, vectors = np.linalg.eigh(0.5 * (np.asarray(matrix) + np.asarray(matrix).T))
    order = np.argsort(values)
    if descending:
        order = order[::-1]
    return values[order], vectors[:, order]


def project_spd(matrix: np.ndarray, *, floor: float = 1.0e-8) -> np.ndarray:
    """Project a numerical tensor onto the symmetric positive-definite cone."""
    values, vectors = np.linalg.eigh(0.5 * (np.asarray(matrix) + np.asarray(matrix).T))
    values = np.maximum(values, float(floor))
    return vectors @ np.diag(values) @ vectors.T


def acute_axis_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Return the acute angle between two sign-degenerate axes."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first /= max(float(np.linalg.norm(first)), 1.0e-30)
    second /= max(float(np.linalg.norm(second)), 1.0e-30)
    cosine = abs(float(np.dot(first, second)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def continuous_eigenframe(current: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    """Resolve eigenvector sign degeneracy relative to the previous frame."""
    output = np.asarray(current, dtype=float).copy()
    if previous is None:
        return output
    for column in range(output.shape[1]):
        if np.dot(output[:, column], previous[:, column]) < 0.0:
            output[:, column] *= -1.0
    return output


def equilibrium_shape_from_tide(
    tidal_tensor: np.ndarray,
    mass_fraction: float,
    *,
    base_axes: Sequence[float] = (1.0, 0.75, 0.55),
    mass_exponents: Sequence[float] = (0.10, 0.18, 0.30),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the instantaneous tidal-aligned equilibrium shape tensor."""
    _, eigenvectors = sorted_eigh_symmetric(tidal_tensor)
    fraction = float(np.clip(mass_fraction, 1.0e-4, 1.0))
    axes = np.asarray(base_axes, dtype=float) * fraction ** np.asarray(mass_exponents, dtype=float)
    tensor = eigenvectors @ np.diag(axes**2) @ eigenvectors.T
    return tensor, axes, eigenvectors


def angular_frequency_gyr_inverse(
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    physical_kpc_per_position_unit: float = 1.0,
) -> np.ndarray:
    """Estimate ``|r x v|/r^2`` and convert it to inverse Gyr."""
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    radius2 = np.einsum("ij,ij->i", positions, positions)
    angular = np.linalg.norm(np.cross(positions, velocities), axis=1)
    omega = angular / np.maximum(radius2, 1.0e-30)
    return omega * KPC_PER_GYR_PER_KMS / max(float(physical_kpc_per_position_unit), 1.0e-30)


def evolve_shape_tensor(
    time: np.ndarray,
    tidal_tensors: np.ndarray,
    mass_fraction: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    initial_tensor: np.ndarray,
    *,
    n_shape_orbits: float = 0.5,
    physical_kpc_per_position_unit: float = 1.0,
    base_axes: Sequence[float] = (1.0, 0.75, 0.55),
    mass_exponents: Sequence[float] = (0.10, 0.18, 0.30),
) -> dict[str, np.ndarray]:
    """Evolve ``dS/dt = -(S-S_eq)/tau`` with an exact per-step response."""
    time = np.asarray(time, dtype=float)
    tidal_tensors = np.asarray(tidal_tensors, dtype=float)
    mass_fraction = np.asarray(mass_fraction, dtype=float)
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    count = time.size
    if tidal_tensors.shape != (count, 3, 3):
        raise ValueError("`tidal_tensors` must have shape (N, 3, 3)")
    if positions.shape != (count, 3) or velocities.shape != (count, 3):
        raise ValueError("`positions` and `velocities` must have shape (N, 3)")
    if mass_fraction.shape != (count,):
        raise ValueError("`mass_fraction` must have shape (N,)")

    omega = angular_frequency_gyr_inverse(
        positions,
        velocities,
        physical_kpc_per_position_unit=physical_kpc_per_position_unit,
    )
    tensors = np.empty((count, 3, 3), dtype=float)
    axes_series = np.empty((count, 3), dtype=float)
    ratios = np.empty((count, 2), dtype=float)
    angles = np.empty(count, dtype=float)
    relaxation = np.empty(count, dtype=float)
    tensor = project_spd(initial_tensor)
    tide_previous = None
    shape_previous = None

    for index in range(count):
        equilibrium, _, tide_frame = equilibrium_shape_from_tide(
            tidal_tensors[index],
            mass_fraction[index],
            base_axes=base_axes,
            mass_exponents=mass_exponents,
        )
        tide_frame = continuous_eigenframe(tide_frame, tide_previous)
        tide_previous = tide_frame
        tau = float(n_shape_orbits) * 2.0 * np.pi / max(abs(float(omega[index])), 1.0e-6)
        relaxation[index] = tau
        if index:
            delta_time = float(time[index] - time[index - 1])
            response = 1.0 - np.exp(-delta_time / max(tau, 1.0e-30))
            tensor = project_spd(tensor + response * (equilibrium - tensor))
        values, shape_frame = sorted_eigh_symmetric(tensor)
        shape_frame = continuous_eigenframe(shape_frame, shape_previous)
        shape_previous = shape_frame
        axes = np.sqrt(np.maximum(values, 0.0))
        tensors[index] = tensor
        axes_series[index] = axes
        ratios[index] = axes[1:] / max(float(axes[0]), 1.0e-30)
        angles[index] = acute_axis_angle_deg(shape_frame[:, 0], tide_frame[:, 0])

    return {
        "S": tensors,
        "axes": axes_series,
        "q": ratios,
        "angle_major_tide_major_deg": angles,
        "tau_shape_gyr": relaxation,
        "omega_gyr_inv": omega,
    }


__all__ = [
    "sorted_eigh_symmetric",
    "project_spd",
    "acute_axis_angle_deg",
    "continuous_eigenframe",
    "equilibrium_shape_from_tide",
    "angular_frequency_gyr_inverse",
    "evolve_shape_tensor",
]
