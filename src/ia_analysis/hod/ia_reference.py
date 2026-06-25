"""Reference-vector bank for component-based intrinsic-alignment measurements."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

REFERENCE_NAMES = (
    "host_major_axis", "host_intermediate_axis", "host_minor_axis", "subhalo_major_axis",
    "subhalo_minor_axis", "radial_vector", "tidal_major_axis", "tidal_intermediate_axis",
    "tidal_minor_axis", "velocity_direction", "angular_momentum", "spin",
    "figure_rotation_axis", "binding_energy_layer_axis", "environment_axis", "custom",
)


def safe_unit_vector(vector: Any, *, fill_value: float = np.nan) -> np.ndarray:
    """Normalize one vector, returning a fill vector for zero/non-finite input."""
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,):
        raise ValueError("vector must have shape (3,)")
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 0.0:
        return np.full(3, fill_value, dtype=float)
    return vector / norm


def normalize_vectors(vectors: Any) -> np.ndarray:
    """Normalize an array of vectors and mark invalid rows as NaN."""
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim == 1:
        return safe_unit_vector(vectors)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("vectors must have shape (N, 3)")
    norms = np.linalg.norm(vectors, axis=1)
    output = np.full(vectors.shape, np.nan)
    valid = np.isfinite(vectors).all(axis=1) & (norms > 0.0)
    output[valid] = vectors[valid] / norms[valid, None]
    return output


def alignment_cos2_minus_one_third(orientations: Any, references: Any) -> np.ndarray:
    """Return axial alignment ``|e.q|^2 - 1/3``, invariant to either sign."""
    left = normalize_vectors(orientations)
    right = normalize_vectors(references)
    if left.ndim == 1:
        left = left[None, :]
    if right.ndim == 1:
        right = np.repeat(right[None, :], left.shape[0], axis=0)
    if left.shape != right.shape:
        raise ValueError("orientations and references must have matching vector shapes")
    return np.einsum("ij,ij->i", left, right) ** 2 - 1.0 / 3.0


def _column_vectors(table: Any, name: str) -> np.ndarray:
    frame = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
    values = frame[name]
    if len(values) == 0:
        return np.empty((0, 3))
    first = values.iloc[0]
    if np.ndim(first) == 0 and all(f"{name}_{axis}" in frame for axis in "xyz"):
        return frame[[f"{name}_{axis}" for axis in "xyz"]].to_numpy(dtype=float)
    return np.vstack(values.to_numpy()).astype(float)


def resolve_reference_vectors(
    table: Any,
    reference: str,
    *,
    custom: Any | None = None,
) -> np.ndarray:
    """Resolve one named reference from a standardized joined catalog."""
    frame = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
    key = str(reference)
    if key == "custom":
        if custom is None:
            raise ValueError("custom reference vectors are required")
        return normalize_vectors(custom)
    if key == "velocity_direction":
        return normalize_vectors(_column_vectors(frame, "velocity"))
    if key not in REFERENCE_NAMES:
        raise KeyError(f"Unknown IA reference {reference!r}")
    return normalize_vectors(_column_vectors(frame, key))


def build_reference_bank(
    table: Any,
    *,
    custom: Mapping[str, Any] | None = None,
    ignore_missing: bool = True,
) -> dict[str, np.ndarray]:
    """Build all available named reference-vector arrays."""
    bank: dict[str, np.ndarray] = {}
    for name in REFERENCE_NAMES:
        if name == "custom":
            continue
        try:
            bank[name] = resolve_reference_vectors(table, name)
        except (KeyError, ValueError):
            if not ignore_missing:
                raise
    for name, values in dict(custom or {}).items():
        bank[str(name)] = normalize_vectors(values)
    return bank


__all__ = [
    "REFERENCE_NAMES", "safe_unit_vector", "normalize_vectors",
    "alignment_cos2_minus_one_third", "resolve_reference_vectors", "build_reference_bank",
]
