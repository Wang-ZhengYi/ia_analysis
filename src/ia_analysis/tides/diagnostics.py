"""Array-level diagnostics for tidal-tensor time series."""

from __future__ import annotations

import numpy as np


def tidal_eigenvalues(tensors: np.ndarray, *, descending: bool = True) -> np.ndarray:
    """Return ordered eigenvalues for one tensor or a tensor series."""
    tensors = np.asarray(tensors, dtype=float)
    symmetric = 0.5 * (tensors + np.swapaxes(tensors, -1, -2))
    values = np.linalg.eigvalsh(symmetric)
    return values[..., ::-1] if descending else values


def tidal_strength(tensors: np.ndarray) -> np.ndarray:
    """Return the maximum absolute tidal eigenvalue."""
    return np.max(np.abs(tidal_eigenvalues(tensors)), axis=-1)


def tidal_anisotropy(tensors: np.ndarray) -> np.ndarray:
    """Return eigenvalue range divided by maximum absolute eigenvalue."""
    values = tidal_eigenvalues(tensors)
    scale = np.maximum(np.max(np.abs(values), axis=-1), 1.0e-30)
    return np.ptp(values, axis=-1) / scale


def summarize_tidal_series(tensors: np.ndarray) -> dict[str, float]:
    """Return compact peak and median diagnostics for a tensor series."""
    strength = tidal_strength(tensors)
    anisotropy = tidal_anisotropy(tensors)
    return {
        "maximum_tidal_strength": float(np.nanmax(strength)),
        "median_tidal_strength": float(np.nanmedian(strength)),
        "maximum_tidal_anisotropy": float(np.nanmax(anisotropy)),
        "median_tidal_anisotropy": float(np.nanmedian(anisotropy)),
    }


__all__ = ["tidal_eigenvalues", "tidal_strength", "tidal_anisotropy", "summarize_tidal_series"]
