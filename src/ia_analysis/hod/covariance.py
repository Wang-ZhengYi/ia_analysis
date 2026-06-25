"""Small empirical covariance utilities for HOD and IA measurements."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def diagonal_covariance(errors: Any, *, floor: float = 0.0) -> np.ndarray:
    error = np.maximum(np.asarray(errors, dtype=float), float(floor))
    return np.diag(error.ravel() ** 2)


def regularize_covariance(covariance: Any, *, shrinkage: float = 0.05, floor: float = 1.0e-12) -> np.ndarray:
    """Shrink covariance toward its diagonal and apply an eigenvalue floor."""
    covariance = np.asarray(covariance, dtype=float)
    symmetric = 0.5 * (covariance + covariance.T)
    diagonal = np.diag(np.diag(symmetric))
    regularized = (1.0 - shrinkage) * symmetric + shrinkage * diagonal
    values, vectors = np.linalg.eigh(regularized)
    values = np.maximum(values, floor)
    return vectors @ np.diag(values) @ vectors.T


def bootstrap_measurement(
    data: Any,
    statistic: Callable[[np.ndarray], Any],
    *,
    n_resamples: int = 256,
    random_state: int | None = None,
) -> dict[str, np.ndarray]:
    array = np.asarray(data)
    rng = np.random.default_rng(random_state)
    samples = []
    for _ in range(int(n_resamples)):
        take = rng.integers(0, len(array), len(array))
        samples.append(np.asarray(statistic(array[take]), dtype=float).ravel())
    samples = np.asarray(samples)
    return {"mean": np.mean(samples, axis=0), "covariance": np.cov(samples, rowvar=False), "samples": samples}


def jackknife_measurement(
    data: Any,
    statistic: Callable[[np.ndarray], Any],
) -> dict[str, np.ndarray]:
    array = np.asarray(data)
    samples = np.asarray([np.asarray(statistic(np.delete(array, i, axis=0)), dtype=float).ravel() for i in range(len(array))])
    mean = np.mean(samples, axis=0)
    delta = samples - mean
    covariance = (len(array) - 1.0) / len(array) * delta.T @ delta
    return {"mean": mean, "covariance": covariance, "samples": samples}


__all__ = ["diagonal_covariance", "bootstrap_measurement", "jackknife_measurement", "regularize_covariance"]
