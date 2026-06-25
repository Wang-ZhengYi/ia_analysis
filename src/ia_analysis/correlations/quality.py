"""Quality diagnostics for persisted correlation-function products."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


def discover_correlation_files(
    root: str | Path,
    *,
    patterns: Iterable[str] = ("*.hdf5", "*.h5"),
    recursive: bool = True,
) -> list[Path]:
    """Return sorted HDF5 correlation products."""
    root = Path(root).expanduser()
    paths: set[Path] = set()
    for pattern in patterns:
        finder = root.rglob if recursive else root.glob
        paths.update(path for path in finder(pattern) if path.is_file())
    return sorted(paths)


def covariance_diagnostics(covariance: np.ndarray) -> dict[str, float]:
    """Return finite fraction, symmetry error, rank, and condition diagnostics."""
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("`covariance` must be a square matrix")
    finite = np.isfinite(covariance)
    clean = np.where(finite, covariance, 0.0)
    symmetric = 0.5 * (clean + clean.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    positive = eigenvalues[eigenvalues > 0.0]
    condition = np.inf if positive.size < 2 else float(positive.max() / positive.min())
    scale = max(float(np.nanmax(np.abs(clean))), 1.0e-30)
    return {
        "size": float(covariance.shape[0]),
        "finite_fraction": float(finite.mean()),
        "symmetry_relative_error": float(np.nanmax(np.abs(clean - clean.T)) / scale),
        "rank": float(np.linalg.matrix_rank(symmetric)),
        "minimum_eigenvalue": float(eigenvalues.min()),
        "condition_number_positive": condition,
    }


def signal_to_noise(values: np.ndarray, covariance: np.ndarray) -> float:
    """Return ``sqrt(x^T C^+ x)`` using a stable pseudo-inverse."""
    values = np.asarray(values, dtype=float).reshape(-1)
    covariance = np.asarray(covariance, dtype=float)
    valid = np.isfinite(values) & np.isfinite(np.diag(covariance))
    if not np.any(valid):
        return float("nan")
    vector = values[valid]
    matrix = covariance[np.ix_(valid, valid)]
    return float(np.sqrt(max(vector @ np.linalg.pinv(matrix, hermitian=True) @ vector, 0.0)))


def inspect_correlation_file(path: str | Path) -> pd.DataFrame:
    """Inventory mean/covariance pairs in one HDF5 correlation product."""
    rows: list[dict[str, object]] = []
    with h5py.File(path, "r") as handle:
        datasets: dict[str, np.ndarray] = {}

        def collect(name: str, obj: object) -> None:
            if isinstance(obj, h5py.Dataset):
                datasets[name] = np.asarray(obj)

        handle.visititems(collect)
        for name, covariance in datasets.items():
            if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
                continue
            lower = name.lower()
            if "cov" not in lower:
                continue
            candidate_names = (
                name.replace("covariance", "mean"),
                name.replace("cov", "mean"),
                name.rsplit("/", 1)[0] + "/mean" if "/" in name else "mean",
            )
            mean_name = next(
                (candidate for candidate in candidate_names if candidate in datasets and datasets[candidate].size == covariance.shape[0]),
                None,
            )
            diagnostics = covariance_diagnostics(covariance)
            rows.append(
                {
                    "path": str(Path(path).resolve()),
                    "covariance_dataset": name,
                    "mean_dataset": mean_name,
                    "signal_to_noise": np.nan if mean_name is None else signal_to_noise(datasets[mean_name], covariance),
                    **diagnostics,
                }
            )
    return pd.DataFrame(rows)


def summarize_correlation_quality(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Combine covariance diagnostics from several correlation files."""
    frames = [inspect_correlation_file(path) for path in paths]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


__all__ = [
    "discover_correlation_files",
    "covariance_diagnostics",
    "signal_to_noise",
    "inspect_correlation_file",
    "summarize_correlation_quality",
]
