"""Halo-occupation measurements for LRG/ELG-style catalog samples."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def normalize_host_indices(
    host_indices: np.ndarray,
    n_halo: int,
    *,
    index_base: int | str = 0,
    invalid_value: int = -1,
) -> np.ndarray:
    """Convert one- or zero-based host indices to validated zero-based indices."""
    host = np.asarray(host_indices, dtype=int).copy()
    if index_base == "auto":
        valid = host[host >= 0]
        index_base = 1 if valid.size and valid.min() >= 1 and valid.max() <= n_halo else 0
    host -= int(index_base)
    host[(host < 0) | (host >= int(n_halo))] = int(invalid_value)
    return host


def occupation_counts(
    halo_mass: np.ndarray,
    galaxy_host_index: np.ndarray,
    *,
    selected: np.ndarray | None = None,
    central: np.ndarray | None = None,
    environment: np.ndarray | None = None,
) -> pd.DataFrame:
    """Count selected central and satellite galaxies in every halo."""
    halo_mass = np.asarray(halo_mass, dtype=float)
    host = np.asarray(galaxy_host_index, dtype=int)
    if selected is None:
        selected = np.ones(host.size, dtype=bool)
    selected = np.asarray(selected, dtype=bool)
    if selected.shape != host.shape:
        raise ValueError("`selected` and `galaxy_host_index` must have matching shapes")
    valid = selected & (host >= 0) & (host < halo_mass.size)
    total = np.bincount(host[valid], minlength=halo_mass.size)
    if central is None:
        central_count = np.zeros(halo_mass.size, dtype=int)
    else:
        central = np.asarray(central, dtype=bool)
        if central.shape != host.shape:
            raise ValueError("`central` and `galaxy_host_index` must have matching shapes")
        central_count = np.bincount(host[valid & central], minlength=halo_mass.size)
    frame = pd.DataFrame(
        {
            "halo_index": np.arange(halo_mass.size),
            "halo_mass": halo_mass,
            "Ngal": total,
            "Ncen": central_count,
            "Nsat": total - central_count,
        }
    )
    if environment is not None:
        environment = np.asarray(environment, dtype=float)
        if environment.shape != halo_mass.shape:
            raise ValueError("`environment` must have one value per halo")
        frame["environment"] = environment
    return frame


def binned_hod(
    counts: pd.DataFrame,
    mass_bins: Sequence[float],
    *,
    components: Iterable[str] = ("Ngal", "Ncen", "Nsat"),
    minimum_haloes: int = 3,
) -> pd.DataFrame:
    """Measure mean occupation and standard error in logarithmic mass bins."""
    edges = np.asarray(mass_bins, dtype=float)
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("`mass_bins` must be strictly increasing")
    mass = counts["halo_mass"].to_numpy(dtype=float)
    bin_index = np.digitize(mass, edges) - 1
    rows: list[dict[str, float | int | str]] = []
    for index in range(edges.size - 1):
        mask = bin_index == index
        number = int(mask.sum())
        if number < int(minimum_haloes):
            continue
        for component in components:
            values = counts.loc[mask, component].to_numpy(dtype=float)
            rows.append(
                {
                    "component": component,
                    "mass_low": edges[index],
                    "mass_high": edges[index + 1],
                    "mass_center": np.sqrt(edges[index] * edges[index + 1]),
                    "n_halo": number,
                    "mean_occupation": float(np.mean(values)),
                    "occupation_std": float(np.std(values, ddof=1)) if number > 1 else 0.0,
                    "occupation_sem": float(np.std(values, ddof=1) / np.sqrt(number)) if number > 1 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def add_environment_quantiles(
    counts: pd.DataFrame,
    *,
    environment_column: str = "environment",
    quantiles: int = 4,
    output_column: str = "environment_quantile",
) -> pd.DataFrame:
    """Assign approximately equal-count environment quantiles."""
    output = counts.copy()
    valid = np.isfinite(output[environment_column].to_numpy(dtype=float))
    output[output_column] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    if valid.any():
        labels = pd.qcut(
            output.loc[valid, environment_column],
            q=int(quantiles),
            labels=False,
            duplicates="drop",
        )
        output.loc[valid, output_column] = labels.astype("Int64")
    return output


def binned_hod_by_environment(
    counts: pd.DataFrame,
    mass_bins: Sequence[float],
    *,
    environment_column: str = "environment_quantile",
    **kwargs: object,
) -> pd.DataFrame:
    """Measure HOD curves independently for each environment quantile."""
    frames: list[pd.DataFrame] = []
    for environment, panel in counts.groupby(environment_column, dropna=True):
        curve = binned_hod(panel, mass_bins, **kwargs)
        curve[environment_column] = environment
        frames.append(curve)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


__all__ = [
    "normalize_host_indices",
    "occupation_counts",
    "binned_hod",
    "add_environment_quantiles",
    "binned_hod_by_environment",
]
