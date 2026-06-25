"""Ordinary HOD measurements and lightweight pairwise IA validation statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ia_analysis.hod.catalog import HODCatalog, select_galaxy_sample, standardize_hod_catalog


@dataclass(frozen=True)
class HODMeasurement:
    """Binned central, satellite, and total halo occupation measurement."""

    mass_edges: np.ndarray
    mass_centers: np.ndarray
    n_halo: np.ndarray
    n_cen: np.ndarray
    n_sat: np.ndarray
    n_tot: np.ndarray
    mean_cen: np.ndarray
    mean_sat: np.ndarray
    mean_tot: np.ndarray
    var_tot: np.ndarray
    sample_label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _catalog(halos: Any, galaxies: Any | None) -> HODCatalog:
    if isinstance(halos, HODCatalog):
        return halos
    if galaxies is None:
        raise ValueError("galaxies are required unless a HODCatalog is supplied")
    return standardize_hod_catalog(halos, galaxies)


def _edges(mass: np.ndarray, mass_bins: int | Sequence[float]) -> np.ndarray:
    if np.isscalar(mass_bins):
        valid = mass[np.isfinite(mass) & (mass > 0.0)]
        return np.logspace(np.log10(valid.min()), np.log10(valid.max()), int(mass_bins) + 1)
    edges = np.asarray(mass_bins, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("mass_bins must be an integer or strictly increasing edges")
    return edges


def measure_hod(
    halos: Any,
    galaxies: Any | None = None,
    *,
    mass_bins: int | Sequence[float] = 10,
    sample_label: str | Sequence[str] | None = None,
    galaxy_mask: Sequence[bool] | None = None,
) -> HODMeasurement:
    """Measure central, satellite, total means and total occupation variance."""
    catalog = _catalog(halos, galaxies)
    selected = select_galaxy_sample(catalog.galaxies, sample_label, mask=galaxy_mask)
    halo_ids = catalog.halos["halo_id"].to_numpy()
    index = {value: i for i, value in enumerate(halo_ids)}
    central = np.zeros(len(halo_ids), dtype=int)
    satellite = np.zeros(len(halo_ids), dtype=int)
    for halo_id, is_central in zip(selected["halo_id"], selected["is_central"].astype(bool)):
        if halo_id in index:
            if is_central:
                central[index[halo_id]] += 1
            else:
                satellite[index[halo_id]] += 1
    total = central + satellite
    mass = catalog.halos["mass"].to_numpy(dtype=float)
    edges = _edges(mass, mass_bins)
    centers = np.sqrt(edges[:-1] * edges[1:])
    arrays = [np.zeros(centers.size, dtype=int) for _ in range(4)]
    n_halo, n_cen, n_sat, n_tot = arrays
    means = [np.full(centers.size, np.nan) for _ in range(4)]
    mean_cen, mean_sat, mean_tot, var_tot = means
    digit = np.digitize(mass, edges) - 1
    for i in range(centers.size):
        use = digit == i
        n_halo[i] = int(use.sum())
        if not np.any(use):
            continue
        n_cen[i], n_sat[i], n_tot[i] = central[use].sum(), satellite[use].sum(), total[use].sum()
        mean_cen[i], mean_sat[i], mean_tot[i] = central[use].mean(), satellite[use].mean(), total[use].mean()
        var_tot[i] = total[use].var()
    label = sample_label if isinstance(sample_label, str) else None
    return HODMeasurement(edges, centers, n_halo, n_cen, n_sat, n_tot, mean_cen, mean_sat, mean_tot, var_tot, label)


def measure_central_hod(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    measurement = measure_hod(*args, **kwargs)
    return measurement.mass_centers, measurement.mean_cen


def measure_satellite_hod(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    measurement = measure_hod(*args, **kwargs)
    return measurement.mass_centers, measurement.mean_sat


def measure_total_hod(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    measurement = measure_hod(*args, **kwargs)
    return measurement.mass_centers, measurement.mean_tot


def measure_occupation_distribution(
    halos: Any,
    galaxies: Any | None = None,
    *,
    mass_bins: int | Sequence[float] = 10,
    sample_label: str | None = None,
) -> pd.DataFrame:
    """Return P(N|M) as a tidy table."""
    catalog = _catalog(halos, galaxies)
    selected = select_galaxy_sample(catalog.galaxies, sample_label)
    counts = selected.groupby("halo_id").size().reindex(catalog.halos["halo_id"], fill_value=0).to_numpy()
    mass = catalog.halos["mass"].to_numpy(dtype=float)
    edges = _edges(mass, mass_bins)
    rows = []
    digit = np.digitize(mass, edges) - 1
    for i in range(edges.size - 1):
        values = counts[digit == i]
        if values.size == 0:
            continue
        unique, number = np.unique(values, return_counts=True)
        rows.extend(
            {"mass_low": edges[i], "mass_high": edges[i + 1], "occupation": int(n), "probability": count / values.size}
            for n, count in zip(unique, number)
        )
    return pd.DataFrame(rows)


def measure_satellite_fraction(halos: Any, galaxies: Any | None = None, **kwargs: Any) -> float:
    catalog = _catalog(halos, galaxies)
    selected = select_galaxy_sample(catalog.galaxies, kwargs.get("sample_label"))
    return float(selected["is_satellite"].astype(bool).mean()) if len(selected) else np.nan


def measure_number_density(
    halos: Any,
    galaxies: Any | None = None,
    *,
    volume: float,
    sample_label: str | None = None,
) -> float:
    if volume <= 0.0:
        raise ValueError("volume must be positive")
    catalog = _catalog(halos, galaxies)
    return float(len(select_galaxy_sample(catalog.galaxies, sample_label)) / volume)


def _pair_category(is_central_i: bool, is_central_j: bool, same_host: bool) -> tuple[str, str]:
    pair = "central-central" if is_central_i and is_central_j else (
        "satellite-satellite" if not is_central_i and not is_central_j else "central-satellite"
    )
    return pair, "1-halo" if same_host else "2-halo"


def measure_pairwise_ia(
    positions: Any,
    orientations: Any,
    rbins: Sequence[float],
    *,
    is_central: Any | None = None,
    host_id: Any | None = None,
    category: str = "all-all",
) -> dict[str, np.ndarray]:
    """Measure omega and eta with simple NumPy pair loops."""
    pos = np.asarray(positions, dtype=float)
    orient = np.asarray(orientations, dtype=float)
    orient /= np.maximum(np.linalg.norm(orient, axis=1)[:, None], 1.0e-30)
    edges = np.asarray(rbins, dtype=float)
    omega_sum = np.zeros(edges.size - 1)
    eta_sum = np.zeros(edges.size - 1)
    counts = np.zeros(edges.size - 1, dtype=int)
    central = np.zeros(len(pos), dtype=bool) if is_central is None else np.asarray(is_central, dtype=bool)
    hosts = np.arange(len(pos)) if host_id is None else np.asarray(host_id)
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            delta = pos[j] - pos[i]
            radius = np.linalg.norm(delta)
            index = np.searchsorted(edges, radius, side="right") - 1
            if index < 0 or index >= counts.size or radius == 0.0:
                continue
            pair, halo_pair = _pair_category(central[i], central[j], hosts[i] == hosts[j])
            if category not in {"all-all", pair, halo_pair}:
                continue
            rhat = delta / radius
            omega_sum[index] += 0.5 * ((np.dot(orient[i], rhat) ** 2) + (np.dot(orient[j], rhat) ** 2)) - 1.0 / 3.0
            eta_sum[index] += np.dot(orient[i], orient[j]) ** 2 - 1.0 / 3.0
            counts[index] += 1
    return {
        "rbins": edges,
        "rmid": 0.5 * (edges[:-1] + edges[1:]),
        "omega": np.divide(omega_sum, counts, out=np.full_like(omega_sum, np.nan), where=counts > 0),
        "eta": np.divide(eta_sum, counts, out=np.full_like(eta_sum, np.nan), where=counts > 0),
        "counts": counts,
    }


def measure_omega_ed(*args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
    return measure_pairwise_ia(*args, **kwargs)


def measure_eta_ee(*args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
    return measure_pairwise_ia(*args, **kwargs)


def measure_xi_gg(positions: Any, rbins: Sequence[float]) -> dict[str, np.ndarray]:
    """Return unnormalized pair counts as a minimal xi_gg validation summary."""
    result = measure_pairwise_ia(positions, np.ones((len(positions), 3)), rbins)
    return {"rbins": result["rbins"], "rmid": result["rmid"], "pair_counts": result["counts"]}


__all__ = [
    "HODMeasurement", "measure_hod", "measure_central_hod", "measure_satellite_hod",
    "measure_total_hod", "measure_occupation_distribution", "measure_satellite_fraction",
    "measure_number_density", "measure_xi_gg", "measure_omega_ed", "measure_eta_ee",
    "measure_pairwise_ia",
]
