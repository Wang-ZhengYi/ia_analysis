"""Fixed-mass secondary-property splits and assembly-biased HOD measurements."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from ia_analysis.hod.catalog import HODCatalog, standardize_hod_catalog
from ia_analysis.hod.statistics import HODMeasurement, measure_hod


def standardize_secondary_property_at_fixed_mass(
    mass: Any,
    secondary: Any,
    *,
    mass_bins: int | Sequence[float] = 10,
) -> np.ndarray:
    """Remove the binned mean mass trend and scale residuals to unit variance."""
    mass = np.asarray(mass, dtype=float)
    secondary = np.asarray(secondary, dtype=float)
    if mass.shape != secondary.shape:
        raise ValueError("mass and secondary must have matching shapes")
    if np.isscalar(mass_bins):
        edges = np.linspace(np.nanmin(np.log10(mass)), np.nanmax(np.log10(mass)), int(mass_bins) + 1)
        coordinate = np.log10(mass)
    else:
        edges = np.asarray(mass_bins, dtype=float)
        coordinate = mass
    output = np.full(mass.shape, np.nan)
    digit = np.digitize(coordinate, edges) - 1
    for i in range(edges.size - 1):
        use = (digit == i) & np.isfinite(secondary)
        if not np.any(use):
            continue
        residual = secondary[use] - np.mean(secondary[use])
        scale = np.std(residual)
        output[use] = residual / (scale if scale > 0.0 else 1.0)
    return output


def split_by_secondary_property_quantiles(
    mass: Any,
    secondary: Any,
    *,
    quantiles: int = 2,
    mass_bins: int | Sequence[float] = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return integer fixed-mass quantile labels and standardized residuals."""
    standardized = standardize_secondary_property_at_fixed_mass(mass, secondary, mass_bins=mass_bins)
    valid = np.isfinite(standardized)
    labels = np.full(standardized.shape, -1, dtype=int)
    if np.any(valid):
        labels[valid] = pd.qcut(standardized[valid], q=int(quantiles), labels=False, duplicates="drop")
    return labels, standardized


def measure_assembly_hod(
    halos: Any,
    galaxies: Any | None = None,
    *,
    secondary_property: str,
    quantiles: int = 2,
    mass_bins: int | Sequence[float] = 10,
    sample_label: str | None = None,
) -> dict[str, Any]:
    """Measure HOD in fixed-mass secondary-property quantiles."""
    catalog = halos if isinstance(halos, HODCatalog) else standardize_hod_catalog(halos, galaxies)
    labels, standardized = split_by_secondary_property_quantiles(
        catalog.halos["mass"], catalog.halos[secondary_property], quantiles=quantiles, mass_bins=mass_bins
    )
    measurements: dict[str, HODMeasurement] = {}
    halo_ids = catalog.halos["halo_id"].to_numpy()
    for quantile in sorted(set(labels[labels >= 0])):
        chosen_ids = set(halo_ids[labels == quantile])
        halo_subset = catalog.halos.loc[labels == quantile]
        galaxy_subset = catalog.galaxies.loc[catalog.galaxies["halo_id"].isin(chosen_ids)]
        measurements[f"q{quantile}"] = measure_hod(
            halo_subset, galaxy_subset, mass_bins=mass_bins, sample_label=sample_label
        )
    comparison = {}
    if len(measurements) >= 2:
        low = measurements[sorted(measurements)[0]]
        high = measurements[sorted(measurements)[-1]]
        comparison = {
            "ratio_high_low": np.divide(high.mean_tot, low.mean_tot, out=np.full_like(high.mean_tot, np.nan), where=low.mean_tot != 0),
            "difference_high_low": high.mean_tot - low.mean_tot,
        }
    return {"measurements": measurements, "quantile": labels, "standardized_secondary": standardized, **comparison}


__all__ = [
    "standardize_secondary_property_at_fixed_mass", "split_by_secondary_property_quantiles",
    "measure_assembly_hod",
]
