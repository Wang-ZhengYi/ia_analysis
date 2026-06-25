"""Satellite phase-space, anisotropy, and binding-energy layer HOD statistics."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from ia_analysis.hod.catalog import HODCatalog, join_halo_galaxy_properties, select_galaxy_sample, standardize_hod_catalog
from ia_analysis.hod.ia_reference import alignment_cos2_minus_one_third, normalize_vectors


def _satellites(halos: Any, galaxies: Any | None, sample_label: str | None = None) -> pd.DataFrame:
    catalog = halos if isinstance(halos, HODCatalog) else standardize_hod_catalog(halos, galaxies)
    selected = select_galaxy_sample(catalog.galaxies, sample_label)
    selected = selected.loc[selected["is_satellite"].astype(bool)]
    return join_halo_galaxy_properties(catalog.halos, selected)


def measure_radial_profile_hod(
    halos: Any,
    galaxies: Any | None = None,
    *,
    radius_edges: Sequence[float] = np.linspace(0.0, 1.5, 16),
    mass_edges: Sequence[float] | None = None,
    sample_label: str | None = None,
) -> pd.DataFrame:
    """Measure mean satellite counts per halo in r/Rvir and optional mass bins."""
    satellites = _satellites(halos, galaxies, sample_label)
    if "r_over_rvir" not in satellites:
        raise KeyError("r_over_rvir is required")
    radius_edges = np.asarray(radius_edges, dtype=float)
    if mass_edges is None:
        mass_edges = [satellites["mass"].min() * 0.99, satellites["mass"].max() * 1.01]
    mass_edges = np.asarray(mass_edges, dtype=float)
    rows = []
    for mi in range(mass_edges.size - 1):
        mass_mask = (satellites["mass"] >= mass_edges[mi]) & (satellites["mass"] < mass_edges[mi + 1])
        panel = satellites.loc[mass_mask]
        n_hosts = max(panel["halo_id"].nunique(), 1)
        count, _ = np.histogram(panel["r_over_rvir"], bins=radius_edges)
        for ri, value in enumerate(count):
            rows.append({
                "mass_low": mass_edges[mi], "mass_high": mass_edges[mi + 1],
                "radius_low": radius_edges[ri], "radius_high": radius_edges[ri + 1],
                "n_satellite": int(value), "mean_per_halo": value / n_hosts,
            })
    return pd.DataFrame(rows)


def measure_velocity_anisotropy_hod(
    halos: Any,
    galaxies: Any | None = None,
    *,
    mass_edges: Sequence[float] | None = None,
    sample_label: str | None = None,
) -> pd.DataFrame:
    """Measure beta=1-(sigma_t1^2+sigma_t2^2)/(2 sigma_r^2) by halo mass."""
    satellites = _satellites(halos, galaxies, sample_label)
    required = {"velocity", "radial_vector", "mass"}
    if not required.issubset(satellites):
        raise KeyError(f"Required columns: {sorted(required)}")
    velocity = np.vstack(satellites["velocity"].to_numpy())
    radial = normalize_vectors(np.vstack(satellites["radial_vector"].to_numpy()))
    vr = np.einsum("ij,ij->i", velocity, radial)
    vt2 = np.sum(velocity * velocity, axis=1) - vr * vr
    mass = satellites["mass"].to_numpy(dtype=float)
    edges = np.asarray(mass_edges if mass_edges is not None else [mass.min() * 0.99, mass.max() * 1.01])
    rows = []
    for i in range(edges.size - 1):
        use = (mass >= edges[i]) & (mass < edges[i + 1])
        if not np.any(use):
            continue
        sigma_r2 = np.var(vr[use])
        beta = np.nan if sigma_r2 == 0.0 else 1.0 - np.mean(vt2[use]) / (2.0 * sigma_r2)
        rows.append({"mass_low": edges[i], "mass_high": edges[i + 1], "beta": beta, "count": int(use.sum())})
    return pd.DataFrame(rows)


def measure_binding_energy_layer_occupation(
    halos: Any,
    galaxies: Any | None = None,
    *,
    sample_label: str | None = None,
) -> pd.DataFrame:
    """Count satellites in named binding-energy layers by host halo."""
    satellites = _satellites(halos, galaxies, sample_label)
    if "binding_energy_layer" not in satellites:
        raise KeyError("binding_energy_layer is required")
    return (
        satellites.groupby(["halo_id", "binding_energy_layer"], dropna=False)
        .size()
        .rename("occupation")
        .reset_index()
    )


def measure_phase_space_hod(halos: Any, galaxies: Any | None = None, **kwargs: Any) -> dict[str, pd.DataFrame]:
    """Return radial, velocity-anisotropy, and binding-layer measurements."""
    return {
        "radial_profile": measure_radial_profile_hod(halos, galaxies, **{k: v for k, v in kwargs.items() if k in {"radius_edges", "mass_edges", "sample_label"}}),
        "velocity_anisotropy": measure_velocity_anisotropy_hod(halos, galaxies, **{k: v for k, v in kwargs.items() if k in {"mass_edges", "sample_label"}}),
        "binding_layers": measure_binding_energy_layer_occupation(halos, galaxies, sample_label=kwargs.get("sample_label")),
    }


def measure_phase_space_assembly_bias(
    halos: Any,
    galaxies: Any | None = None,
    *,
    quantile_column: str,
    sample_label: str | None = None,
) -> dict[Any, dict[str, pd.DataFrame]]:
    """Measure phase-space summaries separately for precomputed assembly quantiles."""
    catalog = halos if isinstance(halos, HODCatalog) else standardize_hod_catalog(halos, galaxies)
    output = {}
    for quantile, halo_panel in catalog.halos.groupby(quantile_column):
        ids = set(halo_panel["halo_id"])
        galaxy_panel = catalog.galaxies.loc[catalog.galaxies["halo_id"].isin(ids)]
        output[quantile] = measure_phase_space_hod(halo_panel, galaxy_panel, sample_label=sample_label)
    return output


def measure_host_axis_phase_space_alignment(
    positions: Any,
    host_major_axes: Any,
) -> float:
    """Measure satellite-position alignment with the host major axis."""
    values = alignment_cos2_minus_one_third(positions, host_major_axes)
    return float(np.nanmean(values))


__all__ = [
    "measure_radial_profile_hod", "measure_velocity_anisotropy_hod",
    "measure_binding_energy_layer_occupation", "measure_phase_space_hod",
    "measure_phase_space_assembly_bias", "measure_host_axis_phase_space_alignment",
]
