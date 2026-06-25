"""Component-level intrinsic-alignment measurements for HOD catalogs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ia_analysis.hod.catalog import HODCatalog, join_halo_galaxy_properties, select_galaxy_sample, standardize_hod_catalog
from ia_analysis.hod.ia_reference import alignment_cos2_minus_one_third, resolve_reference_vectors


@dataclass(frozen=True)
class IAComponentMeasurement:
    """Binned measurement for one physical IA reference component."""

    component: str
    reference: str
    population: str
    sample_label: str | None
    mass_edges: np.ndarray | None
    radius_edges: np.ndarray | None
    secondary_edges: np.ndarray | None
    layer_labels: tuple[str, ...] | None
    values: np.ndarray
    errors: np.ndarray | None = None
    counts: np.ndarray | None = None
    covariance: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _joined(halos: Any, galaxies: Any | None, sample_label: str | None, population: str) -> pd.DataFrame:
    catalog = halos if isinstance(halos, HODCatalog) else standardize_hod_catalog(halos, galaxies)
    selected = select_galaxy_sample(catalog.galaxies, sample_label)
    if population == "central":
        selected = selected.loc[selected["is_central"].astype(bool)]
    elif population == "satellite":
        selected = selected.loc[selected["is_satellite"].astype(bool)]
    elif population != "all":
        raise ValueError("population must be 'central', 'satellite', or 'all'")
    return join_halo_galaxy_properties(catalog.halos, selected)


def _vectors(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise KeyError(f"Missing vector column {column!r}")
    return np.vstack(frame[column].to_numpy()).astype(float)


def measure_reference_alignment(
    orientations: Any,
    references: Any,
    *,
    weights: Any | None = None,
) -> dict[str, float]:
    """Measure mean axial alignment, standard error, and valid count."""
    values = alignment_cos2_minus_one_third(orientations, references)
    valid = np.isfinite(values)
    if not np.any(valid):
        return {"value": np.nan, "error": np.nan, "count": 0}
    if weights is None:
        mean = np.mean(values[valid])
    else:
        weight = np.asarray(weights, dtype=float)[valid]
        mean = np.average(values[valid], weights=weight)
    error = np.std(values[valid], ddof=1) / np.sqrt(valid.sum()) if valid.sum() > 1 else 0.0
    return {"value": float(mean), "error": float(error), "count": int(valid.sum())}


def measure_alignment_hod_components(
    halos: Any,
    galaxies: Any | None = None,
    *,
    component: str,
    reference: str,
    orientation_column: str = "shape_major_axis",
    population: str = "all",
    sample_label: str | None = None,
    mass_edges: Sequence[float] | None = None,
    radius_edges: Sequence[float] | None = None,
    secondary_column: str | None = None,
    secondary_edges: Sequence[float] | None = None,
    layer_column: str | None = None,
) -> IAComponentMeasurement:
    """Measure one alignment component on an optional mass/radius/secondary/layer grid."""
    frame = _joined(halos, galaxies, sample_label, population)
    orientation = _vectors(frame, orientation_column)
    reference_vectors = resolve_reference_vectors(frame, reference)
    alignment = alignment_cos2_minus_one_third(orientation, reference_vectors)
    axes: list[np.ndarray] = []
    coordinates: list[np.ndarray] = []
    if mass_edges is not None:
        axes.append(np.asarray(mass_edges, dtype=float))
        coordinates.append(frame["mass"].to_numpy(dtype=float))
    if radius_edges is not None:
        axes.append(np.asarray(radius_edges, dtype=float))
        coordinates.append(frame["r_over_rvir"].to_numpy(dtype=float))
    if secondary_edges is not None:
        if secondary_column is None:
            raise ValueError("secondary_column is required with secondary_edges")
        axes.append(np.asarray(secondary_edges, dtype=float))
        coordinates.append(frame[secondary_column].to_numpy(dtype=float))
    layer_labels = None
    if layer_column is not None:
        layer_labels = tuple(str(value) for value in pd.unique(frame[layer_column].dropna()))
        axes.append(np.arange(len(layer_labels) + 1, dtype=float) - 0.5)
        mapping = {label: i for i, label in enumerate(layer_labels)}
        coordinates.append(frame[layer_column].astype(str).map(mapping).to_numpy(dtype=float))
    if not axes:
        summary = measure_reference_alignment(orientation, reference_vectors)
        values = np.asarray(summary["value"])
        errors = np.asarray(summary["error"])
        counts = np.asarray(summary["count"])
    else:
        shape = tuple(len(edges) - 1 for edges in axes)
        values = np.full(shape, np.nan)
        errors = np.full(shape, np.nan)
        counts = np.zeros(shape, dtype=int)
        digitized = [np.digitize(values_i, edges) - 1 for values_i, edges in zip(coordinates, axes)]
        for index in np.ndindex(shape):
            use = np.ones(len(frame), dtype=bool)
            for dimension, bin_index in enumerate(index):
                use &= digitized[dimension] == bin_index
            valid = use & np.isfinite(alignment)
            counts[index] = valid.sum()
            if counts[index]:
                values[index] = np.mean(alignment[valid])
                errors[index] = np.std(alignment[valid], ddof=1) / np.sqrt(counts[index]) if counts[index] > 1 else 0.0
    return IAComponentMeasurement(
        component, reference, population, sample_label,
        None if mass_edges is None else np.asarray(mass_edges),
        None if radius_edges is None else np.asarray(radius_edges),
        None if secondary_edges is None else np.asarray(secondary_edges),
        layer_labels, values, errors, counts,
    )


def _component_wrapper(reference: str, population: str, component: str):
    def measure(halos: Any, galaxies: Any | None = None, **kwargs: Any) -> IAComponentMeasurement:
        return measure_alignment_hod_components(
            halos, galaxies, reference=reference, population=population, component=component, **kwargs
        )
    return measure


measure_central_host_alignment = _component_wrapper("host_major_axis", "central", "central_host")
measure_central_tidal_alignment = _component_wrapper("tidal_major_axis", "central", "central_tidal")
measure_satellite_radial_alignment = _component_wrapper("radial_vector", "satellite", "satellite_radial")
measure_satellite_host_alignment = _component_wrapper("host_major_axis", "satellite", "satellite_host")
measure_satellite_subhalo_alignment = _component_wrapper("subhalo_major_axis", "satellite", "satellite_subhalo")
measure_satellite_tidal_alignment = _component_wrapper("tidal_major_axis", "satellite", "satellite_tidal")
measure_satellite_velocity_alignment = _component_wrapper("velocity_direction", "satellite", "satellite_velocity")
measure_satellite_spin_alignment = _component_wrapper("spin", "satellite", "satellite_spin")
measure_binding_layer_alignment = _component_wrapper("binding_energy_layer_axis", "satellite", "binding_layer")
measure_figure_rotation_alignment = _component_wrapper("figure_rotation_axis", "all", "figure_rotation")


def measure_mass_radius_alignment_grid(halos: Any, galaxies: Any | None = None, **kwargs: Any) -> IAComponentMeasurement:
    if "mass_edges" not in kwargs or "radius_edges" not in kwargs:
        raise ValueError("mass_edges and radius_edges are required")
    return measure_alignment_hod_components(halos, galaxies, **kwargs)


def measure_sample_dependent_ia_hod(
    halos: Any,
    galaxies: Any | None = None,
    *,
    sample_labels: Sequence[str] = ("LRG", "ELG"),
    **kwargs: Any,
) -> dict[str, IAComponentMeasurement]:
    return {label: measure_alignment_hod_components(halos, galaxies, sample_label=label, **kwargs) for label in sample_labels}


def measure_assembly_dependent_ia_hod(
    halos: Any,
    galaxies: Any | None = None,
    *,
    secondary_column: str,
    secondary_edges: Sequence[float],
    **kwargs: Any,
) -> IAComponentMeasurement:
    return measure_alignment_hod_components(
        halos, galaxies, secondary_column=secondary_column, secondary_edges=secondary_edges, **kwargs
    )


__all__ = [
    "IAComponentMeasurement", "measure_reference_alignment", "measure_alignment_hod_components",
    "measure_central_host_alignment", "measure_central_tidal_alignment",
    "measure_satellite_radial_alignment", "measure_satellite_host_alignment",
    "measure_satellite_subhalo_alignment", "measure_satellite_tidal_alignment",
    "measure_satellite_velocity_alignment", "measure_satellite_spin_alignment",
    "measure_binding_layer_alignment", "measure_figure_rotation_alignment",
    "measure_mass_radius_alignment_grid", "measure_sample_dependent_ia_hod",
    "measure_assembly_dependent_ia_hod",
]
