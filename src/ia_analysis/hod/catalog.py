"""Lightweight catalog adapters for HOD and component-based IA-HOD analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

HALO_COLUMNS = (
    "halo_id", "host_id", "mass", "rvir", "position", "velocity", "concentration",
    "environment", "tidal_anisotropy", "formation_time", "spin", "axis_ratio_ba",
    "axis_ratio_ca", "triaxiality", "host_major_axis", "host_intermediate_axis",
    "host_minor_axis", "tidal_major_axis", "tidal_intermediate_axis", "tidal_minor_axis",
)
GALAXY_COLUMNS = (
    "galaxy_id", "halo_id", "host_id", "is_central", "is_satellite", "sample_label",
    "stellar_mass", "sfr", "color", "position", "velocity", "orientation",
    "shape_major_axis", "shape_intermediate_axis", "shape_minor_axis", "subhalo_major_axis",
    "subhalo_minor_axis", "spin", "angular_momentum", "radial_vector", "r_over_rvir",
    "binding_energy", "binding_energy_layer", "figure_rotation_axis",
)


@dataclass(frozen=True)
class HODCatalog:
    """Standardized halo and galaxy tables used by this package."""

    halos: pd.DataFrame
    galaxies: pd.DataFrame
    metadata: Mapping[str, Any] | None = None


def _to_dataframe(table: Any, *, name: str) -> pd.DataFrame:
    """Convert a DataFrame, mapping, or structured array to a copied DataFrame."""
    if isinstance(table, pd.DataFrame):
        return table.copy()
    if isinstance(table, Mapping):
        return pd.DataFrame(dict(table))
    array = np.asarray(table)
    if array.dtype.names:
        return pd.DataFrame.from_records(array)
    if array.ndim == 2:
        return pd.DataFrame(array)
    raise TypeError(f"{name} must be a pandas DataFrame, mapping, or structured NumPy array")


def _rename_columns(frame: pd.DataFrame, columns: Mapping[str, str] | None) -> pd.DataFrame:
    """Rename input columns where the mapping is canonical-name to input-name."""
    if not columns:
        return frame
    rename = {source: canonical for canonical, source in columns.items() if source in frame}
    return frame.rename(columns=rename)


def infer_central_satellite_flags(
    galaxies: Any,
    *,
    galaxy_id: str = "galaxy_id",
    halo_id: str = "halo_id",
    host_id: str = "host_id",
    is_central: str = "is_central",
) -> pd.DataFrame:
    """Infer mutually exclusive central/satellite flags from available columns."""
    frame = _to_dataframe(galaxies, name="galaxies")
    if is_central in frame:
        central = frame[is_central].astype(bool).to_numpy()
    elif galaxy_id in frame and halo_id in frame:
        central = frame[galaxy_id].to_numpy() == frame[halo_id].to_numpy()
    elif halo_id in frame and host_id in frame:
        central = frame[halo_id].to_numpy() == frame[host_id].to_numpy()
    else:
        central = np.zeros(len(frame), dtype=bool)
        if halo_id in frame:
            first = ~frame[halo_id].duplicated(keep="first")
            central = first.to_numpy()
    frame["is_central"] = central
    frame["is_satellite"] = ~central
    return frame


def standardize_hod_catalog(
    halos: Any,
    galaxies: Any,
    *,
    halo_columns: Mapping[str, str] | None = None,
    galaxy_columns: Mapping[str, str] | None = None,
    infer_flags: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> HODCatalog:
    """Standardize input tables while preserving extra user-defined columns."""
    halo_frame = _rename_columns(_to_dataframe(halos, name="halos"), halo_columns)
    galaxy_frame = _rename_columns(_to_dataframe(galaxies, name="galaxies"), galaxy_columns)
    if infer_flags:
        galaxy_frame = infer_central_satellite_flags(galaxy_frame)
    if "host_id" not in halo_frame and "halo_id" in halo_frame:
        halo_frame["host_id"] = halo_frame["halo_id"]
    if "host_id" not in galaxy_frame and "halo_id" in galaxy_frame:
        galaxy_frame["host_id"] = galaxy_frame["halo_id"]
    if "sample_label" not in galaxy_frame:
        galaxy_frame["sample_label"] = "all"
    catalog = HODCatalog(halo_frame.reset_index(drop=True), galaxy_frame.reset_index(drop=True), dict(metadata or {}))
    validate_hod_catalog(catalog)
    return catalog


def validate_hod_catalog(catalog: HODCatalog | tuple[Any, Any]) -> None:
    """Validate required IDs, masses, flag lengths, and galaxy host links."""
    if not isinstance(catalog, HODCatalog):
        catalog = standardize_hod_catalog(*catalog)
    halos, galaxies = catalog.halos, catalog.galaxies
    missing_halo = {"halo_id", "mass"} - set(halos)
    missing_galaxy = {"galaxy_id", "halo_id", "is_central", "is_satellite"} - set(galaxies)
    if missing_halo:
        raise ValueError(f"Missing required halo columns: {sorted(missing_halo)}")
    if missing_galaxy:
        raise ValueError(f"Missing required galaxy columns: {sorted(missing_galaxy)}")
    if halos["halo_id"].duplicated().any():
        raise ValueError("halo_id values must be unique")
    mass = halos["mass"].to_numpy(dtype=float)
    if not np.all(np.isfinite(mass) & (mass > 0.0)):
        raise ValueError("halo masses must be positive and finite")
    if np.any(galaxies["is_central"].astype(bool) & galaxies["is_satellite"].astype(bool)):
        raise ValueError("galaxies cannot be both central and satellite")


def select_galaxy_sample(
    galaxies: Any,
    sample_label: str | Sequence[str] | None = None,
    *,
    mask: Sequence[bool] | None = None,
    sample_column: str = "sample_label",
) -> pd.DataFrame:
    """Select LRG, ELG, all, custom labels, and/or an explicit boolean mask."""
    frame = _to_dataframe(galaxies, name="galaxies")
    selected = np.ones(len(frame), dtype=bool)
    if sample_label is not None and str(sample_label).lower() != "all":
        labels = {str(sample_label)} if isinstance(sample_label, str) else {str(value) for value in sample_label}
        selected &= frame[sample_column].astype(str).isin(labels).to_numpy()
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != selected.shape:
            raise ValueError("sample mask must match the galaxy table length")
        selected &= mask_array
    return frame.loc[selected].reset_index(drop=True)


def join_halo_galaxy_properties(
    halos: Any,
    galaxies: Any,
    *,
    halo_id: str = "halo_id",
    galaxy_halo_id: str = "halo_id",
    halo_columns: Sequence[str] | None = None,
    suffix: str = "_halo",
) -> pd.DataFrame:
    """Attach selected halo properties to every galaxy by halo ID."""
    halo_frame = _to_dataframe(halos, name="halos")
    galaxy_frame = _to_dataframe(galaxies, name="galaxies")
    selected = [halo_id, *(halo_columns or [column for column in halo_frame if column != halo_id])]
    selected = list(dict.fromkeys(selected))
    return galaxy_frame.merge(
        halo_frame[selected],
        left_on=galaxy_halo_id,
        right_on=halo_id,
        how="left",
        suffixes=("", suffix),
        validate="many_to_one",
    )


__all__ = [
    "HALO_COLUMNS", "GALAXY_COLUMNS", "HODCatalog", "standardize_hod_catalog",
    "validate_hod_catalog", "infer_central_satellite_flags", "select_galaxy_sample",
    "join_halo_galaxy_properties",
]
