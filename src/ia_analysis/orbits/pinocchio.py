"""Adapters for Pinocchio-like 2LPT merger-tree products.

Purpose
-------
Pinocchio and similar fast 2LPT mock tools can provide approximate group merger
trees and phase-space histories.  This module defines a small adapter layer that
converts already-loaded tables into the orbit-template containers used by
``ia_analysis.orbits.template_orbits``.

Provides
--------
- Column-map configuration for Pinocchio-like group and subhalo tables.
- Table-to-track conversion without imposing a single file schema.
- Group-subhalo template library construction from host links.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ia_analysis.orbits.template_orbits import OrbitTemplateLibrary, TreeTrack, build_template_library


@dataclass(frozen=True)
class PinocchioColumnMap:
    """Column names used to extract tracks from Pinocchio-like tables."""

    object_id: str = "id"
    host_id: str = "host_id"
    snapshot: str = "snapshot"
    x: str = "x"
    y: str = "y"
    z: str = "z"
    vx: str = "vx"
    vy: str = "vy"
    vz: str = "vz"
    mass: str = "mass"
    scale_factor: str | None = "scale_factor"


def read_pinocchio_table(
    path: str | Path,
    *,
    format: str = "auto",
    columns: Sequence[str] | None = None,
) -> Any:
    """Read a lightweight CSV or ASCII table without assuming a tree schema.

    The returned pandas ``DataFrame`` can be passed directly to
    :func:`tracks_from_table`.  ``PinocchioColumnMap`` remains responsible for
    mapping project-specific column names to orbit fields.
    """
    import pandas as pd

    table_path = Path(path)
    if not table_path.is_file():
        raise FileNotFoundError(f"Pinocchio-like table does not exist: {table_path}")
    mode = str(format).strip().lower()
    if mode == "auto":
        suffix = table_path.suffix.lower()
        mode = {".csv": "csv", ".tsv": "tsv"}.get(suffix, "ascii")
    read_options: dict[str, Any] = {"usecols": None if columns is None else list(columns)}
    if mode == "csv":
        return pd.read_csv(table_path, **read_options)
    if mode == "tsv":
        return pd.read_csv(table_path, sep="\t", **read_options)
    if mode in {"ascii", "txt", "whitespace"}:
        return pd.read_csv(table_path, sep=r"\s+", comment="#", **read_options)
    raise ValueError("format must be 'auto', 'csv', 'tsv', or 'ascii'")


def _column(table: Any, name: str) -> np.ndarray:
    """Read one column from a pandas-like table or mapping."""
    values = table[name]
    return values.to_numpy() if hasattr(values, "to_numpy") else np.asarray(values)


def _unique(values: Sequence[Any]) -> list[Any]:
    """Return stable unique values from a one-dimensional array."""
    out = []
    seen = set()
    for value in values:
        key = value.item() if hasattr(value, "item") else value
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def tracks_from_table(
    table: Any,
    *,
    columns: PinocchioColumnMap = PinocchioColumnMap(),
    metadata: Mapping[str, Any] | None = None,
) -> dict[int | str, TreeTrack]:
    """Convert one table of object histories into ``TreeTrack`` objects."""
    ids = _column(table, columns.object_id)
    tracks: dict[int | str, TreeTrack] = {}
    for object_id in _unique(ids):
        mask = ids == object_id
        snapshots = _column(table, columns.snapshot)[mask]
        order = np.argsort(snapshots)
        positions = np.column_stack(
            (
                _column(table, columns.x)[mask],
                _column(table, columns.y)[mask],
                _column(table, columns.z)[mask],
            )
        )[order]
        velocities = np.column_stack(
            (
                _column(table, columns.vx)[mask],
                _column(table, columns.vy)[mask],
                _column(table, columns.vz)[mask],
            )
        )[order]
        mass = _column(table, columns.mass)[mask][order] if columns.mass else None
        scale = None
        if columns.scale_factor is not None and columns.scale_factor in table:
            scale = _column(table, columns.scale_factor)[mask][order]
        tracks[object_id] = TreeTrack(
            object_id=object_id,
            snapshots=snapshots[order],
            positions=positions,
            velocities=velocities,
            mass=mass,
            scale_factor=scale,
            metadata=dict(metadata or {}),
        )
    return tracks


def host_map_from_table(
    subhalo_table: Any,
    *,
    columns: PinocchioColumnMap = PinocchioColumnMap(),
) -> dict[int | str, int | str]:
    """Return a subhalo-to-host map from a Pinocchio-like table."""
    sub_ids = _column(subhalo_table, columns.object_id)
    host_ids = _column(subhalo_table, columns.host_id)
    out: dict[int | str, int | str] = {}
    for sub_id in _unique(sub_ids):
        mask = sub_ids == sub_id
        candidates = host_ids[mask]
        if candidates.size == 0:
            continue
        values, counts = np.unique(candidates, return_counts=True)
        out[sub_id] = values[int(np.argmax(counts))]
    return out


def build_pinocchio_template_library(
    group_table: Any,
    subhalo_table: Any,
    *,
    columns: PinocchioColumnMap = PinocchioColumnMap(),
    boxsize: float | Sequence[float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> OrbitTemplateLibrary:
    """Build an orbit-template library from Pinocchio-like group/subhalo tables."""
    group_tracks = tracks_from_table(group_table, columns=columns, metadata={"kind": "group"})
    subhalo_tracks = tracks_from_table(subhalo_table, columns=columns, metadata={"kind": "subhalo"})
    host_map = host_map_from_table(subhalo_table, columns=columns)
    return build_template_library(
        group_tracks,
        subhalo_tracks,
        host_map,
        boxsize=boxsize,
        metadata={"source": "pinocchio_like_tables", **dict(metadata or {})},
    )


__all__ = [
    "PinocchioColumnMap",
    "read_pinocchio_table",
    "tracks_from_table",
    "host_map_from_table",
    "build_pinocchio_template_library",
]
