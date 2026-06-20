"""Merger-tree readers and snapshot-track selectors.

Purpose
-------
This module is the cross-time IO layer.  It loads merger-tree branches, converts
tree dictionaries to tables, selects requested snapshots, and builds matched
target/reference tracks.  It delegates catalog-specific work to
``ia_analysis.catalogs`` and compatibility helpers in ``ia_analysis.dynamics``.

Provides
--------
- SubLink main-progenitor branch loading.
- Generic TNG merger-tree loading through ``TNGCatalog.loadMergerTree``.
- Track tables for target subhaloes and optional host-central references.
- A small host-central lookup helper used by cross-time workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_TREE_FIELDS = (
    "SnapNum",
    "SubfindID",
    "SubhaloID",
    "SubhaloGrNr",
    "SubhaloMass",
    "SubhaloMassType",
    "SubhaloPos",
    "SubhaloVel",
)


@dataclass(frozen=True)
class MergerTrackConfig:
    """Configuration for reading one tracked subhalo branch."""

    base_path: str | Path
    snap0: int
    subhalo_id0: int
    snap_track: tuple[int, ...]
    fields: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_TREE_FIELDS))
    tree_name: str = "sublink"
    only_mpb: bool = True
    sort: str = "input"
    tng_catalog_kwargs: Mapping[str, Any] | None = None


def _hd_tng():
    """Import the TNG dynamics helper module only when data access is needed."""
    from ia_analysis.dynamics import hd_tng

    return hd_tng


def _normalise_snap_track(snap_track: Sequence[int], *, sort: str = "input") -> tuple[int, ...]:
    snaps = tuple(int(s) for s in snap_track)
    mode = str(sort).strip().lower()
    if mode in {"ascending", "asc", "time"}:
        return tuple(sorted(snaps))
    if mode in {"descending", "desc", "redshift"}:
        return tuple(sorted(snaps, reverse=True))
    return snaps


def load_sublink_mpb(
    base_path: str | Path,
    snap: int,
    subhalo_id: int,
    fields: Sequence[str] | None = None,
    *,
    tng_catalog_kwargs: Mapping[str, Any] | None = None,
) -> Mapping[str, np.ndarray]:
    """Load a SubLink main-progenitor branch using the existing TNG IO layer."""
    return _hd_tng().load_sublink_mpb(
        base_path,
        int(snap),
        int(subhalo_id),
        fields=fields,
        tng_catalog_kwargs=tng_catalog_kwargs,
    )


def load_merger_tree(
    base_path: str | Path,
    snap: int,
    subhalo_id: int,
    fields: Sequence[str] | None = None,
    *,
    tree_name: str = "sublink",
    only_mpb: bool = True,
    tng_catalog_kwargs: Mapping[str, Any] | None = None,
) -> Mapping[str, np.ndarray]:
    """Load a TNG merger tree and return the requested tree fields."""
    tree_key = str(tree_name).strip().lower()
    if tree_key in {"sublink", "sublinkdm", "sublinkgal"} and bool(only_mpb):
        return load_sublink_mpb(
            base_path,
            snap,
            subhalo_id,
            fields=fields,
            tng_catalog_kwargs=tng_catalog_kwargs,
        )

    from ia_analysis.catalogs.TNGCatLoader import TNGCatalog

    cat = TNGCatalog(str(base_path), int(snap), **dict(tng_catalog_kwargs or {}))
    try:
        return cat.loadMergerTree(
            sid=int(subhalo_id),
            tree_name=tree_name,
            onlyMPB=bool(only_mpb),
            fields=list(fields or DEFAULT_TREE_FIELDS),
        )
    finally:
        try:
            cat.cleanup()
        except Exception:
            pass


def tree_to_dataframe(tree: Mapping[str, Any]) -> pd.DataFrame:
    """Convert a merger-tree dictionary of one-dimensional arrays to a table."""
    columns: dict[str, np.ndarray] = {}
    for key, value in tree.items():
        arr = np.asarray(value)
        if arr.ndim == 1:
            columns[str(key)] = arr
    return pd.DataFrame(columns)


def select_tree_rows_for_snapshots(
    tree_df: pd.DataFrame,
    snap_track: Sequence[int],
    *,
    sort: str = "input",
) -> pd.DataFrame:
    """Select one tree row for each requested snapshot."""
    if "SnapNum" not in tree_df:
        raise KeyError("Merger-tree table must contain a 'SnapNum' column.")
    snaps = _normalise_snap_track(snap_track, sort=sort)
    rows = []
    snap_values = tree_df["SnapNum"].astype(int)
    for snap in snaps:
        hit = tree_df.loc[snap_values == int(snap)]
        if len(hit):
            rows.append(hit.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()


def build_main_progenitor_track(
    base_path: str | Path,
    snap0: int,
    subhalo_id0: int,
    snap_track: Sequence[int],
    *,
    fields: Sequence[str] | None = None,
    tree_name: str = "sublink",
    only_mpb: bool = True,
    sort: str = "input",
    tng_catalog_kwargs: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Load a tree branch and select the requested snapshots."""
    tree = load_merger_tree(
        base_path,
        int(snap0),
        int(subhalo_id0),
        fields=fields,
        tree_name=tree_name,
        only_mpb=only_mpb,
        tng_catalog_kwargs=tng_catalog_kwargs,
    )
    table = tree_to_dataframe(tree)
    return select_tree_rows_for_snapshots(table, snap_track, sort=sort)


def build_target_reference_tracks(
    base_path: str | Path,
    snap0: int,
    target_subhalo_id0: int,
    reference_subhalo_id0: int,
    snap_track: Sequence[int],
    *,
    fields: Sequence[str] | None = None,
    sort: str = "input",
    tng_catalog_kwargs: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build a common target/reference track table for cross-time analyses."""
    target = build_main_progenitor_track(
        base_path,
        snap0,
        target_subhalo_id0,
        snap_track,
        fields=fields,
        sort=sort,
        tng_catalog_kwargs=tng_catalog_kwargs,
    )
    reference = build_main_progenitor_track(
        base_path,
        snap0,
        reference_subhalo_id0,
        snap_track,
        fields=fields,
        sort=sort,
        tng_catalog_kwargs=tng_catalog_kwargs,
    )
    if target.empty or reference.empty:
        return pd.DataFrame(
            columns=["SnapNum", "TargetSubfindID", "ReferenceSubfindID", "TargetRow", "ReferenceRow"]
        )

    merged = target.merge(reference, on="SnapNum", suffixes=("_target", "_reference"))
    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "SnapNum": int(row["SnapNum"]),
                "TargetSubfindID": int(row["SubfindID_target"]),
                "ReferenceSubfindID": int(row["SubfindID_reference"]),
                "TargetRow": row.filter(like="_target").to_dict(),
                "ReferenceRow": row.filter(like="_reference").to_dict(),
            }
        )
    return pd.DataFrame(rows)


def find_host_central_at_snapshot(
    base_path: str | Path,
    snap: int,
    subhalo_id: int,
    *,
    tng_catalog_kwargs: Mapping[str, Any] | None = None,
    retry_cfg: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Find the FoF group and central subhalo for one subhalo at one snapshot."""
    hd_tng = _hd_tng()
    cat, halos, subs = hd_tng.open_catalog(
        base_path,
        int(snap),
        group_fields=["GroupFirstSub", "GroupNsubs", "GroupLenType"],
        subhalo_fields=["SubhaloGrNr", "SubhaloLenType", "SubhaloPos", "SubhaloVel"],
        tng_catalog_kwargs=tng_catalog_kwargs,
        retry_cfg=retry_cfg,
    )
    try:
        group_id = int(np.asarray(subs["SubhaloGrNr"], dtype=np.int64)[int(subhalo_id)])
        central_id = int(np.asarray(halos["GroupFirstSub"], dtype=np.int64)[group_id])
        return {
            "snap": int(snap),
            "target_subhalo_id": int(subhalo_id),
            "target_group_id": group_id,
            "central_subhalo_id": central_id,
        }
    finally:
        try:
            cat.cleanup()
        except Exception:
            pass

