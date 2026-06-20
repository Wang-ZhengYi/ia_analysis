"""Merger-tree and cross-time workflow namespace.

Purpose
-------
The MergerTree package owns cross-time reading and orchestration.  It follows
SubLink or other TNG merger-tree tracks, builds snapshot task tables, and calls
the existing catalog, shape, and dynamics modules for each scientific step.

Provides
--------
- Main-progenitor branch loading and snapshot-track selection.
- Target/reference track matching for host-centric cross-time analyses.
- Snapshot-level component loading, shape measurement, and shell dynamics.
- Save/load helpers for computed cross-time products.
"""

from __future__ import annotations

from typing import Any

from ia_analysis._lazy_imports import ExportMap, load_export

_EXPORTS: ExportMap = {
    "MergerTrackConfig": ("ia_analysis.MergerTree.reader", "MergerTrackConfig"),
    "SnapshotTask": ("ia_analysis.MergerTree.workflow", "SnapshotTask"),
    "load_merger_tree": ("ia_analysis.MergerTree.reader", "load_merger_tree"),
    "load_sublink_mpb": ("ia_analysis.MergerTree.reader", "load_sublink_mpb"),
    "tree_to_dataframe": ("ia_analysis.MergerTree.reader", "tree_to_dataframe"),
    "select_tree_rows_for_snapshots": ("ia_analysis.MergerTree.reader", "select_tree_rows_for_snapshots"),
    "build_main_progenitor_track": ("ia_analysis.MergerTree.reader", "build_main_progenitor_track"),
    "build_target_reference_tracks": ("ia_analysis.MergerTree.reader", "build_target_reference_tracks"),
    "find_host_central_at_snapshot": ("ia_analysis.MergerTree.reader", "find_host_central_at_snapshot"),
    "load_snapshot_components": ("ia_analysis.MergerTree.workflow", "load_snapshot_components"),
    "measure_snapshot_shapes": ("ia_analysis.MergerTree.workflow", "measure_snapshot_shapes"),
    "analyze_snapshot_shells": ("ia_analysis.MergerTree.workflow", "analyze_snapshot_shells"),
    "process_snapshot_task": ("ia_analysis.MergerTree.workflow", "process_snapshot_task"),
    "run_cross_time_workflow": ("ia_analysis.MergerTree.workflow", "run_cross_time_workflow"),
    "cross_time_pattern_speed_for_subhalo": (
        "ia_analysis.MergerTree.workflow",
        "cross_time_pattern_speed_for_subhalo",
    ),
    "save_cross_time_products": ("ia_analysis.MergerTree.storage", "save_cross_time_products"),
    "load_cross_time_products": ("ia_analysis.MergerTree.storage", "load_cross_time_products"),
}

__all__ = [*list(_EXPORTS), "api", "reader", "workflow", "storage"]


def __getattr__(name: str) -> Any:
    """Resolve public merger-tree helpers lazily from their domain modules."""
    return load_export(_EXPORTS, name)

