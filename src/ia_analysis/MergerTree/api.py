"""Structured API facade for merger-tree and cross-time analysis.

Purpose
-------
This facade makes the cross-time workflow discoverable without importing TNG
catalog readers, HDF5 libraries, or plotting dependencies at import time.  The
implementation modules are loaded only when a specific function is used.

Provides
--------
- Main-progenitor branch readers and snapshot-track selectors.
- Snapshot workflow helpers that call catalogs, shapes, and dynamics modules.
- Cross-time product persistence helpers for compute-once/read-many notebooks.
"""

from __future__ import annotations

from typing import Any

from ia_analysis._lazy_imports import ExportMap, call_export, export_names, load_export

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
    "product_tables": ("ia_analysis.MergerTree.diagnostics", "product_tables"),
    "combine_product_tables": ("ia_analysis.MergerTree.diagnostics", "combine_product_tables"),
    "summarize_numeric_table": ("ia_analysis.MergerTree.diagnostics", "summarize_numeric_table"),
    "pi_closure_residuals": ("ia_analysis.MergerTree.diagnostics", "pi_closure_residuals"),
}

__all__ = [*export_names(_EXPORTS), "load_track", "run_workflow"]


def __getattr__(name: str) -> Any:
    """Resolve merger-tree API names lazily."""
    return load_export(_EXPORTS, name)


def load_track(*args: Any, **kwargs: Any) -> Any:
    """Load and select a main-progenitor track."""
    return call_export(_EXPORTS, "build_main_progenitor_track", *args, **kwargs)


def run_workflow(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run the standard cross-time workflow."""
    return call_export(_EXPORTS, "run_cross_time_workflow", *args, **kwargs)
