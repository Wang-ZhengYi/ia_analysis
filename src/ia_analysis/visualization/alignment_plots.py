"""Alignment plotting facade for profiles, grids, and figure suites.

Purpose
-------
This module exposes the high-level plotting API for IA alignment figures while
hiding the historical implementation layout behind lazy imports.

Provides
--------
- Alignment specification lookup and chapter listing.
- Binned profile calculation and axis formatting helpers.
- Snapshot grids, redshift evolution plots, pair plots, and suite runners.
"""


from __future__ import annotations

from typing import Any

from ia_analysis.visualization._lazy import call_export, load_export

_EXPORTS = {
    "set_plot_output_root": ("ia_analysis.visualization.arts_IA", "set_plot_output_root"),
    "set_paper_style": ("ia_analysis.visualization.arts_IA", "set_paper_style"),
    "AlignmentSpec": ("ia_analysis.visualization.arts_IA", "AlignmentSpec"),
    "build_alignment_specs": ("ia_analysis.visualization.arts_IA", "build_alignment_specs"),
    "get_alignment_spec_by_name": ("ia_analysis.visualization.arts_IA", "get_alignment_spec_by_name"),
    "list_alignment_specs": ("ia_analysis.visualization.arts_IA", "list_alignment_specs"),
    "list_alignment_chapters": ("ia_analysis.visualization.arts_IA", "list_alignment_chapters"),
    "get_alignment_arrays": ("ia_analysis.visualization.arts_IA", "get_alignment_arrays"),
    "binned_profile": ("ia_analysis.visualization.arts_IA", "binned_profile"),
    "plot_alignment_on_axis": ("ia_analysis.visualization.arts_IA", "plot_alignment_on_axis"),
    "apply_alignment_axis_format": ("ia_analysis.visualization.arts_IA", "apply_alignment_axis_format"),
    "plot_alignment_snapshot_grid": (
        "ia_analysis.visualization.arts_IA",
        "plot_alignment_snapshot_grid",
    ),
    "plot_alignment_redshift_evolution": (
        "ia_analysis.visualization.arts_IA",
        "plot_alignment_redshift_evolution",
    ),
    "plot_alignment": ("ia_analysis.visualization.arts_IA", "plot_alignment"),
    "plot_alignment_pair": ("ia_analysis.visualization.arts_IA", "plot_alignment_pair"),
    "plot_alignment_chapter": ("ia_analysis.visualization.arts_IA", "plot_alignment_chapter"),
    "PhysicalSpec": ("ia_analysis.visualization.arts_IA", "PhysicalSpec"),
    "plot_physical": ("ia_analysis.visualization.arts_IA", "plot_physical"),
    "plot_shape_cluster_alignment_suite": (
        "ia_analysis.visualization.arts_IA",
        "plot_shape_cluster_alignment_suite",
    ),
}

__all__ = [*list(_EXPORTS), "configure_paper_style", "plot_alignment_suite"]


def __getattr__(name: str) -> Any:
    return load_export(_EXPORTS, name)


def configure_paper_style(*args: Any, **kwargs: Any) -> Any:
    """Apply the project paper plotting style."""
    return call_export(_EXPORTS, "set_paper_style", *args, **kwargs)


def plot_alignment_suite(*args: Any, **kwargs: Any) -> Any:
    """Run the shape-cluster alignment figure suite."""
    return call_export(_EXPORTS, "plot_shape_cluster_alignment_suite", *args, **kwargs)

