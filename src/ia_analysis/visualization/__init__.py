"""Visualization namespace for plotting, animation, and figure utilities.

Purpose
-------
The visualization package is split into lightweight facade modules with clear
responsibilities, while preserving access to older implementation modules used
by notebooks.

Provides
--------
- Figure IO, style, legend, profile, spectrum, correlation, TNG-dynamics,
  merger-tree, alignment, orbit-animation, and distribution helper modules.
- Lazy facade imports so package import does not immediately require plotting
  libraries.

Notes
-----
Prefer the structured facade modules for new code.  Historical modules such as
``arts`` and ``arts_IA`` remain available for compatibility inside ``src``.
"""

from __future__ import annotations

from typing import Any

from ia_analysis._lazy_imports import ExportMap, load_export

_EXPORTS: ExportMap = {
    "available_groups": ("ia_analysis.visualization.api", "available_groups"),
    "load_group": ("ia_analysis.visualization.api", "load_group"),
    "group_exports": ("ia_analysis.visualization.api", "group_exports"),
    "resolve": ("ia_analysis.visualization.api", "resolve"),
    "save_figure": ("ia_analysis.visualization.api", "save_figure"),
    "set_project_style": ("ia_analysis.visualization.api", "set_project_style"),
    "configure_alignment_context": ("ia_analysis.visualization.api", "configure_alignment_context"),
    "plot_metric_atlas": ("ia_analysis.visualization.api", "plot_metric_atlas"),
    "plot_alignment_suite": ("ia_analysis.visualization.api", "plot_alignment_suite"),
    "plot_spectrum_grid": ("ia_analysis.visualization.api", "plot_spectrum_grid"),
    "plot_correlation_grid": ("ia_analysis.visualization.api", "plot_correlation_grid"),
    "plot_pi_closure_table": ("ia_analysis.visualization.api", "plot_pi_closure_table"),
    "plot_orbit_plane_evolution": ("ia_analysis.visualization.api", "plot_orbit_plane_evolution"),
    "save_six_panel_orbit_movie": ("ia_analysis.visualization.api", "save_six_panel_orbit_movie"),
    "save_three_panel_orbit_movie": ("ia_analysis.visualization.api", "save_three_panel_orbit_movie"),
    "plot_catalog_inventory": ("ia_analysis.visualization.api", "plot_catalog_inventory"),
    "plot_orbit_shape_suite": ("ia_analysis.visualization.api", "plot_orbit_shape_suite"),
    "plot_spectrum_ratios": ("ia_analysis.visualization.api", "plot_spectrum_ratios"),
    "plot_correlation_quality": ("ia_analysis.visualization.api", "plot_correlation_quality"),
    "plot_hod_components": ("ia_analysis.visualization.api", "plot_hod_components"),
}

__all__ = [
    *list(_EXPORTS),
    "api",
    "alignment_catalogs",
    "alignment_atlas",
    "alignment_metrics",
    "alignment_plots",
    "color_tools",
    "correlation_plots",
    "distribution_fits",
    "figure_io",
    "legends",
    "merger_tree_plots",
    "orbit_animation",
    "parallel_alignment",
    "pipeline_plots",
    "hod_plots",
    "plot_styles",
    "profile_plots",
    "projection_geometry",
    "scene3d",
    "shell_plots",
    "spectrum_plots",
    "tng_dynamics_plots",
]


def __getattr__(name: str) -> Any:
    """Resolve public visualization registry helpers lazily."""
    return load_export(_EXPORTS, name)
