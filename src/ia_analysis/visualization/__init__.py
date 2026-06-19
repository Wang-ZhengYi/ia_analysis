"""Visualization namespace for plotting, animation, and figure utilities.

Purpose
-------
The visualization package is split into lightweight facade modules with clear
responsibilities, while preserving access to older implementation modules used
by notebooks.

Provides
--------
- Color, projection, shell-plot, alignment, orbit-animation, and distribution
  helper modules.
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
    "configure_alignment_context": ("ia_analysis.visualization.api", "configure_alignment_context"),
    "plot_alignment_suite": ("ia_analysis.visualization.api", "plot_alignment_suite"),
    "save_six_panel_orbit_movie": ("ia_analysis.visualization.api", "save_six_panel_orbit_movie"),
    "save_three_panel_orbit_movie": ("ia_analysis.visualization.api", "save_three_panel_orbit_movie"),
}

__all__ = [
    *list(_EXPORTS),
    "api",
    "alignment_catalogs",
    "alignment_metrics",
    "alignment_plots",
    "color_tools",
    "distribution_fits",
    "orbit_animation",
    "parallel_alignment",
    "projection_geometry",
    "scene3d",
    "shell_plots",
]


def __getattr__(name: str) -> Any:
    """Resolve public visualization registry helpers lazily."""
    return load_export(_EXPORTS, name)
