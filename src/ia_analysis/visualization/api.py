"""Structured visualization API registry.

Purpose
-------
Visualization functionality is split into focused facades for alignment
catalogs, metrics, plots, colors, projection geometry, shell plots, scenes,
orbit animation, distributions, and parallel figure production.  This registry
helps notebooks discover those layers and provides a few high-level shortcuts.

Provides
--------
- Group discovery and lazy facade-module loading.
- Export listing for each visualization layer.
- Shortcuts for the most common alignment and orbit-animation workflows.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

_GROUPS = {
    "alignment_catalogs": "ia_analysis.visualization.alignment_catalogs",
    "alignment_metrics": "ia_analysis.visualization.alignment_metrics",
    "alignment_plots": "ia_analysis.visualization.alignment_plots",
    "colors": "ia_analysis.visualization.color_tools",
    "distributions": "ia_analysis.visualization.distribution_fits",
    "orbit_animation": "ia_analysis.visualization.orbit_animation",
    "parallel_alignment": "ia_analysis.visualization.parallel_alignment",
    "projection_geometry": "ia_analysis.visualization.projection_geometry",
    "scene3d": "ia_analysis.visualization.scene3d",
    "shell_plots": "ia_analysis.visualization.shell_plots",
}

__all__ = [
    "available_groups",
    "load_group",
    "group_exports",
    "resolve",
    "configure_alignment_context",
    "plot_alignment_suite",
    "save_six_panel_orbit_movie",
    "save_three_panel_orbit_movie",
]


def available_groups() -> tuple[str, ...]:
    """Return the structured visualization facade groups."""
    return tuple(_GROUPS)


def load_group(group: str) -> ModuleType:
    """Import and return one visualization facade module by group name."""
    key = str(group).strip().lower()
    if key not in _GROUPS:
        valid = ", ".join(available_groups())
        raise KeyError(f"Unknown visualization group {group!r}. Available groups: {valid}")
    return import_module(_GROUPS[key])


def group_exports(group: str) -> tuple[str, ...]:
    """Return the public exports advertised by one visualization facade."""
    module = load_group(group)
    return tuple(getattr(module, "__all__", ()))


def resolve(group: str, name: str) -> Any:
    """Resolve one public object from a visualization facade group."""
    return getattr(load_group(group), name)


def configure_alignment_context(*args: Any, **kwargs: Any) -> Any:
    """Configure alignment catalog paths and sample metadata."""
    return resolve("alignment_catalogs", "configure_alignment_context")(*args, **kwargs)


def plot_alignment_suite(*args: Any, **kwargs: Any) -> Any:
    """Run the high-level alignment figure suite."""
    return resolve("alignment_plots", "plot_alignment_suite")(*args, **kwargs)


def save_six_panel_orbit_movie(*args: Any, **kwargs: Any) -> Any:
    """Save the six-panel orbit movie through the orbit-animation facade."""
    return resolve("orbit_animation", "save_six_panel_orbit_movie")(*args, **kwargs)


def save_three_panel_orbit_movie(*args: Any, **kwargs: Any) -> Any:
    """Save the three-panel orbit movie through the orbit-animation facade."""
    return resolve("orbit_animation", "save_three_panel_orbit_movie")(*args, **kwargs)

