"""Structured visualization API registry.

Purpose
-------
Visualization functionality is split into focused facades for figure IO,
project styles, legends, alignment atlases, profiles, spectra, correlations,
TNG dynamics, merger-tree views, shell plots, scenes, orbit animation,
distributions, and parallel figure production.  This registry helps notebooks
discover those layers and provides a few high-level shortcuts.

Provides
--------
- Group discovery and lazy facade-module loading.
- Export listing for each visualization layer.
- Shortcuts for common figure-saving, alignment, spectra, and dynamics plots.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

_GROUPS = {
    "alignment_catalogs": "ia_analysis.visualization.alignment_catalogs",
    "alignment_metrics": "ia_analysis.visualization.alignment_metrics",
    "alignment_atlas": "ia_analysis.visualization.alignment_atlas",
    "alignment_plots": "ia_analysis.visualization.alignment_plots",
    "colors": "ia_analysis.visualization.color_tools",
    "correlations": "ia_analysis.visualization.correlation_plots",
    "distributions": "ia_analysis.visualization.distribution_fits",
    "figure_io": "ia_analysis.visualization.figure_io",
    "legends": "ia_analysis.visualization.legends",
    "merger_tree": "ia_analysis.visualization.merger_tree_plots",
    "orbit_animation": "ia_analysis.visualization.orbit_animation",
    "parallel_alignment": "ia_analysis.visualization.parallel_alignment",
    "plot_styles": "ia_analysis.visualization.plot_styles",
    "profiles": "ia_analysis.visualization.profile_plots",
    "projection_geometry": "ia_analysis.visualization.projection_geometry",
    "scene3d": "ia_analysis.visualization.scene3d",
    "shell_plots": "ia_analysis.visualization.shell_plots",
    "spectra": "ia_analysis.visualization.spectrum_plots",
    "tng_dynamics": "ia_analysis.visualization.tng_dynamics_plots",
}

__all__ = [
    "available_groups",
    "load_group",
    "group_exports",
    "resolve",
    "save_figure",
    "set_project_style",
    "configure_alignment_context",
    "plot_metric_atlas",
    "plot_alignment_suite",
    "plot_spectrum_grid",
    "plot_correlation_grid",
    "plot_pi_closure_table",
    "plot_orbit_plane_evolution",
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


def save_figure(*args: Any, **kwargs: Any) -> Any:
    """Save a Matplotlib figure with project directory conventions."""
    return resolve("figure_io", "save_figure")(*args, **kwargs)


def set_project_style(*args: Any, **kwargs: Any) -> Any:
    """Apply the project-wide plotting style."""
    return resolve("plot_styles", "set_project_style")(*args, **kwargs)


def plot_metric_atlas(*args: Any, **kwargs: Any) -> Any:
    """Draw a reusable alignment metric atlas."""
    return resolve("alignment_atlas", "plot_metric_atlas")(*args, **kwargs)


def plot_alignment_suite(*args: Any, **kwargs: Any) -> Any:
    """Run the high-level alignment figure suite."""
    return resolve("alignment_plots", "plot_alignment_suite")(*args, **kwargs)


def plot_spectrum_grid(*args: Any, **kwargs: Any) -> Any:
    """Draw a tidy power-spectrum grid."""
    return resolve("spectra", "plot_spectrum_grid")(*args, **kwargs)


def plot_correlation_grid(*args: Any, **kwargs: Any) -> Any:
    """Draw a grid of correlation-function statistic keys."""
    return resolve("correlations", "plot_correlation_grid")(*args, **kwargs)


def plot_pi_closure_table(*args: Any, **kwargs: Any) -> Any:
    """Draw the TNG pi-closure diagnostic table."""
    return resolve("tng_dynamics", "plot_pi_closure_table")(*args, **kwargs)


def plot_orbit_plane_evolution(*args: Any, **kwargs: Any) -> Any:
    """Draw merger-tree orbit-plane evolution tracks."""
    return resolve("merger_tree", "plot_orbit_plane_evolution")(*args, **kwargs)


def save_six_panel_orbit_movie(*args: Any, **kwargs: Any) -> Any:
    """Save the six-panel orbit movie through the orbit-animation facade."""
    return resolve("orbit_animation", "save_six_panel_orbit_movie")(*args, **kwargs)


def save_three_panel_orbit_movie(*args: Any, **kwargs: Any) -> Any:
    """Save the three-panel orbit movie through the orbit-animation facade."""
    return resolve("orbit_animation", "save_three_panel_orbit_movie")(*args, **kwargs)
