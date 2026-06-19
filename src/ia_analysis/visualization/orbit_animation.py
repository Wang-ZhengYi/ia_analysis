"""Orbit animation and preview entrypoints."""

from __future__ import annotations

from typing import Any

from ia_analysis.visualization._lazy import call_export, load_export

_EXPORTS = {
    "save_orbit_movie6": ("ia_analysis.visualization.orbit_viz", "save_orbit_movie6"),
    "save_orbit_movie3": ("ia_analysis.visualization.orbit_viz", "save_orbit_movie3"),
    "make_orbit_animation": ("ia_analysis.visualization.orbit_viz2", "make_orbit_animation"),
    "preview_orbit_frame": ("ia_analysis.visualization.orbit_viz2", "preview_orbit_frame"),
}

__all__ = [
    *list(_EXPORTS),
    "save_six_panel_orbit_movie",
    "save_three_panel_orbit_movie",
]


def __getattr__(name: str) -> Any:
    return load_export(_EXPORTS, name)


def save_six_panel_orbit_movie(*args: Any, **kwargs: Any) -> Any:
    """Save the historical six-panel orbit movie."""
    return call_export(_EXPORTS, "save_orbit_movie6", *args, **kwargs)


def save_three_panel_orbit_movie(*args: Any, **kwargs: Any) -> Any:
    """Save the historical three-panel orbit movie."""
    return call_export(_EXPORTS, "save_orbit_movie3", *args, **kwargs)

