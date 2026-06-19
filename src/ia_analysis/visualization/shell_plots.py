"""Radial-shell and binding-energy shell plotting routines."""

from __future__ import annotations

from typing import Any

from ia_analysis.visualization._lazy import load_export

_EXPORTS = {
    "plot_radial_shells_pretty": ("ia_analysis.visualization.arts", "plot_radial_shells_pretty"),
    "plot_binding_shell_panels_pretty": (
        "ia_analysis.visualization.arts",
        "plot_binding_shell_panels_pretty",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    return load_export(_EXPORTS, name)

