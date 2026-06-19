"""Shell-plot facade for radial and binding-energy panels.

Purpose
-------
This module provides the preferred import path for visualizing halo shells and
binding-energy shell decompositions.

Provides
--------
- Radial shell plots projected onto principal planes.
- Binding-shell panel plots with consistent limits, ellipses, and scale bars.
"""


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

