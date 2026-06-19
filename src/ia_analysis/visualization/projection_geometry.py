"""Projection bases, principal-plane coordinates, ellipses, and scale bars."""

from __future__ import annotations

from typing import Any

from ia_analysis.visualization._lazy import load_export

_EXPORTS = {
    "principal_plane_basis_from_points": (
        "ia_analysis.visualization.arts",
        "principal_plane_basis_from_points",
    ),
    "project_to_principal_plane": ("ia_analysis.visualization.arts", "project_to_principal_plane"),
    "ellipse_from_projected_points": ("ia_analysis.visualization.arts", "ellipse_from_projected_points"),
    "same_xy_limits": ("ia_analysis.visualization.arts", "same_xy_limits"),
    "add_kpc_scalebar": ("ia_analysis.visualization.arts", "add_kpc_scalebar"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    return load_export(_EXPORTS, name)

