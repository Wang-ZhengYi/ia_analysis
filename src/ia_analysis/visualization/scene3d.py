"""3D scene facade for galaxy-system visual diagnostics.

Purpose
-------
This module groups 3D scatter, animated camera, galaxy-system, and clustering
scene helpers behind a lightweight import path.

Provides
--------
- Lazy access to 3D scatter plotting.
- Lazy access to galaxy-system visualization.
- Access to the historical 2D k-means helper used by scene construction.
"""


from __future__ import annotations

from typing import Any

from ia_analysis.visualization._lazy import load_export

_EXPORTS = {
    "plot_3d_scatter": ("ia_analysis.visualization.arts", "plot_3d_scatter"),
    "plot_3d_scatter_animated": ("ia_analysis.visualization.arts", "plot_3d_scatter_animated"),
    "visualize_galaxy_system": ("ia_analysis.visualization.arts", "visualize_galaxy_system"),
    "KMeans2DClusterer": ("ia_analysis.visualization.arts", "KMeans2DClusterer"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    return load_export(_EXPORTS, name)

