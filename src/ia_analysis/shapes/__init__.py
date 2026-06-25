"""Shape and alignment namespace.

Purpose
-------
The shapes package contains tensor, axis, kinematic, spin, and IA projection
utilities that operate on arrays and do not orchestrate full pipelines.

Provides
--------
- Iterative inertia-tensor and principal-axis measurements.
- Figure-rotation and kinematic tensor helpers.
- IA ellipticity/projection helpers used by mesh construction and spectra.
"""

from __future__ import annotations

from typing import Any

from ia_analysis._lazy_imports import ExportMap, load_export

_EXPORTS: ExportMap = {
    "ShapeKin": ("ia_analysis.shapes.api", "ShapeKin"),
    "I_iters": ("ia_analysis.shapes.api", "I_iters"),
    "compute_axis": ("ia_analysis.shapes.api", "compute_axis"),
    "measure_iterative_shape": ("ia_analysis.shapes.api", "measure_iterative_shape"),
    "measure_principal_axes": ("ia_analysis.shapes.api", "measure_principal_axes"),
    "project_shape_ellipticity": ("ia_analysis.shapes.api", "project_shape_ellipticity"),
    "project_spin_ellipticity": ("ia_analysis.shapes.api", "project_spin_ellipticity"),
    "equilibrium_shape_from_tide": ("ia_analysis.shapes.api", "equilibrium_shape_from_tide"),
    "evolve_shape_tensor": ("ia_analysis.shapes.api", "evolve_shape_tensor"),
}

__all__ = [*list(_EXPORTS), "api", "shape", "evolution", "Iana"]


def __getattr__(name: str) -> Any:
    """Resolve public shape helpers lazily from the structured API facade."""
    return load_export(_EXPORTS, name)

