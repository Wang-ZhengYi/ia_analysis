"""Structured shape, spin, and IA-projection API facade.

Purpose
-------
This module groups the most commonly used shape-analysis operations under
clear names.  It keeps the historical ``shape.py`` and ``Iana.py`` modules as
the scientific implementations, while giving new code one compact import path.

Provides
--------
- Iterative inertia tensor measurement with principal-axis packaging.
- Stable aliases for spin, kinetic-energy, kappa-rot, and figure-rotation tools.
- IA ellipticity projection wrappers for shape matrices and spin vectors.
"""

from __future__ import annotations

from typing import Any

from ia_analysis._lazy_imports import ExportMap, call_export, export_names, load_export

_EXPORTS: ExportMap = {
    "ShapeKin": ("ia_analysis.shapes.shape", "ShapeKin"),
    "I_iters": ("ia_analysis.shapes.shape", "I_iters"),
    "compute_axis": ("ia_analysis.shapes.shape", "compute_axis"),
    "compute_inertia_tensor": ("ia_analysis.shapes.shape", "compute_inertia_tensor"),
    "eig_var": ("ia_analysis.shapes.shape", "eig_var"),
    "eig_var_safe": ("ia_analysis.shapes.shape", "eig_var_safe"),
    "iters_var": ("ia_analysis.shapes.shape", "iters_var"),
    "angular_momentum": ("ia_analysis.shapes.shape", "ang_mom"),
    "kinetic_energy": ("ia_analysis.shapes.shape", "kin_energy"),
    "kappa_rot": ("ia_analysis.shapes.shape", "kappa_rot"),
    "beta_fig": ("ia_analysis.shapes.shape", "beta_fig"),
    "alignment_vector_vector": ("ia_analysis.shapes.Iana", "VV"),
    "alignment_tensor_tensor": ("ia_analysis.shapes.Iana", "II"),
    "alignment_vector_tensor": ("ia_analysis.shapes.Iana", "VI"),
    "omega_fig": ("ia_analysis.shapes.Iana", "omega_fig"),
    "chi_shape_orientation": ("ia_analysis.shapes.Iana", "chiSO"),
    "epsilon_from_shape_matrix": ("ia_analysis.shapes.Iana", "epsilon_from_shape_matrix"),
    "epsilon_from_spin": ("ia_analysis.shapes.Iana", "epsilon_from_spin"),
    "estimate_kappa_rot_from_subhalo": ("ia_analysis.shapes.Iana", "estimate_kappa_rot_from_subhalo"),
    "fit_enfw_profile": ("ia_analysis.shapes.Iana", "fit_enfw_profile"),
}

__all__ = [
    *export_names(_EXPORTS),
    "measure_iterative_shape",
    "measure_principal_axes",
    "project_shape_ellipticity",
    "project_spin_ellipticity",
]


def __getattr__(name: str) -> Any:
    """Resolve public shape-analysis aliases lazily."""
    return load_export(_EXPORTS, name)


def measure_iterative_shape(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Measure an iterative shape tensor and return tensor, axes, and vectors."""
    tensor = call_export(_EXPORTS, "I_iters", *args, **kwargs)
    if isinstance(tensor, tuple):
        tensor = tensor[0]
    axes, vectors = call_export(_EXPORTS, "compute_axis", tensor)
    return {"tensor": tensor, "axes": axes, "vectors": vectors}


def measure_principal_axes(tensor: Any) -> dict[str, Any]:
    """Convert a 3D tensor into a principal-axis result dictionary."""
    axes, vectors = call_export(_EXPORTS, "compute_axis", tensor)
    return {"axes": axes, "vectors": vectors}


def project_shape_ellipticity(*args: Any, **kwargs: Any) -> Any:
    """Project a 3D shape matrix to IA ellipticity components."""
    return call_export(_EXPORTS, "epsilon_from_shape_matrix", *args, **kwargs)


def project_spin_ellipticity(*args: Any, **kwargs: Any) -> Any:
    """Project spin vectors to IA ellipticity components."""
    return call_export(_EXPORTS, "epsilon_from_spin", *args, **kwargs)

