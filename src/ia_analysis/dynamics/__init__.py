"""Halo and subhalo dynamics namespace.

Purpose
-------
The dynamics package contains shell-wise kinematics, affine-flow estimators,
figure-rotation diagnostics, and TNG-specific high-level wrappers.

Provides
--------
- Tensor-based halo dynamics utilities.
- TNG cross-redshift and layered-dynamics helpers.
- Interfaces that combine catalog, shape, and tidal information without
  depending on spectra measurement code.
"""

from __future__ import annotations

from typing import Any

from ia_analysis._lazy_imports import ExportMap, load_export

_EXPORTS: ExportMap = {
    "SnapshotTarget": ("ia_analysis.dynamics.api", "SnapshotTarget"),
    "analyze_snapshot_target": ("ia_analysis.dynamics.api", "analyze_snapshot_target"),
    "measure_target_shape": ("ia_analysis.dynamics.api", "measure_target_shape"),
    "measure_target_principal_axes": ("ia_analysis.dynamics.api", "measure_target_principal_axes"),
    "measure_target_kinematics": ("ia_analysis.dynamics.api", "measure_target_kinematics"),
    "measure_figure_rotation": ("ia_analysis.dynamics.api", "measure_figure_rotation"),
    "measure_affine_figure_rotation": ("ia_analysis.dynamics.api", "measure_affine_figure_rotation"),
    "measure_hessian_omega": ("ia_analysis.dynamics.api", "measure_hessian_omega"),
    "measure_target_binding": ("ia_analysis.dynamics.api", "measure_target_binding"),
    "measure_component_binding_profiles": ("ia_analysis.dynamics.api", "measure_component_binding_profiles"),
    "measure_target_shell_dynamics": ("ia_analysis.dynamics.api", "measure_target_shell_dynamics"),
    "measure_tidal_response": ("ia_analysis.dynamics.api", "measure_tidal_response"),
    "symmetrize_matrix": ("ia_analysis.dynamics.api", "symmetrize_matrix"),
    "sorted_eigensystem": ("ia_analysis.dynamics.api", "sorted_eigensystem"),
    "measure_shape_tensor": ("ia_analysis.dynamics.api", "measure_shape_tensor"),
    "measure_mixed_moment": ("ia_analysis.dynamics.api", "measure_mixed_moment"),
    "measure_moment_derivative": ("ia_analysis.dynamics.api", "measure_moment_derivative"),
    "coerce_tidal_hessian": ("ia_analysis.dynamics.api", "coerce_tidal_hessian"),
    "tidal_hessian_from_components": ("ia_analysis.dynamics.api", "tidal_hessian_from_components"),
    "tidal_stretch_eigensystem": ("ia_analysis.dynamics.api", "tidal_stretch_eigensystem"),
    "make_shell_masks": ("ia_analysis.dynamics.api", "make_shell_masks"),
    "component_binding_energy": ("ia_analysis.dynamics.api", "component_binding_energy"),
    "binding_energy_mass_distribution": ("ia_analysis.dynamics.api", "binding_energy_mass_distribution"),
    "component_binding_energy_profiles": ("ia_analysis.dynamics.api", "component_binding_energy_profiles"),
    "analyze_shell": ("ia_analysis.dynamics.api", "analyze_shell"),
    "analyze_halo_shells": ("ia_analysis.dynamics.api", "analyze_halo_shells"),
    "analyze_particle_halo": ("ia_analysis.dynamics.api", "analyze_particle_halo"),
    "compute_affine_kinematics": ("ia_analysis.dynamics.api", "compute_affine_kinematics"),
    "figure_rotation_from_dI": ("ia_analysis.dynamics.api", "figure_rotation_from_dI"),
    "figure_rotation_from_affine": ("ia_analysis.dynamics.api", "figure_rotation_from_affine"),
    "tidal_response_terms": ("ia_analysis.dynamics.api", "tidal_response_terms"),
    "compute_tng_subhalo_dynamics": ("ia_analysis.dynamics.api", "compute_tng_subhalo_dynamics"),
    "compute_tng_halo_sample": ("ia_analysis.dynamics.api", "compute_tng_halo_sample"),
    "compute_tng_component_binding_profiles": ("ia_analysis.dynamics.api", "compute_tng_component_binding_profiles"),
    "compute_halo_component_binding_profiles": ("ia_analysis.dynamics.api", "compute_halo_component_binding_profiles"),
}

__all__ = [
    *list(_EXPORTS),
    "api",
    "shape_measurements",
    "kinematics",
    "dynamics_measurements",
    "matrix_analysis",
    "snapshot_analysis",
    "halo_dynamics",
    "hd_tng",
]


def __getattr__(name: str) -> Any:
    """Resolve public dynamics helpers lazily from the structured API facade."""
    return load_export(_EXPORTS, name)
