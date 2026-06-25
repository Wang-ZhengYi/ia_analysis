"""Structured halo-dynamics API facade.

Purpose
-------
This module separates the recommended public entrypoints from the historical
implementation filenames.  Array-level halo dynamics live in
``halo_dynamics.py`` and TNG orchestration lives in ``hd_tng.py``; both remain
lazy-loaded behind this facade.

Provides
--------
- Shell mask construction and shell-wise dynamics analysis.
- Affine-flow, figure-rotation, tidal-response, and torque helpers.
- TNG subhalo dynamics wrappers for catalog-backed workflows.
"""

from __future__ import annotations

from typing import Any

from ia_analysis._lazy_imports import ExportMap, call_export, export_names, load_export

_EXPORTS: ExportMap = {
    "make_shell_masks": ("ia_analysis.dynamics.halo_dynamics", "make_shell_masks"),
    "radial_shell_masks": ("ia_analysis.dynamics.halo_dynamics", "radial_shell_masks"),
    "binding_energy_shell_masks": ("ia_analysis.dynamics.halo_dynamics", "binding_energy_shell_masks"),
    "spherical_potential_from_radial_mass": ("ia_analysis.dynamics.halo_dynamics", "spherical_potential_from_radial_mass"),
    "component_binding_energy": ("ia_analysis.dynamics.halo_dynamics", "component_binding_energy"),
    "binding_energy_mass_distribution": ("ia_analysis.dynamics.halo_dynamics", "binding_energy_mass_distribution"),
    "component_binding_energy_profiles": ("ia_analysis.dynamics.halo_dynamics", "component_binding_energy_profiles"),
    "analyze_shell": ("ia_analysis.dynamics.halo_dynamics", "analyze_shell"),
    "analyze_halo_shells": ("ia_analysis.dynamics.halo_dynamics", "analyze_halo_shells"),
    "compute_affine_kinematics": ("ia_analysis.dynamics.halo_dynamics", "compute_affine_kinematics"),
    "figure_rotation_from_dI": ("ia_analysis.dynamics.halo_dynamics", "figure_rotation_from_dI"),
    "figure_rotation_from_affine": ("ia_analysis.dynamics.halo_dynamics", "figure_rotation_from_affine"),
    "measure_hessian_and_omega": ("ia_analysis.dynamics.halo_dynamics", "measure_hessian_and_omega"),
    "tidal_tensor_from_components": ("ia_analysis.dynamics.halo_dynamics", "tidal_tensor_from_components"),
    "sample_tidal_hessian": ("ia_analysis.dynamics.halo_dynamics", "sample_tidal_hessian"),
    "torque_from_hessian": ("ia_analysis.dynamics.halo_dynamics", "torque_from_hessian"),
    "tidal_response_terms": ("ia_analysis.dynamics.halo_dynamics", "tidal_response_terms"),
    "stack_shell_quantity": ("ia_analysis.dynamics.halo_dynamics", "stack_shell_quantity"),
    "shape_beta_fig_from_moments": ("ia_analysis.dynamics.halo_dynamics", "shape_beta_fig_from_moments"),
    "open_tng_catalog_for_dynamics": ("ia_analysis.dynamics.hd_tng", "open_catalog"),
    "select_tng_subhaloes": ("ia_analysis.dynamics.hd_tng", "select_subhaloes_in_top_groups"),
    "load_subhalo_component_particles": ("ia_analysis.dynamics.hd_tng", "load_subhalo_component_particles"),
    "load_subhalo_components": ("ia_analysis.dynamics.hd_tng", "load_subhalo_components"),
    "load_subhalo_dm_particles": ("ia_analysis.dynamics.hd_tng", "load_subhalo_dm_particles"),
    "compute_subhalo_component_binding_profiles": (
        "ia_analysis.dynamics.hd_tng",
        "compute_subhalo_component_binding_profiles",
    ),
    "compute_halo_component_binding_profiles": (
        "ia_analysis.dynamics.hd_tng",
        "compute_halo_component_binding_profiles",
    ),
    "OrbitInitialCondition": ("ia_analysis.dynamics.orbit_shape", "OrbitInitialCondition"),
    "DEFAULT_ORBIT_CASES": ("ia_analysis.dynamics.orbit_shape", "DEFAULT_ORBIT_CASES"),
    "run_orbit_shape_case": ("ia_analysis.dynamics.orbit_shape", "run_orbit_shape_case"),
    "run_orbit_shape_suite": ("ia_analysis.dynamics.orbit_shape", "run_orbit_shape_suite"),
    "analyse_particle_data": ("ia_analysis.dynamics.hd_tng", "analyse_particle_data"),
    "compute_one_subhalo": ("ia_analysis.dynamics.hd_tng", "compute_one_subhalo"),
    "compute_haloes": ("ia_analysis.dynamics.hd_tng", "compute_haloes"),
    "enrich_run_with_group_metadata": ("ia_analysis.dynamics.hd_tng", "enrich_run_with_group_metadata"),
    "load_sublink_mpb": ("ia_analysis.MergerTree.reader", "load_sublink_mpb"),
    "cross_time_pattern_speed_for_subhalo": (
        "ia_analysis.MergerTree.workflow",
        "cross_time_pattern_speed_for_subhalo",
    ),
}

__all__ = [
    *export_names(_EXPORTS),
    "analyze_particle_halo",
    "compute_tng_subhalo_dynamics",
    "compute_tng_halo_sample",
    "compute_tng_component_binding_profiles",
]


def __getattr__(name: str) -> Any:
    """Resolve halo-dynamics aliases lazily."""
    return load_export(_EXPORTS, name)


def analyze_particle_halo(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run shell-wise halo dynamics on particle phase-space arrays."""
    return call_export(_EXPORTS, "analyze_halo_shells", *args, **kwargs)


def compute_tng_subhalo_dynamics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compute dynamics for one TNG subhalo using the catalog-backed wrapper."""
    return call_export(_EXPORTS, "compute_one_subhalo", *args, **kwargs)


def compute_tng_halo_sample(*args: Any, **kwargs: Any) -> Any:
    """Compute dynamics for a selected sample of TNG haloes or subhaloes."""
    return call_export(_EXPORTS, "compute_haloes", *args, **kwargs)


def compute_tng_component_binding_profiles(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compute mass distributions over binding energy for one TNG subhalo."""
    return call_export(_EXPORTS, "compute_halo_component_binding_profiles", *args, **kwargs)
