"""Orbit experiment namespace.

Purpose
-------
The orbits package contains NFW orbit experiments, synthetic halo generation,
2LPT merger-tree orbit templates, and ellipsoidal group approximations used for
controlled visual and dynamical tests.

Provides
--------
- Mock halo point-cloud generation.
- Orbit integration and shell-visualization inputs.
- Pinocchio-like tree adapters for group-internal subhalo orbit templates.
- Ellipsoidal tidal and phase-space perturbation models for HOD one-halo terms.
- Tidal-stripping post-processing and tidal-track diagnostics.
- Lightweight utilities used by curated orbit notebooks.
"""

from __future__ import annotations

from typing import Any

from ia_analysis._lazy_imports import ExportMap, load_export

_EXPORTS: ExportMap = {
    "generate_mock_halo": ("ia_analysis.orbits.api", "generate_mock_halo"),
    "generate_mock_ellipsoid": ("ia_analysis.orbits.api", "generate_mock_ellipsoid"),
    "run_orbit": ("ia_analysis.orbits.api", "run_orbit"),
    "NFWHost": ("ia_analysis.orbits.api", "NFWHost"),
    "OrbitSimulator": ("ia_analysis.orbits.api", "OrbitSimulator"),
    "OrbitResult": ("ia_analysis.orbits.api", "OrbitResult"),
    "TreeTrack": ("ia_analysis.orbits.api", "TreeTrack"),
    "OrbitTemplate": ("ia_analysis.orbits.api", "OrbitTemplate"),
    "OrbitTemplateLibrary": ("ia_analysis.orbits.api", "OrbitTemplateLibrary"),
    "DEFAULT_TEMPLATE_FEATURES": ("ia_analysis.orbits.api", "DEFAULT_TEMPLATE_FEATURES"),
    "build_orbit_template": ("ia_analysis.orbits.api", "build_orbit_template"),
    "build_template_library": ("ia_analysis.orbits.api", "build_template_library"),
    "hod_1h_orbit_kernel": ("ia_analysis.orbits.api", "hod_1h_orbit_kernel"),
    "orbit_template_roadmap": ("ia_analysis.orbits.api", "orbit_template_roadmap"),
    "PinocchioColumnMap": ("ia_analysis.orbits.api", "PinocchioColumnMap"),
    "read_pinocchio_table": ("ia_analysis.orbits.api", "read_pinocchio_table"),
    "build_pinocchio_template_library": ("ia_analysis.orbits.api", "build_pinocchio_template_library"),
    "EllipsoidalGroupModel": ("ia_analysis.orbits.api", "EllipsoidalGroupModel"),
    "PhaseSpacePerturbationModel": ("ia_analysis.orbits.api", "PhaseSpacePerturbationModel"),
    "ellipsoid_shape_coefficients": ("ia_analysis.orbits.api", "ellipsoid_shape_coefficients"),
    "homogeneous_ellipsoid_tidal_tensor": ("ia_analysis.orbits.api", "homogeneous_ellipsoid_tidal_tensor"),
    "shape_tensor_from_axes": ("ia_analysis.orbits.api", "shape_tensor_from_axes"),
    "initial_shape_alignment_model": ("ia_analysis.orbits.api", "initial_shape_alignment_model"),
    "perturbation_average_features": ("ia_analysis.orbits.api", "perturbation_average_features"),
    "TidalStrippingOptions": ("ia_analysis.orbits.api", "TidalStrippingOptions"),
    "TidalStrippingHistory": ("ia_analysis.orbits.api", "TidalStrippingHistory"),
    "jacobi_tidal_radius": ("ia_analysis.orbits.api", "jacobi_tidal_radius"),
    "instantaneous_power_law_target": ("ia_analysis.orbits.api", "instantaneous_power_law_target"),
    "build_stripping_history": ("ia_analysis.orbits.api", "build_stripping_history"),
    "stripping_history_from_template": ("ia_analysis.orbits.api", "stripping_history_from_template"),
    "tidal_track_vmax_rmax": ("ia_analysis.orbits.api", "tidal_track_vmax_rmax"),
    "stripping_summary": ("ia_analysis.orbits.api", "stripping_summary"),
}

__all__ = [
    *list(_EXPORTS),
    "api",
    "halo_maker",
    "orbit_nfw",
    "template_orbits",
    "pinocchio",
    "ellipsoidal_model",
    "tidal_stripping",
]


def __getattr__(name: str) -> Any:
    """Resolve public orbit helpers lazily from the structured API facade."""
    return load_export(_EXPORTS, name)
