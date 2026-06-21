"""Structured orbit-experiment API facade.

Purpose
-------
Orbit experiments combine mock halo generation, NFW host profiles, integration,
2LPT merger-tree orbit templates, ellipsoidal group approximations, and
visualization inputs.  This facade exposes those pieces with clear names while
leaving the historical numerical implementations in place.

Provides
--------
- Mock NFW point-cloud generation and ellipsoidal transformations.
- NFW host, orbit simulator, and result-container aliases.
- Pinocchio-like tree adapters and relative subhalo orbit templates.
- Ellipsoidal tidal approximations and phase-space perturbation averages.
- Unit-conversion and dynamical-friction helpers for controlled experiments.
"""

from __future__ import annotations

from typing import Any

from ia_analysis._lazy_imports import ExportMap, call_export, export_names, load_export

_EXPORTS: ExportMap = {
    "generate_nfw_points": ("ia_analysis.orbits.halo_maker", "gen_nfw"),
    "transform_points_to_ellipsoid": ("ia_analysis.orbits.halo_maker", "transform_points_to_ellipsoid"),
    "create_rotation_matrix": ("ia_analysis.orbits.halo_maker", "create_rotation_matrix"),
    "NFWHost": ("ia_analysis.orbits.orbit_nfw", "NFWHost"),
    "OrbitSimulator": ("ia_analysis.orbits.orbit_nfw", "OrbitSimulator"),
    "OrbitResult": ("ia_analysis.orbits.orbit_nfw", "OrbitResult"),
    "TreeTrack": ("ia_analysis.orbits.template_orbits", "TreeTrack"),
    "OrbitTemplate": ("ia_analysis.orbits.template_orbits", "OrbitTemplate"),
    "OrbitTemplateLibrary": ("ia_analysis.orbits.template_orbits", "OrbitTemplateLibrary"),
    "build_orbit_template": ("ia_analysis.orbits.template_orbits", "build_orbit_template"),
    "build_template_library": ("ia_analysis.orbits.template_orbits", "build_template_library"),
    "template_feature_vector": ("ia_analysis.orbits.template_orbits", "template_feature_vector"),
    "hod_1h_orbit_kernel": ("ia_analysis.orbits.template_orbits", "hod_1h_orbit_kernel"),
    "orbit_template_roadmap": ("ia_analysis.orbits.template_orbits", "orbit_template_roadmap"),
    "PinocchioColumnMap": ("ia_analysis.orbits.pinocchio", "PinocchioColumnMap"),
    "tracks_from_table": ("ia_analysis.orbits.pinocchio", "tracks_from_table"),
    "host_map_from_table": ("ia_analysis.orbits.pinocchio", "host_map_from_table"),
    "build_pinocchio_template_library": ("ia_analysis.orbits.pinocchio", "build_pinocchio_template_library"),
    "EllipsoidalGroupModel": ("ia_analysis.orbits.ellipsoidal_model", "EllipsoidalGroupModel"),
    "PhaseSpacePerturbationModel": ("ia_analysis.orbits.ellipsoidal_model", "PhaseSpacePerturbationModel"),
    "homogeneous_ellipsoid_tidal_tensor": (
        "ia_analysis.orbits.ellipsoidal_model",
        "homogeneous_ellipsoid_tidal_tensor",
    ),
    "coherent_layer_shapes": ("ia_analysis.orbits.ellipsoidal_model", "coherent_layer_shapes"),
    "tidal_aligned_shape": ("ia_analysis.orbits.ellipsoidal_model", "tidal_aligned_shape"),
    "perturb_orbit_template": ("ia_analysis.orbits.ellipsoidal_model", "perturb_orbit_template"),
    "perturbation_average_features": ("ia_analysis.orbits.ellipsoidal_model", "perturbation_average_features"),
    "initial_shape_alignment_model": ("ia_analysis.orbits.ellipsoidal_model", "initial_shape_alignment_model"),
    "default_tng_cosmology": ("ia_analysis.orbits.orbit_nfw", "default_tng_cosmology"),
    "chandrasekhar_df_accel_kpc": ("ia_analysis.orbits.orbit_nfw", "chandrasekhar_df_accel_kpc"),
    "ckpc_h_to_kpc_phys": ("ia_analysis.orbits.orbit_nfw", "ckpc_h_to_kpc_phys"),
    "kpc_phys_to_ckpc_h": ("ia_analysis.orbits.orbit_nfw", "kpc_phys_to_ckpc_h"),
    "soften_ckpc_h_to_kpc_phys": ("ia_analysis.orbits.orbit_nfw", "soften_ckpc_h_to_kpc_phys"),
    "mass_1e10_msun_h_to_msun": ("ia_analysis.orbits.orbit_nfw", "mass_1e10_msun_h_to_msun"),
    "mass_msun_to_1e10_msun_h": ("ia_analysis.orbits.orbit_nfw", "mass_msun_to_1e10_msun_h"),
}

__all__ = [
    *export_names(_EXPORTS),
    "generate_mock_halo",
    "generate_mock_ellipsoid",
    "run_orbit",
]


def __getattr__(name: str) -> Any:
    """Resolve orbit-experiment aliases lazily."""
    return load_export(_EXPORTS, name)


def generate_mock_halo(*args: Any, **kwargs: Any) -> Any:
    """Generate a spherical mock NFW point cloud."""
    return call_export(_EXPORTS, "generate_nfw_points", *args, **kwargs)


def generate_mock_ellipsoid(*args: Any, a: float, b: float, c: float, principal_axis: Any, **kwargs: Any) -> Any:
    """Generate a mock NFW point cloud and transform it into an ellipsoid."""
    points = generate_mock_halo(*args, **kwargs)
    return call_export(_EXPORTS, "transform_points_to_ellipsoid", points, a, b, c, principal_axis)


def run_orbit(
    *,
    simulator: Any | None = None,
    simulator_kwargs: dict[str, Any] | None = None,
    **run_kwargs: Any,
) -> Any:
    """Run an NFW orbit using an existing simulator or simulator parameters."""
    if simulator is None:
        cls = load_export(_EXPORTS, "OrbitSimulator")
        simulator = cls(**(simulator_kwargs or {}))
    return simulator.run(**run_kwargs)
