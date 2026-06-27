"""Dynamical measurements for one target in one snapshot."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ia_analysis.dynamics import halo_dynamics as hd


def measure_target_binding(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: Optional[np.ndarray] = None,
    *,
    center: Optional[np.ndarray] = None,
    v_ref: Optional[np.ndarray] = None,
    potentials: Optional[np.ndarray] = None,
    component: str = "matter",
    **kwargs: Any,
) -> dict[str, Any]:
    """Measure per-particle specific binding energy for one target component."""
    return hd.component_binding_energy(
        positions,
        velocities,
        masses=masses,
        center=center,
        v_ref=v_ref,
        potentials=potentials,
        component=component,
        **kwargs,
    )


def measure_component_binding_profiles(
    components: Mapping[str, Mapping[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Measure binding-energy mass profiles for several target components."""
    return hd.component_binding_energy_profiles(components, **kwargs)


def measure_target_shell_dynamics(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: Optional[np.ndarray] = None,
    *,
    center: Optional[np.ndarray] = None,
    v_ref: Optional[np.ndarray] = None,
    shell_method: str = "radial",
    shell_masks: Optional[Sequence[np.ndarray]] = None,
    shell_kwargs: Optional[dict[str, Any]] = None,
    host_tidal_tensor: Optional[np.ndarray] = None,
    internal_hessians: Optional[Sequence[np.ndarray]] = None,
    min_particles: int = 10,
) -> dict[str, Any]:
    """Run shell-wise dynamics for a target component."""
    return hd.analyze_halo_shells(
        positions,
        velocities,
        masses=masses,
        center=center,
        v_ref=v_ref,
        shell_method=shell_method,
        shell_masks=shell_masks,
        shell_kwargs=shell_kwargs,
        host_tidal_tensor=host_tidal_tensor,
        internal_hessians=internal_hessians,
        min_particles=min_particles,
    )


def measure_tidal_response(
    shape_tensor: np.ndarray,
    affine_gradient: np.ndarray,
    *,
    residual_dispersion: Optional[np.ndarray] = None,
    internal_hessian: Optional[np.ndarray] = None,
    external_hessian: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Measure slow-shape source terms from flow, dispersion, gravity, and tides."""
    return hd.tidal_response_terms(
        shape_tensor,
        affine_gradient,
        S=residual_dispersion,
        G_int=internal_hessian,
        T_ext=external_hessian,
    )


__all__ = [
    "measure_target_binding",
    "measure_component_binding_profiles",
    "measure_target_shell_dynamics",
    "measure_tidal_response",
]
