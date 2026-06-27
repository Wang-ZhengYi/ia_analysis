"""Kinematic measurements for one target in one snapshot."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ia_analysis.dynamics import halo_dynamics as hd


def measure_target_kinematics(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: Optional[np.ndarray] = None,
    *,
    center: Optional[np.ndarray] = None,
    v_ref: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    normalize_mass: bool = False,
    min_particles: int = 10,
    include_figure_rotation: bool = True,
) -> dict[str, Any]:
    """Measure affine flow, angular momentum, and optional figure rotation."""
    out = hd.compute_affine_kinematics(
        positions,
        velocities,
        masses=masses,
        center=center,
        v_ref=v_ref,
        mask=mask,
        normalize_mass=normalize_mass,
        min_particles=min_particles,
    )
    if include_figure_rotation and out.get("valid", False):
        out["figure"] = hd.figure_rotation_from_dI(out["I"], out["dI"])
        out["figure_affine"] = hd.figure_rotation_from_affine(out["I"], out["A"])
    return out


def measure_figure_rotation(shape_tensor: np.ndarray, d_shape_tensor: np.ndarray) -> dict[str, Any]:
    """Measure body-frame figure rotation directly from ``I`` and ``dI``."""
    return hd.figure_rotation_from_dI(shape_tensor, d_shape_tensor)


def measure_affine_figure_rotation(shape_tensor: np.ndarray, affine_gradient: np.ndarray) -> dict[str, Any]:
    """Measure the affine ``Omega + eta H`` figure-rotation model."""
    return hd.figure_rotation_from_affine(shape_tensor, affine_gradient)


def measure_hessian_omega(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility wrapper for measuring ``H``, ``Omega``, and figure terms."""
    return hd.measure_hessian_and_omega(*args, **kwargs)


__all__ = [
    "measure_target_kinematics",
    "measure_figure_rotation",
    "measure_affine_figure_rotation",
    "measure_hessian_omega",
]
