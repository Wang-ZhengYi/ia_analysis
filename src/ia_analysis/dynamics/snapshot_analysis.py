"""Integrated target analysis for one simulation snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ia_analysis.dynamics.dynamics_measurements import (
    measure_target_binding,
    measure_target_shell_dynamics,
)
from ia_analysis.dynamics.kinematics import measure_target_kinematics
from ia_analysis.dynamics.matrix_analysis import (
    measure_mixed_moment,
    measure_moment_derivative,
    measure_shape_tensor,
)
from ia_analysis.dynamics.shape_measurements import measure_target_shape


@dataclass
class SnapshotTarget:
    """Particle data for one target component in one snapshot."""

    positions: np.ndarray
    velocities: Optional[np.ndarray] = None
    masses: Optional[np.ndarray] = None
    center: Optional[np.ndarray] = None
    v_ref: Optional[np.ndarray] = None
    potentials: Optional[np.ndarray] = None
    accelerations: Optional[np.ndarray] = None
    component: str = "matter"
    metadata: dict[str, Any] = field(default_factory=dict)


def analyze_snapshot_target(
    target: SnapshotTarget | Mapping[str, Any],
    *,
    include_shape: bool = True,
    include_kinematics: bool = True,
    include_dynamics: bool = True,
    include_matrix: bool = True,
    shape_kwargs: Optional[dict[str, Any]] = None,
    kinematics_kwargs: Optional[dict[str, Any]] = None,
    dynamics_kwargs: Optional[dict[str, Any]] = None,
    shell_masks: Optional[Sequence[np.ndarray]] = None,
    host_tidal_tensor: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """
    Analyze one snapshot target with separated shape/kinematic/dynamic sections.

    The returned dictionary is intentionally sectional:
    ``shape`` contains iterative shape measurement, ``kinematics`` contains
    velocity-gradient and figure-rotation quantities, ``dynamics`` contains
    binding and shell dynamics, and ``matrix`` contains raw moment tensors.
    """
    tgt = _coerce_target(target)
    pos = np.asarray(tgt.positions, dtype=np.float64)
    vel = None if tgt.velocities is None else np.asarray(tgt.velocities, dtype=np.float64)
    masses = None if tgt.masses is None else np.asarray(tgt.masses, dtype=np.float64)
    center = None if tgt.center is None else np.asarray(tgt.center, dtype=np.float64)
    v_ref = None if tgt.v_ref is None else np.asarray(tgt.v_ref, dtype=np.float64)
    rel_pos = pos if center is None else pos - center[None, :]

    out: dict[str, Any] = {
        "component": tgt.component,
        "center": center,
        "v_ref": v_ref,
        "metadata": dict(tgt.metadata),
    }

    shape_section = None
    if include_shape:
        opts = {} if shape_kwargs is None else dict(shape_kwargs)
        shape_section = measure_target_shape(
            pos,
            masses=masses,
            center=center,
            velocities=vel,
            accelerations=tgt.accelerations,
            **opts,
        )
        out["shape"] = shape_section

    kin_section = None
    if include_kinematics and vel is not None:
        opts = {} if kinematics_kwargs is None else dict(kinematics_kwargs)
        if "mask" not in opts and shape_section is not None:
            opts["mask"] = shape_section.get("mask")
        kin_section = measure_target_kinematics(
            pos,
            vel,
            masses=masses,
            center=center,
            v_ref=v_ref,
            **opts,
        )
        out["kinematics"] = kin_section

    if include_matrix:
        matrix: dict[str, Any] = {"shape_tensor": measure_shape_tensor(rel_pos, masses=masses)}
        if vel is not None:
            rel_vel = vel if v_ref is None else vel - v_ref[None, :]
            matrix["mixed_moment"] = measure_mixed_moment(rel_pos, rel_vel, masses=masses)
            matrix["d_shape_tensor"] = measure_moment_derivative(rel_pos, rel_vel, masses=masses)
        out["matrix"] = matrix

    if include_dynamics and vel is not None:
        opts = {} if dynamics_kwargs is None else dict(dynamics_kwargs)
        dyn: dict[str, Any] = {}
        dyn["binding"] = measure_target_binding(
            pos,
            vel,
            masses=masses,
            center=center,
            v_ref=v_ref,
            potentials=tgt.potentials,
            component=tgt.component,
            **opts.pop("binding_kwargs", {}),
        )
        dyn["shells"] = measure_target_shell_dynamics(
            pos,
            vel,
            masses=masses,
            center=center,
            v_ref=v_ref,
            shell_masks=shell_masks,
            host_tidal_tensor=host_tidal_tensor,
            **opts.pop("shell_kwargs", {}),
        )
        if opts:
            dyn["unused_options"] = opts
        out["dynamics"] = dyn

    return out


def _coerce_target(target: SnapshotTarget | Mapping[str, Any]) -> SnapshotTarget:
    if isinstance(target, SnapshotTarget):
        return target
    data = dict(target)
    return SnapshotTarget(**data)


__all__ = ["SnapshotTarget", "analyze_snapshot_target"]
