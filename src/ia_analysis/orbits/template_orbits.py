"""Template orbit models linked to 2LPT merger trees.

Purpose
-------
This module defines the first reusable layer for connecting fast 2LPT
mock-merger trees, such as Pinocchio products, to group-internal subhalo orbit
templates.  The goal is to generate many phase-space templates that can later
be matched to data or used to enrich HOD one-halo terms with nonlinear orbital
information.

Provides
--------
- Lightweight containers for group/subhalo tracks and relative orbit templates.
- Generic builders from already-loaded tree tables or arrays.
- Summary feature extraction for HOD one-halo augmentation.
- A documented orbit-model roadmap that can be exposed in notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TreeTrack:
    """Phase-space track for one group, halo, or subhalo across snapshots."""

    object_id: int | str
    snapshots: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    mass: np.ndarray | None = None
    scale_factor: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate track arrays and coerce them to numeric NumPy arrays."""
        snapshots = np.asarray(self.snapshots)
        positions = np.asarray(self.positions, dtype=float)
        velocities = np.asarray(self.velocities, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("`positions` must have shape (N, 3)")
        if velocities.ndim != 2 or velocities.shape[1] != 3:
            raise ValueError("`velocities` must have shape (N, 3)")
        if positions.shape != velocities.shape:
            raise ValueError("`positions` and `velocities` must have matching shapes")
        if snapshots.shape[0] != positions.shape[0]:
            raise ValueError("`snapshots` must have length N")
        object.__setattr__(self, "snapshots", snapshots)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "velocities", velocities)
        if self.mass is not None:
            mass = np.asarray(self.mass, dtype=float)
            if mass.shape[0] != positions.shape[0]:
                raise ValueError("`mass` must have length N")
            object.__setattr__(self, "mass", mass)
        if self.scale_factor is not None:
            scale = np.asarray(self.scale_factor, dtype=float)
            if scale.shape[0] != positions.shape[0]:
                raise ValueError("`scale_factor` must have length N")
            object.__setattr__(self, "scale_factor", scale)


@dataclass(frozen=True)
class OrbitTemplate:
    """Relative group-internal orbit template for one subhalo."""

    group_id: int | str
    subhalo_id: int | str
    snapshots: np.ndarray
    relative_position: np.ndarray
    relative_velocity: np.ndarray
    host_mass: np.ndarray | None = None
    subhalo_mass: np.ndarray | None = None
    scale_factor: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def radius(self) -> np.ndarray:
        """Return group-centric radius for each snapshot."""
        return np.linalg.norm(self.relative_position, axis=1)

    @property
    def speed(self) -> np.ndarray:
        """Return group-centric speed for each snapshot."""
        return np.linalg.norm(self.relative_velocity, axis=1)


@dataclass(frozen=True)
class OrbitTemplateLibrary:
    """Collection of orbit templates generated from one mock or tree catalog."""

    templates: tuple[OrbitTemplate, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def feature_matrix(self, *, keys: Sequence[str] | None = None) -> tuple[np.ndarray, tuple[str, ...]]:
        """Return an array of HOD-oriented template summary features."""
        names = tuple(keys or DEFAULT_TEMPLATE_FEATURES)
        rows = [template_feature_vector(template, names) for template in self.templates]
        if not rows:
            return np.empty((0, len(names)), dtype=float), names
        return np.vstack(rows), names


DEFAULT_TEMPLATE_FEATURES = (
    "r_final",
    "v_final",
    "r_min",
    "r_max",
    "v_radial_final",
    "v_tangential_final",
    "specific_angular_momentum_final",
    "radial_action_proxy",
)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Integrate with NumPy's current API while retaining older compatibility."""
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = getattr(np, "trapz")
    return float(integrate(y, x))


def _radial_action_proxy(radius: np.ndarray, radial_velocity: np.ndarray) -> float:
    """Integrate ``|v_r|`` along cumulative radial path length.

    This quantity is a robust orbit-shape feature for tracks whose radius can
    reverse direction.  It is not a canonical radial action: snapshots need
    not cover a full orbit, and no Hamiltonian phase-space integral is
    evaluated.
    """
    if radius.size < 2:
        return 0.0
    radial_path = np.concatenate(([0.0], np.cumsum(np.abs(np.diff(radius)))))
    return _trapezoid(np.abs(radial_velocity), radial_path)


def _minimum_image(delta: np.ndarray, boxsize: float | Sequence[float] | None) -> np.ndarray:
    """Apply a minimum-image convention when a periodic box is supplied."""
    if boxsize is None:
        return delta
    box = np.asarray(boxsize, dtype=float)
    if box.ndim == 0:
        box = np.repeat(float(box), 3)
    return delta - box * np.rint(delta / box)


def _match_snapshots(group: TreeTrack, subhalo: TreeTrack) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return common snapshot values and row indices for two tracks."""
    common, group_idx, sub_idx = np.intersect1d(group.snapshots, subhalo.snapshots, return_indices=True)
    order = np.argsort(common)
    return common[order], group_idx[order], sub_idx[order]


def build_orbit_template(
    group_track: TreeTrack,
    subhalo_track: TreeTrack,
    *,
    group_id: int | str | None = None,
    subhalo_id: int | str | None = None,
    boxsize: float | Sequence[float] | None = None,
) -> OrbitTemplate:
    """Build one relative subhalo orbit template from matched tracks."""
    snapshots, gi, si = _match_snapshots(group_track, subhalo_track)
    if snapshots.size == 0:
        raise ValueError("Tracks do not share any snapshots")
    rel_pos = _minimum_image(subhalo_track.positions[si] - group_track.positions[gi], boxsize)
    rel_vel = subhalo_track.velocities[si] - group_track.velocities[gi]
    scale = None
    if group_track.scale_factor is not None:
        scale = group_track.scale_factor[gi]
    elif subhalo_track.scale_factor is not None:
        scale = subhalo_track.scale_factor[si]
    return OrbitTemplate(
        group_id=group_track.object_id if group_id is None else group_id,
        subhalo_id=subhalo_track.object_id if subhalo_id is None else subhalo_id,
        snapshots=snapshots,
        relative_position=rel_pos,
        relative_velocity=rel_vel,
        host_mass=None if group_track.mass is None else group_track.mass[gi],
        subhalo_mass=None if subhalo_track.mass is None else subhalo_track.mass[si],
        scale_factor=scale,
        metadata={"source": "matched_tree_tracks", "n_snapshots": int(snapshots.size)},
    )


def template_feature_vector(template: OrbitTemplate, keys: Sequence[str] = DEFAULT_TEMPLATE_FEATURES) -> np.ndarray:
    """Return HOD-oriented summary features for one orbit template."""
    r = template.radius
    v = template.speed
    rel_pos = np.asarray(template.relative_position, dtype=float)
    rel_vel = np.asarray(template.relative_velocity, dtype=float)
    rhat = rel_pos / np.maximum(r[:, None], 1e-30)
    v_radial = np.einsum("ij,ij->i", rel_vel, rhat)
    v_tangential = np.sqrt(np.maximum(v * v - v_radial * v_radial, 0.0))
    angular = np.linalg.norm(np.cross(rel_pos, rel_vel), axis=1)
    radial_action_proxy = _radial_action_proxy(r, v_radial)
    values = {
        "r_final": float(r[-1]),
        "v_final": float(v[-1]),
        "r_min": float(np.nanmin(r)),
        "r_max": float(np.nanmax(r)),
        "v_radial_final": float(v_radial[-1]),
        "v_tangential_final": float(v_tangential[-1]),
        "specific_angular_momentum_final": float(angular[-1]),
        "radial_action_proxy": radial_action_proxy,
        "mass_loss_fraction": np.nan,
    }
    if template.subhalo_mass is not None and np.isfinite(template.subhalo_mass[0]) and template.subhalo_mass[0] > 0.0:
        values["mass_loss_fraction"] = float(1.0 - template.subhalo_mass[-1] / template.subhalo_mass[0])
    return np.asarray([values.get(str(key), np.nan) for key in keys], dtype=float)


def hod_1h_orbit_kernel(
    library: OrbitTemplateLibrary,
    *,
    feature_weights: Mapping[str, float] | None = None,
    keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compress a template library into mean/covariance features for HOD 1h terms."""
    matrix, names = library.feature_matrix(keys=keys)
    if matrix.size == 0:
        count = len(names)
        return {
            "feature_names": names,
            "mean": np.full(count, np.nan),
            "cov": np.full((count, count), np.nan),
            "score": np.empty(0, dtype=float),
        }
    mean = np.nanmean(matrix, axis=0)
    centered = np.nan_to_num(matrix - mean[None, :], nan=0.0)
    cov = centered.T @ centered / max(matrix.shape[0] - 1, 1)
    weights = np.asarray([float((feature_weights or {}).get(name, 1.0)) for name in names], dtype=float)
    score = np.nan_to_num(matrix, nan=0.0) @ weights
    return {"feature_names": names, "mean": mean, "cov": cov, "score": score}


def build_template_library(
    group_tracks: Mapping[int | str, TreeTrack],
    subhalo_tracks: Mapping[int | str, TreeTrack],
    host_map: Mapping[int | str, int | str],
    *,
    boxsize: float | Sequence[float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> OrbitTemplateLibrary:
    """Build a template library from group tracks, subhalo tracks, and host links."""
    templates = []
    for sub_id, sub_track in subhalo_tracks.items():
        host_id = host_map.get(sub_id)
        if host_id not in group_tracks:
            continue
        templates.append(
            build_orbit_template(
                group_tracks[host_id],
                sub_track,
                group_id=host_id,
                subhalo_id=sub_id,
                boxsize=boxsize,
            )
        )
    return OrbitTemplateLibrary(tuple(templates), metadata=dict(metadata or {}))


def orbit_template_roadmap() -> tuple[dict[str, Any], ...]:
    """Return the planned development stages for the 2LPT-template orbit model."""
    return (
        {
            "stage": "tree_ingestion",
            "goal": "Read Pinocchio or other 2LPT group trees into group and subhalo tracks.",
            "outputs": ("TreeTrack", "host_map", "snapshot metadata"),
        },
        {
            "stage": "relative_phase_space",
            "goal": "Convert tracks to group-centric subhalo positions and velocities.",
            "outputs": ("OrbitTemplate", "OrbitTemplateLibrary"),
        },
        {
            "stage": "template_generation",
            "goal": "Sample phase-space perturbations around 2LPT tracks and build many orbit templates.",
            "outputs": ("template features", "template weights", "HOD one-halo orbit kernel"),
        },
        {
            "stage": "ellipsoidal_group_model",
            "goal": "Replace spherical host assumptions with ellipsoidal group shapes and analytic tidal tensors.",
            "outputs": ("ellipsoid tidal tensor", "phase-space perturbation covariance", "aligned shape templates"),
        },
        {
            "stage": "shape_layers",
            "goal": "Start with coherent inner/outer shapes aligned with the group or tidal field, then add radial and energy-layer differences.",
            "outputs": ("layered shape parameters", "radial/energy alignment priors"),
        },
        {
            "stage": "hod_matching",
            "goal": "Fit or match template libraries to enhance HOD 1h clustering and nonlinear velocity terms.",
            "outputs": ("one-halo correction kernels", "velocity-dispersion priors", "assembly-dependent template weights"),
        },
    )


__all__ = [
    "TreeTrack",
    "OrbitTemplate",
    "OrbitTemplateLibrary",
    "DEFAULT_TEMPLATE_FEATURES",
    "build_orbit_template",
    "build_template_library",
    "template_feature_vector",
    "hod_1h_orbit_kernel",
    "orbit_template_roadmap",
]
