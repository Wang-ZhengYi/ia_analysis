"""Ellipsoidal group and phase-space perturbation models.

Purpose
-------
The orbit-template program should not be restricted to spherical halos.  This
module provides a lightweight ellipsoidal approximation for group tidal fields,
coherent inner/outer subhalo shape templates, and phase-space perturbation
averages around fast 2LPT orbit tracks.

Provides
--------
- Homogeneous-ellipsoid interior tidal tensor in the principal-axis frame.
- Rotation of the ellipsoid tidal tensor into the simulation frame.
- Coherent and tidal-aligned layer shape approximations.
- Gaussian phase-space perturbation sampling around orbit templates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ia_analysis.orbits.template_orbits import OrbitTemplate, OrbitTemplateLibrary


@dataclass(frozen=True)
class EllipsoidalGroupModel:
    """Approximate ellipsoidal group model used for orbit and shape templates."""

    axes: tuple[float, float, float]
    orientation: np.ndarray
    mass: float = 1.0
    gravitational_constant: float = 1.0

    def __post_init__(self) -> None:
        """Validate axes and orientation matrix."""
        axes = tuple(float(x) for x in self.axes)
        if len(axes) != 3 or min(axes) <= 0.0:
            raise ValueError("`axes` must contain three positive semi-axis lengths")
        orient = np.asarray(self.orientation, dtype=float)
        if orient.shape != (3, 3):
            raise ValueError("`orientation` must have shape (3, 3)")
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "orientation", orient)


@dataclass(frozen=True)
class PhaseSpacePerturbationModel:
    """Gaussian perturbation model for orbit-template libraries."""

    radial_position_sigma: float = 0.05
    tangential_position_sigma: float = 0.05
    radial_velocity_sigma: float = 0.10
    tangential_velocity_sigma: float = 0.10
    mass_log_sigma: float = 0.0
    seed: int | None = None


def ellipsoid_shape_coefficients(axes: Sequence[float], *, n_quad: int = 160) -> np.ndarray:
    """Return homogeneous-ellipsoid shape coefficients A_i with sum close to 2.

    The coefficients use the standard integral
    ``A_i = abc int_0^inf d tau / ((a_i^2 + tau) Delta(tau))``.
    A change of variable maps the infinite interval to [0, 1].
    """
    a, b, c = np.asarray(axes, dtype=float)
    nodes, weights = np.polynomial.legendre.leggauss(int(n_quad))
    u = 0.5 * (nodes + 1.0)
    w = 0.5 * weights
    tau = u / np.maximum(1.0 - u, 1e-14)
    jac = 1.0 / np.maximum(1.0 - u, 1e-14) ** 2
    delta = np.sqrt((a * a + tau) * (b * b + tau) * (c * c + tau))
    coeffs = []
    for axis in (a, b, c):
        integrand = a * b * c * jac / ((axis * axis + tau) * delta)
        coeffs.append(float(np.sum(w * integrand)))
    return np.asarray(coeffs, dtype=float)


def homogeneous_ellipsoid_tidal_tensor(model: EllipsoidalGroupModel, *, n_quad: int = 160) -> np.ndarray:
    """Return the constant interior tidal tensor for a homogeneous ellipsoid."""
    axes = np.asarray(model.axes, dtype=float)
    volume = 4.0 * np.pi / 3.0 * float(np.prod(axes))
    density = float(model.mass) / max(volume, 1e-30)
    coeffs = ellipsoid_shape_coefficients(axes, n_quad=n_quad)
    principal = np.diag(2.0 * np.pi * float(model.gravitational_constant) * density * coeffs)
    orient = np.asarray(model.orientation, dtype=float)
    return orient @ principal @ orient.T


def shape_tensor_from_axes(axes: Sequence[float], orientation: np.ndarray) -> np.ndarray:
    """Return a positive-definite shape tensor aligned with ``orientation``."""
    values = np.asarray(axes, dtype=float) ** 2
    orient = np.asarray(orientation, dtype=float)
    return orient @ np.diag(values) @ orient.T


def coherent_layer_shapes(
    group_model: EllipsoidalGroupModel,
    *,
    inner_axis_ratios: Sequence[float] = (1.0, 0.75, 0.55),
    outer_axis_ratios: Sequence[float] = (1.0, 0.70, 0.50),
) -> dict[str, np.ndarray]:
    """Return same-orientation inner and outer shape tensors."""
    return {
        "inner": shape_tensor_from_axes(inner_axis_ratios, group_model.orientation),
        "outer": shape_tensor_from_axes(outer_axis_ratios, group_model.orientation),
    }


def tidal_aligned_shape(
    tidal_tensor: np.ndarray,
    *,
    axis_ratios: Sequence[float] = (1.0, 0.75, 0.55),
    align_with: str = "most_compressive",
) -> np.ndarray:
    """Return a shape tensor aligned with one eigenframe of a tidal tensor."""
    tensor = np.asarray(tidal_tensor, dtype=float)
    evals, evecs = np.linalg.eigh(tensor)
    if align_with == "most_extensive":
        order = np.argsort(evals)[::-1]
    else:
        order = np.argsort(evals)
    orientation = evecs[:, order]
    return shape_tensor_from_axes(axis_ratios, orientation)


def _orthonormal_radial_frame(position: np.ndarray) -> np.ndarray:
    """Return radial, tangential-1, tangential-2 unit vectors."""
    r = np.asarray(position, dtype=float)
    er = r / max(float(np.linalg.norm(r)), 1e-30)
    trial = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(er, trial))) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    et1 = np.cross(er, trial)
    et1 /= max(float(np.linalg.norm(et1)), 1e-30)
    et2 = np.cross(er, et1)
    return np.vstack((er, et1, et2))


def perturb_orbit_template(
    template: OrbitTemplate,
    model: PhaseSpacePerturbationModel,
    *,
    n_samples: int = 64,
) -> OrbitTemplateLibrary:
    """Sample phase-space perturbations around one orbit template."""
    rng = np.random.default_rng(model.seed)
    templates = []
    rel_pos = np.asarray(template.relative_position, dtype=float)
    rel_vel = np.asarray(template.relative_velocity, dtype=float)
    for sample_id in range(int(n_samples)):
        pos_out = np.empty_like(rel_pos)
        vel_out = np.empty_like(rel_vel)
        for i, (pos, vel) in enumerate(zip(rel_pos, rel_vel)):
            frame = _orthonormal_radial_frame(pos)
            r_scale = max(float(np.linalg.norm(pos)), 1.0)
            v_scale = max(float(np.linalg.norm(vel)), 1.0)
            dpos_local = np.array(
                [
                    rng.normal(0.0, model.radial_position_sigma * r_scale),
                    rng.normal(0.0, model.tangential_position_sigma * r_scale),
                    rng.normal(0.0, model.tangential_position_sigma * r_scale),
                ]
            )
            dvel_local = np.array(
                [
                    rng.normal(0.0, model.radial_velocity_sigma * v_scale),
                    rng.normal(0.0, model.tangential_velocity_sigma * v_scale),
                    rng.normal(0.0, model.tangential_velocity_sigma * v_scale),
                ]
            )
            pos_out[i] = pos + dpos_local @ frame
            vel_out[i] = vel + dvel_local @ frame
        sub_mass = template.subhalo_mass
        if sub_mass is not None and model.mass_log_sigma > 0.0:
            sub_mass = sub_mass * np.exp(rng.normal(0.0, model.mass_log_sigma, size=sub_mass.shape))
        templates.append(
            OrbitTemplate(
                group_id=template.group_id,
                subhalo_id=f"{template.subhalo_id}:perturbed:{sample_id}",
                snapshots=template.snapshots,
                relative_position=pos_out,
                relative_velocity=vel_out,
                host_mass=template.host_mass,
                subhalo_mass=sub_mass,
                scale_factor=template.scale_factor,
                metadata={**dict(template.metadata), "perturbation_sample": sample_id},
            )
        )
    return OrbitTemplateLibrary(tuple(templates), metadata={"source": "phase_space_perturbations", "n_samples": int(n_samples)})


def perturbation_average_features(
    template: OrbitTemplate,
    model: PhaseSpacePerturbationModel,
    *,
    n_samples: int = 64,
    keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return mean/covariance of template features under phase-space perturbations."""
    library = perturb_orbit_template(template, model, n_samples=n_samples)
    matrix, names = library.feature_matrix(keys=keys)
    mean = np.nanmean(matrix, axis=0)
    centered = np.nan_to_num(matrix - mean[None, :], nan=0.0)
    cov = centered.T @ centered / max(matrix.shape[0] - 1, 1)
    return {"feature_names": names, "mean": mean, "cov": cov, "samples": matrix}


def initial_shape_alignment_model(
    group_model: EllipsoidalGroupModel,
    *,
    mode: str = "coherent",
    inner_axis_ratios: Sequence[float] = (1.0, 0.75, 0.55),
    outer_axis_ratios: Sequence[float] = (1.0, 0.70, 0.50),
) -> dict[str, np.ndarray]:
    """Return first-stage subhalo shape approximations for the orbit program."""
    if mode == "coherent":
        return coherent_layer_shapes(
            group_model,
            inner_axis_ratios=inner_axis_ratios,
            outer_axis_ratios=outer_axis_ratios,
        )
    if mode == "tidal_aligned":
        tide = homogeneous_ellipsoid_tidal_tensor(group_model)
        return {
            "inner": tidal_aligned_shape(tide, axis_ratios=inner_axis_ratios),
            "outer": tidal_aligned_shape(tide, axis_ratios=outer_axis_ratios),
        }
    raise ValueError("`mode` must be 'coherent' or 'tidal_aligned'")


__all__ = [
    "EllipsoidalGroupModel",
    "PhaseSpacePerturbationModel",
    "ellipsoid_shape_coefficients",
    "homogeneous_ellipsoid_tidal_tensor",
    "shape_tensor_from_axes",
    "coherent_layer_shapes",
    "tidal_aligned_shape",
    "perturb_orbit_template",
    "perturbation_average_features",
    "initial_shape_alignment_model",
]
