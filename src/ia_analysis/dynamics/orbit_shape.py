"""Multi-initial-condition orbit, stripping, and shape-response analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ia_analysis.shapes.evolution import acute_axis_angle_deg, evolve_shape_tensor, sorted_eigh_symmetric
from ia_analysis.tides.diagnostics import tidal_anisotropy, tidal_strength


@dataclass(frozen=True)
class OrbitInitialCondition:
    """One planar orbit initial condition in the orbit simulator's units."""

    name: str
    r0: float
    phi0: float
    vr0: float
    vt0: float


DEFAULT_ORBIT_CASES = (
    OrbitInitialCondition("fiducial_infall", 450.0, 0.25, -80.0, 140.0),
    OrbitInitialCondition("near_circular", 450.0, 0.25, -10.0, 210.0),
    OrbitInitialCondition("radial_plunge", 450.0, 0.25, -180.0, 70.0),
    OrbitInitialCondition("outer_high_L", 650.0, 0.60, -35.0, 220.0),
    OrbitInitialCondition("inner_fast", 260.0, 1.10, -60.0, 185.0),
    OrbitInitialCondition("outgoing_phase", 320.0, 2.00, 110.0, 145.0),
)


def _position_velocity(result: Any) -> tuple[np.ndarray, np.ndarray]:
    """Build three-dimensional position and velocity arrays from an orbit result."""
    zeros = np.zeros_like(np.asarray(result.x, dtype=float))
    return (
        np.column_stack((result.x, result.y, zeros)),
        np.column_stack((result.vx, result.vy, zeros)),
    )


def run_orbit_shape_case(
    simulator: Any,
    case: OrbitInitialCondition,
    *,
    initial_shape_tensor: np.ndarray,
    m_sub: float,
    soften: float = 0.0,
    alpha_sub: float = 2.0,
    t_end: float = 3.0,
    dt: float = 0.01,
    n_shape_orbits: float = 0.5,
    with_df: bool = False,
    ln_lambda: float = 3.0,
) -> dict[str, Any]:
    """Run one orbit and return orbit, shape history, tidy table, and summary."""
    length_factor = float(simulator.a / simulator.h)
    energy = 0.5 * (case.vr0**2 + case.vt0**2) + simulator.host.phi(
        case.r0 * length_factor,
        soften_kpc=float(soften) * length_factor,
    )
    angular_momentum = case.r0 * case.vt0
    orbit = simulator.run(
        E=energy,
        L=angular_momentum,
        r0=case.r0,
        phi0=case.phi0,
        vr_sign=1 if case.vr0 >= 0.0 else -1,
        t_end=t_end,
        dt=dt,
        soften=soften,
        with_df=with_df,
        with_strip=True,
        m_sub=m_sub,
        lnLambda=ln_lambda,
        alpha_sub=alpha_sub,
        gamma_p=alpha_sub,
        show_progress=False,
    )
    positions, velocities = _position_velocity(orbit)
    mass = np.asarray(orbit.M, dtype=float)
    mass_fraction = np.clip(mass / max(float(mass[0]), 1.0e-30), 1.0e-4, 1.0)
    shape = evolve_shape_tensor(
        orbit.t,
        orbit.T_in,
        mass_fraction,
        positions,
        velocities,
        initial_shape_tensor,
        n_shape_orbits=n_shape_orbits,
        physical_kpc_per_position_unit=length_factor,
    )
    strength = tidal_strength(orbit.T_in)
    anisotropy = tidal_anisotropy(orbit.T_in)
    radial_angle = np.empty(len(orbit.t), dtype=float)
    for index, (tensor, position) in enumerate(zip(shape["S"], positions)):
        _, frame = sorted_eigh_symmetric(tensor)
        radial_angle[index] = acute_axis_angle_deg(frame[:, 0], position)
    mode = "with_df" if with_df else "conservative"
    table = pd.DataFrame(
        {
            "time_gyr": orbit.t,
            "radius_ckpc_h": orbit.r,
            "bound_mass_1e10_msun_h": mass,
            "mass_fraction": mass_fraction,
            "b_over_a": shape["q"][:, 0],
            "c_over_a": shape["q"][:, 1],
            "shape_tide_angle_deg": shape["angle_major_tide_major_deg"],
            "shape_radial_angle_deg": radial_angle,
            "tau_shape_gyr": shape["tau_shape_gyr"],
            "tidal_strength": strength,
            "tidal_anisotropy": anisotropy,
        }
    )
    summary = {
        "case": case.name,
        "mode": mode,
        "r0_ckpc_h": case.r0,
        "vr0_kms": case.vr0,
        "vt0_kms": case.vt0,
        "specific_energy_kms2": energy,
        "specific_angular_momentum": angular_momentum,
        "samples": len(orbit.t),
        "merged": bool(orbit.merged),
        "pericentre_ckpc_h": float(np.nanmin(orbit.r)),
        "apocentre_ckpc_h": float(np.nanmax(orbit.r)),
        "final_mass_fraction": float(mass_fraction[-1]),
        "maximum_tidal_strength": float(np.nanmax(strength)),
        "maximum_tidal_anisotropy": float(np.nanmax(anisotropy)),
        "final_b_over_a": float(shape["q"][-1, 0]),
        "final_c_over_a": float(shape["q"][-1, 1]),
        "final_shape_tide_angle_deg": float(shape["angle_major_tide_major_deg"][-1]),
        "final_shape_radial_angle_deg": float(radial_angle[-1]),
        "median_shape_tide_angle_deg": float(np.nanmedian(shape["angle_major_tide_major_deg"])),
        "median_shape_radial_angle_deg": float(np.nanmedian(radial_angle)),
        "minimum_shape_eigenvalue": float(min(np.linalg.eigvalsh(tensor).min() for tensor in shape["S"])),
    }
    return {"orbit": orbit, "shape": shape, "table": table, "summary": summary}


def run_orbit_shape_suite(
    simulator: Any,
    *,
    initial_shape_tensor: np.ndarray,
    m_sub: float,
    cases: Iterable[OrbitInitialCondition] = DEFAULT_ORBIT_CASES,
    friction_modes: Iterable[bool] = (False, True),
    output_dir: str | Path | None = None,
    **case_kwargs: Any,
) -> tuple[dict[tuple[str, str], dict[str, Any]], pd.DataFrame]:
    """Run a grid of orbit initial conditions and optionally write CSV products."""
    runs: dict[tuple[str, str], dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    out = None if output_dir is None else Path(output_dir)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    for case in cases:
        for with_df in friction_modes:
            product = run_orbit_shape_case(
                simulator,
                case,
                initial_shape_tensor=initial_shape_tensor,
                m_sub=m_sub,
                with_df=bool(with_df),
                **case_kwargs,
            )
            mode = product["summary"]["mode"]
            runs[(case.name, mode)] = product
            summaries.append(product["summary"])
            if out is not None:
                product["table"].to_csv(out / f"{case.name}__{mode}.csv", index=False)
    summary = pd.DataFrame(summaries)
    if out is not None:
        summary.to_csv(out / "orbit_shape_initial_condition_summary.csv", index=False)
    return runs, summary


__all__ = [
    "OrbitInitialCondition",
    "DEFAULT_ORBIT_CASES",
    "run_orbit_shape_case",
    "run_orbit_shape_suite",
]
