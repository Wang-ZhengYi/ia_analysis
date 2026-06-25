"""Tidal stripping utilities for orbit-template experiments.

Purpose
-------
This module provides lightweight, dependency-free stripping models that can be
used by orbit notebooks, Pinocchio-like merger-tree templates, and later HOD
one-halo modeling.  The functions are deliberately separated from the full NFW
integrator so that template post-processing can run in a small NumPy-only
environment.

Provides
--------
- Jacobi tidal-radius estimates from local orbital curvature.
- Instantaneous power-law target masses for a truncated subhalo profile.
- Delayed, irreversible mass-loss histories with a configurable stripping
  timescale.
- Penarrubia-style tidal-track ratios for $Vmax$ and $rmax$ diagnostics.
- Convenience conversion from an ``OrbitTemplate`` relative phase-space track
  to a stripping history.

Notes
-----
The default delayed model should be treated as a controlled semi-analytic
approximation, not a calibrated universal law.  Real applications should fit
``tau_orbits`` and the tidal-track coefficients against simulations matched to
the host mass, concentration, orbit distribution, and numerical resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

import numpy as np


# Newton's constant in kpc * (km/s)^2 / Msun.
G_KPC_KMS2_PER_MSUN = 4.30091727003628e-6


@dataclass(frozen=True)
class TidalStrippingOptions:
    """Configuration for the semi-analytic stripping post-processor.

    Parameters
    ----------
    mode
        ``"instantaneous_powerlaw"`` applies the target tidal mass at each
        sample and enforces monotonic stripping. ``"delayed_powerlaw"`` relaxes
        toward the same target over ``tau_orbits`` local orbital periods.
    density_slope
        Inner power-law slope used in the closure ``rho_sub ~ r^-alpha``.
        The enclosed mass scales as ``M(<r) ~ r^(3-alpha)``.
    tau_orbits
        Mass-loss relaxation time in units of the local orbital period.  This
        parameter regularizes strong pericentric impulses that would otherwise
        remove all exterior mass instantly.
    minimum_bound_fraction
        Numerical floor on the bound mass fraction.  It avoids hard disruption
        in lightweight template experiments.
    irreversible
        If True, bound mass never grows when the instantaneous target mass
        becomes larger after pericentre.
    tidal-track coefficients
        The ``vmax_*`` and ``rmax_*`` values define normalized tidal-track
        ratios of the form ``2^c x^s / (1+x)^c`` with ``x = M/M0``.  The
        defaults are common Penarrubia-style starting values and should be
        recalibrated for production work.
    """

    mode: str = "delayed_powerlaw"
    density_slope: float = 2.0
    curvature_floor: float = 1.0e-12
    tau_orbits: float = 0.75
    minimum_bound_fraction: float = 1.0e-4
    irreversible: bool = True
    vmax_slope: float = 0.30
    vmax_curvature: float = 0.40
    rmax_slope: float = 0.40
    rmax_curvature: float = -0.30

    def validate(self) -> None:
        """Raise a clear error when option values are outside model bounds."""
        if self.mode not in {"instantaneous_powerlaw", "delayed_powerlaw"}:
            raise ValueError("mode must be 'instantaneous_powerlaw' or 'delayed_powerlaw'")
        if not (0.0 < float(self.density_slope) < 3.0):
            raise ValueError("density_slope must satisfy 0 < density_slope < 3")
        if float(self.curvature_floor) <= 0.0:
            raise ValueError("curvature_floor must be positive")
        if float(self.tau_orbits) <= 0.0:
            raise ValueError("tau_orbits must be positive")
        if not (0.0 < float(self.minimum_bound_fraction) <= 1.0):
            raise ValueError("minimum_bound_fraction must be in (0, 1]")
        if not isinstance(self.irreversible, (bool, np.bool_)):
            raise ValueError("irreversible must be a boolean")
        coefficients = (
            self.vmax_slope,
            self.vmax_curvature,
            self.rmax_slope,
            self.rmax_curvature,
        )
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("tidal-track coefficients must be finite")

    def updated(self, **kwargs: Any) -> "TidalStrippingOptions":
        """Return a copy with selected option values replaced."""
        out = replace(self, **kwargs)
        out.validate()
        return out


@dataclass(frozen=True)
class TidalStrippingHistory:
    """Container for a post-processed stripping time series."""

    time: np.ndarray
    radius: np.ndarray
    omega: np.ndarray
    host_curvature: np.ndarray
    tidal_radius: np.ndarray
    target_mass: np.ndarray
    bound_mass: np.ndarray
    mass_fraction: np.ndarray
    mass_loss_rate: np.ndarray
    orbital_time: np.ndarray
    vmax_ratio: np.ndarray
    rmax_ratio: np.ndarray
    options: TidalStrippingOptions

    def as_dict(self) -> dict[str, Any]:
        """Return a plain dictionary useful for HDF5, plotting, or notebooks."""
        return {
            "time": self.time,
            "radius": self.radius,
            "omega": self.omega,
            "host_curvature": self.host_curvature,
            "tidal_radius": self.tidal_radius,
            "target_mass": self.target_mass,
            "bound_mass": self.bound_mass,
            "mass_fraction": self.mass_fraction,
            "mass_loss_rate": self.mass_loss_rate,
            "orbital_time": self.orbital_time,
            "vmax_ratio": self.vmax_ratio,
            "rmax_ratio": self.rmax_ratio,
            "options": self.options,
        }


def _as_1d(name: str, value: Any, *, length: int | None = None) -> np.ndarray:
    """Convert ``value`` to a finite one-dimensional float array."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        if length is None:
            arr = arr.reshape(1)
        else:
            arr = np.full(length, float(arr), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if length is not None and arr.shape[0] != length:
        raise ValueError(f"{name} must have length {length}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def jacobi_tidal_radius(
    bound_mass: Any,
    omega: Any,
    host_curvature: Any,
    *,
    gravitational_constant: float = G_KPC_KMS2_PER_MSUN,
    curvature_floor: float = 1.0e-12,
) -> np.ndarray:
    """Estimate the Jacobi radius from local orbital curvature.

    Parameters
    ----------
    bound_mass
        Subhalo mass in the mass unit paired with ``gravitational_constant``.
    omega
        Instantaneous angular frequency in velocity per length units.
    host_curvature
        Radial second derivative of the host potential in velocity squared per
        length squared.  The effective denominator is ``Omega^2 - Phi''``.
    gravitational_constant
        Unit-consistent gravitational constant.  Use the module default for
        ``Msun``, ``kpc``, and ``km/s``.  Use ``1`` for dimensionless demos.
    curvature_floor
        Positive floor for the denominator.
    """
    omega_arr = _as_1d("omega", omega)
    curvature_arr = _as_1d("host_curvature", host_curvature, length=omega_arr.size)
    mass_arr = _as_1d("bound_mass", bound_mass, length=omega_arr.size)
    if float(gravitational_constant) <= 0.0:
        raise ValueError("gravitational_constant must be positive")
    if float(curvature_floor) <= 0.0:
        raise ValueError("curvature_floor must be positive")

    mass_arr = np.maximum(mass_arr, 0.0)
    denominator = np.maximum(omega_arr * omega_arr - curvature_arr, float(curvature_floor))
    return np.power(float(gravitational_constant) * mass_arr / denominator, 1.0 / 3.0)


def power_law_mass_fraction(
    tidal_radius: Any,
    reference_radius: float,
    *,
    density_slope: float = 2.0,
    minimum_fraction: float = 0.0,
) -> np.ndarray:
    """Return the retained mass fraction for a power-law subhalo profile."""
    if not (0.0 < float(density_slope) < 3.0):
        raise ValueError("density_slope must satisfy 0 < density_slope < 3")
    if float(reference_radius) <= 0.0:
        raise ValueError("reference_radius must be positive")
    if float(minimum_fraction) < 0.0:
        raise ValueError("minimum_fraction must be non-negative")

    tidal = _as_1d("tidal_radius", tidal_radius)
    exponent = 3.0 - float(density_slope)
    fraction = np.power(np.clip(tidal / float(reference_radius), 0.0, 1.0), exponent)
    return np.maximum(fraction, float(minimum_fraction))


def instantaneous_power_law_target(
    *,
    mass0: float,
    reference_radius: float,
    omega: Any,
    host_curvature: Any,
    options: TidalStrippingOptions | None = None,
    gravitational_constant: float = G_KPC_KMS2_PER_MSUN,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the instantaneous target bound mass and tidal radius.

    The target uses the same closure as the historical orbit notebook: a
    fixed-mass Jacobi radius is first compared with the reference subhalo size.
    Once that radius falls inside the reference size, the bound mass is solved
    algebraically for a power-law density profile.
    """
    opts = options or TidalStrippingOptions()
    opts.validate()
    if float(mass0) <= 0.0:
        raise ValueError("mass0 must be positive")
    if float(reference_radius) <= 0.0:
        raise ValueError("reference_radius must be positive")

    omega_arr = _as_1d("omega", omega)
    curvature_arr = _as_1d("host_curvature", host_curvature, length=omega_arr.size)
    denominator = np.maximum(omega_arr * omega_arr - curvature_arr, float(opts.curvature_floor))
    fixed_radius = np.power(float(gravitational_constant) * float(mass0) / denominator, 1.0 / 3.0)

    alpha = float(opts.density_slope)
    gamma = 3.0 - alpha
    reference = float(reference_radius)
    stripped_radius = np.power(float(gravitational_constant) * float(mass0) / denominator, 1.0 / alpha)
    stripped_radius *= reference ** (-gamma / alpha)

    is_stripped = fixed_radius < reference
    tidal_radius = np.where(is_stripped, stripped_radius, fixed_radius)
    fraction = np.where(
        is_stripped,
        power_law_mass_fraction(
            stripped_radius,
            reference,
            density_slope=alpha,
            minimum_fraction=opts.minimum_bound_fraction,
        ),
        1.0,
    )
    target_mass = float(mass0) * np.clip(fraction, opts.minimum_bound_fraction, 1.0)
    return target_mass, tidal_radius


def orbital_time_from_omega(omega: Any, *, minimum_abs_omega: float = 1.0e-12) -> np.ndarray:
    """Return a local orbital period estimate ``2*pi / |Omega|``."""
    omega_arr = np.abs(_as_1d("omega", omega))
    if float(minimum_abs_omega) <= 0.0:
        raise ValueError("minimum_abs_omega must be positive")
    return 2.0 * np.pi / np.maximum(omega_arr, float(minimum_abs_omega))


def tidal_track_ratio(mass_fraction: Any, *, slope: float, curvature: float, floor: float = 1.0e-6) -> np.ndarray:
    """Return a normalized tidal-track ratio as a smooth function of mass loss."""
    x = np.clip(_as_1d("mass_fraction", mass_fraction), float(floor), 1.0)
    return (2.0 ** float(curvature)) * np.power(x, float(slope)) / np.power(1.0 + x, float(curvature))


def tidal_track_vmax_rmax(
    mass_fraction: Any,
    *,
    options: TidalStrippingOptions | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return approximate tidal-track ratios for ``Vmax`` and ``rmax``."""
    opts = options or TidalStrippingOptions()
    opts.validate()
    vmax = tidal_track_ratio(
        mass_fraction,
        slope=opts.vmax_slope,
        curvature=opts.vmax_curvature,
        floor=opts.minimum_bound_fraction,
    )
    rmax = tidal_track_ratio(
        mass_fraction,
        slope=opts.rmax_slope,
        curvature=opts.rmax_curvature,
        floor=opts.minimum_bound_fraction,
    )
    return vmax, rmax


def build_stripping_history(
    *,
    time: Any,
    radius: Any,
    omega: Any,
    host_curvature: Any,
    mass0: float,
    reference_radius: float,
    options: TidalStrippingOptions | None = None,
    gravitational_constant: float = G_KPC_KMS2_PER_MSUN,
) -> TidalStrippingHistory:
    """Build an instantaneous or delayed semi-analytic stripping history.

    This is a controlled approximation for orbit-template experiments, not a
    calibrated universal subhalo stripping law.  With ``irreversible=True``,
    the returned bound mass is guaranteed to be monotonic non-increasing.
    """
    opts = options or TidalStrippingOptions()
    opts.validate()

    time_arr = _as_1d("time", time)
    if time_arr.size == 0:
        raise ValueError("time must contain at least one sample")
    if time_arr.size > 1 and np.any(np.diff(time_arr) <= 0.0):
        raise ValueError("time must be strictly increasing")

    radius_arr = _as_1d("radius", radius, length=time_arr.size)
    omega_arr = _as_1d("omega", omega, length=time_arr.size)
    curvature_arr = _as_1d("host_curvature", host_curvature, length=time_arr.size)
    target_mass, tidal_radius = instantaneous_power_law_target(
        mass0=mass0,
        reference_radius=reference_radius,
        omega=omega_arr,
        host_curvature=curvature_arr,
        options=opts,
        gravitational_constant=gravitational_constant,
    )

    floor_mass = float(mass0) * float(opts.minimum_bound_fraction)
    orbital_time = orbital_time_from_omega(omega_arr)
    bound_mass = np.empty_like(target_mass)

    if opts.mode == "instantaneous_powerlaw":
        bound_mass[:] = target_mass
        if opts.irreversible:
            bound_mass[:] = np.minimum.accumulate(bound_mass)
        bound_mass[:] = np.maximum(bound_mass, floor_mass)
    else:
        bound_mass[0] = float(mass0)
        for i in range(1, time_arr.size):
            dt = float(time_arr[i] - time_arr[i - 1])
            previous = float(bound_mass[i - 1])
            target = float(target_mass[i])
            if opts.irreversible:
                target = min(target, previous)
            tau = max(float(opts.tau_orbits) * float(orbital_time[i]), 1.0e-30)
            response = 1.0 - np.exp(-dt / tau)
            if target < previous:
                next_mass = previous + (target - previous) * response
            else:
                next_mass = target if not opts.irreversible else previous
            bound_mass[i] = max(float(next_mass), floor_mass)
        if opts.irreversible:
            bound_mass[:] = np.minimum.accumulate(bound_mass)

    if time_arr.size >= 3:
        mass_loss_rate = np.gradient(bound_mass, time_arr)
    elif time_arr.size == 2:
        slope = (bound_mass[1] - bound_mass[0]) / (time_arr[1] - time_arr[0])
        mass_loss_rate = np.array([slope, slope], dtype=float)
    else:
        mass_loss_rate = np.zeros_like(bound_mass)

    mass_fraction = np.clip(bound_mass / float(mass0), opts.minimum_bound_fraction, 1.0)
    vmax_ratio, rmax_ratio = tidal_track_vmax_rmax(mass_fraction, options=opts)
    return TidalStrippingHistory(
        time=time_arr,
        radius=radius_arr,
        omega=omega_arr,
        host_curvature=curvature_arr,
        tidal_radius=tidal_radius,
        target_mass=target_mass,
        bound_mass=bound_mass,
        mass_fraction=mass_fraction,
        mass_loss_rate=mass_loss_rate,
        orbital_time=orbital_time,
        vmax_ratio=vmax_ratio,
        rmax_ratio=rmax_ratio,
        options=opts,
    )


def angular_frequency_from_phase_space(position: Any, velocity: Any, *, radius_floor: float = 1.0e-12) -> np.ndarray:
    """Estimate ``|r x v| / |r|^2`` for a relative phase-space track."""
    pos = np.asarray(position, dtype=float)
    vel = np.asarray(velocity, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("position must have shape (N, 3)")
    if vel.shape != pos.shape:
        raise ValueError("velocity must have the same shape as position")
    if float(radius_floor) <= 0.0:
        raise ValueError("radius_floor must be positive")
    radius = np.linalg.norm(pos, axis=1)
    angular = np.linalg.norm(np.cross(pos, vel), axis=1)
    return angular / np.maximum(radius * radius, float(radius_floor))


def template_host_curvature_powerlaw(
    radius: Any,
    *,
    amplitude: float = 1.0,
    scale_radius: float = 1.0,
    exponent: float = 3.0,
) -> np.ndarray:
    """Return a simple negative curvature proxy for dimensionless demos.

    This helper is useful for orbit-template notebooks where a full host
    potential has not been attached yet.  Production analyses should replace it
    with curvatures sampled from an NFW, Einasto, or ellipsoidal potential.
    """
    if float(amplitude) < 0.0:
        raise ValueError("amplitude must be non-negative")
    if float(scale_radius) <= 0.0:
        raise ValueError("scale_radius must be positive")
    if float(exponent) <= 0.0:
        raise ValueError("exponent must be positive")
    radius_arr = np.maximum(_as_1d("radius", radius), 0.0)
    return -float(amplitude) / np.power(radius_arr + float(scale_radius), float(exponent))


def stripping_history_from_template(
    template: Any,
    *,
    time: Any | None = None,
    mass0: float | None = None,
    reference_radius: float | None = None,
    host_curvature: Any | None = None,
    host_curvature_model: Callable[[np.ndarray], np.ndarray] | None = None,
    options: TidalStrippingOptions | None = None,
    gravitational_constant: float = 1.0,
) -> TidalStrippingHistory:
    """Build a stripping history from an ``OrbitTemplate``-like object."""
    pos = np.asarray(template.relative_position, dtype=float)
    vel = np.asarray(template.relative_velocity, dtype=float)
    radius = np.linalg.norm(pos, axis=1)
    omega = angular_frequency_from_phase_space(pos, vel)

    if time is None:
        time_arr = np.asarray(template.snapshots, dtype=float)
    else:
        time_arr = _as_1d("time", time, length=radius.size)
    if mass0 is None:
        if getattr(template, "subhalo_mass", None) is None:
            mass0 = 1.0
        else:
            mass_arr = np.asarray(template.subhalo_mass, dtype=float)
            mass0 = float(mass_arr[0])
    if reference_radius is None:
        reference_radius = 0.2 * float(np.nanmax(radius)) if np.nanmax(radius) > 0.0 else 1.0
    if host_curvature is None:
        model = host_curvature_model or template_host_curvature_powerlaw
        curvature = model(radius)
    else:
        curvature = _as_1d("host_curvature", host_curvature, length=radius.size)

    return build_stripping_history(
        time=time_arr,
        radius=radius,
        omega=omega,
        host_curvature=curvature,
        mass0=float(mass0),
        reference_radius=float(reference_radius),
        options=options,
        gravitational_constant=float(gravitational_constant),
    )


def stripping_summary(history: TidalStrippingHistory) -> Mapping[str, float]:
    """Return compact features that are useful for matching orbit templates."""
    return {
        "final_mass_fraction": float(history.mass_fraction[-1]),
        "minimum_tidal_radius": float(np.nanmin(history.tidal_radius)),
        "maximum_mass_loss_rate": float(max(0.0, -np.nanmin(history.mass_loss_rate))),
        "final_vmax_ratio": float(history.vmax_ratio[-1]),
        "final_rmax_ratio": float(history.rmax_ratio[-1]),
    }


__all__ = [
    "G_KPC_KMS2_PER_MSUN",
    "TidalStrippingOptions",
    "TidalStrippingHistory",
    "jacobi_tidal_radius",
    "power_law_mass_fraction",
    "instantaneous_power_law_target",
    "orbital_time_from_omega",
    "tidal_track_ratio",
    "tidal_track_vmax_rmax",
    "build_stripping_history",
    "angular_frequency_from_phase_space",
    "template_host_curvature_powerlaw",
    "stripping_history_from_template",
    "stripping_summary",
]
