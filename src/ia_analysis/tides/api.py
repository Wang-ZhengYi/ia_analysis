"""Structured potential and tidal-field API facade.

Purpose
-------
This facade exposes one compact entrypoint for mass-derived, potential-derived,
and acceleration-derived tidal tensors.  The numerical implementation remains
in ``tidal_field.py`` and is imported lazily.

Provides
--------
- Tidal-grid construction from particles, scalar potentials, or accelerations.
- Stable aliases for potential interpolation helpers.
- Small wrappers with explicit names for notebook and pipeline readability.
"""

from __future__ import annotations

from typing import Any

from ia_analysis._lazy_imports import ExportMap, call_export, export_names, load_export

_EXPORTS: ExportMap = {
    "compute_gravitational_potential": ("ia_analysis.tides.tidal_field", "compute_gravitational_potential"),
    "grid_potential_and_tidal": ("ia_analysis.tides.tidal_field", "grid_potential_and_tidal"),
    "PotentialInterpolator": ("ia_analysis.tides.tidal_field", "PotentialInterpolator"),
    "interpolate_potential_and_tidal": ("ia_analysis.tides.tidal_field", "interpolate_potential_and_tidal"),
}

__all__ = [
    *export_names(_EXPORTS),
    "build_mass_tidal_grid",
    "build_sampled_tidal_grid",
    "build_tidal_grid",
    "sample_tidal_grid",
]


def __getattr__(name: str) -> Any:
    """Resolve tidal-field aliases lazily from the implementation module."""
    return load_export(_EXPORTS, name)


def build_mass_tidal_grid(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compute potential and tidal tensors from particle positions and masses."""
    return call_export(_EXPORTS, "compute_gravitational_potential", *args, **kwargs)


def build_sampled_tidal_grid(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Grid sampled potentials or accelerations and compute tidal tensors."""
    return call_export(_EXPORTS, "grid_potential_and_tidal", *args, **kwargs)


def build_tidal_grid(positions: Any, values: Any, *, source: str = "mass", **kwargs: Any) -> dict[str, Any]:
    """Build a tidal grid from mass, potential, or acceleration samples."""
    mode = str(source).strip().lower()
    if mode in {"mass", "masses", "density"}:
        return build_mass_tidal_grid(positions, values, **kwargs)
    if mode in {"potential", "phi"}:
        return build_sampled_tidal_grid(positions, values, input_type="potential", **kwargs)
    if mode in {"acceleration", "accelerations", "accel"}:
        return build_sampled_tidal_grid(positions, values, input_type="acceleration", **kwargs)
    raise ValueError("Tidal-grid source must be 'mass', 'potential', or 'acceleration'.")


def sample_tidal_grid(*args: Any, **kwargs: Any) -> Any:
    """Interpolate potential and tidal tensor values at one query position."""
    return call_export(_EXPORTS, "interpolate_potential_and_tidal", *args, **kwargs)

