"""Potential and tidal-field namespace.

Purpose
-------
The tides package computes gravitational potential grids, tidal tensors, and
interpolated tensor samples from array inputs.

Provides
--------
- CIC gridding utilities.
- Fourier-space potential and tidal tensor construction.
- Interpolation helpers for sampling tensor fields at object positions.
"""

from __future__ import annotations

from typing import Any

from ia_analysis._lazy_imports import ExportMap, load_export

_EXPORTS: ExportMap = {
    "compute_gravitational_potential": ("ia_analysis.tides.api", "compute_gravitational_potential"),
    "grid_potential_and_tidal": ("ia_analysis.tides.api", "grid_potential_and_tidal"),
    "PotentialInterpolator": ("ia_analysis.tides.api", "PotentialInterpolator"),
    "interpolate_potential_and_tidal": ("ia_analysis.tides.api", "interpolate_potential_and_tidal"),
    "build_mass_tidal_grid": ("ia_analysis.tides.api", "build_mass_tidal_grid"),
    "build_sampled_tidal_grid": ("ia_analysis.tides.api", "build_sampled_tidal_grid"),
    "build_tidal_grid": ("ia_analysis.tides.api", "build_tidal_grid"),
    "sample_tidal_grid": ("ia_analysis.tides.api", "sample_tidal_grid"),
    "tidal_eigenvalues": ("ia_analysis.tides.api", "tidal_eigenvalues"),
    "tidal_strength": ("ia_analysis.tides.api", "tidal_strength"),
    "tidal_anisotropy": ("ia_analysis.tides.api", "tidal_anisotropy"),
    "summarize_tidal_series": ("ia_analysis.tides.api", "summarize_tidal_series"),
}

__all__ = [*list(_EXPORTS), "api", "tidal_field", "diagnostics"]


def __getattr__(name: str) -> Any:
    """Resolve public tidal-field helpers lazily from the structured API facade."""
    return load_export(_EXPORTS, name)
