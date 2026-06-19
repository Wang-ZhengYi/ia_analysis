"""Orbit experiment namespace.

Purpose
-------
The orbits package contains NFW orbit experiments and synthetic halo generation
helpers used for controlled visual and dynamical tests.

Provides
--------
- Mock halo point-cloud generation.
- Orbit integration and shell-visualization inputs.
- Lightweight utilities used by curated orbit notebooks.
"""

from __future__ import annotations

from typing import Any

from ia_analysis._lazy_imports import ExportMap, load_export

_EXPORTS: ExportMap = {
    "generate_mock_halo": ("ia_analysis.orbits.api", "generate_mock_halo"),
    "generate_mock_ellipsoid": ("ia_analysis.orbits.api", "generate_mock_ellipsoid"),
    "run_orbit": ("ia_analysis.orbits.api", "run_orbit"),
    "NFWHost": ("ia_analysis.orbits.api", "NFWHost"),
    "OrbitSimulator": ("ia_analysis.orbits.api", "OrbitSimulator"),
    "OrbitResult": ("ia_analysis.orbits.api", "OrbitResult"),
}

__all__ = [*list(_EXPORTS), "api", "halo_maker", "orbit_nfw"]


def __getattr__(name: str) -> Any:
    """Resolve public orbit helpers lazily from the structured API facade."""
    return load_export(_EXPORTS, name)
