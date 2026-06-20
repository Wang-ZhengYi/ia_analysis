"""Top-level convenience registry for IA analysis domain APIs.

Purpose
-------
The project is organized by scientific responsibility.  This module provides a
small discovery layer so scripts and notebooks can find the recommended API
facade for each domain without importing heavy catalog, plotting, or HPC
dependencies at package-import time.

Provides
--------
- A registry of supported functional domains.
- Lazy loading of domain API modules.
- Human-readable descriptions for notebooks and command-line helpers.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

_DOMAIN_APIS = {
    "catalogs": ("ia_analysis.catalogs.api", "ClusterSims and IllustrisTNG catalog loading."),
    "shapes": ("ia_analysis.shapes.api", "Shape tensors, axes, spin, and IA projections."),
    "tides": ("ia_analysis.tides.api", "Potential grids, tidal tensors, and interpolation."),
    "dynamics": ("ia_analysis.dynamics.api", "Shell-wise halo dynamics and TNG wrappers."),
    "merger_tree": ("ia_analysis.MergerTree.api", "Merger-tree reading and cross-time workflow orchestration."),
    "correlations": ("ia_analysis.correlations.api", "Real-space IA, density, velocity, and rotation correlations."),
    "covariance": ("ia_analysis.covariance.api", "Gaussian, cNG, and SSC covariance helpers."),
    "pipelines": ("ia_analysis.pipelines.api", "Discoverable end-to-end pipeline entrypoints."),
    "orbits": ("ia_analysis.orbits.api", "NFW orbit experiments and mock halo generation."),
    "visualization": ("ia_analysis.visualization.api", "Plotting, animation, and figure helpers."),
}

__all__ = ["available_domains", "describe_domains", "load_domain_api"]


def available_domains() -> tuple[str, ...]:
    """Return the functional domains that expose structured API facades."""
    return tuple(_DOMAIN_APIS)


def describe_domains() -> dict[str, str]:
    """Return short descriptions for all structured API domains."""
    return {name: description for name, (_, description) in _DOMAIN_APIS.items()}


def load_domain_api(domain: str) -> ModuleType:
    """Import and return the API facade module for one functional domain."""
    key = str(domain).strip().lower()
    if key not in _DOMAIN_APIS:
        valid = ", ".join(available_domains())
        raise KeyError(f"Unknown IA analysis API domain {domain!r}. Available domains: {valid}")
    module_name, _ = _DOMAIN_APIS[key]
    return import_module(module_name)
