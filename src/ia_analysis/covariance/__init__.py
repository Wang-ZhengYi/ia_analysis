"""Covariance model namespace for IA power-spectrum products.

Purpose
-------
The covariance package groups Gaussian, connected non-Gaussian, and
super-sample covariance tools used after spectra have been measured.

Provides
--------
- File-oriented covariance entrypoints for HDF5 power-spectrum products.
- Array-level helpers for covariance assembly and noise bookkeeping.

Notes
-----
Covariance modules consume spectra outputs and should not orchestrate catalog or
mesh-generation pipelines directly.
"""

from __future__ import annotations

from typing import Any

from ia_analysis._lazy_imports import ExportMap, load_export

_EXPORTS: ExportMap = {
    "HaloModelOptions": ("ia_analysis.covariance.api", "HaloModelOptions"),
    "CovarianceModel": ("ia_analysis.covariance.api", "CovarianceModel"),
    "build_covariance": ("ia_analysis.covariance.api", "build_covariance"),
    "build_covariance_from_arrays": ("ia_analysis.covariance.api", "build_covariance_from_arrays"),
    "write_covariance_hdf5_group": ("ia_analysis.covariance.api", "write_covariance_hdf5_group"),
    "gaussian_covariance": ("ia_analysis.covariance.api", "gaussian_covariance"),
    "cng_covariance_halomodel": ("ia_analysis.covariance.api", "cng_covariance_halomodel"),
    "ssc_covariance": ("ia_analysis.covariance.api", "ssc_covariance"),
}

__all__ = [*list(_EXPORTS), "api", "Cov"]


def __getattr__(name: str) -> Any:
    """Resolve public covariance helpers lazily from the structured API facade."""
    return load_export(_EXPORTS, name)
