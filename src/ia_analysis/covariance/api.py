"""Structured covariance API facade.

Purpose
-------
This module exposes the covariance builder, model containers, and individual
Gaussian/cNG/SSC components under concise names.  It keeps the implementation
in ``Cov.py`` for compatibility with existing command-line usage.

Provides
--------
- Array-level covariance construction for tests and notebooks.
- HDF5 writer access for measured spectra products.
- Lazy command entrypoint dispatch for ``python -m ia_analysis.covariance.Cov``.
"""

from __future__ import annotations

from typing import Any

from ia_analysis._lazy_imports import ExportMap, call_export, export_names, load_export

_EXPORTS: ExportMap = {
    "HaloModelOptions": ("ia_analysis.covariance.Cov", "HaloModelOptions"),
    "CovarianceModel": ("ia_analysis.covariance.Cov", "CovarianceModel"),
    "build_covariance_from_arrays": ("ia_analysis.covariance.Cov", "build_covariance_from_arrays"),
    "write_covariance_hdf5_group": ("ia_analysis.covariance.Cov", "write_covariance_hdf5_group"),
    "merge_spectra_for_covariance": ("ia_analysis.covariance.Cov", "merge_spectra_for_covariance"),
    "gaussian_covariance": ("ia_analysis.covariance.Cov", "gaussian_covariance"),
    "cng_covariance_halomodel": ("ia_analysis.covariance.Cov", "cng_covariance_halomodel"),
    "ssc_covariance": ("ia_analysis.covariance.Cov", "ssc_covariance"),
    "nla_spectra": ("ia_analysis.covariance.Cov", "nla_spectra"),
    "infer_bias_and_aia": ("ia_analysis.covariance.Cov", "infer_bias_and_aia"),
    "run_covariance_cli": ("ia_analysis.covariance.Cov", "main"),
}

__all__ = [*export_names(_EXPORTS), "build_covariance", "run_cli"]


def __getattr__(name: str) -> Any:
    """Resolve covariance aliases lazily."""
    return load_export(_EXPORTS, name)


def build_covariance(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Build covariance products directly from in-memory arrays."""
    return call_export(_EXPORTS, "build_covariance_from_arrays", *args, **kwargs)


def run_cli() -> None:
    """Dispatch to the historical covariance command-line entrypoint."""
    call_export(_EXPORTS, "run_covariance_cli")

