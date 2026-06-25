"""Structured API facade for real-space correlations.

Purpose
-------
This facade exposes the stable public functions for measuring IA, density,
velocity, and figure-rotation correlation functions without importing heavy
catalog or plotting code at package import time.

Provides
--------
- Catalog and pair-spec containers.
- Two-point and default-suite measurement entrypoints.
- Empirical covariance and compressed four-point helpers.
- HDF5 output for measured correlation products.
"""

from __future__ import annotations

from typing import Any

from ia_analysis._lazy_imports import ExportMap, call_export, export_names, load_export

_EXPORTS: ExportMap = {
    "CorrelationCatalog": ("ia_analysis.correlations.fields", "CorrelationCatalog"),
    "PairSpec": ("ia_analysis.correlations.fields", "PairSpec"),
    "default_pair_specs": ("ia_analysis.correlations.fields", "default_pair_specs"),
    "SUMMARY_CATEGORIES": ("ia_analysis.correlations.estimators", "SUMMARY_CATEGORIES"),
    "DETAILED_CATEGORIES": ("ia_analysis.correlations.estimators", "DETAILED_CATEGORIES"),
    "DEFAULT_CATEGORIES": ("ia_analysis.correlations.estimators", "DEFAULT_CATEGORIES"),
    "CorrelationResult": ("ia_analysis.correlations.estimators", "CorrelationResult"),
    "measure_two_point": ("ia_analysis.correlations.estimators", "measure_two_point"),
    "measure_many_two_point": ("ia_analysis.correlations.estimators", "measure_many_two_point"),
    "CorrelationSuiteResult": ("ia_analysis.correlations.suite", "CorrelationSuiteResult"),
    "measure_default_correlations": ("ia_analysis.correlations.suite", "measure_default_correlations"),
    "estimate_vedomega_four_point": ("ia_analysis.correlations.four_point", "estimate_vedomega_four_point"),
    "connected_vedomega_four_point": ("ia_analysis.correlations.four_point", "connected_vedomega_four_point"),
    "jackknife_tags": ("ia_analysis.correlations.covariance", "jackknife_tags"),
    "covariance_from_samples": ("ia_analysis.correlations.covariance", "covariance_from_samples"),
    "jackknife_covariance": ("ia_analysis.correlations.covariance", "jackknife_covariance"),
    "bootstrap_covariance": ("ia_analysis.correlations.covariance", "bootstrap_covariance"),
    "write_results_hdf5": ("ia_analysis.correlations.io", "write_results_hdf5"),
    "discover_correlation_files": ("ia_analysis.correlations.quality", "discover_correlation_files"),
    "covariance_diagnostics": ("ia_analysis.correlations.quality", "covariance_diagnostics"),
    "signal_to_noise": ("ia_analysis.correlations.quality", "signal_to_noise"),
    "inspect_correlation_file": ("ia_analysis.correlations.quality", "inspect_correlation_file"),
    "summarize_correlation_quality": ("ia_analysis.correlations.quality", "summarize_correlation_quality"),
}

__all__ = [*export_names(_EXPORTS), "measure_suite"]


def __getattr__(name: str) -> Any:
    """Resolve public correlation helpers lazily."""
    return load_export(_EXPORTS, name)


def measure_suite(*args: Any, **kwargs: Any) -> Any:
    """Compatibility alias for the default correlation suite."""
    return call_export(_EXPORTS, "measure_default_correlations", *args, **kwargs)
