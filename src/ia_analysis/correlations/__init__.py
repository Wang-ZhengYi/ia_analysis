"""Real-space correlation-function namespace.

Purpose
-------
This package contains array-level estimators for scalar, vector, and tensor
correlation functions used by IA, velocity, and halo-dynamics analyses.  It is
kept separate from the power-spectrum package because the estimators operate
directly on catalog positions and object-level fields.

Provides
--------
- Catalog and field containers for scalar, vector, and tensor measurements.
- Pair-binned two-point correlations for density, shape, velocity, and figure
  rotation fields.
- Total, one-halo, two-halo, central-central, central-satellite, and
  satellite-satellite categories.
- Jackknife/bootstrap covariance helpers and a compressed four-point estimator.
"""

from __future__ import annotations

from typing import Any

from ia_analysis._lazy_imports import ExportMap, load_export

_EXPORTS: ExportMap = {
    "CorrelationCatalog": ("ia_analysis.correlations.api", "CorrelationCatalog"),
    "PairSpec": ("ia_analysis.correlations.api", "PairSpec"),
    "default_pair_specs": ("ia_analysis.correlations.api", "default_pair_specs"),
    "SUMMARY_CATEGORIES": ("ia_analysis.correlations.api", "SUMMARY_CATEGORIES"),
    "DETAILED_CATEGORIES": ("ia_analysis.correlations.api", "DETAILED_CATEGORIES"),
    "DEFAULT_CATEGORIES": ("ia_analysis.correlations.api", "DEFAULT_CATEGORIES"),
    "CorrelationResult": ("ia_analysis.correlations.api", "CorrelationResult"),
    "CorrelationSuiteResult": ("ia_analysis.correlations.api", "CorrelationSuiteResult"),
    "measure_two_point": ("ia_analysis.correlations.api", "measure_two_point"),
    "measure_many_two_point": ("ia_analysis.correlations.api", "measure_many_two_point"),
    "measure_default_correlations": ("ia_analysis.correlations.api", "measure_default_correlations"),
    "estimate_vedomega_four_point": ("ia_analysis.correlations.api", "estimate_vedomega_four_point"),
    "connected_vedomega_four_point": ("ia_analysis.correlations.api", "connected_vedomega_four_point"),
    "jackknife_tags": ("ia_analysis.correlations.api", "jackknife_tags"),
    "covariance_from_samples": ("ia_analysis.correlations.api", "covariance_from_samples"),
    "jackknife_covariance": ("ia_analysis.correlations.api", "jackknife_covariance"),
    "bootstrap_covariance": ("ia_analysis.correlations.api", "bootstrap_covariance"),
    "write_results_hdf5": ("ia_analysis.correlations.api", "write_results_hdf5"),
}

__all__ = [
    *list(_EXPORTS),
    "api",
    "fields",
    "estimators",
    "covariance",
    "four_point",
    "suite",
    "io",
]


def __getattr__(name: str) -> Any:
    """Resolve public correlation helpers lazily from the API facade."""
    return load_export(_EXPORTS, name)
