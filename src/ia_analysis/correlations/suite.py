"""High-level correlation measurement suite.

Purpose
-------
This module orchestrates the default IA, velocity, and figure-rotation
correlation products from in-memory arrays.  It keeps catalog loading outside
the correlations package and focuses on combining field pairs, halo/sample
categories, optional four-point products, and empirical covariance.

Provides
--------
- Default pair suite for ``ee``, ``ed``, ``dd``, ``vv``, ``dv``, ``ev``, and
  omega cross-correlations.
- Total/1h/2h summaries plus five detailed central/satellite halo categories.
- Optional jackknife covariance for the complete suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ia_analysis.correlations.covariance import covariance_from_samples, jackknife_tags
from ia_analysis.correlations.estimators import DEFAULT_CATEGORIES, CorrelationResult, measure_many_two_point
from ia_analysis.correlations.fields import CorrelationCatalog, PairSpec, default_pair_specs
from ia_analysis.correlations.four_point import connected_vedomega_four_point, estimate_vedomega_four_point


@dataclass
class CorrelationSuiteResult:
    """Container for the default correlation suite."""

    results: dict[str, CorrelationResult]
    covariance: dict[str, dict[str, dict[str, np.ndarray]]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def names(self) -> tuple[str, ...]:
        """Return measured statistic names."""
        return tuple(self.results)


def _measure_suite_once(
    catalog: CorrelationCatalog,
    rbins: Sequence[float],
    *,
    specs: Sequence[PairSpec],
    categories: Sequence[str],
    include_four_point: bool,
    include_connected_four_point: bool,
) -> dict[str, CorrelationResult]:
    """Measure the suite once without covariance recursion."""
    results = measure_many_two_point(catalog, specs, rbins, categories=categories)
    if include_four_point:
        raw = estimate_vedomega_four_point(catalog, rbins, categories=categories)
        results[raw.name] = raw
        if include_connected_four_point:
            connected = connected_vedomega_four_point(raw, results)
            results[connected.name] = connected
    return results


def _jackknife_suite_covariance(
    catalog: CorrelationCatalog,
    rbins: Sequence[float],
    *,
    specs: Sequence[PairSpec],
    categories: Sequence[str],
    nsub: int,
    include_four_point: bool,
    include_connected_four_point: bool,
    include_samples: bool,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Jackknife covariance for all two-point and four-point suite products."""
    tags = jackknife_tags(catalog, nsub=nsub)
    regions = np.unique(tags)
    stacks: dict[str, dict[str, list[np.ndarray]]] = {}
    for region in regions:
        sub = catalog.subset(tags != region)
        measured = _measure_suite_once(
            sub,
            rbins,
            specs=specs,
            categories=categories,
            include_four_point=include_four_point,
            include_connected_four_point=include_connected_four_point,
        )
        for name, result in measured.items():
            stacks.setdefault(name, {str(category): [] for category in categories})
            for category in categories:
                stacks[name][str(category)].append(result.values[str(category)])

    out: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for name, by_category in stacks.items():
        out[name] = {}
        for category, sample_list in by_category.items():
            samples = np.asarray(sample_list, dtype=float)
            mean, cov = covariance_from_samples(samples, kind="jackknife")
            entry = {"mean": mean, "cov": cov}
            if include_samples:
                entry["samples"] = samples
            out[name][category] = entry
    return out


def measure_default_correlations(
    catalog: CorrelationCatalog,
    rbins: Sequence[float],
    *,
    specs: Sequence[PairSpec] | None = None,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
    include_four_point: bool = True,
    include_connected_four_point: bool = True,
    covariance: str | None = None,
    nsub: int = 3,
    include_covariance_samples: bool = False,
) -> CorrelationSuiteResult:
    """Measure the default two-point and four-point correlation suite.

    The detailed categories are the five non-overlapping central/satellite
    halo classes: ``1h_cs``, ``1h_ss``, ``2h_cc``, ``2h_cs``, and ``2h_ss``.
    The summary categories ``total``, ``1h``, and ``2h`` are also retained for
    direct comparison with older correlation-function products.
    """
    pair_specs = tuple(default_pair_specs(include_omega=True) if specs is None else specs)
    measured = _measure_suite_once(
        catalog,
        rbins,
        specs=pair_specs,
        categories=categories,
        include_four_point=include_four_point,
        include_connected_four_point=include_connected_four_point,
    )
    cov: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    cov_mode = None if covariance is None else str(covariance).strip().lower()
    if cov_mode in {"jackknife", "jk"}:
        cov = _jackknife_suite_covariance(
            catalog,
            rbins,
            specs=pair_specs,
            categories=categories,
            nsub=nsub,
            include_four_point=include_four_point,
            include_connected_four_point=include_connected_four_point,
            include_samples=include_covariance_samples,
        )
    elif cov_mode not in {None, "", "none"}:
        raise ValueError("Only covariance=None or covariance='jackknife' is supported by the suite")

    return CorrelationSuiteResult(
        results=measured,
        covariance=cov,
        metadata={
            "n_objects": catalog.size,
            "catalog": catalog.name,
            "categories": tuple(str(c) for c in categories),
            "detailed_categories": ("1h_cs", "1h_ss", "2h_cc", "2h_cs", "2h_ss"),
            "covariance": cov_mode or "none",
        },
    )


__all__ = [
    "CorrelationSuiteResult",
    "measure_default_correlations",
]
