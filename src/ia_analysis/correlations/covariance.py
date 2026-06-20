"""Empirical covariance helpers for real-space correlations.

Purpose
-------
Correlation measurements need covariance estimates for each field pair and each
halo/sample category.  This module provides lightweight empirical covariance
tools that work directly with the array-level estimators in this package.

Provides
--------
- Cubic jackknife region assignment for periodic or non-periodic boxes.
- Jackknife covariance for all requested field pairs and categories.
- Bootstrap covariance for stacked measurements from mocks, jackknife samples,
  or independent realizations.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ia_analysis.correlations.estimators import DEFAULT_CATEGORIES, measure_many_two_point
from ia_analysis.correlations.fields import CorrelationCatalog, PairSpec


def jackknife_tags(catalog: CorrelationCatalog, nsub: int = 3) -> np.ndarray:
    """Assign each object to one cubic jackknife region."""
    nsub = int(nsub)
    if nsub <= 0:
        raise ValueError("`nsub` must be positive")
    pos = np.asarray(catalog.positions, dtype=float)
    if catalog.boxsize is None:
        lo = np.nanmin(pos, axis=0)
        hi = np.nanmax(pos, axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)
        unit = (pos - lo) / span
    else:
        box = np.asarray(catalog.boxsize, dtype=float)
        unit = np.mod(pos, box) / box
    idx = np.minimum((unit * nsub).astype(int), nsub - 1)
    return idx[:, 0] + nsub * idx[:, 1] + nsub * nsub * idx[:, 2]


def covariance_from_samples(samples: Sequence[Sequence[float]], *, kind: str = "sample") -> tuple[np.ndarray, np.ndarray]:
    """Return mean and covariance from stacked one-dimensional measurements."""
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 2:
        raise ValueError("`samples` must have shape (Nsample, Nbin)")
    finite = np.isfinite(arr)
    counts = np.count_nonzero(finite, axis=0)
    summed = np.sum(np.where(finite, arr, 0.0), axis=0)
    mean = np.full(arr.shape[1], np.nan, dtype=float)
    good_mean = counts > 0
    mean[good_mean] = summed[good_mean] / counts[good_mean]
    diff = arr - mean[None, :]
    if kind == "jackknife":
        factor = (arr.shape[0] - 1.0) / max(arr.shape[0], 1)
    else:
        factor = 1.0 / max(arr.shape[0] - 1, 1)
    cov = factor * np.nan_to_num(diff, nan=0.0).T @ np.nan_to_num(diff, nan=0.0)
    bad = counts == 0
    if np.any(bad):
        cov[bad, :] = np.nan
        cov[:, bad] = np.nan
    return mean, cov


def jackknife_covariance(
    catalog: CorrelationCatalog,
    specs: Sequence[PairSpec],
    rbins: Sequence[float],
    *,
    nsub: int = 3,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
    include_samples: bool = False,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Estimate covariance by leaving out one cubic region at a time."""
    tags = jackknife_tags(catalog, nsub=nsub)
    regions = np.unique(tags)
    stacks: dict[str, dict[str, list[np.ndarray]]] = {
        spec.output_name(): {str(category): [] for category in categories} for spec in specs
    }
    for region in regions:
        sub = catalog.subset(tags != region)
        measured = measure_many_two_point(sub, specs, rbins, categories=categories)
        for name, result in measured.items():
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


def bootstrap_covariance(
    samples: Sequence[Sequence[float]],
    *,
    n_boot: int = 256,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Estimate covariance of the mean by bootstrap resampling rows."""
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 2:
        raise ValueError("`samples` must have shape (Nsample, Nbin)")
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)
    n_sample = arr.shape[0]
    boot_means = np.empty((int(n_boot), arr.shape[1]), dtype=float)
    for i in range(int(n_boot)):
        take = rng.integers(0, n_sample, size=n_sample)
        finite = np.isfinite(arr[take])
        counts = np.count_nonzero(finite, axis=0)
        summed = np.sum(np.where(finite, arr[take], 0.0), axis=0)
        boot_means[i] = np.where(counts > 0, summed / np.maximum(counts, 1), np.nan)
    mean, cov = covariance_from_samples(boot_means, kind="sample")
    return {"mean": mean, "cov": cov, "samples": boot_means}


def merge_covariance_into_results(
    results: Mapping[str, Any],
    covariance: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> dict[str, Any]:
    """Return a serializable nested dictionary with values and covariance."""
    merged: dict[str, Any] = {}
    for name, result in results.items():
        merged[name] = {
            "rbins": np.asarray(result.rbins),
            "rmid": np.asarray(result.rmid),
            "categories": {},
            "metadata": dict(result.metadata),
        }
        for category, values in result.values.items():
            entry = {
                "value": np.asarray(values),
                "count": np.asarray(result.counts[category]),
                "weight_sum": np.asarray(result.weight_sums[category]),
            }
            if name in covariance and category in covariance[name]:
                entry["cov"] = np.asarray(covariance[name][category]["cov"])
                entry["jackknife_mean"] = np.asarray(covariance[name][category]["mean"])
            merged[name]["categories"][category] = entry
    return merged


__all__ = [
    "jackknife_tags",
    "covariance_from_samples",
    "jackknife_covariance",
    "bootstrap_covariance",
    "merge_covariance_into_results",
]
