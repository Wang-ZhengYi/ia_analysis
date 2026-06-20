"""Compressed four-point estimators for velocity-shape-density-rotation fields.

Purpose
-------
The full four-point function of ``v``, ``e``, ``d``, and ``omega`` has many
geometric configurations.  For the halo-catalog workflow in this project we
provide a compact pair-binned estimator that lives on the same radial bins and
halo/sample categories as the two-point functions.

Provides
--------
- Raw symmetric ``v-e-d-omega`` pair estimator.
- Connected approximation using measured two-point products.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ia_analysis.correlations.estimators import (
    DEFAULT_CATEGORIES,
    CorrelationResult,
    _accumulate_by_bin,
    pair_category_masks,
    query_pair_geometry,
    radial_bin_centers,
)
from ia_analysis.correlations.fields import CorrelationCatalog, as_scalar_field, radial_projection


def _endpoint_four_product(catalog: CorrelationCatalog, pairs: np.ndarray, rhat: np.ndarray) -> np.ndarray:
    """Return the symmetric pair product for ``v e d omega``."""
    v = catalog.field("v")
    e = catalog.field("e")
    d = as_scalar_field(catalog.field("d"))
    omega = catalog.field("omega")
    i = pairs[:, 0]
    j = pairs[:, 1]

    vi = radial_projection(v[i], "vector", rhat)
    ei = radial_projection(e[i], None, rhat)
    dj = d[j]
    omegaj = radial_projection(omega[j], "vector", rhat)

    vj = radial_projection(v[j], "vector", -rhat)
    ej = radial_projection(e[j], None, -rhat)
    di = d[i]
    omegai = radial_projection(omega[i], "vector", -rhat)
    return 0.5 * (vi * ei * dj * omegaj + vj * ej * di * omegai)


def estimate_vedomega_four_point(
    catalog: CorrelationCatalog,
    rbins: Sequence[float],
    *,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
    name: str = "vedomega4",
) -> CorrelationResult:
    """Estimate a compressed ``v-e-d-omega`` four-point function."""
    edges = np.asarray(rbins, dtype=float)
    centers = radial_bin_centers(edges)
    nbin = centers.size
    pairs, radius, _rhat, bin_index = query_pair_geometry(catalog, edges)
    category_masks = pair_category_masks(catalog, pairs)
    values = {str(category): np.full(nbin, np.nan, dtype=float) for category in categories}
    counts = {str(category): np.zeros(nbin, dtype=int) for category in categories}
    weight_sums = {str(category): np.zeros(nbin, dtype=float) for category in categories}
    if pairs.size:
        signal = _endpoint_four_product(catalog, pairs, _rhat)
        weights = catalog.weights[pairs[:, 0]] * catalog.weights[pairs[:, 1]]
        valid_radius = np.isfinite(radius) & (radius >= edges[0]) & (radius < edges[-1])
        for category in categories:
            key = str(category)
            mask = category_masks.get(key, np.zeros(pairs.shape[0], dtype=bool)) & valid_radius
            val, cnt, den = _accumulate_by_bin(signal, weights, bin_index, mask, nbin)
            values[key] = val
            counts[key] = cnt
            weight_sums[key] = den
    return CorrelationResult(
        name=name,
        rbins=edges,
        rmid=centers,
        values=values,
        counts=counts,
        weight_sums=weight_sums,
        metadata={"estimator": "symmetric_endpoint_product", "n_objects": catalog.size},
    )


def connected_vedomega_four_point(
    raw_four_point: CorrelationResult,
    two_point_results: Mapping[str, CorrelationResult],
    *,
    name: str = "vedomega4_connected",
) -> CorrelationResult:
    """Subtract a Gaussian-like disconnected approximation from ``v e d omega``.

    The approximation uses products already requested by the two-point suite:
    ``ev * omegad + dv * omegae + omegav * ed`` in each radial bin and category.
    Missing terms leave the corresponding category as NaN.
    """
    values: dict[str, np.ndarray] = {}
    for category, raw in raw_four_point.values.items():
        try:
            disconnected = (
                two_point_results["ev"].values[category] * two_point_results["omegad"].values[category]
                + two_point_results["dv"].values[category] * two_point_results["omegae"].values[category]
                + two_point_results["omegav"].values[category] * two_point_results["ed"].values[category]
            )
            values[category] = raw - disconnected
        except KeyError:
            values[category] = np.full_like(raw, np.nan, dtype=float)
    return CorrelationResult(
        name=name,
        rbins=np.asarray(raw_four_point.rbins),
        rmid=np.asarray(raw_four_point.rmid),
        values=values,
        counts={key: np.asarray(val) for key, val in raw_four_point.counts.items()},
        weight_sums={key: np.asarray(val) for key, val in raw_four_point.weight_sums.items()},
        metadata={"estimator": "raw_minus_pair_products", "raw_name": raw_four_point.name},
    )


__all__ = [
    "estimate_vedomega_four_point",
    "connected_vedomega_four_point",
]
