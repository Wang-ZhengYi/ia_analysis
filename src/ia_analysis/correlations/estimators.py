"""Pair-binned real-space correlation estimators.

Purpose
-------
This module provides the lightweight numerical core used by the correlations
subpackage.  It measures weighted pair averages in radial bins and separates
pairs into total, one-halo, two-halo, and the five detailed central/satellite
halo categories requested for IA, velocity, and figure-rotation analyses.

Provides
--------
- Pair querying with optional periodic wrapping.
- Total, 1h, 2h, 1h-cs, 1h-ss, 2h-cc, 2h-cs, and 2h-ss categories.
- Symmetric scalar/vector/tensor field-pair estimators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ia_analysis.correlations.fields import CorrelationCatalog, PairSpec, normalize_sample_class, pair_signal

SUMMARY_CATEGORIES = ("total", "1h", "2h")
DETAILED_CATEGORIES = ("1h_cs", "1h_ss", "2h_cc", "2h_cs", "2h_ss")
DEFAULT_CATEGORIES = SUMMARY_CATEGORIES + DETAILED_CATEGORIES


@dataclass
class CorrelationResult:
    """Measured radial correlation for one field pair."""

    name: str
    rbins: np.ndarray
    rmid: np.ndarray
    values: dict[str, np.ndarray]
    counts: dict[str, np.ndarray]
    weight_sums: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)

    def category(self, name: str) -> np.ndarray:
        """Return one measured category by name."""
        return self.values[str(name)]


def radial_bin_centers(rbins: Sequence[float]) -> np.ndarray:
    """Return geometric centers for positive bins and arithmetic centers otherwise."""
    edges = np.asarray(rbins, dtype=float)
    if np.any(edges[1:] <= edges[:-1]):
        raise ValueError("`rbins` must be strictly increasing")
    if np.all(edges > 0.0):
        return np.sqrt(edges[:-1] * edges[1:])
    return 0.5 * (edges[:-1] + edges[1:])


def _periodic_delta(delta: np.ndarray, boxsize: np.ndarray | None) -> np.ndarray:
    """Apply the minimum-image convention to displacement vectors."""
    if boxsize is None:
        return delta
    return delta - boxsize * np.rint(delta / boxsize)


def query_pair_geometry(catalog: CorrelationCatalog, rbins: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pair indices, distances, radial unit vectors, and bin indices."""
    edges = np.asarray(rbins, dtype=float)
    rmax = float(edges[-1])
    positions = np.asarray(catalog.positions, dtype=float)
    box = None if catalog.boxsize is None else np.asarray(catalog.boxsize, dtype=float)
    tree_positions = np.mod(positions, box) if box is not None else positions
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(tree_positions, boxsize=box)
        pairs = tree.query_pairs(rmax, output_type="ndarray")
    except ImportError:
        ii, jj = np.triu_indices(tree_positions.shape[0], k=1)
        all_pairs = np.column_stack((ii, jj))
        if all_pairs.size == 0:
            pairs = np.empty((0, 2), dtype=int)
        else:
            all_delta = tree_positions[all_pairs[:, 1]] - tree_positions[all_pairs[:, 0]]
            all_delta = _periodic_delta(all_delta, box)
            all_radius = np.linalg.norm(all_delta, axis=1)
            pairs = all_pairs[all_radius < rmax]
    if pairs.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.empty(0, dtype=float),
            np.empty((0, 3), dtype=float),
            np.empty(0, dtype=int),
        )

    delta = tree_positions[pairs[:, 1]] - tree_positions[pairs[:, 0]]
    delta = _periodic_delta(delta, box)
    radius = np.linalg.norm(delta, axis=1)
    good_radius = np.isfinite(radius) & (radius > 0.0)
    rhat = np.zeros_like(delta)
    rhat[good_radius] = delta[good_radius] / radius[good_radius, None]
    bin_index = np.searchsorted(edges, radius, side="right") - 1
    return pairs, radius, rhat, bin_index


def pair_category_masks(catalog: CorrelationCatalog, pairs: np.ndarray) -> dict[str, np.ndarray]:
    """Return boolean masks for summary and detailed pair categories."""
    n_pair = int(pairs.shape[0])
    masks = {name: np.zeros(n_pair, dtype=bool) for name in DEFAULT_CATEGORIES}
    masks["total"][:] = True
    if n_pair == 0:
        return masks

    same_host = None
    if catalog.host_id is not None:
        host = np.asarray(catalog.host_id)
        same_host = host[pairs[:, 0]] == host[pairs[:, 1]]
        masks["1h"] = same_host
        masks["2h"] = ~same_host

    sample = normalize_sample_class(catalog.sample_type)
    if same_host is None or sample is None:
        return masks

    left = sample[pairs[:, 0]]
    right = sample[pairs[:, 1]]
    cc = (left == "c") & (right == "c")
    ss = (left == "s") & (right == "s")
    cs = ((left == "c") & (right == "s")) | ((left == "s") & (right == "c"))

    masks["1h_cs"] = same_host & cs
    masks["1h_ss"] = same_host & ss
    masks["2h_cc"] = (~same_host) & cc
    masks["2h_cs"] = (~same_host) & cs
    masks["2h_ss"] = (~same_host) & ss
    return masks


def _accumulate_by_bin(
    signal: np.ndarray,
    weights: np.ndarray,
    bin_index: np.ndarray,
    mask: np.ndarray,
    nbin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate weighted pair means for one category mask."""
    valid = mask & np.isfinite(signal) & np.isfinite(weights) & (weights > 0.0) & (bin_index >= 0) & (bin_index < nbin)
    numerator = np.zeros(nbin, dtype=float)
    denominator = np.zeros(nbin, dtype=float)
    counts = np.zeros(nbin, dtype=int)
    if np.any(valid):
        idx = bin_index[valid]
        np.add.at(numerator, idx, signal[valid] * weights[valid])
        np.add.at(denominator, idx, weights[valid])
        np.add.at(counts, idx, 1)
    values = np.full(nbin, np.nan, dtype=float)
    good = denominator > 0.0
    values[good] = numerator[good] / denominator[good]
    return values, counts, denominator


def _pair_values(catalog: CorrelationCatalog, spec: PairSpec, pairs: np.ndarray, rhat: np.ndarray) -> np.ndarray:
    """Evaluate one pair signal with optional symmetric exchange averaging."""
    left_field = catalog.field(spec.left)
    right_field = catalog.field(spec.right)
    i = pairs[:, 0]
    j = pairs[:, 1]
    forward = pair_signal(
        left_field[i],
        right_field[j],
        rhat,
        left_kind=spec.left_kind,
        right_kind=spec.right_kind,
        vector_vector_mode=spec.vector_vector_mode,
    )
    if not spec.symmetrize or (spec.left == spec.right):
        return forward
    reverse = pair_signal(
        left_field[j],
        right_field[i],
        -rhat,
        left_kind=spec.left_kind,
        right_kind=spec.right_kind,
        vector_vector_mode=spec.vector_vector_mode,
    )
    return 0.5 * (forward + reverse)


def measure_two_point(
    catalog: CorrelationCatalog,
    spec: PairSpec | tuple[str, str] | str,
    rbins: Sequence[float],
    *,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
) -> CorrelationResult:
    """Measure one weighted two-point correlation in radial bins."""
    if isinstance(spec, PairSpec):
        pair_spec = spec
    elif isinstance(spec, tuple):
        pair_spec = PairSpec(str(spec[0]), str(spec[1]))
    else:
        text = str(spec)
        if len(text) == 2:
            pair_spec = PairSpec(text[0], text[1], text)
        elif text.startswith("omega"):
            pair_spec = PairSpec("omega", text[5:], text)
        else:
            raise ValueError("String pair specs must look like 'ee', 'ed', 'dv', or 'omegav'")

    edges = np.asarray(rbins, dtype=float)
    centers = radial_bin_centers(edges)
    nbin = centers.size
    pairs, radius, rhat, bin_index = query_pair_geometry(catalog, edges)
    category_masks = pair_category_masks(catalog, pairs)

    values = {category: np.full(nbin, np.nan, dtype=float) for category in categories}
    counts = {category: np.zeros(nbin, dtype=int) for category in categories}
    weight_sums = {category: np.zeros(nbin, dtype=float) for category in categories}
    if pairs.size:
        signal = _pair_values(catalog, pair_spec, pairs, rhat)
        weights = catalog.weights[pairs[:, 0]] * catalog.weights[pairs[:, 1]]
        valid_radius = np.isfinite(radius) & (radius >= edges[0]) & (radius < edges[-1])
        for category in categories:
            mask = category_masks.get(str(category), np.zeros(pairs.shape[0], dtype=bool)) & valid_radius
            val, cnt, den = _accumulate_by_bin(signal, weights, bin_index, mask, nbin)
            values[str(category)] = val
            counts[str(category)] = cnt
            weight_sums[str(category)] = den

    return CorrelationResult(
        name=pair_spec.output_name(),
        rbins=edges,
        rmid=centers,
        values=values,
        counts=counts,
        weight_sums=weight_sums,
        metadata={
            "left": pair_spec.left,
            "right": pair_spec.right,
            "n_objects": catalog.size,
            "n_pairs_total": int(pairs.shape[0]),
            "categories": tuple(str(c) for c in categories),
        },
    )


def measure_many_two_point(
    catalog: CorrelationCatalog,
    specs: Sequence[PairSpec],
    rbins: Sequence[float],
    *,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
) -> dict[str, CorrelationResult]:
    """Measure several field-pair correlations with the same categories."""
    return {spec.output_name(): measure_two_point(catalog, spec, rbins, categories=categories) for spec in specs}


__all__ = [
    "SUMMARY_CATEGORIES",
    "DETAILED_CATEGORIES",
    "DEFAULT_CATEGORIES",
    "CorrelationResult",
    "radial_bin_centers",
    "query_pair_geometry",
    "pair_category_masks",
    "measure_two_point",
    "measure_many_two_point",
]
