"""Forward-model hooks for deterministic moments and simple axial orientation mocks.

The long-term target is a multi-reference axial distribution
``p(e) proportional to exp(sum_k kappa_k (e.q_k)^2)``.  This first
implementation provides deterministic component moments and a tested
single-reference sampler.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ia_analysis.hod.ia_reference import normalize_vectors


def predict_orientation_moments(component_strengths: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return deterministic axial second-moment summaries for each component."""
    return {name: np.clip(np.asarray(value, dtype=float), -1.0 / 3.0, 2.0 / 3.0) for name, value in component_strengths.items()}


def sample_orientations_from_reference(
    references: Any,
    *,
    kappa: float = 0.0,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Sample axial orientations with density proportional to exp(kappa cos^2)."""
    reference = normalize_vectors(references)
    if reference.ndim == 1:
        reference = reference[None, :]
    if not np.isfinite(reference).all():
        raise ValueError("references must contain finite non-zero vectors")
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    output = np.empty_like(reference)
    envelope = np.exp(max(float(kappa), 0.0))
    for i, axis in enumerate(reference):
        while True:
            candidate = rng.normal(size=3)
            candidate /= np.linalg.norm(candidate)
            weight = np.exp(float(kappa) * np.dot(candidate, axis) ** 2)
            if rng.random() <= weight / envelope:
                output[i] = candidate
                break
    return output


def assign_mock_orientations(
    catalog: Any,
    references: Any,
    *,
    kappa: float = 0.0,
    column: str = "orientation",
    random_state: int | None = None,
) -> Any:
    """Return a copied DataFrame with sampled axial orientations."""
    import pandas as pd

    frame = catalog.copy() if isinstance(catalog, pd.DataFrame) else pd.DataFrame(catalog)
    frame[column] = list(sample_orientations_from_reference(references, kappa=kappa, random_state=random_state))
    return frame


__all__ = ["predict_orientation_moments", "sample_orientations_from_reference", "assign_mock_orientations"]
