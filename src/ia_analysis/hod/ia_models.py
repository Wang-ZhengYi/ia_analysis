"""Conditional bounded IA-strength models and multi-component IA-HOD predictions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class IAStrengthParameters:
    """Parameters for a bounded conditional alignment-strength field."""

    mu0: float
    beta_mass: float = 0.0
    beta_radius: float = 0.0
    beta_secondary: float = 0.0
    pivot_mass: float = 1.0e13
    pivot_radius: float = 0.3


@dataclass(frozen=True)
class IAComponentModel:
    """One named IA component tied to a physical reference and population."""

    name: str
    reference: str
    population: str
    parameters: IAStrengthParameters
    sample_terms: Mapping[str, float] = field(default_factory=dict)
    layer_terms: Mapping[str, float] = field(default_factory=dict)


def constant_alignment_strength(size: int | tuple[int, ...], mu0: float) -> np.ndarray:
    return np.full(size, np.tanh(float(mu0)))


def mass_powerlaw_alignment_strength(mass: Any, parameters: IAStrengthParameters) -> np.ndarray:
    mass = np.asarray(mass, dtype=float)
    argument = parameters.mu0 + parameters.beta_mass * np.log10(mass / parameters.pivot_mass)
    return np.tanh(argument)


def radius_powerlaw_alignment_strength(radius: Any, parameters: IAStrengthParameters) -> np.ndarray:
    radius = np.asarray(radius, dtype=float)
    argument = parameters.mu0 + parameters.beta_radius * np.log10(np.maximum(radius, 1.0e-8) / parameters.pivot_radius)
    return np.tanh(argument)


def mass_radius_alignment_strength(mass: Any, radius: Any, parameters: IAStrengthParameters) -> np.ndarray:
    mass = np.asarray(mass, dtype=float)
    radius = np.asarray(radius, dtype=float)
    argument = (
        parameters.mu0
        + parameters.beta_mass * np.log10(mass / parameters.pivot_mass)
        + parameters.beta_radius * np.log10(np.maximum(radius, 1.0e-8) / parameters.pivot_radius)
    )
    return np.tanh(argument)


def assembly_modulated_alignment_strength(
    base_argument: Any,
    standardized_secondary: Any,
    *,
    beta_secondary: float,
) -> np.ndarray:
    return np.tanh(np.asarray(base_argument, dtype=float) + float(beta_secondary) * np.asarray(standardized_secondary, dtype=float))


def sample_modulated_alignment_strength(base_argument: Any, sample: Any, sample_terms: Mapping[str, float]) -> np.ndarray:
    sample = np.asarray(sample).astype(str)
    terms = np.asarray([float(sample_terms.get(label, 0.0)) for label in sample])
    return np.tanh(np.asarray(base_argument, dtype=float) + terms)


def layer_modulated_alignment_strength(base_argument: Any, layer: Any, layer_terms: Mapping[str, float]) -> np.ndarray:
    layer = np.asarray(layer).astype(str)
    terms = np.asarray([float(layer_terms.get(label, 0.0)) for label in layer])
    return np.tanh(np.asarray(base_argument, dtype=float) + terms)


def predict_ia_component(
    model: IAComponentModel,
    *,
    mass: Any,
    radius: Any = 1.0,
    secondary: Any = 0.0,
    sample: Any | None = None,
    layer: Any | None = None,
) -> np.ndarray:
    """Predict bounded component strength for conditional catalog properties."""
    mass, radius, secondary = np.broadcast_arrays(
        np.asarray(mass, dtype=float), np.asarray(radius, dtype=float), np.asarray(secondary, dtype=float)
    )
    p = model.parameters
    argument = (
        p.mu0
        + p.beta_mass * np.log10(mass / p.pivot_mass)
        + p.beta_radius * np.log10(np.maximum(radius, 1.0e-8) / p.pivot_radius)
        + p.beta_secondary * secondary
    )
    if sample is not None:
        sample_array = np.broadcast_to(np.asarray(sample), mass.shape)
        argument += np.asarray([model.sample_terms.get(str(value), 0.0) for value in sample_array.ravel()]).reshape(mass.shape)
    if layer is not None:
        layer_array = np.broadcast_to(np.asarray(layer), mass.shape)
        argument += np.asarray([model.layer_terms.get(str(value), 0.0) for value in layer_array.ravel()]).reshape(mass.shape)
    return np.tanh(argument)


def predict_ia_component_grid(
    model: IAComponentModel,
    mass: Sequence[float],
    radius: Sequence[float],
    *,
    secondary: float = 0.0,
) -> np.ndarray:
    mass_grid, radius_grid = np.meshgrid(mass, radius, indexing="ij")
    return predict_ia_component(model, mass=mass_grid, radius=radius_grid, secondary=secondary)


def combine_ia_components(
    predictions: Mapping[str, Any],
    *,
    weights: Mapping[str, float] | None = None,
    mode: str = "sum",
) -> np.ndarray:
    """Combine component predictions as a weighted sum or normalized mean."""
    names = list(predictions)
    stack = np.stack([np.asarray(predictions[name], dtype=float) * float((weights or {}).get(name, 1.0)) for name in names])
    if mode == "sum":
        return np.sum(stack, axis=0)
    if mode == "mean":
        denominator = sum(abs(float((weights or {}).get(name, 1.0))) for name in names)
        return np.sum(stack, axis=0) / max(denominator, 1.0e-30)
    raise ValueError("mode must be 'sum' or 'mean'")


class ComponentIAHODModel:
    """Collection of physical IA components with conditional strength fields."""

    def __init__(self, components: Sequence[IAComponentModel]):
        self.components = {component.name: component for component in components}

    def predict(self, **conditions: Any) -> dict[str, np.ndarray]:
        return {name: predict_ia_component(component, **conditions) for name, component in self.components.items()}

    def combined(self, *, weights: Mapping[str, float] | None = None, **conditions: Any) -> np.ndarray:
        return combine_ia_components(self.predict(**conditions), weights=weights)


__all__ = [
    "IAStrengthParameters", "IAComponentModel", "ComponentIAHODModel",
    "constant_alignment_strength", "mass_powerlaw_alignment_strength",
    "radius_powerlaw_alignment_strength", "mass_radius_alignment_strength",
    "assembly_modulated_alignment_strength", "sample_modulated_alignment_strength",
    "layer_modulated_alignment_strength", "predict_ia_component",
    "predict_ia_component_grid", "combine_ia_components",
]
