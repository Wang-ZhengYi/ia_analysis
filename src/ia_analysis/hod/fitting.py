"""Least-squares fitting for IA components and staged HOD/assembly/IA models."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from ia_analysis.hod.ia_models import IAComponentModel, IAStrengthParameters, predict_ia_component
from ia_analysis.hod.models import fit_zheng_hod


def ia_component_chi2(
    prediction: Any,
    measurement: Any,
    *,
    errors: Any | None = None,
    covariance: Any | None = None,
) -> float:
    """Return IA component chi-square using diagonal or full covariance."""
    residual = np.asarray(measurement, dtype=float).ravel() - np.asarray(prediction, dtype=float).ravel()
    valid = np.isfinite(residual)
    residual = residual[valid]
    if covariance is not None:
        cov = np.asarray(covariance, dtype=float)[np.ix_(valid, valid)]
        return float(residual @ np.linalg.pinv(cov, hermitian=True) @ residual)
    sigma = np.ones_like(residual) if errors is None else np.asarray(errors, dtype=float).ravel()[valid]
    return float(np.sum((residual / np.maximum(sigma, 1.0e-30)) ** 2))


def fit_ia_component_model(
    mass: Any,
    values: Any,
    *,
    radius: Any = 1.0,
    secondary: Any = 0.0,
    errors: Any | None = None,
    initial: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
    name: str = "component",
    reference: str = "custom",
    population: str = "all",
    pivot_mass: float = 1.0e13,
    pivot_radius: float = 0.3,
) -> dict[str, Any]:
    """Fit mu0, mass, radius, and secondary slopes for one component."""
    mass, radius, secondary, values = np.broadcast_arrays(
        np.asarray(mass, dtype=float), np.asarray(radius, dtype=float),
        np.asarray(secondary, dtype=float), np.asarray(values, dtype=float)
    )
    sigma = np.ones_like(values) if errors is None else np.broadcast_to(np.asarray(errors, dtype=float), values.shape)
    valid = np.isfinite(mass) & np.isfinite(radius) & np.isfinite(secondary) & np.isfinite(values) & (mass > 0.0) & (radius > 0.0)

    def residual(theta: np.ndarray) -> np.ndarray:
        parameters = IAStrengthParameters(theta[0], theta[1], theta[2], theta[3], pivot_mass, pivot_radius)
        model = IAComponentModel(name, reference, population, parameters)
        prediction = predict_ia_component(model, mass=mass[valid], radius=radius[valid], secondary=secondary[valid])
        return (prediction - values[valid]) / np.maximum(sigma[valid], 1.0e-6)

    fit = least_squares(residual, np.asarray(initial, dtype=float))
    parameters = IAStrengthParameters(fit.x[0], fit.x[1], fit.x[2], fit.x[3], pivot_mass, pivot_radius)
    return {"model": IAComponentModel(name, reference, population, parameters), "success": bool(fit.success), "cost": float(fit.cost), "result": fit}


def fit_joint_hod_assembly_ia(
    *,
    hod_data: Mapping[str, Any],
    ia_datasets: Mapping[str, Mapping[str, Any]],
    assembly_fit: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Run the recommended staged first-PR fitting sequence."""
    output = {
        "hod": fit_zheng_hod(
            hod_data["mass"], hod_data["mean_cen"], hod_data["mean_sat"],
            errors_cen=hod_data.get("errors_cen"), errors_sat=hod_data.get("errors_sat"),
        ),
        "assembly": None if assembly_fit is None else assembly_fit(),
        "ia": {},
    }
    for name, data in ia_datasets.items():
        output["ia"][name] = fit_ia_component_model(name=name, **data)
    return output


__all__ = ["ia_component_chi2", "fit_ia_component_model", "fit_joint_hod_assembly_ia"]
