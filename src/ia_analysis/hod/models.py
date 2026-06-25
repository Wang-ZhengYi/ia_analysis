"""Standard Zheng-style and decorated HOD prediction models."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf


def zheng_central_occupation(
    mass: Any,
    *,
    log_m_min: float,
    sigma_log_m: float,
) -> np.ndarray:
    """Return the softened central occupation threshold."""
    mass = np.asarray(mass, dtype=float)
    width = max(float(sigma_log_m), 1.0e-6)
    return 0.5 * (1.0 + erf((np.log10(mass) - float(log_m_min)) / width))


def zheng_satellite_occupation(
    mass: Any,
    *,
    log_m0: float,
    log_m1: float,
    alpha: float,
    central_occupation: Any | None = None,
) -> np.ndarray:
    """Return the Zheng power-law satellite occupation."""
    mass = np.asarray(mass, dtype=float)
    m0, m1 = 10.0 ** float(log_m0), 10.0 ** float(log_m1)
    satellite = np.power(np.maximum(mass - m0, 0.0) / max(m1, 1.0e-30), float(alpha))
    if central_occupation is not None:
        satellite *= np.asarray(central_occupation, dtype=float)
    return satellite


def zheng_total_occupation(mass: Any, **parameters: float) -> np.ndarray:
    """Return central plus satellite occupation."""
    central = zheng_central_occupation(
        mass,
        log_m_min=parameters["log_m_min"],
        sigma_log_m=parameters["sigma_log_m"],
    )
    satellite = zheng_satellite_occupation(
        mass,
        log_m0=parameters["log_m0"],
        log_m1=parameters["log_m1"],
        alpha=parameters["alpha"],
        central_occupation=central,
    )
    return central + satellite


def predict_zheng_hod(mass: Any, **parameters: float) -> dict[str, np.ndarray]:
    """Return central, satellite, and total Zheng predictions."""
    central = zheng_central_occupation(mass, log_m_min=parameters["log_m_min"], sigma_log_m=parameters["sigma_log_m"])
    satellite = zheng_satellite_occupation(
        mass,
        log_m0=parameters["log_m0"],
        log_m1=parameters["log_m1"],
        alpha=parameters["alpha"],
        central_occupation=central,
    )
    return {"central": central, "satellite": satellite, "total": central + satellite}


def hod_chi2(
    model: Any,
    observed: Any,
    *,
    errors: Any | None = None,
    covariance: Any | None = None,
) -> float:
    """Return a diagonal- or full-covariance HOD chi-square."""
    residual = np.asarray(observed, dtype=float) - np.asarray(model, dtype=float)
    valid = np.isfinite(residual)
    residual = residual[valid]
    if covariance is not None:
        cov = np.asarray(covariance, dtype=float)[np.ix_(valid, valid)]
        return float(residual @ np.linalg.pinv(cov, hermitian=True) @ residual)
    sigma = np.ones_like(residual) if errors is None else np.asarray(errors, dtype=float)[valid]
    return float(np.sum((residual / np.maximum(sigma, 1.0e-30)) ** 2))


def fit_zheng_hod(
    mass: Any,
    mean_cen: Any,
    mean_sat: Any,
    *,
    initial: Sequence[float] = (12.0, 0.4, 12.0, 13.0, 1.0),
    errors_cen: Any | None = None,
    errors_sat: Any | None = None,
) -> dict[str, Any]:
    """Fit five Zheng HOD parameters with SciPy least squares."""
    mass = np.asarray(mass, dtype=float)
    cen = np.asarray(mean_cen, dtype=float)
    sat = np.asarray(mean_sat, dtype=float)
    err_c = np.ones_like(cen) if errors_cen is None else np.asarray(errors_cen, dtype=float)
    err_s = np.ones_like(sat) if errors_sat is None else np.asarray(errors_sat, dtype=float)
    valid = np.isfinite(mass) & np.isfinite(cen) & np.isfinite(sat) & (mass > 0.0)

    def residual(theta: np.ndarray) -> np.ndarray:
        prediction = predict_zheng_hod(
            mass[valid],
            log_m_min=theta[0],
            sigma_log_m=theta[1],
            log_m0=theta[2],
            log_m1=theta[3],
            alpha=theta[4],
        )
        return np.concatenate(
            ((prediction["central"] - cen[valid]) / np.maximum(err_c[valid], 1.0e-6),
             (prediction["satellite"] - sat[valid]) / np.maximum(err_s[valid], 1.0e-6))
        )

    fit = least_squares(residual, np.asarray(initial, dtype=float), bounds=([8, 0.01, 8, 8, 0.1], [16, 3, 16, 16, 3]))
    names = ("log_m_min", "sigma_log_m", "log_m0", "log_m1", "alpha")
    return {"parameters": dict(zip(names, fit.x)), "success": bool(fit.success), "cost": float(fit.cost), "result": fit}


def decorated_central_occupation(base: Any, standardized_secondary: Any, *, amplitude: float = 0.0) -> np.ndarray:
    """Decorate central occupation while preserving the [0, 1] range."""
    base = np.asarray(base, dtype=float)
    secondary = np.asarray(standardized_secondary, dtype=float)
    room = np.minimum(base, 1.0 - base)
    return np.clip(base + float(amplitude) * np.tanh(secondary) * room, 0.0, 1.0)


def decorated_satellite_occupation(base: Any, standardized_secondary: Any, *, amplitude: float = 0.0) -> np.ndarray:
    """Apply a positive assembly modulation to satellite occupation."""
    base = np.asarray(base, dtype=float)
    secondary = np.asarray(standardized_secondary, dtype=float)
    return np.maximum(base * np.exp(float(amplitude) * np.tanh(secondary)), 0.0)


def decorated_hod_prediction(
    mass: Any,
    standardized_secondary: Any,
    *,
    central_amplitude: float = 0.0,
    satellite_amplitude: float = 0.0,
    parameters: Mapping[str, float],
) -> dict[str, np.ndarray]:
    """Return a decorated Zheng HOD prediction for object-level secondary values."""
    base = predict_zheng_hod(mass, **dict(parameters))
    central = decorated_central_occupation(base["central"], standardized_secondary, amplitude=central_amplitude)
    satellite = decorated_satellite_occupation(base["satellite"], standardized_secondary, amplitude=satellite_amplitude)
    return {"central": central, "satellite": satellite, "total": central + satellite}


__all__ = [
    "zheng_central_occupation", "zheng_satellite_occupation", "zheng_total_occupation",
    "predict_zheng_hod", "fit_zheng_hod", "hod_chi2", "decorated_central_occupation",
    "decorated_satellite_occupation", "decorated_hod_prediction",
]
