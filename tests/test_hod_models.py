"""Tests for Zheng and decorated HOD models."""

import numpy as np


def test_zheng_prediction_and_fit():
    from ia_analysis.hod.models import fit_zheng_hod, predict_zheng_hod

    mass = np.logspace(11, 15, 30)
    parameters = dict(log_m_min=12.2, sigma_log_m=0.4, log_m0=12.0, log_m1=13.2, alpha=1.0)
    prediction = predict_zheng_hod(mass, **parameters)
    fit = fit_zheng_hod(mass, prediction["central"], prediction["satellite"])
    assert fit["success"]
    assert np.allclose(
        predict_zheng_hod(mass, **fit["parameters"])["total"],
        prediction["total"],
        atol=2e-3,
    )


def test_decorated_hod_prediction_is_bounded():
    from ia_analysis.hod.models import decorated_hod_prediction

    mass = np.logspace(11, 15, 20)
    secondary = np.linspace(-2.0, 2.0, mass.size)
    result = decorated_hod_prediction(
        mass,
        secondary,
        central_amplitude=0.8,
        satellite_amplitude=0.3,
        parameters=dict(log_m_min=12.0, sigma_log_m=0.4, log_m0=12.0, log_m1=13.0, alpha=1.0),
    )
    assert np.all((result["central"] >= 0.0) & (result["central"] <= 1.0))
    assert np.all(result["satellite"] >= 0.0)
