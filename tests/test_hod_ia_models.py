"""Tests for bounded conditional IA-strength models and fitting."""

import numpy as np


def test_conditional_ia_strength_is_bounded():
    from ia_analysis.hod.ia_models import IAComponentModel, IAStrengthParameters, predict_ia_component

    model = IAComponentModel(
        "satellite_radial",
        "radial_vector",
        "satellite",
        IAStrengthParameters(0.2, beta_mass=0.4, beta_radius=-0.3, beta_secondary=0.5),
    )
    value = predict_ia_component(
        model,
        mass=np.logspace(11, 15, 20),
        radius=np.linspace(0.1, 1.0, 20),
        secondary=np.linspace(-2.0, 2.0, 20),
    )
    assert np.all(np.abs(value) <= 1.0)


def test_simple_ia_component_fit_recovers_synthetic_trend():
    from ia_analysis.hod.fitting import fit_ia_component_model
    from ia_analysis.hod.ia_models import IAComponentModel, IAStrengthParameters, predict_ia_component

    mass = np.logspace(12, 14, 40)
    radius = np.linspace(0.1, 1.0, 40)
    secondary = np.linspace(-1.0, 1.0, 40)
    truth = IAComponentModel(
        "test", "radial_vector", "satellite",
        IAStrengthParameters(0.1, 0.3, -0.2, 0.25),
    )
    values = predict_ia_component(truth, mass=mass, radius=radius, secondary=secondary)
    fit = fit_ia_component_model(
        mass,
        values,
        radius=radius,
        secondary=secondary,
        name="test",
        reference="radial_vector",
        population="satellite",
    )
    predicted = predict_ia_component(fit["model"], mass=mass, radius=radius, secondary=secondary)
    assert fit["success"]
    assert np.max(np.abs(predicted - values)) < 2e-3


def test_pairwise_omega_eta_small_catalog():
    from ia_analysis.hod.statistics import measure_pairwise_ia

    positions = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]], dtype=float)
    orientations = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    result = measure_pairwise_ia(positions, orientations, [0.5, 2.0, 4.0])
    assert result["counts"].sum() == 3
    assert np.isfinite(result["omega"]).all()
    assert np.isfinite(result["eta"]).all()


def test_single_reference_orientation_sampler_has_axial_bias():
    from ia_analysis.hod.ia_forward import sample_orientations_from_reference
    from ia_analysis.hod.ia_reference import alignment_cos2_minus_one_third

    references = np.repeat([[1.0, 0.0, 0.0]], 400, axis=0)
    orientations = sample_orientations_from_reference(references, kappa=4.0, random_state=7)
    assert np.allclose(np.linalg.norm(orientations, axis=1), 1.0)
    assert np.mean(alignment_cos2_minus_one_third(orientations, references)) > 0.15
