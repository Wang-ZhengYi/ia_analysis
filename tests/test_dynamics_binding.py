"""Binding-energy profile tests for the dynamics package."""

import numpy as np

from ia_analysis.dynamics import halo_dynamics as hd


def test_component_binding_energy_uses_reference_velocity():
    """Kinetic energy must use velocity relative to the requested subhalo frame."""
    out = hd.component_binding_energy(
        positions=np.array([[1.0, 0.0, 0.0]]),
        velocities=np.array([[11.0, 0.0, 0.0]]),
        masses=np.array([1.0]),
        v_ref=np.array([10.0, 0.0, 0.0]),
        potentials=np.array([-10.0]),
    )

    assert np.allclose(out["kinetic"], [0.5])
    assert np.allclose(out["specific_total_energy"], [-9.5])
    assert np.allclose(out["specific_binding_energy"], [9.5])
    assert bool(out["bound_mask"][0])


def test_gas_enthalpy_term_enters_binding_energy():
    """Gas binding energy should include pressure support through enthalpy."""
    out = hd.component_binding_energy(
        positions=np.array([[1.0, 0.0, 0.0]]),
        velocities=np.zeros((1, 3)),
        masses=np.array([1.0]),
        potentials=np.array([-10.0]),
        component="gas",
        internal_energy=np.array([2.0]),
        gas_energy_mode="enthalpy",
        gas_gamma=5.0 / 3.0,
    )

    assert out["gas_term_source"] == "enthalpy_from_internal_energy"
    assert np.allclose(out["gas_term"], [10.0 / 3.0])
    assert np.allclose(out["specific_binding_energy"], [20.0 / 3.0])


def test_component_binding_profiles_return_component_mass_histograms():
    """The multi-component helper should return summaries and common histograms."""
    components = {
        "dm": {
            "X_kpc": np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            "U_kms": np.zeros((2, 3)),
            "masses": np.array([5.0, 5.0]),
        },
        "stars": {
            "X_kpc": np.array([[1.5, 0.0, 0.0]]),
            "U_kms": np.zeros((1, 3)),
            "masses": np.array([2.0]),
        },
        "gas": {
            "X_kpc": np.array([[0.8, 0.0, 0.0]]),
            "U_kms": np.zeros((1, 3)),
            "masses": np.array([1.0]),
            "internal_energy": np.array([1.0]),
        },
    }

    profile = hd.component_binding_energy_profiles(
        components,
        bins=4,
        gas_energy_mode="enthalpy",
        G=1.0,
        softening=0.01,
    )

    summary = {row["component"]: row for row in profile["summary"]}
    assert set(summary) == {"dm", "stars", "gas"}
    assert summary["gas"]["gas_term_source"] == "enthalpy_from_internal_energy"
    assert summary["dm"]["mass_total"] == 10.0
    assert summary["stars"]["mass_total"] == 2.0

    edges = profile["energy_edges"]
    assert edges.size == 5
    for name, hist in profile["binding_distribution"].items():
        assert hist["mass"].size == 4
        assert hist["count"].size == 4
        assert np.sum(hist["mass"]) <= summary[name]["mass_total"] + 1e-12
