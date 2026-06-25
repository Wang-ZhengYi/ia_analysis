"""Tests for satellite phase-space and layer statistics."""

import numpy as np


def test_satellite_radial_profile_and_binding_layers(synthetic_hod_tables):
    from ia_analysis.hod.phase_space import (
        measure_binding_energy_layer_occupation,
        measure_radial_profile_hod,
    )

    halos, galaxies = synthetic_hod_tables
    radial = measure_radial_profile_hod(halos, galaxies, radius_edges=[0.0, 0.5, 1.0])
    layers = measure_binding_energy_layer_occupation(halos, galaxies)
    assert radial["n_satellite"].sum() == galaxies["is_satellite"].sum()
    assert set(layers["binding_energy_layer"]) == {"inner", "outer"}


def test_velocity_anisotropy_is_measured(synthetic_hod_tables):
    from ia_analysis.hod.phase_space import measure_velocity_anisotropy_hod

    halos, galaxies = synthetic_hod_tables
    result = measure_velocity_anisotropy_hod(halos, galaxies, mass_edges=[5e11, 5e12, 5e13])
    assert len(result) == 2
    assert np.isfinite(result["beta"]).all()
