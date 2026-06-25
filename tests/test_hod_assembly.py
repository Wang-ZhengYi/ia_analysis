"""Tests for fixed-mass HOD assembly splits."""

import numpy as np


def test_concentration_and_environment_quantiles_remove_mass_trend(synthetic_hod_tables):
    from ia_analysis.hod.concentration import split_by_concentration_quantiles
    from ia_analysis.hod.environment import split_by_environment_quantiles

    halos, _ = synthetic_hod_tables
    mass_bins = [5e11, 2e12, 5e12, 2e13, 5e13]
    concentration, standardized_c = split_by_concentration_quantiles(
        halos["mass"], halos["concentration"], mass_bins=mass_bins
    )
    environment, standardized_e = split_by_environment_quantiles(
        halos["mass"], halos["environment"], mass_bins=mass_bins
    )
    assert set(concentration) == {0, 1}
    assert set(environment) == {0, 1}
    assert np.isfinite(standardized_c).all()
    assert np.isfinite(standardized_e).all()


def test_measure_assembly_hod_returns_high_low_comparison(synthetic_hod_tables):
    from ia_analysis.hod.assembly import measure_assembly_hod

    halos, galaxies = synthetic_hod_tables
    result = measure_assembly_hod(
        halos,
        galaxies,
        secondary_property="concentration",
        mass_bins=[5e11, 2e12, 5e12, 2e13, 5e13],
    )
    assert set(result["measurements"]) == {"q0", "q1"}
    assert "ratio_high_low" in result
