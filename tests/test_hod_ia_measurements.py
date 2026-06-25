"""Tests for component-level IA-HOD measurements."""

import numpy as np


def test_central_host_and_satellite_radial_alignment(synthetic_hod_tables):
    from ia_analysis.hod.ia_measurements import (
        measure_central_host_alignment,
        measure_satellite_radial_alignment,
    )

    halos, galaxies = synthetic_hod_tables
    central = measure_central_host_alignment(halos, galaxies)
    radial = measure_satellite_radial_alignment(halos, galaxies)
    assert np.isclose(central.values, 2.0 / 3.0)
    assert np.isclose(radial.values, 2.0 / 3.0)


def test_satellite_tidal_and_assembly_dependent_alignment(synthetic_hod_tables):
    from ia_analysis.hod.ia_measurements import (
        measure_assembly_dependent_ia_hod,
        measure_satellite_tidal_alignment,
    )

    halos, galaxies = synthetic_hod_tables
    tidal = measure_satellite_tidal_alignment(halos, galaxies)
    assembly = measure_assembly_dependent_ia_hod(
        halos,
        galaxies,
        component="satellite_tidal",
        reference="tidal_major_axis",
        population="satellite",
        secondary_column="tidal_anisotropy",
        secondary_edges=[-2.0, 0.0, 2.0],
    )
    assert tidal.counts > 0
    assert assembly.values.shape == (2,)


def test_mass_radius_alignment_grid(synthetic_hod_tables):
    from ia_analysis.hod.ia_measurements import measure_mass_radius_alignment_grid

    halos, galaxies = synthetic_hod_tables
    result = measure_mass_radius_alignment_grid(
        halos,
        galaxies,
        component="satellite_radial",
        reference="radial_vector",
        population="satellite",
        mass_edges=[5e11, 5e12, 5e13],
        radius_edges=[0.0, 0.5, 1.0],
    )
    assert result.values.shape == (2, 2)
    assert result.counts.sum() == galaxies["is_satellite"].sum()
