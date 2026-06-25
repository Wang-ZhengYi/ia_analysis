"""Synthetic tests for ordinary HOD statistics."""

import numpy as np


def test_ordinary_hod_and_central_satellite_split(synthetic_hod_tables):
    from ia_analysis.hod.statistics import measure_hod

    halos, galaxies = synthetic_hod_tables
    result = measure_hod(halos, galaxies, mass_bins=[5e11, 5e12, 5e13])
    assert np.allclose(result.mean_cen, 1.0)
    assert np.allclose(result.mean_sat, [1.0, 2.0])
    assert np.allclose(result.mean_tot, result.mean_cen + result.mean_sat)


def test_lrg_elg_sample_split_and_basic_summaries(synthetic_hod_tables):
    from ia_analysis.hod.statistics import (
        measure_hod,
        measure_number_density,
        measure_occupation_distribution,
        measure_satellite_fraction,
    )

    halos, galaxies = synthetic_hod_tables
    lrg = measure_hod(halos, galaxies, mass_bins=[5e11, 5e12, 5e13], sample_label="LRG")
    elg = measure_hod(halos, galaxies, mass_bins=[5e11, 5e12, 5e13], sample_label="ELG")
    assert lrg.n_tot.sum() == elg.n_tot.sum()
    assert 0.0 < measure_satellite_fraction(halos, galaxies) < 1.0
    assert measure_number_density(halos, galaxies, volume=100.0) == len(galaxies) / 100.0
    assert not measure_occupation_distribution(halos, galaxies, mass_bins=2).empty


def test_catalog_adapter_accepts_mapping_and_structured_array():
    from ia_analysis.hod.catalog import standardize_hod_catalog

    halos = np.array(
        [(1, 1.0e12), (2, 2.0e12)],
        dtype=[("halo_id", "i8"), ("mass", "f8")],
    )
    galaxies = {
        "galaxy_id": np.array([1, 20, 2]),
        "halo_id": np.array([1, 1, 2]),
        "sample_label": np.array(["LRG", "LRG", "ELG"]),
    }
    catalog = standardize_hod_catalog(halos, galaxies)
    assert catalog.galaxies["is_central"].tolist() == [True, False, True]
