"""Tests for the lazy HOD public API and registry."""


def test_hod_domain_api_is_discoverable_and_lightweight():
    from ia_analysis import api

    assert "hod" in api.available_domains()
    hod = api.load_domain_api("hod")
    assert "measure_hod" in hod.__all__
    assert "ComponentIAHODModel" in hod.__all__
    assert callable(hod.measure_hod)


def test_hod_and_ia_hdf5_roundtrip(tmp_path, synthetic_hod_tables):
    from ia_analysis.hod.ia_measurements import measure_central_host_alignment
    from ia_analysis.hod.io import (
        load_hod_measurement_hdf5,
        load_ia_component_hdf5,
        save_hod_measurement_hdf5,
        save_ia_component_hdf5,
    )
    from ia_analysis.hod.statistics import measure_hod

    halos, galaxies = synthetic_hod_tables
    hod = measure_hod(halos, galaxies, mass_bins=[5e11, 5e12, 5e13])
    ia = measure_central_host_alignment(halos, galaxies)
    hod_path = save_hod_measurement_hdf5(hod, tmp_path / "hod.hdf5")
    ia_path = save_ia_component_hdf5(ia, tmp_path / "ia.hdf5")
    assert load_hod_measurement_hdf5(hod_path).mean_tot.shape == hod.mean_tot.shape
    assert load_ia_component_hdf5(ia_path).component == "central_host"
