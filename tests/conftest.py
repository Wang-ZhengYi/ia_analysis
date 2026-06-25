"""Shared synthetic HOD and IA-HOD catalogs."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_hod_tables():
    """Return small halo and galaxy tables with all core HOD/IA fields."""
    halo_id = np.arange(8)
    mass = np.repeat([1.0e12, 3.0e12, 1.0e13, 3.0e13], 2)
    major = [np.array([1.0, 0.0, 0.0])] * 8
    tidal = [np.array([0.0, 1.0, 0.0])] * 8
    halos = pd.DataFrame({
        "halo_id": halo_id,
        "host_id": halo_id,
        "mass": mass,
        "rvir": np.linspace(0.2, 0.8, 8),
        "concentration": np.tile([5.0, 10.0], 4) + 0.1 * np.log10(mass),
        "environment": np.tile([-1.0, 1.0], 4) + 0.05 * np.log10(mass),
        "tidal_anisotropy": np.linspace(-1.0, 1.0, 8),
        "host_major_axis": major,
        "host_intermediate_axis": tidal,
        "host_minor_axis": [np.array([0.0, 0.0, 1.0])] * 8,
        "tidal_major_axis": tidal,
        "tidal_intermediate_axis": major,
        "tidal_minor_axis": [np.array([0.0, 0.0, 1.0])] * 8,
    })
    rows = []
    galaxy_id = 100
    for hid, halo_mass in zip(halo_id, mass):
        sample = "LRG" if hid % 2 == 0 else "ELG"
        n_sat = 1 + int(halo_mass >= 1.0e13)
        rows.append({
            "galaxy_id": galaxy_id, "halo_id": hid, "host_id": hid,
            "is_central": True, "is_satellite": False, "sample_label": sample,
            "position": np.zeros(3), "velocity": np.zeros(3),
            "shape_major_axis": np.array([1.0, 0.0, 0.0]),
            "orientation": np.array([1.0, 0.0, 0.0]),
            "radial_vector": np.array([1.0, 0.0, 0.0]),
            "r_over_rvir": 0.0, "binding_energy_layer": "inner",
            "binding_energy_layer_axis": np.array([1.0, 0.0, 0.0]),
            "subhalo_major_axis": np.array([1.0, 0.0, 0.0]),
            "spin": np.array([0.0, 0.0, 1.0]),
            "angular_momentum": np.array([0.0, 0.0, 1.0]),
            "figure_rotation_axis": np.array([0.0, 0.0, 1.0]),
        })
        galaxy_id += 1
        for sat in range(n_sat):
            radial = np.array([1.0, 0.0, 0.0]) if sat == 0 else np.array([0.0, 1.0, 0.0])
            rows.append({
                "galaxy_id": galaxy_id, "halo_id": hid, "host_id": hid,
                "is_central": False, "is_satellite": True, "sample_label": sample,
                "position": radial * (sat + 1), "velocity": np.array([1.0 + 0.2 * hid, sat + 1.0, 0.0]),
                "shape_major_axis": radial, "orientation": radial, "radial_vector": radial,
                "r_over_rvir": 0.25 + 0.35 * sat,
                "binding_energy": -2.0 + sat, "binding_energy_layer": "inner" if sat == 0 else "outer",
                "binding_energy_layer_axis": radial, "subhalo_major_axis": radial,
                "spin": np.array([0.0, 0.0, 1.0]),
                "angular_momentum": np.array([0.0, 0.0, 1.0]),
                "figure_rotation_axis": np.array([0.0, 0.0, 1.0]),
            })
            galaxy_id += 1
    return halos, pd.DataFrame(rows)
