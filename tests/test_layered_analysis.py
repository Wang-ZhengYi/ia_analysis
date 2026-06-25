"""Synthetic tests for notebook-derived layered analysis APIs."""

from __future__ import annotations

import h5py
import numpy as np


def test_catalog_inventory_and_spectrum_ratio(tmp_path):
    from ia_analysis.catalogs.analysis import inventory_catalogs
    from ia_analysis.spectra.analysis import read_spectrum, relative_to_reference

    for flag, scale in (("GR", 1.0), ("F50", 1.2)):
        path = tmp_path / f"pks_{flag}_s021.hdf5"
        with h5py.File(path, "w") as handle:
            group = handle.create_group("all")
            group["k"] = np.array([0.1, 0.2, 0.4])
            group["P_dd"] = scale * np.array([10.0, 6.0, 3.0])

    inventory = inventory_catalogs(tmp_path)
    assert len(inventory) == 2
    spectra = [
        read_spectrum(tmp_path / f"pks_{flag}_s021.hdf5", "all", "P_dd")
        for flag in ("GR", "F50")
    ]
    ratios = relative_to_reference(__import__("pandas").concat(spectra, ignore_index=True))
    modified = ratios.loc[ratios["flag"] == "F50", "ratio_to_reference"]
    assert np.allclose(modified, 1.2)


def test_shape_evolution_stays_positive_definite():
    from ia_analysis.shapes.evolution import evolve_shape_tensor
    from ia_analysis.tides.diagnostics import tidal_anisotropy, tidal_strength

    time = np.linspace(0.0, 1.0, 8)
    tides = np.repeat(np.diag([2.0, 0.5, -1.0])[None, :, :], time.size, axis=0)
    positions = np.column_stack((np.cos(time), np.sin(time), np.zeros_like(time)))
    velocities = np.column_stack((-np.sin(time), np.cos(time), np.zeros_like(time)))
    history = evolve_shape_tensor(
        time,
        tides,
        np.linspace(1.0, 0.5, time.size),
        positions,
        velocities,
        np.diag([1.0, 0.7**2, 0.5**2]),
    )
    assert history["S"].shape == (time.size, 3, 3)
    assert min(np.linalg.eigvalsh(tensor).min() for tensor in history["S"]) > 0.0
    assert np.all(tidal_strength(tides) == 2.0)
    assert np.all(tidal_anisotropy(tides) > 0.0)


def test_hod_counts_and_environment_bins():
    from ia_analysis.catalogs.hod import (
        add_environment_quantiles,
        binned_hod,
        occupation_counts,
    )

    counts = occupation_counts(
        np.array([1.0e12, 2.0e12, 8.0e12]),
        np.array([0, 0, 1, 2, 2, 2]),
        central=np.array([True, False, True, True, False, False]),
        environment=np.array([0.1, 0.5, 0.9]),
    )
    assert counts["Ngal"].tolist() == [2, 1, 3]
    assert counts["Nsat"].tolist() == [1, 0, 2]
    curve = binned_hod(counts, [5.0e11, 3.0e12, 1.0e13], minimum_haloes=1)
    assert set(curve["component"]) == {"Ngal", "Ncen", "Nsat"}
    split = add_environment_quantiles(counts, quantiles=2)
    assert split["environment_quantile"].notna().all()


def test_correlation_covariance_quality():
    from ia_analysis.correlations.quality import covariance_diagnostics, signal_to_noise

    covariance = np.diag([1.0, 4.0, 9.0])
    diagnostics = covariance_diagnostics(covariance)
    assert diagnostics["rank"] == 3.0
    assert diagnostics["condition_number_positive"] == 9.0
    assert np.isclose(signal_to_noise(np.array([1.0, 2.0, 3.0]), covariance), np.sqrt(3.0))
