"""Tests for IA reference-vector normalization and axial invariance."""

import numpy as np


def test_reference_normalization_and_sign_invariance():
    from ia_analysis.hod.ia_reference import alignment_cos2_minus_one_third, normalize_vectors

    orientation = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    reference = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    normalized = normalize_vectors(orientation)
    value = alignment_cos2_minus_one_third(orientation, reference)
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)
    assert np.allclose(value, alignment_cos2_minus_one_third(-orientation, reference))
    assert np.allclose(value, alignment_cos2_minus_one_third(orientation, -reference))
    assert np.allclose(value, 2.0 / 3.0)


def test_reference_bank_resolves_available_axes(synthetic_hod_tables):
    from ia_analysis.hod.catalog import join_halo_galaxy_properties
    from ia_analysis.hod.ia_reference import build_reference_bank

    halos, galaxies = synthetic_hod_tables
    joined = join_halo_galaxy_properties(halos, galaxies)
    bank = build_reference_bank(joined)
    assert {"host_major_axis", "tidal_major_axis", "radial_vector", "spin"}.issubset(bank)
