"""Smoke tests for the real-space correlations package."""

import numpy as np


def _synthetic_catalog():
    from ia_analysis.correlations import CorrelationCatalog

    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.6, 0.0],
            [3.0, 0.0, 0.0],
            [3.4, 0.0, 0.0],
            [0.0, 3.2, 0.0],
        ],
        dtype=float,
    )
    e = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.1],
            [0.9, 0.0, 0.0],
            [0.0, 1.0, 0.1],
        ],
        dtype=float,
    )
    v = np.array(
        [
            [10.0, 1.0, 0.0],
            [8.0, 0.0, 0.0],
            [0.0, 9.0, 0.0],
            [7.0, 1.0, 0.0],
            [6.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
        ],
        dtype=float,
    )
    omega = 0.1 * v[:, ::-1]
    return CorrelationCatalog(
        positions=positions,
        fields={"e": e, "v": v, "omega": omega, "d": np.ones(positions.shape[0])},
        host_id=np.array([0, 0, 0, 1, 1, 2]),
        sample_type=np.array(["c", "s", "s", "c", "s", "c"]),
        boxsize=20.0,
        name="synthetic",
    )


def test_default_correlations_measure_five_detailed_categories():
    from ia_analysis.correlations import measure_default_correlations
    from ia_analysis.correlations.estimators import DETAILED_CATEGORIES

    rbins = np.array([0.1, 1.0, 10.0])
    suite = measure_default_correlations(_synthetic_catalog(), rbins, covariance="jackknife", nsub=2)

    assert {"ee", "ed", "dd", "vv", "dv", "ev", "omegae", "omegad", "omegav"} <= set(suite.results)
    assert {"vedomega4", "vedomega4_connected"} <= set(suite.results)
    assert suite.metadata["detailed_categories"] == DETAILED_CATEGORIES

    dd = suite.results["dd"]
    for category in DETAILED_CATEGORIES:
        assert category in dd.values
        assert int(dd.counts[category].sum()) > 0

    cov = suite.covariance["ee"]["total"]["cov"]
    assert cov.shape == (rbins.size - 1, rbins.size - 1)


def test_correlation_hdf5_writer(tmp_path):
    import h5py
    from ia_analysis.correlations import measure_default_correlations, write_results_hdf5

    rbins = np.array([0.1, 1.0, 10.0])
    suite = measure_default_correlations(_synthetic_catalog(), rbins, covariance=None)
    out = write_results_hdf5(tmp_path / "corr.hdf5", suite)

    with h5py.File(out, "r") as h5:
        assert "statistics/ee/categories/total/value" in h5
        assert "statistics/vedomega4/categories/1h_cs/count" in h5
