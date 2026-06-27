import warnings

import numpy as np

from ia_analysis.visualization.arts_IA import (
    axis_vector,
    baryon_dm_ratio,
    stretch_tidal_tensor,
    tidal_tensor,
)


def test_baryon_dm_ratio_ignores_nonpositive_dm_without_warning():
    masses = np.zeros((4, 6), dtype=float)
    masses[:, 0] = [1.0, 1.0, 1.0, 0.0]
    masses[:, 1] = [2.0, 0.0, -1.0, 2.0]
    masses[:, 4] = [3.0, 3.0, 3.0, 0.0]

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        ratio = baryon_dm_ratio({"SubhaloMassInRadType": masses}, log=False)

    assert np.isclose(ratio[0], 2.0)
    assert np.isnan(ratio[1])
    assert np.isnan(ratio[2])
    assert np.isnan(ratio[3])


def test_tidal_major_axis_uses_largest_stretching_direction():
    legacy_hessian = np.array([[[-3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]])
    MA = {"Tidal_tot": legacy_hessian}

    stretch_axis = axis_vector(stretch_tidal_tensor(MA, "GR"), "major")[0]

    assert np.isclose(abs(np.dot(stretch_axis, [1.0, 0.0, 0.0])), 1.0)


def test_grmg_tidal_tensor_combines_gr_and_mg_extra_components():
    gr = np.arange(9, dtype=float).reshape(1, 3, 3)
    mg_extra = np.full((1, 3, 3), 2.0)
    MA = {"Tidal_tot": gr, "Tidal_tot_mg": mg_extra}

    np.testing.assert_allclose(tidal_tensor(MA, "GRMG"), gr + mg_extra)


def test_tidal_tensor_reads_four_final_branches_with_grouped_paths():
    self_t = np.full((1, 3, 3), 1.0)
    gr = np.full((1, 3, 3), 2.0)
    group = np.full((1, 3, 3), 3.0)
    mg = np.full((1, 3, 3), 4.0)
    MA = {
        "Tidal/Tidal_self": self_t,
        "Tidal/Tidal_tot": gr,
        "Tidal/Tidal_grp": group,
        "Tidal/Tidal_tot_mg": mg,
    }

    np.testing.assert_allclose(tidal_tensor(MA, "self"), self_t)
    np.testing.assert_allclose(tidal_tensor(MA, "GR"), gr)
    np.testing.assert_allclose(tidal_tensor(MA, "group"), group)
    np.testing.assert_allclose(tidal_tensor(MA, "MG"), mg)
    np.testing.assert_allclose(tidal_tensor(MA, "GRMG"), gr + mg)
