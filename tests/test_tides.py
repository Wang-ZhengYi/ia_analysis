import numpy as np


def test_tidal_tensor_components_are_finite_and_symmetric_schema():
    import importlib.util

    if importlib.util.find_spec("scipy") is None or importlib.util.find_spec("numba") is None:
        return

    from ia_analysis.tides.tidal_field import compute_gravitational_potential

    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    masses = np.ones(positions.shape[0], dtype=float)

    out = compute_gravitational_potential(
        positions,
        masses,
        grid_size=4,
        boundary_padding=0.1,
        softening=0.1,
    )

    for key in ["potential", "Txx", "Txy", "Txz", "Tyy", "Tyz", "Tzz"]:
        assert key in out
        assert np.all(np.isfinite(out[key]))
