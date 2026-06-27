"""Tests for modular snapshot target analysis."""

import numpy as np

from ia_analysis.dynamics import (
    SnapshotTarget,
    analyze_snapshot_target,
    measure_target_kinematics,
    measure_target_shape,
    tidal_stretch_eigensystem,
)


def _target_particles():
    positions = np.array(
        [
            [-2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, 0.5],
        ],
        dtype=float,
    )
    velocities = np.array(
        [
            [0.0, -2.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.1],
            [0.0, 0.0, -0.1],
        ],
        dtype=float,
    )
    masses = np.ones(positions.shape[0])
    return positions, velocities, masses


def test_measure_target_shape_is_separate_from_kinematics():
    positions, velocities, masses = _target_particles()

    shape = measure_target_shape(positions, masses=masses, percentile=100.0, max_iter=4, tol=1e-8)
    kin = measure_target_kinematics(
        positions,
        velocities,
        masses=masses,
        min_particles=3,
        mask=shape["mask"],
    )

    assert shape["tensor"].shape == (3, 3)
    assert shape["axes"]["a"] >= shape["axes"]["b"] >= shape["axes"]["c"]
    assert kin["valid"]
    assert kin["I"].shape == (3, 3)
    assert kin["figure"]["source"] == "direct_dI"


def test_analyze_snapshot_target_returns_sectional_payload():
    positions, velocities, masses = _target_particles()
    target = SnapshotTarget(
        positions=positions,
        velocities=velocities,
        masses=masses,
        potentials=np.full(positions.shape[0], -10.0),
        component="dm",
        metadata={"snap": 42, "id": 7},
    )

    out = analyze_snapshot_target(
        target,
        shape_kwargs={"percentile": 100.0, "max_iter": 4, "tol": 1e-8},
        kinematics_kwargs={"min_particles": 3},
        dynamics_kwargs={"shell_kwargs": {"min_particles": 3, "shell_kwargs": {"n_shells": 2}}},
    )

    assert set(["shape", "kinematics", "matrix", "dynamics"]).issubset(out)
    assert out["component"] == "dm"
    assert out["metadata"]["snap"] == 42
    assert out["matrix"]["shape_tensor"].shape == (3, 3)
    assert out["dynamics"]["binding"]["potential_source"] == "input"
    assert len(out["dynamics"]["shells"]["shells"]) == 2


def test_tidal_stretch_eigensystem_uses_minus_hessian_convention():
    hessian = np.diag([-3.0, -1.0, 2.0])

    stretch = tidal_stretch_eigensystem(hessian)

    assert np.allclose(stretch["stretch_evals"], [3.0, 1.0, -2.0])
    assert np.allclose(np.abs(stretch["stretch_evecs"][:, 0]), [1.0, 0.0, 0.0])
