"""Tests for 2LPT-style orbit template helpers."""

import numpy as np


def _tracks():
    from ia_analysis.orbits.template_orbits import TreeTrack

    group = TreeTrack(
        object_id=10,
        snapshots=np.array([0, 1, 2]),
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        velocities=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        mass=np.array([100.0, 110.0, 120.0]),
    )
    subhalo = TreeTrack(
        object_id=20,
        snapshots=np.array([0, 1, 2]),
        positions=np.array([[1.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 3.0, 0.0]]),
        velocities=np.array([[1.0, 5.0, 0.0], [11.0, 4.0, 0.0], [8.0, 3.0, 0.0]]),
        mass=np.array([10.0, 8.0, 6.0]),
    )
    return group, subhalo


def test_orbit_template_features_and_kernel():
    from ia_analysis.orbits.template_orbits import OrbitTemplateLibrary, build_orbit_template, hod_1h_orbit_kernel

    group, subhalo = _tracks()
    template = build_orbit_template(group, subhalo)
    assert template.relative_position.shape == (3, 3)
    assert np.allclose(template.relative_velocity[1], [1.0, 4.0, 0.0])

    kernel = hod_1h_orbit_kernel(OrbitTemplateLibrary((template,)))
    assert "r_final" in kernel["feature_names"]
    assert kernel["mean"].shape[0] == len(kernel["feature_names"])


def test_pinocchio_like_table_adapter_builds_template_library():
    from ia_analysis.orbits.pinocchio import build_pinocchio_template_library

    group_table = {
        "id": np.array([10, 10]),
        "snapshot": np.array([0, 1]),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0, 0.0]),
        "z": np.array([0.0, 0.0]),
        "vx": np.array([0.0, 10.0]),
        "vy": np.array([0.0, 0.0]),
        "vz": np.array([0.0, 0.0]),
        "mass": np.array([100.0, 110.0]),
    }
    subhalo_table = {
        "id": np.array([20, 20]),
        "host_id": np.array([10, 10]),
        "snapshot": np.array([0, 1]),
        "x": np.array([1.0, 1.0]),
        "y": np.array([0.0, 2.0]),
        "z": np.array([0.0, 0.0]),
        "vx": np.array([1.0, 11.0]),
        "vy": np.array([5.0, 4.0]),
        "vz": np.array([0.0, 0.0]),
        "mass": np.array([10.0, 8.0]),
    }
    library = build_pinocchio_template_library(group_table, subhalo_table)
    assert len(library.templates) == 1
    assert library.templates[0].group_id == 10
    assert library.templates[0].subhalo_id == 20


def test_ellipsoidal_tide_shapes_and_perturbation_average():
    from ia_analysis.orbits.ellipsoidal_model import (
        EllipsoidalGroupModel,
        PhaseSpacePerturbationModel,
        homogeneous_ellipsoid_tidal_tensor,
        initial_shape_alignment_model,
        perturbation_average_features,
    )
    from ia_analysis.orbits.template_orbits import build_orbit_template

    group, subhalo = _tracks()
    template = build_orbit_template(group, subhalo)
    model = EllipsoidalGroupModel(axes=(2.0, 1.2, 0.8), orientation=np.eye(3), mass=100.0)
    tide = homogeneous_ellipsoid_tidal_tensor(model)
    assert tide.shape == (3, 3)
    assert np.allclose(tide, tide.T)
    assert np.isfinite(tide).all()

    shapes = initial_shape_alignment_model(model, mode="tidal_aligned")
    assert set(shapes) == {"inner", "outer"}
    assert shapes["inner"].shape == (3, 3)

    avg = perturbation_average_features(template, PhaseSpacePerturbationModel(seed=5), n_samples=8)
    assert avg["samples"].shape[0] == 8
    assert avg["cov"].shape[0] == avg["cov"].shape[1]


def test_delayed_tidal_stripping_history_is_monotonic():
    from ia_analysis.orbits.tidal_stripping import (
        TidalStrippingOptions,
        build_stripping_history,
        stripping_summary,
    )

    time = np.linspace(0.0, 4.0, 12)
    radius = 1.0 + 0.35 * np.cos(2.0 * np.pi * time / time[-1])
    omega = 2.0 / np.maximum(radius, 0.2)
    host_curvature = -0.35 / (radius + 0.3) ** 3
    instant = build_stripping_history(
        time=time,
        radius=radius,
        omega=omega,
        host_curvature=host_curvature,
        mass0=1.0,
        reference_radius=0.55,
        options=TidalStrippingOptions(mode="instantaneous_powerlaw", density_slope=2.0),
        gravitational_constant=1.0,
    )
    delayed = build_stripping_history(
        time=time,
        radius=radius,
        omega=omega,
        host_curvature=host_curvature,
        mass0=1.0,
        reference_radius=0.55,
        options=TidalStrippingOptions(mode="delayed_powerlaw", density_slope=2.0, tau_orbits=0.5),
        gravitational_constant=1.0,
    )

    assert np.all(np.diff(delayed.bound_mass) <= 1.0e-12)
    assert delayed.mass_fraction[-1] >= instant.mass_fraction[-1]
    assert delayed.vmax_ratio.shape == delayed.rmax_ratio.shape == time.shape
    assert stripping_summary(delayed)["maximum_mass_loss_rate"] >= 0.0


def test_stripping_history_from_orbit_template():
    from ia_analysis.orbits.tidal_stripping import TidalStrippingOptions, stripping_history_from_template
    from ia_analysis.orbits.template_orbits import build_orbit_template

    group, subhalo = _tracks()
    template = build_orbit_template(group, subhalo)
    history = stripping_history_from_template(
        template,
        options=TidalStrippingOptions(mode="delayed_powerlaw", tau_orbits=0.25),
        gravitational_constant=1.0,
    )
    assert history.time.shape == template.snapshots.shape
    assert np.all(history.mass_fraction <= 1.0)
    assert np.isfinite(history.tidal_radius).all()
