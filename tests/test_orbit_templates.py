"""Tests for 2LPT-style orbit template helpers."""

import numpy as np
import pytest


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
    assert np.allclose(kernel["cov"], 0.0)


def test_empty_orbit_template_library_has_stable_feature_shapes():
    from ia_analysis.orbits.template_orbits import (
        DEFAULT_TEMPLATE_FEATURES,
        OrbitTemplateLibrary,
        hod_1h_orbit_kernel,
    )

    kernel = hod_1h_orbit_kernel(OrbitTemplateLibrary(()))
    count = len(DEFAULT_TEMPLATE_FEATURES)
    assert kernel["mean"].shape == (count,)
    assert kernel["cov"].shape == (count, count)
    assert kernel["score"].shape == (0,)
    assert np.isnan(kernel["mean"]).all()


def test_radial_action_proxy_handles_non_monotonic_radius_without_cancellation():
    from ia_analysis.orbits.template_orbits import OrbitTemplate, template_feature_vector

    template = OrbitTemplate(
        group_id=1,
        subhalo_id=2,
        snapshots=np.array([0, 1, 2]),
        relative_position=np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        relative_velocity=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-3.0, 0.0, 0.0]]),
    )
    proxy = template_feature_vector(template, keys=("radial_action_proxy",))[0]
    assert np.isclose(proxy, 5.5)


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


def test_read_pinocchio_table_supports_csv_and_ascii(tmp_path):
    from ia_analysis.orbits.pinocchio import read_pinocchio_table, tracks_from_table

    header = "id,snapshot,x,y,z,vx,vy,vz,mass\n"
    csv_path = tmp_path / "groups.csv"
    csv_path.write_text(header + "10,1,1,0,0,10,0,0,110\n10,0,0,0,0,0,0,0,100\n", encoding="utf-8")
    csv_table = read_pinocchio_table(csv_path)
    tracks = tracks_from_table(csv_table)
    assert tracks[10].snapshots.tolist() == [0, 1]

    ascii_path = tmp_path / "subhaloes.dat"
    ascii_path.write_text(
        "id host_id snapshot x y z vx vy vz mass\n"
        "20 10 0 1 0 0 1 5 0 10\n"
        "20 10 1 1 2 0 11 4 0 8\n",
        encoding="utf-8",
    )
    ascii_table = read_pinocchio_table(ascii_path, columns=("id", "host_id", "snapshot", "x", "y", "z", "vx", "vy", "vz", "mass"))
    assert list(ascii_table.columns) == ["id", "host_id", "snapshot", "x", "y", "z", "vx", "vy", "vz", "mass"]


def test_read_pinocchio_table_rejects_unknown_format(tmp_path):
    from ia_analysis.orbits.pinocchio import read_pinocchio_table

    path = tmp_path / "table.data"
    path.write_text("id snapshot\n1 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="format must be"):
        read_pinocchio_table(path, format="binary")


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


@pytest.mark.parametrize("axes", [(1.0, 1.0, 1.0), (2.0, 1.2, 0.8), (4.0, 1.0, 0.4)])
def test_ellipsoid_shape_coefficients_sum_to_two(axes):
    from ia_analysis.orbits.ellipsoidal_model import ellipsoid_shape_coefficients

    coefficients = ellipsoid_shape_coefficients(axes, n_quad=240)
    assert np.all(coefficients > 0.0)
    assert np.isclose(coefficients.sum(), 2.0, rtol=2.0e-4, atol=2.0e-4)


def test_ellipsoidal_group_rejects_non_orthonormal_orientation():
    from ia_analysis.orbits.ellipsoidal_model import EllipsoidalGroupModel

    bad_orientation = np.eye(3)
    bad_orientation[0, 1] = 0.1
    with pytest.raises(ValueError, match="orthonormal"):
        EllipsoidalGroupModel(axes=(2.0, 1.0, 0.5), orientation=bad_orientation)
    with pytest.raises(ValueError, match="determinant"):
        EllipsoidalGroupModel(axes=(2.0, 1.0, 0.5), orientation=np.diag([-1.0, 1.0, 1.0]))


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


def _stripping_inputs():
    time = np.linspace(0.0, 3.0, 16)
    radius = 1.0 + 0.4 * np.cos(2.0 * np.pi * time / time[-1])
    omega = 2.5 / np.maximum(radius, 0.2)
    curvature = -0.4 / (radius + 0.2) ** 3
    return time, radius, omega, curvature


@pytest.mark.parametrize("mode", ["instantaneous_powerlaw", "delayed_powerlaw"])
def test_stripping_modes_respect_irreversibility_and_floor(mode):
    from ia_analysis.orbits.tidal_stripping import TidalStrippingOptions, build_stripping_history

    time, radius, omega, curvature = _stripping_inputs()
    floor = 0.2
    history = build_stripping_history(
        time=time,
        radius=radius,
        omega=omega,
        host_curvature=curvature,
        mass0=1.0,
        reference_radius=0.8,
        options=TidalStrippingOptions(
            mode=mode,
            tau_orbits=0.2,
            minimum_bound_fraction=floor,
            irreversible=True,
        ),
        gravitational_constant=1.0,
    )
    assert np.all(np.diff(history.bound_mass) <= 1.0e-12)
    assert np.all(history.mass_fraction >= floor)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": "unknown"}, "mode must be"),
        ({"density_slope": 0.0}, "density_slope"),
        ({"curvature_floor": 0.0}, "curvature_floor"),
        ({"tau_orbits": 0.0}, "tau_orbits"),
        ({"minimum_bound_fraction": 1.1}, "minimum_bound_fraction"),
        ({"irreversible": "yes"}, "irreversible"),
        ({"vmax_slope": np.nan}, "coefficients"),
    ],
)
def test_invalid_tidal_stripping_options(kwargs, message):
    from ia_analysis.orbits.tidal_stripping import TidalStrippingOptions

    with pytest.raises(ValueError, match=message):
        TidalStrippingOptions(**kwargs).validate()


def test_stripping_history_requires_strictly_increasing_time():
    from ia_analysis.orbits.tidal_stripping import build_stripping_history

    with pytest.raises(ValueError, match="strictly increasing"):
        build_stripping_history(
            time=[0.0, 1.0, 1.0],
            radius=[1.0, 0.8, 0.9],
            omega=[1.0, 2.0, 1.5],
            host_curvature=[-0.1, -0.2, -0.1],
            mass0=1.0,
            reference_radius=0.5,
            gravitational_constant=1.0,
        )


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


def test_minimal_pinocchio_to_stripping_integration():
    from ia_analysis.orbits.pinocchio import build_pinocchio_template_library
    from ia_analysis.orbits.template_orbits import hod_1h_orbit_kernel
    from ia_analysis.orbits.tidal_stripping import stripping_history_from_template, stripping_summary

    group_table = {
        "id": np.array([10, 10, 10]),
        "snapshot": np.array([0, 1, 2]),
        "x": np.array([0.0, 0.5, 1.0]),
        "y": np.zeros(3),
        "z": np.zeros(3),
        "vx": np.array([0.0, 1.0, 1.0]),
        "vy": np.zeros(3),
        "vz": np.zeros(3),
        "mass": np.array([100.0, 105.0, 110.0]),
    }
    subhalo_table = {
        "id": np.array([20, 20, 20]),
        "host_id": np.array([10, 10, 10]),
        "snapshot": np.array([0, 1, 2]),
        "x": np.array([2.0, 1.5, 2.2]),
        "y": np.array([0.0, 1.0, 0.2]),
        "z": np.zeros(3),
        "vx": np.array([0.0, -1.0, 1.0]),
        "vy": np.array([2.0, 1.0, -1.0]),
        "vz": np.zeros(3),
        "mass": np.array([10.0, 9.0, 8.0]),
    }
    library = build_pinocchio_template_library(group_table, subhalo_table)
    kernel = hod_1h_orbit_kernel(library)
    history = stripping_history_from_template(library.templates[0], gravitational_constant=1.0)
    summary = stripping_summary(history)

    assert kernel["score"].shape == (1,)
    assert 0.0 < summary["final_mass_fraction"] <= 1.0
    assert summary["minimum_tidal_radius"] > 0.0


def test_orbit_api_exports_stable_workflow_helpers():
    from ia_analysis.orbits import api

    expected = {
        "TreeTrack",
        "OrbitTemplate",
        "OrbitTemplateLibrary",
        "build_orbit_template",
        "build_template_library",
        "template_feature_vector",
        "hod_1h_orbit_kernel",
        "PinocchioColumnMap",
        "read_pinocchio_table",
        "tracks_from_table",
        "build_pinocchio_template_library",
        "ellipsoid_shape_coefficients",
        "homogeneous_ellipsoid_tidal_tensor",
        "TidalStrippingOptions",
        "TidalStrippingHistory",
        "jacobi_tidal_radius",
        "instantaneous_power_law_target",
        "build_stripping_history",
        "stripping_history_from_template",
        "stripping_summary",
    }
    assert expected.issubset(set(api.__all__))
