"""Package-structure and command-entrypoint smoke tests.

Purpose
-------
These tests ensure the structured package imports cleanly, public namespaces are
available, and lightweight CLI help paths keep working without full HPC data.

Provides
--------
- Smoke imports for every major subpackage and facade module.
- Package-style command help checks.
- Basic velocity-momentum helper coverage.
"""

import importlib
from pathlib import Path
import subprocess
import sys


def test_subpackage_smoke_imports():
    for name in [
        "ia_analysis",
        "ia_analysis.api",
        "ia_analysis.catalogs",
        "ia_analysis.catalogs.api",
        "ia_analysis.shapes",
        "ia_analysis.shapes.api",
        "ia_analysis.tides",
        "ia_analysis.tides.api",
        "ia_analysis.dynamics",
        "ia_analysis.dynamics.api",
        "ia_analysis.MergerTree",
        "ia_analysis.MergerTree.api",
        "ia_analysis.MergerTree.reader",
        "ia_analysis.MergerTree.workflow",
        "ia_analysis.MergerTree.storage",
        "ia_analysis.meshes",
        "ia_analysis.meshes.CatMesh",
        "ia_analysis.meshes.SnapMesh",
        "ia_analysis.spectra",
        "ia_analysis.spectra.CatMesh",
        "ia_analysis.spectra.SnapMesh",
        "ia_analysis.spectra.catalog_mesh",
        "ia_analysis.spectra.snapshot_mesh",
        "ia_analysis.covariance",
        "ia_analysis.covariance.api",
        "ia_analysis.pipelines",
        "ia_analysis.pipelines.api",
        "ia_analysis.orbits",
        "ia_analysis.orbits.api",
        "ia_analysis.visualization",
        "ia_analysis.visualization.api",
        "ia_analysis.visualization.color_tools",
        "ia_analysis.visualization.projection_geometry",
        "ia_analysis.visualization.shell_plots",
        "ia_analysis.visualization.scene3d",
        "ia_analysis.visualization.alignment_catalogs",
        "ia_analysis.visualization.alignment_metrics",
        "ia_analysis.visualization.alignment_plots",
        "ia_analysis.visualization.orbit_animation",
        "ia_analysis.visualization.distribution_fits",
        "ia_analysis.visualization.parallel_alignment",
        "ia_analysis.spectra.velocity_momentum",
    ]:
        importlib.import_module(name)


def test_shape_package_imports():
    shape = importlib.import_module("ia_analysis.shapes.shape")
    assert hasattr(shape, "ShapeKin")
    assert hasattr(shape, "compute_axis")


def test_structured_api_registries_are_lightweight_and_discoverable():
    api = importlib.import_module("ia_analysis.api")
    assert "catalogs" in api.available_domains()
    assert "merger_tree" in api.available_domains()
    assert api.load_domain_api("pipelines").pipeline_module("cs-global") == "ia_analysis.pipelines.run_cs"

    catalogs = importlib.import_module("ia_analysis.catalogs.api")
    ordered = catalogs.sort_hdf5_chunks(["snap_099.10.hdf5", "snap_099.2.hdf5", "snap_099.0.hdf5"])
    assert ordered == ["snap_099.0.hdf5", "snap_099.2.hdf5", "snap_099.10.hdf5"]

    pipelines = importlib.import_module("ia_analysis.pipelines")
    assert pipelines.pipeline_command("tng-layered") == ("python", "-m", "ia_analysis.pipelines.tng_layered_shape_tide")

    visualization = importlib.import_module("ia_analysis.visualization.api")
    assert "alignment_plots" in visualization.available_groups()
    assert "plot_alignment_suite" in visualization.group_exports("alignment_plots")


def test_merger_tree_track_selection_from_synthetic_table():
    import pandas as pd

    reader = importlib.import_module("ia_analysis.MergerTree.reader")
    tree = {
        "SnapNum": [99, 84, 67, 50],
        "SubfindID": [10, 20, 30, 40],
        "SubhaloMass": [1.0, 0.8, 0.5, 0.2],
        "MatrixField": [[[1, 0], [0, 1]]] * 4,
    }
    table = reader.tree_to_dataframe(tree)
    assert isinstance(table, pd.DataFrame)
    assert "MatrixField" not in table
    selected = reader.select_tree_rows_for_snapshots(table, [67, 99], sort="input")
    assert list(selected["SnapNum"].astype(int)) == [67, 99]
    assert list(selected["SubfindID"].astype(int)) == [30, 10]


def test_dynamics_has_single_tng_driver_file():
    dynamics_dir = Path(__file__).resolve().parents[1] / "src" / "ia_analysis" / "dynamics"
    assert sorted(path.name for path in dynamics_dir.glob("hd_tng*.py")) == ["hd_tng.py"]


def test_package_run_cs_help():
    proc = subprocess.run(
        [sys.executable, "-m", "ia_analysis.pipelines.run_cs", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "ClusterSims" in proc.stdout


def test_velocity_momentum_status_and_pairs():
    vm = importlib.import_module("ia_analysis.spectra.velocity_momentum")
    status = vm.velocity_divergence_self_folding_status()
    assert status["status"] == "not_exact_for_local_velocity_divergence"
    assert status["recommended_field"] == "theta_momentum_mesh"

    pairs = vm.append_momentum_divergence_cross_pairs(
        {"d": object(), "E": object(), "g": object(), "tm": object()},
        theta_key="tm",
    )
    assert pairs == [("d", "tm"), ("E", "tm"), ("g", "tm")]
