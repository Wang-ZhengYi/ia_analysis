import importlib
import subprocess
import sys


def test_subpackage_smoke_imports():
    for name in [
        "ia_analysis",
        "ia_analysis.catalogs",
        "ia_analysis.shapes",
        "ia_analysis.tides",
        "ia_analysis.dynamics",
        "ia_analysis.meshes",
        "ia_analysis.meshes.CatMesh",
        "ia_analysis.meshes.SnapMesh",
        "ia_analysis.spectra",
        "ia_analysis.spectra.CatMesh",
        "ia_analysis.spectra.SnapMesh",
        "ia_analysis.spectra.catalog_mesh",
        "ia_analysis.spectra.snapshot_mesh",
        "ia_analysis.covariance",
        "ia_analysis.pipelines",
        "ia_analysis.orbits",
        "ia_analysis.visualization",
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


def test_legacy_shape_wrapper_imports():
    shape = importlib.import_module("shape")
    assert hasattr(shape, "ShapeKin")
    assert hasattr(shape, "compute_axis")


def test_legacy_run_cs_help():
    proc = subprocess.run(
        [sys.executable, "run_cs.py", "--help"],
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
