from ia_analysis.notebook_pipelines import legacy_catalog


def test_legacy_catalog_lists_and_reads_function_source():
    items = legacy_catalog.definitions(["orbit_nb.py"])
    names = {item.name for item in items}

    assert "make_nfw_profiles" in names
    assert "multiangle" in names

    text = legacy_catalog.source("orbit_nb.py", "make_nfw_profiles")
    assert text.startswith("def make_nfw_profiles")


def test_legacy_catalog_separates_plotting_functions():
    grouped = legacy_catalog.manifest(["plot_tcfs_3x5_nb.py"])

    assert any(item.name == "plot_3x5" for item in grouped["plotting"])
    assert any(item.name == "load_mean_covdiag" for item in grouped["pipeline"])
