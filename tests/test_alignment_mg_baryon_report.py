import h5py
import numpy as np
import pandas as pd

from ia_analysis.visualization.alignment_mg_baryon_report import (
    assign_categories,
    build_report,
    canonical_table_from_mock_summary,
    compute_all_metrics,
    compute_baryon_metrics,
    compute_mg_metrics,
    compute_mg_residuals,
    load_alignment_table,
    write_metrics_hdf5,
)


def _synthetic_table():
    rows = []
    for obs in [
        "StarShape_groupTidal_Mstar_major",
        "CGHA_BaryonDM_major",
        "TidalMajorRadial_GRMG_Mstar",
        "VelRadial_R",
    ]:
        xvar = "BaryonDM" if "BaryonDM" in obs else ("R" if obs == "VelRadial_R" else "Mstar")
        for model, offset in [("GR", 0.0), ("F4", 0.12), ("F5", -0.06)]:
            for snap, z in [(1, 1.0), (2, 0.0)]:
                for x in [0.0, 1.0, 2.0]:
                    baryon_term = 0.08 * x if xvar == "BaryonDM" else 0.01 * x
                    rows.append({
                        "observable_name": obs,
                        "x_variable": xvar,
                        "x_value": x,
                        "redshift": z,
                        "snapshot": snap,
                        "gravity_model": model,
                        "mu": 0.5 + baryon_term + offset + 0.01 * snap,
                        "mu_error": 0.05,
                        "count": 20,
                    })
    return pd.DataFrame(rows)


def test_canonical_table_adapter_from_mock_summary():
    summary = {
        "CGHA_Mstar_major": {
            "GR": {1: {"x": [1.0, 2.0], "mu": [0.5, 0.6], "mu_error": [0.1, 0.2], "count": [10, 11], "redshift": 0.0}},
        }
    }
    table = canonical_table_from_mock_summary(summary)
    assert list(table["observable_name"].unique()) == ["CGHA_Mstar_major"]
    assert set(["component", "population", "axis", "reference"]).issubset(table.columns)
    assert table["population"].iloc[0] == "central"
    assert table["axis"].iloc[0] == "major"


def test_gr_residual_calculation():
    table = pd.DataFrame({
        "observable_name": ["CGHA_Mstar_major", "CGHA_Mstar_major"],
        "x_variable": ["Mstar", "Mstar"],
        "x_value": [1.0, 1.0],
        "redshift": [0.0, 0.0],
        "snapshot": [1, 1],
        "gravity_model": ["GR", "F4"],
        "mu": [0.4, 0.55],
        "mu_error": [0.1, 0.2],
        "count": [10, 10],
    })
    residuals = compute_mg_residuals(load_alignment_table_from_frame(table))
    assert np.isclose(residuals["delta_mu"].iloc[0], 0.15)


def load_alignment_table_from_frame(frame):
    from ia_analysis.visualization.alignment_mg_baryon_report import canonicalize_alignment_table

    return canonicalize_alignment_table(frame)


def test_mg_metrics_known_array():
    residuals = pd.DataFrame({
        "observable_name": ["obs"] * 3,
        "component": ["c"] * 3,
        "population": ["all"] * 3,
        "axis": ["major"] * 3,
        "reference": ["halo"] * 3,
        "x_variable": ["Mstar"] * 3,
        "gravity_model": ["F4", "F4", "F5"],
        "delta_mu": [1.0, 2.0, -2.0],
        "delta_snr": [2.0, 4.0, -4.0],
        "redshift": [0.0, 1.0, 0.0],
        "x_value": [1.0, 1.0, 2.0],
        "snapshot": [1, 2, 1],
    })
    metrics = compute_mg_metrics(residuals)
    assert np.isclose(metrics["MG_RMS"].iloc[0], np.sqrt(3.0))
    assert np.isclose(metrics["MG_MAX"].iloc[0], 2.0)
    assert np.isclose(metrics["MG_SIGN_COHERENCE"].iloc[0], 2 / 3)


def test_baryon_metrics_slope_and_range():
    table = pd.DataFrame({
        "observable_name": ["CGHA_BaryonDM_major"] * 3,
        "component": ["CGHA"] * 3,
        "population": ["central"] * 3,
        "axis": ["major"] * 3,
        "reference": ["halo"] * 3,
        "x_variable": ["BaryonDM"] * 3,
        "x_value": [0.0, 1.0, 2.0],
        "redshift": [0.0, 0.0, 0.0],
        "snapshot": [1, 1, 1],
        "gravity_model": ["GR", "GR", "GR"],
        "mu": [0.2, 0.4, 0.6],
        "mu_error": [np.nan, np.nan, np.nan],
        "count": [10, 10, 10],
    })
    metrics = compute_baryon_metrics(table)
    assert np.isclose(metrics["BARYON_SLOPE"].iloc[0], 0.2)
    assert np.isclose(metrics["BARYON_RANGE"].iloc[0], 0.4)


def test_category_assignment_rules():
    base = pd.DataFrame({
        "observable_name": ["StarShape_groupTidal_Mstar_major", "CGHA_BaryonDM_major", "TidalMajorRadial_GRMG_Mstar"],
        "component": ["StarShape_groupTidal", "CGHA", "TidalMajorRadial_GRMG"],
        "population": ["satellite", "central", "satellite"],
        "axis": ["major", "major", "none"],
        "reference": ["Tgroup", "halo", "T_GR+MG"],
        "x_variable": ["Mstar", "BaryonDM", "Mstar"],
        "MG_RMS": [0.2, 0.01, 0.3],
        "MG_MAX": [0.3, 0.02, 0.4],
        "MG_SNR_PROXY": [2.0, np.nan, 3.0],
        "BARYON_RMS": [np.nan, 0.5, np.nan],
        "BARYON_SLOPE": [np.nan, 0.4, np.nan],
    })
    out = assign_categories(base)
    by_name = out.set_index("observable_name")
    assert by_name.loc["StarShape_groupTidal_Mstar_major", "survey_accessibility"] >= 2
    assert by_name.loc["CGHA_BaryonDM_major", "category"] == "baryon_control"
    assert by_name.loc["TidalMajorRadial_GRMG_Mstar", "category"] == "theory_template"


def test_pdf_csv_hdf5_smoke(tmp_path):
    table = load_alignment_table_from_frame(_synthetic_table())
    pdf = tmp_path / "report.pdf"
    csv = tmp_path / "ranking.csv"
    h5 = tmp_path / "metrics.hdf5"
    metrics, outputs = build_report(table, output_pdf=pdf, output_table=csv, output_hdf5=h5, max_detail_pages=2)
    assert outputs.pdf.exists() and outputs.pdf.stat().st_size > 0
    assert outputs.csv.exists() and len(pd.read_csv(csv)) == len(metrics)
    assert outputs.hdf5.exists()
    with h5py.File(h5, "r") as handle:
        assert "metrics" in handle
        assert "residuals" in handle


def test_write_and_reload_metrics_hdf5(tmp_path):
    table = load_alignment_table_from_frame(_synthetic_table())
    metrics, residuals = compute_all_metrics(table)
    path = tmp_path / "metrics.hdf5"
    write_metrics_hdf5(path, metrics, residuals, table)
    reloaded = load_alignment_table(path)
    assert len(reloaded) == len(table)
