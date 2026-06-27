"""Modified-gravity versus baryonic-response diagnostics for alignment summaries.

This module consumes a canonical long-form table of binned alignment fits and
builds a compact diagnostic report.  It is intentionally downstream of the
existing alignment plotting machinery: no PDF/image parsing is used.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

_mpl_config_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "ia_analysis_matplotlib"
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm

try:  # Reuse the current plotting style when available.
    from ia_analysis.visualization import arts_IA
except Exception:  # pragma: no cover
    arts_IA = None


LOGGER = logging.getLogger(__name__)

GRAVITY_ORDER = ("GR", "F4", "F4.5", "F5", "F5.5", "F6")
FLAG_ALIASES = {"F40": "F4", "F45": "F4.5", "F50": "F5", "F55": "F5.5", "F60": "F6"}

CANONICAL_COLUMNS = [
    "observable_name",
    "component",
    "population",
    "axis",
    "reference",
    "x_variable",
    "x_value",
    "x_low",
    "x_high",
    "redshift",
    "snapshot",
    "gravity_model",
    "mu",
    "mu_error",
    "count",
]


@dataclass(frozen=True)
class ReportOutputs:
    """Paths written by :func:`build_report`."""

    pdf: Path | None
    csv: Path | None
    hdf5: Path | None


def _as_model_label(model: Any) -> str:
    value = str(model)
    return FLAG_ALIASES.get(value, value)


def _norm_text(value: Any) -> str:
    return "" if pd.isna(value) else str(value).lower()


def infer_metadata_from_name(name: str) -> dict[str, str]:
    """Infer practical metadata from an existing alignment spec name."""

    tokens = str(name).split("_")
    axis = "none"
    if tokens[-1] in {"major", "medium", "minor", "intermediate"}:
        axis = "intermediate" if tokens[-1] == "medium" else tokens[-1]
        x_variable = tokens[-2] if len(tokens) >= 2 else "unknown"
        base = "_".join(tokens[:-2])
    else:
        x_variable = tokens[-1] if tokens else "unknown"
        base = "_".join(tokens[:-1]) if len(tokens) > 1 else str(name)

    lower = str(name).lower()
    population = "all"
    if lower.startswith("cg") or "central" in lower:
        population = "central"
    elif any(key in lower for key in ["sat", "sgha", "subdm", "vel", "omega", "tidalmajorradial"]):
        population = "satellite"

    reference = "unknown"
    if "grmg" in lower or "t_gr+mg" in lower:
        reference = "T_GR+MG"
    elif "grtidal" in lower or "tidal_gr" in lower:
        reference = "T_GR"
    elif "grouptidal" in lower or "tgroup" in lower or "_group" in lower:
        reference = "Tgroup"
    elif "radial" in lower:
        reference = "radial"
    elif "vel" in lower:
        reference = "velocity"
    elif "sub" in lower:
        reference = "subhalo"
    elif "halo" in lower or "cgha" in lower:
        reference = "halo"

    component = base or str(name)
    return {
        "component": component,
        "population": population,
        "axis": axis,
        "reference": reference,
        "x_variable": x_variable,
    }


def canonicalize_alignment_table(table: pd.DataFrame, *, min_count: int = 0) -> pd.DataFrame:
    """Validate and normalize a long-form alignment table."""

    df = table.copy()
    aliases = {
        "spec": "observable_name",
        "name": "observable_name",
        "flag": "gravity_model",
        "model": "gravity_model",
        "snap": "snapshot",
        "z": "redshift",
        "error": "mu_error",
        "n": "count",
        "nn": "count",
        "x": "x_value",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns and v not in df.columns})
    if "observable_name" not in df:
        raise ValueError("Alignment table requires an observable_name/spec column.")
    if "gravity_model" not in df:
        raise ValueError("Alignment table requires a gravity_model/flag column.")
    if "mu" not in df:
        raise ValueError("Alignment table requires a mu column.")

    inferred = df["observable_name"].map(infer_metadata_from_name)
    for col in ["component", "population", "axis", "reference", "x_variable"]:
        if col not in df:
            df[col] = [item[col] for item in inferred]
        else:
            df[col] = df[col].fillna(pd.Series([item[col] for item in inferred], index=df.index))

    defaults = {
        "x_value": np.nan,
        "x_low": np.nan,
        "x_high": np.nan,
        "redshift": np.nan,
        "snapshot": -1,
        "mu_error": np.nan,
        "count": np.nan,
    }
    for col, default in defaults.items():
        if col not in df:
            df[col] = default

    for col in ["x_value", "x_low", "x_high", "redshift", "mu", "mu_error", "count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["snapshot"] = pd.to_numeric(df["snapshot"], errors="coerce").fillna(-1).astype(int)
    df["gravity_model"] = df["gravity_model"].map(_as_model_label)

    before = len(df)
    df = df[np.isfinite(df["mu"])].copy()
    dropped = before - len(df)
    if dropped:
        LOGGER.warning("Dropped %d rows with non-finite mu.", dropped)

    if min_count > 0:
        known = df["count"].notna()
        low = known & (df["count"] < min_count)
        if low.any():
            LOGGER.info("Marked %d low-count rows as NaN using min_count=%d.", int(low.sum()), min_count)
            df.loc[low, "mu"] = np.nan
            df = df[np.isfinite(df["mu"])].copy()

    for col in CANONICAL_COLUMNS:
        if col not in df:
            df[col] = np.nan
    return df[CANONICAL_COLUMNS].sort_values(
        ["observable_name", "x_variable", "gravity_model", "snapshot", "x_value"]
    ).reset_index(drop=True)


def canonical_table_from_mock_summary(summary: dict[str, Any]) -> pd.DataFrame:
    """Adapter for tests and simple in-memory summary dictionaries.

    Expected shape:
    ``{observable: {model: {snapshot: {"x": ..., "mu": ..., "mu_error": ..., "count": ...}}}}``.
    """

    rows: list[dict[str, Any]] = []
    for obs, by_model in summary.items():
        meta = infer_metadata_from_name(obs)
        for model, by_snap in by_model.items():
            for snap, payload in by_snap.items():
                x = np.asarray(payload.get("x", payload.get("x_value", [])), dtype=float)
                mu = np.asarray(payload.get("mu", []), dtype=float)
                err = np.asarray(payload.get("mu_error", np.full_like(mu, np.nan)), dtype=float)
                count = np.asarray(payload.get("count", np.full_like(mu, np.nan)), dtype=float)
                if x.size == 0:
                    x = np.arange(mu.size, dtype=float)
                edges = np.asarray(payload.get("x_edges", []), dtype=float)
                for i, value in enumerate(mu):
                    row = {
                        "observable_name": obs,
                        **meta,
                        "x_value": float(x[i]),
                        "x_low": float(edges[i]) if edges.size == mu.size + 1 else np.nan,
                        "x_high": float(edges[i + 1]) if edges.size == mu.size + 1 else np.nan,
                        "snapshot": int(snap),
                        "redshift": float(payload.get("redshift", np.nan)),
                        "gravity_model": model,
                        "mu": float(value),
                        "mu_error": float(err[i]) if i < err.size else np.nan,
                        "count": float(count[i]) if i < count.size else np.nan,
                    }
                    rows.append(row)
    return canonicalize_alignment_table(pd.DataFrame(rows))


def canonical_table_from_arts_profiles(
    *,
    root_dir: str | Path,
    requested_flags: Iterable[str] = ("GR", "F40", "F45", "F50", "F55", "F60"),
    snap_list: Iterable[int] = (1, 3, 6, 8, 10, 12, 15, 18, 21),
    specs: Iterable[str] | None = None,
    min_count: int = 8,
) -> pd.DataFrame:
    """Materialize the canonical table from the existing ``arts_IA`` pipeline."""

    if arts_IA is None:
        raise RuntimeError("ia_analysis.visualization.arts_IA is unavailable.")
    maset, flags, missing = arts_IA.load_alignment_maset(
        Path(root_dir),
        requested_flags=list(requested_flags),
        snap_list=list(snap_list),
        verbose=True,
        strict=False,
    )
    if missing:
        LOGGER.warning("Missing/failed input catalogues while materializing table: %d", len(missing))
    snaps = [int(s) for s in snap_list]
    zmap = {snap: arts_IA.ZMAP_ALL.get(snap, np.nan) for snap in snaps}
    arts_IA.set_alignment_context(maset, flags, zmap, snap_list=snaps)
    selected = [arts_IA.get_alignment_spec_by_name(s) for s in specs] if specs else list(arts_IA.ALIGNMENT_SPECS)

    rows: list[dict[str, Any]] = []
    for spec in selected:
        meta = infer_metadata_from_name(spec.name)
        for flag in flags:
            for snap in snaps:
                profile = arts_IA.get_binned_alignment_profile(spec, flag, snap, min_count=min_count)
                if profile is None:
                    LOGGER.info("No valid profile for %s %s snap=%s.", spec.name, flag, snap)
                    continue
                x, mu, err, count = profile
                for xv, yv, ev, nv in zip(x, mu, err, count):
                    if not np.isfinite(yv):
                        continue
                    rows.append({
                        "observable_name": spec.name,
                        **meta,
                        "x_value": xv,
                        "x_low": np.nan,
                        "x_high": np.nan,
                        "redshift": zmap.get(int(snap), np.nan),
                        "snapshot": int(snap),
                        "gravity_model": flag,
                        "mu": yv,
                        "mu_error": ev,
                        "count": nv,
                    })
    return canonicalize_alignment_table(pd.DataFrame(rows), min_count=min_count)


def load_alignment_table(path: str | Path, *, min_count: int = 0) -> pd.DataFrame:
    """Load a canonical alignment table from CSV, HDF5, or NPZ."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        table = pd.read_csv(path)
    elif suffix in {".hdf5", ".h5"}:
        try:
            table = pd.read_hdf(path, key="alignment_points")
        except Exception:
            with h5py.File(path, "r") as h5:
                group = h5["alignment_points"] if "alignment_points" in h5 else h5
                data = {}
                for key, ds in group.items():
                    values = ds[()]
                    if values.dtype.kind == "S":
                        values = values.astype(str)
                    data[key] = values
                table = pd.DataFrame(data)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            table = pd.DataFrame({key: data[key] for key in data.files})
    else:
        raise ValueError(f"Unsupported alignment summary format: {path.suffix}")
    return canonicalize_alignment_table(table, min_count=min_count)


def validate_alignment_table(table: pd.DataFrame, *, baseline_model: str = "GR") -> None:
    """Log table-level validation information."""

    models = set(table["gravity_model"].dropna().astype(str))
    if baseline_model not in models:
        raise ValueError(f"Baseline model {baseline_model!r} is required for residual calculations.")
    if len(models) < 2:
        LOGGER.warning("Only one gravity model is present; MG residual diagnostics will be empty.")
    grouped = table.groupby(["observable_name", "x_variable"], dropna=False)
    for key, group in grouped:
        if baseline_model not in set(group["gravity_model"]):
            LOGGER.warning("Skipping MG residuals for %s: no baseline model.", key)


def compute_mg_residuals(table: pd.DataFrame, *, baseline_model: str = "GR") -> pd.DataFrame:
    """Compute ``Delta_mu = mu(model) - mu(baseline)`` per matched bin."""

    keys = [
        "observable_name",
        "component",
        "population",
        "axis",
        "reference",
        "x_variable",
        "x_value",
        "redshift",
        "snapshot",
    ]
    gr = table[table["gravity_model"] == baseline_model][keys + ["mu", "mu_error"]].rename(
        columns={"mu": "mu_baseline", "mu_error": "mu_error_baseline"}
    )
    other = table[table["gravity_model"] != baseline_model].copy()
    if other.empty:
        return pd.DataFrame(columns=keys + ["gravity_model", "delta_mu", "sigma_combined", "delta_snr"])
    merged = other.merge(gr, on=keys, how="inner")
    missing = len(other) - len(merged)
    if missing > 0:
        LOGGER.warning("Could not baseline-match %d non-GR alignment rows.", missing)
    merged["delta_mu"] = merged["mu"] - merged["mu_baseline"]
    merged["sigma_combined"] = np.sqrt(merged["mu_error"] ** 2 + merged["mu_error_baseline"] ** 2)
    merged.loc[~np.isfinite(merged["sigma_combined"]) | (merged["sigma_combined"] <= 0), "sigma_combined"] = np.nan
    merged["delta_snr"] = merged["delta_mu"] / merged["sigma_combined"]
    return merged[keys + ["gravity_model", "mu", "mu_baseline", "delta_mu", "sigma_combined", "delta_snr"]]


def _rms(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.sqrt(np.nanmean(arr**2))) if arr.size else np.nan


def _sign_coherence(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr) & (arr != 0)]
    if arr.size == 0:
        return np.nan
    return float(max(np.mean(arr > 0), np.mean(arr < 0)))


def compute_mg_metrics(residuals: pd.DataFrame) -> pd.DataFrame:
    """Summarize residual arrays per observable/component/x-variable."""

    group_cols = ["observable_name", "component", "population", "axis", "reference", "x_variable"]
    rows: list[dict[str, Any]] = []
    for key, group in residuals.groupby(group_cols, dropna=False):
        delta = group["delta_mu"].to_numpy(dtype=float)
        row = dict(zip(group_cols, key))
        row["MG_RMS"] = _rms(delta)
        row["MG_MAX"] = float(np.nanmax(np.abs(delta))) if np.isfinite(delta).any() else np.nan
        row["MG_SIGN_COHERENCE"] = _sign_coherence(delta)
        row["MG_SNR_PROXY"] = _rms(group["delta_snr"].to_numpy(dtype=float))

        rz = []
        for _, sub in group.groupby(["gravity_model", "x_value"], dropna=False):
            vals = sub.sort_values("redshift")["delta_mu"].to_numpy(dtype=float)
            if np.isfinite(vals).sum() > 1:
                rz.append(float(np.nanstd(vals)))
        rx = []
        for _, sub in group.groupby(["gravity_model", "redshift", "snapshot"], dropna=False):
            vals = sub.sort_values("x_value")["delta_mu"].to_numpy(dtype=float)
            if np.isfinite(vals).sum() > 1:
                rx.append(float(np.nanstd(vals)))
        row["MG_REDSHIFT_LEVERAGE"] = _rms(rz)
        row["MG_MASS_LEVERAGE"] = _rms(rx)
        row["n_residual_bins"] = int(np.isfinite(delta).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def compute_baryon_metrics(table: pd.DataFrame) -> pd.DataFrame:
    """Compute baryonic-response metrics for BaryonDM-like x variables."""

    group_cols = ["observable_name", "component", "population", "axis", "reference", "x_variable"]
    rows: list[dict[str, Any]] = []
    baryon = table[table["x_variable"].astype(str).str.lower().str.contains("baryon|gas|fb|fbar", regex=True)]
    for key, group in baryon.groupby(group_cols, dropna=False):
        slopes = []
        intercepts = []
        ranges = []
        rms_vals = []
        for _, sub in group.groupby(["gravity_model", "snapshot", "redshift"], dropna=False):
            x = sub["x_value"].to_numpy(dtype=float)
            y = sub["mu"].to_numpy(dtype=float)
            good = np.isfinite(x) & np.isfinite(y)
            if good.sum() < 2:
                continue
            coeff = np.polyfit(x[good], y[good], 1)
            slopes.append(float(coeff[0]))
            intercepts.append(float(coeff[1]))
            ranges.append(float(np.nanmax(y[good]) - np.nanmin(y[good])))
            rms_vals.append(float(np.nanstd(y[good])))
        row = dict(zip(group_cols, key))
        row["BARYON_SLOPE"] = float(np.nanmedian(slopes)) if slopes else np.nan
        row["BARYON_RMS"] = _rms(rms_vals)
        row["BARYON_RANGE"] = float(np.nanmedian(ranges)) if ranges else np.nan
        row["BARYON_SLOPE_MODEL_SCATTER"] = float(np.nanstd(slopes)) if slopes else np.nan
        row["BARYON_INTERCEPT_MODEL_SCATTER"] = float(np.nanstd(intercepts)) if intercepts else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def survey_accessibility_score(row: pd.Series | dict[str, Any]) -> int:
    """Rule-based survey accessibility score in [0, 3]."""

    text = " ".join(_norm_text(row.get(k, "")) for k in ["observable_name", "component", "reference", "x_variable"])
    if any(key in text for key in ["grmg", "t_gr+mg", "total tidal", "model-specific", "mg tidal"]):
        return 0
    if any(key in text for key in ["subdm", "subhalo", "baryondm", "stellar-dm", "star-dm", "binding"]):
        return 1
    if any(key in text for key in ["tgroup", "group", "velocity", "velradial", "radial"]):
        return 2
    if any(key in text for key in ["cgha", "sgha", "shape-density", "ed", "ee"]):
        return 3
    return 1


def _has_phase_space(row: pd.Series | dict[str, Any]) -> bool:
    text = " ".join(_norm_text(row.get(k, "")) for k in ["observable_name", "component", "reference"])
    return any(key in text for key in ["radial", "velocity", "vel", "omega", "orbit", "phase"])


def _is_theory_template(row: pd.Series | dict[str, Any]) -> bool:
    text = " ".join(_norm_text(row.get(k, "")) for k in ["observable_name", "component", "reference"])
    return any(key in text for key in ["grmg", "t_gr+mg", "mg tidal", "total tidal"])


def assign_categories(metrics: pd.DataFrame) -> pd.DataFrame:
    """Assign accessibility, category, ranking score, and notes."""

    df = metrics.copy()
    if df.empty:
        return df
    for col in ["MG_RMS", "MG_MAX", "MG_SNR_PROXY", "BARYON_RMS", "BARYON_SLOPE"]:
        if col not in df:
            df[col] = np.nan
    df["survey_accessibility"] = [survey_accessibility_score(row) for _, row in df.iterrows()]
    df["MG_TO_BARYON_RATIO"] = df["MG_RMS"] / df["BARYON_RMS"].replace(0, np.nan)

    mg_cut = np.nanpercentile(df["MG_RMS"].dropna(), 60) if df["MG_RMS"].notna().any() else np.inf
    bary_cut = np.nanpercentile(df["BARYON_RMS"].dropna(), 60) if df["BARYON_RMS"].notna().any() else np.inf
    slope_cut = np.nanpercentile(np.abs(df["BARYON_SLOPE"].dropna()), 60) if df["BARYON_SLOPE"].notna().any() else np.inf

    categories = []
    notes = []
    for _, row in df.iterrows():
        baryon_candidates = [row.get("BARYON_RMS", np.nan), abs(row.get("BARYON_SLOPE", np.nan))]
        baryon_candidates = [float(v) for v in baryon_candidates if np.isfinite(v)]
        baryon_strength = max(baryon_candidates) if baryon_candidates else np.nan
        mg_strength = row.get("MG_SNR_PROXY", np.nan)
        if not np.isfinite(mg_strength):
            mg_strength = row.get("MG_RMS", np.nan)
        if _is_theory_template(row):
            cat = "theory_template"
            note = "Simulation-level MG response template; not directly survey-facing."
        elif np.isfinite(baryon_strength) and (row.get("BARYON_RMS", np.nan) >= bary_cut or abs(row.get("BARYON_SLOPE", np.nan)) >= slope_cut):
            cat = "baryon_control"
            note = "BaryonDM trend is useful for galaxy-formation nuisance priors."
        elif _has_phase_space(row):
            cat = "phase_space_nuisance"
            note = "One-halo satellite/phase-space term for HOD and RSD consistency."
        elif row["survey_accessibility"] >= 2 and np.isfinite(mg_strength) and row.get("MG_RMS", np.nan) >= mg_cut:
            cat = "survey_mg_probe"
            note = "Survey-facing IA observable with coherent MG residual response."
        else:
            cat = "low_priority_or_redundant"
            note = "Weak response, low accessibility, or redundant with stronger diagnostics."
        if "tgroup" in _norm_text(row.get("reference", "")):
            note = "Tgroup proxy is survey-facing through group catalogs or reconstructed tidal fields."
        categories.append(cat)
        notes.append(note)
    df["category"] = categories
    df["notes"] = notes
    df["ranking_score"] = (
        df["MG_RMS"].fillna(0)
        + 0.25 * df["MG_MAX"].fillna(0)
        + 0.1 * df["MG_SNR_PROXY"].fillna(0)
        + 0.02 * df["survey_accessibility"].fillna(0)
    )
    return df.sort_values(["category", "ranking_score"], ascending=[True, False]).reset_index(drop=True)


def compute_all_metrics(table: pd.DataFrame, *, baseline_model: str = "GR") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(ranking_metrics, residual_rows)``."""

    validate_alignment_table(table, baseline_model=baseline_model)
    residuals = compute_mg_residuals(table, baseline_model=baseline_model)
    mg = compute_mg_metrics(residuals) if not residuals.empty else pd.DataFrame()
    bary = compute_baryon_metrics(table)
    key_cols = ["observable_name", "component", "population", "axis", "reference", "x_variable"]
    if mg.empty:
        base = table[key_cols].drop_duplicates().copy()
    else:
        base = mg
    if not bary.empty:
        base = base.merge(bary, on=key_cols, how="outer")
    metrics = assign_categories(base)
    return metrics, residuals


def _write_string_dataset(group: h5py.Group, name: str, values: Iterable[Any]) -> None:
    data = np.asarray(["" if pd.isna(v) else str(v) for v in values], dtype=h5py.string_dtype("utf-8"))
    group.create_dataset(name, data=data)


def write_dataframe_hdf5(h5: h5py.File, name: str, df: pd.DataFrame) -> None:
    group = h5.create_group(name)
    for col in df.columns:
        values = df[col].to_numpy()
        if values.dtype.kind in {"O", "U", "S"}:
            _write_string_dataset(group, col, values)
        else:
            group.create_dataset(col, data=values)


def write_metrics_hdf5(path: str | Path, metrics: pd.DataFrame, residuals: pd.DataFrame, table: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        write_dataframe_hdf5(h5, "metrics", metrics)
        write_dataframe_hdf5(h5, "residuals", residuals)
        write_dataframe_hdf5(h5, "alignment_points", table)


def _format_value(value: Any, precision: int = 4) -> str:
    try:
        val = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(val):
        return "nan"
    return f"{val:.{precision}g}"


def _text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 8.5))
    ax.axis("off")
    ax.text(0.02, 0.96, title, ha="left", va="top", fontsize=18, weight="bold")
    ax.text(0.02, 0.88, "\n".join(lines), ha="left", va="top", fontsize=10.5, linespacing=1.45, wrap=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _table_page(pdf: PdfPages, title: str, df: pd.DataFrame, columns: list[str], *, max_rows: int = 14) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 8.5))
    ax.axis("off")
    ax.set_title(title, fontsize=15, weight="bold", pad=10)
    view = df.head(max_rows).copy()
    if view.empty:
        ax.text(0.5, 0.5, "No rows available", ha="center", va="center")
    else:
        display = []
        for i, (_, row) in enumerate(view.iterrows(), start=1):
            vals = [i]
            for col in columns:
                vals.append(_format_value(row.get(col, "")) if col.isupper() or col.startswith("MG_") or col.startswith("BARYON") else str(row.get(col, "")))
            display.append(vals)
        col_labels = ["rank"] + columns
        table = ax.table(cellText=display, colLabels=col_labels, loc="center", cellLoc="left", colLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1.0, 1.35)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _inventory_page(pdf: PdfPages, table: pd.DataFrame, metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5))
    axes = axes.ravel()
    items = [
        ("Category", metrics["category"].value_counts()),
        ("Population", table["population"].value_counts()),
        ("Reference", table["reference"].value_counts().head(12)),
        ("x variable", table["x_variable"].value_counts().head(12)),
    ]
    for ax, (title, counts) in zip(axes, items):
        counts.sort_values().plot.barh(ax=ax, color="#4c78a8")
        ax.set_title(title, weight="bold")
        ax.set_xlabel("count")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Observable Inventory", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _mg_vs_baryon_page(pdf: PdfPages, metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.6))
    marker_map = {
        "survey_mg_probe": "o",
        "baryon_control": "s",
        "phase_space_nuisance": "^",
        "theory_template": "D",
        "low_priority_or_redundant": "x",
    }
    for cat, sub in metrics.groupby("category"):
        ax.scatter(
            sub["BARYON_RMS"].fillna(0),
            sub["MG_SNR_PROXY"].where(sub["MG_SNR_PROXY"].notna(), sub["MG_RMS"]).fillna(0),
            c=sub["survey_accessibility"],
            cmap="viridis",
            vmin=0,
            vmax=3,
            marker=marker_map.get(cat, "o"),
            label=cat,
            edgecolors="k" if marker_map.get(cat, "o") != "x" else "none",
            alpha=0.85,
        )
    top = metrics.sort_values("ranking_score", ascending=False).head(8)
    for _, row in top.iterrows():
        x = 0 if not np.isfinite(row.get("BARYON_RMS", np.nan)) else row["BARYON_RMS"]
        y = row["MG_SNR_PROXY"] if np.isfinite(row.get("MG_SNR_PROXY", np.nan)) else row.get("MG_RMS", 0)
        ax.annotate(str(row["observable_name"])[:28], (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Baryonic response: BARYON_RMS")
    ax.set_ylabel("MG response: MG_SNR_PROXY or MG_RMS")
    ax.set_title("MG-vs-baryon separation", weight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="best")
    caption = (
        "Upper-left points are baryon-robust MG probes; lower-right points are baryonic nuisance controls; "
        "upper-right points need joint MG+baryon modeling."
    )
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _heatmap_page(pdf: PdfPages, residuals: pd.DataFrame, observable: str) -> None:
    sub = residuals[residuals["observable_name"] == observable]
    if sub.empty:
        return
    models = [m for m in GRAVITY_ORDER if m != "GR" and m in set(sub["gravity_model"])]
    ncols = min(3, max(1, len(models)))
    nrows = int(math.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows), squeeze=False)
    vmax = np.nanpercentile(np.abs(sub["delta_mu"]), 95) if sub["delta_mu"].notna().any() else 1.0
    vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0
    for ax, model in zip(axes.ravel(), models):
        sm = sub[sub["gravity_model"] == model]
        pivot = sm.pivot_table(index="redshift", columns="x_value", values="delta_mu", aggfunc="mean").sort_index()
        im = ax.imshow(
            pivot.to_numpy(dtype=float),
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        )
        ax.set_title(model)
        ax.set_xlabel(str(sm["x_variable"].iloc[0]))
        ax.set_ylabel("redshift")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([_format_value(v, 3) for v in pivot.columns], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([_format_value(v, 3) for v in pivot.index], fontsize=8)
    for ax in axes.ravel()[len(models):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label=r"$\Delta\mu = \mu(F_i)-\mu({\rm GR})$")
    fig.suptitle(f"MG residual heatmap: {observable}", fontsize=14, weight="bold")
    fig.text(0.5, 0.02, "Diverging colors are centered on GR; masked bins lack a matched GR/non-GR measurement.", ha="center", fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.16, top=0.88, wspace=0.3, hspace=0.4)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _detail_page(pdf: PdfPages, table: pd.DataFrame, residuals: pd.DataFrame, metrics_row: pd.Series) -> None:
    obs = metrics_row["observable_name"]
    xvar = metrics_row["x_variable"]
    sub = table[(table["observable_name"] == obs) & (table["x_variable"] == xvar)]
    if sub.empty:
        return
    snaps = list(sub["snapshot"].drop_duplicates().sort_values())
    if len(snaps) > 3:
        snaps = [snaps[0], snaps[len(snaps) // 2], snaps[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for snap in snaps:
        ss = sub[sub["snapshot"] == snap]
        for model in [m for m in GRAVITY_ORDER if m in set(ss["gravity_model"])]:
            line = ss[ss["gravity_model"] == model].sort_values("x_value")
            axes[0].plot(line["x_value"], line["mu"], marker="o", ms=2.5, lw=1.2, label=f"{model} s{snap}")
    rsub = residuals[(residuals["observable_name"] == obs) & (residuals["x_variable"] == xvar)]
    for model in [m for m in GRAVITY_ORDER if m != "GR" and m in set(rsub["gravity_model"])]:
        line = rsub[rsub["gravity_model"] == model].sort_values("x_value")
        axes[1].plot(line["x_value"], line["delta_mu"], marker="o", ms=2.5, lw=1.2, label=model)
    axes[0].set_title(r"$\mu(x)$ curves")
    axes[1].set_title(r"Residual curves")
    for ax in axes:
        ax.set_xlabel(str(xvar))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6, frameon=False, ncol=2)
    axes[0].set_ylabel(r"Dimroth-Watson $\mu$")
    axes[1].set_ylabel(r"$\Delta\mu$")
    summary = f"{metrics_row['category']}: {metrics_row['notes']}"
    fig.suptitle(f"Detail: {obs}", fontsize=14, weight="bold")
    fig.text(0.5, 0.02, summary, ha="center", fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_report_pdf(path: str | Path, table: pd.DataFrame, metrics: pd.DataFrame, residuals: pd.DataFrame, *, max_detail_pages: int = 12) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arts_IA is not None:
        arts_IA.set_paper_style()
    with PdfPages(path) as pdf:
        top_mg = metrics[metrics["survey_accessibility"] >= 2].sort_values("ranking_score", ascending=False).head(10)
        top_bary = metrics[metrics["category"] == "baryon_control"].sort_values("BARYON_RMS", ascending=False).head(10)
        top_theory = metrics[metrics["category"] == "theory_template"].sort_values("MG_RMS", ascending=False).head(10)
        lines = [
            f"Alignment points analyzed: {len(table)}",
            f"Observable/x-variable summaries: {len(metrics)}",
            f"Gravity models: {', '.join([m for m in GRAVITY_ORDER if m in set(table['gravity_model'])])}",
            f"Redshifts/snapshots: {table['redshift'].nunique(dropna=True)} / {table['snapshot'].nunique(dropna=True)}",
            "",
            "Top survey-accessible MG probes: " + ", ".join(top_mg["observable_name"].astype(str).head(10)),
            "Top baryon controls: " + ", ".join(top_bary["observable_name"].astype(str).head(10)),
            "Top theory templates: " + ", ".join(top_theory["observable_name"].astype(str).head(10)),
            "",
            "Future joint likelihood: combine clustering/RSD/lensing with central/group tidal IA, satellite radial IA, HOD occupation, phase-space controls, and BaryonDM calibration priors.",
        ]
        _text_page(pdf, "Alignment MG-Baryon Diagnostic Report", lines)
        _inventory_page(pdf, table, metrics)
        for obs in metrics.sort_values("ranking_score", ascending=False)["observable_name"].drop_duplicates().head(8):
            _heatmap_page(pdf, residuals, str(obs))
        _mg_vs_baryon_page(pdf, metrics)
        cols = ["observable_name", "component", "x_variable", "survey_accessibility", "MG_RMS", "MG_MAX", "MG_SNR_PROXY", "BARYON_RMS", "MG_TO_BARYON_RATIO", "category"]
        tables = [
            ("Best Survey-Accessible MG Probes", metrics[metrics["survey_accessibility"] >= 2].sort_values("ranking_score", ascending=False)),
            ("Best Baryon Controls", metrics[metrics["category"] == "baryon_control"].sort_values("BARYON_RMS", ascending=False)),
            ("Best Phase-Space Nuisance Controls", metrics[metrics["category"] == "phase_space_nuisance"].sort_values("ranking_score", ascending=False)),
            ("Best Theory-Template MG Responses", metrics[metrics["category"] == "theory_template"].sort_values("MG_RMS", ascending=False)),
            ("Strong MG-Baryon Degeneracy", metrics.sort_values("MG_TO_BARYON_RATIO", ascending=True, na_position="last")),
        ]
        for title, df in tables:
            _table_page(pdf, title, df, cols)
        _text_page(pdf, "Recommended Data Vector", [
            "Clustering: xi_gg / P_gg / RSD multipoles.",
            "Lensing: DeltaSigma plus halo mass and concentration priors.",
            "IA: central-Tgroup alignment, satellite radial alignment, satellite-Tgroup alignment, and ED/EE summaries when available.",
            "HOD: N_cen(M), N_sat(M), satellite fraction, radial profile, and velocity-radial consistency terms.",
            "Baryon controls: BaryonDM proxy dependence and stellar-DM or galaxy-subhalo misalignment calibration.",
            "Survey-facing observables use galaxy/group/tidal-field proxies; T_GR+MG entries are simulation templates and should enter as theory priors, not observed data-vector elements.",
        ])
        for _, row in metrics.sort_values("ranking_score", ascending=False).head(max_detail_pages).iterrows():
            _detail_page(pdf, table, residuals, row)


def build_report(
    table: pd.DataFrame,
    *,
    output_pdf: str | Path | None = None,
    output_table: str | Path | None = None,
    output_hdf5: str | Path | None = None,
    baseline_model: str = "GR",
    gravity_models: Iterable[str] | None = None,
    selected_observables: Iterable[str] | None = None,
    min_count: int = 0,
    make_pdf: bool = True,
    make_csv: bool = True,
    make_hdf5: bool = True,
    max_detail_pages: int = 12,
) -> tuple[pd.DataFrame, ReportOutputs]:
    """Compute diagnostics and write requested report products."""

    table = canonicalize_alignment_table(table, min_count=min_count)
    if gravity_models:
        keep = {_as_model_label(m) for m in gravity_models}
        keep.add(baseline_model)
        table = table[table["gravity_model"].isin(keep)].copy()
    if selected_observables:
        patterns = [re.compile(str(p)) for p in selected_observables]
        mask = table["observable_name"].map(lambda x: any(p.search(str(x)) for p in patterns))
        table = table[mask].copy()
    metrics, residuals = compute_all_metrics(table, baseline_model=baseline_model)

    out_pdf = Path(output_pdf) if output_pdf else None
    out_csv = Path(output_table) if output_table else None
    out_h5 = Path(output_hdf5) if output_hdf5 else None
    if make_csv and out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(out_csv, index=False)
    if make_hdf5 and out_h5 is not None:
        write_metrics_hdf5(out_h5, metrics, residuals, table)
    if make_pdf and out_pdf is not None:
        write_report_pdf(out_pdf, table, metrics, residuals, max_detail_pages=max_detail_pages)
    return metrics, ReportOutputs(pdf=out_pdf if make_pdf else None, csv=out_csv if make_csv else None, hdf5=out_h5 if make_hdf5 else None)


def _parse_csv_or_space(values: list[str] | None, cast=str):
    if not values:
        return None
    out = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            out.append(cast(part))
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MG-vs-baryon diagnostics from alignment summary tables.")
    parser.add_argument("--input-summary", help="Canonical or compatible alignment table: CSV, HDF5, or NPZ.")
    parser.add_argument("--catalog-root", help="Optional existing alignment catalogue root for arts_IA materialization.")
    parser.add_argument("--output-pdf", default="alignment_mg_baryon_diagnostic_report.pdf")
    parser.add_argument("--output-table", default="alignment_mg_baryon_ranking.csv")
    parser.add_argument("--output-hdf5", default="alignment_mg_baryon_metrics.hdf5")
    parser.add_argument("--materialized-table", help="Optional CSV path to save the canonical long-form table.")
    parser.add_argument("--min-count", type=int, default=0)
    parser.add_argument("--gravity-models", nargs="+", default=None)
    parser.add_argument("--baseline-model", default="GR")
    parser.add_argument("--selected-observables", nargs="+", default=None)
    parser.add_argument("--max-detail-pages", type=int, default=12)
    parser.add_argument("--make-pdf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make-csv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make-hdf5", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--snaps", nargs="+", default=["1", "3", "6", "8", "10", "12", "15", "18", "21"])
    parser.add_argument("--specs", nargs="+", default=None, help="Spec subset when using --catalog-root.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    models = _parse_csv_or_space(args.gravity_models, str)
    if args.input_summary:
        table = load_alignment_table(args.input_summary, min_count=args.min_count)
    elif args.catalog_root:
        table = canonical_table_from_arts_profiles(
            root_dir=args.catalog_root,
            requested_flags=models or ("GR", "F40", "F45", "F50", "F55", "F60"),
            snap_list=_parse_csv_or_space(args.snaps, int),
            specs=_parse_csv_or_space(args.specs, str),
            min_count=args.min_count,
        )
    else:
        raise SystemExit("Provide --input-summary or --catalog-root.")

    if args.materialized_table:
        out = Path(args.materialized_table)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        LOGGER.info("Wrote canonical alignment table: %s", out)

    _, outputs = build_report(
        table,
        output_pdf=args.output_pdf,
        output_table=args.output_table,
        output_hdf5=args.output_hdf5,
        baseline_model=args.baseline_model,
        gravity_models=models,
        selected_observables=_parse_csv_or_space(args.selected_observables, str),
        min_count=args.min_count,
        make_pdf=args.make_pdf,
        make_csv=args.make_csv,
        make_hdf5=args.make_hdf5,
        max_detail_pages=args.max_detail_pages,
    )
    for label, path in [("PDF", outputs.pdf), ("CSV", outputs.csv), ("HDF5", outputs.hdf5)]:
        if path is not None:
            LOGGER.info("%s: %s", label, path)


if __name__ == "__main__":  # pragma: no cover
    main()
