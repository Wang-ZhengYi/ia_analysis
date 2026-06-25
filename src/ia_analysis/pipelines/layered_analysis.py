"""Thin orchestration for the layered analysis APIs.

Dependency direction:
catalogs -> shapes/tides -> dynamics/MergerTree ->
spectra/correlations -> visualization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ia_analysis.catalogs.analysis import inventory_catalogs
from ia_analysis.correlations.quality import discover_correlation_files, summarize_correlation_quality
from ia_analysis.dynamics.orbit_shape import run_orbit_shape_suite
from ia_analysis.spectra.analysis import (
    discover_spectrum_files,
    load_spectrum_collection,
    relative_to_reference,
)
from ia_analysis.visualization.figure_io import save_figure
from ia_analysis.visualization.pipeline_plots import (
    plot_catalog_inventory,
    plot_correlation_quality,
    plot_orbit_shape_suite,
    plot_spectrum_ratios,
)


def _output_directory(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def analyze_catalog_products(catalog_roots: Any, output_dir: str | Path) -> Any:
    """Inventory global catalogs and write a CSV plus summary figure."""
    output = _output_directory(output_dir)
    frame = inventory_catalogs(catalog_roots)
    frame.to_csv(output / "catalog_inventory.csv", index=False)
    figure, _ = plot_catalog_inventory(frame)
    save_figure(figure, "catalog_inventory", root=output, close=True)
    return frame


def analyze_orbit_shape_suite(
    simulator: Any,
    output_dir: str | Path,
    *,
    initial_shape_tensor: Any,
    m_sub: float,
    **kwargs: Any,
) -> tuple[dict[Any, Any], Any]:
    """Run and visualize the multi-initial-condition orbit-shape experiment."""
    output = _output_directory(output_dir)
    runs, summary = run_orbit_shape_suite(
        simulator,
        initial_shape_tensor=initial_shape_tensor,
        m_sub=m_sub,
        output_dir=output,
        **kwargs,
    )
    figure, _ = plot_orbit_shape_suite(runs)
    save_figure(figure, "multi_initial_condition_orbit_shape_evolution", root=output, close=True)
    return runs, summary


def analyze_spectrum_products(
    spectrum_root: str | Path,
    output_dir: str | Path,
    *,
    samples: Iterable[str],
    spectra: Iterable[str],
    source_group: str | None = None,
    reference: str = "GR",
) -> Any:
    """Read persisted spectra, compute reference ratios, and write products."""
    output = _output_directory(output_dir)
    paths = discover_spectrum_files(spectrum_root)
    frame = load_spectrum_collection(paths, samples=samples, spectra=spectra, source_group=source_group)
    ratios = relative_to_reference(frame, reference=reference) if not frame.empty else frame
    ratios.to_csv(output / "spectrum_reference_ratios.csv", index=False)
    figure, _ = plot_spectrum_ratios(ratios)
    save_figure(figure, "spectrum_reference_ratios", root=output, close=True)
    return ratios


def analyze_correlation_products(correlation_root: str | Path, output_dir: str | Path) -> Any:
    """Inspect correlation covariances and write quality diagnostics."""
    output = _output_directory(output_dir)
    frame = summarize_correlation_quality(discover_correlation_files(correlation_root))
    frame.to_csv(output / "correlation_quality.csv", index=False)
    figure, _ = plot_correlation_quality(frame)
    save_figure(figure, "correlation_quality", root=output, close=True)
    return frame


__all__ = [
    "analyze_catalog_products",
    "analyze_orbit_shape_suite",
    "analyze_spectrum_products",
    "analyze_correlation_products",
]
