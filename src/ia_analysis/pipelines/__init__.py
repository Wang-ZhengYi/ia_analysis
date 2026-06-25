"""End-to-end pipeline namespace.

Purpose
-------
The pipelines package contains command-line and orchestration modules that tie
together catalog loading, shape measurement, tidal fields, and output writing.

Provides
--------
- ClusterSims global catalog generation entrypoints.
- TNG global and layered shape-tide entrypoints.
- Thin orchestration layers that keep low-level math in domain packages.
"""

from __future__ import annotations

from typing import Any

from ia_analysis._lazy_imports import ExportMap, load_export

_EXPORTS: ExportMap = {
    "PipelineEntrypoint": ("ia_analysis.pipelines.api", "PipelineEntrypoint"),
    "list_pipelines": ("ia_analysis.pipelines.api", "list_pipelines"),
    "describe_pipelines": ("ia_analysis.pipelines.api", "describe_pipelines"),
    "get_pipeline": ("ia_analysis.pipelines.api", "get_pipeline"),
    "pipeline_module": ("ia_analysis.pipelines.api", "pipeline_module"),
    "pipeline_command": ("ia_analysis.pipelines.api", "pipeline_command"),
    "analyze_catalog_products": ("ia_analysis.pipelines.api", "analyze_catalog_products"),
    "analyze_orbit_shape_suite": ("ia_analysis.pipelines.api", "analyze_orbit_shape_suite"),
    "analyze_spectrum_products": ("ia_analysis.pipelines.api", "analyze_spectrum_products"),
    "analyze_correlation_products": ("ia_analysis.pipelines.api", "analyze_correlation_products"),
}

__all__ = [
    *list(_EXPORTS),
    "api",
    "global_cs",
    "global_tng",
    "run_cs",
    "run_tng",
    "tng_layered_shape_tide",
    "layered_analysis",
]


def __getattr__(name: str) -> Any:
    """Resolve public pipeline registry helpers lazily."""
    return load_export(_EXPORTS, name)
