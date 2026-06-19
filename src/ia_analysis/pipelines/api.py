"""Discoverable registry for end-to-end pipeline entrypoints.

Purpose
-------
Pipeline modules are intentionally orchestration-only and should remain thin.
This registry lets notebooks, documentation, and launch scripts discover the
recommended ``python -m`` commands without importing pipeline implementation
modules or hard-coding historical filenames.

Provides
--------
- A typed record for supported pipeline entrypoints.
- Lookup helpers for module names and command tuples.
- Short descriptions for human-readable notebook pipeline sections.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineEntrypoint:
    """Description of one runnable project pipeline."""

    name: str
    module: str
    description: str
    legacy_script: str | None = None

    def command(self, python: str = "python") -> tuple[str, str, str]:
        """Return a ``python -m`` command tuple for this pipeline."""
        return (python, "-m", self.module)


_PIPELINES = {
    "cs-global": PipelineEntrypoint(
        name="cs-global",
        module="ia_analysis.pipelines.run_cs",
        legacy_script="run_cs.py",
        description="Build ClusterSims global shape, tidal, and alignment catalogs.",
    ),
    "tng-global": PipelineEntrypoint(
        name="tng-global",
        module="ia_analysis.pipelines.run_tng",
        legacy_script="run_tng.py",
        description="Build IllustrisTNG global shape, tidal, and alignment catalogs.",
    ),
    "cs-orchestration": PipelineEntrypoint(
        name="cs-orchestration",
        module="ia_analysis.pipelines.global_cs",
        legacy_script="global_cs.py",
        description="Run lower-level ClusterSims orchestration helpers.",
    ),
    "tng-orchestration": PipelineEntrypoint(
        name="tng-orchestration",
        module="ia_analysis.pipelines.global_tng",
        legacy_script="global_tng.py",
        description="Run lower-level TNG orchestration helpers.",
    ),
    "tng-layered": PipelineEntrypoint(
        name="tng-layered",
        module="ia_analysis.pipelines.tng_layered_shape_tide",
        legacy_script=None,
        description="Run layered TNG shape-tide measurements for shell analyses.",
    ),
}

__all__ = [
    "PipelineEntrypoint",
    "list_pipelines",
    "describe_pipelines",
    "get_pipeline",
    "pipeline_module",
    "pipeline_command",
]


def list_pipelines() -> tuple[str, ...]:
    """Return the supported end-to-end pipeline names."""
    return tuple(_PIPELINES)


def describe_pipelines() -> dict[str, str]:
    """Return short descriptions for all registered pipelines."""
    return {name: entry.description for name, entry in _PIPELINES.items()}


def get_pipeline(name: str) -> PipelineEntrypoint:
    """Return one pipeline registry entry by name."""
    key = str(name).strip().lower()
    if key not in _PIPELINES:
        valid = ", ".join(list_pipelines())
        raise KeyError(f"Unknown pipeline {name!r}. Available pipelines: {valid}")
    return _PIPELINES[key]


def pipeline_module(name: str) -> str:
    """Return the importable module path for one pipeline."""
    return get_pipeline(name).module


def pipeline_command(name: str, python: str = "python") -> tuple[str, str, str]:
    """Return the recommended ``python -m`` command tuple for one pipeline."""
    return get_pipeline(name).command(python=python)

