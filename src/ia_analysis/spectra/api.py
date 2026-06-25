"""Structured API for persisted and measured spectrum products."""

from __future__ import annotations

from typing import Any

from ia_analysis._lazy_imports import ExportMap, export_names, load_export

_EXPORTS: ExportMap = {
    "discover_spectrum_files": ("ia_analysis.spectra.analysis", "discover_spectrum_files"),
    "parse_flag_snapshot": ("ia_analysis.spectra.analysis", "parse_flag_snapshot"),
    "list_spectra": ("ia_analysis.spectra.analysis", "list_spectra"),
    "read_spectrum": ("ia_analysis.spectra.analysis", "read_spectrum"),
    "load_spectrum_collection": ("ia_analysis.spectra.analysis", "load_spectrum_collection"),
    "relative_to_reference": ("ia_analysis.spectra.analysis", "relative_to_reference"),
}

__all__ = export_names(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve spectrum-analysis helpers lazily."""
    return load_export(_EXPORTS, name)
