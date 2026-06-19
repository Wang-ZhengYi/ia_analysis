"""Alignment catalog loading, field access, and population masks."""

from __future__ import annotations

from typing import Any

from ia_analysis.visualization._lazy import call_export, load_export

_EXPORTS = {
    "set_alignment_context": ("ia_analysis.visualization.arts_IA", "set_alignment_context"),
    "available_flags_for_snap": ("ia_analysis.visualization.arts_IA", "available_flags_for_snap"),
    "load_alignment_maset": ("ia_analysis.visualization.arts_IA", "load_alignment_maset"),
    "load_marenew_pickle": ("ia_analysis.visualization.arts_IA", "load_marenew_pickle"),
    "has_field": ("ia_analysis.visualization.arts_IA", "has_field"),
    "get_field": ("ia_analysis.visualization.arts_IA", "get_field"),
    "maybe_field": ("ia_analysis.visualization.arts_IA", "maybe_field"),
    "safe_log10": ("ia_analysis.visualization.arts_IA", "safe_log10"),
    "mask_population": ("ia_analysis.visualization.arts_IA", "mask_population"),
    "apply_range_mask": ("ia_analysis.visualization.arts_IA", "apply_range_mask"),
    "flag_label": ("ia_analysis.visualization.arts_IA", "flag_label"),
    "flag_color": ("ia_analysis.visualization.arts_IA", "flag_color"),
}

__all__ = [
    *list(_EXPORTS),
    "configure_alignment_context",
    "load_alignment_catalogs",
    "load_legacy_alignment_pickle",
]


def __getattr__(name: str) -> Any:
    return load_export(_EXPORTS, name)


def configure_alignment_context(*args: Any, **kwargs: Any) -> Any:
    """Configure the global alignment plotting context."""
    return call_export(_EXPORTS, "set_alignment_context", *args, **kwargs)


def load_alignment_catalogs(*args: Any, **kwargs: Any) -> Any:
    """Load MAset-style alignment catalogs."""
    return call_export(_EXPORTS, "load_alignment_maset", *args, **kwargs)


def load_legacy_alignment_pickle(*args: Any, **kwargs: Any) -> Any:
    """Load the historical MArenew pickle format."""
    return call_export(_EXPORTS, "load_marenew_pickle", *args, **kwargs)

