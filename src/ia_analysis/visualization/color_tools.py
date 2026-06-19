"""Color palette extraction and indexed-color helpers."""

from __future__ import annotations

from typing import Any

from ia_analysis.visualization._lazy import call_export, load_export

_EXPORTS = {
    "extract_dominant_colors": ("ia_analysis.visualization.arts", "extract_dominant_colors"),
    "get_colors": ("ia_analysis.visualization.arts", "get_colors"),
}

__all__ = ["extract_dominant_colors", "get_colors", "build_indexed_color_list"]


def __getattr__(name: str) -> Any:
    return load_export(_EXPORTS, name)


def build_indexed_color_list(*args: Any, **kwargs: Any) -> Any:
    """Return the historical indexed color list produced by ``get_colors``."""
    return call_export(_EXPORTS, "get_colors", *args, **kwargs)

