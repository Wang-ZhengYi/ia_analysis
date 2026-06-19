"""Color extraction facade for visualization workflows.

Purpose
-------
This module provides a small, named entrypoint for palette extraction and
indexed color-list construction used across figure notebooks.

Provides
--------
- Lazy access to dominant-color extraction.
- Lazy access to the historical indexed color helper.
- A clearer alias named ``build_indexed_color_list`` for new code.
"""


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

