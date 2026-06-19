"""Distribution models and fitting helpers used by visualization routines."""

from __future__ import annotations

from typing import Any

from ia_analysis.visualization._lazy import load_export

_EXPORTS = {
    "DimrothWatson": ("ia_analysis.visualization.DWE", "DimrothWatson"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    return load_export(_EXPORTS, name)

