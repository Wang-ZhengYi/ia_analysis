"""Distribution-fitting facade for visualization diagnostics.

Purpose
-------
This module exposes probability distribution models used by alignment and angle
histogram plots without importing the heavier plotting stack at package import
time.

Provides
--------
- Lazy access to the enhanced Dimroth-Watson distribution class.
- A stable import path for future distribution models used by figures.
"""


from __future__ import annotations

from typing import Any

from ia_analysis.visualization._lazy import load_export

_EXPORTS = {
    "DimrothWatson": ("ia_analysis.visualization.DWE", "DimrothWatson"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    return load_export(_EXPORTS, name)

