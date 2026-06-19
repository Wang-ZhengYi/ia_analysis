"""Parallel alignment-grid facade and command helper.

Purpose
-------
This module exposes the parallel galaxy-halo alignment grid builder and its CLI
main function under a concise structured module name.

Provides
--------
- Lazy access to the multiprocessing grid-generation implementation.
- A stable package path for batch plotting jobs.
"""


from __future__ import annotations

from typing import Any

from ia_analysis.visualization._lazy import load_export

_EXPORTS = {
    "build_alignment_grids_parallel": (
        "ia_analysis.visualization.plot_GH_alignment_parallel",
        "build_alignment_grids_parallel",
    ),
    "main": ("ia_analysis.visualization.plot_GH_alignment_parallel", "main"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    return load_export(_EXPORTS, name)

