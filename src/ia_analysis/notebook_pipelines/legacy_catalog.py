"""Safely discover functions preserved in historical notebook exports.

Exported notebook scripts mix useful definitions with top-level data loading
and cluster-specific state. This module parses them without importing them.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


EXPORT_DIR = Path(__file__).resolve().parent / "exports"
PLOT_PREFIXES = (
    "configure_",
    "despine",
    "draw",
    "maybe_close",
    "plot",
    "save_fig",
    "savefig",
    "set_",
    "setup_plot",
)


@dataclass(frozen=True)
class LegacyDefinition:
    """One top-level function or class found in an exported notebook."""

    export: str
    name: str
    kind: str
    category: str
    lineno: int
    end_lineno: int


def _export_path(export: str) -> Path:
    name = export if export.endswith(".py") else f"{export}.py"
    path = EXPORT_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Unknown notebook export: {name}")
    return path


def _category(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith(PLOT_PREFIXES) or any(
        token in lowered
        for token in (
            "axis_style",
            "colorbar",
            "figure",
            "legend",
            "plot",
            "savefig",
            "subplot_title",
        )
    ):
        return "plotting"
    return "pipeline"


def definitions(exports: Iterable[str] | None = None) -> tuple[LegacyDefinition, ...]:
    """Return top-level definitions without importing notebook exports."""
    paths = [_export_path(export) for export in exports] if exports is not None else sorted(EXPORT_DIR.glob("*_nb.py"))
    found: list[LegacyDefinition] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            found.append(
                LegacyDefinition(
                    export=path.name,
                    name=node.name,
                    kind="class" if isinstance(node, ast.ClassDef) else "function",
                    category=_category(node.name),
                    lineno=node.lineno,
                    end_lineno=node.end_lineno or node.lineno,
                )
            )
    return tuple(found)


def source(export: str, name: str, occurrence: int = 1) -> str:
    """Return source for one legacy definition without executing its export."""
    if occurrence < 1:
        raise ValueError("occurrence must be at least 1")
    path = _export_path(export)
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name
    ]
    if len(matches) < occurrence:
        raise KeyError(f"{name!r} occurrence {occurrence} not found in {path.name}")
    segment = ast.get_source_segment(text, matches[occurrence - 1])
    if segment is None:  # pragma: no cover
        node = matches[occurrence - 1]
        segment = "\n".join(text.splitlines()[node.lineno - 1 : node.end_lineno])
    return segment


def manifest(exports: Iterable[str]) -> dict[str, tuple[LegacyDefinition, ...]]:
    """Group definitions into pipeline and plotting entries."""
    items = definitions(exports)
    return {
        "pipeline": tuple(item for item in items if item.category == "pipeline"),
        "plotting": tuple(item for item in items if item.category == "plotting"),
    }


def search(query: str, exports: Iterable[str] | None = None) -> tuple[LegacyDefinition, ...]:
    """Search definition names case-insensitively."""
    needle = query.casefold()
    return tuple(item for item in definitions(exports) if needle in item.name.casefold())


__all__ = ["LegacyDefinition", "definitions", "manifest", "search", "source"]
