"""Small internal helpers for lightweight public API facades.

Purpose
-------
This module centralizes lazy-import behavior used by subpackage API facades.
It keeps top-level package imports cheap while still giving users short,
stable names for functions that live in historical implementation modules.

Provides
--------
- Attribute loading from a ``{public_name: (module, attribute)}`` export map.
- Function dispatch helpers used by thin wrapper functions.
- Export-list helpers for documentation, tests, and interactive notebooks.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping

ExportMap = Mapping[str, tuple[str, str]]


def load_export(exports: ExportMap, name: str) -> Any:
    """Load one public attribute from a lazy export map."""
    try:
        module_name, attr_name = exports[name]
    except KeyError as exc:
        valid = ", ".join(sorted(exports))
        raise AttributeError(f"Unknown API export {name!r}. Available exports: {valid}") from exc

    module = import_module(module_name)
    return getattr(module, attr_name)


def call_export(exports: ExportMap, name: str, *args: Any, **kwargs: Any) -> Any:
    """Load a callable export and invoke it with the supplied arguments."""
    return load_export(exports, name)(*args, **kwargs)


def export_names(exports: ExportMap) -> tuple[str, ...]:
    """Return the stable public names provided by an export map."""
    return tuple(exports)

