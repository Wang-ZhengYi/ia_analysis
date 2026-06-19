"""Small lazy-export helpers for visualization facade modules."""

from __future__ import annotations

from importlib import import_module
from typing import Any


ExportMap = dict[str, tuple[str, str]]


def load_export(exports: ExportMap, name: str) -> Any:
    """Load one exported attribute from its legacy implementation module."""
    if name not in exports:
        raise AttributeError(name)
    module_name, attr_name = exports[name]
    module = import_module(module_name)
    return getattr(module, attr_name)


def call_export(exports: ExportMap, name: str, *args: Any, **kwargs: Any) -> Any:
    """Load and call one exported function."""
    return load_export(exports, name)(*args, **kwargs)

