"""Helpers for legacy root-level compatibility wrappers."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_src_path() -> None:
    """Make the local src-layout package importable when running wrappers in-place."""
    src = Path(__file__).resolve().parent / "src"
    src_text = str(src)
    if src.is_dir() and src_text not in sys.path:
        sys.path.insert(0, src_text)

