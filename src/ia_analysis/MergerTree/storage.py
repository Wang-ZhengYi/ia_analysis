"""Persistence helpers for cross-time merger-tree products.

Purpose
-------
Cross-time TNG products can be expensive to compute because they may load
catalogs, particles, merger trees, and shell analyses across many snapshots.
This module provides small save/load helpers so plotting notebooks can read
precomputed products without triggering new data access.

Provides
--------
- Pickle-based save/load helpers for workflow product dictionaries.
- Parent-directory creation for reproducible output paths.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save_cross_time_products(products: dict[str, Any], path: str | Path) -> Path:
    """Save cross-time products to a pickle file and return the output path."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(products, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path


def load_cross_time_products(path: str | Path) -> dict[str, Any]:
    """Load cross-time products saved by ``save_cross_time_products``."""
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Cross-time products file does not exist: {in_path}")
    with in_path.open("rb") as handle:
        products = pickle.load(handle)
    if not isinstance(products, dict):
        raise TypeError(f"Expected a product dictionary, got {type(products)!r}.")
    return products

