"""Catalog-product discovery and lightweight inventory analysis.

This module contains the reusable, data-only part of the global-catalog
notebook.  It deliberately does not import shape, dynamics, spectra, or
visualization code.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd

_GLOBAL_NAME = re.compile(
    r"(?P<label>.+?)(?:_s|_snap)(?P<snap>\d+)(?:\.[^.]+)?$", re.IGNORECASE
)


def parse_catalog_identity(path: str | Path) -> dict[str, Any]:
    """Infer a model label and snapshot number from a catalog filename."""
    item = Path(path)
    match = _GLOBAL_NAME.match(item.stem)
    if match is None:
        return {"label": item.stem, "snap": np.nan}
    label = match.group("label")
    for prefix in ("global_", "catalog_"):
        if label.lower().startswith(prefix):
            label = label[len(prefix) :]
            break
    return {"label": label, "snap": int(match.group("snap"))}


def _dataset_rows(dataset: h5py.Dataset) -> int:
    """Return the leading row count of one HDF5 dataset."""
    return int(dataset.shape[0]) if dataset.shape else 1


def summarize_catalog_file(path: str | Path) -> dict[str, Any]:
    """Summarize one HDF5 catalog without loading large arrays into memory."""
    item = Path(path)
    identity = parse_catalog_identity(item)
    datasets: list[tuple[str, h5py.Dataset]] = []
    with h5py.File(item, "r") as handle:
        def collect(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                datasets.append((name, obj))

        handle.visititems(collect)
        row_counts = [_dataset_rows(dataset) for _, dataset in datasets]
        attrs = {str(key): value for key, value in handle.attrs.items()}
    return {
        "path": str(item.resolve()),
        "filename": item.name,
        **identity,
        "size_bytes": int(item.stat().st_size),
        "n_datasets": len(datasets),
        "n_rows_max": max(row_counts, default=0),
        "dataset_names": tuple(name for name, _ in datasets),
        "file_attributes": attrs,
    }


def inventory_catalogs(
    roots: str | Path | Iterable[str | Path],
    *,
    patterns: Iterable[str] = ("*.hdf5", "*.h5"),
    recursive: bool = True,
) -> pd.DataFrame:
    """Discover and summarize global-catalog products under one or more roots."""
    if isinstance(roots, (str, Path)):
        roots = (roots,)
    paths: set[Path] = set()
    for root_value in roots:
        root = Path(root_value).expanduser()
        if root.is_file():
            paths.add(root)
            continue
        for pattern in patterns:
            finder = root.rglob if recursive else root.glob
            paths.update(path for path in finder(pattern) if path.is_file())
    rows = [summarize_catalog_file(path) for path in sorted(paths)]
    return pd.DataFrame(rows)


__all__ = ["parse_catalog_identity", "summarize_catalog_file", "inventory_catalogs"]
