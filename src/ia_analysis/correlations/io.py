"""HDF5 IO for real-space correlation products.

Purpose
-------
The correlations package produces nested Python objects for two-point,
four-point, category, and covariance products.  This module writes those
objects to a stable HDF5 layout for downstream plotting and covariance analysis.

Provides
--------
- HDF5 writer for ``CorrelationSuiteResult``.
- Small metadata serialization helpers for scalar and array attributes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _write_metadata(group: Any, metadata: Mapping[str, Any]) -> None:
    """Write scalar metadata as attrs and array metadata as datasets."""
    import h5py

    for key, value in dict(metadata).items():
        if value is None:
            continue
        if isinstance(value, (str, bytes, int, float, bool, np.integer, np.floating, np.bool_)):
            group.attrs[str(key)] = value
            continue
        arr = np.asarray(value)
        if arr.dtype.kind in {"U", "O"}:
            arr = arr.astype(h5py.string_dtype(encoding="utf-8"))
        group.create_dataset(str(key), data=arr)


def write_results_hdf5(
    out_path: str | Path,
    suite_result: Any,
    *,
    overwrite: bool = True,
) -> Path:
    """Write a correlation suite result to HDF5."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    import h5py

    with h5py.File(path, mode) as h5:
        meta = h5.create_group("meta")
        _write_metadata(meta, getattr(suite_result, "metadata", {}))
        stats = h5.create_group("statistics")
        for name, result in suite_result.results.items():
            g = stats.create_group(str(name))
            g.create_dataset("rbins", data=np.asarray(result.rbins))
            g.create_dataset("rmid", data=np.asarray(result.rmid))
            _write_metadata(g.create_group("metadata"), result.metadata)
            cats = g.create_group("categories")
            for category, values in result.values.items():
                cg = cats.create_group(str(category))
                cg.create_dataset("value", data=np.asarray(values))
                cg.create_dataset("count", data=np.asarray(result.counts[category]))
                cg.create_dataset("weight_sum", data=np.asarray(result.weight_sums[category]))
                if name in suite_result.covariance and category in suite_result.covariance[name]:
                    cov_entry = suite_result.covariance[name][category]
                    cg.create_dataset("cov", data=np.asarray(cov_entry["cov"]))
                    cg.create_dataset("cov_mean", data=np.asarray(cov_entry["mean"]))
                    if "samples" in cov_entry:
                        cg.create_dataset("cov_samples", data=np.asarray(cov_entry["samples"]))
    return path


__all__ = ["write_results_hdf5"]
