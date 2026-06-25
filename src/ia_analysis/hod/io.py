"""HDF5 and DataFrame serialization for HOD and IA component measurements."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from ia_analysis.hod.ia_measurements import IAComponentMeasurement
from ia_analysis.hod.statistics import HODMeasurement


def _write_value(group: h5py.Group, name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, dict):
        group.attrs[name] = json.dumps(value, default=str)
    elif isinstance(value, str):
        group.attrs[name] = value
    elif isinstance(value, tuple) and all(isinstance(item, str) for item in value):
        group.create_dataset(name, data=np.asarray(value, dtype=h5py.string_dtype("utf-8")))
    else:
        group.create_dataset(name, data=np.asarray(value))


def save_hod_measurement_hdf5(measurement: HODMeasurement, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["type"] = "HODMeasurement"
        for name, value in asdict(measurement).items():
            _write_value(handle, name, value)
    return path


def load_hod_measurement_hdf5(path: str | Path) -> HODMeasurement:
    with h5py.File(path, "r") as handle:
        data = {name: np.asarray(handle[name]) for name in HODMeasurement.__dataclass_fields__ if name in handle}
        data["sample_label"] = handle.attrs.get("sample_label")
        data["metadata"] = json.loads(handle.attrs.get("metadata", "{}"))
    return HODMeasurement(**data)


def save_ia_component_hdf5(measurement: IAComponentMeasurement, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["type"] = "IAComponentMeasurement"
        for name, value in asdict(measurement).items():
            _write_value(handle, name, value)
    return path


def load_ia_component_hdf5(path: str | Path) -> IAComponentMeasurement:
    with h5py.File(path, "r") as handle:
        data = {}
        for name in IAComponentMeasurement.__dataclass_fields__:
            if name in handle:
                value = np.asarray(handle[name])
                if name == "layer_labels":
                    value = tuple(item.decode() if isinstance(item, bytes) else str(item) for item in value)
                data[name] = value
            elif name in handle.attrs:
                data[name] = handle.attrs[name]
            else:
                data[name] = None
        data["metadata"] = json.loads(handle.attrs.get("metadata", "{}"))
    return IAComponentMeasurement(**data)


def measurement_to_dataframe(measurement: HODMeasurement | IAComponentMeasurement) -> pd.DataFrame:
    """Convert one-dimensional measurements to a tidy DataFrame."""
    if isinstance(measurement, HODMeasurement):
        return pd.DataFrame({
            "mass_center": measurement.mass_centers,
            "n_halo": measurement.n_halo,
            "mean_cen": measurement.mean_cen,
            "mean_sat": measurement.mean_sat,
            "mean_tot": measurement.mean_tot,
            "var_tot": measurement.var_tot,
        })
    values = np.asarray(measurement.values)
    return pd.DataFrame({"index": np.arange(values.size), "value": values.ravel()})


def measurement_from_dataframe(frame: pd.DataFrame, *, kind: str, **metadata: Any):
    """Rebuild a basic HOD or one-dimensional IA measurement from a DataFrame."""
    if kind == "hod":
        centers = frame["mass_center"].to_numpy()
        edges = np.empty(centers.size + 1)
        edges[1:-1] = np.sqrt(centers[:-1] * centers[1:])
        edges[0], edges[-1] = centers[0] ** 2 / edges[1], centers[-1] ** 2 / edges[-2]
        zeros = np.zeros(centers.size, dtype=int)

        def column(name: str, default: np.ndarray) -> np.ndarray:
            return frame[name].to_numpy() if name in frame else default.copy()

        return HODMeasurement(
            edges, centers, column("n_halo", zeros), zeros, zeros, zeros,
            frame["mean_cen"].to_numpy(), frame["mean_sat"].to_numpy(),
            frame["mean_tot"].to_numpy(), frame["var_tot"].to_numpy(), metadata=metadata,
        )
    if kind == "ia":
        return IAComponentMeasurement(
            metadata.pop("component", "component"), metadata.pop("reference", "custom"),
            metadata.pop("population", "all"), metadata.pop("sample_label", None),
            None, None, None, None, frame["value"].to_numpy(), metadata=metadata,
        )
    raise ValueError("kind must be 'hod' or 'ia'")


__all__ = [
    "save_hod_measurement_hdf5", "load_hod_measurement_hdf5",
    "save_ia_component_hdf5", "load_ia_component_hdf5",
    "measurement_to_dataframe", "measurement_from_dataframe",
]
