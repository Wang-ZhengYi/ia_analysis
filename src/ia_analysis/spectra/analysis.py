"""Discovery, reading, and comparison of persisted power-spectrum products."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import pandas as pd

_FLAG_SNAP = re.compile(r"(?P<flag>GR|F\d+).*?(?:s|snap)?(?P<snap>\d{1,3})(?:\D|$)", re.IGNORECASE)


def discover_spectrum_files(
    root: str | Path,
    *,
    patterns: Iterable[str] = ("*.hdf5", "*.h5"),
    recursive: bool = True,
) -> list[Path]:
    """Return sorted HDF5 spectrum products under a directory."""
    root = Path(root).expanduser()
    files: set[Path] = set()
    for pattern in patterns:
        finder = root.rglob if recursive else root.glob
        files.update(path for path in finder(pattern) if path.is_file())
    return sorted(files)


def parse_flag_snapshot(path: str | Path) -> tuple[str | None, int | None]:
    """Infer the gravity-model flag and snapshot from a spectrum filename."""
    match = _FLAG_SNAP.search(Path(path).stem)
    if match is None:
        return None, None
    return match.group("flag").upper(), int(match.group("snap"))


def _resolve_group(handle: h5py.File, sample: str, source_group: str | None) -> h5py.Group:
    candidates = []
    if source_group:
        candidates.extend((f"{source_group}/{sample}", source_group))
    candidates.extend((sample, "/"))
    for candidate in candidates:
        if candidate == "/":
            return handle
        if candidate in handle and isinstance(handle[candidate], h5py.Group):
            return handle[candidate]
    raise KeyError(f"Could not find sample {sample!r} in {handle.filename}")


def list_spectra(path: str | Path, sample: str = "all", *, source_group: str | None = None) -> tuple[str, ...]:
    """List one-dimensional datasets that can be read as spectra."""
    with h5py.File(path, "r") as handle:
        group = _resolve_group(handle, sample, source_group)
        return tuple(name for name, value in group.items() if isinstance(value, h5py.Dataset) and value.ndim == 1)


def read_spectrum(
    path: str | Path,
    sample: str,
    spectrum: str,
    *,
    source_group: str | None = None,
    k_candidates: Sequence[str] = ("k", "k3D", "k_centers"),
) -> pd.DataFrame:
    """Read one spectrum into a tidy table with file/model metadata."""
    with h5py.File(path, "r") as handle:
        group = _resolve_group(handle, sample, source_group)
        candidates = (spectrum, spectrum.replace("P_", ""), f"Pk_{spectrum}", f"P_{spectrum}")
        dataset_name = next((name for name in candidates if name in group), None)
        if dataset_name is None:
            raise KeyError(f"Spectrum {spectrum!r} is not present in {path}")
        power = np.asarray(group[dataset_name], dtype=float).reshape(-1)
        k_name = next((name for name in k_candidates if name in group), None)
        if k_name is None:
            raise KeyError(f"No k-axis dataset found in group {group.name}")
        wave_number = np.asarray(group[k_name], dtype=float).reshape(-1)
        if wave_number.size != power.size:
            raise ValueError("k and spectrum arrays have different lengths")
        sigma = None
        for name in (f"{dataset_name}_sigma", f"sigma_{dataset_name}", f"{spectrum}_sigma"):
            if name in group:
                sigma = np.asarray(group[name], dtype=float).reshape(-1)
                break
    flag, snap = parse_flag_snapshot(path)
    frame = pd.DataFrame({"k": wave_number, "P": power})
    frame["sigma"] = np.nan if sigma is None else sigma
    frame["sample"] = sample
    frame["spectrum"] = spectrum
    frame["flag"] = flag
    frame["snap"] = snap
    frame["path"] = str(Path(path).resolve())
    return frame


def load_spectrum_collection(
    paths: Iterable[str | Path],
    *,
    samples: Iterable[str],
    spectra: Iterable[str],
    source_group: str | None = None,
    skip_missing: bool = True,
) -> pd.DataFrame:
    """Read a grid of files, samples, and spectra into one tidy table."""
    frames: list[pd.DataFrame] = []
    for path in paths:
        for sample in samples:
            for spectrum in spectra:
                try:
                    frames.append(read_spectrum(path, sample, spectrum, source_group=source_group))
                except (KeyError, ValueError):
                    if not skip_missing:
                        raise
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def relative_to_reference(
    frame: pd.DataFrame,
    *,
    reference: str = "GR",
    group_columns: Iterable[str] = ("sample", "spectrum", "snap"),
    value_column: str = "P",
    x_column: str = "k",
) -> pd.DataFrame:
    """Add model/reference ratios after interpolation onto each model k-grid."""
    output: list[pd.DataFrame] = []
    groups = [column for column in group_columns if column in frame]
    for _, panel in frame.groupby(groups, dropna=False):
        reference_rows = panel.loc[panel["flag"] == reference].sort_values(x_column)
        if reference_rows.empty:
            continue
        for flag, model_rows in panel.groupby("flag", dropna=False):
            model_rows = model_rows.sort_values(x_column).copy()
            reference_values = np.interp(
                model_rows[x_column],
                reference_rows[x_column],
                reference_rows[value_column],
                left=np.nan,
                right=np.nan,
            )
            values = model_rows[value_column].to_numpy(dtype=float)
            ratio = np.full(values.shape, np.nan)
            valid = np.isfinite(values) & np.isfinite(reference_values) & (reference_values != 0.0)
            ratio[valid] = values[valid] / reference_values[valid]
            model_rows["reference_flag"] = reference
            model_rows["ratio_to_reference"] = ratio
            model_rows["fractional_difference"] = ratio - 1.0
            output.append(model_rows)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


__all__ = [
    "discover_spectrum_files",
    "parse_flag_snapshot",
    "list_spectra",
    "read_spectrum",
    "load_spectrum_collection",
    "relative_to_reference",
]
