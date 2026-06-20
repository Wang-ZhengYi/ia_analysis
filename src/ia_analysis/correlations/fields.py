"""Catalog and field helpers for real-space correlations.

Purpose
-------
The correlation estimators work with object positions plus named scalar,
vector, or tensor fields.  This module keeps validation and projection rules in
one place so the pair-counting code can stay focused on bin accumulation.

Provides
--------
- ``CorrelationCatalog`` for positions, object fields, halo IDs, sample classes,
  and optional weights.
- ``PairSpec`` for naming one measured field pair.
- Field-kind inference and radial projection helpers used by the estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

FIELD_KIND_SCALAR = "scalar"
FIELD_KIND_VECTOR = "vector"
FIELD_KIND_TENSOR = "tensor"
VALID_FIELD_KINDS = {FIELD_KIND_SCALAR, FIELD_KIND_VECTOR, FIELD_KIND_TENSOR}


@dataclass(frozen=True)
class PairSpec:
    """Describe one two-point field correlation to measure."""

    left: str
    right: str
    name: str | None = None
    left_kind: str | None = None
    right_kind: str | None = None
    vector_vector_mode: str = "dot"
    symmetrize: bool = True

    def output_name(self) -> str:
        """Return a stable ASCII name for this field pair."""
        if self.name:
            return str(self.name)
        return f"{self.left}{self.right}".replace("omega", "omega")


@dataclass
class CorrelationCatalog:
    """Object-level catalog used by the real-space correlation estimators."""

    positions: np.ndarray
    fields: Mapping[str, Any]
    host_id: Any | None = None
    sample_type: Any | None = None
    weights: Any | None = None
    boxsize: float | Sequence[float] | None = None
    name: str = "catalog"

    def __post_init__(self) -> None:
        """Validate array lengths and coerce numeric fields."""
        positions = np.asarray(self.positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("`positions` must have shape (N, 3)")
        object.__setattr__(self, "positions", positions)

        nobj = positions.shape[0]
        coerced: dict[str, np.ndarray] = {}
        for key, value in dict(self.fields).items():
            arr = np.asarray(value, dtype=float)
            if arr.shape[0] != nobj:
                raise ValueError(f"Field {key!r} has length {arr.shape[0]}, expected {nobj}")
            coerced[str(key)] = arr
        object.__setattr__(self, "fields", coerced)

        if self.host_id is not None:
            host = np.asarray(self.host_id)
            if host.shape[0] != nobj:
                raise ValueError("`host_id` must have length N")
            object.__setattr__(self, "host_id", host)

        if self.sample_type is not None:
            sample_type = np.asarray(self.sample_type)
            if sample_type.shape[0] != nobj:
                raise ValueError("`sample_type` must have length N")
            object.__setattr__(self, "sample_type", sample_type)

        if self.weights is None:
            weights = np.ones(nobj, dtype=float)
        else:
            weights = np.asarray(self.weights, dtype=float)
            if weights.shape[0] != nobj:
                raise ValueError("`weights` must have length N")
        object.__setattr__(self, "weights", weights)

        if self.boxsize is not None:
            box = np.asarray(self.boxsize, dtype=float)
            if box.ndim == 0:
                box = np.repeat(float(box), 3)
            if box.shape != (3,):
                raise ValueError("`boxsize` must be a scalar or length-3 sequence")
            object.__setattr__(self, "boxsize", box)

    @property
    def size(self) -> int:
        """Return the number of objects in the catalog."""
        return int(self.positions.shape[0])

    def field(self, name: str) -> np.ndarray:
        """Return one named field, using ones for the conventional density field."""
        key = str(name)
        if key in self.fields:
            return np.asarray(self.fields[key])
        if key == "d":
            return np.ones(self.size, dtype=float)
        raise KeyError(f"Field {name!r} is not present in catalog {self.name!r}")

    def subset(self, mask: Sequence[bool]) -> "CorrelationCatalog":
        """Return a new catalog containing only selected objects."""
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape[0] != self.size:
            raise ValueError("Subset mask must have length N")
        return CorrelationCatalog(
            positions=self.positions[mask_arr],
            fields={name: values[mask_arr] for name, values in self.fields.items()},
            host_id=None if self.host_id is None else self.host_id[mask_arr],
            sample_type=None if self.sample_type is None else self.sample_type[mask_arr],
            weights=self.weights[mask_arr],
            boxsize=self.boxsize,
            name=self.name,
        )


def infer_field_kind(values: np.ndarray, explicit: str | None = None) -> str:
    """Infer whether a field is scalar, vector, or tensor."""
    if explicit is not None:
        kind = str(explicit).strip().lower()
        if kind not in VALID_FIELD_KINDS:
            raise ValueError(f"Unknown field kind {explicit!r}")
        return kind

    arr = np.asarray(values)
    if arr.ndim == 1:
        return FIELD_KIND_SCALAR
    if arr.ndim == 2 and arr.shape[1] == 1:
        return FIELD_KIND_SCALAR
    if arr.ndim == 2 and arr.shape[1] == 3:
        return FIELD_KIND_VECTOR
    if arr.ndim == 3 and arr.shape[1:] == (3, 3):
        return FIELD_KIND_TENSOR
    raise ValueError(f"Cannot infer field kind from shape {arr.shape}")


def as_scalar_field(values: np.ndarray) -> np.ndarray:
    """Return a scalar field as a one-dimensional array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError("Scalar fields must have shape (N,) or (N, 1)")
    return arr


def radial_projection(values: np.ndarray, kind: str, rhat: np.ndarray) -> np.ndarray:
    """Project scalar, vector, or tensor field values onto the pair direction."""
    kind = infer_field_kind(values, kind)
    arr = np.asarray(values, dtype=float)
    if kind == FIELD_KIND_SCALAR:
        return as_scalar_field(arr)
    if kind == FIELD_KIND_VECTOR:
        return np.einsum("ij,ij->i", arr, rhat)
    if kind == FIELD_KIND_TENSOR:
        return np.einsum("ni,nij,nj->n", rhat, arr, rhat)
    raise ValueError(f"Unknown field kind {kind!r}")


def pair_signal(
    left_values: np.ndarray,
    right_values: np.ndarray,
    rhat: np.ndarray,
    *,
    left_kind: str | None = None,
    right_kind: str | None = None,
    vector_vector_mode: str = "dot",
) -> np.ndarray:
    """Evaluate the pair signal for one field pair.

    Scalar-scalar pairs use a direct product.  Scalar-vector and scalar-tensor
    pairs use the radial projection of the non-scalar field.  Vector-vector
    pairs use a dot product by default, while tensor-containing pairs use radial
    projections.  This gives a compact estimator for ``dd``, ``ed``, ``ee``,
    ``dv``, ``ev``, ``vv``, and omega-field correlations without imposing a
    particular catalog schema.
    """
    left_arr = np.asarray(left_values, dtype=float)
    right_arr = np.asarray(right_values, dtype=float)
    lk = infer_field_kind(left_arr, left_kind)
    rk = infer_field_kind(right_arr, right_kind)

    if lk == FIELD_KIND_SCALAR and rk == FIELD_KIND_SCALAR:
        return as_scalar_field(left_arr) * as_scalar_field(right_arr)

    if lk == FIELD_KIND_VECTOR and rk == FIELD_KIND_VECTOR and vector_vector_mode == "dot":
        return np.einsum("ij,ij->i", left_arr, right_arr)

    left_proj = radial_projection(left_arr, lk, rhat)
    right_proj = radial_projection(right_arr, rk, rhat)
    return left_proj * right_proj


def normalize_sample_class(values: np.ndarray | None) -> np.ndarray | None:
    """Map sample labels to ``c`` for centrals and ``s`` for satellites."""
    if values is None:
        return None
    arr = np.asarray(values)
    out = np.full(arr.shape[0], "", dtype=object)
    for i, value in enumerate(arr):
        if isinstance(value, bytes):
            text = value.decode(errors="ignore").strip().lower()
        else:
            text = str(value).strip().lower()
        if text in {"central", "centre", "center", "cen", "c", "0", "true"}:
            out[i] = "c"
        elif text in {"satellite", "sat", "s", "1", "false"}:
            out[i] = "s"
        else:
            out[i] = text
    return out


def default_pair_specs(include_omega: bool = True) -> tuple[PairSpec, ...]:
    """Return the standard field-pair suite requested for IA correlations."""
    specs = [
        PairSpec("e", "e", "ee"),
        PairSpec("e", "d", "ed"),
        PairSpec("d", "d", "dd"),
        PairSpec("v", "v", "vv"),
        PairSpec("d", "v", "dv"),
        PairSpec("e", "v", "ev"),
    ]
    if include_omega:
        specs.extend(
            [
                PairSpec("omega", "e", "omegae"),
                PairSpec("omega", "d", "omegad"),
                PairSpec("omega", "v", "omegav"),
            ]
        )
    return tuple(specs)


__all__ = [
    "FIELD_KIND_SCALAR",
    "FIELD_KIND_VECTOR",
    "FIELD_KIND_TENSOR",
    "VALID_FIELD_KINDS",
    "PairSpec",
    "CorrelationCatalog",
    "infer_field_kind",
    "as_scalar_field",
    "radial_projection",
    "pair_signal",
    "normalize_sample_class",
    "default_pair_specs",
]
