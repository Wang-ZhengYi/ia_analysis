"""Shape measurements for one target in one snapshot."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ia_analysis.shapes import api as shape_api


def measure_target_shape(
    positions: np.ndarray,
    masses: Optional[np.ndarray] = None,
    *,
    center: Optional[np.ndarray] = None,
    velocities: Optional[np.ndarray] = None,
    accelerations: Optional[np.ndarray] = None,
    r_ell: Optional[float] = None,
    percentile: float = 100.0,
    max_iter: int = 100,
    tol: float = 0.01,
    tensor_mode: str = "reduced",
    return_mask: bool = True,
) -> dict[str, Any]:
    """
    Measure an iterative shape tensor for target particles.

    Positions may be absolute snapshot coordinates; pass ``center`` to make
    them relative before measuring the tensor.
    """
    pos = np.asarray(positions, dtype=np.float64)
    rel_pos = pos if center is None else pos - np.asarray(center, dtype=np.float64)[None, :]

    result = shape_api.I_iters(
        rel_pos,
        masses=masses,
        velocities=velocities,
        accelerations=accelerations,
        r_ell=r_ell,
        percentile=percentile,
        max_iter=max_iter,
        tol=tol,
        return_dI=velocities is not None,
        return_ddI=velocities is not None and accelerations is not None,
        return_mask=return_mask,
        return_converged=True,
        tensor_mode=tensor_mode,
    )
    if not isinstance(result, tuple):
        result = (result,)

    idx = 0
    tensor = result[idx]
    idx += 1
    d_tensor = None
    dd_tensor = None
    if velocities is not None:
        d_tensor = result[idx]
        idx += 1
    if velocities is not None and accelerations is not None:
        dd_tensor = result[idx]
        idx += 1
    mask = result[idx] if return_mask else None
    if return_mask:
        idx += 1
    converged = bool(result[idx])

    axes, vectors = shape_api.compute_axis(tensor)
    return {
        "tensor": tensor,
        "d_tensor": d_tensor,
        "dd_tensor": dd_tensor,
        "axes": axes,
        "vectors": vectors,
        "mask": mask,
        "converged": converged,
        "center": None if center is None else np.asarray(center, dtype=np.float64),
        "tensor_mode": tensor_mode,
    }


def measure_target_principal_axes(tensor: np.ndarray) -> dict[str, Any]:
    """Convert a shape tensor into axis lengths and principal vectors."""
    return shape_api.measure_principal_axes(tensor)


__all__ = ["measure_target_shape", "measure_target_principal_axes"]
