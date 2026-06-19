"""Self-foldable momentum-divergence meshes and cross spectra.

The historical velocity-divergence mesh in ``CatMesh``/``SnapMesh`` is built as
``v(x) = p(x) / rho(x)`` followed by ``theta = -div(v)/(aH)``.  The division by
the folded density field is local and nonlinear, so it is not an exact
self-folding operation.

This module provides a linear, self-foldable alternative based on the momentum
density field ``q(x) = (1 + delta_w) u(x)``.  Its divergence is useful for
folded cross spectra with additive fields such as matter density ``d``, galaxy
density ``g``, or IA ``E``.  It should be treated as a momentum-divergence
proxy, not as the same estimator as the local velocity divergence ``theta``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


def velocity_divergence_self_folding_status() -> Dict[str, str]:
    """Return the project decision on velocity-divergence self-folding."""
    return {
        "status": "not_exact_for_local_velocity_divergence",
        "reason": (
            "The existing theta estimator forms v(x)=p(x)/rho(x) after painting. "
            "That cell-wise ratio is nonlinear, so folding positions before the "
            "ratio mixes streams and does not reproduce the unfolded high-k "
            "velocity field exactly."
        ),
        "recommended_field": "theta_momentum_mesh",
        "recommended_method": (
            "Paint the additive momentum density components on the folded mesh, "
            "take their spectral divergence, and cross-correlate that field with "
            "other additive folded fields such as d, g, and E."
        ),
    }


def _require_mas_library():
    try:
        import MAS_library as MASL  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional HPC stack
        raise ImportError(
            "build_momentum_divergence_mesh requires MAS_library from Pylians3."
        ) from exc
    return MASL


def _validate_particle_arrays(
    positions: np.ndarray,
    velocities: np.ndarray,
    weights: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.asarray(positions, dtype=np.float64)
    vel = np.asarray(velocities, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3).")
    if vel.shape != pos.shape:
        raise ValueError("velocities must have shape (N, 3), matching positions.")

    if weights is None:
        w = np.ones(pos.shape[0], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (pos.shape[0],):
            raise ValueError("weights must have shape (N,), matching positions.")

    finite = np.isfinite(pos).all(axis=1) & np.isfinite(vel).all(axis=1) & np.isfinite(w)
    if not np.all(finite):
        pos = pos[finite]
        vel = vel[finite]
        w = w[finite]

    return pos, vel, w


def _cosmo_h(cosmo: Any) -> float:
    try:
        return float(cosmo["h"])
    except Exception:
        return float(getattr(cosmo, "h"))


def _Hz_kms_per_Mpc(cosmo: Any, z: float) -> float:
    """Evaluate H(z) in km/s/Mpc using the project pyccl convention."""
    try:
        import pyccl as ccl  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional HPC stack
        raise ImportError("pyccl is required when cosmo and z are provided.") from exc

    a = 1.0 / (1.0 + float(z))
    Ez = float(ccl.h_over_h0(cosmo, a))
    return 100.0 * _cosmo_h(cosmo) * Ez


def _paint_weighted(
    positions: np.ndarray,
    weights: np.ndarray,
    *,
    boxsize: float,
    nmesh: int,
    mas: str,
) -> np.ndarray:
    MASL = _require_mas_library()
    mesh = np.zeros((int(nmesh), int(nmesh), int(nmesh)), dtype=np.float32)
    MASL.MA(
        positions.astype(np.float32, copy=False),
        mesh,
        float(boxsize),
        str(mas),
        W=weights.astype(np.float32, copy=False),
    )
    return mesh


def _spectral_divergence(
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    *,
    boxsize: float,
) -> np.ndarray:
    nmesh = int(qx.shape[0])
    dx = float(boxsize) / float(nmesh)
    k1 = 2.0 * np.pi * np.fft.fftfreq(nmesh, d=dx)

    qx_k = np.fft.fftn(qx)
    qy_k = np.fft.fftn(qy)
    qz_k = np.fft.fftn(qz)
    div_k = 1j * (
        k1[:, None, None] * qx_k
        + k1[None, :, None] * qy_k
        + k1[None, None, :] * qz_k
    )
    return np.fft.ifftn(div_k).real.astype(np.float32)


def build_momentum_divergence_mesh(
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    boxsize: float,
    nmesh: int,
    fold: int = 1,
    z: float | None = None,
    cosmo: Any | None = None,
    weights: np.ndarray | None = None,
    mas: str = "CIC",
    normalize_by_mean_weight: bool = True,
    subtract_bulk_velocity: bool = True,
) -> Dict[str, Any]:
    """Build a folded, self-foldable momentum-divergence field.

    Parameters
    ----------
    positions, velocities
        Arrays with shape ``(N, 3)``. Positions are in Mpc/h and velocities in
        km/s, matching the existing mesh builders.
    boxsize
        Original simulation box size in Mpc/h.
    nmesh
        Number of cells per folded-box side.
    fold
        Self-folding factor. Positions are wrapped into a box of side
        ``boxsize / fold``.
    z, cosmo
        When both are supplied, the returned ``theta_momentum_mesh`` is
        normalized by ``aH`` using the same convention as the existing theta
        meshes.  Otherwise it is the raw negative divergence in
        ``(km/s)/(Mpc/h)``.
    weights
        Optional particle or tracer weights.
    normalize_by_mean_weight
        If True, return the divergence of ``q=(sum w u)/<sum w>``.  This is the
        recommended folded proxy because it shares the usual dimensionless
        density normalization.
    subtract_bulk_velocity
        If True, subtract the weighted bulk velocity before painting momentum.

    Returns
    -------
    dict
        Contains ``theta_momentum_mesh``, ``momentum_divergence_mesh``,
        component meshes, and metadata.
    """
    fold = int(fold)
    if fold < 1:
        raise ValueError("fold must be >= 1.")
    nmesh = int(nmesh)
    if nmesh <= 0:
        raise ValueError("nmesh must be positive.")
    if float(boxsize) <= 0.0:
        raise ValueError("boxsize must be positive.")

    pos, vel, w = _validate_particle_arrays(positions, velocities, weights)
    folded_boxsize = float(boxsize) / float(fold)
    pos_folded = np.mod(pos, folded_boxsize)

    total_weight = float(np.sum(w, dtype=np.float64))
    if total_weight <= 0.0:
        raise ValueError("total input weight must be positive.")

    if subtract_bulk_velocity:
        bulk = np.sum(w[:, None] * vel, axis=0, dtype=np.float64) / total_weight
        vel_use = vel - bulk[None, :]
    else:
        bulk = np.zeros(3, dtype=np.float64)
        vel_use = vel

    qx = _paint_weighted(pos_folded, w * vel_use[:, 0], boxsize=folded_boxsize, nmesh=nmesh, mas=mas)
    qy = _paint_weighted(pos_folded, w * vel_use[:, 1], boxsize=folded_boxsize, nmesh=nmesh, mas=mas)
    qz = _paint_weighted(pos_folded, w * vel_use[:, 2], boxsize=folded_boxsize, nmesh=nmesh, mas=mas)

    mean_weight_per_cell = total_weight / float(nmesh**3)
    if normalize_by_mean_weight:
        qx /= np.float32(mean_weight_per_cell)
        qy /= np.float32(mean_weight_per_cell)
        qz /= np.float32(mean_weight_per_cell)

    div_q = _spectral_divergence(qx, qy, qz, boxsize=folded_boxsize)
    negative_divergence = (-div_q).astype(np.float32, copy=False)

    theta_normalized = False
    theta = negative_divergence
    if cosmo is not None or z is not None:
        if cosmo is None or z is None:
            raise ValueError("cosmo and z must be provided together.")
        a = 1.0 / (1.0 + float(z))
        Hz = _Hz_kms_per_Mpc(cosmo, float(z))
        h = _cosmo_h(cosmo)
        theta = (negative_divergence * np.float32(h / (a * Hz))).astype(np.float32, copy=False)
        theta_normalized = True

    return {
        "theta_momentum_mesh": theta.astype(np.float32, copy=False),
        "momentum_divergence_mesh": negative_divergence,
        "momentum_x_mesh": qx.astype(np.float32, copy=False),
        "momentum_y_mesh": qy.astype(np.float32, copy=False),
        "momentum_z_mesh": qz.astype(np.float32, copy=False),
        "meta": {
            "field": "theta_momentum_mesh",
            "estimator": "folded_momentum_divergence",
            "local_velocity_divergence": False,
            "folding_factor": fold,
            "boxsize": float(boxsize),
            "folded_boxsize": folded_boxsize,
            "nmesh": nmesh,
            "mas": str(mas),
            "normalize_by_mean_weight": bool(normalize_by_mean_weight),
            "subtract_bulk_velocity": bool(subtract_bulk_velocity),
            "bulk_velocity_kms": bulk.astype(float),
            "total_weight": total_weight,
            "mean_weight_per_cell": mean_weight_per_cell,
            "theta_normalized_by_aH": bool(theta_normalized),
            "z": None if z is None else float(z),
        },
    }


def _resolve_theta_key(meshes: Mapping[str, np.ndarray], theta_key: str) -> str:
    candidates = [theta_key, "theta_momentum_mesh", "tm_mesh", "tm"]
    for key in candidates:
        if key in meshes:
            return key
    raise ValueError(
        "Could not find a momentum-divergence field. "
        "Expected one of: " + ", ".join(dict.fromkeys(candidates))
    )


def append_momentum_divergence_cross_pairs(
    meshes: Mapping[str, np.ndarray],
    *,
    theta_key: str = "tm",
    fields: Iterable[str] = ("d", "E", "g"),
) -> list[tuple[str, str]]:
    """Return available cross pairs between ``theta_key`` and other fields."""
    resolved_theta_key = _resolve_theta_key(meshes, theta_key)
    return [
        (str(field), resolved_theta_key)
        for field in fields
        if str(field) in meshes and str(field) != resolved_theta_key
    ]


def measure_momentum_divergence_cross_spectra(
    meshes: Mapping[str, np.ndarray],
    *,
    boxsize: float,
    fold: int = 1,
    fields: Iterable[str] = ("d", "E", "g"),
    theta_key: str = "tm",
    power_theta_name: str = "tm",
    los: Sequence[float] = (0.0, 0.0, 1.0),
    threads: int = 8,
    assign: str = "CIC",
    theta_mas: str = "None",
    k_edges: np.ndarray | None = None,
    include_auto: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Measure cross spectra of momentum divergence with additive fields.

    The input ``meshes`` should use short power-estimator names such as ``d``,
    ``E``, and ``g``.  The momentum-divergence mesh may be named ``tm`` or
    ``theta_momentum_mesh``; the estimator output uses ``power_theta_name``.
    """
    from ia_analysis.spectra.powers import PowerConfig, PowerSpectrumEstimator

    resolved_theta_key = _resolve_theta_key(meshes, theta_key)
    power_meshes: Dict[str, np.ndarray] = {
        str(power_theta_name): np.asarray(meshes[resolved_theta_key]),
    }

    selected_fields = []
    for field in fields:
        field_name = str(field)
        if field_name in meshes:
            power_meshes[field_name] = np.asarray(meshes[field_name])
            selected_fields.append(field_name)

    if not selected_fields:
        raise ValueError("No requested cross fields were found in meshes.")

    pairs = [(field, str(power_theta_name)) for field in selected_fields]
    if include_auto:
        pairs.append((str(power_theta_name), str(power_theta_name)))

    mas_map = {field: str(assign) for field in selected_fields}
    mas_map[str(power_theta_name)] = str(theta_mas)

    fold = int(fold)
    if fold < 1:
        raise ValueError("fold must be >= 1.")

    estimator = PowerSpectrumEstimator(
        PowerConfig(
            boxsize=float(boxsize) / float(fold),
            power_norm_boxsize=float(boxsize),
            los=tuple(float(x) for x in los),  # type: ignore[arg-type]
            threads=int(threads),
        )
    )
    out = estimator.compute(
        meshes=power_meshes,
        pairs=pairs,
        mas=mas_map,
        k_edges=k_edges,
        verbose=verbose,
    )
    out["momentum_divergence"] = {
        "input_theta_key": resolved_theta_key,
        "power_theta_name": str(power_theta_name),
        "cross_fields": selected_fields,
        "self_folding_status": velocity_divergence_self_folding_status(),
    }
    return out

