#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hd_tng.py
---------
High-level IllustrisTNG driver for shell-wise halo/subhalo dynamics.

This module wraps the lower-level utilities in ``halo_dynamics.py`` and the
TNG IO layer in ``TNGCatLoader.py``.  It is designed for external scripts and
notebooks that need a single callable interface, in the same spirit as a
``global_tng.compute_many`` backend, but focused on halo dynamics:

    * select subhaloes in the largest FoF groups;
    * download/read the required TNG catalogues and subhalo cutouts;
    * build radial and/or binding-energy shell masks;
    * measure the affine flow tensor A, the symmetric velocity-flow Hessian H,
      the material-flow rotation Omega, and the figure-rotation estimate Pi;
    * compare Pi from Omega+H with the direct moment relation from dot(I);
    * optionally follow a SubLink main progenitor branch for a cross-time
      finite-difference pattern-speed test.

The default API cache policy is temporary and clean: downloaded TNG cutouts are
placed in a private system temp directory (normally under /tmp) and deleted when
``compute_haloes`` returns.  Set ``tng_catalog_kwargs={'delete_cache': False}``
if you intentionally want to keep API cutouts for debugging.
"""

from __future__ import annotations

import atexit
import glob
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from urllib import request as urlrequest

import h5py
import numpy as np
import pandas as pd

from ia_analysis.catalogs.TNGCatLoader import TNGCatalog
try:
    import halo_dynamics_mea as hd
except Exception:  # pragma: no cover
    from ia_analysis.dynamics import halo_dynamics as hd

try:  # only needed for cross-time SO(3) logarithms
    from scipy.spatial.transform import Rotation
except Exception:  # pragma: no cover
    Rotation = None

KM_S_PER_KPC_TO_GYR_INV = 1.0227121650537077
G_KPC_KMS2_MSUN = 4.30091727003628e-6

DEFAULT_CFG: Dict[str, Any] = {
    "sim_name": "TNG50-1",
    "snap": 99,
    "top_n_groups": 5,
    "max_subhaloes_per_group": 3,
    "min_dm_particles": 800,
    "include_central": False,
    "n_radial_shells": 5,
    "n_binding_shells": 5,
    "min_particles_per_shell": 100,
    "shell_methods": ("radial", "binding_energy"),
    "equal_number_radial_shells": True,
    "compute_binding_potential_if_missing": True,
    "keep_particles": True,
    "api_max_retries": 6,
    "api_retry_base_sleep": 5.0,
    "api_retry_max_sleep": 90.0,
    "verbose": True,
}

# A small registry used only when callers want an explicit cleanup hook.
_OPEN_CATALOGS: List[TNGCatalog] = []


def cleanup_open_catalogs() -> None:
    """Delete temporary API files from all catalogues opened by this module."""
    for cat in list(_OPEN_CATALOGS):
        try:
            cat.cleanup()
        except Exception:
            pass
    _OPEN_CATALOGS.clear()


atexit.register(cleanup_open_catalogs)


# -----------------------------------------------------------------------------
# Robust TNG metadata and unit conversion
# -----------------------------------------------------------------------------


def _first_hdf5(directory: Union[str, Path]) -> Optional[str]:
    files = sorted(glob.glob(str(Path(directory) / "*.hdf5")))
    return files[0] if files else None


def _request_tng_json(url: str, api_key: Optional[str], timeout: int = 60) -> Optional[dict]:
    if api_key is None:
        return None
    req = urlrequest.Request(
        url,
        headers={"api-key": str(api_key), "User-Agent": "hd_tng/1.0"},
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        warnings.warn(f"TNG API metadata request failed: {url}; reason: {exc}")
        return None


def read_header_for_snap(
    base_path: Union[str, Path],
    snap: int,
    *,
    sim_name: str = "TNG50-1",
    api_key: Optional[str] = None,
    api_base_url: str = "https://www.tng-project.org/api",
) -> Dict[str, Any]:
    """Read a TNG snapshot/group-catalog Header robustly.

    Priority: local HDF5 header, ``illustris_python.groupcat.loadHeader``, TNG
    API metadata, and finally a safe TNG50-1 snap-99 fallback.
    """
    base_path = Path(base_path)
    snap = int(snap)

    for fp in [
        _first_hdf5(base_path / f"groups_{snap:03d}"),
        _first_hdf5(base_path / f"snapdir_{snap:03d}"),
    ]:
        if fp is None:
            continue
        with h5py.File(fp, "r") as f:
            return dict(f["Header"].attrs)

    try:
        import illustris_python as il  # type: ignore

        groupcat = getattr(il, "groupcat", None)
        if groupcat is None:
            import illustris_python.groupcat as groupcat  # type: ignore
        return dict(groupcat.loadHeader(str(base_path), snap))
    except Exception:
        pass

    meta = _request_tng_json(
        f"{api_base_url.rstrip('/')}/{sim_name}/snapshots/{snap}/",
        api_key=api_key,
    )
    if meta is not None:
        a = 1.0 / (1.0 + float(meta.get("redshift", 0.0)))
        h = float(meta.get("hubble", meta.get("HubbleParam", 0.6774)))
        return {
            "Time": a,
            "Redshift": float(meta.get("redshift", 1.0 / a - 1.0)),
            "HubbleParam": h,
            "Omega0": float(meta.get("omega_m", meta.get("Omega0", 0.3089))),
            "OmegaLambda": float(meta.get("omega_l", meta.get("OmegaLambda", 0.6911))),
            "BoxSize": float(meta.get("boxsize", 35000.0 if sim_name.startswith("TNG50") else np.nan)),
        }

    if sim_name.startswith("TNG50") and snap == 99:
        return {
            "Time": 1.0,
            "Redshift": 0.0,
            "HubbleParam": 0.6774,
            "Omega0": 0.3089,
            "OmegaLambda": 0.6911,
            "BoxSize": 35000.0,
            "MassTable": np.array([0.0, 4.5e-6, 0, 0, 0, 0]),
        }

    raise RuntimeError(f"Could not read a Header for snap={snap}, base_path={base_path}")


def scale_factor_from_header(header: Mapping[str, Any]) -> float:
    return float(header.get("Time", 1.0))


def hubble_from_header(header: Mapping[str, Any]) -> float:
    return float(header.get("HubbleParam", 0.6774))


def boxsize_ckpc_h_from_header(header: Mapping[str, Any]) -> float:
    return float(header.get("BoxSize", np.nan))


def tng_relative_positions_to_physical_kpc(
    coords_ckpc_h: np.ndarray,
    center_ckpc_h: np.ndarray,
    header: Mapping[str, Any],
) -> np.ndarray:
    """Return periodic relative coordinates in physical kpc."""
    coords = np.asarray(coords_ckpc_h, dtype=np.float64)
    center = np.asarray(center_ckpc_h, dtype=np.float64)
    box = boxsize_ckpc_h_from_header(header)
    d = coords - center[None, :]
    if np.isfinite(box) and box > 0:
        d = d - box * np.rint(d / box)
    return d * scale_factor_from_header(header) / hubble_from_header(header)


def tng_velocity_to_kms(vel_code: np.ndarray, header: Mapping[str, Any]) -> np.ndarray:
    """Convert TNG stored peculiar velocities to km/s."""
    return np.asarray(vel_code, dtype=np.float64) * np.sqrt(scale_factor_from_header(header))


def dm_mass_msun_from_header(header: Mapping[str, Any], n: int = 1) -> np.ndarray:
    """Return a constant DM particle mass array in Msun."""
    h = hubble_from_header(header)
    mt = np.asarray(header.get("MassTable", np.zeros(6)), dtype=np.float64)
    if mt.size >= 2 and mt[1] > 0:
        m = mt[1] * 1.0e10 / h
    else:
        # TNG50-1 DM particle mass, approximate fallback.  Exact value is not
        # critical for equal-mass DM shell shapes, but matters for absolute units.
        m = 4.5e4 / h
    return np.full(int(n), float(m), dtype=np.float64)


def cosmic_time_gyr_from_header(header: Mapping[str, Any]) -> float:
    """Flat-LCDM cosmic age at the snapshot scale factor, in Gyr."""
    a = scale_factor_from_header(header)
    h = hubble_from_header(header)
    om = float(header.get("Omega0", 0.3089))
    ol = float(header.get("OmegaLambda", 1.0 - om))
    if not (np.isfinite(a) and a > 0 and om > 0 and ol > 0 and h > 0):
        return np.nan
    hubble_time_gyr = 9.778 / h
    return float(2.0 / (3.0 * np.sqrt(ol)) * np.arcsinh(np.sqrt(ol / om) * a ** 1.5) * hubble_time_gyr)


# -----------------------------------------------------------------------------
# Small retry wrapper and TNG catalogue opening
# -----------------------------------------------------------------------------


def _is_retryable_exception(exc: Exception) -> bool:
    text = repr(exc)
    retry_tokens = ["HTTP Error 429", "HTTP Error 500", "HTTP Error 502", "HTTP Error 503", "HTTP Error 504", "Gateway Time-out", "timed out"]
    return any(tok in text for tok in retry_tokens)


def retry_call(func, *args, max_retries: int = 6, base_sleep: float = 5.0, max_sleep: float = 90.0, verbose: bool = True, **kwargs):
    """Retry a fragile TNG API/local IO call on transient HTTP errors."""
    last_exc = None
    for attempt in range(int(max_retries) + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt >= int(max_retries) or not _is_retryable_exception(exc):
                raise
            sleep = min(float(max_sleep), float(base_sleep) * (2 ** attempt))
            if verbose:
                print(f"[retry] {func.__name__} failed with {exc!r}; sleeping {sleep:.1f}s before retry {attempt+1}/{max_retries}")
            time.sleep(sleep)
    raise last_exc  # pragma: no cover


def default_tng_catalog_kwargs(
    *,
    sim_name: str = "TNG50-1",
    api_key: Optional[str] = None,
    download_if_missing: bool = True,
    delete_cache: bool = True,
    cache_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True,
    timeout: int = 180,
) -> Dict[str, Any]:
    """Default clean API-cache policy: use /tmp and delete after use."""
    return {
        "sim_name": sim_name,
        "api_key": api_key if api_key is not None else os.environ.get("TNG_API_KEY"),
        "download_if_missing": download_if_missing,
        "delete_cache": delete_cache,
        "cache_dir": None if cache_dir is None else str(cache_dir),
        "verbose": verbose,
        "timeout": timeout,
    }


def open_catalog(
    base_path: Union[str, Path],
    snap: int,
    *,
    group_fields: Optional[Sequence[str]] = None,
    subhalo_fields: Optional[Sequence[str]] = None,
    tng_catalog_kwargs: Optional[Mapping[str, Any]] = None,
    retry_cfg: Optional[Mapping[str, Any]] = None,
) -> Tuple[TNGCatalog, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Open a TNGCatalog and load an API-safe minimal group catalogue."""
    group_fields = list(group_fields or ["GroupLenType", "GroupFirstSub", "GroupNsubs"])
    subhalo_fields = list(subhalo_fields or ["SubhaloGrNr", "SubhaloLenType", "SubhaloPos", "SubhaloVel", "SubhaloHalfmassRadType"])
    kwargs = dict(tng_catalog_kwargs or {})
    cat = TNGCatalog(str(base_path), int(snap), **kwargs)
    _OPEN_CATALOGS.append(cat)
    rcfg = dict(retry_cfg or {})
    halos, subs = retry_call(cat.loadFoF, group_fields=group_fields, subhalo_fields=subhalo_fields, **rcfg)
    return cat, halos, subs


# -----------------------------------------------------------------------------
# Target selection and particle loading
# -----------------------------------------------------------------------------


def select_subhaloes_in_top_groups(
    halos: Mapping[str, np.ndarray],
    subs: Mapping[str, np.ndarray],
    *,
    top_n_groups: int = 5,
    max_subhaloes_per_group: int = 3,
    min_dm_particles: int = 800,
    include_central: bool = False,
) -> pd.DataFrame:
    """Select subhaloes in the largest FoF groups, ranked by DM particle count."""
    group_len = np.asarray(halos["GroupLenType"], dtype=np.int64)
    group_first = np.asarray(halos["GroupFirstSub"], dtype=np.int64)
    group_nsubs = np.asarray(halos["GroupNsubs"], dtype=np.int64)
    sub_len = np.asarray(subs["SubhaloLenType"], dtype=np.int64)
    sub_gid = np.asarray(subs["SubhaloGrNr"], dtype=np.int64)

    group_rank = np.argsort(group_len[:, 1])[::-1][: int(top_n_groups)]
    rows = []
    for rank, gid in enumerate(group_rank):
        gid = int(gid)
        sids = np.flatnonzero(sub_gid == gid)
        if not include_central:
            cen = int(group_first[gid]) if group_first[gid] >= 0 else -1
            sids = sids[sids != cen]
        sids = sids[sub_len[sids, 1] >= int(min_dm_particles)]
        if sids.size == 0:
            continue
        # Largest DM subhaloes first.
        sids = sids[np.argsort(sub_len[sids, 1])[::-1]]
        sids = sids[: int(max_subhaloes_per_group)]
        for local_rank, sid in enumerate(sids):
            rows.append(
                {
                    "FoFRank": int(rank),
                    "GroupID": int(gid),
                    "SubhaloID": int(sid),
                    "RankInGroup": int(local_rank),
                    "GroupNdm": int(group_len[gid, 1]),
                    "GroupNsubs": int(group_nsubs[gid]),
                    "Ndm": int(sub_len[sid, 1]),
                    "Nstar": int(sub_len[sid, 4]) if sub_len.shape[1] > 4 else 0,
                    "IsCentral": bool(sid == int(group_first[gid])),
                }
            )
    return pd.DataFrame(rows)


def _load_subhalo_dm_particles_raw(cat: TNGCatalog, sid: int, retry_cfg: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    fields = ["Coordinates", "Velocities", "ParticleIDs", "Potential"]
    try:
        pdata = retry_call(cat.loadSubhalos, int(sid), ptypes=[1], fields=fields, **retry_cfg)
    except Exception as exc:
        warnings.warn(f"Potential read failed for sid={sid}; retrying without Potential. Reason: {exc}")
        pdata = retry_call(cat.loadSubhalos, int(sid), ptypes=[1], fields=fields[:-1], **retry_cfg)
    return pdata.get("PartType1", {})


def load_subhalo_dm_particles(
    cat: TNGCatalog,
    subs: Mapping[str, np.ndarray],
    sid: int,
    *,
    snap: int,
    base_path: Union[str, Path],
    header: Mapping[str, Any],
    retry_cfg: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Load and convert one subhalo's DM particles into relative physical units."""
    retry_cfg = dict(retry_cfg or {})
    p1 = _load_subhalo_dm_particles_raw(cat, int(sid), retry_cfg)
    coords_raw = np.asarray(p1["Coordinates"], dtype=np.float64)
    vel_raw = np.asarray(p1["Velocities"], dtype=np.float64)
    ids = np.asarray(p1.get("ParticleIDs", np.arange(coords_raw.shape[0])), dtype=np.int64)

    center_ckpc_h = np.asarray(subs["SubhaloPos"][int(sid)], dtype=np.float64)
    v_ref_raw = np.asarray(subs["SubhaloVel"][int(sid)], dtype=np.float64)

    X_kpc = tng_relative_positions_to_physical_kpc(coords_raw, center_ckpc_h, header)
    U_kms = tng_velocity_to_kms(vel_raw, header) - tng_velocity_to_kms(v_ref_raw[None, :], header)[0]
    masses = dm_mass_msun_from_header(header, coords_raw.shape[0])

    potential = None
    if "Potential" in p1:
        potential = np.asarray(p1["Potential"], dtype=np.float64)
        if potential.shape != (coords_raw.shape[0],):
            potential = None

    return {
        "sid": int(sid),
        "snap": int(snap),
        "coords_ckpc_h": coords_raw,
        "vel_code": vel_raw,
        "center_ckpc_h": center_ckpc_h,
        "v_ref_code": v_ref_raw,
        "X_kpc": X_kpc,
        "U_kms": U_kms,
        "masses": masses,
        "ids": ids,
        "potential": potential,
    }


# -----------------------------------------------------------------------------
# Shell analysis and closure diagnostics
# -----------------------------------------------------------------------------


def direct_pi_from_P(I: np.ndarray, P: np.ndarray, rel_gap_min: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Pi directly from R^T dot(I) R = R^T(P+P^T)R."""
    evals, R = hd.eigh_sorted_desc(I)
    scale = max(np.max(np.abs(evals)), 1.0)
    gaps = np.array([abs(evals[1] - evals[0]), abs(evals[2] - evals[0]), abs(evals[2] - evals[1])])
    if np.min(gaps) < rel_gap_min * scale:
        return np.full((3, 3), np.nan), evals, R
    Phat = R.T @ P @ R
    dIhat = Phat + Phat.T
    Pi = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(i + 1, 3):
            Pi[i, j] = dIhat[i, j] / (evals[j] - evals[i])
            Pi[j, i] = -Pi[i, j]
    return Pi, evals, R


def skew_components(M: np.ndarray) -> np.ndarray:
    return np.array([M[0, 1], M[0, 2], M[1, 2]], dtype=np.float64)


def build_shell_masks_for_particles(
    pdata: Mapping[str, Any],
    *,
    method: str,
    n_shells: int,
    equal_number: bool = True,
    compute_binding_potential_if_missing: bool = True,
    min_particles: int = 10,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Build shell masks for a loaded particle dictionary."""
    X = np.asarray(pdata["X_kpc"], dtype=np.float64)
    U = np.asarray(pdata["U_kms"], dtype=np.float64)
    m = np.asarray(pdata["masses"], dtype=np.float64)
    method = method.lower().strip()
    if method == "radial":
        return hd.make_shell_masks(
            X,
            velocities=U,
            masses=m,
            center=np.zeros(3),
            v_ref=np.zeros(3),
            method="radial",
            n_shells=int(n_shells),
            equal_number=bool(equal_number),
        )
    if method in ("binding", "binding_energy", "energy"):
        return hd.make_shell_masks(
            X,
            velocities=U,
            masses=m,
            center=np.zeros(3),
            v_ref=np.zeros(3),
            method="binding_energy",
            n_shells=int(n_shells),
            potentials=pdata.get("potential", None),
            compute_potential_if_missing=bool(compute_binding_potential_if_missing),
            G=G_KPC_KMS2_MSUN,
            softening=0.05,
        )
    raise ValueError("method must be 'radial' or 'binding_energy'")


def _safe_component(M: np.ndarray, i: int, j: int) -> float:
    """Safely read one matrix component."""
    M = np.asarray(M, dtype=float)
    if M.shape != (3, 3):
        return float("nan")
    val = M[i, j]
    return float(val) if np.isfinite(val) else float("nan")


def _relative_residual_pct(y: float, x: float) -> float:
    """
    Relative residual in percent:
        100 * (y - x) / |x|.
    """
    if np.isfinite(x) and abs(x) > 0 and np.isfinite(y):
        return float(100.0 * (y - x) / abs(x))
    return float("nan")


def closure_table_from_analysis(
    sid: int,
    snap: int,
    analysis: Mapping[str, Any],
    *,
    shell_method: str,
    unit_factor: float = KM_S_PER_KPC_TO_GYR_INV,
) -> pd.DataFrame:
    """
    Compare the affine Omega+H model against the directly measured dI Pi.

    Definitions
    -----------
    Pi_direct:
        Measured from primitive dI,

            dI_ij = sum m (U_i X_j + X_i U_j),
            Pi_ij = (R^T dI R)_ij / (lambda_j - lambda_i).

        In halo_dynamics_mea this is stored in

            sh["figure"]["Pi"].

    Pi_aff:
        The affine model/decomposition,

            Pi_ij = Omega_hat_ij + eta_ij H_hat_ij.

        In halo_dynamics_mea this is stored in

            sh["figure_affine"]["Pi"].

    This replaces the old use of direct_pi_from_P(...) for Pi_direct.
    """
    rows = []
    comps = [("01", 0, 1), ("02", 0, 2), ("12", 1, 2)]
    sym6 = [
        ("00", 0, 0),
        ("11", 1, 1),
        ("22", 2, 2),
        ("01", 0, 1),
        ("02", 0, 2),
        ("12", 1, 2),
    ]

    for ish, sh in enumerate(analysis["shells"]):
        if not sh.get("valid", False):
            continue

        fig_direct = sh.get("figure", {})
        fig_affine = sh.get("figure_affine", {})

        # Compatibility fallback for old halo_dynamics outputs.
        if not fig_affine:
            fig_affine = fig_direct

        Pi_direct = np.asarray(
            fig_direct.get("Pi", np.full((3, 3), np.nan)),
            dtype=np.float64,
        )
        Pi_aff = np.asarray(
            fig_affine.get("Pi", np.full((3, 3), np.nan)),
            dtype=np.float64,
        )

        evals = np.asarray(
            fig_direct.get("evals", fig_affine.get("evals", np.full(3, np.nan))),
            dtype=np.float64,
        )

        dI = np.asarray(sh.get("dI", np.full((3, 3), np.nan)), dtype=np.float64)
        dI_hat = np.asarray(fig_direct.get("dI_hat", np.full((3, 3), np.nan)), dtype=np.float64)
        lambda_dot = np.asarray(fig_direct.get("lambda_dot", np.full(3, np.nan)), dtype=np.float64)

        Pi_aff_gyr = Pi_aff * float(unit_factor)
        Pi_direct_gyr = Pi_direct * float(unit_factor)
        diff = Pi_aff_gyr - Pi_direct_gyr

        # Omega/H decomposition is attached to the affine model figure.
        Omega_hat = np.asarray(fig_affine.get("Omega_hat", np.full((3, 3), np.nan)), dtype=float)
        H_hat = np.asarray(fig_affine.get("H_hat", np.full((3, 3), np.nan)), dtype=float)
        eta = np.asarray(fig_affine.get("eta", np.full((3, 3), np.nan)), dtype=float)

        Pi_Omega = Omega_hat * float(unit_factor)
        Pi_H = eta * H_hat * float(unit_factor)

        row: Dict[str, Any] = {
            "snap": int(snap),
            "SubhaloID": int(sid),
            "shell_method": str(shell_method),
            "shell": int(ish),
            "N": int(sh.get("N_used", sh.get("N", 0))),
            "Pi_direct_source": str(fig_direct.get("source", "direct_dI")),
            "Pi_direct_valid": bool(fig_direct.get("valid", False)),
            "Pi_affine_valid": bool(fig_affine.get("valid", False)),
            "lambda1": float(evals[0]) if evals.size > 0 else np.nan,
            "lambda2": float(evals[1]) if evals.size > 1 else np.nan,
            "lambda3": float(evals[2]) if evals.size > 2 else np.nan,
            "lambda_dot1": float(lambda_dot[0]) if lambda_dot.size > 0 else np.nan,
            "lambda_dot2": float(lambda_dot[1]) if lambda_dot.size > 1 else np.nan,
            "lambda_dot3": float(lambda_dot[2]) if lambda_dot.size > 2 else np.nan,
            "rms_Pi_direct": float(np.sqrt(np.nanmean(skew_components(Pi_direct_gyr) ** 2))),
            "rms_diff": float(np.sqrt(np.nanmean(skew_components(diff) ** 2))),
        }

        for lab, i, j in sym6:
            row[f"dI_raw_{lab}"] = _safe_component(dI, i, j)
            row[f"dI_hat_{lab}"] = _safe_component(dI_hat, i, j)

        for lab, i, j in comps:
            pa = _safe_component(Pi_aff_gyr, i, j)
            pd_ = _safe_component(Pi_direct_gyr, i, j)
            rr = _relative_residual_pct(pa, pd_)

            row[f"Pi_aff_{lab}"] = pa
            row[f"Pi_direct_{lab}"] = pd_
            row[f"residual_{lab}"] = pa - pd_
            row[f"rel_residual_{lab}_pct"] = rr

            row[f"Pi_Omega_{lab}"] = _safe_component(Pi_Omega, i, j)
            row[f"Pi_H_{lab}"] = _safe_component(Pi_H, i, j)

            denom = abs(row[f"Pi_Omega_{lab}"]) + abs(row[f"Pi_H_{lab}"])
            if np.isfinite(denom) and denom > 0:
                row[f"f_Omega_abs_{lab}"] = abs(row[f"Pi_Omega_{lab}"]) / denom
                row[f"f_H_abs_{lab}"] = abs(row[f"Pi_H_{lab}"]) / denom
            else:
                row[f"f_Omega_abs_{lab}"] = np.nan
                row[f"f_H_abs_{lab}"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)

def analyse_particle_data(
    pdata: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any],
    shell_methods: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run radial/binding shell dynamics on one loaded particle set."""
    shell_methods = tuple(shell_methods or cfg.get("shell_methods", ("radial", "binding_energy")))
    out: Dict[str, Any] = {}
    for method in shell_methods:
        method_key = "binding_energy" if str(method).lower().startswith("binding") else "radial"
        n_shells = int(cfg.get("n_binding_shells" if method_key == "binding_energy" else "n_radial_shells", 5))
        masks, info = build_shell_masks_for_particles(
            pdata,
            method=method_key,
            n_shells=n_shells,
            equal_number=bool(cfg.get("equal_number_radial_shells", True)),
            compute_binding_potential_if_missing=bool(cfg.get("compute_binding_potential_if_missing", True)),
        )
        analysis = hd.analyze_halo_shells(
            pdata["X_kpc"],
            pdata["U_kms"],
            masses=pdata["masses"],
            center=np.zeros(3),
            v_ref=np.zeros(3),
            shell_masks=masks,
            min_particles=int(cfg.get("min_particles_per_shell", 100)),
        )
        closure = closure_table_from_analysis(pdata["sid"], pdata["snap"], analysis, shell_method=method_key)
        out[method_key] = {
            "masks": masks,
            "info": info,
            "analysis": analysis,
            "closure": closure,
        }
    return out


def compute_one_subhalo(
    cat: TNGCatalog,
    halos: Mapping[str, np.ndarray],
    subs: Mapping[str, np.ndarray],
    sid: int,
    *,
    gid: Optional[int],
    snap: int,
    base_path: Union[str, Path],
    header: Mapping[str, Any],
    cfg: Mapping[str, Any],
    retry_cfg: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Load one subhalo and compute all requested shell-dynamics quantities."""
    sid = int(sid)
    gid_cat = int(subs["SubhaloGrNr"][sid])
    gid = gid_cat if gid is None else int(gid)
    pdata = load_subhalo_dm_particles(cat, subs, sid, snap=int(snap), base_path=base_path, header=header, retry_cfg=retry_cfg)
    shells = analyse_particle_data(pdata, cfg=cfg)

    sub_info = {
        "SubhaloID": sid,
        "GroupID": gid,
        "GroupID_catalog": gid_cat,
        "SubhaloPos_ckpc_h": np.asarray(subs["SubhaloPos"][sid], dtype=np.float64),
        "SubhaloVel_code": np.asarray(subs["SubhaloVel"][sid], dtype=np.float64),
        "SubhaloLenType": np.asarray(subs["SubhaloLenType"][sid], dtype=np.int64),
    }
    if "SubhaloHalfmassRadType" in subs:
        sub_info["SubhaloHalfmassRadType_ckpc_h"] = np.asarray(subs["SubhaloHalfmassRadType"][sid], dtype=np.float64)

    result = {
        "Sub_info": sub_info,
        "shells": shells,
    }
    if bool(cfg.get("keep_particles", True)):
        result["particles"] = {"dm": pdata}
    return result


# -----------------------------------------------------------------------------
# Public batch API
# -----------------------------------------------------------------------------


def compute_haloes(
    base_path: Union[str, Path],
    snap: int = 99,
    *,
    cfg: Optional[Mapping[str, Any]] = None,
    target_table: Optional[pd.DataFrame] = None,
    sid_array: Optional[Sequence[int]] = None,
    gid_array: Optional[Sequence[int]] = None,
    group_fields: Optional[Sequence[str]] = None,
    subhalo_fields: Optional[Sequence[str]] = None,
    tng_catalog_kwargs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute shell-wise dynamics for a batch of TNG subhaloes.

    Parameters
    ----------
    base_path, snap
        TNG base path and snapshot number.
    cfg
        Runtime configuration.  Missing keys are filled from ``DEFAULT_CFG``.
    target_table
        Optional table with at least ``SubhaloID`` and ``GroupID`` columns.
    sid_array, gid_array
        Optional explicit arrays, similar to ``global_tng.compute_many``.
    tng_catalog_kwargs
        Forwarded to :class:`TNGCatLoader.TNGCatalog`.  Defaults use API
        fallback, system temp cache, and delete-on-cleanup.
    """
    run_cfg = dict(DEFAULT_CFG)
    if cfg is not None:
        run_cfg.update(dict(cfg))
    sim_name = str(run_cfg.get("sim_name", "TNG50-1"))
    api_key = run_cfg.get("api_key", os.environ.get("TNG_API_KEY"))
    header = read_header_for_snap(base_path, int(snap), sim_name=sim_name, api_key=api_key)

    cat_kwargs = default_tng_catalog_kwargs(
        sim_name=sim_name,
        api_key=api_key,
        download_if_missing=bool(run_cfg.get("download_if_missing", True)),
        delete_cache=bool(run_cfg.get("delete_cache", True)),
        cache_dir=run_cfg.get("cache_dir", None),
        verbose=bool(run_cfg.get("verbose", True)),
        timeout=int(run_cfg.get("timeout", 180)),
    )
    if tng_catalog_kwargs is not None:
        cat_kwargs.update(dict(tng_catalog_kwargs))

    retry_cfg = {
        "max_retries": int(run_cfg.get("api_max_retries", 6)),
        "base_sleep": float(run_cfg.get("api_retry_base_sleep", 5.0)),
        "max_sleep": float(run_cfg.get("api_retry_max_sleep", 90.0)),
        "verbose": bool(run_cfg.get("verbose", True)),
    }

    cat = None
    failures: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    try:
        cat, halos, subs = open_catalog(
            base_path,
            int(snap),
            group_fields=group_fields,
            subhalo_fields=subhalo_fields,
            tng_catalog_kwargs=cat_kwargs,
            retry_cfg=retry_cfg,
        )

        if target_table is None:
            if sid_array is not None:
                sids = np.asarray(sid_array, dtype=np.int64)
                if gid_array is None:
                    gids = np.asarray(subs["SubhaloGrNr"], dtype=np.int64)[sids]
                else:
                    gids = np.asarray(gid_array, dtype=np.int64)
                target_table = pd.DataFrame({"SubhaloID": sids, "GroupID": gids})
            else:
                target_table = select_subhaloes_in_top_groups(
                    halos,
                    subs,
                    top_n_groups=int(run_cfg.get("top_n_groups", 5)),
                    max_subhaloes_per_group=int(run_cfg.get("max_subhaloes_per_group", 3)),
                    min_dm_particles=int(run_cfg.get("min_dm_particles", 800)),
                    include_central=bool(run_cfg.get("include_central", False)),
                )

        for _, row in target_table.iterrows():
            sid = int(row["SubhaloID"])
            gid = int(row["GroupID"]) if "GroupID" in row else None
            try:
                if bool(run_cfg.get("verbose", True)):
                    print(f"[hd_tng] computing sid={sid}, gid={gid}")
                results.append(
                    compute_one_subhalo(
                        cat,
                        halos,
                        subs,
                        sid,
                        gid=gid,
                        snap=int(snap),
                        base_path=base_path,
                        header=header,
                        cfg=run_cfg,
                        retry_cfg=retry_cfg,
                    )
                )
            except Exception as exc:
                failures.append({"SubhaloID": sid, "GroupID": gid, "error": repr(exc)})
                if bool(run_cfg.get("verbose", True)):
                    print(f"[hd_tng] failed sid={sid}: {exc}")

        closure_tables = []
        for res in results:
            for method, block in res.get("shells", {}).items():
                if "closure" in block and len(block["closure"]):
                    closure_tables.append(block["closure"])
        closure_all = pd.concat(closure_tables, ignore_index=True) if closure_tables else pd.DataFrame()

        return {
            "metadata": {
                "sim_name": sim_name,
                "base_path": str(base_path),
                "snap": int(snap),
                "header": header,
                "unit_factor_gyr_inv": KM_S_PER_KPC_TO_GYR_INV,
                "cfg": run_cfg,
            },
            "target_table": target_table.reset_index(drop=True),
            "results": results,
            "closure_all": closure_all,
            "failures": pd.DataFrame(failures),
        }
    finally:
        if cat is not None:
            try:
                cat.cleanup()
            except Exception:
                pass
            if cat in _OPEN_CATALOGS:
                _OPEN_CATALOGS.remove(cat)


# Compatibility alias with the global_tng naming style.
compute_many = compute_haloes



# -----------------------------------------------------------------------------
# FoF/subhalo metadata enrichment
# -----------------------------------------------------------------------------


def _array_or_none(d: Mapping[str, Any], key: str) -> Optional[np.ndarray]:
    if isinstance(d, Mapping) and key in d:
        return np.asarray(d[key])
    return None


def _safe_item(x: Any) -> Any:
    arr = np.asarray(x)
    if arr.shape == ():
        return arr.item()
    return x


def _infer_sid_from_result(res: Mapping[str, Any]) -> Optional[int]:
    """Infer SubfindID from one result block."""
    if not isinstance(res, Mapping):
        return None

    candidates: List[Any] = [
        res.get("SubhaloID", None),
        res.get("SubfindID", None),
        res.get("sid", None),
        res.get("SID", None),
    ]

    sub_info = res.get("Sub_info", None)
    if isinstance(sub_info, Mapping):
        candidates.extend(
            [
                sub_info.get("SubhaloID", None),
                sub_info.get("SubfindID", None),
                sub_info.get("sid", None),
                sub_info.get("SID", None),
            ]
        )

    for x in candidates:
        if x is None:
            continue
        try:
            arr = np.asarray(x)
            if arr.size == 1:
                return int(arr.reshape(-1)[0])
        except Exception:
            pass

    return None


def _metadata_for_one_subhalo(
    halos: Mapping[str, np.ndarray],
    subs: Mapping[str, np.ndarray],
    sid: int,
    *,
    header: Mapping[str, Any],
    gid_input: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a compact FoF-centric metadata dictionary for one subhalo."""
    sid = int(sid)

    sub_grnr = _array_or_none(subs, "SubhaloGrNr")
    group_first = _array_or_none(halos, "GroupFirstSub")

    if sub_grnr is None:
        raise KeyError("SubhaloGrNr not found in subhalo catalog.")
    if group_first is None:
        raise KeyError("GroupFirstSub not found in group catalog.")

    gid = int(sub_grnr[sid])
    cen_id = int(group_first[gid]) if gid >= 0 else -1

    sub_pos = _array_or_none(subs, "SubhaloPos")
    sub_vel = _array_or_none(subs, "SubhaloVel")
    group_pos = _array_or_none(halos, "GroupPos")
    group_vel = _array_or_none(halos, "GroupVel")

    pos_abs_ckpc_h = np.asarray(sub_pos[sid], dtype=float) if sub_pos is not None else np.full(3, np.nan)

    if cen_id >= 0 and sub_pos is not None:
        cen_pos_ckpc_h = np.asarray(sub_pos[cen_id], dtype=float)
    elif group_pos is not None:
        cen_pos_ckpc_h = np.asarray(group_pos[gid], dtype=float)
    else:
        cen_pos_ckpc_h = np.full(3, np.nan)

    group_pos_ckpc_h = np.asarray(group_pos[gid], dtype=float) if group_pos is not None else np.full(3, np.nan)

    pos_rel_cen_kpc = tng_relative_positions_to_physical_kpc(
        pos_abs_ckpc_h[None, :],
        cen_pos_ckpc_h,
        header,
    )[0]
    pos_rel_group_kpc = tng_relative_positions_to_physical_kpc(
        pos_abs_ckpc_h[None, :],
        group_pos_ckpc_h,
        header,
    )[0]

    meta: Dict[str, Any] = {
        "SubhaloID": int(sid),
        "GroupID": int(gid),
        "GroupID_input": None if gid_input is None else int(gid_input),
        "CenID": int(cen_id),
        "IsCentral": bool(sid == cen_id),
        "SubhaloPos_ckpc_h": pos_abs_ckpc_h,
        "CenPos_ckpc_h": cen_pos_ckpc_h,
        "GroupPos_ckpc_h": group_pos_ckpc_h,
        "pos_rel_cen_kpc": pos_rel_cen_kpc,
        "pos_rel_group_kpc": pos_rel_group_kpc,
        "r_cen_kpc": float(np.linalg.norm(pos_rel_cen_kpc)) if np.all(np.isfinite(pos_rel_cen_kpc)) else np.nan,
        "r_group_kpc": float(np.linalg.norm(pos_rel_group_kpc)) if np.all(np.isfinite(pos_rel_group_kpc)) else np.nan,
    }

    if sub_vel is not None:
        meta["SubhaloVel_code"] = np.asarray(sub_vel[sid], dtype=float)
        if cen_id >= 0:
            meta["CenVel_code"] = np.asarray(sub_vel[cen_id], dtype=float)
            meta["vel_rel_cen_kms"] = (
                tng_velocity_to_kms(meta["SubhaloVel_code"][None, :], header)[0]
                - tng_velocity_to_kms(meta["CenVel_code"][None, :], header)[0]
            )

    if group_vel is not None:
        meta["GroupVel_code"] = np.asarray(group_vel[gid], dtype=float)

    sub_len_type = _array_or_none(subs, "SubhaloLenType")
    if sub_len_type is not None:
        meta["SubhaloLenType"] = np.asarray(sub_len_type[sid], dtype=np.int64)
        if sub_len_type.ndim == 2 and sub_len_type.shape[1] > 1:
            meta["Ndm"] = int(sub_len_type[sid, 1])
        if sub_len_type.ndim == 2 and sub_len_type.shape[1] > 4:
            meta["Nstar"] = int(sub_len_type[sid, 4])

    group_len_type = _array_or_none(halos, "GroupLenType")
    if group_len_type is not None:
        meta["GroupLenType"] = np.asarray(group_len_type[gid], dtype=np.int64)

    for key in ["SubhaloMass", "SubhaloVmax", "SubhaloFlag"]:
        arr = _array_or_none(subs, key)
        if arr is not None:
            meta[key] = _safe_item(arr[sid])

    for key in [
        "SubhaloMassType",
        "SubhaloHalfmassRadType",
    ]:
        arr = _array_or_none(subs, key)
        if arr is not None:
            meta[key] = np.asarray(arr[sid])

    for key in [
        "GroupMass",
        "GroupNsubs",
        "Group_R_Crit200",
        "Group_M_Crit200",
        "Group_R_Mean200",
        "Group_M_Mean200",
        "Group_R_TopHat200",
        "Group_M_TopHat200",
    ]:
        arr = _array_or_none(halos, key)
        if arr is not None:
            meta[key] = _safe_item(arr[gid])

    if "Group_R_Crit200" in meta and np.isfinite(float(meta["Group_R_Crit200"])):
        r200_kpc = ckpc_h_to_physical_kpc(float(meta["Group_R_Crit200"]), header)
        meta["Group_R_Crit200_kpc"] = float(r200_kpc)
        if np.isfinite(meta["r_cen_kpc"]) and r200_kpc > 0:
            meta["r_cen_over_R200c"] = float(meta["r_cen_kpc"] / r200_kpc)

    return meta


def _flatten_metadata_for_table(meta: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten scalar and short-vector metadata for a pandas table."""
    row: Dict[str, Any] = {}

    for key, val in meta.items():
        arr = np.asarray(val)

        if arr.shape == ():
            try:
                row[key] = arr.item()
            except Exception:
                row[key] = val
        elif arr.shape == (3,):
            row[f"{key}_x"] = float(arr[0])
            row[f"{key}_y"] = float(arr[1])
            row[f"{key}_z"] = float(arr[2])
        elif key in ("SubhaloLenType", "GroupLenType", "SubhaloMassType", "GroupMassType"):
            labels = ["gas", "dm", "tracer", "bh", "stars", "wind"]
            for i in range(min(arr.size, len(labels))):
                row[f"{key}_{labels[i]}"] = arr.reshape(-1)[i]
        else:
            # Keep non-scalar arrays only inside Sub_info, not the summary table.
            continue

    return row


def _build_subhalo_metadata_table(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for res in results:
        sub_info = res.get("Sub_info", {}) if isinstance(res, Mapping) else {}
        if not isinstance(sub_info, Mapping):
            continue

        row = _flatten_metadata_for_table(sub_info)
        if "SubhaloID" in sub_info:
            row["SubhaloID"] = int(sub_info["SubhaloID"])
        rows.append(row)

    return pd.DataFrame(rows)


def _attach_metadata_to_closure(closure: pd.DataFrame, meta_table: pd.DataFrame) -> pd.DataFrame:
    if closure is None or not isinstance(closure, pd.DataFrame) or len(closure) == 0:
        return closure
    if meta_table is None or not isinstance(meta_table, pd.DataFrame) or len(meta_table) == 0:
        return closure
    if "SubhaloID" not in closure.columns or "SubhaloID" not in meta_table.columns:
        return closure

    drop_cols = [c for c in meta_table.columns if c in closure.columns and c != "SubhaloID"]
    meta_use = meta_table.drop(columns=drop_cols, errors="ignore")
    return closure.merge(meta_use, on="SubhaloID", how="left")


def enrich_run_with_group_metadata(
    run: Dict[str, Any],
    *,
    base_path: Optional[Union[str, Path]] = None,
    snap: Optional[int] = None,
    group_fields: Optional[Sequence[str]] = None,
    subhalo_fields: Optional[Sequence[str]] = None,
    tng_catalog_kwargs: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """
    Enrich an existing ``compute_haloes`` run with FoF/group metadata.

    This function does not recompute particle or shell dynamics.  It reloads
    the group catalog for the target snapshot and patches each result's
    ``Sub_info`` with FoF-centric quantities:

      - GroupID, CenID, IsCentral;
      - subhalo / central / FoF positions;
      - FoF-centric distances and r/R200c;
      - particle counts and common mass/radius fields when available.

    It also writes:

      - ``run["subhalo_metadata"]``;
      - metadata columns into ``run["closure_all"]`` when possible.

    Returns
    -------
    pandas.DataFrame
        One row per result/subhalo.
    """
    meta_run = run.get("metadata", {}) if isinstance(run, Mapping) else {}
    cfg = dict(meta_run.get("cfg", {})) if isinstance(meta_run, Mapping) else {}

    if base_path is None:
        base_path = meta_run.get("base_path", None)
    if base_path is None:
        raise ValueError("base_path must be supplied if run['metadata']['base_path'] is missing.")

    if snap is None:
        snap = int(meta_run.get("snap", cfg.get("snap", 99)))
    snap = int(snap)

    sim_name = str(meta_run.get("sim_name", cfg.get("sim_name", DEFAULT_CFG.get("sim_name", "TNG50-1"))))
    api_key = cfg.get("api_key", os.environ.get("TNG_API_KEY"))

    header = meta_run.get("header", None)
    if header is None:
        header = read_header_for_snap(base_path, snap, sim_name=sim_name, api_key=api_key)

    group_fields_use = list(
        group_fields
        or [
            "GroupFirstSub",
            "GroupNsubs",
            "GroupLenType",
            "GroupMass",
            "GroupMassType",
            "GroupPos",
            "GroupVel",
            "Group_R_Crit200",
            "Group_M_Crit200",
            "Group_R_Mean200",
            "Group_M_Mean200",
            "Group_R_TopHat200",
            "Group_M_TopHat200",
        ]
    )
    subhalo_fields_use = list(
        subhalo_fields
        or [
            "SubhaloGrNr",
            "SubhaloLenType",
            "SubhaloMass",
            "SubhaloMassType",
            "SubhaloPos",
            "SubhaloVel",
            "SubhaloVmax",
            "SubhaloHalfmassRadType",
            "SubhaloFlag",
        ]
    )

    cat_kwargs = default_tng_catalog_kwargs(
        sim_name=sim_name,
        api_key=api_key,
        download_if_missing=bool(cfg.get("download_if_missing", True)),
        delete_cache=bool(cfg.get("delete_cache", True)),
        cache_dir=cfg.get("cache_dir", None),
        verbose=bool(cfg.get("verbose", True)),
        timeout=int(cfg.get("timeout", 180)),
    )

    if tng_catalog_kwargs is not None:
        cat_kwargs.update(dict(tng_catalog_kwargs))

    retry_cfg = {
        "max_retries": int(cfg.get("api_max_retries", 6)),
        "base_sleep": float(cfg.get("api_retry_base_sleep", 5.0)),
        "max_sleep": float(cfg.get("api_retry_max_sleep", 90.0)),
        "verbose": bool(cfg.get("verbose", True)),
    }

    cat = None
    try:
        cat, halos, subs = open_catalog(
            base_path,
            snap,
            group_fields=group_fields_use,
            subhalo_fields=subhalo_fields_use,
            tng_catalog_kwargs=cat_kwargs,
            retry_cfg=retry_cfg,
        )

        for res in run.get("results", []):
            sid = _infer_sid_from_result(res)
            if sid is None:
                continue

            sub_info = res.setdefault("Sub_info", {})
            gid_input = sub_info.get("GroupID", None) if isinstance(sub_info, Mapping) else None

            md = _metadata_for_one_subhalo(
                halos,
                subs,
                sid,
                header=header,
                gid_input=gid_input,
            )

            sub_info.update(md)
            sub_info["SubhaloID"] = int(sid)

        subhalo_metadata = _build_subhalo_metadata_table(run.get("results", []))
        run["subhalo_metadata"] = subhalo_metadata

        if "closure_all" in run and isinstance(run["closure_all"], pd.DataFrame):
            run["closure_all"] = _attach_metadata_to_closure(run["closure_all"], subhalo_metadata)

        return subhalo_metadata

    finally:
        if cat is not None:
            try:
                cat.cleanup()
            except Exception:
                pass

            if cat in _OPEN_CATALOGS:
                _OPEN_CATALOGS.remove(cat)


# -----------------------------------------------------------------------------
# Cross-time finite-difference pattern speed
# -----------------------------------------------------------------------------


def load_sublink_mpb(
    base_path: Union[str, Path],
    snap: int,
    subhalo_id: int,
    fields: Optional[Sequence[str]] = None,
    *,
    tng_catalog_kwargs: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, np.ndarray]:
    """
    Load the SubLink main progenitor branch.

    Priority
    --------
    1. Try local ``illustris_python.sublink.loadTree``.
    2. If unavailable/failed, try ``TNGCatalog.loadMergerTree(..., onlyMPB=True)``.
       This enables API fallback and persistent cache-first behaviour when the
       active ``TNGCatLoader`` supports it.

    Notes
    -----
    ``fields`` is kept compatible with ``illustris_python`` and the TNG API
    tree cutout HDF5 reader.
    """
    if fields is None:
        fields = [
            "SnapNum",
            "SubfindID",
            "SubhaloID",
            "SubhaloGrNr",
            "SubhaloMass",
            "SubhaloMassType",
            "SubhaloPos",
            "SubhaloVel",
        ]

    local_exc = None

    # 1. Local tree files through illustris_python.
    try:
        import illustris_python as il  # type: ignore

        sublink = getattr(il, "sublink", None)
        if sublink is None:
            import illustris_python.sublink as sublink  # type: ignore

        tree = sublink.loadTree(
            str(base_path),
            int(snap),
            int(subhalo_id),
            fields=list(fields),
            onlyMPB=True,
        )

        if tree is not None and len(tree) > 0:
            return tree

    except Exception as exc:
        local_exc = exc

    # 2. TNGCatLoader API/cache fallback.
    cat = None
    try:
        cat_kwargs = dict(tng_catalog_kwargs or {})
        cat = TNGCatalog(str(base_path), int(snap), **cat_kwargs)
        _OPEN_CATALOGS.append(cat)

        if not hasattr(cat, "loadMergerTree"):
            raise AttributeError("TNGCatalog.loadMergerTree(...) is not available.")

        tree = cat.loadMergerTree(
            sid=int(subhalo_id),
            tree_name="sublink",
            onlyMPB=True,
            fields=list(fields),
        )

        if tree is not None and len(tree) > 0:
            return tree

    except Exception as api_exc:
        raise RuntimeError(
            "Could not load SubLink MPB. Local illustris_python loading failed "
            f"with {local_exc!r}; API/cache fallback failed with {api_exc!r}."
        ) from api_exc

    finally:
        if cat is not None:
            try:
                cat.cleanup()
            except Exception:
                pass

            if cat in _OPEN_CATALOGS:
                _OPEN_CATALOGS.remove(cat)

    raise RuntimeError(f"No SubLink MPB returned for snap={snap}, subhalo={subhalo_id}")


def tree_to_dataframe(tree: Mapping[str, Any]) -> pd.DataFrame:
    data = {}
    for k, v in tree.items():
        arr = np.asarray(v)
        if arr.ndim == 1:
            data[k] = arr
    return pd.DataFrame(data)


def select_tree_rows_for_snaps(tree_df: pd.DataFrame, snap_list: Sequence[int]) -> pd.DataFrame:
    rows = []
    for snap in snap_list:
        hit = tree_df.loc[tree_df["SnapNum"].astype(int) == int(snap)]
        if len(hit):
            rows.append(hit.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()


def _proper_rotation_frame(R: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Project a 3x3 axis frame to the nearest right-handed SO(3) frame.

    Principal-axis eigenvectors are sign-degenerate; this fixes the geometric
    convention needed by SO(3) logarithms without changing the physical axes.
    """
    R = np.asarray(R, dtype=np.float64)

    if R.shape != (3, 3) or not np.all(np.isfinite(R)):
        raise ValueError("R must be a finite 3x3 matrix.")

    Q = R.copy()
    for j in range(3):
        n = np.linalg.norm(Q[:, j])
        if not np.isfinite(n) or n < eps:
            raise ValueError("Degenerate principal-axis frame.")
        Q[:, j] /= n

    if np.linalg.det(Q) < 0:
        Q[:, 2] *= -1.0

    U, _, Vt = np.linalg.svd(Q)
    Q = U @ Vt

    if np.linalg.det(Q) < 0:
        U[:, -1] *= -1.0
        Q = U @ Vt

    return Q.astype(np.float64)


def _align_basis_to_previous(R: np.ndarray, R_prev: np.ndarray) -> np.ndarray:
    """
    Align an unoriented principal-axis triad to the previous triad.

    Only sign flips with determinant +1 are allowed, so the result stays in
    SO(3).  This avoids left-handed frames in cross-time finite differences.
    """
    Rp = _proper_rotation_frame(R_prev)
    Rn = _proper_rotation_frame(R)

    dots = np.diag(Rp.T @ Rn)

    sign_candidates = np.array(
        [
            [ 1.0,  1.0,  1.0],
            [ 1.0, -1.0, -1.0],
            [-1.0,  1.0, -1.0],
            [-1.0, -1.0,  1.0],
        ],
        dtype=np.float64,
    )

    scores = sign_candidates @ dots
    best = sign_candidates[int(np.argmax(scores))]

    Rn = Rn @ np.diag(best)
    return _proper_rotation_frame(Rn)


def _rotation_log_so3(R0: np.ndarray, R1: np.ndarray) -> np.ndarray:
    """
    Robust SO(3) logarithm for principal-axis frames.
    """
    if Rotation is None:
        raise ImportError("scipy.spatial.transform.Rotation is required for SO(3) logarithms.")

    R0p = _proper_rotation_frame(R0)
    R1p = _align_basis_to_previous(R1, R0p)

    dR = R0p.T @ R1p
    dR = _proper_rotation_frame(dR)

    rotvec = Rotation.from_matrix(dR).as_rotvec()

    if hasattr(hd, "vector_to_skew"):
        return hd.vector_to_skew(rotvec)

    S = np.zeros((3, 3), dtype=float)
    S[0, 1] = -rotvec[2]
    S[1, 0] = rotvec[2]
    S[0, 2] = rotvec[1]
    S[2, 0] = -rotvec[1]
    S[1, 2] = -rotvec[0]
    S[2, 1] = rotvec[0]
    return S


def _basis_for_id_subset(pdata: Mapping[str, Any], id_subset: np.ndarray, min_particles: int) -> Optional[Dict[str, Any]]:
    ids = np.asarray(pdata["ids"], dtype=np.int64)
    take = np.isin(ids, np.asarray(id_subset, dtype=np.int64))
    if np.count_nonzero(take) < int(min_particles):
        return None
    I = hd.shape_tensor(np.asarray(pdata["X_kpc"])[take], masses=np.asarray(pdata["masses"])[take])
    evals, R = hd.eigh_sorted_desc(I)
    return {"I": I, "evals": evals, "R": R, "mask": take, "N": int(np.count_nonzero(take))}


def cross_time_pattern_speed_for_subhalo(
    base_path: Union[str, Path],
    snap0: int,
    subhalo_id0: int,
    *,
    snap_track: Sequence[int],
    cfg: Optional[Mapping[str, Any]] = None,
    tng_catalog_kwargs: Optional[Mapping[str, Any]] = None,
    shell_method: str = "radial",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Finite-difference pattern-speed test along the main progenitor branch.

    Shell membership is defined at the later snapshot of each pair and matched
    by ParticleIDs to the earlier snapshot.
    """
    run_cfg = dict(DEFAULT_CFG)
    if cfg is not None:
        run_cfg.update(dict(cfg))
    sim_name = str(run_cfg.get("sim_name", "TNG50-1"))
    api_key = run_cfg.get("api_key", os.environ.get("TNG_API_KEY"))
    retry_cfg = {
        "max_retries": int(run_cfg.get("api_max_retries", 6)),
        "base_sleep": float(run_cfg.get("api_retry_base_sleep", 5.0)),
        "max_sleep": float(run_cfg.get("api_retry_max_sleep", 90.0)),
        "verbose": bool(run_cfg.get("verbose", True)),
    }
    cat_kwargs = default_tng_catalog_kwargs(
        sim_name=sim_name,
        api_key=api_key,
        download_if_missing=bool(run_cfg.get("download_if_missing", True)),
        delete_cache=bool(run_cfg.get("delete_cache", True)),
        cache_dir=run_cfg.get("cache_dir", None),
        verbose=bool(run_cfg.get("verbose", True)),
    )
    if tng_catalog_kwargs is not None:
        cat_kwargs.update(dict(tng_catalog_kwargs))

    tree = load_sublink_mpb(base_path, int(snap0), int(subhalo_id0))
    tree_df = tree_to_dataframe(tree)
    track = select_tree_rows_for_snaps(tree_df, snap_track)
    if len(track) < 2:
        raise RuntimeError("Need at least two available branch snapshots for a cross-time test.")
    track = track.sort_values("SnapNum").reset_index(drop=True)

    pdata_by_snap: Dict[int, Dict[str, Any]] = {}
    for _, row in track.iterrows():
        snap = int(row["SnapNum"])
        sid = int(row["SubfindID"])
        header = read_header_for_snap(base_path, snap, sim_name=sim_name, api_key=api_key)
        cat, halos, subs = open_catalog(base_path, snap, tng_catalog_kwargs=cat_kwargs, retry_cfg=retry_cfg)
        try:
            pdata_by_snap[snap] = load_subhalo_dm_particles(cat, subs, sid, snap=snap, base_path=base_path, header=header, retry_cfg=retry_cfg)
            pdata_by_snap[snap]["time_gyr"] = cosmic_time_gyr_from_header(header)
            pdata_by_snap[snap]["SubfindID"] = sid
        finally:
            cat.cleanup()
            if cat in _OPEN_CATALOGS:
                _OPEN_CATALOGS.remove(cat)

    rows = []
    snaps = list(track["SnapNum"].astype(int).values)
    for s0, s1 in zip(snaps[:-1], snaps[1:]):
        p0 = pdata_by_snap[int(s0)]
        p1 = pdata_by_snap[int(s1)]
        dt = float(p1["time_gyr"] - p0["time_gyr"])
        if not np.isfinite(dt) or dt <= 0:
            continue
        n_shells = int(run_cfg.get("n_binding_shells" if shell_method.startswith("binding") else "n_radial_shells", 5))
        masks1, _info = build_shell_masks_for_particles(
            p1,
            method="binding_energy" if shell_method.startswith("binding") else "radial",
            n_shells=n_shells,
            equal_number=bool(run_cfg.get("equal_number_radial_shells", True)),
            compute_binding_potential_if_missing=bool(run_cfg.get("compute_binding_potential_if_missing", True)),
        )
        for ish, mi in enumerate(masks1):
            ids_shell = np.asarray(p1["ids"], dtype=np.int64)[np.asarray(mi, dtype=bool)]
            ids_common = np.intersect1d(ids_shell, np.asarray(p0["ids"], dtype=np.int64), assume_unique=False)
            if ids_common.size < int(run_cfg.get("min_particles_per_shell", 100)):
                continue
            b0 = _basis_for_id_subset(p0, ids_common, int(run_cfg.get("min_particles_per_shell", 100)))
            b1 = _basis_for_id_subset(p1, ids_common, int(run_cfg.get("min_particles_per_shell", 100)))
            if b0 is None or b1 is None:
                continue
            R1 = _align_basis_to_previous(b1["R"], b0["R"])
            Pi_fd = _rotation_log_so3(b0["R"], R1) / dt
            kin1 = hd.compute_affine_kinematics(p1["X_kpc"], p1["U_kms"], masses=p1["masses"], center=np.zeros(3), v_ref=np.zeros(3), mask=mi, min_particles=int(run_cfg.get("min_particles_per_shell", 100)))
            fig1_aff = hd.figure_rotation_from_affine(kin1["I"], kin1["A"])
            Pi_aff = np.asarray(fig1_aff["Pi"], dtype=np.float64) * KM_S_PER_KPC_TO_GYR_INV

            if "dI" in kin1 and hasattr(hd, "figure_rotation_from_dI"):
                fig1_dir = hd.figure_rotation_from_dI(kin1["I"], kin1["dI"])
                Pi_dir = np.asarray(fig1_dir["Pi"], dtype=np.float64) * KM_S_PER_KPC_TO_GYR_INV
            else:
                Pi_dir = np.full((3, 3), np.nan, dtype=np.float64)

            rows.append(
                {
                    "snap_early": int(s0),
                    "snap_late": int(s1),
                    "sid_early": int(p0["SubfindID"]),
                    "sid_late": int(p1["SubfindID"]),
                    "dt_gyr": dt,
                    "shell": int(ish),
                    "N_common": int(ids_common.size),
                    "Pi_fd_01": float(Pi_fd[0, 1]),
                    "Pi_fd_02": float(Pi_fd[0, 2]),
                    "Pi_fd_12": float(Pi_fd[1, 2]),
                    "Pi_direct_late_01": float(Pi_dir[0, 1]),
                    "Pi_direct_late_02": float(Pi_dir[0, 2]),
                    "Pi_direct_late_12": float(Pi_dir[1, 2]),
                    "Pi_aff_late_01": float(Pi_aff[0, 1]),
                    "Pi_aff_late_02": float(Pi_aff[0, 2]),
                    "Pi_aff_late_12": float(Pi_aff[1, 2]),
                }
            )
    return pd.DataFrame(rows), track


__all__ = [
    "DEFAULT_CFG",
    "KM_S_PER_KPC_TO_GYR_INV",
    "cleanup_open_catalogs",
    "read_header_for_snap",
    "compute_haloes",
    "compute_many",
    "compute_one_subhalo",
    "select_subhaloes_in_top_groups",
    "load_subhalo_dm_particles",
    "analyse_particle_data",
    "closure_table_from_analysis",
    "enrich_run_with_group_metadata",
    "direct_pi_from_P",
    "cross_time_pattern_speed_for_subhalo",
    "load_sublink_mpb",
]
