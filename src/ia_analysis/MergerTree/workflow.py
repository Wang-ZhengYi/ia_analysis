"""Cross-time merger-tree workflow orchestration.

Purpose
-------
This module coordinates the steps needed by merger-tree notebooks.  It decides
which snapshot/subhalo pairs to process, then delegates all scientific work to
the existing catalog, shape, and dynamics modules.

Provides
--------
- Snapshot component loading for dark matter, stars, gas, and black holes.
- Shape measurement wrappers for loaded particle components.
- Halo shell-dynamics wrappers for loaded dark-matter particles.
- A standard compute-once cross-time product builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ia_analysis.MergerTree.reader import (
    build_main_progenitor_track,
    build_target_reference_tracks,
    find_host_central_at_snapshot,
)

PARTICLE_COMPONENTS = {
    "gas": 0,
    "dm": 1,
    "dark_matter": 1,
    "stars": 4,
    "star": 4,
    "bh": 5,
    "black_holes": 5,
}


@dataclass(frozen=True)
class SnapshotTask:
    """One snapshot/subhalo pair selected from a merger-tree track."""

    snap: int
    subhalo_id: int
    reference_subhalo_id: int | None = None
    row: Mapping[str, Any] | None = None


def _hd_tng():
    """Import the TNG dynamics helper module only when a workflow step runs."""
    from ia_analysis.dynamics import hd_tng

    return hd_tng


def _retry_cfg_from_config(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = dict(cfg or {})
    return {
        "max_retries": int(cfg.get("api_max_retries", 6)),
        "base_sleep": float(cfg.get("api_retry_base_sleep", 5.0)),
        "max_sleep": float(cfg.get("api_retry_max_sleep", 90.0)),
        "verbose": bool(cfg.get("verbose", True)),
    }


def _component_name(component: str) -> str:
    key = str(component).strip().lower()
    if key not in PARTICLE_COMPONENTS:
        valid = ", ".join(sorted(PARTICLE_COMPONENTS))
        raise KeyError(f"Unknown particle component {component!r}. Available components: {valid}")
    ptype = PARTICLE_COMPONENTS[key]
    if ptype == 0:
        return "gas"
    if ptype == 1:
        return "dm"
    if ptype == 4:
        return "stars"
    if ptype == 5:
        return "bh"
    return key


def _mass_from_tng_units(values: Any, header: Mapping[str, Any]) -> np.ndarray:
    h = float(header.get("HubbleParam", 0.6774))
    return np.asarray(values, dtype=float) * 1.0e10 / h


def load_particle_component(
    catalog: Any,
    subs: Mapping[str, np.ndarray],
    subhalo_id: int,
    *,
    ptype: int,
    snap: int,
    base_path: str | Path,
    header: Mapping[str, Any],
    retry_cfg: Mapping[str, Any] | None = None,
    fields: Sequence[str] | None = None,
    component_name: str | None = None,
) -> dict[str, Any]:
    """Load one particle component and convert it to relative physical units."""
    hd_tng = _hd_tng()
    ptype = int(ptype)
    component_name = component_name or f"PartType{ptype}"
    if fields is None:
        fields = ["Coordinates", "Velocities", "ParticleIDs"]
        if ptype in (0, 4, 5):
            fields = [*fields, "Masses"]
        if ptype == 0:
            fields = [*fields, "InternalEnergy"]

    try:
        block = hd_tng.retry_call(
            catalog.loadSubhalos,
            int(subhalo_id),
            ptypes=[ptype],
            fields=list(fields),
            **dict(retry_cfg or {}),
        )
        pdata = block.get(f"PartType{ptype}", {})
    except Exception:
        if ptype == 0 and "InternalEnergy" in fields:
            reduced_fields = [field for field in fields if field != "InternalEnergy"]
            block = hd_tng.retry_call(
                catalog.loadSubhalos,
                int(subhalo_id),
                ptypes=[ptype],
                fields=reduced_fields,
                **dict(retry_cfg or {}),
            )
            pdata = block.get(f"PartType{ptype}", {})
        else:
            raise

    if "Coordinates" not in pdata or "Velocities" not in pdata:
        return {
            "component": component_name,
            "ptype": ptype,
            "sid": int(subhalo_id),
            "snap": int(snap),
            "X_kpc": np.empty((0, 3), dtype=float),
            "U_kms": np.empty((0, 3), dtype=float),
            "coords_ckpc_h": np.empty((0, 3), dtype=float),
            "masses": np.empty(0, dtype=float),
            "ids": np.empty(0, dtype=np.int64),
        }

    coords = np.asarray(pdata["Coordinates"], dtype=float)
    velocities = np.asarray(pdata["Velocities"], dtype=float)
    center = np.asarray(subs["SubhaloPos"][int(subhalo_id)], dtype=float)
    v_ref = np.asarray(subs["SubhaloVel"][int(subhalo_id)], dtype=float)
    x_kpc = hd_tng.tng_relative_positions_to_physical_kpc(coords, center, header)
    u_kms = hd_tng.tng_velocity_to_kms(velocities, header) - hd_tng.tng_velocity_to_kms(v_ref[None, :], header)[0]

    if ptype == 1:
        masses = hd_tng.dm_mass_msun_from_header(header, coords.shape[0])
    elif "Masses" in pdata:
        masses = _mass_from_tng_units(pdata["Masses"], header)
    else:
        masses = np.ones(coords.shape[0], dtype=float)

    out = {
        "component": component_name,
        "ptype": ptype,
        "sid": int(subhalo_id),
        "snap": int(snap),
        "X_kpc": x_kpc,
        "U_kms": u_kms,
        "coords_ckpc_h": coords,
        "vel_code": velocities,
        "center_ckpc_h": center,
        "v_ref_code": v_ref,
        "masses": np.asarray(masses, dtype=float),
        "ids": np.asarray(pdata.get("ParticleIDs", np.arange(coords.shape[0])), dtype=np.int64),
    }
    if "InternalEnergy" in pdata:
        out["internal_energy"] = np.asarray(pdata["InternalEnergy"], dtype=float)
    return out


def load_snapshot_components(
    base_path: str | Path,
    snap: int,
    subhalo_id: int,
    *,
    components: Sequence[str] = ("dm", "stars"),
    sim_name: str = "TNG300-1",
    api_key: str | None = None,
    cfg: Mapping[str, Any] | None = None,
    tng_catalog_kwargs: Mapping[str, Any] | None = None,
    group_fields: Sequence[str] | None = None,
    subhalo_fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load requested particle components for one track snapshot."""
    hd_tng = _hd_tng()
    header = hd_tng.read_header_for_snap(base_path, int(snap), sim_name=sim_name, api_key=api_key)
    retry_cfg = _retry_cfg_from_config(cfg)
    group_fields = list(group_fields or ["GroupFirstSub", "GroupNsubs", "GroupLenType", "Group_R_Crit200", "GroupPos"])
    subhalo_fields = list(
        subhalo_fields
        or ["SubhaloGrNr", "SubhaloLenType", "SubhaloPos", "SubhaloVel", "SubhaloMassType", "SubhaloHalfmassRadType"]
    )
    cat, halos, subs = hd_tng.open_catalog(
        base_path,
        int(snap),
        group_fields=group_fields,
        subhalo_fields=subhalo_fields,
        tng_catalog_kwargs=tng_catalog_kwargs,
        retry_cfg=retry_cfg,
    )
    try:
        loaded: dict[str, Any] = {}
        for component in components:
            key = _component_name(component)
            ptype = PARTICLE_COMPONENTS[str(component).strip().lower()]
            if ptype == 1:
                loaded["dm"] = hd_tng.load_subhalo_dm_particles(
                    cat,
                    subs,
                    int(subhalo_id),
                    snap=int(snap),
                    base_path=base_path,
                    header=header,
                    retry_cfg=retry_cfg,
                )
                loaded["dm"]["component"] = "dm"
                loaded["dm"]["ptype"] = 1
            else:
                loaded[key] = load_particle_component(
                    cat,
                    subs,
                    int(subhalo_id),
                    ptype=ptype,
                    snap=int(snap),
                    base_path=base_path,
                    header=header,
                    retry_cfg=retry_cfg,
                    component_name=key,
                )

        gid = int(np.asarray(subs["SubhaloGrNr"], dtype=np.int64)[int(subhalo_id)])
        return {
            "snap": int(snap),
            "subhalo_id": int(subhalo_id),
            "group_id": gid,
            "header": dict(header),
            "components": loaded,
            "subhalo": {field: np.asarray(values)[int(subhalo_id)] for field, values in subs.items()},
            "group": {field: np.asarray(values)[gid] for field, values in halos.items()},
        }
    finally:
        try:
            cat.cleanup()
        except Exception:
            pass


def measure_component_shape(
    component: Mapping[str, Any],
    *,
    min_particles: int = 20,
    shape_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure one component's shape by calling the shapes API."""
    positions = np.asarray(component.get("X_kpc", np.empty((0, 3))), dtype=float)
    masses = np.asarray(component.get("masses", np.empty(0)), dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < int(min_particles):
        return {"ok": False, "N": int(positions.shape[0] if positions.ndim == 2 else 0)}
    from ia_analysis.shapes.api import measure_iterative_shape

    payload = measure_iterative_shape(positions, masses=masses, **dict(shape_kwargs or {}))
    payload["ok"] = True
    payload["N"] = int(positions.shape[0])
    return payload


def measure_snapshot_shapes(
    snapshot: Mapping[str, Any],
    *,
    components: Sequence[str] = ("dm", "stars"),
    min_particles: int = 20,
    shape_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Measure shapes for selected loaded components."""
    loaded = snapshot.get("components", {})
    shapes: dict[str, dict[str, Any]] = {}
    for component in components:
        key = _component_name(component)
        if key in loaded:
            shapes[key] = measure_component_shape(
                loaded[key],
                min_particles=min_particles,
                shape_kwargs=shape_kwargs,
            )
    return shapes


def analyze_snapshot_shells(
    snapshot: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any],
    shell_methods: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run halo shell dynamics for the loaded dark-matter component."""
    dm = snapshot.get("components", {}).get("dm")
    if dm is None:
        return {}
    return _hd_tng().analyse_particle_data(dm, cfg=cfg, shell_methods=shell_methods)


def process_snapshot_task(
    task: SnapshotTask,
    *,
    base_path: str | Path,
    sim_name: str = "TNG300-1",
    api_key: str | None = None,
    cfg: Mapping[str, Any] | None = None,
    tng_catalog_kwargs: Mapping[str, Any] | None = None,
    components: Sequence[str] = ("dm", "stars"),
    shape_components: Sequence[str] = ("dm", "stars"),
    shell_methods: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load one snapshot and run requested shape and shell-dynamics steps."""
    cfg = dict(cfg or _hd_tng().DEFAULT_CFG)
    snapshot = load_snapshot_components(
        base_path,
        task.snap,
        task.subhalo_id,
        components=components,
        sim_name=sim_name,
        api_key=api_key,
        cfg=cfg,
        tng_catalog_kwargs=tng_catalog_kwargs,
    )
    snapshot["reference_subhalo_id"] = task.reference_subhalo_id
    snapshot["track_row"] = dict(task.row or {})
    snapshot["shapes"] = measure_snapshot_shapes(
        snapshot,
        components=shape_components,
        min_particles=int(cfg.get("min_particles_for_shape", cfg.get("min_particles_per_shell", 100))),
    )
    snapshot["shells"] = analyze_snapshot_shells(snapshot, cfg=cfg, shell_methods=shell_methods)
    return snapshot


def cross_time_pattern_speed_for_subhalo(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute finite-difference pattern-speed closure through the dynamics backend."""
    return _hd_tng().cross_time_pattern_speed_for_subhalo(*args, **kwargs)


def run_cross_time_workflow(
    base_path: str | Path,
    snap0: int,
    subhalo_id0: int,
    snap_track: Sequence[int],
    *,
    sim_name: str = "TNG300-1",
    api_key: str | None = None,
    cfg: Mapping[str, Any] | None = None,
    tng_catalog_kwargs: Mapping[str, Any] | None = None,
    components: Sequence[str] = ("dm", "stars"),
    shape_components: Sequence[str] = ("dm", "stars"),
    shell_methods: Sequence[str] = ("radial", "binding_energy"),
    reference: str | int | None = "host_central",
    compute_closure: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the standard merger-tree cross-time reading and analysis workflow."""
    cfg = dict(cfg or _hd_tng().DEFAULT_CFG)
    if tng_catalog_kwargs is None:
        tng_catalog_kwargs = _hd_tng().default_tng_catalog_kwargs(
            sim_name=sim_name,
            api_key=api_key,
            download_if_missing=bool(cfg.get("download_if_missing", True)),
            delete_cache=bool(cfg.get("delete_cache", True)),
            cache_dir=cfg.get("cache_dir", None),
            verbose=bool(cfg.get("verbose", verbose)),
        )

    reference_id: int | None
    if reference == "host_central":
        ref_info = find_host_central_at_snapshot(
            base_path,
            int(snap0),
            int(subhalo_id0),
            tng_catalog_kwargs=tng_catalog_kwargs,
            retry_cfg=_retry_cfg_from_config(cfg),
        )
        reference_id = int(ref_info["central_subhalo_id"])
    elif reference is None:
        ref_info = None
        reference_id = None
    else:
        ref_info = {"central_subhalo_id": int(reference)}
        reference_id = int(reference)

    if reference_id is None:
        target_track = build_main_progenitor_track(
            base_path,
            snap0,
            subhalo_id0,
            snap_track,
            tng_catalog_kwargs=tng_catalog_kwargs,
        )
        tasks = [
            SnapshotTask(int(row["SnapNum"]), int(row["SubfindID"]), row=row.to_dict())
            for _, row in target_track.iterrows()
        ]
        track_table = target_track
    else:
        track_table = build_target_reference_tracks(
            base_path,
            snap0,
            subhalo_id0,
            reference_id,
            snap_track,
            tng_catalog_kwargs=tng_catalog_kwargs,
        )
        tasks = [
            SnapshotTask(
                int(row["SnapNum"]),
                int(row["TargetSubfindID"]),
                reference_subhalo_id=int(row["ReferenceSubfindID"]),
                row=row.to_dict(),
            )
            for _, row in track_table.iterrows()
        ]

    records = []
    for task in tasks:
        if verbose:
            print(f"[MergerTree] snap={task.snap}, subhalo={task.subhalo_id}")
        records.append(
            process_snapshot_task(
                task,
                base_path=base_path,
                sim_name=sim_name,
                api_key=api_key,
                cfg=cfg,
                tng_catalog_kwargs=tng_catalog_kwargs,
                components=components,
                shape_components=shape_components,
                shell_methods=shell_methods,
            )
        )

    closures: dict[str, Any] = {}
    if compute_closure:
        for method in shell_methods:
            try:
                closure, closure_track = cross_time_pattern_speed_for_subhalo(
                    base_path,
                    int(snap0),
                    int(subhalo_id0),
                    snap_track=tuple(int(s) for s in snap_track),
                    cfg=cfg,
                    tng_catalog_kwargs=tng_catalog_kwargs,
                    shell_method=str(method),
                )
                closures[str(method)] = {"closure": closure, "track": closure_track}
            except Exception as exc:
                closures[str(method)] = {"closure": pd.DataFrame(), "track": pd.DataFrame(), "error": repr(exc)}

    return {
        "snap0": int(snap0),
        "subhalo_id0": int(subhalo_id0),
        "snap_track": tuple(int(s) for s in snap_track),
        "reference": ref_info,
        "track": track_table,
        "tasks": tasks,
        "records": records,
        "closures": closures,
        "tng_catalog_kwargs": dict(tng_catalog_kwargs or {}),
    }
