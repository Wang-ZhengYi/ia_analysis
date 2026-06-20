#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TNGCatLoader.py

Standalone IllustrisTNG loader with CSCatalog-compatible methods:

    cat = TNGCatalog(base_path, snap, sim_name="TNG300-1", api_key="...",
                     download_if_missing=None, delete_cache=True, verbose=True)
    halos, subs = cat.loadFoF(group_fields=[...], subhalo_fields=[...])
    p = cat.loadSubhalos(sid=123, ptypes=[1, 4], fields=[...])
    q = cat.loadHalos(gid=10, ptypes=[0, 1, 4, 5], fields=[...])
    cat.cleanup()

Design
------
1. Prefer local files whenever they exist.
2. For local group catalogs and particle data, try ``illustris_python`` first.
   This is the most reliable path for official TNG split files.
3. If ``illustris_python`` is unavailable or fails, use the internal local HDF5
   reader as a local-only fallback.
4. Only if the local directory is absent and API download is enabled, download
   the requested group-catalog fields or halo/subhalo cutouts from the API.
5. API downloads use deterministic persistent cache filenames in cache_dir: if a readable HDF5 cache file exists it is reused; otherwise it is downloaded.
6. Downloaded files are temporary only when delete_cache=True; set delete_cache=False to preserve cache across runs.
7. Download progress is printed by default.

Notes
-----
- Full local TNG layout is supported through groups_XXX/, snapdir_XXX/, and
  offsets_XXX.hdf5 or postprocessing/offsets/offsets_XXX.hdf5.
- The API fallback requires a TNG API key, either passed explicitly or via
  the TNG_API_KEY environment variable.
- API fallback for group catalogs uses the one-field-at-a-time subset endpoint.
  This avoids downloading whole groupcat chunks.
- API fallback for particles uses halo/subhalo cutouts. This avoids downloading
  whole snapshot chunks.
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

import h5py

logger = logging.getLogger(__name__)

PTYPE_TO_API = {
    0: "gas",
    1: "dm",
    4: "stars",
    5: "bhs",
}
API_TO_PTYPE = {v: k for k, v in PTYPE_TO_API.items()}

_VEC3_FIELDS = {
    "Coordinates",
    "Velocities",
    "Acceleration",
    "Accelerations",
    "ModifiedGravityAcceleration",
}
_OPTIONAL_ZERO_VEC3_FIELDS = {
    "ModifiedGravityAcceleration",
}
_OPTIONAL_ZERO_SCALAR_FIELDS = {
    "Potential",
    "ModifiedGravityPotential",
}


def _hdf5_chunk_sort_key(filepath: str) -> tuple:
    """Sort split HDF5 chunks by numeric suffix, not lexicographically."""
    base = os.path.basename(str(filepath))

    # Standard split-file pattern: snap_099.10.hdf5, fof_subhalo_tab_099.10.hdf5.
    m = re.search(r"\.(\d+)\.hdf5$", base)
    if m is not None:
        return (0, int(m.group(1)), base)

    # Fallback for names ending in _10.hdf5.
    m = re.search(r"_(\d+)\.hdf5$", base)
    if m is not None:
        return (1, int(m.group(1)), base)

    # Last fallback: natural sorting.
    parts = re.split(r"(\d+)", base)
    natural_parts = tuple(int(p) if p.isdigit() else p for p in parts)
    return (2, natural_parts, base)



def _hdf5_chunk_id(filepath: str) -> Optional[int]:
    base = os.path.basename(str(filepath))
    m = re.search(r"\.(\d+)\.hdf5$", base)
    if m is not None:
        return int(m.group(1))
    m = re.search(r"_(\d+)\.hdf5$", base)
    if m is not None:
        return int(m.group(1))
    return None


def _check_split_file_completeness(files: Sequence[str], attr_names: Sequence[str]) -> None:
    """Raise if Header says split files are incomplete and chunk ids reveal a gap."""
    if not files:
        return
    n_expected = None
    try:
        with h5py.File(files[0], "r") as f:
            header = f.get("Header", None)
            if header is not None:
                for attr in attr_names:
                    if attr in header.attrs:
                        n_expected = int(header.attrs[attr])
                        break
    except Exception:
        return

    if n_expected is None or n_expected <= 0:
        return

    ids = [_hdf5_chunk_id(fp) for fp in files]
    if any(x is None for x in ids):
        if len(files) < n_expected:
            raise FileNotFoundError(
                f"Local split HDF5 set is incomplete: found {len(files)} files, expected {n_expected}."
            )
        return

    have = set(int(x) for x in ids if x is not None)
    want = set(range(n_expected))
    missing = sorted(want - have)
    if missing:
        raise FileNotFoundError(
            f"Local split HDF5 set is incomplete: missing chunk ids {missing[:20]}"
            + (" ..." if len(missing) > 20 else "")
        )

def _unique(xs: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(text)).strip("_")


class TNGCatalog:
    """
    IllustrisTNG catalog/particle loader.

    Parameters
    ----------
    base_path : str
        Local TNG base path. If local files exist, they are used first.
    snap : int
        Snapshot number.
    sim_name : str
        TNG API simulation name, e.g. "TNG300-1".
    api_key : str or None
        TNG API key. If None, ``TNG_API_KEY`` from the environment is used.
    download_if_missing : bool or None
        If True, missing local data are fetched from the TNG API. If None,
        this is enabled automatically when an API key is available.
    cache_dir : str or None
        Directory for temporary API files. If None, a private temp directory is
        created.
    delete_cache : bool
        Delete API-downloaded files after use. Default True.
    verbose : bool
        Print download progress. Default True.
    use_illustris_python : bool
        If True, local group catalogs and particles are read with the official
        ``illustris_python`` helpers before trying the internal HDF5 reader.
        Default True.
    prefer_cutout : bool
        For missing particle data, use halo/subhalo cutouts instead of snapshot
        chunks. Default True.
    """

    def __init__(
        self,
        base_path: str,
        snap: int,
        *,
        sim_name: Optional[str] = None,
        api_key: Optional[str] = None,
        download_if_missing: Optional[bool] = None,
        cache_dir: Optional[str] = None,
        delete_cache: bool = True,
        verbose: bool = True,
        use_illustris_python: bool = True,
        prefer_cutout: bool = True,
        api_base_url: str = "https://www.tng-project.org/api",
        timeout: int = 120,
        max_retries: int = 5,
        retry_base_sleep: float = 3.0,
        retry_max_sleep: float = 60.0,
    ):
        self.base_path = str(base_path)
        self.snap = int(snap)
        self.sim_name = str(sim_name or os.environ.get("TNG_SIM_NAME", "TNG300-1"))
        self.api_key = api_key if api_key is not None else os.environ.get("TNG_API_KEY")
        self.download_if_missing = bool(self.api_key) if download_if_missing is None else bool(download_if_missing)
        self.delete_cache = bool(delete_cache)
        self.verbose = bool(verbose)
        self.use_illustris_python = bool(use_illustris_python)
        self.prefer_cutout = bool(prefer_cutout)
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout = int(timeout)
        self.max_retries = int(max_retries)
        self.retry_base_sleep = float(retry_base_sleep)
        self.retry_max_sleep = float(retry_max_sleep)

        self.halos: Optional[Dict[str, np.ndarray]] = None
        self.subhalos: Optional[Dict[str, np.ndarray]] = None

        self._snap_index: Dict[int, Tuple[List[str], np.ndarray]] = {}
        self._offsets_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._file_offsets_snap: Optional[np.ndarray] = None
        self._cutout_cache: Dict[Tuple[str, int, Tuple[Tuple[int, Tuple[str, ...]], ...]], str] = {}
        self._downloaded_files: List[str] = []
        self._closed = False

        if cache_dir is None:
            self._owned_cache_dir = tempfile.mkdtemp(prefix=f"tng_api_{self.sim_name}_snap{self.snap:03d}_")
            self.cache_dir = self._owned_cache_dir
        else:
            self._owned_cache_dir = None
            self.cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
            os.makedirs(self.cache_dir, exist_ok=True)

        atexit.register(self.cleanup)

    # ------------------------------------------------------------------
    # Context manager and cleanup
    # ------------------------------------------------------------------

    def __enter__(self) -> "TNGCatalog":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()

    def close(self) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """Delete API-downloaded temporary files unless delete_cache=False."""
        if self._closed:
            return
        self._closed = True

        if not self.delete_cache:
            return

        # Remove individually tracked files first. If cache_dir is user-owned,
        # this avoids deleting unrelated files.
        for fp in list(dict.fromkeys(self._downloaded_files)):
            try:
                if fp and os.path.isfile(fp):
                    os.remove(fp)
            except Exception as e:
                logger.debug("Failed to remove temporary TNG API file %s: %s", fp, e)

        # Remove the private temp directory if we created it.
        if self._owned_cache_dir and os.path.isdir(self._owned_cache_dir):
            try:
                shutil.rmtree(self._owned_cache_dir, ignore_errors=True)
            except Exception as e:
                logger.debug("Failed to remove temporary TNG API directory %s: %s", self._owned_cache_dir, e)

    def __del__(self):  # pragma: no cover - best-effort cleanup only
        try:
            self.cleanup()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public catalog loaders
    # ------------------------------------------------------------------

    def loadFoF(self, group_fields: Sequence[str], subhalo_fields: Sequence[str]):
        """
        Load FoF group and subhalo catalogs.

        If local groupcat chunks exist, this reads them directly. If not, and
        API download is enabled, it downloads only the requested fields from the
        groupcat subset API, one field at a time.
        """
        req_g = ["GroupFirstSub", "GroupNsubs", "GroupLenType"]
        req_s = ["SubhaloGrNr", "SubhaloLenType"]
        g_fields = _unique(list(group_fields or []) + req_g)
        computed = {"SubhaloID", "GroupID", "CenID", "SubhaloOffsetType"}
        s_fields = [x for x in _unique(list(subhalo_fields or []) + req_s) if x not in computed]

        if self._has_local_groupcat():
            # Important policy: if the local group catalog directory exists, do
            # not fall back to the API. First try the official illustris_python
            # reader, then the internal HDF5 reader. If both local readers fail,
            # raise the local error so the bad/mismatched local files are visible.
            halos, subs = self._load_fof_local(g_fields, s_fields)
        else:
            self._require_api("local group catalog is missing")
            halos, subs = self._load_fof_api_fields(g_fields, s_fields)

        nsub = int(subs["SubhaloLenType"].shape[0])
        subs["SubhaloID"] = np.arange(nsub, dtype=np.int64)

        gid = np.asarray(subs["SubhaloGrNr"], dtype=np.int64)
        subs["GroupID"] = gid

        first = np.asarray(halos["GroupFirstSub"], dtype=np.int64)
        cen = np.full(nsub, -1, dtype=np.int64)
        ok = (gid >= 0) & (gid < first.shape[0])
        cen[ok] = first[gid[ok]]
        subs["CenID"] = cen

        # Offsets are useful for the internal local snapshot slicer. They are
        # not needed to load the FoF catalog itself. Do not download offsets here:
        # if local catalogs exist, this method must remain fully local.
        local_offsets = self._find_local_offsets_file()
        if local_offsets is not None:
            try:
                off_sub, off_grp = self._read_offsets_file()
                subs["SubhaloOffsetType"] = off_sub
                halos["_GroupOffsetType"] = off_grp
            except Exception as e:
                logger.info(
                    "Local offsets file could not be read (%s). FoF catalog loading continues; "
                    "particle-level local slicing may require offsets later.",
                    e,
                )
                subs["SubhaloOffsetType"] = np.zeros((nsub, 6), dtype=np.int64)
                halos["_GroupOffsetType"] = self._compute_group_offsets(np.asarray(halos["GroupLenType"], dtype=np.int64))
        else:
            logger.info(
                "Local offsets file not found. FoF catalog loading continues; "
                "particle-level local slicing may require offsets later."
            )
            subs["SubhaloOffsetType"] = np.zeros((nsub, 6), dtype=np.int64)
            halos["_GroupOffsetType"] = self._compute_group_offsets(np.asarray(halos["GroupLenType"], dtype=np.int64))

        self.halos = halos
        self.subhalos = subs
        return halos, subs

    def loadSubhalos(self, sid: int, ptypes: Sequence[int], fields: Sequence[str]):
        """Load particle fields for one subhalo."""
        self._require_fof_loaded()
        sid = int(sid)

        # Important policy: if the local snapdir exists, do not fall back to
        # the API. Prefer illustris_python, then the internal offset-based local
        # slicer. API cutouts are only used when no local snapdir exists.
        if self._has_local_snapshots():
            return self._load_particles_local(kind="subhalo", object_id=sid, ptypes=ptypes, fields=fields)

        self._require_api("local snapshot files are missing")
        return self._load_cutout(kind="subhalo", object_id=sid, ptypes=ptypes, fields=fields)

    def loadHalos(self, ptypes: Sequence[int], fields: Sequence[str], gid: Optional[int] = None, sid: Optional[int] = None):
        """Load particle fields for one FoF halo."""
        self._require_fof_loaded()
        if gid is None:
            if sid is None:
                raise ValueError("loadHalos: provide gid or sid.")
            gid = int(self.subhalos["GroupID"][int(sid)])
        else:
            gid = int(gid)

        if self._has_local_snapshots():
            return self._load_particles_local(kind="halo", object_id=gid, ptypes=ptypes, fields=fields)

        self._require_api("local snapshot files are missing")
        return self._load_cutout(kind="halo", object_id=gid, ptypes=ptypes, fields=fields)

    def loadMergerTree(
        self,
        sid: Optional[int] = None,
        *,
        subhalo_id: Optional[int] = None,
        subfind_id: Optional[int] = None,
        fields: Optional[Sequence[str]] = None,
        tree_name: str = "sublink",
        treeName: Optional[str] = None,
        onlyMPB: bool = True,
        only_mpb: Optional[bool] = None,
        full_tree: bool = False,
        mode: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Load an object-level merger tree through the IllustrisTNG API.

        This is intentionally an object-level fallback, not a raw tree-chunk
        reader.  It is designed for workflows such as hd_tng.load_sublink_mpb(),
        where a single subhalo at ``self.snap`` needs its SubLink MPB or full
        tree.  If the file is already present in ``cache_dir``, it is reused;
        otherwise it is downloaded with the existing retry/progress machinery.

        Parameters
        ----------
        sid, subhalo_id, subfind_id : int
            Subfind ID at ``self.snap``.  ``sid`` is kept for compatibility with
            the rest of this loader and with CSCatalog-style calls.
        fields : sequence of str or None
            Datasets to return.  If None, all datasets in the downloaded HDF5
            tree are returned.
        tree_name / treeName : str
            Currently supports "sublink" and "lhalotree" API endpoints.  The
            default is "sublink".
        onlyMPB / only_mpb : bool
            If True, download ``mpb.hdf5``.  If False, download ``full.hdf5``.
        full_tree : bool
            Convenience flag equivalent to ``onlyMPB=False``.
        mode : {"mpb", "full"} or None
            Explicit tree mode.  Overrides ``onlyMPB``/``full_tree`` if given.

        Returns
        -------
        dict
            Mapping from dataset names to numpy arrays.
        """
        # Accept common aliases used by Illustris/TNG helper styles.
        if sid is None:
            sid = subhalo_id
        if sid is None:
            sid = subfind_id
        if sid is None:
            sid = kwargs.pop("id", None)
        if sid is None:
            raise ValueError("loadMergerTree requires sid=..., subhalo_id=..., or subfind_id=...")
        sid = int(sid)

        if treeName is not None:
            tree_name = treeName
        if only_mpb is not None:
            onlyMPB = bool(only_mpb)
        if "onlyMPB" in kwargs:
            onlyMPB = bool(kwargs.pop("onlyMPB"))
        if "only_mpb" in kwargs:
            onlyMPB = bool(kwargs.pop("only_mpb"))
        if "fullTree" in kwargs:
            full_tree = bool(kwargs.pop("fullTree"))
        if "full_tree" in kwargs:
            full_tree = bool(kwargs.pop("full_tree"))

        tree_key = str(tree_name).lower().replace("_", "").replace("-", "")
        if tree_key in ("sublink", "sublinkdm", "sublinkgal"):
            endpoint = "sublink"
        elif tree_key in ("lhalotree", "lhalo", "lh"):
            endpoint = "lhalotree"
        else:
            raise ValueError(f"Unsupported merger tree '{tree_name}'. Use 'sublink' or 'lhalotree'.")

        if mode is not None:
            mode_key = str(mode).lower()
        elif full_tree:
            mode_key = "full"
        else:
            mode_key = "mpb" if bool(onlyMPB) else "full"

        if mode_key in ("mpb", "main", "mainbranch", "main_progenitor", "mainprogenitor"):
            filename = "mpb.hdf5"
            mode_tag = "mpb"
        elif mode_key in ("full", "tree", "fulltree", "all"):
            filename = "full.hdf5"
            mode_tag = "full"
        else:
            raise ValueError("mode must be 'mpb' or 'full'")

        path = self._merger_tree_cache_path(endpoint=endpoint, sid=sid, mode_tag=mode_tag)
        url = (
            f"{self.api_base_url}/{self.sim_name}/snapshots/{self.snap}/"
            f"subhalos/{sid}/{endpoint}/{filename}"
        )

        if not self._is_valid_hdf5_file(path):
            self._require_api(f"local {endpoint} {mode_tag} merger tree is missing or invalid")

        path = self._cache_or_download_hdf5(
            url,
            path,
            label=f"{endpoint} {mode_tag} tree sid={sid} snap={self.snap}",
        )

        return self._read_merger_tree_hdf5(path, fields=fields)

    def _merger_tree_cache_path(self, *, endpoint: str, sid: int, mode_tag: str) -> str:
        fname = (
            f"{_safe_name(self.sim_name)}_snap{self.snap:03d}_subhalo{int(sid)}_"
            f"{_safe_name(endpoint)}_{_safe_name(mode_tag)}.hdf5"
        )
        return os.path.join(self.cache_dir, fname)

    def _read_merger_tree_hdf5(self, path: str, fields: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
        """Read selected datasets from a downloaded object-level tree HDF5 file."""
        out: Dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as f:
            dataset_names: List[str] = []

            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    dataset_names.append(name)

            f.visititems(visitor)

            if fields is None:
                wanted = dataset_names
            else:
                # Match either the exact HDF5 path or the basename.  The object
                # tree files normally store datasets at the root, but this makes
                # the reader robust to minor layout differences.
                by_base = {os.path.basename(name): name for name in dataset_names}
                wanted = []
                missing = []
                for field in fields:
                    field = str(field)
                    if field in f and isinstance(f[field], h5py.Dataset):
                        wanted.append(field)
                    elif field in by_base:
                        wanted.append(by_base[field])
                    else:
                        missing.append(field)
                if missing:
                    raise KeyError(
                        f"Merger tree file {path} is missing requested fields {missing}. "
                        f"Available fields include: {[os.path.basename(x) for x in dataset_names[:40]]}"
                    )

            for name in wanted:
                key = os.path.basename(name)
                out[key] = np.asarray(f[name][()])

        return out

    # ------------------------------------------------------------------
    # Local group catalog
    # ------------------------------------------------------------------

    def _get_illustris_modules(self):
        """
        Return (groupcat_module, snapshot_module, import_error).

        The official TNG examples normally use::

            import illustris_python as il
            il.groupcat.loadHalos(...)
            il.snapshot.loadSubhalo(...)

        Some installations expose the submodules only as
        ``illustris_python.groupcat`` and ``illustris_python.snapshot``. This
        helper supports both layouts without requiring the caller to care.
        """
        try:
            import illustris_python as il  # type: ignore

            groupcat = getattr(il, "groupcat", None)
            snapshot = getattr(il, "snapshot", None)
            if groupcat is None:
                import illustris_python.groupcat as groupcat  # type: ignore
            if snapshot is None:
                import illustris_python.snapshot as snapshot  # type: ignore
            return groupcat, snapshot, None
        except Exception as e:  # pragma: no cover - depends on user's environment
            return None, None, e

    def _has_local_groupcat(self) -> bool:
        try:
            return len(self._list_hdf5_files(prefix="groups")) > 0
        except FileNotFoundError:
            return False

    def _load_fof_local(self, group_fields: Sequence[str], subhalo_fields: Sequence[str]):
        """
        Load local FoF catalogs.

        Priority:
        1. ``illustris_python.groupcat`` if available.
        2. Internal split-HDF5 concatenation as a local-only fallback.

        This method intentionally never calls the API. If local files exist but
        both local readers fail, the raised error exposes the local-file problem.
        """
        errors = []

        if self.use_illustris_python:
            try:
                return self._load_fof_local_illustris(group_fields, subhalo_fields)
            except Exception as e:
                errors.append(("illustris_python.groupcat", e))
                logger.warning("illustris_python groupcat read failed; trying internal local HDF5 reader. Reason: %s", e)

        try:
            return self._load_fof_local_hdf5(group_fields, subhalo_fields)
        except Exception as e:
            if errors:
                detail = "; ".join(f"{name}: {err}" for name, err in errors)
                raise RuntimeError(
                    "Local group catalog read failed. Tried illustris_python first and internal HDF5 second. "
                    f"Earlier local-reader errors: {detail}. Final internal-reader error: {e}"
                ) from e
            raise

    def _load_fof_local_illustris(self, group_fields: Sequence[str], subhalo_fields: Sequence[str]):
        groupcat, _snapshot, import_error = self._get_illustris_modules()
        if groupcat is None:
            raise ImportError(
                "illustris_python is not importable. Install/put illustris_python on PYTHONPATH, "
                "or rely on the internal local HDF5 fallback."
            ) from import_error

        if self.verbose:
            print("[TNG local] reading group catalog with illustris_python.groupcat", file=sys.stderr)

        halos_raw = groupcat.loadHalos(self.base_path, self.snap, fields=list(group_fields))
        subs_raw = groupcat.loadSubhalos(self.base_path, self.snap, fields=list(subhalo_fields))

        halos = self._normalise_illustris_catalog_result(halos_raw, group_fields, "Group")
        subs = self._normalise_illustris_catalog_result(subs_raw, subhalo_fields, "Subhalo")
        return halos, subs

    def _normalise_illustris_catalog_result(self, raw, fields: Sequence[str], label: str) -> Dict[str, np.ndarray]:
        fields = list(fields or [])
        if not fields:
            return {}

        if isinstance(raw, Mapping):
            out = {}
            missing = []
            for field in fields:
                if field in raw:
                    out[field] = np.asarray(raw[field])
                else:
                    missing.append(field)
            if missing:
                raise KeyError(f"illustris_python did not return {label} fields {missing}")
            return out

        if len(fields) == 1:
            return {fields[0]: np.asarray(raw)}

        raise TypeError(
            f"illustris_python returned a non-dict object for multiple {label} fields: {type(raw)!r}"
        )

    def _load_fof_local_hdf5(self, group_fields: Sequence[str], subhalo_fields: Sequence[str]):
        files = self._list_hdf5_files(prefix="groups")
        _check_split_file_completeness(files, attr_names=("NumFiles", "NumFilesPerSnapshot"))
        halos = self._h5_concat(files, "Group", group_fields)
        subs = self._h5_concat(files, "Subhalo", subhalo_fields)
        return halos, subs

    def _h5_concat(self, files: Sequence[str], group: str, fields: Sequence[str]) -> Dict[str, np.ndarray]:
        out = {k: [] for k in fields}
        field_meta: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {}

        for fp in files:
            with h5py.File(fp, "r") as f:
                g = f.get(group, None)
                n_this = self._groupcat_rows_this_file(f, group)
                if g is None:
                    if n_this == 0:
                        for k in fields:
                            out[k].append(self._empty_like_groupcat_field(k, n_this, field_meta))
                        continue
                    raise KeyError(f"Group '{group}' not found in {fp}")

                for k in fields:
                    if k in g:
                        arr = np.asarray(g[k][()])
                        out[k].append(arr)
                        field_meta.setdefault(k, (tuple(arr.shape[1:]), arr.dtype))
                    elif n_this == 0:
                        out[k].append(self._empty_like_groupcat_field(k, n_this, field_meta))
                    else:
                        raise KeyError(f"Field '{k}' not found in {group} of {fp}")

        cat = {}
        for k, arrs in out.items():
            if arrs:
                cat[k] = np.concatenate(arrs, axis=0)
            else:
                cat[k] = np.empty((0,), dtype=np.float64)
        return cat

    def _groupcat_rows_this_file(self, h5: h5py.File, group: str) -> Optional[int]:
        header = h5.get("Header", None)
        if header is None:
            return None
        if group == "Group":
            for key in ("Ngroups_ThisFile", "NgroupsThisFile"):
                if key in header.attrs:
                    return int(header.attrs[key])
        if group == "Subhalo":
            for key in ("Nsubgroups_ThisFile", "Nsubhalos_ThisFile", "NsubgroupsThisFile", "NsubhalosThisFile"):
                if key in header.attrs:
                    return int(header.attrs[key])
        return None

    def _empty_like_groupcat_field(self, field: str, n_rows: Optional[int], meta: Mapping[str, Tuple[Tuple[int, ...], np.dtype]]) -> np.ndarray:
        n = 0 if n_rows is None else int(n_rows)
        if field in meta:
            tail, dtype = meta[field]
            return np.empty((n,) + tuple(tail), dtype=dtype)
        return np.empty((n,), dtype=np.float64)

    # ------------------------------------------------------------------
    # API group catalog subset fallback
    # ------------------------------------------------------------------

    def _load_fof_api_fields(self, group_fields: Sequence[str], subhalo_fields: Sequence[str]):
        halos = {}
        subs = {}
        for field in group_fields:
            halos[field] = self._download_groupcat_field("Group", field)
        for field in subhalo_fields:
            subs[field] = self._download_groupcat_field("Subhalo", field)
        return halos, subs

    def _is_valid_hdf5_file(self, path: str) -> bool:
        """
        Return True if ``path`` exists, is non-empty, and can be opened as HDF5.

        This is used for persistent API cache files.  A stale ``.part`` file is
        never considered valid.  If a cached file is corrupt or incomplete, the
        caller will remove it and re-download.
        """
        if (not path) or (not os.path.isfile(path)) or os.path.getsize(path) <= 0:
            return False
        if str(path).endswith(".part"):
            return False
        try:
            with h5py.File(path, "r"):
                return True
        except Exception:
            return False

    def _remove_bad_cache_file(self, path: str) -> None:
        """Best-effort removal of a corrupt cache file and its .part neighbor."""
        for fp in [path, path + ".part"]:
            try:
                if fp and os.path.exists(fp):
                    os.remove(fp)
            except Exception as e:
                logger.debug("Failed to remove bad cache file %s: %s", fp, e)

    def _cache_or_download_hdf5(self, url: str, path: str, *, label: str) -> str:
        """
        Persistent cache-first downloader for API HDF5 files.

        Policy
        ------
        1. If ``path`` exists and is a readable non-empty HDF5 file, reuse it.
        2. If ``path`` exists but is corrupt/incomplete, remove and re-download.
        3. If ``path`` is absent, download it with retry/progress support.

        Only files downloaded in the current session are tracked for cleanup.
        Already-existing cache hits are never deleted by this object.
        """
        if self._is_valid_hdf5_file(path):
            if self.verbose:
                print(f"[TNG cache] using cached {label}: {path}", file=sys.stderr)
            return path

        if os.path.exists(path) or os.path.exists(path + ".part"):
            if self.verbose:
                print(f"[TNG cache] invalid/incomplete cached {label}; re-downloading: {path}", file=sys.stderr)
            self._remove_bad_cache_file(path)

        self._download_file(url, path, label=label)
        self._track_downloaded(path)
        return path


    def _download_groupcat_field(self, obj_type: str, field: str) -> np.ndarray:
        query = urlencode({obj_type: field})
        url = f"{self.api_base_url}/{self.sim_name}/files/groupcat-{self.snap}/?{query}"

        # Deterministic cache filename: reuse across notebook/kernel sessions.
        fname = (
            f"{_safe_name(self.sim_name)}_snap{self.snap:03d}_"
            f"groupcat_{_safe_name(obj_type)}_{_safe_name(field)}.hdf5"
        )
        path = os.path.join(self.cache_dir, fname)

        path = self._cache_or_download_hdf5(
            url,
            path,
            label=f"groupcat {obj_type}/{field}",
        )
        return self._read_groupcat_subset(path, obj_type, field)

    def _read_groupcat_subset(self, path: str, obj_type: str, field: str) -> np.ndarray:
        candidates = [
            f"{obj_type}/{field}",
            f"/{obj_type}/{field}",
            field,
            f"/{field}",
        ]
        with h5py.File(path, "r") as f:
            for key in candidates:
                key2 = key.strip("/")
                if key2 in f and isinstance(f[key2], h5py.Dataset):
                    return np.asarray(f[key2][()])

            # Flexible fallback: find a dataset with the requested basename.
            found = []

            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset) and os.path.basename(name) == field:
                    found.append(name)

            f.visititems(visitor)
            if found:
                return np.asarray(f[found[0]][()])

            available = []
            f.visititems(lambda name, obj: available.append(name) if isinstance(obj, h5py.Dataset) else None)
            raise KeyError(
                f"Could not find dataset for {obj_type}/{field} in downloaded groupcat subset {path}. "
                f"Available datasets: {available[:20]}"
            )

    # ------------------------------------------------------------------
    # Local particle loading
    # ------------------------------------------------------------------

    def _load_particles_local(self, kind: str, object_id: int, ptypes: Sequence[int], fields: Sequence[str]):
        """
        Load halo/subhalo particles from local snapshot files.

        Priority:
        1. ``illustris_python.snapshot.loadHalo/loadSubhalo``.
        2. Internal offset-based local HDF5 slicing.

        This method intentionally never calls the API. The public methods call
        it only when ``snapdir_XXX`` exists.
        """
        errors = []

        if self.use_illustris_python:
            try:
                return self._load_particles_local_illustris(kind, object_id, ptypes, fields)
            except Exception as e:
                errors.append(("illustris_python.snapshot", e))
                logger.warning(
                    "illustris_python snapshot read failed for %s=%s; trying internal local HDF5 reader. Reason: %s",
                    kind, object_id, e,
                )

        try:
            return self._load_particles_local_hdf5(kind, object_id, ptypes, fields)
        except Exception as e:
            if errors:
                detail = "; ".join(f"{name}: {err}" for name, err in errors)
                raise RuntimeError(
                    f"Local particle read failed for {kind}={object_id}. Tried illustris_python first and "
                    f"internal HDF5 second. Earlier local-reader errors: {detail}. "
                    f"Final internal-reader error: {e}"
                ) from e
            raise

    def _load_particles_local_illustris(self, kind: str, object_id: int, ptypes: Sequence[int], fields: Sequence[str]):
        _groupcat, snapshot, import_error = self._get_illustris_modules()
        if snapshot is None:
            raise ImportError(
                "illustris_python is not importable. Install/put illustris_python on PYTHONPATH, "
                "or rely on the internal local HDF5 fallback."
            ) from import_error

        if kind not in ("halo", "subhalo"):
            raise ValueError("kind must be 'halo' or 'subhalo'")

        loader = snapshot.loadHalo if kind == "halo" else snapshot.loadSubhalo
        object_id = int(object_id)
        fields = _unique(list(fields or []))
        out = {}

        if self.verbose:
            print(f"[TNG local] reading {kind} particles with illustris_python.snapshot", file=sys.stderr)

        for pt in ptypes:
            pt = int(pt)
            gname = f"PartType{pt}"
            n_expected = self._expected_particle_count(kind, object_id, pt)
            out[gname] = {}

            if n_expected <= 0:
                out[gname] = {field: self._empty_for_field(field) for field in fields}
                continue

            normal_fields = [
                f for f in fields
                if f not in _OPTIONAL_ZERO_VEC3_FIELDS and f not in _OPTIONAL_ZERO_SCALAR_FIELDS
            ]
            optional_fields = [f for f in fields if f not in normal_fields]

            if normal_fields:
                raw = loader(self.base_path, self.snap, object_id, pt, fields=list(normal_fields))
                out[gname].update(self._normalise_illustris_particle_result(raw, normal_fields, gname))

            # Try optional fields if they exist locally; otherwise fill zeros.
            for field in optional_fields:
                try:
                    raw = loader(self.base_path, self.snap, object_id, pt, fields=[field])
                    value = self._normalise_illustris_particle_result(raw, [field], gname)[field]
                    out[gname][field] = value
                except Exception:
                    if field in _OPTIONAL_ZERO_VEC3_FIELDS:
                        out[gname][field] = np.zeros((n_expected, 3), dtype=np.float64)
                    elif field in _OPTIONAL_ZERO_SCALAR_FIELDS:
                        out[gname][field] = np.zeros((n_expected,), dtype=np.float64)
                    else:
                        raise

            # Preserve requested field order in the nested dict.
            out[gname] = {field: out[gname][field] for field in fields}

        return out

    def _normalise_illustris_particle_result(self, raw, fields: Sequence[str], ptype_label: str) -> Dict[str, np.ndarray]:
        fields = list(fields or [])
        if not fields:
            return {}

        if isinstance(raw, Mapping):
            out = {}
            missing = []
            for field in fields:
                if field in raw:
                    out[field] = np.asarray(raw[field])
                else:
                    missing.append(field)
            if missing:
                raise KeyError(f"illustris_python did not return {ptype_label} fields {missing}")
            return out

        if len(fields) == 1:
            return {fields[0]: np.asarray(raw)}

        raise TypeError(
            f"illustris_python returned a non-dict object for multiple {ptype_label} fields: {type(raw)!r}"
        )

    def _load_particles_local_hdf5(self, kind: str, object_id: int, ptypes: Sequence[int], fields: Sequence[str]):
        object_id = int(object_id)
        fields = _unique(list(fields or []))

        if kind == "subhalo":
            soff = np.asarray(self.subhalos["SubhaloOffsetType"], dtype=np.int64)
            slt = np.asarray(self.subhalos["SubhaloLenType"], dtype=np.int64)
            out = {}
            for pt in ptypes:
                pt = int(pt)
                n = int(slt[object_id, pt])
                if n <= 0:
                    out[f"PartType{pt}"] = {k: self._empty_for_field(k) for k in fields}
                    continue
                off = int(soff[object_id, pt])
                out[f"PartType{pt}"] = self._read_fields_slice(pt, fields, off, n)
            return out

        if kind == "halo":
            glt = np.asarray(self.halos["GroupLenType"], dtype=np.int64)
            if "_GroupOffsetType" in self.halos:
                goff = np.asarray(self.halos["_GroupOffsetType"], dtype=np.int64)
            else:
                goff = self._compute_group_offsets(glt)

            out = {}
            for pt in ptypes:
                pt = int(pt)
                n = int(glt[object_id, pt])
                if n <= 0:
                    out[f"PartType{pt}"] = {k: self._empty_for_field(k) for k in fields}
                    continue
                off = int(goff[object_id, pt])
                out[f"PartType{pt}"] = self._read_fields_slice(pt, fields, off, n)
            return out

        raise ValueError("kind must be 'halo' or 'subhalo'")

    def _expected_particle_count(self, kind: str, object_id: int, ptype: int) -> int:
        if kind == "subhalo":
            return int(np.asarray(self.subhalos["SubhaloLenType"], dtype=np.int64)[int(object_id), int(ptype)])
        if kind == "halo":
            return int(np.asarray(self.halos["GroupLenType"], dtype=np.int64)[int(object_id), int(ptype)])
        raise ValueError("kind must be 'halo' or 'subhalo'")

    # ------------------------------------------------------------------
    # API cutout fallback
    # ------------------------------------------------------------------

    def _load_cutout(self, kind: str, object_id: int, ptypes: Sequence[int], fields: Sequence[str]):
        fields_by_ptype = {int(pt): _unique(list(fields or [])) for pt in ptypes}
        key = (
            str(kind),
            int(object_id),
            tuple((pt, tuple(fields_by_ptype[pt])) for pt in sorted(fields_by_ptype)),
        )
        if key not in self._cutout_cache:
            self._cutout_cache[key] = self._download_cutout(kind, int(object_id), fields_by_ptype)

        path = self._cutout_cache[key]
        out = {}
        with h5py.File(path, "r") as f:
            for pt in ptypes:
                pt = int(pt)
                gname = f"PartType{pt}"
                out[gname] = {}
                if gname not in f:
                    for field in fields:
                        out[gname][field] = self._empty_for_field(field)
                    continue
                g = f[gname]
                n_ref = None
                if "Coordinates" in g:
                    n_ref = int(g["Coordinates"].shape[0])
                for field in fields:
                    if field in g:
                        out[gname][field] = np.asarray(g[field][()])
                    elif field in _OPTIONAL_ZERO_VEC3_FIELDS:
                        n = n_ref if n_ref is not None else 0
                        out[gname][field] = np.zeros((n, 3), dtype=np.float64)
                    elif field in _OPTIONAL_ZERO_SCALAR_FIELDS:
                        n = n_ref if n_ref is not None else 0
                        out[gname][field] = np.zeros((n,), dtype=np.float64)
                    else:
                        raise KeyError(f"Field '{field}' not found in {gname} of API cutout {path}")
        return out

    def _download_cutout(self, kind: str, object_id: int, fields_by_ptype: Mapping[int, Sequence[str]]) -> str:
        if kind not in ("halo", "subhalo"):
            raise ValueError("kind must be 'halo' or 'subhalo'")
        endpoint = "halos" if kind == "halo" else "subhalos"

        params = {}
        normalized_fields_by_ptype = {}
        for pt, fields in fields_by_ptype.items():
            pt = int(pt)
            if pt not in PTYPE_TO_API:
                raise ValueError(f"API cutout does not support PartType{pt}; supported ptypes: {sorted(PTYPE_TO_API)}")
            fields = _unique(list(fields or []))
            normalized_fields_by_ptype[pt] = fields
            if not fields:
                continue
            params[PTYPE_TO_API[pt]] = ",".join(fields)

        query = urlencode(params)
        url = f"{self.api_base_url}/{self.sim_name}/snapshots/{self.snap}/{endpoint}/{int(object_id)}/cutout.hdf5"
        if query:
            url += "?" + query

        # Deterministic cache filename: if the same object/ptype/field request
        # has already been downloaded into cache_dir, reuse it instead of
        # downloading again.  The field order is normalized by _unique(...).
        field_tag = _safe_name(
            "__".join(
                f"{PTYPE_TO_API[pt]}={','.join(normalized_fields_by_ptype[pt])}"
                for pt in sorted(normalized_fields_by_ptype)
            )
        )
        if not field_tag:
            field_tag = "all"

        fname = (
            f"{_safe_name(self.sim_name)}_snap{self.snap:03d}_"
            f"{_safe_name(kind)}_{int(object_id)}_{field_tag}.hdf5"
        )
        path = os.path.join(self.cache_dir, fname)

        return self._cache_or_download_hdf5(
            url,
            path,
            label=f"{kind} cutout id={object_id}",
        )

    # ------------------------------------------------------------------
    # Offsets and local snapshot slicing
    # ------------------------------------------------------------------

    def _read_offsets_file(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._offsets_cache is not None:
            return self._offsets_cache

        fp = self._find_local_offsets_file()
        if fp is None:
            self._require_api("local offsets file is missing")
            fp = self._download_offsets_file()

        with h5py.File(fp, "r") as f:
            if "Subhalo" not in f or "Group" not in f:
                raise KeyError(f"Offsets file missing 'Subhalo'/'Group' groups: {fp}")
            if "SnapByType" not in f["Subhalo"] or "SnapByType" not in f["Group"]:
                raise KeyError(f"Offsets file missing '*/SnapByType' datasets: {fp}")
            off_sub = np.asarray(f["Subhalo/SnapByType"][()], dtype=np.int64)
            off_grp = np.asarray(f["Group/SnapByType"][()], dtype=np.int64)
            if "FileOffsets" in f and "SnapByType" in f["FileOffsets"]:
                fs = np.asarray(f["FileOffsets/SnapByType"][()], dtype=np.int64)
                # TNG stores [6, Nchunk]; use [Nchunk, 6] internally.
                self._file_offsets_snap = fs.T if fs.ndim == 2 and fs.shape[0] == 6 else fs

        self._offsets_cache = (off_sub, off_grp)
        return off_sub, off_grp

    def _find_local_offsets_file(self) -> Optional[str]:
        s = f"{self.snap:03d}"
        candidates = [
            os.path.join(self.base_path, f"offsets_{s}.hdf5"),
            os.path.join(self.base_path, f"offsets_{s}", f"offsets_{s}.hdf5"),
            os.path.join(os.path.dirname(self.base_path), "postprocessing", "offsets", f"offsets_{s}.hdf5"),
            os.path.join(os.path.dirname(self.base_path), "postprocessing", "offsets", f"offsets_{s}", f"offsets_{s}.hdf5"),
        ]
        for c in candidates:
            if c and os.path.isfile(c):
                return c
        return None

    def _download_offsets_file(self) -> str:
        url = f"{self.api_base_url}/{self.sim_name}/files/offsets.{self.snap}.hdf5"

        # Deterministic cache filename.
        fname = f"{_safe_name(self.sim_name)}_offsets_{self.snap:03d}.hdf5"
        path = os.path.join(self.cache_dir, fname)

        return self._cache_or_download_hdf5(
            url,
            path,
            label=f"offsets snap={self.snap}",
        )

    def _has_local_snapshots(self) -> bool:
        try:
            return len(self._list_hdf5_files(prefix="snapdir")) > 0
        except FileNotFoundError:
            return False

    def _read_fields_slice(self, partType: int, fields: Sequence[str], off: int, n: int):
        pt = int(partType)
        n = int(n)
        d = {}
        for k in fields:
            if k in _OPTIONAL_ZERO_VEC3_FIELDS:
                try:
                    d[k] = self._read_global_slice(pt, k, off, n)
                except Exception:
                    d[k] = np.zeros((n, 3), dtype=np.float64)
            elif k in _OPTIONAL_ZERO_SCALAR_FIELDS:
                try:
                    d[k] = self._read_global_slice(pt, k, off, n)
                except Exception:
                    d[k] = np.zeros((n,), dtype=np.float64)
            else:
                d[k] = self._read_global_slice(pt, k, off, n)
        return d

    def _read_global_slice(self, partType: int, field: str, off: int, n: int) -> np.ndarray:
        if int(n) <= 0:
            return self._empty_for_field(field)

        pt = int(partType)
        off = int(off)
        n = int(n)
        files, pref = self._build_snap_index(pt)
        name = f"PartType{pt}"
        need0 = off
        need1 = off + n

        out = None
        for i, fp in enumerate(files):
            a0 = int(pref[i])
            a1 = int(pref[i + 1])
            if a1 <= need0:
                continue
            if a0 >= need1:
                break
            b0 = max(need0, a0) - a0
            b1 = min(need1, a1) - a0
            with h5py.File(fp, "r") as f:
                g = f.get(name, None)
                if g is None or field not in g:
                    raise KeyError(f"Field '{field}' not found in {name} of {fp}")
                arr = g[field][b0:b1]
            out = arr if out is None else np.concatenate([out, arr], axis=0)

        if out is None:
            return self._empty_for_field(field)
        out = np.asarray(out)
        if out.shape[0] != n:
            raise RuntimeError(
                f"Incomplete local snapshot slice for PartType{pt}/{field}: expected {n}, got {out.shape[0]}. "
                "The local snapdir is probably incomplete."
            )
        return out

    def _build_snap_index(self, partType: int):
        pt = int(partType)
        if pt in self._snap_index:
            return self._snap_index[pt]

        files = self._list_hdf5_files(prefix="snapdir")
        _check_split_file_completeness(files, attr_names=("NumFilesPerSnapshot", "NumFiles"))
        name = f"PartType{pt}"
        counts = []
        for fp in files:
            with h5py.File(fp, "r") as f:
                g = f.get(name, None)
                if g is None or "Coordinates" not in g:
                    counts.append(0)
                else:
                    counts.append(int(g["Coordinates"].shape[0]))
        pref = np.concatenate([[0], np.cumsum(np.asarray(counts, dtype=np.int64))])
        self._snap_index[pt] = (files, pref)
        return files, pref

    # ------------------------------------------------------------------
    # API download utilities
    # ------------------------------------------------------------------

    def _require_api(self, reason: str) -> None:
        if not self.download_if_missing:
            raise FileNotFoundError(
                f"{reason}; API fallback is disabled. Set download_if_missing=True and provide a TNG API key."
            )
        if not self.api_key:
            raise RuntimeError(
                f"{reason}; API fallback requires a TNG API key. Pass api_key=... or export TNG_API_KEY."
            )

    def _download_file(self, url: str, path: str, *, label: str) -> str:
        """
        Download one API file with retry support for transient server errors.

        The TNG API occasionally returns 502/503/504/429 for group-catalog
        subset requests.  These should be treated as transient.  Temporary
        ``.part`` files are removed after each failed attempt so a later retry
        always starts from a clean state.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".part"
        headers = {"api-key": str(self.api_key), "User-Agent": "TNGCatLoader/1.0"}
        retry_statuses = {429, 500, 502, 503, 504}
        last_error = None

        for attempt in range(max(1, self.max_retries + 1)):
            req = Request(url, headers=headers)

            if self.verbose:
                if attempt == 0:
                    print(f"[TNG API] downloading {label}", file=sys.stderr)
                    print(f"[TNG API] url: {url}", file=sys.stderr)
                else:
                    print(f"[TNG API] retry {attempt}/{self.max_retries} for {label}", file=sys.stderr)

            try:
                with urlopen(req, timeout=self.timeout) as resp:
                    total = resp.headers.get("Content-Length")
                    total_i = int(total) if total and total.isdigit() else None
                    downloaded = 0
                    last_print = 0.0
                    chunk_size = 1024 * 1024
                    with open(tmp, "wb") as fh:
                        while True:
                            chunk = resp.read(chunk_size)
                            if not chunk:
                                break
                            fh.write(chunk)
                            downloaded += len(chunk)
                            now = time.monotonic()
                            if self.verbose and (now - last_print > 0.5):
                                self._print_progress(downloaded, total_i)
                                last_print = now
                    if self.verbose:
                        self._print_progress(downloaded, total_i, final=True)
                os.replace(tmp, path)
                return path

            except HTTPError as e:
                last_error = e
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass
                if int(getattr(e, "code", -1)) not in retry_statuses or attempt >= self.max_retries:
                    raise RuntimeError(f"TNG API download failed for {label}: {e}") from e

            except (URLError, TimeoutError) as e:
                last_error = e
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass
                if attempt >= self.max_retries:
                    raise RuntimeError(f"TNG API download failed for {label}: {e}") from e

            except Exception:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                finally:
                    raise

            sleep = min(self.retry_max_sleep, self.retry_base_sleep * (2 ** attempt))
            if self.verbose:
                print(f"[TNG API] transient failure for {label}: {last_error}; sleeping {sleep:.1f} s", file=sys.stderr)
            time.sleep(sleep)

        raise RuntimeError(f"TNG API download failed for {label}: {last_error}")

    def _print_progress(self, downloaded: int, total: Optional[int], final: bool = False) -> None:
        mb = downloaded / 1024.0 / 1024.0
        if total:
            pct = 100.0 * downloaded / total
            tmb = total / 1024.0 / 1024.0
            msg = f"\r[TNG API] {mb:.1f}/{tmb:.1f} MiB ({pct:5.1f}%)"
        else:
            msg = f"\r[TNG API] {mb:.1f} MiB"
        end = "\n" if final else ""
        print(msg, end=end, file=sys.stderr, flush=True)

    def _track_downloaded(self, path: str) -> None:
        if path and os.path.isfile(path):
            self._downloaded_files.append(path)

    # ------------------------------------------------------------------
    # Filesystem helpers
    # ------------------------------------------------------------------

    def _list_hdf5_files(self, prefix: str) -> List[str]:
        """
        List local split HDF5 files.

        prefix="groups" reads base_path/groups_XXX/*.hdf5.
        prefix="snapdir" reads base_path/snapdir_XXX/*.hdf5.
        """
        d = os.path.join(self.base_path, f"{prefix}_{self.snap:03d}")
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Directory not found: {d}")
        files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".hdf5")]
        if not files:
            raise FileNotFoundError(f"No .hdf5 files found in: {d}")
        files.sort(key=_hdf5_chunk_sort_key)
        return files

    def _compute_group_offsets(self, group_len_type: np.ndarray) -> np.ndarray:
        gl = np.asarray(group_len_type, dtype=np.int64)
        offsets = np.zeros_like(gl, dtype=np.int64)
        offsets[1:, :] = np.cumsum(gl[:-1, :], axis=0)
        return offsets

    def _require_fof_loaded(self) -> None:
        if self.halos is None or self.subhalos is None:
            raise RuntimeError("FoF catalogs not loaded. Call loadFoF(...) first.")

    def _empty_for_field(self, field: str) -> np.ndarray:
        if field in _VEC3_FIELDS:
            return np.empty((0, 3), dtype=np.float64)
        return np.empty((0,), dtype=np.float64)
