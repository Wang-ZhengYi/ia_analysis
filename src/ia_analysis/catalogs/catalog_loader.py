# catalog_loader.py
# -*- coding: utf-8 -*-

"""
Catalog loaders for ClusterSims and IllustrisTNG-like data products.

This file provides two classes:
    - CSCatalog: loader for ClusterSims group catalogs and particle snapshots.
    - TNGCatalog: loader for IllustrisTNG catalogs with a CSCatalog-compatible API.

Important implementation note
-----------------------------
The global ordering of HDF5 chunks is critical. Group/subhalo catalogs and
particle snapshots are split into multiple files such as:
    groups_021.0.hdf5, groups_021.1.hdf5, ..., groups_021.10.hdf5
or
    snapdir_021.0.hdf5, snapdir_021.1.hdf5, ..., snapdir_021.10.hdf5

A plain lexicographic sort would incorrectly order them as 0, 1, 10, 11, 2, ...
This breaks global subhalo IDs, GroupFirstSub, GroupNsubs, and particle offsets.
Therefore all HDF5 chunks are sorted by their numeric chunk id.
"""

import os
import re
import json
import logging
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

import h5py


logger = logging.getLogger(__name__)


def _hdf5_chunk_sort_key(filepath):
    """
    Return a robust sort key for split HDF5 catalog/snapshot files.

    The preferred pattern is the standard AREPO/TNG-like chunk suffix:
        something.0.hdf5
        something.1.hdf5
        something.10.hdf5

    The function first extracts the integer immediately before '.hdf5'.
    This guarantees numeric ordering, avoiding the dangerous lexicographic
    order 0, 1, 10, 11, 2, ... .

    If no standard chunk suffix is found, it falls back to a natural sort key
    based on all integers in the filename.
    """
    base = os.path.basename(filepath)

    # Standard split-file pattern: groups_021.10.hdf5, snap_021.10.hdf5, etc.
    m = re.search(r"\.(\d+)\.hdf5$", base)
    if m is not None:
        return (0, int(m.group(1)), base)

    # Fallback for filenames such as groups_021_10.hdf5, if ever present.
    m = re.search(r"_(\d+)\.hdf5$", base)
    if m is not None:
        return (1, int(m.group(1)), base)

    # Last fallback: natural sorting over the full filename.
    parts = re.split(r"(\d+)", base)
    natural_parts = tuple(int(p) if p.isdigit() else p for p in parts)
    return (2, natural_parts, base)


class CSCatalog:
    """
    ClusterSims catalog loader with group-contiguous subhalo layout.

    Key idea
    --------
    Subhalos of a FoF group g occupy a contiguous range:
        lo = GroupFirstSub[g]
        hi = lo + GroupNsubs[g]

    Therefore any Subhalo* array can be sliced by [lo:hi] to get all subhalos
    in group g. The first subhalo in this range is the central subhalo, and the
    remaining subhalos are satellites.

    Workflow
    --------
    cat = CSCatalog(base_path, snap)
    halos, subs = cat.loadFoF(group_fields=[...], subhalo_fields=[...])

    # Slice subhalo properties of a group.
    g = 10
    lo = halos["GroupFirstSub"][g]
    hi = lo + halos["GroupNsubs"][g]
    sub_mass = subs["SubhaloMass"][lo:hi]

    # Load particles.
    parts_sub = cat.loadSubhalos(sid=123, ptypes=[1], fields=["Coordinates", "Velocities"])
    parts_grp = cat.loadHalos(gid=10, ptypes=[1], fields=["Coordinates", "Masses"])
    parts_grp2 = cat.loadHalos(sid=123, ptypes=[1], fields=["Coordinates", "Masses"])
    """

    def __init__(self, base_path, snap):
        self.base_path = str(base_path)
        self.snap = int(snap)

        self.halos = None     # Dict containing Group catalog fields.
        self.subhalos = None  # Dict containing Subhalo catalog fields plus injected arrays.

        # Internal cache for snapdir indexing:
        #   partType -> (ordered_files, cumulative_particle_counts)
        self._snap_index = {}

    # =========================================================
    # Public: FoF catalogs
    # =========================================================

    def loadFoF(self, group_fields, subhalo_fields):
        """
        Load FoF group and subhalo catalogs for this snapshot.

        Besides the requested fields, this function injects into the returned
        subhalo dictionary:
          - SubhaloID:         int64, [0..Nsub-1]
          - GroupID:           int64, subhalo -> group mapping
          - CenID:             int64, central subhalo id of each subhalo's group
          - SubhaloOffsetType: int64 (Nsub, Npt), global offsets for particle slicing

        These dictionaries are also stored as self.halos and self.subhalos.
        """
        # Minimal fields required to construct central/satellite mapping and offsets.
        req_g = ["GroupFirstSub", "GroupNsubs", "GroupLenType"]
        req_s = ["SubhaloLenType"]

        g_fields = self._unique(list(group_fields) + req_g)

        # These fields are computed internally and should not be read from disk.
        computed = {"SubhaloID", "GroupID", "CenID", "SubhaloOffsetType"}
        s_fields = [x for x in self._unique(list(subhalo_fields) + req_s) if x not in computed]

        files = self._list_hdf5_files(prefix="groups")

        halos = self._h5_concat(files, "Group", g_fields)
        subs = self._h5_concat(files, "Subhalo", s_fields)

        # Inject global subhalo IDs. This is only valid if group files are sorted
        # by numeric chunk id, which _list_hdf5_files now guarantees.
        nsub = self._infer_len(subs)
        subs["SubhaloID"] = np.arange(nsub, dtype=np.int64)

        # Inject GroupID and CenID using the contiguous subhalo range of each FoF group.
        group_id, cen_id = self._build_subhalo_group_maps(halos, nsub)
        subs["GroupID"] = group_id
        subs["CenID"] = cen_id

        # Inject particle offsets for each subhalo.
        subs["SubhaloOffsetType"] = self._compute_subhalo_offsets(halos, subs)

        self.halos = halos
        self.subhalos = subs
        return halos, subs

    # =========================================================
    # Public: particle loaders
    # =========================================================

    def loadSubhalos(self, sid, ptypes, fields):
        """
        Load particle fields for a subhalo.

        Parameters
        ----------
        sid : int
            Global subhalo id.
        ptypes : iterable of int
            Particle types to load, e.g. [1] for dark matter.
        fields : iterable of str
            Particle fields to load, e.g. ["Coordinates", "Velocities"].

        Requires
        --------
        loadFoF(...) must be called first, because particle offsets are built
        from GroupFirstSub, GroupNsubs, GroupLenType, and SubhaloLenType.
        """
        self._require_fof_loaded()

        sid = int(sid)
        soff = np.asarray(self.subhalos["SubhaloOffsetType"], dtype=np.int64)
        slt = np.asarray(self.subhalos["SubhaloLenType"], dtype=np.int64)

        out = {}
        for pt in ptypes:
            pt = int(pt)
            n = int(slt[sid, pt])
            if n <= 0:
                out[f"PartType{pt}"] = {k: self._empty_for_field(k) for k in fields}
                continue

            off = int(soff[sid, pt])
            out[f"PartType{pt}"] = self._read_fields_slice(pt, fields, off, n)

        return out

    def loadHalos(self, ptypes, fields, gid=None, sid=None):
        """
        Load particle fields for a FoF group.

        You can provide either:
          - gid: direct group index
          - sid: subhalo id; gid will be inferred from self.subhalos["GroupID"][sid]

        Requires
        --------
        loadFoF(...) must be called first.
        """
        self._require_fof_loaded()

        if gid is None:
            if sid is None:
                raise ValueError("loadHalos: provide gid or sid.")
            sid = int(sid)
            gid = int(self.subhalos["GroupID"][sid])
        else:
            gid = int(gid)

        glt = np.asarray(self.halos["GroupLenType"], dtype=np.int64)
        goff = self._compute_group_offsets(glt)

        out = {}
        for pt in ptypes:
            pt = int(pt)
            n = int(glt[gid, pt])
            if n <= 0:
                out[f"PartType{pt}"] = {k: self._empty_for_field(k) for k in fields}
                continue

            off = int(goff[gid, pt])
            out[f"PartType{pt}"] = self._read_fields_slice(pt, fields, off, n)

        return out

    # =========================================================
    # Internal: maps / offsets
    # =========================================================

    def _build_subhalo_group_maps(self, halos, nsub):
        """
        Build two length-Nsub arrays:
          GroupID[sid] = gid
          CenID[sid]   = GroupFirstSub[gid]

        This uses the AREPO/TNG-like contiguous subhalo layout:
          subhalos of group g occupy [GroupFirstSub[g], GroupFirstSub[g] + GroupNsubs[g]).
        """
        first = np.asarray(halos["GroupFirstSub"], dtype=np.int64)
        nsubs = np.asarray(halos["GroupNsubs"], dtype=np.int64)

        group_id = np.full(int(nsub), -1, dtype=np.int64)
        cen_id = np.full(int(nsub), -1, dtype=np.int64)

        Ng = int(first.shape[0])
        for g in range(Ng):
            fs = int(first[g])
            ns = int(nsubs[g])
            if ns <= 0 or fs < 0:
                continue

            lo = fs
            hi = fs + ns
            if lo >= nsub:
                continue
            if hi > nsub:
                # This should not happen for a consistent catalog. Clip to keep
                # the loader usable while preserving valid objects.
                hi = nsub

            group_id[lo:hi] = g
            cen_id[lo:hi] = fs

        return group_id, cen_id

    def _compute_group_offsets(self, group_len_type):
        """
        Compute cumulative particle offsets of each FoF group from GroupLenType.
        """
        gl = np.asarray(group_len_type, dtype=np.int64)
        offsets = np.zeros_like(gl, dtype=np.int64)
        offsets[1:, :] = np.cumsum(gl[:-1, :], axis=0)
        return offsets

    def _compute_subhalo_offsets(self, halos, subhalos):
        """
        Compute SubhaloOffsetType (Nsub, Npt) from:
          GroupFirstSub, GroupNsubs, GroupLenType, SubhaloLenType.

        The computation assumes that subhalos are stored contiguously within each
        group and in the same global order as GroupFirstSub/GroupNsubs.
        """
        first = np.asarray(halos["GroupFirstSub"], dtype=np.int64)
        nsubs = np.asarray(halos["GroupNsubs"], dtype=np.int64)
        glt = np.asarray(halos["GroupLenType"], dtype=np.int64)
        slt = np.asarray(subhalos["SubhaloLenType"], dtype=np.int64)

        Ng = int(first.shape[0])
        Nsub = int(slt.shape[0])
        Npt = int(slt.shape[1])

        goff = self._compute_group_offsets(glt)
        soff = np.zeros((Nsub, Npt), dtype=np.int64)

        for g in range(Ng):
            fs = int(first[g])
            ns = int(nsubs[g])
            if ns <= 0:
                continue

            lo = fs
            hi = fs + ns
            if lo < 0 or lo >= Nsub:
                continue
            if hi > Nsub:
                hi = Nsub

            run = goff[g, :].copy()
            for s in range(lo, hi):
                soff[s, :] = run
                run += slt[s, :]

        return soff

    # =========================================================
    # Internal: reading snapdir_* particle slices
    # =========================================================

    def _read_fields_slice(self, partType, fields, off, n):
        """
        Read multiple fields for PartType slice [off:off+n].

        Special case:
          - ModifiedGravityAcceleration is optional. If missing, return zeros.
        """
        pt = int(partType)
        n = int(n)

        d = {}
        for k in fields:
            if k == "ModifiedGravityAcceleration":
                try:
                    d[k] = self._read_global_slice(pt, k, off, n)
                except Exception:
                    d[k] = np.zeros((n, 3), dtype=np.float64)
            else:
                d[k] = self._read_global_slice(pt, k, off, n)
        return d

    def _read_global_slice(self, partType, field, off, n):
        """
        Read a global particle slice from split snapshot files.

        The global slice [off, off+n) is mapped onto the numerically ordered
        snapdir chunks using cumulative particle counts.
        """
        if n <= 0:
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
        return np.asarray(out)

    def _build_snap_index(self, partType):
        """
        Build a particle-count prefix array for numerically ordered snapdir chunks.
        """
        pt = int(partType)
        if pt in self._snap_index:
            return self._snap_index[pt]

        files = self._list_hdf5_files(prefix="snapdir")
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

    # =========================================================
    # Internal: HDF5 concatenation and utilities
    # =========================================================

    def _list_hdf5_files(self, prefix):
        """
        List split HDF5 files in numeric chunk order.

        This is deliberately not a plain files.sort(). Numeric chunk ordering is
        required to keep global subhalo IDs, GroupFirstSub, GroupNsubs, and
        particle offsets consistent.
        """
        d = os.path.join(self.base_path, f"{prefix}_{self.snap:03d}")
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Directory not found: {d}")

        files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".hdf5")]
        if not files:
            raise FileNotFoundError(f"No .hdf5 files found in: {d}")

        files.sort(key=_hdf5_chunk_sort_key)
        return files

    def _h5_concat(self, files, group, fields):
        """
        Concatenate requested HDF5 fields across numerically ordered chunks.
        """
        out = {k: [] for k in fields}
        for fp in files:
            with h5py.File(fp, "r") as f:
                g = f.get(group, None)
                if g is None:
                    raise KeyError(f"Group '{group}' not found in {fp}")
                for k in fields:
                    if k not in g:
                        raise KeyError(f"Field '{k}' not found in {group} of {fp}")
                    out[k].append(g[k][:])

        cat = {}
        for k, arrs in out.items():
            cat[k] = np.concatenate(arrs, axis=0) if arrs else np.empty((0,), dtype=np.float64)
        return cat

    def _require_fof_loaded(self):
        if self.halos is None or self.subhalos is None:
            raise RuntimeError("FoF catalogs not loaded. Call loadFoF(...) first.")

    def _unique(self, xs):
        """
        Preserve order while removing duplicates.
        """
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    def _infer_len(self, dct):
        if not dct:
            return 0
        arr = next(iter(dct.values()))
        return int(arr.shape[0])

    def _empty_for_field(self, field):
        if field in (
            "Coordinates", "Velocities",
            "Acceleration", "Accelerations",
            "ModifiedGravityAcceleration",
        ):
            return np.empty((0, 3), dtype=np.float64)
        return np.empty((0,), dtype=np.float64)


class TNGCatalog:
    """
    IllustrisTNG catalog loader with a CSCatalog-compatible interface.

    New in this version
    -------------------
    This class can optionally download missing IllustrisTNG files from the
    official API when ``download_if_missing=True``. The default remains fully
    local/offline and is therefore backward compatible.

    Supported automatic downloads
    -----------------------------
    - FoF/Subfind group catalog chunks:
        ``base_path/groups_XXX/groups_XXX.N.hdf5``
    - Snapshot offsets file:
        ``dirname(base_path)/postprocessing/offsets/offsets_XXX.hdf5``
    - Snapshot chunks needed by a requested particle slice:
        ``base_path/snapdir_XXX/snap_XXX.N.hdf5``

    Notes
    -----
    - Snapshot chunks can be very large. The loader downloads only chunks that
      intersect the requested global particle slice, using
      ``FileOffsets/SnapByType`` from the offsets file when available.
    - API access normally requires an IllustrisTNG API key. Pass ``api_key=...``
      or set environment variable ``TNG_API_KEY`` or ``ILLUSTRIS_API_KEY``.
    - ``sim_name`` is needed for API download, e.g. ``TNG100-1`` or ``TNG300-1``.
      If omitted, the loader tries to infer it from ``base_path``.

    Workflow
    --------
    Local-only behavior, unchanged::

        cat = TNGCatalog(base_path, snap)

    API-backed behavior::

        cat = TNGCatalog(
            base_path,
            snap,
            sim_name="TNG300-1",
            api_key="...",
            download_if_missing=True,
        )
    """

    def __init__(
        self,
        base_path,
        snap,
        *,
        sim_name=None,
        api_key=None,
        api_url="https://www.tng-project.org/api",
        download_if_missing=False,
        cache_dir=None,
        download_groupcat=True,
        download_offsets=True,
        download_snapshots=True,
        overwrite=False,
        timeout=120,
        max_retries=3,
    ):
        self.requested_base_path = str(base_path)
        self.snap = int(snap)

        self.sim_name = sim_name or self._infer_sim_name(self.requested_base_path)
        self.api_key = api_key or os.environ.get("TNG_API_KEY") or os.environ.get("ILLUSTRIS_API_KEY")
        self.api_url = str(api_url).rstrip("/")
        self.download_if_missing = bool(download_if_missing)
        self.download_groupcat = bool(download_groupcat)
        self.download_offsets = bool(download_offsets)
        self.download_snapshots = bool(download_snapshots)
        self.overwrite = bool(overwrite)
        self.timeout = int(timeout)
        self.max_retries = int(max_retries)

        # If a cache directory is provided and the requested base_path does not
        # exist yet, use cache_dir/<sim_name>/output as the local TNG output root.
        # Existing base_path always wins to avoid surprising path changes.
        if cache_dir is not None and (not os.path.isdir(self.requested_base_path)):
            if self.sim_name is None:
                raise ValueError("cache_dir was provided, but sim_name could not be inferred. Pass sim_name='TNG100-1' etc.")
            self.base_path = os.path.join(str(cache_dir), self.sim_name, "output")
        else:
            self.base_path = self.requested_base_path

        self.halos = None
        self.subhalos = None

        # Internal cache for snapdir indexing:
        #   partType -> (ordered_files, cumulative_particle_counts)
        self._snap_index = {}

        # Optional FileOffsets/SnapByType from offsets_XXX.hdf5, normalized to
        # shape (Nfile, 6). This lets us lazily download only touched snapshot
        # chunks instead of requiring a complete local snapdir.
        self._snapshot_file_offsets = None

    # =========================================================
    # Public: FoF catalogs
    # =========================================================

    def loadFoF(self, group_fields, subhalo_fields):
        """
        Load FoF group and subhalo catalogs for this snapshot using illustris_python.

        Besides requested fields, inject into the returned subhalo dictionary:
          - SubhaloID:         int64, [0..Nsub-1]
          - GroupID:           int64, subhalo -> group mapping
          - CenID:             int64, central subhalo id of each subhalo's group
          - SubhaloOffsetType: int64 (Nsub, 6), global offsets for particle slicing
        """
        if self.download_if_missing and self.download_groupcat:
            self._ensure_groupcat_available()

        il = self._import_il()

        # Minimal requirements for TNG naming conventions.
        req_g = ["GroupFirstSub", "GroupNsubs", "GroupLenType"]
        req_s = ["SubhaloGrNr", "SubhaloLenType"]

        g_fields = self._unique(list(group_fields) + req_g)

        # These fields are injected internally.
        computed = {"SubhaloID", "GroupID", "CenID", "SubhaloOffsetType"}
        s_fields = [x for x in self._unique(list(subhalo_fields) + req_s) if x not in computed]

        halos = il.groupcat.loadHalos(self.base_path, int(self.snap), fields=g_fields)
        subs = il.groupcat.loadSubhalos(self.base_path, int(self.snap), fields=s_fields)

        # Inject SubhaloID.
        nsub = int(subs["SubhaloLenType"].shape[0])
        subs["SubhaloID"] = np.arange(nsub, dtype=np.int64)

        # For TNG, SubhaloGrNr directly gives the subhalo -> FoF group mapping.
        gid = np.asarray(subs["SubhaloGrNr"], dtype=np.int64)
        subs["GroupID"] = gid

        # Inject CenID = GroupFirstSub[GroupID].
        first = np.asarray(halos["GroupFirstSub"], dtype=np.int64)
        cen = np.full(nsub, -1, dtype=np.int64)
        ok = (gid >= 0) & (gid < first.shape[0])
        cen[ok] = first[gid[ok]]
        subs["CenID"] = cen

        # Inject SubhaloOffsetType from the offsets file.
        off_sub, off_grp = self._read_offsets_file()
        subs["SubhaloOffsetType"] = off_sub

        # Store group offsets internally. Prefix '_' avoids collision with real catalog fields.
        halos["_GroupOffsetType"] = off_grp

        self.halos = halos
        self.subhalos = subs
        return halos, subs

    # =========================================================
    # Public: particle loaders
    # =========================================================

    def loadSubhalos(self, sid, ptypes, fields):
        """
        Load particle fields for a subhalo.
        """
        self._require_fof_loaded()

        sid = int(sid)
        soff = np.asarray(self.subhalos["SubhaloOffsetType"], dtype=np.int64)
        slt = np.asarray(self.subhalos["SubhaloLenType"], dtype=np.int64)

        out = {}
        for pt in ptypes:
            pt = int(pt)
            n = int(slt[sid, pt])
            if n <= 0:
                out[f"PartType{pt}"] = {k: self._empty_for_field(k) for k in fields}
                continue

            off = int(soff[sid, pt])
            out[f"PartType{pt}"] = self._read_fields_slice(pt, fields, off, n)

        return out

    def loadHalos(self, ptypes, fields, gid=None, sid=None):
        """
        Load particle fields for a FoF group.

        Provide either:
          - gid: direct group index
          - sid: subhalo id; gid will be inferred from self.subhalos["GroupID"][sid]
        """
        self._require_fof_loaded()

        if gid is None:
            if sid is None:
                raise ValueError("loadHalos: provide gid or sid.")
            sid = int(sid)
            gid = int(self.subhalos["GroupID"][sid])
        else:
            gid = int(gid)

        glt = np.asarray(self.halos["GroupLenType"], dtype=np.int64)

        # Prefer exact offsets from the offsets file. Fall back to cumulative sum if absent.
        if "_GroupOffsetType" in self.halos:
            goff = np.asarray(self.halos["_GroupOffsetType"], dtype=np.int64)
        else:
            goff = self._compute_group_offsets(glt)

        out = {}
        for pt in ptypes:
            pt = int(pt)
            n = int(glt[gid, pt])
            if n <= 0:
                out[f"PartType{pt}"] = {k: self._empty_for_field(k) for k in fields}
                continue

            off = int(goff[gid, pt])
            out[f"PartType{pt}"] = self._read_fields_slice(pt, fields, off, n)

        return out

    # =========================================================
    # API download helpers
    # =========================================================

    def _infer_sim_name(self, path):
        """
        Infer a TNG/Illustris simulation name from a local path if possible.
        """
        text = str(path)
        # Common names: TNG100-1, TNG300-1, TNG50-1, TNG100-1-Dark, Illustris-1.
        matches = re.findall(r"(?:TNG\d+-\d+(?:-Dark)?|Illustris-\d+)", text)
        return matches[-1] if matches else None

    def _require_download_config(self):
        if not self.download_if_missing:
            return False
        if self.sim_name is None:
            raise ValueError(
                "download_if_missing=True requires sim_name, e.g. sim_name='TNG100-1' or 'TNG300-1'. "
                "The loader could not infer it from base_path."
            )
        if not self.api_key:
            logger.warning(
                "No IllustrisTNG API key was provided. Set TNG_API_KEY/ILLUSTRIS_API_KEY "
                "or pass api_key=... if the API returns 401/403."
            )
        return True

    def _api_headers(self):
        headers = {"User-Agent": "shape-tide-tng-loader/1.0"}
        if self.api_key:
            headers["API-Key"] = self.api_key
        return headers

    def _api_files_url(self, endpoint, params=None):
        endpoint = str(endpoint).lstrip("/")
        url = f"{self.api_url}/{self.sim_name}/files/{endpoint}"
        if params:
            url += "?" + urlencode(params, doseq=True)
        return url

    def _api_sim_url(self, endpoint="", params=None):
        endpoint = str(endpoint).lstrip("/")
        if endpoint:
            url = f"{self.api_url}/{self.sim_name}/{endpoint}"
        else:
            url = f"{self.api_url}/{self.sim_name}/"
        if params:
            url += "?" + urlencode(params, doseq=True)
        return url

    def _api_get_json(self, url):
        last_err = None
        for attempt in range(max(1, self.max_retries)):
            try:
                req = Request(url, headers=self._api_headers())
                with urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8")
                return json.loads(raw)
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
                last_err = e
                if attempt + 1 < self.max_retries:
                    time.sleep(min(2.0 * (attempt + 1), 10.0))
        raise RuntimeError(f"Failed to fetch JSON from {url}: {last_err}")

    def _extract_file_count(self, payload, key_hint):
        """
        Extract a number of chunks from the flexible API JSON payload.
        """
        if isinstance(payload, list):
            return len(payload)

        if not isinstance(payload, dict):
            return None

        # Direct count keys.
        for key in (
            "count",
            "num_files",
            f"num_files_{key_hint}",
            "num_files_snapshot",
            "num_files_groupcat",
        ):
            if key in payload:
                try:
                    return int(payload[key])
                except Exception:
                    pass

        # Common list-like keys.
        for key in ("files", "file_urls", "fileurls", "results", "chunks"):
            val = payload.get(key, None)
            if isinstance(val, list):
                return len(val)
            if isinstance(val, dict):
                return len(val)

        # Very defensive fallback: count links that look like HDF5 chunk URLs.
        text = json.dumps(payload)
        if key_hint == "snapshot":
            hits = re.findall(r"snapshot-\d+\.(\d+)\.hdf5", text)
        else:
            hits = re.findall(r"groupcat-\d+\.(\d+)\.hdf5", text)
        if hits:
            return max(int(x) for x in hits) + 1
        return None

    def _get_num_files(self, kind):
        """
        Return the number of groupcat/snapshot chunks according to the API.
        """
        self._require_download_config()
        kind = str(kind)
        if kind not in ("groupcat", "snapshot"):
            raise ValueError("kind must be 'groupcat' or 'snapshot'")

        list_endpoint = f"{kind}-{self.snap}/"
        try:
            payload = self._api_get_json(self._api_files_url(list_endpoint, params={"format": "api"}))
            n = self._extract_file_count(payload, kind)
            if n is not None and n > 0:
                return int(n)
        except Exception as e:
            logger.warning("Could not get %s chunk list from API files endpoint: %s", kind, str(e))

        # Metadata fallback. The simulation endpoint commonly contains
        # num_files_snapshot and num_files_groupcat.
        payload = self._api_get_json(self._api_sim_url(params={"format": "api"}))
        n = self._extract_file_count(payload, kind)
        if n is None or n <= 0:
            raise RuntimeError(f"Could not determine number of {kind} chunks for {self.sim_name} snap={self.snap}")
        return int(n)

    def _download_api_file(self, endpoint, target_path, params=None):
        """
        Download one API file endpoint to target_path using an atomic .part file.
        """
        self._require_download_config()
        target_path = os.path.abspath(str(target_path))
        if os.path.isfile(target_path) and not self.overwrite:
            return target_path

        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        tmp_path = target_path + ".part"
        url = self._api_files_url(endpoint, params=params)

        last_err = None
        for attempt in range(max(1, self.max_retries)):
            try:
                logger.info("Downloading TNG file: %s -> %s", url, target_path)
                req = Request(url, headers=self._api_headers())
                with urlopen(req, timeout=self.timeout) as resp, open(tmp_path, "wb") as out:
                    while True:
                        block = resp.read(1024 * 1024)
                        if not block:
                            break
                        out.write(block)
                # Basic HDF5 validation catches HTML error pages saved as .hdf5.
                with h5py.File(tmp_path, "r"):
                    pass
                os.replace(tmp_path, target_path)
                return target_path
            except HTTPError as e:
                last_err = e
                try:
                    msg = e.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    msg = ""
                if e.code in (401, 403):
                    raise RuntimeError(
                        f"TNG API permission error {e.code} for {url}. "
                        "Pass a valid api_key or set TNG_API_KEY. "
                        f"Server message: {msg}"
                    ) from e
            except (URLError, TimeoutError, OSError) as e:
                last_err = e
            finally:
                if os.path.isfile(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

            if attempt + 1 < self.max_retries:
                time.sleep(min(2.0 * (attempt + 1), 10.0))

        raise RuntimeError(f"Failed to download {url} -> {target_path}: {last_err}")

    def _local_hdf5_files(self, prefix):
        d = os.path.join(self.base_path, f"{prefix}_{self.snap:03d}")
        if not os.path.isdir(d):
            return []
        files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".hdf5")]
        files.sort(key=_hdf5_chunk_sort_key)
        return files

    def _ensure_groupcat_available(self):
        """
        Ensure the group catalog directory exists locally. Downloads all chunks
        because illustris_python.groupcat expects a complete local group catalog.
        """
        if self._local_hdf5_files("groups"):
            return
        if not self.download_if_missing or not self.download_groupcat:
            return

        nfiles = self._get_num_files("groupcat")
        s = f"{self.snap:03d}"
        out_dir = os.path.join(self.base_path, f"groups_{s}")
        os.makedirs(out_dir, exist_ok=True)

        for chunk in range(nfiles):
            target = os.path.join(out_dir, f"groups_{s}.{chunk}.hdf5")
            self._download_api_file(f"groupcat-{self.snap}.{chunk}.hdf5", target)

    def _offset_candidates(self):
        il = None
        candidates = []
        try:
            il = self._import_il()
            candidates.append(il.groupcat.offsetPath(self.base_path, int(self.snap)))
        except Exception:
            pass

        s = f"{int(self.snap):03d}"
        candidates += [
            os.path.join(self.base_path, f"offsets_{s}.hdf5"),
            os.path.join(self.base_path, f"offsets_{s}", f"offsets_{s}.hdf5"),
            os.path.join(os.path.dirname(self.base_path), "postprocessing", "offsets", f"offsets_{s}.hdf5"),
            os.path.join(os.path.dirname(self.base_path), "postprocessing", "offsets", f"offsets_{s}", f"offsets_{s}.hdf5"),
        ]

        # Preserve order while removing duplicates/empty values.
        out = []
        seen = set()
        for c in candidates:
            if c and c not in seen:
                out.append(c)
                seen.add(c)
        return out

    def _ensure_offsets_available(self):
        for c in self._offset_candidates():
            if os.path.isfile(c):
                return c

        if not self.download_if_missing or not self.download_offsets:
            return None

        self._require_download_config()
        # Prefer the standard illustris_python layout.
        target = self._offset_candidates()[0]
        self._download_api_file(f"offsets.{self.snap}.hdf5", target)
        return target

    def _snapshot_dir(self):
        return os.path.join(self.base_path, f"snapdir_{self.snap:03d}")

    def _snapshot_chunk_path(self, chunk):
        return os.path.join(self._snapshot_dir(), f"snap_{self.snap:03d}.{int(chunk)}.hdf5")

    def _find_snapshot_chunk_file(self, chunk):
        """
        Find an existing local file for one snapshot chunk. Different public
        releases/scripts can use slightly different filenames, so check several.
        """
        chunk = int(chunk)
        d = self._snapshot_dir()
        s = f"{self.snap:03d}"
        candidates = [
            os.path.join(d, f"snap_{s}.{chunk}.hdf5"),
            os.path.join(d, f"snapshot_{s}.{chunk}.hdf5"),
            os.path.join(d, f"snapshot-{self.snap}.{chunk}.hdf5"),
            os.path.join(d, f"snapshot-{s}.{chunk}.hdf5"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c

        if os.path.isdir(d):
            pat = re.compile(rf"(?:snap|snapshot)[_-]?0*{self.snap}\.{chunk}\.hdf5$|snapshot-0*{self.snap}\.{chunk}\.hdf5$")
            for name in os.listdir(d):
                if name.endswith(".hdf5") and pat.search(name):
                    return os.path.join(d, name)
        return None

    def _ensure_snapshot_chunk_available(self, chunk):
        fp = self._find_snapshot_chunk_file(chunk)
        if fp is not None:
            return fp

        if not self.download_if_missing or not self.download_snapshots:
            return None

        self._require_download_config()
        target = self._snapshot_chunk_path(chunk)
        self._download_api_file(f"snapshot-{self.snap}.{int(chunk)}.hdf5", target)
        return target

    # =========================================================
    # Internal: offsets file
    # =========================================================

    def _normalize_file_offsets(self, arr):
        """
        Normalize FileOffsets/SnapByType to shape (Nfile, 6).
        """
        a = np.asarray(arr, dtype=np.int64)
        if a.ndim != 2:
            raise ValueError("FileOffsets/SnapByType must be a 2D array")
        if a.shape[1] == 6:
            return a
        if a.shape[0] == 6:
            return a.T
        raise ValueError(f"Unexpected FileOffsets/SnapByType shape: {a.shape}")

    def _read_offsets_file(self):
        """
        Read offsets_XXX.hdf5.

        Returns
        -------
        off_sub : ndarray, shape (Nsub, 6)
            Subhalo particle offsets by type, from Subhalo/SnapByType.
        off_grp : ndarray, shape (Ng, 6)
            FoF group particle offsets by type, from Group/SnapByType.
        """
        fp = self._ensure_offsets_available()

        if fp is None:
            raise FileNotFoundError(
                "TNG offsets file not found. Tried:\n  "
                + "\n  ".join([str(x) for x in self._offset_candidates() if x])
            )

        with h5py.File(fp, "r") as f:
            if "Subhalo" not in f or "Group" not in f:
                raise KeyError(f"Offsets file missing 'Subhalo'/'Group' groups: {fp}")
            if "SnapByType" not in f["Subhalo"] or "SnapByType" not in f["Group"]:
                raise KeyError(f"Offsets file missing '*/SnapByType' datasets: {fp}")

            off_sub = np.asarray(f["Subhalo/SnapByType"][()], dtype=np.int64)
            off_grp = np.asarray(f["Group/SnapByType"][()], dtype=np.int64)

            if "FileOffsets" in f and "SnapByType" in f["FileOffsets"]:
                self._snapshot_file_offsets = self._normalize_file_offsets(f["FileOffsets/SnapByType"][()])

        return off_sub, off_grp

    # =========================================================
    # Internal: reading snapdir_* particle slices
    # =========================================================

    def _read_fields_slice(self, partType, fields, off, n):
        """
        Read multiple fields for PartType slice [off:off+n].

        Optional fields are returned as zeros if missing:
          - ModifiedGravityAcceleration
          - Potential
          - ModifiedGravityPotential
        """
        pt = int(partType)
        n = int(n)

        d = {}
        for k in fields:
            if k == "ModifiedGravityAcceleration":
                try:
                    d[k] = self._read_global_slice(pt, k, off, n)
                except Exception:
                    d[k] = np.zeros((n, 3), dtype=np.float64)
            elif k in ("Potential", "ModifiedGravityPotential"):
                try:
                    d[k] = self._read_global_slice(pt, k, off, n)
                except Exception:
                    d[k] = np.zeros((n,), dtype=np.float64)
            else:
                d[k] = self._read_global_slice(pt, k, off, n)
        return d

    def _read_global_slice(self, partType, field, off, n):
        """
        Read a global particle slice from numerically ordered snapdir chunks.
        Missing snapshot chunks are downloaded lazily when enabled.
        """
        if n <= 0:
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
            if b1 <= b0:
                continue

            if not os.path.isfile(fp):
                got = self._ensure_snapshot_chunk_available(i)
                if got is None or not os.path.isfile(got):
                    raise FileNotFoundError(f"Snapshot chunk {i} not found and automatic download is disabled: {fp}")
                fp = got
                files[i] = fp

            with h5py.File(fp, "r") as f:
                g = f.get(name, None)
                if g is None or field not in g:
                    raise KeyError(f"Field '{field}' not found in {name} of {fp}")
                arr = g[field][b0:b1]

            out = arr if out is None else np.concatenate([out, arr], axis=0)

        if out is None:
            return self._empty_for_field(field)
        return np.asarray(out)

    def _build_snap_index_from_local_files(self, partType, files):
        """
        Build a particle-count prefix array by scanning local HDF5 chunk headers.
        """
        pt = int(partType)
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
        return files, pref

    def _build_snap_index_from_offsets(self, partType):
        """
        Build a logical snapshot index from FileOffsets/SnapByType.

        This supports lazy chunk download. The last upper bound is set to a very
        large integer; slice reads always cap it by the requested ``need1``.
        """
        pt = int(partType)

        if self._snapshot_file_offsets is None:
            # Load offsets if not already loaded; this also normalizes file offsets.
            self._read_offsets_file()

        if self._snapshot_file_offsets is None:
            raise RuntimeError(
                "Cannot build lazy snapshot index because offsets file lacks FileOffsets/SnapByType. "
                "Download the complete snapdir locally or use a full offsets file."
            )

        starts = np.asarray(self._snapshot_file_offsets[:, pt], dtype=np.int64)
        if starts.size == 0:
            raise RuntimeError("FileOffsets/SnapByType contains no snapshot chunks")

        files = []
        for i in range(starts.size):
            files.append(self._find_snapshot_chunk_file(i) or self._snapshot_chunk_path(i))

        pref = np.concatenate([starts, np.asarray([np.iinfo(np.int64).max], dtype=np.int64)])
        return files, pref

    def _build_snap_index(self, partType):
        """
        Build a particle-count prefix array for numerically ordered snapdir chunks.

        Preference order:
        1. complete local snapdir scan, same as the old code;
        2. lazy index from offsets file, enabling on-demand chunk download.
        """
        pt = int(partType)
        if pt in self._snap_index:
            return self._snap_index[pt]

        local_files = self._local_hdf5_files(prefix="snapdir")
        if local_files:
            # If auto-download is disabled, keep original behavior exactly: the
            # local snapdir must be present and readable.
            if not self.download_if_missing:
                files, pref = self._build_snap_index_from_local_files(pt, local_files)
                self._snap_index[pt] = (files, pref)
                return files, pref

            # With auto-download enabled, use the complete local scan only when
            # the number of local chunks matches the API metadata. Otherwise use
            # offsets-based lazy indexing so missing chunks can be fetched.
            try:
                n_expected = self._get_num_files("snapshot")
            except Exception:
                n_expected = None
            if n_expected is not None and len(local_files) >= int(n_expected):
                files, pref = self._build_snap_index_from_local_files(pt, local_files)
                self._snap_index[pt] = (files, pref)
                return files, pref

            # The snapdir is missing some chunks, or the API metadata could not
            # be queried. Prefer offsets-based indexing so local partial data
            # are interpreted with their correct global chunk ids. This also
            # enables lazy downloading when download_snapshots=True.
            try:
                files, pref = self._build_snap_index_from_offsets(pt)
                self._snap_index[pt] = (files, pref)
                return files, pref
            except Exception:
                if n_expected is None:
                    # Last fallback: keep the old local-only behavior when no
                    # reliable offsets are available.
                    files, pref = self._build_snap_index_from_local_files(pt, local_files)
                    self._snap_index[pt] = (files, pref)
                    return files, pref
                raise

        if self.download_if_missing and self.download_snapshots:
            files, pref = self._build_snap_index_from_offsets(pt)
            self._snap_index[pt] = (files, pref)
            return files, pref

        # Original failure mode when no local snapdir is available.
        files = self._list_hdf5_files(prefix="snapdir")
        files, pref = self._build_snap_index_from_local_files(pt, files)
        self._snap_index[pt] = (files, pref)
        return files, pref

    # =========================================================
    # Internal: utilities
    # =========================================================

    def _import_il(self):
        try:
            import illustris_python as il
        except Exception as e:
            raise ImportError("illustris_python is required for TNGCatalog.") from e
        return il

    def _require_fof_loaded(self):
        if self.halos is None or self.subhalos is None:
            raise RuntimeError("FoF catalogs not loaded. Call loadFoF(...) first.")

    def _unique(self, xs):
        """
        Preserve order while removing duplicates.
        """
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    def _compute_group_offsets(self, group_len_type):
        """
        Compute cumulative particle offsets of each FoF group from GroupLenType.
        """
        gl = np.asarray(group_len_type, dtype=np.int64)
        offsets = np.zeros_like(gl, dtype=np.int64)
        offsets[1:, :] = np.cumsum(gl[:-1, :], axis=0)
        return offsets

    def _list_hdf5_files(self, prefix):
        """
        List split HDF5 files in numeric chunk order.

        If automatic download is enabled, this method may create/download the
        requested data product first. Numeric chunk ordering is preserved.
        """
        if prefix == "groups" and self.download_if_missing and self.download_groupcat:
            self._ensure_groupcat_available()

        d = os.path.join(self.base_path, f"{prefix}_{self.snap:03d}")
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Directory not found: {d}")

        files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".hdf5")]
        if not files:
            raise FileNotFoundError(f"No .hdf5 files found in: {d}")

        files.sort(key=_hdf5_chunk_sort_key)
        return files

    def _empty_for_field(self, field):
        if field in (
            "Coordinates", "Velocities",
            "Acceleration", "Accelerations",
            "ModifiedGravityAcceleration",
        ):
            return np.empty((0, 3), dtype=np.float64)
        return np.empty((0,), dtype=np.float64)
