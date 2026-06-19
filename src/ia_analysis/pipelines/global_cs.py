#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""global_cs.py

ClusterSims backend grouped by FoF gid.

Key updates relative to the previous version
-------------------------------------------
- Add explicit finite cleaning before every ShapeKin call.
- Add explicit finite cleaning before every tidal-grid build.
- Keep shape and tidal failures local to a gid / sid instead of aborting a worker.
- Add more diagnostic logging so that invalid aperture, bad particles, or broken
  tidal inputs are visible before ShapeKin emits a generic eigen warning.

NEW (minimal change)
--------------------
- Include Neff in the Shape output dict for both DM and stars.
  Neff is the effective number of particles used in the variance estimate:
      Neff = 1 / sum(w_i^2)
  where w_i are the (possibly mass-normalized) weights used by ShapeKin's
  cached variance stage on the FINAL ellipsoidal subset.
- Tidal_grp is target-exclusive: the target subhalo's own particles are removed
  from the host-group mass distribution before the group tidal tensor is built.
- Tidal_tot and Tidal_tot_mg keep the legacy inclusive definitions from GR and
  MG/fifth-force acceleration fields.
- Add tidal_self, computed from the target subhalo's own all-type mass
  distribution after an independent self-shape ellipsoidal selection.
  All four tidal branches use the same legacy_tidal_sign setting and the same
  tidal units: km^2/s^2/(ckpc/h)^2.

Interface compatibility is preserved:
    compute_many(base_path, snap, sid_array, gid_array, cfg, ...)
    compute_one_subhalo(base_path, snap, sid, gid=None, cfg=None)
"""

import logging
import numpy as np

from ia_analysis.catalogs.catalog_loader import CSCatalog
from ia_analysis.shapes.shape import ShapeKin
from ia_analysis.tides.tidal_field import (
    compute_gravitational_potential,
    grid_potential_and_tidal,
    PotentialInterpolator,
)

logger = logging.getLogger("global_cs")

PTYPE_DM = 1
PTYPE_STAR = 4
PTYPE_GAS = 0
PTYPE_BH = 5


# ============================================================
# Small helpers
# ============================================================


def _nan_shape(include_ddI=True):
    """
    Return a NaN-filled shape dict with the expected schema.

    NOTE:
    - We include Neff here as well, so downstream code can rely on the key
      existing even when the shape/variance computation failed.
    """
    nan3 = np.full(3, np.nan, dtype=np.float64)
    nan33 = np.full((3, 3), np.nan, dtype=np.float64)
    out = dict(
        I=nan33.copy(),
        dI=nan33.copy(),
        ddI=nan33.copy() if include_ddI else None,  # will be removed below if not requested
        mass=np.nan,
        L=nan3.copy(),
        K_tot=np.nan,
        kappa_rot=np.nan,
        # Effective sample size from the variance stage (1/sum w^2); NaN if unavailable.
        Neff=np.nan,
        axis_relerr=nan3.copy(),
        cos_err=nan3.copy(),
        converged=False,
    )
    if not include_ddI:
        out.pop("ddI", None)
    return out


def _nan_tidal(include_mg=True, include_self=True):
    """Return a NaN-filled tidal dict with the expected schema.

    The keys follow the internal output naming.  run_cs.py maps them to
    columnar HDF5 datasets named Tidal_grp, Tidal_tot, Tidal_tot_mg,
    and Tidal_self.
    """
    nan33 = np.full((3, 3), np.nan, dtype=np.float64)
    out = dict(
        tidal_grp=nan33.copy(),
        tidal_tot=nan33.copy(),
    )
    if include_mg:
        out["tidal_tot_mg"] = nan33.copy()
    if include_self:
        out["tidal_self"] = nan33.copy()
    return out


def _to_vec3_array(arr, name):
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return a.reshape(0, 3)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N,3)")
    return a


def _to_vec1_array(arr, name, n_expected=None):
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name} must have shape (N,)")
    if n_expected is not None and a.shape[0] != int(n_expected):
        raise ValueError(f"{name} length mismatch: expected {n_expected}, got {a.shape[0]}")
    return a


def _clean_particle_block(pos, mass=None, vel=None, acc=None, *, label, sid=None, gid=None):
    """
    Remove non-finite rows before passing data to ShapeKin or tidal builders.

    Returns
    -------
    pos, mass, vel, acc, stats
        Cleaned arrays and a small diagnostic dict.
    """
    pos = _to_vec3_array(pos, f"{label}.pos")
    n0 = pos.shape[0]

    if mass is None:
        mass_arr = None
        bad_mass = np.zeros(n0, dtype=bool)
    else:
        mass_arr = _to_vec1_array(mass, f"{label}.mass", n_expected=n0)
        bad_mass = (~np.isfinite(mass_arr)) | (mass_arr <= 0.0)

    if vel is None:
        vel_arr = None
        bad_vel = np.zeros(n0, dtype=bool)
    else:
        vel_arr = _to_vec3_array(vel, f"{label}.vel")
        if vel_arr.shape[0] != n0:
            raise ValueError(f"{label}.vel length mismatch")
        bad_vel = ~np.all(np.isfinite(vel_arr), axis=1)

    if acc is None:
        acc_arr = None
        bad_acc = np.zeros(n0, dtype=bool)
    else:
        acc_arr = _to_vec3_array(acc, f"{label}.acc")
        if acc_arr.shape[0] != n0:
            raise ValueError(f"{label}.acc length mismatch")
        bad_acc = ~np.all(np.isfinite(acc_arr), axis=1)

    bad_pos = ~np.all(np.isfinite(pos), axis=1)
    good = ~(bad_pos | bad_mass | bad_vel | bad_acc)

    n_keep = int(np.count_nonzero(good))
    n_drop = int(n0 - n_keep)
    if n_drop > 0:
        logger.warning(
            "[%s sid=%s gid=%s] dropped %d/%d particles before shape/tidal "
            "(bad_pos=%d bad_mass=%d bad_vel=%d bad_acc=%d)",
            label,
            str(sid),
            str(gid),
            n_drop,
            n0,
            int(np.count_nonzero(bad_pos)),
            int(np.count_nonzero(bad_mass)),
            int(np.count_nonzero(bad_vel)),
            int(np.count_nonzero(bad_acc)),
        )

    pos = pos[good]
    if mass_arr is not None:
        mass_arr = mass_arr[good]
    if vel_arr is not None:
        vel_arr = vel_arr[good]
    if acc_arr is not None:
        acc_arr = acc_arr[good]

    stats = dict(n_in=int(n0), n_keep=int(n_keep), n_drop=int(n_drop))
    return pos, mass_arr, vel_arr, acc_arr, stats


def _count_selected(sk):
    if getattr(sk, "mask", None) is None:
        return 0
    return int(np.count_nonzero(sk.mask))


# ============================================================
# Tidal mass collection
# ============================================================


def _collect_halo_mass_distribution(cat, gid, cfg):
    """Collect raw (pos, mass) arrays for the gid mass distribution tidal estimate."""
    gid = int(gid)
    dm_fixed = float(cfg["dm_particle_mass"])

    pos_list = []
    mass_list = []

    # Gas
    try:
        gas = cat.loadHalos(gid=gid, ptypes=[PTYPE_GAS], fields=["Coordinates", "Masses"])
        gp = np.asarray(gas.get("PartType0", {}).get("Coordinates", []), dtype=np.float64)
        if gp.size:
            gm = np.asarray(gas["PartType0"]["Masses"], dtype=np.float64)
            pos_list.append(gp)
            mass_list.append(gm)
    except Exception as e:
        logger.warning("[gid=%d gas] loadHalos failed for mass distribution. Reason: %s", gid, str(e))

    # DM
    try:
        dm = cat.loadHalos(gid=gid, ptypes=[PTYPE_DM], fields=["Coordinates"])
        dp = np.asarray(dm.get("PartType1", {}).get("Coordinates", []), dtype=np.float64)
        if dp.size:
            dm_m = np.full(dp.shape[0], dm_fixed, dtype=np.float64)
            pos_list.append(dp)
            mass_list.append(dm_m)
    except Exception as e:
        logger.warning("[gid=%d dm] loadHalos failed for mass distribution. Reason: %s", gid, str(e))

    # Stars
    try:
        st = cat.loadHalos(gid=gid, ptypes=[PTYPE_STAR], fields=["Coordinates", "Masses"])
        sp = np.asarray(st.get("PartType4", {}).get("Coordinates", []), dtype=np.float64)
        if sp.size:
            sm = np.asarray(st["PartType4"]["Masses"], dtype=np.float64)
            pos_list.append(sp)
            mass_list.append(sm)
    except Exception as e:
        logger.warning("[gid=%d stars] loadHalos failed for mass distribution. Reason: %s", gid, str(e))

    # BH
    try:
        bh = cat.loadHalos(gid=gid, ptypes=[PTYPE_BH], fields=["Coordinates", "Masses"])
        bp = np.asarray(bh.get("PartType5", {}).get("Coordinates", []), dtype=np.float64)
        if bp.size:
            bm = np.asarray(bh["PartType5"]["Masses"], dtype=np.float64)
            pos_list.append(bp)
            mass_list.append(bm)
    except Exception as e:
        logger.warning("[gid=%d bh] loadHalos failed for mass distribution. Reason: %s", gid, str(e))

    if len(pos_list) == 0:
        raise RuntimeError(f"No mass particles found for gid={gid} in any supported PartType")

    return np.concatenate(pos_list, axis=0), np.concatenate(mass_list, axis=0)


def _exclude_subhalo_particles_from_group(cat, halos, subhalos, gid, sid, pos, mass, ptype):
    """Remove the target subhalo particle slice from the host-group particle block.

    This uses the catalog offset tables rather than position matching, so the
    exclusion is exact as long as the loader preserves the standard AREPO/TNG
    group-contiguous particle ordering.
    """
    gid = int(gid)
    sid = int(sid)
    pt = int(ptype)

    pos = np.asarray(pos)
    mass = np.asarray(mass)

    if pos.shape[0] == 0:
        return pos, mass

    try:
        sub_len = int(np.asarray(subhalos["SubhaloLenType"], dtype=np.int64)[sid, pt])
        if sub_len <= 0:
            return pos, mass

        sub_off = int(np.asarray(subhalos["SubhaloOffsetType"], dtype=np.int64)[sid, pt])

        if "_GroupOffsetType" in halos:
            grp_off = int(np.asarray(halos["_GroupOffsetType"], dtype=np.int64)[gid, pt])
        else:
            grp_len = np.asarray(halos["GroupLenType"], dtype=np.int64)
            grp_off_all = np.zeros_like(grp_len, dtype=np.int64)
            grp_off_all[1:, :] = np.cumsum(grp_len[:-1, :], axis=0)
            grp_off = int(grp_off_all[gid, pt])

        start = sub_off - grp_off
        stop = start + sub_len

        if start < 0 or stop > pos.shape[0]:
            logger.warning(
                "[gid=%d sid=%d PartType%d] invalid local exclusion slice [%d:%d) for group block of length %d; keeping original group mass block",
                gid, sid, pt, start, stop, int(pos.shape[0])
            )
            return pos, mass

        keep = np.ones(pos.shape[0], dtype=bool)
        keep[start:stop] = False
        return pos[keep], mass[keep]

    except Exception as e:
        logger.warning(
            "[gid=%d sid=%d PartType%d] failed to exclude target subhalo particles from group mass block. Reason: %s",
            gid, sid, pt, str(e)
        )
        return pos, mass


# ============================================================
# Main grouped backend
# ============================================================


class GlobalCSGrouped:
    """ClusterSims global pipeline grouped by FoF group (gid)."""

    def __init__(self, base_path, snap, cfg, group_fields=None, subhalo_fields=None):
        self.base_path = str(base_path)
        self.snap = int(snap)
        self.cfg = dict(cfg)
        self.cat = CSCatalog(self.base_path, self.snap)

        if group_fields is None:
            group_fields = []
        if subhalo_fields is None:
            subhalo_fields = []

        group_fields_extra = list(self.cfg.get("group_fields_extra", []) or [])
        sub_fields_extra = list(self.cfg.get("sub_fields_extra", []) or [])

        group_fields = self._unique(list(group_fields) + group_fields_extra + ["GroupFirstSub", "GroupNsubs", "GroupLenType"])
        subhalo_fields = self._unique(list(subhalo_fields) + sub_fields_extra + ["SubhaloPos", "SubhaloVel", "SubhaloLenType", "SubhaloHalfmassRadType"])

        self.halos, self.subhalos = self.cat.loadFoF(group_fields=group_fields, subhalo_fields=subhalo_fields)

    def run(self, sid_array, gid_array):
        sid_array = np.asarray(sid_array, dtype=np.int64)
        gid_array = np.asarray(gid_array, dtype=np.int64)
        if sid_array.shape != gid_array.shape:
            raise ValueError("sid_array and gid_array must have the same shape")

        gids, groups = self._group_by_gid(gid_array)

        results = [None] * sid_array.size
        for gid, idxs in zip(gids, groups):
            sids = sid_array[idxs]
            outs = self._process_one_group(int(gid), sids)
            for j, out in enumerate(outs):
                results[int(idxs[j])] = out
        return results

    def _process_one_group(self, gid, sids):
        gid = int(gid)
        sids = np.asarray(sids, dtype=np.int64)

        try:
            total_tidal_ctx = self._build_total_tidal_context_for_gid(gid)
        except Exception as e:
            logger.warning("[gid=%d] total tidal context build failed; tidal outputs will be NaN. Reason: %s", gid, str(e))
            total_tidal_ctx = None

        outs = []
        for sid in sids:
            sid = int(sid)
            sub_info = self._sub_info(sid, gid=gid)
            tidal = self._build_and_interpolate_exclusive_tidals(
                gid=gid,
                sid=sid,
                center=sub_info.get("pos_abs", None),
                total_tidal_ctx=total_tidal_ctx,
            )

            # Target-dependent self tidal from the target subhalo's own
            # all-type mass distribution after a self-shape selection.
            tidal["tidal_self"] = self._compute_self_tidal(
                sid=sid,
                gid=gid,
                center=sub_info.get("pos_abs", None),
            )

            shape = self._shapes(sid, sub_info)
            outs.append(dict(Sub_info=sub_info, Shape=shape, Tidal=tidal))
        return outs

    def _build_exclusive_group_mass_distribution(self, gid, sid):
        """Collect host-group mass particles after removing the target subhalo itself.

        The returned mass distribution is used only for ``tidal_grp`` so that the
        group tidal tensor represents the external host environment rather than
        the target subhalo's self-contribution.
        """
        gid = int(gid)
        sid = int(sid)
        dm_fixed = float(self.cfg["dm_particle_mass"])

        pos_list = []
        mass_list = []

        # Gas
        try:
            gas = self.cat.loadHalos(gid=gid, ptypes=[PTYPE_GAS], fields=["Coordinates", "Masses"])
            gp = np.asarray(gas.get("PartType0", {}).get("Coordinates", []), dtype=np.float64)
            if gp.size:
                gm = np.asarray(gas["PartType0"]["Masses"], dtype=np.float64)
                gp, gm = _exclude_subhalo_particles_from_group(self.cat, self.halos, self.subhalos, gid, sid, gp, gm, PTYPE_GAS)
                if gp.size:
                    pos_list.append(gp)
                    mass_list.append(gm)
        except Exception as e:
            logger.warning("[gid=%d sid=%d gas] loadHalos failed for exclusive group mass distribution. Reason: %s", gid, sid, str(e))

        # DM
        try:
            dm = self.cat.loadHalos(gid=gid, ptypes=[PTYPE_DM], fields=["Coordinates"])
            dp = np.asarray(dm.get("PartType1", {}).get("Coordinates", []), dtype=np.float64)
            if dp.size:
                dm_m = np.full(dp.shape[0], dm_fixed, dtype=np.float64)
                dp, dm_m = _exclude_subhalo_particles_from_group(self.cat, self.halos, self.subhalos, gid, sid, dp, dm_m, PTYPE_DM)
                if dp.size:
                    pos_list.append(dp)
                    mass_list.append(dm_m)
        except Exception as e:
            logger.warning("[gid=%d sid=%d dm] loadHalos failed for exclusive group mass distribution. Reason: %s", gid, sid, str(e))

        # Stars
        try:
            st = self.cat.loadHalos(gid=gid, ptypes=[PTYPE_STAR], fields=["Coordinates", "Masses"])
            sp = np.asarray(st.get("PartType4", {}).get("Coordinates", []), dtype=np.float64)
            if sp.size:
                sm = np.asarray(st["PartType4"]["Masses"], dtype=np.float64)
                sp, sm = _exclude_subhalo_particles_from_group(self.cat, self.halos, self.subhalos, gid, sid, sp, sm, PTYPE_STAR)
                if sp.size:
                    pos_list.append(sp)
                    mass_list.append(sm)
        except Exception as e:
            logger.warning("[gid=%d sid=%d stars] loadHalos failed for exclusive group mass distribution. Reason: %s", gid, sid, str(e))

        # BH
        try:
            bh = self.cat.loadHalos(gid=gid, ptypes=[PTYPE_BH], fields=["Coordinates", "Masses"])
            bp = np.asarray(bh.get("PartType5", {}).get("Coordinates", []), dtype=np.float64)
            if bp.size:
                bm = np.asarray(bh["PartType5"]["Masses"], dtype=np.float64)
                bp, bm = _exclude_subhalo_particles_from_group(self.cat, self.halos, self.subhalos, gid, sid, bp, bm, PTYPE_BH)
                if bp.size:
                    pos_list.append(bp)
                    mass_list.append(bm)
        except Exception as e:
            logger.warning("[gid=%d sid=%d bh] loadHalos failed for exclusive group mass distribution. Reason: %s", gid, sid, str(e))

        if len(pos_list) == 0:
            raise RuntimeError(f"No external group-mass particles remain for gid={gid}, sid={sid}")

        return np.concatenate(pos_list, axis=0), np.concatenate(mass_list, axis=0)

    # --------------------------------------------------------
    # Subhalo info
    # --------------------------------------------------------

    def _sub_info(self, sid, gid=None):
        sid = int(sid)
        subs = self.subhalos

        gid_cat = int(subs["GroupID"][sid])
        gid = gid_cat if gid is None else int(gid)
        cen = int(subs["CenID"][sid])

        pos = np.asarray(subs["SubhaloPos"][sid], dtype=np.float64)
        vel = np.asarray(subs["SubhaloVel"][sid], dtype=np.float64)
        posc = np.asarray(subs["SubhaloPos"][cen], dtype=np.float64)
        velc = np.asarray(subs["SubhaloVel"][cen], dtype=np.float64)

        out = dict(
            SubhaloID=sid,
            GroupID=int(gid),
            CenID=cen,
            pos_abs=pos,
            vel_abs=vel,
            pos_rel=(pos - posc),
            vel_rel=(vel - velc),
        )

        for k in self.cfg.get("sub_fields_extra", []) or []:
            out[k] = np.asarray(subs[k][sid]) if k in subs else np.nan

        halos = self.halos
        for k in self.cfg.get("group_fields_extra", []) or []:
            out[k] = np.asarray(halos[k][int(gid)]) if k in halos else np.nan

        if int(gid) != gid_cat:
            out["_GroupID_catalog"] = gid_cat

        return out

    # --------------------------------------------------------
    # Shapes
    # --------------------------------------------------------

    def _pack_shapekin(self, sk, *, include_ddI=True):
        """
        Pack a ShapeKin instance to the output schema.

        NEW:
        - Add Neff (effective sample size) to output.
          Neff is computed inside ShapeKin's variance stage on the FINAL subset.
        """
        nan33 = np.full((3, 3), np.nan, dtype=np.float64)

        if sk.mass is None:
            mass_tot = float(_count_selected(sk)) if sk.mask is not None else float("nan")
        else:
            if sk.mask is None:
                mass_tot = float("nan")
            else:
                mass_tot = float(np.sum(sk.mass[sk.mask]))

        lam = np.maximum(np.abs(np.asarray(sk.evals, dtype=np.float64)), 1e-30)
        var_e = np.asarray(sk.var_evals, dtype=np.float64)
        var_cos = np.asarray(sk.var_cos, dtype=np.float64)

        axis_relerr = 0.5 * np.sqrt(np.maximum(var_e, 0.0)) / lam
        cos_err = np.sqrt(np.maximum(var_cos, 0.0))

        # Neff is cached by ShapeKin (NaN if variance stage failed or was not computed).
        neff = float(getattr(sk, "Neff", np.nan))

        L, _ = sk.L()
        K_tot = sk.K()
        kappa_rot = sk.kappa()["kappa_rot"]

        out = dict(
            I=np.asarray(sk.I, dtype=np.float64) if sk.I is not None else nan33.copy(),
            dI=np.asarray(sk.dI, dtype=np.float64) if sk.dI is not None else nan33.copy(),
            mass=float(mass_tot),
            L=np.asarray(L, dtype=np.float64),
            K_tot=float(K_tot),
            kappa_rot=float(kappa_rot),
            Neff=neff,
            axis_relerr=np.asarray(axis_relerr, dtype=np.float64),
            cos_err=np.asarray(cos_err, dtype=np.float64),
            converged=bool(getattr(sk, "converged", False)),
        )
        if include_ddI:
            out["ddI"] = np.asarray(sk.ddI, dtype=np.float64) if sk.ddI is not None else nan33.copy()
        return out

    def _run_shape_component(self, pos, mass, vel, acc, center, *, sid, gid, label, percentile_key, r_ell=None, include_ddI=True):
        try:
            if center is None:
                logger.warning("[%s sid=%d gid=%d] missing center; returning NaN shape", label, sid, gid)
                return _nan_shape(include_ddI=include_ddI)

            center = np.asarray(center, dtype=np.float64)
            if center.shape != (3,) or (not np.all(np.isfinite(center))):
                logger.warning("[%s sid=%d gid=%d] non-finite center=%s; returning NaN shape", label, sid, gid, repr(center))
                return _nan_shape(include_ddI=include_ddI)

            pos, mass, vel, acc, stats = _clean_particle_block(
                pos,
                mass=mass,
                vel=vel,
                acc=acc,
                label=label,
                sid=sid,
                gid=gid,
            )
            if stats["n_keep"] < 3:
                logger.warning("[%s sid=%d gid=%d] only %d finite particles remain after cleaning; returning NaN shape", label, sid, gid, stats["n_keep"])
                return _nan_shape(include_ddI=include_ddI)

            if r_ell is not None:
                rval = float(r_ell)
                if (not np.isfinite(rval)) or rval <= 0.0:
                    logger.warning("[%s sid=%d gid=%d] invalid r_ell=%s; returning NaN shape", label, sid, gid, str(r_ell))
                    return _nan_shape(include_ddI=include_ddI)

            sk = ShapeKin(
                particles=pos,
                masses=mass,
                velocities=vel,
                accelerations=acc,
                Pos=center,
            )
            sk.run_shape(
                percentile=float(self.cfg.get(percentile_key, 100.0)),
                max_iter=int(self.cfg.get("shape_max_iter", 100)),
                tol=float(self.cfg.get("shape_tol", 0.01)),
                r_ell=r_ell,
                tensor_mode=str(self.cfg.get("shape_tensor_mode", "reduced")),
                return_dI=True,
                return_ddI=bool(include_ddI),
            )

            nsel = _count_selected(sk)
            if nsel < 3:
                logger.warning("[%s sid=%d gid=%d] ShapeKin selected only %d particles; output will be NaN-like", label, sid, gid, nsel)
            if not bool(getattr(sk, "converged", False)):
                logger.warning("[%s sid=%d gid=%d] shape iteration did not reach tolerance; using last iterate", label, sid, gid)
            if not np.all(np.isfinite(np.asarray(sk.I, dtype=np.float64))):
                logger.warning("[%s sid=%d gid=%d] non-finite tensor returned by ShapeKin", label, sid, gid)

            return self._pack_shapekin(sk, include_ddI=include_ddI)

        except Exception as e:
            logger.warning("[%s sid=%d gid=%d] shape computation failed; returning NaNs. Reason: %s", label, sid, gid, str(e))
            return _nan_shape(include_ddI=include_ddI)

    def _shapes(self, sid, sub_info):
        sid = int(sid)
        gid = int(sub_info.get("GroupID", -1))
        center = sub_info.get("pos_abs", None)
        subs = self.subhalos

        # DM
        try:
            dm = self.cat.loadSubhalos(sid=sid, ptypes=[PTYPE_DM], fields=["Coordinates", "Velocities", "Acceleration"])
            dm_pos = np.asarray(dm["PartType1"]["Coordinates"], dtype=np.float64)
            dm_vel = np.asarray(dm["PartType1"]["Velocities"], dtype=np.float64)
            dm_acc = np.asarray(dm["PartType1"]["Acceleration"], dtype=np.float64)
            dm_mass = np.full(dm_pos.shape[0], float(self.cfg["dm_particle_mass"]), dtype=np.float64)
        except Exception as e:
            logger.warning("[dm sid=%d gid=%d] loadSubhalos failed; returning NaN shape. Reason: %s", sid, gid, str(e))
            dm_shape = _nan_shape(include_ddI=True)
        else:
            dm_shape = self._run_shape_component(
                dm_pos,
                dm_mass,
                dm_vel,
                dm_acc,
                center,
                sid=sid,
                gid=gid,
                label="dm",
                percentile_key="dm_shape_percentile",
                r_ell=None,
                include_ddI=True,
            )

        # Stars
        try:
            st = self.cat.loadSubhalos(sid=sid, ptypes=[PTYPE_STAR], fields=["Coordinates", "Velocities", "Acceleration", "Masses"])
            st_pos = np.asarray(st["PartType4"]["Coordinates"], dtype=np.float64)
            st_vel = np.asarray(st["PartType4"]["Velocities"], dtype=np.float64)
            st_acc = np.asarray(st["PartType4"]["Acceleration"], dtype=np.float64)
            st_mass = np.asarray(st["PartType4"]["Masses"], dtype=np.float64)
            r_half = float(np.asarray(subs["SubhaloHalfmassRadType"][sid, 4], dtype=np.float64))
            r_ap = float(self.cfg.get("star_aperture_factor", 2.0)) * r_half
            if (not np.isfinite(r_half)) or (not np.isfinite(r_ap)) or r_ap <= 0.0:
                logger.warning(
                    "[stars sid=%d gid=%d] invalid stellar aperture: halfmass_radius=%s, factor=%s, r_ell=%s",
                    sid,
                    gid,
                    str(r_half),
                    str(self.cfg.get("star_aperture_factor", 2.0)),
                    str(r_ap),
                )
        except Exception as e:
            logger.warning("[stars sid=%d gid=%d] loadSubhalos failed; returning NaN shape. Reason: %s", sid, gid, str(e))
            st_shape = _nan_shape(include_ddI=True)
        else:
            st_shape = self._run_shape_component(
                st_pos,
                st_mass,
                st_vel,
                st_acc,
                center,
                sid=sid,
                gid=gid,
                label="stars",
                percentile_key="star_shape_percentile",
                r_ell=r_ap,
                include_ddI=True,
            )

        return dict(dm=dm_shape, stars=st_shape)

    # --------------------------------------------------------
    # Tidals
    # --------------------------------------------------------

    @staticmethod
    def _make_interpolator_from_flat(res):
        coords = np.asarray(res["coordinates"], dtype=np.float64)
        x = np.unique(coords[:, 0])
        y = np.unique(coords[:, 1])
        z = np.unique(coords[:, 2])
        nx, ny, nz = len(x), len(y), len(z)

        phi = np.asarray(res["potential"], dtype=np.float64).reshape(nx, ny, nz)
        Txx = np.asarray(res["Txx"], dtype=np.float64).reshape(nx, ny, nz)
        Txy = np.asarray(res["Txy"], dtype=np.float64).reshape(nx, ny, nz)
        Txz = np.asarray(res["Txz"], dtype=np.float64).reshape(nx, ny, nz)
        Tyy = np.asarray(res["Tyy"], dtype=np.float64).reshape(nx, ny, nz)
        Tyz = np.asarray(res["Tyz"], dtype=np.float64).reshape(nx, ny, nz)
        Tzz = np.asarray(res["Tzz"], dtype=np.float64).reshape(nx, ny, nz)

        tidal6 = np.stack([Txx, Txy, Txz, Tyy, Tyz, Tzz], axis=0)
        return PotentialInterpolator((x, y, z), phi, tidal6, bounds_error=False, fill_value=0.0)

    def _collect_subhalo_self_mass_distribution(self, sid):
        """Collect the target subhalo's own all-type mass distribution.

        This is used only for ``tidal_self``.  The particles are collected from
        the target subhalo itself (gas + DM + stars + BH when available).  The
        returned block is later passed through an independent ShapeKin
        ellipsoidal selection, and only the final selected particles are used
        to build the self tidal tensor.
        """
        sid = int(sid)
        dm_fixed = float(self.cfg["dm_particle_mass"])

        pos_list = []
        mass_list = []

        # Gas
        try:
            gas = self.cat.loadSubhalos(sid=sid, ptypes=[PTYPE_GAS], fields=["Coordinates", "Masses"])
            gp = np.asarray(gas.get("PartType0", {}).get("Coordinates", []), dtype=np.float64)
            if gp.size:
                gm = np.asarray(gas["PartType0"]["Masses"], dtype=np.float64)
                pos_list.append(gp)
                mass_list.append(gm)
        except Exception as e:
            logger.warning("[sid=%d gas] loadSubhalos failed for self tidal mass distribution. Reason: %s", sid, str(e))

        # DM
        try:
            dm = self.cat.loadSubhalos(sid=sid, ptypes=[PTYPE_DM], fields=["Coordinates"])
            dp = np.asarray(dm.get("PartType1", {}).get("Coordinates", []), dtype=np.float64)
            if dp.size:
                dm_m = np.full(dp.shape[0], dm_fixed, dtype=np.float64)
                pos_list.append(dp)
                mass_list.append(dm_m)
        except Exception as e:
            logger.warning("[sid=%d dm] loadSubhalos failed for self tidal mass distribution. Reason: %s", sid, str(e))

        # Stars
        try:
            st = self.cat.loadSubhalos(sid=sid, ptypes=[PTYPE_STAR], fields=["Coordinates", "Masses"])
            sp = np.asarray(st.get("PartType4", {}).get("Coordinates", []), dtype=np.float64)
            if sp.size:
                sm = np.asarray(st["PartType4"]["Masses"], dtype=np.float64)
                pos_list.append(sp)
                mass_list.append(sm)
        except Exception as e:
            logger.warning("[sid=%d stars] loadSubhalos failed for self tidal mass distribution. Reason: %s", sid, str(e))

        # BH
        try:
            bh = self.cat.loadSubhalos(sid=sid, ptypes=[PTYPE_BH], fields=["Coordinates", "Masses"])
            bp = np.asarray(bh.get("PartType5", {}).get("Coordinates", []), dtype=np.float64)
            if bp.size:
                bm = np.asarray(bh["PartType5"]["Masses"], dtype=np.float64)
                pos_list.append(bp)
                mass_list.append(bm)
        except Exception as e:
            logger.warning("[sid=%d bh] loadSubhalos failed for self tidal mass distribution. Reason: %s", sid, str(e))

        if len(pos_list) == 0:
            raise RuntimeError(f"No self particles found for sid={sid} in any supported PartType")

        return np.concatenate(pos_list, axis=0), np.concatenate(mass_list, axis=0)

    def _compute_self_tidal(self, sid, gid, center):
        """Compute ``tidal_self`` from the target object's own all-type mass distribution.

        Procedure
        ---------
        1. Collect gas + DM + stars + BH particles belonging to the target subhalo.
        2. Run an independent ShapeKin ellipsoidal iteration on these particles.
        3. Use only the final selected particles to build a Newtonian potential grid.
        4. Interpolate the resulting tidal tensor at the subhalo center.

        Sign and units
        --------------
        This uses ``compute_gravitational_potential`` with the same
        ``legacy_tidal_sign`` as ``tidal_grp``.  Therefore ``tidal_self`` and
        ``tidal_grp`` have the same sign convention and units:
        km^2/s^2/(ckpc/h)^2.
        """
        nan33 = np.full((3, 3), np.nan, dtype=np.float64)
        sid = int(sid)
        gid = int(gid)

        try:
            if center is None:
                raise ValueError("missing center")
            center = np.asarray(center, dtype=np.float64)
            if center.shape != (3,) or not np.all(np.isfinite(center)):
                raise ValueError(f"invalid center={center!r}")

            pos_raw, mass_raw = self._collect_subhalo_self_mass_distribution(sid)
            pos, mass, _, _, stats = _clean_particle_block(
                pos_raw,
                mass=mass_raw,
                label="tidal_self_mass",
                sid=sid,
                gid=gid,
            )
            if stats["n_keep"] < 4:
                raise RuntimeError(f"insufficient finite self particles ({stats['n_keep']})")

            sk = ShapeKin(particles=pos, masses=mass, velocities=None, accelerations=None, Pos=center)
            sk.run_shape(
                percentile=float(self.cfg.get("self_shape_percentile", self.cfg.get("dm_shape_percentile", 100.0))),
                max_iter=int(self.cfg.get("shape_max_iter", 100)),
                tol=float(self.cfg.get("shape_tol", 0.01)),
                r_ell=None,
                tensor_mode=str(self.cfg.get("shape_tensor_mode", "reduced")),
                return_dI=False,
                return_ddI=False,
            )

            mask = getattr(sk, "mask", None)
            nsel = 0 if mask is None else int(np.count_nonzero(mask))
            if nsel < 4:
                raise RuntimeError(f"self-shape selected too few particles ({nsel})")

            pos_sel = pos[mask]
            mass_sel = mass[mask]

            ngrid = int(self.cfg.get("tidal_grid_size", 128))
            pad = float(self.cfg.get("tidal_padding", 0.2))
            soft = float(self.cfg.get("tidal_softening", 0.01))
            legacy_sign = bool(self.cfg.get("legacy_tidal_sign", True))

            res_self = compute_gravitational_potential(
                pos_sel,
                mass_sel,
                grid_size=ngrid,
                boundary_padding=pad,
                softening=soft,
                legacy_tidal_sign=legacy_sign,
                nthreads=self.cfg.get("tidal_nthreads", None),
            )
            interp_self = self._make_interpolator_from_flat(res_self)
            _, T_self = interp_self(center)
            return np.asarray(T_self, dtype=np.float64)

        except Exception as e:
            logger.warning("[sid=%d gid=%d] self tidal computation failed; returning NaNs. Reason: %s", sid, gid, str(e))
            return nan33.copy()

    def _build_total_tidal_context_for_gid(self, gid):
        """Build gid-shared total tidal interpolators.

        Only ``tidal_tot`` and ``tidal_tot_mg`` are gid-shared.  The group tidal
        branch ``tidal_grp`` is target-dependent because it excludes the target
        subhalo's own particles and is therefore built per sid.
        """
        gid = int(gid)
        ngrid = int(self.cfg.get("tidal_grid_size", 128))
        pad = float(self.cfg.get("tidal_padding", 0.2))
        soft = float(self.cfg.get("tidal_softening", 0.01))
        legacy_sign = bool(self.cfg.get("legacy_tidal_sign", True))

        # total tidal from GR acceleration on group DM particles
        dm_grp = self.cat.loadHalos(
            gid=gid,
            ptypes=[PTYPE_DM],
            fields=["Coordinates", "Acceleration", "ModifiedGravityAcceleration"],
        )
        dm_pos_raw = np.asarray(dm_grp["PartType1"]["Coordinates"], dtype=np.float64)
        dm_acc_raw = np.asarray(dm_grp["PartType1"]["Acceleration"], dtype=np.float64)
        dm_mg_acc_raw = np.asarray(dm_grp["PartType1"]["ModifiedGravityAcceleration"], dtype=np.float64)

        dm_pos_gr, _, _, dm_acc, stats_gr = _clean_particle_block(
            dm_pos_raw,
            acc=dm_acc_raw,
            label="tidal_dm_gr",
            gid=gid,
        )
        if stats_gr["n_keep"] < 4:
            raise RuntimeError(f"gid={gid}: insufficient finite DM+GR-acc particles for total tidal ({stats_gr['n_keep']})")

        res_gr = grid_potential_and_tidal(
            dm_pos_gr,
            dm_acc,
            grid_size=ngrid,
            boundary_padding=pad,
            input_type="acceleration",
            reconstruct_potential_from_acc=True,
            legacy_tidal_sign=legacy_sign,
        )
        interp_gr = self._make_interpolator_from_flat(res_gr)

        dm_pos_mg, _, _, dm_mg_acc, stats_mg = _clean_particle_block(
            dm_pos_raw,
            acc=dm_mg_acc_raw,
            label="tidal_dm_mg",
            gid=gid,
        )
        if stats_mg["n_keep"] < 4:
            raise RuntimeError(f"gid={gid}: insufficient finite DM+MG-acc particles for MG tidal ({stats_mg['n_keep']})")

        res_mg = grid_potential_and_tidal(
            dm_pos_mg,
            dm_mg_acc,
            grid_size=ngrid,
            boundary_padding=pad,
            input_type="acceleration",
            reconstruct_potential_from_acc=True,
            legacy_tidal_sign=legacy_sign,
        )
        interp_mg = self._make_interpolator_from_flat(res_mg)

        return dict(interp_gr=interp_gr, interp_mg=interp_mg)

    def _build_and_interpolate_exclusive_tidals(self, gid, sid, center, total_tidal_ctx):
        """Build target-exclusive ``tidal_grp`` and combine it with gid-shared totals."""
        gid = int(gid)
        sid = int(sid)

        if center is None:
            logger.warning("[sid=%d gid=%d] missing center for tidal interpolation; returning NaNs", sid, gid)
            return _nan_tidal(include_mg=True)

        center = np.asarray(center, dtype=np.float64)
        if center.shape != (3,) or (not np.all(np.isfinite(center))):
            logger.warning("[sid=%d gid=%d] non-finite tidal interpolation center=%s; returning NaNs", sid, gid, repr(center))
            return _nan_tidal(include_mg=True)

        ngrid = int(self.cfg.get("tidal_grid_size", 128))
        pad = float(self.cfg.get("tidal_padding", 0.2))
        soft = float(self.cfg.get("tidal_softening", 0.01))
        legacy_sign = bool(self.cfg.get("legacy_tidal_sign", True))

        try:
            pos_mass_raw, m_mass_raw = self._build_exclusive_group_mass_distribution(gid, sid)
            pos_mass, m_mass, _, _, stats_mass = _clean_particle_block(
                pos_mass_raw,
                mass=m_mass_raw,
                label="tidal_mass_exclusive",
                gid=gid,
                sid=sid,
            )
            if stats_mass["n_keep"] < 4:
                raise RuntimeError(
                    f"gid={gid} sid={sid}: insufficient finite external mass particles for exclusive group tidal ({stats_mass['n_keep']})"
                )

            res_mass = compute_gravitational_potential(
                pos_mass,
                m_mass,
                grid_size=ngrid,
                boundary_padding=pad,
                softening=soft,
                legacy_tidal_sign=legacy_sign,
                nthreads=self.cfg.get("tidal_nthreads", None),
            )
            interp_grp = self._make_interpolator_from_flat(res_mass)

            _, T_grp = interp_grp(center)
            if total_tidal_ctx is None:
                return dict(
                    tidal_grp=np.asarray(T_grp, dtype=np.float64),
                    tidal_tot=np.full((3, 3), np.nan, dtype=np.float64),
                    tidal_tot_mg=np.full((3, 3), np.nan, dtype=np.float64),
                )

            _, T_tot = total_tidal_ctx["interp_gr"](center)
            _, T_mg = total_tidal_ctx["interp_mg"](center)

            return dict(
                tidal_grp=np.asarray(T_grp, dtype=np.float64),
                tidal_tot=np.asarray(T_tot, dtype=np.float64),
                tidal_tot_mg=np.asarray(T_mg, dtype=np.float64),
            )

        except Exception as e:
            logger.warning("[sid=%d gid=%d] exclusive group tidal build/interpolation failed; returning NaNs. Reason: %s", sid, gid, str(e))
            if total_tidal_ctx is None:
                return _nan_tidal(include_mg=True)
            try:
                _, T_tot = total_tidal_ctx["interp_gr"](center)
                _, T_mg = total_tidal_ctx["interp_mg"](center)
                return dict(
                    tidal_grp=np.full((3, 3), np.nan, dtype=np.float64),
                    tidal_tot=np.asarray(T_tot, dtype=np.float64),
                    tidal_tot_mg=np.asarray(T_mg, dtype=np.float64),
                )
            except Exception:
                return _nan_tidal(include_mg=True)

    def _interpolate_tidals(self, center, tidal_ctx, *, gid=None, sid=None):
        if center is None:
            logger.warning("[sid=%s gid=%s] missing center for tidal interpolation; returning NaNs", str(sid), str(gid))
            return _nan_tidal(include_mg=True)

        center = np.asarray(center, dtype=np.float64)
        if center.shape != (3,) or (not np.all(np.isfinite(center))):
            logger.warning("[sid=%s gid=%s] non-finite tidal interpolation center=%s; returning NaNs", str(sid), str(gid), repr(center))
            return _nan_tidal(include_mg=True)

        try:
            _, T_grp = tidal_ctx["interp_grp"](center)
            _, T_tot = tidal_ctx["interp_gr"](center)
            _, T_mg = tidal_ctx["interp_mg"](center)
        except Exception as e:
            logger.warning("[sid=%s gid=%s] tidal interpolation failed; returning NaNs. Reason: %s", str(sid), str(gid), str(e))
            return _nan_tidal(include_mg=True)

        return dict(
            tidal_grp=np.asarray(T_grp, dtype=np.float64),
            tidal_tot=np.asarray(T_tot, dtype=np.float64),
            tidal_tot_mg=np.asarray(T_mg, dtype=np.float64),
        )

    # --------------------------------------------------------
    # Utility helpers
    # --------------------------------------------------------

    @staticmethod
    def _unique(xs):
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    @staticmethod
    def _group_by_gid(gid_array):
        gid_array = np.asarray(gid_array, dtype=np.int64)
        first_order = {}
        for g in gid_array:
            g = int(g)
            if g not in first_order:
                first_order[g] = len(first_order)

        order_key = np.array([first_order[int(g)] for g in gid_array], dtype=np.int64)
        sort_idx = np.lexsort((np.arange(gid_array.size), order_key))

        gids = []
        groups = []
        cur_g = None
        cur = []
        for idx in sort_idx:
            g = int(gid_array[idx])
            if cur_g is None:
                cur_g = g
            if g != cur_g:
                gids.append(cur_g)
                groups.append(np.asarray(cur, dtype=np.int64))
                cur_g = g
                cur = [int(idx)]
            else:
                cur.append(int(idx))
        if cur_g is not None:
            gids.append(cur_g)
            groups.append(np.asarray(cur, dtype=np.int64))

        return np.asarray(gids, dtype=np.int64), groups


# ============================================================
# Public API
# ============================================================


def compute_many(base_path, snap, sid_array, gid_array, cfg, group_fields=None, subhalo_fields=None):
    state = GlobalCSGrouped(base_path, snap, cfg, group_fields=group_fields, subhalo_fields=subhalo_fields)
    return state.run(sid_array, gid_array)


def compute_one_subhalo(base_path, snap, sid, gid=None, cfg=None):
    if cfg is None:
        raise ValueError("cfg must be provided")
    state = GlobalCSGrouped(base_path, snap, cfg)
    if gid is None:
        gid = int(state.subhalos["GroupID"][int(sid)])
    return state.run([int(sid)], [int(gid)])[0]
