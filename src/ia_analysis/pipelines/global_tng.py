#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""global_tng_updated.py

TNG backend grouped by FoF gid.

Key updates relative to the previous version
-------------------------------------------
- Add explicit finite cleaning before every ShapeKin call.
- Add explicit finite cleaning before tidal-grid construction from mass or potential.
- Keep tidal failures local to a gid instead of aborting the whole worker.
- Emit more diagnostic logs for invalid apertures and broken particle blocks.
- Pass optional TNGCatalog API/cache options through the full pipeline.

TNG constraints are unchanged
-----------------------------
- No per-particle acceleration is used.
- No MG tidal branch.
- Shape outputs contain I and dI, but no ddI.

Output data structure
---------------------
Each element of the returned list (or the single dict from compute_one_subhalo)
has the schema:

{
  "Sub_info": {
      "SubhaloID": int,
      "GroupID": int,
      "CenID": int,
      "pos_abs": (3,) float,
      "vel_abs": (3,) float,
      "pos_rel": (3,) float,
      "vel_rel": (3,) float,
      ... plus any cfg["sub_fields_extra"] and cfg["group_fields_extra"]
  },
  "Shape": {
      "dm": {
          "I": (3,3) float,
          "dI": (3,3) float,
          "mass": float,              # total mass (or count if masses unavailable)
          "L": (3,) float,            # total angular momentum vector
          "K_tot": float,             # total kinetic energy
          "kappa_rot": float,         # rotational kinetic fraction
          "Neff": float,              # effective sample size from variance stage
          "axis_relerr": (3,) float,  # relative errors of sqrt(|lambda_i|)
          "cos_err": (3,) float,      # rms uncertainty proxy of principal-axis cosines
          "converged": bool
      },
      "stars": { ... same keys as dm ... }
  },
  "Tidal": {
      "tidal_grp": (3,3) float,       # target-exclusive host/group mass tidal tensor
      "tidal_tot": (3,3) float,       # inclusive tidal tensor from DM potential samples
      "tidal_self": (3,3) float       # target self tidal tensor from selected all-type particles
  }
}

Notes
-----
- No "tidal_tot_mg" branch for TNG because no MG/fifth-force potential is available.
- No "ddI" output for TNG (accelerations are not used).
- ``tidal_grp`` and ``tidal_self`` both use the mass-to-potential branch of
  tidal_field.py with the same ``legacy_tidal_sign`` setting. ``tidal_tot``
  uses the potential-to-tidal branch with the same sign setting, so all
  available TNG tidal tensors are sign-consistent and directly comparable.
- If a component fails, its fields are filled with NaNs but other components
  are still computed and written.

"""

import logging
import numpy as np

from ia_analysis.catalogs.TNGCatLoader import TNGCatalog
from ia_analysis.shapes.shape import ShapeKin
from ia_analysis.tides.tidal_field import (
    compute_gravitational_potential,
    grid_potential_and_tidal,
    PotentialInterpolator,
)

logger = logging.getLogger("global_tng")

PTYPE_DM = 1
PTYPE_STAR = 4
PTYPE_GAS = 0
PTYPE_BH = 5


# ============================================================
# Small helpers
# ============================================================


def _nan_shape():
    nan3 = np.full(3, np.nan, dtype=np.float64)
    nan33 = np.full((3, 3), np.nan, dtype=np.float64)
    return dict(
        I=nan33.copy(),
        dI=nan33.copy(),
        mass=np.nan,
        L=nan3.copy(),
        K_tot=np.nan,
        kappa_rot=np.nan,
        Neff=np.nan,
        axis_relerr=nan3.copy(),
        cos_err=nan3.copy(),
        converged=False,
    )


def _nan_tidal(include_self=True):
    """Return a NaN-filled tidal dict with the expected TNG schema."""
    nan33 = np.full((3, 3), np.nan, dtype=np.float64)
    out = dict(
        tidal_grp=nan33.copy(),
        tidal_tot=nan33.copy(),
    )
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


def _clean_particle_block(pos, mass=None, vel=None, scalar=None, *, label, sid=None, gid=None):
    """
    Generic cleaner for TNG particle blocks.

    Parameters
    ----------
    scalar : array-like or None
        Optional per-particle scalar field, used here for the DM potential block.
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

    if scalar is None:
        scalar_arr = None
        bad_scalar = np.zeros(n0, dtype=bool)
    else:
        scalar_arr = _to_vec1_array(scalar, f"{label}.scalar", n_expected=n0)
        bad_scalar = ~np.isfinite(scalar_arr)

    bad_pos = ~np.all(np.isfinite(pos), axis=1)
    good = ~(bad_pos | bad_mass | bad_vel | bad_scalar)

    n_keep = int(np.count_nonzero(good))
    n_drop = int(n0 - n_keep)
    if n_drop > 0:
        logger.warning(
            "[%s sid=%s gid=%s] dropped %d/%d particles before shape/tidal "
            "(bad_pos=%d bad_mass=%d bad_vel=%d bad_scalar=%d)",
            label,
            str(sid),
            str(gid),
            n_drop,
            n0,
            int(np.count_nonzero(bad_pos)),
            int(np.count_nonzero(bad_mass)),
            int(np.count_nonzero(bad_vel)),
            int(np.count_nonzero(bad_scalar)),
        )

    pos = pos[good]
    if mass_arr is not None:
        mass_arr = mass_arr[good]
    if vel_arr is not None:
        vel_arr = vel_arr[good]
    if scalar_arr is not None:
        scalar_arr = scalar_arr[good]

    stats = dict(n_in=int(n0), n_keep=int(n_keep), n_drop=int(n_drop))
    return pos, mass_arr, vel_arr, scalar_arr, stats


def _unique(xs):
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _group_by_gid_stable(gid_array):
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


def _infer_grid_shape_from_coords(coords):
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    return int(np.unique(x).size), int(np.unique(y).size), int(np.unique(z).size)


def _res_to_interpolator(res):
    coords = np.asarray(res["coordinates"], dtype=np.float64)
    nx, ny, nz = _infer_grid_shape_from_coords(coords)

    phi = np.asarray(res["potential"], dtype=np.float64).reshape((nx, ny, nz))
    tidal6 = np.stack(
        [
            np.asarray(res["Txx"], dtype=np.float64).reshape((nx, ny, nz)),
            np.asarray(res["Txy"], dtype=np.float64).reshape((nx, ny, nz)),
            np.asarray(res["Txz"], dtype=np.float64).reshape((nx, ny, nz)),
            np.asarray(res["Tyy"], dtype=np.float64).reshape((nx, ny, nz)),
            np.asarray(res["Tyz"], dtype=np.float64).reshape((nx, ny, nz)),
            np.asarray(res["Tzz"], dtype=np.float64).reshape((nx, ny, nz)),
        ],
        axis=0,
    )

    xcoords = np.unique(coords[:, 0])
    ycoords = np.unique(coords[:, 1])
    zcoords = np.unique(coords[:, 2])
    return PotentialInterpolator((xcoords, ycoords, zcoords), phi, tidal6, bounds_error=False, fill_value=0.0)


# ============================================================
# Tidal mass collection
# ============================================================


def _collect_halo_mass_distribution(cat, gid, cfg):
    gid = int(gid)
    dm_fixed = float(cfg["dm_particle_mass"])

    choice = cfg.get("tidal_mass_ptypes", "all")
    if choice is None or choice == "all":
        use = {"gas", "dm", "stars", "bh"}
    else:
        use = set(choice)

    pos_list = []
    mass_list = []

    if "gas" in use:
        try:
            gas = cat.loadHalos(gid=gid, ptypes=[PTYPE_GAS], fields=["Coordinates", "Masses"])
            gp = np.asarray(gas.get("PartType0", {}).get("Coordinates", []), dtype=np.float64)
            if gp.size:
                gm = np.asarray(gas["PartType0"]["Masses"], dtype=np.float64)
                pos_list.append(gp)
                mass_list.append(gm)
        except Exception as e:
            logger.warning("[gid=%d gas] loadHalos failed for mass distribution. Reason: %s", gid, str(e))

    if "dm" in use:
        try:
            dm = cat.loadHalos(gid=gid, ptypes=[PTYPE_DM], fields=["Coordinates"])
            dp = np.asarray(dm.get("PartType1", {}).get("Coordinates", []), dtype=np.float64)
            if dp.size:
                dm_m = np.full(dp.shape[0], dm_fixed, dtype=np.float64)
                pos_list.append(dp)
                mass_list.append(dm_m)
        except Exception as e:
            logger.warning("[gid=%d dm] loadHalos failed for mass distribution. Reason: %s", gid, str(e))

    if "stars" in use:
        try:
            st = cat.loadHalos(gid=gid, ptypes=[PTYPE_STAR], fields=["Coordinates", "Masses"])
            sp = np.asarray(st.get("PartType4", {}).get("Coordinates", []), dtype=np.float64)
            if sp.size:
                sm = np.asarray(st["PartType4"]["Masses"], dtype=np.float64)
                pos_list.append(sp)
                mass_list.append(sm)
        except Exception as e:
            logger.warning("[gid=%d stars] loadHalos failed for mass distribution. Reason: %s", gid, str(e))

    if "bh" in use:
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
    """Remove the target subhalo particle slice from the host-group particle block."""
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


class GlobalTNGGrouped:
    """TNG global pipeline grouped by FoF gid.

    Parameters
    ----------
    tng_catalog_kwargs : dict or None
        Optional keyword arguments forwarded to ``TNGCatalog``.  This keeps
        the original calling pattern unchanged while allowing the driver to
        provide an API key, cache directory, download policy, or simulation
        name when local TNG files are missing.
    """

    def __init__(
        self,
        base_path,
        snap,
        cfg,
        group_fields=None,
        subhalo_fields=None,
        tng_catalog_kwargs=None,
    ):
        self.base_path = str(base_path)
        self.snap = int(snap)
        self.cfg = dict(cfg)
        self.tng_catalog_kwargs = dict(tng_catalog_kwargs or {})
        self.cat = TNGCatalog(self.base_path, self.snap, **self.tng_catalog_kwargs)

        if group_fields is None:
            group_fields = []
        if subhalo_fields is None:
            subhalo_fields = []

        group_fields_extra = list(self.cfg.get("group_fields_extra", []) or [])
        sub_fields_extra = list(self.cfg.get("sub_fields_extra", []) or [])

        group_fields = _unique(list(group_fields) + group_fields_extra + ["GroupFirstSub", "GroupNsubs", "GroupLenType"])
        subhalo_fields = _unique(list(subhalo_fields) + sub_fields_extra + ["SubhaloPos", "SubhaloVel", "SubhaloLenType", "SubhaloGrNr", "SubhaloHalfmassRadType"])

        self.halos, self.subhalos = self.cat.loadFoF(group_fields=group_fields, subhalo_fields=subhalo_fields)

    def run(self, sid_array, gid_array):
        sid_array = np.asarray(sid_array, dtype=np.int64)
        gid_array = np.asarray(gid_array, dtype=np.int64)
        if sid_array.shape != gid_array.shape:
            raise ValueError("sid_array and gid_array must have the same shape")

        gids, groups = _group_by_gid_stable(gid_array)

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
            info = self._sub_info(sid, gid=gid)
            tidal = self._build_and_interpolate_exclusive_tidals(
                gid=gid,
                sid=sid,
                center=info.get("pos_abs", None),
                total_tidal_ctx=total_tidal_ctx,
            )

            # Target-dependent self tidal from the target subhalo's own
            # all-type mass distribution after a self-shape selection.
            tidal["tidal_self"] = self._compute_self_tidal(
                sid=sid,
                gid=gid,
                center=info.get("pos_abs", None),
            )

            shape = self._compute_shapes(sid, info)
            outs.append(dict(Sub_info=info, Shape=shape, Tidal=tidal))
        return outs

    def _build_exclusive_group_mass_distribution(self, gid, sid):
        """Collect host-group mass particles after removing the target subhalo itself."""
        gid = int(gid)
        sid = int(sid)
        dm_fixed = float(self.cfg["dm_particle_mass"])

        choice = self.cfg.get("tidal_mass_ptypes", "all")
        if choice is None or choice == "all":
            use = {"gas", "dm", "stars", "bh"}
        else:
            use = set(choice)

        pos_list = []
        mass_list = []

        if "gas" in use:
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

        if "dm" in use:
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

        if "stars" in use:
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

        if "bh" in use:
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
    # Sub_info
    # --------------------------------------------------------

    def _sub_info(self, sid, gid=None):
        sid = int(sid)
        subs = self.subhalos

        gid_cat = int(subs["GroupID"][sid]) if "GroupID" in subs else int(subs["SubhaloGrNr"][sid])
        gid = gid_cat if gid is None else int(gid)
        cen = int(subs["CenID"][sid]) if "CenID" in subs else int(self.halos["GroupFirstSub"][gid])

        pos = np.asarray(subs["SubhaloPos"][sid], dtype=np.float64)
        vel = np.asarray(subs["SubhaloVel"][sid], dtype=np.float64)
        posc = np.asarray(subs["SubhaloPos"][cen], dtype=np.float64)
        velc = np.asarray(subs["SubhaloVel"][cen], dtype=np.float64)

        out = dict(
            SubhaloID=int(sid),
            GroupID=int(gid),
            CenID=int(cen),
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
            out["_GroupID_catalog"] = int(gid_cat)

        return out

    # --------------------------------------------------------
    # Shapes
    # --------------------------------------------------------

    def _pack_shapekin(self, sk):
        nan33 = np.full((3, 3), np.nan, dtype=np.float64)

        if sk.mass is None:
            mass_tot = float(np.count_nonzero(sk.mask)) if sk.mask is not None else float("nan")
        else:
            mass_tot = float(np.sum(sk.mass[sk.mask])) if sk.mask is not None else float("nan")

        lam = np.maximum(np.abs(np.asarray(sk.evals, dtype=np.float64)), 1e-30)
        var_e = np.asarray(sk.var_evals, dtype=np.float64)
        var_cos = np.asarray(sk.var_cos, dtype=np.float64)

        axis_relerr = 0.5 * np.sqrt(np.maximum(var_e, 0.0)) / lam
        cos_err = np.sqrt(np.maximum(var_cos, 0.0))

        L, _ = sk.L()
        K_tot = sk.K()
        kappa_rot = sk.kappa()["kappa_rot"]

        # Neff is cached by ShapeKin variance stage (NaN if variance failed).
        neff = float(getattr(sk, "Neff", np.nan))

        return dict(
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

    def _run_shape_component(self, pos, mass, vel, center, *, sid, gid, label, percentile_key, r_ell=None):
        try:
            if center is None:
                logger.warning("[%s sid=%d gid=%d] missing center; returning NaN shape", label, sid, gid)
                return _nan_shape()

            center = np.asarray(center, dtype=np.float64)
            if center.shape != (3,) or (not np.all(np.isfinite(center))):
                logger.warning("[%s sid=%d gid=%d] non-finite center=%s; returning NaN shape", label, sid, gid, repr(center))
                return _nan_shape()

            pos, mass, vel, _, stats = _clean_particle_block(
                pos,
                mass=mass,
                vel=vel,
                scalar=None,
                label=label,
                sid=sid,
                gid=gid,
            )
            if stats["n_keep"] < 3:
                logger.warning("[%s sid=%d gid=%d] only %d finite particles remain after cleaning; returning NaN shape", label, sid, gid, stats["n_keep"])
                return _nan_shape()

            if r_ell is not None:
                rval = float(r_ell)
                if (not np.isfinite(rval)) or rval <= 0.0:
                    logger.warning("[%s sid=%d gid=%d] invalid r_ell=%s; returning NaN shape", label, sid, gid, str(r_ell))
                    return _nan_shape()

            sk = ShapeKin(
                particles=pos,
                masses=mass,
                velocities=vel,
                accelerations=None,
                Pos=center,
            )
            sk.run_shape(
                percentile=float(self.cfg.get(percentile_key, 100.0)),
                max_iter=int(self.cfg.get("shape_max_iter", 100)),
                tol=float(self.cfg.get("shape_tol", 0.01)),
                r_ell=r_ell,
                tensor_mode=str(self.cfg.get("shape_tensor_mode", "reduced")),
                return_dI=True,
                return_ddI=False,
            )

            nsel = int(np.count_nonzero(sk.mask)) if sk.mask is not None else 0
            if nsel < 3:
                logger.warning("[%s sid=%d gid=%d] ShapeKin selected only %d particles; output will be NaN-like", label, sid, gid, nsel)
            if not bool(getattr(sk, "converged", False)):
                logger.warning("[%s sid=%d gid=%d] shape iteration did not reach tolerance; using last iterate", label, sid, gid)
            if not np.all(np.isfinite(np.asarray(sk.I, dtype=np.float64))):
                logger.warning("[%s sid=%d gid=%d] non-finite tensor returned by ShapeKin", label, sid, gid)

            return self._pack_shapekin(sk)

        except Exception as e:
            logger.warning("[%s sid=%d gid=%d] shape computation failed; returning NaNs. Reason: %s", label, sid, gid, str(e))
            return _nan_shape()

    def _compute_shapes(self, sid, info):
        sid = int(sid)
        gid = int(info.get("GroupID", -1))
        center = info.get("pos_abs", None)
        subs = self.subhalos

        # DM
        try:
            dm = self.cat.loadSubhalos(sid=sid, ptypes=[PTYPE_DM], fields=["Coordinates", "Velocities"])
            dm_pos = np.asarray(dm["PartType1"]["Coordinates"], dtype=np.float64)
            dm_vel = np.asarray(dm["PartType1"]["Velocities"], dtype=np.float64)
            dm_mass = np.full(dm_pos.shape[0], float(self.cfg["dm_particle_mass"]), dtype=np.float64)
        except Exception as e:
            logger.warning("[dm sid=%d gid=%d] loadSubhalos failed; returning NaN shape. Reason: %s", sid, gid, str(e))
            dm_shape = _nan_shape()
        else:
            dm_shape = self._run_shape_component(dm_pos, dm_mass, dm_vel, center, sid=sid, gid=gid, label="dm", percentile_key="dm_shape_percentile", r_ell=None)

        # Stars
        try:
            st = self.cat.loadSubhalos(sid=sid, ptypes=[PTYPE_STAR], fields=["Coordinates", "Velocities", "Masses"])
            st_pos = np.asarray(st["PartType4"]["Coordinates"], dtype=np.float64)
            st_vel = np.asarray(st["PartType4"]["Velocities"], dtype=np.float64)
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
            st_shape = _nan_shape()
        else:
            st_shape = self._run_shape_component(st_pos, st_mass, st_vel, center, sid=sid, gid=gid, label="stars", percentile_key="star_shape_percentile", r_ell=r_ap)

        return dict(dm=dm_shape, stars=st_shape)

    # --------------------------------------------------------
    # Tidals
    # --------------------------------------------------------

    def _collect_subhalo_self_mass_distribution(self, sid):
        """Collect the target subhalo's own all-type mass distribution for T_self.

        The returned block is used only for ``tidal_self``.  The particles are
        collected from the target subhalo itself (gas + DM + stars + BH when
        available), then passed through an independent ShapeKin ellipsoidal
        selection.  Only the final selected particles contribute to T_self.
        """
        sid = int(sid)
        dm_fixed = float(self.cfg["dm_particle_mass"])

        choice = self.cfg.get("tidal_mass_ptypes", "all")
        if choice is None or choice == "all":
            use = {"gas", "dm", "stars", "bh"}
        else:
            use = set(choice)

        pos_list = []
        mass_list = []

        if "gas" in use:
            try:
                gas = self.cat.loadSubhalos(sid=sid, ptypes=[PTYPE_GAS], fields=["Coordinates", "Masses"])
                gp = np.asarray(gas.get("PartType0", {}).get("Coordinates", []), dtype=np.float64)
                if gp.size:
                    gm = np.asarray(gas["PartType0"]["Masses"], dtype=np.float64)
                    pos_list.append(gp)
                    mass_list.append(gm)
            except Exception as e:
                logger.warning("[sid=%d gas] loadSubhalos failed for self tidal mass distribution. Reason: %s", sid, str(e))

        if "dm" in use:
            try:
                dm = self.cat.loadSubhalos(sid=sid, ptypes=[PTYPE_DM], fields=["Coordinates"])
                dp = np.asarray(dm.get("PartType1", {}).get("Coordinates", []), dtype=np.float64)
                if dp.size:
                    dm_m = np.full(dp.shape[0], dm_fixed, dtype=np.float64)
                    pos_list.append(dp)
                    mass_list.append(dm_m)
            except Exception as e:
                logger.warning("[sid=%d dm] loadSubhalos failed for self tidal mass distribution. Reason: %s", sid, str(e))

        if "stars" in use:
            try:
                st = self.cat.loadSubhalos(sid=sid, ptypes=[PTYPE_STAR], fields=["Coordinates", "Masses"])
                sp = np.asarray(st.get("PartType4", {}).get("Coordinates", []), dtype=np.float64)
                if sp.size:
                    sm = np.asarray(st["PartType4"]["Masses"], dtype=np.float64)
                    pos_list.append(sp)
                    mass_list.append(sm)
            except Exception as e:
                logger.warning("[sid=%d stars] loadSubhalos failed for self tidal mass distribution. Reason: %s", sid, str(e))

        if "bh" in use:
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

        Sign and units are kept consistent with the other available TNG tidal
        branches by forwarding the same ``legacy_tidal_sign`` setting to
        ``compute_gravitational_potential``.  The output units are
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
                scalar=None,
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

            cfg = self.cfg
            ngrid = int(cfg.get("tidal_grid_size", 32))
            pad = float(cfg.get("tidal_padding", 0.2))
            soft = float(cfg.get("tidal_softening", 0.01))
            legacy_sign = bool(cfg.get("legacy_tidal_sign", True))

            res_self = compute_gravitational_potential(
                positions=pos_sel,
                masses=mass_sel,
                grid_size=ngrid,
                boundary_padding=pad,
                softening=soft,
                legacy_tidal_sign=legacy_sign,
            )
            interp_self = _res_to_interpolator(res_self)
            _, T_self = interp_self(center)
            return np.asarray(T_self, dtype=np.float64)

        except Exception as e:
            logger.warning("[sid=%d gid=%d] self tidal computation failed; returning NaNs. Reason: %s", sid, gid, str(e))
            return nan33.copy()

    def _build_total_tidal_context_for_gid(self, gid):
        """Build gid-shared total tidal interpolators.

        ``tidal_tot`` is still shared at fixed gid.  ``tidal_grp`` is now
        target-dependent because it is computed from the host-group mass
        distribution after removing the target subhalo itself.
        """
        gid = int(gid)
        cfg = self.cfg

        ngrid = int(cfg.get("tidal_grid_size", 32))
        pad = float(cfg.get("tidal_padding", 0.2))
        legacy_sign = bool(cfg.get("legacy_tidal_sign", True))

        # total tidal from DM potential samples
        dm_p = self.cat.loadHalos(gid=gid, ptypes=[PTYPE_DM], fields=["Coordinates", "Potential"])
        pot_pos_raw = np.asarray(dm_p["PartType1"]["Coordinates"], dtype=np.float64)
        pot_val_raw = np.asarray(dm_p["PartType1"]["Potential"], dtype=np.float64)

        pot_pos, _, _, pot_val, stats_pot = _clean_particle_block(
            pot_pos_raw,
            scalar=pot_val_raw,
            label="tidal_dm_potential",
            gid=gid,
        )
        if stats_pot["n_keep"] < 4:
            raise RuntimeError(f"gid={gid}: insufficient finite DM potential samples for total tidal ({stats_pot['n_keep']})")

        res_tot = grid_potential_and_tidal(
            pot_pos,
            pot_val,
            grid_size=ngrid,
            boundary_padding=pad,
            input_type="potential",
            legacy_tidal_sign=legacy_sign,
        )
        interp_tot = _res_to_interpolator(res_tot)

        return dict(interp_tot=interp_tot)

    def _build_and_interpolate_exclusive_tidals(self, gid, sid, center, total_tidal_ctx):
        """Build target-exclusive ``tidal_grp`` and combine it with gid-shared ``tidal_tot``."""
        gid = int(gid)
        sid = int(sid)

        if center is None:
            logger.warning("[sid=%d gid=%d] missing center for tidal interpolation; returning NaNs", sid, gid)
            return _nan_tidal()

        center = np.asarray(center, dtype=np.float64)
        if center.shape != (3,) or (not np.all(np.isfinite(center))):
            logger.warning("[sid=%d gid=%d] non-finite tidal interpolation center=%s; returning NaNs", sid, gid, repr(center))
            return _nan_tidal()

        cfg = self.cfg
        ngrid = int(cfg.get("tidal_grid_size", 32))
        pad = float(cfg.get("tidal_padding", 0.2))
        soft = float(cfg.get("tidal_softening", 0.01))
        legacy_sign = bool(cfg.get("legacy_tidal_sign", True))

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

            res_grp = compute_gravitational_potential(
                positions=pos_mass,
                masses=m_mass,
                grid_size=ngrid,
                boundary_padding=pad,
                softening=soft,
                legacy_tidal_sign=legacy_sign,
            )
            interp_grp = _res_to_interpolator(res_grp)

            _, T_grp = interp_grp(center)
            if total_tidal_ctx is None:
                return dict(
                    tidal_grp=np.asarray(T_grp, dtype=np.float64),
                    tidal_tot=np.full((3, 3), np.nan, dtype=np.float64),
                )

            _, T_tot = total_tidal_ctx["interp_tot"](center)
            return dict(
                tidal_grp=np.asarray(T_grp, dtype=np.float64),
                tidal_tot=np.asarray(T_tot, dtype=np.float64),
            )

        except Exception as e:
            logger.warning("[sid=%d gid=%d] exclusive group tidal build/interpolation failed; returning NaNs. Reason: %s", sid, gid, str(e))
            if total_tidal_ctx is None:
                return _nan_tidal()
            try:
                _, T_tot = total_tidal_ctx["interp_tot"](center)
                return dict(
                    tidal_grp=np.full((3, 3), np.nan, dtype=np.float64),
                    tidal_tot=np.asarray(T_tot, dtype=np.float64),
                )
            except Exception:
                return _nan_tidal()

    def _interp_tidals(self, center, tidal_ctx, *, gid=None, sid=None):
        if center is None:
            logger.warning("[sid=%s gid=%s] missing center for tidal interpolation; returning NaNs", str(sid), str(gid))
            return _nan_tidal()

        center = np.asarray(center, dtype=np.float64)
        if center.shape != (3,) or (not np.all(np.isfinite(center))):
            logger.warning("[sid=%s gid=%s] non-finite tidal interpolation center=%s; returning NaNs", str(sid), str(gid), repr(center))
            return _nan_tidal()

        try:
            _, T_grp = tidal_ctx["interp_grp"](center)
            _, T_tot = tidal_ctx["interp_tot"](center)
        except Exception as e:
            logger.warning("[sid=%s gid=%s] tidal interpolation failed; returning NaNs. Reason: %s", str(sid), str(gid), str(e))
            return _nan_tidal()

        return dict(
            tidal_grp=np.asarray(T_grp, dtype=np.float64),
            tidal_tot=np.asarray(T_tot, dtype=np.float64),
        )


# ============================================================
# Public API
# ============================================================


def compute_many(
    base_path,
    snap,
    sid_array,
    gid_array,
    cfg,
    group_fields=None,
    subhalo_fields=None,
    tng_catalog_kwargs=None,
):
    """Compute a batch of TNG subhalos.

    ``tng_catalog_kwargs`` is forwarded to :class:`TNGCatLoader.TNGCatalog`.
    If API cutouts/groupcat subsets are downloaded temporarily, they are deleted
    before this function returns by default. Set ``delete_cache=False`` or pass
    ``--keep-cache`` in ``run_tng.py`` to keep them.
    """
    state = GlobalTNGGrouped(
        base_path,
        snap,
        cfg,
        group_fields=group_fields,
        subhalo_fields=subhalo_fields,
        tng_catalog_kwargs=tng_catalog_kwargs,
    )
    try:
        return state.run(sid_array, gid_array)
    finally:
        cat = getattr(state, "cat", None)
        if cat is not None and hasattr(cat, "cleanup"):
            cat.cleanup()


def compute_one_subhalo(base_path, snap, sid, gid=None, cfg=None, tng_catalog_kwargs=None):
    if cfg is None:
        raise ValueError("cfg must be provided")
    state = GlobalTNGGrouped(base_path, snap, cfg, tng_catalog_kwargs=tng_catalog_kwargs)
    try:
        if gid is None:
            if "GroupID" in state.subhalos:
                gid = int(state.subhalos["GroupID"][int(sid)])
            else:
                gid = int(state.subhalos["SubhaloGrNr"][int(sid)])
        return state.run([int(sid)], [int(gid)])[0]
    finally:
        cat = getattr(state, "cat", None)
        if cat is not None and hasattr(cat, "cleanup"):
            cat.cleanup()
