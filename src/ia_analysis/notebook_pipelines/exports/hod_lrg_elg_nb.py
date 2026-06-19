"""Exported code from notebooks/raw_20260618/HOD_LRG_ELG.ipynb.

This file is generated for project management and refactoring. Review before running end to end.
"""


# %% [markdown] cell 1
# # LRG / ELG HOD from raw simulation data This notebook does **not** read the pre-built galaxy catalog / `MA.pkl`. It reads the FoF/Subhalo data directly from the simulation via `CSCatalog.loadFoF`, then measures the HOD curves for LRG / ELG samples. ## LRG / ELG selection used in this version The default LRG / ELG criteria are changed to the redshift-dependent cuts in the provided table: | $z$ | ELG: $\log_{10}({\rm sSFR}\ [{\rm yr}^{-1}])$ | LRG: $M_\star[10^{10}M_\odot]$ | |---:|---:|---:| | 1

# %% code cell 2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.lines import Line2D
import arts
try:
    from scipy.interpolate import PchipInterpolator
    HAS_SCIPY_PCHIP = True
except Exception:
    PchipInterpolator = None
    HAS_SCIPY_PCHIP = False

from catalog_loader import CSCatalog

sns.set(style='ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.fontset'] = 'cm'

# ============================================================
# Paths and simulation setup
# ============================================================
BASE_ROOT = "/cosma8/data/dp203/bl267/Data/ClusterSims"
SIM_TEMPLATE = "L302_N1136_{flag}"

flags = ['F40', 'F45', 'F50', 'F55', 'F60', 'GR']
snaps = [1, 3, 6, 8, 10, 12, 15, 18, 21]
zmap = {s: arts.ZMAP_ALL[s] for s in snaps}
# {
#     3: 1.48,
#     6: 0.97,
#     8: 0.78,
#     10: 0.64,
#     12: 0.51,
#     15: 0.33,
#     18: 0.16,
#     21: 0.00,
# }

# ============================================================
# Raw simulation field names
# ============================================================
HALO_MASS_KEY = "Group_M_Crit200"
HALO_RADIUS_KEY = "Group_R_Crit200"
GROUP_POS_KEY = "GroupPos"
GROUP_FIRST_SUB_KEY = "GroupFirstSub"
GROUP_NSUBS_KEY = "GroupNsubs"

SUBHALO_GROUP_KEY = "SubhaloGrNr"
SUBHALO_POS_KEY = "SubhaloPos"
MSTAR_KEY_PREFERRED = "SubhaloMassInRadType"
MSTAR_KEY_FALLBACK = "SubhaloMassType"
SFR_KEY = "SubhaloSFR"
STELLAR_COMPONENT_INDEX = 4

# ============================================================
# Unit conversions
# ============================================================
# TNG / AREPO group-catalog convention:
#   Group_M_Crit200          : 1e10 Msun/h
#   SubhaloMassInRadType[:,4]: 1e10 Msun/h
#   SubhaloSFR               : Msun/yr
#   GroupPos/SubhaloPos      : ckpc/h
#   Group_R_Crit200          : ckpc/h
#
# Therefore:
#   M200c plotted in Msun/h = raw * 1e10
#   Mstar cuts below are also in Msun/h = raw * 1e10
#   sSFR should use physical Msun, so Mstar[Msun] = Mstar[Msun/h] / h
#   R200c selection only compares quantities in the same position unit.
HUBBLE_h = 0.6774
MASS_UNIT_TO_MSUNH = 1e10
HALO_MASS_SCALE = MASS_UNIT_TO_MSUNH
STELLAR_MASS_SCALE = MASS_UNIT_TO_MSUNH

# Box size in the same units as GroupPos/SubhaloPos.
# For L302 simulations with TNG-like ckpc/h positions: 302 Mpc/h = 302000 ckpc/h.
# If your positions are already in cMpc/h, set this to 302.0.
BOXSIZE_POSITION_UNIT = 302000.0
USE_PERIODIC_DISTANCE = True

# ============================================================
# HOD binning and plotting setup
# ============================================================
M200C_RANGE = (1e11, 1e15)
NBINS = 20
MASS_BINS = np.logspace(np.log10(M200C_RANGE[0]), np.log10(M200C_RANGE[1]), NBINS + 1)
MIN_HALOS_PER_BIN = 5
STAT = "mean"
SHOW_YLOG = True
SHOW_BAND = False

# Smooth HOD curves in log10(M200c). This only affects plotting.
# PCHIP is shape-preserving and does not create spline/Gaussian over-smoothing wiggles.
SMOOTH_HOD_CURVES = True
SMOOTH_NGRID = 300
SHOW_BINNED_POINTS = True   # keep points visible while checking the result

# Regions to plot.
# "fof" reproduces the original FoF-member HOD.
# "r200c" applies d(SubhaloPos, GroupPos) <= Group_R_Crit200.
HOD_REGIONS = ["fof", "r200c"]
region_labels = {
    "fof": "FoF members",
    "r200c": r"within $R_{200c}$",
}
region_alpha = {
    "fof": 0.35,
    "r200c": 0.95,
}
region_lw = {
    "fof": 1.6,
    "r200c": 2.5,
}

COMPONENTS = ["Ncen", "Nsat", "Ntot"]
clist = ['#c02c38','#c2c116','#3c9566','#1177b0','#ff7c38','#be8936',
         '#e03e36','#b80d57','#700961','#11659a','#abcdef','#fedcba']
component_colors = {
    "Ncen": clist[0],
    "Nsat": clist[3],
    "Ntot": clist[2],
}
component_labels = {
    "Ncen": r"$\langle N_{\rm cen}\rangle$",
    "Nsat": r"$\langle N_{\rm sat}\rangle$",
    "Ntot": r"$\langle N_{\rm tot}\rangle$",
}
component_linestyles = {
    "Ncen": "--",
    "Nsat": "-",
    "Ntot": "-.",
}

# ============================================================
# LRG / ELG criteria from the provided redshift-dependent table
# ============================================================
# The table gives:
#   ELG: log10(sSFR / yr^-1)
#   LRG: Mstar in units of 1e10 physical Msun
#
# For snapshots whose redshifts are not exactly in the table, the default
# behaviour is linear interpolation in z. Values outside the table range are
# clamped to the nearest tabulated edge.
VERBOSE = False
DESI_CUT_INTERP_MODE = "linear"  # options: "linear", "nearest"

SELECTION_TABLE_Z = np.array([1.16, 0.97, 0.73, 0.51, 0.27], dtype=float)
ELG_LOGSSFR_CUT_TABLE = np.array([-9.3, -9.4, -9.5, -9.6, -9.9], dtype=float)
LRG_MSTAR_CUT_1E10_MSUN_TABLE = np.array([7.5, 7.4, 8.2, 8.2, 8.6], dtype=float)

if VERBOSE:
    print("Direct-simulation mode: no MA.pkl / external galaxy table is read.")
    print("Mass units follow TNG-like catalog convention: raw mass * 1e10 = Msun/h.")
    print("sSFR uses Mstar[Msun] = Mstar[Msun/h] / h.")
    print("R200c cut uses GroupPos, SubhaloPos, and Group_R_Crit200 in their native common position unit.")
    print("LRG/ELG selection follows the redshift-dependent table supplied in the notebook.")

# %% code cell 3
# ============================================================
# Direct raw-simulation loading helpers
# ============================================================

_sim_cache = {}


def _try_load_fof(cat, group_fields, subhalo_fields):
    """Small wrapper so field fallback logic stays readable."""
    return cat.loadFoF(group_fields=group_fields, subhalo_fields=subhalo_fields)


def _choose_first_sub_indexing(first_raw, gid, nhalo, nsub):
    """
    Determine whether GroupFirstSub is 0-based or 1-based.
    TNG should be 0-based, but this check prevents silent central/satellite mistakes.
    """
    first_raw = np.asarray(first_raw, dtype=np.int64)
    gid = np.asarray(gid, dtype=np.int64)
    gidx = np.arange(min(nhalo, len(first_raw)), dtype=np.int64)

    m0 = (first_raw >= 0) & (first_raw < nsub) & (gidx < nhalo)
    frac0 = np.nan
    if np.any(m0):
        frac0 = np.mean(gid[first_raw[m0]] == gidx[m0])

    first1 = first_raw - 1
    m1 = (first_raw > 0) & (first1 >= 0) & (first1 < nsub) & (gidx < nhalo)
    frac1 = np.nan
    if np.any(m1):
        frac1 = np.mean(gid[first1[m1]] == gidx[m1])

    if np.isfinite(frac1) and (not np.isfinite(frac0) or frac1 > frac0 + 0.1):
        return first1, "1-based converted to 0-based", frac1
    return first_raw, "0-based", frac0


def _periodic_delta(delta, boxsize):
    """Minimum-image displacement in the same unit as the input positions."""
    if boxsize is None or not np.isfinite(boxsize) or boxsize <= 0:
        return delta
    return delta - boxsize * np.rint(delta / boxsize)


def _compute_subhalo_radius_to_host(group_pos, sub_pos, gid, nhalo):
    """Return 3D subhalo-host distance in the native position unit."""
    nsub = len(gid)
    r_host = np.full(nsub, np.nan, dtype=float)

    group_pos = np.asarray(group_pos, dtype=float)
    sub_pos = np.asarray(sub_pos, dtype=float)
    gid = np.asarray(gid, dtype=np.int64)

    good = (
        (gid >= 0) & (gid < nhalo)
        & np.all(np.isfinite(sub_pos), axis=1)
        & np.all(np.isfinite(group_pos[gid.clip(0, max(nhalo - 1, 0))]), axis=1)
    )
    if not np.any(good):
        return r_host

    delta = sub_pos[good] - group_pos[gid[good]]
    if USE_PERIODIC_DISTANCE:
        delta = _periodic_delta(delta, BOXSIZE_POSITION_UNIT)
    r_host[good] = np.linalg.norm(delta, axis=1)
    return r_host


def load_sim_direct(flag, snap):
    """
    Read FoF halo and subhalo data directly from the raw simulation.

    Unit convention assumed here is the same as TNG/AREPO group catalogs:
    masses are stored in 1e10 Msun/h, SFR is stored in Msun/yr,
    and GroupPos/SubhaloPos/Group_R_Crit200 share the same native position unit.
    """
    key = (flag, int(snap))
    if key in _sim_cache:
        return _sim_cache[key]

    base_path = Path(BASE_ROOT) / SIM_TEMPLATE.format(flag=flag)
    cat = CSCatalog(base_path, int(snap))

    group_field_attempts = [
        [HALO_MASS_KEY, HALO_RADIUS_KEY, GROUP_POS_KEY, GROUP_FIRST_SUB_KEY, GROUP_NSUBS_KEY],
        [HALO_MASS_KEY, HALO_RADIUS_KEY, GROUP_POS_KEY, GROUP_FIRST_SUB_KEY],
        [HALO_MASS_KEY, HALO_RADIUS_KEY, GROUP_POS_KEY],
        [HALO_MASS_KEY, GROUP_FIRST_SUB_KEY, GROUP_NSUBS_KEY],
        [HALO_MASS_KEY, GROUP_FIRST_SUB_KEY],
        [HALO_MASS_KEY],
    ]

    # Prefer loading SubhaloGrNr explicitly. If unavailable, a fallback from
    # GroupFirstSub/GroupNsubs is used, but the summary will report this.
    subhalo_field_attempts = [
        [SUBHALO_GROUP_KEY, SUBHALO_POS_KEY, MSTAR_KEY_PREFERRED, SFR_KEY],
        [SUBHALO_GROUP_KEY, SUBHALO_POS_KEY, MSTAR_KEY_FALLBACK, SFR_KEY],
        [SUBHALO_GROUP_KEY, MSTAR_KEY_PREFERRED, SFR_KEY],
        [SUBHALO_GROUP_KEY, MSTAR_KEY_FALLBACK, SFR_KEY],
        [SUBHALO_POS_KEY, MSTAR_KEY_PREFERRED, SFR_KEY],
        [SUBHALO_POS_KEY, MSTAR_KEY_FALLBACK, SFR_KEY],
        [MSTAR_KEY_PREFERRED, SFR_KEY],
        [MSTAR_KEY_FALLBACK, SFR_KEY],
    ]

    last_error = None
    halos = subs = None
    used_group_fields = used_subhalo_fields = None

    for gf in group_field_attempts:
        for sf in subhalo_field_attempts:
            try:
                halos, subs = _try_load_fof(cat, group_fields=gf, subhalo_fields=sf)
                used_group_fields = gf
                used_subhalo_fields = sf
                break
            except Exception as err:
                last_error = err
        if halos is not None:
            break

    if halos is None or subs is None:
        raise RuntimeError(
            f"Could not load raw FoF/Subhalo fields for flag={flag}, snap={snap}. "
            f"Last error was: {last_error}"
        )

    if HALO_MASS_KEY not in halos:
        raise KeyError(f"Required halo field {HALO_MASS_KEY!r} is missing.")

    # M200c in Msun/h for HOD x-axis.
    m200c = np.asarray(halos[HALO_MASS_KEY], dtype=float) * HALO_MASS_SCALE
    nhalo = len(m200c)

    # Stellar mass in Msun/h for mass cuts; physical Msun for sSFR.
    stellar_mass_key = None
    if MSTAR_KEY_PREFERRED in subs:
        stellar_mass_key = MSTAR_KEY_PREFERRED
    elif MSTAR_KEY_FALLBACK in subs:
        stellar_mass_key = MSTAR_KEY_FALLBACK
    else:
        raise KeyError(
            f"Neither {MSTAR_KEY_PREFERRED!r} nor {MSTAR_KEY_FALLBACK!r} was loaded."
        )

    mtype = np.asarray(subs[stellar_mass_key], dtype=float)
    if mtype.ndim != 2 or mtype.shape[1] <= STELLAR_COMPONENT_INDEX:
        raise ValueError(
            f"{stellar_mass_key} should be a 2D mass-type array with stellar column index "
            f"{STELLAR_COMPONENT_INDEX}; got shape={mtype.shape}."
        )
    mstar_msunh = mtype[:, STELLAR_COMPONENT_INDEX] * STELLAR_MASS_SCALE
    mstar_msun = mstar_msunh / HUBBLE_h

    if SFR_KEY not in subs:
        raise KeyError(f"Required subhalo field {SFR_KEY!r} is missing.")
    sfr = np.asarray(subs[SFR_KEY], dtype=float)

    nsub = len(mstar_msunh)
    sid = np.arange(nsub, dtype=np.int64)

    # Subhalo -> host group ID.
    if SUBHALO_GROUP_KEY in subs:
        gid = np.asarray(subs[SUBHALO_GROUP_KEY], dtype=np.int64)
        gid_source = SUBHALO_GROUP_KEY
    elif GROUP_FIRST_SUB_KEY in halos and GROUP_NSUBS_KEY in halos:
        gid = np.full(nsub, -1, dtype=np.int64)
        first_raw = np.asarray(halos[GROUP_FIRST_SUB_KEY], dtype=np.int64)
        nsubs = np.asarray(halos[GROUP_NSUBS_KEY], dtype=np.int64)
        # Assume 0-based first-sub indexing for fallback assignment.
        for g, (s0, ns) in enumerate(zip(first_raw, nsubs)):
            if s0 >= 0 and ns > 0 and s0 + ns <= nsub:
                gid[s0:s0 + ns] = g
        gid_source = f"fallback from {GROUP_FIRST_SUB_KEY}+{GROUP_NSUBS_KEY}"
    else:
        raise KeyError(
            f"Cannot determine host group IDs: need {SUBHALO_GROUP_KEY!r} "
            f"or both {GROUP_FIRST_SUB_KEY!r} and {GROUP_NSUBS_KEY!r}."
        )

    # Central subhalo ID for each FoF halo.
    central_sid = np.full(nhalo, -1, dtype=np.int64)
    first_sub_indexing = "unknown"
    first_sub_match_fraction = np.nan
    if GROUP_FIRST_SUB_KEY in halos:
        first_raw = np.asarray(halos[GROUP_FIRST_SUB_KEY], dtype=np.int64)
        first, first_sub_indexing, first_sub_match_fraction = _choose_first_sub_indexing(
            first_raw, gid, nhalo, nsub
        )
        central_sid[:len(first)] = first
    else:
        # Fallback: use the first subhalo appearing in each group.
        valid_gid = (gid >= 0) & (gid < nhalo)
        order = np.argsort(gid[valid_gid], kind="stable")
        valid_sid = sid[valid_gid]
        valid_gid_sorted = gid[valid_gid][order]
        valid_sid_sorted = valid_sid[order]
        unique_gid, first_idx = np.unique(valid_gid_sorted, return_index=True)
        central_sid[unique_gid] = valid_sid_sorted[first_idx]
        first_sub_indexing = "first subhalo by sorted gid fallback"

    valid_sub = (
        np.isfinite(mstar_msunh)
        & np.isfinite(mstar_msun)
        & np.isfinite(sfr)
        & (mstar_msunh > 0)
        & (gid >= 0)
        & (gid < nhalo)
    )
    is_cen = valid_sub & (sid == central_sid[gid])
    is_sat = valid_sub & ~is_cen

    host_m200c = np.full(nsub, np.nan, dtype=float)
    good_gid = (gid >= 0) & (gid < nhalo)
    host_m200c[good_gid] = m200c[gid[good_gid]]

    # R200c geometry. The radius cut is performed in native position units.
    has_r200c_geometry = (
        (HALO_RADIUS_KEY in halos)
        and (GROUP_POS_KEY in halos)
        and (SUBHALO_POS_KEY in subs)
    )

    host_r200c = np.full(nsub, np.nan, dtype=float)
    r_to_host = np.full(nsub, np.nan, dtype=float)
    inside_r200c = np.zeros(nsub, dtype=bool)

    if has_r200c_geometry:
        r200c_native = np.asarray(halos[HALO_RADIUS_KEY], dtype=float)
        group_pos = np.asarray(halos[GROUP_POS_KEY], dtype=float)
        sub_pos = np.asarray(subs[SUBHALO_POS_KEY], dtype=float)

        if group_pos.ndim != 2 or group_pos.shape[1] != 3:
            raise ValueError(f"{GROUP_POS_KEY} should have shape (Nhalo, 3); got {group_pos.shape}.")
        if sub_pos.ndim != 2 or sub_pos.shape[1] != 3:
            raise ValueError(f"{SUBHALO_POS_KEY} should have shape (Nsub, 3); got {sub_pos.shape}.")

        host_r200c[good_gid] = r200c_native[gid[good_gid]]
        r_to_host = _compute_subhalo_radius_to_host(group_pos, sub_pos, gid, nhalo)
        inside_r200c = (
            valid_sub
            & np.isfinite(r_to_host)
            & np.isfinite(host_r200c)
            & (host_r200c > 0)
            & (r_to_host <= host_r200c)
        )
        # Centrals should belong to the R200c-selected sample by construction.
        inside_r200c |= is_cen
    else:
        # Keep notebook runnable if geometry fields are unavailable.
        # The summary explicitly reports has_r200c_geometry=False.
        inside_r200c = valid_sub.copy()

    out = {
        "M200c": m200c,
        "Ng": nhalo,
        "gid": gid,
        "sid": sid,
        "central_sid": central_sid,
        "is_cen": is_cen,
        "is_sat": is_sat,
        "inside_r200c": inside_r200c,
        "r_to_host": r_to_host,
        "host_r200c": host_r200c,
        "has_r200c_geometry": bool(has_r200c_geometry),
        "mstar_msunh": mstar_msunh,
        "mstar_msun": mstar_msun,
        "host_m200c": host_m200c,
        "sfr": sfr,
        "stellar_mass_key": stellar_mass_key,
        "used_group_fields": used_group_fields,
        "used_subhalo_fields": used_subhalo_fields,
        "gid_source": gid_source,
        "first_sub_indexing": first_sub_indexing,
        "first_sub_match_fraction": first_sub_match_fraction,
        "snap": int(snap),
        "z": zmap.get(int(snap), np.nan),
    }
    _sim_cache[key] = out
    return out


def safe_ssfr(sfr, mstar_msun):
    """Return SFR / Mstar in yr^-1; invalid or zero-mass objects become NaN."""
    sfr = np.asarray(sfr, dtype=float)
    mstar_msun = np.asarray(mstar_msun, dtype=float)
    ssfr = np.full_like(sfr, np.nan, dtype=float)
    m = np.isfinite(sfr) & np.isfinite(mstar_msun) & (mstar_msun > 0)
    ssfr[m] = sfr[m] / mstar_msun[m]
    return ssfr


def _redshift_dependent_cut(z, values, mode=DESI_CUT_INTERP_MODE):
    """
    Return the redshift-dependent cut value from the supplied table.

    Parameters
    ----------
    z : float
        Snapshot redshift.
    values : array-like
        Values tabulated at SELECTION_TABLE_Z.
    mode : {"linear", "nearest"}
        "linear" interpolates in z and clamps to edge values outside the table.
        "nearest" uses the nearest tabulated redshift.
    """
    z = float(z)
    ztab = np.asarray(SELECTION_TABLE_Z, dtype=float)
    vals = np.asarray(values, dtype=float)

    if mode == "nearest":
        return float(vals[np.nanargmin(np.abs(ztab - z))])

    if mode != "linear":
        raise ValueError("DESI_CUT_INTERP_MODE must be 'linear' or 'nearest'.")

    # np.interp expects ascending x.
    order = np.argsort(ztab)
    return float(np.interp(z, ztab[order], vals[order], left=vals[order][0], right=vals[order][-1]))


def elg_logssfr_cut(z):
    """ELG selection threshold log10(sSFR / yr^-1) at redshift z."""
    return _redshift_dependent_cut(z, ELG_LOGSSFR_CUT_TABLE)


def lrg_mstar_cut_msun(z):
    """LRG selection threshold Mstar in physical Msun at redshift z."""
    return 1e10 * _redshift_dependent_cut(z, LRG_MSTAR_CUT_1E10_MSUN_TABLE)


def select_lrg(mstar_msunh, sfr, mstar_msun, z=None, snap=None):
    """
    Select LRG-like galaxies using the redshift-dependent stellar-mass cut
    from the provided table.

    The table threshold is in physical Msun, so this function uses
    mstar_msun rather than mstar_msunh.
    """
    if z is None:
        z = zmap.get(int(snap), np.nan) if snap is not None else np.nan

    mcut = lrg_mstar_cut_msun(z)

    return (
        np.isfinite(mstar_msun)
        & (mstar_msun >= mcut)
    )


def select_elg(mstar_msunh, sfr, mstar_msun, z=None, snap=None):
    """
    Select ELG-like galaxies using the redshift-dependent sSFR cut
    from the provided table.
    """
    if z is None:
        z = zmap.get(int(snap), np.nan) if snap is not None else np.nan

    ssfr = safe_ssfr(sfr, mstar_msun)
    log_ssfr = np.full_like(ssfr, np.nan, dtype=float)
    m = np.isfinite(ssfr) & (ssfr > 0)
    log_ssfr[m] = np.log10(ssfr[m])

    return (
        np.isfinite(log_ssfr)
        & (log_ssfr >= elg_logssfr_cut(z))
    )


def make_analysis_specs():
    return [
        {
            "name": "LRG",
            "title": r"LRG: redshift-dependent $M_\star$ cut",
            "selector": select_lrg,
        },
        {
            "name": "ELG",
            "title": r"ELG: redshift-dependent sSFR cut",
            "selector": select_elg,
        },
    ]

# %% code cell 4
# ============================================================
# HOD measurement helpers
# ============================================================

def compute_occupation_counts_from_sim(sim, selector, region="fof"):
    """
    Compute Ncen, Nsat and Ntot per FoF halo for a given galaxy selector.

    Parameters
    ----------
    region : {"fof", "r200c"}
        "fof"   : use all selected subhalos assigned to the FoF host.
        "r200c" : use only selected subhalos with r_to_host <= Group_R_Crit200.
    """
    if region not in ("fof", "r200c"):
        raise ValueError("region must be either 'fof' or 'r200c'.")

    m200c = np.asarray(sim["M200c"], dtype=float)
    nhalo = int(sim["Ng"])
    gid = np.asarray(sim["gid"], dtype=np.int64)
    mstar_msunh = np.asarray(sim["mstar_msunh"], dtype=float)
    mstar_msun = np.asarray(sim["mstar_msun"], dtype=float)
    sfr = np.asarray(sim["sfr"], dtype=float)
    is_cen = np.asarray(sim["is_cen"], dtype=bool)
    is_sat = np.asarray(sim["is_sat"], dtype=bool)

    sample = np.asarray(selector(mstar_msunh, sfr, mstar_msun, z=sim.get("z"), snap=sim.get("snap")), dtype=bool)
    valid = sample & (gid >= 0) & (gid < nhalo)

    if region == "r200c":
        valid = valid & np.asarray(sim["inside_r200c"], dtype=bool)

    ncen = np.zeros(nhalo, dtype=np.int64)
    nsat = np.zeros(nhalo, dtype=np.int64)

    m_cen = valid & is_cen
    if np.any(m_cen):
        ncen += np.bincount(gid[m_cen], minlength=nhalo).astype(np.int64)

    m_sat = valid & is_sat
    if np.any(m_sat):
        nsat += np.bincount(gid[m_sat], minlength=nhalo).astype(np.int64)

    return {
        "M200c": m200c,
        "Ncen": ncen,
        "Nsat": nsat,
        "Ntot": ncen + nsat,
        "Ng": nhalo,
        "region": region,
        "Ngal_selected": int(valid.sum()),
        "Ncen_selected": int(m_cen.sum()),
        "Nsat_selected": int(m_sat.sum()),
        "selected_mask": valid,
    }


def bin_occupation_curve(occ, component="Nsat", *, mass_bins=MASS_BINS, stat="mean", min_halos_per_bin=3):
    x = np.asarray(occ["M200c"], dtype=float)
    y = np.asarray(occ[component], dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[valid]
    y = y[valid]

    ibin = np.digitize(x, mass_bins) - 1
    nb = len(mass_bins) - 1

    xc, yc, y16, y84, nh = [], [], [], [], []
    xl, xr = [], []

    for i in range(nb):
        m = (ibin == i)
        if m.sum() < min_halos_per_bin:
            continue

        vals = y[m]
        left, right = mass_bins[i], mass_bins[i + 1]

        xl.append(left)
        xr.append(right)
        xc.append(np.sqrt(left * right))
        nh.append(m.sum())
        y16.append(np.percentile(vals, 16))
        y84.append(np.percentile(vals, 84))

        if stat == "mean":
            yc.append(np.mean(vals))
        elif stat == "median":
            yc.append(np.median(vals))
        else:
            raise ValueError("stat must be 'mean' or 'median'.")

    return {
        "x": np.asarray(xc),
        "y": np.asarray(yc),
        "y16": np.asarray(y16),
        "y84": np.asarray(y84),
        "n_halo": np.asarray(nh),
        "x_left": np.asarray(xl),
        "x_right": np.asarray(xr),
        "component": component,
        "region": occ.get("region", "unknown"),
        "stat": stat,
        "mass_bins": np.asarray(mass_bins),
    }


def _finite_log10_range(arr):
    arr = np.asarray(arr, dtype=float)
    m = np.isfinite(arr) & (arr > 0)
    if not np.any(m):
        return (np.nan, np.nan)
    return (float(np.nanmin(np.log10(arr[m]))), float(np.nanmax(np.log10(arr[m]))))


def _selected_stellar_to_host_summary(sim, selected_mask):
    selected_mask = np.asarray(selected_mask, dtype=bool)
    ratio = np.asarray(sim["mstar_msunh"], dtype=float) / np.asarray(sim["host_m200c"], dtype=float)
    m = selected_mask & np.isfinite(ratio) & (ratio > 0)
    if not np.any(m):
        return {"median_Mstar_over_M200c": np.nan, "frac_Mstar_gt_0p2_M200c": np.nan}
    return {
        "median_Mstar_over_M200c": float(np.nanmedian(ratio[m])),
        "frac_Mstar_gt_0p2_M200c": float(np.mean(ratio[m] > 0.2)),
    }


def _selected_r_over_r200c_summary(sim, selected_mask):
    selected_mask = np.asarray(selected_mask, dtype=bool)
    r = np.asarray(sim["r_to_host"], dtype=float)
    r200 = np.asarray(sim["host_r200c"], dtype=float)
    ratio = r / r200
    m = selected_mask & np.isfinite(ratio) & (ratio >= 0)
    if not np.any(m):
        return {"median_r_over_r200c": np.nan, "frac_within_r200c": np.nan}
    return {
        "median_r_over_r200c": float(np.nanmedian(ratio[m])),
        "frac_within_r200c": float(np.mean(ratio[m] <= 1.0)),
    }


def build_curves_for_snapshot(flags, snap, analysis_specs):
    curves = {}
    summary_rows = []

    for spec in analysis_specs:
        sname = spec["name"]
        curves[sname] = {}
        selector = spec["selector"]

        for flag in flags:
            sim = load_sim_direct(flag, snap)
            curves[sname][flag] = {}
            occ_by_region = {}

            for region in HOD_REGIONS:
                occ = compute_occupation_counts_from_sim(sim, selector, region=region)
                occ_by_region[region] = occ
                curves[sname][flag][region] = {}

                for component in COMPONENTS:
                    curves[sname][flag][region][component] = bin_occupation_curve(
                        occ,
                        component=component,
                        mass_bins=MASS_BINS,
                        stat=STAT,
                        min_halos_per_bin=MIN_HALOS_PER_BIN,
                    )

            # Summaries use the original FoF-selected sample plus R200c counts.
            occ_fof = occ_by_region["fof"]
            occ_r200c = occ_by_region["r200c"]
            log_m200c_min, log_m200c_max = _finite_log10_range(sim["M200c"])
            log_mstar_min, log_mstar_max = _finite_log10_range(sim["mstar_msunh"])
            ratio_summary = _selected_stellar_to_host_summary(sim, occ_fof["selected_mask"])
            r_summary = _selected_r_over_r200c_summary(sim, occ_fof["selected_mask"])

            summary_rows.append({
                "sample": sname,
                "flag": flag,
                "snap": int(snap),
                "z": zmap.get(int(snap), np.nan),
                "Ng_selected_fof": occ_fof["Ngal_selected"],
                "Ncen_selected_fof": occ_fof["Ncen_selected"],
                "Nsat_selected_fof": occ_fof["Nsat_selected"],
                "Ng_selected_R200c": occ_r200c["Ngal_selected"],
                "Ncen_selected_R200c": occ_r200c["Ncen_selected"],
                "Nsat_selected_R200c": occ_r200c["Nsat_selected"],
                "has_r200c_geometry": sim["has_r200c_geometry"],
                "stellar_mass_key": sim["stellar_mass_key"],
                "gid_source": sim["gid_source"],
                "first_sub_indexing": sim["first_sub_indexing"],
                "first_sub_match_fraction": sim["first_sub_match_fraction"],
                "log10_M200c_range": (round(log_m200c_min, 2), round(log_m200c_max, 2)),
                "log10_Mstar_range": (round(log_mstar_min, 2), round(log_mstar_max, 2)),
                "median_Mstar_over_M200c": ratio_summary["median_Mstar_over_M200c"],
                "frac_Mstar_gt_0p2_M200c": ratio_summary["frac_Mstar_gt_0p2_M200c"],
                "median_r_over_r200c": r_summary["median_r_over_r200c"],
                "frac_selected_within_R200c": r_summary["frac_within_r200c"],
            })

    return curves, summary_rows

# %% code cell 5
# ============================================================
# Plotting helpers
# ============================================================

def smooth_curve_logx(x, y, *, ngrid=300, ylog=False):
    """
    Return a smooth HOD curve on a dense log10(M) grid.

    Uses PCHIP when scipy is available. PCHIP is shape-preserving and is safer
    for HOD curves than Gaussian smoothing through sparse/noisy bins.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    good = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if ylog:
        good &= (y > 0)

    x = x[good]
    y = y[good]
    if len(x) < 3:
        return x, y

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    lx = np.log10(x)

    # Remove repeated x values if any.
    unique_lx, inverse = np.unique(lx, return_inverse=True)
    if len(unique_lx) < len(lx):
        yy = np.zeros(len(unique_lx), dtype=float)
        nn = np.zeros(len(unique_lx), dtype=float)
        for k, val in zip(inverse, y):
            yy[k] += val
            nn[k] += 1
        lx = unique_lx
        y = yy / nn

    if len(lx) < 3:
        return 10**lx, y

    work_y = np.log10(y) if ylog else y.copy()
    grid_lx = np.linspace(lx.min(), lx.max(), int(ngrid))

    if HAS_SCIPY_PCHIP:
        interpolator = PchipInterpolator(lx, work_y, extrapolate=False)
        smooth_y = interpolator(grid_lx)
    else:
        smooth_y = np.interp(grid_lx, lx, work_y)

    y_smooth = 10**smooth_y if ylog else smooth_y
    if not ylog:
        y_smooth = np.clip(y_smooth, 0, None)

    return 10**grid_lx, y_smooth


def _plot_one_curve(ax, curve, component, region, *, ylog, smooth, smooth_ngrid, show_binned_points):
    x = curve["x"]
    y = curve["y"]
    y16 = curve["y16"]
    y84 = curve["y84"]

    if len(x) == 0:
        return

    color = component_colors[component]
    alpha = region_alpha[region]
    lw = region_lw[region]

    if show_binned_points:
        y_points = np.where(y > 0, y, np.nan) if ylog else y
        ax.plot(
            x, y_points,
            marker="o",
            markersize=3.0 if region == "fof" else 4.0,
            linestyle="None",
            color=color,
            alpha=0.22 if region == "fof" else 0.45,
        )

    if smooth:
        xplot, yplot = smooth_curve_logx(
            x,
            y,
            ngrid=smooth_ngrid,
            ylog=ylog,
        )
    else:
        xplot = x
        yplot = np.where(y > 0, y, np.nan) if ylog else y

    if len(xplot) == 0:
        return

    ax.plot(
        xplot, yplot,
        color=color,
        linestyle=component_linestyles[component],
        lw=lw,
        alpha=alpha,
    )

    if SHOW_BAND:
        if ylog:
            good = (y16 > 0) & (y84 > 0)
            ax.fill_between(
                x[good], y16[good], y84[good],
                color=color,
                alpha=0.04 if region == "fof" else 0.08,
            )
        else:
            ax.fill_between(
                x, y16, y84,
                color=color,
                alpha=0.04 if region == "fof" else 0.08,
            )


def plot_hod_grid(
    curves,
    analysis_specs,
    *,
    flags=flags,
    snap=None,
    zmap=None,
    figsize=(22, 7),
    xlog=True,
    ylog=False,
    smooth=SMOOTH_HOD_CURVES,
    smooth_ngrid=SMOOTH_NGRID,
    show_binned_points=SHOW_BINNED_POINTS,
):
    nrows = len(analysis_specs)
    ncols = len(flags)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes = np.asarray(axes)
    # Robust axis shape handling for both 2x6 CS plots and single-column TNG plots.
    if nrows == 1 and ncols == 1:
        axes = axes.reshape(1, 1)
    elif nrows == 1:
        axes = axes.reshape(1, ncols)
    elif ncols == 1:
        axes = axes.reshape(nrows, 1)
    else:
        axes = axes.reshape(nrows, ncols)

    for i, spec in enumerate(analysis_specs):
        sname = spec["name"]

        for j, flag in enumerate(flags):
            ax = axes[i, j]

            # Plot original FoF-member curves first, then R200c curves on top.
            for region in HOD_REGIONS:
                for component in COMPONENTS:
                    curve = curves[sname][flag][region][component]
                    _plot_one_curve(
                        ax,
                        curve,
                        component,
                        region,
                        ylog=ylog,
                        smooth=smooth,
                        smooth_ngrid=smooth_ngrid,
                        show_binned_points=show_binned_points,
                    )

            if i == 0:
                ax.set_title(flag, fontsize=13)

            if j == 0:
                ax.set_ylabel(
                    spec["title"] + "\n" + (r"$\langle N\rangle$" if STAT == "mean" else r"${\rm median}(N)$"),
                    fontsize=10,
                )

            if xlog:
                ax.set_xscale("log")
            if ylog:
                ax.set_yscale("log")

            ax.set_xlim(M200C_RANGE)
            ax.grid(True, alpha=0.3)

    for j in range(ncols):
        axes[-1, j].set_xlabel(r"$M_{200\mathrm{c}}\,[M_\odot/h]$", fontsize=11)

    stitle = "LRG / ELG HOD from raw simulation data"
    if snap is not None:
        if zmap is not None and snap in zmap:
            stitle += f" at snap={snap:03d} (z={zmap[snap]:.2f})"
        else:
            stitle += f" at snap={snap:03d}"
    stitle += r"; faint = FoF, bold = $R_{200c}$"
    fig.suptitle(stitle, y=1.02, fontsize=17)

    legend_handles = []
    legend_labels = []
    for region in HOD_REGIONS:
        for component in COMPONENTS:
            legend_handles.append(
                Line2D(
                    [0], [0],
                    color=component_colors[component],
                    lw=region_lw[region],
                    alpha=region_alpha[region],
                    linestyle=component_linestyles[component],
                )
            )
            legend_labels.append(f"{region_labels[region]} {component_labels[component]}")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=3,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.5,
        title="Occupation components and radial selections",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    return fig, axes

# %% code cell 6
# ============================================================
# Curve-data saving helpers: HDF5 output
# ============================================================
# Every figure-producing cell below can save the numerical curves used in the plot.
# All curve data are saved as HDF5 files, with one file per simulation label/model
# and snapshot:
#
#   curve_data_hdf5/<dataset>_<label>_snapXXX.hdf5
#
# The same file is appended when HOD and radial-profile curves are produced for
# the same label and snapshot. For example:
#
#   ClusterSims_GR_snap021.hdf5
#       /hod/...
#       /radial_profiles/...
#       /meta
#
# This keeps all numerical curves in one folder while preserving a clean
# one-model-one-snapshot file structure.

import json
import re
import h5py
from pathlib import Path

SAVE_CURVE_DATA = True
SAVE_SMOOTH_CURVE_DATA = True
CURVE_DATA_ROOT = Path("./curve_data_hdf5")
HDF5_CURVE_DATA_DIR = CURVE_DATA_ROOT


def _ensure_curve_data_dirs():
    """Create the HDF5 curve-data output folder."""
    HDF5_CURVE_DATA_DIR.mkdir(parents=True, exist_ok=True)


_ensure_curve_data_dirs()


def _safe_name(x):
    """Return a filesystem/HDF5-safe name component."""
    s = str(x)
    s = s.replace("/", "_").replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_.+\-]+", "_", s)
    return s.strip("_") or "unknown"


def _curve_hdf5_path(dataset_label, label, snap, outdir=None):
    """
    HDF5 path for one dataset/model/snapshot.

    Parameters
    ----------
    dataset_label : str
        Dataset family, e.g. "ClusterSims" or "TNG".
    label : str
        Gravity model or TNG run label, e.g. "GR", "F50", "TNG300-1".
    snap : int
        Snapshot number.
    outdir : str or Path or None
        Optional override for the output directory.
    """
    outdir = HDF5_CURVE_DATA_DIR if outdir is None else Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"{_safe_name(dataset_label)}_{_safe_name(label)}_snap{int(snap):03d}.hdf5"
    return outdir / fname


def _to_python_scalar(x):
    """Convert numpy scalars/arrays to JSON-serializable Python objects."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        val = float(x)
        return val if np.isfinite(val) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return [_to_python_scalar(v) for v in np.ravel(x)]
    if isinstance(x, tuple):
        return [_to_python_scalar(v) for v in x]
    if isinstance(x, list):
        return [_to_python_scalar(v) for v in x]
    return x


def _jsonable_row(row):
    """Convert one metadata row to JSON-serializable values."""
    return {str(k): _to_python_scalar(v) for k, v in dict(row).items()}


def _as_1d(a, dtype=float):
    """Return a 1D numpy array; empty if input is missing."""
    if a is None:
        return np.asarray([], dtype=dtype)
    arr = np.atleast_1d(np.asarray(a))
    if dtype is not None:
        try:
            arr = arr.astype(dtype)
        except Exception:
            pass
    return arr


def _replace_group(parent, name):
    """Create a fresh group, replacing an old group with the same name."""
    if name in parent:
        del parent[name]
    return parent.create_group(name)


def _write_dataset(group, name, data, dtype=None, compression="gzip"):
    """Write a numeric dataset, replacing an existing dataset if needed."""
    if name in group:
        del group[name]
    arr = np.asarray(data)
    if dtype is not None:
        arr = arr.astype(dtype)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    kwargs = {}
    if arr.size > 0 and arr.ndim > 0:
        kwargs["compression"] = compression
    group.create_dataset(name, data=arr, **kwargs)


def _write_attrs(group, attrs):
    """Write simple attributes, skipping values that HDF5 cannot store directly."""
    for key, value in attrs.items():
        if value is None:
            continue
        try:
            if isinstance(value, (dict, list, tuple, np.ndarray)):
                group.attrs[key] = json.dumps(_to_python_scalar(value))
            else:
                group.attrs[key] = _to_python_scalar(value)
        except Exception:
            group.attrs[key] = str(value)


def _write_json_dataset(group, name, obj):
    """Write JSON metadata as a UTF-8 variable-length string dataset."""
    if name in group:
        del group[name]
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    dt = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=text, dtype=dt)


def _write_common_meta(f, *, dataset_label, label, snap, zval):
    """Write/overwrite common file-level metadata."""
    meta = f.require_group("meta")
    meta.attrs["dataset"] = str(dataset_label)
    meta.attrs["label"] = str(label)
    meta.attrs["snap"] = int(snap)
    if np.isfinite(zval):
        meta.attrs["z"] = float(zval)
    meta.attrs["format"] = "HOD_LRG_ELG_curve_data_v2_hdf5"
    meta.attrs["description"] = (
        "Numerical curves used by the HOD_LRG_ELG notebook. "
        "One HDF5 file is written for each simulation label/model and snapshot."
    )


def save_hod_curve_data(
    curves,
    analysis_specs,
    labels,
    *,
    snap,
    zmap=None,
    dataset_label="ClusterSims",
    summary_rows=None,
    outdir=None,
    ylog=SHOW_YLOG,
    smooth=SMOOTH_HOD_CURVES,
    smooth_ngrid=SMOOTH_NGRID,
):
    """
    Save all HOD curves used in a plot to HDF5 files.

    Output structure for each model/snapshot file:

        /meta
        /hod/<sample>/<region>/<component>/binned/{x,y,y16,y84,n_halo,x_left,x_right}
        /hod/<sample>/<region>/<component>/smooth/{x,y}
        /hod/summary_json

    The output file is:

        curve_data_hdf5/<dataset>_<label>_snapXXX.hdf5
    """
    if not SAVE_CURVE_DATA:
        return []

    outdir = HDF5_CURVE_DATA_DIR if outdir is None else Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    zval = np.nan if zmap is None else zmap.get(int(snap), np.nan)
    written_paths = []

    for label in labels:
        outpath = _curve_hdf5_path(dataset_label, label, snap, outdir=outdir)
        with h5py.File(outpath, "a") as f:
            _write_common_meta(f, dataset_label=dataset_label, label=label, snap=snap, zval=zval)
            hod_grp = _replace_group(f, "hod")
            _write_attrs(hod_grp, {
                "x_variable": "M200c_Msun_over_h",
                "ylog": bool(ylog),
                "smooth_requested": bool(smooth),
                "smooth_ngrid": int(smooth_ngrid),
                "statistic": STAT,
                "regions": list(HOD_REGIONS),
                "components": list(COMPONENTS),
            })

            for spec in analysis_specs:
                sname = spec["name"]
                if sname not in curves or label not in curves[sname]:
                    continue
                sample_grp = hod_grp.create_group(_safe_name(sname))
                _write_attrs(sample_grp, {
                    "sample": sname,
                    "title": spec.get("title", sname),
                })

                for region in HOD_REGIONS:
                    if region not in curves[sname][label]:
                        continue
                    region_grp = sample_grp.create_group(_safe_name(region))
                    _write_attrs(region_grp, {"region": region})

                    for component in COMPONENTS:
                        curve = curves[sname][label][region].get(component)
                        if curve is None:
                            continue
                        comp_grp = region_grp.create_group(_safe_name(component))
                        _write_attrs(comp_grp, {
                            "component": component,
                            "stat": curve.get("stat", STAT),
                            "y_variable": component,
                        })

                        x = _as_1d(curve.get("x"), dtype=float)
                        y = _as_1d(curve.get("y"), dtype=float)
                        y16 = _as_1d(curve.get("y16"), dtype=float)
                        y84 = _as_1d(curve.get("y84"), dtype=float)
                        n_halo = _as_1d(curve.get("n_halo"), dtype=float)
                        x_left = _as_1d(curve.get("x_left"), dtype=float)
                        x_right = _as_1d(curve.get("x_right"), dtype=float)

                        bgrp = comp_grp.create_group("binned")
                        _write_dataset(bgrp, "x", x)
                        _write_dataset(bgrp, "y", y)
                        _write_dataset(bgrp, "y16", y16)
                        _write_dataset(bgrp, "y84", y84)
                        _write_dataset(bgrp, "n_halo", n_halo)
                        _write_dataset(bgrp, "x_left", x_left)
                        _write_dataset(bgrp, "x_right", x_right)
                        _write_attrs(bgrp, {
                            "curve_kind": "binned",
                            "x_variable": "M200c_Msun_over_h",
                            "y_variable": component,
                        })

                        if SAVE_SMOOTH_CURVE_DATA and smooth and len(x) > 0:
                            xs, ys = smooth_curve_logx(x, y, ngrid=smooth_ngrid, ylog=ylog)
                            sgrp = comp_grp.create_group("smooth")
                            _write_dataset(sgrp, "x", _as_1d(xs, dtype=float))
                            _write_dataset(sgrp, "y", _as_1d(ys, dtype=float))
                            _write_attrs(sgrp, {
                                "curve_kind": "smooth",
                                "x_variable": "M200c_Msun_over_h",
                                "y_variable": component,
                            })

            if summary_rows is not None:
                rows_for_label = []
                for row in summary_rows:
                    row_label = row.get("label", row.get("flag", ""))
                    if str(row_label) == str(label):
                        rr = dict(row)
                        rr.setdefault("dataset", dataset_label)
                        rr.setdefault("label", label)
                        rr.setdefault("snap", int(snap))
                        rr.setdefault("z", zval)
                        rows_for_label.append(_jsonable_row(rr))
                _write_json_dataset(hod_grp, "summary_json", rows_for_label)

        written_paths.append(outpath)

    if VERBOSE:
        for p in written_paths:
            print("Saved HOD curve HDF5:", p)

    return written_paths


def save_radial_profile_curve_data(
    radial_profiles,
    analysis_specs,
    labels,
    *,
    snap,
    zmap=None,
    dataset_label="ClusterSims",
    summary_rows=None,
    outdir=None,
    ykey=None,
    ylog=True,
    smooth=True,
):
    """
    Save all radial-profile curves used in a plot to HDF5 files.

    Output structure for each model/snapshot file:

        /meta
        /radial_profiles/<sample>/<mass_bin>/binned/{x,r_left,r_right,counts,...}
        /radial_profiles/<sample>/<mass_bin>/smooth/{x,y}
        /radial_profiles/summary_json

    The file name is the same as the HOD output, so HOD and radial-profile
    curves for the same model/snapshot are stored together.
    """
    if not SAVE_CURVE_DATA:
        return []

    if ykey is None:
        ykey = globals().get("RADIAL_PROFILE_Y", "mean_shell_count")

    outdir = HDF5_CURVE_DATA_DIR if outdir is None else Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    zval = np.nan if zmap is None else zmap.get(int(snap), np.nan)
    written_paths = []

    for label in labels:
        outpath = _curve_hdf5_path(dataset_label, label, snap, outdir=outdir)
        with h5py.File(outpath, "a") as f:
            _write_common_meta(f, dataset_label=dataset_label, label=label, snap=snap, zval=zval)
            rad_grp = _replace_group(f, "radial_profiles")
            _write_attrs(rad_grp, {
                "x_variable": "r_over_R200c",
                "y_variable": ykey,
                "ylog": bool(ylog),
                "smooth_requested": bool(smooth),
                "profile_uses_FoF_members_not_R200c_cut": True,
                "include_centrals": globals().get("RADIAL_INCLUDE_CENTRALS", False),
            })

            for spec in analysis_specs:
                sname = spec["name"]
                if sname not in radial_profiles or label not in radial_profiles[sname]:
                    continue
                sample_grp = rad_grp.create_group(_safe_name(sname))
                _write_attrs(sample_grp, {
                    "sample": sname,
                    "title": spec.get("title", sname),
                })

                profiles = radial_profiles[sname][label]
                for (lo, hi), prof in profiles.items():
                    mb_name = _safe_name(f"M{lo:.3e}_{hi:.3e}")
                    mb_grp = sample_grp.create_group(mb_name)
                    _write_attrs(mb_grp, {
                        "mass_bin_label": prof.get("mass_label", ""),
                        "mass_bin_lo": float(lo),
                        "mass_bin_hi": float(hi),
                        "n_halo": int(prof.get("n_halo", 0)),
                        "n_gal": int(prof.get("n_gal", 0)),
                        "include_centrals": bool(prof.get("include_centrals", globals().get("RADIAL_INCLUDE_CENTRALS", False))),
                        "profile_ykey": ykey,
                    })

                    x = _as_1d(prof.get("x"), dtype=float)
                    y = _as_1d(prof.get(ykey), dtype=float)
                    bgrp = mb_grp.create_group("binned")
                    _write_dataset(bgrp, "x", x)
                    _write_dataset(bgrp, "r_left", _as_1d(prof.get("r_left"), dtype=float))
                    _write_dataset(bgrp, "r_right", _as_1d(prof.get("r_right"), dtype=float))
                    _write_dataset(bgrp, "counts", _as_1d(prof.get("counts"), dtype=float))
                    _write_dataset(bgrp, "mean_shell_count", _as_1d(prof.get("mean_shell_count"), dtype=float))
                    _write_dataset(bgrp, "number_density", _as_1d(prof.get("number_density"), dtype=float))
                    _write_dataset(bgrp, "cumulative", _as_1d(prof.get("cumulative"), dtype=float))
                    _write_dataset(bgrp, "y", y)
                    _write_attrs(bgrp, {
                        "curve_kind": "binned",
                        "x_variable": "r_over_R200c",
                        "y_variable": ykey,
                    })

                    if SAVE_SMOOTH_CURVE_DATA and smooth and len(x) > 0:
                        xs, ys = _smooth_profile_x(x, y, ylog=ylog, ngrid=250)
                        sgrp = mb_grp.create_group("smooth")
                        _write_dataset(sgrp, "x", _as_1d(xs, dtype=float))
                        _write_dataset(sgrp, "y", _as_1d(ys, dtype=float))
                        _write_attrs(sgrp, {
                            "curve_kind": "smooth",
                            "x_variable": "r_over_R200c",
                            "y_variable": ykey,
                        })

            if summary_rows is not None:
                rows_for_label = []
                for row in summary_rows:
                    if str(row.get("label", "")) == str(label):
                        rr = dict(row)
                        rr.setdefault("dataset", dataset_label)
                        rr.setdefault("label", label)
                        rr.setdefault("snap", int(snap))
                        rr.setdefault("z", zval)
                        rows_for_label.append(_jsonable_row(rr))
                _write_json_dataset(rad_grp, "summary_json", rows_for_label)

        written_paths.append(outpath)

    if VERBOSE:
        for p in written_paths:
            print("Saved radial-profile curve HDF5:", p)

    return written_paths


def inspect_curve_hdf5(path):
    """Small helper to inspect the saved HDF5 curve-data structure."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        print(path)
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name:80s} shape={obj.shape} dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  {name}/")
        f.visititems(visitor)

# %% code cell 7
# ============================================================
# Curve-data loading helpers and catalog/HDF5 fallback control
# ============================================================
# Default behaviour:
#   1. Try to read the raw catalog and rebuild the curves.
#   2. Save the newly built curves to HDF5.
#   3. If the raw catalog is unavailable, fall back to the previously saved HDF5 curves.
#
# You can change CURVE_DATA_SOURCE to:
#   "catalog_first" : default; use catalog, fallback to HDF5 if catalog loading fails.
#   "hdf5_first"    : use existing HDF5 if available; otherwise build from catalog.
#   "hdf5_only"     : only read HDF5; never touch the raw catalog.
#   "catalog_only"  : only read the raw catalog; never fall back to HDF5.

CURVE_DATA_SOURCE = "catalog_first"


def _read_json_dataset(group, name, default=None):
    """Read a JSON string dataset written by _write_json_dataset()."""
    if default is None:
        default = []
    if group is None or name not in group:
        return default
    obj = group[name][()]
    if isinstance(obj, bytes):
        text = obj.decode("utf-8")
    elif isinstance(obj, np.ndarray):
        val = obj[()]
        text = val.decode("utf-8") if isinstance(val, bytes) else str(val)
    else:
        text = str(obj)
    try:
        return json.loads(text)
    except Exception:
        return default


def _read_dataset_if_exists(group, name, default=None, dtype=float):
    """Read one dataset from an HDF5 group, returning an empty/default array if absent."""
    if default is None:
        default = np.asarray([], dtype=dtype)
    if group is None or name not in group:
        return np.asarray(default, dtype=dtype)
    arr = np.asarray(group[name][()])
    if dtype is not None:
        try:
            arr = arr.astype(dtype)
        except Exception:
            pass
    return np.atleast_1d(arr)


def _curve_hdf5_has_group(dataset_label, label, snap, group_name, outdir=None):
    """Return True if the model/snapshot curve HDF5 file exists and contains a group."""
    path = _curve_hdf5_path(dataset_label, label, snap, outdir=outdir)
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            return group_name in f
    except OSError:
        return False


def all_curve_hdf5_available(labels, snap, dataset_label, group_name, outdir=None):
    """Check whether all labels have saved HDF5 curve data for one group."""
    return all(_curve_hdf5_has_group(dataset_label, label, snap, group_name, outdir=outdir) for label in labels)


def load_hod_curve_data_from_hdf5(labels, snap, analysis_specs, *, dataset_label="ClusterSims", indir=None):
    """
    Load HOD curves from previously saved HDF5 files.

    The returned structure is compatible with plot_hod_grid().
    """
    indir = HDF5_CURVE_DATA_DIR if indir is None else Path(indir)
    curves = {spec["name"]: {} for spec in analysis_specs}
    summary_rows = []

    missing = []
    for label in labels:
        path = _curve_hdf5_path(dataset_label, label, snap, outdir=indir)
        if not path.exists():
            missing.append(str(path))
            continue

        with h5py.File(path, "r") as f:
            if "hod" not in f:
                missing.append(str(path) + "::/hod")
                continue
            hod_grp = f["hod"]
            summary_rows.extend(_read_json_dataset(hod_grp, "summary_json", default=[]))

            for spec in analysis_specs:
                sname = spec["name"]
                sample_key = _safe_name(sname)
                curves[sname].setdefault(label, {})
                if sample_key not in hod_grp:
                    continue
                sample_grp = hod_grp[sample_key]

                for region in HOD_REGIONS:
                    region_key = _safe_name(region)
                    curves[sname][label].setdefault(region, {})
                    if region_key not in sample_grp:
                        continue
                    region_grp = sample_grp[region_key]

                    for component in COMPONENTS:
                        component_key = _safe_name(component)
                        if component_key not in region_grp:
                            continue
                        comp_grp = region_grp[component_key]
                        bgrp = comp_grp.get("binned")
                        curves[sname][label][region][component] = {
                            "x": _read_dataset_if_exists(bgrp, "x"),
                            "y": _read_dataset_if_exists(bgrp, "y"),
                            "y16": _read_dataset_if_exists(bgrp, "y16"),
                            "y84": _read_dataset_if_exists(bgrp, "y84"),
                            "n_halo": _read_dataset_if_exists(bgrp, "n_halo"),
                            "x_left": _read_dataset_if_exists(bgrp, "x_left"),
                            "x_right": _read_dataset_if_exists(bgrp, "x_right"),
                            "component": component,
                            "region": region,
                            "stat": comp_grp.attrs.get("stat", STAT),
                            "mass_bins": MASS_BINS,
                            "source": "hdf5",
                            "hdf5_path": str(path),
                        }

    if missing:
        raise FileNotFoundError(
            "Missing HOD curve HDF5 data for dataset/model/snapshot.\n"
            + "\n".join(missing[:20])
            + ("\n..." if len(missing) > 20 else "")
        )

    if VERBOSE:
        print(f"Loaded HOD curves from HDF5: dataset={dataset_label}, snap={snap:03d}")
    return curves, summary_rows


def load_radial_profile_curve_data_from_hdf5(
    labels,
    snap,
    analysis_specs,
    *,
    dataset_label="ClusterSims",
    indir=None,
    ykey=None,
):
    """
    Load radial-profile curves from previously saved HDF5 files.

    The returned structure is compatible with plot_radial_profile_grid().
    """
    if ykey is None:
        ykey = globals().get("RADIAL_PROFILE_Y", "mean_shell_count")

    indir = HDF5_CURVE_DATA_DIR if indir is None else Path(indir)
    radial_profiles = {spec["name"]: {} for spec in analysis_specs}
    summary_rows = []

    missing = []
    for label in labels:
        path = _curve_hdf5_path(dataset_label, label, snap, outdir=indir)
        if not path.exists():
            missing.append(str(path))
            continue

        with h5py.File(path, "r") as f:
            if "radial_profiles" not in f:
                missing.append(str(path) + "::/radial_profiles")
                continue
            rad_grp = f["radial_profiles"]
            summary_rows.extend(_read_json_dataset(rad_grp, "summary_json", default=[]))

            for spec in analysis_specs:
                sname = spec["name"]
                sample_key = _safe_name(sname)
                radial_profiles[sname].setdefault(label, {})
                if sample_key not in rad_grp:
                    continue
                sample_grp = rad_grp[sample_key]

                for mb_key in sample_grp.keys():
                    mb_grp = sample_grp[mb_key]
                    if not isinstance(mb_grp, h5py.Group):
                        continue
                    bgrp = mb_grp.get("binned")
                    lo = float(mb_grp.attrs.get("mass_bin_lo", np.nan))
                    hi = float(mb_grp.attrs.get("mass_bin_hi", np.nan))
                    if not np.isfinite(lo) or not np.isfinite(hi):
                        continue

                    prof = {
                        "x": _read_dataset_if_exists(bgrp, "x"),
                        "r_left": _read_dataset_if_exists(bgrp, "r_left"),
                        "r_right": _read_dataset_if_exists(bgrp, "r_right"),
                        "counts": _read_dataset_if_exists(bgrp, "counts"),
                        "mean_shell_count": _read_dataset_if_exists(bgrp, "mean_shell_count"),
                        "number_density": _read_dataset_if_exists(bgrp, "number_density"),
                        "cumulative": _read_dataset_if_exists(bgrp, "cumulative"),
                        "mass_label": mb_grp.attrs.get("mass_bin_label", _radial_mass_bin_label(lo, hi)),
                        "n_halo": int(mb_grp.attrs.get("n_halo", 0)),
                        "n_gal": int(mb_grp.attrs.get("n_gal", 0)),
                        "include_centrals": bool(mb_grp.attrs.get("include_centrals", False)),
                        "source": "hdf5",
                        "hdf5_path": str(path),
                    }
                    # Ensure the requested ykey exists even for older cache files.
                    if ykey not in prof and bgrp is not None and "y" in bgrp:
                        prof[ykey] = _read_dataset_if_exists(bgrp, "y")
                    radial_profiles[sname][label][(lo, hi)] = prof

    if missing:
        raise FileNotFoundError(
            "Missing radial-profile HDF5 curve data for dataset/model/snapshot.\n"
            + "\n".join(missing[:20])
            + ("\n..." if len(missing) > 20 else "")
        )

    if VERBOSE:
        print(f"Loaded radial profiles from HDF5: dataset={dataset_label}, snap={snap:03d}")
    return radial_profiles, summary_rows


def get_or_build_hod_curves(
    labels,
    snap,
    analysis_specs,
    build_func,
    *,
    dataset_label="ClusterSims",
    zmap=None,
    source=None,
    ylog=SHOW_YLOG,
    smooth=SMOOTH_HOD_CURVES,
):
    """
    Get HOD curves either from raw catalogs or from saved HDF5 curve files.

    By default, this uses CURVE_DATA_SOURCE="catalog_first": build from catalog
    and save to HDF5; if the catalog is unavailable, load the saved HDF5 curves.
    """
    source = CURVE_DATA_SOURCE if source is None else source
    labels = list(labels)

    if source not in {"catalog_first", "hdf5_first", "hdf5_only", "catalog_only"}:
        raise ValueError("source must be catalog_first, hdf5_first, hdf5_only, or catalog_only")

    if source in {"hdf5_first", "hdf5_only"}:
        if all_curve_hdf5_available(labels, snap, dataset_label, "hod"):
            curves, rows = load_hod_curve_data_from_hdf5(labels, snap, analysis_specs, dataset_label=dataset_label)
            return curves, rows, "hdf5"
        if source == "hdf5_only":
            # This raises a detailed missing-file error.
            curves, rows = load_hod_curve_data_from_hdf5(labels, snap, analysis_specs, dataset_label=dataset_label)
            return curves, rows, "hdf5"

    try:
        curves, rows = build_func()
        save_hod_curve_data(
            curves,
            analysis_specs,
            labels,
            snap=snap,
            zmap=zmap,
            dataset_label=dataset_label,
            summary_rows=rows,
            ylog=ylog,
            smooth=smooth,
        )
        return curves, rows, "catalog"
    except Exception as catalog_error:
        if source == "catalog_only":
            raise
        if VERBOSE:
            print(f"[fallback] Catalog loading failed for {dataset_label} snap={snap:03d}: {catalog_error}")
            print("[fallback] Trying saved HDF5 curve data instead.")
        try:
            curves, rows = load_hod_curve_data_from_hdf5(labels, snap, analysis_specs, dataset_label=dataset_label)
            return curves, rows, "hdf5"
        except Exception as hdf5_error:
            raise RuntimeError(
                f"Could not build HOD curves from catalog and could not load saved HDF5 curves.\n"
                f"Catalog error: {catalog_error}\n"
                f"HDF5 error: {hdf5_error}"
            ) from hdf5_error


def get_or_build_radial_profiles(
    labels,
    snap,
    analysis_specs,
    build_func,
    *,
    dataset_label="ClusterSims",
    zmap=None,
    source=None,
    ykey=None,
    ylog=True,
    smooth=True,
):
    """
    Get radial profiles either from raw catalogs or from saved HDF5 curve files.
    """
    source = CURVE_DATA_SOURCE if source is None else source
    labels = list(labels)
    if ykey is None:
        ykey = globals().get("RADIAL_PROFILE_Y", "mean_shell_count")

    if source not in {"catalog_first", "hdf5_first", "hdf5_only", "catalog_only"}:
        raise ValueError("source must be catalog_first, hdf5_first, hdf5_only, or catalog_only")

    if source in {"hdf5_first", "hdf5_only"}:
        if all_curve_hdf5_available(labels, snap, dataset_label, "radial_profiles"):
            profiles, rows = load_radial_profile_curve_data_from_hdf5(
                labels, snap, analysis_specs, dataset_label=dataset_label, ykey=ykey
            )
            return profiles, rows, "hdf5"
        if source == "hdf5_only":
            profiles, rows = load_radial_profile_curve_data_from_hdf5(
                labels, snap, analysis_specs, dataset_label=dataset_label, ykey=ykey
            )
            return profiles, rows, "hdf5"

    try:
        profiles, rows = build_func()
        save_radial_profile_curve_data(
            profiles,
            analysis_specs,
            labels,
            snap=snap,
            zmap=zmap,
            dataset_label=dataset_label,
            summary_rows=rows,
            ykey=ykey,
            ylog=ylog,
            smooth=smooth,
        )
        return profiles, rows, "catalog"
    except Exception as catalog_error:
        if source == "catalog_only":
            raise
        if VERBOSE:
            print(f"[fallback] Catalog loading failed for {dataset_label} radial snap={snap:03d}: {catalog_error}")
            print("[fallback] Trying saved HDF5 radial-profile data instead.")
        try:
            profiles, rows = load_radial_profile_curve_data_from_hdf5(
                labels, snap, analysis_specs, dataset_label=dataset_label, ykey=ykey
            )
            return profiles, rows, "hdf5"
        except Exception as hdf5_error:
            raise RuntimeError(
                f"Could not build radial profiles from catalog and could not load saved HDF5 curves.\n"
                f"Catalog error: {catalog_error}\n"
                f"HDF5 error: {hdf5_error}"
            ) from hdf5_error

print("HDF5 fallback helpers loaded.")
print("CURVE_DATA_SOURCE =", CURVE_DATA_SOURCE)

# %% [markdown] cell 8
# ## Single-snapshot example This cell reads the raw simulation data directly and plots the LRG/ELG HOD grid for one snapshot.

# %% code cell 9
# IPython-only: %%time
# ============================================================
# ClusterSims single-snapshot HOD example
# ============================================================
snap = 21  # single-snapshot preview; the loop below uses all snapshots in `snaps`
analysis_specs = make_analysis_specs()

curves, summary_rows, curve_source = get_or_build_hod_curves(
    flags,
    snap,
    analysis_specs,
    build_func=lambda: build_curves_for_snapshot(flags, snap, analysis_specs),
    dataset_label="ClusterSims",
    zmap=zmap,
    ylog=SHOW_YLOG,
    smooth=SMOOTH_HOD_CURVES,
)

if VERBOSE:
    print(f"Curve source: {curve_source}")
    print("Selection summary:")
    for row in summary_rows:
        print(row)

fig, axes = plot_hod_grid(
    curves,
    analysis_specs,
    flags=flags,
    snap=snap,
    zmap=zmap,
    figsize=(22, 7),
    xlog=True,
    ylog=SHOW_YLOG,
)

plt.show()

# %% [markdown] cell 10
# ## Loop over all snapshots Run this cell to generate and save one LRG/ELG HOD figure for each snapshot.

# %% code cell 11
# %%time
# ============================================================
# ClusterSims HOD loop over all snapshots
# ============================================================
outdir = Path("./plots")
outdir.mkdir(parents=True, exist_ok=True)

analysis_specs = make_analysis_specs()

for snap in snaps:
    curves, summary_rows, curve_source = get_or_build_hod_curves(
        flags,
        snap,
        analysis_specs,
        build_func=lambda snap=snap: build_curves_for_snapshot(flags, snap, analysis_specs),
        dataset_label="ClusterSims",
        zmap=zmap,
        ylog=SHOW_YLOG,
        smooth=SMOOTH_HOD_CURVES,
    )

    if VERBOSE:
        print(f"\n=== snap={snap:03d}, z={zmap.get(snap, np.nan):.2f}, source={curve_source} ===")
        for row in summary_rows:
            print(row)

    fig, axes = plot_hod_grid(
        curves,
        analysis_specs,
        flags=flags,
        snap=snap,
        zmap=zmap,
        figsize=(22, 7),
        xlog=True,
        ylog=SHOW_YLOG,
    )

    fout = outdir / f"HOD_LRG_ELG_direct_smooth_FoF_R200c_snap{snap:03d}.png"
    fig.savefig(fout, dpi=200, bbox_inches="tight")
    if VERBOSE:
        print("Saved:", fout)
    plt.show()
    # plt.close(fig)

# %% [markdown] cell 12
# ## TNGCatalog: LRG / ELG HOD from TNG The following cells add a parallel TNG workflow. They use the same LRG / ELG proxy cuts, the same HOD binning, and the same three curves $$ \langle N_{\rm cen}|M\rangle,\quad \langle N_{\rm sat}|M\rangle,\quad \langle N_{\rm tot}|M\rangle . $$ The TNG loader returns a dictionary with the same keys as `load_sim_direct`, so the existing HOD measurement and plotting functions are reused. Edit `TNG_RUNS` and `TNG_SNAPS` below to match your local TNG directory an

# %% code cell 13

# ============================================================
# TNGCatalog setup
# ============================================================
# This block is intentionally separate from the ClusterSims setup above.
# Edit TNG_RUNS to your actual TNG group-catalog root directories on COSMA.

try:
    from catalog_loader import TNGCatalog
except Exception:
    # Fallback if your local class is defined in a separate module.
    from global_tng import TNGCatalog

# Example layout. Change these paths if your TNG data are elsewhere.
# The value should be the directory accepted by TNGCatalog(base_path, snap).
TNG_RUNS = {
    "TNG": Path("/cosma8/data/dp203/dc-wang17/TNG/tng_data"),
}

# Standard TNG z=0 snapshot is usually 99. Add more snapshots if needed.
TNG_SNAPS = [99]
TNG_ZMAP = {
    99: 0.00,
    # Add entries if you use more TNG snapshots, e.g. 67: 0.50, 50: 1.00.
}

if VERBOSE:
    print("TNGCatalog mode enabled.")
    print("TNG runs:")
    for name, path in TNG_RUNS.items():
        print(f"  {name}: {path}")
    print("TNG snapshots:", TNG_SNAPS)

# %% code cell 14

# ============================================================
# TNGCatalog loading helpers
# ============================================================
_tng_cache = {}


def _try_load_tng_fof(cat, group_fields, subhalo_fields):
    """Wrapper for TNGCatalog.loadFoF with the same interface as CSCatalog.loadFoF."""
    return cat.loadFoF(group_fields=group_fields, subhalo_fields=subhalo_fields)


def load_tng_direct(run_label, snap):
    """
    Read FoF halo and subhalo data directly from TNG via TNGCatalog.

    The output dictionary intentionally matches the one returned by load_sim_direct,
    so compute_occupation_counts_from_sim(), bin_occupation_curve(), and plot_hod_grid()
    can be reused without changes.

    Assumed TNG / AREPO group-catalog convention:
      Group_M_Crit200           : 1e10 Msun/h
      SubhaloMassInRadType[:,4] : 1e10 Msun/h
      SubhaloSFR                : Msun/yr
      GroupPos/SubhaloPos       : ckpc/h
      Group_R_Crit200           : ckpc/h
    """
    key = (str(run_label), int(snap))
    if key in _tng_cache:
        return _tng_cache[key]

    if run_label not in TNG_RUNS:
        raise KeyError(f"Unknown TNG run {run_label!r}. Available keys: {list(TNG_RUNS)}")

    base_path = Path(TNG_RUNS[run_label])
    cat = TNGCatalog(base_path, int(snap))

    group_field_attempts = [
        [HALO_MASS_KEY, HALO_RADIUS_KEY, GROUP_POS_KEY, GROUP_FIRST_SUB_KEY, GROUP_NSUBS_KEY],
        [HALO_MASS_KEY, HALO_RADIUS_KEY, GROUP_POS_KEY, GROUP_FIRST_SUB_KEY],
        [HALO_MASS_KEY, HALO_RADIUS_KEY, GROUP_POS_KEY],
        [HALO_MASS_KEY, GROUP_FIRST_SUB_KEY, GROUP_NSUBS_KEY],
        [HALO_MASS_KEY, GROUP_FIRST_SUB_KEY],
        [HALO_MASS_KEY],
    ]

    subhalo_field_attempts = [
        [SUBHALO_GROUP_KEY, SUBHALO_POS_KEY, MSTAR_KEY_PREFERRED, SFR_KEY],
        [SUBHALO_GROUP_KEY, SUBHALO_POS_KEY, MSTAR_KEY_FALLBACK, SFR_KEY],
        [SUBHALO_GROUP_KEY, MSTAR_KEY_PREFERRED, SFR_KEY],
        [SUBHALO_GROUP_KEY, MSTAR_KEY_FALLBACK, SFR_KEY],
        [SUBHALO_POS_KEY, MSTAR_KEY_PREFERRED, SFR_KEY],
        [SUBHALO_POS_KEY, MSTAR_KEY_FALLBACK, SFR_KEY],
        [MSTAR_KEY_PREFERRED, SFR_KEY],
        [MSTAR_KEY_FALLBACK, SFR_KEY],
    ]

    last_error = None
    halos = subs = None
    used_group_fields = used_subhalo_fields = None

    for gf in group_field_attempts:
        for sf in subhalo_field_attempts:
            try:
                halos, subs = _try_load_tng_fof(cat, group_fields=gf, subhalo_fields=sf)
                used_group_fields = gf
                used_subhalo_fields = sf
                break
            except Exception as err:
                last_error = err
        if halos is not None:
            break

    if halos is None or subs is None:
        raise RuntimeError(
            f"Could not load TNG FoF/Subhalo fields for run={run_label}, snap={snap}. "
            f"Last error was: {last_error}"
        )

    if HALO_MASS_KEY not in halos:
        raise KeyError(f"Required halo field {HALO_MASS_KEY!r} is missing from TNG data.")

    m200c = np.asarray(halos[HALO_MASS_KEY], dtype=float) * HALO_MASS_SCALE
    nhalo = len(m200c)

    stellar_mass_key = None
    if MSTAR_KEY_PREFERRED in subs:
        stellar_mass_key = MSTAR_KEY_PREFERRED
    elif MSTAR_KEY_FALLBACK in subs:
        stellar_mass_key = MSTAR_KEY_FALLBACK
    else:
        raise KeyError(f"Neither {MSTAR_KEY_PREFERRED!r} nor {MSTAR_KEY_FALLBACK!r} was loaded.")

    mstar_raw = np.asarray(subs[stellar_mass_key], dtype=float)
    if mstar_raw.ndim == 2:
        mstar_raw = mstar_raw[:, STELLAR_COMPONENT_INDEX]
    mstar_msunh = mstar_raw * STELLAR_MASS_SCALE
    mstar_msun = mstar_msunh / HUBBLE_h

    if SFR_KEY not in subs:
        raise KeyError(f"Required subhalo field {SFR_KEY!r} is missing from TNG data.")
    sfr = np.asarray(subs[SFR_KEY], dtype=float)

    nsub = len(mstar_msunh)
    sid = np.arange(nsub, dtype=np.int64)

    # Host group ID for each subhalo.
    if SUBHALO_GROUP_KEY in subs:
        gid = np.asarray(subs[SUBHALO_GROUP_KEY], dtype=np.int64)
        gid_source = SUBHALO_GROUP_KEY
    elif GROUP_FIRST_SUB_KEY in halos and GROUP_NSUBS_KEY in halos:
        gid = np.full(nsub, -1, dtype=np.int64)
        first_raw = np.asarray(halos[GROUP_FIRST_SUB_KEY], dtype=np.int64)
        nsubs = np.asarray(halos[GROUP_NSUBS_KEY], dtype=np.int64)
        for g, (s0, ns) in enumerate(zip(first_raw, nsubs)):
            if s0 >= 0 and ns > 0 and s0 + ns <= nsub:
                gid[s0:s0 + ns] = g
        gid_source = f"fallback from {GROUP_FIRST_SUB_KEY}+{GROUP_NSUBS_KEY}"
    else:
        raise KeyError(
            f"Cannot determine host group IDs: need {SUBHALO_GROUP_KEY!r} "
            f"or both {GROUP_FIRST_SUB_KEY!r} and {GROUP_NSUBS_KEY!r}."
        )

    central_sid = np.full(nhalo, -1, dtype=np.int64)
    first_sub_indexing = "unknown"
    first_sub_match_fraction = np.nan
    if GROUP_FIRST_SUB_KEY in halos:
        first_raw = np.asarray(halos[GROUP_FIRST_SUB_KEY], dtype=np.int64)
        first, first_sub_indexing, first_sub_match_fraction = _choose_first_sub_indexing(
            first_raw, gid, nhalo, nsub
        )
        n_first = min(len(first), nhalo)
        central_sid[:n_first] = first[:n_first]
    else:
        valid_gid = (gid >= 0) & (gid < nhalo)
        order = np.argsort(gid[valid_gid], kind="stable")
        valid_sid = sid[valid_gid]
        valid_gid_sorted = gid[valid_gid][order]
        valid_sid_sorted = valid_sid[order]
        unique_gid, first_idx = np.unique(valid_gid_sorted, return_index=True)
        central_sid[unique_gid] = valid_sid_sorted[first_idx]
        first_sub_indexing = "first subhalo by sorted gid fallback"

    valid_sub = (
        np.isfinite(mstar_msunh)
        & np.isfinite(mstar_msun)
        & np.isfinite(sfr)
        & (mstar_msunh > 0)
        & (gid >= 0)
        & (gid < nhalo)
    )

    is_cen = np.zeros(nsub, dtype=bool)
    good_for_cen = valid_sub & (gid >= 0) & (gid < nhalo)
    is_cen[good_for_cen] = sid[good_for_cen] == central_sid[gid[good_for_cen]]
    is_sat = valid_sub & ~is_cen

    host_m200c = np.full(nsub, np.nan, dtype=float)
    good_gid = (gid >= 0) & (gid < nhalo)
    host_m200c[good_gid] = m200c[gid[good_gid]]

    has_r200c_geometry = (
        (HALO_RADIUS_KEY in halos)
        and (GROUP_POS_KEY in halos)
        and (SUBHALO_POS_KEY in subs)
    )

    host_r200c = np.full(nsub, np.nan, dtype=float)
    r_to_host = np.full(nsub, np.nan, dtype=float)
    inside_r200c = np.zeros(nsub, dtype=bool)

    if has_r200c_geometry:
        r200c_native = np.asarray(halos[HALO_RADIUS_KEY], dtype=float)
        group_pos = np.asarray(halos[GROUP_POS_KEY], dtype=float)
        sub_pos = np.asarray(subs[SUBHALO_POS_KEY], dtype=float)

        if group_pos.ndim != 2 or group_pos.shape[1] != 3:
            raise ValueError(f"{GROUP_POS_KEY} should have shape (Nhalo, 3); got {group_pos.shape}.")
        if sub_pos.ndim != 2 or sub_pos.shape[1] != 3:
            raise ValueError(f"{SUBHALO_POS_KEY} should have shape (Nsub, 3); got {sub_pos.shape}.")

        host_r200c[good_gid] = r200c_native[gid[good_gid]]
        r_to_host = _compute_subhalo_radius_to_host(group_pos, sub_pos, gid, nhalo)
        inside_r200c = (
            valid_sub
            & np.isfinite(r_to_host)
            & np.isfinite(host_r200c)
            & (host_r200c > 0)
            & (r_to_host <= host_r200c)
        )
        inside_r200c |= is_cen
    else:
        inside_r200c = valid_sub.copy()

    out = {
        "M200c": m200c,
        "Ng": nhalo,
        "gid": gid,
        "sid": sid,
        "central_sid": central_sid,
        "is_cen": is_cen,
        "is_sat": is_sat,
        "inside_r200c": inside_r200c,
        "r_to_host": r_to_host,
        "host_r200c": host_r200c,
        "has_r200c_geometry": bool(has_r200c_geometry),
        "mstar_msunh": mstar_msunh,
        "mstar_msun": mstar_msun,
        "host_m200c": host_m200c,
        "sfr": sfr,
        "stellar_mass_key": stellar_mass_key,
        "used_group_fields": used_group_fields,
        "used_subhalo_fields": used_subhalo_fields,
        "gid_source": gid_source,
        "first_sub_indexing": first_sub_indexing,
        "first_sub_match_fraction": first_sub_match_fraction,
        "run_label": run_label,
        "snap": int(snap),
        "z": TNG_ZMAP.get(int(snap), np.nan),
    }

    _tng_cache[key] = out
    return out


def build_tng_curves_for_snapshot(tng_runs, snap, analysis_specs):
    """Build LRG/ELG HOD curves for one TNG snapshot and one or more TNG runs."""
    run_labels = list(tng_runs.keys())
    curves = {}
    summary_rows = []

    for spec in analysis_specs:
        sname = spec["name"]
        curves[sname] = {}
        selector = spec["selector"]

        for run_label in run_labels:
            sim = load_tng_direct(run_label, snap)
            curves[sname][run_label] = {}
            occ_by_region = {}

            for region in HOD_REGIONS:
                occ = compute_occupation_counts_from_sim(sim, selector, region=region)
                occ_by_region[region] = occ
                curves[sname][run_label][region] = {}

                for component in COMPONENTS:
                    curves[sname][run_label][region][component] = bin_occupation_curve(
                        occ,
                        component=component,
                        mass_bins=MASS_BINS,
                        stat=STAT,
                        min_halos_per_bin=MIN_HALOS_PER_BIN,
                    )

            occ_fof = occ_by_region["fof"]
            occ_r200c = occ_by_region["r200c"]
            log_m200c_min, log_m200c_max = _finite_log10_range(sim["M200c"])
            log_mstar_min, log_mstar_max = _finite_log10_range(sim["mstar_msunh"])
            ratio_summary = _selected_stellar_to_host_summary(sim, occ_fof["selected_mask"])
            r_summary = _selected_r_over_r200c_summary(sim, occ_fof["selected_mask"])

            summary_rows.append({
                "sample": sname,
                "run": run_label,
                "snap": int(snap),
                "z": TNG_ZMAP.get(int(snap), np.nan),
                "Ng_selected_fof": occ_fof["Ngal_selected"],
                "Ncen_selected_fof": occ_fof["Ncen_selected"],
                "Nsat_selected_fof": occ_fof["Nsat_selected"],
                "Ng_selected_R200c": occ_r200c["Ngal_selected"],
                "Ncen_selected_R200c": occ_r200c["Ncen_selected"],
                "Nsat_selected_R200c": occ_r200c["Nsat_selected"],
                "has_r200c_geometry": sim["has_r200c_geometry"],
                "stellar_mass_key": sim["stellar_mass_key"],
                "gid_source": sim["gid_source"],
                "first_sub_indexing": sim["first_sub_indexing"],
                "first_sub_match_fraction": sim["first_sub_match_fraction"],
                "log10_M200c_range": (round(log_m200c_min, 2), round(log_m200c_max, 2)),
                "log10_Mstar_range": (round(log_mstar_min, 2), round(log_mstar_max, 2)),
                "median_Mstar_over_M200c": ratio_summary["median_Mstar_over_M200c"],
                "frac_Mstar_gt_0p2_M200c": ratio_summary["frac_Mstar_gt_0p2_M200c"],
                "median_r_over_r200c": r_summary["median_r_over_r200c"],
                "frac_selected_within_R200c": r_summary["frac_within_r200c"],
            })

    return curves, summary_rows

# %% code cell 15
# IPython-only: %%time
# ============================================================
# TNG single-snapshot HOD example
# ============================================================
# Use snap=99 for the usual TNG z=0 catalog. Change if needed.
tng_snap = TNG_SNAPS[0]
tng_labels = list(TNG_RUNS.keys())
analysis_specs = make_analysis_specs()

tng_curves, tng_summary_rows, curve_source = get_or_build_hod_curves(
    tng_labels,
    tng_snap,
    analysis_specs,
    build_func=lambda: build_tng_curves_for_snapshot(TNG_RUNS, tng_snap, analysis_specs),
    dataset_label="TNG",
    zmap=TNG_ZMAP,
    ylog=SHOW_YLOG,
    smooth=SMOOTH_HOD_CURVES,
)

if VERBOSE:
    print(f"Curve source: {curve_source}")
    print("TNG selection summary:")
    for row in tng_summary_rows:
        print(row)

fig, axes = plot_hod_grid(
    tng_curves,
    analysis_specs,
    flags=tng_labels,
    snap=tng_snap,
    zmap=TNG_ZMAP,
    figsize=(max(5.2 * len(tng_labels), 6.0), 7),
    xlog=True,
    ylog=SHOW_YLOG,
)
fig.suptitle(
    f"TNG LRG / ELG HOD from TNGCatalog at snap={tng_snap:03d}"
    + (f" (z={TNG_ZMAP[tng_snap]:.2f})" if tng_snap in TNG_ZMAP else "")
    + r"; faint = FoF, bold = $R_{200c}$",
    y=1.02,
    fontsize=17,
)
plt.show()

# %% code cell 16
# %%time
# ============================================================
# TNG HOD loop over selected snapshots
# ============================================================
tng_outdir = Path("./plots_tng")
tng_outdir.mkdir(parents=True, exist_ok=True)

tng_labels = list(TNG_RUNS.keys())
analysis_specs = make_analysis_specs()

for tng_snap in TNG_SNAPS:
    tng_curves, tng_summary_rows, curve_source = get_or_build_hod_curves(
        tng_labels,
        tng_snap,
        analysis_specs,
        build_func=lambda tng_snap=tng_snap: build_tng_curves_for_snapshot(TNG_RUNS, tng_snap, analysis_specs),
        dataset_label="TNG",
        zmap=TNG_ZMAP,
        ylog=SHOW_YLOG,
        smooth=SMOOTH_HOD_CURVES,
    )

    if VERBOSE:
        print(f"\n=== TNG snap={tng_snap:03d}, z={TNG_ZMAP.get(tng_snap, np.nan):.2f}, source={curve_source} ===")
        for row in tng_summary_rows:
            print(row)

    fig, axes = plot_hod_grid(
        tng_curves,
        analysis_specs,
        flags=tng_labels,
        snap=tng_snap,
        zmap=TNG_ZMAP,
        figsize=(max(5.2 * len(tng_labels), 6.0), 7),
        xlog=True,
        ylog=SHOW_YLOG,
    )
    fig.suptitle(
        f"TNG LRG / ELG HOD from TNGCatalog at snap={tng_snap:03d}"
        + (f" (z={TNG_ZMAP[tng_snap]:.2f})" if tng_snap in TNG_ZMAP else "")
        + r"; faint = FoF, bold = $R_{200c}$",
        y=1.02,
        fontsize=17,
    )

    fig.savefig(
        tng_outdir / f"TNG_HOD_LRG_ELG_FoF_R200c_snap{tng_snap:03d}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()
    # plt.close(fig)

# %% [markdown] cell 17
# ## LRG / ELG radial profiles inside host halos This section measures the radial distribution of the selected LRG / ELG galaxies around their host halo center. The key difference from the earlier $R_{200c}$ HOD curves is that this calculation **does not cut at** $R_{200c}$. It uses FoF membership and then bins the selected galaxies by $$ q \equiv \frac{r_{\rm gal-host}}{R_{200c}} . $$ Thus the profile can extend to $q>1$. By default, centrals are excluded from the radial profile because they form

# %% code cell 18

# ============================================================
# Radial-profile configuration and helpers
# ============================================================
# The radial profile uses FoF membership and is NOT restricted to R200c.
# It is plotted as a function of q = r_to_host / R200c.

RADIAL_R_OVER_R200C_RANGE = (1e-2, 5.0)
RADIAL_NBINS = 24
RADIAL_BINS = np.logspace(
    np.log10(RADIAL_R_OVER_R200C_RANGE[0]),
    np.log10(RADIAL_R_OVER_R200C_RANGE[1]),
    RADIAL_NBINS + 1,
)

# Host-halo mass bins used to stack radial profiles.
# Edit this list if you want finer/coarser mass conditioning.
RADIAL_HOST_MASS_BINS = [
    (1e11, 1e12),
    (1e12, 1e13),
    (1e13, 1e14),
    (1e14, 1e15),
]

# By default, do not include centrals in the radial profile.
# The central contribution is a near-delta at r/R200c=0 and is better handled
# through <Ncen|M> in the HOD panel above.
RADIAL_INCLUDE_CENTRALS = False

# What to plot:
#   "mean_shell_count" : <N_gal in radial shell | Mbin>
#   "number_density"   : <N_shell>/Delta[(r/R200c)^3 volume]
#   "cumulative"       : <N_gal(<r) | Mbin>
RADIAL_PROFILE_Y = "mean_shell_count"

radial_mass_colors = [clist[0], clist[3], clist[2], clist[4], clist[7], clist[9]]


def _radial_mass_bin_label(lo, hi):
    return rf"$10^{{{np.log10(lo):.0f}}}$--$10^{{{np.log10(hi):.0f}}}$"


def compute_radial_profile_from_sim(
    sim,
    selector,
    *,
    radial_bins=RADIAL_BINS,
    host_mass_bins=RADIAL_HOST_MASS_BINS,
    include_centrals=RADIAL_INCLUDE_CENTRALS,
):
    """
    Compute LRG/ELG radial profiles around host halos.

    The calculation uses FoF membership and does not impose r <= R200c.
    The x-axis is q = r_to_host / R200c, so selected galaxies can contribute
    to q > 1 if their FoF host assignment places them outside R200c.

    Returns
    -------
    profiles : dict
        One entry per host-mass bin. Each profile contains:
        - x: radial-bin centers in r/R200c
        - mean_shell_count: <N in shell | host mass bin>
        - number_density: mean_shell_count divided by dimensionless shell volume
        - cumulative: <N(<r) | host mass bin>
        - n_halo: number of host halos in the host-mass bin
        - n_gal: number of selected galaxies contributing to the radial profile
    """
    m200c = np.asarray(sim["M200c"], dtype=float)
    nhalo = int(sim["Ng"])
    gid = np.asarray(sim["gid"], dtype=np.int64)
    r_to_host = np.asarray(sim["r_to_host"], dtype=float)
    host_r200c = np.asarray(sim["host_r200c"], dtype=float)
    mstar_msunh = np.asarray(sim["mstar_msunh"], dtype=float)
    mstar_msun = np.asarray(sim["mstar_msun"], dtype=float)
    sfr = np.asarray(sim["sfr"], dtype=float)
    is_cen = np.asarray(sim["is_cen"], dtype=bool)

    if not sim.get("has_r200c_geometry", False):
        raise RuntimeError(
            "Radial profiles require GroupPos, SubhaloPos, and Group_R_Crit200. "
            "The current catalog was loaded without complete R200c geometry."
        )

    selected = np.asarray(selector(mstar_msunh, sfr, mstar_msun, z=sim.get("z"), snap=sim.get("snap")), dtype=bool)
    valid_gid = (gid >= 0) & (gid < nhalo)
    q = np.full(len(gid), np.nan, dtype=float)
    good_q = valid_gid & np.isfinite(r_to_host) & np.isfinite(host_r200c) & (host_r200c > 0)
    q[good_q] = r_to_host[good_q] / host_r200c[good_q]

    # Use FoF membership only; deliberately do not require q <= 1.
    valid_gal = selected & good_q & np.isfinite(q) & (q >= 0)
    if not include_centrals:
        valid_gal &= ~is_cen

    radial_bins = np.asarray(radial_bins, dtype=float)
    x = np.sqrt(radial_bins[:-1] * radial_bins[1:])
    shell_volume = (4.0 / 3.0) * np.pi * (radial_bins[1:]**3 - radial_bins[:-1]**3)

    profiles = {}
    for lo, hi in host_mass_bins:
        halo_mask = np.isfinite(m200c) & (m200c >= lo) & (m200c < hi)
        n_halo = int(np.sum(halo_mask))
        if n_halo == 0:
            counts = np.zeros(len(x), dtype=float)
            mean_shell = np.full(len(x), np.nan, dtype=float)
        else:
            gal_mask = valid_gal & halo_mask[gid]
            counts, _ = np.histogram(q[gal_mask], bins=radial_bins)
            counts = counts.astype(float)
            mean_shell = counts / n_halo

        number_density = mean_shell / shell_volume
        cumulative = np.cumsum(np.nan_to_num(mean_shell, nan=0.0))

        key = (float(lo), float(hi))
        profiles[key] = {
            "x": x,
            "r_left": radial_bins[:-1],
            "r_right": radial_bins[1:],
            "counts": counts,
            "mean_shell_count": mean_shell,
            "number_density": number_density,
            "cumulative": cumulative,
            "n_halo": n_halo,
            "n_gal": int(np.sum(valid_gal & halo_mask[gid])) if n_halo > 0 else 0,
            "mass_bin": key,
            "mass_label": _radial_mass_bin_label(lo, hi),
            "include_centrals": bool(include_centrals),
        }

    return profiles


def build_radial_profiles_for_snapshot(labels, snap, analysis_specs, loader_func):
    """
    Build LRG/ELG radial profiles for one snapshot.

    Parameters
    ----------
    labels : list[str]
        Simulation labels, e.g. flags for ClusterSims or run labels for TNG.
    loader_func : callable
        Function with signature loader_func(label, snap) -> sim dictionary.
    """
    radial_profiles = {}
    summary_rows = []

    for spec in analysis_specs:
        sname = spec["name"]
        selector = spec["selector"]
        radial_profiles[sname] = {}

        for label in labels:
            sim = loader_func(label, snap)
            prof = compute_radial_profile_from_sim(
                sim,
                selector,
                radial_bins=RADIAL_BINS,
                host_mass_bins=RADIAL_HOST_MASS_BINS,
                include_centrals=RADIAL_INCLUDE_CENTRALS,
            )
            radial_profiles[sname][label] = prof

            total_ngal = int(sum(p["n_gal"] for p in prof.values()))
            total_nhalo = int(sum(p["n_halo"] for p in prof.values()))
            summary_rows.append({
                "sample": sname,
                "label": label,
                "snap": int(snap),
                "include_centrals": RADIAL_INCLUDE_CENTRALS,
                "profile_uses_FoF_members_not_R200c_cut": True,
                "radial_range_r_over_R200c": RADIAL_R_OVER_R200C_RANGE,
                "total_profile_galaxies_in_mass_bins": total_ngal,
                "total_halos_in_mass_bins": total_nhalo,
                "has_r200c_geometry": sim.get("has_r200c_geometry", False),
            })

    return radial_profiles, summary_rows


def _smooth_profile_x(x, y, *, ylog=True, ngrid=250):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if ylog:
        good &= (y > 0)
    if np.sum(good) < 3:
        return x[good], y[good]

    lx = np.log10(x[good])
    yy = np.log10(y[good]) if ylog else y[good]
    order = np.argsort(lx)
    lx = lx[order]
    yy = yy[order]
    grid = np.linspace(lx.min(), lx.max(), ngrid)

    if HAS_SCIPY_PCHIP:
        interp = PchipInterpolator(lx, yy, extrapolate=False)
        out = interp(grid)
    else:
        out = np.interp(grid, lx, yy)

    return 10**grid, (10**out if ylog else np.clip(out, 0, None))


def plot_radial_profile_grid(
    radial_profiles,
    analysis_specs,
    *,
    labels,
    snap=None,
    zmap=None,
    title_prefix="LRG / ELG radial profiles",
    figsize=None,
    ykey=RADIAL_PROFILE_Y,
    ylog=True,
    smooth=True,
):
    """Plot stacked radial profiles in a sample x label grid."""
    nrows = len(analysis_specs)
    ncols = len(labels)
    if figsize is None:
        figsize = (max(4.2 * ncols, 6.0), 6.8)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes = np.asarray(axes)
    if nrows == 1 and ncols == 1:
        axes = axes.reshape(1, 1)
    elif nrows == 1:
        axes = axes.reshape(1, ncols)
    elif ncols == 1:
        axes = axes.reshape(nrows, 1)
    else:
        axes = axes.reshape(nrows, ncols)

    for i, spec in enumerate(analysis_specs):
        sname = spec["name"]
        for j, label in enumerate(labels):
            ax = axes[i, j]
            profiles = radial_profiles[sname][label]

            for k, ((lo, hi), prof) in enumerate(profiles.items()):
                x = prof["x"]
                y = prof[ykey]
                color = radial_mass_colors[k % len(radial_mass_colors)]
                plot_y = np.where(y > 0, y, np.nan) if ylog else y

                # Binned points
                ax.plot(
                    x,
                    plot_y,
                    marker="o",
                    markersize=3,
                    linestyle="None",
                    color=color,
                    alpha=0.45,
                )

                # Smooth curve for readability
                if smooth:
                    xs, ys = _smooth_profile_x(x, y, ylog=ylog, ngrid=250)
                else:
                    xs, ys = x, plot_y

                if len(xs) > 0:
                    ax.plot(
                        xs,
                        ys,
                        color=color,
                        lw=2.0,
                        label=prof["mass_label"] + rf"; $N_h={prof['n_halo']}$",
                    )

            ax.axvline(1.0, color="k", lw=1.0, ls="--", alpha=0.35)
            ax.text(
                1.03,
                0.93,
                r"$R_{200c}$",
                transform=ax.get_xaxis_transform(),
                fontsize=9,
                alpha=0.65,
            )
            ax.set_xscale("log")
            if ylog:
                ax.set_yscale("log")
            ax.set_xlim(RADIAL_R_OVER_R200C_RANGE)
            ax.grid(True, alpha=0.25)

            if i == 0:
                ax.set_title(str(label), fontsize=12)
            if j == 0:
                if ykey == "mean_shell_count":
                    ylabel = r"$\langle N_{\rm shell}\mid M_{200c}\rangle$"
                elif ykey == "number_density":
                    ylabel = r"$\langle N_{\rm shell}\rangle/\Delta V[(r/R_{200c})^3]$"
                elif ykey == "cumulative":
                    ylabel = r"$\langle N(<r)\mid M_{200c}\rangle$"
                else:
                    ylabel = ykey
                ax.set_ylabel(spec["name"] + "\n" + ylabel, fontsize=10)

    for j in range(ncols):
        axes[-1, j].set_xlabel(r"$r/R_{200c}$", fontsize=11)

    handles, labels_legend = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels_legend,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=min(len(handles), 4),
            frameon=False,
            title=r"Host-halo mass bins $[M_\odot/h]$",
        )

    stitle = title_prefix
    if snap is not None:
        if zmap is not None and snap in zmap:
            stitle += f" at snap={snap:03d} (z={zmap[snap]:.2f})"
        else:
            stitle += f" at snap={snap:03d}"
    stitle += r"; FoF members, no $r<R_{200c}$ cut"
    if not RADIAL_INCLUDE_CENTRALS:
        stitle += "; satellites only"
    fig.suptitle(stitle, y=1.04, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig, axes

print("Radial-profile helpers loaded.")
print("Profiles use FoF membership and can extend beyond r/R200c=1.")
print("RADIAL_INCLUDE_CENTRALS =", RADIAL_INCLUDE_CENTRALS)

# %% [markdown] cell 19
# ## ClusterSims radial-profile example This cell plots the LRG / ELG radial profiles for the ClusterSims flags at one snapshot. The radial profile is computed for selected galaxies in FoF halos and is not restricted to $R_{200c}$.

# %% code cell 20
# IPython-only: %%time
# ============================================================
# ClusterSims radial-profile single-snapshot example
# ============================================================
radial_snap = 21
analysis_specs = make_analysis_specs()

radial_profiles, radial_summary_rows, curve_source = get_or_build_radial_profiles(
    flags,
    radial_snap,
    analysis_specs,
    build_func=lambda: build_radial_profiles_for_snapshot(
        flags,
        radial_snap,
        analysis_specs,
        load_sim_direct,
    ),
    dataset_label="ClusterSims",
    zmap=zmap,
    ykey=RADIAL_PROFILE_Y,
    ylog=True,
    smooth=True,
)

if VERBOSE:
    print(f"Curve source: {curve_source}")
    print("ClusterSims radial-profile summary:")
    for row in radial_summary_rows:
        print(row)

fig, axes = plot_radial_profile_grid(
    radial_profiles,
    analysis_specs,
    labels=flags,
    snap=radial_snap,
    zmap=zmap,
    title_prefix="ClusterSims LRG / ELG radial profiles",
    figsize=(22, 7),
    ykey=RADIAL_PROFILE_Y,
    ylog=True,
    smooth=True,
)
plt.show()

# %% [markdown] cell 21
# ## ClusterSims radial-profile loop over all snapshots Run this cell to save the radial-profile figures for all snapshots in `snaps`.

# %% code cell 22
# %%time
# ============================================================
# ClusterSims radial-profile loop over all snapshots
# ============================================================
radial_outdir = Path("./plots_radial_profiles")
radial_outdir.mkdir(parents=True, exist_ok=True)

analysis_specs = make_analysis_specs()

for radial_snap in snaps:
    radial_profiles, radial_summary_rows, curve_source = get_or_build_radial_profiles(
        flags,
        radial_snap,
        analysis_specs,
        build_func=lambda radial_snap=radial_snap: build_radial_profiles_for_snapshot(
            flags,
            radial_snap,
            analysis_specs,
            load_sim_direct,
        ),
        dataset_label="ClusterSims",
        zmap=zmap,
        ykey=RADIAL_PROFILE_Y,
        ylog=True,
        smooth=True,
    )

    if VERBOSE:
        print(f"\n=== ClusterSims radial profiles: snap={radial_snap:03d}, z={zmap.get(radial_snap, np.nan):.2f}, source={curve_source} ===")
        for row in radial_summary_rows:
            print(row)

    fig, axes = plot_radial_profile_grid(
        radial_profiles,
        analysis_specs,
        labels=flags,
        snap=radial_snap,
        zmap=zmap,
        title_prefix="ClusterSims LRG / ELG radial profiles",
        figsize=(22, 7),
        ykey=RADIAL_PROFILE_Y,
        ylog=True,
        smooth=True,
    )

    fig.savefig(
        radial_outdir / f"RadialProfile_LRG_ELG_FoF_noR200cCut_snap{radial_snap:03d}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()
    # plt.close(fig)

# %% [markdown] cell 23
# ## TNG radial-profile example This uses `TNGCatalog` with your TNG path ```python /cosma8/data/dp203/dc-wang17/TNG/tng_data ``` and applies the same LRG / ELG proxy cuts and radial-profile estimator.

# %% code cell 24
# IPython-only: %%time
# ============================================================
# TNG radial-profile single-snapshot example
# ============================================================
tng_radial_snap = TNG_SNAPS[0]
tng_labels = list(TNG_RUNS.keys())
analysis_specs = make_analysis_specs()

tng_radial_profiles, tng_radial_summary_rows, curve_source = get_or_build_radial_profiles(
    tng_labels,
    tng_radial_snap,
    analysis_specs,
    build_func=lambda: build_radial_profiles_for_snapshot(
        tng_labels,
        tng_radial_snap,
        analysis_specs,
        load_tng_direct,
    ),
    dataset_label="TNG",
    zmap=TNG_ZMAP,
    ykey=RADIAL_PROFILE_Y,
    ylog=True,
    smooth=True,
)

if VERBOSE:
    print(f"Curve source: {curve_source}")
    print("TNG radial-profile summary:")
    for row in tng_radial_summary_rows:
        print(row)

fig, axes = plot_radial_profile_grid(
    tng_radial_profiles,
    analysis_specs,
    labels=tng_labels,
    snap=tng_radial_snap,
    zmap=TNG_ZMAP,
    title_prefix="TNG LRG / ELG radial profiles",
    figsize=(max(5.2 * len(tng_labels), 6.0), 7),
    ykey=RADIAL_PROFILE_Y,
    ylog=True,
    smooth=True,
)
plt.show()

# %% [markdown] cell 25
# ## TNG radial-profile loop over selected snapshots Run this cell if you add more entries to `TNG_SNAPS`.

# %% code cell 26
# %%time
# ============================================================
# TNG radial-profile loop over selected snapshots
# ============================================================
tng_radial_outdir = Path("./plots_tng_radial_profiles")
tng_radial_outdir.mkdir(parents=True, exist_ok=True)

tng_labels = list(TNG_RUNS.keys())
analysis_specs = make_analysis_specs()

for tng_radial_snap in TNG_SNAPS:
    tng_radial_profiles, tng_radial_summary_rows, curve_source = get_or_build_radial_profiles(
        tng_labels,
        tng_radial_snap,
        analysis_specs,
        build_func=lambda tng_radial_snap=tng_radial_snap: build_radial_profiles_for_snapshot(
            tng_labels,
            tng_radial_snap,
            analysis_specs,
            load_tng_direct,
        ),
        dataset_label="TNG",
        zmap=TNG_ZMAP,
        ykey=RADIAL_PROFILE_Y,
        ylog=True,
        smooth=True,
    )

    if VERBOSE:
        print(f"\n=== TNG radial profiles: snap={tng_radial_snap:03d}, z={TNG_ZMAP.get(tng_radial_snap, np.nan):.2f}, source={curve_source} ===")
        for row in tng_radial_summary_rows:
            print(row)

    fig, axes = plot_radial_profile_grid(
        tng_radial_profiles,
        analysis_specs,
        labels=tng_labels,
        snap=tng_radial_snap,
        zmap=TNG_ZMAP,
        title_prefix="TNG LRG / ELG radial profiles",
        figsize=(max(5.2 * len(tng_labels), 6.0), 7),
        ykey=RADIAL_PROFILE_Y,
        ylog=True,
        smooth=True,
    )

    fig.savefig(
        tng_radial_outdir / f"TNG_RadialProfile_LRG_ELG_FoF_noR200cCut_snap{tng_radial_snap:03d}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()
    # plt.close(fig)

# %% code cell 27

# %% code cell 28
