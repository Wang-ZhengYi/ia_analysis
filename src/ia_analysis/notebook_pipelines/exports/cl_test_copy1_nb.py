"""Exported code from notebooks/raw_20260618/CL_test-Copy1.ipynb.

This file is generated for project management and refactoring. Review before running end to end.
"""


# %% code cell 1

# %% code cell 2

# %% code cell 3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import h5py
# import .Mesh
# import powers

from catalog_loader import CSCatalog

# %% code cell 4
from Iana import *

# %% code cell 5

from shape import ShapeKin
# import arts  # contains plot_3d_scatter

# %% code cell 6
# choose which backend
kind = "tng"   # "cs" or "tng"
snap = 1
sid  = 1

# paths
base_cs  = "/cosma8/data/dp203/bl267/Data/ClusterSims/L302_N1136_GR"
base_tng = "/cosma8/data/dp203/dc-wang17/TNG/tng_data"
# Fixed DM masses (no Masses field for DM in CS; for TNG we still use fixed mass)
TNG_DM_FIXED_MASS = 5.9e7
CS_DM_FIXED_MASS = 1.35401e9
base = base_cs if kind == "cs" else base_tng

# %% code cell 7
# IPython-only: %%time
cat = CSCatalog(base_cs, 21)

# %% code cell 8
group_fields = ["GroupMass", "GroupPos", "GroupFirstSub", "GroupNsubs", "GroupLenType"]
subhalo_fields = ["SubhaloMass", "SubhaloPos", "SubhaloLenType"]

# %% code cell 9
# IPython-only: %%time
halos, subs = cat.loadFoF(group_fields=group_fields, subhalo_fields=subhalo_fields)

# %% code cell 10
print("Loaded halos keys:", list(halos.keys())[:10], "...")
print("Loaded subs keys :", list(subs.keys())[:10], "...")
print("Ngroups =", halos["GroupFirstSub"].shape[0])
print("Nsub    =", subs["SubhaloID"].shape[0])

# Check injected arrays exist and lengths match
for k in ["SubhaloID", "GroupID", "CenID"]:
    assert k in subs, f"Missing injected field: {k}"
    assert subs[k].shape[0] == subs["SubhaloID"].shape[0], f"Length mismatch for {k}"
print("✅ Injected arrays SubhaloID/GroupID/CenID OK")

# Basic sanity: GroupID range
gid_min, gid_max = subs["GroupID"].min(), subs["GroupID"].max()
print("GroupID range in subs:", gid_min, gid_max)

# %% code cell 11
# IPython-only: %%time
gids_with_sub = np.where(halos["GroupNsubs"] > 0)[0]
assert gids_with_sub.size > 0, "No groups with subhalos found?!"
gid = int(gids_with_sub[0])

lo = int(halos["GroupFirstSub"][gid])
hi = lo + int(halos["GroupNsubs"][gid])

print(f"Test group gid={gid}: lo={lo}, hi={hi}, nsubs={hi-lo}")

# slice SubhaloMass (if present)
if "SubhaloMass" in subs:
    m_slice = subs["SubhaloMass"][lo:hi]
    print("SubhaloMass slice shape:", m_slice.shape)
else:
    print("⚠️ SubhaloMass not loaded; skip mass-slice check")

# slice GroupID and check all equal gid
gid_slice = subs["GroupID"][lo:hi]
assert np.all(gid_slice == gid), "GroupID slice is not constant == gid (contiguous assumption broken?)"
print("✅ GroupID slice constant -> contiguous subhalo layout verified for this gid")

# check CenID slice is constant == GroupFirstSub[gid]
cen_expected = lo
cen_slice = subs["CenID"][lo:hi]
assert np.all(cen_slice == cen_expected), "CenID slice not constant == GroupFirstSub[gid]"
print("✅ CenID slice constant == GroupFirstSub[gid]")

# %% code cell 12
# IPython-only: %%time
# pick a subhalo id inside this group (central if you like)
sid = lo  # central subhalo of the selected group
print(f"Test sid={sid} (central of gid={gid})")

# Choose a partType to test; typical DM is 1 in TNG-like layout
ptypes = [1]
fields = ["Coordinates", "Velocities"]  #

parts_sub = cat.loadSubhalos(sid=sid, ptypes=ptypes, fields=fields)
print("loadSubhalo returned keys:", parts_sub.keys())

# %% code cell 13
# IPython-only: %%time
# Verify particle count matches SubhaloLenType
n_expected_sub = int(subs["SubhaloLenType"][sid, 1])
n_got_sub = int(parts_sub["PartType1"]["Coordinates"].shape[0])
print("Subhalo PartType1 N_expected =", n_expected_sub, "N_got =", n_got_sub)
assert n_expected_sub == n_got_sub, "Particle count mismatch for loadSubhalo!"
print("✅ loadSubhalo particle count matches SubhaloLenType")

# %% code cell 14
# IPython-only: %%time
# loadHalo by gid
parts_halo_gid = cat.loadHalos(gid=gid, ptypes=ptypes, fields=fields)
n_expected_halo = int(halos["GroupLenType"][gid, 1])
n_got_halo = int(parts_halo_gid["PartType1"]["Coordinates"].shape[0])
print("Halo(gid) PartType1 N_expected =", n_expected_halo, "N_got =", n_got_halo)
assert n_expected_halo == n_got_halo, "Particle count mismatch for loadHalo(gid)!"
print("✅ loadHalo(gid) particle count matches GroupLenType")

# %% code cell 15
# IPython-only: %%time
# loadHalo by sid (infer gid)
parts_halo_sid = cat.loadHalos(sid=sid, ptypes=ptypes, fields=fields)
n_got_halo2 = int(parts_halo_sid["PartType1"]["Coordinates"].shape[0])
print("Halo(sid->gid) PartType1 N_got =", n_got_halo2)
assert n_got_halo2 == n_expected_halo, "loadHalo(sid=...) result differs from loadHalo(gid=...)!"
print("✅ loadHalo(sid) agrees with loadHalo(gid)")

print("\nALL TESTS PASSED ✅")

# %% code cell 16
from global_cs import compute_many as compute_many_cs

# %% code cell 17

# %% code cell 18
cfg = dict(
    # boxsize in comoving Mpc/h
    boxsize=205.0,

    # fixed DM particle mass (same convention as your pipeline)
    dm_particle_mass=CS_DM_FIXED_MASS,   # <-- set your real DM particle mass here

    # stage1: copy extra subhalo catalog fields into Sub_info
    sub_fields_extra=[
    # baryons / feedback-ish
    "SubhaloSFR",
    "SubhaloGasMetallicity",
    "SubhaloMass",
    "SubhaloMassInRadType",
    "SubhaloVmax",
    "SubhaloWindMass",
    # common BH proxies (may not exist; will become NaN)
    "SubhaloBHMass",
    "SubhaloBHMdot",
],
    group_fields_extra=[
    "Group_M_Crit500",
    "Group_M_Crit200",
    "Group_R_Crit500",
    "Group_R_Crit200",
],
    # stage2: shape params
    dm_shape_percentile=98.0,
    star_shape_percentile=100,
    shape_max_iter=80,
    shape_tol=1e-2,
    star_aperture_factor=2,
    # stage3: tidal params
    tidal_grid_size=128,
    tidal_padding=0.2,
    tidal_periodic_wrap=False,
)

# %% code cell 19
# IPython-only: %%time
# in each worker:
sids=[0]
gids=[0]
results = compute_many_cs(base_cs, 21, sids, gids, cfg)

# %% code cell 20
277614*21.1/128/60/60

# %% code cell 21
results[0]

# %% code cell 22
I=np.array([[275.92125389,  34.45106385,  30.51854816],
          [ 34.45106385, 353.74085689,  -1.27789713],
          [ 30.51854816,  -1.27789713, 387.65500379]])

# %% code cell 23
chiSO(I)

# %% code cell 24
from global_tng import compute_many as compute_tng

# %% code cell 25
cfg_tng = dict(
    # boxsize in comoving Mpc/h
    boxsize=205.0,

    # fixed DM particle mass (same convention as your pipeline)
    dm_particle_mass=TNG_DM_FIXED_MASS,   # <-- set your real DM particle mass here

    # stage1: copy extra subhalo catalog fields into Sub_info
    sub_fields_extra=[
       "SubhaloSFR", "SubhaloGasMetallicity","SubhaloMass", "SubhaloHalfmassRadType"
    ],
    group_fields_extra=[
        "Group_M_Crit500",
    ],

    # stage2: shape params
    dm_shape_percentile=98.0,
    star_shape_percentile=100,
    shape_max_iter=80,
    shape_tol=1e-2,
    star_aperture_factor=2,
    # stage3: tidal params
    tidal_grid_size=128,
    tidal_padding=0.2,
)

# %% code cell 26
test_GR_21_200 = h5py.File('/cosma8/data/dp203/dc-wang17/MG_global/test_GR_s021.hdf5','r')

# %% code cell 27
test_GR_21_200.keys()

# %% code cell 28
print(test_GR_21_200['meta']['cfg_json'][...])

# %% code cell 29
test_GR_21_200['Star'].keys()

# %% code cell 30
test_GR_21_200['Star']['Neff'][:]

# %% code cell 31

# %% code cell 32

# %% code cell 33
# IPython-only: %%time
results_tng = compute_tng(base_tng, 99, sids, gids, cfg_tng)

# %% code cell 34
results_tng[0]

# %% code cell 35
