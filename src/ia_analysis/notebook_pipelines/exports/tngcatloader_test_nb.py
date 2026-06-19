"""Exported code from notebooks/raw_20260618/TNGCatLoader_test.ipynb.

This file is generated for project management and refactoring. Review before running end to end.
"""


# %% [markdown] cell 1
# # TNGCatLoader partial-download test This notebook tests the new `TNGCatLoader.TNGCatalog` workflow. It prefers local TNG files. If local files are missing and `TNG_API_KEY` is available, it downloads only the needed group-catalog fields and halo/subhalo cutouts. Temporary downloads are deleted by default.

# %% code cell 2
import os
import sys
import numpy as np

# Make sure the local directory containing TNGCatLoader.py/global_tng.py is importable.
sys.path.insert(0, os.getcwd())

from TNGCatLoader import TNGCatalog
from global_tng import compute_many

# %% code cell 3
SNAP_LIST = [99, 84, 72, 67, 59, 50, 40, 33]
# python tng_downloader.py --snaps 99, 72, 67, 59, 50, 40, 33 --retry-base-sleep 3 --retry-max-sleep 30 --max-retries 20
SNAP_TO_Z = {
    99: 0.00,
    84: 0.20,
    72: 0.40,
    67: 0.50,
    59: 0.70,
    50: 1.00,
    40: 1.50,
    33: 2.00,
}

# %% code cell 4
# ---- User settings ----
BASE_TNG = os.environ.get('TNG_BASE_PATH', '/cosma8/data/dp203/dc-wang17/TNG/tng_data')
SNAP = int(os.environ.get('TNG_SNAP', '99'))
SIM_NAME = os.environ.get('TNG_SIM_NAME', 'TNG300-1')
API_KEY = os.environ.get('TNG_API_KEY','ec7a0419719cacfd0a27d964d8993b9d')

# Keep this small for a smoke test.
N_TEST = 2
MIN_STAR_PARTICLES = 150

tng_catalog_kwargs = dict(
    sim_name=SIM_NAME,
    api_key=API_KEY,
    download_if_missing=None,   # None => enabled automatically if API_KEY exists
    cache_dir=None,             # private temporary directory
    delete_cache=True,          # default: delete downloaded subset/cutout files after use
    verbose=True,               # default: show download progress
    prefer_cutout=True,
)

print('BASE_TNG =', BASE_TNG)
print('SNAP =', SNAP)
print('SIM_NAME =', SIM_NAME)
print('API key available =', bool(API_KEY))

# %% code cell 5
API_KEY

# %% code cell 6
GROUP_FIELDS = [
    'GroupFirstSub', 'GroupNsubs', 'GroupLenType',
    'Group_M_Crit500', 'Group_M_Crit200',
    'Group_R_Crit500', 'Group_R_Crit200',
]

SUBHALO_FIELDS = [
    'SubhaloPos', 'SubhaloVel', 'SubhaloLenType', 'SubhaloGrNr', 'SubhaloHalfmassRadType',
    'SubhaloSFR', 'SubhaloGasMetallicity', 'SubhaloMass', 'SubhaloMassInRadType',
    'SubhaloVmax', 'SubhaloWindMass', 'SubhaloBHMass', 'SubhaloBHMdot',
]

CFG_TNG = dict(
    dm_shape_percentile=98.0,
    star_shape_percentile=100.0,
    star_aperture_factor=2.0,
    shape_max_iter=100,
    shape_tol=1e-2,
    shape_tensor_mode='reduced',
    tidal_grid_size=32,          # smaller than production for a fast smoke test
    tidal_padding=0.20,
    tidal_softening=0.01,
    legacy_tidal_sign=True,
    dm_particle_mass=5.9e7,
    sub_fields_extra=[
        'SubhaloSFR', 'SubhaloGasMetallicity', 'SubhaloMass', 'SubhaloMassInRadType',
        'SubhaloVmax', 'SubhaloWindMass', 'SubhaloBHMass', 'SubhaloBHMdot',
    ],
    group_fields_extra=[
        'Group_M_Crit500', 'Group_M_Crit200', 'Group_R_Crit500', 'Group_R_Crit200',
    ],
)

# %% [markdown] cell 7
# ## 1. Test `TNGCatalog.loadFoF()` This will read local group catalogs if present. If they are absent and `TNG_API_KEY` is set, it downloads only the requested catalog fields.

# %% code cell 8
with TNGCatalog(BASE_TNG, SNAP, **tng_catalog_kwargs) as cat:
    halos, subs = cat.loadFoF(group_fields=GROUP_FIELDS, subhalo_fields=SUBHALO_FIELDS)

print('N groups    =', len(halos['GroupFirstSub']))
print('N subhalos =', len(subs['SubhaloID']))
print('Subhalo fields:', sorted([k for k in subs.keys() if not k.startswith('_')])[:20], '...')

# %% [markdown] cell 9
# ## 2. Select a tiny test sample

# %% code cell 10
slt = np.asarray(subs['SubhaloLenType'], dtype=np.int64)
nstar = slt[:, 4]
sid_sel = np.where(nstar >= MIN_STAR_PARTICLES)[0].astype(np.int64)[:N_TEST]
gid_sel = np.asarray(subs['GroupID'], dtype=np.int64)[sid_sel]

print('sid_sel =', sid_sel)
print('gid_sel =', gid_sel)
assert sid_sel.size > 0, 'No subhalos passed the selection. Lower MIN_STAR_PARTICLES.'

# %% [markdown] cell 11
# ## 3. Run `global_tng.compute_many()` If local `snapdir_XXX` is missing or incomplete, this uses halo/subhalo cutouts for only the requested particle fields. The temporary downloaded cutouts are deleted when each worker finishes.

# %% code cell 12
results = compute_many(
    BASE_TNG,
    SNAP,
    sid_sel,
    gid_sel,
    CFG_TNG,
    group_fields=GROUP_FIELDS,
    subhalo_fields=SUBHALO_FIELDS,
    tng_catalog_kwargs=tng_catalog_kwargs,
)

print('N results =', len(results))
for r in results:
    print('sid=', r['Sub_info']['SubhaloID'],
          'gid=', r['Sub_info']['GroupID'],
          'DM converged=', r['Shape']['dm']['converged'],
          'Star converged=', r['Shape']['stars']['converged'])

# %% [markdown] cell 13
# ## 4. Command-line production example The production driver uses the same loader. By default, downloaded API subset/cutout files are deleted. Add `--keep-cache` only if you want to inspect or reuse them.

# %% code cell 14
print('Example command:')
print('python run_tng.py --nworker 4 --out tng_shape_test.hdf5 --max-sub 10 --api-key $TNG_API_KEY')
print('Keep temporary API files: add --keep-cache')
print('Suppress progress: add --quiet-tng-download')

# %% code cell 15

# %% code cell 16
