"""Exported code from notebooks/raw_20260618/global_test.ipynb.

This file is generated for project management and refactoring. Review before running end to end.
"""


# %% code cell 1
import sys
import numpy as np

sys.path.insert(0, "/cosma/home/dp203/dc-wang17/IA_analysis/anaIA")

from catalog_loader import CSCatalog
from run_cs import _default_cfg
from global_cs import compute_many as cm_cs

# %% code cell 2
base = "/cosma8/data/dp203/bl267/Data/ClusterSims/L302_N1136_GR"
snap = 21
cfg = _default_cfg()

cat_cs = CSCatalog(base, snap)
halos_cs , subs_cs  = cat_cs .loadFoF(
    group_fields=["GroupFirstSub", "GroupNsubs", "GroupLenType"],
    subhalo_fields=["SubhaloLenType", "SubhaloPos", "SubhaloVel"],
)

# %% code cell 3
sid = [0]
gid = [0]

# %% code cell 4
res_cs  = cm_cs(base, snap, sid, gid, cfg)

# %% code cell 5
for r in res_cs:
    print(
        r["Sub_info"]["SubhaloID"],
        r["Shape"]["dm"]["converged"],
        np.isfinite(r["Shape"]["dm"]["I"]).all(),
        np.isfinite(r["Tidal"]["tidal_grp"]).all(),
    )

# %% code cell 6
res_cs

# %% code cell 7
