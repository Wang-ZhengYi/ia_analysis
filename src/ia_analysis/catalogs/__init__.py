"""Catalog and particle data loading namespace.

Purpose
-------
The catalogs package contains readers for ClusterSims and IllustrisTNG-style
FoF, subhalo, and particle data products.

Provides
--------
- HDF5 chunk discovery and numeric sorting utilities.
- ClusterSims catalog access helpers.
- TNG catalog wrappers that keep API/cache details outside science modules.

Notes
-----
This package is intentionally low level.  It should not import shapes, spectra,
or pipeline orchestration code.
"""


