"""Deprecated mesh-building namespace.

New code should import catalog and snapshot mesh utilities from
``ia_analysis.spectra``.  This package remains as a compatibility layer for
older scripts and notebooks.
"""

__all__ = ["CatMesh", "SnapMesh"]
