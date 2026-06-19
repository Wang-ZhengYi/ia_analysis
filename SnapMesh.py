"""Compatibility wrapper for :mod:`ia_analysis.spectra.SnapMesh`."""

from _compat import ensure_src_path

ensure_src_path()

from ia_analysis.spectra.SnapMesh import *  # noqa: F401,F403,E402
