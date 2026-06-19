"""Compatibility wrapper for :mod:`ia_analysis.spectra.CatMesh`."""

from _compat import ensure_src_path

ensure_src_path()

from ia_analysis.spectra.CatMesh import *  # noqa: F401,F403,E402
