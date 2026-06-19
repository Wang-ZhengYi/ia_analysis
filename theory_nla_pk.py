"""Compatibility CLI wrapper for :mod:ia_analysis.spectra.theory_nla_pk."""

from _compat import ensure_src_path

ensure_src_path()

import runpy

from ia_analysis.spectra.theory_nla_pk import *  # noqa: F401,F403,E402

if __name__ == "__main__":
    runpy.run_module("ia_analysis.spectra.theory_nla_pk", run_name="__main__")