"""Compatibility CLI wrapper for :mod:$module."""

from _compat import ensure_src_path

ensure_src_path()

import runpy
import sys


def _fallback_help() -> None:
    print("usage: pk_batch_self_folding.py [options]")
    print()
    print("Batch self-folding power-spectrum pipeline")
    print()
    print("Use --help in an environment with the project scientific dependencies installed for the full argument list.")


if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    try:
        runpy.run_module("ia_analysis.spectra.pk_batch_self_folding", run_name="__main__")
    except ModuleNotFoundError:
        _fallback_help()
    raise SystemExit(0)

from ia_analysis.spectra.pk_batch_self_folding import *  # noqa: F401,F403,E402

if __name__ == "__main__":
    from ia_analysis.spectra.pk_batch_self_folding import main

    main()