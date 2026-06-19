"""Compatibility CLI wrapper for :mod:`ia_analysis.visualization.plot_GH_alignment_parallel`."""

from _compat import ensure_src_path

ensure_src_path()

import runpy
import sys


def _fallback_help() -> None:
    print("usage: plot_GH_alignment_parallel.py [options]")
    print()
    print("Parallel galaxy-halo alignment plotting pipeline.")
    print()
    print("Use --help in an environment with matplotlib, seaborn, scipy, and tqdm installed for the full argument list.")


if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    try:
        runpy.run_module("ia_analysis.visualization.plot_GH_alignment_parallel", run_name="__main__")
    except ModuleNotFoundError:
        _fallback_help()
    raise SystemExit(0)

from ia_analysis.visualization.plot_GH_alignment_parallel import *  # noqa: F401,F403,E402

if __name__ == "__main__":
    from ia_analysis.visualization.plot_GH_alignment_parallel import main

    main()

