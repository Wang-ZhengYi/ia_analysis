"""Compatibility CLI wrapper for :mod:`ia_analysis.covariance.Cov`."""

from _compat import ensure_src_path

ensure_src_path()

from ia_analysis.covariance.Cov import *  # noqa: F401,F403,E402

if __name__ == "__main__":
    from ia_analysis.covariance.Cov import main

    main()

