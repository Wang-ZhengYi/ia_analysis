"""Structured Python package for intrinsic-alignment analysis workflows.

Purpose
-------
This top-level package exposes only lightweight metadata so importing
``ia_analysis`` does not require optional HPC or plotting dependencies.

Provides
--------
- The canonical package namespace for catalogs, shapes, tides, dynamics,
  merger trees, correlations, spectra, covariance, pipelines, orbits, and
  visualization.
- Version metadata for downstream scripts and notebooks.

Notes
-----
Heavy dependencies such as Pylians, pyccl, illustris_python, halotools, and
matplotlib are imported only by the submodules that actually need them.
"""


__version__ = "0.1.0"

__all__ = ["__version__"]

