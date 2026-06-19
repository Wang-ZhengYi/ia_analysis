"""Covariance model namespace for IA power-spectrum products.

Purpose
-------
The covariance package groups Gaussian, connected non-Gaussian, and
super-sample covariance tools used after spectra have been measured.

Provides
--------
- File-oriented covariance entrypoints for HDF5 power-spectrum products.
- Array-level helpers for covariance assembly and noise bookkeeping.

Notes
-----
Covariance modules consume spectra outputs and should not orchestrate catalog or
mesh-generation pipelines directly.
"""


