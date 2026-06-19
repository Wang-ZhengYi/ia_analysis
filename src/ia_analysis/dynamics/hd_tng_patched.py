"""Compatibility wrapper for the canonical TNG dynamics implementation.

Purpose
-------
Historically this file carried a patched copy of ``hd_tng.py``.  The patched
copy matched the canonical implementation byte-for-byte in the structured
package, so keeping two full files would make future maintenance error-prone.

Provides
--------
- Backward-compatible imports from ``ia_analysis.dynamics.hd_tng_patched``.
- Direct forwarding of public names to ``ia_analysis.dynamics.hd_tng``.
- Attribute forwarding for private helpers used by older notebooks.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "DEFAULT_CFG",
    "KM_S_PER_KPC_TO_GYR_INV",
    "cleanup_open_catalogs",
    "read_header_for_snap",
    "compute_haloes",
    "compute_many",
    "compute_one_subhalo",
    "select_subhaloes_in_top_groups",
    "load_subhalo_dm_particles",
    "analyse_particle_data",
    "closure_table_from_analysis",
    "enrich_run_with_group_metadata",
    "direct_pi_from_P",
    "cross_time_pattern_speed_for_subhalo",
    "load_sublink_mpb",
]


def _implementation_module():
    """Import the canonical TNG dynamics module only when a name is requested."""
    return import_module("ia_analysis.dynamics.hd_tng")


def __getattr__(name: str):
    """Forward any compatibility attribute lookup to ``hd_tng``."""
    return getattr(_implementation_module(), name)


def __dir__() -> list[str]:
    """Return the forwarded implementation namespace for interactive use."""
    return sorted(set(globals()) | set(__all__))
