"""Alignment metric facade for vectors, tensors, and scalar diagnostics.

Purpose
-------
This module groups tensor-axis extraction, vector normalization, angular
alignment, tidal-tensor access, and mass/radius diagnostics under one import
location.

Provides
--------
- Shape-shape, vector-shape, and vector-vector alignment metrics.
- Lazy access to stellar, dark-matter, tidal, radial, velocity, and spin fields.
- Scalar diagnostics such as stellar mass, halo mass, radius, and baryon ratio.
"""


from __future__ import annotations

from typing import Any

from ia_analysis.visualization._lazy import load_export

_EXPORTS = {
    "tensor_axes": ("ia_analysis.visualization.arts_IA", "tensor_axes"),
    "axis_vector": ("ia_analysis.visualization.arts_IA", "axis_vector"),
    "abs_cos_vectors": ("ia_analysis.visualization.arts_IA", "abs_cos_vectors"),
    "tensor_tensor_alignment": ("ia_analysis.visualization.arts_IA", "tensor_tensor_alignment"),
    "vector_tensor_alignment": ("ia_analysis.visualization.arts_IA", "vector_tensor_alignment"),
    "vector_vector_alignment": ("ia_analysis.visualization.arts_IA", "vector_vector_alignment"),
    "I_star": ("ia_analysis.visualization.arts_IA", "I_star"),
    "I_dm": ("ia_analysis.visualization.arts_IA", "I_dm"),
    "tidal_tensor": ("ia_analysis.visualization.arts_IA", "tidal_tensor"),
    "R_vec": ("ia_analysis.visualization.arts_IA", "R_vec"),
    "V_vec": ("ia_analysis.visualization.arts_IA", "V_vec"),
    "omega_star_vec": ("ia_analysis.visualization.arts_IA", "omega_star_vec"),
    "omega_dm_vec": ("ia_analysis.visualization.arts_IA", "omega_dm_vec"),
    "log_stellar_mass": ("ia_analysis.visualization.arts_IA", "log_stellar_mass"),
    "log_subhalo_mass": ("ia_analysis.visualization.arts_IA", "log_subhalo_mass"),
    "log_halo_mass": ("ia_analysis.visualization.arts_IA", "log_halo_mass"),
    "r_over_r200c": ("ia_analysis.visualization.arts_IA", "r_over_r200c"),
    "baryon_dm_ratio": ("ia_analysis.visualization.arts_IA", "baryon_dm_ratio"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    return load_export(_EXPORTS, name)

