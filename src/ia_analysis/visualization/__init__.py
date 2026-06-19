"""Plotting, animation, and visualization helpers.

This namespace is split into lightweight facade modules:

- ``color_tools`` for palettes and color extraction.
- ``projection_geometry`` for principal-plane projections and scale bars.
- ``scene3d`` for 3D scatter and galaxy-system scenes.
- ``shell_plots`` for radial and binding shell panels.
- ``alignment_catalogs`` for MAset loading and population masks.
- ``alignment_metrics`` for vector/tensor alignment diagnostics.
- ``alignment_plots`` for alignment figures and paper suites.
- ``orbit_animation`` for orbit movies and previews.
- ``distribution_fits`` for probability models used by plots.
- ``parallel_alignment`` for parallel grid-generation CLIs.

Legacy modules such as ``arts`` and ``arts_IA`` remain importable.
"""

__all__ = [
    "alignment_catalogs",
    "alignment_metrics",
    "alignment_plots",
    "color_tools",
    "distribution_fits",
    "orbit_animation",
    "parallel_alignment",
    "projection_geometry",
    "scene3d",
    "shell_plots",
]
