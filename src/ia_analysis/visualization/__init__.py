"""Visualization namespace for plotting, animation, and figure utilities.

Purpose
-------
The visualization package is split into lightweight facade modules with clear
responsibilities, while preserving access to older implementation modules used
by notebooks.

Provides
--------
- Color, projection, shell-plot, alignment, orbit-animation, and distribution
  helper modules.
- Lazy facade imports so package import does not immediately require plotting
  libraries.

Notes
-----
Prefer the structured facade modules for new code.  Historical modules such as
``arts`` and ``arts_IA`` remain available for compatibility inside ``src``.
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
