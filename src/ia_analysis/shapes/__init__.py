"""Shape and alignment namespace.

Purpose
-------
The shapes package contains tensor, axis, kinematic, spin, and IA projection
utilities that operate on arrays and do not orchestrate full pipelines.

Provides
--------
- Iterative inertia-tensor and principal-axis measurements.
- Figure-rotation and kinematic tensor helpers.
- IA ellipticity/projection helpers used by mesh construction and spectra.
"""


from .shape import ShapeKin, I_iters, compute_axis

__all__ = ["ShapeKin", "I_iters", "compute_axis"]

