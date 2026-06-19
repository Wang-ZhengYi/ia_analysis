"""Numerical smoke tests for shape-tensor utilities.

Purpose
-------
These tests use synthetic particle distributions to verify that the shape module
returns finite tensors, axes, and axis-ratio outputs.

Provides
--------
- Basic coverage for inertia tensor measurement.
- Sanity checks for axis extraction and shape output structure.
"""

import numpy as np

from ia_analysis.shapes.shape import I_iters, compute_axis


def test_shape_tensor_axes_from_synthetic_cloud():
    x = np.linspace(-2.0, 2.0, 7)
    y = np.linspace(-1.0, 1.0, 5)
    z = np.linspace(-0.5, 0.5, 3)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    particles = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    I = I_iters(particles, percentile=100.0, max_iter=5, tol=1e-8)
    axes, vecs = compute_axis(I)

    assert I.shape == (3, 3)
    assert set(axes) == {"a", "b", "c"}
    assert set(vecs) == {"e1", "e2", "e3"}
    assert np.all(np.isfinite(I))
    assert np.all(np.isfinite([axes["a"], axes["b"], axes["c"]]))
    assert all(np.asarray(vecs[k]).shape == (3,) for k in ("e1", "e2", "e3"))
    assert axes["a"] >= axes["b"] >= axes["c"]
