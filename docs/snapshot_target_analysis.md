# Snapshot Target Analysis

Shape and halo dynamics are both measurements on a target object in one
snapshot.  New code should use the modular dynamics entrypoints below instead
of importing the large historical implementation file directly.

## Modules

- `ia_analysis.dynamics.shape_measurements`: iterative shape tensor, axes, and
  optional `dI`/`ddI` measurements.
- `ia_analysis.dynamics.kinematics`: affine velocity gradient, spin,
  `H`/`Omega`, and direct `dI`-based figure rotation.
- `ia_analysis.dynamics.dynamics_measurements`: binding energy, shell dynamics,
  tidal response, and multi-component binding profiles.
- `ia_analysis.dynamics.matrix_analysis`: pure tensor algebra and tidal Hessian
  convention helpers.
- `ia_analysis.dynamics.snapshot_analysis`: one integrated wrapper for a
  target component in one snapshot.

## Integrated Use

```python
from ia_analysis.dynamics import SnapshotTarget, analyze_snapshot_target

target = SnapshotTarget(
    positions=pos,
    velocities=vel,
    masses=mass,
    center=center,
    v_ref=v_ref,
    potentials=phi,
    component="dm",
    metadata={"snap": snap, "subhalo_id": sid},
)

result = analyze_snapshot_target(
    target,
    shape_kwargs={"percentile": 100.0, "tensor_mode": "reduced"},
    kinematics_kwargs={"min_particles": 20},
    dynamics_kwargs={
        "binding_kwargs": {"compute_potential_if_missing": False},
        "shell_kwargs": {"shell_method": "radial", "shell_kwargs": {"n_shells": 5}},
    },
)
```

The output is intentionally split into sections:

- `result["shape"]`: iterative tensor, axes, principal vectors, mask, and
  convergence flag.
- `result["kinematics"]`: affine flow tensors, angular momentum, residual
  dispersion, and figure rotation.
- `result["dynamics"]`: binding energy and shell-wise dynamical response.
- `result["matrix"]`: raw moment tensors useful for diagnostics and custom
  matrix analysis.

## Tidal Sign Convention

The Hessian helpers use the gravitational-potential convention
`H = d_i d_j Phi`.  The local differential acceleration tensor is `-H`, because
`a = -grad Phi`.  Therefore the tidal tensor major axis for the largest
stretching direction is the eigenvector with the largest eigenvalue of `-H`.
Use `tidal_stretch_eigensystem(H)` when selecting that direction.
