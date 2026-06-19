# Velocity Divergence And Self-Folding

## Decision

The existing local velocity-divergence estimator should not be treated as an
exact self-foldable field.

The current mesh builders paint density and momentum, form a local velocity
field with $v(x)=p(x)/rho(x)$, and then compute $theta=-div(v)/(aH)$.  The
cell-wise division by $rho(x)$ is nonlinear.  If particles are folded before the
division, different images are mixed in the same cell, so the folded result is
not guaranteed to match the high-$k$ content of the unfolded local velocity
field.

## Implemented Alternative

`ia_analysis.spectra.velocity_momentum` provides a folded momentum-divergence
proxy:

- `build_momentum_divergence_mesh`: paints additive momentum components on the
  folded mesh and computes their spectral divergence.
- `append_momentum_divergence_cross_pairs`: builds available cross-pair lists.
- `measure_momentum_divergence_cross_spectra`: measures cross spectra with
  existing fields such as `d`, `E`, and `g`.
- `velocity_divergence_self_folding_status`: returns the decision summary for
  logs or notebooks.

The recommended short field name for power-spectrum work is `tm`.

## Interpretation

`tm` is not the same estimator as the historical local velocity divergence `t`
or particle velocity divergence `tp`.  It is useful for folded cross spectra
because it is built from additive momentum density, while `t` and `tp` remain
the appropriate names for the local velocity-divergence meshes.

