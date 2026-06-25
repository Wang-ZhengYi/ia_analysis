# Layered pipeline architecture

The reusable pipeline implementation follows one dependency direction:

```text
catalogs
   |
   v
shapes + tides
   |
   v
dynamics + MergerTree
   |
   v
spectra + correlations
   |
   v
visualization
```

`ia_analysis.pipelines` is a thin orchestration layer. It may compose all
domain layers, but scientific calculations should remain in the domain that
owns them. Notebooks configure paths and plotting options, call these APIs,
and write their outputs beside the notebook.

## Notebook-derived APIs

- `ia_analysis.catalogs.analysis`: persisted catalog discovery and inventory.
- `ia_analysis.catalogs.hod`: LRG/ELG occupation counts, mass bins, and environment splits.
- `ia_analysis.shapes.evolution`: positive-definite tidal shape relaxation.
- `ia_analysis.tides.diagnostics`: tidal eigenvalue strength and anisotropy.
- `ia_analysis.dynamics.orbit_shape`: multi-orbit stripping and shape suite.
- `ia_analysis.MergerTree.diagnostics`: cross-time table and closure summaries.
- `ia_analysis.spectra.analysis`: spectrum reading and GR-relative comparison.
- `ia_analysis.correlations.quality`: covariance and signal-to-noise checks.
- `ia_analysis.visualization.pipeline_plots`: final-stage summary figures.
- `ia_analysis.pipelines.layered_analysis`: output-writing orchestration.

The architecture test in `tests/test_layer_dependencies.py` rejects imports
from an earlier layer into a later layer.
