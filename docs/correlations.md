# Correlations Package

The `ia_analysis.correlations` package measures real-space correlation
functions directly from object catalogs.  It is designed for IA, density,
velocity, and halo figure-rotation analyses where all fields live on the same
catalog rows.

## Field Convention

- `d`: scalar density or object weight field.  If it is absent, the estimator
  uses an all-ones scalar field.
- `e`: shape field.  It can be scalar, a 3-vector, or a 3 by 3 tensor.  Vector
  and tensor fields are radially projected along each pair separation.
- `v`: velocity vector field.
- `omega`: figure-rotation angular-velocity vector field.

The default two-point suite measures:

- `ee`
- `ed`
- `dd`
- `vv`
- `dv`
- `ev`
- `omegae`
- `omegad`
- `omegav`

Cross-correlations are symmetrized over the two pair directions by default.

## Halo And Sample Categories

The output always keeps summary categories:

- `total`
- `1h`
- `2h`

It also keeps five detailed non-overlapping central/satellite halo categories:

- `1h_cs`
- `1h_ss`
- `2h_cc`
- `2h_cs`
- `2h_ss`

The category `1h_cc` is not included by default because the standard halo model
has at most one central per host halo, making same-halo central-central pairs
empty.  If an input catalog violates that convention, those pairs still
contribute to the summary `1h` category.

## Four-Point Product

The package estimates a compressed pair-binned four-point product named
`vedomega4`.  It uses the same radial bins and categories as the two-point
statistics.  A connected approximation named `vedomega4_connected` is also
available:

`vedomega4_connected = vedomega4 - ev * omegad - dv * omegae - omegav * ed`

This is a compact diagnostic, not the full configuration-dependent four-point
tensor.

## Minimal Usage

```python
import numpy as np
from ia_analysis.correlations import CorrelationCatalog, measure_default_correlations

catalog = CorrelationCatalog(
    positions=pos,
    fields={"e": e, "v": vel, "omega": omega},
    host_id=host_id,
    sample_type=sample_type,
    boxsize=205.0,
)

rbins = np.logspace(-1, 1.5, 16)
suite = measure_default_correlations(
    catalog,
    rbins,
    covariance="jackknife",
    nsub=3,
)

ee_total = suite.results["ee"].values["total"]
ev_2h_cs_cov = suite.covariance["ev"]["2h_cs"]["cov"]
```

## HDF5 Output

```python
from ia_analysis.correlations import write_results_hdf5

write_results_hdf5("correlations.hdf5", suite)
```

The writer stores statistics under `statistics/<name>/categories/<category>/`
with `value`, `count`, `weight_sum`, and optional covariance datasets.
