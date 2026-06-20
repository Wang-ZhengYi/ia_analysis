# Structured API Facades

The project keeps historical implementation modules in place for notebook and
script compatibility, but new code should prefer the structured API facades.
Each facade is lightweight at import time and lazy-loads the scientific
implementation only when a function or class is actually used.

## Domain Registry

Use `ia_analysis.api` when a notebook needs to discover package capabilities:

```python
from ia_analysis import api

api.available_domains()
api.load_domain_api("catalogs")
```

The registry currently covers `catalogs`, `shapes`, `tides`, `dynamics`,
`merger_tree`, `covariance`, `pipelines`, `orbits`, and `visualization`.

## Recommended Imports

Catalog readers:

```python
from ia_analysis.catalogs import open_cluster_catalog, open_tng_catalog
```

Shape measurements and IA projections:

```python
from ia_analysis.shapes import measure_iterative_shape, project_shape_ellipticity
```

Tidal fields:

```python
from ia_analysis.tides import build_tidal_grid, sample_tidal_grid
```

Halo dynamics:

```python
from ia_analysis.dynamics import analyze_particle_halo, compute_tng_halo_sample
```

Merger-tree tracks and cross-time workflows:

```python
from ia_analysis.MergerTree import build_main_progenitor_track, run_cross_time_workflow
```

Covariance products:

```python
from ia_analysis.covariance import build_covariance, write_covariance_hdf5_group
```

Pipeline discovery:

```python
from ia_analysis.pipelines import list_pipelines, pipeline_command

pipeline_command("cs-global")
```

Orbit experiments:

```python
from ia_analysis.orbits import generate_mock_halo, run_orbit
```

Visualization:

```python
from ia_analysis.visualization import available_groups, plot_alignment_suite
```

## Compatibility Policy

Legacy modules such as `catalog_loader.py`, `TNGCatLoader.py`, `shape.py`,
`Iana.py`, `tidal_field.py`, `halo_dynamics.py`, `hd_tng.py`, `Cov.py`,
`orbit_nfw.py`, `halo_maker.py`, `arts.py`, and `arts_IA.py` remain importable.
The structured facades are now the preferred surface for new code because they
make dependency direction and function ownership easier to see.

TNG dynamics are now consolidated into `ia_analysis.dynamics.hd_tng`.  The
former `hd_tng_patched.py` and `hd_tng_mea_enriched.py` copies were removed so
there is only one maintained implementation for catalog-backed halo dynamics
and cross-time pattern-speed diagnostics.
