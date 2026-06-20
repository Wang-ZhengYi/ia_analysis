# MergerTree Workflows

`ia_analysis.MergerTree` owns cross-time reading and orchestration.  It should
not duplicate catalog, shape, or halo-dynamics algorithms.  Instead, it decides
which snapshots and subhaloes belong to a track, then calls the relevant domain
modules for each step.

## Responsibilities

The package is split into three layers:

- `ia_analysis.MergerTree.reader`: load SubLink or TNG merger trees, convert
  tree dictionaries to tables, select requested snapshots, and match target
  tracks to reference tracks such as a z=0 host central.
- `ia_analysis.MergerTree.workflow`: run snapshot-level tasks.  It calls
  `catalogs`/`hd_tng` to load particles, `shapes` to measure component shapes,
  and `dynamics`/`hd_tng` to run halo shell analyses.
- `ia_analysis.MergerTree.storage`: save and reload computed cross-time
  products so plotting notebooks do not re-download or recompute data.

## Standard Step Order

1. Load the target main-progenitor branch with `build_main_progenitor_track`.
2. Optionally find the z=0 host central and build a matched reference track with
   `build_target_reference_tracks`.
3. For each selected snapshot, create a `SnapshotTask`.
4. Load requested particle components with `load_snapshot_components`.
   The default components are dark matter and stars; gas and black holes can be
   requested when needed.
5. Measure component shapes with `measure_snapshot_shapes`.
6. Run dark-matter shell dynamics with `analyze_snapshot_shells`.
7. Optionally compute cross-time figure-rotation closure with
   `cross_time_pattern_speed_for_subhalo`.
8. Save products with `save_cross_time_products` and reuse them in plotting
   notebooks.

## Example

```python
from ia_analysis.MergerTree import run_cross_time_workflow, save_cross_time_products

products = run_cross_time_workflow(
    base_path="/path/to/TNG300-1",
    snap0=99,
    subhalo_id0=12345,
    snap_track=[99, 91, 84, 67],
    sim_name="TNG300-1",
    api_key=None,
    components=("dm", "stars"),
    shell_methods=("radial", "binding_energy"),
)

save_cross_time_products(products, "outputs/subhalo_12345_cross_time.pkl")
```

The workflow returns a dictionary with the selected track, snapshot tasks,
per-snapshot records, optional cross-time closure tables, and the TNG catalog
configuration used for the run.
