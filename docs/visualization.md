# Visualization Structure

The visualization package now has lightweight facade modules with clear
responsibilities.  Legacy modules remain available, but new notebooks and
scripts should prefer the structured paths below.

## Recommended Modules

- `ia_analysis.visualization.color_tools`: palette extraction and indexed color lists.
- `ia_analysis.visualization.projection_geometry`: principal-plane bases, projected coordinates, ellipses, and scale bars.
- `ia_analysis.visualization.scene3d`: 3D scatter views and galaxy-system scenes.
- `ia_analysis.visualization.shell_plots`: radial shell and binding-shell panels.
- `ia_analysis.visualization.alignment_catalogs`: MAset loading, field access, flag labels, and population masks.
- `ia_analysis.visualization.alignment_metrics`: vector and tensor alignment diagnostics.
- `ia_analysis.visualization.alignment_plots`: alignment plot specs, binned profiles, grids, and figure suites.
- `ia_analysis.visualization.orbit_animation`: orbit movie and preview entrypoints.
- `ia_analysis.visualization.distribution_fits`: distribution models used by plotting code.
- `ia_analysis.visualization.parallel_alignment`: parallel alignment-grid CLI helpers.

## Compatibility

The old implementation files are still importable:

- `ia_analysis.visualization.arts`
- `ia_analysis.visualization.arts_IA`
- `ia_analysis.visualization.DWE`
- `ia_analysis.visualization.orbit_viz`
- `ia_analysis.visualization.orbit_viz2`
- `ia_analysis.visualization.plot_GH_alignment_parallel`

The new modules lazily load those implementations, so importing the package does
not immediately require plotting libraries such as `matplotlib` or `seaborn`.

## Naming Guidance

Use descriptive module names for new work:

- Put catalog and mask logic in `alignment_catalogs`.
- Put scalar alignment calculations in `alignment_metrics`.
- Put figure construction in `alignment_plots`.
- Put shell-specific panels in `shell_plots`.
- Put orbit animations in `orbit_animation`.

When adding new functions, prefer verbs that describe the action, for example
`load_alignment_catalogs`, `plot_alignment_suite`, or
`save_six_panel_orbit_movie`.

