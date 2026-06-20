# Visualization Structure

The visualization package now has lightweight facade modules with clear
responsibilities.  Legacy modules remain available, but new notebooks and
scripts should prefer the structured paths below.

## Recommended Modules

- `ia_analysis.visualization.figure_io`: directory creation, filename
  sanitizing, figure saving, and figure-grid helpers.
- `ia_analysis.visualization.plot_styles`: project style setup, model labels,
  component labels, and scalar colormap helpers.
- `ia_analysis.visualization.legends`: reusable legend handles, model legends,
  component legends, and compact axis colorbars.
- `ia_analysis.visualization.color_tools`: palette extraction and indexed color lists.
- `ia_analysis.visualization.projection_geometry`: principal-plane bases, projected coordinates, ellipses, and scale bars.
- `ia_analysis.visualization.scene3d`: 3D scatter views and galaxy-system scenes.
- `ia_analysis.visualization.shell_plots`: radial shell and binding-shell panels.
- `ia_analysis.visualization.alignment_catalogs`: MAset loading, field access, flag labels, and population masks.
- `ia_analysis.visualization.alignment_metrics`: vector and tensor alignment diagnostics.
- `ia_analysis.visualization.alignment_plots`: alignment plot specs, binned profiles, grids, and figure suites.
- `ia_analysis.visualization.alignment_atlas`: reusable metric-atlas,
  redshift-evolution, heatmap, and property-distribution plots extracted from
  exploratory notebooks.
- `ia_analysis.visualization.profile_plots`: HOD, radial-profile, and generic
  DataFrame curve-grid helpers.
- `ia_analysis.visualization.spectrum_plots`: power-spectrum, AIA, and ratio
  plot helpers for measured spectrum tables.
- `ia_analysis.visualization.correlation_plots`: correlation-function key and
  tidy-grid plotting helpers.
- `ia_analysis.visualization.tng_dynamics_plots`: pi-closure, residual,
  alignment-distribution, component-fraction, and binding-energy diagnostic
  plots.
- `ia_analysis.visualization.merger_tree_plots`: cross-snapshot orbit-plane,
  closure-evolution, shell-density, and particle-profile comparison plots.
- `ia_analysis.visualization.orbit_animation`: orbit movie and preview entrypoints.
- `ia_analysis.visualization.distribution_fits`: distribution models used by plotting code.
- `ia_analysis.visualization.parallel_alignment`: parallel alignment-grid CLI helpers.

## Notebook Helper Migration

Reusable plotting code found in notebooks is now grouped by behavior instead of
by notebook filename:

- Repeated `save_fig` and `save_figure` helpers belong in `figure_io`.
- Paper or notebook style dictionaries, model colors, component colors, and
  snapshot labels belong in `plot_styles`.
- Repeated legend and colorbar snippets belong in `legends`.
- HOD curves, radial profiles, satellite profiles, and model-comparison grids
  belong in `profile_plots`.
- `P(k)`, `AIA(k)`, ratio-to-GR, and redshift spectrum grids belong in
  `spectrum_plots`.
- Correlation-function key plots and multi-statistic grids belong in
  `correlation_plots`.
- TNG pi-closure tables, closure residual histograms, direction-distribution
  panels, component-fraction plots, and binding-energy distribution plots
  belong in `tng_dynamics_plots`.
- Merger-tree orbit-plane overlays, cross-time closure evolution, shell-density
  sequences, and particle-profile comparisons belong in `merger_tree_plots`.
- Exploratory alignment metric atlases, redshift evolution summaries, heatmaps,
  and property-distribution atlases belong in `alignment_atlas`.

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

- Put shared saving and layout logic in `figure_io`.
- Put shared style, labels, and color maps in `plot_styles`.
- Put shared legends and colorbars in `legends`.
- Put catalog and mask logic in `alignment_catalogs`.
- Put scalar alignment calculations in `alignment_metrics`.
- Put figure construction in `alignment_plots`.
- Put generic notebook alignment atlases in `alignment_atlas`.
- Put generic curve grids and radial profiles in `profile_plots`.
- Put spectrum-display logic in `spectrum_plots`.
- Put correlation-display logic in `correlation_plots`.
- Put TNG dynamics diagnostic plots in `tng_dynamics_plots`.
- Put cross-time merger-tree plots in `merger_tree_plots`.
- Put shell-specific panels in `shell_plots`.
- Put orbit animations in `orbit_animation`.

When adding new functions, prefer verbs that describe the action, for example
`load_alignment_catalogs`, `plot_alignment_suite`, or
`save_six_panel_orbit_movie`.
