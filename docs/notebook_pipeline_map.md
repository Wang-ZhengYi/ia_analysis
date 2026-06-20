# Notebook Pipeline Map

This document maps raw notebooks to curated workflows and reusable Python code.

## Alignment And Figures

- Raw notebooks: `full.ipynb`, `full_alignments.ipynb`, `cluster_ia_paper_figure_suite_errorband_smooth.ipynb`, `plot_all_pks.ipynb`, `plot_pks.ipynb`, `plot_tcfs_3x5.ipynb`.
- Maintained notebook: `notebooks/pipelines/02_alignment_figure_suite.ipynb`.
- Reusable modules: `ia_analysis.visualization.alignment_catalogs`, `ia_analysis.visualization.alignment_metrics`, `ia_analysis.visualization.alignment_plots`, `ia_analysis.visualization.parallel_alignment`.

## HOD And Population

- Raw notebooks: `HOD_data.ipynb`, `HOD_LRG_ELG.ipynb`, `hod_measure_lrg_elg.ipynb`, `MAset_satellite_radial_distribution.ipynb`, `MAset_satellite_radial_distribution_compare.ipynb`, `merger_align.ipynb`, `merger_stripping.ipynb`.
- Maintained notebook: `notebooks/pipelines/03_hod_population_pipeline.ipynb`.
- Exported scripts: `ia_analysis.notebook_pipelines.exports`.

## Power Spectra And Correlations

- Raw notebooks: `pks_PK_AIA.ipynb`, `plot_pks.ipynb`, `plot_all_pks.ipynb`, `ia_corr.ipynb`, `ia_corr_abundance.ipynb`.
- Maintained notebooks: `04_power_spectrum_pipeline.ipynb`, `05_correlation_pipeline.ipynb`.
- Reusable modules: `ia_analysis.spectra`, `ia_analysis.covariance`.

## TNG Dynamics And Cross-Z

- Raw notebooks: `TNGCatLoader.ipynb`, `TNGCatLoader_test.ipynb`, `hd_tng_crossZ.ipynb`, `hd_tng_plot.ipynb`, `crossz.ipynb`, `global_test.ipynb`.
- Maintained notebook: `06_tng_dynamics_layered_pipeline.ipynb`.
- Reusable modules: `ia_analysis.catalogs`, `ia_analysis.MergerTree`,
  `ia_analysis.dynamics`, `ia_analysis.pipelines.tng_layered_shape_tide`.

## Orbits And 3D Visualization

- Raw notebooks: `orbit.ipynb`, `tri3D.ipynb`.
- Maintained notebook: `07_orbits_and_shell_visualization.ipynb`.
- Reusable modules: `ia_analysis.orbits`, `ia_analysis.visualization`.
