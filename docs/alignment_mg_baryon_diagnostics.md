# Alignment MG-Baryon Diagnostics

`tools/build_alignment_mg_baryon_report.py` builds a downstream diagnostic layer
from fitted alignment-summary data. It does not parse plots or PDF pages. The
input is a long-form table with one row per measured alignment point, or an
existing `arts_IA` catalogue root that can be materialized into that table.

The report is designed to answer which IA observables are survey-facing
modified-gravity probes, which are baryonic-response controls, which describe
one-halo satellite phase space, and which are simulation-only theory templates.

## Why These Classes Matter

`Tgroup`-based alignments are treated as survey-facing because group catalogues,
reconstructed density/tidal fields, and group major-axis proxies can connect
them to galaxy surveys. They are useful candidates for joint clustering,
lensing, RSD, and IA modified-gravity tests.

`BaryonDM` trends are treated primarily as baryonic-response controls. They
encode galaxy-formation or baryonic mass-fraction sensitivity and are most
useful as nuisance priors or marginalization axes rather than primary MG probes.

Satellite radial and velocity-radial alignments are treated as one-halo IA and
phase-space nuisance controls. They help connect IA, HOD satellite fractions,
radial profiles, and RSD/phase-space consistency.

Central galaxy--halo and central galaxy--`Tgroup` alignments are useful for
LRG-like central IA and large-scale IA modeling. They are natural ingredients for
joint clustering + lensing + IA analyses.

`T_GR+MG` alignments are marked as theory templates. They can show strong
simulation-level MG response, but they depend on an assumed gravity model and
should not be treated as directly observed survey quantities.

## Metrics

For each observable/component/x-variable combination the code computes:

- `MG_RMS`: RMS of `mu(model) - mu(GR)` over non-GR models, redshifts, and bins.
- `MG_MAX`: maximum absolute MG residual.
- `MG_SIGN_COHERENCE`: fraction of residual bins sharing the dominant sign.
- `MG_REDSHIFT_LEVERAGE`: RMS residual variation across redshift.
- `MG_MASS_LEVERAGE`: RMS residual variation across x bins.
- `MG_SNR_PROXY`: RMS of residual divided by combined GR and non-GR errors, when
  errors are available.
- `BARYON_SLOPE`: fitted slope of `mu` versus the `BaryonDM` proxy.
- `BARYON_RMS`: RMS variation along the baryon-proxy axis.
- `BARYON_RANGE`: max-minus-min response along the baryon-proxy axis.
- `MG_TO_BARYON_RATIO`: `MG_RMS / BARYON_RMS` when both are available.

The category assignment combines these metrics with rule-based survey
accessibility:

- `survey_mg_probe`
- `baryon_control`
- `phase_space_nuisance`
- `theory_template`
- `low_priority_or_redundant`

## Running

From a saved long-form summary table:

```bash
python tools/build_alignment_mg_baryon_report.py \
  --input-summary /path/to/alignment_summary_data.csv \
  --output-pdf /path/to/alignment_mg_baryon_diagnostic_report.pdf \
  --output-table /path/to/alignment_mg_baryon_ranking.csv \
  --output-hdf5 /path/to/alignment_mg_baryon_metrics.hdf5
```

If the table has not yet been saved, materialize it from the existing `arts_IA`
catalogue/profile machinery:

```bash
python tools/build_alignment_mg_baryon_report.py \
  --catalog-root /path/to/global/catalogues \
  --materialized-table /path/to/alignment_summary_long_form.csv \
  --output-pdf /path/to/alignment_mg_baryon_diagnostic_report.pdf \
  --output-table /path/to/alignment_mg_baryon_ranking.csv \
  --output-hdf5 /path/to/alignment_mg_baryon_metrics.hdf5
```

Useful options:

```bash
--min-count 20
--gravity-models GR F40 F45 F50 F55 F60
--baseline-model GR
--selected-observables CGHA StarShape_groupTidal TidalMajorRadial_GRMG
--max-detail-pages 20
--no-make-pdf
```

## Interpreting The PDF

The executive summary lists the number of alignment points, gravity models,
redshifts, and top-ranked observables. The inventory page shows how the sample is
distributed across categories, populations, references, and x variables.

MG residual heatmaps show `Delta_mu = mu(F_i) - mu(GR)` with a diverging colormap
centered at zero. Coherent colors across redshift or x bins indicate a stable MG
response.

The MG-vs-baryon scatter plot separates responses:

- upper-left: MG-sensitive and baryon-robust probes;
- lower-right: baryon-dominated nuisance controls;
- upper-right: degenerate MG+baryon observables needing joint modeling;
- lower-left: weak or low-priority observables.

Ranking tables provide survey-facing MG probes, baryon controls, phase-space
nuisance controls, theory templates, and MG-baryon degeneracy candidates. The
recommended data-vector section states which entries should be treated as direct
survey observables and which should enter as simulation priors/templates.
