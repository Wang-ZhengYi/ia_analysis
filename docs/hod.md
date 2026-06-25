# HOD and Component-Based IA-HOD

`ia_analysis.hod` provides a lightweight modeling layer between standardized
halo/galaxy catalogs and clustering or intrinsic-alignment validation
statistics. It uses NumPy, SciPy, pandas, h5py, and Matplotlib only.

## Workflow

```text
halo / galaxy catalog
    -> ordinary HOD
    -> fixed-mass assembly HOD
    -> satellite phase-space and layer statistics
    -> alignment-component measurements
    -> conditional IA-strength models
    -> optional mock orientations
    -> xi / omega / eta validation
```

The package separates three assembly-bias channels:

1. occupation assembly bias: changes in central or satellite counts at fixed
   halo mass;
2. phase-space assembly bias: changes in satellite radius, velocity
   anisotropy, or binding-energy layers at fixed mass;
3. IA assembly bias: changes in alignment strength at fixed mass and radius.

## Catalog Adapter

`standardize_hod_catalog` accepts dictionaries, structured NumPy arrays, and
pandas DataFrames. Column names can be mapped to the canonical schema. Required
fields are `halo_id` and `mass` for haloes, and `galaxy_id` plus `halo_id` for
galaxies. Central/satellite flags can be supplied or inferred.

Optional halo fields include concentration, environment, tidal anisotropy,
formation time, spin, shape parameters, host principal axes, and tidal
eigenvectors. Optional galaxy fields include LRG/ELG labels, position,
velocity, shape axes, subhalo axes, spin, angular momentum, radial direction,
`r_over_rvir`, binding energy/layer, and figure-rotation axis.

## Ordinary and Assembly-Biased HOD

`measure_hod` returns `HODMeasurement` with central, satellite, and total
occupation means and total occupation variance. Related helpers measure
`P(N|M)`, satellite fraction, and number density.

Zheng-style predictions and least-squares fitting are provided by
`predict_zheng_hod` and `fit_zheng_hod`.

Assembly splitting always starts with
`standardize_secondary_property_at_fixed_mass`. The secondary-property mean
trend with halo mass is removed before concentration, environment, tidal
anisotropy, formation-time, spin, shape, or custom quantiles are assigned.
Decorated HOD functions provide bounded central and positive satellite
modulations.

## Phase-Space HOD

`phase_space.py` measures:

- satellite radial occupation in `r/Rvir`;
- velocity anisotropy by halo mass;
- binding-energy-layer occupation;
- the same summaries in precomputed environment or concentration quantiles;
- optional satellite-position alignment with a host major axis.

These measurements do not assume that all host diagnostics are spherical.

## IA Reference Components

The reference bank supports host and tidal principal axes, subhalo axes, radial
direction, velocity, angular momentum, spin, figure rotation, binding-energy
layer axes, environment axes, and custom vectors.

The scalar component statistic is

```text
A = <|e . q_ref|^2> - 1/3
```

and is invariant under either axial sign flip.

`IAComponentMeasurement` stores component identity, population, sample label,
mass/radius/secondary bins, layer labels, values, errors, counts, covariance,
and metadata. Measurements include central-host, central-tidal,
satellite-radial, satellite-host, satellite-subhalo, satellite-tidal,
satellite-velocity, satellite-spin, binding-layer, and figure-rotation
components.

## Conditional IA Strength

Each component has its own bounded field:

```text
mu_k = tanh(mu0
            + beta_mass log10(M / M0)
            + beta_radius log10(x / x0)
            + beta_secondary S_tilde
            + sample_term
            + layer_term)
```

`ComponentIAHODModel` combines several `IAComponentModel` objects without
forcing them to share a global amplitude or physical reference.

## Relation to arXiv:2311.07374v3

The paper motivates extending HOD mocks with galaxy orientation and IA-strength
parameters. This implementation uses that idea as a starting point, but exposes
multiple physical reference axes and conditions each component on mass, radius,
population, galaxy sample, assembly variable, and structural layer. It is
therefore designed for the broader alignment measurements already produced by
this repository rather than as a reproduction of one paper model.

Reference: <https://arxiv.org/abs/2311.07374>

## Synthetic Example

```python
from ia_analysis.hod import (
    standardize_hod_catalog,
    measure_hod,
    measure_assembly_hod,
    measure_alignment_hod_components,
)

catalog = standardize_hod_catalog(halo_table, galaxy_table)
ordinary = measure_hod(catalog, mass_bins=mass_edges, sample_label="LRG")
assembly = measure_assembly_hod(
    catalog,
    secondary_property="concentration",
    mass_bins=mass_edges,
)
radial_ia = measure_alignment_hod_components(
    catalog,
    component="satellite_radial",
    reference="radial_vector",
    population="satellite",
    mass_edges=mass_edges,
    radius_edges=radius_edges,
)
```

## Limitations and Future Work

The first implementation uses simple pair loops, binned summaries, and
least-squares fits. It does not provide Corrfunc acceleration, weak-lensing or
cosmic-shear likelihoods, MCMC, emulators, simulation-specific readers, or a
calibrated multi-reference axial sampler. The documented forward-model target
is `p(e) proportional to exp(sum_k kappa_k (e.q_k)^2)`.
