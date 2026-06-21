# 2LPT Orbit Template Plan

This document describes the planned orbit-template layer for connecting fast
2LPT merger-tree products, with Pinocchio as the first target, to subhalo
motion, one-halo HOD modeling, and approximate subhalo shapes.

## Scientific Goal

The goal is to generate many group-internal subhalo orbit templates from
approximate merger trees.  These templates can be matched to simulations or
data-like catalogs, then compressed into parameters that improve HOD one-halo
terms and nonlinear velocity information.

The long-term model should not assume purely spherical halos.  It should allow
ellipsoidal groups, phase-space perturbations around 2LPT tracks, and shape
models that can start simple and later become layer-dependent.

## Stage 1: 2LPT Tree Ingestion

Input examples:

- Pinocchio group catalogs and merger trees.
- Other 2LPT or fast mock catalogs with object IDs, host IDs, snapshots,
  positions, velocities, and masses.

Implemented starting point:

- `ia_analysis.orbits.pinocchio.PinocchioColumnMap`
- `ia_analysis.orbits.pinocchio.tracks_from_table`
- `ia_analysis.orbits.pinocchio.build_pinocchio_template_library`

The adapter intentionally accepts already-loaded table-like objects.  This lets
the project support multiple Pinocchio file layouts without hard-coding a
single schema too early.

## Stage 2: Relative Phase-Space Templates

For each subhalo linked to a host group:

1. Match group and subhalo rows by snapshot.
2. Compute group-centric relative position.
3. Compute group-centric relative velocity.
4. Store host mass, subhalo mass, and scale factor when available.

Implemented containers:

- `TreeTrack`
- `OrbitTemplate`
- `OrbitTemplateLibrary`

The first HOD-facing feature set includes:

- final group-centric radius
- final speed
- minimum and maximum radius
- final radial velocity
- final tangential velocity
- final specific angular momentum
- radial-action proxy
- optional mass-loss fraction

## Stage 3: Template Generation And Matching

The template library can be enlarged by perturbing phase-space tracks.  This is
useful because 2LPT trees contain approximate large-scale history but not the
full nonlinear internal orbit distribution.

Implemented first approximation:

- `PhaseSpacePerturbationModel`
- `perturb_orbit_template`
- `perturbation_average_features`

Perturbations are defined in a local radial/tangential frame.  This is a
controlled way to estimate average effects from unresolved phase-space scatter.

Future matching targets:

- radial satellite profiles
- velocity dispersion and pairwise velocity distributions
- one-halo real-space correlation functions
- one-halo redshift-space distortion terms
- IA shape-position and shape-velocity correlations

## Stage 4: Ellipsoidal Group Approximation

The model should not be restricted to spherical halos.  The first ellipsoidal
layer uses a homogeneous ellipsoid approximation for the group tidal tensor.

Implemented first approximation:

- `EllipsoidalGroupModel`
- `homogeneous_ellipsoid_tidal_tensor`
- `coherent_layer_shapes`
- `tidal_aligned_shape`
- `initial_shape_alignment_model`

This gives two initial shape modes:

- `coherent`: inner and outer layers share the group orientation.
- `tidal_aligned`: inner and outer layers align with the eigenframe of the
  ellipsoidal tidal tensor.

## Stage 5: Shape Layers

The first shape model should be intentionally simple:

- inner and outer layers have the same orientation;
- their axis ratios may differ;
- the orientation can follow either the group ellipsoid or the analytic tidal
  tensor eigenframe.

The next extension should allow:

- radial shells with different axis ratios;
- energy shells with different axis ratios;
- misalignment angles between inner and outer layers;
- correlations between layer misalignment and orbital phase;
- coupling to the measured tidal field from `ia_analysis.tides` when simulation
  fields are available.

## Stage 6: HOD One-Halo Augmentation

The orbit templates should be compressed into HOD-facing kernels rather than
used as raw tracks in every downstream calculation.

Initial products:

- mean feature vector from a template library;
- feature covariance;
- template scores under user-defined feature weights.

Implemented helper:

- `hod_1h_orbit_kernel`

Future HOD parameters may include:

- satellite radial concentration correction;
- radial and tangential velocity bias;
- central-satellite velocity offset;
- orbit anisotropy parameter;
- ellipsoidal alignment strength;
- inner/outer shape coherence;
- energy-layer shape coherence.

## Development Checklist

1. Confirm the exact Pinocchio table outputs used in the production workflow.
2. Add a file reader that maps those outputs into `PinocchioColumnMap`.
3. Generate template libraries for several mass and redshift bins.
4. Compare template feature distributions with TNG or ClusterSims subhalo
   tracks where available.
5. Fit a small HOD augmentation model to one-halo clustering and velocity
   statistics.
6. Add shape-layer templates and test their IA correlations.
7. Promote stable outputs into maintained notebooks and pipeline functions.
