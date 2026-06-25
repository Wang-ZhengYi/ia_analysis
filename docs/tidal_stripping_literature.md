# Tidal Stripping Algorithm Notes

This note records the lightweight stripping model used by the orbit-template
notebook and explains how it should be upgraded for calibrated HOD work.

## Literature Guidance

- Taylor & Babul, *The Evolution of Substructure in Galaxy, Group and Cluster
  Haloes I: Basic Dynamics*, introduce a semi-analytic merger-tree treatment of
  subhalo orbits, dynamical friction, tidal mass loss, and disruption.  Their
  key practical message for this project is that stripping should be tied to
  orbital history and characteristic orbital timescales, not only to a single
  instantaneous radius.
  <https://arxiv.org/abs/astro-ph/0301612>

- Hayashi et al., *The Structural Evolution of Substructure*, show with
  controlled simulations that stripped NFW subhaloes evolve mainly as a function
  of bound mass loss and that the density profile should be treated as a
  gradually modified structure rather than an instantly truncated sphere.
  <https://arxiv.org/abs/astro-ph/0203004>

- Peñarrubia, Navarro & McConnachie, *The Tidal Evolution of Local Group Dwarf
  Spheroidals*, and Peñarrubia et al., *The impact of dark matter cusps and
  cores on the satellite galaxy population around spiral galaxies*, motivate
  tidal tracks where structural quantities such as $V_\mathrm{max}$ and
  $r_\mathrm{max}$ are functions of the retained bound mass fraction.
  <https://arxiv.org/abs/0708.3087>
  <https://arxiv.org/abs/1002.3376>

- van den Bosch & Ogiya, *Dark Matter Substructure in Numerical Simulations*,
  emphasize that hard subhalo disruption can be a numerical artifact.  For this
  reason the lightweight model keeps a small bound-remnant floor instead of
  deleting templates unless a calibrated disruption criterion is supplied.
  <https://arxiv.org/abs/1801.05427>

- Ogiya et al., *DASH: a library of dynamical subhalo evolution*, and Green &
  van den Bosch, *The tidal evolution of dark matter substructure I*, provide a
  better calibration target for future work.  DASH is especially useful because
  it isolates controlled subhalo evolution in analytic host potentials and
  supplies interpolation-friendly evolution data.
  <https://arxiv.org/abs/1901.08601>
  <https://arxiv.org/abs/1908.08537>

## Implemented Lightweight Model

The implemented module is `ia_analysis.orbits.tidal_stripping`.  It does not
depend on `pyccl`, catalog readers, or the full NFW orbit integrator.

The model has three layers.

1. The instantaneous target computes a Jacobi-like radius
   $r_t = (G M / D)^{1/3}$ with
   $D = \Omega^2 - \Phi^{\prime\prime}$ and a positive floor.  When this radius
   falls inside the reference subhalo size, the retained target mass is solved
   using a power-law closure $M(<r) \propto r^{3-\alpha}$.

2. The default delayed mode relaxes the actual bound mass toward the
   instantaneous target over
   $\tau_\mathrm{strip} = N_\mathrm{orb} T_\mathrm{orb}$.  This prevents a
   single noisy near-pericentre sample from removing too much mass instantly.

3. A Peñarrubia et al.-inspired tidal-track helper converts the retained mass fraction
   into approximate $V_\mathrm{max}$ and $r_\mathrm{max}$ ratios.  The default
   coefficients are intended for demonstrations and should be calibrated before
   scientific production runs.

## Recommended Upgrade Path

- Keep the instantaneous target as a diagnostic.  It is useful for visualizing
  where the orbit is tidally stressed.
- Use delayed stripping for template libraries and HOD matching.  Tune
  `tau_orbits`, `density_slope`, and the tidal-track coefficients against
  controlled simulations or DASH/SatGen-like references.
- Replace the demo power-law host curvature with curvatures sampled from the
  actual spherical or ellipsoidal host model.  For the ellipsoidal group model,
  evaluate the local Hessian along the template orbit rather than using a
  radius-only proxy.
- Do not introduce hard disruption by default.  Use survival flags only after
  adding a resolution-aware or calibration-aware criterion.
- Store both `target_mass` and `bound_mass`.  The difference between them is a
  useful measure of stripping lag and should help when matching one-halo HOD
  velocity and shape templates.
