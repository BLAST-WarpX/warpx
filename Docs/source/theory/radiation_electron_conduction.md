# Isotropic ideal-electron conduction increment

This records the original ideal-electron increment. The subsequent
[nonideal Cartesian conduction implementation](radiation_nonlinear_electron_conduction.md)
adds nonlinear latent-energy and single-table EOS support without changing
this ideal solver.

Status, 13 September 2026: implemented; bounded Cartesian/RZ CPU and native
RZ CUDA/restart checks pass. No hohlraum application is used as qualification.

This closes the previous absence of electron heat flux for a defined model:
ideal hybrid electrons coupled to fixed-charge ions on native finite volumes, with
isotropic conductivity, insulating/periodic faces and no AMR or EB. Ions remain
kinetic. Tables, anisotropy and nonlocal electron heat transport are not enabled.

## Discrete model

With `Cv = ne kb/(gamma-1)` and symmetric face conductance
`g_ij = kappa_face A_face/distance`, each accepted substep solves

`(Cv_i + h sum(g_ij)/V_i) T_i(new) - h sum(g_ij T_j(new))/V_i = Cv_i T_i(old)`.

The existing audited positive graph solve is reused with Cv as capacity and
T as the specific unknown. Its iteration, positivity, 128-epsilon matrix
residual and 4096-epsilon-scale integrated inventory checks are unchanged.
The caller's model name now gives conduction-specific failure messages; the
original charge-energy transport algebra and defaults are unchanged.

Face conductivity is the harmonic mean. The optional harmonic flux limiter
and conductivity parser are evaluated on the old substep state. This is a
first-order linearized model: it does not claim a fully implicit nonlinear
Spitzer solution or an unconditional hard cap on the final implicit flux.
The substep stiffness is bounded by eight, with an explicit maximum number
of substeps. This is a performance limitation to document, not a reason to
weaken the algebraic solve gates. No multigrid scalability is claimed.

The RZ measure uses the same axis-volume correction and physical-wall factors
as native caloric transport. Exactly empty support conducts no heat. A uniform
temperature or identically zero conductivity is an exact no-op, including
temperature ghosts. Live temperature and the local redistribution diagnostic
are committed only after the entire requested interval succeeds. The new
manifest preserves model/expression/limiter identity on restart.

PICMI's `HybridPICSolver` accepts `electron_energy_transport`,
`electron_heat_conduction`, `electron_thermal_conductivity`, and the three
`electron_conduction_*` controls with the same names and units as the input
deck. `electron_thermal_conductivity` omits the `(rho,Te)` suffix in Python.
Numeric values and expressions both work; solver keyword constants are
name-mangled consistently. Omitted controls do not change legacy defaults.

## Fixed-charge composition dependence

The optional `electron_thermal_conductivity(rho,Te,Zbar,Zeff)` signature can
distinguish materials at equal charge density and temperature. Select exactly
one conductivity signature. PICMI calls the new expression
`electron_thermal_conductivity_composition`.

For depositing positive fixed-charge ions, `Z_s=q_s/e`,
`Zbar=sum(Z_s n_s)/sum(n_s)` and `Zeff=sum(Z_s^2 n_s)/sum(Z_s n_s)`.
Species charge densities come from the same physical FV deposits as total
charge, including the corrected RZ walls. The builder rejects inconsistent
total/species charge, rather than adding a floor or inventing composition.
Vacuum has zero moments and no conductivity evaluation. Neutral species are
excluded: this is a charged-ion average, not the ionization fraction of a
partly neutral gas. Mixture moments change as particles redistribute, but
remain frozen with density during one heat-conduction interval.

`hybrid_conduction_mean_charge_fp` and
`hybrid_conduction_effective_charge_fp` expose the last evaluated nodal
moments, initially zero. Their checkpoint fields and the distinct signature
manifest preserve the interpretation on restart. The two-argument signature
retains its existing manifest; external constants/charges remain the caller's
responsibility.

The local FLASH4.8 `ConductivityMain/SpitzerHighZ` implementation uses an EOS
mean charge and its Coulomb-log model, while `Spitzer` uses a temperature
power law. This motivates making composition available, not claiming a
calibrated preset. The new interface deliberately exposes both different
charge averages without choosing a high-Z atomic or mixing closure for the
user. No FLASH source or new FLASH numerical result is included here.

Bounded checks distinguish Zbar=4/3 from Zeff=3/2 for equal charge-density
contributions with charges one and two. Vacuum-bounded contacts with **equal
charge density but unequal conductivity** match independent two-capacity
classical and limited solutions (normalized error below 2e-13 on local CPU).
A native two-species case follows 50 finite Qei exchanges with composition-
dependent conduction: temperature contrast decreases to 0.621 of its prior
value, total-energy residual is 1.62e-13, and the disabled heavy-species
velocities remain bitwise unchanged. MPI and signature/charge-consistency
rejection gates pass in the focused CPU selection (15/15). These are operator
and coupling contracts. The same focused local CUDA selection passes 15/15;
Cartesian CPU serial/MPI operator checks pass 9/9. This is not front-propagation
agreement with a FLASH fluid closure.

## Bounded results

- A periodic Fourier mode loses more than half its initial amplitude over ten
  steps. The test checks the independent discrete backward-Euler solution,
  not just conservation. RZ normalized field error is 2.63e-12 and relative
  energy residual is -2.47e-12. This is not a continuum refinement claim.
- A vacuum-bounded two-node contact has unequal heat capacities and
  conductivities. Independent two-capacity solutions check harmonic interface
  conductance, classical and flux-limited exchange, and no heat deposited in
  vacuum. Exact zero-conductivity and uniform-temperature identities pass.
- Cartesian 1D, 2D and 3D serial/MPI operators and invalid-conductivity
  rejection pass 9/9 in 3.26 seconds. The first fresh-build invocation exposed
  a missing directory-scope Python interpreter discovery; CMake now explicitly
  finds the interpreter for these direct CTests. No physics assertion changed.
- Native RZ pressure-work plus conduction completes 400 steps and passes the
  existing 1e-10 total-energy bound. Measured relative residual is -1.83e-11,
  with 1.122e-6 J transferred into ion kinetic energy. Same-rank and
  one-to-two-rank restarts pass their field/particle comparisons.
- Local RTX 4000 Ada CUDA operators, native integration, same/changed-rank
  restart and changed-model rejection pass 9/9 in 243.62 seconds. These are
  local CUDA results, not a claim of new remote P40 qualification.

No tolerance has been changed to reduce these residuals. The bounded results
are sufficient to proceed with feature development; extensive timestep/mesh
studies and high-stiffness performance work remain separate qualification.
FLASH4.8's local Spitzer conductivity implementation was inspected as reference
context, but no new FLASH run is claimed from source inspection alone.
