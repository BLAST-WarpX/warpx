# Radiation qualification checkpoint: WIP

This update publishes the current qualification work, not a declaration of
production readiness. It does not merge the PR, include circuit-coupling packs,
or replace kinetic ions with a FLASH fluid closure. AMR is not a completion
requirement for this work.

## Implemented scope

- Opt-in stationary implicit diffusion and native electron caloric exchange,
  with private candidate state, residual checks and bounded substep retries.
- Opt-in gray moving moment transport in periodic Cartesian geometry, coupled
  to native kinetic ions and hybrid electron energy. Actual finite ion work and
  deferred particle-owned impulse/work are separately accounted and restarted.
- Native shape-consistent force assignment and adjoint work, with independent
  moving-beam ODE, trapped-pulse transport and oblique-interface checks.
- Particle diagnostics use output copies for unit conversion, preserving live
  particle state. Hybrid deposited moment history is preserved on restart.
- Guarded boundary building blocks: positive M1 mirror flux, independent face
  accounting, reflecting-wall force assignment and transactional native
  particle-carry reflection with a checkpointed pending-only wall ledger.

Existing legacy defaults remain. Experimental options and unsupported-path
guards are documented in the input reference. In particular, these boundary
building blocks do **not** enable the full moving radiation/material wall path.

## Evidence and unresolved gates

The public-branch CPU regression subset passes 328/328 checks after restoring
the existing parent carry-ownership validation ahead of its new wall ledger.
This includes the legacy radiation cases and the new implicit/moving exchange,
boundary, rejection, diagnostic and restart checks. The full-run log is
`build-production-cpu/pr-owner-full-tests.log` in the PR qualification worktree.

The latest carry-boundary tests passed on CPU with one rank, two ranks and a
one-to-two-rank checkpoint continuation. They include exact live-state rejection,
normal/reflect-all reflection, two carry owners, migration, overflow rejection,
and compensated cancellation across restart. CUDA qualification of this latest
increment was outstanding when it was packaged.

Earlier CPU/P40 qualification covers the periodic moving beam, trapped pulse,
shape assignment and several independent boundary kernels. Those results do
not qualify all dimensions or all boundary combinations. The strict full cold
material-wall GPU trajectory and cross-backend bounds still fail, even though
total energy conservation and restored deposited-history checks pass. No
physics tolerance has been loosened to turn those failures into passes.

Remaining work includes the integrated physical-boundary material feedback and
complete represented-plus-deferred wall inventory, resolution of those strict
GPU trajectory gates, moving multigroup frequency redistribution, packet/moment
conversion with moving material, and practical full 3D/RZ qualification.
Green CI is a regression/build milestone, not evidence that these open physics
requirements are complete.

### CI checksum follow-up

The September 10 stabilization pass merges upstream resistive-drag and
vacuum-seam changes while retaining the conservative caloric source and
cell-local ion moment projection. The latent-Qei source oracle remains
unmagnetized; the upstream ideal-gas Qei case separately exercises force-free
electron-ion drift.

Local validation with test cleanup enabled passes 434/434 selected 1D
radiation/coupling checks (`build-ci-tools/resume-cleanup-gates-fresh.log`).
The carry-ownership negative test now holds its producer checkpoint until it
finishes. Moving-beam restart preparation archives prior diagnostic output,
preventing duplicate rows during repeated local runs. Single-precision
analysis of the affected source/test translation units also catches and fixes
the RZ implicit-diffusion test warnings hidden by the canceled CI matrix job.
These are selected local gates, not a claim that the complete CI matrix or
the remaining production qualification is green.

An isolated control executable restores only the old live-particle diagnostic
SI round trip. Both `test_1d_collision_z` and its modulus-shuffle variant then
reproduce their existing checksums exactly (maximum relative error zero).
The corrected executable instead reproduces CI's changed checksums, while
both original collision physics analyses pass. This establishes the cause
for these two cases, not yet for every failing reference. At that stage no
failing reference was regenerated; incoming upstream references were retained
as part of the merge. The completed attribution and approved refresh follow
below.

The 2D CI matrix also exposed a real initialization defect: native bulk plasma
injection does not call the generic runtime-attribute initializer, so newly
registered radiation carries could contain allocator residue. Both bulk
injection paths now explicitly zero newborn radiation accounts. The flux path
does so after redistribution of its temporary container, whose runtime
components are not communicated. Existing particle accounts are not reset.
The new pre-filled-capacity birth test passes with the fix and fails when the
initializer is removed. The affected 2D moment/particle-carry subset then
passes 77/77 locally, including MPI, rollback and restart cases.
The expanded 1D radiation/coupling subset also passes 329/329 checks
(`build-ci-tools/birth-final-1d-tests.log`).

Single-precision CI compilation additionally requires native-precision flux
literals and explicit ownership/query declarations. Those changes retain the
moving runtime's double-precision guard. Invalid-tolerance rejection now has
explicit zero, negative, out-of-range, infinity and NaN coverage; no accepted
tolerance range or physics assertion was relaxed.

Azure build 6472 passed the RZ pinch simulation and its original physics
analysis, but its existing `test_rz_theta_implicit_dynamic_pinch` checksum
differed by up to 1.58e-6. A local PETSc/MPI A/B comparison changes only the
diagnostic unit-conversion behavior: restoring the old live SI round trip
changes checksums by up to 4.80e-6 relative to the corrected copy-only writer.
Both variants pass the original 1e-12 energy and charge-conservation bounds
and the solver-iteration limits. Corrected/legacy maximum relative energy
errors are 1.97e-15 / 3.10e-14; charge RMS errors are 8.45e-14 / 8.00e-14.

Neither initial local variant reproduced the exact CI checksum. This established
sensitivity to diagnostic mutation, not unique attribution of the CI delta;
that local PETSc/MPI environment differed from CI. No generated benchmark or
physics tolerance was changed at that stage. Artifacts are under `build-rz-checksum/` and
`build-ci-tools/rz-diagnostic-comparison.json` in the PR worktree. The temporary
legacy writer was removed immediately after building the control executable.

CodeQL also identified two reference-test products evaluated in `double`
before conversion to `long double`. Promoting the operands before arithmetic
fixes those intermediate-precision issues; the existing moving-flux test
passes without changing its bound.

### Completed attribution and approved reference refresh

Matching CI's PETSc 3.25.5/OpenMPI and Ubuntu BLAS/LAPACK/SuperLU setup
resolved the earlier pinch-environment ambiguity. For all 21 diagnostic-related
checksum differences, isolated legacy-writer controls reproduce the existing
references exactly, while corrected local checksum dictionaries are identical
to Azure build 6498. This covers the Cartesian, radial-1D and RZ cases,
including Python/openPMD output. Every available original physics analysis
passes; the legacy 3D PEC case has no separate analysis CTest.

The additional Qei reference difference has a separate cause. Restoring only
upstream's two thermal-exchange functions reproduces that reference exactly.
Both implementations pass the original analysis; the conservative exchange
reduces measured thermal-energy drift from 0.605% to 0.159% and normalized
ion-current projection from 0.0534 to 0.0038. The original limits remain 2%
and 0.08, respectively.

A real RZ merge defect was fixed separately: applying the Cartesian reflective
density operator after radial folding corrupted species density near the axis.
Matching the native radial deposition convention reduces the tested density
error from 0.32808 to zero. An isolated old-code control reproduces the failure;
the unchanged RZ analysis passes in Azure 6498. That completed CI run has no
simulation or physics-analysis failures, only the 22 reference comparisons.

After explicit approval, those 22 references are regenerated from the verified
Azure 6498 records using the checksum framework. No solver code, input or test
tolerance is changed by this refresh. The fresh CI run remains a separate gate;
these reference updates do not close the production-qualification gaps above.

## Detailed records

### Application-exposed initialization and no-op defects

The September 12 exploratory RZ shell/foam/capsule setup exposed two general
defects. These fixes do not qualify the complete hohlraum application:

- Nonuniform radial particle loading culled candidates using density at
  untransformed logical cell coordinates. A zero-step spherical-layer loading
  regression found only 4678 of 5654 expected particles for radial power one,
  and 2132 of 3368 for power two. The physical quadrature sites and cylindrical
  weights now pass for powers zero, one and two, including two-rank loading.
  Only the invalid early density cull is bypassed for nonzero radial power;
  final physical-position density and bounds checks remain authoritative.
- A zero electron-energy increment unnecessarily inverted the caloric EOS,
  which could change temperature and produce a spurious conjugate energy
  request at a particle-support edge. The shared caloric source helper now
  preserves valid state and zero ledgers exactly without an inverse round trip.
  The independent ideal/latent-EOS check improves from 124/256 to 256/256;
  invalid input states remain rejected. Nonzero-source Qei support failures
  are a separate, unresolved application issue.

The new local regression selection passes 9/9 CTest entries. Existing selected
RZ radiation run/analysis checks pass 23/23. No checksum reference or physical
assertion tolerance was changed. These are local checks, not a claim of a new
complete CI pass or production radiation-driven capsule compression.

```{toctree}
:maxdepth: 1

radiation_moving_runtime
radiation_moving_material_plan
radiation_moving_moment_discretization
radiation_moving_beam_verification
radiation_particle_shape_exchange
radiation_moment_boundary_plan
radiation_material_wall_qualification
radiation_oblique_interface_qualification
radiation_reflecting_cavity_qualification
```

Build/test artifacts referenced in these records are local qualification logs,
not bundled simulation data. No FLASH source or remote-machine credentials are
included in this update.
