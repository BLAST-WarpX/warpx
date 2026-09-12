# Moving-moment physical-boundary implementation contract

Status: derivation/qualification work, not an enabled runtime boundary. Keep the
current periodic, particle-boundary and conversion guards until the joint
operator and independent boundary inventories are verified.

The low-level implicit solver now has experimental prescribed-flux and fixed
optical-mirror entry points. Neither enables native-PIC physical boundaries;
the runtime and material-feedback increment helper remain periodic-only.

## Live deferred-carry reflection checkpoint

The native particle boundary dispatcher now stages elastic Cartesian reflection
of every registered radiation carry together with particle position and velocity.
It accepts the whole species only after validating all carry paths and the
globally reduced pending-only wall transfers. Invalid values, overflow, particle
loss, thermalization, radial geometry, refinement, moving windows and non-double
precision reject without modifying live particles. The existing full radiation
runtime physical-boundary guard is unchanged.

The wall ledger records direct event transfers of deferred momentum, not the
represented ion-wall impulse and not a conservation-residual estimate. Its
compensated sum and correction are checkpointed per species/path in
`RadiationCarryWallMomentum_data.txt`. They are already global replicated values;
they must not be summed again across MPI ranks. A nonperiodic carry restart
requires this ledger. Legacy periodic restarts may initialize zero wall history.

The CPU tests `test_1d_particle_impulse_boundary_carry_1`, `_2` and `_restart`
pass 3/3 in 1.58 seconds (`build-production-cpu/carry-boundary-complete-tests.log`).
They check two independent carry paths, normal and reflect-all events, zero
normal velocity, particle migration, unchanged live bits on rejected states,
native dispatch, compensated cancellation and checkpoint continuation from one
to two ranks. This increment has not yet been CUDA-qualified. Full radiation-wall
material feedback, represented wall impulse accounting and the strict GPU
cold-wall trajectory gates remain unfinished. No physics tolerance was changed.

## Low-level physical-edge material assignment

`ParticleImpulseAssignment::ReflectingNodalCellAverage` is a new explicit
low-level 1D PEC/reflecting adapter. No runtime input selects it, and the
existing radiation particle-boundary guard remains in force. The assignment
adapter itself does not reflect carry; the native boundary transaction above
handles deferred carry and its separate pending-only wall momentum export.

Native B-spline nodal weights are averaged onto cells, and their exterior
cell contributions are folded evenly at the physical nodal wall: cell -1
maps to 0, -2 to 1, and correspondingly at the upper face. Every Cartesian
radiation force component uses the same scalar assignment weights. Applying
the component-dependent PEC electric-field parity here would incorrectly
cancel tangential radiation force. Gather, material mass, requested work and
work-velocity deposition use the same folded assignment. Actual finite-kick
impulse/work and their numerical carries retain particle-center ownership.
At the exact upper face that center is the last physical cell, not cell N.
Out-of-domain and nonfinite particle positions reject before any scatter.

The independent test first mirrors nodal charge contributions, doubles the
physical endpoint density and averages physical corner values. It also calls
the native charge-deposition/boundary routine and compares its half-volume
cell mass to the radiation adapter. Normal force covers shape orders 1-4;
fourth-order tangential force and two-rank decomposition provide complementary
checks. Each reflecting case includes 112 overlapping-cloud finite/sub-ULP
kick, opposing/cancelling force and boundary/interior placements, followed
by three invalid-position rejections with unchanged live particle bits.

The initial tangential test passed all 112 candidate checks, then failed a
shared final assertion that assumed the fixture velocity was still along z.
Restoring the fixture's canonical orientation after its tangential checks
fixes that test-harness mismatch; no assertion bound was altered. The combined
CPU reflecting, existing periodic-shape and diagnostic subset passes 16/16
in 8.63 seconds. Logs are in `build-production-cpu/reflecting-assignment-tests.log`
(initial failure) and `reflecting-assignment-native-tests.log` (completed set).
The complete 1D/2D CPU impulse/ownership/restart set subsequently passes 31/31
in 16.28 seconds, including a negative check that the new adapter rejects
periodic geometry. The P40 CUDA 1D set passes 19/19 in 35.67 seconds, including
the two-rank reflecting assignment, native charge comparison and invalid-state
checks. Its log is copied to `build-production-cpu/p40-reflecting-assignment-tests.log`.
After relinking the native executables, moving-beam and trapped-pulse integration/
restart regressions pass 8/8 in 26.65 seconds (`reflecting-assignment-runtime-tests.log`).
These are adapter and periodic-runtime results, not evolving radiation/material-
wall production qualification. Full independent wall ledgers,
physical-boundary material feedback and the unresolved full GPU wall trajectory
gates remain before runtime enablement.

Kinetic reconstruction provides explicit outgoing half-space moment fluxes;
see equations 28, 38 and 43 of
[Kanno, Harada and Hanawa (2013)](https://arxiv.org/abs/1303.6805).
Their cell-to-face treatment also matters in optically thick cells. This is a
radiation closure, not a replacement for the native hybrid-PIC material model.

## Candidate outgoing flux and stable evaluation

For a lab state (E,q=F/c), let f=|q|/E and define the **radiation angular**
parameter b=3(q/E)/(2+sqrt(4-3f^2)). It is not material velocity. For outward
normal n, set bn=b.n, D=1-|b|^2+bn^2, a=-bn/sqrt(D), s=1-a, and bt=b-bn*n.
Factoring the half-space expressions gives

    Phi_E_out/c  = E sqrt(D) s^3 (3+a) / [4(3+|b|^2)]
    Phi_qn_out/c = E D s^3 / [2(3+|b|^2)]
    Phi_qt_out   = bt Phi_E_out.

The transported q flux divided by c is physical momentum flux. For inward,
nearly collimated radiation, evaluate s as
`(1-|b|^2)/(sqrt(D)*(sqrt(D)-bn))`. Obtain `1-|b|^2` from
`12*(1-f)*(1+f)/[(sqrt(4-3f^2)+1)*(sqrt(4-3f^2)+2)]` to avoid cancellation.
Exact vacuum and beam/grazing limits need explicit branches. Near the cone,
norm uncertainty must be accounted for; do not silently clip the stored state.

The independent `reference_moment_boundary.py` integrates a normalized angular
distribution directly. This is an algebra check, not an integrator test.
The initial 48 isotropic/oblique, positive/negative-face checks through f=0.95
pass with maximum normalized absolute difference 4.330e-14.

## Qualified standalone kernel (2026-09-09)

`MomentBoundaryFlux.H` supplies outgoing and fixed-specular-wall helpers,
without enabling native runtime boundaries. The double-precision CTest pair
`test_moment_boundary_flux` and `test_moment_boundary_flux_analysis` passes on
CPU (1.25 seconds) and Tesla P40 CUDA (3.34 seconds, MPI-enabled build).
This is a single-process kernel check, not a boundary decomposition test.

The 334 cases include all Cartesian normals and signs, oblique distributions,
vacuum, exact axial and grazing beams, energies from 1e-200 to 1e200, and
invalid-state rejection. The 325 nonvacuum valid results are exported for an
independent reference: angular quadrature for moderate reduced flux and
long-double axial/grazing limits near a beam. Nonzero backward tails have a
relative-error gate, so silently zeroing tiny outgoing radiation does not pass.
The existing closure's precision collar is retained without clipping stored
energy or momentum. Paired outgoing fluxes recover the full M1 flux; the fixed
mirror has exactly zero energy/tangential flux and twice the outgoing normal
pressure. These are algebraic checks, not production boundary results.

## Independent boundary accounting

`ComputeMomentBoundaryExchange` integrates supplied Cartesian face fluxes, not
the radiation/material residual. The transport fields are cell-integrated, so
the outward increment is the high-minus-low face sum multiplied by `dt/dx`
in each nonperiodic direction. The three q increments divided by c are physical
momentum. Periodic directions contribute zero, including in mixed geometries.
The helper validates face centering and complete, disjoint underlying cell
coverage; it excludes ghosts and duplicated internal faces. Invalid geometry,
layout, timestep or nonfinite physical flux leaves the caller's output unchanged.
It returns a candidate only: retry acceptance and diagnostic accumulation remain
the caller's responsibility. The low-level nonperiodic solver consumes it;
the native moving-material runtime remains periodic-only.

CPU accounting checks pass in 1D/2D/3D, each on one and two MPI ranks (6/6).
A separate four-rank 1D check also passes, exercising ranks with no external
faces. Together with the outgoing-flux/reference pair, 8/8 CTests pass in
4.18 seconds. These are dimension-dependent face-layout unit tests, not copies
of a production simulation or evidence of physical-boundary runtime support.
The MPI/CUDA P40 build also passes its 1D/2D serial and two-rank accounting
checks plus the outgoing-flux/reference pair (6/6, 9.54 seconds). No 3D CUDA
boundary accounting result is claimed.

Reproduce the CPU checks from the worktree root:

```sh
cmake --build build-production-cpu -j 8 --target test_moment_boundary_exchange_1d test_moment_boundary_exchange_2d test_moment_boundary_exchange_3d
ctest --test-dir build-production-cpu -R '^test_moment_boundary_(exchange_|flux)' --output-on-failure
```

## Required integration work

### Prescribed-flux solver stage

`TryImplicitMomentTransport` accepts optional, immutable Cartesian face arrays
for a stage. On nonperiodic faces these replace the numerical flux with the
supplied lab `(energy,q)` flux. The arrays must match the radiation face layout
and distribution mapping. Physical ghosts of private solver states use
constant extrapolation only to complete interior transverse stencils; their
values do not define the boundary flux. A prescribed flux is fixed forcing,
so its derivative in the Krylov correction is zero.

The projected energy-minus-work boundary flux is formed from the same lab flux
and face velocity. Independent signed face sums enter both the global balance
and the conserved-total projection. Only a successfully accepted result exports
`boundary_exchange`; failed results export zero and leave supplied radiation,
material transfer and caloric outputs unchanged. This prevents the periodic
projection from restoring injected/escaped energy to the closed-system total.

Qualification in progress: a 20-step moving-frame scattering case has nonzero
prescribed injection or net outflow, independent accumulated
material/radiation/boundary inventories, insufficient-iteration rejection and
nonfinite-forcing rejection. The serial inflow/outflow and two-rank inflow CPU
checks pass. The test's first rollback check
used an incorrect `MultiFab::norm0` overload; the corrected check explicitly
selects all four components and one ghost layer, with the original exact-zero
assertion retained. This is solver integration evidence, not native particle
boundary or optically thick vacuum-boundary qualification.

The refreshed CPU periodic transport-refinement/coupled-source batch passes
59/59 (51.10 seconds). The native moving-beam ODE, changed-rank restart and
three prescribed-boundary checks pass 7/7 (11.20 seconds) with a fresh restart
output directory. An earlier reused-directory run failed the diagnostic
uniqueness check: 400 rows contained two copies of each of the 200 resumed
steps. The old output was preserved under
`build-production-cpu/restart-history.kJytXc`, not filtered or deleted; the
one-record-per-step assertion remains unchanged. The P40 MPI/CUDA inflow and
two-rank inflow CTests pass (2/2, 5.13 seconds), and the separate CUDA outflow
invocation with `test.injection=-0.1` also passes its full balance/rollback checks.

### State-dependent fixed optical mirror

The generic dissipative interior face flux is not used as a mirror pressure.
For E=1, qx=0.7 pointing away from the low wall, its reflected-ghost construction
gives a negative coordinate pressure, Pxx-0.7. A fixed radiation mirror cannot
exert tension. The selected half-space mirror pressure remains nonnegative and
has identically zero physical energy and tangential momentum flux.

`EvaluateM1MirrorJacobian` differentiates that same pressure. With
`r=sqrt(1-|bt|^2)`, `s=1+bn/r`, and `h=r^2*s^3/(3+|b|^2)`, the outward-normal
q pressure is c E h. Its angular derivatives are

    dh/dbn = [3*r*s^2 - 2*bn*h] / (3+|b|^2)
    dh/dbt = bt*[s^2*(s-3) - 2*h] / (3+|b|^2).

The chain rule uses scaled `E*db/dq` and `E*db/dE` rather than forming 1/E.
The stable backward-tail expression for s is retained. At vacuum the solver
uses the isotropic limiting Jacobian, and at an exact grazing beam the symmetric
angular limit; the physical flux is unchanged. The CPU and P40 CUDA kernel tests
pass finite-difference, homogeneity, scale-extreme and relative near-beam-tail
derivative checks, alongside the independent angular-flux reference (CPU 2/2,
1.12 seconds; CUDA 2/2, 3.32 seconds).

`reflecting_boundaries` is an opt-in low-level solver option, mutually exclusive
with prescribed fluxes. The physical wall pressure is reevaluated on every
nonlinear candidate and its derivative is frozen only for the Krylov correction.
The projected wall flux is `-beta_normal * Phi_qnormal`, from the same lab-frame
pressure; the physical energy flux remains zero. The signed face inventory is
also reevaluated, including after the conserved-total projection, and only the
accepted stage exports it. No boundary diagnostic is inferred from a residual.

Initial serial/two-rank CPU solver checks pass with a nonzero wall impulse and
tangential material drift. They retain exact zero wall energy/tangential exchange
and rejected-state preservation. An additional check reevaluates wall pressure
on the accepted stored state outside the operator to detect a stale candidate
ledger. This is not yet a production reflection/refinement result or particle-
wall implementation. The refreshed CPU solver/periodic-refinement/coupled-source
batch passes 61/61 (52.31 seconds), and native moving-beam/changed-rank restart
passes 4/4 (10.77 seconds) with preserved old restart output and a fresh run
directory. A separate RZ-only CMake configuration keeps the geometry-independent
frame test registered but excludes these unsupported Cartesian boundary-solver
tests. The refreshed P40 MPI/CUDA prescribed/reflecting-wall solver batch also
passes 5/5 (12.44 seconds), including the accepted-state pressure check.

Next accuracy gate: a closed reflecting cavity with a cosine radiation-energy
mode and prescribed tangential material drift. The gray moving diffusion limit
has normal diffusivity `c/(3*kappa*gamma)` and zero normal energy flux. Compare
the Neumann eigenmode's decay and full spatial profile under separate mesh and
timestep refinement. This will test the thick boundary treatment, not merely
its zero-leakage identity. Keep native particle boundaries guarded throughout.

CPU production-resolution and CPU/CUDA CI-sized results for this gate now pass; see
[the reflecting-cavity accuracy report](radiation_reflecting_cavity_qualification.md)
for the separate spatial/time errors, executed visualization, scope and remaining
native-boundary obligations. The finer production-resolution study was CPU-only.

### Native stationary material-wall ownership contract

The next native integration must distinguish an optical radiation wall from
the particle operation. For a stationary Cartesian specular reflection R,
apply the same reflection to particle proper velocity u and each path/group's
pending proper-velocity increment delta-u. The signed scalar pending kinetic-
work account is unchanged: both |u| and |u+delta-u| are invariant under R.

| Quantity | Reflected value | Wall bookkeeping |
| --- | --- | --- |
| Represented particle momentum m u | m R u | Actual material-wall transfer m(u-Ru) |
| Pending radiation momentum m delta-u | m R delta-u | Pending-source transfer m(delta-u-R delta-u) |
| Pending kinetic-work account | Unchanged | Zero for this stationary elastic operation |

The existing radiation momentum diagnostic records cumulative source impulse,
not instantaneous material momentum. Consequently, its pending-carry boundary
entry must not also count the full represented particle-wall impulse: that
would double-count momentum already assigned to material by the radiation
source. Independent total material/wall inventories still need the full actual
particle reflection impulse. Optical-wall radiation momentum is another
distinct transfer, already computed from radiation face fluxes.

Use the actual reflection decision from `ApplyParticleBoundaries`, not a test of
whether a velocity happened to change sign (zero components and stochastic
reflection invalidate that inference). Native `reflect_all_velocities` changes
the transformation and must either transform every carry component accordingly
or remain explicitly unsupported. Thermal reemission and particle removal are
not elastic reflection and need separate energy/export rules.

Before enabling: integrate the event before particle invalidation/redistribution,
persist its accumulated diagnostic state, verify shape-weight deposition/gather
and native nodal boundary control-volume weights, and test boundary-crossing
trajectories with sub-ULP carries and changed-rank restart. The current native
particle boundary and shaped-assignment periodic guards remain in force.

Implementation checkpoint: `ApplyParticleBoundaries::apply_boundaries` now has
an optional `BoundaryEvent` output for its actual coordinate reflections,
thermalization and loss decision. The default call is unchanged and the event
adds no random draw. Radial-coordinate flags are explicitly not Cartesian
momentum reflection flags. The kernel now includes its own constants header;
the standalone test exposed the previous include-order dependency.

`EvaluateMaterialCarryReflection` supplies a pure stationary Cartesian candidate
from that event: reflected pending momentum, unchanged signed pending work and
the pending-only wall transfer. It rejects thermalization, loss and nonfinite
or overflowing accounts. It does not yet write live particle attributes or
accumulate checkpointed wall diagnostics.

The native-kernel test covers inside particles, both walls, zero-velocity
reflection, reflect-all-velocities, open loss, deterministic absorbing-boundary
loss/reflection, and zero/nonzero-temperature reemission. Elastic cases carry
increments of order 1e-30: the test requires nonzero wall export even when the
represented normal velocity is zero, exact pending momentum accounting and
invariant finite requested kinetic work under reflection. Unsupported events
must reject the carry candidate. The CPU event/carry, finite-work, shape and
changed-rank impulse-restart batch passes 12/12 (6.15 seconds). The native event/
carry check also passes on P40 CUDA (1/1, 1.59 seconds). These are kernel and
existing impulse-path checks, not a live reflected-carry checkpoint test.
Live attribute/diagnostic integration, native shape-edge treatment and boundary
restart remain open. Next check physical-edge charge/control-volume weights
with radiation disabled before connecting the reflected carry to native walls.

That native check found and led to an experimental 1D PEC pressure-work adjoint;
see [the material-wall qualification report](radiation_material_wall_qualification.md).
It documents the initial EOS failure, the corrected local-pressure boundary,
passing CPU/P40 energy inventories, CPU restart, and still-failing strict GPU
trajectory comparisons. No radiation carry boundary guard has been removed.

### Still required before native runtime enablement

- Define incoming radiation, reflection and vacuum separately. A fixed mirror
  exchanges normal momentum but no energy or tangential momentum; an incoming
  bath/beam supplies independently counted energy and momentum.
- Extend face evaluation, its linearization and roundoff bounds consistently.
  Filling ghosts periodically at a physical boundary is not an implementation.
- Derive the cell-to-face/stiff-limit treatment with the existing projected
  energy-work flux. Pasting a thin-cell kinetic flux into the implicit operator
  does not establish the correct moving diffusion limit.
- Accumulate accepted face fluxes transactionally, including interval retries.
  Never infer escaped or injected momentum from a residual.
- Replace the periodic-only global balance and conserved-total projection in
  `ImplicitMomentTransport.cpp` with balances that include independently summed
  accepted boundary fluxes. Updating face ghosts alone would otherwise project
  genuine escaped radiation back into the domain. The projected energy-minus-
  work flux must use the same boundary face velocity and momentum flux as the
  physical energy balance.
- Retain particle-loss/reflection guards until particle-owned pending impulse
  and work have corresponding boundary rules and independent tests.

Before enabling: verify isotropic half-space limits, oblique and grazing beams,
specular wall pressure, cancellation of paired interior fluxes, nonlinear
rejection rollback, thick-limit refinement against a qualified FLASH reference,
and moving-material energy/momentum inventories under restart and decomposition.
