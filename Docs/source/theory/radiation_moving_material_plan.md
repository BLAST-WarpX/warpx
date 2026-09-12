# Radiation–moving-material coupling implementation plan

Status: active implementation, 2026-09-09. Baseline private commit
`7cad05976a0918d5941c2c7fb424a22945c400e1`. Nothing in this plan constitutes
production qualification. The public transport PR remains WIP.

Current runtime checkpoint: `80b27c6ff` integrates the opt-in gray periodic
moving-material path into native PIC, including shape orders 1-4, actual
particle work diagnostics and persistent moment fields. See
[radiation_moving_runtime.md](radiation_moving_runtime.md) for the current
evidence, including passing strong-interface changed-rank restart and failing
cubic fine-grid material controls. The historical increments below describe
their state at the time; they do not supersede that current report. Boundary,
conversion, multigroup and radial obligations below remain open.

## Objective and scope

Couple radiation momentum and energy consistently to moving kinetic ions and
native hybrid electron caloric evolution on a fixed mesh. Do not introduce a
FLASH hydrodynamic closure, a moving mesh, or an interface tracker. AMR is not
an exit requirement. Retain stationary behavior and explicit unsupported-input
guards until replacement paths have passed their own physical gates.

Completion is evidence-based, not a six-hour time limit. Each stage below needs
implementation and tests; a primitive helper or stationary comparison alone is
not completion. New functionality stays opt-in during qualification.

## 1. Establish the discrete contract and material ownership

- Audit radiation/PIC ordering, mass/current deposition, native electron energy
  advection, species creation, particle migration, restart, and boundary loss.
- Separate requested radiation impulse, represented ion impulse, deferred
  impulse, actual ion kinetic work, and electron internal-energy exchange.
- Replace cell ownership of unrepresented material impulse with physical
  ownership. Particle attributes are the candidate, but their energy bookkeeping,
  boundary export, injection initialization, resampling and restart compatibility
  must be designed before enabling the path. Unsupported particle operations
  must be guarded rather than silently losing residual state.
- Preserve the vector identity requested + old deferred = realized + new
  deferred. Use stable actual finite-kick kinetic-energy differences, not only
  force dotted with an old velocity. Account for the energy associated with
  delayed realization without withdrawing unrelated material's energy.
- Make invalid updates rejectable before committed particle or material changes.

Gate: finite kicks, sub-ULP kicks, cancellation, mixed species, empty cells,
particle crossing into cells containing different material, and restart/rank
changes preserve the declared energy and momentum identities.

## 2. Select and implement the moving radiation equations

The existing scalar FLD field cannot simply be assigned an arbitrary momentum
reservoir. Compare an explicitly bounded mixed-frame diffusion approximation
with a radiation moment state that evolves physical momentum and stress.
Document the selected closure, frame, velocity/optical-depth envelope, and its
boundary/conversion rules before removing any conversion/force guard.

- Use material-frame opacity and emission with consistent lab-frame exchange.
- Include radiation advection and compression work where the retained ordering
  requires them. Small velocity alone does not establish the diffusion ordering.
- Distinguish gray transforms from finite-frequency-group transforms; retain
  a guard on moving multigroup use until frequency shifts/group-edge exchange
  are implemented and verified. Gray-only progress is not full feature completion.
- Conversion must balance actual sampled packet four-momentum, including finite
  sampling recoil, with the chosen radiation state and material exchange.
- Shared face energy/momentum fluxes must cancel internally and feed independent
  boundary diagnostics. Do not infer boundary transfer from a residual.

Derivation references:
[Krumholz et al., mixed-frame FLD](https://arxiv.org/abs/astro-ph/0611003) and
[Skinner & Ostriker, two moments](https://arxiv.org/abs/1306.0010).
These supply radiation equations, not a replacement for hybrid-PIC material
closure. The already tested FrameTransform helper is only a building block.

Gate: boosted equilibrium, moving absorber/scatterer, trapped-pulse transport,
compression, and actual conversion energy/momentum balances against independent
analytic/manufactured solutions with time and spatial refinement.

## 3. Integrate native material evolution transactionally

- Define which deposited material velocity evaluates the radiation source and
  at what time level; preserve genuine species drifts rather than a hidden
  single-fluid replacement.
- Integrate radiation source, realized ion work and native electron caloric
  response with timestep rejection/retry that restores every affected state.
- Reevaluate density/opacity at the documented stage; ensure electron advection
  and pressure work are neither omitted nor counted twice.
- Extend coupled implicit support only when moving-particle integration passes;
  do not merely remove the current do_not_push assertion.
- Persist authoritative state and cumulative diagnostics across checkpoints.

Gate: closed radiation/material exchange, finite electron thermal inventory,
force reversal, nonlinear opacity/EOS, and intentionally rejected/retried steps
preserve independent physical inventories and reproduce accepted trajectories.

## 4. Practical qualification and handoff

Use different geometries for distinct physics, not copies of one smoke test:

| Case | Purpose | Independent evidence |
| --- | --- | --- |
| 1D translating illuminated foil | Doppler exchange, acceleration, work, crossings | Analytic beam transfer and actual ion/electron/radiation inventories |
| 1D trapped pulse carried by matter | Dynamic diffusion and advection | Translated solution, diffusion broadening, mesh/time convergence |
| 2D oblique moving opacity interface | Vector force, interface transport and conversion | Both momentum components, rotated reference and refined trajectory |
| 3D finite moving cloud | Material ownership, transverse leakage and migration | Boundary fluxes, particle-resolved ownership, MPI decomposition |
| RZ compressing column | Geometric pressure work and trapped radiation | Physical volume/area metrics and compression scaling |
| Moving multigroup absorber | Frequency redistribution and spectral force | Group-edge accounting and independent transformed spectrum |

Run FLASH comparisons only where radiation equations and prescribed material
states/closures match. First document and qualify the FLASH reference and its
visualization; then compare WarpX. Where closures differ, test the corresponding
radiation subproblem or exact conservation law instead of demanding identical
fluid trajectories. Existing stationary F1/F2 results remain regression evidence,
not substitutes for these cases.

Keep quick analytic/regression gates suitable for CI and longer production-shaped
refinement studies outside the default suite. No weakened assertion tolerances,
checksum hand edits, positivity clipping, or hidden residual-based fluxes.

Exit requires the new physical gates, stationary regression suite, CPU/GPU
agreement within declared precision, changed-rank restart, and documented
supported/unsupported inputs. Publish measured errors, plots, failed attempts,
source identities and reproducible commands. Keep the public PR WIP; completion
of private qualification does not authorize merging it.

## Initial audit evidence

Implementation decision: develop an opt-in lab-frame moment state `(E,F/c)`
with M1 closure, retaining physical light speed in the conservation equations.
Do not reinterpret legacy scalar FLD state or its checkpoint as that state.
The initial gray closure/source kernels precede the stiff transport integrator;
their tests cannot qualify diffusion-limit accuracy or moving transport. M1
does not preserve arbitrary crossing beams, which remain a packet-path use case.
The transport discretization must demonstrate the correct optically thick limit
under refinement before production-shaped dynamic-diffusion tests are accepted.

First code increment isolates the actual finite-kick kinetic work already used
by the runtime and tests it against extended-precision independent arithmetic.
180 CPU cases pass (worst scaled error 4.34e-16); the selected 25 existing 1D
momentum/carry/restart CTests pass in 20.07 seconds. Material ownership, moving
source integration and transport remain unimplemented in this increment.

M1 reconstruction now passes the existing 102 frame cases, including comparison
to independently boosted isotropic pressure and an oblique beam (worst combined
error 8.46e-16). The initial implementation rejected that beam: component-first
normalization rounded its reduced flux above one. Taking the scaled hypot norm
of the dimensional flux before dividing by energy fixes the CPU failure. CUDA
still rounded the beam norm above the boundary. The final closure explicitly
accepts an eight-machine-epsilon boundary collar and uses the limiting beam
constitutive square root there, without altering input momentum. Tests check
that a two-epsilon excess is preserved exactly and a 64-epsilon excess is
rejected. No physical assertion threshold was changed. Both initial failures
are recorded here rather than treating the first CPU pass as GPU qualification.

The first increment also implements a pure particle-impulse candidate (not yet
runtime particle attributes): 32 sequences of 4096 sub-ULP kicks, including force
reversals, pass bounded momentum/work-account tests. Worst CPU balance is
7.08e-14. The frame/M1 and work/candidate tests both pass on the local RTX 4000
Ada CUDA build (2/2, 0.61 seconds), as well as CPU (2/2, 0.97 seconds). These
checks do not test particle motion, migration, restart, or production transport.

### Next implementation contract: particle-owned numerical carry

Candidate design for the next increment, not yet wired into runtime:

- Store signed residual momentum and signed residual kinetic-work bookkeeping
  with the receiving particle, in Cartesian coordinates. These are bounded
  representation-error accounts, not a substitute radiation momentum/stress
  field and not a new thermodynamic EOS component.
- For each particle, allocate the new requested momentum `dp` by the declared
  material mass weights. Let `Wreq` be the stable finite-kick work for that new
  unrounded request and `Wactual` the measured work after applying new impulse
  plus old residual to represented particle velocity. Then enforce
  `rp_new = rp_old + dp - dp_actual` and
  `re_new = re_old + Wreq - Wactual`.
- Partition the material energy source into `Wreq` and native electron heating
  at assignment. Do not postpone withdrawing work from radiation until a
  different cell happens to receive the particle. Subsequent realization changes
  the particle-owned signed work account, not an unrelated cell's reservoir.
- Sum actual kinetic, native thermal and the numerical work account in the
  exact exchange ledger. Independently bound the numerical account relative to
  particle precision and absolute transfer; a small conservation residual with
  an unbounded carry is not acceptance. Audit interaction with intervening EM
  pushes, force reversals and repeated cancellation before enabling motion.
- Runtime real components communicate and checkpoint when `comm=1` in this
  checkout. Initialization must explicitly zero new state; injection, smart copy,
  resampling, species-changing processes, particle boundary export and old
  checkpoint compatibility each need a tested rule or a retained guard.
- Candidate particle velocities/carries and material/radiation energy changes
  must be staged together before commit. Repeating an atomic scatter during a
  second commit pass is not an authoritative transaction: its reduction ordering
  can change the previously validated work. Commit the accepted candidate state.

The new implicit moment source should consume a material callback, as the
stationary coupled solver already does. This callback must evaluate genuine
particle kinetic response plus native electron caloric response; replacing the
ions by a single fluid kinetic-energy formula is not acceptable for drifting
species or finite thermal spread.

### Particle adapter increment: implemented and mechanically qualified

`ParticleImpulseTransaction` now stages actual per-particle velocities, Cartesian
momentum carry and signed work carry in separate candidate storage. It exposes
cell-integrated requested/actual work, actual impulse, carry-energy change and
rounding residual. Discarding or rejecting a candidate leaves live state alone;
commit copies the accepted particle candidates without repeating atomic work
reductions. It rejects non-finite requests and aggregates, missing cell mass,
duplicate species, incompatible layouts and unbounded representation accounts.
The caller still must stage and accept its radiation/native caloric state before
calling commit. This is not yet a complete radiation-stage transaction.

Communicated runtime attributes are registered before injection and initialized
to zero. The low-level adapter tests use four particles per cell, apply sub-ULP
impulses to one half of the material, move the particles across half the periodic
domain with the actual WarpX position pusher, redistribute, checkpoint, and
continue with a representable impulse in the relocated owners' new cells.
They check every particle's expected position-dependent ownership and total
particle count, not only a global sum that could hide reassignment.

Measured checks on this increment:

- CPU 1D/2D ownership plus one-to-two-rank restart: 4/4, 2.25 seconds.
- CUDA 2D ownership/checkpoint: 1/1; warm final run 0.53 seconds.
- One-to-four-rank 1D restart with two grids (empty ranks): passed.
- CUDA-written 2D particle checkpoint read by two CPU MPI ranks: passed.
- CPU-written 2D particle checkpoint read on CUDA: passed.
- Existing selected 1D radiation momentum/carry/restart suite: 25/25, 21.71 seconds.

The first changed-rank test failed because the low-level AMReX particle reader
appends to existing particles. This standalone driver had already initialized a
fresh fixture, unlike WarpX's normal restart path. It now explicitly clears that
fixture before reading; a particle-count assertion prevents a vacuous or duplicate
population pass. No physics tolerance was changed.

Reproduce the registered checks with `ctest --test-dir build-production-cpu -R
'^test_[12]d_particle_impulse' --output-on-failure` and `ctest --test-dir
build-coupled-cuda -R '^test_2d_particle_impulse$' --output-on-failure`.
For cross-backend checks, run the destination build's `test_2d_particle_impulse`
with `inputs_base_2d_particle_impulse` and
`test.restart=<source-build>/Examples/Tests/radiation_transport/particle_impulse_2d/checkpoint`;
set `AMREX_INPUTS_FILE_PREFIX` to the source example directory and
`OMP_NUM_THREADS=1`. The CPU changed-rank destination uses `mpiexec -n 2`.

Important limits: these are low-level particle checkpoints, not full radiation
checkpoints. This adapter is not yet called by `RadiationTransport::Advance`.
No new moving-radiation input has been enabled. Remaining integration includes
material boundary export/reflection, resampling/species-change policy, full
checkpoint schema, independent radiation/material diagnostics, native thermal
acceptance, moving source/transport and group/conversion physics. None is
satisfied by the passing ownership tests alone.

### Runtime carry integration increment

An opt-in `radiation_transport.momentum_carry=particle` now calls the staged
adapter from the radiation work paths. It debits newly assigned work immediately,
tracks represented kinetic work separately, and includes the change of the
particle-owned work account in radiation/material energy closure. Live diagnostics
reduce the carried state from particles after motion, not stale Eulerian fields.
Full checkpoints record and require the same carry mode, group count and momentum
species. This is an ownership integration option, not the new moving radiation
equations. The legacy cell mode and unsupported conversion/implicit-motion guards
remain unchanged.

The registered 1D and 2D integration cases are precision regressions with
prescribed lab opacity, not practical production benchmarks. They move ions
through several cells/domain crossings while photon absorption supplies both
represented and deferred impulse. Analysis checks every output step against
analytic attenuation, particle identity and trajectory, actual ion kinetic work,
native electron caloric inventory reconstructed from Te/rho, and stored particle
work/impulse accounts. It does not substitute cumulative source columns for the
independent material inventory.

- CPU integration, changed-rank full restarts and negative guards: 12/12,
  8.98 seconds, including two output-preparation fixtures.
- CUDA 2D full runtime and independent analysis: 2/2, 9.01 seconds; worst
  actual energy-balance error 1.443e-15 against the unchanged 1e-10 gate.
- CPU 1D/2D corresponding worst errors: 1.373e-15 and 1.401e-15.
- Full CUDA radiation checkpoint at step 15 continued on two CPU ranks through
  step 32 and compared to the uninterrupted CUDA trajectory: passed; worst
  actual energy-balance error 5.713e-16. Its outputs are in
  `build-production-cpu/bin/test_2d_radiation_particle_carry_cuda_restart`.
- Existing selected 1D momentum/carry/restart regressions: 25/25, 19.99 seconds.
- Ruff and `git diff --check`: passed.

Failures found and retained in this record:

1. The new boundary rejection test initially matched text across WarpX's wrapped
   error lines. Its expected substring now stays within the actual message line;
   the runtime rejection was working throughout.
2. Replaying a restart in the same output directory appended duplicate diagnostic
   steps. The new regression fixture archives old `diags` into `previous_runs`
   before replay. Analysis still rejects duplicate steps; no rows are discarded
   or tolerated silently.
3. The first full CUDA run crashed before step one while packing runtime particle
   attributes into a plotfile. The polymorphic temporary container's payload had
   been set to pinned memory, but its runtime pointer tables remained device-only
   in AMReX 26.09. Constructing the entire temporary with a pinned allocator fixes
   CPU packing, without changing the shared AMReX checkout. The full CUDA
   trajectory and output analysis then passed.
4. The CUDA build initially selected a Python interpreter without yt. It is now
   configured with the same `/home/tomzhu0225/.venv/bin/python` used by CPU analysis.

Current runtime guard envelope: native hybrid electrons, periodic Cartesian
particle boundaries, fixed species, no collisions/ionization/resampling, and
double field/particle precision. Radiation and particle work are staged together;
the downstream native electron update is not yet part of an all-state retry.
The adapter still uses NGP cell mass sharing, so a new impulse in a particle-empty
cell is rejected even if interpolated electron density is nonzero. Moving vacuum
interfaces need a consistent reaction-force allocation, not just a density floor.
The diffusion/group work call sites are wired, but the new runtime qualification
above exercises streaming absorption; independent diffusion/group carry tests
remain required. Boundary export/reflection, RZ, physical moving-frame sources,
moment transport, conservative packet conversion, finite-group transforms and
the production-shaped FLASH/native campaign are still open.

### Implicit moving gray source and material-feedback preparation

`ImplicitMomentSource.H` now solves the gray M1 source with prescribed material
velocity and temperature. Its rates are `c*alpha_abs*dt` and `c*alpha_scat*dt`.
It uses the exact gray frame transform and an analytic M1 pressure Jacobian.
This is not yet called by the production radiation evolution loop and does not
by itself include finite-mass/thermal feedback or spatial transport.

The nonlinear increment coordinates are

```
z = (delta E - beta dot delta(F/c), delta(F/c))
r0 = z0 + (c*alpha_abs*dt)/gamma * (E_rest - B)
ri = zi + c*dt*G_i,lab
```

This equivalent projection removes the cancellation of large scattering work
terms from the energy equation. Keeping `z0` as an independent unknown also
retains weak caloric exchange alongside large mechanical work. The returned
energy-minus-work transfer is assigned from this increment, not reconstructed
by subtracting two large output energies. Separate state/increment updates avoid
losing stiffly attenuated radiation to cancellation. Compensated bookkeeping
reports radiation-storage and work-projection rounding residuals explicitly.
The acceptance test uses component-wise source scales and derivative-based
floating-point allowances; line search measures residual excess above the same
bounds rather than chasing noise in an already resolved stiff component.

The first implementation failed four stiff scattering cases in line search.
Projecting the energy equation reduced those failures, and the component-wise
line-search treatment resolved the remainder without changing the analytic or
conservation gates. The final test includes axial and diagonal velocities, both
signs, beta through 0.6, rates from 1e-20 through 1e8, independent stationary and
moving-beam backward-Euler solutions, boosted LTE, scattering work, weak heating
with strong scattering, and unchanged-state failure checks. Its 210 cases pass
on CPU and CUDA; worst analytic/ledger error is 1.253e-12 against a 1e-10 gate.
The M1 Jacobian is also checked against centered differences away from the
realizability boundary and against its homogeneous tensor reconstruction.
These are source-equation tests, not a relativistic hybrid-fluid qualification.

The particle adapter can now optionally provide its mass-weighted finite-kick
secant velocity and actual cell mass for the forthcoming material-feedback
iteration. It does not replace the scalar per-particle work ledger with a dot
product of a rounded bulk velocity. CPU 1D/2D ownership/restart checks remain
passing (4/4, 2.03 seconds), including the added velocity/mass checks. The final
CUDA frame/source/particle-adapter selection passes 3/3 in 0.97 seconds.

Next integration must use actual kinetic particles and native nodal caloric
response in scratch state, committing radiation, particle and electron state
only after joint convergence. Optical coefficients that are distinct spectral
means must not be silently interpreted as a physical gray absorption-plus-
scattering pair. Finite-frequency group transforms remain a separate derivation.

Spatial design warning: the stationary-fluid asymptotic correction in
[Bloch et al., ARK-RT](https://arxiv.org/html/2011.13926#S6.SS2) is not, by itself,
a moving-fluid diffusion-limit method. Its paper also records realizability and
opacity-interface issues. A copied static correction or flux clipping would not
satisfy our moving trapped-pulse and interface gates. The spatial discretization
still needs its own moving-limit derivation and verification.

### Finite-mass/native-electron gray source stage (in progress)

`CoupledMomentSource` now stages the gray moment source, actual kinetic-particle
impulses and native nodal caloric response together. Its prescribed-velocity
source is re-evaluated at the resulting finite-kick secant velocity and native
temperature before acceptance. It checks heat and impulse equation residuals,
actual particle work, particle-owned momentum/work carry changes, and raw
energy/momentum balances before committing supplied state. This remains a
source-operator API, not a spatial radiation runtime mode or interval-wide
rollback implementation.

The homogeneous trajectory tests advance 400 source steps with actual particle
motion and redistribution. Independent extended-precision particle and radiation
inventories check every-step conservation at 1e-10. An independently solved
rest-isotropic equilibrium checks acceleration, radiation momentum, thermal
energy and radiation-energy **change** at 1e-8; a zero-update implementation
cannot pass. Ideal and fixed-charge latent native electron closures are covered.
The CPU 1D/2D, one-/two-rank matrix including stiff heating and hot-material
cooling (c*kappa*dt=100) passes 20/20 in 15.01 seconds. A separate 1D heating
run at 1e8 also passes. The expanded five-case CUDA drag/thermal/latent/stiff/
cooling matrix passes 5/5 in 229.27 seconds. CPU particle ownership and MPI
restart regressions were rebuilt with the new carry-momentum inventory field.

Failures found and addressed include a one-ULP lagged-temperature convergence
stall, cancellation in a double-precision test inventory of a small radiation
energy change, and inadmissible fixed-relaxation stiff heating trials. Acceptance
now checks the actual source equation at the candidate native temperature;
the independent inventory uses extended precision; safeguarded backtracking
uses a fixed residual scale rather than a changing source-amplitude denominator.
No physical accuracy gate was relaxed.

The hot 10-eV cooling stress initially exhausted native energy in its frozen-T
guess, then exposed a stalled velocity block throttled by the stiff thermal
relaxation. Scratch-only admissible initialization and independent advancement
of the velocity block once the thermal source resolves now pass this case.
Backtracking measures residual excess above the unchanged acceptance bounds,
so resolved-component noise cannot impede a still-unresolved component. A tiny
line-search step must not snap to a far-away target merely because its fractional
update rounds to zero. Rejected-state checks include every radiation/temperature
component and ghost cell.

Nonuniform material response, all-state interval retry,
moving spatial transport, boundary export, group/conversion physics and the
planned production geometries are not implied by the homogeneous source tests.

### Nonuniform source equations and native caloric remap

The next source qualification uses sinusoidal nodal temperature and radiation
energy profiles with oblique radiation momentum, in 1D/2D with one/two MPI ranks.
It covers ideal electrons, fixed-charge latent energy, and stiff absorption.
An independent Python/extended-precision checker reconstructs the M1 stress,
Lorentz-transformed four-force, actual particle momentum/work and old-capacity-
weighted cell-to-node heat allocation. It imports no production frame, source,
work or caloric helper. The test records stable particle **momentum changes**,
not differences of two large summed momenta. This remains one source stage;
material positions are fixed during it, with no claim of spatial evolution.

The first stiff cases passed the producer fixed-point criterion but failed the
independent backward-Euler equation gate (up to 8.97e-10 versus 1e-10). The
source stage now also checks the unscaled equation on actual candidate radiation
at candidate temperature/velocity, and uses a tighter inner solve. Stiffness
must not amplify a nominally converged iterate past the physical equation gate.
Backtracking permits equation-evaluation roundoff in merit comparisons without
changing any final acceptance gate. The final CPU source matrix passes 44/44
in 22.68 seconds; worst independent nonuniform four-force error is 1.404e-11,
nodal caloric error 4.495e-15, and integrated actual energy error 4.849e-16.
The corresponding CUDA matrix passes 11/11 in 238.99 seconds. A separate
four-rank/two-grid 2D stiff-gradient run passes the same independent analysis;
its state file is byte-identical to the one-rank result. The 1D 400-step heating
stress at c*kappa*dt=1e8 was rerun after strengthening the equation gate and
passes, with actual energy error 1.155e-16 and endpoint error 1.025e-9.

### Private material intervals and late-substep failure

The source interval driver now evaluates every substep on private radiation,
native temperature and kinetic-particle copies. Failed attempts are discarded
in full and restarted at twice the subdivision. A successful attempt checks
the accumulated actual energy balance before committing particles and supplied
fields. No accepted trial substep changes live particles; this is not a scheme
that kicks live ions and later tries to undo their rounded velocities.

The material copy preserves runtime carry attributes, particle identities and
the original memory arena. Before any copy-back it checks layout, particle
identity, weight, position, and unchanged live velocities/carry against the
captured baseline. An intervening live kick makes the commit fail without
overwriting that kick. Empty tiles are not confused with changed particle
ownership. The copy has an explicit memory cost: a complete private selected-
species particle state plus a baseline of mutable velocity/carry components.

Transaction tests deliberately fail the second substep after the first accepted
privately. A callback observer checks that live particle/field state remains
unchanged throughout. Budget exhaustion preserves every supplied field,
including ghosts and sentinel heat output. With one retry, the result matches
a clean four-substep reference and actual particle/native caloric inventories.
Both pure scattering and thermal exchange are included. These are transaction
tests supplementing the 400-step physical source tests, not production spatial
benchmarks. A four-rank/two-grid run also exercises empty-rank participation.

After rebuilding, the combined CPU source/transaction/particle selection passes
56/56 in 30.69 seconds, the full CUDA selection passes 14/14 in 312.63 seconds,
and the existing runtime particle-carry/restart/negative-input selection passes
12/12 in 19.41 seconds. Assertions and checksum expectations were not loosened.

This driver does not yet advance spatial radiation fluxes, move ions, redeposit
density, or refresh derived native pressure. Those operations belong to the
forthcoming full transport/PIC integration; no current guard is removed by
introducing the private source interval.

The next spatial-stage derivation and its unresolved proof obligations are in
[radiation_moving_moment_discretization.md](radiation_moving_moment_discretization.md).
It derives the constant-velocity coherent-scattering diffusion tensor and a
candidate correction of the projected energy/work flux. The private face kernel
and prescribed-material implicit transport prototype now have independent
limiting-flux and spatial/temporal refinement checks; detailed results and their
scope are recorded there. This is not yet a coupled production transport mode.

### Spatial feedback development and qualification

The private spatial integration now includes joint M1 transport, full analytic
pressure-Jacobian corrections, independent source increments, native caloric
equation validation and private interval retries. The CPU selection passes
102/102 in 251.63 seconds. Its log is retained at
`build-production-cpu/coupled-spatial-cpu-102-qualification.log`. This is
periodic Cartesian prototype qualification, not a full production/runtime claim.

The native response can validate a nonlinear trial temperature against the same
old-Cv-weighted nodal energy equation used by the ordinary caloric inverse.
It returns signed energy-density residual, source scale and arithmetic bound,
and independently records the actual old-to-new EOS energy at cell corners.
The 1e-11 nodal equation gate, local projected-heat check, actual particle work,
and raw energy/momentum gates remain. Legacy/source-only callers retain the
ordinary inverse-EOS response. No extra energy reservoir or fluid EOS was added.

This resolved a stiff-feedback failure in which substituting a second rounded
caloric-inverse temperature amplified a tiny discrepancy into a failing
four-force residual. Source finalization now uses direct force evaluation when
c*dt*(kappa_a+kappa_s) <= 1, and independently retained implicit increments at
larger rates. Compensated cellwise conservation corrections avoid subtracting
large global inventories. A source-cell heat check cannot borrow the larger
transport-divergence rounding allowance to hide inconsistent local heating.

Opacity alone also proved insufficient to select the thermal Newton solver.
Accepted temperature secants now detect noncontractive native feedback. A new
unit-rate regression passes in 1D/2D and one/two-rank layouts; independent probes
at rates 0.999999, 1.000001, 10 and 1000 pass the original analysis.

The 204 frame and 420 implicit-source analytic cases pass on CPU and CUDA.
The existing CPU transport-refinement/analytic selection passes 4/4 after
rebuilding. The prior complete coupled-source CPU matrix includes 400-step
moving thermal and drag trajectories, ordinary/latent/stiff nonuniform
four-force and caloric checks, forced private-interval failures, and the native
prescribed-temperature rejection/inventory checks. No assertion tolerance or
checksum expectation was loosened.

CUDA spatial qualification remains open. The earlier batch passed the frame,
source and drag checks, but its 400-step spatial thermal case hit CTest's
1500-second timeout. That obsolete batch was stopped after the solver update;
a rebuilt targeted run now reports timestep progress. The timeout is not a
physical pass. A trial inexact inner solve improved some CPU timings but failed
the unchanged thermal gate and was removed. Large-grid performance still needs
measurement; these very small GPU cases incur substantial launch overhead.

The subsequent local-source initial guess reduces unnecessary constant-mode
Krylov work without changing the transport equations or acceptance bounds. It
is a realizable predictor, not an operator split; failed predictions retain the
original physical state as the guess. The complete CPU selection with this
predictor passes 103/103 in 213.65 seconds, including transport refinement.
The log is `build-production-cpu/coupled-spatial-cpu-103-source-guess.log`.
The 2D 400-step thermal CPU test falls from about 19 seconds to about 2 seconds.
The CUDA trajectory subsequently passes all 400 steps in 173.30 seconds, with
actual energy error 7.608e-13, momentum error 1.063e-15 and equilibrium error
1.016e-9. This supersedes its earlier timeout, not the unfinished nonuniform
CUDA qualification.

The current CPU configuration combines batched conservation reductions, a
40-iteration Krylov budget per coupled Newton correction, and an inner iteration
target of 1/16 of the existing arithmetic acceptance bound. Merely shortening
the Krylov solve failed an independent momentum check and was not accepted;
the additional accuracy headroom passes the full 103-test selection in
133.53 seconds. Final physical bounds and external assertions are unchanged.
A native Linux two-P40 worker is being prepared with the same source and
CUDA 12.4 toolchain for independent backend qualification. The WSL nonuniform
GPU cases remain under test and must not yet be reported as passing.

Subsequent native P40 qualification of the capped/predictor periodic solver
passes the ordinary and latent nonuniform spatial cases in 49.08 and 50.48
seconds, respectively (two independent jobs on the two P40s). Independent
four-force errors are 8.054e-12 and 1.028e-11; native nodal caloric errors are
9.677e-12 and 5.538e-12. This closes those native-CUDA baseline cases, not the
entire GPU matrix or the later shaped-force implementation. The latter is
being rebuilt and qualified separately.

The receiver-support derivation and current qualification are in
[radiation_particle_shape_exchange.md](radiation_particle_shape_exchange.md).
It separates adjoint source-origin work from actual deposited particle
inventories and states interface, migration and parallel proof obligations.
The linear nodal/cell-average assignment is now an opt-in private transaction
and coupled-solver option. Collapsed-cloud transfer checks pass CPU and CUDA;
nonuniform per-particle force, four-force and native caloric checks pass CPU on
one/two ranks, as do complementary 400-step shaped drag/thermal cases. Existing
NGP tests keep their original contract and assertions. This addresses the
empty-center receiver support problem, not the full moving-interface feature.
Moving density-edge benchmarks,
nonperiodic inventories, full PIC/runtime/restart integration and subsequent
group/radial qualification remain open. The public PR remains WIP and unchanged.

- ApplyMaterialImpulse currently retains sub-representable impulse in cell fields.
- ApplyRadiationMomentumWork pairs represented kicks with actual kinetic work,
  but delayed work can be drawn from the cell's current reservoir.
- coupled_implicit explicitly requires do_not_push species and clears its
  material momentum/kinetic diagnostics; stationary tests cannot qualify motion.
- Conversion plus momentum remains guarded because scalar diffusion does not
  retain converted packet momentum. The separate moving-window guard concerns
  mesh-window shifts and must not be confused with material motion.
