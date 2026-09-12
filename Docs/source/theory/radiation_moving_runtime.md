# Experimental gray moving-material runtime

Status: private qualification branch; not a declaration that the complete
radiation module is production-ready. The public PR stays WIP. This runtime
path now calls the coupled M1 solver from the actual native PIC evolution loop.

## Supported contract

Select `radiation_transport.diffusion_solver = coupled_moment`, with diffusion,
LTE and momentum coupling enabled and `momentum_carry = particle`. Initially:

- one gray group, one kinetic ion species and an empty photon species;
- explicit native hybrid-PIC evolution, double fields and particles;
- native B-spline particle shapes 1-4, fixed periodic Cartesian grid, no EB;
- native ideal or fixed-charge analytic latent electron caloric response;
- gray electron-state opacity or analytic coefficients, no material/species or
  spectral opacity tables; transport extinction must be at least absorption;
- no packet injection/conversion, collisions, species changes or moving window;
- material speed limited to 0.01c by the coupled source/transport solver.

The Planck coefficient is gray absorption. In this explicitly selected model,
the transport coefficient means total gray extinction; their difference is
isotropic coherent scattering. A negative difference is rejected, not clipped.
The existing native cell-state adapter supplies the caloric-weighted cell
temperature and its gray LTE bath. Truly empty cells have zero material source;
invalid nonempty material is not silently turned into vacuum.

Old scalar diffusion and packet modes retain their existing guards and defaults.
This mode does not implement finite-frequency-group frame redistribution or
packet conversion simply because it stores a radiation momentum state.

## State, ordering and transactions

The energy remains the cell-integrated `radiation_diffusion_energy` [J/cell].
Three additional scalar fields, `radiation_moment_qx`, `_qy`, `_qz`, store
cell-integrated F/c [J/cell]; physical radiation momentum is their sum divided
by c. `initial_moment_flux_ratio` supplies three Cartesian components of F/(cE)
at a cold start. Its norm must not exceed one. Each q component is independently
plottable and mandatory in a moment-model checkpoint.

The radiation stage precedes the normal PIC push. It advances radiation,
particle velocities/carry and native electron caloric state on private copies,
including whole-source-interval retries. Only an accepted interval commits.
Native PIC then performs its existing particle drift, density/current
deposition, electron transport/pressure work and field evolution. No FLASH
fluid closure is substituted for that native update. This is an operator-split
runtime, not a claim of second-order coupling or rollback of arbitrary failures
in the subsequent native PIC step.

Actual particle-center kinetic work and momentum changes are returned through
`CoupledMomentExchange` only on success. Failed attempts preserve prior output
fields, including ghosts; accepted substeps accumulate into private diagnostics.
Runtime material diagnostics use these actual fields, not a redistributed
source-origin allocation. Source-origin work remains the shape-adjoint pairing.

`RadiationMomentModel_data.txt` identifies the gray M1 model and particle order.
The v2 schema records that order explicitly; the earlier private linear v1
schema is accepted only with particle order one. Changing order at restart is
not an implicit conversion.
Restart must preserve it; neither dropping q nor inventing it from an old scalar
checkpoint is an implicit conversion. The existing particle-owned carry schema
also remains mandatory. Changed-rank restart uses the registered field and
particle redistribution paths.

## Current evidence and remaining work

The FLASH moving trapped-pulse reference was run and documented first in
`practical_campaign/moving_references_20260909/FLASH_MOVING_PULSE_RESULTS.md`.
It passes separate mesh/time refinement against a Fourier solution. The native
WarpX runtime has completed 800 steps over 0.3 domain lengths at 128, 256 and
512 cells. Continuous-reference profile errors are 1.284%, 0.666% and 0.354%.
This is first-order spatial convergence, less accurate on these meshes than
FLASH's reference discretization. A two-rank run restarted on one rank at step
400 reproduces its step-800 fields and particle trajectory within explicit gates.

The registered CI integration pair uses 400 steps over the same physical time,
including changed-rank restart, and passes in 14.48 seconds locally. The source
and transaction tests remain separate gates, including deliberately failed late
substeps with untouched diagnostic sentinels.

The trapped-pulse case deliberately has high material inertia. Its raw total
energy/momentum residuals and background arithmetic bounds are reported; it is
not a sensitive acceleration/work benchmark. Appreciable moving-interface
force/work, physical boundary inventories, stronger native pressure feedback,
full runtime CUDA qualification and the remaining geometries/groups still need
qualification. Do not infer those results from the trapped-pulse pass.

## Strong-interface development findings

FLASH F-M2 now has a separate recorded 256/512/1024-cell and timestep-refinement
reference, with appreciable radiation work and pressure feedback. The native
linear-shape run conserves radiation + actual ion kinetic + native nodal
caloric + particle-carry energy, but develops particle crossings. A no-radiation
contact control reproduces this behavior: 27.6% peak velocity deviation and
crossings at 256 cells. It is therefore not solely a radiation-force defect.
Cubic-shape controls remain laminar, with 3.32% and 2.02% peak deviations at
256/512 cells; their spurious kinetic changes decrease from 295503 J to 80058 J.
These are material discretization errors, not a reason to substitute FLASH's
fluid closure or to waive source gates.

The receiver now uses the native B-spline nodal factors followed by cell-corner
averaging, with the corresponding 1/2/2/3 ghost layers for orders 1/2/3/4.
Independent cardinal-spline mass/support, particle kick and adjoint-work tests
pass for all four orders in 1D/2D and on one/two MPI ranks. The current combined
CPU selection passes 135 tests in 153.28 seconds. All 11 P40 CUDA particle
tests also pass (14.91 seconds), including orders 1-4 in 1D/2D and carry
ownership. Existing explicit linear and NGP options keep their contracts.

The cubic native F-M2 runs at 256 and 512 cells complete 6400 steps without
particle crossings. Relative raw energy errors are 5.164e-12 and 5.170e-12 of
initial radiation energy. The 256-cell checkpoint at step 3200, restarted from
one rank onto two, reproduces the final particle positions, proper velocities,
radiation moments, nodal density and nodal temperature under explicit gates.

The 1024-cell run conserves energy to 5.169e-12 of initial radiation energy,
but develops particle crossings and a 12.77% peak velocity deviation. Its
no-radiation control also crosses, reaching 10.39% deviation. Thus the fixed
6400-step mesh sequence does **not** qualify converged interface dynamics;
conservation alone is insufficient. A 12800-step no-radiation control still
crosses and reaches 9.06% deviation, so halving the timestep does not remove
the defect. Particle-count and interpolation-order controls are investigating
the material discretization. Increasing cubic sampling from 4 to 16 particles
per cell still gives crossings (9.79% final peak deviation). The quartic control
stays laminar at every recorded time with 1.15% final peak deviation and 21276 J
spurious kinetic change. Its full radiation counterpart is under test; this
control does not qualify radiation dynamics or indefinite material stability.
No control subtraction or tolerance relaxation is applied. The interface
analysis has an explicit `--require-laminar` gate for these cold, single-stream
benchmarks, separate from its inventory checks; this is not a generic ban on
physically valid multistream kinetic simulations.
At 2048 cells the quartic no-radiation control also crosses (5.63% final peak
deviation). A separate diagnostic initializes temperature from the actual
quartic nodal deposition stencil, making initial pressure uniform to 1.52e-13
relative; it still crosses and reaches 6.40%. Initial pressure imbalance is
therefore not the sole cause. The native FV transport explicitly forbids
filtering until charge/energy fluxes share a consistent filtered continuity
equation; that guard remains. No unsupported filter was enabled to hide this
failure, and finer cold-interface production accuracy remains unqualified.
The full 2048-cell radiation run now also fails the laminar gate: final peak
velocity deviation is 8.68%, despite raw energy closure at 5.157e-12 of initial
radiation energy. Its 768898 J kinetic gain is close to the FLASH integrated
value, but that agreement does not qualify the incorrect or contaminated
particle trajectory. The 1024-cell half-dt run stays laminar and changes kinetic
gain from 626679 to 624113 J (about 0.4%); timestep error alone does not account
for the remaining partition discrepancy. The 3200/6400/12800-step sequence at
1024 cells now passes the temporal field gates, with ratios 2.004 (E), 2.005
(qz), 2.022 (charge) and 2.047 (caloric density). Kinetic gains are 631870,
626679 and 624113 J, also showing approximately first-order time convergence.
The spatial and temporal gates are distinct: this does not waive the failed
2048-cell trajectory or establish a converged spatial work partition.
The corresponding strong-interface CUDA runtime studies are still in progress.

`moment_relaxation` is a bounded iteration control in (0,1], default 0.5.
For this weak radiation-inertia case, setting it to one reduced source iterations
from 23 to 3 and an eight-step probe from 4.649 to 0.8013 seconds. The same force,
work and actual conservation gates remain in force; probe field differences
were below 1e-10. `diffusion_solver_verbosity` exposes accepted substeps and the
last source iteration count; values above one also expose inner residuals.

The full native-PIC trapped pulse also passes on a P40 CUDA build. This does not
qualify the subsequent higher-order strong-interface changes on CUDA.

Subsequent cubic 256-cell P40 qualification completes all 6400 interface steps,
passes the same raw inventories and remains laminar. Its energy residual is
5.175e-12 of initial radiation energy. Final CPU/P40 maximum differences,
normalized by the CPU maximum of each quantity, are 5.279e-15 for radiation E,
3.156e-13 for qz, 1.196e-12 for nodal charge, 1.362e-13 for nodal temperature,
and 4.627e-12 for particle proper velocity (matching particle identities).
These are measured backend differences, not a changed-rank restart tolerance
or a claim that the known cubic fine-grid instability is resolved. The refreshed
1D radiation regression selection passes all 212 tests in 156.99 seconds.

The quartic 256/512/1024-cell interface runs are laminar and pass raw inventories
(relative energy errors 5.174e-12, 5.175e-12 and 5.180e-12).
The new three-mesh analysis requires identical physics, shape, sampling and
timestep, rejects cold-particle crossings, and requires an L1 self-refinement
ratio of at least 1.5 for radiation E/qz and native charge/caloric densities.
It rejects the cubic sequence. The quartic field ratios pass: E 1.779, qz 1.824,
charge 1.537 and caloric density 1.761. The fine-pair relative L1 differences
remain 0.470%, 0.637%, 2.044% and 0.371%, respectively. These are self-differences,
not exact-solution errors or evidence of second-order accuracy.
Actual ion kinetic gains are 438422, 527582 and 626679 J. Their successive
increments have not yet entered a decreasing asymptotic sequence. Thus this
field-refinement pass does not close kinetic-work partition accuracy; finer
spatial and temporal qualification remain necessary. The independent FLASH
reference has a 755317 J kinetic gain, with the already documented material
closure differences; agreement must not be manufactured by changing closures.

The full 2D vector case is described in
[radiation_oblique_interface_qualification.md](radiation_oblique_interface_qualification.md).
Its eight-step setup probe is not production evidence. Its analysis reads native
periodic nodal fields in both dimensions and checks the actual vector inventories;
the full 6400-step runs and their refinement evidence are recorded in that report.
The 128x256 checkpoint also reproduces its final trajectory after a one-to-two-GPU
restart. Density-profile accuracy remains incomplete; these passes do not close
the broader geometry, boundary or spectral obligations.

Independent radiation-specific acceleration is now verified separately in
[radiation_moving_beam_verification.md](radiation_moving_beam_verification.md):
actual moving ions gain 0.4435% of the initial radiation energy, travel 0.646
domain lengths and preserve electron caloric energy under coherent scattering.
The 200/400/800-step sequence converges against an independent four-force ODE.
It passes CPU and two-P40 MPI execution, without replacing or waiving the
failed cold-contact tests. MPI/CUDA particle-transfer and ownership tests also
pass all 22 cases in 39.51 seconds.

Four corrupted-checkpoint tests require failure before advancing if the model
record or any one of the three moment fields is absent. All four and their
fresh checkpoint producer pass in 12.08 seconds. Corruption is applied only to
temporary copies; the original checkpoint remains untouched.

## Built-in momentum inventory and diagnostic restart

`RadiationMomentum` remains an impulse ledger by default. In moment mode,
`<diagnostic>.include_moment_inventory=1` appends three instantaneous field
momentum columns, obtained directly from sum(q)/c. It does not reinterpret a
scalar diffusion flux as an independent momentum state. Existing default
columns and legacy checkpoint data remain unchanged. The opt-in output schema
is recorded and must be preserved at restart.

The moving-beam analysis independently compares these columns against plotfile
moments and checks their balance with cumulative actual material impulse and
pending impulse. Its positive changed-rank restart and rejection checks pass
all six tests in 10.73 seconds. The legacy momentum selection passes 27 tests
in 38.01 seconds. An isolated cleanup-on CTest audit, reusing the qualified CPU
executable, passes 19 tests in 15.20 seconds; fixture lifetimes keep checkpoints
alive through both valid and rejected restarts. Only disposable audit outputs
are cleaned, not the production datasets.

The inventory extension also passes all six MPI/CUDA beam, restart and
unsupported-schema checks in 54.13 seconds after rebuilding on P40. The pending
boundary integration contract is recorded in
[radiation_moment_boundary_plan.md](radiation_moment_boundary_plan.md); its
angular algebra checks do not remove any current runtime guard.
