# Native material-wall prerequisite for moving radiation coupling

Status: experimental 1D PEC pressure-work implementation. CPU conservation and
changed-rank restart pass; P40 conservation passes, but strict GPU trajectory
comparisons remain failed and unresolved. Radiation is disabled in this control.
It does not enable radiation carry at physical particle boundaries.

## What the control found

The native conservative pressure-work path originally rejected nonperiodic
geometry because its pressure-energy adjoint was periodic-only. Initial-only
deposition with pressure work disabled nevertheless passed: fourth-order
particle mass and the physical half-volume-weighted nodal charge agreed to
1.90e-16 relative error. This was a zero-step check, not an evolving solution.

The first boundary-adjoint attempt conserved total energy but failed the EOS
gate at step 93. At step 92 the wall-node electron temperature had fallen to
0.435 eV, while its neighbor remained at 60.6 eV. The existing pressure boundary
copied the neighbor's pressure onto the wall node even though the wall node had
its own evolving half-volume energy. Debiting that borrowed pressure exhausted
the local energy. Global energy closure alone did not detect this problem.

The experimental `hybrid_pic_model.conservative_pressure_work_pec=1` path now:

- Retains the physical wall-node EOS pressure and fills pressure ghosts evenly.
- Applies the same PEC mask/parity to isolated pressure E as to total gathered E.
- Folds work-current ghost deposits with the transpose electric-field parity,
  before the ordinary inter-FAB/MPI sum. This is not the physical-current wall rule.
- Uses the negative transpose pressure gradient with physical nodal volume
  weights, including half-volume wall nodes, for the electron energy debit.

In discrete notation, particle pressure work contains
`-dt * J_work^T C M G P * cell_volume`. Here C extends the pressure electric
field to gather ghosts, M is the frozen masked reciprocal charge density and G
includes the pressure-gradient/wall-node constraints. The electron update uses
`delta U_j = dt P_j [W^-1 G^T M C^T J_work]_j`, with W the nodal control-volume
weights. This pairs
actual Boris work with electron energy; it does not introduce a fluid-ion
closure or a global energy correction.

Legacy behavior is unchanged without the new option. The option is currently
restricted to 1D, stationary PEC fields and reflecting particles. Radiation's
particle-carry and shaped-force periodic guards remain unchanged.

## Executed native control

`inputs_base_1d_material_wall_control` starts uniform kinetic protons at 90 km/s,
rho=100 kg/m^3 and native electrons at 100 eV, with the existing fixed-charge
latent-energy EOS. It uses 128 cells, four particles/cell, fourth-order shapes,
dt=1 ps and 3200 steps over a 1 mm interval. Integrals assume unit transverse
area. Opposing ion streams after reflection are retained as kinetic states;
they are not required to match a FLASH hydrodynamic trajectory.

| Run | Ion kinetic-energy change [J] | Electron-energy change [J] | Relative total-energy residual |
| --- | --- | --- | --- |
| CPU, uninterrupted | -311690347.2246701 | 311690347.22426295 | 2.19e-13 |
| CPU, one-to-two-rank restart at step 1600 | -311690347.2246701 | 311690347.22426295 | 2.19e-13 |
| P40 CUDA, uninterrupted | -311690347.2248559 | 311690347.22446644 | 2.09e-13 |

Final mass/charge errors are below 2e-16 for these runs. The fraction of ions
with negative longitudinal momentum is 0.3828125; this is a phase-space measure,
not an inferred count of wall events. The CPU restart passes the strict final
particle/field checks. The five native wall/CPU restart/input-rejection CTests
pass in 11.28 seconds. Existing coupled-source cases also passed; one initial
negative-test failure was a line-wrapped error-message match, corrected by
shortening the error message without changing the rejected condition.

The local energy/phase-space plot is generated at
`build-production-cpu/material-wall-even-pressure/material_wall_control.png`;
it is a qualification artifact, not bundled with the source documentation.

Reproduce the full control from an empty output directory under the build root:

```sh
OMP_NUM_THREADS=1 AMREX_INPUTS_FILE_PREFIX=/path/to/worktree/Examples/Tests/radiation_transport/ \
  ../bin/warpx.1d.MPI.OMP.DP.PDP inputs_base_1d_material_wall_control \
  hybrid_pic_model.conservative_pressure_work_pec=1
python ../../Examples/Tests/radiation_transport/analysis_material_wall_control.py . --plot
```

## Remaining strict trajectory failures

No trajectory tolerance has been changed. The separate CPU–GPU comparison used
a maximum velocity bound of 9e-6 m/s; one final particle differed by 2.3846e-5
m/s. Position, charge and temperature met their comparison bounds. Normalized
velocity differences grew from about 3.4e-14 at step 400 to 2.65e-10 at step
3200. This growth is consistent with sensitivity to small perturbations, but
does not by itself prove whether its origin is physical or numerical.

The one-to-two-GPU restart also fails the stricter 9e-8 m/s velocity assertion:
its maximum difference is 1.6442e-7 m/s. Its position difference is 3.21e-17 m,
charge difference 0.00848 C/m^3, and temperature difference 2.37e-7 K. The analysis
now saves failure metrics before propagating assertions; a failed comparison
is explicitly marked `all_gates_passed=false`. These runs are not restart passes.

CPU timestep refinement at fixed mesh and end time gives:

| Steps | Kinetic-energy change [J] | Relative energy residual |
| --- | --- | --- |
| 3200 | -311690347.2246701 | 2.19e-13 |
| 6400 | -312308857.12789464 | 4.34e-13 |
| 12800 | -312482321.9287864 | 8.69e-13 |

Successive weighted-L1 self-differences decrease by factors 2.50 for charge,
3.45 for temperature and 2.39 for particle velocity. The last mean velocity
self-difference is still 0.01408 of the initial drift (about 1.27 km/s). This
establishes a discretization-error scale, not permission to relax the strict
GPU comparisons. An error-budget decision has been requested from the user;
the current assertions and failed evidence remain intact.

## Artifacts and next work

### Diagnostic state-mutation regression

While investigating the strict trajectory comparisons, a separate defect was
reproduced: plotfile writing converted live particle proper velocity to SI
momentum and back. Floating-point multiplication by mass and reciprocal mass
does not preserve all input bits. Diagnostic output therefore changed the live
state without a corresponding update to the particle-owned deferred work.

The writer now filters/copies native particles first and converts only the output
copy to SI. Parser filters use native normalized proper velocity. The regression
writes unfiltered, uniform-stride and parser-selected output and requires exact
equality of every live particle real attribute after each write, including four
nonzero deferred impulse/work components. Separate output analysis checks SI
momentum magnitude and exact selected-particle attributes. This test failed on
the original writer (`plot-immutable-red/`) and passes after the change
(`plot-immutable-green/`); both registered CPU checks pass in 2.22 seconds.

An additional 1D MPI/OpenMP build enables openPMD 0.17.1 with its JSON backend.
The plotfile/openPMD live-state checks and exported-data analyses pass 4/4 in
3.31 seconds. The two formats agree exactly on SI momentum, position, weight,
nonzero carry and uniform/parser selections. This tests ordinary live-container
output, not the reusable pinned back-transformed buffer branch or every
openPMD backend. Earlier failed wall comparisons and unchanged tolerance gates
remain evidence, not superseded passes.

Fresh post-fix CPU runs (`material-wall-readonly/` and
`material-wall-readonly-restart/`) pass all physics gates; the one-to-two-rank
restart has exactly zero final differences in position, proper velocity, charge
and temperature. CPU energy error is 2.1860e-13. The selected CPU regression set
passes 72/72 in 56.75 seconds; the two diagnostic checks also pass on P40 in
4.34 seconds.

However, the fresh P40 comparisons still fail. The maximum CPU/P40 velocity
difference normalized to 90 km/s is 2.55608e-10, above the original 1e-10 bound.
The P40 one-to-two-rank restart differs by 1.60856e-7 m/s, above 9e-8 m/s.
Its position, charge and temperature differences are 2.94903e-17 m,
0.00367355 C/m^3 and 1.49943e-7 K, respectively. Thus the diagnostic defect
was real but does not explain away the failed trajectory gates.
`analysis_material_wall_backends.py` makes the original cross-backend checks
reproducible and saves the failure history in `backend_comparison.json`.
It also reproduces the failure on the preserved pre-fix outputs. No bounds
were increased. Post-fix P40 outputs are copied locally under
`p40-material-wall-readonly/` and `p40-material-wall-readonly-restart/`.

A same-rank, one-P40 restart from the same step-1600 checkpoint also fails the
9e-8 m/s gate: its maximum velocity difference is 1.07372e-7 m/s. Position,
charge and temperature differences are 2.61293e-17 m, 0.0101070 C/m^3 and
4.40516e-7 K. Thus changed MPI rank count is not necessary for this discrepancy;
the evidence does not yet distinguish restart reconstruction, GPU deposition
ordering and subsequent kinetic amplification. This deliberate same-rank
control is retained under the worker's `material-wall-readonly-restart-single/`.

The input left sorting at backend defaults: every four steps on GPU, disabled
on CPU. An additional CPU run explicitly setting `warpx.sort_intervals=4`
also conserves energy but still fails the comparison to the P40 baseline:
maximum normalized velocity difference 3.24776e-10. Matching sorting schedules
therefore does not resolve the failure. This control is separate from the
original cases, under `material-wall-readonly-sort4/`; its comparison has a
separate `material-wall-readonly-sort4-backends.json` report with source paths.
The added negative input test also confirms that this experimental PEC
pressure-work option rejects an absorbing particle face (0.65 seconds).

The final CUDA diagnostic rebuild passes both plotfile checks in 4.86 seconds.
An earlier test invocation overlapped executable linking and could not start
(`permission denied`); its log is preserved as `immutable-build-overlap-test.log`.
That launch failure was not a numerical test result.

A restore-only P40 run (`max_step=1600`, checkpoint at step 1600) reproduces
particle IDs, positions, momenta, weights and checkpointed electron temperature
exactly. Its raw `rho_fp` output is zero: hybrid charge/current initialization
is performed inside evolution by `HybridPICInitializeRhoJandB`, which this
no-advance probe does not reach. Therefore this probe verifies particle and
temperature serialization, not reconstructed charge/current or restart dynamics.
The next useful isolation point is immediately after hybrid redeposition and
before the first resumed push; do not interpret the zero raw diagnostic as
evidence that the evolving solver uses zero charge.

### Isolation at restart redeposition

The existing particle test executable now has a qualification-only
`test.hybrid_restore_probe=1` mode. It loads an ordinary native-material
checkpoint without adding radiation test attributes, invokes the real
`HybridPICInitializeRhoJandB` bootstrap, asserts exact preservation of all
particle real attributes and writes `probe_rho`, `probe_temperature` and
`probe_current_{0,1,2}` MultiFabs. It does not advance the simulation or declare
the full wall test passed.

At step 1600 the CPU bootstrap reconstructs all charge/current values exactly.
On P40 the largest charge difference is 7.62939e-6 C/m^3 against a maximum
2.41685e10 C/m^3; longitudinal current differs by 0.25 A/m^2 against
8.62095e14 A/m^2. Transverse currents remain exactly zero and the particle-state
assertion passes. These are relative perturbations of about 3.2e-16 and 2.9e-16
before the first resumed push, not proof that all later error comes from this
one operation. Results reside in `material-wall-restore-probe/` and
`p40-material-wall-restore-probe/` locally.

The current initialization comment claiming deterministic reconstruction is
therefore too strong on GPU. The next implementation candidate is preserving
the solver's deposited moment history in checkpoints. This requires an explicit
valid-history marker: a step-zero checkpoint can precede hybrid bootstrap, so
the mere existence of temporary moment fields must not authorize using their
initial zero values. Legacy checkpoints must still reconstruct their missing
history, partial new history must be rejected, and species-resolved/auxiliary
deposits must remain consistent. Restart fidelity cannot be fixed by blindly
skipping all redeposition. This change is not yet implemented or qualified.

### Deposited-history checkpoint implementation

The next increment adds `HybridMomentHistory.txt` plus independent MultiFabs
for total charge/current, every allocated charged-species density, material
ion-count densities and the species-density sum. This is solver history, not
a fitted correction to a final conservation residual. Existing field and
particle checkpoints remain unchanged.

The manifest records version, step, history validity and expected fields. Its
deposition contract includes particle shape, deposition method, filtering and
communication precision, species identity, charge, mass and deposition flag.
A new restart restores the complete history and consumes it once during hybrid
bootstrap, then copies it into the usual old-time buffers. Derived pressure is
rebuilt from restored charge and native checkpointed temperature. Auxiliary
trajectory fluxes are still deposited by the first actual evolved step.

Before bootstrap, checkpoints explicitly mark history invalid and write no
history MultiFabs. Such checkpoints reconstruct deposits normally. A legacy
checkpoint with neither manifest nor history fields does likewise. Missing
members of a new history, orphaned history without its manifest, mismatched
deposition settings and trailing metadata reject startup; they do not silently
fall back to reconstruction.

The seven-case schema check passes new, step-zero and legacy continuation and
the expected missing-manifest, missing-field, changed-shape and trailing-data
rejections. The full CPU wall/changed-rank restart plus native moving-beam and
trapped-pulse integration/restart set passes 15/15 in 40.13 seconds. The earlier
run could not launch eight moving tests after their output directories were
archived; reconfiguration recreated the directories. That launch failure is
retained in `moment-history-regression.log`, and the completed run is in
`moment-history-regression2.log`. No numerical assertion was changed.

CPU wall conservation remains 2.1860e-13 relative, with the same kinetic and
electron-energy changes as before this checkpoint change. The 2D oblique
integration/restart checks pass 4/4 in 5.46 seconds, and the existing native
ionization/restart checks pass 4/4 in 3.37 seconds. Both 2D and RZ executables
compile; this is not radial moving-radiation physics qualification.

The seven schema cases pass on P40 as well. The extended probe additionally
compares stored double-precision bits (including signed zero), not just a
floating-point tolerance. Both CPU and P40 restore charge, native temperature
and all three current components bitwise exactly, without changing particles.
Thus the earlier GPU bootstrap redeposition perturbation is eliminated.

The complete wall trajectories nevertheless remain failed under the original
bounds. P40 full-run conservation is 2.09126e-13 relative. Its CPU/P40 maximum
velocity difference normalized to 90 km/s is 2.87421e-10 (bound 1e-10).
The one-to-two-GPU restart differs by 2.08868e-7 m/s (bound 9e-8 m/s); its
position, charge and temperature differences are 3.68629e-17 m,
0.0090332 C/m^3 and 2.44007e-7 K. A same-rank, one-GPU restart also fails,
at 2.81452e-7 m/s. Exact restart bootstrap is therefore insufficient to meet
the final trajectory gate. No threshold was changed, and no failing run is
reclassified as passed. A decision about independently validating a physics-based
reproducibility budget has been requested; it is not authorization to relax one.

Final-schema artifacts are under `material-wall-history-final/` and
`p40-material-wall-history-final/`, with corresponding `-probe` and `-restart`
directories. The worker also retains `material-wall-history-final-restart-single/`.
Both source directories retain the schema test's copied checkpoints and logs;
these are private copies, not mutations of the original source checkpoints.

Local artifacts live under `build-production-cpu/`:

- `material-wall-shape4/`: original guard and initial-only deposition checks.
- `material-wall-adjoint-probe/`: the failed borrowed-pressure attempt through step 92.
- `material-wall-even-pressure/` and `material-wall-even-pressure-restart/`: CPU full runs.
- `material-wall-even-pressure-s6400/` and `material-wall-even-pressure-s12800/`: timestep study.
- `p40-material-wall-even-pressure/`: copied full P40 output.
- `material-wall-cpu-gpu-comparison.log` and `material-wall-time-refinement.log`: measured differences.

The worker retains its failed changed-rank GPU comparison under
`build-p40-mpi/material-wall-even-pressure-restart/`. Live radiation-carry
reflection, checkpointed carry-wall diagnostics, physical-edge radiation force
assignment and the coupled radiation/material wall problem remain unfinished.
