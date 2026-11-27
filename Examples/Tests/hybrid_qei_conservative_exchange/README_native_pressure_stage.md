# Private native RZ pressure timestep

Owner review and the declared 12-test CPU selection are complete. Final
independent Grok review confirmed the midpoint/native-work identity and led
to a separate admission correction: reject EB, non-collocated/multimode
configurations and native gather-boundary fills narrower than one cell.
Caloric volumes are checked for finiteness before native calls. No
native-stage CUDA result is claimed. The four new native-stage cases were
rerun after this correction: 4/4 passed in 2.32 seconds.

`NativeRZPressureStage` provides a callable pressure-only timestep on native
level-zero magnetic fields and independently owned electron internal-energy
and temperature arrays. The fixture passes the actual HybridPIC FV internal
energy and temperature fields. It freezes positive deposited charge, uses the
charge-only ideal EOS, and admits only collocated mode zero with an axis,
stationary outer PEC wall, periodic z and no PML.

This helper is not called by `HybridPICEvolveFields`, `FieldPush`, or the
production RK4/RKF45 owner. It adds no input parameter. The existing
zero-Ampere guard in `HybridPICModel.cpp` remains unchanged.

## Discrete update

Let L be the actual native pressure-Ohm, electric boundary and Faraday map,
including `WarpX::EvolveB`'s magnetic boundary projection. V is the native
caloric node volume; W is the distinct native magnetic quadrature. Shared
nodes and periodic endpoints represent a single degree of freedom.

A half-step predictor uses P0 from the EOS and the conservative secant
magnetic midpoint to obtain Uhalf. The EOS reconstructs Pmid from Uhalf.
The final update is

```
B1 = B0 + dt L Pmid
V (U1 - U0) = -dt diag(Pmid) L^T W (B0 + B1) / (2 mu0)
T1 = EOS.inverse(rho, U1)
```

Both forward updates call the real native Ohm/BC/Faraday path. Only its
isolated pressure electric output enters Faraday. The Ampere current is
measured as a diagnostic, never substituted for the weighted transpose.
The final work uses the actual native B1; the standalone qualified primitive
is only a rejection witness. The predictor supplies second-order pressure
feedback, while the endpoint midpoint includes the exact magnetic cross term.

The stage snapshots all three B and E components plus U/T, including ghosts.
All physical candidate rejections are collective. A scope guard restores the
snapshot on rejected or recoverably throwing calls; exceptions propagate.
Successful calls restore E and the other B components, then commit Btheta,
U and T together. Scalar physical ghosts are even; Btheta is odd at the axis
and even at the outer wall. Charge below the native Ohm floor is rejected,
not floored. Backing-array aliasing is outside the caller contract.

## Qualification

The four new CTest entries are:

- `wave5_rz_native_pressure_stage`: four 8-by-8 boxes, serial.
- `wave5_rz_native_pressure_stage_single_box`: one 16-by-16 box, serial.
- `wave5_rz_native_pressure_stage_mpi`: four boxes, two MPI ranks.
- `wave5_rz_native_pressure_stage_mpi_small_boxes`: sixteen 4-by-4 boxes,
  two MPI ranks.

Each runs a nonzero-Ampere pressure exchange, a radial-only null-curl
control, huge-step caloric rejection, rank-local invalid-volume rejection,
and a Btheta/U/T save/reload followed by continuation. The actual native
material fields are checked for byte-identical rollback including ghosts.
All B and E components are also checked on rejection. Reload continuation
matches an uninterrupted run bitwise including Btheta/U/T ghosts.

An independent local oracle measures every column of L using pressure basis
perturbations through native Ohm/BC/Faraday. It predicts the half-step
pressure and checks each final node's work and magnetic update without
calling the production pressure-work stencil. Temporal refinement uses
16, 32 and 128 equal steps over the same interval; the observed coarse/fine
error ratios are 4.0526 for B and 4.6839 for U. The initial four/eight-step
sequence was outside the selected convergence criterion; the step sequence
was refined with the assertions unchanged.

Representative four-box results (double precision):

- Maximum initial Ampere current: 174.3865685 A/m2.
- Magnetic/caloric exchange: +9.002656474e-6 J / -9.002656474e-6 J.
- Global defect: approximately 7.18e-17 J.
- Maximum independent local work: 6.079946831e-6 J; discrepancy
  4.427972679e-18 J.
- Independent predicted-pressure magnetic discrepancy: below 6e-21 T.

The eight existing pressure operator/stage/primitive tests also pass in the
RZ-only CPU build. No existing analysis tolerance or checksum was changed.
Build options: Release, RZ, OMP, MPI, double field/particle precision,
APP and LIB on; EB, FFT, QED, openPMD, Python, Singularity-EOS and IPO off.
AMReX comes from the supplied local 26.09 source tree. Compilation uses two
jobs and an in-worktree build directory. CUDA qualification is separate.

## Remaining integration boundary

The production owner still stores only B in its RK state. A rejected RKF45
substep must restore the last accepted B/U/T state and associated ledgers;
an RK4 failure may restart an entire half-step, including earlier accepted
substeps. Neither scope is activated here. The state I/O fixture does not
test WarpX's complete checkpoint schema, particle history or RK restart.

Hall, resistive, ion-current, hyper-resistive and external-field increments
need distinct, time-consistent work attribution before pressure can join
the production field integrator. Moving charge/composition, caloric remap,
pressure-advection/enthalpy flux, vacuum, nonideal EOS and AMR are outside
this qualification. There is no fault injection into an actual native call;
the exception restoration is enforced structurally by the scope guard.
Process/device failure or failure while restoring is not recoverable through
this in-memory transaction.
