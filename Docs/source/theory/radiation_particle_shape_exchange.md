# Particle-shape-consistent radiation force: next interface contract

Status (9 September 2026): implemented as an explicitly selected private
transaction/solver option. Nearest-cell (NGP) assignment remains the transaction
default; the experimental `coupled_moment` runtime selects the native shape.
See [runtime qualification](radiation_moving_runtime.md) for supported scope,
strong-interface results and the unresolved fine-grid material instability.
All orders pass the independent 1D/2D CPU and P40 CUDA transfer tests.

`ParticleImpulseAssignment::LinearNodalCellAverage` retains its explicit linear
contract. `NativeNodalCellAverage` extends the same adjoint construction to native
B-spline orders 1-4. For order p, the cell shape is half the sum of the native
nodal weights on the cell's two corners in each dimension (tensor product),
with p+2 cells of support per axis. The ghost requirements are 1/2/2/3 for
orders 1/2/3/4. Mass, requested impulse and source-origin work use those ghost
layers with periodic/shared-grid summation and filling.
Actual work, actual impulse and carry changes retain particle-center ownership.
`WorkPartitionResidual()` records source work minus the independently evaluated
particle finite work, with a per-particle uncancelled-operand rounding gate.

Initial linear qualification added four CTest cases (1D/2D, one/two MPI ranks).
The current order-1-to-4 cases each cover 96 collapsed-cloud configurations:
fractional locations (including nonbinary fractions), mesh/grid and
periodic edges, empty ranks, opposing and exactly cancelling forces, finite and
sub-ULP kicks. References explicitly deposit to charge nodes and average cell
corners; they do not call the implementation stencil. Tests check actual and
carried momentum after commit as well as the global work ledger. All four and
the four existing NGP ownership/restart cases pass. Two 400-step spatial coupled
runs also pass unchanged endpoint/conservation gates: 1D drag (equilibrium error
2.174e-9, actual energy residual 2.672e-16) and 2D native-electron thermal coupling
(equilibrium error 1.016e-9, actual energy residual 7.604e-13).

The nonuniform oracle additionally exports raw before/after particle records,
independently reconstructs the nodal/cell-average weights and secant work frame,
and checks each particle's requested increment against the gathered radiation
impulse. It keeps the global actual inventory check and the native nodal caloric
and gray four-force gates. Source-cell heat subtracts adjoint source-cell work,
not particle-center deposited kinetic work. Complementary 1D source and 2D
spatial cases pass on one/two ranks; together with the sustained cases this
ten-test CPU selection passes in 12.44 seconds. The broader 121-test CPU selection
(old source/caloric/refinement gates plus shaped transfer/coupling) passes in
151.87 seconds. Two additional 2D shaped-ownership tests pass: a weak spatially
varying kick, ballistic translation/redistribution, checkpoint, one-to-two-rank
restart and a continued representable kick, with particle-resolved momentum and
work/carry checks. The unchanged NGP tests run alongside them.

These are discrete transfer and uniform sustained-coupling tests, not a moving
density-edge benchmark. Mixed clouds/species, physical boundaries, 3D and RZ
remain separate qualification work. Runtime integration, strong-interface
restart and incomplete refinement evidence are recorded in the linked report.

## Why the existing receiver support is insufficient

The hybrid material state samples nodal charge density at cell corners (see
the native cell-state construction in `RadiationTransport.cpp`). With linear
particle shapes, a particle contributes to both neighboring charge nodes. A
cell reconstructed from those nodes can therefore contain material opacity
without containing an NGP particle center. `ParticleImpulseTransaction::Stage`
rejects nonzero impulse in such an empty NGP receiver cell in its default mode.

Do not fix this by zeroing the opacity, discarding that impulse, or assigning it
to an arbitrary distant particle. The force receiver support must match the
material-density construction. This is about matter moving through a fixed
mesh, not moving the mesh or introducing a FLASH fluid closure.

## Cartesian linear-shape construction

Let a particle lie a fraction xi along cell j. Linear nodal deposition followed
by arithmetic cell-corner averaging gives these cell-integrated shape weights:

    S_(j-1,p) = (1-xi)/2
    S_(j,p)   = 1/2
    S_(j+1,p) = xi/2.

Their sum is one. In Cartesian multiple dimensions, take the tensor product.
This is the interior/periodic construction; physical-boundary dual volumes
require a separate boundary construction.
For a single fixed-charge material, this has the same support as the native
cell electron-density reconstruction. For mixtures, receiver mass and charge
density have different weights; all opacity-bearing material must have an
explicit, physically justified momentum receiver. Equal support does not imply
species-resolved opacity or a separate comoving frame for each species.

For macroparticle rest mass m_p (including statistical weight), define

    M_i = sum_p m_p S_(i,p)
    delta u_p = sum_i S_(i,p) I_i/M_i,

where I_i is requested physical impulse integrated over cell i, and u is
proper velocity. Then, before floating-point rounding,

    sum_p m_p delta u_p = sum_i I_i.

Reject nonzero I_i when M_i is zero. Do not replace M_i with a floor. Gather
all contributions before evaluating one particle candidate; applying the
different cell kicks sequentially would make finite work depend on ordering.
Existing particle-owned momentum/work carry must remain attached to that
particle through migration and restart.

## Work is the adjoint pairing, not an interpolated energy correction

For each particle, form the finite-kick secant velocity v*_p corresponding to
its total requested increment. Its stable scalar requested kinetic work obeys

    W_p = m_p v*_p . delta u_p.

The source-cell work and velocity are therefore

    W_i  = sum_p m_p S_(i,p) v*_p . I_i/M_i
    v*_i = sum_p m_p S_(i,p) v*_p/M_i.

In exact arithmetic, sum_i W_i = sum_p W_p and W_i = v*_i.I_i. Retain the
particle scalar finite-work calculation as the independent authority. Opposing
cell forces can nearly cancel at one particle, so computing only a dot product
of a rounded averaged velocity is insufficient. Measure and bound the rounding
difference between the source-origin work partition and the independent
particle total; do not introduce a freely adjustable energy reservoir.

The radiation source can use v*_i as its discrete work frame, subject to the
existing low-velocity and single-bulk-frame assumptions. Native electron heat
still uses the configured EOS and old-Cv-weighted nodal equation. This does not
replace kinetic ion distributions with a hydrodynamic closure.

## Do not confuse source origin with deposited particle inventory

With overlapping shapes, a particle receives impulse from several cells.
Depositing its actual velocity change back to the mesh is **not** the identity
operator on the source-cell impulse:

    delta P_deposited = S diag(m) S^T diag(1/M) I.

Only its global sum equals the requested global impulse. Thus an exact
cell-by-cell identity between radiation loss and a deposited particle moment
would be the wrong test for this finite-size particle coupling. It must not be
made to look exact by relabeling a source allocation as an actual local kinetic
inventory.

Keep these diagnostics distinct:

- requested impulse/work by radiation source cell;
- actual particle momentum/kinetic changes deposited at particle locations;
- particle-owned numerical carry changes, deposited with a declared convention;
- independent global particle inventories and bounded arithmetic residuals.

Qualify the transfer on individual particles against independently reconstructed
shape weights, the global actual energy/momentum identities, and spatial
refinement of the interface response. Existing NGP tests keep their current
meaning and thresholds; new shape tests need this stronger, appropriate
contract rather than relaxed versions of an inapplicable local identity.

## Transaction and parallel requirements

1. Deposit mass into ghosted cell fields using non-SIMD particle scatter loops
   (`amrex::For`) and GPU atomics, then sum shared/periodic contributions.
2. Fill mass and impulse ghosts before gathering across a particle-grid edge.
   Each physical particle must be processed once, including on empty ranks.
3. Stage one total candidate per particle, without live writes. Aggregate work
   and carry diagnostics with the same declared ownership conventions.
4. Validate native caloric response, radiation equations, actual inventories and
   finite carry before committing. A failed retry must preserve every live
   particle and supplied field, including ghosts.
5. Keep physical boundary export separate from periodic ghost summation. A
   boundary flux cannot be inferred from a global residual.

Radial volume factors, rotation of vector bases, angular momentum and unmatched
material/receiver species need their own
derivations and gates. The Cartesian formula above is not authority to enable
those paths.

## Required qualification before interface integration

- A particle whose charge cloud overlaps a cell without a particle center:
  nonzero force reaches the real particle, with no discarded opacity or impulse.
- Independently reconstructed weights at fractional cell positions, grid edges
  and periodic edges; mixed masses and both force signs.
- Opposing forces on one overlapping cloud, including near-cancellation and
  sub-ULP total kicks; stable scalar work and bounded particle carry.
- One/two/more-rank layouts, empty ranks, CPU/CUDA, migration and restart.
- A moving density edge under sustained radiation: compare actual impulse,
  work, native heating and interface profiles under mesh/time refinement.
- Practical foil/cloud reference runs with FLASH performed and documented
  before the corresponding production WarpX comparison, while distinguishing
  native hybrid-PIC and FLASH caloric/pressure closures.

The standalone stencil and transaction checks are prerequisites, not substitutes
for the moving foil, oblique interface, 3D cloud and radial production cases in
the main implementation plan.
