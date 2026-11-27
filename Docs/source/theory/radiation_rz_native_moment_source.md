# Native-material meridional RZ moment source

This follow-up connects the low-level RZ M1 operator to **actual finite-mass
particles and native nodal electron thermodynamics**. The source transaction
updates radiation, ion velocity, particle-owned compensation and electron
temperature together. It may include spatial radiation transport.

This page describes the low-level, opt-in meridional source adapter. The bounded
native drifting integration is documented separately in
[the RZ moving-runtime qualification](radiation_rz_moving_runtime.md).
The separate [angular-conservative source building blocks](radiation_rz_angular_source.md)
do not remove the native runtime's meridional guard.
No particle drift, density
advance, packet conversion, current-driven magnetic work or evolved charge
state is performed by this source stage. The native integration tests freeze
positions while evolving the source variables; they are not hohlraum runs.

## Source and momentum ownership

- Particle work is the finite change of relativistic kinetic energy from the
  committed particle velocities. Unrepresentable increments retain their
  particle-owned momentum/energy compensation.
- The electron source is the caloric part of radiation exchange, evaluated
  through the native electron-energy response and checked at the accepted
  nodal temperature.
- Spatial stages use the annular M1 flux operator. Optical-wall transfer and
  cylindrical stress are recomputed directly on the **actual accepted
  candidate**, not reconstructed from a global balance residual.
- Accepted interval output accumulates only accepted substeps. Failure after
  an earlier private success leaves live radiation, particles, temperature,
  heat, and the caller's exchange output unchanged. Retrying the interval
  discards the entire previous attempt's wall and geometric contributions.

The radiation moment and material impulse use local `(r, theta, z)` components;
particle-owned carry remains Cartesian. The radial-component inventory includes
the cylindrical stress term and is not mistaken for a globally conserved
Cartesian momentum. Optical-wall transfer is not charged to material recoil.

Exactly meridional particle motion can acquire a tiny theta component when its
secant velocity is rotated from Cartesian storage. The adapter canonicalizes
only a `64*eps`-scaled meridional-velocity rounding bound; meaningful azimuthal
flow rejects the candidate. It does not canonicalize arbitrary radiation states.

Similarly, the raw relative theta momentum residual can be large when both its
numerator and denominator are essentially zero. That raw diagnostic is retained.
The RZ acceptance gate separately bounds the rotation products using the
**uncancelled transverse requested/actual/carry impulse**, never radiation
energy or the measured imbalance. The physical momentum tolerance remains
`1e-10`; the new `momentum_balance_residual` reports the ratio to the combined
physical and explicitly derived arithmetic bound. The Cartesian gate is unchanged.

## Bounded evidence

The native tests use a fixed `8 x 8` RZ domain, shaped nodal electron fields,
nearest-cell radiation-to-ion assignment, one fixed-charge ion species and
ideal finite-volume electrons. They compare against independently accumulated
particle kinetic energy and independently weighted nodal electron energy,
not merely the source's own energy ledger.

| Case | Finite effect | Maximum independent total-energy residual, CPU |
|---|---|---|
| 64 recoil/scattering stages | Actual ion work `-6.6386e-8 J` | `4.04e-16` |
| 64 LTE exchange stages | Electron caloric transfer `4.3968e-8 J`; finite ion work also checked | `3.32e-17` |
| 64 spatial M1/recoil stages | Actual ion work `-6.8667e-8 J`, with direct wall/geometric accounting | `3.30e-16` |
| Failed second substep and whole-interval retry | Live-state and output immutability; accepted four-step result compared with a private four-step reference | Pass, serial and two-rank MPI |

The five-case CPU/MPI selection passes in 24.95 s locally. The combined RZ
source/operator address/undefined-behavior sanitizer matrix passes 9/9 in
236.83 s. The surrounding Cartesian transport/source/boundary and ideal
conduction selection passes 127/127. Single-precision compiler checks pass
12/12 affected source translation units across all four supported geometries.

CUDA source-only recoil and LTE cases pass all 64 stages. The sustained
spatial native source hit the original 1500 s wall-clock limit after at least
23 accepted stages, with no reported physics-gate failure. This is incomplete
GPU qualification and poor measured small-grid performance, not a full pass.
The follow-up uses a single block for sustained serial cases and retains
multiple blocks for rollback/MPI ownership tests. GPU spatial cases are marked
slow and have a 5400 s execution budget; the 64-stage physics integration and
all assertion tolerances remain unchanged. No scalable GPU performance claim
is made. No checksum reference was refreshed for this feature.

Rollback testing exposed undefined physical radiation ghosts in the new
nonperiodic source candidate. The source now preserves caller-owned physical
ghosts while synchronizing interior/periodic aliases. The verification scratch
has no ghosts and copies only its own allocated extent. Both failure rollback
and successful retry compare the complete advertised state, including ghosts.

## Separate runtime qualification

These source tests use explicit gray coefficients and native caloric callbacks.
The bounded native adapter, its annular coefficient normalization, actual drift,
pressure-work requirements and restart evidence are documented in the
[RZ moving-runtime qualification](radiation_rz_moving_runtime.md). Its supported
contract must not be inferred from the source-only tests on this page.

Packet–diffusion momentum conversion, general azimuthal/angular-momentum
transport, table-EOS material coupling and conservative magnetic work remain
distinct feature gaps. Passing this source transaction must not silently
activate any of them.

## Persistent transport accounting (September 14 integration work)

`MomentTransportLedger` independently accumulates outward optical-wall transfer
and cylindrical stress in `(E,c*p)` units. It uses compensated sums, including
when subtracting the two accounts: weak net exchange must not disappear merely
because their cumulative values are large. These are direct accepted transport
terms, not an inferred conservation residual or an additional material reservoir.

The interval API optionally accepts this ledger. Each attempt starts from a
private copy, accumulates its accepted substeps and publishes it only with the
final particle/field commit. Failed later substeps discard the trial history.
Its versioned stream record preserves the compensation as well as the totals;
malformed, nonfinite or overflowing records/updates reject without changing the
previous ledger.

The native moment runtime now owns this ledger and writes it alongside the
`gray_m1_low_beta_nodal_shape_ledger_v3` checkpoint manifest. Restart requires the
declared ledger. Older v1/v2 native Cartesian checkpoints retain their original
shape checks and initialize these identically-zero periodic transport accounts.
This v3 schema remains Cartesian. The bounded RZ runtime uses a distinct
meridional nearest-cell model manifest rather than reinterpreting it.

Qualification adds cancellation/serialization/overflow unit checks and extends
the existing serial/MPI interval-rejection gates with persistent-ledger checks.
The native checkpoint corruption harness also tests a missing declared ledger.
Passing these checks is infrastructure evidence, not a moving-material trajectory
or changed-rank RZ restart result.

The ledger plus source selection passes 6/6 CPU tests in 22.04 s, including
serial and two-rank rollback. Twelve affected single-precision compiler checks
pass across Cartesian and RZ builds. The native Cartesian moving-pulse producer,
physics analysis, shape guard and missing-manifest/moment/ledger checks pass
8/8 in 14.63 s; normal restarted evolution and its comparison pass 2/2 in 5.11 s.
The CUDA-build ledger unit test passes in 0.71 s.

A bounded performance probe of the existing CUDA source binary completes one
spatial stage in 78.38 s versus 0.2302 s on CPU. The profile attributes 48.89%
of GPU time to dot products and 24.33% to vector increments in repeated Krylov
orthogonalization. The default half-relaxed source requires 32 outer iterations.
The test-only full-relaxation probe requires six on its first stage and passes
all 64 CPU spatial stages in 2.468 s, with independent energy residual
`1.64e-15`. The current CUDA binary also passes the full-relaxation single-stage
probe in 12.55 s with six outer iterations. Its sustained 64-stage run completes
in 993.8 s with independent energy residual `2.75e-16`.
This is a bounded solver-control comparison, not a change to defaults
or acceptance tolerances. It qualifies this source case, not native CUDA drift.
