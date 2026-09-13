# September 13 feature-first increment

This increment follows **features → bounded validation → application**.
Application pictures are not acceptance evidence. Once an implemented feature
passes its conservation, ownership, invalid-state and bounded portability
checks, remaining accuracy/performance gaps are recorded instead of consuming
the campaign in repeated refinement. A genuine safety or conservation failure
still blocks the affected feature; no assertion is relaxed to move past it.

## What was added

| Capability | Implemented subset | Important remaining boundary |
|---|---|---|
| RZ particle/electron pressure work | Centered gradient paired with actual Boris gather/scatter work; native axis/wall volumes; restart | Ideal FV electrons, zero Ampere plasma current, supported PEC/periodic walls only |
| Face-exact RZ packet transport | Radial and axial intersections, corners, axis traversal, escape and periodic axial ownership | No EB or coupled direct recoil with this policy; continuous opacity is midpoint-sampled inside a cell |
| Packet/diffusion conversion on that path | Gray/multigroup conversion and reemission, zero-residence exclusion, CPU/GPU-safe shared scatter | Does not close momentum ownership during conversion or moving spectral transport |
| Electron heat conduction | Conservative isotropic ideal-electron heat flux, lagged harmonic limiter, insulating/periodic faces | No nonlinear table-EOS/anisotropic/nonlocal conduction; bounded substep stiffness, not multigrid scalability |
| Material-specific Qei rates | Named species overrides, global fallback, fixed-charge support, PICMI and restart manifest | User-supplied rates, not calibrated high-Z atomic/Coulomb-log physics |
| Python input access | PICMI material-energy controls and constant-name mangling | The same C++ model/backend guards remain authoritative |
| RZ particle-owned absorption recoil | Finite material work, local diagnostic projection, Cartesian carry ownership and elastic material-wall transfer | Absorption-only bounded-segment lab-frame model; no RZ moment transport, face-exact recoil or conversion momentum closure |
| Strict material metadata audit | Optional opacity/EOS/PIC atomic-mass matching, mean nuclear-charge matching and a fixed-ion charge bound | Matching composition moments does not establish calibration, identical mixtures or equilibrium ionization consistency |
| Low-level meridional RZ M1 transport | Annular fluxes, cylindrical stress, prescribed moving-medium work and optical mirrors | No native RZ coupled-moment activation, azimuthal transport or packet/diffusion momentum closure |

The material additions support radiation-driven matter response; they are not
themselves photon transport. In particular, conservative **electron pressure
work** must not be mistaken for completion of **radiation momentum coupling**
in every geometry.

The follow-up RZ absorption/carry increment is described in
[radiation_rz_particle_carry.md](radiation_rz_particle_carry.md). Its 256-step
native-state energy check is distinct from its radiation-source ledger check;
neither is a declaration of complete moving-frame or packet/diffusion coupling.

The subsequent low-level moment operator and its independent radial diffusion,
moving-mode and compression-work targets are described in
[radiation_rz_moment_transport.md](radiation_rz_moment_transport.md).

The metadata audit is enabled with
`radiation_transport.require_material_metadata_consistency=1`; its exact scope
and limitations are documented in the input reference. Six manufactured
acceptance/rejection scenarios and the surrounding HDF5 regression selection
pass (21/21 stages). The fixtures test tungsten metadata only, not calibrated
tungsten opacity. No physical coefficient or particle charge is adjusted.

## Bounded evidence and defects found

- RZ CPU feature/continuity/restart selection: 80/80 stages, 44.88 s locally.
- Public-worktree RZ/Python selection: 91/91 stages, 43.54 s, including the
  original species-drag regression (zero density error) and PICMI input test.
- Cartesian radiation/conduction/Qei/pressure regressions: 292/292 stages,
  109.09 s locally. These include existing tests, not 292 new physics cases.
- Single-precision fields/double-precision particles, exact-face and conversion
  selection: 44/44 stages, 13.62 s. An existing emitted-momentum reference now
  accounts for the declared mixed-precision representation. Its tolerance and
  bitwise restart comparison are unchanged.
- Independent pressure transpose and actual-particle work checks, finite
  compression, conduction decay/contact tests and two-material Qei checks are
  described in the linked records below. CPU and local CUDA evidence are
  distinguished; no remote P40 result is inferred from a local CUDA pass.
- Combined local CUDA feature/restart selection: 70/70 stages, 1042.01 s.
  The longer timing includes repeated 400-step material integrations, not
  extra physical features. Single-precision compiler checking separately
  caught an implicit double-to-field cast in RZ work deposition; conversions
  at the particle/field boundary are now explicit.

Implementation/qualification found and fixed stale RZ pressure-gather ghosts,
conversion in a cell with zero path residence, host tile scatter races, missing
FV species-density wall contributions, an RZ caloric-volume mismatch in Qei,
and non-finite Qei propagation. The old non-FV RZ drag density convention is
preserved. Test reference and configuration errors are distinguished from
solver defects in the detailed records. Generated checksums were not edited.

## Relationship to FLASH4.8 and the application

These additions close specific missing equations and coupling contracts, not
a percentage of universal FLASH equivalence. Hybrid PIC retains kinetic ions;
FLASH fluid-ion closure need not produce the same interface trajectory or
thermal support. Compare matched reduced physics first, then use each model's
own conserved inventories and numerical-resolution evidence. No AMR milestone
is required for the intended production scope.

The largest remaining feature work is full RZ moving-material radiation
momentum/energy and spectral ownership, magnetic pressure-work accounting,
and consistent material/EOS/opacity/charge-state closures over the intended
high-Z range. Stiff nonlinear and anisotropic conduction are additional
model/performance work. These are not gaps to hide by tuning a hohlraum movie.

In particular, the present hybrid Faraday solve omits electron-pressure
induction. The pressure-work checks are not evidence for Biermann generation
under nonparallel density/temperature gradients. Qualify that together with
magnetic-energy work, not by removing the present guard alone.

No new hohlraum run or FLASH run is claimed by this increment. Existing
application results predate these features and remain exploratory. The next
application step is a bounded integration check of supported combinations,
with energy/escape ledgers and explicit model guards, after feature gates.
Do not silently activate unqualified magnetic or moving-frame paths simply
because the initial shell is ballistic.

Detailed records:

- [RZ pressure work](radiation_rz_pressure_work.md)
- [RZ exact-face transport and conversion](radiation_rz_face_streaming.md)
- [Electron conduction](radiation_electron_conduction.md)
- [Species Qei rates](radiation_species_qei_rates.md)

CI status belongs to the actual PR head and must be read there; local passes
in this record do not imply that a later commit has passed hosted CI. The WIP
merge guard remains intentional.
