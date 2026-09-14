# Angular-conservative RZ moment/source building blocks

This opt-in capability includes low-level source/transport and a bounded native
angular runtime. The default adapter remains the
[meridional runtime](radiation_rz_moving_runtime.md). The experimental
`rz_angular_transport` integration passes full reflected-motion and changed-rank
restart checks on CPU and CUDA, within the contract below. This does not finish
the general RZ radiation module; the stiff GPU source limitation below remains.

## Which momentum is conserved?

Radiation cells store integrated `(E, Qr, Qtheta, Qz)`, with `Q = F/c`.
The axial angular inventory is `sum(Rbar*Qtheta/c)`, where the annular
volume-mean radius is

```text
Rbar = integral(r dV)/V = r_mid + dr^2/(12*r_mid).
```

It is not `sum(Qtheta/c)`. The radial angular-flux divergence is therefore
the upper-minus-lower face sum of `r*A*c*Prtheta`, divided by `Rbar`.
The axial term uses the usual face-area divergence. Internal angular fluxes
telescope with the same `Rbar` inventory weight. Fixed optical mirrors
exchange no tangential torque; periodic axial faces cancel.

The ordinary theta-component geometric term remains separately computable
from these face fluxes. It is not a residual correction. Its contribution
also enters the projected `E-beta.Q` row. Omitting that term would introduce
a frame-dependent energy equation even with zero opacity.

## Actual particle torque and finite work

Equal nearest-cell theta kicks do not, in general, assign the requested
radiation torque: particle mass-weighted radius and radiation volume-mean
radius need not coincide. `ParticleImpulseAssignment::RZAngularConservative`
instead interprets the requested theta impulse as `Ptheta = DeltaJ/Rbar` and
uses actual particle inertia:

```text
Ieff = sum(m_p*(r_p/Rbar)^2)
Delta utheta,p = (r_p/Rbar)*Ptheta/Ieff.
```

Radial and axial assignment retain their mass weights. The theta component
of the finite-work velocity has the corresponding inertia/conjugate weight.
Work is still obtained from finite relativistic particle kinetic-energy
changes, not an infinitesimal `v dot dp` approximation. Representational
momentum and energy remainders remain owned by the particles.

`ActualImpulse()` and `MomentumCarryChange()` retain physical `(r, theta, z)`
impulse semantics. `ConjugateImpulse()` and `ConjugateMomentumCarryChange()`
provide torque divided by `Rbar` in their theta component. They must not be
silently substituted in a physical theta-momentum diagnostic. The source's
`exchange.momentum` remains the actual physical impulse.

The source adapter requires matched angular assignment and
`transport.rz_angular_transport=true`. Unmatched options reject. Nonzero
torque with zero recipient angular inertia, and particles outside the owned
physical domain, reject before committing state.

## Independent checks and bounded accuracy

- Geometry checks use binary and nonbinary mesh spacing, show angular
  telescoping, and explicitly demonstrate that ordinary theta momentum is
  not the invariant.
- A rotating photon-gas M1 equilibrium runs for 64 transport stages. Refining
  32 to 64 radial cells reduces the measured profile error from 4.991% to
  2.745%, while retaining the independent energy/angular conservation gates.
  Finite Rusanov shear diffusion remains; exact discrete stationarity is not
  claimed.
- Prescribed rotating material, including mixed radial/axial motion, develops
  finite radiation work without elastic caloric heating. A zero-opacity
  mixed-component frame comparison agrees to about `9.25e-17` normalized
  difference.
- Actual finite-mass particles undergo 64 angular source stages. Spatial
  elastic exchange produces about `-3.19e-8 J` ion work. Strong LTE exchange
  produces about `4.40e-8 J` electron caloric transfer and separately checked
  finite ion work. Independent particle and radiation inventories test total
  energy and axial angular momentum, rather than trusting source ledgers.
- Failure after a successful private substep must leave all live state,
  physical ghosts, exchange output and the compensated transport ledger
  unchanged. Retrying must reproduce the accepted-only subdivided reference.

The current CPU source/operator/particle selection passes 21/21 checks;
20 affected single-precision source translation-unit checks also pass.
The six-case source/operator/particle ASAN/UBSAN selection passes in 229.69 s.
The subsequent CUDA source selection passes four of five checks; the stiff
LTE case times out as documented below and remains unqualified on GPU.
These angular particle-source tests hold positions fixed. Native meridional
drift evidence belongs to the separately documented runtime and does not
qualify this new angular assignment automatically.

## Native integration: detected failures and repairs

Native angular motion required a distinct checkpoint/model contract, actual
angular inventory and transfer diagnostics, and independent drifting,
reflection, decomposition and restart checks. In particular, during drift
`r cross carry` changes by `v cross carry`; its representational contribution
must be bounded or conservatively owned, not hidden in a fitted torque ledger.

The initial native probe supplied a distinct angular model identifier and
an optional `RadiationMomentum.include_angular_inventory` diagnostic containing
radiation, represented-ion and particle-owned pending angular inventories.
At 1.28 ns, independent plotfile inventories show about 0.127 radians mean
material rotation and matched radiation/ion angular transfer of `2.33e-11`
kg m^2/s. At the first wall encounters the full run fails the unchanged
electron charge-continuity gate (approximately `2.09e-5`, versus `1e-7`).
Reconstructing the previous position from the reflected velocity is not the
actual particle path. Saved RZ Cartesian endpoints and their use in the
auxiliary charge flux repair that failure in the subsequent full 256-step run.
That run has about 0.251 radians mean rotation, 25.1 micrometres maximum radial
motion, and total-energy error `-2.91e-10 J` against a `3.14e-6 J` bound.
It nevertheless fails angular conservation at the wall: the independent error
is `-9.47e-10 kg m^2/s`, versus a `1.49e-17` bound. The endpoint-radius reflection
had to be replaced with a collision-point path/momentum/carry reflection before
native angular qualification. Neither gate was loosened.

The subsequent collision-point repair follows actual circle/segment crossings
and reflects the remaining displacement, represented momentum, and every carry
through the same Cartesian matrix. It supports bounded multiple crossings and
rejects exhausted collision budgets before commit. The corrected azimuthal
position must also commit; keeping the old angle was separately detected by
the independent angular gate. The geometry unit checks energy, angular momentum,
arbitrary carry-vector norm, and time reversal, including grazing/multiple hits.

With both endpoint deposition and collision-point commit repaired, the full
256-step CPU run passes: radiation gains `2.327294534e-11 kg m^2/s` of angular
momentum, matched by the represented-ion loss plus pending change. Independent
angular error is `1.08e-20 kg m^2/s` against `1.49e-17`; energy error is
`-2.88e-10 J` against `3.14e-6 J`. Mean rotation is 0.250698 radians and maximum
radial displacement is 25.0631 micrometres. No physical gate was changed across
these failing and passing controls.

The native CPU selection passes 12/12 checks, including one-to-two-rank restart,
missing-ledger and wrong-assignment rejection. A separate finite-carry wall test
passes on one and two ranks: it imposes deliberately nonzero pending momentum,
checks represented angular momentum and kinetic energy, checks the actual
pending-vector reflection/transfer, and rejects a staged callback without
changing particles or the caller's accounts. The surrounding meridional
runtime/restart/unsupported-path selection passes 13/13. These results do not
substitute for broader material or transport qualification. The native angular
CUDA selection subsequently passes 7/7 checks in 2249.94 s, and the native
address/undefined-behavior sanitizer matrix passes 8/8 in 233.97 s. Deliberately
nonzero carry/wall transactions pass on one and two GPU/MPI ranks as well.

### Known stiff-source GPU limit

The separate fixed-position angular LTE stress case reaches its 5400 s GPU
deadline after 13 of 64 accepted stages. No physical assertion fails in those
completed stages, but the case is incomplete and is not counted as a pass.
Its full CPU run passes. A smaller fixed-point relaxation control increased
the CPU work, so it was not adopted. The test's physics gates and deadline are
unchanged. Small-grid, very-stiff GPU source performance needs further work;
native moving-run success does not erase this limitation.

A separate moving absorption/emission case retains all 256 native steps and
wall encounters. On CPU, radiation loses 167.714 J, electrons gain 183.040 J,
and ions lose 15.326 J. Independent energy error is `-5.34e-10 J` against
`3.17e-8 J`; angular error is `-6.88e-21 kg m^2/s` against `4.57e-18`.
Run/analysis and changed-rank restart checks pass 4/4 on CPU and 4/4 on CUDA
(1570.82 s locally). This case does not replace the stiff-source stress gate.

### Experimental native contract

Set `radiation_transport.rz_angular_transport=1` with `diffusion_solver=coupled_moment`
and the existing meridional runtime requirements (gray, fixed-charge, ideal
electrons, level zero, shape one, conservative pressure work, axis/PEC radial
wall, periodic z, no packet conversion or plasma-current magnetic work).
The ion species additionally requires `save_previous_position=1` and actual
particle pushing. Saved `prev_x/prev_y/prev_z` are Cartesian coordinates in RZ;
the auxiliary charge flux uses the previous endpoint and the nearest periodic
axial image, rather than backtracking the post-reflection velocity. RCYLINDER
and RSPHERE previous-position support is not enabled by this change.

The native model identifier is `gray_m1_angular_rz_inertia_ledger_v1`, distinct
from the meridional assignment. Angular diagnostics use a separate schema.
Reflect-all walls are unsupported in this angular contract; they can exchange
tangential momentum and are not a smooth specular wall.

Packet/moment conversion, pressure-driven magnetic induction and conservative
magnetic work, calibrated high-Z material closures, and moving nonideal EOS coupling
remain separate unresolved production capabilities. This source increment
does not qualify those paths.

Nonideal RZ thermal conduction has a separate
[qualification](radiation_nonlinear_electron_conduction.md); it does not remove
the ideal-electron guard on this moving radiation/pressure-work model.
