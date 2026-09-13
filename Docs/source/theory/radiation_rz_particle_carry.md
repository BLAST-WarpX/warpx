# RZ particle-owned radiation impulse: staged extension

Status: development, September 13. This adds a bounded absorption-only runtime
path, not completion of general RZ radiation–moving-material transport. The
RZ moment, face-exact momentum and packet-conversion momentum guards remain.

## Runtime scope

`radiation_transport.momentum_carry=particle` now accepts absorption-only RZ
transport with native hybrid electrons, double field/particle precision, one
azimuthal mode, level zero, no EB, an axis/reflecting radial domain and periodic
or reflecting axial material boundaries. The existing fixed-species, no-loss,
no-thermalization and no-collision guards remain. Diffusion, LTE emission and
packet/diffusion conversion are rejected for this RZ particle-carry subset.

This retains the existing bounded-segment lab-frame absorption model. It does
not add comoving opacity, Doppler group redistribution, exact radial interaction
quadrature, or radiation advection/compression in an optically thick material.
Finite material work is accounted for, but that alone is not a moving-frame
transport model. Defaults and the legacy cell-owned carry path are unchanged.

## Coordinate and energy contract

Particles store Cartesian proper velocities and Cartesian deferred impulse per
unit rest mass. A cell impulse in RZ has local `(r,theta,z)` components. At the
particle's azimuth, the existing nearest-cell adapter rotates the requested
increment into Cartesian coordinates before evaluating the finite relativistic
kick. Its actual kinetic work is computed from the represented velocity change;
pending work and numerical roundoff residual remain separate accounts. Field
work velocities and realized impulse are projected back into the local basis.

A sum of cylindrical radial components is **not** a conserved global Cartesian
momentum. The local source identity and the global Cartesian particle/carry
inventory are different checks. Likewise, an outer-wall impulse is not a
measurement of photon escape or packet-to-moment conversion recoil.

`RadiationMomentum` uses the local projection for its existing `(r,theta,z)`
pending-carry columns. The particle inventory API defaults to Cartesian and the
wall ledger remains Cartesian. Pending energy is independent of this projection.

## Elastic material walls

The boundary adapter is extended to an axis-containing RZ domain with an outer
reflecting radial wall and periodic or reflecting axial faces. The axis uses
the native `none` boundary, not a planar mirror. The native particle boundary
event determines which cylindrical velocity components reverse. Carry storage
and the pending-only wall transfer remain Cartesian.

For one reflected transverse direction with unit Cartesian normal `n`, the
carry map is `a_new = a_old - 2 (a_old dot n) n`; the wall receives
`m_weighted (a_old - a_new)`. A simultaneous radial and azimuthal reversal is
an exact transverse sign reversal. The axial map is an exact sign reversal.
An interior particle undergoes no coordinate round trip. Pending scalar work
is unchanged by this stationary orthogonal reflection.

All particles and all registered carry paths are staged before committing.
Invalid state, unsupported boundaries, an overshoot still outside the domain
after the native reflection, or rejection by the collective accounting callback
leaves live particles and the caller's transfer vector unchanged. The existing
compensated wall ledger can then accept the direct event transfers. Loss and
thermalization remain unsupported.

The mirror uses the native end-of-step position reflection. This does not add
an exact curved-wall collision-time/orbit integrator, nor establish global
angular-momentum accuracy for finite wall overshoots. The wall diagnostic counts
deferred radiation impulse only; it must not be presented as total ion-wall
force or work.

## Qualification gates

The new native RZ fixture tests 128 finite, non-collinear kicks on moving ions,
the actual cylindrical projection, independent long-double finite kinetic work,
the requested/represented/deferred identities, and unchanged staged particles.
Its wall section checks independent local-basis reflection at multiple angles,
pending impulse inventory plus direct wall transfer, unchanged pending work,
interior no-ops and rejected transactions. Serial and MPI exercise different
particle ownership. GPU execution checks the same physical assertions.

Passing this fixture does not qualify nodal RZ source assignment, moving-frame
radiation transport, conversion momentum ownership, or a magnetic-energy ledger.
Those remain separate release gates. Numerical results are recorded only after
the final implementation passes; intermediate fixture failures are not passes.

Current checks: the consolidated RZ CPU selection passes 7/7 (4.99 s), including native
particle-boundary calls, radial/axial corners, compensated wall history,
redistribution, particle/wall checkpoint round trips and unsupported-path guards.
All corresponding CUDA checks pass, including both MPI cases. The native-energy
and analytic attenuation gates share one simulation, avoiding duplicate runs.
The guard-output fixture was rerun after accounting for the wrapped error message.
The unchanged Cartesian particle-impulse/reflecting-wall/restart selection passes
36/36 (14.04 s). CI-style single-precision compiler analysis of the changed
boundary translation unit passes in 1D, 2D, 3D and RZ. The expanded compiler
check covers 16 changed translation-unit/dimension combinations and passes all.

The 256-step native absorption case transfers 438.6761141 J (53.58% of the
initial pulse). Packet attenuation agrees with `E(t)=E(0) exp(-alpha c t)` and
the source/impulse ledgers pass unchanged 2e-12 relative gates. A separate
native-state inventory sums ion kinetic energy, independently integrated nodal
ideal-electron energy, photon energy and particle-owned pending work. Its
relative total-energy change is 1.35e-16 on the local CPU run, against a 1e-10
gate; the corresponding local CUDA result is 4.06e-16. Native conservative
pressure work and charge-consistent electron transport
are active; the comparison is not constructed from the source ledger itself.
This short material-displacement absorption case is not a trapped-pulse or
moving-interface qualification. A full changed-rank radiation-runtime restart
campaign remains separate from these particle/wall round trips.
