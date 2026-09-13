# Meridional RZ moving-moment transport

This is a low-level gray M1 transport/source capability with prescribed material
velocity, opacity and equilibrium radiation energy. It is **not** activation of
the native RZ coupled-moment runtime, packet–diffusion momentum closure, or a
completed radiation-hydrodynamics model.

## Equations and ownership

Radiation cells store integrated energy and flux moment,
`U = (E, Qr, Qtheta, Qz)`, where `Q = F/c` has energy units. The operator divides
by the annular cell volume before evaluating face fluxes. It multiplies radial
fluxes by `2*pi*r_face*dz`, and axial fluxes by the annular cross-section.
WarpX's RZ geometry object does not have to report an AMReX cylindrical
coordinate type: the dimensionality and explicit annular measures define this
adapter.

The radial momentum equation includes the cylindrical `P_theta_theta/r`
stress term. Its discretization cancels the radial divergence of a uniform
isotropic pressure exactly. An axis face has zero area and no optical-wall
transfer. An outer optical mirror carries pressure but no energy.

The accepted result reports two independently evaluated quantities:

- `boundary_exchange`: outward wall energy and `c*momentum`.
- `geometric_exchange`: cylindrical radial stress, **not** a material force,
  lost photon momentum, or a boundary transfer.

The checked integrated equation is
`U_new - U_old + material_transfer + boundary_exchange - geometric_exchange = 0`.
Only energy and axial momentum have a Cartesian conserved-total projection.
Radial momentum is never projected into a fictitious globally conserved scalar.
The geometric contribution is also included in the `E - beta.Q` equation used
by the stiff solve; omitting that term would corrupt moving-material work.

The material output is the prescribed-medium four-force. Its energy component
includes radiation work, and a separate caloric output removes that work. It is
not yet the kinetic-energy change of native finite-mass particles.

## Supported and rejected configurations

Supported here: double precision, a fixed axis-containing RZ domain, stationary
radial optical mirror, periodic or stationary reflecting axial boundaries,
`|beta| <= 0.01`, and exactly zero azimuthal radiation moment/material velocity.
The state is meridional, not a general rotating axisymmetric radiation field.
Material coefficients are fixed within each backward-Euler stage.

Nonzero azimuthal state or velocity is rejected without modifying caller data.
General azimuthal transport needs torque/lever-arm accounting; changing that
guard is not sufficient. Prescribed external face fluxes and native particle
boundary coupling are not included in this RZ adapter. AMR, embedded boundaries,
multigroup moving spectral transfer, and moving optical walls are outside its
contract.

## Bounded qualification

The new tests use long enough integrations to produce finite transport or work,
not just a few initialization steps. They do not replace the existing native
particle/electron energy and carry tests.

| Case | Independent physical target | Local CPU result |
|---|---|---|
| Isotropic equilibrium, periodic and reflecting axial walls | Zero flux; outer pressure balanced by cylindrical stress over 16 stages | Exact zero radiation momentum; direct wall/stress integrals pass |
| Radial Bessel mode, 64 stages | Annular average of a cylindrical P1/telegraph diffusion eigenmode, approximately half-amplitude decay | 0.71% mode-profile error; also tested on two MPI ranks |
| Axially moving trapped mode, 128 stages | Diffusion plus advection through 0.3 domain lengths at `beta_z=0.003` | 9.19% error at 64 cells; 5.16% at 128 cells, passing the refinement gate |
| Radial prescribed compression, 128 stages | Independent cylindrical diffusion/advection/compression-work PDE, refined from 512 to 1024 cells | 0.43% profile error; reference refinement 0.0094%; material work about -0.004487 J |
| Static and moving LTE baths, 64 stages | Independent boosted equilibrium moments and radiation/material energy conservation | Both equilibrium and original `1e-10` energy gates pass |
| Invalid azimuthal states and exhausted nonlinear/linear budgets | Transactional rejection, including ghosts and output sentinels | Rejected without publishing state or boundary/geometric exchange |

The axial mode exposes appreciable first-order spatial diffusion. The result is
useful evidence of the correct advection direction, magnitude and refinement,
not high-order accuracy. It is recorded rather than hidden by weakening a gate
or repeatedly tuning one mesh. The radial compression target is a prescribed
nonrelativistic diffusion limit; it is not a FLASH fluid-ion trajectory or a
claim that hybrid PIC and fluid closures are interchangeable.

The local CPU matrix passes 6/6 cases in 9.90 s, with unchanged acceptance gates;
the surrounding Cartesian transport/source/boundary matrix passes 118/118.
Single-precision compiler checks pass for all 16 affected translation units
across 1D, 2D, 3D and RZ; this is compiler portability evidence, not SP physics
qualification. Strict Sphinx documentation generation also passes.

CUDA qualification exposed an asymmetric fused multiply/subtract of equal-area
axial pressure fluxes that could inject a false axial force. Factoring out the
common area restores exact cancellation; the previously rejected radial case
then passes without changing a tolerance. The full local CUDA matrix passes
6/6 in 567.77 s, including two-rank MPI. Most of that time is the small-grid
axial refinement run (423 s); this is not scalable GPU performance evidence.

Run the matrix with `ctest -R '^test_rz_moment_transport_' --output-on-failure`
in a configured double-precision RZ build. The compression analysis generates
`rz_m1_compression.csv` in its CTest working directory. Its reference is assembled
independently in Python and advanced using a sparse matrix exponential; it does
not call the C++ radiation operator.

## Next integration boundary

The [low-level native RZ source transaction](radiation_rz_native_moment_source.md)
now pairs this transport with actual particle work, native caloric deposition
and separately counted optical-wall/geometric terms, including interval rollback.
Its source/drift ordering and persistent runtime diagnostics are the next
integration boundary. Packet–diffusion conversion then needs the same momentum
ownership across both representations. The public runtime remains guarded;
these operators do not qualify a dynamic hohlraum or conservative magnetic work.
