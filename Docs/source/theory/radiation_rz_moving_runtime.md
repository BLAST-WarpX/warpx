# Bounded meridional RZ moving-material runtime

The native `coupled_moment` adapter now connects the annular radiation source
transaction to actual PIC drift, force gathering, density/current deposition,
and native electron-energy evolution. It is no longer only a frozen-position
source experiment. This is a **bounded meridional capability**, not general RZ
radiation hydrodynamics or a FLASH-equivalent production module.

## Supported contract

- One gray moment group, one fixed-charge ion species and an empty photon
  species; native ideal-gas electrons, double-precision fields and particles.
- Level zero, no EB or moving window, axis-containing RZ with one azimuthal
  mode; stationary PEC/reflecting outer radial wall and periodic axial faces.
- Particle shape one. Radiation-to-ion force uses the qualified nearest-cell
  assignment; native nodal electron fields retain their own particle-shape and
  caloric-volume mapping. Higher-order native RZ force assignment is not enabled.
- The existing low-beta moment/frame restriction remains. Radiation theta
  moment must be zero; meaningful azimuthal bulk material velocity is rejected
  by the source adapter. This is not a general angular-momentum/swirl model.
- `hybrid_pic_model.conservative_pressure_work=1` and
  `conservative_pressure_work_pec=1` are required. Their separate restrictions,
  including zero **Ampere plasma current**, remain enforced. That condition is
  not zero ion velocity or zero ion current: the examples have moving kinetic
  ions and a quasineutral electron return current. It does not qualify general
  magnetic work or pressure-driven magnetic induction.
- No packet conversion, spectral/species/material opacity tables, material
  tables, evolved charge states, ionization, collisions, resampling or open
  material loss are promoted by this adapter.

The source runs before the normal native PIC push. Its accepted temperature
refreshes native pressure, and the established pressure-work transaction pairs
the pressure force actually gathered by ions with the electron-energy debit.
The native finite-volume material remap then follows the moved material. This
is a split evolution scheme; conservation alone is not a high-order time-
accuracy claim.

## Measures, ownership and restart

Equilibrium radiation energy uses physical annular cell volume, not the
Cartesian product of grid spacings. Nodal electron heat uses its own native
axis/wall control-volume weights. They are distinct measures and must not be
substituted for each other.

Radiation moments, kinetic ion work, particle-owned compensation and nodal heat
are committed transactionally by the source interval. Optical-wall transfer
and cylindrical stress are accumulated independently in
`MomentTransportLedger`, including compensation for weak differences between
large accounts. Cylindrical stress is neither material recoil nor a residual
reservoir. The RZ radial-component inventory is not globally conserved
Cartesian momentum.

The checkpoint model is `gray_m1_meridional_rz_nearest_cell_ledger_v1`, with its
particle shape order and required persistent transport ledger. A Cartesian
moment checkpoint is rejected rather than reinterpreted as RZ. The normal
particle compensation and pressure-work checkpoint contracts also remain.

`RadiationMomentum.include_moment_transport=1`, together with
`include_moment_inventory=1`, appends cumulative wall, geometric and compensated
wall-minus-geometric accounts. Each account has energy and three momentum
components. Output cadence does not own or advance these counters; both
diagnostic options must remain unchanged at restart.

## Two practical native integration gates

These use actual particles, including finite cell crossing, rather than only
prescribed velocities or a handful of steps. The C++ driver independently
integrates native nodal electron energy; the Python analyses independently
read particle kinetic energy, positions, carry and radiation fields. They do
not infer electron energy from cell-averaged plotfile temperature.

### Axially moving trapped radiation pulse

`inputs_base_rz_moving_moment_pulse` uses an `8 x 64` annular grid and 256 steps
of `1e-11 s`. A 100 km/s material flow advects a smooth trapped radiation
modulation. Scattering, ion recoil/work, native pressure work and density
evolution remain active.

- Actual axial particle displacement is `0.255755--0.256245 mm`, about a
  quarter of the 1 mm domain, crossing many axial cells.
- Ion kinetic energy gains `0.021255748 J`; nodal electrons change by
  `-0.000639486 J`; radiation changes by `-0.020616266 J`.
- Independent energy error is `-4.319e-9 J`, within the unchanged physical
  plus background-arithmetic bound `3.188e-8 J`. Whole-system native residual
  is `-2.560e-13`.
- Maximum density and ion axial-velocity changes are `0.2855%` and `0.1619%`.
- The independent linear trapped-pulse transport reference gives `0.9915%`
  normalized L1 profile error and `0.1509%` domain-length phase error.
- A halfway checkpoint restarted on two MPI ranks matches uninterrupted
  serial particle states, fields and persistent accounts.

The radiation reference tests transport, not equality between PIC ions and a
FLASH fluid-ion closure. Coarser spatial/time runs are retained as bounded
refinement evidence: 64 axial cells at twice the timestep missed the existing
`0.2%` phase gate; halving the timestep passes it without changing that gate.
Residual first-order diffusion is documented, not removed by tolerance changes.

### Radial compression with LTE absorption and heating

`inputs_base_rz_moving_moment_compression` uses `32 x 8` cells and the same
256-step duration. A smooth inward radial velocity, with peak 20 km/s, vanishes
at the axis and stationary outer wall. Gray absorption is `1/m` and total
extinction is `1e7/m`. No current drive or moving physical wall is imposed.

- Maximum inward displacement is `51.009 micrometres`; `79.69%` of particles
  cross a radial cell boundary. Maximum density change is `42.37%`.
- Ions lose `6.747703 J`, electrons gain `174.363865 J`, and radiation loses
  `167.616161 J`. Independent energy error is `-1.588e-11 J`, below its
  unchanged `3.145e-8 J` bound.
- Radiation gives material an outward radial impulse `4.735827e-5 kg m/s`,
  opposing the inward motion. Optical-wall impulse is `3.470419e-4 kg m/s`
  and the geometric contribution is `3.944002e-4 kg m/s`. Keeping them distinct
  closes the source radial inventory to `7.454e-20 kg m/s`.
- Changed-rank restart matches the uninterrupted radial run, including the
  nonzero wall/geometric history.
- Doubling radial resolution changes ion work from `-6.74770` to `-6.75788 J`
  and radial radiative impulse by approximately `0.085%`; the refined run also
  passes the independent gates. No exact fluid-ion trajectory is claimed.

The analyses write `rz_moving_moment.json/png` and
`rz_moving_compression.json/png` in their run directories. These are numerical
evidence and visualization artifacts, not generated checksum references.

## What the tests exposed

The first moving case used the legacy pressure-work treatment and failed the
unchanged independent energy gate at `7.67e-9`. Selecting ideal EOS alone did
not fix it. Enabling the conservative native pressure-work pair reduced the
residual to `1.26e-13`. The RZ radiation adapter therefore explicitly requires
that pair instead of allowing a configuration that breaks the combined
material-energy contract.

The source-only CUDA performance investigation is separate: full relaxation
reduces the first source stage from 32 to six outer iterations. All 64 spatial
source stages pass on CUDA in 993.8 s with independent energy residual
`2.75e-16`. That is source-layer evidence, **not** CUDA qualification of these
new native drifting cases. Their current evidence is CPU/MPI; backend
performance and further qualification remain visible obligations.

## Still outside this qualification

General swirl/angular momentum, packet--moment conversion, nonideal/multimaterial
moving EOS/opacity coupling, magnetic pressure-work/induction, open material
boundaries and calibrated high-Z materials are not finished by these tests.
The present smooth, positive-material-support runs also do not qualify all
moving exact-vacuum fronts or strongly multistreaming material interfaces.
Keep the PR WIP and retain those distinctions when discussing production
readiness or a dynamic-hohlraum application.
