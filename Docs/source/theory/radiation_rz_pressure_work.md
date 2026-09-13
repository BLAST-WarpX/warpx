# Opt-in native RZ pressure-work exchange

Status, 13 September 2026: implemented in the private qualification branch;
bounded CPU and local CUDA qualification pass. Packaging is in progress.
This is a material-coupling prerequisite, not a claim that the
hohlraum application or full moving radiation module is qualified.

## Model contract

Select `hybrid_pic_model.conservative_pressure_work=1` and
`hybrid_pic_model.conservative_pressure_work_pec=1` with ideal finite-volume
electrons. The first RZ contract requires m=0, collocated double-precision
fields/particles, full-order direct Boris gather (orders 1-4), no EB or material
tables, an axis-containing domain, an outer PEC/reflecting wall, and periodic
or PEC/reflecting axial faces. Ampere plasma current must be zero; otherwise
the electron update rejects the unqualified electromagnetic work channel.
Existing filter, particle-operation and diagnostic synchronization guards stay.

Legacy RZ calls the forward cylindrical Yee pressure difference even on a
collocated grid. The new option instead selects the centered nodal pressure
gradient G, with zero radial gradient at the axis. Axial force can be nonzero
at the axis. Physical PEC pressure forces vanish at the wall nodes: scalar
pressure is extended evenly without replacing the wall's own EOS pressure,
and tangential E is masked. This is an opt-in stencil change, not just a new
diagnostic or a global correction of the final energy discrepancy.

The pressure field at the push is `E_p = C[-rho_inverse G P_old]`. Exact Boris
electric-work velocity is scattered through the transpose cylindrical gather
rotation and shape. RZ scatter retains integrated charge-work velocities
[C m/s]. The transpose C^T folds radial/azimuthal odd and axial even axis
ghosts, PEC component parities, and MPI/periodic aliases. Then

`delta U_e = dt P_old W^-1 G^T rho_inverse C^T J_work`.

W is the same native caloric measure as charge-consistent transport, including
the independently selected axis volume and physical wall half-volumes. It is
applied exactly once. Electron work is loaded before the charge-energy remap;
EOS-invalid local cooling is rejected. Particle work, Qei and radiation sources
are not counted twice. The stored pressure and inverse denominator remain the
ones that generated the actual particle push, including after restart.

## Implementation-exposed defect

The first multiple-FAB native check had a 1.32e-9 relative energy residual at
200 steps, while the independent gather/scatter and gradient/transpose checks
passed at roundoff. Axis physical ghosts at axial FAB corners could use stale
neighbor data: applying the axis extension before inter-FAB exchange did not
implement the C paired by C^T. The opt-in path now synchronizes valid/periodic
E first, then rebuilds PEC and axis extensions before particle gathering. It
does the same for isolated pressure E, including on restart. Legacy paths are
unchanged. The original failure is retained in `build-demo/rz-work-first-tests.log`.

## Bounded evidence

- Independent host sparse-matrix assembly checks device G and -W^-1 G^T,
  both axis-volume policies and periodic/physical axial faces. Largest
  pointwise transpose error is 1.78e-15; power residuals are below 2.7e-16.
- The real RZ gather/scatter pairing passes orders 1-4, including nonzero
  particle angles; an order-mismatched negative control remains detectable.
- A 400-step native pressure-only evolution on eight FABs gains 1.320e-6 J
  ion kinetic energy and loses the same native electron energy. Relative
  total-energy residual is -2.63e-13 of 0.004719 J initial energy.
- Nine CPU operator/native/shape/axis/wall/restart gates pass in 14.51 seconds.
  Separate same-rank and one-to-two-rank restart field/particle comparisons
  pass without changing their 2e-12 bounds.
- The CUDA operator, native evolution, shape, wall, axis, compression,
  same-rank and one-to-two-rank restart/comparison selection passes 13/13 on
  the local RTX 4000 Ada. This is not a claim of a P40 run or bitwise
  CPU/GPU equivalence. CPU negative controls reject changed axis metrics,
  disabling the restart model, and an imposed nonzero Ampere plasma current.
- A separate 4 ns (400 x 10 ps) control also passes: ion work 6.955e-6 J,
  relative energy residual -2.85e-13. This is bounded integration evidence,
  not a spatial accuracy or cold-contact stability qualification.
- Inward-moving material also heats the electrons: a 400 x 2 ps radial
  compression transfers 7.892e-4 J from ions to electrons (8.40% of initial
  total energy), with -4.12e-13 relative total-energy residual. This tests the
  opposite work sign with appreciable transfer, not only a weak cold push.

Tests are under `Examples/Tests/hybrid_qei_conservative_exchange`; use CTest
regex `^test_rz_pressure_work_`. No FLASH fluid-ion closure is substituted.
No hohlraum run has yet been used to qualify this increment. Magnetic work,
material tables/mixtures, moving radiation integration and longer application
accuracy remain separate obligations. Existing tolerances remain intact.
