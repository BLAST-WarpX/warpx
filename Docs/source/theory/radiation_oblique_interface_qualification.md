# Oblique moving-interface qualification

Status: prepared; no production pass claimed. The eight-step CPU execution
probe only verifies setup and measures runtime. It is not a physics gate.

The interface normal is n=(2,0,1)/sqrt(5), rather than an axis or a symmetry
diagonal. Set Lx=L sqrt(5)/2 and Lz=L sqrt(5), so the periodic normal coordinate
s=(2x+z)/sqrt(5) modulo L reproduces the one-dimensional F-M2 profile. Radiation
flux and material drift are rotated into n; opacity, density and native caloric
parameters are unchanged. Four particles per cell sample a genuine 2D tensor
shape. This exercises both physical force components and transverse errors.

The FLASH F-M2 planar solution has already been independently run, refined and
visualized. Its rotation supplies an **indirect continuum reference**, not an
additional 2D FLASH run or a native kinetic-ion closure. The atomic-mass/caloric
differences documented for F-M2 remain. The stronger direct comparison is to
the refined native 1D case with the same particle-shape order and native EOS.

Qualification requires the full 3.333333333 ns interval (0.3 L translation),
not the initialization probe. Preserve actual radiation E/q, ion finite kinetic
work and particle-owned carries, and native nodal caloric energy. Require raw
energy closure within 1e-8 of initial radiation energy and each Cartesian
momentum balance within 1e-8 E0/c plus the declared initial-material rounding
bound. Require appreciable ion work, exceeding 1e-3 E0. These are the existing
strong-interface inventory gates, not relaxed multidimensional alternatives.

The exact planar continuum solution has zero transverse material acceleration
and radiation flux. Report both transverse fields independently of global
cancellation. Compare their reduction, and the normal density/radiation/caloric
profiles, under mesh refinement against the native 1D sequence. Do not accept
global conservation alone as evidence of vector-force accuracy. Changed-rank
restart and CPU/GPU execution must preserve the accepted trajectory. A physical
open-boundary interface remains a separate, currently unsupported obligation.

## Current evidence

The P40 64x128 run completes all 6400 steps. Its raw energy residual is
5.135e-12 of initial radiation energy, with Cartesian momentum residuals
(-3.822e-14, 0, -3.198e-14) in the run's unit-depth inventory. Radiation loses
2636.64 J, ions gain 5495.60 J and native electron caloric energy loses
2858.96 J. This coarse-grid partition differs substantially from the refined
planar case; conservation is not a vector-accuracy or partition-accuracy pass.
Final mass-weighted transverse velocity RMS is 0.003280 of the initial drift,
and transverse radiation q has L1/E=1.235e-7.

The 128x256 P40 run also completes 6400 steps and passes raw inventories
(energy residual 5.170e-12). Ion kinetic gain is 1559.50 J and electron caloric
gain 2352.52 J. Transverse velocity RMS falls to 0.0003091 of the drift (10.61x
reduction), and transverse q L1/E falls to 2.540e-8 (4.862x). Normal-profile
L1 differences from the **finite** native 1D N=1024 reference fall from
4.693% to 2.426% (E), 6.211% to 3.350% (q), 15.613% to 9.166% (charge), and
3.854% to 1.936% (caloric density). These are useful reductions, but not an
exact reference or a claim that the density interface is adequately resolved.

The no-radiation 64x128 control gains 8384.19 J of ion kinetic energy and has
transverse velocity RMS 0.002971 of the drift. Thus material-grid errors are
substantial at this coarse resolution; their work must not be subtracted from
the radiation case as a fitted correction. The full CPU 64x128 run agrees with
P40: maximum normalized field differences are 4.894e-15 (E), 2.196e-14 (q),
2.238e-14 (charge), and 2.681e-14 (temperature). The full 128x256 one-to-two-GPU
restart now passes the explicit particle and field trajectory gates; this is
separate evidence from the short CI restart below.

`analysis_moving_moment_oblique.py` exports normal-coordinate profiles without
phase fitting, preserving their volume averages, and plots the actual 2D
radiation/charge and normal/transverse q fields. Its changed-rank comparison
checks particles and fields separately from the inventory gates.

A deliberately short 16x32, 16-step CI pair checks runtime wiring, native 2D
nodal inventory reading and two-to-one-rank restart. It is not the production
qualification above. The combined 1D/2D moving integration selection passes
13 tests in 29.93 seconds. The first CI attempt exposed an unused parameter;
the valid spelling is `interpolation.galerkin_scheme`, not
`warpx.galerkin_interpolation`. Collocated gathering already had the intended
value in the recorded production runs, so correcting the input does not change
those results. Repeated-run analysis now excludes AMReX `.old.*` plot archives;
it does not delete or score those archived outputs as current results.
