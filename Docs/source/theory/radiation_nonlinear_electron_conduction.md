# Conduction with a nonideal electron EOS

The heat-conduction path now supports fixed-charge latent-energy electrons and
one file-backed Singularity-EOS electron material in Cartesian geometry, with
the RZ CPU extension qualified below.
It solves the nonlinear caloric equation; it does not approximate a finite
energy change as a frozen heat capacity multiplied by a temperature change.
The original ideal-electron Cartesian/RZ solver and its arithmetic remain
unchanged. Insulating scratch ghosts are now initialized deterministically;
they are not allowed to inherit allocator contents.

This is useful material-energy infrastructure for radiation calculations,
not a calibrated tungsten transport model. Ions remain kinetic; latent-energy
terms do not evolve the PIC charge state. Table EOSs retain the existing
host-only allocation/backend restriction.

## What is solved

At fixed density, one backward-Euler substep solves

`U_i(T_new) - U_i(T_old) + h*sum(g_ij*(T_i_new-T_j_new))/V_i = 0`.

The symmetric face conductance and native nodal volumes are the same quantities
used by the ideal solver. Each nonlinear graph iteration solves a scalar
monotone EOS equation at every node, using safeguarded Newton/bisection within
the previous global temperature range. EOS state/heat-capacity checks reject
invalid candidates. The accepted state must pass both the local nonlinear
equations and an independently integrated energy inventory.

The implementation does not assume that table energy is positive. An arbitrary
negative energy reference is allowed; neither stored energy nor temperature is
clipped to make a solve pass. The reported local redistribution is the actual
EOS energy difference, `U(T_new)-U(T_old)`.

Conductivity and the optional flux limiter remain **lagged per substep**. This
is not a jointly converged nonlinear conductivity/limiter solve. The initial
diagonal rate bound is eight; a rejected nonlinear stage may be retried with a
halved step, at most twenty refinements. All work stays private until the full
requested interval succeeds. Budget exhaustion preserves the caller's complete
temperature and redistribution output, including ghosts.

For a native table material, mass density is derived from its deposited
**unit-ion charge density**, `q_e*n_i`, with the material mass conversion. It is
not derived from physical electron charge assuming `Z=1`. A native fixed-`Z=20`
test exercises this distinction.

## Bounded qualification

The profile tests run 64 nonlinear steps through a finite redistribution of a
hot/cold periodic profile. Independent Python references solve
`Cv(T)*dT/dt = kappa*d2T/dx2` with continuous-time BDF integration and 256/512
point spatial refinement. The table fixture has nonlinear energy and heat
capacity, tungsten metadata, and an electron-only SP5 file; its coefficients
are manufactured, not measured tungsten data.

| Case | Relative profile error, normalized by the finite change | Relative energy-inventory error |
|---|---|---|
| Latent-energy conduction | 0.262% | `8.8e-15` |
| File-backed nonlinear EOS | 0.505% | `2.8e-12` |
| Same EOS with negative energy reference | 0.505% | `5.7e-12` |
| Native fixed-`Z=20` simulation, 64 PIC steps with stationary ions | 0.505%; density unchanged | The core's EOS energy gate remains active |

The refined-reference differences are below 0.003% of the physical profile
change. The native comparison explicitly selects classical conduction
(`electron_conduction_flux_limiter=0`), matching its PDE reference; retaining
the default limiter would compare different heat-flux closures. Plotfile
temperatures are compared using the actual nodal-to-cell sampling, not by
inverting an averaged nonlinear internal energy.

The local eight-stage CTest selection passes, including a substep-budget
rollback, two-rank latent conduction, the generated table fixture and native
run/analysis stages. Single-precision compiler checks pass 16/16 ordinary
translation units and 4/4 table-enabled translation units. Table-enabled
checking also made existing EOS precision/index conversions explicit; it did
not qualify single-precision table physics or relax a numerical tolerance.

The existing OpenMP HDF5 CI job now enables the table EOS and runs these tests
and the original EOS unit gates, with explicit Python analysis dependencies.
CI success must still be checked on the actual PR head. Local CPU/compiler
passes are not a claim of completed CUDA runtime qualification.

The resumed 1D/2D table-enabled build completes. Its exact proposed CI
selection passes 16/16 stages in 14.46 s, including the material-opacity and
metadata checks, thermodynamics tests, generated fixtures, nonlinear profiles,
budget rollback and native table run/analysis.

## Remaining boundaries

- Moving nonideal RZ radiation/pressure-work coupling remains guarded; the
  conduction extension does not remove those independent restrictions.
- Multiple table materials, evolving charge state and calibrated high-Z
  conductivity/opacity/EOS consistency are not supplied by this increment.
- Conductivity is isotropic; no magnetic anisotropy or nonlocal heat transport
  is implemented.
- The bounded graph iteration is not a scalable multigrid replacement for very
  stiff, large problems. The lagged limiter is not a hard bound on final flux.
- No FLASH fluid-ion closure is imposed on hybrid-PIC ions, and no new FLASH or
  hohlraum application run is claimed here.

The implementation and tests address a missing material-energy feature first.
The next conduction work should provide a scalable stiff solve, then add
anisotropy where the intended physical regime requires it.

## RZ nonlinear calorics and native thermal interface

The radial tests evolve an insulating hot core/cool exterior over 30 ns in a
1 mm-radius cylinder, using 64 radial cells and 64 time steps. The low-level
reference independently solves the cylindrical nonlinear heat equation on
256/512-point grids. Native tests additionally evolve the real hybrid electron
state at the actual PIC-deposited nodal density. They inspect nodal EOS energy,
not a nonlinear EOS evaluated at a cell-averaged temperature. No flat-density
FLASH fluid closure is imposed near the axis or wall.

| RZ case | Relative profile error | Relative energy error |
|---|---|---|
| Latent-energy radial heat redistribution | 0.2314% | `5.87e-13` |
| Manufactured table EOS | 0.3629% | `3.33e-12` |
| Same table with negative energy reference | 0.3629% | `3.16e-12` |
| Native latent-energy thermal evolution | 0.2827% | `5.95e-13` |
| Native fixed-Z=20 table thermal evolution | 0.3931% | `3.32e-12` |

The registered table-enabled RZ CPU matrix passes 14/14 checks, including
serial/MPI native profiles, generated fixtures, rollback and both table energy
references. The surrounding Cartesian latent/budget/MPI selection passes 3/3.
The 12 affected single-precision source checks pass. The RZ latent-energy CUDA
matrix passes 7/7 native/reference/rollback/MPI checks in 70.99 s; table EOS
remains host-only. The table-enabled CI job now includes the RZ cases alongside
the existing Cartesian gates.

Native qualification exposed uninitialized insulating ghosts in the private
temperature candidate. The nonlinear solver correctly rejected its non-finite
input. Filling those ghosts before the solve repaired the defect without
weakening the finite-state check. A poisoned-input-ghost case now requires
finite completed output. Retry failures also distinguish invalid inputs,
local solves, iteration exhaustion and energy balance, with residuals reported
in scientific notation.

Native nonideal thermal cases select `electron_energy_transport=finite_volume`.
The separate implicit advective option remains ideal-only; implicit nonlinear
conduction does not qualify that advective solver. These native thermal tests
hold ion positions fixed. Actual moving radiation evidence is documented
separately, and is not automatically extended to nonideal EOSs.
