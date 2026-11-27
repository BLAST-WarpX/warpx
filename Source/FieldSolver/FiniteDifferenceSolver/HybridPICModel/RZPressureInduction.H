/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

namespace warpx::hybrid
{
/** Private-stage m=0 pressure induction and its conjugate caloric work.
 * All fields are scalar collocated nodal values. Pressure and positive charge
 * density are frozen over the stage; caloric_volume is the native electron
 * control volume, which need not equal the magnetic FieldEnergy quadrature.
 * The axis and a stationary PEC radial wall enclose a periodic axial domain.
 * No vacuum floor, magnetic-current work from other Ohm terms, or particle
 * update is supplied here. This operator is not enabled in native evolution.
 *
 * Only B_theta changes under an axisymmetric scalar-pressure curl. Its secant
 * midpoint defines the weighted-transpose current; the same masked pressure
 * gradient supplies the opposite electron compression work. A nonfinite or
 * nonpositive candidate returns false without changing any output or ghost.
 * On success only valid B_theta/internal-energy cells commit; callers refill
 * their ghosts and invert their EOS before using the new thermodynamic state.
 */
bool tryAdvanceRZPressureInduction (amrex::MultiFab& magnetic_theta,
                                    amrex::MultiFab& internal_energy,
                                    amrex::MultiFab const& pressure, amrex::MultiFab const& charge,
                                    amrex::MultiFab const& caloric_volume,
                                    amrex::Geometry const& geometry, amrex::Real dt);
} // namespace warpx::hybrid
