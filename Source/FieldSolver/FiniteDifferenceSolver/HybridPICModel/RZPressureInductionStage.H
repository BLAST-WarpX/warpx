/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#pragma once

#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronThermodynamics.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

namespace warpx::hybrid {
/** \brief Transactional ideal-electron pressure-induction/caloric stage.
 *
 * B_theta, U, and T are collocated nodal electron/magnetic states.  The
 * positive electron charge-density magnitude and native electron control
 * volume are read-only.  Pressure is reconstructed only with the supplied EOS
 * from U and charge; it is never an independent input.
 *
 * The caller must supply independently owned, nonaliasing MultiFabs: distinct
 * C++ objects do not by themselves prove that AMReX alias backing is absent.
 * `temperature_floor` is a finite, nonnegative private stage-admissibility
 * floor.  It is combined with the executor's own floor and ceiling without
 * replacing either EOS limit or changing the EOS law.  In particular, the
 * charge-only ideal executor currently reports a zero minimum temperature and
 * a finite maximum-Real ceiling; an explicit stage floor makes candidate-floor
 * rejection testable without claiming the ideal EOS itself has a physical
 * floor.
 *
 * The complete update is first performed in scratch.  Every initial and
 * candidate EOS inversion is checked before B_theta, U, and T are committed
 * together.  Any failed update, including one rejected only by the candidate
 * thermal state, leaves every valid cell and every pre-existing ghost cell of
 * all three mutable fields unchanged.  This private helper does not enable a
 * native evolution path and does not qualify the zero-Ampere-current guard.
 */
bool tryAdvanceRZPressureInductionStage (
    amrex::MultiFab& B_theta, amrex::MultiFab& U, amrex::MultiFab& T,
    amrex::MultiFab const& charge, amrex::MultiFab const& caloric_volume,
    amrex::Geometry const& geometry, amrex::Real dt,
    ElectronThermodynamicsExecutor eos, amrex::Real temperature_floor = 0);
} // namespace warpx::hybrid
