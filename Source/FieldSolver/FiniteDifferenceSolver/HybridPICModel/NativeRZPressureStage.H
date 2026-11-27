/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#pragma once

#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronThermodynamics.H"

#include <AMReX_MultiFab.H>

class HybridPICModel;
class WarpX;

namespace warpx::hybrid
{

/** Diagnostics from the private native RZ pressure-stage trial. */
struct NativeRZPressureDiagnostics
{
    amrex::Real max_abs_ampere_current = amrex::Real(0.0);
    amrex::Real max_native_transpose_B_error = amrex::Real(0.0);
    amrex::Real max_native_transpose_B_scale = amrex::Real(0.0);
    amrex::Real max_native_transpose_U_error = amrex::Real(0.0);
    amrex::Real max_native_transpose_U_scale = amrex::Real(0.0);
};

/** \brief Perform one private, transactional native RZ pressure-only trial.
 *
 * This is a bounded integration container, not a production runtime
 * path and not a replacement for HybridPICModel::BfieldEvolve.  The magnetic
 * candidate is produced by the actual native cylindrical Ohm, E-boundary and
 * Faraday calls.  Ohm is invoked with its isolated pressure-electric output
 * copied into the Faraday E state, so Hall, ion-current, resistive, external
 * and hyper-resistive channels cannot enter this trial.  Charge is frozen and
 * the charge-only ideal EOS is mandatory.
 *
 * Initial pressure is integrated over dt/2; its caloric work produces a trial U/T
 * state, and pressure reconstructed from that trial U drives the final native
 * Faraday candidate.  Final caloric work uses the weighted transpose paired
 * with B0 and the actual final B candidate, including the midpoint cross term.
 * A disagreement with the qualified standalone operator rejects the trial.
 *
 * On rejection every rank rejects together and restores all valid and ghost
 * cells of live B/E and caller-owned U/T byte-for-byte.  E is scratch in this
 * interface and is restored on acceptance as well.  On acceptance B_theta and
 * U/T valid and ghost cells are replaced by the candidate with explicitly
 * filled periodic/even-scalar ghosts and toroidal ghosts odd at the axis,
 * even at the PEC wall; B_r and B_z remain frozen. Independently owned,
 * nonaliasing B/E/U/T/charge/volume allocations are a caller precondition.
 * EB, non-collocated/multimode configurations and particle-gather boundary
 * fills narrower than one cell are rejected before any native operator call.
 * Recoverable native-call exceptions propagate after local restoration;
 * process/device failure and failure of restoration itself are excluded.
 */
[[nodiscard]] bool tryAdvanceNativeRZPressureStage (
    WarpX& sim, HybridPICModel const& hybrid, amrex::MultiFab& internal_energy,
    amrex::MultiFab& temperature, amrex::MultiFab const& charge,
    amrex::MultiFab const& caloric_volume, amrex::Real dt, ElectronThermodynamicsExecutor eos,
    NativeRZPressureDiagnostics* diagnostics = nullptr);

} // namespace warpx::hybrid
