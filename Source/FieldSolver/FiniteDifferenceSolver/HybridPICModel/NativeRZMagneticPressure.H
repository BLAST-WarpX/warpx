/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#pragma once

#include "ElectronThermodynamics.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

class HybridPICModel;
class WarpX;

namespace warpx::hybrid {
/** Counters belong to one bounded native half-step, including rejected trials. */
struct NativeRZMagneticPressureDiagnostics
{
    int accepted = 0;
    int rejected = 0;
    int rk4_restarts = 0;
    amrex::Real max_ampere_current = 0;
};

/** Private fixed-charge, ideal-electron pressure channel on native RZ meshes.
 *
 * C maps edge E to cell-centered B_theta using the actual native Faraday call.
 * G is the forward nodal pressure gradient and A is the logarithmic-mean edge
 * charge. The outer PEC tangential edge is masked. With W the FieldEnergy
 * annular-cell quadrature, local caloric work is
 *     V dU/dt = P G^T A^-1 C^T W B / mu0.
 * The final local RK magnetic-work defect includes B's secant midpoint and
 * quadratic cross terms. Its opposite is partitioned equally among each
 * cell's four nodal caloric inventories on Yee, or applied at the same node on
 * a collocated mesh; there is no global redistribution. Collocated C/G and
 * their transposes include the axis and PEC coefficients and outer half-weight.
 * Trial U, T and B share the Butcher tableau; all work is
 * undone on rejection and a failed RK4 restarts the whole half-step in RKF45.
 *
 * This callable increment enters HybridPICModel::BfieldEvolve and FieldPush.
 * Ordinary evolution uses it only for the explicit pressure_only channel.
 * It does not extend collocated ion-work admission or supply
 * Hall/resistive/ion-current ledgers to the default full Ohm channel.
 * Independently owned nonaliasing U/T/charge/volume allocations are required.
 */
class NativeRZMagneticPressure
{
  public:
    NativeRZMagneticPressure (WarpX& sim, amrex::MultiFab& U, amrex::MultiFab& T,
                              amrex::MultiFab const& charge, amrex::MultiFab const& volume,
                              ElectronThermodynamicsExecutor eos,
                              NativeRZMagneticPressureDiagnostics& diagnostics);

    bool evolve (HybridPICModel& hybrid, amrex::Real half_dt, int step);
    bool refresh (HybridPICModel& hybrid);
    amrex::MultiFab const& pressure () const noexcept { return m_pressure; }
    void fillElectricField ();
    bool
    accepted () const noexcept
    {
        return m_accepted;
    }

  private:
    // CUDA extended lambdas require namespace entry points, not private
    // enclosing member functions. Keep the stage API and its data private.
    friend void FillNativeMagneticPressureScalar (
        NativeRZMagneticPressure const& state, amrex::MultiFab& field);
    friend bool ReconstructNativeMagneticPressure (
        NativeRZMagneticPressure& state, bool check_temperature);
    friend void EvaluateNativeMagneticPressureWork (
        NativeRZMagneticPressure const& state, amrex::MultiFab const& pressure,
        amrex::MultiFab const& magnetic, amrex::MultiFab& derivative);
    friend bool TrialNativeMagneticPressure (
        NativeRZMagneticPressure& state, HybridPICModel& hybrid,
        amrex::Real dt, bool adaptive, amrex::Real& error);

    bool reconstruct (bool check_temperature = false);
    void work (amrex::MultiFab const& pressure, amrex::MultiFab const& magnetic,
               amrex::MultiFab& derivative) const;
    bool trial (HybridPICModel& hybrid, amrex::Real dt, bool adaptive, amrex::Real& error);
    void fillScalar (amrex::MultiFab& field) const;

    WarpX& m_sim;
    amrex::MultiFab& m_U;
    amrex::MultiFab& m_T;
    amrex::MultiFab m_charge;
    amrex::MultiFab const& m_volume;
    ElectronThermodynamicsExecutor m_eos;
    NativeRZMagneticPressureDiagnostics& m_diagnostics;
    amrex::MultiFab m_pressure;
    bool m_collocated = false;
    bool m_accepted = false;
};
} // namespace warpx::hybrid
