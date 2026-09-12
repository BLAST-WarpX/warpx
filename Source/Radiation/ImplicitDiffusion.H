/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_IMPLICITDIFFUSION_H_
#define WARPX_RADIATION_IMPLICITDIFFUSION_H_

#include "EnergyGroups.H"

#include <AMReX_Array.H>
#include <AMReX_BaseFwd.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

namespace warpx::radiation
{
    /** Stationary, optically thick, transport-only backward-Euler FLD controls.
     * Opacity is fixed during this solve; the flux limiter is nonlinear.
     * This is not a coupled implicit material-temperature solver. */
    struct ImplicitDiffusionOptions
    {
        amrex::Real tolerance;
        amrex::Real linear_tolerance;
        amrex::Real minimum_optical_depth;
        int max_iterations;
        int max_linear_iterations;
        int verbosity;
        amrex::GpuArray<int, 3> boundary_lo;
        amrex::GpuArray<int, 3> boundary_hi;
        amrex::GpuArray<int, 6> bath_is_temperature;
        amrex::ParserExecutor<4> const* bath_executors;
        EnergyGroupsExecutor energy_groups;
        bool use_full_gradient = false;
        // Optional frozen material trial: dE/dt = emissivity - absorption_rate*E.
        // Both fields are cell centered with one component per group. Rates
        // have units 1/s and J/m^3/s. This alone is NOT a coupled material solve.
        amrex::MultiFab const* absorption_rate = nullptr;
        amrex::MultiFab const* emissivity = nullptr;
        amrex::Real nonlinear_relaxation = 0.8;
        // Solve for a correction using the independently evaluated face/source
        // residual, avoiding cancellation against a large equilibrium field.
        bool use_incremental_form = false;
    };

    enum class ImplicitDiffusionFailure
    {
        None,
        LinearConvergence,
        NonlinearConvergence,
        NonnegativeIterate
    };

    struct ImplicitDiffusionResult
    {
        amrex::Real escaped_energy = 0;
        amrex::Real injected_energy = 0;
        amrex::Real numerical_energy_residual = 0;
        amrex::Real maximum_relative_residual = 0;
        int nonlinear_iterations = 0;
        int linear_iterations = 0;
        ImplicitDiffusionFailure failure = ImplicitDiffusionFailure::None;
        amrex::Vector<amrex::Real> group_escaped_energy;
        amrex::Vector<amrex::Real> group_injected_energy;
        amrex::Vector<amrex::Real> group_material_energy;
    };

    /** Attempt a radiation-only update. A solver failure leaves every input
     * group and guard cell unchanged and returns zero accepted ledger transfers.
     * Invalid configuration/input states are fatal errors, not timestep retries.
     * The caller owns retry policy; this does not roll back any prior PIC update. */
    ImplicitDiffusionResult TryAdvanceImplicitDiffusion (amrex::MultiFab& radiation,
                                                         amrex::MultiFab const& opacity,
                                                         amrex::Geometry const& geometry,
                                                         amrex::Real time, amrex::Real dt,
                                                         ImplicitDiffusionOptions const& options);

    /** Re-evaluate the discrete equation at a supplied candidate and current
     * material coefficients, using exactly the solver's accepted-face formula.
     * Does not advance or mutate either state. Returns per-group trial transfers
     * and the maximum group-scaled equation residual, not accepted live ledgers. */
    ImplicitDiffusionResult EvaluateImplicitDiffusionResidual (
        amrex::MultiFab const& old_radiation, amrex::MultiFab const& candidate,
        amrex::MultiFab const& opacity, amrex::Geometry const& geometry, amrex::Real time,
        amrex::Real dt, ImplicitDiffusionOptions const& options);

    /** Solve all groups into scratch storage, committing only after every group
     * meets the positivity and nonlinear equation-residual checks. Input/output
     * radiation is energy per cell, not density. The same accepted face flux
     * defines the operator residual and the independent boundary ledger. */
    ImplicitDiffusionResult AdvanceImplicitDiffusion (amrex::MultiFab& radiation,
                                                      amrex::MultiFab const& opacity,
                                                      amrex::Geometry const& geometry,
                                                      amrex::Real time, amrex::Real dt,
                                                      ImplicitDiffusionOptions const& options);
} // namespace warpx::radiation
#endif
