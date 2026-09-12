/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_COUPLEDIMPLICITDIFFUSION_H_
#define WARPX_RADIATION_COUPLEDIMPLICITDIFFUSION_H_

#include "ImplicitDiffusion.H"

#include <functional>

namespace warpx::radiation
{
    /** Private stationary native-electron prototype. Coefficients
     * are evaluated at each supplied nodal temperature trial. material_response
     * is the native cell-to-node caloric map: temperature starts at T_old and
     * source starts at requested J/cell, returning candidate T and realized
     * J/cell. Callbacks may mutate only their explicit scratch outputs. */
    struct CoupledImplicitCallbacks
    {
        std::function<void(amrex::MultiFab const&, amrex::MultiFab&, amrex::MultiFab&,
                           amrex::MultiFab&, amrex::Real)>
            coefficients;
        std::function<amrex::Real(amrex::MultiFab&, amrex::MultiFab&)> material_response;
        // Optional native caloric metric: global maximum equation mismatch and
        // heating scale for (old, candidate, check). Required for nonlinear EOS.
        std::function<amrex::GpuArray<amrex::Real, 2>(
            amrex::MultiFab const&, amrex::MultiFab const&, amrex::MultiFab const&)>
            material_residual;
    };

    struct CoupledImplicitOptions
    {
        ImplicitDiffusionOptions diffusion;
        int max_iterations = 100;
        amrex::Real material_tolerance = 1.e-9;
        amrex::Real energy_tolerance = 1.e-10;
        amrex::Real relaxation = 0.5;
        // Retry this radiation/material stage with 2^n uniform substeps.
        // Zero preserves the original single-attempt behavior.
        int max_subdivisions = 0;
    };

    enum class CoupledImplicitFailure
    {
        None,
        Spatial,
        Material,
        IterationBudget,
        IntervalEnergyBalance
    };

    struct CoupledImplicitResult
    {
        ImplicitDiffusionResult diffusion;
        CoupledImplicitFailure failure = CoupledImplicitFailure::None;
        int iterations = 0;
        int source_consistency_corrections = 0;
        int rejected_attempts = 0;
        int accepted_substeps = 0;
        amrex::Real material_relative_residual = 0;
        amrex::Real raw_energy_relative_residual = 0;
        amrex::Real material_realization_residual = 0;
        amrex::Real native_source_realization_residual = 0;
    };

    /** All-or-nothing radiation/material stage, not whole-PIC rollback. On
     * failure radiation, nodal temperature and material source including ghosts
     * remain unchanged. No boundary/source transfer is accepted on failure.
     * The material residual is scaled by nodal heating, not the background
     * state; nonlinear EOS adapters supply their caloric-energy metric.
     * Runtime integration remains guarded. */
    CoupledImplicitResult TryAdvanceCoupledImplicitDiffusion (
        amrex::MultiFab& radiation, amrex::MultiFab& nodal_temperature,
        amrex::MultiFab& realized_material_energy, amrex::Geometry const& geometry,
        amrex::Real time, amrex::Real dt, CoupledImplicitOptions const& options,
        CoupledImplicitCallbacks const& callbacks);

    /** Retry only the stationary radiation/material stage. Coefficient callbacks
     * receive each substep's endpoint time; material callbacks start from that
     * substep's supplied old temperature. Nothing is committed unless the entire
     * interval succeeds. This is not rollback of an earlier PIC update. */
    CoupledImplicitResult TryAdvanceCoupledImplicitSubcycled (
        amrex::MultiFab& radiation, amrex::MultiFab& nodal_temperature,
        amrex::MultiFab& realized_material_energy, amrex::Geometry const& geometry,
        amrex::Real time, amrex::Real dt, CoupledImplicitOptions const& options,
        CoupledImplicitCallbacks const& callbacks);
} // namespace warpx::radiation
#endif
