/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_COUPLEDIMPLICITDIFFUSION_H_
#define WARPX_RADIATION_COUPLEDIMPLICITDIFFUSION_H_

#include "CaloricConvergence.H"
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
        // Optional energy-equation elimination for the primary caloric trial
        // only: fill its request [J/cell] from the supplied spatial candidate.
        // The callback must use immutable old state and the accepted spatial
        // flux, and may mutate only the source. Final acceptance still evaluates
        // the full constitutive source and its native material response.
        std::function<void(amrex::MultiFab const&, amrex::MultiFab&)>
            conservative_material_request;
        /** Optional nonlinear radiation RHS, refreshed immediately after each
         * coefficient evaluation. The first argument is the immutable incoming
         * radiation; fill the second with the complete extensive RHS for the
         * current coefficient trial of a conservative material source split.
         * Both RHS/rebase hooks must be supplied together or both absent.
         * This is not an external source API. Spatial solves, independent equation
         * checks and energy balance all use that same RHS. It must be finite
         * and nonnegative. This single-step hook is not supported by the
         * subcycled wrapper, whose interval ledger assumes an unchanged RHS. */
        std::function<void(amrex::MultiFab const&, amrex::MultiFab&)> radiation_rhs;
        /** Optional rebasing when radiation_rhs changes with temperature.
         * Given previous/current extensive RHS, adjust an existing static
         * material request/realization to the current convention, preserving
         * physical caloric heat: add sum_groups(current_rhs-previous_rhs).
         * This algebraic change is checked independently. Only the supplied
         * source may be mutated. Required together with radiation_rhs. */
        std::function<void(amrex::MultiFab const&, amrex::MultiFab const&, amrex::MultiFab&)>
            rebase_material_source;
        /** Optional node-local caloric acceptance. Arguments are immutable old,
         * candidate, physical check, lower/upper independently perturbed-source
         * temperature images and the requested material tolerance. The adapter
         * must validate the check against that independent bracket, and use
         * local heating/representation scales, never a hotter node's allowance.
         * Requires material_residual for the separate raw caloric diagnostic.
         * Radiation-equation and combined-energy gates remain independent. */
        std::function<LocalCaloricConvergence(
            amrex::MultiFab const&, amrex::MultiFab const&, amrex::MultiFab const&,
            amrex::MultiFab const&, amrex::MultiFab const&, amrex::Real)>
            material_local_residual;
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
        // Disable source realization reconciliation for independently audited sources.
        bool reconcile_material_source = true;
        // Backtrack an inadmissible caloric response only to generate the next
        // nonlinear temperature trial. Full admissible images use bounded vector
        // Aitken relaxation with unique periodic-node dot products. Accepted
        // equations always use the full source.
        bool safeguard_material_iteration = false;
        // Legacy ideal native caloric-map arithmetic allowance, only for the
        // default temperature metric. Custom caloric metrics retain their units.
        bool allow_ideal_temperature_roundoff = false;
        // Require node-local ideal-temperature convergence in addition to the
        // legacy global arithmetic allowance. Moving-thermal uses this for
        // every state; custom caloric metrics keep their own units.
        bool require_local_ideal_accuracy = false;
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
        // Raw mismatch remains visible even if the arithmetic allowance is used.
        amrex::Real material_relative_residual = 0;
        amrex::Real material_absolute_residual = 0;
        amrex::Real material_roundoff_bound = 0;
        /** Maximum selected node-local fixed-point defect, ideal temperature or
         * custom caloric energy. Zero when no local metric is requested. */
        amrex::Real material_maximum_local_relative_residual = 0;
        amrex::Real raw_energy_relative_residual = 0;
        amrex::Real material_realization_residual = 0;
        amrex::Real native_source_realization_residual = 0;
    };

    /** All-or-nothing radiation/material stage, not whole-PIC rollback. On
     * failure radiation, nodal temperature and material source including ghosts
     * remain unchanged. No boundary/source transfer is accepted on failure.
     * The material residual is scaled by nodal heating, not the background
     * state; nonlinear EOS adapters supply their caloric-energy metric.
     * Moving-thermal trials additionally require a node-local fixed-point
     * residual within that node's heat scale and representation allowance,
     * using either the ideal-temperature or an adapter-supplied caloric metric.
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
