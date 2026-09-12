/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_COUPLEDMOMENTSOURCE_H_
#define WARPX_RADIATION_COUPLEDMOMENTSOURCE_H_

#include "ImplicitMomentTransport.H"
#include "ParticleImpulse.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <functional>
#include <string>
#include <vector>

class MultiParticleContainer;

namespace warpx::radiation
{
    class ParticleImpulseMaterial;
    struct CoupledMomentCallbacks {
        // Physical inverse-length absorption/scattering and cell-integrated
        // equilibrium radiation energy, evaluated at a native nodal trial T.
        std::function<void(amrex::MultiFab const&, amrex::MultiFab&, amrex::MultiFab&,
                           amrex::MultiFab&, amrex::Real)>
            coefficients;
        // Old native T and requested J/cell enter; candidate T and realized
        // J/cell exit. Non-finite return rejects the candidate without commit.
        std::function<amrex::Real(amrex::MultiFab&, amrex::MultiFab&)> material_response;
        // Optional prescribed-T equation check. Old T/requested heat enter;
        // actual T/realized heat exit. Nodal output components are energy
        // residual, source scale and arithmetic bound, respectively.
        std::function<amrex::Real(amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab const&,
                                  amrex::MultiFab&)>
            material_at_temperature;
    };

    struct CoupledMomentOptions {
        int max_iterations = 100;
        amrex::Real tolerance = 1.e-11;
        amrex::Real energy_tolerance = 1.e-10;
        amrex::Real relaxation = 0.5;
        // Native electron fluid is nonrelativistic, even though gray frame
        // algebra and the particle kinetic-work calculation retain gamma.
        amrex::Real maximum_beta = 0.01;
        bool verbose = false;
        bool spatial_transport = false;
        ParticleImpulseAssignment particle_assignment = ParticleImpulseAssignment::NearestCell;
        // Bound each inexact Newton correction; the complete nonlinear gates
        // still determine acceptance, and callers can increase this budget.
        ImplicitMomentTransportOptions transport{.max_linear_iterations = 40};
    };

    enum class CoupledMomentFailure { None, Radiation, Material, Particle, IterationBudget };

    /** Actual particle-center inventory changes for an accepted source/interval.
     * Optional output, replaced only on success; failures preserve its previous
     * fields. These are not source-origin allocations of radiation impulse.
     */
    struct CoupledMomentExchange {
        amrex::MultiFab kinetic_work;
        amrex::MultiFab momentum;
    };

    struct CoupledMomentResult {
        CoupledMomentFailure failure = CoupledMomentFailure::None;
        int iterations = 0;
        amrex::Real material_residual = 0;
        amrex::Real source_residual = 0;
        amrex::Real work_residual = 0;
        amrex::Real raw_energy_residual = 0;
        amrex::Real raw_momentum_residual = 0;
        amrex::Real actual_kinetic_work = 0;
        amrex::Real carry_energy_change = 0;
        amrex::Real native_realization_residual = 0;
        bool valid = false;
    };

    /** Atomic gray source stage: radiation stores cell-integrated (E,F/c),
     * not flux density. On failure all supplied fields and particles remain
     * unchanged. On success the caller must refresh derived native pressure
     * from the accepted nodal temperature. Spatial moment transport is optional;
     * no PIC drift or density advance is performed,
     * and no interval-wide subcycling/retry is implied by this single stage.
     * Initial boundary scope is periodic Cartesian geometry. */
    CoupledMomentResult TryAdvanceCoupledMomentSource (
        MultiParticleContainer& particles, std::vector<std::string> const& momentum_species,
        std::string const& carry_path, amrex::MultiFab& radiation,
        amrex::MultiFab& nodal_temperature, amrex::MultiFab& realized_material_energy,
        amrex::Geometry const& geometry, amrex::Real time, amrex::Real dt,
        CoupledMomentOptions const& options, CoupledMomentCallbacks const& callbacks,
        CoupledMomentExchange* exchange = nullptr);

    // Same source stage applied to a private multi-stage material copy. Its
    // accepted kicks remain private until that material's explicit Commit.
    CoupledMomentResult
    TryAdvanceCoupledMomentSource (ParticleImpulseMaterial& material, std::string const& carry_path,
                                   amrex::MultiFab& radiation, amrex::MultiFab& nodal_temperature,
                                   amrex::MultiFab& realized_material_energy,
                                   amrex::Geometry const& geometry, amrex::Real time,
                                   amrex::Real dt, CoupledMomentOptions const& options,
                                   CoupledMomentCallbacks const& callbacks,
                                   CoupledMomentExchange* exchange = nullptr);

    struct CoupledMomentIntervalOptions {
        CoupledMomentOptions source;
        int initial_substeps = 1;
        int max_refinements = 8;
    };

    struct CoupledMomentIntervalResult {
        CoupledMomentResult last_source;
        int attempts = 0;
        int substeps = 0;
        int completed_trial_substeps = 0;
        amrex::Real actual_kinetic_work = 0;
        amrex::Real carry_energy_change = 0;
        amrex::Real raw_energy_residual = 0;
        bool valid = false;
    };

    /** Retry an entire source interval on private radiation, temperature and
     * kinetic-particle copies. On a failed substep, discard the whole attempt
     * and double its subdivision. Neither failed nor intermediate accepted
     * substeps touch live state. Callbacks must not mutate external state.
     * The optional spatial moment solve can be included in each substep. No
     * PIC drift, density update or derived native-pressure commit is performed.
     */
    CoupledMomentIntervalResult TryAdvanceCoupledMomentInterval (
        MultiParticleContainer& particles, std::vector<std::string> const& momentum_species,
        std::string const& carry_path, amrex::MultiFab& radiation,
        amrex::MultiFab& nodal_temperature, amrex::MultiFab& realized_material_energy,
        amrex::Geometry const& geometry, amrex::Real time, amrex::Real dt,
        CoupledMomentIntervalOptions const& options, CoupledMomentCallbacks const& callbacks,
        CoupledMomentExchange* exchange = nullptr);
} // namespace warpx::radiation
#endif
