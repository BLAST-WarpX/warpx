/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_COUPLEDMOMENTSOURCE_H_
#define WARPX_RADIATION_COUPLEDMOMENTSOURCE_H_

#include "ImplicitDipoleTransport.H"
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
    class MomentTransportLedger;
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

    enum class CoupledRadiationClosure { M1, ComovingDipole };

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
        // Opt-in prescribed comoving P1 backend. M1 defaults/operation order
        // are unchanged. Periodic Cartesian or guarded meridional RZ gray
        // source transactions; user-facing RZ activation remains separate.
        CoupledRadiationClosure closure = CoupledRadiationClosure::M1;
        // Use this backend's own policy, not M1's inexact-Newton inner cap.
        ImplicitDipoleTransportOptions dipole_transport;
    };

    enum class CoupledMomentFailure { None, Radiation, Material, Particle, IterationBudget };

    /** Actual particle-center inventory changes for an accepted source/interval.
     * Optional output, replaced only on success; failures preserve its previous
     * fields. These are not source-origin allocations of radiation impulse.
     */
    struct CoupledMomentExchange {
        amrex::MultiFab kinetic_work;
        amrex::MultiFab momentum;
        /** Global actual particle-owned pending (px,py,pz,work) changes.
         * Momentum uses kg m/s, work J; these are not inferred from radiation. */
        amrex::GpuArray<amrex::Real, 4> pending_change{};
        MomentTransportAccounting transport;
    };

    struct CoupledMomentResult {
        CoupledMomentFailure failure = CoupledMomentFailure::None;
        int iterations = 0;
        amrex::Real material_residual = 0;
        amrex::Real source_residual = 0;
        amrex::Real work_residual = 0;
        amrex::Real raw_energy_residual = 0;
        amrex::Real raw_momentum_residual = 0;
        // Source-origin versus particle-center torque/rbar partition difference
        // [J]. Not a momentum deficit; physical Lz is independently gated below.
        amrex::Real azimuthal_partition_difference = 0;
        // RZ basis-rotation balance divided by its physical + arithmetic bound.
        // raw_momentum_residual remains the unadjusted relative diagnostic.
        amrex::Real momentum_balance_residual = 0;
        // Closed RZ angular inventory uses annular mean radius times Qtheta.
        // The acceptance ratio includes explicit transverse rotation operands,
        // not a radiation-energy reservoir or an inferred torque correction.
        amrex::Real raw_angular_momentum_residual = 0;
        amrex::Real angular_balance_residual = 0;
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
     * Boundary scope is periodic Cartesian geometry or experimental meridional
     * RZ with opt-in fixed optical mirrors and nearest-cell meridional assignment,
     * or opt-in P1 with the native reflecting radial/closed nodal
     * assignment through cubic. Overlapping work/momentum checks are global
     * source-origin/particle-center identities with native operand-derived
     * arithmetic bounds, never per-source-cell actual-work identities.
     * Matched RZAngularConservative/rz_angular_transport options admit nearest
     * torque coupling; nodal P1 keeps its separately explicit angular transport
     * option. Angular
     * residuals use torque-conjugate impulses; exchange.momentum retains the
     * actual physical particle impulse. Native angular runtime activation and
     * particle drift remain separate guards. */
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
     * An optional cumulative transport ledger commits atomically with the
     * accepted interval; rejected attempts do not change its compensation.
     */
    CoupledMomentIntervalResult TryAdvanceCoupledMomentInterval (
        MultiParticleContainer& particles, std::vector<std::string> const& momentum_species,
        std::string const& carry_path, amrex::MultiFab& radiation,
        amrex::MultiFab& nodal_temperature, amrex::MultiFab& realized_material_energy,
        amrex::Geometry const& geometry, amrex::Real time, amrex::Real dt,
        CoupledMomentIntervalOptions const& options, CoupledMomentCallbacks const& callbacks,
        CoupledMomentExchange* exchange = nullptr, MomentTransportLedger* ledger = nullptr);
} // namespace warpx::radiation
#endif
