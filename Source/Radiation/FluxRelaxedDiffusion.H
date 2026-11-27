/* Copyright 2026 The WarpX Community
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_FLUXRELAXEDDIFFUSION_H_
#define WARPX_RADIATION_FLUXRELAXEDDIFFUSION_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <functional>

namespace warpx::radiation
{

/** Callback contract for the caller-owned kinetic-material candidate.
 *
 * requested_impulse is cell-integrated ion impulse [kg m/s].  The callback
 * must stage it with ParticleImpulseTransaction/ParticleImpulseMaterial and
 * fill requested_work with the corresponding radiation reservoir debit [J]:
 * ActualWork for cell carry, or RequestedWork = ActualWork +
 * EnergyCarryChange for particle carry.  It must not commit live particles.
 * The Boolean return is collective over ranks participating in this call.
 */
using FluxRelaxedWorkCallback =
    std::function<bool(amrex::MultiFab const& requested_impulse, amrex::MultiFab& requested_work)>;

struct FluxRelaxedDiffusionOptions
{
    int max_iterations = 100;
    amrex::Real relaxation = 0.8;
    bool verbose = false;
    int max_linear_iterations = 200;
    amrex::Real linear_tolerance = 1.e-14;
    amrex::Real tolerance = 1.e-13;
    amrex::Real energy_tolerance = 1.e-12;
    amrex::Real maximum_beta = 1.e-3;
    amrex::Real maximum_advective_cfl = 0.5;
};

enum class FluxRelaxedDiffusionFailure
{
    None,
    InvalidState,
    InvalidCallback,
    Iteration,
    Admissibility,
    LinearConvergence
};

enum class FluxRelaxedDiffusionAdmissibilitySite
{
    None,
    NonfiniteRightHandSide,
    EvaluateState,
    PhotonCone,
    RawGlobalLedger
};

struct FluxRelaxedDiffusionResult
{
    FluxRelaxedDiffusionResult () = default;
    FluxRelaxedDiffusionResult (FluxRelaxedDiffusionResult&&) = default;
    FluxRelaxedDiffusionResult& operator=(FluxRelaxedDiffusionResult&&) = default;
    FluxRelaxedDiffusionResult (FluxRelaxedDiffusionResult const&) = delete;
    FluxRelaxedDiffusionResult& operator=(FluxRelaxedDiffusionResult const&) = delete;
    FluxRelaxedDiffusionFailure failure = FluxRelaxedDiffusionFailure::None;
    int iterations = 0;
    // Maxima of independently evaluated equations divided by their local
    // physical operand scales.
    amrex::Real energy_equation_residual = 0.0;
    // Near vacuum, successful energy convergence may instead use only the
    // machine-roundoff absolute floor below. Both are in cell-integrated J.
    amrex::Real energy_equation_absolute_residual = 0.0;
    amrex::Real energy_roundoff_bound = 0.0;
    amrex::Real momentum_equation_residual = 0.0;
    amrex::Real collision_identity_residual = 0.0;
    // Absolute global raw conservation residuals [J] and [J] (the latter is
    // c times momentum).  Acceptance divides each by its physical scale.
    amrex::Real raw_energy_residual = 0.0;
    amrex::Real raw_momentum_residual = 0.0;
    // Diagnostic-only identity of the first aggregate admissibility rejection.
    // It never participates in acceptance or numerical decisions.
    FluxRelaxedDiffusionAdmissibilitySite admissibility_site =
        FluxRelaxedDiffusionAdmissibilitySite::None;
    bool valid = false;

    /** Accepted, caller-owned transaction inputs; zero on every failure. */
    amrex::MultiFab requested_impulse;
    amrex::MultiFab requested_work;
};

/** Advance gray scattering-only flux-relaxed P1 diffusion.
 *
 * radiation is cell-integrated [E,qx,qy,qz] with q=F/c=c*p [J].  The first
 * supported geometry is periodic Cartesian with one or more valid ghost
 * cells.  beta is a frozen Cartesian field with |beta| <= 1e-3.  Relabeling
 * packets into radiation does not kick ions: only the collision impulse
 * returned through the callback does.  Pressure flux transports radiation
 * momentum and is never charged to ions a second time.
 *
 * On failure radiation and its ghosts are unchanged and both returned
 * ledger MultiFabs are zero.  On success the caller may commit the already
 * staged kinetic transaction and then commit radiation in one coordinated
 * native step.
 */
FluxRelaxedDiffusionResult
TryAdvanceFluxRelaxedDiffusion (amrex::MultiFab& radiation, amrex::MultiFab const& opacity,
                                amrex::MultiFab const& frozen_beta, amrex::Geometry const& geometry,
                                amrex::Real dt, FluxRelaxedDiffusionOptions const& options,
                                FluxRelaxedWorkCallback const& work_callback);

} // namespace warpx::radiation

#endif // WARPX_RADIATION_FLUXRELAXEDDIFFUSION_H_
