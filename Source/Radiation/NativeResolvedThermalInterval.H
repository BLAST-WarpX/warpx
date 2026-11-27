/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_NATIVERESOLVEDTHERMALINTERVAL_H_
#define WARPX_RADIATION_NATIVERESOLVEDTHERMALINTERVAL_H_

#include "CollidedDiffusionAdmission.H"
#include "NativePhotonPathReplay.H"
#include "NativeResolvedThermalSource.H"
#include "ResolvedThermalDiffusionField.H"

namespace warpx::radiation
{
struct NativeResolvedThermalIntervalOptions
{
    NativeResolvedThermalIntervalOptions ()
    {
        field.lte_centered_moving_source = true;
        field.maximum_beta = .001;
        thermal.allow_vacuum_noop = true;
    }
    NativeResolvedThermalSourceOptions thermal;
    ResolvedThermalDiffusionFieldOptions field;
    NativePhotonCollisionOptions collisions;
    CollidedDiffusionAdmissionComponents census;
    CollidedDiffusionAdmissionControls admission;
    int packet_subsegment_limit = 1024;
};

struct NativeResolvedThermalIntervalResult
{
    bool valid = false;
    std::string reason;
    NativeResolvedThermalSourceResult candidate;
    NativePhotonPathSubsegmentReport path;
    CollidedDiffusionAdmissionReport census;
    ResolvedThermalDiffusionFieldResult field;
    amrex::Long deferred_admissions = 0;
    amrex::Real raw_energy_residual = 0;
    amrex::GpuArray<amrex::Real, 3> raw_momentum_residual{};
};

/** ONE caller-private resolved gray thermal interval.
 *
 * Unlike transport followed by a pure-D source, this coordinator owns D spatial
 * diffusion, pressure/retained-q relaxation and thermal absorption together.
 * Every T/secant/work trial restarts the same supplied photon/material/D state,
 * replays coherent chronological paths using chi_s, solves masked spatial and
 * thermal D using chi_t=chi_s+chi_a, and applies continuous endpoint packet
 * capture/emission using proper chi_a. q_const comes from THAT total-opacity
 * flux exactly once. D recoil is (q_old-q_const)/c minus outward stress, with
 * actual emitted angular traction at D/K faces. There is no second chi_a q
 * damping, duplicate pressure force or separate material/caloric publication.
 *
 * One combined finite native kick and one native nodal caloric solve use all
 * packet/D energy debit, actual finite work and numerical carry. Coherent
 * path histories and birth logical keys are reused across all trials. Source
 * endpoint splitting and accepted-time upwind enthalpy are first order; capture is NOT
 * integrated along packet residence paths. Leakage/census are private pure
 * representation transfers, independently remeasured in the complete ledger.
 * New cohorts failing the next-interval thermal/diffusion envelope remain K.
 * The old D population requires a positive comoving angular dipole; the new
 * LTE-centered field mode requires chi_t*dx>=8, c*dt*chi_t>=8, beta<=.001 and
 * beta*chi_t*L<=.01, positive accepted fields and its implicit drift-CFL guard.
 * This is static constitutive diffusion, not finite-rate causal P1 storage.
 * Fixed-charge ideal/latent native electrons
 * and frozen proper gray coefficients/rate frame are initially supported.
 * Cartesian periodic level zero only. All native physical/rank failures reject
 * without live mutation, native IDs or logical interval consumption.
 *
 * Input owners/fields remain unchanged. candidate.material_stage is UNCOMMITTED
 * against caller material; candidate.photons is a nested private result. Outer
 * joint acceptance alone may adopt/commit these into private owners and publish
 * live material/photons/D/T. Do NOT first advance D through NativeInterval.
 * Initial endpoint-source mode requires OUTER c*dt<=min(dx)/2; packet-only
 * path subsegments do not establish absorption shadow accuracy for larger dt.
 */
[[nodiscard]] NativeResolvedThermalIntervalResult TryAdvanceNativeResolvedThermalInterval (
    WarpXParticleContainer& live_photons, PhotonPacketTransaction const& interval_photons,
    ParticleImpulseMaterial& interval_material, std::string const& carry_path,
    amrex::MultiFab const& old_diffusion_radiation, amrex::iMultiFab const& desired_diffusion_mask,
    amrex::MultiFab const& old_nodal_temperature,
    amrex::MultiFab const& coherent_scattering_opacity,
    amrex::MultiFab const& proper_absorption_opacity, amrex::MultiFab const& initial_rate_velocity,
    HybridPICModel const& hybrid, amrex::Geometry const& geometry, amrex::Real dt,
    NativeResolvedThermalIntervalOptions const& options);
} // namespace warpx::radiation
#endif
