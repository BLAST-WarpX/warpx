/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_NATIVERESOLVEDRZADAPTIVETHERMALHISTORY_H_
#define WARPX_RADIATION_NATIVERESOLVEDRZADAPTIVETHERMALHISTORY_H_

#include "NativeResolvedRZThermalHistory.H"
#include "NativeRZAdaptiveHistoryAcceptanceBudget.H"

namespace warpx::radiation
{
struct NativeResolvedRZAdaptiveThermalHistoryOptions
{
    /** Required declaration: input clocks are NOT carried from a prior outer.
     * Persistent logical keys remain native identity, not old event evidence. */
    bool fresh_outer_endpoint = false;
    NativeResolvedRZThermalHistoryOptions history;
    AdaptiveDiffusionMaskPolicy mask_policy;
    CollidedDiffusionAdmissionComponents census;
    CollidedDiffusionAdmissionControls admission;
};

struct NativeRZAdaptiveThermalHistoryAcceptanceWitness
{
    bool valid = false;
    /** Independent ORIGINAL material to accepted actual native endpoint,
     * including pending material carry; NOT sums of source-stage reports. */
    NativeRZAdaptiveHistoryOwnerWitness owners;
    NativeRZAdaptiveHistoryAcceptanceLimits limits;
    /** Physical [E,cPz,cLz]. Composed=inner-entry-exit; independently measured
     * includes actual original/final radiation carry exactly once. GLOBAL. */
    std::array<long double, 3> composed_residual{}, actual_residual{};
};

struct NativeResolvedRZAdaptiveThermalHistoryResult
{
    bool valid = false;
    std::string reason, policy_identity;
    int completed_private_subintervals = 0;
    /** Accepted FIXED-middle-mask history. Its material/T/work/heat are final;
     * its photon/D/clocks are PRE-promotion physical evidence only. No returned
     * clock is restart state or a clock for a promoted-away final ray. */
    NativeResolvedRZThermalHistoryResult history;
    /** FINAL private publication owners. Publish these photons/D/mask/carry
     * jointly with history material/T/force/work/heat, once after all audits. */
    std::unique_ptr<PhotonPacketTransaction> photons;
    amrex::MultiFab diffusion_radiation, radiation_representation_carry;
    amrex::iMultiFab accepted_diffusion_mask;
    AdaptivePacketDiffusionTransitionResult entry_transition, endpoint_transition;
    /** Failed deterministic endpoint constitutive admission defers the pure
     * promotion; its attempted transition is evidence, never published. */
    bool endpoint_promotion_deferred = false;
    NativeRZAdaptiveThermalHistoryAcceptanceWitness acceptance_witness;
};

/** PRIVATE CPU builtin ideal/latent periodic-z gray L0/m0 coordinator. NOT
 * normal-runtime activation, a serialized policy, or an in-progress restart.
 * Full OUTER half-cell admission occurs before any clone or demotion. Entry
 * D->K is conservative; the middle mask stays frozen for EVERY existing
 * accepted history slice; endpoint K->D uses this outer's actual collision
 * census and existing deterministic constitutive admission. No second kick,
 * solver replay, carried clocks, photon-weight proposal, table, closed wall,
 * GPU, adaptive slice, M1 or spectral/AMR scope is admitted.
 * Existing history null carry/adaptive/clock guards remain internal and exact.
 * Failure discards ALL candidates, including a late failure after an accepted
 * private slice. No owner/native ID/allocator is published by this function.
 */
[[nodiscard]] NativeResolvedRZAdaptiveThermalHistoryResult
TryAdvanceNativeResolvedRZAdaptiveThermalHistory (
    WarpXParticleContainer& live_photons,
    PhotonPacketTransaction const& photon_start,
    ParticleImpulseMaterial const& material_start,
    std::string const& carry_path,
    amrex::MultiFab const& old_diffusion_radiation,
    amrex::iMultiFab const& old_mask,
    amrex::MultiFab const& old_radiation_representation_carry,
    amrex::MultiFab const& old_nodal_temperature, HybridPICModel const& hybrid,
    amrex::Geometry const& geometry, amrex::Real start_time, amrex::Real outer_dt,
    NativeRZThermalHistoryOpacityEvaluator const& evaluate_opacity,
    NativeResolvedRZAdaptiveThermalHistoryOptions const& options);
} // namespace warpx::radiation
#endif
