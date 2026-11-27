/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_NATIVERESOLVEDRZTHERMALINTERVAL_H_
#define WARPX_RADIATION_NATIVERESOLVEDRZTHERMALINTERVAL_H_

#include "AdaptivePacketDiffusionTransition.H"
#include "NativeResolvedThermalSource.H"
#include "NativeRZThermalHistoryAdmission.H"
#include "RZNativePhotonPathReplay.H"
#include "ResolvedRZThermalDiffusionField.H"

namespace warpx::radiation
{
struct NativeResolvedRZThermalIntervalOptions
{
    NativeResolvedRZThermalIntervalOptions ();
    NativeResolvedThermalSourceOptions thermal;
    ResolvedRZThermalDiffusionFieldOptions field;
    NativePhotonCollisionOptions collisions;
    int packet_subsegment_limit = 1024;
    /** Engine-owned private History seam. Opt-in requires its immutable actual
     * step context and original outer clocks. Callers cannot issue contexts;
     * neither this pointer nor its authority escapes the synchronous call. */
    NativeRZThermalAbsorptionDepthPolicy absorption_depth_policy =
        NativeRZThermalAbsorptionDepthPolicy::WholeOuter16;
    /** Extended duration is a History-only synchronous capability. A caller
     * selecting this enum without the matching actual context/witness fails;
     * standalone Interval and all old path entries remain half-cell bounded. */
    NativeRZThermalPhotonTimePolicy photon_time_policy =
        NativeRZThermalPhotonTimePolicy::PhotonHalfCell;
    NativeRZThermalHistoryStepContext const* authenticated_history_step = nullptr;
    /** ONLY History may issue mode2's immutable no-K authority. Neither this
     * pointer nor its duration/owner authority escapes the synchronous leaf. */
    NativeRZThermalAllDNoKEventWitness const* all_d_no_k_event_witness = nullptr;
    /** PRIVATE CPU built-in fixed-mask closed-z radiation trial, not material
     * transport or normal runtime. Must agree with thermal spatial endpoint,
     * reciprocal packet and field reflecting selectors. Adaptive calls reject. */
    bool reflecting_axial_fixed_mask_trial = false;
    /** PRIVATE default-OFF one-fixed-charge CPU TABLE extension of that
     * fixed-mask closed source; must match thermal's private table selector.
     * Existing closed TABLE refusals remain when this option is false. */
    bool reflecting_axial_table_trial = false;
    bool single_selected_table_trial = false;
    /** PRIVATE fixed-mask history seam. Nonnull supplies complete current
     * photon clocks, frozen identically for EVERY nonlinear trial. Collision
     * seed/interval remain the enclosing outer clock namespace; thermal packet
     * seed/interval may use an explicitly distinct subinterval birth namespace.
     * Default null retains all existing same-namespace/prepare contracts.
     * No optical, spatial, material or caloric acceptance guard is relaxed.
     */
    std::vector<NativePhotonPathCarry> const* initial_photon_clocks = nullptr;
    /** Opt-in pure representation transfer AFTER the accepted physical source.
     * The caller conservatively demotes before entering this interval. Both
     * pointers are required together; default fixed-mask calls are unchanged.
     */
    amrex::iMultiFab const* desired_diffusion_mask = nullptr;
    amrex::MultiFab const* radiation_representation_carry = nullptr;
    CollidedDiffusionAdmissionComponents census;
    CollidedDiffusionAdmissionControls admission;
};

struct NativeResolvedRZThermalIntervalResult
{
    bool valid = false;
    std::string reason;
    NativeResolvedThermalSourceResult candidate;
    NativePhotonPathSubsegmentReport path;
    /** Defined only for adaptive calls. These and candidate photons/D/T/native
     * material are one joint private publication; carry is signed radiation
     * representation roundoff, never material work, heat or force. */
    amrex::iMultiFab accepted_diffusion_mask;
    amrex::MultiFab radiation_representation_carry;
    amrex::MultiFab admitted_four_inventory;
    amrex::Long admitted_photons = 0;
    amrex::Long promoted_cells = 0;
    amrex::Long deferred_cells = 0;
    /** Independent complete photon+D+native finite work/carry+EOS ledger:
     * energy [J], axial cPz [J], and physical cLz [J m]. Radial momentum is
     * deliberately absent. Stationary wall momentum is an external account. */
    amrex::GpuArray<amrex::Real, 3> raw_inventory_residual{};
    /** External axial wall accounts [kg m/s], already GLOBAL. K remains in
     * path.path.wall_momentum[2]; D is copied once from candidate. No wall
     * energy/work/physical-Lz torque, and no additional material kick. */
    amrex::GpuArray<amrex::Real, 2> diffusion_wall_momentum_by_side{};
    amrex::Real diffusion_wall_momentum = 0;
    amrex::Real axial_wall_momentum_operand = 0;
    /** Count-derived external-wall summation arithmetic allowance [J=cPz]. */
    amrex::Real axial_wall_roundoff_bound = 0;
};

/** Read-only collective option/boundary preflight shared by interval/history.
 * Does not capture an owner or consume any random draw. */
[[nodiscard]] bool ValidateNativeResolvedRZThermalIntervalScope (
    HybridPICModel const& hybrid, amrex::Geometry const& geometry,
    NativeResolvedRZThermalIntervalOptions const& options);

/** ONE caller-private chronological RZ packet+implicit thermal-D interval.
 * Every temperature/secant/finite-work trial starts the exact same immutable
 * outer photon/material/D/T snapshot and keyed chronological path. Actual
 * path E and (cPr,cLz/rbar,cPz), reciprocal endpoint capture/births, and actual
 * accepted-face D/K photons all enter ONE total finite native kick, carry and
 * owner-local reciprocal caloric solve. The accepted annular current/stress
 * and cylindrical connection remain independently auditable. The prior total
 * native RequestedWork is authoritative; component projections using the
 * same secant velocity partition it without replacing finite native work.
 *
 * Optional adaptive promotion is a pure representation transfer of actual
 * path-collision census only, AFTER this one accepted physical source. Thermal
 * and leakage births are not collision evidence; wall-only rays also remain K.
 * Admission failing the actual accepted endpoint/frozen-input annular source
 * envelope is deferred. Future macrostep T/Cv/density/rate changes require
 * retained-D constitutive screening and constructive demotion before paths.
 * Representation carry joins the independently remeasured E/Pz/Lz account.
 *
 * Scope: axis-containing L0,m0, fixed or explicitly adaptive mixed mask,
 * ideal/builtin latent raw-density electrons, kinetic fixed-charge ions, frozen
 * proper gray chi_s/chi_a, reflecting radial walls and periodic z. The distinct
 * CPU built-in reflecting_axial_fixed_mask_trial admits stationary z endcaps
 * ONLY with all matching spatial-source/reciprocal-packet/field selectors and
 * no adaptive representation/census transfer. Material positions/density stay
 * frozen. K-path and D-stress wall accounts are external, not another kick.
 * beta<=.001,
 * chi_t*min(dx)>=8, c*dt*chi_t>=8 and beta*chi_t*L<=.01 in D, and OUTER
 * c*dt<=min(dx)/2 for first-order endpoint absorption. Packet subsegments do
 * not relax that outer accuracy guard. The separately witnessed History-only
 * AllDNoPhotonImplicitStep has no packet endpoint/path population and omits
 * ONLY that photon half-cell comparison, retaining all implicit D guards.
 * Arbitrary dynamics, AMR, tables and M1
 * remain unsupported. This private leaf does not activate normal runtime or
 * checkpoints and does not imply a FLASH fluid momentum/energy closure.
 *
 * All input owners/fields and the native allocator remain unchanged on both
 * success and rejection. candidate.material_stage remains uncommitted into
 * the caller's material owner, and candidate.photons is a nested private
 * transaction. Only outer joint acceptance may adopt/commit private stages;
 * final publication and birth IDs belong to the outer transaction.
 */
[[nodiscard]] NativeResolvedRZThermalIntervalResult TryAdvanceNativeResolvedRZThermalInterval (
    WarpXParticleContainer& live_photons, PhotonPacketTransaction const& interval_photons,
    ParticleImpulseMaterial& interval_material, std::string const& carry_path,
    amrex::MultiFab const& old_diffusion_radiation, amrex::iMultiFab const& diffusion_mask,
    amrex::MultiFab const& old_nodal_temperature,
    amrex::MultiFab const& coherent_scattering_opacity,
    amrex::MultiFab const& proper_absorption_opacity, amrex::MultiFab const& initial_rate_velocity,
    HybridPICModel const& hybrid, amrex::Geometry const& geometry, amrex::Real dt,
    NativeResolvedRZThermalIntervalOptions const& options);
} // namespace warpx::radiation
#endif
