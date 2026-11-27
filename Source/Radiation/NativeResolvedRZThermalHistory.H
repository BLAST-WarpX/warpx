/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_NATIVERESOLVEDRZTHERMALHISTORY_H_
#define WARPX_RADIATION_NATIVERESOLVEDRZTHERMALHISTORY_H_

#include "NativeResolvedRZThermalInterval.H"
#include "NativeRZHistoryAcceptanceBudget.H"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace warpx::radiation
{
/** Declared left-endpoint time and ACTUAL accepted private nodal T/native
 * rate frame. Density and particle positions remain frozen in the native
 * hybrid/source context for the entire history. Write proper gray chi_s/chi_a
 * into supplied private cell fields. False is a collective outer rejection.
 * The evaluator must not mutate any input/live owner or use stochastic draws.
 */
using NativeRZThermalHistoryOpacityEvaluator = std::function<bool(
    amrex::Real, amrex::MultiFab const&, amrex::MultiFab const&,
    amrex::MultiFab&, amrex::MultiFab&, std::string&)>;

struct NativeResolvedRZThermalHistoryOptions
{
    int subintervals = 1;
    int maximum_subintervals = 32;
    /** Opt-in CPU periodic all-D true-selected TABLE material. History issues
     * a fresh immutable R/P/S epoch from native charge/unit-ion fields and
     * accepted-left T in each slice. Trial T changes the bath, not opacity.
     * The supplied table must remain loaded and immutable until return. */
    MaterialOpacityTable const* independent_gray_table = nullptr;
    detail::ResolvedIndependentGrayMaterialTableOpacityOptions independent_gray_opacity;
    /** Separate fixed-mask packet/FLD epoch. Retains whole-outer photon
     * optical16 and half-cell guards; never consumes an all-D/no-K witness. */
    bool independent_gray_mixed = false;
    /** Default preserves the runtime's original whole-outer depth audit. The
     * opt-in CPU true-TABLE pure-absorption branch authenticates every actual
     * History step. Mode2 requires engine-proven all-D/zero stored photons;
     * only implicit D absorption loses its packet-depth cap. Scattering16,
     * packet/D/clock/outer physical guards remain unchanged. */
    NativeRZThermalAbsorptionDepthPolicy absorption_depth_policy =
        NativeRZThermalAbsorptionDepthPolicy::WholeOuter16;
    /** Default leaves ALL prior duration admission and identities unchanged.
     * Opt-in requires mode2 and engine-proven all-D/zero stored photons. It
     * removes only photon half-cell duration, never the actual D guards. */
    NativeRZThermalPhotonTimePolicy photon_time_policy =
        NativeRZThermalPhotonTimePolicy::PhotonHalfCell;
    std::string opacity_model_identity;
    NativeResolvedRZThermalIntervalOptions interval;
    /** PRIVATE CPU built-in L0/m0 gray axis/reflecting-r, periodic-z census hook;
     * requires raw-density native hybrid PIC. Default OFF leaves the
     * existing fixed-history path untouched. The caller supplies its actual
     * post-entry-demotion photon_start. Snapshot only the history-private
     * clone, and initialize actual accepted births before their later paths.
     * This is metadata, not adaptive conversion or a normal runtime selector.
     */
    bool retain_collided_census = false;
    CollidedDiffusionAdmissionComponents collided_census;
};

struct NativeRZThermalHistoryCensusReceipt
{
    bool valid = false;
    /** LOCAL final-survivor clocks, in the same traversal order as the
     * history's photon_clocks, restricted to actual completed_events>0.
     * Fresh-outer initialization was event zero; counts are cumulative over
     * this outer only, including earlier slices, never just the last slice.
     * Native idcpu and tagged provisional birth keys remain distinct.
     * Absorbed rays are NOT final clocks or promotion evidence.
     */
    std::vector<NativePhotonPathCarry> collided_clocks;
    /** LOCAL actual staging receipts, concatenated in accepted slice order.
     * Producer ownership need not equal final particle ownership. Includes
     * births later absorbed; does NOT claim these are final surviving rays.
     * Each accepted staging batch's global uniqueness/completeness is checked
     * by the existing reconciliation directory, not by summing a replicated
     * key/clock payload. Neither vector is restart state.
     */
    std::vector<std::uint64_t> actual_birth_keys;
};

struct NativeResolvedRZThermalHistoryStep
{
    amrex::Real time = 0;
    amrex::Real dt = 0;
    amrex::MultiFab initial_rate_velocity;
    amrex::MultiFab endpoint_rate_velocity;
    amrex::MultiFab scattering_opacity;
    amrex::MultiFab absorption_opacity;
    /** Actual accepted original leaf diagnostics/face accounts. Its material
     * stage has committed ONLY into the result-private material; its photons
     * are consumed by outer private AdoptCandidate, never live publication.
     */
    NativeResolvedRZThermalIntervalResult accepted;
};

struct NativeResolvedRZThermalHistoryResult
{
    bool valid = false;
    std::string reason;
    int completed_private_subintervals = 0;
    std::string policy_identity;
    std::unique_ptr<ParticleImpulseMaterial> material;
    std::unique_ptr<PhotonPacketTransaction> photons;
    amrex::MultiFab nodal_temperature;
    amrex::MultiFab diffusion_radiation;
    amrex::MultiFab actual_material_work;
    amrex::MultiFab material_energy_carry_change;
    amrex::MultiFab actual_heat;
    /** Physical local cylindrical impulse published to the ordinary material
     * momentum diagnostic. This is distinct from the work/torque-conjugate
     * field used by the E/Pz/Lz ledger below. */
    amrex::MultiFab actual_material_impulse;
    amrex::MultiFab actual_conjugate_impulse;
    amrex::MultiFab conjugate_momentum_carry_change;
    amrex::GpuArray<amrex::Real, 3> wall_momentum{};
    /** Separate GLOBAL stationary K and D axial wall accounts [kg m/s].
     * Accumulated once per accepted private slice, never MPI-summed again. */
    amrex::GpuArray<amrex::Real, 2> kinetic_wall_momentum_by_side{};
    amrex::GpuArray<amrex::Real, 2> diffusion_wall_momentum_by_side{};
    amrex::Real diffusion_wall_momentum = 0;
    amrex::Real axial_wall_momentum_operand = 0;
    /** Independent outer endpoint E, cPz, physical cLz residuals, NOT sums
     * of reported leaf residuals. Bounds use uncancelled actual operands and
     * accepted per-step storage/stage/caloric arithmetic bounds. */
    amrex::GpuArray<amrex::Real, 3> endpoint_residual{};
    amrex::GpuArray<amrex::Real, 3> endpoint_roundoff_bound{};
    /** Sum of qualified leaf physical E/Pz/Lz acceptance scales: actual slice
     * radiation plus uncancelled heat/work/carry changes. Lz also has a lever;
     * background material KE/EOS enters arithmetic only, never physical
     * tolerance.
     */
    amrex::Real endpoint_physical_energy_scale = 0;
    /** Exact immutable acceptance-law witness, including private closed wall
     * operands/residuals. Neither rounded diagnostic Real residuals nor old
     * count-dependent allowances certify photon coarsening. */
    NativeRZHistoryAcceptanceWitness acceptance_witness;
    std::vector<NativePhotonPathCarry> photon_clocks;
    NativeRZThermalHistoryCensusReceipt census_receipt;
    std::vector<NativeResolvedRZThermalHistoryStep> steps;
};

/** PRIVATE fixed-mask gray RZ material/opacity history solver. Normal runtime
 * consumes this API under its qualified accepted-outer fresh-clock policy.
 * Clone actual caller-private particles/carries with original live provenance.
 * Each qualified physical subinterval updates private T and native endpoint
 * rate, evaluates opacity at its declared left time/state, then performs ONE
 * existing packet+implicit-D+native work/carry/caloric solve. No second kick,
 * copied nonlinear solver or beta-force replacement occurs within a slice.
 * Source positions/density stay frozen; photons genuinely advance. Individual
 * slices retain optical16/event, half-cell and D temporal/spatial/beta guards.
 * The distinct AllDNoPhotonImplicitStep policy permits an actual implicit D
 * duration beyond photon half-cell ONLY under immutable all-D/empty owners,
 * clocks, mask, allocator and namespace authority. It retains scattering16
 * and every D/frame/work/caloric/outer law; it is not long photon transport.
 * Clock depth/event counters persist under the SAME enclosing draw namespace;
 * subinterval birth namespaces are distinct and all logical keys audited.
 * Native IDs are consumed ONLY by future explicit outer photon publication.
 * All supplied live/private particles/fields/carries/IDs remain unchanged on
 * success AND rejection. Invalid results expose no partial candidate owners.
 * Policy identity names private partition/time/state/namespace semantics;
 * normal runtime additionally binds its provider/epoch/checkpoint identity.
 * Carries here are in-progress PRIVATE clocks inside this enclosing history.
 * Accepted-outer publication/restart keeps ordinary native particle and draw-key
 * state, discards these completed outer clocks and starts fresh event-zero
 * clocks in the next namespace. That policy needs no carry remap and does not
 * support in-progress checkpoint/restore or exposing provisional clock tags to
 * a live consumer. This API itself never publishes or serializes clocks.
 * An explicit private periodic single_selected_table_trial retains the actual
 * one-material registered EOS selector and requires every native corner select
 * material0. It does not activate normal registered-opacity transport.
 * CPU periodic-z additionally admits one fixed-charge unresolved-composition
 * table material with frozen physical-corner calorics/raw unit-ion density;
 * normal runtime binds the same table/provider and fresh accepted-outer policy.
 * Stationary closed-z radiation is a distinct default-false CPU built-in
 * ideal/latent fixed-mask interval option with separate globally accepted K/D
 * wall accounts. Normal runtime may consume it ONLY under the explicit
 * resolved_thermal_reflecting_z CPU boundary/checkpoint policy; this private
 * API still neither publishes nor serializes any state. This PRIVATE API
 * requires full outer half-cell admission for closed z; private periodic-z
 * histories retain their individual slice guards. NORMAL fixed histories and
 * the separate adaptive fresh-outer coordinator require full outer admission.
 * A distinct default-OFF reflecting_axial_table_trial admits the private CPU
 * one-fixed-charge TABLE composition with actual clipped corner calorics.
 * It does not lift normal runtime/checkpoint guards. Table+photon-weights and
 * closed TABLE+photon-weights remain unsupported. CPU built-in closed histories
 * may feed the separately qualified original-direction endpoint proposal; its
 * immutable signed K/D wall witness is retained without a second kick.
 * GPU tables/closed boundaries, table-opacity data, M1, adaptive masks,
 * multigroup or moving source positions/density remain unsupported. Native
 * material motion outside this frozen radiation phase is owned by hybrid PIC,
 * not a closed fluid momentum/energy closure claimed by this API.
 */
[[nodiscard]] NativeResolvedRZThermalHistoryResult
TryAdvanceNativeResolvedRZThermalHistory (
    WarpXParticleContainer& live_photons,
    PhotonPacketTransaction const& photon_start,
    ParticleImpulseMaterial const& material_start,
    std::string const& carry_path,
    amrex::MultiFab const& old_diffusion_radiation,
    amrex::iMultiFab const& fixed_mask,
    amrex::MultiFab const& old_nodal_temperature, HybridPICModel const& hybrid,
    amrex::Geometry const& geometry, amrex::Real start_time,
    amrex::Real outer_dt,
    NativeRZThermalHistoryOpacityEvaluator const& evaluate_opacity,
    NativeResolvedRZThermalHistoryOptions const& options);
} // namespace warpx::radiation
#endif
