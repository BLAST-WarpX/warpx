/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_NATIVERESOLVEDTHERMALSOURCE_H_
#define WARPX_RADIATION_NATIVERESOLVEDTHERMALSOURCE_H_

#include "NativePacketThermalEventReplay.H"
#include "NativeRZThermalLeakage.H"
#include "ParticleImpulse.H"
#include "PhotonPacketTransaction.H"

#include <AMReX_Array.H>

#include <array>
#include <functional>
#include <cmath>
#include <limits>

class HybridPICModel;

namespace warpx::radiation
{
struct ResolvedRZThermalDiffusionFieldOptions;
class NativeRZIndependentGrayOpacitySlice;
class NativeRZThermalHistoryStepContext;
enum class NativeThermalCaloricBackend
{
    TrialCvCellMean,
    ReciprocalNodalEmission
};
struct NativeResolvedThermalSourceOptions
{
    NativeResolvedThermalSourceOptions ()
    {
        packets.survival_weighted_emission = true;
    }
    NativePacketThermalEventOptions packets;
    int max_temperature_iterations = 40;
    int max_linear_iterations = 40;
    int max_material_iterations = 20;
    int max_line_search = 20;
    amrex::Real material_tolerance = 1.e-10;
    amrex::Real work_tolerance = 1.e-10;
    amrex::Real energy_tolerance = 1.e-10;
    amrex::Real minimum_electron_density = 1;
    /** Explicit true-vacuum no-op. Exactly zero native charge has no EOS
     * floor/capacity/source weight and retains its old T. Fully vacant cells
     * require zero opacity/D/source; nonzero source into absent matter rejects. */
    bool allow_vacuum_noop = false;
    /** Explicit periodic Cartesian native shapes 1-3 force/work support.
     * Source-origin heat uses requested work; physical kinetic work/carry
     * retain particle-center ownership and independent global accounting. */
    bool cartesian_native_nodal_force = false;
    /** Native RZ corner-support force for built-in ideal/latent calorics,
     * including the live hybrid reflecting-radial charge deposit and, under
     * the separately admitted closed spatial selector, both axial endcaps.
     * Distinct from registered TABLE vacuum admission. Source work and native
     * center-owned kinetic/carry accounts have independent global ownership. */
    bool rz_native_nodal_force = false;
    /** Explicit reciprocal corner bath/emission, initially qualified only
     * through the caller-private resolved spatial interval backend. */
    NativeThermalCaloricBackend caloric_backend = NativeThermalCaloricBackend::TrialCvCellMean;
    /** Separate new-model sampling control; legacy rays_per_cell is unchanged. */
    int reciprocal_emission_rays_per_corner = 8;
    /** Opt in to centered composite old-corner energy for roundoff-sized
     * physical copies of nodal temperature. The strict reciprocal contract
     * remains the default. */
    bool reciprocal_composite_old_temperature = false;
    /** Explicit m=0 RZ reciprocal native K+D source. Requires the reciprocal
     * backend, composite old-corner caloric contract, angular event replay,
     * and the caller-private RZ spatial trial. The Cartesian source and the
     * old mean-radius RZ endpoint remain unchanged when false. */
    bool rz_reciprocal_implicit = false;
    /** Caller-private CPU built-in zero-flight reciprocal endpoint with z
     * PEC/reflecting boundaries. Native material positions/density stay frozen;
     * finite kick/work is not closed material transport.
     * All cells must be K and every D component exactly zero. No D, leakage,
     * path, census, interval/history or normal-runtime admission is lifted. */
    bool rz_closed_reciprocal_endpoint_trial = false;
    /** Separate CPU built-in closed K+D endpoint scope, with accepted external
     * D-wall traction. No photon flight or closed material transport is owned.
     * Mutually exclusive with the exact all-K endpoint selector above. */
    bool rz_closed_reciprocal_spatial_endpoint_trial = false;
    /** PRIVATE default-OFF CPU one-fixed-charge TABLE composition of the
     * spatial closed endpoint and actual clipped physical-corner calorics.
     * Requires the spatial selector above, not the exact all-K endpoint.
     * This does not activate normal runtime, adaptive masks or GPU tables. */
    bool rz_closed_reciprocal_table_trial = false;
    /** Private CPU periodic sole selected material0; no normal activation. */
    bool rz_single_selected_table_trial = false;
    /** Borrowed independent mixed epoch must separately opt into exact vacuum. */
    bool rz_single_selected_table_vacuum = false;
    /** Caller-private CPU ideal all-D implicit trial coordinates. Internal
     * nodal increments retain sub-ULP corrections relative to the immutable
     * old temperature. Final independent native caloric realization, work
     * and conservation gates are unchanged. No input/runtime selector. */
    bool rz_centered_ideal_temperature_trial = false;
    /** Borrowed synchronous History-issued opacity epoch. Never retained in
     * a candidate or checkpoint. Default null preserves matched transport. */
    NativeRZIndependentGrayOpacitySlice const* rz_independent_gray_opacity = nullptr;

    [[nodiscard]] bool UsesRZNativeNodalForce () const noexcept
    {
        return rz_native_nodal_force || rz_single_selected_table_vacuum;
    }
    [[nodiscard]] bool UsesOverlappingMaterialForce () const noexcept
    {
        return cartesian_native_nodal_force || UsesRZNativeNodalForce();
    }
    [[nodiscard]] ParticleImpulseAssignment RZForceAssignment () const noexcept
    {
        return rz_native_nodal_force
                   ? (rz_closed_reciprocal_spatial_endpoint_trial
                          ? ParticleImpulseAssignment::RZAngularReflectingClosedNodalCellAverage
                          : ParticleImpulseAssignment::RZAngularReflectingRadialNodalCellAverage)
               : rz_single_selected_table_vacuum
                   ? ParticleImpulseAssignment::RZAngularNodalCellAverage
                   : ParticleImpulseAssignment::RZAngularConservative;
    }
};

struct NativeResolvedThermalSourceResult
{
    bool valid = false;
    std::string reason;
    int temperature_iterations = 0;
    int material_iterations = 0;
    /** Local unique-owner Krylov coordinate count and scalar global sum.
     * Empty ranks hold zero coordinates but join every field/scalar exchange. */
    amrex::Long caloric_owned_nodes = 0;
    amrex::Long caloric_global_nodes = 0;
    int caloric_peak_basis_vectors = 0;
    /** Number of table iterations whose interim requested Q was not directly
     * invertible and therefore used the prescribed-provider Newton direction.
     * Final accepted table states still require a certified inverse. */
    int table_inverse_newton_fallbacks = 0;
    /** Subset caused by an unaffordable interim table inverse. Prescribed
     * trials may still reach an affordable coupled root; the final state may not. */
    int table_unaffordable_newton_fallbacks = 0;
    /** Valid represented table probes meeting the unchanged globally reduced
     * caloric norm<=1 despite a nondecreasing raw merit. Subsequent native
     * work, certified final inverse and global inventories still must pass. */
    int table_converged_nondescent_probes = 0;
    /** The very first frozen-work source trial requested more cooling than
     * the table state could supply, before a coupled colder root was sought. */
    bool table_initial_inverse_unaffordable = false;
    amrex::Real maximum_caloric_residual = 0;
    amrex::Real raw_energy_residual = 0;
    amrex::Real energy_roundoff_bound = 0;
    amrex::MultiFab diffusion_radiation;
    amrex::MultiFab nodal_temperature;
    amrex::MultiFab physical_debit;
    amrex::MultiFab raw_debit;
    amrex::MultiFab operand_scale;
    amrex::MultiFab storage_roundoff_bound;
    amrex::MultiFab requested_heat;
    amrex::MultiFab actual_heat;
    amrex::MultiFab thermal_redistribution;
    /** Actual reciprocal origin terms, excluding the common W*total heat:
     * packet birth energy and the matching D corner bath contribution. */
    amrex::MultiFab reciprocal_corner_heat_offset;
    amrex::MultiFab nodal_requested_heat_density;
    /** Opt-in RZ field geometry accounts [J/cell]. The theta component of
     * outward stress includes its conservative connection; no global radial
     * momentum conservation is claimed. Empty for Cartesian/default paths. */
    amrex::MultiFab rz_outward_pressure_stress;
    amrex::MultiFab rz_theta_connection;
    int rz_field_linear_iterations = 0;
    amrex::Real rz_field_linear_residual = 0;
    /** Signed external D endcap impulses [kg m/s], low/high z. No wall energy
     * work or Lz torque. Flight's K-wall impulse remains a separate owner. */
    amrex::GpuArray<amrex::Real, 2> rz_axial_wall_momentum_by_side{};
    amrex::Real rz_axial_wall_momentum = 0;
    amrex::MultiFab rz_packet_debit;
    /** Path/thermal endpoint packet debit whose same-frame momentum work is
     * excluded from D's work input. Leakage's donor correction belongs to D,
     * so its newly emitted packet is not subtracted here. */
    amrex::MultiFab rz_work_excluded_packet_debit;
    /** Authoritative previous total native finite-work iterate minus the
     * same-frame endpoint packet momentum projection in D; zero in K.
     * This is the field work input, not beta dot the total impulse. */
    amrex::MultiFab rz_diffusion_work_input;
    /** Genuine frozen caloric corner weights for the independent-opacity
     * accepted-temperature/post-kick screen. Interval discards this scratch
     * before returning; empty on all existing paths. */
    amrex::MultiFab rz_independent_gray_corner_weights;
    /** Actual accepted D->K face photons and donor correction. The latter
     * replaces face leakage/stress by actual stored packet transfer within
     * this ONE source kick; it is not a heat reconciliation. */
    std::unique_ptr<NativeRZThermalLeakageResult> rz_leakage;
    /** Exact accepted D face currents/stresses and signed positive D->K
     * leakage. rz_leakage owns the actual staged face photons; an outer
     * interval must adopt them once, never emit or debit them again. */
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>
        rz_face_lab_energy_current;
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>
        rz_face_lab_stress;
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>
        rz_face_leaked_energy;
    std::unique_ptr<PhotonPacketTransaction> photons;
    /** RZ private-history ancestry receipt from ACTUAL thermal emission and
     * accepted D/K leakage staging, not a difference of endpoint key sets. */
    std::vector<std::uint64_t> rz_history_birth_keys;
    std::unique_ptr<ParticleImpulseTransaction> material_stage;
};

namespace detail
{
/** Existing strict descent, with an exact terminal exception ONLY for a valid
 * composite table probe. This changes no caloric acceptance or provider bound.
 * A nonconverged/invalid probe has no such exception. */
[[nodiscard]] inline bool AcceptedNativeThermalLineSearchProbe (
    bool valid, bool composite_table, amrex::Real caloric_norm,
    amrex::Real probe_merit, amrex::Real base_merit) noexcept
{
    return valid && ((composite_table && caloric_norm <= 1) || probe_merit < base_merit);
}

/** Exact opt-in RZ conjugate momentum acceptance predicate. The stage bound
 * accounts for transverse Cartesian/RZ representability; actual native carry
 * remains a separate measured field and is never filled from this residual. */
[[nodiscard]] inline bool AcceptedRZConjugateMomentumLedger (
    amrex::Real raw_cmomentum, amrex::Real actual_impulse,
    amrex::Real actual_carry, amrex::Real stored_cmomentum_roundoff,
    amrex::Real cmomentum_operand_scale, amrex::Real material_mass,
    amrex::Real stage_roundoff, amrex::Real& allowance) noexcept
{
    auto const epsilon = std::numeric_limits<amrex::Real>::epsilon();
    allowance = stored_cmomentum_roundoff / PhysConst::c + stage_roundoff +
        128 * epsilon *
            (cmomentum_operand_scale / PhysConst::c +
             std::abs(actual_impulse) + std::abs(actual_carry)) +
        4096 * epsilon * epsilon * material_mass * PhysConst::c;
    auto const defect = raw_cmomentum / PhysConst::c -
                        actual_impulse - actual_carry;
    return std::isfinite(defect + allowance) && allowance >= 0 &&
           std::abs(defect) <= allowance;
}

/** Global RZ E/Pz/Lz sum acceptance. The caller supplies independently
 * accumulated signed ledgers and their unsigned *uncancelled* operands. The
 * operation budget covers at most 24 rounded operations per physical cell
 * (radius conversion, signed combinations, scale, and local reductions),
 * eight per actual leakage face, two per MPI rank, and 64 terminal operations.
 * It bounds arithmetic, not a measured residual or physical closure error. */
[[nodiscard]] inline bool AcceptedRZGlobalNativeInventory (
    std::array<amrex::Real, 11> const& totals, amrex::Long cells, amrex::Long faces,
    int ranks, amrex::Real energy_roundoff, amrex::Real energy_tolerance,
    std::array<amrex::Real, 3>& allowance) noexcept
{
    auto const epsilon = std::numeric_limits<amrex::Real>::epsilon();
    auto const operations = 24.L * cells + 8.L * faces + 2.L * ranks + 64.L;
    auto const product = operations * epsilon;
    if (!(cells >= 0 && faces >= 0 && ranks > 0 && product < 0.5L))
    {
        return false;
    }
    // The nonnegative scale and stage-bound sums also round downward. Use
    // gamma/(1-gamma) to express their arithmetic error against the observed
    // scale, rather than an inaccessible exact sum.
    auto const gamma = product / (1 - product);
    auto const arithmetic = static_cast<amrex::Real>(gamma / (1 - gamma));
    allowance = {
        energy_roundoff + (energy_tolerance + arithmetic) * totals[2],
        totals[9] + arithmetic * (totals[5] + totals[9]),
        totals[10] + arithmetic * (totals[8] + totals[10])};
    return std::isfinite(totals[0] + totals[1] + allowance[0] +
                         totals[3] + totals[4] + allowance[1] +
                         totals[6] + totals[7] + allowance[2]) &&
           allowance[0] >= 0 && allowance[1] >= 0 && allowance[2] >= 0 &&
           std::abs(totals[0] - totals[1]) <= allowance[0] &&
           std::abs(totals[3] - totals[4]) <= allowance[1] &&
           std::abs(totals[6] - totals[7]) <= allowance[2];
}
} // namespace detail

/** Caller-private spatial/thermal radiation trial. Receives (nodal T, sampled
 * cell T, secant velocity, previous total native RequestedWork), returning
 * candidate photons/D fields, independently reconstructed local radiation
 * accounts and a stable physical energy-minus-frame-work field. Raw debit
 * subtracts outward spatial E/P transfers. Every call restarts the immutable
 * outer interval and its logical draws. No native material/caloric or live
 * mutation is allowed. The common solver converges the work input, ONE total
 * native kick and ONE caloric response, retaining all independent ledger gates. */
using NativeResolvedThermalTrialBackend =
    std::function<bool(amrex::MultiFab const&, amrex::MultiFab const&, amrex::MultiFab const&,
                       amrex::MultiFab const&, NativeThermalCornerEmission const*,
                       NativeResolvedThermalSourceResult&, amrex::MultiFab&, std::string&)>;

/** One implicit gray THERMAL source at a resolved transport endpoint.
 *
 * Transport/coherent scattering is a preceding first-order split operator.
 * Proper chi_a here is separate from that operator's chi_s and chi_t. Packet
 * capture and keyed thermal emission use NativePacketThermalEventReplay's
 * fixed-position endpoint convention. D's extensive LAB (E,F/c) follows the
 * exact Lorentz stress map of COMOVING P0=E0 I/3 with a positive angular dipole
 * |F0/c|<=E0/3. Its pure absorption source uses the four-vector force map; no
 * M1 closure, coherent scattering or spatial pressure work is applied here.
 *
 * Temperature trials reuse one immutable private photon endpoint, frozen
 * density, proper opacity, rate frame and logical angular draws. Every trial
 * stages ONE total packet+D native material kick, then ONE nodal source
 * Q=packet_actual_debit+D_retained_transfer-(actual_ion_work+delta_energy_carry).
 * D's independently measured stored-field transfer and arithmetic bounds
 * remain separate. No state clipping, source correction or free carry is used.
 * Ideal/latent native caloric response, or the guarded single-material RZ
 * composite-table response, and independent cell-corner EOS quadrature, raw
 * E/P inventories, finite ion work/carry and secant-frame consistency must all
 * pass before a result is returned.
 * Opacity is the proper inverse length and is temperature-independent in this
 * bounded gray operator. Cell temperatures use trial native heat capacities,
 * centered at the coldest active corner to preserve constant native T exactly
 * and avoid hot/cold weighted-mean cancellation; empty cells sample zero.
 * caloric cell-to-node deposition uses the native old-state caloric weights.
 * Packet optical depths obey the replay's declared maximum-depth envelope.
 * Endpoint splitting is first order in time, not along-path attenuation or a
 * combined spatial/thermal diffusion solve. Newton/GMRES vectors contain only
 * this rank's unique periodic owner nodes. Nodal fields exchange duplicate and
 * halo values; dot products, residual norms and decisions use scalar MPI
 * reductions. The small Hessenberg matrix remains replicated. No global nodal
 * vector/basis is allocated, including on ranks with no owned boxes.
 * Backtracking uses the GMRES-scaled residual's globally summed squared L2
 * merit with scales frozen at that Newton base. Final acceptance still requires
 * the unchanged physical per-node caloric ratio and every material/field gate.
 *
 * Scope: Cartesian periodic level zero with ideal/builtin latent electrons, or
 * fixed-mask gray m=0 RZ with either those models or one unresolved-composition
 * Singularity/Spiner table material. The table path is CPU-only, raw-density
 * native FV, fixed-charge and reciprocal-nodal; adaptive representation,
 * table opacity, ionization, multigroup and Cartesian table calorics remain
 * unsupported. All paths require double fields/particles, every represented
 * ion and frame beta<=.01. Positive native
 * source density is required at each active node; allow_vacuum_noop separately
 * admits exactly zero-density nodes with unchanged T, zero EOS capacity/weight
 * and zero source. Wholly vacant cells require zero opacity and zero D state.
 * The input D state is zero in K cells; the mask is binary, 1 for D and 0 for K.
 *
 * All supplied fields and BOTH caller private owners remain immutable. On
 * success, photons is a nested candidate and material_stage is an UNCOMMITTED
 * stage pointing into the caller's material owner. The caller may adopt these
 * accepted candidates ONLY into its private outer interval, using
 * material_stage->Commit() and photons_owner.AdoptCandidate(*result.photons),
 * and copy returned private fields. Final live publication and native birth
 * IDs remain exclusively owned by the outer jointly accepted interval.
 */
[[nodiscard]] NativeResolvedThermalSourceResult TryAdvanceNativeResolvedThermalSource (
    WarpXParticleContainer& live_photon_owner, PhotonPacketTransaction const& photon_endpoint,
    ParticleImpulseMaterial& material_endpoint, std::string const& carry_path,
    amrex::MultiFab const& diffusion_radiation, amrex::iMultiFab const& diffusion_mask,
    amrex::MultiFab const& nodal_temperature, amrex::MultiFab const& proper_absorption_opacity,
    amrex::MultiFab const& initial_rate_velocity, HybridPICModel const& hybrid,
    amrex::Geometry const& geometry, amrex::Real dt,
    NativeResolvedThermalSourceOptions const& options,
    NativeResolvedThermalTrialBackend const* private_trial_backend = nullptr);

/** Build the immutable m=0 RZ packet+D trial backend and solve the existing
 * owner-local temperature/finite-work coordinator. proper_transport_opacity
 * is chi_t for D and may differ from thermal chi_a; K uses packet chi_a only.
 * All outputs remain caller-private. This wrapper does not publish or activate
 * a normal-runtime model. */
[[nodiscard]] NativeResolvedThermalSourceResult TryAdvanceNativeResolvedRZThermalSource (
    WarpXParticleContainer& live_photon_owner, PhotonPacketTransaction const& photon_endpoint,
    ParticleImpulseMaterial& material_endpoint, std::string const& carry_path,
    amrex::MultiFab const& diffusion_radiation, amrex::iMultiFab const& diffusion_mask,
    amrex::MultiFab const& nodal_temperature, amrex::MultiFab const& proper_transport_opacity,
    amrex::MultiFab const& proper_absorption_opacity,
    amrex::MultiFab const& initial_rate_velocity, HybridPICModel const& hybrid,
    amrex::Geometry const& geometry, amrex::Real dt,
    NativeResolvedThermalSourceOptions const& options,
    ResolvedRZThermalDiffusionFieldOptions const& field_options);

/** Reusable private RZ endpoint packet+direct-B D trial for an outer
 * chronological interval. replay must have been prepared against exactly
 * private_endpoint; d_chi_t/d_chi_a are frozen D-only opacities (zero in K),
 * while replay retains the full thermal chi_a including K. old_diffusion is
 * the immutable old D state for this trial. prior_work starts from the
 * authoritative total native RequestedWork; an outer chronological caller
 * first subtracts only its upstream packet cP secant projection at this SAME
 * velocity, and this helper subtracts the thermal endpoint packet projection.
 * No material stage or live mutation occurs here. The caller adds upstream
 * path accounts before the common coordinator stages ONE total native kick. */
struct NativeResolvedRZThermalTrialInputs
{
    WarpXParticleContainer& live_photon_owner;
    PhotonPacketTransaction const& private_endpoint;
    NativePacketThermalEventReplay const& prepared_replay;
    amrex::MultiFab const& old_diffusion;
    amrex::iMultiFab const& diffusion_mask;
    amrex::MultiFab const& d_chi_t;
    amrex::MultiFab const& d_chi_a;
    amrex::Geometry const& geometry;
    amrex::Real dt;
    NativeResolvedThermalSourceOptions const& source_options;
    ResolvedRZThermalDiffusionFieldOptions const& field_options;
    NativeRZThermalHistoryStepContext const* history_step = nullptr;
    /** Borrowed actual interval path impulse [kg m/s], in the three native
     * RZ conjugate components. Combine it with endpoint photon debit before
     * subtracting work from the full prior material work. Null is the original
     * endpoint-only source. No lifetime extends beyond the synchronous trial. */
    amrex::MultiFab const* path_impulse = nullptr;
};

/** Private direct bath from genuine frozen caloric incidence and actual nodal
 * T. Trial and accepted-state screens share the same operation order. Failure
 * leaves the caller's cell output untouched. */
[[nodiscard]] bool TryBuildNativeResolvedRZReciprocalBath (
    amrex::MultiFab const& nodal_temperature, amrex::MultiFab const& corner_weights,
    amrex::Real radiation_constant, amrex::MultiFab& proper_bath);

[[nodiscard]] bool TryBuildNativeResolvedRZThermalTrial (
    NativeResolvedRZThermalTrialInputs const& inputs,
    amrex::MultiFab const& trial_nodal_temperature,
    amrex::MultiFab const& trial_cell_temperature,
    amrex::MultiFab const& secant_work_velocity,
    amrex::MultiFab const& prior_work_after_upstream_projection,
    NativeThermalCornerEmission const* frozen_emission,
    NativeResolvedThermalSourceResult& candidate,
    amrex::MultiFab& projected_energy_minus_work, std::string& reason);
} // namespace warpx::radiation

#endif
