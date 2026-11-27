/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_RESOLVED_PACKET_RUNTIME_H_
#define WARPX_RADIATION_RESOLVED_PACKET_RUNTIME_H_

#include "AdaptivePacketDiffusionTransition.H"
#include "ConservativePhotonWeightOriginalRowBasis.H"
#include "NativeRZHistoryAcceptanceBudget.H"

#include <AMReX_INT.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace warpx::radiation {
/** Preserve the legacy three-variable parser schema, including constants. */
struct ResolvedOpacityExecutor {
    amrex::ParserExecutor<3> spatial;
    amrex::ParserExecutor<4> temporal;
    amrex::ParserExecutor<6> material;
    bool time_dependent = false;
    bool material_dependent = false;

    AMREX_GPU_HOST_DEVICE amrex::Real operator() (
        amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real time,
        amrex::Real ne, amrex::Real te) const noexcept
    {
        return material_dependent ? material(x,y,z,time,ne,te) :
            time_dependent ? temporal(x,y,z,time) : spatial(x,y,z);
    }
};

/** GLOBAL stationary radiation axial-wall accounts [kg m/s], never native
 * material reflection/pending carry. Last and cumulative sides/net are stored
 * separately to preserve their actual accepted arithmetic on restart. */
struct ResolvedThermalAxialWallAccounts
{
    std::array<amrex::Real, 2> kinetic{}, diffusion{};
    amrex::Real kinetic_net = 0, diffusion_net = 0;
    std::array<amrex::Real, 2> cumulative_kinetic{}, cumulative_diffusion{};
    amrex::Real cumulative_kinetic_net = 0, cumulative_diffusion_net = 0;
    amrex::Real last_arithmetic_bound = 0, cumulative_arithmetic_bound = 0;
    amrex::Real last_operand = 0, cumulative_operand = 0;
};

/** Optional transient diagnostic only. Physical endface slabs retain source
 * owner/rank incidence (no nodal duplicates), not full-domain face arrays.
 * Scalars elsewhere are GLOBAL; these immutable FABs and provenance keys are LOCAL.
 * No physical identity/checkpoint state and no second solve or kick. */
struct ResolvedThermalAxialWallSurface
{
    amrex::Real dt = 0;
    std::shared_ptr<amrex::MultiFab const> energy_current, stress, leakage;
    std::vector<std::uint64_t> kinetic_mirror_keys;
    std::vector<std::uint64_t> birth_keys, leakage_birth_keys;
    /** Same-outer provisional keys with actual accepted events. History clocks
     * start at zero at the fresh outer entry; these counts are cumulative
     * within that outer history, NOT newly incurred events in this slice. */
    std::vector<std::uint64_t> collided_same_outer_birth_keys;
};

/** Configuration and accepted history of the bounded gray resolved runtime.
 * Conserved cell E/q and photon/material attributes belong to native registries.
 * This counter advances only after a complete accepted radiation interval.
 */
struct ResolvedPacketRuntime {
    std::string mask_expression;
    std::string opacity_expression;
    amrex::Parser mask_parser;
    amrex::Parser opacity_parser;
    /** Prescribed lab opacity sampled at the accepted interval endpoint.
     * Static spatial input retains opacity_time=0 and is never refreshed. */
    bool time_dependent_opacity = false;
    bool material_dependent_opacity = false;
    mutable bool material_opacity_initialized = false;
    mutable amrex::Real opacity_time = 0;
    // Native restart overwrites species mass/charge from its header. Retain
    // the requested identity so incompatible input is rejected, not ignored.
    std::vector<std::array<amrex::Real, 2>> configured_material_identity;
    amrex::Real photon_energy = amrex::Real(1.e-16);
    bool packet_substeps = true;
    bool moving_diffusion = true;
    /** Fresh-state LAB cell Q/E only; no frame/closure inference or restart overwrite.
     * RZ Q is [cPr,cLz/rbar,cPz]. Absence keeps the original exact-zero initialization. */
    std::array<amrex::Real, 3> initial_momentum_ratio{};
    bool has_initial_momentum_ratio = false;
    mutable bool initial_momentum_initialized = false;
    bool kinetic_empty_cells = false;
    // Opt-in through enable_lte_exchange with the resolved backend. Proper
    // chi_a and chi_s remain distinct; one total-opacity D reconstruction and
    // one actual native caloric/recoil stage own the thermal interval.
    bool thermal_exchange = false;
    /** Default-OFF periodic Cartesian native force support through cubic. */
    bool thermal_native_nodal_force = false;
    int thermal_emission_rays = 8;
    bool thermal_reciprocal_nodal = false;
    int thermal_reciprocal_rays_per_corner = 8;
    int thermal_linear_iterations = 40;
    /** Opt-in uniform left-state material/opacity history inside one accepted
     * outer RZ radiation interval. Zero preserves the ordinary single-interval
     * endpoint-frozen runtime. Residual Poisson clocks never cross an accepted
     * outer boundary. */
    int thermal_history_subintervals = 0;
    /** Default-OFF CPU builtin periodic RZ policy: conservative entry transfer,
     * mask frozen through the entire history, actual-census endpoint transfer.
     * Distinct accepted-outer policy; never in-progress clock restart state. */
    bool thermal_adaptive_history = false;
    /** Default-off, one native weight proposal at a fully accepted fresh outer
     * history endpoint. Not a population cap or in-progress checkpoint policy. */
    bool thermal_history_photon_weights = false;
    /** Explicit direction policy; Legacy preserves the established enabled
     * policy identity and arithmetic. Original-row RRQR retains every native
     * guard and the fresh-outer one-attempt certificate, not a population cap. */
    PhotonWeightDirectionMethod thermal_history_photon_weights_direction_method =
        PhotonWeightDirectionMethod::Legacy;
    /** Separate CPU-first original-equation certificate with derived fixed-pose
     * formation bounds. Requires RRQR direction method; existing policies stay off. */
    bool thermal_history_photon_weights_original_certificate = false;
    // Accepted diagnostic only: preflight pairs, choices, removed records.
    // Native checkpoint state remains photons/keys/allocator plus model identity.
    mutable std::array<amrex::Long, 3> last_history_photon_weights{};
    // Reporting-only immutable accepted inputs; transient, not checkpoint state.
    mutable NativeRZHistoryAcceptanceWitness last_history_weight_witness;
    /** Explicit default-OFF CPU built-in fixed-mask reflecting-z thermal
     * boundary policy. It never enables normal photon-weight control. */
    bool thermal_reflecting_z = false;
    /** Separate default-OFF CPU one-fixed-charge table extension of reflecting
     * z. Requires positive fixed-mask material history; no N0 or GPU activation. */
    bool thermal_reflecting_z_table = false;
    /** Default-OFF CPU periodic fixed-mask gray history with one registered
     * fixed-charge TABLE material. Actual S/P/R channels must be arithmetic-
     * compatible; this is not independent Planck/Rosseland transport. */
    bool thermal_material_opacity_table = false;
    /** Separate default-OFF owner-issued exact binary64 axis-node correlation.
     * Interpolation and physical response arithmetic are never replaced by
     * corner lookup. Nonidentical queries retain strict-interior admission. */
    bool thermal_material_opacity_exact_axis_nodes = false;
    /** Separate default-OFF schema-1.1 linear policy. Every scattering node
     * must be canonical +0 with its exact mask; no opacity floor or R-P
     * reconstruction. The existing whole-step optical guards remain. */
    bool thermal_material_opacity_zero_scattering = false;
    /** Independent R/P/true-S CPU all-D TABLE history, frozen accepted-left
     * coefficients per slice. Opt-in and half-cell bounded. */
    bool thermal_material_opacity_independent_gray = false;
    /** Fixed-mask packet/FLD extension, retaining whole-outer optical16. */
    bool thermal_material_opacity_independent_mixed = false;
    bool thermal_material_opacity_independent_vacuum = false;
    /** Separate default-OFF, CPU pure-absorption history policy. The History
     * engine authenticates actual leaf durations and checks every frame;
     * the original whole-outer optical policy is unchanged when disabled. */
    bool thermal_history_pure_absorption_step_depth = false;
    /** Separate default-OFF CPU fixed all-D history admission. Only an
     * engine-issued zero-stored-photon witness can exempt D absorption16;
     * scattering, frames, finite native work and all physical gates remain. */
    bool thermal_history_all_d_implicit_absorption = false;
    /** Independent default-OFF CPU all-D/zero-stored-photon implicit duration
     * request. History must issue and revalidate the actual duration authority;
     * this flag never permits long packet paths or relaxes implicit D guards. */
    bool thermal_history_all_d_implicit_timestep = false;
    mutable ResolvedThermalAxialWallAccounts thermal_axial_wall;
    bool thermal_axial_wall_witness = false;
    mutable std::vector<ResolvedThermalAxialWallSurface> thermal_axial_wall_surfaces;
    std::string thermal_identity;
    mutable amrex::Real thermal_opacity_time = 0;
    /** Conservative geometry-aware optical-depth representation policy. RZ
     * stores local [E,cPr,cLz/rbar,cPz], with E/Pz/rbar-Lz global budgets only. The
     * persistent mask is then accepted history, not the initial parser value.
     */
    bool adaptive_representation = false;
    AdaptiveDiffusionMaskPolicy adaptive_policy;
    std::string parser_identity;
    int packet_subsegment_limit = 1024;
    int max_iterations = 128;
    int verbosity = 0;
    mutable std::uint64_t interval = 0;

    [[nodiscard]] ResolvedOpacityExecutor opacityExecutor () const
    {
        ResolvedOpacityExecutor executor;
        executor.time_dependent = time_dependent_opacity;
        executor.material_dependent = material_dependent_opacity;
        if (material_dependent_opacity) {
            executor.material = opacity_parser.compile<6>();
        } else if (time_dependent_opacity) {
            executor.temporal = opacity_parser.compile<4>();
        } else {
            executor.spatial = opacity_parser.compile<3>();
        }
        return executor;
    }

    [[nodiscard]] int opacityMode () const
    {
        return material_dependent_opacity ? 2 : time_dependent_opacity ? 1 : 0;
    }
};

inline std::string
ResolvedMomentumFieldName (int direction)
{
    return std::string("radiation_resolved_q") + "xyz"[direction];
}
inline constexpr char ResolvedMaskFieldName[] = "radiation_resolved_diffusion_mask";
inline constexpr char ResolvedOpacityFieldName[] = "radiation_resolved_opacity";
inline constexpr char ResolvedAdmissionFieldName[] = "radiation_resolved_admitted_energy";
inline constexpr char ResolvedElectronDensityFieldName[] = "radiation_resolved_opacity_ne";
inline constexpr char ResolvedElectronTemperatureFieldName[] = "radiation_resolved_opacity_Te";
inline constexpr char ResolvedRepresentationCarryFieldName[] =
    "radiation_resolved_representation_carry";
inline constexpr char ResolvedAbsorptionFieldName[] = "radiation_resolved_proper_absorption";
} // namespace warpx::radiation
#endif
