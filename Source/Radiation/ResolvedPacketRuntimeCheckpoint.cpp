/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "RadiationTransport.H"

#include "CartesianPhotonBoundary.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "CartesianPhotonBoundary.H"
#include "PhotonLogicalDrawKey.H"
#include "ResolvedPacketRuntime.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_Geometry.H>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>

namespace {
constexpr char manifest_name[] = "/RadiationResolvedPackets_data.txt";
constexpr char manifest_version[] = "resolved_packet_fld_v9";
constexpr char legacy_fixed_version[] = "resolved_packet_fld_v7";
constexpr char adaptive_manifest_version[] = "resolved_packet_fld_v8";
#if defined(WARPX_DIM_RZ)
constexpr char adaptive_model[] = "adaptive_rz_conjugate_v1";
#else
constexpr char adaptive_model[] = "adaptive_cartesian_v1";
#endif

bool
ParseUnsigned (std::string const& token, std::uint64_t& value)
{
    if (token.empty() || token.find_first_not_of("0123456789") != std::string::npos) {
        return false;
    }
    auto const parsed = std::from_chars(token.data(), token.data() + token.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == token.data() + token.size();
}

bool
HasFieldHeader (std::string const& directory, std::string const& field)
{
    std::ifstream const header{directory + "/Level_0/" + field + "[level=0]_H"};
    return header.good();
}
} // namespace

void
RadiationTransport::WriteResolvedPacketCheckpoint (std::string const& directory) const
{
    if (!m_use_resolved_packet_fld) {
        return;
    }

    auto const& geometry = WarpX::GetInstance().Geom(0);
    auto const& state = m_resolved_packet;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!usesResolvedMaterialOpacity() ||
        state.material_opacity_initialized, "Cannot checkpoint uninitialized material opacity.");
    std::ofstream manifest{directory + manifest_name};
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(manifest.good(),
                                     "Could not create resolved packet-FLD checkpoint manifest.");
    manifest << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10);
    manifest << (usesAdaptivePacketRepresentation() ? adaptive_manifest_version
                                                    : manifest_version)
             << '\n'
             << std::quoted(state.mask_expression) << '\n'
             << std::quoted(state.opacity_expression) << ' '
             << state.opacityMode() << ' ' << state.opacity_time << '\n'
             << state.photon_energy << ' '
             << static_cast<int>(state.moving_diffusion) << ' '
             << static_cast<int>(state.packet_substeps) << ' '
             << state.packet_subsegment_limit << ' ' << state.max_iterations
             << ' ' << static_cast<int>(state.kinetic_empty_cells) << '\n';
    // These are the fixed admission limits of the resolved runtime. A
    // future change to either limit needs a new manifest version or explicit
    // migration, even if the user-facing input remains unchanged.
    manifest << amrex::Real(1) / 3 << ' ' << amrex::Real(0.1) << '\n';
    manifest << m_particle_conversion_seed << ' ' << state.interval << '\n';
    manifest << std::quoted(m_photon_species) << ' ' << m_momentum_species.size() << ' '
             << m_num_groups << '\n';
    for (auto const& species : m_momentum_species) {
        auto const& material =
            WarpX::GetInstance().GetPartContainer().GetParticleContainerFromName(species);
        manifest << std::quoted(species) << ' ' << material.getMass() << ' ' << material.getCharge()
                 << '\n';
    }
    manifest << AMREX_SPACEDIM << ' ' << geometry.Coord() << ' ' << WarpX::nox << '\n';
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        manifest << geometry.Domain().smallEnd(d) << ' ' << geometry.Domain().bigEnd(d) << ' '
                 << geometry.ProbLo(d) << ' ' << geometry.ProbHi(d) << ' '
                 << static_cast<int>(geometry.isPeriodic(d)) << ' '
                 << static_cast<int>(WarpX::particle_boundary_lo[d]) << ' '
                 << static_cast<int>(WarpX::particle_boundary_hi[d]) << ' '
                 << static_cast<int>(WarpX::field_boundary_lo[d]) << ' '
                 << static_cast<int>(WarpX::field_boundary_hi[d]) << ' '
                 << m_diffusion_boundary_lo[d] << ' ' << m_diffusion_boundary_hi[d] << ' '
                 << m_photon_boundary_lo[d] << ' ' << m_photon_boundary_hi[d] << '\n';
    }
    if (!usesAdaptivePacketRepresentation()) {
        manifest << std::quoted(state.parser_identity) << '\n';
    }
    if (usesResolvedMaterialOpacity()) {
        manifest << std::quoted(ResolvedMaterialOpacityIdentity()) << '\n';
    }
    if (usesAdaptivePacketRepresentation()) {
        manifest << adaptive_model << ' ' << state.adaptive_policy.demote_depth << ' '
                 << state.adaptive_policy.promote_depth << ' '
                 << std::quoted(state.parser_identity) << '\n';
    }
    manifest.flush();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(manifest.good(),
                                     "Could not write resolved packet-FLD checkpoint manifest.");
}

void
RadiationTransport::ReadResolvedPacketCheckpoint (std::string const& directory)
{
    if (!m_use_resolved_packet_fld) {
        return;
    }

    std::ifstream manifest{directory + manifest_name};
    std::string version, mask_expression, opacity_expression;
    bool const same_initial_model =
        static_cast<bool>(manifest >> version >> std::quoted(mask_expression) >>
                          std::quoted(opacity_expression)) &&
        (version == adaptive_manifest_version || version == manifest_version || version == legacy_fixed_version ||
         version == "resolved_packet_fld_v6" || version == "resolved_packet_fld_v5" ||
         version == "resolved_packet_fld_v4" || version == "resolved_packet_fld_v3") &&
        mask_expression == m_resolved_packet.mask_expression &&
        opacity_expression == m_resolved_packet.opacity_expression;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        same_initial_model, "Resolved packet-FLD restart changed its version, mask, or opacity "
                            "expression.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE((version == adaptive_manifest_version) ==
                                         usesAdaptivePacketRepresentation(),
                                     "Resolved packet-FLD restart changed its "
                                     "adaptive representation ownership.");

    std::string opacity_mode = "0";
    amrex::Real opacity_time = 0;
    bool const stored_opacity_controls = version == adaptive_manifest_version ||
                                         version == manifest_version || version == legacy_fixed_version ||
                                         version == "resolved_packet_fld_v6";
    bool const opacity_controls_read = !stored_opacity_controls ||
        static_cast<bool>(manifest >> opacity_mode >> opacity_time);
    auto const checkpoint_time = WarpX::GetInstance().gett_new(0);
    auto const time_roundoff = 64 * std::numeric_limits<amrex::Real>::epsilon() *
        std::max(std::abs(checkpoint_time), std::abs(opacity_time));
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        opacity_controls_read &&
            opacity_mode == std::to_string(m_resolved_packet.opacityMode()) &&
            (!usesResolvedMaterialOpacity() || version == manifest_version || version == legacy_fixed_version ||
             version == adaptive_manifest_version) &&
            std::isfinite(opacity_time) && std::isfinite(checkpoint_time) &&
            (opacity_mode == "0"
                 ? opacity_time == 0
                 : std::abs(opacity_time - checkpoint_time) <= time_roundoff),
        "Resolved packet-FLD restart changed its opacity mode or evaluation "
        "time.");
    m_resolved_packet.opacity_time = opacity_time;

    amrex::Real photon_energy = 0;
    int moving = -1, substeps = -1, subsegment_limit = -1, max_iterations = -1;
    bool const same_replay_controls =
        static_cast<bool>(manifest >> photon_energy >> moving >> substeps >> subsegment_limit >>
                          max_iterations) &&
        std::isfinite(photon_energy) && photon_energy == m_resolved_packet.photon_energy &&
        moving == static_cast<int>(m_resolved_packet.moving_diffusion) &&
        substeps == static_cast<int>(m_resolved_packet.packet_substeps) &&
        subsegment_limit == m_resolved_packet.packet_subsegment_limit &&
        max_iterations == m_resolved_packet.max_iterations;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        same_replay_controls,
        "Resolved packet-FLD restart changed photon energy or replay controls.");

    // Version3 has the same persisted fields and keys, and always used the
    // legacy fixed scalar partition. It can migrate only with the policy off.
    std::string front_policy = "0";
    bool const front_policy_read = version == "resolved_packet_fld_v3" ||
        static_cast<bool>(manifest >> front_policy);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(front_policy_read &&
        (front_policy == "0" || front_policy == "1") &&
        (front_policy == "1") == m_resolved_packet.kinetic_empty_cells,
        "Resolved packet-FLD restart changed the empty-front policy.");

    amrex::Real max_flux_fraction = -1, max_pressure_anisotropy = -1;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        static_cast<bool>(manifest >> max_flux_fraction >> max_pressure_anisotropy) &&
            std::isfinite(max_flux_fraction) && std::isfinite(max_pressure_anisotropy) &&
            max_flux_fraction == amrex::Real(1) / 3 && max_pressure_anisotropy == amrex::Real(0.1),
        "Resolved packet-FLD restart changed its fixed admission limits.");

    std::string seed_token, interval_token;
    std::uint64_t seed = 0, interval = 0;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        static_cast<bool>(manifest >> seed_token >> interval_token) &&
            ParseUnsigned(seed_token, seed) && seed > 0 &&
            ParseUnsigned(interval_token, interval) && seed == m_particle_conversion_seed,
        "Resolved packet-FLD restart has a missing, invalid, or mismatched event seed/counter.");

    std::string photon_species;
    std::size_t species_count = 0;
    int groups = 0;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        static_cast<bool>(manifest >> std::quoted(photon_species) >> species_count >> groups) &&
            photon_species == m_photon_species && species_count == m_momentum_species.size() &&
            groups == m_num_groups,
        "Resolved packet-FLD restart changed photon or material species ownership.");
    for (std::size_t index = 0; index < m_momentum_species.size(); ++index) {
        auto const& expected = m_momentum_species[index];
        std::string actual;
        amrex::ParticleReal mass = 0, charge = 0;
        auto const& material =
            WarpX::GetInstance().GetPartContainer().GetParticleContainerFromName(expected);
        bool const same_material_identity =
            static_cast<bool>(manifest >> std::quoted(actual) >> mass >> charge) &&
            actual == expected && std::isfinite(mass) && std::isfinite(charge) &&
            mass == material.getMass() && charge == material.getCharge() &&
            mass == m_resolved_packet.configured_material_identity[index][0] &&
            charge == m_resolved_packet.configured_material_identity[index][1];
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            same_material_identity,
            "Resolved packet-FLD restart changed material species identity, mass, or charge.");
    }

    auto const& geometry = WarpX::GetInstance().Geom(0);
    int dimensions = -1, coordinate = -1, shape = -1;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        static_cast<bool>(manifest >> dimensions >> coordinate >> shape) &&
            dimensions == AMREX_SPACEDIM && coordinate == geometry.Coord() && shape == WarpX::nox,
        "Resolved packet-FLD restart changed coordinate system or particle shape.");
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        int lo = 0, hi = 0, periodic = -1;
        int particle_lo = -1, particle_hi = -1, field_lo = -1, field_hi = -1;
        int diffusion_lo = -1, diffusion_hi = -1;
        amrex::Real physical_lo = 0, physical_hi = 0;
        bool const same_geometry_and_boundaries =
            static_cast<bool>(manifest >> lo >> hi >> physical_lo >> physical_hi >> periodic >>
                              particle_lo >> particle_hi >> field_lo >> field_hi >> diffusion_lo >>
                              diffusion_hi) &&
                std::isfinite(physical_lo) && std::isfinite(physical_hi) &&
                lo == geometry.Domain().smallEnd(d) && hi == geometry.Domain().bigEnd(d) &&
                physical_lo == geometry.ProbLo(d) && physical_hi == geometry.ProbHi(d) &&
                periodic == static_cast<int>(geometry.isPeriodic(d)) &&
                particle_lo == static_cast<int>(WarpX::particle_boundary_lo[d]) &&
                particle_hi == static_cast<int>(WarpX::particle_boundary_hi[d]) &&
                field_lo == static_cast<int>(WarpX::field_boundary_lo[d]) &&
                field_hi == static_cast<int>(WarpX::field_boundary_hi[d]) &&
                diffusion_lo == m_diffusion_boundary_lo[d] &&
                diffusion_hi == m_diffusion_boundary_hi[d];
        // Keep the assertion expression short so the actionable diagnostic is
        // not truncated by the backend's fixed-size exception buffer.
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(same_geometry_and_boundaries,
            "Resolved packet-FLD restart changed its mesh, physical domain, or boundaries.");
        int photon_lo = m_photon_boundary_lo[d], photon_hi = m_photon_boundary_hi[d];
        bool const stored_photon_faces = version == adaptive_manifest_version ||
                                         version == manifest_version || version == legacy_fixed_version ||
                                         version == "resolved_packet_fld_v6" ||
                                         version == "resolved_packet_fld_v5";
        bool const faces_read = !stored_photon_faces ||
            static_cast<bool>(manifest >> photon_lo >> photon_hi);
        auto const absorbing = static_cast<int>(warpx::radiation::PhotonFacePolicy::Absorbing);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(faces_read &&
            photon_lo == m_photon_boundary_lo[d] && photon_hi == m_photon_boundary_hi[d] &&
            (stored_photon_faces || (photon_lo != absorbing && photon_hi != absorbing)),
            "Resolved packet-FLD restart changed independent photon boundaries.");
    }
    if (version == manifest_version) {
        std::string bindings;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE((manifest >> std::quoted(bindings)) &&
            bindings == m_resolved_packet.parser_identity,
            "Fixed-mask packet-FLD restart changed mask or opacity parser constants.");
    } else if (!usesAdaptivePacketRepresentation() && !m_resolved_packet.parser_identity.empty()) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_assume_legacy_checkpoint_model,
            "Missing fixed-mask parser constants. Verify their original values before setting "
            "radiation_transport.assume_legacy_checkpoint_model=1 for a legacy restart.");
        ablastr::warn_manager::WMRecordWarning("Radiation transport",
            "Using caller-verified parser constants for a legacy fixed-mask checkpoint.",
            ablastr::warn_manager::WarnPriority::medium);
    }
    if (usesResolvedMaterialOpacity()) {
        std::string identity;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE((manifest >> std::quoted(identity)) &&
            identity == ResolvedMaterialOpacityIdentity(),
            "Resolved packet-FLD restart changed its native material sampling identity.");
    }
    if (usesAdaptivePacketRepresentation()) {
        std::string model, parser_identity;
        amrex::Real demote = 0, promote = 0;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            static_cast<bool>(manifest >> model >> demote >> promote >>
                              std::quoted(parser_identity)) &&
                model == adaptive_model &&
                demote == m_resolved_packet.adaptive_policy.demote_depth &&
                promote == m_resolved_packet.adaptive_policy.promote_depth &&
                parser_identity == m_resolved_packet.parser_identity,
            "Adaptive packet restart changed hysteresis, parser constants or "
            "model identity.");
    }
    std::string trailing;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !(manifest >> trailing) && manifest.eof(),
        "Resolved packet-FLD checkpoint manifest has unexpected trailing data.");

    auto const& photons = WarpX::GetInstance().GetPartContainer()
                              .GetParticleContainerFromName(m_photon_species);
    std::array<int, 2> draw_key_components{-1, -1};
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        warpx::radiation::FindPhotonDrawKeyComponents(photons.GetRealSoANames(),
                                                     draw_key_components) &&
            draw_key_components[0] >= 0 &&
            std::numeric_limits<amrex::ParticleReal>::digits >= 32,
        "Resolved packet-FLD restart is missing exact photon draw-key columns.");

    for (int d = 0; d < 3; ++d) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            HasFieldHeader(directory, warpx::radiation::ResolvedMomentumFieldName(d)),
            "Resolved packet-FLD restart is missing persistent radiation q state.");
    }
    for (auto const* field : {"radiation_diffusion_energy", warpx::radiation::ResolvedMaskFieldName,
                              warpx::radiation::ResolvedOpacityFieldName,
                              warpx::radiation::ResolvedAdmissionFieldName}) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            HasFieldHeader(directory, field),
            "Resolved packet-FLD restart is missing a persistent radiation field.");
    }
    if (usesResolvedMaterialOpacity()) {
        for (auto const* field : {warpx::radiation::ResolvedElectronDensityFieldName,
                                 warpx::radiation::ResolvedElectronTemperatureFieldName}) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(HasFieldHeader(directory, field),
                "Resolved material-opacity restart is missing its frozen native samples.");
        }
    }
    if (usesAdaptivePacketRepresentation()) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            HasFieldHeader(
                directory,
                warpx::radiation::ResolvedRepresentationCarryFieldName),
            "Adaptive packet restart is missing its conservative "
            "representation carry.");
        auto const& carry = *WarpX::GetInstance().m_fields.get(
            warpx::radiation::ResolvedRepresentationCarryFieldName, 0);
#if defined(WARPX_DIM_RZ)
        for (auto const* name : {"radiation_material_energy", "radiation_material_kinetic_energy",
                                 "radiation_material_momentum"})
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                HasFieldHeader(directory, name),
                "Adaptive RZ restart is missing its accepted native material diagnostics.");
        }
        auto const carry_ghosts = carry.nGrowVect();
#else
        auto const carry_ghosts = amrex::IntVect(0);
#endif
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(carry.nComp() == 4 && carry.is_finite(0, 4, carry_ghosts),
                                         "Adaptive packet restart has invalid "
                                         "conservative representation carry.");
    }
    ValidateResolvedPacketRestartFields();
    m_resolved_packet.material_opacity_initialized = usesResolvedMaterialOpacity();
    m_resolved_packet.interval = interval;
}
