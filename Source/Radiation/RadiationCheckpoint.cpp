/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "RadiationTransport.H"

#include "CartesianPhotonBoundary.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/MaterialTableIdentity.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_Math.H>

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

using namespace amrex::literals;

namespace
{
// Canonical single-token identity: mesh/physical faces are source-model state,
// whereas box decomposition and MPI rank count are deliberately not included.
std::string MovingThermalGeometryIdentity ()
{
    auto const& geometry = WarpX::GetInstance().Geom(0);
    std::ostringstream identity;
    identity << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10);
#if defined(WARPX_DIM_RZ)
    identity << "rz:" << WarpX::n_rz_azimuthal_modes;
#else
    identity << "cartesian:" << AMREX_SPACEDIM;
#endif
    identity << ':' << geometry.Coord();
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        identity << ':' << geometry.Domain().smallEnd(d) << ':' << geometry.Domain().bigEnd(d)
            << ':' << geometry.ProbLo(d) << ':' << geometry.ProbHi(d)
            << ':' << geometry.isPeriodic(d)
            << ':' << static_cast<int>(WarpX::field_boundary_lo[d])
            << ':' << static_cast<int>(WarpX::field_boundary_hi[d])
            << ':' << static_cast<int>(WarpX::particle_boundary_lo[d])
            << ':' << static_cast<int>(WarpX::particle_boundary_hi[d]);
    }
    return identity.str();
}
}

void
RadiationTransport::WriteCheckpointData (std::string const& dir) const
{
    WriteParticleCarryWallCheckpoint(dir);
    if (!m_enabled) { return; }
    if (!m_use_moving_multigroup &&
        (m_num_groups > 1 || m_group_photon_energies_h.front() > 0.0_rt))
    {
        std::ofstream groups{dir + "/RadiationEnergyGroups_data.txt"};
        groups << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10)
               << "energy_groups_v1 " << m_num_groups << '\n';
        for (auto edge : m_energy_group_boundaries_h) { groups << edge << '\n'; }
        for (auto energy : m_group_photon_energies_h) { groups << energy << '\n'; }
        groups.close();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(groups.good(), "Could not write radiation energy groups.");
    }
    WriteResolvedPacketCheckpoint(dir);
    WriteMovingMultigroupCheckpoint(dir);
    WriteResolvedThermalCheckpoint(dir);
    if (m_use_moving_diffusion) {
        std::ofstream model{dir + "/RadiationMovingDiffusion_data.txt"};
        auto const eos = m_hybrid_model->electronThermodynamicsExecutor();
        bool const extended = m_use_moving_thermal_diffusion &&
            (m_moving_thermal_variable_opacity || eos.m_num_materials > 0 ||
             eos.isSingularitySpiner() || m_momentum_species.size() > 1);
        model << (m_use_moving_thermal_diffusion ?
                  (extended ? "moving_thermal_static_v5\n" : "moving_thermal_static_v4\n") :
                  "moving_asymptotic_v1\n");
        if (m_use_moving_thermal_diffusion) {
            model.precision(17);
            model << m_moving_thermal_absorption << ' ' << m_moving_thermal_transport << '\n';
            model << eos.m_gamma << ' ' << m_hybrid_model->electronDensityFloor() << ' '
                  << m_minimum_electron_density << ' '
                  << m_hybrid_model->transportsElectronInternalEnergyAtRawDensity() << '\n';
            model << (eos.isIdealGas() ? "ideal_gas" :
                      (eos.isSingularitySpiner() ? "singularity_spiner" :
                                                  "fixed_charge_latent_energy"))
                  << ' ' << eos.m_num_latent_transitions << '\n';
            for (int n = 0; n < eos.m_num_latent_transitions; ++n) {
                model << eos.m_latent_transition_temperature[n] << ' '
                      << eos.m_latent_energy_per_electron[n] << ' '
                      << eos.m_latent_transition_sharpness[n] << '\n';
            }
            model << MovingThermalGeometryIdentity() << '\n';
            if (extended)
            {
                model << std::quoted(m_moving_thermal_material_identity) << '\n'
                      << std::quoted(m_moving_thermal_coefficient_identity) << '\n';
            }
        }
        model.flush();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(model.good(),
            "Could not checkpoint moving diffusion model metadata.");
    }
    if (m_use_flux_relaxed_diffusion) {
        std::ofstream flux_state{dir + "/RadiationFluxRelaxed_data.txt"};
        flux_state << (m_flux_relaxed_limited_m1 ?
            warpx::radiation::detail::LimitedM1PacketStateMarker() :
            warpx::radiation::detail::FluxRelaxedStateMarker())
                   << '\n';
        flux_state.flush();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            flux_state.good(),
            "Could not checkpoint flux-relaxed state metadata.");
    }
    if (m_coherent_packet_transport) {
        std::ofstream coherent{dir + "/RadiationCoherentPackets_data.txt"};
#if defined(WARPX_DIM_RZ)
        auto const& material = WarpX::GetInstance().GetPartContainer()
            .GetParticleContainerFromName(m_momentum_species.front());
        coherent << std::setprecision(std::numeric_limits<amrex::ParticleReal>::max_digits10)
                 << "chronological_coherent_packets_rz_angular_v3 "
                 << m_coherent_segment_counter << ' ' << std::quoted(m_photon_species)
                 << ' ' << std::quoted(m_momentum_species.front()) << ' '
                 << material.getMass() << ' ' << material.getCharge()
                 << ' ' << WarpX::nox << '\n';
        auto const& geometry = WarpX::GetInstance().Geom(0);
        coherent << geometry.Coord() << ' ' << WarpX::n_rz_azimuthal_modes << '\n';
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            coherent << geometry.Domain().smallEnd(d) << ' ' << geometry.Domain().bigEnd(d) << ' '
                     << geometry.ProbLo(d) << ' ' << geometry.ProbHi(d) << ' '
                     << static_cast<int>(geometry.isPeriodic(d)) << ' '
                     << static_cast<int>(WarpX::particle_boundary_lo[d]) << ' '
                     << static_cast<int>(WarpX::particle_boundary_hi[d]) << ' '
                     << static_cast<int>(WarpX::field_boundary_lo[d]) << ' '
                     << static_cast<int>(WarpX::field_boundary_hi[d]) << '\n';
        }
#else
        coherent << "chronological_coherent_packets_v1 " << m_coherent_segment_counter << '\n';
#endif
        coherent.flush();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(coherent.good(),
            "Could not checkpoint coherent packet event counters.");
    }
    if (m_absorb_nonperiodic_photons) {
        std::ofstream boundary{dir + "/RadiationPhotonBoundary_data.txt"};
        boundary << "absorbing_nonperiodic_v1 " << m_photon_species << '\n';
        boundary.flush();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(boundary.good(),
            "Could not checkpoint independent photon boundaries.");
    } else if (m_has_explicit_photon_faces) {
        std::ofstream boundary{dir + "/RadiationPhotonBoundary_data.txt"};
        boundary << "photon_boundary_faces_v1 " << m_photon_species
                 << ' ' << AMREX_SPACEDIM;
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            boundary << ' ' << warpx::radiation::PhotonFacePolicyName(
                m_photon_boundary_lo[direction]);
        }
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            boundary << ' ' << warpx::radiation::PhotonFacePolicyName(
                m_photon_boundary_hi[direction]);
        }
        boundary << '\n';
        boundary.flush();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(boundary.good(),
            "Could not checkpoint independent photon boundaries.");
    }
    if (m_use_coupled_moment_transport) {
        std::ofstream model{dir + "/RadiationMomentModel_data.txt"};
#if defined(WARPX_DIM_RZ)
        model << (m_rz_angular_transport ? "gray_m1_angular_rz_inertia_ledger_v1 "
                                       : "gray_m1_meridional_rz_nearest_cell_ledger_v1 ")
              << WarpX::nox << '\n';
#else
        model << "gray_m1_low_beta_nodal_shape_ledger_v3 " << WarpX::nox << '\n';
#endif
        model.flush();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(model.good(),
            "Could not checkpoint moving radiation model.");
        std::ofstream ledger{dir + "/RadiationMomentTransportLedger_data.txt"};
        bool const written = m_moment_transport_ledger.Write(ledger);
        ledger.flush();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(written && ledger.good(),
            "Could not checkpoint moment wall/geometric accounting.");
    }
    if (m_particle_momentum_carry) {
        std::ofstream owner{dir + "/RadiationParticleCarry_data.txt"};
        owner << "particle_carry_v1 " << m_num_groups << ' ' << m_momentum_species.size() << '\n';
        for (auto const& name : m_momentum_species) { owner << name << '\n'; }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            owner.good(), "Could not checkpoint radiation carry schema.");
    }
    // These counters feed live diagnostics independently of output cadence.
    // The resolved conversion seed must also survive a restart when the input
    // requested warpx.random_seed=random.
    std::ofstream checkpoint{
        dir + "/RadiationTransport_data.txt", std::ofstream::out};
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        checkpoint.good(),
        "RadiationTransport could not write its checkpoint state.");
    checkpoint.precision(17);
    checkpoint << m_cumulative_boundary_energy_loss << "\n"
               << m_cumulative_numerical_energy_residual << "\n"
               << m_particle_conversion_seed << "\n";
    if (m_has_diffusion_bath || m_cumulative_boundary_energy_injection > 0.0_rt) {
        checkpoint << "bath_ledger_v1 " << m_cumulative_boundary_energy_injection << "\n";
    }
}

void
RadiationTransport::ReadCheckpointData (std::string const& dir)
{
    warpx::materials::MaterialTableIdentity::ValidateRestart(dir);
    std::ifstream group_manifest{dir + "/RadiationEnergyGroups_data.txt"};
    if (group_manifest.is_open()) {
        std::string version, trailing;
        int count = 0;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_enabled && !m_use_moving_multigroup && (group_manifest >> version >> count) &&
                version == "energy_groups_v1" && count == m_num_groups,
            "Radiation restart changed its spectral group count or model.");
        for (auto const& values : {&m_energy_group_boundaries_h, &m_group_photon_energies_h}) {
            for (auto expected : *values) {
                amrex::Real stored = 0;
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE((group_manifest >> stored) && stored == expected,
                    "Radiation restart changed its group edges or representative photon energies.");
            }
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!(group_manifest >> trailing) && group_manifest.eof(),
            "Radiation energy-group checkpoint has unexpected trailing data.");
    } else if (m_enabled && !m_use_moving_multigroup &&
               (m_num_groups > 1 || m_group_photon_energies_h.front() > 0.0_rt)) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_assume_legacy_checkpoint_model,
            "Missing spectral group identity. Verify the original group definition before "
            "setting radiation_transport.assume_legacy_checkpoint_model=1 for a legacy restart.");
        ablastr::warn_manager::WMRecordWarning("Radiation transport",
            "Using caller-verified spectral groups for a legacy radiation checkpoint.",
            ablastr::warn_manager::WarnPriority::medium);
    }
    // Check mode ownership even for a run that has disabled radiation. The
    // resolved E/q/packet census cannot be silently interpreted as another mode.
    std::ifstream resolved{dir + "/RadiationResolvedPackets_data.txt"};
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(resolved.good() == m_use_resolved_packet_fld,
        "Restart must preserve the resolved packet-FLD mode and its manifest.");
    std::ifstream moving_groups{dir + "/RadiationMovingMultigroup_data.txt"};
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        moving_groups.good() == m_use_moving_multigroup,
        "Restart must preserve the moving multigroup model and its manifest.");
    std::ifstream thermal{dir + "/RadiationResolvedThermal_data.txt"};
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        thermal.good() == usesResolvedThermalExchange(),
        "Restart must preserve resolved thermal source ownership and its manifest.");
    std::ifstream moving{dir + "/RadiationMovingDiffusion_data.txt"};
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(moving.good() == m_use_moving_diffusion,
        "Restart must preserve the moving diffusion model and its manifest.");
    if (m_use_moving_diffusion) {
        std::string marker, trailing;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE((moving >> marker) &&
            (m_use_moving_thermal_diffusion
                 ? marker == "moving_thermal_static_v5" || marker == "moving_thermal_static_v4" ||
                   marker == "moving_thermal_static_v3" ||
                   marker == "moving_thermal_static_v2"
                 : marker == "moving_asymptotic_v1"),
            "Invalid moving diffusion checkpoint model metadata.");
        if (m_use_moving_thermal_diffusion) {
            amrex::Real absorption = -1, transport = -1;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE((moving >> absorption >> transport) &&
                absorption == m_moving_thermal_absorption && transport == m_moving_thermal_transport,
                "Restart must preserve moving thermal constant opacity coefficients.");
            amrex::Real gamma = 0, floor = -1, threshold = -1;
            int raw_density = -1;
            auto const eos = m_hybrid_model->electronThermodynamicsExecutor();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                marker == "moving_thermal_static_v5" ||
                    (!m_moving_thermal_variable_opacity && eos.m_num_materials == 0 &&
                     !eos.isSingularitySpiner() && m_momentum_species.size() == 1),
                "Legacy moving thermal checkpoints require the original constant "
                "single-material model.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE((moving >> gamma >> floor >> threshold >> raw_density) &&
                gamma == eos.m_gamma &&
                floor == m_hybrid_model->electronDensityFloor() &&
                threshold == m_minimum_electron_density &&
                raw_density == static_cast<int>(
                    m_hybrid_model->transportsElectronInternalEnergyAtRawDensity()),
                "Moving thermal restart must preserve caloric and density conventions.");
            if (marker == "moving_thermal_static_v2") {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(eos.isIdealGas(),
                    "Legacy moving thermal checkpoints require ideal electrons.");
            } else {
                std::string model;
                int transitions = -1;
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE((moving >> model >> transitions) &&
                    model == (eos.isIdealGas() ? "ideal_gas" :
                              (eos.isSingularitySpiner() ? "singularity_spiner" :
                                                          "fixed_charge_latent_energy")) &&
                    transitions == eos.m_num_latent_transitions,
                    "Moving thermal restart must preserve the caloric model and transitions.");
                for (int n = 0; n < transitions; ++n) {
                    amrex::Real temperature = -1, energy = -1, sharpness = -1;
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE((moving >> temperature >> energy >> sharpness) &&
                        temperature == eos.m_latent_transition_temperature[n] &&
                        energy == eos.m_latent_energy_per_electron[n] &&
                        sharpness == eos.m_latent_transition_sharpness[n],
                        "Moving thermal restart must preserve latent-energy parameters.");
                }
            }
            if (marker == "moving_thermal_static_v4" || marker == "moving_thermal_static_v5") {
                std::string identity;
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE((moving >> identity) &&
                    identity == MovingThermalGeometryIdentity(),
                    "Moving thermal restart must preserve mesh, geometry and physical boundaries.");
                if (marker == "moving_thermal_static_v5")
                {
                    std::string material_identity, coefficient_identity;
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                        (moving >> std::quoted(material_identity) >>
                         std::quoted(coefficient_identity)) &&
                            material_identity == m_moving_thermal_material_identity &&
                            coefficient_identity == m_moving_thermal_coefficient_identity,
                        "Moving thermal restart must preserve native material and opacity identities.");
                }
            } else {
#if defined(WARPX_DIM_RZ)
                constexpr bool legacy_geometry = false;
#else
                bool const legacy_geometry = WarpX::GetInstance().Geom(0).isAllPeriodic();
#endif
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(legacy_geometry,
                    "Legacy moving thermal checkpoints require periodic Cartesian geometry.");
            }
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!(moving >> trailing),
            m_use_moving_thermal_diffusion
                ? "Trailing moving diffusion checkpoint model metadata."
                : "Invalid moving diffusion checkpoint model metadata.");
    }
    std::ifstream coherent{dir + "/RadiationCoherentPackets_data.txt"};
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(coherent.good() == m_coherent_packet_transport,
        "Restart must preserve the coherent-packet mode and event-counter manifest.");
    if (m_coherent_packet_transport) {
        std::string marker, counter, trailing;
#if defined(WARPX_DIM_RZ)
        int shape = -1;
        std::string photon_name, material_name;
        amrex::ParticleReal mass = 0, charge = 0;
        auto const& material = WarpX::GetInstance().GetPartContainer()
            .GetParticleContainerFromName(m_momentum_species.front());
        bool const parsed_identity = static_cast<bool>(coherent >> marker >> counter
            >> std::quoted(photon_name) >> std::quoted(material_name)
            >> mass >> charge >> shape);
        bool const valid_identity = parsed_identity &&
            (marker == "chronological_coherent_packets_rz_angular_v2" ||
             marker == "chronological_coherent_packets_rz_angular_v3") &&
            photon_name == m_photon_species && material_name == m_momentum_species.front() &&
            amrex::Math::isfinite(mass) && amrex::Math::isfinite(charge) &&
            mass == material.getMass() && charge == material.getCharge() &&
            mass == m_coherent_rz_configured_material_mass &&
            charge == m_coherent_rz_configured_material_charge &&
            shape == WarpX::nox &&
            counter.find_first_not_of("0123456789") == std::string::npos;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(valid_identity,
            "Invalid RZ coherent-packet checkpoint manifest or changed ion mass/charge input.");
        auto const& geometry = WarpX::GetInstance().Geom(0);
        if (marker == "chronological_coherent_packets_rz_angular_v3") {
            int coordinates = -1, modes = -1;
            bool const same_coordinates = static_cast<bool>(coherent >> coordinates >> modes) &&
                coordinates == geometry.Coord() && modes == WarpX::n_rz_azimuthal_modes;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(same_coordinates,
                "RZ coherent-packet restart changed coordinates or angular modes.");
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                int lo = 0, hi = 0, periodic = -1;
                int particle_lo = -1, particle_hi = -1, field_lo = -1, field_hi = -1;
                amrex::Real physical_lo = 0, physical_hi = 0;
                bool const same_geometry = static_cast<bool>(coherent >> lo >> hi
                    >> physical_lo >> physical_hi >> periodic >> particle_lo >> particle_hi
                    >> field_lo >> field_hi) &&
                    std::isfinite(physical_lo) && std::isfinite(physical_hi) &&
                    lo == geometry.Domain().smallEnd(d) && hi == geometry.Domain().bigEnd(d) &&
                    physical_lo == geometry.ProbLo(d) && physical_hi == geometry.ProbHi(d) &&
                    periodic == static_cast<int>(geometry.isPeriodic(d)) &&
                    particle_lo == static_cast<int>(WarpX::particle_boundary_lo[d]) &&
                    particle_hi == static_cast<int>(WarpX::particle_boundary_hi[d]) &&
                    field_lo == static_cast<int>(WarpX::field_boundary_lo[d]) &&
                    field_hi == static_cast<int>(WarpX::field_boundary_hi[d]);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(same_geometry,
                    "RZ coherent-packet restart changed mesh, physical domain or boundaries.");
            }
        } else {
            // v2 was qualified only with periodic z; it cannot attest to a
            // reflecting-end checkpoint identity introduced by v3.
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(geometry.isPeriodic(1),
                "Legacy RZ coherent-packet checkpoint requires periodic z.");
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!(coherent >> trailing),
            "Invalid trailing RZ coherent-packet checkpoint metadata.");
#else
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE((coherent >> marker >> counter) &&
            marker == "chronological_coherent_packets_v1" &&
            counter.find_first_not_of("0123456789") == std::string::npos &&
            !(coherent >> trailing), "Invalid coherent-packet checkpoint manifest.");
#endif
        auto const parsed = std::from_chars(counter.data(), counter.data()+counter.size(),
                                             m_coherent_segment_counter);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(parsed.ec == std::errc{} &&
            parsed.ptr == counter.data()+counter.size(), "Invalid coherent-packet event counter.");
    }
    std::ifstream flux_state{dir + "/RadiationFluxRelaxed_data.txt"};
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        flux_state.good() == m_use_flux_relaxed_diffusion,
        "Restart must preserve the flux-relaxed state marker; switching "
        "transport modes or silently discarding this marker is not allowed.");
    if (m_use_flux_relaxed_diffusion) {
        std::string marker, trailing;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            static_cast<bool>(std::getline(flux_state, marker)) &&
                marker == (m_flux_relaxed_limited_m1 ?
                    warpx::radiation::detail::LimitedM1PacketStateMarker() :
                    warpx::radiation::detail::FluxRelaxedStateMarker()) &&
                !(flux_state >> trailing),
            "Invalid or mismatched flux-relaxed checkpoint metadata.");
        for (int direction = 0; direction < 3; ++direction) {
            std::ifstream const q_header{
                dir + "/Level_0/" +
                warpx::radiation::detail::FluxRelaxedFieldName(direction) +
                "[level=0]_H"};
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                q_header.good(),
                "Flux-relaxed restart is missing persistent q state.");
        }
    }
    std::ifstream boundary{dir + "/RadiationPhotonBoundary_data.txt"};
    bool const expect_photon_boundary_manifest =
        m_absorb_nonperiodic_photons || m_has_explicit_photon_faces;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        boundary.good() == expect_photon_boundary_manifest,
        "Restart must preserve independent photon boundaries and their manifest.");
    if (!m_enabled) {
        ReadParticleCarryWallCheckpoint(dir);
        return;
    }
    if (m_absorb_nonperiodic_photons) {
        std::string version, species, trailing;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE((boundary >> version >> species)
            && version == "absorbing_nonperiodic_v1" && species == m_photon_species
            && !(boundary >> trailing), "Invalid independent photon boundary manifest.");
    } else if (m_has_explicit_photon_faces) {
        std::string version, species, trailing;
        int dims = 0;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (boundary >> version >> species >> dims)
                && version == "photon_boundary_faces_v1"
                && species == m_photon_species
                && dims == AMREX_SPACEDIM,
            "Invalid independent photon boundary manifest.");
        auto const require_face = [&](int const expected, char const* const side)
        {
            std::string token;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                static_cast<bool>(boundary >> token)
                    && warpx::radiation::PhotonFacePolicyFromName(token.c_str())
                        == expected,
                std::string("Restart changed independent photon boundary ")
                    + side + " faces.");
        };
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            require_face(m_photon_boundary_lo[direction], "lo");
        }
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            require_face(m_photon_boundary_hi[direction], "hi");
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !(boundary >> trailing), "Invalid independent photon boundary manifest.");
    }
    std::ifstream model{dir + "/RadiationMomentModel_data.txt"};
    std::ifstream moment_ledger{dir + "/RadiationMomentTransportLedger_data.txt"};
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_use_coupled_moment_transport || !moment_ledger.good(),
        "Cannot discard a moment transport ledger on scalar radiation restart.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(model.good() == m_use_coupled_moment_transport,
        "Restart must preserve the radiation moment model; scalar/moment conversion "
        "is not implicit.");
    if (m_use_coupled_moment_transport) {
        std::string version, trailing;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(static_cast<bool>(model >> version),
            "Invalid moving radiation model checkpoint schema.");
        int order = 1;
#if defined(WARPX_DIM_RZ)
        bool const has_ledger = version == (m_rz_angular_transport
            ? "gray_m1_angular_rz_inertia_ledger_v1" : "gray_m1_meridional_rz_nearest_cell_ledger_v1");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(has_ledger && static_cast<bool>(model >> order),
            m_rz_angular_transport
                ? "RZ moment restart requires the angular inertia model and transport ledger."
                : "RZ moment restart requires the meridional nearest-cell model and transport ledger.");
#else
        bool const has_ledger = version == "gray_m1_low_beta_nodal_shape_ledger_v3";
        if (version == "gray_m1_low_beta_nodal_shape_v2" || has_ledger) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(static_cast<bool>(model >> order),
                "Moving radiation checkpoint is missing its particle shape order.");
        } else {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(version == "gray_m1_low_beta_linear_shape_v1",
                "Unknown moving radiation model checkpoint schema.");
        }
#endif
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(order == WarpX::nox && !(model >> trailing),
            "Moving radiation restart must preserve its particle shape order.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(moment_ledger.good() == has_ledger,
            "Moving radiation checkpoint must preserve its declared transport ledger.");
        if (has_ledger) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_moment_transport_ledger.Read(moment_ledger),
                "Invalid moment wall/geometric checkpoint accounting.");
        } else {
            // Legacy native moment runtime is periodic Cartesian: these two
            // independent transport accounts are identically zero there.
            m_moment_transport_ledger = {};
        }
    }
    std::ifstream owner{dir + "/RadiationParticleCarry_data.txt"};
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(owner.good() == m_particle_momentum_carry,
        "Restart must preserve radiation momentum_carry ownership; cell/particle carry "
        "migration is not implicit.");
    if (m_particle_momentum_carry) {
        std::string version;
        int groups = 0;
        std::size_t species_count = 0;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            static_cast<bool>(owner >> version >> groups >> species_count) &&
            version == "particle_carry_v1" && groups == m_num_groups &&
            species_count == m_momentum_species.size(), "Invalid radiation particle carry schema.");
        for (auto const& expected : m_momentum_species) {
            std::string name;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(static_cast<bool>(owner >> name) && name == expected,
                "Restart changed radiation momentum species ownership.");
        }
        std::string trailing;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!(owner >> trailing),
            "Unexpected trailing radiation particle carry schema data.");
    }
    // Validate the parent ownership contract before its per-owner wall history.
    ReadParticleCarryWallCheckpoint(dir);
    if (m_use_moving_thermal_diffusion || m_use_moving_multigroup) {
        // This mode has no legacy checkpoint without complete accounting.
        // Validate before the common reader publishes any cumulative counters.
        std::ifstream accounting{dir + "/RadiationTransport_data.txt"};
        amrex::Real loss = -1, residual = 0;
        std::string seed, trailing;
        bool complete = static_cast<bool>(accounting >> loss >> residual >> seed) &&
            amrex::Math::isfinite(loss) && loss >= 0 && amrex::Math::isfinite(residual) &&
            !seed.empty() && seed.find_first_not_of("0123456789") == std::string::npos &&
            !(accounting >> trailing);
        std::uint64_t parsed_seed = 0;
        if (complete) {
            auto const parsed = std::from_chars(seed.data(), seed.data()+seed.size(), parsed_seed);
            complete = parsed.ec == std::errc{} && parsed.ptr == seed.data()+seed.size() &&
                parsed_seed > 0;
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(complete,
            "Moving thermal restart requires complete finite radiation accounting.");
    }
    std::ifstream checkpoint{
        dir + "/RadiationTransport_data.txt", std::ifstream::in};
    char const* const seed_checkpoint_error = m_use_resolved_packet_fld
        ? "Resolved packet-FLD restart requires complete seed checkpoint accounting."
        : "Coherent-packet restart requires complete resolved-seed checkpoint accounting.";
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !(m_coherent_packet_transport || m_use_resolved_packet_fld) || checkpoint.good(),
        seed_checkpoint_error);
    if (!checkpoint.good()) {
        m_cumulative_boundary_energy_loss = 0.0_rt;
        m_cumulative_boundary_energy_injection = 0.0_rt;
        m_cumulative_numerical_energy_residual = 0.0_rt;
        m_last_numerical_energy_residual = 0.0_rt;
        ablastr::warn_manager::WMRecordWarning(
            "Radiation transport",
            "The restart checkpoint predates RadiationTransport cumulative "
            "boundary accounting. The live cumulative escaped-energy counter "
            "will restart from zero.",
            ablastr::warn_manager::WarnPriority::low);
        return;
    }
    amrex::Real cumulative_boundary_energy_loss = 0.0_rt;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        static_cast<bool>(checkpoint >> cumulative_boundary_energy_loss)
            && amrex::Math::isfinite(cumulative_boundary_energy_loss)
            && cumulative_boundary_energy_loss >= 0.0_rt,
        "RadiationTransport checkpoint state is truncated or invalid.");
    m_cumulative_boundary_energy_loss = cumulative_boundary_energy_loss;

    std::string numerical_energy_residual_token;
    if (!(checkpoint >> numerical_energy_residual_token)) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !(m_coherent_packet_transport || m_use_resolved_packet_fld),
            seed_checkpoint_error);
        if (checkpoint.eof()) {
            checkpoint.clear();
            m_cumulative_numerical_energy_residual = 0.0_rt;
            ablastr::warn_manager::WMRecordWarning(
                "Radiation transport",
                "The restart checkpoint predates signed numerical-energy "
                "residual accounting. The cumulative numerical-energy residual "
                "will restart from zero.",
                ablastr::warn_manager::WarnPriority::low);
        } else {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                false,
                "RadiationTransport checkpoint numerical-energy residual is "
                "malformed.");
        }
    } else {
        amrex::Real cumulative_numerical_energy_residual = 0.0_rt;
        std::istringstream numerical_energy_residual_stream{
            numerical_energy_residual_token};
        std::string residual_trailing_token;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            static_cast<bool>(numerical_energy_residual_stream
                >> cumulative_numerical_energy_residual)
                && !(numerical_energy_residual_stream >> residual_trailing_token),
            "RadiationTransport checkpoint numerical-energy residual is "
            "malformed.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            amrex::Math::isfinite(cumulative_numerical_energy_residual),
            "RadiationTransport checkpoint numerical-energy residual is "
            "non-finite.");
        m_cumulative_numerical_energy_residual =
            cumulative_numerical_energy_residual;

        std::string particle_conversion_seed_token;
        if (!(checkpoint >> particle_conversion_seed_token)) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !(m_coherent_packet_transport || m_use_resolved_packet_fld),
                seed_checkpoint_error);
            if (checkpoint.eof()) {
                checkpoint.clear();
                if (m_enable_particle_conversion) {
                    ablastr::warn_manager::WMRecordWarning(
                        "Radiation transport",
                        "The restart checkpoint predates resolved particle-"
                        "conversion seed accounting. The seed resolved from the "
                        "current input will be used.",
                        ablastr::warn_manager::WarnPriority::low);
                }
            } else {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    false,
                    "RadiationTransport checkpoint particle-conversion seed is "
                    "malformed.");
            }
        } else {
            std::uint64_t particle_conversion_seed = 0;
            std::istringstream particle_conversion_seed_stream{
                particle_conversion_seed_token};
            std::string seed_trailing_token;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                particle_conversion_seed_token.front() != '-'
                    && static_cast<bool>(particle_conversion_seed_stream
                        >> particle_conversion_seed)
                    && particle_conversion_seed > 0
                    && !(particle_conversion_seed_stream
                        >> seed_trailing_token),
                "RadiationTransport checkpoint particle-conversion seed is "
                "malformed.");
            m_particle_conversion_seed = particle_conversion_seed;
        }
        std::string extension_token;
        if (checkpoint >> extension_token) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                extension_token == "bath_ledger_v1" &&
                    static_cast<bool>(checkpoint >> m_cumulative_boundary_energy_injection) &&
                    amrex::Math::isfinite(m_cumulative_boundary_energy_injection) &&
                    m_cumulative_boundary_energy_injection >= 0.0_rt,
                "RadiationTransport checkpoint bath injection ledger is "
                "malformed.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_has_diffusion_bath,
                "A restart with a bath injection ledger must retain a marshak_bath face. "
                "To turn off the drive, retain the bath boundary and set its temperature "
                "or all group energy densities to zero; this preserves injection accounting.");
        } else {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(checkpoint.eof(), "RadiationTransport checkpoint bath "
                                                               "injection ledger is unreadable.");
            checkpoint.clear();
            m_cumulative_boundary_energy_injection = 0.0_rt;
        }
        std::string trailing_token;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !(checkpoint >> trailing_token),
            "RadiationTransport checkpoint state has unexpected trailing data.");
    }
    m_last_numerical_energy_residual = 0.0_rt;
    ReadResolvedPacketCheckpoint(dir);
    ReadMovingMultigroupCheckpoint(dir);
    ReadResolvedThermalCheckpoint(dir);
}
