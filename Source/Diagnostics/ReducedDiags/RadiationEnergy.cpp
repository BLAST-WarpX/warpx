/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "RadiationEnergy.H"

#include "Fields.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/SpeciesPhysicalProperties.H"
#include "Particles/WarpXParticleContainer.H"
#include "Radiation/RadiationTransport.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <fstream>
#include <sstream>
#include <string>

using namespace amrex::literals;
using warpx::fields::FieldType;

RadiationEnergy::RadiationEnergy (std::string const& rd_name)
    : ReducedDiags{rd_name}
{
    amrex::ParmParse const pp_diag(rd_name);
    amrex::ParmParse const pp_radiation("radiation_transport");
    bool radiation_enabled = false;
    pp_radiation.query("enabled", radiation_enabled);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        radiation_enabled,
        "RadiationEnergy requires radiation_transport.enabled=1.");
    pp_radiation.get("photon_species", m_photon_species);
    auto& warpx = WarpX::GetInstance();
    m_num_groups = warpx.GetRadiationTransport().numEnergyGroups();
    m_has_boundary_injection = warpx.GetRadiationTransport().hasDiffusionBath();
    m_has_implicit_diffusion = warpx.GetRadiationTransport().usesImplicitDiffusion();
    m_has_particle_carry = warpx.GetRadiationTransport().usesParticleMomentumCarry();
    pp_diag.query("include_solver_details", m_include_solver_details);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_include_solver_details ||
                                         m_has_implicit_diffusion,
                                     "RadiationEnergy.include_solver_details "
                                     "requires an implicit radiation solver.");
    if (m_include_solver_details) {
        m_cumulative_group_transfers.resize(3 * m_num_groups, 0);
    }
    if (amrex::ParallelDescriptor::IOProcessor() && !m_write_header) {
        std::ifstream previous{m_path + m_rd_name + "." + m_extension};
        std::string header;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(static_cast<bool>(std::getline(previous, header)),
            "RadiationEnergy could not read the existing diagnostic header on restart.");
        bool const previous_has_injection =
            header.find("cumulative_boundary_energy_injection(J)") != std::string::npos;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(previous_has_injection == m_has_boundary_injection,
            "RadiationEnergy bath column layout changed on restart. Use a new diagnostic "
            "output path when adding baths; never append rows with a different schema.");
        bool const previous_has_solver_columns =
            header.find("diffusion_nonlinear_iterations()") != std::string::npos;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(previous_has_solver_columns == m_has_implicit_diffusion,
            "RadiationEnergy solver column layout changed on restart. Use a new diagnostic "
            "output path when switching explicit/implicit spatial solvers.");
        bool const previous_has_details =
            header.find("diffusion_group_0_out(J)") != std::string::npos;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(previous_has_details ==
                                             m_include_solver_details,
                                         "RadiationEnergy detailed solver "
                                         "column layout changed on restart.");
        bool const previous_has_carry =
            header.find("pending_material_carry_energy(J)") != std::string::npos;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(previous_has_carry == m_has_particle_carry,
            "RadiationEnergy particle-carry column layout changed on restart.");
    }
    std::string diagnostic_photon_species = m_photon_species;
    if (pp_diag.query("photon_species", diagnostic_photon_species)) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            diagnostic_photon_species == m_photon_species,
            rd_name + ".photon_species must match "
                "radiation_transport.photon_species so boundary-energy accounting "
                "remains conservative.");
    }

    auto const& photons = warpx.GetPartContainer()
        .GetParticleContainerFromName(m_photon_species);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        photons.AmIA<PhysicalSpecies::photon>(),
        rd_name + ".photon_species must name a photon species.");

    // Total radiation, streaming photons, thick-field radiation, signed
    // current/cumulative material exchange and current/cumulative escape loss.
    int const num_group_columns = m_num_groups > 1 ? m_num_groups : 0;
    m_data.resize(15 + num_group_columns + (m_has_boundary_injection ? 2 : 0) +
                      (m_has_implicit_diffusion ? 3 : 0) +
                      (m_include_solver_details ? 6 * m_num_groups + 8 : 0) +
                      (m_has_particle_carry ? 2 : 0),
                  0.0_rt);

    if (amrex::ParallelDescriptor::IOProcessor() && m_write_header) {
        std::ofstream output{
            m_path + m_rd_name + "." + m_extension, std::ofstream::out};
        output << "#[0]step()" << m_sep
               << "[1]time(s)" << m_sep
               << "[2]total_radiation(J)" << m_sep
               << "[3]streaming_photons(J)" << m_sep
               << "[4]diffusion_radiation(J)" << m_sep
               << "[5]material_exchange(J)" << m_sep
               << "[6]cumulative_material_exchange(J)" << m_sep
               << "[7]boundary_energy_loss(J)" << m_sep
               << "[8]cumulative_boundary_energy_loss(J)";
        if (m_num_groups > 1) {
            for (int group = 0; group < m_num_groups; ++group) {
                output << m_sep << "[" << 9 + group
                       << "]diffusion_radiation_group_" << group << "(J)";
            }
        }
        int const material_column = 9
            + (m_num_groups > 1 ? m_num_groups : 0);
        output << m_sep << "[" << material_column
               << "]material_internal_exchange(J)"
               << m_sep << "[" << material_column + 1
               << "]material_kinetic_exchange(J)"
               << m_sep << "[" << material_column + 2
               << "]streaming_boundary_energy_loss(J)"
               << m_sep << "[" << material_column + 3
               << "]cumulative_streaming_boundary_energy_loss(J)"
               << m_sep << "[" << material_column + 4
               << "]diffusion_boundary_energy_loss(J)"
               << m_sep << "[" << material_column + 5
               << "]cumulative_diffusion_boundary_energy_loss(J)"
               << m_sep << "[" << material_column + 6
               << "]numerical_energy_residual(J)"
               << m_sep << "[" << material_column + 7
               << "]cumulative_numerical_energy_residual(J)";
        if (m_has_boundary_injection) {
            output << m_sep << "[" << material_column + 8 << "]boundary_energy_injection(J)"
                   << m_sep << "[" << material_column + 9
                   << "]cumulative_boundary_energy_injection(J)";
        }
        if (m_has_implicit_diffusion) {
            int const first = material_column + 8 + (m_has_boundary_injection ? 2 : 0);
            output << m_sep << "[" << first << "]diffusion_nonlinear_iterations()"
                   << m_sep << "[" << first + 1 << "]diffusion_linear_iterations()"
                   << m_sep << "[" << first + 2 << "]diffusion_relative_residual()";
        }
        if (m_include_solver_details) {
            int column =
                material_column + 11 + (m_has_boundary_injection ? 2 : 0);
            for (int g = 0; g < m_num_groups; ++g) {
                for (auto const* quantity :
                     {"out", "in", "material", "cumulative_out",
                      "cumulative_in", "cumulative_material"}) {
                    output << m_sep << "[" << column++ << "]diffusion_group_"
                           << g << "_" << quantity << "(J)";
                }
            }
            for (auto const* quantity :
                 {"coupled_iterations()", "accepted_radiation_substeps()",
                  "rejected_radiation_attempts()",
                  "source_consistency_corrections()",
                  "material_relative_residual()",
                  "raw_stage_energy_relative_residual()",
                  "minimum_group_cell_energy(J)",
                  "minimum_material_temperature(K)"}) {
                output << m_sep << "[" << column++ << "]" << quantity;
            }
        }
        if (m_has_particle_carry) {
            output << m_sep << '[' << m_data.size() << "]pending_material_carry_energy(J)"
                   << m_sep << '[' << m_data.size() + 1 << "]material_carry_energy_change(J)";
        }
        output << "\n";
    }
}

void RadiationEnergy::ComputeDiags (int const step)
{
    auto& warpx = WarpX::GetInstance();
    amrex::Real material_internal_exchange = 0.0_rt;
    amrex::Real material_kinetic_exchange = 0.0_rt;
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        if (warpx.m_fields.has(FieldType::radiation_material_energy, lev)) {
            material_internal_exchange += warpx.m_fields.get(
                FieldType::radiation_material_energy, lev)->sum(
                    0, /*local=*/false);
        }
        if (warpx.m_fields.has(
                FieldType::radiation_material_kinetic_energy, lev))
        {
            material_kinetic_exchange += warpx.m_fields.get(
                FieldType::radiation_material_kinetic_energy, lev)->sum(
                    0, /*local=*/false);
        }
    }
    amrex::Real const material_exchange =
        material_internal_exchange + material_kinetic_exchange
        + warpx.GetRadiationTransport().lastMaterialCarryEnergyChange();
    auto const& radiation_transport = warpx.GetRadiationTransport();
    amrex::Real const boundary_energy_loss =
        radiation_transport.lastBoundaryEnergyLoss();
    amrex::Real const streaming_boundary_energy_loss =
        radiation_transport.lastStreamingBoundaryEnergyLoss();
    amrex::Real const diffusion_boundary_energy_loss =
        radiation_transport.lastDiffusionBoundaryEnergyLoss();
    m_cumulative_numerical_energy_residual =
        radiation_transport.cumulativeNumericalEnergyResidual();
    if (step >= 0 && step != m_last_accumulated_step) {
        if (m_include_solver_details) {
            auto const& solve =
                radiation_transport.lastImplicitDiffusionResult();
            for (int g = 0; g < m_num_groups; ++g) {
                if (!solve.group_escaped_energy.empty()) {
                    m_cumulative_group_transfers[3 * g] +=
                        solve.group_escaped_energy[g];
                    m_cumulative_group_transfers[3 * g + 1] +=
                        solve.group_injected_energy[g];
                    m_cumulative_group_transfers[3 * g + 2] +=
                        solve.group_material_energy[g];
                }
            }
        }
        m_cumulative_material_exchange += material_exchange;
        m_cumulative_boundary_energy_loss =
            radiation_transport.cumulativeBoundaryEnergyLoss();
        m_cumulative_streaming_boundary_energy_loss +=
            streaming_boundary_energy_loss;
        m_cumulative_diffusion_boundary_energy_loss +=
            diffusion_boundary_energy_loss;
        m_last_accumulated_step = step;
    }

    if (!m_intervals.contains(step + 1)) { return; }

    auto const& photons = warpx.GetPartContainer()
        .GetParticleContainerFromName(m_photon_species);
    amrex::Real const streaming_energy =
        photons.sumParticleEnergy(/*local=*/false);

    amrex::Real diffusion_energy = 0.0_rt;
    std::vector<amrex::Real> diffusion_group_energy(m_num_groups, 0.0_rt);
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        if (warpx.m_fields.has(FieldType::radiation_diffusion_energy, lev)) {
            auto const* diffusion = warpx.m_fields.get(
                FieldType::radiation_diffusion_energy, lev);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                diffusion->nComp() == m_num_groups,
                "RadiationEnergy group count does not match the diffusion field.");
            for (int group = 0; group < m_num_groups; ++group) {
                diffusion_group_energy[group] += diffusion->sum(
                    group, /*local=*/false);
            }
        }
    }
    for (auto const energy : diffusion_group_energy) {
        diffusion_energy += energy;
    }

    m_data[0] = streaming_energy + diffusion_energy;
    m_data[1] = streaming_energy;
    m_data[2] = diffusion_energy;
    if (m_has_particle_carry) {
        auto& particles = warpx.GetPartContainer();
        auto const streaming = radiation_transport.pendingMaterialImpulse(particles, true);
        auto const diffusion = radiation_transport.pendingMaterialImpulse(particles, false);
        m_data[m_data.size() - 2] = streaming[3] + diffusion[3];
        m_data[m_data.size() - 1] = radiation_transport.lastMaterialCarryEnergyChange();
    }
    m_data[3] = material_exchange;
    m_data[4] = m_cumulative_material_exchange;
    m_data[5] = boundary_energy_loss;
    m_data[6] = m_cumulative_boundary_energy_loss;
    if (m_num_groups > 1) {
        for (int group = 0; group < m_num_groups; ++group) {
            m_data[7 + group] = diffusion_group_energy[group];
        }
    }
    int const material_column = 7
        + (m_num_groups > 1 ? m_num_groups : 0);
    m_data[material_column] = material_internal_exchange;
    m_data[material_column + 1] = material_kinetic_exchange;
    m_data[material_column + 2] = streaming_boundary_energy_loss;
    m_data[material_column + 3] =
        m_cumulative_streaming_boundary_energy_loss;
    m_data[material_column + 4] = diffusion_boundary_energy_loss;
    m_data[material_column + 5] =
        m_cumulative_diffusion_boundary_energy_loss;
    m_data[material_column + 6] =
        radiation_transport.lastNumericalEnergyResidual();
    m_data[material_column + 7] =
        m_cumulative_numerical_energy_residual;
    if (m_has_boundary_injection) {
        m_data[material_column + 8] = radiation_transport.lastBoundaryEnergyInjection();
        m_data[material_column + 9] = radiation_transport.cumulativeBoundaryEnergyInjection();
    }
    if (m_has_implicit_diffusion) {
        int const first = material_column + 8 + (m_has_boundary_injection ? 2 : 0);
        auto const& solve = radiation_transport.lastImplicitDiffusionResult();
        m_data[first] = static_cast<amrex::Real>(solve.nonlinear_iterations);
        m_data[first + 1] = static_cast<amrex::Real>(solve.linear_iterations);
        m_data[first + 2] = solve.maximum_relative_residual;
    }
    if (m_include_solver_details) {
        int column = material_column + 11 + (m_has_boundary_injection ? 2 : 0);
        auto const& solve = radiation_transport.lastImplicitDiffusionResult();
        for (int g = 0; g < m_num_groups; ++g) {
            m_data[column++] = solve.group_escaped_energy.empty()
                                   ? 0
                                   : solve.group_escaped_energy[g];
            m_data[column++] = solve.group_injected_energy.empty()
                                   ? 0
                                   : solve.group_injected_energy[g];
            m_data[column++] = solve.group_material_energy.empty()
                                   ? 0
                                   : solve.group_material_energy[g];
            for (int q = 0; q < 3; ++q) {
                m_data[column++] = m_cumulative_group_transfers[3 * g + q];
            }
        }
        auto const& coupled = radiation_transport.lastCoupledSolveDiagnostics();
        m_data[column++] = coupled.iterations;
        m_data[column++] = coupled.accepted_substeps;
        m_data[column++] = coupled.rejected_attempts;
        m_data[column++] = coupled.source_consistency_corrections;
        m_data[column++] = coupled.material_residual;
        m_data[column++] = coupled.raw_energy_residual;
        auto const& energy =
            *warpx.m_fields.get(FieldType::radiation_diffusion_energy, 0);
        amrex::Real minimum = energy.min(0);
        for (int g = 1; g < m_num_groups; ++g) {
            minimum = amrex::min(minimum, energy.min(g));
        }
        m_data[column++] = minimum;
        m_data[column] =
            radiation_transport.commitsHybridMaterialState()
                ? warpx.m_fields
                      .get(FieldType::hybrid_electron_temperature_fp, 0)
                      ->min(0)
                : -1;
    }
}

void RadiationEnergy::WriteCheckpointData (std::string const& dir)
{
    std::ofstream checkpoint{
        dir + "/" + m_rd_name + "_RadiationEnergy_data.txt",
        std::ofstream::out};
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        checkpoint.good(),
        "RadiationEnergy could not write its checkpoint state.");
    checkpoint.precision(17);
    checkpoint << m_cumulative_material_exchange << "\n"
               << m_cumulative_boundary_energy_loss << "\n"
               << m_cumulative_streaming_boundary_energy_loss << "\n"
               << m_cumulative_diffusion_boundary_energy_loss << "\n"
               << m_last_accumulated_step << "\n"
               << m_cumulative_numerical_energy_residual << "\n";
    if (m_include_solver_details) {
        checkpoint << "solver_details_v1 " << m_num_groups << "\n";
        for (auto value : m_cumulative_group_transfers) {
            checkpoint << value << "\n";
        }
    }
}

void RadiationEnergy::ReadCheckpointData (std::string const& dir)
{
    std::ifstream checkpoint{
        dir + "/" + m_rd_name + "_RadiationEnergy_data.txt",
        std::ifstream::in};
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        checkpoint.good(),
        "RadiationEnergy could not read its checkpoint state.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        static_cast<bool>(checkpoint >> m_cumulative_material_exchange
            >> m_cumulative_boundary_energy_loss
            >> m_cumulative_streaming_boundary_energy_loss
            >> m_cumulative_diffusion_boundary_energy_loss
            >> m_last_accumulated_step),
        "RadiationEnergy checkpoint state is truncated or invalid.");
    std::string numerical_energy_residual_token;
    if (!(checkpoint >> numerical_energy_residual_token)) {
        if (checkpoint.eof()) {
            checkpoint.clear();
            m_cumulative_numerical_energy_residual = 0.0_rt;
            ablastr::warn_manager::WMRecordWarning(
                "Radiation energy",
                "The restart checkpoint predates signed numerical-energy "
                "residual diagnostics. The cumulative numerical-energy residual "
                "will restart from zero.",
                ablastr::warn_manager::WarnPriority::low);
        } else {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                false,
                "RadiationEnergy checkpoint numerical-energy residual is "
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
            "RadiationEnergy checkpoint numerical-energy residual is "
            "malformed.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            amrex::Math::isfinite(cumulative_numerical_energy_residual),
            "RadiationEnergy checkpoint numerical-energy residual is "
            "non-finite.");
        m_cumulative_numerical_energy_residual =
            cumulative_numerical_energy_residual;
    }
    std::string tag;
    bool const has_details = static_cast<bool>(checkpoint >> tag);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        has_details == m_include_solver_details,
        "RadiationEnergy solver-details checkpoint configuration changed on "
        "restart.");
    if (has_details) {
        int groups = 0;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            tag == "solver_details_v1" &&
                static_cast<bool>(checkpoint >> groups) &&
                groups == m_num_groups,
            "RadiationEnergy solver-details checkpoint version/group count is "
            "invalid.");
        int component = 0;
        for (auto& value : m_cumulative_group_transfers) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                static_cast<bool>(checkpoint >> value) &&
                    amrex::Math::isfinite(value) &&
                    (component % 3 == 2 || value >= 0),
                "RadiationEnergy group ledger is truncated, non-finite or has "
                "negative boundary transfer.");
            ++component;
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !(checkpoint >> tag),
            "RadiationEnergy checkpoint state has unexpected trailing data.");
    }
}
