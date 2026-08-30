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
    m_data.resize(15 + num_group_columns, 0.0_rt);

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
        material_internal_exchange + material_kinetic_exchange;
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
        std::string trailing_token;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !(checkpoint >> trailing_token),
            "RadiationEnergy checkpoint state has unexpected trailing data.");
        m_cumulative_numerical_energy_residual =
            cumulative_numerical_energy_residual;
    }
}
