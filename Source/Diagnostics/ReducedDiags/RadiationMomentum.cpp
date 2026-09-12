/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "RadiationMomentum.H"

#include "Fields.H"
#include "Radiation/RadiationTransport.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <array>
#include <fstream>
#include <string>

using namespace amrex::literals;
using warpx::fields::FieldType;

RadiationMomentum::RadiationMomentum (std::string const& rd_name)
    : ReducedDiags{rd_name}
{
    amrex::ParmParse const pp_radiation("radiation_transport");
    bool radiation_enabled = false;
    pp_radiation.query("enabled", radiation_enabled);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        radiation_enabled,
        "RadiationMomentum requires radiation_transport.enabled=1.");

    std::string diffusion_solver = "explicit";
    pp_radiation.query("diffusion_solver", diffusion_solver);
    amrex::ParmParse(rd_name).query("include_moment_inventory", m_include_moment_inventory);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_include_moment_inventory ||
        diffusion_solver == "coupled_moment",
        "RadiationMomentum.include_moment_inventory requires coupled_moment transport.");
    m_data.resize(m_include_moment_inventory ? 27 : 24, 0.0_rt);

#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RZ)
    std::array<std::string, 3> const labels{"r", "theta", "z"};
#elif defined(WARPX_DIM_RSPHERE)
    std::array<std::string, 3> const labels{"r", "theta", "phi"};
#else
    std::array<std::string, 3> const labels{"x", "y", "z"};
#endif

    if (amrex::ParallelDescriptor::IOProcessor() && m_write_header) {
        std::ofstream output{
            m_path + m_rd_name + "." + m_extension, std::ofstream::out};
        output << "#[0]step()" << m_sep << "[1]time(s)";
        int column = 2;
        for (std::string const& label : labels) {
            output << m_sep << "[" << column++ << "]material_"
                   << label << "(kg*m/s)";
        }
        for (std::string const& label : labels) {
            output << m_sep << "[" << column++ << "]cumulative_material_"
                   << label << "(kg*m/s)";
        }
        for (std::string const& label : labels) {
            output << m_sep << "[" << column++ << "]diffusion_boundary_"
                   << label << "(kg*m/s)";
        }
        for (std::string const& label : labels) {
            output << m_sep << "[" << column++
                   << "]cumulative_diffusion_boundary_" << label
                   << "(kg*m/s)";
        }
        for (std::string const& label : labels) {
            output << m_sep << "[" << column++ << "]streaming_boundary_"
                   << label << "(kg*m/s)";
        }
        for (std::string const& label : labels) {
            output << m_sep << "[" << column++
                   << "]cumulative_streaming_boundary_" << label
                   << "(kg*m/s)";
        }
        for (std::string const& label : labels) {
            output << m_sep << "[" << column++
                   << "]pending_streaming_material_" << label
                   << "(kg*m/s)";
        }
        for (std::string const& label : labels) {
            output << m_sep << "[" << column++
                   << "]pending_diffusion_material_" << label
                   << "(kg*m/s)";
        }
        if (m_include_moment_inventory) {
            for (std::string const& label : labels) {
                output << m_sep << "[" << column++ << "]moment_radiation_"
                       << label << "(kg*m/s)";
            }
        }
        output << "\n";
    }
}

void RadiationMomentum::ComputeDiags (int const step)
{
    auto& warpx = WarpX::GetInstance();
    amrex::GpuArray<amrex::Real, 3> material_impulse{0.0_rt, 0.0_rt, 0.0_rt};
    amrex::GpuArray<amrex::Real, 3> boundary_impulse{0.0_rt, 0.0_rt, 0.0_rt};
    amrex::GpuArray<amrex::Real, 3> streaming_boundary_impulse{
        0.0_rt, 0.0_rt, 0.0_rt};
    amrex::GpuArray<amrex::Real, 3> pending_streaming_impulse{
        0.0_rt, 0.0_rt, 0.0_rt};
    amrex::GpuArray<amrex::Real, 3> pending_diffusion_impulse{
        0.0_rt, 0.0_rt, 0.0_rt};
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        if (warpx.m_fields.has(FieldType::radiation_material_momentum, lev)) {
            auto const* momentum = warpx.m_fields.get(
                FieldType::radiation_material_momentum, lev);
            for (int component = 0; component < 3; ++component) {
                material_impulse[component] += momentum->sum(
                    component, /*local=*/false);
            }
        }
        if (warpx.m_fields.has(
                FieldType::radiation_streaming_momentum_carry, lev))
        {
            auto const* carry = warpx.m_fields.get(
                FieldType::radiation_streaming_momentum_carry, lev);
            for (int component = 0; component < 3; ++component) {
                pending_streaming_impulse[component] += carry->sum(
                    component, /*local=*/false);
            }
        }
        if (warpx.m_fields.has(
                FieldType::radiation_diffusion_momentum_carry, lev))
        {
            auto const* carry = warpx.m_fields.get(
                FieldType::radiation_diffusion_momentum_carry, lev);
            for (int component = 0; component < 3; ++component) {
                pending_diffusion_impulse[component] += carry->sum(
                    component, /*local=*/false);
            }
        }
    }
    auto const& radiation = warpx.GetRadiationTransport();
    if (radiation.usesParticleMomentumCarry()) {
        auto& particles = warpx.GetPartContainer();
        auto const streaming = radiation.pendingMaterialImpulse(particles, true);
        auto const diffusion = radiation.pendingMaterialImpulse(particles, false);
        for (int d = 0; d < 3; ++d) {
            pending_streaming_impulse[d] = streaming[d];
            pending_diffusion_impulse[d] = diffusion[d];
        }
    }
    for (int component = 0; component < 3; ++component) {
        boundary_impulse[component] =
            radiation.lastDiffusionBoundaryMomentumLoss(component);
        streaming_boundary_impulse[component] =
            radiation.lastStreamingBoundaryMomentumLoss(component);
    }

    if (step >= 0 && step != m_last_accumulated_step) {
        for (int component = 0; component < 3; ++component) {
            m_cumulative_material_impulse[component] +=
                material_impulse[component];
            m_cumulative_boundary_impulse[component] +=
                boundary_impulse[component];
            m_cumulative_streaming_boundary_impulse[component] +=
                streaming_boundary_impulse[component];
        }
        m_last_accumulated_step = step;
    }

    if (!m_intervals.contains(step + 1)) { return; }

    for (int component = 0; component < 3; ++component) {
        m_data[component] = material_impulse[component];
        m_data[3 + component] = m_cumulative_material_impulse[component];
        m_data[6 + component] = boundary_impulse[component];
        m_data[9 + component] = m_cumulative_boundary_impulse[component];
        m_data[12 + component] = streaming_boundary_impulse[component];
        m_data[15 + component] =
            m_cumulative_streaming_boundary_impulse[component];
        m_data[18 + component] = pending_streaming_impulse[component];
        m_data[21 + component] = pending_diffusion_impulse[component];
    }
    if (m_include_moment_inventory) {
        auto const inventory = radiation.momentMomentumInventory(warpx.m_fields);
        for (int component = 0; component < 3; ++component) {
            m_data[24 + component] = inventory[component];
        }
    }
}

void RadiationMomentum::WriteCheckpointData (std::string const& dir)
{
    std::ofstream checkpoint{
        dir + "/" + m_rd_name + "_RadiationMomentum_data.txt",
        std::ofstream::out};
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        checkpoint.good(),
        "RadiationMomentum could not write its checkpoint state.");
    checkpoint.precision(17);
    for (int component = 0; component < 3; ++component) {
        checkpoint << m_cumulative_material_impulse[component] << "\n";
    }
    for (int component = 0; component < 3; ++component) {
        checkpoint << m_cumulative_boundary_impulse[component] << "\n";
    }
    for (int component = 0; component < 3; ++component) {
        checkpoint << m_cumulative_streaming_boundary_impulse[component]
                   << "\n";
    }
    checkpoint << m_last_accumulated_step << "\n";
    if (m_include_moment_inventory) {
        checkpoint << "moment_inventory_v1\n";
    }
}

void RadiationMomentum::ReadCheckpointData (std::string const& dir)
{
    std::ifstream checkpoint{
        dir + "/" + m_rd_name + "_RadiationMomentum_data.txt",
        std::ifstream::in};
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        checkpoint.good(),
        "RadiationMomentum could not read its checkpoint state.");
    bool valid = true;
    for (int component = 0; component < 3; ++component) {
        valid = valid && static_cast<bool>(
            checkpoint >> m_cumulative_material_impulse[component]);
    }
    for (int component = 0; component < 3; ++component) {
        valid = valid && static_cast<bool>(
            checkpoint >> m_cumulative_boundary_impulse[component]);
    }
    for (int component = 0; component < 3; ++component) {
        valid = valid && static_cast<bool>(
            checkpoint >> m_cumulative_streaming_boundary_impulse[component]);
    }
    valid = valid && static_cast<bool>(checkpoint >> m_last_accumulated_step);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        valid,
        "RadiationMomentum checkpoint state is truncated or invalid.");
    std::string schema, trailing;
    bool const has_inventory_schema = static_cast<bool>(checkpoint >> schema);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        has_inventory_schema == m_include_moment_inventory &&
            (!has_inventory_schema || (schema == "moment_inventory_v1" &&
                                       !(checkpoint >> trailing))),
        "RadiationMomentum restart must preserve include_moment_inventory and its schema.");
}
