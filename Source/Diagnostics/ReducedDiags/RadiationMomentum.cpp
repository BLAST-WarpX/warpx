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

    m_data.resize(18, 0.0_rt);

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
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        if (warpx.m_fields.has(FieldType::radiation_material_momentum, lev)) {
            auto const* momentum = warpx.m_fields.get(
                FieldType::radiation_material_momentum, lev);
            for (int component = 0; component < 3; ++component) {
                material_impulse[component] += momentum->sum(
                    component, /*local=*/false);
            }
        }
    }
    auto const& radiation = warpx.GetRadiationTransport();
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
}
