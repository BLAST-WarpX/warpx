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
#include "Utils/WarpXConst.H"
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
    amrex::ParmParse(rd_name).query("include_moment_transport", m_include_moment_transport);
    amrex::ParmParse(rd_name).query("include_angular_inventory", m_include_angular_inventory);
    amrex::ParmParse(rd_name).query("include_particle_wall", m_include_particle_wall);
    amrex::ParmParse(rd_name).query("include_thermal_axial_wall", m_include_thermal_axial_wall);
    if (m_include_thermal_axial_wall)
    {
        bool reflecting_z = false;
        pp_radiation.query("resolved_thermal_reflecting_z", reflecting_z);
#if defined(WARPX_DIM_RZ)
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(reflecting_z && diffusion_solver == "resolved_packet_fld",
            "RadiationMomentum.include_thermal_axial_wall requires closed thermal RZ runtime.");
#else
        WARPX_ABORT_WITH_MESSAGE("RadiationMomentum.include_thermal_axial_wall requires RZ.");
#endif
    }
#if !defined(WARPX_DIM_1D_Z) && !defined(WARPX_DIM_XZ) && !defined(WARPX_DIM_3D) && \
    !defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_include_particle_wall,
        "RadiationMomentum.include_particle_wall requires Cartesian or RZ geometry.");
#endif
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_include_angular_inventory || m_include_moment_transport ||
            (diffusion_solver == "resolved_packet_fld" && m_include_moment_inventory),
        "RadiationMomentum.include_angular_inventory requires include_moment_transport=1 "
        "or resolved_packet_fld with include_moment_inventory=1.");
#if !defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_include_angular_inventory,
        "RadiationMomentum.include_angular_inventory requires RZ geometry.");
#endif
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_include_moment_transport || m_include_moment_inventory,
        "RadiationMomentum.include_moment_transport requires include_moment_inventory=1.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_include_moment_transport ||
        (diffusion_solver == "coupled_moment" || diffusion_solver == "moving_multigroup"),
        "RadiationMomentum.include_moment_transport requires coupled_moment/moving_multigroup transport; "
        "flux_relaxed currently exposes its inventory, not the M1 transport ledger.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_include_moment_inventory || diffusion_solver == "coupled_moment" ||
            diffusion_solver == "flux_relaxed" ||
            diffusion_solver == "resolved_packet_fld" ||
            diffusion_solver == "moving_multigroup",
        "RadiationMomentum.include_moment_inventory requires coupled_moment "
        "or flux_relaxed/resolved_packet_fld transport.");
    auto const legacy_columns =
        (m_include_moment_transport ? 39 : (m_include_moment_inventory ? 27 : 24)) +
        (m_include_angular_inventory ? 3 : 0);
    m_data.resize(legacy_columns + (m_include_particle_wall ? 3 : 0), 0.0_rt);
    m_particle_wall_column = legacy_columns;
    m_thermal_axial_wall_column = m_data.size();
    if (m_include_thermal_axial_wall) { m_data.resize(m_data.size() + 12, 0.0_rt); }

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
        if (m_include_moment_transport) {
            for (char const* account : {"boundary", "geometric", "boundary_minus_geometric"}) {
                output << m_sep << "[" << column++ << "]cumulative_moment_"
                       << account << "_energy(J)";
                for (std::string const& label : labels) {
                    output << m_sep << "[" << column++ << "]cumulative_moment_"
                           << account << '_' << label << "(kg*m/s)";
                }
            }
        }
        if (m_include_angular_inventory) {
            for (char const* account : {"radiation", "represented_material", "pending_material"}) {
                output << m_sep << "[" << column++ << "]angular_" << account << "_z(kg*m^2/s)";
            }
        }
        if (m_include_particle_wall) {
            // This ledger contains actual native Cartesian particle momentum,
            // also in RZ. It is not a sum of local radial/theta projections.
            for (std::string const& label : std::array<std::string, 3>{"x", "y", "z"}) {
                output << m_sep << "[" << column++ << "]particle_wall_external_"
                       << label << "(kg*m/s)";
            }
        }
        if (m_include_thermal_axial_wall)
        {
            for (char const* epoch : {"last", "cumulative"})
            {
                for (char const* component : {"K_low", "K_high", "K_net", "D_low", "D_high", "D_net"})
                {
                    output << m_sep << "[" << column++ << "]" << epoch <<
                        "_radiation_axial_wall_" << component << "(kg*m/s)";
                }
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
    amrex::GpuArray<amrex::Real, 3> const particle_wall_impulse =
        radiation.lastParticleWallMomentum();
    if (radiation.usesParticleMomentumCarry()) {
        auto& particles = warpx.GetPartContainer();
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
        constexpr bool local_cylindrical = true;
#else
        constexpr bool local_cylindrical = false;
#endif
        auto const streaming = radiation.pendingMaterialImpulse(particles, true, local_cylindrical);
        auto const diffusion = radiation.pendingMaterialImpulse(particles, false, local_cylindrical);
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
        if (m_include_particle_wall) {
            m_data[m_particle_wall_column + component] = particle_wall_impulse[component];
        }
    }
    if (m_include_thermal_axial_wall)
    {
        auto const& wall = radiation.thermalAxialWallAccounts();
        std::array<amrex::Real, 12> const values{
            wall.kinetic[0], wall.kinetic[1], wall.kinetic_net,
            wall.diffusion[0], wall.diffusion[1], wall.diffusion_net,
            wall.cumulative_kinetic[0], wall.cumulative_kinetic[1], wall.cumulative_kinetic_net,
            wall.cumulative_diffusion[0], wall.cumulative_diffusion[1], wall.cumulative_diffusion_net};
        for (std::size_t column = 0; column < values.size(); ++column)
        { m_data[m_thermal_axial_wall_column + column] = values[column]; }
    }
    if (m_include_moment_inventory) {
        auto const inventory = radiation.momentMomentumInventory(warpx.m_fields);
        for (int component = 0; component < 3; ++component) {
            m_data[24 + component] = inventory[component];
        }
    }
    if (m_include_moment_transport) {
        auto const& ledger = radiation.momentTransportLedger();
        std::array<warpx::radiation::FourVector, 3> const accounts{
            ledger.Boundary(), ledger.Geometric(), ledger.Balance()};
        int offset = 27;
        for (auto const& account : accounts) {
            m_data[offset] = account[0];
            for (int component = 1; component < 4; ++component) {
                m_data[offset + component] = account[component] / PhysConst::c;
            }
            offset += 4;
        }
    }
    if (m_include_angular_inventory) {
        auto const inventory = radiation.momentAngularInventory(warpx.GetPartContainer(), warpx.m_fields);
        for (int component = 0; component < 3; ++component) {
            m_data[(m_include_moment_transport ? 39 : 27) + component] = inventory[component];
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
        checkpoint << (m_include_angular_inventory ? "moment_inventory_transport_angular_v3\n" :
            (m_include_moment_transport ? "moment_inventory_transport_v2\n" : "moment_inventory_v1\n"));
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
    std::string const expected_schema = m_include_angular_inventory
        ? "moment_inventory_transport_angular_v3"
        : (m_include_moment_transport ? "moment_inventory_transport_v2" : "moment_inventory_v1");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        has_inventory_schema == m_include_moment_inventory &&
            (!has_inventory_schema || (schema == expected_schema &&
                                       !(checkpoint >> trailing))),
        "RadiationMomentum restart must preserve include_moment_inventory, "
        "include_moment_transport, include_angular_inventory and their schema.");
}
