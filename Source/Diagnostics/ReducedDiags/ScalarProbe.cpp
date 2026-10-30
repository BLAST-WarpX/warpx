/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "ScalarProbe.H"

#include "Fields.H"
#include "ScalarProbeSampling.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <AMReX_Gpu.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <vector>

ScalarProbe::ScalarProbe (std::string const& rd_name)
    : ReducedDiags{rd_name}
{
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    WARPX_ABORT_WITH_MESSAGE("ScalarProbe currently supports Cartesian geometry only");
#endif
    amrex::ParmParse const pp(rd_name);
    std::string geometry = "Point";
    pp.query("probe_geometry", geometry);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        geometry == "Point" || geometry == "Line" || geometry == "Plane",
        "ScalarProbe probe_geometry must be Point, Line or Plane");
    bool integrate = false, follow_window = false;
    pp.query("integrate", integrate);
    pp.query("do_moving_window_FP", follow_window);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!integrate && !follow_window,
        "ScalarProbe samples instantaneous values at fixed simulation coordinates");

    Position center{0,0,0}, end{0,0,0};
    utils::parser::getWithParser(pp, "z_probe", center[2]);
#if !defined(WARPX_DIM_1D_Z)
    utils::parser::queryWithParser(pp, "x_probe", center[0]);
#endif
#if defined(WARPX_DIM_3D)
    utils::parser::queryWithParser(pp, "y_probe", center[1]);
#endif
    int resolution = 0;
    if (geometry != "Point") {
        utils::parser::getWithParser(pp, "resolution", resolution);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(resolution >= 2 &&
            resolution <= std::numeric_limits<int>::max()/2,
            "ScalarProbe Line and Plane require 2 <= resolution <= INT_MAX/2");
    }
    if (geometry == "Point") {
        m_positions.push_back(center);
    } else if (geometry == "Line") {
        utils::parser::getWithParser(pp, "z1_probe", end[2]);
#if !defined(WARPX_DIM_1D_Z)
        utils::parser::queryWithParser(pp, "x1_probe", end[0]);
#endif
#if defined(WARPX_DIM_3D)
        utils::parser::queryWithParser(pp, "y1_probe", end[1]);
#endif
        m_positions.reserve(resolution);
        for (int i = 0; i < resolution; ++i) {
            Position point{};
            amrex::Real const fraction = amrex::Real(i) / (resolution-1);
            for (int d = 0; d < 3; ++d) {
                point[d] = center[d] + fraction * (end[d]-center[d]);
            }
            m_positions.push_back(point);
        }
    } else if (geometry == "Plane") {
#if defined(WARPX_DIM_1D_Z)
        WARPX_ABORT_WITH_MESSAGE("ScalarProbe Plane requires Cartesian 2D or 3D");
#endif
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::size_t(resolution)*std::size_t(resolution) <=
                std::size_t(std::numeric_limits<int>::max()/2),
            "ScalarProbe resolution exceeds the supported sample count");
        Position normal{0,1,0}, up{0,0,1};
        amrex::Real radius = 0;
        utils::parser::getWithParser(pp, "detector_radius", radius);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(radius) && radius > 0,
            "ScalarProbe detector_radius must be finite and positive");
#if defined(WARPX_DIM_3D)
        utils::parser::queryWithParser(pp, "target_normal_x", normal[0]);
        utils::parser::queryWithParser(pp, "target_normal_y", normal[1]);
        utils::parser::queryWithParser(pp, "target_normal_z", normal[2]);
        utils::parser::queryWithParser(pp, "target_up_y", up[1]);
#endif
        utils::parser::queryWithParser(pp, "target_up_x", up[0]);
        utils::parser::queryWithParser(pp, "target_up_z", up[2]);
        auto const normalize = [](Position& v) {
            amrex::Real const norm = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(norm) && norm > 0,
                "ScalarProbe plane normal/up vectors must be finite and nonzero");
            for (auto& x : v) { x /= norm; }
        };
        normalize(normal);
        normalize(up);
        amrex::Real const dot = normal[0]*up[0] + normal[1]*up[1] + normal[2]*up[2];
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(dot) < amrex::Real(0.999999),
            "ScalarProbe plane normal and up vectors must not be parallel");
        // Project the up vector into the plane to make a square detector.
        for (int d = 0; d < 3; ++d) { up[d] -= dot * normal[d]; }
        normalize(up);
        Position const side{normal[1]*up[2]-normal[2]*up[1],
                            normal[2]*up[0]-normal[0]*up[2],
                            normal[0]*up[1]-normal[1]*up[0]};
        // Match FieldProbe's center-to-corner radius and side/up ordering.
        amrex::Real const half_width = radius / std::sqrt(amrex::Real(2));
        m_positions.reserve(std::size_t(resolution)*resolution);
        for (int i = 0; i < resolution; ++i) {
            for (int j = 0; j < resolution; ++j) {
                Position point{};
                for (int d = 0; d < 3; ++d) {
                    point[d] = center[d] + half_width *
                        ((amrex::Real(2)*i/(resolution-1)-1)*side[d] +
                         (amrex::Real(2)*j/(resolution-1)-1)*up[d]);
                }
                m_positions.push_back(point);
            }
        }
    } else {
        WARPX_ABORT_WITH_MESSAGE("ScalarProbe probe_geometry must be Point, Line or Plane");
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_positions.size() <=
        std::size_t(std::numeric_limits<int>::max()/2), "Too many ScalarProbe samples");
    for (auto const& point : m_positions) {
        for (auto x : point) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(x),
                "ScalarProbe coordinates must be finite");
        }
    }
    m_data.resize(2*m_positions.size());
    m_levels.resize(m_positions.size(), -1);
    if (amrex::ParallelDescriptor::IOProcessor() && m_write_header) {
        std::ofstream out(m_path + m_rd_name + "." + m_extension);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(out.good(), "Cannot open ScalarProbe output file");
        out << "step" << m_sep << "time(s)" << m_sep << "probe_id" << m_sep << "level"
            << m_sep << "x(m)" << m_sep << "y(m)" << m_sep << "z(m)"
            << m_sep << "phi(V)" << m_sep << "rho(C/m^3)\n";
    }
}

void ScalarProbe::CheckFields () const
{
    auto const& warpx = WarpX::GetInstance();
    using warpx::fields::FieldType;
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(warpx.m_fields.has(FieldType::phi_fp, lev) &&
            warpx.m_fields.has(FieldType::rho_fp, lev),
            "ScalarProbe requires stored phi_fp and rho_fp fields (an electrostatic solver)");
        for (auto type : {FieldType::phi_fp, FieldType::rho_fp}) {
            auto const& field = *warpx.m_fields.get(type, lev);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(field.is_nodal() && field.nComp() == 1 &&
                field.boxArray() == amrex::convert(warpx.boxArray(lev),
                                                   amrex::IntVect::TheNodeVector()) &&
                field.DistributionMap() == warpx.DistributionMap(lev),
                "ScalarProbe requires scalar nodal phi/rho on the simulation grids");
        }
    }
}

void ScalarProbe::InitData ()
{
    CheckFields();
}

void ScalarProbe::ComputeDiags (int step)
{
    if (!m_intervals.contains(step+1)) { return; }
    CheckFields();
    auto const& warpx = WarpX::GetInstance();
    using warpx::fields::FieldType;
    using BoxSamples = std::vector<std::vector<Sample>>;
    std::vector<BoxSamples> samples(warpx.finestLevel()+1);
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        samples[lev].resize(warpx.boxArray(lev).size());
    }
    std::fill(m_data.begin(), m_data.end(), amrex::Real(0));
    std::fill(m_levels.begin(), m_levels.end(), -1);
    for (int id = 0; id < static_cast<int>(m_positions.size()); ++id) {
        auto const& p = m_positions[id];
#if (AMREX_SPACEDIM == 1)
        amrex::GpuArray<amrex::Real, 1> const position{p[2]};
#elif (AMREX_SPACEDIM == 2)
        amrex::GpuArray<amrex::Real, 2> const position{p[0], p[2]};
#else
        amrex::GpuArray<amrex::Real, 3> const position{p[0], p[1], p[2]};
#endif
        for (int lev = warpx.finestLevel(); lev >= 0; --lev) {
            auto const& geom = warpx.Geom(lev);
            Sample sample{id, amrex::IntVect(0), {}};
            bool inside = true;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                if (position[d] < geom.ProbLo(d) || position[d] > geom.ProbHi(d)) {
                    inside = false;
                    break;
                }
                amrex::Real const coordinate = (position[d]-geom.ProbLo(d)) *
                    geom.InvCellSize(d) + geom.Domain().smallEnd(d);
                sample.m_cell[d] = std::clamp(static_cast<int>(std::floor(coordinate)),
                    geom.Domain().smallEnd(d), geom.Domain().bigEnd(d));
                sample.m_fraction[d] = std::clamp(coordinate-sample.m_cell[d],
                    amrex::Real(0), amrex::Real(1));
            }
            if (!inside) { continue; }
            auto const boxes = warpx.boxArray(lev).intersections(
                amrex::Box(sample.m_cell, sample.m_cell));
            if (boxes.empty()) { continue; }
            int const box = boxes.front().first;
            m_levels[id] = lev;
            if (warpx.DistributionMap(lev)[box] == amrex::ParallelDescriptor::MyProc()) {
                samples[lev][box].push_back(sample);
            }
            break;
        }
    }
    // Match the default first-order interpolation used by FieldProbe. Define
    // the value outside the device lambda, as FieldProbe does for CUDA builds.
    const int temp_interp_order = 1;
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        auto const& phi = *warpx.m_fields.get(FieldType::phi_fp, lev);
        auto const& rho = *warpx.m_fields.get(FieldType::rho_fp, lev);
        for (int box = 0; box < static_cast<int>(samples[lev].size()); ++box) {
            auto const& points = samples[lev][box];
            if (points.empty()) { continue; }
            amrex::Gpu::DeviceVector<Sample> device_points(points.size());
            amrex::Gpu::DeviceVector<amrex::Real> values(2*points.size());
            amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                points.begin(), points.end(), device_points.begin());
            auto const* q = device_points.data();
            auto* result = values.data();
            auto const phi_array = phi[box].const_array();
            auto const rho_array = rho[box].const_array();
            amrex::ParallelFor(static_cast<int>(points.size()),
                [=] AMREX_GPU_DEVICE (int i) {
                    result[2*i] = warpx::diagnostics::sampleNodalScalar(
                        phi_array, q[i].m_cell, q[i].m_fraction, temp_interp_order);
                    result[2*i+1] = warpx::diagnostics::sampleNodalScalar(
                        rho_array, q[i].m_cell, q[i].m_fraction, temp_interp_order);
                });
            std::vector<amrex::Real> host(values.size());
            amrex::Gpu::copyAsync(
                amrex::Gpu::deviceToHost, values.begin(), values.end(), host.begin());
            amrex::Gpu::streamSynchronize();
            for (std::size_t i = 0; i < points.size(); ++i) {
                m_data[2*points[i].m_id] = host[2*i];
                m_data[2*points[i].m_id+1] = host[2*i+1];
            }
        }
    }
    // One owning cell per probe: sum communicates samples without double-counting nodes.
    amrex::ParallelDescriptor::ReduceRealSum(m_data.data(), static_cast<int>(m_data.size()),
        amrex::ParallelDescriptor::IOProcessorNumber());
}

void ScalarProbe::WriteToFile (int step) const
{
    if (!amrex::ParallelDescriptor::IOProcessor()) { return; }
    std::ofstream out(m_path + m_rd_name + "." + m_extension, std::ios::app);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(out.good(), "Cannot open ScalarProbe output file");
    out << std::scientific << std::setprecision(m_precision);
    for (std::size_t id = 0; id < m_positions.size(); ++id) {
        out << step+1 << m_sep << WarpX::GetInstance().gett_new(0)
            << m_sep << id << m_sep << m_levels[id];
        for (auto coordinate : m_positions[id]) { out << m_sep << coordinate; }
        auto const missing = std::numeric_limits<amrex::Real>::quiet_NaN();
        out << m_sep << (m_levels[id] < 0 ? missing : m_data[2*id])
            << m_sep << (m_levels[id] < 0 ? missing : m_data[2*id+1]) << '\n';
    }
}
