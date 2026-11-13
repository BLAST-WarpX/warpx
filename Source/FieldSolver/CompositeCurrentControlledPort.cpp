/* Copyright 2026 WarpX contributors
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "CompositeCurrentControlledPort.H"

#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H"
#include "Fields.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_iMultiFab.H>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <utility>

using namespace amrex::literals;
using warpx::fields::FieldType;

namespace {
AMREX_GPU_HOST_DEVICE constexpr int
physical_direction (int mesh_direction)
{
#ifdef WARPX_DIM_3D
    return mesh_direction;
#else
    return mesh_direction == 0 ? 0 : 2;
#endif
}

AMREX_GPU_HOST_DEVICE constexpr int
mesh_direction (int direction)
{
#ifdef WARPX_DIM_3D
    return direction;
#else
    return direction == 0 ? 0 : direction == 2 ? 1 : -1;
#endif
}

struct CurlGeometry {
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lower;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> spacing;
    amrex::IntVect origin;
    amrex::GpuArray<amrex::Real, 3> port_lower;
    amrex::GpuArray<amrex::Real, 3> port_upper;
    int axis;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
potential (int const* index, CurlGeometry const& geometry)
{
    amrex::Real result = 1.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        int const direction = physical_direction(d);
        if (direction == geometry.axis) {
            continue;
        }
        amrex::Real const coordinate =
            geometry.lower[d] + (index[d] - geometry.origin[d]) * geometry.spacing[d];
        amrex::Real const lo = geometry.port_lower[direction];
        amrex::Real const hi = geometry.port_upper[direction];
        amrex::Real const tolerance = 32.0_rt * std::numeric_limits<amrex::Real>::epsilon() *
                                      amrex::max(1.0_rt, amrex::max(std::abs(lo), std::abs(hi)));
        if (coordinate < lo - tolerance || coordinate > hi + tolerance) {
            return 0.0_rt;
        }
        if (std::abs(coordinate - lo) <= tolerance || std::abs(coordinate - hi) <= tolerance) {
            result *= 0.5_rt;
        }
    }
    return result;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
curl (int component, int i, int j, int k, CurlGeometry const& geometry)
{
    if (component == geometry.axis) {
        return 0.0_rt;
    }
    int const derivative = 3 - component - geometry.axis;
    int const mesh_derivative = mesh_direction(derivative);
    if (mesh_derivative < 0 || mesh_derivative >= AMREX_SPACEDIM) {
        return 0.0_rt;
    }
    int index[3] = {i, j, k};
    amrex::Real const lower = potential(index, geometry);
    ++index[mesh_derivative];
    amrex::Real const sign = derivative == (component + 1) % 3 ? 1.0_rt : -1.0_rt;
    return sign * (potential(index, geometry) - lower) / geometry.spacing[mesh_derivative];
}

CurlGeometry
curl_geometry (amrex::Geometry const& geometry, std::array<amrex::Real, 3> const& lower,
               std::array<amrex::Real, 3> const& upper, int axis)
{
    return {geometry.ProbLoArray(),         geometry.CellSizeArray(),
            geometry.Domain().smallEnd(),   {lower[0], lower[1], lower[2]},
            {upper[0], upper[1], upper[2]}, axis};
}
} // namespace

CompositeCurrentControlledPort::CompositeCurrentControlledPort (std::array<amrex::Real, 3> lower,
                                                                std::array<amrex::Real, 3> upper,
                                                                int direction,
                                                                amrex::Real terminal_0,
                                                                amrex::Real terminal_1)
    : m_lower(lower), m_upper(upper), m_direction(direction),
      m_mesh_direction(mesh_direction(direction))
{
#if !defined(WARPX_DIM_3D) && !defined(WARPX_DIM_XZ)
    WARPX_ABORT_WITH_MESSAGE("Composite current ports require Cartesian 2D or 3D geometry.");
#endif
    auto const& warpx = WarpX::GetInstance();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::Yee &&
            WarpX::grid_type == GridType::Staggered,
        "Ports crossing refinement boundaries require the staggered Yee "
        "solver.");
    m_lower[m_direction] = std::min(terminal_0, terminal_1);
    m_upper[m_direction] = std::max(terminal_0, terminal_1);
    auto const& coarse_geometry = warpx.Geom(0);
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        int const direction_d = physical_direction(d);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_lower[direction_d] >= coarse_geometry.ProbLo(d) + coarse_geometry.CellSize(d) &&
                m_upper[direction_d] <= coarse_geometry.ProbHi(d) - coarse_geometry.CellSize(d),
            "A composite current port must stay one coarse cell inside the "
            "domain boundaries.");
    }

    m_levels.resize(warpx.finestLevel() + 1);
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        auto& level = m_levels[lev];
        auto const& geometry = warpx.Geom(lev);
        amrex::Real const spacing = geometry.CellSize(m_mesh_direction);
        amrex::Real const origin = geometry.ProbLo(m_mesh_direction);
        int const first = geometry.Domain().smallEnd(m_mesh_direction);
        level.first = first + static_cast<int>(
                                  std::floor((m_lower[m_direction] - origin) / spacing - 0.5_rt));
        level.last =
            first + static_cast<int>(std::ceil((m_upper[m_direction] - origin) / spacing - 0.5_rt));
        level.offset = m_unknowns;
        m_unknowns += level.last - level.first + 1;
    }

    auto const& fine_geometry = warpx.Geom(warpx.finestLevel());
    auto const& fine_level = m_levels.back();
    m_planes = {terminal_0, terminal_1};
    for (int plane = fine_level.first; plane <= fine_level.last; ++plane) {
        amrex::Real const coordinate =
            fine_geometry.ProbLo(m_mesh_direction) +
            (plane - fine_geometry.Domain().smallEnd(m_mesh_direction) + 0.5_rt) *
                fine_geometry.CellSize(m_mesh_direction);
        if (coordinate > m_lower[m_direction] && coordinate < m_upper[m_direction]) {
            m_planes.push_back(coordinate);
        }
    }
    std::sort(m_planes.begin(), m_planes.end());
    m_planes.erase(std::unique(m_planes.begin(), m_planes.end()), m_planes.end());
    m_terminal_rows = {
        static_cast<int>(std::lower_bound(m_planes.begin(), m_planes.end(), terminal_0) -
                         m_planes.begin()),
        static_cast<int>(std::lower_bound(m_planes.begin(), m_planes.end(), terminal_1) -
                         m_planes.begin())};
    m_columns.resize(m_planes.size() * 2 * m_levels.size(), -1);
    for (int row = 0; row < static_cast<int>(m_planes.size()); ++row) {
        for (int lev = 0; lev < static_cast<int>(m_levels.size()); ++lev) {
            auto const& geometry = warpx.Geom(lev);
            int const first =
                geometry.Domain().smallEnd(m_mesh_direction) +
                static_cast<int>(std::floor((m_planes[row] - geometry.ProbLo(m_mesh_direction)) /
                                                geometry.CellSize(m_mesh_direction) -
                                            0.5_rt));
            for (int side = 0; side < 2; ++side) {
                int const plane = first + side;
                if (plane >= m_levels[lev].first && plane <= m_levels[lev].last) {
                    m_columns[(row * m_levels.size() + lev) * 2 + side] =
                        m_levels[lev].offset + plane - m_levels[lev].first;
                }
            }
        }
    }
    BuildSamples();
}

void
CompositeCurrentControlledPort::BuildSamples ()
{
    auto& warpx = WarpX::GetInstance();
    using Buckets = std::array<std::vector<std::vector<Sample>>, 3>;
    std::vector<Buckets> buckets(m_levels.size());
    for (int lev = 0; lev < static_cast<int>(m_levels.size()); ++lev) {
        for (int component = 0; component < 3; ++component) {
            buckets[lev][component].resize(warpx.boxArray(lev).size());
            m_levels[lev].samples[component].resize(warpx.boxArray(lev).size());
        }
    }
    auto const add_sample = [&] (std::array<amrex::Real, 3> const& point, int component,
                                 amrex::Real weight, int row) {
        for (int lev = warpx.finestLevel(); lev >= 0; --lev) {
            auto const& geometry = warpx.Geom(lev);
            int owner[3] = {0, 0, 0};
            Sample sample;
            sample.row = row;
            sample.weight = weight;
            auto const* field = warpx.m_fields.get(FieldType::Bfield_aux,
                                                   ablastr::fields::Direction{component}, lev);
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                amrex::Real const coordinate =
                    (point[physical_direction(d)] - geometry.ProbLo(d)) / geometry.CellSize(d);
                owner[d] = geometry.Domain().smallEnd(d) + static_cast<int>(std::floor(coordinate));
                amrex::Real const raw = coordinate - (field->ixType().nodeCentered(d) ? 0 : 0.5_rt);
                amrex::Real const index = std::floor(raw);
                sample.index[d] = geometry.Domain().smallEnd(d) + static_cast<int>(index);
                sample.fraction[d] = raw - index;
            }
            amrex::IntVect const cell(AMREX_D_DECL(owner[0], owner[1], owner[2]));
            auto const intersections = warpx.boxArray(lev).intersections(amrex::Box(cell, cell));
            if (intersections.empty()) {
                continue;
            }
            int const grid = intersections.front().first;
            if (field->DistributionMap()[grid] == amrex::ParallelDescriptor::MyProc()) {
                buckets[lev][component][grid].push_back(sample);
            }
            return;
        }
        WARPX_ABORT_WITH_MESSAGE("A composite current contour lies outside the grid hierarchy.");
    };

    for (int row = 0; row < static_cast<int>(m_planes.size()); ++row) {
#ifdef WARPX_DIM_3D
        int const tangent_0 = (m_direction + 1) % 3;
        int const tangent_1 = (m_direction + 2) % 3;
        for (int edge = 0; edge < 4; ++edge) {
            int const tangent = edge < 2 ? tangent_0 : tangent_1;
            int const fixed = edge < 2 ? tangent_1 : tangent_0;
            bool const upper = edge % 2 != 0;
            amrex::Real const orientation =
                edge < 2 ? (upper ? -1.0_rt : 1.0_rt) : (upper ? 1.0_rt : -1.0_rt);
            amrex::Real const length = m_upper[tangent] - m_lower[tangent];
            int const count =
                std::max(1, static_cast<int>(std::ceil(
                                length / warpx.Geom(warpx.finestLevel()).CellSize(tangent))));
            amrex::Real const spacing = length / count;
            for (int point = 0; point < count; ++point) {
                std::array<amrex::Real, 3> coordinate{};
                coordinate[m_direction] = m_planes[row];
                coordinate[fixed] = upper ? m_upper[fixed] : m_lower[fixed];
                coordinate[tangent] = m_lower[tangent] + (point + 0.5_rt) * spacing;
                add_sample(coordinate, tangent, orientation * spacing, row);
            }
        }
#else
        int const transverse = m_direction == 0 ? 2 : 0;
        for (int side = 0; side < 2; ++side) {
            std::array<amrex::Real, 3> coordinate{};
            coordinate[m_direction] = m_planes[row];
            coordinate[transverse] = side == 0 ? m_lower[transverse] : m_upper[transverse];
            amrex::Real const orientation = m_direction == 2 ? 1.0_rt : -1.0_rt;
            add_sample(coordinate, 1, side == 0 ? -orientation : orientation, row);
        }
#endif
    }
    for (int lev = 0; lev < static_cast<int>(m_levels.size()); ++lev) {
        for (int component = 0; component < 3; ++component) {
            for (int grid = 0; grid < static_cast<int>(buckets[lev][component].size()); ++grid) {
                auto const& source = buckets[lev][component][grid];
                auto& destination = m_levels[lev].samples[component][grid];
                destination.resize(source.size());
                amrex::Gpu::copy(amrex::Gpu::hostToDevice, source.begin(), source.end(),
                                 destination.begin());
            }
        }
    }
}

CompositeCurrentControlledPort::Measurement
CompositeCurrentControlledPort::Measure ()
{
    auto& warpx = WarpX::GetInstance();
    int const channels = 2 * static_cast<int>(m_levels.size()) + 3;
    std::size_t const entries = m_planes.size() * channels;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(entries <=
                                         static_cast<std::size_t>(std::numeric_limits<int>::max()),
                                     "Too many composite current-port contour samples.");
    amrex::Gpu::DeviceVector<amrex::Real> device_data(entries, 0.0_rt);
    amrex::Real* const sums = device_data.data();
    bool const build_response = m_response.empty();
    int const axis = m_mesh_direction;
    for (int lev = 0; lev < static_cast<int>(m_levels.size()); ++lev) {
        auto const geometry = curl_geometry(warpx.Geom(lev), m_lower, m_upper, m_direction);
        auto* const material = warpx.get_pointer_MacroscopicProperties(lev);
        for (int component = 0; component < 3; ++component) {
            auto const& field = *warpx.m_fields.get(FieldType::Bfield_aux,
                                                    ablastr::fields::Direction{component}, lev);
            auto const& flags = warpx.GetEBUpdateBFlag()[lev][component];
            for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
                auto const& samples = m_levels[lev].samples[component][mfi.index()];
                if (samples.empty()) {
                    continue;
                }
                auto const* const points = samples.data();
                auto const values = field.const_array(mfi);
                auto const permeability = material == nullptr
                                              ? amrex::Array4<amrex::Real const>{}
                                              : material->getmu_mf().const_array(mfi);
                auto const active =
                    flags == nullptr ? amrex::Array4<int const>{} : flags->const_array(mfi);
                amrex::ParallelFor(static_cast<int>(samples.size()), [=] AMREX_GPU_DEVICE(int p) {
                    Sample const point = points[p];
                    for (int corner = 0; corner < (1 << AMREX_SPACEDIM); ++corner) {
                        int index[3] = {point.index[0], point.index[1], point.index[2]};
                        amrex::Real weight = point.weight;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                            int const side = (corner >> d) & 1;
                            index[d] += side;
                            weight *= side == 0 ? 1.0_rt - point.fraction[d] : point.fraction[d];
                        }
                        if (weight == 0.0_rt) {
                            continue;
                        }
                        int const i = index[0], j = index[1], k = index[2];
                        weight *= permeability ? PhysConst::mu0 / permeability(i, j, k) : 1.0_rt;
                        std::size_t const row = static_cast<std::size_t>(point.row) * channels;
                        amrex::Real const contribution = weight * values(i, j, k);
                        amrex::Gpu::Atomic::AddNoRet(sums + row, contribution);
                        amrex::Gpu::Atomic::AddNoRet(sums + row + channels - 2,
                                                     std::abs(contribution));
                        if (build_response) {
                            if (active && active(i, j, k) == 0) {
                                amrex::Gpu::Atomic::AddNoRet(sums + row + channels - 1, 1.0_rt);
                            } else {
                                int const side = (corner >> axis) & 1;
                                amrex::Gpu::Atomic::AddNoRet(
                                    sums + row + 1 + 2 * lev + side,
                                    weight * curl(component, i, j, k, geometry));
                            }
                        }
                    }
                });
            }
        }
    }
    std::vector<amrex::Real> data(entries);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_data.begin(), device_data.end(),
                     data.begin());
    amrex::ParallelDescriptor::ReduceRealSum(data.data(), static_cast<int>(data.size()));
    Measurement result;
    result.circulation.resize(m_planes.size());
    result.absolute_circulation.resize(m_planes.size());
    if (build_response) {
        m_response.resize(m_planes.size() * (channels - 3));
    }
    for (int row = 0; row < static_cast<int>(m_planes.size()); ++row) {
        std::size_t const row_offset = static_cast<std::size_t>(row) * channels;
        std::size_t const response_offset = static_cast<std::size_t>(row) * (channels - 3);
        result.circulation[row] = data[row_offset];
        result.absolute_circulation[row] = data[row_offset + channels - 2];
        if (build_response) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                data[row_offset + channels - 1] == 0.0_rt,
                "A composite current-port contour must remain outside covered "
                "EB cells.");
            for (int slot = 0; slot < channels - 3; ++slot) {
                m_response[response_offset + slot] = data[row_offset + 1 + slot];
            }
        }
    }
    return result;
}

std::vector<amrex::Real>
CompositeCurrentControlledPort::Solve (Measurement const& measured,
                                       amrex::Real desired_circulation) const
{
    int const rows = static_cast<int>(m_planes.size());
    int const slots = 2 * static_cast<int>(m_levels.size());
    std::vector<double> response = m_response;
    std::vector<double> residual(rows), normalization(rows);
    double threshold = 0;
    for (int row = 0; row < rows; ++row) {
        std::size_t const row_offset = static_cast<std::size_t>(row) * slots;
        double norm = 0;
        for (int slot = 0; slot < slots; ++slot) {
            double const value = response[row_offset + slot];
            norm += value * value;
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(norm > 0,
                                         "A composite port has no solver-active curl response on "
                                         "its contour.");
        normalization[row] = std::sqrt(norm);
        residual[row] = (desired_circulation - measured.circulation[row]) / normalization[row];
        double const scale = std::max(std::abs(static_cast<double>(desired_circulation)),
                                      static_cast<double>(measured.absolute_circulation[row]));
        double const roundoff = 64 * std::numeric_limits<amrex::Real>::epsilon() *
                                std::max(scale, 1.e-30) / normalization[row];
        threshold += roundoff * roundoff;
        for (int slot = 0; slot < slots; ++slot) {
            response[row_offset + slot] /= normalization[row];
        }
    }
    auto const transpose = [&] (std::vector<double> const& input) {
        std::vector<double> result(m_unknowns, 0);
        for (int row = 0; row < rows; ++row) {
            std::size_t const row_offset = static_cast<std::size_t>(row) * slots;
            for (int slot = 0; slot < slots; ++slot) {
                int const column = m_columns[row_offset + slot];
                if (column >= 0) {
                    result[column] += response[row_offset + slot] * input[row];
                }
            }
        }
        return result;
    };
    auto const multiply = [&] (std::vector<double> const& input) {
        std::vector<double> result(rows, 0);
        for (int row = 0; row < rows; ++row) {
            std::size_t const row_offset = static_cast<std::size_t>(row) * slots;
            for (int slot = 0; slot < slots; ++slot) {
                int const column = m_columns[row_offset + slot];
                if (column >= 0) {
                    result[row] += response[row_offset + slot] * input[column];
                }
            }
        }
        return result;
    };
    auto const dot = [] (std::vector<double> const& left, std::vector<double> const& right) {
        return std::inner_product(left.begin(), left.end(), right.begin(), 0.0);
    };

    // CG on A A^T gives the minimum-norm curl amplitudes A^T y. Redundant
    // constraints on coarse-only contour segments do not require an inverse.
    std::vector<double> solution(rows, 0), direction = residual;
    double squared = dot(residual, residual);
    int const iterations =
        static_cast<int>(std::min(std::size_t{2} * m_planes.size() + 32, std::size_t{2000}));
    for (int iteration = 0; iteration < iterations && squared > threshold; ++iteration) {
        auto const applied = multiply(transpose(direction));
        double const denominator = dot(direction, applied);
        if (denominator <= 0) {
            break;
        }
        double const alpha = squared / denominator;
        for (int row = 0; row < rows; ++row) {
            solution[row] += alpha * direction[row];
            residual[row] -= alpha * applied[row];
        }
        double const next_squared = dot(residual, residual);
        double const beta = next_squared / squared;
        for (int row = 0; row < rows; ++row) {
            direction[row] = residual[row] + beta * direction[row];
        }
        squared = next_squared;
    }
    auto const amplitudes = transpose(solution);
    auto const achieved = multiply(amplitudes);
    for (int row = 0; row < rows; ++row) {
        double const difference =
            (desired_circulation - measured.circulation[row]) - normalization[row] * achieved[row];
        double const scale = std::max(std::abs(static_cast<double>(desired_circulation)),
                                      static_cast<double>(measured.absolute_circulation[row]));
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::abs(difference) <=
                512 * std::numeric_limits<amrex::Real>::epsilon() * std::max(scale, 1.e-30),
            "The coupled composite current-port projection did not converge.");
    }
    return std::vector<amrex::Real>(amplitudes.begin(), amplitudes.end());
}

void
CompositeCurrentControlledPort::AddCorrection (int level, PatchType patch_type,
                                               std::vector<amrex::Real> const& coefficients)
{
    auto& warpx = WarpX::GetInstance();
    bool const fine = patch_type == PatchType::fine;
    int const source_level = fine ? level : level - 1;
    auto const& source = m_levels[source_level];
    amrex::Gpu::DeviceVector<amrex::Real> device_coefficients(source.last - source.first + 1);
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, coefficients.begin() + source.offset,
                     coefficients.begin() + source.offset + source.last - source.first + 1,
                     device_coefficients.begin());
    auto const* const scale = device_coefficients.data();
    auto const& mesh = warpx.Geom(source_level);
    auto const geometry = curl_geometry(mesh, m_lower, m_upper, m_direction);
    int const first = source.first, last = source.last, axis = m_mesh_direction;
    for (int region = 0; region < 2; ++region) {
        FieldType const type = region == 0 ? (fine ? FieldType::Bfield_fp : FieldType::Bfield_cp)
                                           : (fine ? FieldType::pml_B_fp : FieldType::pml_B_cp);
        if (!warpx.m_fields.has_vector(type, level)) {
            continue;
        }
        for (int component = 0; component < 3; ++component) {
            if (component == m_direction) {
                continue;
            }
            auto& field = *warpx.m_fields.get(type, ablastr::fields::Direction{component}, level);
            int const derivative = 3 - component - m_direction;
            int const split = region == 0 ? 0 : (derivative == (component + 1) % 3 ? 0 : 1);
            auto const& flags = warpx.GetEBUpdateBFlag(patch_type)[level][component];
            for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
                amrex::Box box = mfi.validbox();
                box.setSmall(axis, std::max(box.smallEnd(axis), first));
                box.setBig(axis, std::min(box.bigEnd(axis), last));
                if (!box.ok()) {
                    continue;
                }
                auto const values = field.array(mfi);
                auto const active = region != 0 || flags == nullptr ? amrex::Array4<int const>{}
                                                                    : flags->const_array(mfi);
                amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if (active && active(i, j, k) == 0) {
                        return;
                    }
                    int const index[3] = {i, j, k};
                    values(i, j, k, split) +=
                        scale[index[axis] - first] * curl(component, i, j, k, geometry);
                });
            }
        }
    }
    amrex::Gpu::streamSynchronize();
}

void
CompositeCurrentControlledPort::Apply (amrex::Real target_axis_current)
{
    auto const measured = Measure();
    auto const coefficients = Solve(measured, PhysConst::mu0 * target_axis_current);
    for (int lev = 0; lev < static_cast<int>(m_levels.size()); ++lev) {
        AddCorrection(lev, PatchType::fine, coefficients);
        if (lev > 0) {
            AddCorrection(lev, PatchType::coarse, coefficients);
        }
    }
}

std::array<amrex::Real, 2>
CompositeCurrentControlledPort::Currents (amrex::Real target_axis_current)
{
    auto const measured = Measure();
    amrex::Real const desired = PhysConst::mu0 * target_axis_current;
    for (int row = 0; row < static_cast<int>(m_planes.size()); ++row) {
        amrex::Real const scale = std::max(std::abs(desired), measured.absolute_circulation[row]);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::abs(measured.circulation[row] - desired) <=
                1024.0_rt * std::numeric_limits<amrex::Real>::epsilon() *
                    std::max(scale, 1.e-30_rt),
            "The reconstructed magnetic field does not satisfy the composite "
            "port constraint.");
    }
    return {measured.circulation[m_terminal_rows[0]] / PhysConst::mu0,
            measured.circulation[m_terminal_rows[1]] / PhysConst::mu0};
}
