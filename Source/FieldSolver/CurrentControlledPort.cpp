/* Copyright 2026 WarpX contributors
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "CurrentControlledPort.H"

#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <AMReX_iMultiFab.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace amrex;

namespace {
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
[[nodiscard]] int
nearest_index (amrex::MultiFab const& field, int const direction,
               amrex::Real const coordinate) {
    amrex::Geometry const& geometry = WarpX::GetInstance().Geom(0);
    amrex::Box const cell_domain = geometry.Domain();
    amrex::Box const field_domain = amrex::convert(cell_domain, field.ixType());
    int const nodal = field.ixType().nodeCentered(direction) ? 1 : 0;
    amrex::Real const offset = nodal ? 0.0_rt : 0.5_rt;
    amrex::Real const raw = (coordinate - geometry.ProbLo(direction)) /
                                geometry.CellSize(direction) +
                            cell_domain.smallEnd(direction) - offset;
    int const index = static_cast<int>(std::llround(raw));
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        index >= field_domain.smallEnd(direction) &&
            index <= field_domain.bigEnd(direction),
        "Current-controlled port coordinate lies outside the field domain.");
    return index;
}
#endif

#ifdef WARPX_DIM_3D
[[nodiscard]] std::array<int, 2>
index_range (amrex::MultiFab const& field, int const direction,
             amrex::Real const lower, amrex::Real const upper) {
    amrex::Geometry const& geometry = WarpX::GetInstance().Geom(0);
    amrex::Box const cell_domain = geometry.Domain();
    amrex::Box const field_domain = amrex::convert(cell_domain, field.ixType());
    int const nodal = field.ixType().nodeCentered(direction) ? 1 : 0;
    amrex::Real const offset = nodal ? 0.0_rt : 0.5_rt;
    amrex::Real const raw_lower =
        (lower - geometry.ProbLo(direction)) / geometry.CellSize(direction) +
        cell_domain.smallEnd(direction) - offset;
    amrex::Real const raw_upper =
        (upper - geometry.ProbLo(direction)) / geometry.CellSize(direction) +
        cell_domain.smallEnd(direction) - offset;
    amrex::Real constexpr tolerance =
        64.0_rt * std::numeric_limits<amrex::Real>::epsilon();
    int first = static_cast<int>(std::ceil(raw_lower - tolerance));
    int last = static_cast<int>(std::floor(raw_upper + tolerance));
    first = std::max(first, field_domain.smallEnd(direction));
    last = std::min(last, field_domain.bigEnd(direction));
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        last > first, "Each current-controlled port contour edge must span at "
                      "least one grid interval.");
    return {first, last};
}
#endif

#ifdef WARPX_DIM_RZ
[[nodiscard]] amrex::Real
index_coordinate (amrex::MultiFab const& field, int const direction,
                  int const index) {
    amrex::Geometry const& geometry = WarpX::GetInstance().Geom(0);
    amrex::Box const cell_domain = geometry.Domain();
    amrex::Real const offset =
        field.ixType().nodeCentered(direction) ? 0.0_rt : 0.5_rt;
    return geometry.ProbLo(direction) +
           (index - cell_domain.smallEnd(direction) + offset) *
               geometry.CellSize(direction);
}
#endif

void
read_terminal (amrex::ParmParse const& parser, std::string const& name,
               std::array<amrex::Real, 3>& lower_bound,
               std::array<amrex::Real, 3>& upper_bound) {
    amrex::Vector<amrex::Real> lower(3);
    amrex::Vector<amrex::Real> upper(3);
    utils::parser::getArrWithParser(parser, (name + ".lower_bound").c_str(),
                                    lower, 0, 3);
    utils::parser::getArrWithParser(parser, (name + ".upper_bound").c_str(),
                                    upper, 0, 3);
    for (int direction = 0; direction < 3; ++direction) {
        lower_bound[direction] = lower[direction];
        upper_bound[direction] = upper[direction];
    }
}
} // namespace

bool
CurrentControlledPort::is_enabled () {
    bool enabled = false;
    amrex::ParmParse const parser("warpx");
    parser.query("current_controlled_port", enabled);
    return enabled;
}

CurrentControlledPort::CurrentControlledPort () {
    amrex::ParmParse const parser("warpx");
    std::string waveform_file;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        parser.query("current_controlled_port.file", waveform_file),
        "warpx.current_controlled_port.file is required.");
    m_waveform.load(waveform_file);
    utils::parser::getWithParser(parser, "current_controlled_port.direction",
                                 m_direction);
    read_terminal(parser, "current_controlled_port.terminal_0", m_terminal_0.lo,
                  m_terminal_0.hi);
    read_terminal(parser, "current_controlled_port.terminal_1", m_terminal_1.lo,
                  m_terminal_1.hi);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_waveform.max_abs_current() > 0.0_rt,
        "The current-controlled-port waveform is identically zero.");
#ifdef WARPX_DIM_3D
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_direction >= 0 && m_direction < 3,
        "warpx.current_controlled_port.direction must be 0, 1, or 2 in 3D.");
#elif defined(WARPX_DIM_XZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_direction == 0 || m_direction == 2,
        "2D XZ current-controlled ports require direction = 0 (x) or 2 (z); "
        "two distinct terminals cannot be represented along the invariant y "
        "direction.");
#elif defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_direction == 2,
        "RZ current-controlled ports currently require direction = 2 (Jz).");
#else
    WARPX_ABORT_WITH_MESSAGE("Current-controlled paired terminals are "
                             "implemented in 2D XZ, 3D, and RZ.");
#endif

    for (int direction = 0; direction < 3; ++direction) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(m_terminal_0.lo[direction]) &&
                std::isfinite(m_terminal_0.hi[direction]) &&
                std::isfinite(m_terminal_1.lo[direction]) &&
                std::isfinite(m_terminal_1.hi[direction]),
            "Current-controlled terminal bounds must be finite.");
    }

    amrex::Real const position_0 = m_terminal_0.lo[m_direction];
    amrex::Real const position_1 = m_terminal_1.lo[m_direction];
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_terminal_0.hi[m_direction] == position_0 &&
            m_terminal_1.hi[m_direction] == position_1 &&
            position_0 != position_1,
        "Each terminal must have zero thickness in "
        "current_controlled_port.direction, "
        "and the two terminal positions must differ.");
    m_axis_sign = position_1 > position_0 ? 1.0_rt : -1.0_rt;

    for (int direction = 0; direction < 3; ++direction) {
        if (direction == m_direction) {
            continue;
        }
#ifdef WARPX_DIM_RZ
        if (direction == 1) {
            continue;
        }
#elif defined(WARPX_DIM_XZ)
        if (direction == 1) {
            continue;
        }
#endif
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_terminal_0.hi[direction] > m_terminal_0.lo[direction] &&
                m_terminal_1.hi[direction] > m_terminal_1.lo[direction],
            "Terminal transverse upper bounds must exceed lower bounds.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_terminal_0.lo[direction] == m_terminal_1.lo[direction] &&
                m_terminal_0.hi[direction] == m_terminal_1.hi[direction],
            "The first current-controlled-port implementation requires "
            "congruent "
            "terminal contours.");
    }
}

void
CurrentControlledPort::ValidateGeometry (
    ablastr::fields::VectorField const& Bfield) const {
    WarpX const& warpx = WarpX::GetInstance();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        warpx.maxLevel() == 0,
        "Current-controlled ports currently support one mesh level.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        warpx.getdo_moving_window() == 0,
        "Current-controlled ports do not yet support a moving window.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !warpx.get_load_balance_intervals().isActivated(),
        "Current-controlled ports do not yet support dynamic load balancing.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::Yee ||
            WarpX::electromagnetic_solver_id ==
                ElectromagneticSolverAlgo::HybridPIC,
        "Current-controlled ports currently support Yee and Hybrid-PIC field "
        "solvers.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(WarpX::grid_type == GridType::Staggered ||
                                         WarpX::grid_type ==
                                             GridType::Collocated,
                                     "Current-controlled ports currently "
                                     "support staggered or collocated fields, "
                                     "not the hybrid-staggering option.");
    for (int component = 0; component < 3; ++component) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            Bfield[component]->nComp() == 1,
            "Current-controlled ports currently support one field component "
            "per mode.");
    }

#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    amrex::Geometry const& geometry = warpx.Geom(0);
    std::array<amrex::Real, 3> domain_lo{{0.0_rt, 0.0_rt, 0.0_rt}};
    std::array<amrex::Real, 3> domain_hi{{0.0_rt, 0.0_rt, 0.0_rt}};
#ifdef WARPX_DIM_3D
    for (int direction = 0; direction < 3; ++direction) {
        domain_lo[direction] = geometry.ProbLo(direction);
        domain_hi[direction] = geometry.ProbHi(direction);
    }
#elif defined(WARPX_DIM_XZ)
    domain_lo[0] = geometry.ProbLo(0);
    domain_hi[0] = geometry.ProbHi(0);
    domain_lo[2] = geometry.ProbLo(1);
    domain_hi[2] = geometry.ProbHi(1);
#elif defined(WARPX_DIM_RZ)
    domain_lo[0] = geometry.ProbLo(0);
    domain_hi[0] = geometry.ProbHi(0);
    domain_lo[2] = geometry.ProbLo(1);
    domain_hi[2] = geometry.ProbHi(1);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        WarpX::n_rz_azimuthal_modes == 1,
        "RZ current-controlled ports currently support only the m=0 mode.");
#endif

    for (Terminal const* terminal : {&m_terminal_0, &m_terminal_1}) {
        for (int direction = 0; direction < 3; ++direction) {
#ifdef WARPX_DIM_RZ
            if (direction == 1) {
                continue;
            }
#elif defined(WARPX_DIM_XZ)
            if (direction == 1) {
                continue;
            }
#endif
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                terminal->lo[direction] >= domain_lo[direction] &&
                    terminal->hi[direction] <= domain_hi[direction],
                "Current-controlled terminal bounds must lie inside the "
                "simulation domain.");
        }
    }
#endif
}

void
CurrentControlledPort::InitData (
    ablastr::fields::VectorField const& Bfield,
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update_B,
    amrex::Real const time) {
    ValidateGeometry(Bfield);
    amrex::Geometry const& geometry = WarpX::GetInstance().Geom(0);
    for (int component = 0; component < 3; ++component) {
        m_owner_mask[component] =
            Bfield[component]->OwnerMask(geometry.periodicity());
    }
#ifdef WARPX_DIM_3D
    BuildCurlBasis(Bfield, eb_update_B);
#endif
    m_initialized = true;
    ApplyBfield(Bfield, eb_update_B, 0, PatchType::fine, time);
}

std::array<amrex::Real, 3>
CurrentControlledPort::status () const {
    return {m_last_target, m_last_terminal_current[0],
            m_last_terminal_current[1]};
}

#ifdef WARPX_DIM_3D
std::array<CurrentControlledPort::Edge, 4>
CurrentControlledPort::Edges () const {
    int const tangent_a = (m_direction + 1) % 3;
    int const tangent_b = (m_direction + 2) % 3;
    amrex::Geometry const& geometry = WarpX::GetInstance().Geom(0);
    amrex::Real const offset_a = WarpX::grid_type == GridType::Staggered
                                     ? 0.5_rt * geometry.CellSize(tangent_a)
                                     : 0.0_rt;
    amrex::Real const offset_b = WarpX::grid_type == GridType::Staggered
                                     ? 0.5_rt * geometry.CellSize(tangent_b)
                                     : 0.0_rt;
    return {{{tangent_a, tangent_a, tangent_b,
              m_terminal_0.lo[tangent_b] - offset_b, m_terminal_0.lo[tangent_a],
              m_terminal_0.hi[tangent_a], 1.0_rt},
             {tangent_b, tangent_b, tangent_a,
              m_terminal_0.hi[tangent_a] + offset_a, m_terminal_0.lo[tangent_b],
              m_terminal_0.hi[tangent_b], 1.0_rt},
             {tangent_a, tangent_a, tangent_b,
              m_terminal_0.hi[tangent_b] + offset_b, m_terminal_0.lo[tangent_a],
              m_terminal_0.hi[tangent_a], -1.0_rt},
             {tangent_b, tangent_b, tangent_a,
              m_terminal_0.lo[tangent_a] - offset_a, m_terminal_0.lo[tangent_b],
              m_terminal_0.hi[tangent_b], -1.0_rt}}};
}

void
CurrentControlledPort::BuildCurlBasis (
    ablastr::fields::VectorField const& Bfield,
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update_B) {
    WarpX& warpx = WarpX::GetInstance();
    amrex::BoxArray const& cell_boxes = warpx.boxArray(0);
    amrex::DistributionMapping const& distribution = warpx.DistributionMap(0);
    amrex::IntVect const guards(1);

    std::array<std::unique_ptr<amrex::MultiFab>, 3> potential;
    for (int component = 0; component < 3; ++component) {
        amrex::IntVect staggering = amrex::IntVect::TheNodeVector();
        if (WarpX::grid_type == GridType::Staggered) {
            staggering[component] = 0;
        }
        potential[component] = std::make_unique<amrex::MultiFab>(
            amrex::convert(cell_boxes, staggering), distribution, 1, guards);
        potential[component]->setVal(0.0_rt);

        m_curl_basis[component] = std::make_unique<amrex::MultiFab>(
            Bfield[component]->boxArray(), Bfield[component]->DistributionMap(),
            1, Bfield[component]->nGrowVect());
        m_curl_basis[component]->setVal(0.0_rt);
    }

    amrex::MultiFab& axial_potential = *potential[m_direction];
    amrex::Geometry const& geometry = warpx.Geom(0);
    auto const prob_lo = geometry.ProbLoArray();
    auto const cell_size = geometry.CellSizeArray();
    amrex::IntVect const domain_lo = geometry.Domain().smallEnd();
    amrex::GpuArray<amrex::Real, 3> const transverse_lo{
        m_terminal_0.lo[0], m_terminal_0.lo[1], m_terminal_0.lo[2]};
    amrex::GpuArray<amrex::Real, 3> const transverse_hi{
        m_terminal_0.hi[0], m_terminal_0.hi[1], m_terminal_0.hi[2]};
    amrex::IntVect const index_type = axial_potential.ixType().toIntVect();
    int const axis = m_direction;

    for (amrex::MFIter mfi(axial_potential); mfi.isValid(); ++mfi) {
        amrex::Box const& box = mfi.validbox();
        auto const values = axial_potential.array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            int const index[3] = {i, j, k};
            bool inside = true;
            for (int direction = 0; direction < 3; ++direction) {
                if (direction == axis) {
                    continue;
                }
                amrex::Real const offset =
                    index_type[direction] == 1 ? 0.0_rt : 0.5_rt;
                amrex::Real const coordinate =
                    prob_lo[direction] +
                    (index[direction] - domain_lo[direction] + offset) *
                        cell_size[direction];
                inside = inside && coordinate >= transverse_lo[direction] &&
                         coordinate <= transverse_hi[direction];
            }
            if (inside) {
                values(i, j, k) = 1.0_rt;
            }
        });
    }
    for (auto& component : potential) {
        component->FillBoundary(geometry.periodicity());
    }

    ablastr::fields::VectorField const potential_field{
        potential[0].get(), potential[1].get(), potential[2].get()};
    ablastr::fields::VectorField basis_field{
        m_curl_basis[0].get(), m_curl_basis[1].get(), m_curl_basis[2].get()};
    warpx.get_pointer_fdtd_solver_fp(0)->ComputeCurlA(
        basis_field, potential_field, eb_update_B, 0);
    for (auto& component : m_curl_basis) {
        component->FillBoundary(geometry.periodicity());
    }
}

amrex::Real
CurrentControlledPort::MeasureEdgeLocal (amrex::MultiFab const& field,
                                         amrex::iMultiFab const& owner_mask,
                                         amrex::iMultiFab const* const eb_flag,
                                         Edge const& edge,
                                         int const plane_index) const {
    amrex::Geometry const& geometry = WarpX::GetInstance().Geom(0);
    amrex::Box line = amrex::convert(geometry.Domain(), field.ixType());
    int const fixed_index =
        nearest_index(field, edge.fixed_direction, edge.fixed_coordinate);
    auto const tangent_range = index_range(field, edge.tangent_direction,
                                           edge.tangent_lo, edge.tangent_hi);
    line.setSmall(m_direction, plane_index);
    line.setBig(m_direction, plane_index);
    line.setSmall(edge.fixed_direction, fixed_index);
    line.setBig(edge.fixed_direction, fixed_index);
    line.setSmall(edge.tangent_direction, tangent_range[0]);
    line.setBig(edge.tangent_direction, tangent_range[1]);

    amrex::Real circulation = 0.0_rt;
    amrex::Real const spacing = geometry.CellSize(edge.tangent_direction);
    for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
        amrex::Box const section = line & mfi.validbox();
        if (!section.ok()) {
            continue;
        }
        auto const values = field.const_array(mfi);
        auto const owners = owner_mask.const_array(mfi);
        amrex::Array4<int const> flags;
        if (eb_flag != nullptr) {
            flags = eb_flag->const_array(mfi);
        }
        amrex::Real const orientation = edge.orientation;
        amrex::ReduceOps<amrex::ReduceOpSum> reduce_ops;
        amrex::ReduceData<amrex::Real> reduce_data(reduce_ops);
        using ReduceTuple = decltype(reduce_data)::Type;
        reduce_ops.eval(
            section, reduce_data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple {
                if (owners(i, j, k) == 0 || (flags && flags(i, j, k) == 0)) {
                    return {0.0_rt};
                }
                return {orientation * values(i, j, k) * spacing};
            });
        auto const reduced = reduce_data.value();
        circulation += amrex::get<0>(reduced);
    }
    return circulation;
}

void
CurrentControlledPort::AddScaledCurlBasis (
    ablastr::fields::VectorField const& Bfield, int const plane_index,
    amrex::Real const scale) const {
    for (int component = 0; component < 3; ++component) {
        if (component == m_direction) {
            continue;
        }
        amrex::MultiFab& field = *Bfield[component];
        amrex::MultiFab const& basis = *m_curl_basis[component];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
            amrex::Box section = mfi.validbox();
            section.setSmall(m_direction, plane_index);
            section.setBig(m_direction, plane_index);
            section &= mfi.validbox();
            if (!section.ok()) {
                continue;
            }
            auto const values = field.array(mfi);
            auto const basis_values = basis.const_array(mfi);
            amrex::ParallelFor(
                section, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    values(i, j, k) += scale * basis_values(i, j, k);
                });
        }
    }
}

amrex::Real
CurrentControlledPort::MeasureContourLocal (
    ablastr::fields::VectorField const& Bfield,
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update_B,
    int const plane_index) const {
    amrex::Real result = 0.0_rt;
    for (Edge const& edge : Edges()) {
        result += MeasureEdgeLocal(
            *Bfield[edge.component], *m_owner_mask[edge.component],
            eb_update_B[edge.component].get(), edge, plane_index);
    }
    return result;
}
#endif

#ifdef WARPX_DIM_XZ
std::array<amrex::Real, 3>
CurrentControlledPort::MeasureXZContourLocal (
    amrex::MultiFab const& by, amrex::iMultiFab const& owner_mask,
    amrex::iMultiFab const* const eb_flag, int const plane_index) const {
    int const transverse_direction = m_direction == 0 ? 2 : 0;
    int const axial_mesh_direction = m_direction == 0 ? 0 : 1;
    int const transverse_mesh_direction = m_direction == 0 ? 1 : 0;
    int const lower_index = nearest_index(
        by, transverse_mesh_direction, m_terminal_0.lo[transverse_direction]);
    int const upper_index = nearest_index(
        by, transverse_mesh_direction, m_terminal_0.hi[transverse_direction]);
    amrex::Real const upper_weight = m_direction == 2 ? 1.0_rt : -1.0_rt;

    amrex::Real circulation = 0.0_rt;
    amrex::Real response_norm = 0.0_rt;
    amrex::Real active_points = 0.0_rt;
    for (amrex::MFIter mfi(by); mfi.isValid(); ++mfi) {
        auto const values = by.const_array(mfi);
        auto const owners = owner_mask.const_array(mfi);
        amrex::Array4<int const> flags;
        if (eb_flag != nullptr) {
            flags = eb_flag->const_array(mfi);
        }
        for (int side = 0; side < 2; ++side) {
            int const transverse_index = side == 0 ? lower_index : upper_index;
            amrex::IntVect point;
            point[axial_mesh_direction] = plane_index;
            point[transverse_mesh_direction] = transverse_index;
            if (!mfi.validbox().contains(point)) {
                continue;
            }
            amrex::Real const weight = side == 0 ? -upper_weight : upper_weight;
            amrex::Box const point_box(point, point, by.ixType());
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum,
                             amrex::ReduceOpSum>
                reduce_ops;
            amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real>
                reduce_data(reduce_ops);
            using ReduceTuple = decltype(reduce_data)::Type;
            reduce_ops.eval(
                point_box, reduce_data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple {
                    if (owners(i, j, k) == 0 ||
                        (flags && flags(i, j, k) == 0)) {
                        return {0.0_rt, 0.0_rt, 0.0_rt};
                    }
                    return {weight * values(i, j, k), weight * weight, 1.0_rt};
                });
            auto const reduced = reduce_data.value();
            circulation += amrex::get<0>(reduced);
            response_norm += amrex::get<1>(reduced);
            active_points += amrex::get<2>(reduced);
        }
    }
    std::array<amrex::Real, 3> result{
        {circulation, response_norm, active_points}};
    return result;
}

void
CurrentControlledPort::CorrectXZContour (
    amrex::MultiFab& by, amrex::iMultiFab const* const eb_flag,
    int const plane_index, amrex::Real const current_error,
    std::array<amrex::Real, 3> const& contour) const {
    int const transverse_direction = m_direction == 0 ? 2 : 0;
    int const axial_mesh_direction = m_direction == 0 ? 0 : 1;
    int const transverse_mesh_direction = m_direction == 0 ? 1 : 0;
    int const lower_index = nearest_index(
        by, transverse_mesh_direction, m_terminal_0.lo[transverse_direction]);
    int const upper_index = nearest_index(
        by, transverse_mesh_direction, m_terminal_0.hi[transverse_direction]);
    amrex::Real const upper_weight = m_direction == 2 ? 1.0_rt : -1.0_rt;
    amrex::Real const scale = PhysConst::mu0 * current_error / contour[1];

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(by); mfi.isValid(); ++mfi) {
        auto const values = by.array(mfi);
        amrex::Array4<int const> flags;
        if (eb_flag != nullptr) {
            flags = eb_flag->const_array(mfi);
        }
        for (int side = 0; side < 2; ++side) {
            int const transverse_index = side == 0 ? lower_index : upper_index;
            amrex::IntVect point;
            point[axial_mesh_direction] = plane_index;
            point[transverse_mesh_direction] = transverse_index;
            if (!mfi.validbox().contains(point)) {
                continue;
            }
            amrex::Real const weight = side == 0 ? -upper_weight : upper_weight;
            amrex::Box const point_box(point, point, by.ixType());
            amrex::ParallelFor(point_box,
                               [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                                   if (flags && flags(i, j, k) == 0) {
                                       return;
                                   }
                                   values(i, j, k) += scale * weight;
                               });
        }
    }
}
#endif

#ifdef WARPX_DIM_RZ
std::array<amrex::Real, 3>
CurrentControlledPort::MeasureRZContourLocal (
    amrex::MultiFab const& btheta, amrex::iMultiFab const& owner_mask,
    amrex::iMultiFab const* const eb_flag, int const plane_index) const {
    amrex::Real const requested_inner = m_terminal_0.lo[0];
    bool const has_inner =
        requested_inner > WarpX::GetInstance().Geom(0).ProbLo(0);
    int const inner_index =
        has_inner ? nearest_index(btheta, 0, requested_inner) : 0;
    int const outer_index = nearest_index(btheta, 0, m_terminal_0.hi[0]);
    amrex::Real const inner_radius =
        has_inner ? index_coordinate(btheta, 0, inner_index) : 0.0_rt;
    amrex::Real const outer_radius = index_coordinate(btheta, 0, outer_index);

    amrex::Real circulation = 0.0_rt;
    amrex::Real response_norm = 0.0_rt;
    amrex::Real active_points = 0.0_rt;
    for (amrex::MFIter mfi(btheta); mfi.isValid(); ++mfi) {
        auto const values = btheta.const_array(mfi);
        auto const owners = owner_mask.const_array(mfi);
        amrex::Array4<int const> flags;
        if (eb_flag != nullptr) {
            flags = eb_flag->const_array(mfi);
        }
        for (int side = 0; side < 2; ++side) {
            if (side == 0 && !has_inner) {
                continue;
            }
            int const radial_index = side == 0 ? inner_index : outer_index;
            amrex::IntVect const point(
                AMREX_D_DECL(radial_index, plane_index, 0));
            if (!mfi.validbox().contains(point)) {
                continue;
            }
            amrex::Real const radius = side == 0 ? inner_radius : outer_radius;
            amrex::Real const weight = (side == 0 ? -1.0_rt : 1.0_rt) * 2.0_rt *
                                       MathConst::pi * radius;
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum,
                             amrex::ReduceOpSum>
                reduce_ops;
            amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real>
                reduce_data(reduce_ops);
            using ReduceTuple = decltype(reduce_data)::Type;
            amrex::Box const point_box(point, point, btheta.ixType());
            reduce_ops.eval(
                point_box, reduce_data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple {
                    if (owners(i, j, k) == 0 ||
                        (flags && flags(i, j, k) == 0)) {
                        return {0.0_rt, 0.0_rt, 0.0_rt};
                    }
                    return {weight * values(i, j, k), weight * weight, 1.0_rt};
                });
            auto const reduced = reduce_data.value();
            circulation += amrex::get<0>(reduced);
            response_norm += amrex::get<1>(reduced);
            active_points += amrex::get<2>(reduced);
        }
    }
    std::array<amrex::Real, 3> result{
        {circulation, response_norm, active_points}};
    return result;
}

void
CurrentControlledPort::CorrectRZContour (
    amrex::MultiFab& btheta, amrex::iMultiFab const* const eb_flag,
    int const plane_index, amrex::Real const current_error,
    std::array<amrex::Real, 3> const& contour) const {
    amrex::Real const requested_inner = m_terminal_0.lo[0];
    bool const has_inner =
        requested_inner > WarpX::GetInstance().Geom(0).ProbLo(0);
    int const inner_index =
        has_inner ? nearest_index(btheta, 0, requested_inner) : 0;
    int const outer_index = nearest_index(btheta, 0, m_terminal_0.hi[0]);
    amrex::Real const inner_radius =
        has_inner ? index_coordinate(btheta, 0, inner_index) : 0.0_rt;
    amrex::Real const outer_radius = index_coordinate(btheta, 0, outer_index);
    amrex::Real const scale = PhysConst::mu0 * current_error / contour[1];

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(btheta); mfi.isValid(); ++mfi) {
        auto const values = btheta.array(mfi);
        amrex::Array4<int const> flags;
        if (eb_flag != nullptr) {
            flags = eb_flag->const_array(mfi);
        }
        for (int side = 0; side < 2; ++side) {
            if (side == 0 && !has_inner) {
                continue;
            }
            int const radial_index = side == 0 ? inner_index : outer_index;
            amrex::IntVect const point(
                AMREX_D_DECL(radial_index, plane_index, 0));
            if (!mfi.validbox().contains(point)) {
                continue;
            }
            amrex::Real const radius = side == 0 ? inner_radius : outer_radius;
            amrex::Real const weight = (side == 0 ? -1.0_rt : 1.0_rt) * 2.0_rt *
                                       MathConst::pi * radius;
            amrex::Box const point_box(point, point, btheta.ixType());
            amrex::ParallelFor(point_box,
                               [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                                   if (flags && flags(i, j, k) == 0) {
                                       return;
                                   }
                                   values(i, j, k) += scale * weight;
                               });
        }
    }
}
#endif

void
CurrentControlledPort::ApplyBfield (
    ablastr::fields::VectorField const& Bfield,
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update_B,
    int const lev, PatchType const patch_type, amrex::Real const time) {
    if (!m_initialized || lev != 0 || patch_type != PatchType::fine) {
        return;
    }

    amrex::Real const requested_current = m_waveform.value(time);
    amrex::Real const target_axis_current = m_axis_sign * requested_current;
    m_last_target = requested_current;

#ifdef WARPX_DIM_3D
    auto const edges = Edges();
    amrex::MultiFab const& reference_field = *Bfield[edges[0].component];
    int const plane_0 = nearest_index(reference_field, m_direction,
                                      m_terminal_0.lo[m_direction]);
    int const plane_1 = nearest_index(reference_field, m_direction,
                                      m_terminal_1.lo[m_direction]);
    int const first_plane = std::min(plane_0, plane_1);
    int const last_plane = std::max(plane_0, plane_1);
    int const number_of_planes = last_plane - first_plane + 1;
    ablastr::fields::VectorField const basis_field{
        m_curl_basis[0].get(), m_curl_basis[1].get(), m_curl_basis[2].get()};
    std::vector<amrex::Real> contour_data(2 * number_of_planes, 0.0_rt);
    for (int offset = 0; offset < number_of_planes; ++offset) {
        int const plane = first_plane + offset;
        contour_data[2 * offset] =
            MeasureContourLocal(Bfield, eb_update_B, plane);
        contour_data[2 * offset + 1] =
            MeasureContourLocal(basis_field, eb_update_B, plane);
    }
    amrex::ParallelDescriptor::ReduceRealSum(
        contour_data.data(), static_cast<int>(contour_data.size()));

    amrex::Real const desired_circulation =
        PhysConst::mu0 * target_axis_current;
    for (int offset = 0; offset < number_of_planes; ++offset) {
        amrex::Real const circulation = contour_data[2 * offset];
        amrex::Real const basis_response = contour_data[2 * offset + 1];
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::abs(basis_response) > 0.0_rt,
            "The current-controlled port contour has no solver-active curl "
            "basis; "
            "place its contour outside covered EB cells and away from the "
            "domain edge.");
        AddScaledCurlBasis(Bfield, first_plane + offset,
                           (desired_circulation - circulation) /
                               basis_response);
    }
    std::array<amrex::Real, 2> terminal_circulation{
        MeasureContourLocal(Bfield, eb_update_B, plane_0),
        MeasureContourLocal(Bfield, eb_update_B, plane_1)};
    amrex::ParallelDescriptor::ReduceRealSum(terminal_circulation.data(), 2);
    for (int terminal = 0; terminal < 2; ++terminal) {
        m_last_terminal_current[terminal] =
            m_axis_sign * terminal_circulation[terminal] / PhysConst::mu0;
    }
#elif defined(WARPX_DIM_XZ)
    amrex::MultiFab& by = *Bfield[1];
    int const axial_mesh_direction = m_direction == 0 ? 0 : 1;
    int const plane_0 =
        nearest_index(by, axial_mesh_direction, m_terminal_0.lo[m_direction]);
    int const plane_1 =
        nearest_index(by, axial_mesh_direction, m_terminal_1.lo[m_direction]);
    int const first_plane = std::min(plane_0, plane_1);
    int const last_plane = std::max(plane_0, plane_1);
    int const number_of_planes = last_plane - first_plane + 1;
    std::vector<amrex::Real> contour_data(3 * number_of_planes, 0.0_rt);
    for (int offset = 0; offset < number_of_planes; ++offset) {
        auto const contour = MeasureXZContourLocal(
            by, *m_owner_mask[1], eb_update_B[1].get(), first_plane + offset);
        for (int component = 0; component < 3; ++component) {
            contour_data[3 * offset + component] = contour[component];
        }
    }
    amrex::ParallelDescriptor::ReduceRealSum(
        contour_data.data(), static_cast<int>(contour_data.size()));
    for (int offset = 0; offset < number_of_planes; ++offset) {
        std::array<amrex::Real, 3> contour{contour_data[3 * offset],
                                           contour_data[3 * offset + 1],
                                           contour_data[3 * offset + 2]};
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            contour[1] > 0.0_rt && contour[2] > 0.0_rt,
            "The 2D XZ current-controlled contour has no active field points; "
            "place its endpoints outside covered EB cells.");
        amrex::Real const measured_current = contour[0] / PhysConst::mu0;
        CorrectXZContour(by, eb_update_B[1].get(), first_plane + offset,
                         target_axis_current - measured_current, contour);
    }
    std::array<amrex::Real, 2> terminal_circulation{
        MeasureXZContourLocal(by, *m_owner_mask[1], eb_update_B[1].get(),
                              plane_0)[0],
        MeasureXZContourLocal(by, *m_owner_mask[1], eb_update_B[1].get(),
                              plane_1)[0]};
    amrex::ParallelDescriptor::ReduceRealSum(terminal_circulation.data(), 2);
    for (int terminal = 0; terminal < 2; ++terminal) {
        m_last_terminal_current[terminal] =
            m_axis_sign * terminal_circulation[terminal] / PhysConst::mu0;
    }
#elif defined(WARPX_DIM_RZ)
    amrex::MultiFab& btheta = *Bfield[1];
    int const plane_0 = nearest_index(btheta, 1, m_terminal_0.lo[2]);
    int const plane_1 = nearest_index(btheta, 1, m_terminal_1.lo[2]);
    int const first_plane = std::min(plane_0, plane_1);
    int const last_plane = std::max(plane_0, plane_1);
    int const number_of_planes = last_plane - first_plane + 1;
    std::vector<amrex::Real> contour_data(3 * number_of_planes, 0.0_rt);
    for (int offset = 0; offset < number_of_planes; ++offset) {
        auto const contour =
            MeasureRZContourLocal(btheta, *m_owner_mask[1],
                                  eb_update_B[1].get(), first_plane + offset);
        for (int component = 0; component < 3; ++component) {
            contour_data[3 * offset + component] = contour[component];
        }
    }
    amrex::ParallelDescriptor::ReduceRealSum(
        contour_data.data(), static_cast<int>(contour_data.size()));
    for (int offset = 0; offset < number_of_planes; ++offset) {
        std::array<amrex::Real, 3> contour{contour_data[3 * offset],
                                           contour_data[3 * offset + 1],
                                           contour_data[3 * offset + 2]};
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            contour[1] > 0.0_rt && contour[2] > 0.0_rt,
            "The RZ current-controlled contour has no active field points; "
            "place its radial contour outside covered EB cells.");
        amrex::Real const measured_current = contour[0] / PhysConst::mu0;
        CorrectRZContour(btheta, eb_update_B[1].get(), first_plane + offset,
                         target_axis_current - measured_current, contour);
    }
    std::array<amrex::Real, 2> terminal_circulation{
        MeasureRZContourLocal(btheta, *m_owner_mask[1], eb_update_B[1].get(),
                              plane_0)[0],
        MeasureRZContourLocal(btheta, *m_owner_mask[1], eb_update_B[1].get(),
                              plane_1)[0]};
    amrex::ParallelDescriptor::ReduceRealSum(terminal_circulation.data(), 2);
    for (int terminal = 0; terminal < 2; ++terminal) {
        m_last_terminal_current[terminal] =
            m_axis_sign * terminal_circulation[terminal] / PhysConst::mu0;
    }
#else
    amrex::ignore_unused(Bfield, eb_update_B, target_axis_current);
#endif
}

void
WarpX::ApplyCurrentControlledPort (int const lev, PatchType const patch_type,
                                   amrex::Real const time) {
    if (!m_current_controlled_port || patch_type != PatchType::fine) {
        return;
    }
    m_current_controlled_port->ApplyBfield(
        m_fields.get_alldirs(warpx::fields::FieldType::Bfield_fp, lev),
        m_eb_update_B[lev], lev, patch_type, time);
}
