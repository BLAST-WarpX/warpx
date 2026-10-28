/* Copyright 2026 WarpX contributors
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "CurrentControlledPort.H"

#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/CompositeCurrentControlledPort.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuAtomic.H>
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
[[nodiscard]] bool
nearly_equal (amrex::Real const lhs, amrex::Real const rhs) {
    amrex::Real const scale = std::max(std::abs(lhs), std::abs(rhs));
    return std::abs(lhs - rhs) <=
           64.0_rt * std::numeric_limits<amrex::Real>::epsilon() * scale;
}

#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
[[nodiscard]] int
nearest_index (amrex::Geometry const& geometry, amrex::MultiFab const& field, int const direction,
               amrex::Real const coordinate)
{
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
index_range (amrex::Geometry const& geometry, amrex::MultiFab const& field, int const direction,
             amrex::Real const lower, amrex::Real const upper)
{
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
index_coordinate (amrex::Geometry const& geometry, amrex::MultiFab const& field,
                  int const direction, int const index)
{
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

CurrentControlledPort::CurrentControlledPort (std::string parameter_prefix, int const level,
                                              PatchType const patch_type)
    : m_parameter_prefix(parameter_prefix), m_level(level), m_patch_type(patch_type)
{
    amrex::ParmParse const parser("warpx");
    auto const key = [&parameter_prefix] (std::string const& suffix) {
        return parameter_prefix + "." + suffix;
    };
    std::string waveform_file;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        parser.query(key("file"), waveform_file),
        "Each current-controlled port requires a waveform file.");
    m_waveform.load(waveform_file);
    utils::parser::getWithParser(parser, key("direction").c_str(), m_direction);
    utils::parser::queryWithParser(parser, key("current_scale").c_str(),
                                   m_current_scale);
    read_terminal(parser, key("terminal_0"), m_terminal_0.lo, m_terminal_0.hi);
    read_terminal(parser, key("terminal_1"), m_terminal_1.lo, m_terminal_1.hi);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_waveform.max_abs_current() > 0.0_rt,
        "The current-controlled-port waveform is identically zero.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::isfinite(m_current_scale) && m_current_scale > 0.0_rt,
        "Current-controlled-port current_scale must be finite and positive.");
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
        m_direction == 0 || m_direction == 2,
        "RZ current-controlled ports require direction = 0 (Jr) or 2 (Jz); "
        "azimuthal terminals cannot be represented in axisymmetry.");
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
        nearly_equal(m_terminal_0.hi[m_direction], position_0) &&
            nearly_equal(m_terminal_1.hi[m_direction], position_1) &&
            !nearly_equal(position_0, position_1),
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
            nearly_equal(m_terminal_0.lo[direction],
                         m_terminal_1.lo[direction]) &&
                nearly_equal(m_terminal_0.hi[direction],
                             m_terminal_1.hi[direction]),
            "Current-controlled ports require matching terminal transverse "
            "bounds.");
    }
}

amrex::Geometry const&
CurrentControlledPort::Geometry () const
{
    return WarpX::GetInstance().Geom(m_patch_type == PatchType::fine ? m_level : m_level - 1);
}

CurrentControlledPort::~CurrentControlledPort () = default;

amrex::Array4<amrex::Real const>
CurrentControlledPort::Permeability (amrex::MFIter const& mfi) const
{
    auto* const properties =
        WarpX::GetInstance().get_pointer_MacroscopicProperties(m_level, m_patch_type);
    return properties == nullptr ? amrex::Array4<amrex::Real const>{}
                                 : properties->getmu_mf().const_array(mfi);
}

bool
CurrentControlledPort::PatchContainsPort (ablastr::fields::VectorField const& Bfield) const
{
    if (m_level == 0) {
        return true;
    }
    auto const& geometry = Geometry();
    amrex::IntVect lower, upper;
    for (int mesh_direction = 0; mesh_direction < AMREX_SPACEDIM; ++mesh_direction) {
#ifdef WARPX_DIM_3D
        int const direction = mesh_direction;
#else
        int const direction = mesh_direction == 0 ? 0 : 2;
#endif
        amrex::Real const lo = std::min(m_terminal_0.lo[direction], m_terminal_1.lo[direction]);
        amrex::Real const hi = std::max(m_terminal_0.hi[direction], m_terminal_1.hi[direction]);
        lower[mesh_direction] = static_cast<int>(std::floor((lo - geometry.ProbLo(mesh_direction)) /
                                                            geometry.CellSize(mesh_direction))) +
                                geometry.Domain().smallEnd(mesh_direction);
        upper[mesh_direction] = static_cast<int>(std::ceil((hi - geometry.ProbLo(mesh_direction)) /
                                                           geometry.CellSize(mesh_direction))) +
                                geometry.Domain().smallEnd(mesh_direction);
    }
    amrex::Box const region(lower, upper);
    auto const boxes = amrex::convert(Bfield[0]->boxArray(), amrex::IntVect::TheCellVector());
    auto const required = amrex::grow(region, 2) & geometry.Domain();
    if (!boxes.intersects(required)) {
        return false;
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        boxes.contains(required),
        "An SMR current-controlled port must be wholly inside or outside each refined patch, "
        "with two coarse cells of clearance. "
        "Port contours crossing refinement boundaries are not supported.");
    return true;
}

void
CurrentControlledPort::ValidateGeometry (
    ablastr::fields::VectorField const& Bfield) const {
    // Patch contours share a physical time, and their geometry is cached at initialization.
    // Subcycling, moving windows, and load balancing would invalidate those assumptions.
    WarpX const& warpx = WarpX::GetInstance();
    bool subcycling = false;
    amrex::ParmParse("warpx").query("do_subcycling", subcycling);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        warpx.maxLevel() == 0 || ((warpx.evolve_scheme == EvolveScheme::Explicit ||
                                   warpx.evolve_scheme == EvolveScheme::Theta_Implicit_EM) &&
                                  !subcycling),
        "SMR current-controlled ports require explicit or theta-implicit evolution and "
        "synchronized timesteps.");
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
    // Translate each geometry's mesh coordinates to the physical x/y/z terminal bounds.
    // In RZ the physical x coordinate is radius, and the azimuthal bounds are omitted.
    amrex::Geometry const& geometry = Geometry();
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

#ifdef WARPX_DIM_RZ
    if (m_direction == 0) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_terminal_0.lo[0] > domain_lo[0] &&
                m_terminal_1.lo[0] > domain_lo[0],
            "Radial RZ current-controlled terminals must have positive "
            "radius.");
    }
#endif

#ifdef WARPX_DIM_3D
    amrex::MultiFab const& terminal_field = *Bfield[(m_direction + 1) % 3];
    int const terminal_mesh_direction = m_direction;
#elif defined(WARPX_DIM_XZ)
    amrex::MultiFab const& terminal_field = *Bfield[1];
    int const terminal_mesh_direction = m_direction == 0 ? 0 : 1;
#elif defined(WARPX_DIM_RZ)
    amrex::MultiFab const& terminal_field = *Bfield[1];
    int const terminal_mesh_direction = m_direction == 0 ? 0 : 1;
#endif
    // Distinct physical positions can round onto the same native field plane.
    // Reject that degenerate contour before constructing its response.
    int const terminal_plane_0 = nearest_index(Geometry(), terminal_field, terminal_mesh_direction,
                                               m_terminal_0.lo[m_direction]);
    int const terminal_plane_1 = nearest_index(Geometry(), terminal_field, terminal_mesh_direction,
                                               m_terminal_1.lo[m_direction]);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        terminal_plane_0 != terminal_plane_1,
        "The two current-controlled terminals must map to distinct field "
        "planes.");
#endif
}

void
CurrentControlledPort::InitData (
    ablastr::fields::VectorField const& Bfield,
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update_B, amrex::Real const time,
    int const lev, PatchType const patch_type)
{
    if (m_composite != nullptr) {
        return;
    }
    if (lev == 0 && patch_type == PatchType::fine) {
        ValidateGeometry(Bfield);
        auto const& warpx = WarpX::GetInstance();
        for (int refined_level = 1; refined_level <= warpx.finestLevel();
             ++refined_level) {
            for (auto const patch : {PatchType::fine, PatchType::coarse}) {
                CurrentControlledPort const candidate(m_parameter_prefix,
                                                refined_level, patch);
                auto const& geometry = candidate.Geometry();
                amrex::IntVect lower, upper;
                for (int direction = 0; direction < AMREX_SPACEDIM;
                     ++direction) {
#ifdef WARPX_DIM_3D
                    int const physical = direction;
#else
                    int const physical = direction == 0 ? 0 : 2;
#endif
                    amrex::Real const lo = std::min(m_terminal_0.lo[physical],
                                                    m_terminal_1.lo[physical]);
                    amrex::Real const hi = std::max(m_terminal_0.hi[physical],
                                                    m_terminal_1.hi[physical]);
                    lower[direction] = geometry.Domain().smallEnd(direction) +
                                       static_cast<int>(std::floor(
                                           (lo - geometry.ProbLo(direction)) /
                                           geometry.CellSize(direction)));
                    upper[direction] = geometry.Domain().smallEnd(direction) +
                                       static_cast<int>(std::ceil(
                                           (hi - geometry.ProbLo(direction)) /
                                           geometry.CellSize(direction)));
                }
                auto const required = amrex::grow(amrex::Box(lower, upper), 2) &
                                      geometry.Domain();
                auto const boxes = amrex::convert(
                    warpx.m_fields
                        .get(patch == PatchType::fine
                                 ? warpx::fields::FieldType::Bfield_fp
                                 : warpx::fields::FieldType::Bfield_cp,
                             ablastr::fields::Direction{0}, refined_level)
                        ->boxArray(),
                    amrex::IntVect::TheCellVector());
                if (boxes.intersects(required) && !boxes.contains(required)) {
                    m_composite =
                        std::make_unique<CompositeCurrentControlledPort>(
                            m_terminal_0.lo, m_terminal_0.hi, m_direction,
                            m_terminal_0.lo[m_direction],
                            m_terminal_1.lo[m_direction]);
                    m_initialized = true;
                    return;
                }
            }
        }
    }
    if (lev != m_level || patch_type != m_patch_type) {
        auto patch = std::make_unique<CurrentControlledPort>(m_parameter_prefix, lev, patch_type);
        patch->InitData(Bfield, eb_update_B, time, lev, patch_type);
        m_refined_patches.push_back(std::move(patch));
        return;
    }
    ValidateGeometry(Bfield);
    if (!PatchContainsPort(Bfield)) {
        return;
    }
    amrex::Geometry const& geometry = Geometry();
    for (int component = 0; component < 3; ++component) {
        m_owner_mask[component] =
            Bfield[component]->OwnerMask(geometry.periodicity());
    }
#ifdef WARPX_DIM_3D
    BuildCurlBasis(Bfield, eb_update_B);
#endif
    m_initialized = true;
    ApplyBfield(Bfield, eb_update_B, lev, patch_type, time);
}

std::array<amrex::Real, 3>
CurrentControlledPort::status () const {
    return {m_last_target, m_last_terminal_current[0],
            m_last_terminal_current[1]};
}

bool
CurrentControlledPort::correction_region_overlaps (
    CurrentControlledPort const& other,
    std::array<amrex::Real, 3> const& minimum_gap) const {
    auto const bounds = [] (CurrentControlledPort const& port) {
        std::array<amrex::Real, 3> lo = port.m_terminal_0.lo;
        std::array<amrex::Real, 3> hi = port.m_terminal_0.hi;
        lo[port.m_direction] = std::min(port.m_terminal_0.lo[port.m_direction],
                                        port.m_terminal_1.lo[port.m_direction]);
        hi[port.m_direction] = std::max(port.m_terminal_0.lo[port.m_direction],
                                        port.m_terminal_1.lo[port.m_direction]);
        return std::pair{lo, hi};
    };

    auto const [this_lo, this_hi] = bounds(*this);
    auto const [other_lo, other_hi] = bounds(other);
    for (int direction = 0; direction < 3; ++direction) {
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        if (direction == 1) {
            continue;
        }
#endif
        bool const separated =
            (this_hi[direction] + minimum_gap[direction] <
                 other_lo[direction] &&
             !nearly_equal(this_hi[direction] + minimum_gap[direction],
                           other_lo[direction])) ||
            (other_hi[direction] + minimum_gap[direction] <
                 this_lo[direction] &&
             !nearly_equal(other_hi[direction] + minimum_gap[direction],
                           this_lo[direction]));
        if (separated) {
            return false;
        }
    }
    return true;
}

#ifdef WARPX_DIM_3D
std::array<CurrentControlledPort::Edge, 4>
CurrentControlledPort::Edges () const {
    int const tangent_a = (m_direction + 1) % 3;
    int const tangent_b = (m_direction + 2) % 3;
    amrex::Geometry const& geometry = Geometry();
    amrex::Real const offset_a = WarpX::grid_type == GridType::Staggered
                                     ? 0.5_rt * geometry.CellSize(tangent_a)
                                     : 0.0_rt;
    amrex::Real const offset_b = WarpX::grid_type == GridType::Staggered
                                     ? 0.5_rt * geometry.CellSize(tangent_b)
                                     : 0.0_rt;
    return {{{tangent_a, tangent_a, tangent_b,
              m_terminal_0.lo[tangent_b] - offset_b, m_terminal_0.lo[tangent_a],
              m_terminal_0.hi[tangent_a], 1.0_rt, false},
             {tangent_b, tangent_b, tangent_a,
              m_terminal_0.hi[tangent_a] + offset_a, m_terminal_0.lo[tangent_b],
              m_terminal_0.hi[tangent_b], 1.0_rt, true},
             {tangent_a, tangent_a, tangent_b,
              m_terminal_0.hi[tangent_b] + offset_b, m_terminal_0.lo[tangent_a],
              m_terminal_0.hi[tangent_a], -1.0_rt, true},
             {tangent_b, tangent_b, tangent_a,
              m_terminal_0.lo[tangent_a] - offset_a, m_terminal_0.lo[tangent_b],
              m_terminal_0.hi[tangent_b], -1.0_rt, false}}};
}

int
CurrentControlledPort::EdgeFixedIndex (amrex::MultiFab const& field,
                                       Edge const& edge) const {
    if (WarpX::grid_type != GridType::Staggered) {
        return nearest_index(Geometry(), field, edge.fixed_direction, edge.fixed_coordinate);
    }
    amrex::Geometry const& geometry = Geometry();
    amrex::Box const field_domain =
        amrex::convert(geometry.Domain(), field.ixType());
    amrex::Real const bound = edge.upper_boundary
                                  ? m_terminal_0.hi[edge.fixed_direction]
                                  : m_terminal_0.lo[edge.fixed_direction];
    amrex::Real const raw = (bound - geometry.ProbLo(edge.fixed_direction)) /
                                geometry.CellSize(edge.fixed_direction) +
                            geometry.Domain().smallEnd(edge.fixed_direction);
    amrex::Real constexpr tolerance =
        64.0_rt * std::numeric_limits<amrex::Real>::epsilon();
    int const index = edge.upper_boundary
                          ? static_cast<int>(std::floor(raw + tolerance))
                          : static_cast<int>(std::ceil(raw - tolerance)) - 1;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        index >= field_domain.smallEnd(edge.fixed_direction) &&
            index <= field_domain.bigEnd(edge.fixed_direction),
        "Current-controlled port contour lies outside the field domain.");
    return index;
}

void
CurrentControlledPort::BuildCurlBasis (
    ablastr::fields::VectorField const& Bfield,
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update_B) {
    WarpX& warpx = WarpX::GetInstance();
    amrex::BoxArray const cell_boxes =
        amrex::convert(Bfield[0]->boxArray(), amrex::IntVect::TheCellVector());
    amrex::DistributionMapping const& distribution = Bfield[0]->DistributionMap();
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
    amrex::Geometry const& geometry = Geometry();
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
    auto* const solver = m_patch_type == PatchType::fine
                             ? warpx.get_pointer_fdtd_solver_fp(m_level)
                             : warpx.get_pointer_fdtd_solver_cp(m_level);
    solver->ComputeCurlA(basis_field, potential_field, eb_update_B, m_level);
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
    amrex::Geometry const& geometry = Geometry();
    amrex::Box line = amrex::convert(geometry.Domain(), field.ixType());
    int const fixed_index = EdgeFixedIndex(field, edge);
    auto const tangent_range =
        index_range(Geometry(), field, edge.tangent_direction, edge.tangent_lo, edge.tangent_hi);
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
        auto const permeability = Permeability(mfi);
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
                return {orientation * values(i, j, k) * spacing *
                        (permeability ? PhysConst::mu0 / permeability(i, j, k) : 1.0_rt)};
            });
        auto const reduced = reduce_data.value();
        circulation += amrex::get<0>(reduced);
    }
    return circulation;
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

std::vector<amrex::Real>
CurrentControlledPort::MeasureContoursLocalBatched (
    ablastr::fields::VectorField const& Bfield,
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update_B,
    int const first_plane, int const last_plane) const {
    int const number_of_planes = last_plane - first_plane + 1;
    amrex::Gpu::DeviceVector<amrex::Real> device_circulation(number_of_planes,
                                                             0.0_rt);
    amrex::Real* const circulation = device_circulation.data();
    amrex::Geometry const& geometry = Geometry();

    for (Edge const& edge : Edges()) {
        amrex::MultiFab const& field = *Bfield[edge.component];
        amrex::iMultiFab const& owner_mask = *m_owner_mask[edge.component];
        amrex::iMultiFab const* const eb_flag =
            eb_update_B[edge.component].get();
        amrex::Box slab = amrex::convert(geometry.Domain(), field.ixType());
        int const fixed_index = EdgeFixedIndex(field, edge);
        auto const tangent_range = index_range(Geometry(), field, edge.tangent_direction,
                                               edge.tangent_lo, edge.tangent_hi);
        slab.setSmall(m_direction, first_plane);
        slab.setBig(m_direction, last_plane);
        slab.setSmall(edge.fixed_direction, fixed_index);
        slab.setBig(edge.fixed_direction, fixed_index);
        slab.setSmall(edge.tangent_direction, tangent_range[0]);
        slab.setBig(edge.tangent_direction, tangent_range[1]);
        amrex::Real const weight =
            edge.orientation * geometry.CellSize(edge.tangent_direction);
        int const axis = m_direction;

        for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
            amrex::Box const section = slab & mfi.validbox();
            if (!section.ok()) {
                continue;
            }
            auto const values = field.const_array(mfi);
            auto const permeability = Permeability(mfi);
            auto const owners = owner_mask.const_array(mfi);
            amrex::Array4<int const> flags;
            if (eb_flag != nullptr) {
                flags = eb_flag->const_array(mfi);
            }
            amrex::For(section, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                if (owners(i, j, k) == 0 || (flags && flags(i, j, k) == 0)) {
                    return;
                }
                int const index[3] = {i, j, k};
                amrex::Gpu::Atomic::AddNoRet(
                    &circulation[index[axis] - first_plane],
                    weight * values(i, j, k) *
                        (permeability ? PhysConst::mu0 / permeability(i, j, k) : 1.0_rt));
            });
        }
    }

    std::vector<amrex::Real> host_circulation(number_of_planes);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_circulation.begin(),
                     device_circulation.end(), host_circulation.begin());
    return host_circulation;
}

void
CurrentControlledPort::AddScaledCurlBasisBatched (
    ablastr::fields::VectorField const& Bfield, int const first_plane,
    std::vector<amrex::Real> const& scales) const {
    amrex::Gpu::DeviceVector<amrex::Real> device_scales(scales.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, scales.begin(), scales.end(),
                     device_scales.begin());
    amrex::Real const* const scale = device_scales.data();
    int const last_plane = first_plane + static_cast<int>(scales.size()) - 1;
    int const axis = m_direction;

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
            section.setSmall(axis, first_plane);
            section.setBig(axis, last_plane);
            section &= mfi.validbox();
            if (!section.ok()) {
                continue;
            }
            auto const values = field.array(mfi);
            auto const basis_values = basis.const_array(mfi);
            amrex::ParallelFor(section, [=] AMREX_GPU_DEVICE(int i, int j,
                                                             int k) {
                int const index[3] = {i, j, k};
                values(i, j, k) +=
                    scale[index[axis] - first_plane] * basis_values(i, j, k);
            });
        }
    }
    amrex::Gpu::streamSynchronize();
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
    int const lower_index = nearest_index(Geometry(), by, transverse_mesh_direction,
                                          m_terminal_0.lo[transverse_direction]);
    int const upper_index = nearest_index(Geometry(), by, transverse_mesh_direction,
                                          m_terminal_0.hi[transverse_direction]);
    amrex::Real const upper_weight = m_direction == 2 ? 1.0_rt : -1.0_rt;

    amrex::Real circulation = 0.0_rt;
    amrex::Real response_norm = 0.0_rt;
    amrex::Real active_points = 0.0_rt;
    for (amrex::MFIter mfi(by); mfi.isValid(); ++mfi) {
        auto const values = by.const_array(mfi);
        auto const permeability = Permeability(mfi);
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
                    amrex::Real const h_weight =
                        weight * (permeability ? PhysConst::mu0 / permeability(i, j, k) : 1.0_rt);
                    return {h_weight * values(i, j, k), h_weight * h_weight, 1.0_rt};
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

std::vector<amrex::Real>
CurrentControlledPort::MeasureXZContoursLocalBatched (
    amrex::MultiFab const& by, amrex::iMultiFab const& owner_mask,
    amrex::iMultiFab const* const eb_flag, int const first_plane,
    int const last_plane) const {
    int const transverse_direction = m_direction == 0 ? 2 : 0;
    int const axial_mesh_direction = m_direction == 0 ? 0 : 1;
    int const transverse_mesh_direction = m_direction == 0 ? 1 : 0;
    int const lower_index = nearest_index(Geometry(), by, transverse_mesh_direction,
                                          m_terminal_0.lo[transverse_direction]);
    int const upper_index = nearest_index(Geometry(), by, transverse_mesh_direction,
                                          m_terminal_0.hi[transverse_direction]);
    amrex::Real const upper_weight = m_direction == 2 ? 1.0_rt : -1.0_rt;
    int const number_of_planes = last_plane - first_plane + 1;
    amrex::Gpu::DeviceVector<amrex::Real> device_contours(3 * number_of_planes,
                                                          0.0_rt);
    amrex::Real* const contours = device_contours.data();
    amrex::Geometry const& geometry = Geometry();

    for (int side = 0; side < 2; ++side) {
        int const transverse_index = side == 0 ? lower_index : upper_index;
        amrex::Real const weight = side == 0 ? -upper_weight : upper_weight;
        amrex::Box line = amrex::convert(geometry.Domain(), by.ixType());
        line.setSmall(axial_mesh_direction, first_plane);
        line.setBig(axial_mesh_direction, last_plane);
        line.setSmall(transverse_mesh_direction, transverse_index);
        line.setBig(transverse_mesh_direction, transverse_index);

        for (amrex::MFIter mfi(by); mfi.isValid(); ++mfi) {
            amrex::Box const section = line & mfi.validbox();
            if (!section.ok()) {
                continue;
            }
            auto const values = by.const_array(mfi);
            auto const permeability = Permeability(mfi);
            auto const owners = owner_mask.const_array(mfi);
            amrex::Array4<int const> flags;
            if (eb_flag != nullptr) {
                flags = eb_flag->const_array(mfi);
            }
            amrex::For(section, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                if (owners(i, j, k) == 0 || (flags && flags(i, j, k) == 0)) {
                    return;
                }
                int const index[3] = {i, j, k};
                int const offset = index[axial_mesh_direction] - first_plane;
                amrex::Real const h_weight =
                    weight * (permeability ? PhysConst::mu0 / permeability(i, j, k) : 1.0_rt);
                amrex::Gpu::Atomic::AddNoRet(&contours[3 * offset], h_weight * values(i, j, k));
                amrex::Gpu::Atomic::AddNoRet(&contours[3 * offset + 1], h_weight * h_weight);
                amrex::Gpu::Atomic::AddNoRet(&contours[3 * offset + 2], 1.0_rt);
            });
        }
    }

    std::vector<amrex::Real> host_contours(3 * number_of_planes);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_contours.begin(),
                     device_contours.end(), host_contours.begin());
    return host_contours;
}

void
CurrentControlledPort::CorrectXZContoursBatched (
    amrex::MultiFab& by, amrex::iMultiFab const* const eb_flag,
    int const first_plane, std::vector<amrex::Real> const& scales) const {
    int const transverse_direction = m_direction == 0 ? 2 : 0;
    int const axial_mesh_direction = m_direction == 0 ? 0 : 1;
    int const transverse_mesh_direction = m_direction == 0 ? 1 : 0;
    int const lower_index = nearest_index(Geometry(), by, transverse_mesh_direction,
                                          m_terminal_0.lo[transverse_direction]);
    int const upper_index = nearest_index(Geometry(), by, transverse_mesh_direction,
                                          m_terminal_0.hi[transverse_direction]);
    amrex::Real const upper_weight = m_direction == 2 ? 1.0_rt : -1.0_rt;
    int const last_plane = first_plane + static_cast<int>(scales.size()) - 1;
    amrex::Gpu::DeviceVector<amrex::Real> device_scales(scales.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, scales.begin(), scales.end(),
                     device_scales.begin());
    amrex::Real const* const scale = device_scales.data();
    amrex::Geometry const& geometry = Geometry();

    for (int side = 0; side < 2; ++side) {
        int const transverse_index = side == 0 ? lower_index : upper_index;
        amrex::Real const weight = side == 0 ? -upper_weight : upper_weight;
        amrex::Box line = amrex::convert(geometry.Domain(), by.ixType());
        line.setSmall(axial_mesh_direction, first_plane);
        line.setBig(axial_mesh_direction, last_plane);
        line.setSmall(transverse_mesh_direction, transverse_index);
        line.setBig(transverse_mesh_direction, transverse_index);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(by); mfi.isValid(); ++mfi) {
            amrex::Box const section = line & mfi.validbox();
            if (!section.ok()) {
                continue;
            }
            auto const values = by.array(mfi);
            auto const permeability = Permeability(mfi);
            amrex::Array4<int const> flags;
            if (eb_flag != nullptr) {
                flags = eb_flag->const_array(mfi);
            }
            amrex::ParallelFor(section, [=] AMREX_GPU_DEVICE(int i, int j,
                                                             int k) {
                if (flags && flags(i, j, k) == 0) {
                    return;
                }
                int const index[3] = {i, j, k};
                int const offset = index[axial_mesh_direction] - first_plane;
                values(i, j, k) += scale[offset] * weight *
                                   (permeability ? PhysConst::mu0 / permeability(i, j, k) : 1.0_rt);
            });
        }
    }
    amrex::Gpu::streamSynchronize();
}
#endif

#ifdef WARPX_DIM_RZ
std::array<amrex::Real, 3>
CurrentControlledPort::MeasureRZContourLocal (amrex::MultiFab const& btheta,
                                              amrex::iMultiFab const& owner_mask,
                                              amrex::iMultiFab const* const eb_flag,
                                              int const plane_index) const
{
    bool const axial = m_direction == 2;
    int const transverse_mesh_direction = axial ? 0 : 1;
    int const transverse_physical_direction = axial ? 0 : 2;
    amrex::Real const lower_coordinate = m_terminal_0.lo[transverse_physical_direction];
    amrex::Real const upper_coordinate = m_terminal_0.hi[transverse_physical_direction];
    bool const omit_lower = axial && lower_coordinate <= Geometry().ProbLo(0);
    int const lower_index =
        omit_lower ? 0
                   : nearest_index(Geometry(), btheta, transverse_mesh_direction, lower_coordinate);
    int const upper_index =
        nearest_index(Geometry(), btheta, transverse_mesh_direction, upper_coordinate);
    amrex::Real const lower_radius =
        axial && !omit_lower ? index_coordinate(Geometry(), btheta, 0, lower_index) : 0.0_rt;
    amrex::Real const upper_radius =
        axial ? index_coordinate(Geometry(), btheta, 0, upper_index) : 0.0_rt;
    amrex::Real const plane_radius =
        axial ? 0.0_rt : index_coordinate(Geometry(), btheta, 0, plane_index);

    amrex::Real circulation = 0.0_rt;
    amrex::Real response_norm = 0.0_rt;
    amrex::Real active_points = 0.0_rt;
    for (amrex::MFIter mfi(btheta); mfi.isValid(); ++mfi) {
        auto const values = btheta.const_array(mfi);
        auto const permeability = Permeability(mfi);
        auto const owners = owner_mask.const_array(mfi);
        amrex::Array4<int const> flags;
        if (eb_flag != nullptr) {
            flags = eb_flag->const_array(mfi);
        }
        for (int side = 0; side < 2; ++side) {
            if (side == 0 && omit_lower) {
                continue;
            }
            int const transverse_index = side == 0 ? lower_index : upper_index;
            amrex::IntVect const point =
                axial ? amrex::IntVect(AMREX_D_DECL(transverse_index, plane_index, 0))
                      : amrex::IntVect(AMREX_D_DECL(plane_index, transverse_index, 0));
            if (!mfi.validbox().contains(point)) {
                continue;
            }
            amrex::Real const radius =
                axial ? (side == 0 ? lower_radius : upper_radius) : plane_radius;
            amrex::Real const orientation =
                axial ? (side == 0 ? -1.0_rt : 1.0_rt) : (side == 0 ? 1.0_rt : -1.0_rt);
            amrex::Real const weight = orientation * 2.0_rt * MathConst::pi * radius;
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_ops;
            amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real> reduce_data(reduce_ops);
            using ReduceTuple = decltype(reduce_data)::Type;
            amrex::Box const point_box(point, point, btheta.ixType());
            reduce_ops.eval(
                point_box, reduce_data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple {
                    if (owners(i, j, k) == 0 || (flags && flags(i, j, k) == 0)) {
                        return {0.0_rt, 0.0_rt, 0.0_rt};
                    }
                    amrex::Real const h_weight =
                        weight * (permeability ? PhysConst::mu0 / permeability(i, j, k) : 1.0_rt);
                    return {h_weight * values(i, j, k), h_weight * h_weight, 1.0_rt};
                });
            auto const reduced = reduce_data.value();
            circulation += amrex::get<0>(reduced);
            response_norm += amrex::get<1>(reduced);
            active_points += amrex::get<2>(reduced);
        }
    }
    std::array<amrex::Real, 3> result{{circulation, response_norm, active_points}};
    return result;
}

std::vector<amrex::Real>
CurrentControlledPort::MeasureRZContoursLocalBatched (amrex::MultiFab const& btheta,
                                                      amrex::iMultiFab const& owner_mask,
                                                      amrex::iMultiFab const* const eb_flag,
                                                      int const first_plane,
                                                      int const last_plane) const
{
    bool const axial = m_direction == 2;
    int const plane_mesh_direction = axial ? 1 : 0;
    int const transverse_mesh_direction = axial ? 0 : 1;
    int const transverse_physical_direction = axial ? 0 : 2;
    amrex::Real const lower_coordinate = m_terminal_0.lo[transverse_physical_direction];
    amrex::Real const upper_coordinate = m_terminal_0.hi[transverse_physical_direction];
    bool const omit_lower = axial && lower_coordinate <= Geometry().ProbLo(0);
    int const lower_index =
        omit_lower ? 0
                   : nearest_index(Geometry(), btheta, transverse_mesh_direction, lower_coordinate);
    int const upper_index =
        nearest_index(Geometry(), btheta, transverse_mesh_direction, upper_coordinate);
    amrex::Real const lower_radius =
        axial && !omit_lower ? index_coordinate(Geometry(), btheta, 0, lower_index) : 0.0_rt;
    amrex::Real const upper_radius =
        axial ? index_coordinate(Geometry(), btheta, 0, upper_index) : 0.0_rt;
    int const number_of_planes = last_plane - first_plane + 1;
    std::vector<amrex::Real> weights(2 * number_of_planes);
    for (int offset = 0; offset < number_of_planes; ++offset) {
        amrex::Real const plane_radius =
            axial ? 0.0_rt : index_coordinate(Geometry(), btheta, 0, first_plane + offset);
        weights[2 * offset] =
            omit_lower ? 0.0_rt : -2.0_rt * MathConst::pi * (axial ? lower_radius : -plane_radius);
        weights[2 * offset + 1] = 2.0_rt * MathConst::pi * (axial ? upper_radius : -plane_radius);
    }

    amrex::Gpu::DeviceVector<amrex::Real> device_weights(weights.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, weights.begin(), weights.end(),
                     device_weights.begin());
    amrex::Real const* const weight_values = device_weights.data();
    amrex::Gpu::DeviceVector<amrex::Real> device_contours(3 * number_of_planes, 0.0_rt);
    amrex::Real* const contours = device_contours.data();
    amrex::Geometry const& geometry = Geometry();

    for (int side = 0; side < 2; ++side) {
        if (side == 0 && omit_lower) {
            continue;
        }
        int const transverse_index = side == 0 ? lower_index : upper_index;
        amrex::Box line = amrex::convert(geometry.Domain(), btheta.ixType());
        line.setSmall(plane_mesh_direction, first_plane);
        line.setBig(plane_mesh_direction, last_plane);
        line.setSmall(transverse_mesh_direction, transverse_index);
        line.setBig(transverse_mesh_direction, transverse_index);

        for (amrex::MFIter mfi(btheta); mfi.isValid(); ++mfi) {
            amrex::Box const section = line & mfi.validbox();
            if (!section.ok()) {
                continue;
            }
            auto const values = btheta.const_array(mfi);
            auto const permeability = Permeability(mfi);
            auto const owners = owner_mask.const_array(mfi);
            amrex::Array4<int const> flags;
            if (eb_flag != nullptr) {
                flags = eb_flag->const_array(mfi);
            }
            amrex::For(section, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                if (owners(i, j, k) == 0 || (flags && flags(i, j, k) == 0)) {
                    return;
                }
                int const index[3] = {i, j, k};
                int const offset = index[plane_mesh_direction] - first_plane;
                amrex::Real const weight =
                    weight_values[2 * offset + side] *
                    (permeability ? PhysConst::mu0 / permeability(i, j, k) : 1.0_rt);
                amrex::Gpu::Atomic::AddNoRet(&contours[3 * offset], weight * values(i, j, k));
                amrex::Gpu::Atomic::AddNoRet(&contours[3 * offset + 1], weight * weight);
                amrex::Gpu::Atomic::AddNoRet(&contours[3 * offset + 2], 1.0_rt);
            });
        }
    }

    std::vector<amrex::Real> host_contours(3 * number_of_planes);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_contours.begin(), device_contours.end(),
                     host_contours.begin());
    return host_contours;
}

void
CurrentControlledPort::CorrectRZContoursBatched (amrex::MultiFab& btheta,
                                                 amrex::iMultiFab const* const eb_flag,
                                                 int const first_plane,
                                                 std::vector<amrex::Real> const& scales) const
{
    bool const axial = m_direction == 2;
    int const plane_mesh_direction = axial ? 1 : 0;
    int const transverse_mesh_direction = axial ? 0 : 1;
    int const transverse_physical_direction = axial ? 0 : 2;
    amrex::Real const lower_coordinate = m_terminal_0.lo[transverse_physical_direction];
    amrex::Real const upper_coordinate = m_terminal_0.hi[transverse_physical_direction];
    bool const omit_lower = axial && lower_coordinate <= Geometry().ProbLo(0);
    int const lower_index =
        omit_lower ? 0
                   : nearest_index(Geometry(), btheta, transverse_mesh_direction, lower_coordinate);
    int const upper_index =
        nearest_index(Geometry(), btheta, transverse_mesh_direction, upper_coordinate);
    amrex::Real const lower_radius =
        axial && !omit_lower ? index_coordinate(Geometry(), btheta, 0, lower_index) : 0.0_rt;
    amrex::Real const upper_radius =
        axial ? index_coordinate(Geometry(), btheta, 0, upper_index) : 0.0_rt;
    int const number_of_planes = static_cast<int>(scales.size());
    int const last_plane = first_plane + number_of_planes - 1;
    std::vector<amrex::Real> weights(2 * number_of_planes);
    for (int offset = 0; offset < number_of_planes; ++offset) {
        amrex::Real const plane_radius =
            axial ? 0.0_rt : index_coordinate(Geometry(), btheta, 0, first_plane + offset);
        weights[2 * offset] =
            omit_lower ? 0.0_rt : -2.0_rt * MathConst::pi * (axial ? lower_radius : -plane_radius);
        weights[2 * offset + 1] = 2.0_rt * MathConst::pi * (axial ? upper_radius : -plane_radius);
    }

    amrex::Gpu::DeviceVector<amrex::Real> device_weights(weights.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, weights.begin(), weights.end(),
                     device_weights.begin());
    amrex::Real const* const weight_values = device_weights.data();
    amrex::Gpu::DeviceVector<amrex::Real> device_scales(scales.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, scales.begin(), scales.end(), device_scales.begin());
    amrex::Real const* const scale_values = device_scales.data();
    amrex::Geometry const& geometry = Geometry();

    for (int side = 0; side < 2; ++side) {
        if (side == 0 && omit_lower) {
            continue;
        }
        int const transverse_index = side == 0 ? lower_index : upper_index;
        amrex::Box line = amrex::convert(geometry.Domain(), btheta.ixType());
        line.setSmall(plane_mesh_direction, first_plane);
        line.setBig(plane_mesh_direction, last_plane);
        line.setSmall(transverse_mesh_direction, transverse_index);
        line.setBig(transverse_mesh_direction, transverse_index);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(btheta); mfi.isValid(); ++mfi) {
            amrex::Box const section = line & mfi.validbox();
            if (!section.ok()) {
                continue;
            }
            auto const values = btheta.array(mfi);
            auto const permeability = Permeability(mfi);
            amrex::Array4<int const> flags;
            if (eb_flag != nullptr) {
                flags = eb_flag->const_array(mfi);
            }
            amrex::ParallelFor(section, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                if (flags && flags(i, j, k) == 0) {
                    return;
                }
                int const index[3] = {i, j, k};
                int const offset = index[plane_mesh_direction] - first_plane;
                values(i, j, k) += scale_values[offset] * weight_values[2 * offset + side] *
                                   (permeability ? PhysConst::mu0 / permeability(i, j, k) : 1.0_rt);
            });
        }
    }
    amrex::Gpu::streamSynchronize();
}
#endif

#ifdef WARPX_DIM_3D
void
CurrentControlledPort::Apply3D (
    ablastr::fields::VectorField const& Bfield,
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update_B,
    amrex::Real const target_axis_current) {
    auto const edges = Edges();
    amrex::MultiFab const& reference_field = *Bfield[edges[0].component];
    int const plane_0 =
        nearest_index(Geometry(), reference_field, m_direction, m_terminal_0.lo[m_direction]);
    int const plane_1 =
        nearest_index(Geometry(), reference_field, m_direction, m_terminal_1.lo[m_direction]);
    int const first_plane = std::min(plane_0, plane_1);
    int const last_plane = std::max(plane_0, plane_1);
    int const number_of_planes = last_plane - first_plane + 1;
    ablastr::fields::VectorField const basis_field{
        m_curl_basis[0].get(), m_curl_basis[1].get(), m_curl_basis[2].get()};
    auto const field_circulation = MeasureContoursLocalBatched(
        Bfield, eb_update_B, first_plane, last_plane);
    auto const basis_circulation = MeasureContoursLocalBatched(
        basis_field, eb_update_B, first_plane, last_plane);
    std::vector<amrex::Real> contour_data(2 * number_of_planes);
    for (int offset = 0; offset < number_of_planes; ++offset) {
        contour_data[2 * offset] = field_circulation[offset];
        contour_data[2 * offset + 1] = basis_circulation[offset];
    }
    amrex::ParallelDescriptor::ReduceRealSum(
        contour_data.data(), static_cast<int>(contour_data.size()));

    amrex::Real const desired_circulation =
        PhysConst::mu0 * target_axis_current;
    std::vector<amrex::Real> scales(number_of_planes);
    for (int offset = 0; offset < number_of_planes; ++offset) {
        amrex::Real const circulation = contour_data[2 * offset];
        amrex::Real const basis_response = contour_data[2 * offset + 1];
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::abs(basis_response) > 0.0_rt,
            "The current-controlled port contour has no solver-active curl "
            "basis; place its contour outside covered EB cells and away from "
            "the domain edge.");
        scales[offset] = (desired_circulation - circulation) / basis_response;
    }
    AddScaledCurlBasisBatched(Bfield, first_plane, scales);
    std::array<amrex::Real, 2> terminal_circulation{
        MeasureContourLocal(Bfield, eb_update_B, plane_0),
        MeasureContourLocal(Bfield, eb_update_B, plane_1)};
    amrex::ParallelDescriptor::ReduceRealSum(terminal_circulation.data(), 2);
    for (int terminal = 0; terminal < 2; ++terminal) {
        m_last_terminal_current[terminal] =
            m_axis_sign * terminal_circulation[terminal] / PhysConst::mu0;
    }
}
#endif

#ifdef WARPX_DIM_XZ
void
CurrentControlledPort::ApplyXZ (
    ablastr::fields::VectorField const& Bfield,
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update_B,
    amrex::Real const target_axis_current) {
    amrex::MultiFab& by = *Bfield[1];
    int const axial_mesh_direction = m_direction == 0 ? 0 : 1;
    int const plane_0 =
        nearest_index(Geometry(), by, axial_mesh_direction, m_terminal_0.lo[m_direction]);
    int const plane_1 =
        nearest_index(Geometry(), by, axial_mesh_direction, m_terminal_1.lo[m_direction]);
    int const first_plane = std::min(plane_0, plane_1);
    int const last_plane = std::max(plane_0, plane_1);
    int const number_of_planes = last_plane - first_plane + 1;
    std::vector<amrex::Real> contour_data = MeasureXZContoursLocalBatched(
        by, *m_owner_mask[1], eb_update_B[1].get(), first_plane, last_plane);
    amrex::ParallelDescriptor::ReduceRealSum(
        contour_data.data(), static_cast<int>(contour_data.size()));
    std::vector<amrex::Real> scales(number_of_planes);
    for (int offset = 0; offset < number_of_planes; ++offset) {
        std::array<amrex::Real, 3> contour{contour_data[3 * offset],
                                           contour_data[3 * offset + 1],
                                           contour_data[3 * offset + 2]};
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            contour[1] > 0.0_rt && contour[2] > 0.0_rt,
            "The 2D XZ current-controlled contour has no active field points; "
            "place its endpoints outside covered EB cells.");
        amrex::Real const measured_current = contour[0] / PhysConst::mu0;
        scales[offset] = PhysConst::mu0 *
                         (target_axis_current - measured_current) / contour[1];
    }
    CorrectXZContoursBatched(by, eb_update_B[1].get(), first_plane, scales);
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
}
#endif

#ifdef WARPX_DIM_RZ
void
CurrentControlledPort::ApplyRZ (
    ablastr::fields::VectorField const& Bfield,
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update_B,
    amrex::Real const target_axis_current) {
    amrex::MultiFab& btheta = *Bfield[1];
    int const terminal_mesh_direction = m_direction == 0 ? 0 : 1;
    int const plane_0 =
        nearest_index(Geometry(), btheta, terminal_mesh_direction, m_terminal_0.lo[m_direction]);
    int const plane_1 =
        nearest_index(Geometry(), btheta, terminal_mesh_direction, m_terminal_1.lo[m_direction]);
    int const first_plane = std::min(plane_0, plane_1);
    int const last_plane = std::max(plane_0, plane_1);
    int const number_of_planes = last_plane - first_plane + 1;
    std::vector<amrex::Real> contour_data = MeasureRZContoursLocalBatched(
        btheta, *m_owner_mask[1], eb_update_B[1].get(), first_plane,
        last_plane);
    amrex::ParallelDescriptor::ReduceRealSum(
        contour_data.data(), static_cast<int>(contour_data.size()));
    std::vector<amrex::Real> scales(number_of_planes);
    for (int offset = 0; offset < number_of_planes; ++offset) {
        std::array<amrex::Real, 3> contour{contour_data[3 * offset],
                                           contour_data[3 * offset + 1],
                                           contour_data[3 * offset + 2]};
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            contour[1] > 0.0_rt && contour[2] > 0.0_rt,
            "The RZ current-controlled contour has no active field points; "
            "place its contour outside covered EB cells.");
        amrex::Real const measured_current = contour[0] / PhysConst::mu0;
        scales[offset] = PhysConst::mu0 *
                         (target_axis_current - measured_current) / contour[1];
    }
    CorrectRZContoursBatched(btheta, eb_update_B[1].get(), first_plane, scales);
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
}
#endif

void
CurrentControlledPort::ApplyBfield (
    ablastr::fields::VectorField const& Bfield,
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update_B,
    int const lev, PatchType const patch_type, amrex::Real const time) {
    if (m_composite != nullptr) {
        return;
    }
    if (lev != m_level || patch_type != m_patch_type) {
        for (auto& patch : m_refined_patches) {
            if (patch->m_level == lev && patch->m_patch_type == patch_type) {
                patch->ApplyBfield(Bfield, eb_update_B, lev, patch_type, time);
                return;
            }
        }
        return;
    }
    if (!m_initialized) {
        return;
    }

    amrex::Real const requested_current =
        m_current_scale * m_waveform.value(time);
    amrex::Real const target_axis_current = m_axis_sign * requested_current;
    m_last_target = requested_current;

#ifdef WARPX_DIM_3D
    Apply3D(Bfield, eb_update_B, target_axis_current);
#elif defined(WARPX_DIM_XZ)
    ApplyXZ(Bfield, eb_update_B, target_axis_current);
#elif defined(WARPX_DIM_RZ)
    ApplyRZ(Bfield, eb_update_B, target_axis_current);
#else
    amrex::ignore_unused(Bfield, eb_update_B, target_axis_current);
#endif
}

void
CurrentControlledPort::ApplyComposite (amrex::Real time)
{
    m_last_target = m_current_scale * m_waveform.value(time);
    m_composite->Apply(m_axis_sign * m_last_target);
}

void
CurrentControlledPort::UpdateCompositeStatus ()
{
    auto const currents = m_composite->Currents(m_axis_sign * m_last_target);
    for (int terminal = 0; terminal < 2; ++terminal) {
        m_last_terminal_current[terminal] = m_axis_sign * currents[terminal];
    }
}

void
WarpX::ApplyCompositeCurrentControlledPorts (amrex::Real time)
{
    if (std::none_of(m_current_controlled_ports.begin(),
                     m_current_controlled_ports.end(), [] (auto const& port) {
                         return port->uses_composite();
                     })) {
        return;
    }
    FillBoundaryB(guard_cells.ng_alloc_EB, WarpX::sync_nodal_points);
    UpdateAuxiliaryData();
    FillBoundaryAux(guard_cells.ng_UpdateAux);
    FillBoundaryAux(finest_level, guard_cells.ng_UpdateAux);
    for (auto& port : m_current_controlled_ports) {
        if (port->uses_composite()) {
            port->ApplyComposite(time);
            FillBoundaryB(guard_cells.ng_alloc_EB, WarpX::sync_nodal_points);
            UpdateAuxiliaryData();
            FillBoundaryAux(guard_cells.ng_UpdateAux);
            FillBoundaryAux(finest_level, guard_cells.ng_UpdateAux);
            port->UpdateCompositeStatus();
        }
    }
}

void
WarpX::ApplyCurrentControlledPort (int const lev, PatchType const patch_type,
                                   amrex::Real const time) {
    if (m_current_controlled_ports.empty()) {
        return;
    }
    auto const Bfield =
        m_fields.get_alldirs(patch_type == PatchType::fine ? warpx::fields::FieldType::Bfield_fp
                                                           : warpx::fields::FieldType::Bfield_cp,
                             lev);
    for (auto& current_port : m_current_controlled_ports) {
        current_port->ApplyBfield(Bfield, GetEBUpdateBFlag(patch_type)[lev],
                                  lev, patch_type, time);
    }
}
