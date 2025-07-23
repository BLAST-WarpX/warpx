// Quartz.cpp - Implementation of QuartzBoundary for dielectric (quartz) boundary conditions
// Physical parameters are annotated in English for clarity.
#include "BoundaryConditions/Quartz.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Config.H>
#include <AMReX_Extension.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <memory>
#include <sstream>
using namespace amrex;

// Constructor: reads all quartz boundary parameters
QuartzBoundary::QuartzBoundary()
{
    ReadParameters();
}

// Reads physical parameters for quartz boundary from input file
void QuartzBoundary::ReadParameters()
{
    const ParmParse pp_("");
    // Relative permittivity (dielectric constant) of quartz
    pp_.query("epsilon_r", m_epsilon_r); // e.g. 3.8 for quartz
    // Relative permeability of quartz (usually ~1.0 for non-magnetic materials)
    pp_.query("mu_r", m_mu_r);
    // Electrical conductivity of quartz (S/m)
    pp_.query("sigma", m_sigma);
    // Ring geometry parameters for CCP (Capacitively Coupled Plasma) applications
    // Inner radius of the quartz ring (meters)
    pp_.query("ring_inner_radius", m_ring_inner_radius);
    // Outer radius of the quartz ring (meters)
    pp_.query("ring_outer_radius", m_ring_outer_radius);
    // Height of the quartz ring (meters)
    pp_.query("ring_height", m_ring_height);
    // Center x-coordinate of the ring (meters)
    pp_.query("ring_center_x", m_ring_center_x);
    // Center y-coordinate of the ring (meters)
    pp_.query("ring_center_y", m_ring_center_y);
    // Bottom z-coordinate of the ring (meters)
    pp_.query("ring_bottom_z", m_ring_bottom_z);
    // Space-dependent permittivity function (if provided)
    if (pp_.query("epsilon_r_function(x,y,z)", m_str_epsilon_r_function)) {
        m_use_space_dependent = true;
        utils::parser::Store_parserString(pp_, "epsilon_r_function(x,y,z)", m_str_epsilon_r_function);
        m_epsilon_r_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_str_epsilon_r_function, {"x", "y", "z"}));
    }
    // Space-dependent permeability function (if provided)
    if (pp_.query("mu_r_function(x,y,z)", m_str_mu_r_function)) {
        m_use_space_dependent = true;
        utils::parser::Store_parserString(pp_, "mu_r_function(x,y,z)", m_str_mu_r_function);
        m_mu_r_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_str_mu_r_function, {"x", "y", "z"}));
    }
    // Space-dependent conductivity function (if provided)
    if (pp_.query("sigma_function(x,y,z)", m_str_sigma_function)) {
        m_use_space_dependent = true;
        utils::parser::Store_parserString(pp_, "sigma_function(x,y,z)", m_str_sigma_function);
        m_sigma_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_str_sigma_function, {"x", "y", "z"}));
    }
    // Custom ring geometry function (if provided)
    if (pp_.query("ring_geometry_function(x,y,z)", m_str_ring_geometry_function)) {
        m_use_space_dependent = true;
        utils::parser::Store_parserString(pp_, "ring_geometry_function(x,y,z)", m_str_ring_geometry_function);
        m_ring_geometry_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_str_ring_geometry_function, {"x", "y", "z"}));
    }
}

// Applies quartz boundary condition to the E-field
void QuartzBoundary::ApplytoEfield(
    std::array<amrex::MultiFab*,3> Efield,
    amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_lo,
    amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_hi,
    amrex::IntVect const & ng_fieldgather, amrex::Geometry const & geom,
    int lev, PatchType patch_type, amrex::Vector<amrex::IntVect> const & ref_ratios,
    amrex::Real time,
    bool split_pml_field)
{
    // E_tangential is continuous across the boundary
    // D_normal = epsilon_r * E_normal is continuous across the boundary
    // This means E_normal is discontinuous: E_normal_outside = epsilon_r * E_normal_inside
    const auto dx = geom.CellSizeArray(); // Cell size (meters)
    const auto prob_lo = geom.ProbLoArray(); // Physical domain lower bound
    const auto prob_hi = geom.ProbHiArray(); // Physical domain upper bound
    const Box& domain = geom.Domain(); // Simulation domain
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        bool has_quartz_lo = (field_boundary_lo[idim] == FieldBoundaryType::Quartz);
        bool has_quartz_hi = (field_boundary_hi[idim] == FieldBoundaryType::Quartz);
        if (!has_quartz_lo && !has_quartz_hi) continue;
        for (int icomp = 0; icomp < 3; ++icomp) {
            if (Efield[icomp] == nullptr) continue;
            bool is_normal_component = (icomp == idim); // Whether this field component is normal to the boundary
            MultiFab& mf = *Efield[icomp];
            for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
                FArrayBox& fab = mf[mfi];
                const Box& bx = fab.box();
                // Lower boundary
                if (has_quartz_lo) {
                    Box lo_bx = bx;
                    lo_bx.setBig(idim, domain.smallEnd(idim) - 1);
                    lo_bx &= bx;
                    if (lo_bx.ok()) {
                        Array4<Real> const& arr = fab.array();
                        amrex::ParallelFor(lo_bx,
                            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                            {
                                // For normal component: E_outside = epsilon_r * E_inside
                                // For tangential component: E_outside = E_inside
                                if (is_normal_component) {
                                    arr(i,j,k) = m_epsilon_r * arr(i,j,k);
                                }
                            });
                    }
                }
                // Upper boundary
                if (has_quartz_hi) {
                    Box hi_bx = bx;
                    hi_bx.setSmall(idim, domain.bigEnd(idim) + 1);
                    hi_bx &= bx;
                    if (hi_bx.ok()) {
                        Array4<Real> const& arr = fab.array();
                        amrex::ParallelFor(hi_bx,
                            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                            {
                                // For normal component: E_outside = epsilon_r * E_inside
                                // For tangential component: E_outside = E_inside
                                if (is_normal_component) {
                                    arr(i,j,k) = m_epsilon_r * arr(i,j,k);
                                }
                            });
                    }
                }
            }
        }
    }
}

// Applies quartz boundary condition to the B-field
void QuartzBoundary::ApplytoBfield(
    std::array<amrex::MultiFab*,3> Bfield,
    amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_lo,
    amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_hi,
    amrex::IntVect const & ng_fieldgather, amrex::Geometry const & geom,
    int lev, PatchType patch_type, amrex::Vector<amrex::IntVect> const & ref_ratios,
    amrex::Real time)
{
    // For quartz boundary: B_normal and B_tangential are continuous (mu_r ~ 1)
    const auto dx = geom.CellSizeArray(); // Cell size (meters)
    const auto prob_lo = geom.ProbLoArray(); // Physical domain lower bound
    const auto prob_hi = geom.ProbHiArray(); // Physical domain upper bound
    const Box& domain = geom.Domain(); // Simulation domain
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        bool has_quartz_lo = (field_boundary_lo[idim] == FieldBoundaryType::Quartz);
        bool has_quartz_hi = (field_boundary_hi[idim] == FieldBoundaryType::Quartz);
        if (!has_quartz_lo && !has_quartz_hi) continue;
        for (int icomp = 0; icomp < 3; ++icomp) {
            if (Bfield[icomp] == nullptr) continue;
            bool is_normal_component = (icomp == idim); // Whether this field component is normal to the boundary
            MultiFab& mf = *Bfield[icomp];
            for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
                FArrayBox& fab = mf[mfi];
                const Box& bx = fab.box();
                // Lower boundary
                if (has_quartz_lo) {
                    Box lo_bx = bx;
                    lo_bx.setBig(idim, domain.smallEnd(idim) - 1);
                    lo_bx &= bx;
                    if (lo_bx.ok()) {
                        Array4<Real> const& arr = fab.array();
                        amrex::ParallelFor(lo_bx,
                            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                            {
                                // For quartz: B-field is continuous, no change needed
                            });
                    }
                }
                // Upper boundary
                if (has_quartz_hi) {
                    Box hi_bx = bx;
                    hi_bx.setSmall(idim, domain.bigEnd(idim) + 1);
                    hi_bx &= bx;
                    if (hi_bx.ok()) {
                        Array4<Real> const& arr = fab.array();
                        amrex::ParallelFor(hi_bx,
                            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                            {
                                // For quartz: B-field is continuous, no change needed
                            });
                    }
                }
            }
        }
    }
}

// Checks if a point (x, y, z) is inside the quartz ring geometry
bool QuartzBoundary::IsInsideRing(amrex::Real x, amrex::Real y, amrex::Real z) const
{
    // If a custom geometry function is provided, use it
    if (m_ring_geometry_parser) {
        auto const& ring_func = m_ring_geometry_parser->compile<3>();
        return ring_func(x, y, z) < 0.0_rt;
    }
    // Default: cylindrical ring geometry for CCP
    amrex::Real dx = x - m_ring_center_x; // x offset from ring center
    amrex::Real dy = y - m_ring_center_y; // y offset from ring center
    amrex::Real r = std::sqrt(dx*dx + dy*dy); // radial distance from center
    bool in_radius = (r >= m_ring_inner_radius) && (r <= m_ring_outer_radius); // within ring radii
    bool in_height = (z >= m_ring_bottom_z) && (z <= m_ring_bottom_z + m_ring_height); // within ring height
    return in_radius && in_height;
} 