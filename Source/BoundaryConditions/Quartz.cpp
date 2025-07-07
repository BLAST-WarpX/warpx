#include "Quartz.H"

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

Quartz::Quartz()
{
    ReadParameters();
}

void Quartz::ReadParameters()
{
    const ParmParse pp_quartz("quartz");
    
    // Read relative permittivity
    pp_quartz.query("epsilon_r", m_epsilon_r);
    
    // Read relative permeability
    pp_quartz.query("mu_r", m_mu_r);
    
    // Read conductivity
    pp_quartz.query("sigma", m_sigma);
    
    // Read quartz ring geometry parameters for CCP applications
    pp_quartz.query("ring_inner_radius", m_ring_inner_radius);
    pp_quartz.query("ring_outer_radius", m_ring_outer_radius);
    pp_quartz.query("ring_height", m_ring_height);
    pp_quartz.query("ring_center_x", m_ring_center_x);
    pp_quartz.query("ring_center_y", m_ring_center_y);
    pp_quartz.query("ring_bottom_z", m_ring_bottom_z);
    
    // Check for space-dependent properties
    if (pp_quartz.query("epsilon_r_function(x,y,z)", m_str_epsilon_r_function)) {
        m_use_space_dependent = true;
        utils::parser::Store_parserString(pp_quartz, "epsilon_r_function(x,y,z)", m_str_epsilon_r_function);
        m_epsilon_r_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_str_epsilon_r_function, {"x", "y", "z"}));
    }
    
    if (pp_quartz.query("mu_r_function(x,y,z)", m_str_mu_r_function)) {
        m_use_space_dependent = true;
        utils::parser::Store_parserString(pp_quartz, "mu_r_function(x,y,z)", m_str_mu_r_function);
        m_mu_r_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_str_mu_r_function, {"x", "y", "z"}));
    }
    
    if (pp_quartz.query("sigma_function(x,y,z)", m_str_sigma_function)) {
        m_use_space_dependent = true;
        utils::parser::Store_parserString(pp_quartz, "sigma_function(x,y,z)", m_str_sigma_function);
        m_sigma_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_str_sigma_function, {"x", "y", "z"}));
    }
    
    // Check for custom ring geometry function
    if (pp_quartz.query("ring_geometry_function(x,y,z)", m_str_ring_geometry_function)) {
        m_use_space_dependent = true;
        utils::parser::Store_parserString(pp_quartz, "ring_geometry_function(x,y,z)", m_str_ring_geometry_function);
        m_ring_geometry_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_str_ring_geometry_function, {"x", "y", "z"}));
    }
}

void Quartz::ApplyQuartztoEfield(
    std::array<amrex::MultiFab*,3> Efield,
    amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_lo,
    amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_hi,
    amrex::IntVect const & ng_fieldgather, amrex::Geometry const & geom,
    int lev, PatchType patch_type, amrex::Vector<amrex::IntVect> const & ref_ratios,
    amrex::Real time,
    bool split_pml_field)
{
    using namespace amrex::literals;
    
    // For quartz boundary condition:
    // - E_tangential is continuous across the boundary
    // - D_normal = epsilon_r * E_normal is continuous across the boundary
    // - This means E_normal is discontinuous: E_normal_outside = epsilon_r * E_normal_inside
    
    const auto dx = geom.CellSizeArray();
    const auto prob_lo = geom.ProbLoArray();
    const auto prob_hi = geom.ProbHiArray();
    
    // Get the domain box
    const Box& domain = geom.Domain();
    
    // Loop over each direction
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        
        // Check if this direction has quartz boundary
        bool has_quartz_lo = (field_boundary_lo[idim] == FieldBoundaryType::Quartz);
        bool has_quartz_hi = (field_boundary_hi[idim] == FieldBoundaryType::Quartz);
        
        if (!has_quartz_lo && !has_quartz_hi) continue;
        
        // Loop over each field component
        for (int icomp = 0; icomp < 3; ++icomp) {
            
            if (Efield[icomp] == nullptr) continue;
            
            // Determine if this component is normal or tangential to the boundary
            bool is_normal_component = (icomp == idim);
            
            // Get the field data
            MultiFab& mf = *Efield[icomp];
            
            // Loop over the MultiFab
            for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
                
                FArrayBox& fab = mf[mfi];
                const Box& bx = fab.box();
                
                // Apply boundary condition at lower boundary
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
                                // Tangential components remain unchanged (continuous)
                            });
                    }
                }
                
                // Apply boundary condition at upper boundary
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
                                // Tangential components remain unchanged (continuous)
                            });
                    }
                }
            }
        }
    }
}

void Quartz::ApplyQuartztoBfield(
    std::array<amrex::MultiFab*,3> Bfield,
    amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_lo,
    amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_hi,
    amrex::IntVect const & ng_fieldgather, amrex::Geometry const & geom,
    int lev, PatchType patch_type, amrex::Vector<amrex::IntVect> const & ref_ratios,
    amrex::Real time)
{
    using namespace amrex::literals;
    
    // For quartz boundary condition on B-field:
    // - B_normal is continuous across the boundary
    // - H_tangential = B_tangential / mu_r is continuous
    // - Since quartz is non-magnetic (mu_r ~ 1), B_tangential is also continuous
    // - This is similar to PEC boundary condition for B-field
    
    const auto dx = geom.CellSizeArray();
    const auto prob_lo = geom.ProbLoArray();
    const auto prob_hi = geom.ProbHiArray();
    
    // Get the domain box
    const Box& domain = geom.Domain();
    
    // Loop over each direction
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        
        // Check if this direction has quartz boundary
        bool has_quartz_lo = (field_boundary_lo[idim] == FieldBoundaryType::Quartz);
        bool has_quartz_hi = (field_boundary_hi[idim] == FieldBoundaryType::Quartz);
        
        if (!has_quartz_lo && !has_quartz_hi) continue;
        
        // Loop over each field component
        for (int icomp = 0; icomp < 3; ++icomp) {
            
            if (Bfield[icomp] == nullptr) continue;
            
            // Determine if this component is normal or tangential to the boundary
            bool is_normal_component = (icomp == idim);
            
            // Get the field data
            MultiFab& mf = *Bfield[icomp];
            
            // Loop over the MultiFab
            for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
                
                FArrayBox& fab = mf[mfi];
                const Box& bx = fab.box();
                
                // Apply boundary condition at lower boundary
                if (has_quartz_lo) {
                    Box lo_bx = bx;
                    lo_bx.setBig(idim, domain.smallEnd(idim) - 1);
                    lo_bx &= bx;
                    
                    if (lo_bx.ok()) {
                        Array4<Real> const& arr = fab.array();
                        
                        amrex::ParallelFor(lo_bx,
                            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                            {
                                // For normal component: B_outside = B_inside (continuous)
                                // For tangential component: B_outside = B_inside (continuous)
                                // Since quartz is non-magnetic, all components are continuous
                                // No change needed for B-field
                            });
                    }
                }
                
                // Apply boundary condition at upper boundary
                if (has_quartz_hi) {
                    Box hi_bx = bx;
                    hi_bx.setSmall(idim, domain.bigEnd(idim) + 1);
                    hi_bx &= bx;
                    
                    if (hi_bx.ok()) {
                        Array4<Real> const& arr = fab.array();
                        
                        amrex::ParallelFor(hi_bx,
                            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                            {
                                // For normal component: B_outside = B_inside (continuous)
                                // For tangential component: B_outside = B_inside (continuous)
                                // Since quartz is non-magnetic, all components are continuous
                                // No change needed for B-field
                            });
                    }
                }
            }
        }
    }
}

bool Quartz::IsInsideQuartzRing(amrex::Real x, amrex::Real y, amrex::Real z) const
{
    using namespace amrex::literals;

    // 如果有自定义几何函数，优先用它
    if (m_ring_geometry_parser) {
        auto const& ring_func = m_ring_geometry_parser->compile<3>();
        return ring_func(x, y, z) < 0.0_rt;
    }

    // 默认CCP石英环几何
    amrex::Real dx = x - m_ring_center_x;
    amrex::Real dy = y - m_ring_center_y;
    amrex::Real r = std::sqrt(dx*dx + dy*dy);

    bool in_radius = (r >= m_ring_inner_radius) && (r <= m_ring_outer_radius);
    bool in_height = (z >= m_ring_bottom_z) && (z <= m_ring_bottom_z + m_ring_height);

    return in_radius && in_height;
} 