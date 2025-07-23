#ifndef QUARTZ_BOUNDARY_H_
#define QUARTZ_BOUNDARY_H_

#include "Utils/WarpXAlgorithmSelection.H"
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_Vector.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Parser.H>
#include <array>
#include <memory>
#include <string>

class QuartzBoundary
{
public:
    QuartzBoundary();

    void ApplytoEfield(
        std::array<amrex::MultiFab*,3> Efield,
        amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_lo,
        amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_hi,
        amrex::IntVect const & ng_fieldgather, amrex::Geometry const & geom,
        int lev, PatchType patch_type, amrex::Vector<amrex::IntVect> const & ref_ratios,
        amrex::Real time,
        bool split_pml_field = false);

    void ApplytoBfield(
        std::array<amrex::MultiFab*,3> Bfield,
        amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_lo,
        amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_hi,
        amrex::IntVect const & ng_fieldgather, amrex::Geometry const & geom,
        int lev, PatchType patch_type, amrex::Vector<amrex::IntVect> const & ref_ratios,
        amrex::Real time);

    void ReadParameters();

    bool IsInsideRing(amrex::Real x, amrex::Real y, amrex::Real z) const;

private:
    amrex::Real m_epsilon_r = 3.8;
    amrex::Real m_mu_r = 1.0;
    amrex::Real m_sigma = 0.0;
    bool m_use_space_dependent = false;
    std::unique_ptr<amrex::Parser> m_epsilon_r_parser;
    std::unique_ptr<amrex::Parser> m_mu_r_parser;
    std::unique_ptr<amrex::Parser> m_sigma_parser;
    std::string m_str_epsilon_r_function;
    std::string m_str_mu_r_function;
    std::string m_str_sigma_function;
    amrex::Real m_ring_inner_radius = 0.05;
    amrex::Real m_ring_outer_radius = 0.08;
    amrex::Real m_ring_height = 0.02;
    amrex::Real m_ring_center_x = 0.0;
    amrex::Real m_ring_center_y = 0.0;
    amrex::Real m_ring_bottom_z = 0.0;
    std::unique_ptr<amrex::Parser> m_ring_geometry_parser;
    std::string m_str_ring_geometry_function;
};

#endif // QUARTZ_BOUNDARY_H_
