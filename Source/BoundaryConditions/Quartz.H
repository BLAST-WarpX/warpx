#ifndef QUARTZ_H_
#define QUARTZ_H_

#include "Utils/WarpXAlgorithmSelection.H"

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_Vector.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Parser.H>

#include <AMReX_BaseFwd.H>

#include <array>
#include <memory>

class Quartz
{
public:

    Quartz();

    /**
     * \brief Apply quartz boundary condition on the E-field.
     *        This is specifically designed for CCP (Capacitively Coupled Plasma) quartz ring applications.
     *        The quartz ring acts as an insulating barrier between electrodes.
     *        Quartz is a dielectric material with relative permittivity ~3.8.
     *        The boundary condition enforces continuity of D_normal and E_tangential.
     *
     * \param[in,out] Efield
     * \param[in]     field_boundary_lo   lower field boundary conditions
     * \param[in]     field_boundary_hi   upper field boundary conditions
     * \param[in]     ng_fieldgather      number of guard cells used by field gather
     * \param[in]     geom                geometry object of level "lev"
     * \param[in]     lev                 level of the Multifab
     * \param[in]     patch_type          coarse or fine
     * \param[in]     ref_ratios          vector containing the refinement ratios of the refinement levels
     * \param[in]     time                current time of the simulation
     * \param[in]     split_pml_field     whether pml the multifab is the regular Efield or
     *                                    split pml field
     */
    void ApplyQuartztoEfield(
        std::array<amrex::MultiFab*,3> Efield,
        amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_lo,
        amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_hi,
        amrex::IntVect const & ng_fieldgather, amrex::Geometry const & geom,
        int lev, PatchType patch_type, amrex::Vector<amrex::IntVect> const & ref_ratios,
        amrex::Real time,
        bool split_pml_field = false);

    /**
     * \brief Apply quartz boundary condition on the B-field.
     *        For quartz, the magnetic field boundary condition is similar to PEC
     *        since quartz is non-magnetic (mu_r ~ 1).
     *
     * \param[in,out] Bfield
     * \param[in]     field_boundary_lo   lower field boundary conditions
     * \param[in]     field_boundary_hi   upper field boundary conditions
     * \param[in]     ng_fieldgather      number of guard cells used by field gather
     * \param[in]     geom                geometry object of level "lev"
     * \param[in]     lev                 level of the Multifab
     * \param[in]     patch_type          coarse or fine
     * \param[in]     ref_ratios          vector containing the refinement ratios of the refinement levels
     * \param[in]     time                current time of the simulation
     */
    void ApplyQuartztoBfield(
        std::array<amrex::MultiFab*,3> Bfield,
        amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_lo,
        amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const & field_boundary_hi,
        amrex::IntVect const & ng_fieldgather, amrex::Geometry const & geom,
        int lev, PatchType patch_type, amrex::Vector<amrex::IntVect> const & ref_ratios,
        amrex::Real time);

    /**
     * \brief Read quartz boundary parameters from input file
     */
    void ReadParameters();

    /**
     * \brief Check if a point is inside the quartz ring geometry
     *        This defines the ring-shaped geometry for CCP applications
     */
    bool IsInsideQuartzRing(amrex::Real x, amrex::Real y, amrex::Real z) const;

private:

    /** Relative permittivity of quartz (default: 3.8) */
    amrex::Real m_epsilon_r = 3.8;
    
    /** Relative permeability of quartz (default: 1.0) */
    amrex::Real m_mu_r = 1.0;
    
    /** Conductivity of quartz (default: 0.0) */
    amrex::Real m_sigma = 0.0;
    
    /** Whether to use space-dependent properties */
    bool m_use_space_dependent = false;
    
    /** Parser for space-dependent epsilon_r */
    std::unique_ptr<amrex::Parser> m_epsilon_r_parser;
    
    /** Parser for space-dependent mu_r */
    std::unique_ptr<amrex::Parser> m_mu_r_parser;
    
    /** Parser for space-dependent sigma */
    std::unique_ptr<amrex::Parser> m_sigma_parser;

    /** String for epsilon_r function */
    std::string m_str_epsilon_r_function;
    
    /** String for mu_r function */
    std::string m_str_mu_r_function;
    
    /** String for sigma function */
    std::string m_str_sigma_function;
    
    /** Quartz ring geometry parameters for CCP applications */
    amrex::Real m_ring_inner_radius = 0.05;    // Inner radius of the ring (m)
    amrex::Real m_ring_outer_radius = 0.08;    // Outer radius of the ring (m)
    amrex::Real m_ring_height = 0.02;          // Height of the ring (m)
    amrex::Real m_ring_center_x = 0.0;         // Center x-coordinate (m)
    amrex::Real m_ring_center_y = 0.0;         // Center y-coordinate (m)
    amrex::Real m_ring_bottom_z = 0.0;         // Bottom z-coordinate (m)
    
    /** Parser for custom ring geometry */
    std::unique_ptr<amrex::Parser> m_ring_geometry_parser;
    std::string m_str_ring_geometry_function;
};

#endif // QUARTZ_H_ 