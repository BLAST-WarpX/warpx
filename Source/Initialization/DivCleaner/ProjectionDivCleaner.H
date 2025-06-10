 /* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy, Inc.)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PROJECTION_DIV_CLEANER_H_
#define WARPX_PROJECTION_DIV_CLEANER_H_

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_LO_BCTYPES.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFInterp_C.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <AMReX_SPACE.H>
#include <AMReX_Vector.H>

#include "Fields.H"
#include "Utils/Parser/ParserUtils.H"

namespace warpx::initialization {

class ProjectionDivCleaner
{
protected:
    int m_levels = 1; // Hard coded to 1 for now, will only clean first level

    int m_ref_ratio = 1;

    // For MLMG solver
    int m_verbose = 2;
    int m_bottom_verbose = 0;
    int m_max_iter = 5000;
    int m_max_fmg_iter = 1000;
    int m_linop_maxorder = 3;
    bool m_agglomeration = false;
    bool m_consolidation = false;
    bool m_semicoarsening = true;
    int m_max_coarsening_level = 10;
    int m_max_semicoarsening_level = 10;
    amrex::BottomSolver m_bottom_solver = amrex::BottomSolver::bicgstab;


    amrex::Real m_rtol;
    amrex::Real m_atol;

    std::string m_field_name;

public:
    amrex::Vector< std::unique_ptr<amrex::MultiFab> > m_solution;
    amrex::Vector< std::unique_ptr<amrex::MultiFab> > m_source;

    amrex::Vector<amrex::Real> m_h_stencil_coefs_x;
    amrex::Vector<amrex::Real> m_h_stencil_coefs_y;
    amrex::Vector<amrex::Real> m_h_stencil_coefs_z;

    amrex::Gpu::DeviceVector<amrex::Real> m_stencil_coefs_x;
    amrex::Gpu::DeviceVector<amrex::Real> m_stencil_coefs_y;
    amrex::Gpu::DeviceVector<amrex::Real> m_stencil_coefs_z;

    // Default Constructor
    ProjectionDivCleaner (std::string const& a_field_name);

    void ReadParameters ();

    void solve ();
    void setSourceFromBfield ();
    void correctBfield ();

};

} // end namespace warpx::initialization

#endif // WARPX_PROJECTION_DIV_CLEANER_H_
