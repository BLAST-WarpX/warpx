/* Copyright 2024 S. Eric Clark
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"

#include "EmbeddedBoundary/Covered.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"

#  include <AMReX.H>
#  include <AMReX_Array.H>
#  include <AMReX_Array4.H>
#  include <AMReX_Box.H>
#  include <AMReX_BoxArray.H>
#  include <AMReX_BoxList.H>
#  include <AMReX_Config.H>
#  include <AMReX_EB2.H>
#  include <AMReX_EB_utils.H>
#  include <AMReX_FabArray.H>
#  include <AMReX_FabFactory.H>
#  include <AMReX_GpuControl.H>
#  include <AMReX_GpuDevice.H>
#  include <AMReX_GpuQualifiers.H>
#  include <AMReX_IntVect.H>
#  include <AMReX_Loop.H>
#  include <AMReX_MFIter.H>
#  include <AMReX_MultiFab.H>
#  include <AMReX_iMultiFab.H>
#  include <AMReX_ParmParse.H>
#  include <AMReX_Parser.H>
#  include <AMReX_REAL.H>
#  include <AMReX_SPACE.H>
#  include <AMReX_Vector.H>

using namespace amrex;
using namespace warpx::fields;

namespace EB {

// Default Constructor
Covered::Covered (amrex::MFIter &mfi, int lev)
{
    if (EB::enabled()) {
        auto& warpx = WarpX::GetInstance();
        auto edge_lengths = warpx.m_fields.get_alldirs(FieldType::edge_lengths, lev);
        auto face_areas = warpx.m_fields.get_alldirs(FieldType::face_areas, lev);

        lx = edge_lengths[0]->array(mfi);
        ly = edge_lengths[1]->array(mfi);
        lz = edge_lengths[2]->array(mfi);
        Sx = face_areas[0]->array(mfi);
        Sy = face_areas[1]->array(mfi);
        Sz = face_areas[2]->array(mfi);

#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        lx_lo = amrex::lbound(lx);
        lx_hi = amrex::ubound(lx);
        lz_lo = amrex::lbound(lz);
        lz_hi = amrex::ubound(lz);
#endif
    }
}

}