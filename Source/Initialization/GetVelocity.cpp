/* Copyright 2021 Hannah Klion
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "GetVelocity.H"

GetVelocityVector::GetVelocityVector (VelocityProperties const& vel)
    : m_type{vel.m_type}
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    , m_from_file{vel.m_read_u_mean_path, "u_mean", vel.m_geom, vel.m_read_u_mean_distributed,
                  vel.m_type == VelFromFileVector}
#endif
{
    if (m_type == VelConstantVector) {
        m_ux_mean = vel.m_ux_mean;
        m_uy_mean = vel.m_uy_mean;
        m_uz_mean = vel.m_uz_mean;
    }
    else if (m_type == VelParserFunctionVector) {
        m_ux_mean_parser = vel.m_ptr_ux_mean_parser->compile<3>();
        m_uy_mean_parser = vel.m_ptr_uy_mean_parser->compile<3>();
        m_uz_mean_parser = vel.m_ptr_uz_mean_parser->compile<3>();
    }
}

void GetVelocityVector::prepare (
    amrex::BoxArray const& grids,
    amrex::DistributionMapping const& dmap,
    amrex::IntVect const& ngrow)
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    if (m_type == VelFromFileVector) {
        m_from_file.prepare(grids, dmap, ngrow);
    }
#else
    amrex::ignore_unused(grids, dmap, ngrow);
#endif
}

void GetVelocityVector::prepare (
    amrex::RealBox const& pbox, int moving_dir, int moving_sign)
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    if (m_type == VelFromFileVector) {
        m_from_file.prepare(pbox, moving_dir, moving_sign);
    }
#else
    amrex::ignore_unused(pbox, moving_dir, moving_sign);
#endif
}

void GetVelocityVector::prepare (int li)
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    if (m_type == VelFromFileVector) {
        m_from_file.prepare(li);
    }
#else
    amrex::ignore_unused(li);
#endif
}

void GetVelocityVector::clear ()
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    if (m_type == VelFromFileVector) {
        m_from_file.clear();
    }
#endif
}

bool GetVelocityVector::distributed () const noexcept
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    if (m_type == VelFromFileVector) {
        return m_from_file.distributed();
    } else {
        return false;
    }
#else
    return false;
#endif
}
