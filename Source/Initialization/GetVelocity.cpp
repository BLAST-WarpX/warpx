/* Copyright 2021 Hannah Klion
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "GetVelocity.H"

GetVelocityVector::GetVelocityVector (VelocityProperties const& vel) noexcept
    : m_type{vel.m_type}
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    , m_u_mean_x_reader{vel.m_u_mean_x_reader.get()}
    , m_u_mean_y_reader{vel.m_u_mean_y_reader.get()}
    , m_u_mean_z_reader{vel.m_u_mean_z_reader.get()}
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

void GetVelocityVector::prepare (amrex::BoxArray const& grids,
                                 amrex::DistributionMapping const& dmap,
                                 amrex::IntVect const& ngrow,
                                 std::function<amrex::Real(amrex::Real)> const& get_zlab)
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    if (needPreparation()) {
        m_u_mean_x_reader->prepare(grids, dmap, ngrow, get_zlab);
        m_u_mean_y_reader->prepare(grids, dmap, ngrow, get_zlab);
        m_u_mean_z_reader->prepare(grids, dmap, ngrow, get_zlab);
        m_from_file = ExternalFieldVectorView(m_u_mean_x_reader, m_u_mean_y_reader,
                                              m_u_mean_z_reader);
    }
#else
    amrex::ignore_unused(grids, dmap, ngrow, get_zlab);
#endif
}

void GetVelocityVector::prepare (amrex::RealBox const& pbox, int moving_dir, int moving_sign,
                                 std::function<amrex::Real(amrex::Real)> const& get_zlab)
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    if (needPreparation()) {
        m_u_mean_x_reader->prepare(pbox, moving_dir, moving_sign, get_zlab);
        m_u_mean_y_reader->prepare(pbox, moving_dir, moving_sign, get_zlab);
        m_u_mean_z_reader->prepare(pbox, moving_dir, moving_sign, get_zlab);
        m_from_file = ExternalFieldVectorView(m_u_mean_x_reader, m_u_mean_y_reader,
                                              m_u_mean_z_reader);
    }
#else
    amrex::ignore_unused(pbox, moving_dir, moving_sign, get_zlab);
#endif
}

void GetVelocityVector::prepare (int li)
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    if (needPreparation()) {
        m_from_file = ExternalFieldVectorView(m_u_mean_x_reader, m_u_mean_y_reader,
                                              m_u_mean_z_reader, li);
    }
#else
    amrex::ignore_unused(li);
#endif
}

bool GetVelocityVector::needPreparation () const
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    return m_type == VelFromFileVector;
#else
    return false;
#endif
}

bool GetVelocityVector::distributed () const
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    if (needPreparation()) {
        return m_u_mean_x_reader->distributed();
    }
#endif
    return false;
}
