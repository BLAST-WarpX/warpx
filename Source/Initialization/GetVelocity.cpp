/* Copyright 2021 Hannah Klion
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "GetVelocity.H"

#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
GetVelocityVectorFromFile::GetVelocityVectorFromFile (VelocityProperties const& vel)
{
    if (vel.m_type != VelFromFileVector) {
        return;
    }

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const problo = vel.m_geom.ProbLoArray();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const pdx = vel.m_geom.CellSizeArray();
    amrex::Box const dombox = amrex::convert(vel.m_geom.Domain(), amrex::IntVect(1));
    bool const distributed = vel.m_read_u_mean_distributed;
    m_ux_mean_reader = new ExternalFieldReader(
        vel.m_read_u_mean_path, "u_mean", "x", problo, pdx, dombox, distributed);
    m_uy_mean_reader = new ExternalFieldReader(
        vel.m_read_u_mean_path, "u_mean", "y", problo, pdx, dombox, distributed);
    m_uz_mean_reader = new ExternalFieldReader(
        vel.m_read_u_mean_path, "u_mean", "z", problo, pdx, dombox, distributed);
}

void GetVelocityVectorFromFile::clear ()
{
    delete m_ux_mean_reader;
    m_ux_mean_reader = nullptr;
    delete m_uy_mean_reader;
    m_uy_mean_reader = nullptr;
    delete m_uz_mean_reader;
    m_uz_mean_reader = nullptr;
}

void GetVelocityVectorFromFile::prepare (
    amrex::BoxArray const& grids,
    amrex::DistributionMapping const& dmap,
    amrex::IntVect const& ngrow)
{
    if (m_ux_mean_reader) {
        m_ux_mean_reader->prepare(grids, dmap, ngrow);
        m_ux_mean_view = m_ux_mean_reader->getView();
    }
    if (m_uy_mean_reader) {
        m_uy_mean_reader->prepare(grids, dmap, ngrow);
        m_uy_mean_view = m_uy_mean_reader->getView();
    }
    if (m_uz_mean_reader) {
        m_uz_mean_reader->prepare(grids, dmap, ngrow);
        m_uz_mean_view = m_uz_mean_reader->getView();
    }
}

void GetVelocityVectorFromFile::prepare (
    amrex::RealBox const& pbox, int moving_dir, int moving_sign)
{
    if (m_ux_mean_reader) {
        m_ux_mean_reader->prepare(pbox, moving_dir, moving_sign);
        m_ux_mean_view = m_ux_mean_reader->getView();
    }
    if (m_uy_mean_reader) {
        m_uy_mean_reader->prepare(pbox, moving_dir, moving_sign);
        m_uy_mean_view = m_uy_mean_reader->getView();
    }
    if (m_uz_mean_reader) {
        m_uz_mean_reader->prepare(pbox, moving_dir, moving_sign);
        m_uz_mean_view = m_uz_mean_reader->getView();
    }
}

void GetVelocityVectorFromFile::prepare (int li)
{
    if (m_ux_mean_reader) {
        m_ux_mean_view = m_ux_mean_reader->getView(li);
    }
    if (m_uy_mean_reader) {
        m_uy_mean_view = m_uy_mean_reader->getView(li);
    }
    if (m_uz_mean_reader) {
        m_uz_mean_view = m_uz_mean_reader->getView(li);
    }
}

bool GetVelocityVectorFromFile::distributed () const noexcept
{
    if (m_ux_mean_reader) {
        return m_ux_mean_reader->distributed();
    } else {
        return false;
    }
}
#endif

// Constructor for single-component (scalar) velocity
GetVelocity::GetVelocity (VelocityProperties const& vel) noexcept
    : m_type{vel.m_type}, m_dir{vel.m_dir}, m_sign_dir{vel.m_sign_dir}
{
    if (m_type == VelConstantValue) {
        m_velocity = vel.m_velocity;
    }
    else if (m_type == VelParserFunction) {
        m_velocity_parser = vel.m_ptr_velocity_parser->compile<3>();
    }
}

// Constructor for three-component (vector) velocity
GetVelocityVector::GetVelocityVector (VelocityProperties const& vel)
    : m_type{vel.m_type}
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    , m_from_file{vel}
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
