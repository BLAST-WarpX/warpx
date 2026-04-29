/* Copyright 2021 Hannah Klion
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "GetVelocity.H"

#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
namespace
{
    void prepareExternalFieldReader (
        ExternalFieldReader* reader,
        amrex::BoxArray const& grids,
        amrex::DistributionMapping const& dmap,
        amrex::IntVect const& ngrow,
        std::function<amrex::Real(amrex::Real)> const& get_zlab,
        ExternalFieldView& view)
    {
        if (reader == nullptr) {
            return;
        }
        reader->prepare(grids, dmap, ngrow, get_zlab);
        view = reader->getView();
    }

    void prepareExternalFieldReader (
        ExternalFieldReader* reader,
        amrex::RealBox const& pbox,
        int moving_dir,
        int moving_sign,
        std::function<amrex::Real(amrex::Real)> const& get_zlab,
        ExternalFieldView& view)
    {
        if (reader == nullptr) {
            return;
        }
        reader->prepare(pbox, moving_dir, moving_sign, get_zlab);
        view = reader->getView();
    }
}
#endif

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

GetVelocityVector::GetVelocityVector (VelocityProperties const& vel) noexcept
    : m_type{vel.m_type}
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
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    else if (m_type == VelFromFileVector) {
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const problo = vel.m_geom.ProbLoArray();
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const pdx = vel.m_geom.CellSizeArray();
        amrex::Box const dombox = amrex::convert(vel.m_geom.Domain(), amrex::IntVect(1));
        bool const distributed = vel.m_read_u_mean_distributed;
        m_ux_mean_reader = std::make_shared<ExternalFieldReader>(
            vel.m_read_ux_mean_path, vel.m_ux_mean_openpmd_mesh, "", problo, pdx, dombox, distributed);
        m_uy_mean_reader = std::make_shared<ExternalFieldReader>(
            vel.m_read_uy_mean_path, vel.m_uy_mean_openpmd_mesh, "", problo, pdx, dombox, distributed);
        m_uz_mean_reader = std::make_shared<ExternalFieldReader>(
            vel.m_read_uz_mean_path, vel.m_uz_mean_openpmd_mesh, "", problo, pdx, dombox, distributed);
    }
#endif
}

void GetVelocityVector::prepare (
    amrex::BoxArray const& grids,
    amrex::DistributionMapping const& dmap,
    amrex::IntVect const& ngrow,
    std::function<amrex::Real(amrex::Real)> const& get_zlab)
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    if (m_type != VelFromFileVector) {
        return;
    }
    prepareExternalFieldReader(m_ux_mean_reader.get(), grids, dmap, ngrow, get_zlab, m_ux_mean_view);
    prepareExternalFieldReader(m_uy_mean_reader.get(), grids, dmap, ngrow, get_zlab, m_uy_mean_view);
    prepareExternalFieldReader(m_uz_mean_reader.get(), grids, dmap, ngrow, get_zlab, m_uz_mean_view);
#else
    amrex::ignore_unused(grids, dmap, ngrow, get_zlab);
#endif
}

void GetVelocityVector::prepare (
    amrex::RealBox const& pbox, int moving_dir, int moving_sign,
    std::function<amrex::Real(amrex::Real)> const& get_zlab)
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    if (m_type != VelFromFileVector) {
        return;
    }
    prepareExternalFieldReader(
        m_ux_mean_reader.get(), pbox, moving_dir, moving_sign, get_zlab, m_ux_mean_view);
    prepareExternalFieldReader(
        m_uy_mean_reader.get(), pbox, moving_dir, moving_sign, get_zlab, m_uy_mean_view);
    prepareExternalFieldReader(
        m_uz_mean_reader.get(), pbox, moving_dir, moving_sign, get_zlab, m_uz_mean_view);
#else
    amrex::ignore_unused(pbox, moving_dir, moving_sign, get_zlab);
#endif
}

void GetVelocityVector::prepare (int li)
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    if (m_type != VelFromFileVector) {
        return;
    }
    if (m_ux_mean_reader) {
        m_ux_mean_view = m_ux_mean_reader->getView(li);
    }
    if (m_uy_mean_reader) {
        m_uy_mean_view = m_uy_mean_reader->getView(li);
    }
    if (m_uz_mean_reader) {
        m_uz_mean_view = m_uz_mean_reader->getView(li);
    }
#else
    amrex::ignore_unused(li);
#endif
}

void GetVelocityVector::clear ()
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    m_ux_mean_reader.reset();
    m_uy_mean_reader.reset();
    m_uz_mean_reader.reset();
#endif
}

bool GetVelocityVector::distributed () const noexcept
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    if (m_type != VelFromFileVector || !m_ux_mean_reader) {
        return false;
    }
    return m_ux_mean_reader->distributed();
#else
    return false;
#endif
}
