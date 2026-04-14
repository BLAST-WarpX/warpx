/* Copyright 2021 Hannah Klion
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "GetTemperature.H"

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

// Constructor for single-component (scalar) temperature
GetTemperature::GetTemperature (TemperatureProperties const& temp) noexcept
    : m_type{temp.m_type}
{
    if (m_type == TempConstantValue) {
        m_temperature = temp.m_temperature;
    }
    else if (m_type == TempParserFunction) {
        m_temperature_parser = temp.m_ptr_temperature_parser->compile<3>();
    }
}

// Constructor for three-component (vector) temperature
GetTemperatureVector::GetTemperatureVector (TemperatureProperties const& temp) noexcept
    : m_type{temp.m_type}
{
    if (m_type == TempConstantVector) {
        m_ux_std = temp.m_ux_std;
        m_uy_std = temp.m_uy_std;
        m_uz_std = temp.m_uz_std;
    }
    else if (m_type == TempParserFunctionVector) {
        m_ux_std_parser = temp.m_ptr_ux_std_parser->compile<3>();
        m_uy_std_parser = temp.m_ptr_uy_std_parser->compile<3>();
        m_uz_std_parser = temp.m_ptr_uz_std_parser->compile<3>();
    }
    else if (m_type == TempParserScalarTeV) {
        m_species_mass = temp.m_species_mass;
        m_T_eV_parser = temp.m_ptr_T_eV_parser->compile<3>();
    }
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    else if (m_type == TempFromFileVector) {
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const problo = temp.m_geom.ProbLoArray();
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const pdx = temp.m_geom.CellSizeArray();
        amrex::Box const dombox = amrex::convert(temp.m_geom.Domain(), amrex::IntVect(1));
        bool const distributed = temp.m_read_u_std_distributed;
        m_ux_std_reader = new ExternalFieldReader(
            temp.m_read_ux_std_path, temp.m_ux_std_openpmd_mesh, "", problo, pdx, dombox, distributed);
        m_uy_std_reader = new ExternalFieldReader(
            temp.m_read_uy_std_path, temp.m_uy_std_openpmd_mesh, "", problo, pdx, dombox, distributed);
        m_uz_std_reader = new ExternalFieldReader(
            temp.m_read_uz_std_path, temp.m_uz_std_openpmd_mesh, "", problo, pdx, dombox, distributed);
    }
#endif
}

void GetTemperatureVector::prepare (
    amrex::BoxArray const& grids,
    amrex::DistributionMapping const& dmap,
    amrex::IntVect const& ngrow,
    std::function<amrex::Real(amrex::Real)> const& get_zlab)
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    if (m_type != TempFromFileVector) {
        return;
    }
    prepareExternalFieldReader(m_ux_std_reader, grids, dmap, ngrow, get_zlab, m_ux_std_view);
    prepareExternalFieldReader(m_uy_std_reader, grids, dmap, ngrow, get_zlab, m_uy_std_view);
    prepareExternalFieldReader(m_uz_std_reader, grids, dmap, ngrow, get_zlab, m_uz_std_view);
#else
    amrex::ignore_unused(grids, dmap, ngrow, get_zlab);
#endif
}

void GetTemperatureVector::prepare (
    amrex::RealBox const& pbox, int moving_dir, int moving_sign,
    std::function<amrex::Real(amrex::Real)> const& get_zlab)
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    if (m_type != TempFromFileVector) {
        return;
    }
    prepareExternalFieldReader(
        m_ux_std_reader, pbox, moving_dir, moving_sign, get_zlab, m_ux_std_view);
    prepareExternalFieldReader(
        m_uy_std_reader, pbox, moving_dir, moving_sign, get_zlab, m_uy_std_view);
    prepareExternalFieldReader(
        m_uz_std_reader, pbox, moving_dir, moving_sign, get_zlab, m_uz_std_view);
#else
    amrex::ignore_unused(pbox, moving_dir, moving_sign, get_zlab);
#endif
}

void GetTemperatureVector::prepare (int li)
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    if (m_type != TempFromFileVector) {
        return;
    }
    if (m_ux_std_reader) {
        m_ux_std_view = m_ux_std_reader->getView(li);
    }
    if (m_uy_std_reader) {
        m_uy_std_view = m_uy_std_reader->getView(li);
    }
    if (m_uz_std_reader) {
        m_uz_std_view = m_uz_std_reader->getView(li);
    }
#else
    amrex::ignore_unused(li);
#endif
}

void GetTemperatureVector::clear ()
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    delete m_ux_std_reader;
    m_ux_std_reader = nullptr;
    delete m_uy_std_reader;
    m_uy_std_reader = nullptr;
    delete m_uz_std_reader;
    m_uz_std_reader = nullptr;
#endif
}

bool GetTemperatureVector::distributed () const noexcept
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    if (m_type != TempFromFileVector || m_ux_std_reader == nullptr) {
        return false;
    }
    return m_ux_std_reader->distributed();
#else
    return false;
#endif
}
