/* Copyright 2021 Hannah Klion
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "GetTemperature.H"

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
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    , m_u_std_x_reader{temp.m_u_std_x_reader.get()}
    , m_u_std_y_reader{temp.m_u_std_y_reader.get()}
    , m_u_std_z_reader{temp.m_u_std_z_reader.get()}
#endif
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
}

void GetTemperatureVector::prepare (amrex::BoxArray const& grids,
                                    amrex::DistributionMapping const& dmap,
                                    amrex::IntVect const& ngrow,
                                    std::function<amrex::Real(amrex::Real)> const& get_zlab)
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    if (needPreparation()) {
        m_u_std_x_reader->prepare(grids, dmap, ngrow, get_zlab);
        m_u_std_y_reader->prepare(grids, dmap, ngrow, get_zlab);
        m_u_std_z_reader->prepare(grids, dmap, ngrow, get_zlab);
        m_from_file = ExternalFieldVectorView(m_u_std_x_reader, m_u_std_y_reader,
                                              m_u_std_z_reader);
    }
#else
    amrex::ignore_unused(grids, dmap, ngrow, get_zlab);
#endif
}

void GetTemperatureVector::prepare (amrex::RealBox const& pbox, int moving_dir, int moving_sign,
                                    std::function<amrex::Real(amrex::Real)> const& get_zlab)
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    if (needPreparation()) {
        m_u_std_x_reader->prepare(pbox, moving_dir, moving_sign, get_zlab);
        m_u_std_y_reader->prepare(pbox, moving_dir, moving_sign, get_zlab);
        m_u_std_z_reader->prepare(pbox, moving_dir, moving_sign, get_zlab);
        m_from_file = ExternalFieldVectorView(m_u_std_x_reader, m_u_std_y_reader,
                                              m_u_std_z_reader);
    }
#else
    amrex::ignore_unused(pbox, moving_dir, moving_sign, get_zlab);
#endif
}

void GetTemperatureVector::prepare (int li)
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    if (needPreparation()) {
        m_from_file = ExternalFieldVectorView(m_u_std_x_reader, m_u_std_y_reader,
                                              m_u_std_z_reader, li);
    }
#else
    amrex::ignore_unused(li);
#endif
}

bool GetTemperatureVector::needPreparation () const
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    return m_type == TempFromFileVector;
#else
    return false;
#endif
}

bool GetTemperatureVector::distributed () const
{
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    if (needPreparation()) {
        return m_u_std_x_reader->distributed();
    }
#endif
    return false;
}
