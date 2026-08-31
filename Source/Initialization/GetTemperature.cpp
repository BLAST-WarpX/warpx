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
    if (m_type == TemperatureInitType::TempConstantValue) {
        m_temperature = temp.m_temperature;
    }
    else if (m_type == TemperatureInitType::TempParserFunction) {
        m_temperature_parser = temp.m_ptr_temperature_parser->compile<3>();
    }
}

// Constructor for three-component (vector) temperature
GetTemperatureVector::GetTemperatureVector (TemperatureProperties const& temp) noexcept
    : m_type{temp.m_type}
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    , m_from_file{temp.m_u_std_x_reader.get(), temp.m_u_std_y_reader.get(),
                  temp.m_u_std_z_reader.get()}
#endif
{
    if (m_type == TemperatureInitType::TempConstantVector) {
        m_ux_std = temp.m_ux_std;
        m_uy_std = temp.m_uy_std;
        m_uz_std = temp.m_uz_std;
    }
    else if (m_type == TemperatureInitType::TempParserFunctionVector) {
        m_ux_std_parser = temp.m_ptr_ux_std_parser->compile<3>();
        m_uy_std_parser = temp.m_ptr_uy_std_parser->compile<3>();
        m_uz_std_parser = temp.m_ptr_uz_std_parser->compile<3>();
    }
}
