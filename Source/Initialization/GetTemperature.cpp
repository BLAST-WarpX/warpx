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
