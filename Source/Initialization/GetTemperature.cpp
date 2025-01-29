/* Copyright 2021 Hannah Klion
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "GetTemperature.H"

GetTemperature::GetTemperature (TemperatureProperties const& temp) noexcept :
    m_type{temp.m_type}
{
    if (m_type == TempConstantValue) {
        m_temperature = temp.m_temperature;
    }
    else if (m_type == TempParserFunction) {
        m_temperature_parser = temp.m_ptr_temperature_parser->compile<3>();
    }
}
