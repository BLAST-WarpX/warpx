/* Copyright 2021 Hannah Klion
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_GET_TEMPERATURE_H_
#define WARPX_GET_TEMPERATURE_H_

#include "TemperatureProperties.H"

/**
 * \brief Get temperature at a point on the grid
 *
 * Functor to return temperature at a point on the grid, either constant (m_temperature)
 * or a spatially varying value computed using the parser function (m_temperature_parser).
 * It provides the temperature information held by the TemperatureProperties instance
 * passed to the constructor.
 */
struct GetTemperature
{
    /* Type of temperature initialization */
    TemperatureInitType m_type;

    /* Constant temperature value, if m_type == TempConstantValue */
    amrex::Real m_temperature;
    /* Temperature parser function, if m_type == TempParserFunction */
    amrex::ParserExecutor<3> m_temperature_parser;

    /**
     * \brief Construct the functor with information provided by temp
     *
     * \param[in] temp: const reference to the TemperatureProperties object that will be used to
     * populate the functor
     */
    explicit GetTemperature (TemperatureProperties const& temp) noexcept;

    /**
     * \brief Functor call. Returns the value of temperature at the location (x,y,z)
     *
     * \param[in] x: x-coordinate of given location
     * \param[in] y: y-coordinate of given location
     * \param[in] z: z-cooridnate of given location
     *
     *\return value of temperature at (x,y,z).
     *        m_temperature if m_type is TempConstantValue
     *        m_temperature_parser(x,y,z) if m_type is TempParserFunction
     */
    AMREX_GPU_HOST_DEVICE
    amrex::Real operator() (amrex::Real const x, amrex::Real const y, amrex::Real const z) const noexcept
    {
        switch (m_type)
        {
            case (TempConstantValue):
            {
                return m_temperature;
            }
            case (TempParserFunction):
            {
                return m_temperature_parser(x,y,z);
            }
            default:
            {
                amrex::Abort("Get initial temperature: unknown type");
                return 0.0;
            }
        }
    }
};
#endif //WARPX_GET_TEMPERATURE_H_
