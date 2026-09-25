/* Copyright 2021 Hannah Klion
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_GET_TEMPERATURE_H_
#define WARPX_GET_TEMPERATURE_H_

#include "ExternalField.H"
#include "TemperatureProperties.H"

#include <AMReX.H>

#include <cmath>

/** Convert a scalar temperature in eV to an isotropic u_std vector. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::XDim3
ConvertTemperatureInEvToUStd (amrex::Real const T_eV,
                              amrex::Real const q_e_over_mc2) noexcept
{
    if (T_eV < 0.0) {
        amrex::Abort("The Maxwellian temperature in eV must be non-negative.");
    }
    const amrex::Real u_std = std::sqrt(T_eV * q_e_over_mc2);
    return amrex::XDim3{u_std, u_std, u_std};
}

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
struct GetTemperatureVector
{
    /* Type of temperature initialization */
    TemperatureInitType m_type;

    amrex::Real m_ux_std{0}, m_uy_std{0}, m_uz_std{0};
    amrex::ParserExecutor<3> m_ux_std_parser, m_uy_std_parser, m_uz_std_parser;
    GetTemperature m_temperature;
    amrex::Real m_q_e_over_mc2{0}; // q_e / (m c^2)
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    ExternalFieldVectorView m_from_file; // Vector openPMD u_std field, if m_type == TempFromFileVector
    ExternalFieldView m_temperature_in_eV_from_file; // scalar openPMD temperature_in_eV field, if m_type == TempFromFileValue
#endif
    /**
     * \brief Construct the functor with information provided by temp
     *
     * \param[in] temp: const reference to the TemperatureProperties object that will be used to
     * populate the functor
     */
    explicit GetTemperatureVector (TemperatureProperties const& temp) noexcept;

    /**
     * \brief Returns the thermal spread as normalized momentum standard deviations
     *        (ux_std, uy_std, uz_std) in the drift frame at (x,y,z).
     *
     * \param[in] x: x-coordinate of given location
     * \param[in] y: y-coordinate of given location
     * \param[in] z: z-cooridnate of given location
     *
     * \return amrex::XDim3{ux_std, uy_std, uz_std} — isotropic or anisotropic
     * depending on m_type (direct u_std, or converted from temperature_in_eV).
     */
    AMREX_GPU_HOST_DEVICE
    amrex::XDim3 operator() (amrex::Real const x, amrex::Real const y, amrex::Real const z) const noexcept
    {
        switch (m_type)
        {
            case (TempConstantVector):
            {
                return amrex::XDim3{m_ux_std, m_uy_std, m_uz_std};
            }
            case (TempParserFunctionVector):
            {
                return amrex::XDim3{m_ux_std_parser(x,y,z), m_uy_std_parser(x,y,z), m_uz_std_parser(x,y,z)};
            }
            case (TempConstantValue):
            case (TempParserFunction):
            {
                return ConvertTemperatureInEvToUStd(
                    m_temperature(x,y,z), m_q_e_over_mc2);
            }
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
            case (TempFromFileVector):
            {
                return m_from_file.getValue(x, y, z);
            }
            case (TempFromFileValue):
            {
#if (AMREX_SPACEDIM < 3)
                amrex::ignore_unused(x,y,z);
#endif
#if (AMREX_SPACEDIM == 1)
                amrex::RealVect const pos{z};
#elif defined(WARPX_DIM_XZ)
                amrex::RealVect const pos{x,z};
#else
                amrex::RealVect const pos{x,y,z};
#endif
#if defined(AMREX_USE_GPU)
                AMREX_IF_ON_HOST((amrex::Abort(
                    "For GPU builds, TempFromFileValue only works on device.");))
                AMREX_IF_ON_DEVICE((return ConvertTemperatureInEvToUStd(
                    m_temperature_in_eV_from_file(pos), m_q_e_over_mc2);))
                return amrex::XDim3{0.0, 0.0, 0.0};
#else
                return ConvertTemperatureInEvToUStd(
                    m_temperature_in_eV_from_file(pos), m_q_e_over_mc2);
#endif
            }
#endif
            default:
            {
                amrex::Abort("Get initial temperature: unknown type");
                return amrex::XDim3{0.0, 0.0, 0.0};
            }
        }
    }
};
#endif //WARPX_GET_TEMPERATURE_H_
