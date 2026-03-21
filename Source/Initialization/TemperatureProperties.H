/* Copyright 2021 Hannah Klion
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_TEMPERATURE_PROPERTIES_H_
#define WARPX_TEMPERATURE_PROPERTIES_H_

#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>

/* Type of temperature initialization. Used by TemperatureProperties and GetTemperature. */
enum TemperatureInitType {TempConstantValue, TempParserFunction, TempParserFunctionVector, TempConstantVector};

/**
 * \brief Struct to store temperature properties, for use in momentum initialization.
 *
 * Reads in and stores temperature used to initialize the Maxwell-Boltzmann and Maxwell-Juttner
 * momentum distributions in InjectorMomentum. The information is read from the parameters of
 * the species being initialized, and will be accessed by GetTemperature.
 */
struct TemperatureProperties
{
    /**
     * \brief Read runtime parameters to populate constant or spatially-varying temperature
     * information
     *
     * \param[in] pp: Reference to the parameter parser object for the species being initialized
     * \param[in] source_name: Optional group name of the input parameters
     */
    TemperatureProperties (const amrex::ParmParse& pp, std::string const& source_name);

    /* Type of temperature initialization */
    TemperatureInitType m_type;

    /* Constant temperature value, if m_type == TempConstantValue */
    amrex::Real m_temperature;
    /* Storage of the parser function, if m_type == TempParserFunction */
    std::unique_ptr<amrex::Parser> m_ptr_temperature_parser;

    /* Standard deviations for temperature vector, if m_type == TempConstantVector */
    amrex::Real m_ux_std{0.0}, m_uy_std{0.0}, m_uz_std{0.0};
    /* Storage of the parser functions for temperature vector, if m_type == TempParserFunctionVector */
    std::unique_ptr<amrex::Parser> m_ptr_ux_std_parser, m_ptr_uy_std_parser, m_ptr_uz_std_parser;
};

#endif //WARPX_TEMPERATURE_PROPERTIES_H_
