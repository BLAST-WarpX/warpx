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
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Dim3.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_IntVect.H>
#include <AMReX_RealBox.H>

#include "Utils/WarpXConst.H"

#include <functional>

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

#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    ExternalFieldVectorFromFile m_from_file;
#endif

    /**
     * \brief Construct the functor with information provided by temp
     *
     * \param[in] temp: const reference to the TemperatureProperties object that will be used to
     * populate the functor
     */
    explicit GetTemperatureVector (TemperatureProperties const& temp);

    [[nodiscard]] bool needPreparation () const noexcept
    {
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
        return m_type == TempFromFileVector;
#else
        return false;
#endif
    }

    // When u_std is loaded from openPMD (TempFromFileVector), same distributed-data
    // prepare requirement as GetVelocityVector.
    // get_zlab converts simulation-frame z coordinate to lab-frame z before reading
    // the file (needed in boosted-frame runs).
    // u_mean must already be prepared; see InjectorMomentumMaxwellian::prepare.
    void prepare (amrex::BoxArray const& grids,
                  amrex::DistributionMapping const& dmap,
                  amrex::IntVect const& ngrow,
                  std::function<amrex::Real(amrex::Real)> const& get_zlab);

    void prepare (amrex::RealBox const& pbox, int moving_dir, int moving_sign,
                  std::function<amrex::Real(amrex::Real)> const& get_zlab);

    void prepare (int li);

    void clear ();

    [[nodiscard]] bool distributed () const noexcept;

    /**
     * \brief Functor call. Returns the value of temperature at the location (x,y,z)
     *
     * \param[in] x: x-coordinate of given location
     * \param[in] y: y-coordinate of given location
     * \param[in] z: z-cooridnate of given location
     *
     *\return value of temperature at (x,y,z).
     *        m_ux_std, m_uy_std, m_uz_std if m_type is TempConstantVector
     *        m_ux_std_parser(x,y,z), m_uy_std_parser(x,y,z), m_uz_std_parser(x,y,z) if m_type is TempParserFunctionVector
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
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
            case (TempFromFileVector):
            {
                return m_from_file.getValue(x, y, z);
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
