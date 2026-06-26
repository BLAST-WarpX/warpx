/* Copyright 2021 Hannah Klion
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_GET_VELOCITY_H_
#define WARPX_GET_VELOCITY_H_

#include "VelocityProperties.H"

/** Get the bulk drift momentum vector at a point on the grid
 *
 * Class to get the bulk drift momentum vector (ux_mean, uy_mean, uz_mean) at a point on
 * the grid, either constant or a spatially varying value computed using parser functions.
 * It provides the velocity information held by the VelocityProperties instance passed to
 * the constructor, and is used by both the Maxwellian and Maxwell-Juttner distributions.
 */
struct GetVelocityVector
{
    /* Type of velocity initialization */
    VelocityInitType m_type;

    /** Constant velocity value, if m_type == VelConstantVector */
    amrex::Real m_ux_mean{0}, m_uy_mean{0}, m_uz_mean{0};
    /** Velocity parser function, if m_type == VelParserFunctionVector */
    amrex::ParserExecutor<3> m_ux_mean_parser, m_uy_mean_parser, m_uz_mean_parser;
    /**
     * \brief Construct the functor with information provided by vel
     *
     * \param[in] vel: const reference to the VelocityProperties object that will be used to
     * populate the functor
     */
    explicit GetVelocityVector (VelocityProperties const& vel) noexcept;

    /**
     * \brief Functor call. Returns the value of velocity at the location (x,y,z)
     *
     * \param[in] x x-coordinate of given location
     * \param[in] y y-coordinate of given location
     * \param[in] z z-cooridnate of given location
     *
     *\return value of velocity at (x,y,z).
     *        m_ux_mean, m_uy_mean, m_uz_mean if m_type is VelConstantVector
     *        m_ux_mean_parser(x,y,z), m_uy_mean_parser(x,y,z), m_uz_mean_parser(x,y,z) if m_type is VelParserFunctionVector
     */
    AMREX_GPU_HOST_DEVICE
    amrex::XDim3 operator() (amrex::Real const x, amrex::Real const y, amrex::Real const z) const noexcept
    {
        switch (m_type)
        {
            case (VelConstantVector):
            {
                return amrex::XDim3{ m_ux_mean, m_uy_mean, m_uz_mean };
            }
            case (VelParserFunctionVector):
            {
                return amrex::XDim3{ m_ux_mean_parser(x,y,z), m_uy_mean_parser(x,y,z), m_uz_mean_parser(x,y,z) };
            }
            default:
            {
                amrex::Abort("Get initial velocity: unknown type");
                return amrex::XDim3{0.0, 0.0, 0.0};
            }
        }
    }
};

#endif //WARPX_GET_VELOCITY_H_
