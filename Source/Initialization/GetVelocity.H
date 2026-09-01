/* Copyright 2021 Hannah Klion
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_GET_VELOCITY_H_
#define WARPX_GET_VELOCITY_H_

#include "ExternalField.H"
#include "VelocityProperties.H"

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Dim3.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_REAL.H>

#include <functional>

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

#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    /** View on the openPMD data, if m_type == VelFromFileVector */
    ExternalFieldVectorView m_from_file;
    /** Readers of the three components of `u_mean`. These are owned by the
     * VelocityProperties object that was passed to the constructor. */
    ExternalFieldReader* m_u_mean_x_reader = nullptr;
    ExternalFieldReader* m_u_mean_y_reader = nullptr;
    ExternalFieldReader* m_u_mean_z_reader = nullptr;
#endif

    /**
     * \brief Construct the functor with information provided by vel
     *
     * \param[in] vel: const reference to the VelocityProperties object that will be used to
     * populate the functor
     */
    explicit GetVelocityVector (VelocityProperties const& vel) noexcept;

    // This function needs to be called before initializing the momentum of the
    // particles, when `u_mean` is read from a file. When the readers are
    // created in the constructor of VelocityProperties, we do not know the
    // particle containers' BoxArray and DistributionMapping yet.
    void prepare (amrex::BoxArray const& grids,
                  amrex::DistributionMapping const& dmap,
                  amrex::IntVect const& ngrow,
                  std::function<amrex::Real(amrex::Real)> const& get_zlab);

    // This function needs to be called before we do continuous injection of
    // particles to inform the readers to load the data we need.
    void prepare (amrex::RealBox const& pbox, int moving_dir, int moving_sign,
                  std::function<amrex::Real(amrex::Real)> const& get_zlab);

    // For distributed data, this function prepares the functor for reading
    // data in this Box. Here `li` is the local index for the Box in a
    // particle container.
    void prepare (int li);

    [[nodiscard]] bool needPreparation () const;

    [[nodiscard]] bool distributed () const;

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
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
            case (VelFromFileVector):
            {
                return m_from_file.getValue(x, y, z);
            }
#endif
            default:
            {
                amrex::Abort("Get initial velocity: unknown type");
                return amrex::XDim3{0.0, 0.0, 0.0};
            }
        }
    }
};

#endif //WARPX_GET_VELOCITY_H_
