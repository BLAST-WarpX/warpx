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
#include <AMReX_Box.H>
#include <AMReX_Dim3.H>
#include <AMReX_BoxArray.H>
#include <AMReX_RealBox.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_IntVect.H>
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
    ExternalFieldVectorFromFile m_from_file;
#endif

    /**
     * \brief Construct the functor with information provided by vel
     *
     * \param[in] vel: const reference to the VelocityProperties object that will be used to
     * populate the functor
     */
    explicit GetVelocityVector (VelocityProperties const& vel);

    //! True when bulk mean velocity is loaded from openPMD (cf. ``InjectorDensity`` ``fromfile``).
    [[nodiscard]] bool needPreparation () const noexcept
    {
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RZ) && \
    !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
        return m_type == VelFromFileVector;
#else
        return false;
#endif
    }

    // When u_mean is loaded from openPMD (VelFromFileVector), prepare() must be
    // called before particle init or continuous injection if data are distributed,
    // because the particle BoxArray and DistributionMapping are not known at
    // construction time.
    // Does not take get_zlab (see GetTemperatureVector::prepare).
    void prepare (amrex::BoxArray const& grids,
                  amrex::DistributionMapping const& dmap,
                  amrex::IntVect const& ngrow);

    void prepare (amrex::RealBox const& pbox, int moving_dir, int moving_sign);

    void prepare (int li);

    void clear ();

    [[nodiscard]] bool distributed () const noexcept;

    /**
     * \brief Returns the velocity vector at the location (x,y,z)
     *
     * This definition is in the header because this function is called from GPU kernels.
     */
    AMREX_GPU_HOST_DEVICE amrex::XDim3
    operator() (amrex::Real const x, amrex::Real const y, amrex::Real const z) const noexcept
    {
        switch (m_type)
        {
        case (VelConstantVector):
        {
            return amrex::XDim3{m_ux_mean, m_uy_mean, m_uz_mean};
        }
        case (VelParserFunctionVector):
        {
            return amrex::XDim3{m_ux_mean_parser(x, y, z), m_uy_mean_parser(x, y, z),
                                m_uz_mean_parser(x, y, z)};
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
            amrex::Abort("Get initial velocity (vector): unknown type");
            return amrex::XDim3{0.0, 0.0, 0.0};
        }
        }
    }
};

#endif //WARPX_GET_VELOCITY_H_
