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

#include <cmath>
#include <functional>

/** Get velocity at a point on the grid
 *
 * Class to get velocity at a point on the grid, either constant (m_velocity)
 * or a spatially varying value computed using the parser function (m_velocity_parser).
 * It also stores the direction of the velocity field. It provides the velocity information
 * held by the VelocityProperties instance passed to the constructor.
 */
struct GetVelocity
{
    /* Type of velocity initialization */
    VelocityInitType m_type;

    /* Velocity direction */
    int m_dir; //! Index x=0, y=1, z=2
    int m_sign_dir; //! Sign of the velocity direction positive=1, negative=-1

    /** Constant velocity value, if m_type == VelConstantValue */
    amrex::Real m_velocity{0};
    /** Velocity parser function, if m_type == VelParserFunction */
    amrex::ParserExecutor<3> m_velocity_parser;

    /**
     * \brief Construct the functor with information provided by vel
     *
     * \param[in] vel: const reference to the VelocityProperties object that will be used to
     * populate the functor
     */
    explicit GetVelocity (VelocityProperties const& vel) noexcept;

    /**
     * \brief Functor call. Returns the value of velocity at the location (x,y,z)
     *
     * \param[in] x x-coordinate of given location
     * \param[in] y y-coordinate of given location
     * \param[in] z z-cooridnate of given location
     *
     *\return value of velocity at (x,y,z).
     *        m_velocity if m_type is VelConstantValue
     *        m_velocity_parser(x,y,z) if m_type is VelParserFunction
     */
    AMREX_GPU_HOST_DEVICE
    amrex::Real operator() (amrex::Real const x, amrex::Real const y, amrex::Real const z) const noexcept
    {
        switch (m_type)
        {
            case (VelConstantValue):
            {
                return m_sign_dir * m_velocity;
            }
            case (VelParserFunction):
            {
                return m_sign_dir * m_velocity_parser(x,y,z);
            }
            default:
            {
                amrex::Abort("Get initial velocity: unknown type");
                return 0.0;
            }
        }
    }

    /**
     * \brief Returns the index of the direction of the bulk velocity
     *
     *\return index of direction of velocity.
     *        0: x
     *        1: y
     *        2: z
     */
    [[nodiscard]]
    AMREX_GPU_HOST_DEVICE
    int direction () const noexcept
    {
        return m_dir;
    }
};

struct GetVelocityVector
{
    /* Type of velocity initialization */
    VelocityInitType m_type;

    /** Constant velocity value, if m_type == VelConstantVector */
    amrex::Real m_ux_mean{0}, m_uy_mean{0}, m_uz_mean{0};
    /** Velocity parser function, if m_type == VelParserFunctionVector */
    amrex::ParserExecutor<3> m_ux_mean_parser, m_uy_mean_parser, m_uz_mean_parser;

#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    ExternalFieldView m_ux_mean_view;
    ExternalFieldView m_uy_mean_view;
    ExternalFieldView m_uz_mean_view;
    ExternalFieldReader* m_ux_mean_reader{nullptr};
    ExternalFieldReader* m_uy_mean_reader{nullptr};
    ExternalFieldReader* m_uz_mean_reader{nullptr};
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
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
        return m_type == VelFromFileVector;
#else
        return false;
#endif
    }

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
     * \brief Functor call. Returns the value of velocity at the location (x,y,z)
     *
     * Defined inline so HIP/CUDA device code is emitted in TUs that launch kernels
     * capturing this functor (device link cannot resolve out-of-line .cpp definitions).
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
#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
        case (VelFromFileVector):
        {
#if (AMREX_SPACEDIM < 3)
            amrex::ignore_unused(x, y, z);
#endif
#if (AMREX_SPACEDIM == 1)
            amrex::RealVect const pos{z};
#elif defined(WARPX_DIM_RZ)
            amrex::RealVect const pos{std::sqrt(x * x + y * y), z};
#elif defined(WARPX_DIM_XZ)
            amrex::RealVect const pos{x, z};
#else
            amrex::RealVect const pos{x, y, z};
#endif
            return amrex::XDim3{m_ux_mean_view(pos), m_uy_mean_view(pos), m_uz_mean_view(pos)};
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
