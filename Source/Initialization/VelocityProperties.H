/* Copyright 2021 Hannah Klion
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_VELOCITY_PROPERTIES_H_
#define WARPX_VELOCITY_PROPERTIES_H_

#include <AMReX_Geometry.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>

#include <memory>
#include <string>

/* Type of velocity initialization. Used by VelocityProperties and GetVelocityVector. */
enum VelocityInitType {VelConstantVector, VelParserFunctionVector, VelFromFileVector};

/**
 * \brief Struct to store velocity properties, for use in momentum initialization.
 *
 * Reads in and stores velocity used to initialize the Maxwellian and Maxwell-Juttner
 * momentum distributions in InjectorMomentum. The information is read from the parameters of
 * the species being initialized, and will be accessed by GetVelocityVector.
 */
struct VelocityProperties
{
    /**
     * \brief Read runtime parameters to populate constant or spatially-varying velocity
     * information
     *
     * Construct VelocityProperties based on the passed parameters.
     * If velocity is a constant, store value. If a parser, make and
     * store the parser function
     *
     * \param[in] pp: Reference to the parameter parser object for the species being initialized
     * \param[in] source_name: Optional group name of the input parameters
     * \param[in] geom: Simulation geometry (for aligning openPMD external fields)
     */
    VelocityProperties (const amrex::ParmParse& pp, std::string const& source_name,
                        amrex::Geometry const& geom);

    [[nodiscard]] bool needPreparation () const noexcept
    {
        return m_type == VelFromFileVector;
    }

    /* Type of velocity initialization */
    VelocityInitType m_type;

    /* Mean velocity components for velocity vector, if m_type == VelConstantVector */
    amrex::Real m_ux_mean{0.0}, m_uy_mean{0.0}, m_uz_mean{0.0};
    /* Storage of the parser functions for velocity vector, if m_type == VelParserFunctionVector */
    std::unique_ptr<amrex::Parser> m_ptr_ux_mean_parser, m_ptr_uy_mean_parser, m_ptr_uz_mean_parser;

    std::string m_read_u_mean_path;
    bool m_read_u_mean_distributed{true};

    amrex::Geometry m_geom;
};

#endif //WARPX_VELOCITY_PROPERTIES_H_
