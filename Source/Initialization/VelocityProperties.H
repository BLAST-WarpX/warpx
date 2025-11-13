/* Copyright 2021 Hannah Klion
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_VELOCITY_PROPERTIES_H_
#define WARPX_VELOCITY_PROPERTIES_H_

#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>

/* Type of velocity initialization. Used by VelocityProperties and GetVelocity. */
enum VelocityInitType {VelConstantValue, VelParserFunction};

/**
 * \brief Struct to store velocity properties, for use in momentum initialization.
 *
 * Reads in and stores velocity used to initialize the Maxwell-Boltzmann and Maxwell-Juttner
 * momentum distributions in InjectorMomentum. The information is read from the parameters of
 * the species being initialized, and will be accessed by GetVelocity.
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
     */
    VelocityProperties (const amrex::ParmParse& pp, std::string const& source_name);

    /* Type of velocity initialization */
    VelocityInitType m_type;

    /* Velocity direction */
    int m_dir; // Index x=0, y=1, z=2
    int m_sign_dir; // Sign of the velocity direction positive=1, negative=-1

    /* Constant velocity value, if m_type == VelConstantValue */
    amrex::Real m_velocity{0};
    /* Storage of the parser function, if m_type == VelParserFunction */
    std::unique_ptr<amrex::Parser> m_ptr_velocity_parser;
};

#endif //WARPX_VELOCITY_PROPERTIES_H_
