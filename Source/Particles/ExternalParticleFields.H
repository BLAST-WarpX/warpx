/* Copyright 2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Diederichs (CERN)
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_EXTERNAL_PARTICLE_FIELDS_H_
#define WARPX_EXTERNAL_PARTICLE_FIELDS_H_

#include <AMReX_Parser.H>
#include <string>
#include <vector>
#include <memory>

// /**
//  * \brief This struct contains the meta data needed to describe a single external particle
//  * field:
//  * The name of the field, the path to the file, the time dependency function,
//  * and then the parser for the time depenendcy function
//  *
//  */
struct ParticleFieldMetaData {
    //! Path to the openPMD file containing the field
    std::string path;
    //! Time dependency function as a string
    std::string time_function = "1.0";
    //! User-defined parser for time dependency of the field
    std::unique_ptr<amrex::Parser> time_parser;
    //! Compiled executor of the parser
    amrex::ParserExecutor<1> time_executor;
};

// /**
//  * \brief This class contains the parameters for the external particle fields.
//  *
//  */
class ExternalParticleFields {
public:

    //! Array of E field metadata
    std::vector<ParticleFieldMetaData> m_E_field_metadata;
    //! Array of B field metadata
    std::vector<ParticleFieldMetaData> m_B_field_metadata;
    //! Number of external E fields
    int m_nEfields;
    //! Number of external B fields
    int m_nBfields;

    /// @brief Read in the parameters via the parser and create the external particle
    // field meta data arrays
    void ReadParameters();
};

#endif // WARPX_EXTERNAL_PARTICLE_FIELDS_H_
