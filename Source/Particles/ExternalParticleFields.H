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
//  * \brief This struct contains the meta data needed to describe a single external particle field:
//  * The name of the field, the path to the file, the time dependency function,
//. * and then the parser for the time depenendcy function
//  *
//  */
struct ParticleFieldMetaData {
    std::string name;               // e.g., "constant_field"
    std::string path;               // openPMD file path
    std::string time_function = "1.0";
    std::unique_ptr<amrex::Parser> time_parser;   // owns the parser
    amrex::ParserExecutor<1> time_executor;       // compiled f(t)
};

// /**
//  * \brief This class contains the parameters for the external particle fields.
//  *
//  */
class ExternalParticleFields {
public:

    std::vector<ParticleFieldMetaData> m_E_field_metadata;
    std::vector<ParticleFieldMetaData> m_B_field_metadata;
    int m_nEfields;
    int m_nBfields;

  void ReadParameters();
};

#endif // WARPX_EXTERNAL_PARTICLE_FIELDS_H_
