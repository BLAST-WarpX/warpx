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

// #include "Fields.H"

// #include "Utils/WarpXAlgorithmSelection.H"

// #include "EmbeddedBoundary/Enabled.H"
// #include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
// #include "Utils/Parser/ParserUtils.H"
// #include "Utils/WarpXConst.H"
// #include "Utils/WarpXProfilerWrapper.H"

// #include <ablastr/fields/MultiFabRegister.H>

// #include <AMReX_Array.H>
// #include <AMReX_REAL.H>
// #include <AMReX_BoxArray.H>
// #include <AMReX_IntVect.H>
// #include <AMReX_DistributionMapping.H>

// #include <optional>

#include <AMReX_Parser.H>
#include <string>
#include <vector>
#include <memory>

struct ParticleFieldMetaData {
    std::string name;               // e.g., "constant_field"
    std::string path;               // openPMD file path
    std::string time_function = "1.0";
    std::unique_ptr<amrex::Parser> time_parser;   // owns the parser
    amrex::ParserExecutor<1> time_executor;       // compiled f(t)
};

class ExternalParticleFields {
public:

    std::vector<ParticleFieldMetaData> m_E_field_metadata;
    std::vector<ParticleFieldMetaData> m_B_field_metadata;
    int m_nEfields;
    int m_nBfields;



  void ReadParameters();
};

// /**
//  * \brief This class contains the parameters needed to evaluate a
//  * time varying external vector potential, leading to external E/B
//  * fields to be applied in Hybrid Solver. This class is used to break up
//  * the passed in fields into a spatial and time dependent solution.
//  *
//  * Eventually this can be used in a list to control independent external
//  * fields with different time profiles.
//  *
//  */
// class ExternalVectorPotential
// {
// protected:
//     int m_nFields;

//     std::vector<std::string> m_field_names;

//     std::vector<std::string> m_Ax_ext_grid_function;
//     std::vector<std::string> m_Ay_ext_grid_function;
//     std::vector<std::string> m_Az_ext_grid_function;
//     std::vector<std::array< std::unique_ptr<amrex::Parser>, 3>> m_A_external_parser;
//     std::vector<std::array< amrex::ParserExecutor<4>, 3>> m_A_external;

//     std::vector<std::string> m_A_ext_time_function;
//     std::vector<std::unique_ptr<amrex::Parser>> m_A_external_time_parser;
//     std::vector<amrex::ParserExecutor<1>> m_A_time_scale;

//     std::vector<bool> m_read_A_from_file;
//     std::vector<std::string> m_external_file_path;

//     // Clean div A
//     bool m_do_clean_divA = true;

// public:

//     // Default Constructor
//     ExternalVectorPotential ();

//     void ReadParameters ();

//     void AllocateLevelMFs (
//         ablastr::fields::MultiFabRegister & fields,
//         int lev, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm,
//         int ncomps,
//         const amrex::IntVect& ngEB,
//         const amrex::IntVect& Ex_nodal_flag,
//         const amrex::IntVect& Ey_nodal_flag,
//         const amrex::IntVect& Ez_nodal_flag,
//         const amrex::IntVect& Bx_nodal_flag,
//         const amrex::IntVect& By_nodal_flag,
//         const amrex::IntVect& Bz_nodal_flag
//     );

//     void InitData ();

//     void CalculateExternalCurlA ();
//     void CalculateExternalCurlA (std::string& coil_name);

//     AMREX_FORCE_INLINE
//     void AddExternalFieldFromVectorPotential (
//         ablastr::fields::VectorField const& dstField,
//         amrex::Real scale_factor,
//         ablastr::fields::VectorField const& srcField,
//         std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update);

//     void UpdateHybridExternalFields (
//         amrex::Real t,
//         amrex::Real dt
//     );
// };

#endif // WARPX_EXTERNAL_PARTICLE_FIELDS_H_
