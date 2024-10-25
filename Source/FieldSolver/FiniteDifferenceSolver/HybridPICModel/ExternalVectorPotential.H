/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_EXTERNAL_VECTOR_POTENTIAL_H_
#define WARPX_EXTERNAL_VECTOR_POTENTIAL_H_

#include "Fields.H"

#include "Utils/WarpXAlgorithmSelection.H"

#include "EmbeddedBoundary/Covered.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXProfilerWrapper.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_Array.H>
#include <AMReX_REAL.H>
#include <AMReX_BoxArray.H>
#include <AMReX_IntVect.H>
#include <AMReX_DistributionMapping.H>

#include <optional>

using namespace amrex;

/**
 * \brief This class contains the parameters needed to evaluate a
 * time varying external vector potential, leading to external E/B
 * fields to be applied in Hybrid Solver. This class is used to break up
 * the passed in fields into a spatial and time dependent solution.
 * 
 * Eventually this can be used in a list to control independent external 
 * fields with different time profiles.
 * 
 */
class ExternalVectorPotential
{
protected:
    std::string m_Ax_ext_grid_function = "0.0";
    std::string m_Ay_ext_grid_function = "0.0";
    std::string m_Az_ext_grid_function = "0.0";
    std::array< std::unique_ptr<amrex::Parser>, 3> m_A_external_parser;
    std::array< amrex::ParserExecutor<4>, 3> m_A_external;

    std::string m_A_ext_time_function = "1.0";
    std::unique_ptr<amrex::Parser> m_A_external_time_parser;
    amrex::ParserExecutor<1> m_A_time_scale;

    bool m_read_A_from_file = false;
    std::string m_external_file_path = ""; 

public:


    // Default Constructor
    ExternalVectorPotential ();

    void ReadParameters (); 

    void AllocateLevelMFs (
        ablastr::fields::MultiFabRegister & fields,
        int lev, const BoxArray& ba, const DistributionMapping& dm,
        const int ncomps,
        const IntVect& ngEB,
        const IntVect& Ex_nodal_flag,
        const IntVect& Ey_nodal_flag,
        const IntVect& Ez_nodal_flag,
        const IntVect& Bx_nodal_flag,
        const IntVect& By_nodal_flag,
        const IntVect& Bz_nodal_flag
    );

    void InitData ();

    AMREX_FORCE_INLINE
    void ZeroFieldinEB (
        ablastr::fields::VectorField const& Field,
        EB::CoverTopology topology,
        const int lev);

    void UpdateHybridExternalFields (
        const amrex::Real t,
        const amrex::Real dt
    );
};

#endif //WARPX_TIME_DEPENDENT_VECTOR_POTENTIAL_H_