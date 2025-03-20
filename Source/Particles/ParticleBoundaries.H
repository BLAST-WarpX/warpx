/* Copyright 2021 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLEBOUNDARIES_H_
#define WARPX_PARTICLEBOUNDARIES_H_

#include "Utils/WarpXAlgorithmSelection.H"

#include <AMReX_AmrCore.H>
#include <AMReX_Parser.H>

/**
 *  This allows a different boundary condition on each edge of the grid.
 */
struct ParticleBoundaries
{

    ParticleBoundaries () noexcept;

    void Set_reflect_all_velocities (bool flag);
    void SetAll (ParticleBoundaryType bc);
    /** Sets thermal velocity in ParticleBoundariesData 'data.m_uth' to u_th */
    void SetThermalVelocity (amrex::Real u_th);

    void SetBoundsX (ParticleBoundaryType bc_lo, ParticleBoundaryType bc_hi);
    void SetBoundsY (ParticleBoundaryType bc_lo, ParticleBoundaryType bc_hi);
    void SetBoundsZ (ParticleBoundaryType bc_lo, ParticleBoundaryType bc_hi);

    [[nodiscard]] bool CheckAll (ParticleBoundaryType bc) const;

    void BuildReflectionModelParsers ();

    // reflection models for absorbing boundaries
    std::string reflection_model_xlo_str = "0.0";
    std::string reflection_model_xhi_str = "0.0";
    std::string reflection_model_ylo_str = "0.0";
    std::string reflection_model_yhi_str = "0.0";
    std::string reflection_model_zlo_str = "0.0";
    std::string reflection_model_zhi_str = "0.0";

    std::unique_ptr<amrex::Parser> reflection_model_xlo_parser;
    std::unique_ptr<amrex::Parser> reflection_model_xhi_parser;
    std::unique_ptr<amrex::Parser> reflection_model_ylo_parser;
    std::unique_ptr<amrex::Parser> reflection_model_yhi_parser;
    std::unique_ptr<amrex::Parser> reflection_model_zlo_parser;
    std::unique_ptr<amrex::Parser> reflection_model_zhi_parser;

    struct ParticleBoundariesData {

        ParticleBoundaryType xmin_bc;
        ParticleBoundaryType xmax_bc;
        ParticleBoundaryType ymin_bc;
        ParticleBoundaryType ymax_bc;
        ParticleBoundaryType zmin_bc;
        ParticleBoundaryType zmax_bc;
        amrex::Real m_uth = 0.;

        amrex::ParserExecutor<1> reflection_model_xlo;
        amrex::ParserExecutor<1> reflection_model_xhi;
        amrex::ParserExecutor<1> reflection_model_ylo;
        amrex::ParserExecutor<1> reflection_model_yhi;
        amrex::ParserExecutor<1> reflection_model_zlo;
        amrex::ParserExecutor<1> reflection_model_zhi;

        bool reflect_all_velocities;
    };
    ParticleBoundariesData data;

};
#endif /*WARPX_PARTICLEBOUNDARIES_H_*/
