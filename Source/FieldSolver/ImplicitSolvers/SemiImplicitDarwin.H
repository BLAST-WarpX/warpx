/* Copyright 2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef SEMI_IMPLICIT_DARWIN_H_
#define SEMI_IMPLICIT_DARWIN_H_

// #include "NonlinearSolvers/LinearSolverLibrary.H"
#include "NonlinearSolvers/LinearFunctionMF.H"
// #include "NonlinearSolvers/NonlinearSolverLibrary.H"
#include "WarpXSolverVec.H"

#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "Initialization/DivCleaner/ProjectionDivCleaner.H"
#include "Particles/MultiParticleContainer.H"

#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include "ImplicitSolver.H"

/** @file
 *  INSERT DESCRIPTION!
 */

class SemiImplicitDarwin : public ImplicitSolver
{
public:

    SemiImplicitDarwin() = default;

    ~SemiImplicitDarwin() override = default;

    // Prohibit Move and Copy operations
    SemiImplicitDarwin(const SemiImplicitDarwin&) = delete;
    SemiImplicitDarwin& operator=(const SemiImplicitDarwin&) = delete;
    SemiImplicitDarwin(SemiImplicitDarwin&&) = delete;
    SemiImplicitDarwin& operator=(SemiImplicitDarwin&&) = delete;

    void Define ( WarpX*  a_WarpX ) override;

    void PrintParameters () const override;

    void OneStep ( amrex::Real  start_time,
                   amrex::Real  a_dt,
                   int          a_step ) override;

    void ComputeRHS ( WarpXSolverVec&  a_RHS,
                      const WarpXSolverVec&  a_dA,
                      amrex::Real      start_time,
                      int              a_nl_iter,
                      bool             a_from_jacobian ) override;

    void ApplySusceptibility ( ablastr::fields::MultiLevelVectorField& rhs,
                               const ablastr::fields::MultiLevelVectorField& dA);

    void PrepareCurrentDeposition ();
    void AccumulateCurrentAndSusceptibility ();
    void CalculateSourceVector ();
    void UpdateEandAfromdA( int astep );
    void ClearParticleVelocities ();
    void FinishVelocityUpdate ();

    // This parameter is used for the time-step fraction in the PC for implicit
    // treatment of light waves in the curl-curl MLMG solver.
    // This function should return zero if light waves are not treated implicitly
    [[nodiscard]] virtual amrex::Real GetThetaForPC () const override { return 0.0; }

private:

    /**
     * \brief Solver vectors for delta A
     */
    WarpXSolverVec m_dA, m_source;

    /**
     * \brief The linear function used by linear solver to compute A*v.
     */
    std::unique_ptr<LinearFunctionMF<WarpXSolverVec,SemiImplicitDarwin>> m_linear_function;

    /**
     * \brief Choice of linear solver
     */
    LinearSolverType m_linear_solver_type = LinearSolverType::amrex_gmres;

    /**
     * \brief The linear solver object.
     */
    std::unique_ptr<LinearSolver<WarpXSolverVec,LinearFunctionMF<WarpXSolverVec,SemiImplicitDarwin>>> m_linear_solver;

    /**
     * \brief Projection divergence cleaner to get solenoidal current.
     */
    // warpx::initialization::ProjectionDivCleaner m_div_cleaner;

    // Linear solver defaults
    int m_linsol_verbose_int = 2;
    int m_linsol_maxits = 1000;
    int m_linsol_restart_length = 30;
    amrex::Real m_linsol_atol = 0.;
    amrex::Real m_linsol_rtol = 1.0e-4;
};

#endif
