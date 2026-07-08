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

#include "NonlinearSolvers/LinearFunctionMF.H"
#include "WarpXSolverVec.H"

#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "Particles/MultiParticleContainer.H"

#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include "ImplicitSolver.H"

/** @file
 *  Semi-implicit Darwin field solver: advances the vector potential A
 *  (through an auxiliary variable Z) and derives E and B from it each step.
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

    void Define ( WarpX* a_WarpX, bool from_restart ) override;

    void PrintParameters () const override;

    int OneStep ( amrex::Real  start_time,
                  amrex::Real  a_dt,
                  int          a_step ) override;

    void ComputeRHS ( WarpXSolverVec&  a_RHS,
                      const WarpXSolverVec&  a_Z,
                      amrex::Real      start_time,
                      int              a_nl_iter,
                      bool             a_from_jacobian ) override;

    void ApplySusceptibility ( ablastr::fields::MultiLevelVectorField& rhs,
                               const ablastr::fields::MultiLevelVectorField& dA);

    void PrepareCurrentDeposition ();
    void AccumulateCurrentAndSusceptibility ();
    void CalculateSourceVector ();
    void UpdateEfromdA( int astep );
    void ClearParticleVelocities ();
    void FinishVelocityUpdate ();

    /**
     * \brief Apply conducting-wall (PEC) boundary conditions to a Z-staggered
     *  (i.e., B-staggered) vector field: zero the wall-normal components on
     *  the wall plane and fill `ng` guard layers with the B-type image
     *  symmetry (tangential components even-mirrored, normal components
     *  odd-mirrored). With ng = 0 only the on-wall zeroing (the projection of
     *  the constrained DOFs) is performed. No-op unless a PEC boundary is
     *  present.
     */
    void ApplyPECtoZ ( const ablastr::fields::VectorField& a_field,
                       int lev, const amrex::IntVect& ng );

    /**
     * \brief Apply conducting-wall (PEC) boundary conditions to a dA-staggered
     *  (i.e., E-staggered) vector field: zero the wall-tangential components
     *  on the wall plane and fill `ng` guard layers with the E-type image
     *  symmetry (tangential components odd-mirrored, normal components
     *  even-mirrored). No-op unless a PEC boundary is present.
     */
    void ApplyPECtodA ( const ablastr::fields::VectorField& a_field,
                        int lev, const amrex::IntVect& ng );

    // This parameter is used for the time-step fraction in the PC for implicit
    // treatment of light waves in the curl-curl MLMG solver.
    // This function should return zero if light waves are not treated implicitly
    [[nodiscard]] amrex::Real GetThetaForPC () const override { return 0.0; }

private:

    /**
     * \brief Solver vectors for delta A
     */
    WarpXSolverVec m_Z, m_source;

    /**
     * \brief True if any field boundary is a conducting (PEC) wall; set in
     *  Define(). Gates all wall-BC handling so periodic runs are untouched.
     */
    bool m_has_pec = false;

    /**
     * \brief Scratch space used by ComputeRHS(), allocated once here rather
     *  than on every GMRES iteration since every iterate of m_Z shares the
     *  same layout.
     */
    amrex::MultiFab m_lapZ_x, m_lapZ_y, m_lapZ_z;
    amrex::MultiFab m_Zscratch_x, m_Zscratch_y, m_Zscratch_z;

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

    // Linear solver defaults
    int m_linsol_verbose_int = 2;
    int m_linsol_maxits = 1000;
    int m_linsol_restart_length = 30;
    amrex::Real m_linsol_atol = 0.;
    amrex::Real m_linsol_rtol = 1.0e-4;
};

#endif
