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

#include "DarwinLinearFieldOperator.H"
#include "WarpXSolverVec.H"

#include "NonlinearSolvers/LinearSolverLibrary.H"

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

    /**
     * \brief Not implemented for this solver.
     *
     *  `ImplicitSolver::ComputeRHS()` is the residual `RHS(U)` of the nonlinear
     *  equation `U = b + RHS(U)`, evaluated repeatedly by the Picard and Newton
     *  solvers. The Darwin scheme is instead linear in its unknown: it never
     *  builds a nonlinear residual and never installs a nonlinear solver, so
     *  this override exists only to satisfy the pure-virtual base declaration
     *  and aborts if it is ever reached. The linear operator applied by GMRES
     *  is `DarwinLinearFieldOperator::apply()` instead.
     */
    void ComputeRHS ( WarpXSolverVec&  a_RHS,
                      const WarpXSolverVec&  a_Z,
                      amrex::Real      start_time,
                      int              a_nl_iter,
                      bool             a_from_jacobian ) override;

    /** \brief Pointer back to the main WarpX class, used by DarwinLinearFieldOperator */
    [[nodiscard]] WarpX* GetWarpX () const { return m_WarpX; }

    /**
     * \brief Accumulate `chi dA` into `rhs`, where `chi` is the deposited mass
     *  matrix scaled by `2 mu_0 / dt`. This is the plasma response term of the
     *  magnetoinductive operator, i.e. the linear response of the deposited
     *  current to the inductive E-field being solved for.
     */
    void ApplyScaledMassMatrices ( ablastr::fields::MultiLevelVectorField& rhs,
                                   const ablastr::fields::MultiLevelVectorField& dA);

    /**
     * \brief Fill a cell-centered MultiFab with the scalar susceptibility
     *  chi(x): the (2 mu0/dt)-scaled row-sum of the diagonal mass-matrix
     *  bands, averaged over the three diagonal blocks and interpolated from
     *  their native (E-type) staggering to cell centers. This is the local
     *  chi scale the curl(chi curl Z) operator term applies to a uniform dA,
     *  used by the pc_darwin_mlmg preconditioner's Helmholtz factor. Must be
     *  called after the mass matrices have been deposited and synced for the
     *  current step.
     *
     *  The full mass matrix is a 3x3 block tensor (one block per Jx/Jy/Jz
     *  row/column pair) at every point of each block's native E-type
     *  staggering, and each block is itself stored sparsely as several
     *  components representing the deposition-stencil bands rather than as
     *  a dense matrix. a_chi_cc collapses all of that to a single scalar per
     *  cell: it keeps only the 3 diagonal blocks (xx, yy, zz), sums each
     *  block's stencil-band components into a per-point row-sum, averages
     *  that row-sum over the 3 diagonal blocks, and interpolates the result
     *  from the staggered points onto the cell center.
     */
    void ComputeScaledMassMatrixCC ( amrex::MultiFab& a_chi_cc ) const;

    void PrepareVelocitiesForCurrentDeposition ();
    void AccumulateCurrentAndMassMatrices ();
    void CalculateSourceVector ();
    void ComputeInductiveEfromdA( int astep );
    void ClearParticleVelocities ();
    void FinishVelocityUpdate ();

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
     * \brief The linear operator whose action on a vector the linear solver
     *  evaluates, i.e. the left-hand side of the magnetoinductive equation
     *
     *      bilaplacian(Z) + curl(chi curl(Z)) = 2 * laplacian(B) + 2 * mu_0 curl(J)
     *
     *  where chi is the mass matrix scaled by 2 * mu_0 / dt. It owns the
     *  scratch space that evaluation needs.
     */
    std::unique_ptr<DarwinLinearFieldOperator> m_linear_function;

    /**
     * \brief Whether the predictor velocity push in OneStep() should
     *  temporarily override the global galerkin_interpolation flag to true.
     *  Set once in Define(); false if the user explicitly selected
     *  momentum-conserving gathering, to avoid silently overriding that choice.
     */
    bool m_predictor_use_galerkin = false;

    /**
     * \brief Choice of linear solver
     */
    LinearSolverType m_linear_solver_type = LinearSolverType::amrex_gmres;

    /**
     * \brief The linear solver object.
     */
    std::unique_ptr<LinearSolver<WarpXSolverVec,DarwinLinearFieldOperator>> m_linear_solver;

    // Linear solver defaults
    int m_linsol_verbose_int = 2;
    int m_linsol_maxits = 1000;
    int m_linsol_restart_length = 30;
    amrex::Real m_linsol_atol = 0.;
    amrex::Real m_linsol_rtol = 1.0e-4;

    /**
     * \brief Preconditioner for the GMRES solve (amrex_gmres.pc_type; off by
     *  default). Only PreconditionerType::pc_darwin_mlmg is supported: the
     *  factored-Laplacian MLMG preconditioner (see DarwinMLMGPC.H).
     */
    PreconditionerType m_pc_type = PreconditionerType::none;
};

#endif
