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

#include <array>

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

    /**
     * \brief Add the grad-div (Coulomb-gauge penalty) stabilization term
     *  -grad(alpha * div(Z)) to the operator output. Curl-free Z modes are
     *  otherwise constrained only by the bare biharmonic (the curl(chi curl)
     *  term annihilates them), leaving them orders of magnitude more singular
     *  than the solenoidal modes in domains large compared to the electron
     *  skin depth; this term lifts them to the same chi*k^2 eigenvalue scale.
     *  Discretely curl(grad(.)) = 0, so the term cannot change dA = curl(Z).
     *  `a_Z` must have >= 2 guard layers filled (periodic and, at conducting
     *  walls, image-symmetry values); `a_rhs` is modified on valid cells only.
     */
    void AddGradDivZTerm ( const ablastr::fields::VectorField& a_rhs,
                           const ablastr::fields::VectorField& a_Z,
                           int lev );

    /**
     * \brief Subtract the component means of a Z-staggered vector field for
     *  every vector component whose uniform (constant) field is an exact null
     *  vector of the wall-projected operator (zero curl, zero divergence,
     *  zero biharmonic, untouched by the wall projection, which only zeroes
     *  wall-normal DOFs). Any such component in the GMRES source is
     *  irreducible residual; any such component in the operator output is
     *  roundoff. Both have zero curl, so removing them cannot change
     *  dA = curl(Z). No-op unless a PEC boundary is present (fully periodic
     *  sources telescope to zero mean automatically).
     */
    void ProjectOutNullConstants ( const ablastr::fields::VectorField& a_field );

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
     * \brief Cell-centered scratch holding div(Z) inside AddGradDivZTerm()
     *  (Z is B-staggered, so its discrete divergence is fully cell-centered
     *  in every Cartesian geometry).
     */
    amrex::MultiFab m_divZ;

    /**
     * \brief User-adjustable multiplier on the grad-div penalty coefficient
     *  (semi_implicit_darwin.graddiv_factor; 0 disables the term).
     */
    amrex::Real m_graddiv_factor = 1.0;

    /**
     * \brief Grad-div penalty coefficient alpha, refreshed each step in
     *  AccumulateCurrentAndSusceptibility() as m_graddiv_factor x (2 mu0/dt)
     *  x the domain-mean diagonal mass-matrix band row-sum - i.e., the mean
     *  chi scale the curl(chi curl Z) term applies to solenoidal modes.
     */
    amrex::Real m_graddiv_alpha = 0.0;

    /**
     * \brief Which vector components of Z admit an exact constant null
     *  vector of the wall-projected operator: those that are not wall-normal
     *  to any PEC direction (see ProjectOutNullConstants). Set in Define();
     *  all-false unless a PEC boundary is present.
     */
    std::array<bool, 3> m_null_constant_comp = {false, false, false};

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
