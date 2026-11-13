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
                  int          a_step,
                  bool verbose_step) override;

    int DoSolve (amrex::Real  /* start_time */,
                 int          /* a_step */,
                 bool /* verbose_step */) override { return 0; }

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
     * \brief Fill three cell-centered MultiFabs with the diagonal
     *  susceptibility components chi_xx(x), chi_yy(x), chi_zz(x): the
     *  (2 mu0/dt)-scaled row-sum of each diagonal mass-matrix block,
     *  collocated from its native (E-type) staggering onto cell centers by
     *  index identification. These are the local chi scales the
     *  curl(chi curl Z) operator term applies to a uniform dA, used by the
     *  pc_darwin_mlmg preconditioner's per-component Helmholtz factors. Must
     *  be called after the mass matrices have been deposited and synced for
     *  the current step.
     *
     *  The full mass matrix is a 3x3 block tensor (one block per Jx/Jy/Jz
     *  row/column pair) at every point of each block's native E-type
     *  staggering, and each block is itself stored sparsely as several
     *  components representing the deposition-stencil bands rather than as
     *  a dense matrix. a_chi_xx_cc, a_chi_yy_cc, a_chi_zz_cc each collapse
     *  one diagonal block to a single scalar per cell: sum that block's
     *  stencil-band components into a per-point row-sum at (i,j,k), then
     *  read that same (i,j,k) index into the cell-centered array (a plain
     *  index identification, not an average of neighboring staggered
     *  points). This is the same convention DarwinMLMGPC uses to collocate
     *  the B-staggered Z components, so chi_dd(x) ends up collocated with
     *  where curl(Z)_d itself lands under that convention.
     */
    void ComputeScaledMassMatrixCC ( amrex::MultiFab& a_chi_xx_cc,
                                     amrex::MultiFab& a_chi_yy_cc,
                                     amrex::MultiFab& a_chi_zz_cc ) const;

    void PrepareVelocitiesForCurrentDeposition ();
    void AccumulateCurrentAndMassMatrices ();
    void CalculateSourceVector ();
    void ComputeInductiveEfromdA( int astep );
    void ClearParticleVelocities ();
    void FinishVelocityUpdate ();

    /**
     * \brief Apply conducting-wall (PEC) boundary conditions to a Z-staggered
     *  (i.e., B-staggered) vector field: zero the wall-normal components on
     *  the wall plane and fill `ng` guard layers with the B-type image
     *  symmetry (tangential components even-mirrored, normal components
     *  odd-mirrored). With ng = 0 only the on-wall zeroing (the projection of
     *  the constrained degrees of freedom) is performed. No-op unless a PEC
     *  boundary is present.
     */
    void ApplyPECtoZ ( const ablastr::fields::VectorField& a_field,
                       int lev, const amrex::IntVect& ng ) const;

    /**
     * \brief Apply conducting-wall (PEC) boundary conditions to a dA-staggered
     *  (i.e., E-staggered) vector field: zero the wall-tangential components
     *  on the wall plane and fill `ng` guard layers with the E-type image
     *  symmetry (tangential components odd-mirrored, normal components
     *  even-mirrored). No-op unless a PEC boundary is present.
     */
    void ApplyPECtodA ( const ablastr::fields::VectorField& a_field,
                        int lev, const amrex::IntVect& ng ) const;

    /**
     * \brief Add the grad-div (Coulomb-gauge penalty) term
     *  -grad(alpha div(Z)) to the operator output `a_rhs`, on its valid cells.
     *  The curl(chi curl(Z)) term annihilates curl-free Z modes, which are then
     *  only constrained by the bilaplacian (eigenvalues ~k^4, many orders of
     *  magnitude below the k^4 + chi k^2 of the solenoidal modes in domains
     *  large compared to the electron skin depth), and this term lifts them to
     *  k^4 + alpha k^2. With alpha spatially uniform, the term is discretely
     *  curl-free, so it does not change dA = curl(Z). `a_Z` must have 2 guard
     *  layers filled (with the periodic and PEC image values). No-op if
     *  implicit_evolve.graddiv_factor is zero.
     */
    void AddGradDivZTerm ( const ablastr::fields::VectorField& a_rhs,
                           const ablastr::fields::VectorField& a_Z,
                           int lev );

    /**
     * \brief At conducting (PEC) walls, fold the guard-cell values of an
     *  E-staggered vector field deposited from the particles (the current
     *  density, or the product of the mass matrices with dA) back into the
     *  domain, with the image symmetry of the E-field (tangential components
     *  odd, normal components even), and set the image guard values. No-op
     *  unless a PEC boundary is present.
     */
    void FoldGuardCellsAtPEC ( const ablastr::fields::VectorField& a_field, int lev ) const;

    /**
     * \brief Subtract the mean of each component of a Z-staggered vector
     *  field whose uniform (constant) field is an exact null vector of the
     *  wall-projected operator: zero curl, zero bilaplacian, and untouched by
     *  the wall projection, which only zeroes wall-normal components. With
     *  PEC walls, the source has a component along these null vectors (the
     *  domain sum of curl(J) and laplacian(B) telescopes to wall values),
     *  which GMRES can never reduce. Being constant, these components have
     *  zero curl, so removing them does not change dA = curl(Z). No-op unless
     *  a PEC boundary is present (with periodic boundaries the source has
     *  zero mean).
     */
    void ProjectOutNullConstants ( const ablastr::fields::VectorField& a_field, int lev ) const;

    /** \brief Whether any field boundary is a conducting (PEC) wall */
    [[nodiscard]] bool HasPEC () const { return m_has_pec; }

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
     *  Define(). Gates all wall boundary handling so periodic runs are untouched.
     */
    bool m_has_pec = false;

    /**
     * \brief Which components of Z admit a constant null vector of the
     *  wall-projected operator: those that are not normal to any PEC wall
     *  (see ProjectOutNullConstants). Set in Define(); all false unless a PEC
     *  boundary is present.
     */
    std::array<bool, 3> m_null_constant_comp = {false, false, false};

    /**
     * \brief Multiplier of the grad-div penalty coefficient
     *  (implicit_evolve.graddiv_factor; 0, the default, disables the term)
     */
    amrex::Real m_graddiv_factor = 0.0;

    /**
     * \brief Grad-div penalty coefficient alpha, refreshed every step from
     *  the mass matrices: m_graddiv_factor times the domain mean of the
     *  (2 mu0/dt)-scaled row sums of the diagonal mass-matrix blocks, i.e.,
     *  the mean chi that curl(chi curl(Z)) applies to solenoidal modes
     */
    amrex::Real m_graddiv_alpha = 0.0;

    /**
     * \brief Cell-centered scratch holding div(Z) in AddGradDivZTerm()
     */
    amrex::MultiFab m_divZ;

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
     * \brief Whether the predictor velocity push in OneStep() should gather
     *  the electrostatic E-field with the Galerkin scheme.
     *  Set once in Define(); false if the user explicitly selected
     *  momentum-conserving gathering, to avoid silently overriding that choice.
     */
    bool m_predictor_use_galerkin = false;

    /**
     * \brief Order of the particle shape factors used for the magnetostatic part
     *  of the solve: the current and mass matrices deposition, the E-field gather
     *  in the corrector push and the B-field gather in both pushes. Must not
     *  exceed algo.particle_shape, which is used for the electrostatic part
     *  and is the default.
     */
    int m_magnetostatic_particle_shape;

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
