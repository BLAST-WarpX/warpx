/* Copyright 2024 Justin Angus, Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef Implicit_Solver_H_
#define Implicit_Solver_H_

#include "FieldSolver/ImplicitSolvers/ImplicitOptions.H"

#include "FieldSolver/ImplicitSolvers/WarpXSolverVec.H"
#include "NonlinearSolvers/NonlinearSolverLibrary.H"

#include <AMReX_Array.H>
#include <AMReX_REAL.H>
#include <AMReX_LO_BCTYPES.H>

/**
 * \brief Base class for implicit time solvers. The base functions are those
 *  needed by an implicit solver to be used through WarpX and those needed
 *  to use the nonlinear solvers, such as Picard or Newton (i.e., JFNK).
 */

class WarpX;
enum class FieldBoundaryType;
class ImplicitSolver
{
public:

    ImplicitSolver() = default;

    virtual ~ImplicitSolver() = default;

    // Prohibit Move and Copy operations
    ImplicitSolver(const ImplicitSolver&) = delete;
    ImplicitSolver& operator=(const ImplicitSolver&) = delete;
    ImplicitSolver(ImplicitSolver&&) = delete;
    ImplicitSolver& operator=(ImplicitSolver&&) = delete;

    //
    // the following routines are called by WarpX
    //

    /**
     * \brief Read user-provided parameters that control the implicit solver.
     * Allocate internal arrays for intermediate field values needed by the solver.
     */
    virtual void Define ( WarpX*  a_WarpX ) = 0;

    [[nodiscard]] bool IsDefined () const { return m_is_defined; }

    virtual void PrintParameters () const = 0;

    void CreateParticleAttributes () const;

    bool DoParticleSuborbits () { return m_particle_suborbits; }

    /**
     * \brief Advance fields and particles by one time step using the specified implicit algorithm
     */
    virtual void OneStep ( amrex::Real  a_time,
                           amrex::Real  a_dt,
                           int          a_step ) = 0;

    //
    // the following routines are called by the linear and nonlinear solvers
    //

    /**
     * \brief Computes the RHS of the equation corresponding to the specified implicit algorithm.
     * The discrete equations corresponding to numerical integration of ODEs are often
     * written in the form U = b + RHS(U), where U is the variable to be solved for (e.g.,
     * the solution at the next time step), b is a constant (i.e., the solution from the
     * previous time step), and RHS(U) is the right-hand-side of the equation. Iterative
     * solvers, such as Picard and Newton, and higher-order Runge-Kutta methods, need to
     * compute RHS(U) multiple times for different values of U. Thus, a routine that
     * returns this value is needed.
     * e.g., Ebar - E^n = cvac^2*0.5*dt*(curl(Bbar) - mu0*Jbar(Ebar,Bbar))
     * Here, U = Ebar, b = E^n, and the expression on the right is RHS(U).
     */
    virtual void ComputeRHS ( WarpXSolverVec&  a_RHS,
                        const WarpXSolverVec&  a_E,
                              amrex::Real      a_time,
                              int              a_nl_iter,
                              bool             a_from_jacobian ) = 0;

    [[nodiscard]] int numAMRLevels () const { return m_num_amr_levels; }

    [[nodiscard]] const amrex::Geometry& GetGeometry (int) const;
    [[nodiscard]] const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& GetFieldBoundaryLo () const;
    [[nodiscard]] const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& GetFieldBoundaryHi () const;
    [[nodiscard]] amrex::Array<amrex::LinOpBCType,AMREX_SPACEDIM> GetLinOpBCLo () const;
    [[nodiscard]] amrex::Array<amrex::LinOpBCType,AMREX_SPACEDIM> GetLinOpBCHi () const;

    /**
     * \brief Return pointer to MultiFab array for mass matrix
     */
    [[nodiscard]] const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>* GetMassMatricesCoeff() const {
        if (m_use_mass_matrices_pc) { return &m_mmpc_mfarrvec; }
        else { return nullptr; }
    }

    // This parameter is used for the time-step fraction in the PC for implicit
    // treatment of light waves in the curl-curl MLMG solver.
    // This function should return zero if light waves are not treated implicitly
    [[nodiscard]] virtual amrex::Real  GetThetaForPC () const = 0;

    void ComputeJfromMassMatrices ();

protected:

    /**
     * \brief Pointer back to main WarpX class
     */
    WarpX* m_WarpX;

    bool m_is_defined = false;

    /**
     * \brief Number of AMR levels
     */
    int m_num_amr_levels = 1;

    /**
     * \brief Time step
     */
    mutable amrex::Real m_dt = 0.0;

    /**
     * \brief Time-biasing parameter for fields used on RHS to advance system.
     */
    amrex::Real m_theta = 0.5;

    /**
     * \brief Nonlinear solver type and object
     */
    NonlinearSolverType m_nlsolver_type;
    std::unique_ptr<NonlinearSolver<WarpXSolverVec,ImplicitSolver>> m_nlsolver;

    /**
     * \brief tolerance used by the iterative method used to obtain a self-consistent
     *  update of the particle positions and velocities for given E and B on the grid
     */
    amrex::ParticleReal m_particle_tolerance = amrex::ParticleReal(1.0e-10);

    /**
     * \brief maximum iterations for the iterative method used to obtain a self-consistent
     *  update of the particle positions and velocities for given E and B on the grid
     */
    int m_max_particle_iterations = 21;

    /**
     * \brief whether to use suborbits for particles that fail to converge in
     * m_max_particle_iterations iterations
     */
    bool m_particle_suborbits = false;

    /**
     * \brief whether the details of unconverged particles are printed out during
     * the particle evolve loops
     */
    bool m_print_unconverged_particle_details = false;

    /**
     * \brief Whether to use mass matrices for the implicit solver
     */
    bool m_use_mass_matrices = false;

    /**
     * \brief Whether to use mass matrices to compute J during linear stage of JFNK
     */
    bool m_use_mass_matrices_jacobian = false;

    /**
     * \brief Whether to use mass matrices in the preconditioner
     */
    bool m_use_mass_matrices_pc = false;

    /**
     * \brief Direction-dependent component number of mass matrices elements
     */
    amrex::IntVect m_ncomp_xx = amrex::IntVect{0};
    amrex::IntVect m_ncomp_xy = amrex::IntVect{0};
    amrex::IntVect m_ncomp_xz = amrex::IntVect{0};
    amrex::IntVect m_ncomp_yx = amrex::IntVect{0};
    amrex::IntVect m_ncomp_yy = amrex::IntVect{0};
    amrex::IntVect m_ncomp_yz = amrex::IntVect{0};
    amrex::IntVect m_ncomp_zx = amrex::IntVect{0};
    amrex::IntVect m_ncomp_zy = amrex::IntVect{0};
    amrex::IntVect m_ncomp_zz = amrex::IntVect{0};

    /**
     * \brief Array of multifab pointers to mass matrix preconditioner
     */
    amrex::Vector<amrex::Array<amrex::MultiFab*,3>> m_mmpc_mfarrvec;

    /**
     * \brief parse nonlinear solver parameters (if one is used)
     */
    void parseNonlinearSolverParams ( const amrex::ParmParse&  pp );

    /**
     * \brief Initialize the Mass Matrices used for plasma response in nonlinear Newton solver
     */
    void InitializeMassMatrices ();

    void PrintBaseImplicitSolverParameters () const;

    /**
     * \brief Perform operations needed before computing the Right Hand Side
     */
    void PreRHSOp ( const amrex::Real  a_cur_time,
                    const int          a_nl_iter,
                    const bool         a_from_jacobian );

    /**
     * \brief Communicate Mass Matrices used for PC and apply boundary conditions
     */
    void SyncMassMatricesPCAndApplyBCs ();

    /**
     * \brief Scale mass matrices used for PC by c^2*mu0*theta*dt and add 1 to diagonal terms
     */
    void SetMassMatricesForPC ( const amrex::Real a_theta_dt );

    /**
     * \brief Save E and J at start of each Newton step
     */
    void SaveEandJ ();

    /**
     * \brief Convert from WarpX FieldBoundaryType to amrex::LinOpBCType
     */
    [[nodiscard]] amrex::Array<amrex::LinOpBCType,AMREX_SPACEDIM> convertFieldBCToLinOpBC ( const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& ) const;

};

#endif
