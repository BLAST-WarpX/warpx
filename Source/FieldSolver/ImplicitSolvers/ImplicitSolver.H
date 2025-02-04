/* Copyright 2024 Justin Angus, Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef Implicit_Solver_H_
#define Implicit_Solver_H_

#include "FieldSolver/ImplicitSolvers/WarpXSolverVec.H"
#include "NonlinearSolvers/NonlinearSolverLibrary.H"
#include "Utils/WarpXAlgorithmSelection.H"

#include <AMReX_Array.H>
#include <AMReX_REAL.H>
#include <AMReX_LO_BCTYPES.H>

/**
 * \brief Base class for implicit time solvers. The base functions are those
 *  needed by an implicit solver to be used through WarpX and those needed
 *  to use the nonlinear solvers, such as Picard or Newton (i.e., JFNK).
 */

class WarpX;
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

    void GetParticleSolverParams (int&  a_max_particle_iter,
                                  amrex::ParticleReal&  a_particle_tol ) const
    {
        a_max_particle_iter = m_max_particle_iterations;
        a_particle_tol = m_particle_tolerance;
    }

    void CreateParticleAttributes () const;

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
     * \brief Nonlinear solver type and object
     */
    NonlinearSolverType m_nlsolver_type;
    std::unique_ptr<NonlinearSolver<WarpXSolverVec,ImplicitSolver>> m_nlsolver;

    /**
     * \brief tolerance used by the iterative method used to obtain a self-consistent
     *  update of the particle positions and velocities for given E and B on the grid
     */
    amrex::ParticleReal m_particle_tolerance = 1.0e-10;

    /**
     * \brief maximum iterations for the iterative method used to obtain a self-consistent
     *  update of the particle positions and velocities for given E and B on the grid
     */
    int m_max_particle_iterations = 21;

    /**
     * \brief parse nonlinear solver parameters (if one is used)
     */
    void parseNonlinearSolverParams( const amrex::ParmParse&  pp )
    {

        std::string nlsolver_type_str;
        pp.get("nonlinear_solver", nlsolver_type_str);

        if (nlsolver_type_str=="picard") {
            m_nlsolver_type = NonlinearSolverType::Picard;
            m_nlsolver = std::make_unique<PicardSolver<WarpXSolverVec,ImplicitSolver>>();
            m_max_particle_iterations = 1;
            m_particle_tolerance = 0.0;
        }
        else if (nlsolver_type_str=="newton") {
            m_nlsolver_type = NonlinearSolverType::Newton;
            m_nlsolver = std::make_unique<NewtonSolver<WarpXSolverVec,ImplicitSolver>>();
            pp.query("max_particle_iterations", m_max_particle_iterations);
            pp.query("particle_tolerance", m_particle_tolerance);
        }
        else {
            WARPX_ABORT_WITH_MESSAGE(
                "invalid nonlinear_solver specified. Valid options are picard and newton.");
        }

    }

    /**
     * \brief Convert from WarpX FieldBoundaryType to amrex::LinOpBCType
     */
    [[nodiscard]] amrex::Array<amrex::LinOpBCType,AMREX_SPACEDIM> convertFieldBCToLinOpBC ( const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& ) const;

};

#endif
