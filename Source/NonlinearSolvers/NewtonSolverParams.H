/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef NEWTON_SOLVER_PARAMS_H_
#define NEWTON_SOLVER_PARAMS_H_

#include "LinearSolverLibrary.H"
#include "PreconditionerLibrary.H"
#include "Utils/TextMsg.H"

#include <AMReX_Enum.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>

#include <string>

/**
 * \brief Input parameters specific to the Newton-type nonlinear solvers (NewtonSolver
 * and PETScSNES), in addition to NonlinearSolverParams. A solver may change a default
 * before calling Parse().
 */
struct NewtonSolverParams
{
    //! The solve has converged when the norm of the update is below this factor times
    //! the norm of the solution (not an input parameter).
    amrex::Real step_tolerance = 1.0e-16;
    //! The solve has diverged when the residual norm exceeds this factor times the initial norm.
    amrex::Real divergence_tolerance = 1.0e4;

    //! Zero updates the PC only at iteration zero; positive values give a Newton interval.
    int pc_update_newton_interval = 1;
    //! Update the PC on time steps whose zero-based index is divisible by this interval.
    int pc_update_time_step_interval = 1;

    //! Choice of linear solver.
    LinearSolverType linear_solver_type = LinearSolverType::amrex_gmres;
    //! Relative tolerance for the linear solver.
    amrex::Real linsol_rtol = 1.0e-4;
    //! Absolute tolerance for the linear solver.
    amrex::Real linsol_atol = 0.;
    //! Maximum iterations for the linear solver.
    int linsol_maxits = 1000;
    //! Verbosity flag for the linear solver.
    int linsol_verbose_int = 2;
    //! Restart iteration for the linear solver.
    int linsol_restart_length = 30;

    //! Preconditioner type.
    PreconditionerType pc_type = PreconditionerType::none;

    void Parse ()
    {
        const amrex::ParmParse pp_newton("newton");
        pp_newton.query("linear_solver",       linear_solver_type);
        pp_newton.query("divergence_tolerance", divergence_tolerance);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            divergence_tolerance >= 1.,
            "newton.divergence_tolerance must not be less than 1");
        pp_newton.query("pc_update_newton_interval", pc_update_newton_interval);
        pp_newton.query("pc_update_time_step_interval", pc_update_time_step_interval);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            pc_update_newton_interval >= 0,
            "newton.pc_update_newton_interval must be nonnegative");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            pc_update_time_step_interval > 0,
            "newton.pc_update_time_step_interval must be positive");

        const amrex::ParmParse pp_l(amrex::getEnumNameString(linear_solver_type));
        pp_l.query("verbose_int",         linsol_verbose_int);
        pp_l.query("restart_length",      linsol_restart_length);
        pp_l.query("absolute_tolerance",  linsol_atol);
        pp_l.query("relative_tolerance",  linsol_rtol);
        pp_l.query("max_iterations",      linsol_maxits);

        /* backward compatibility */
        const amrex::ParmParse pp_gmres("gmres");
        pp_gmres.query("verbose_int",         linsol_verbose_int);
        pp_gmres.query("restart_length",      linsol_restart_length);
        pp_gmres.query("absolute_tolerance",  linsol_atol);
        pp_gmres.query("relative_tolerance",  linsol_rtol);
        pp_gmres.query("max_iterations",      linsol_maxits);

        const amrex::ParmParse pp_jac("jacobian");
        pp_jac.query("pc_type", pc_type);

        if (pc_type == PreconditionerType::pc_jacobi) {
            if (pc_update_time_step_interval > 1) {
            // Jacobi reads MassMatrices_PC directly. A scheduled PC update can
            // clear these coefficients during residual evaluation, then leave
            // them unfinished if Newton converges before PreLinearSolve().
            // Restrict Newton update intervals when lagging across time steps.
            // This does not cover convergence at iteration 0 on an update step.
            // The underlying issue is that suborbit particles deposit their PC
            // contributions during residual evaluation. A future PR will move
            // this deposition to PreLinearSolve(), removing the need for this
            // restriction.
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    pc_update_newton_interval == 0,
                    "newton.pc_update_newton_interval must be 0 when using "
                    "pc_type = pc_jacobi and pc_update_time_step_interval > 1.");
            }
        }
    }

    /**
     * \brief Whether the PC is updated at Newton iteration a_iter of time step a_step.
     * The first required linear solve always initializes the PC.
     */
    [[nodiscard]] bool UpdatePC (int a_step, int a_iter, bool a_pc_initialized) const
    {
        const bool pc_update_step = (a_step % pc_update_time_step_interval == 0);
        const bool pc_update_iteration = (a_iter == 0) ||
            (pc_update_newton_interval > 0 && a_iter % pc_update_newton_interval == 0);
        return !a_pc_initialized || (pc_update_step && pc_update_iteration);
    }

    void Print (const std::string& a_name) const
    {
        amrex::Print() << a_name << " divergence tolerance: " << divergence_tolerance << "\n";
        amrex::Print() << a_name << " PC update Newton interval:    " << pc_update_newton_interval << "\n";
        amrex::Print() << a_name << " PC update time-step interval: " << pc_update_time_step_interval << "\n";
        auto linsol_name = amrex::getEnumNameString(linear_solver_type);
        amrex::Print() << a_name << " linear solver:       " << linsol_name << "\n";
        amrex::Print() << "Linear solver (" << linsol_name << ") verbose:            " << linsol_verbose_int << "\n";
        amrex::Print() << "Linear solver (" << linsol_name << ") restart length:     " << linsol_restart_length << "\n";
        amrex::Print() << "Linear solver (" << linsol_name << ") max iterations:     " << linsol_maxits << "\n";
        amrex::Print() << "Linear solver (" << linsol_name << ") relative tolerance: " << linsol_rtol << "\n";
        amrex::Print() << "Linear solver (" << linsol_name << ") absolute tolerance: " << linsol_atol << "\n";
        amrex::Print() << "Preconditioner type:      " << amrex::getEnumNameString(pc_type) << "\n";
    }
};

#endif
