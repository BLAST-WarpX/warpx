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
#include <AMReX_REAL.H>

#include <string>

/**
 * \brief Input parameters of the Newton-type nonlinear solvers (NewtonSolver and
 * PETScSNES), read in one place. A solver may change a default before calling Parse().
 */
struct NewtonSolverParams
{
    //! Verbosity flag for the nonlinear solver.
    bool verbose = true;
    //! Diagnostic file and the time-step interval at which it is written.
    std::string diagnostic_file;
    int diagnostic_interval = 1;

    //! Relative tolerance, absolute tolerance and maximum iterations for the nonlinear solver.
    amrex::Real rtol = 1.0e-6;
    amrex::Real atol = 0.;
    int maxits = 100;
    //! Flag to determine whether convergence is required.
    bool require_convergence = true;

    //! Zero updates the PC only at iteration zero; positive values give a Newton interval.
    int pc_update_newton_interval = 1;
    //! Update the PC on time steps whose zero-based index is divisible by this interval.
    int pc_update_time_step_interval = 1;

    //! Choice of linear solver.
    LinearSolverType linear_solver_type = LinearSolverType::amrex_gmres;
    //! Relative tolerance, absolute tolerance and maximum iterations for the linear solver.
    amrex::Real linsol_rtol = 1.0e-4;
    amrex::Real linsol_atol = 0.;
    int linsol_maxits = 1000;
    //! Verbosity flag and restart iteration for the linear solver.
    int linsol_verbose_int = 2;
    int linsol_restart_length = 30;

    //! Preconditioner type.
    PreconditionerType pc_type = PreconditionerType::none;

    void Parse ()
    {
        const amrex::ParmParse pp_newton("newton");
        pp_newton.query("verbose",             verbose);
        pp_newton.query("linear_solver",       linear_solver_type);
        pp_newton.query("absolute_tolerance",  atol);
        pp_newton.query("relative_tolerance",  rtol);
        pp_newton.query("max_iterations",      maxits);
        pp_newton.query("pc_update_newton_interval", pc_update_newton_interval);
        pp_newton.query("pc_update_time_step_interval", pc_update_time_step_interval);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            pc_update_newton_interval >= 0,
            "newton.pc_update_newton_interval must be nonnegative");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            pc_update_time_step_interval > 0,
            "newton.pc_update_time_step_interval must be positive");
        pp_newton.query("require_convergence", require_convergence);
        pp_newton.query("diagnostic_file",     diagnostic_file);
        pp_newton.query("diagnostic_interval", diagnostic_interval);

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
    }
};

#endif
