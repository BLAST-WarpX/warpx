/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef NEWTON_SOLVER_PARAMS_H_
#define NEWTON_SOLVER_PARAMS_H_

#include "LinearSolverLibrary.H"
#include "NonlinearSolver.H"
#include "PreconditionerLibrary.H"
#include "Utils/TextMsg.H"

#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_Enum.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>

#include <sstream>
#include <string>

/**
 * \brief Input parameters specific to the Newton-type nonlinear solvers (NewtonSolver
 * and PETScSNES), in addition to NonlinearSolverParams. A solver may change a default
 * before calling Parse().
 */
struct NewtonSolverParams
{
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
    //! Restart iteration for the linear solver (amrex_gmres and petsc_ksp).
    int linsol_restart_length = 30;
    //! Weighted Jacobi and Chebyshev linear solvers: iterate on the assembled PC matrix.
    bool use_pcmat = true;

    //! Preconditioner type.
    PreconditionerType pc_type = PreconditionerType::none;

    /**
     * \brief Read the parameters. A solver with a fixed linear solver sets
     * linear_solver_type and passes a_query_linear_solver = false.
     */
    void Parse (bool a_query_linear_solver = true)
    {
        // Invalid values are replaced by the defaults, with a warning
        const NewtonSolverParams defaults(*this);

        const amrex::ParmParse pp_newton("newton");
        if (a_query_linear_solver) { pp_newton.query("linear_solver", linear_solver_type); }
        pp_newton.query("divergence_tolerance", divergence_tolerance);
        ResetInvalidInput(divergence_tolerance, (divergence_tolerance >= 1.),
                          defaults.divergence_tolerance,
                          "newton.divergence_tolerance", "is less than 1");
        pp_newton.query("pc_update_newton_interval", pc_update_newton_interval);
        ResetInvalidInput(pc_update_newton_interval, (pc_update_newton_interval >= 0),
                          defaults.pc_update_newton_interval,
                          "newton.pc_update_newton_interval", "is negative");
        pp_newton.query("pc_update_time_step_interval", pc_update_time_step_interval);
        ResetInvalidInput(pc_update_time_step_interval, (pc_update_time_step_interval > 0),
                          defaults.pc_update_time_step_interval,
                          "newton.pc_update_time_step_interval", "is not positive");

        const amrex::ParmParse pp_l(amrex::getEnumNameString(linear_solver_type));
        pp_l.query("verbose_int",         linsol_verbose_int);
        pp_l.query("absolute_tolerance",  linsol_atol);
        pp_l.query("relative_tolerance",  linsol_rtol);
        pp_l.query("max_iterations",      linsol_maxits);
        if (IsKrylov()) { pp_l.query("restart_length", linsol_restart_length); }
        else { pp_l.query("use_pcmat", use_pcmat); }

        /* backward compatibility */
        if (IsKrylov()) {
            const amrex::ParmParse pp_gmres("gmres");
            pp_gmres.query("verbose_int",         linsol_verbose_int);
            pp_gmres.query("restart_length",      linsol_restart_length);
            pp_gmres.query("absolute_tolerance",  linsol_atol);
            pp_gmres.query("relative_tolerance",  linsol_rtol);
            pp_gmres.query("max_iterations",      linsol_maxits);
        }

        const amrex::ParmParse pp_jac("jacobian");
        pp_jac.query("pc_type", pc_type);
        // The weighted Jacobi and Chebyshev linear solvers do not apply a preconditioner
        if (!IsKrylov() && pc_type != PreconditionerType::none) {
            ablastr::warn_manager::WMRecordWarning("NewtonSolver",
                amrex::getEnumNameString(linear_solver_type) +
                " linear solver does not use a preconditioner; ignoring jacobian.pc_type = " +
                amrex::getEnumNameString(pc_type));
            pc_type = PreconditionerType::none;
        }

        // Jacobi and Chebyshev may read MassMatrices_PC directly. A scheduled PC update can
        // clear these coefficients during residual evaluation and leave them unfinished if
        // Newton converges before PreLinearSolve(). Restrict the Newton interval when lagging
        // across time steps, until the PC deposition is moved to PreLinearSolve().
        const bool reads_mass_matrices_pc =
            pc_type == PreconditionerType::pc_jacobi ||
            pc_type == PreconditionerType::pc_chebyshev ||
            (!IsKrylov() && use_pcmat);
        if (reads_mass_matrices_pc && pc_update_time_step_interval > 1) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                pc_update_newton_interval == 0,
                "newton.pc_update_newton_interval must be 0 when using pc_type = pc_jacobi "
                "or pc_chebyshev, or the weighted_jacobi or chebyshev linear solver with "
                "use_pcmat = true, and pc_update_time_step_interval > 1.");
        }
    }

    //! Whether the linear solver is a Krylov method (amrex_gmres or petsc_ksp)
    [[nodiscard]] bool IsKrylov () const
    {
        return linear_solver_type == LinearSolverType::amrex_gmres ||
               linear_solver_type == LinearSolverType::petsc_ksp;
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
        if (IsKrylov()) {
            amrex::Print() << "Linear solver (" << linsol_name << ") restart length:     " << linsol_restart_length << "\n";
        }
        amrex::Print() << "Linear solver (" << linsol_name << ") max iterations:     " << linsol_maxits << "\n";
        amrex::Print() << "Linear solver (" << linsol_name << ") relative tolerance: " << linsol_rtol << "\n";
        amrex::Print() << "Linear solver (" << linsol_name << ") absolute tolerance: " << linsol_atol << "\n";
        if (!IsKrylov()) {
            amrex::Print() << "Linear solver (" << linsol_name << ") use_pcmat:           " << (use_pcmat?"true":"false") << "\n";
        }
        amrex::Print() << "Preconditioner type:      " << amrex::getEnumNameString(pc_type) << "\n";
    }
};

/**
 * \brief Failure messages common to the Newton-type nonlinear solvers.
 */
namespace newton_solver_msg
{
    //! The residual norm exceeds the divergence tolerance times the initial norm
    inline std::string Diverged (int a_iter, int a_step, amrex::Real a_norm_abs,
                                 amrex::Real a_divergence_tolerance, amrex::Real a_norm0)
    {
        std::stringstream msg;
        msg << "Newton solver diverged at iteration " << a_iter <<
               " of step " << a_step <<
               ". Absolute norm " << a_norm_abs <<
               " is more than " << a_divergence_tolerance <<
               " times the initial norm " << a_norm0;
        return msg.str();
    }

    //! The linear solver did not converge
    inline std::string LinearSolveFailed (int a_iter, int a_step, int a_linear_iters)
    {
        std::stringstream msg;
        msg << "Newton solver stopped at iteration " << a_iter <<
               " of step " << a_step <<
               ": the linear solver failed to converge in " <<
               a_linear_iters << " iterations";
        return msg.str();
    }

    //! The maximum number of iterations is reached
    inline std::string MaxIterations (int a_iter, int a_step,
                                      amrex::Real a_norm_rel, amrex::Real a_rtol,
                                      amrex::Real a_norm_abs, amrex::Real a_atol)
    {
        std::stringstream msg;
        msg << "Newton solver failed to converge after " << a_iter <<
               " iterations at step " << a_step <<
               ". Relative norm is " << a_norm_rel <<
               " and the relative tolerance is " << a_rtol <<
               ". Absolute norm is " << a_norm_abs <<
               " and the absolute tolerance is " << a_atol;
        return msg.str();
    }
}

#endif
