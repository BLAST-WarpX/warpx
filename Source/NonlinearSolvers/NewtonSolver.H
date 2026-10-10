/* Copyright 2024 Justin Angus
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef NEWTON_SOLVER_H_
#define NEWTON_SOLVER_H_

#include "LinearSolverLibrary.H"
#include "NewtonSolverParams.H"
#include "NonlinearSolver.H"
#include "JacobianFunctionMF.H"
#include "Preconditioner.H"
#include "Utils/TextMsg.H"

#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_ParmParse.H>
#include <AMReX_Config.H>

#include <vector>
#include <istream>
#include <filesystem>

/**
 * \brief Newton method to solve nonlinear equation of form:
 * F(U) = U - b - R(U) = 0. U is the solution vector, b is a constant,
 * and R(U) is some nonlinear function of U, which is computed in the
 * ComputeRHS() Ops function.
 */

template<class Vec, class Ops>
class NewtonSolver : public NonlinearSolver<Vec,Ops>
{
public:

    NewtonSolver() = default;

    ~NewtonSolver() override = default;

    // Prohibit Move and Copy operations
    NewtonSolver(const NewtonSolver&) = delete;
    NewtonSolver& operator=(const NewtonSolver&) = delete;
    NewtonSolver(NewtonSolver&&) noexcept = delete;
    NewtonSolver& operator=(NewtonSolver&&) noexcept = delete;

    void Define (const Vec&  a_U,
                       Ops*  a_ops) override;

    void Solve (Vec&   a_U,
          const Vec&   a_b,
          amrex::Real  a_time,
          amrex::Real  a_dt,
          int          a_step,
          bool verbose_step) const override;

    inline void CurTime ( amrex::Real  a_time ) const
    {
        m_cur_time = a_time;
        m_linear_function->curTime( a_time );
    }

    inline void CurTimeStep ( amrex::Real  a_dt ) const
    {
        m_dt = a_dt;
        m_linear_function->curTimeStep( a_dt );
    }

    PreconditionerType GetPreconditionerType () const override
    {
        return m_linear_function->pcType();
    }

    void PrintParams () const override
    {
        this->m_params.Print("Newton");
        m_newton_params.Print("Newton");
        m_linear_function->printParams();
    }

private:

    /**
     * \brief Intermediate Vec containers used by the solver.
     */
    mutable Vec m_dU, m_F, m_R;

    /**
     * \brief Pointer to Ops class.
     */
    Ops* m_ops = nullptr;

    /**
     * \brief Parameters specific to Newton-type solvers.
     */
    NewtonSolverParams m_newton_params;

    //! Ensure the first required linear solve initializes the PC, including after restart.
    mutable bool m_pc_initialized = false;

    /**
     * \brief Total nonlinear iterations for the diagnostic file
     */
    mutable int m_total_iters = 0;

    /**
     * \brief Total linear iterations for the diagnostic file
     */
    mutable int m_total_linsol_iters = 0;

    mutable amrex::Real m_cur_time, m_dt;

    /**
     * \brief The linear function used by linear solver to compute A*v.
     * In the contect of JFNK, A = dF/dU (i.e., system Jacobian)
     */
    std::unique_ptr<JacobianFunctionMF<Vec,Ops>> m_linear_function;

    /**
     * \brief The linear solver object.
     */
    std::unique_ptr<LinearSolver<Vec,JacobianFunctionMF<Vec,Ops>>> m_linear_solver;

    void ParseParameters ();

    /**
     * \brief Compute the nonlinear residual: F(U) = U - b - R(U).
     */
    void EvalResidual (Vec&        a_F,
                 const Vec&        a_U,
                 const Vec&        a_b,
                       amrex::Real a_time,
                       int         a_iter,
                       bool        update_pc) const;

};

template <class Vec, class Ops>
void NewtonSolver<Vec,Ops>::Define ( const Vec&  a_U,
                                     Ops*        a_ops )
{
    BL_PROFILE("NewtonSolver::Define()");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !this->m_is_defined,
        "Newton nonlinear solver object is already defined!");

    ParseParameters();

    m_dU.Define(a_U);
    m_F.Define(a_U); // residual function F(U) = U - b - R(U) = 0
    m_R.Define(a_U); // right hand side function R(U)

    m_ops = a_ops;

    m_linear_function = std::make_unique<JacobianFunctionMF<Vec,Ops>>();
    m_linear_function->define(m_F, m_ops, m_newton_params.pc_type);
    this->m_usePC = m_linear_function->usePreconditioner();

    if (m_newton_params.linear_solver_type == LinearSolverType::amrex_gmres) {
        m_linear_solver = std::make_unique<AMReXGMRES<Vec,JacobianFunctionMF<Vec,Ops>>>();
    } else if (m_newton_params.linear_solver_type == LinearSolverType::petsc_ksp) {
#ifdef AMREX_USE_PETSC
        m_linear_solver = std::make_unique<PETScKSP<Vec,JacobianFunctionMF<Vec,Ops>>>();
#else
        WARPX_ABORT_WITH_MESSAGE("NewtonSolver::Define(): must compile with PETSc to use petsc_ksp (AMREX_USE_PETSC must be defined)");
#endif
    } else {
        WARPX_ABORT_WITH_MESSAGE("NewtonSolver::Define(): invalid linear solver type");
    }
    m_linear_solver->define(*m_linear_function);
    m_linear_solver->setVerbose( m_newton_params.linsol_verbose_int );
    m_linear_solver->setRestartLength( m_newton_params.linsol_restart_length );
    m_linear_solver->setMaxIters( m_newton_params.linsol_maxits );

    this->m_is_defined = true;

    this->InitDiagnosticFile({"gmres_iters", "gmres_total_iters", "gmres_last_res"});
}

template <class Vec, class Ops>
void NewtonSolver<Vec,Ops>::ParseParameters ()
{
    this->m_params.Parse("newton");
    m_newton_params.Parse();
}

template <class Vec, class Ops>
void NewtonSolver<Vec,Ops>::Solve (Vec&         a_U,
                             const Vec&         a_b,
                                   amrex::Real  a_time,
                                   amrex::Real  a_dt,
                                   int          a_step,
                                   bool verbose_step) const
{
    BL_PROFILE("NewtonSolver::Solve()");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        this->m_is_defined,
        "NewtonSolver::Solve() called on undefined object");
    using namespace amrex::literals;

    //
    // Newton routine to solve nonlinear equation of form:
    // F(U) = U - b - R(U) = 0
    //

    CurTime(a_time);
    CurTimeStep(a_dt);

    amrex::Real norm_abs = 0.;
    amrex::Real norm0 = 1._rt;
    amrex::Real norm_rel = 0.;

    int iter;
    int linear_solver_iters = 0;
    bool reached_maxits = false;
    for (iter = 0; iter <= this->m_params.maxits; iter++) {

        // Determine whether to update the PC on this iteration
        const bool update_pc = m_newton_params.UpdatePC(a_step, iter, m_pc_initialized);

        // Compute residual: F(U) = U - b - R(U)
        EvalResidual(m_F, a_U, a_b, a_time, iter, update_pc);

        // Compute norm of the residual
        norm_abs = m_F.norm2();
        if (iter == 0) {
            if (norm_abs > 0.) { norm0 = norm_abs; }
            else { norm0 = 1._rt; }
        }
        norm_rel = norm_abs/norm0;

        // Check for convergence criteria
        if (this->m_params.verbose and verbose_step) {
            amrex::Print() << "Newton: iteration = " << std::setw(3) << iter <<  ", norm = "
                           << std::scientific << std::setprecision(5) << norm_abs << " (abs.), "
                           << std::scientific << std::setprecision(5) << norm_rel << " (rel.)" << "\n";
        }

        if (norm_abs < this->m_params.atol) {
            this->m_exit_status = 2;
            if (this->m_params.verbose and verbose_step) {
                amrex::Print() << "Newton: exiting at iteration = " << std::setw(3) << iter
                               << ". Satisfied absolute tolerance " << this->m_params.atol << "\n";
            }
            break;
        }

        // As PETSc SNES, the relative test is skipped at iteration 0
        if (iter > 0 and norm_abs <= this->m_params.rtol*norm0) {
            this->m_exit_status = 3;
            if (this->m_params.verbose and verbose_step) {
                amrex::Print() << "Newton: exiting at iteration = " << std::setw(3) << iter
                               << ". Satisfied relative tolerance " << this->m_params.rtol << "\n";
            }
            break;
        }

        // As PETSc SNES, a diverged solve returns a negative exit status
        if (norm_abs > m_newton_params.divergence_tolerance*norm0) {
            this->m_exit_status = -9;
            std::stringstream convergenceMsg;
            convergenceMsg << "Newton solver diverged at iteration " << iter <<
                              " of step " << a_step <<
                              ". Absolute norm " << norm_abs <<
                              " is more than " << m_newton_params.divergence_tolerance <<
                              " times the initial norm " << norm0;
            if (this->m_params.verbose) { amrex::Print() << convergenceMsg.str() << "\n"; }
            ablastr::warn_manager::WMRecordWarning("NewtonSolver", convergenceMsg.str());
            break;
        }

        if (iter == this->m_params.maxits) {
            reached_maxits = true;
            if (this->m_params.verbose and verbose_step) {
                amrex::Print() << "Newton: exiting at iter = " << std::setw(3) << iter
                               << ". Maximum iteration reached: iter = " << this->m_params.maxits << "\n";
            }
            break;
        }

        // Prepare for the linear solve
        m_linear_function->setBaseSolution(a_U);
        m_linear_function->setBaseRHS(m_R);
        m_ops->PreLinearSolve(update_pc);
        if (update_pc) {
            m_linear_function->updatePreCondMat();
            // Keep WarpX sparse assembly and cached PETSc factors on the same schedule.
            m_linear_solver->invalidatePreconditioner();
            m_pc_initialized = true;
        }

        // Solve linear system for Newton step [Jac]*dU = F
        m_dU.zero();
        m_linear_solver->solve( m_dU, m_F, m_newton_params.linsol_rtol, m_newton_params.linsol_atol );
        linear_solver_iters += m_linear_solver->getNumIters();

        // As PETSc SNES, stop at a failed linear solve without updating the solution
        if (!m_linear_solver->converged()) {
            this->m_exit_status = -3;
            std::stringstream convergenceMsg;
            convergenceMsg << "Newton solver stopped at iteration " << iter <<
                              " of step " << a_step <<
                              ": the linear solver failed to converge in " <<
                              m_linear_solver->getNumIters() << " iterations";
            if (this->m_params.verbose) { amrex::Print() << convergenceMsg.str() << "\n"; }
            ablastr::warn_manager::WMRecordWarning("NewtonSolver", convergenceMsg.str());
            break;
        }

        // Update solution
        a_U -= m_dU;
    }

    // Update total iteration count
    m_total_iters += iter;
    m_total_linsol_iters += linear_solver_iters;

    // Note that the failure warning is skipped in the case when
    // rtol == 0 to allow a fixed number of iterations without a flood of warning.
    if (reached_maxits and this->m_params.rtol == 0.) { this->m_exit_status = 0; }
    if (reached_maxits and this->m_params.rtol > 0.) {
        std::stringstream convergenceMsg;
        convergenceMsg << "Newton solver failed to converge after " << iter <<
                          " iterations at step " << a_step <<
                          ". Relative norm is " << norm_rel <<
                          " and the relative tolerance is " << this->m_params.rtol <<
                          ". Absolute norm is " << norm_abs <<
                          " and the absolute tolerance is " << this->m_params.atol;
        if (this->m_params.verbose) { amrex::Print() << convergenceMsg.str() << "\n"; }
        ablastr::warn_manager::WMRecordWarning("NewtonSolver", convergenceMsg.str());

        if (this->m_params.require_convergence) { this->m_exit_status = -5; }
        else { this->m_exit_status = 0; }
    }

    this->WriteDiagnostics(a_step, a_time + a_dt, iter, m_total_iters, norm_abs, norm_rel,
                           linear_solver_iters, m_total_linsol_iters,
                           m_linear_solver->getResidualNorm());

}

template <class Vec, class Ops>
void NewtonSolver<Vec,Ops>::EvalResidual (Vec&        a_F,
                                    const Vec&        a_U,
                                    const Vec&        a_b,
                                          amrex::Real a_time,
                                          int         a_iter,
                                          bool        update_pc) const
{
    BL_PROFILE("NewtonSolver::EvalResidual()");
    m_ops->ComputeRHS(m_R, a_U, a_time, a_iter, false, update_pc);

    // Compute residual: F(U) = U - b - R(U)
    a_F.Copy(a_U);
    a_F -= m_R;
    a_F -= a_b;

}

#endif
