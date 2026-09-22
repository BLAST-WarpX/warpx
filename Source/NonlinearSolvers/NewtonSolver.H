/* Copyright 2024 Justin Angus
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef NEWTON_SOLVER_H_
#define NEWTON_SOLVER_H_

#include "LinearSolverLibrary.H"
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
          int          a_step) const override;

    void GetSolverParams (amrex::Real&  a_rtol,
                          amrex::Real&  a_atol,
                          int&          a_maxits) override
    {
        a_rtol = m_rtol;
        a_atol = m_atol;
        a_maxits = m_maxits;
    }

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
        amrex::Print()     << "Newton verbose:             " << (this->m_verbose?"true":"false") << "\n";
        amrex::Print()     << "Newton max iterations:      " << m_maxits << "\n";
        amrex::Print()     << "Newton relative tolerance:  " << m_rtol << "\n";
        amrex::Print()     << "Newton absolute tolerance:  " << m_atol << "\n";
        amrex::Print()     << "Newton require convergence: " << (m_require_convergence?"true":"false") << "\n";
        if (m_adaptive_forcing) {
            amrex::Print() << "Newton adaptive forcing:    true"
                           << " (gamma = "    << m_forcing_gamma
                           << ", xi = "       << m_forcing_xi
                           << ", zeta_max = " << m_forcing_zeta_max
                           << ", zeta_min = " << m_forcing_zeta_min << ")\n";
        } else {
            amrex::Print() << "Newton adaptive forcing:    false\n";
        }
        auto linsol_name = amrex::getEnumNameString(m_linear_solver_type);
        amrex::Print()     << "Newton linear solver:       " << linsol_name << "\n";
        amrex::Print()     << "Linear solver (" << linsol_name << ") verbose:            " << m_linsol_verbose_int << "\n";
        amrex::Print()     << "Linear solver (" << linsol_name << ") restart length:     " << m_linsol_restart_length << "\n";
        amrex::Print()     << "Linear solver (" << linsol_name << ") max iterations:     " << m_linsol_maxits << "\n";
        amrex::Print()     << "Linear solver (" << linsol_name << ") relative tolerance: " << m_linsol_rtol << "\n";
        amrex::Print()     << "Linear solver (" << linsol_name << ") absolute tolerance: " << m_linsol_atol << "\n";
        amrex::Print()     << "Preconditioner type:      " << amrex::getEnumNameString(m_pc_type) << "\n";

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
     * \brief Flag to determine whether convergence is required.
     */
    bool m_require_convergence = true;

    /**
     * \brief Relative tolerance for the Newton solver.
     */
    amrex::Real m_rtol = 1.0e-6;

    /**
     * \brief Absolute tolerance for the Newton solver.
     */
    amrex::Real m_atol = 0.;

    /**
     * \brief Relative perturbation scale for the finite-difference JVP
     * (matrix-free Jacobian action) in the linear solve.
     */
    amrex::Real m_jfnk_epsilon = 1.0e-6;

    /**
     * \brief Maximum iterations for the Newton solver.
     */
    int m_maxits = 100;

    /**
     * \brief Total nonlinear iterations for the diagnostic file
     */
    mutable int m_total_iters = 0;

    /**
     * \brief Relative tolerance for linear solver.
     */
    amrex::Real m_linsol_rtol = 1.0e-4;

    /**
     * \brief Absolute tolerance for linear solver.
     */
    amrex::Real m_linsol_atol = 0.;

    /**
     * \brief Maximum iterations for linear solver.
     */
    int m_linsol_maxits = 1000;

    /**
     * \brief Total linear iterations for the diagnostic file
     */
    mutable int m_total_linsol_iters = 0;

    /**
     * \brief Verbosity flag for linear solver.
     */
    int m_linsol_verbose_int = 2;

    /**
     * \brief Restart iteration for linear solver.
     */
    int m_linsol_restart_length = 30;

    /**
     * \brief Preconditioner type
     */
    PreconditionerType m_pc_type = PreconditionerType::none;

    mutable amrex::Real m_cur_time, m_dt;

    /**
     * \brief The linear function used by linear solver to compute A*v.
     * In the contect of JFNK, A = dF/dU (i.e., system Jacobian)
     */
    std::unique_ptr<JacobianFunctionMF<Vec,Ops>> m_linear_function;

    /**
     * \brief Choice of linear solver
     */
    LinearSolverType m_linear_solver_type = LinearSolverType::amrex_gmres;

    /**
     * \brief The linear solver object.
     */
    std::unique_ptr<LinearSolver<Vec,JacobianFunctionMF<Vec,Ops>>> m_linear_solver;

    void ParseParameters ();

    /**
     * \brief Compute the nonlinear residual: F(U) = U - b - R(U).
     */
    /** Optional backtracking line search (newton.line_search = 1):
     *  after the linear solve, accept the largest step fraction alpha in
     *  {1, 1/2, ..., 2^-max_halvings} that reduces the residual norm
     *  (keeping the best trial seen). Stabilizes Newton when the update
     *  overshoots -- stiff or state-dependent coefficients (vacuum
     *  resistivity ramps, flux limiters) and noisy residual landscapes. */
    bool m_use_line_search = false;
    int m_line_search_max_halvings = 6;
    /** newton.line_search_reuse_residual = 1: reuse the accepted line-search trial
     *  residual as the next iteration's residual instead of re-evaluating it
     *  (one residual evaluation less per Newton iteration; see Solve()). */
    bool m_line_search_reuse = false;
    mutable Vec m_U_ls;

    /** Eisenstat-Walker adaptive forcing (newton.adaptive_forcing = 1):
     *  choose the GMRES relative tolerance each Newton iteration from the
     *  observed nonlinear residual decrease instead of a fixed value
     *  (Choice 2 with safeguards, Chacon JCP 2025 Eq. 21). Avoids
     *  oversolving the linear system when far from the root. Off by
     *  default; the fixed-tolerance path is unchanged. */
    bool m_adaptive_forcing = false;
    amrex::Real m_forcing_gamma = 0.9;
    amrex::Real m_forcing_xi = 1.5;
    amrex::Real m_forcing_zeta_max = 0.8;
    amrex::Real m_forcing_zeta_min = 1.0e-4;
    /** True when the user explicitly set the linear-solver relative
     *  tolerance; adaptive forcing then starts from that value on the
     *  first Newton iteration instead of zeta_max. */
    bool m_linsol_rtol_set = false;

    void EvalResidual ( Vec&         a_F,
                  const Vec&         a_U,
                  const Vec&         a_b,
                        amrex::Real  a_time,
                        int          a_iter ) const;

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
    m_U_ls.Define(a_U); // line-search base state
    m_F.Define(a_U); // residual function F(U) = U - b - R(U) = 0
    m_R.Define(a_U); // right hand side function R(U)

    m_ops = a_ops;

    m_linear_function = std::make_unique<JacobianFunctionMF<Vec,Ops>>();
    m_linear_function->define(m_F, m_ops, m_pc_type);
    m_linear_function->setJFNKEps(m_jfnk_epsilon);
    this->m_usePC = m_linear_function->usePreconditioner();

    if (m_linear_solver_type == LinearSolverType::amrex_gmres) {
        m_linear_solver = std::make_unique<AMReXGMRES<Vec,JacobianFunctionMF<Vec,Ops>>>();
    } else if (m_linear_solver_type == LinearSolverType::petsc_ksp) {
#ifdef AMREX_USE_PETSC
        m_linear_solver = std::make_unique<PETScKSP<Vec,JacobianFunctionMF<Vec,Ops>>>();
#else
        WARPX_ABORT_WITH_MESSAGE("NewtonSolver::Define(): must compile with PETSc to use petsc_ksp (AMREX_USE_PETSC must be defined)");
#endif
    } else {
        WARPX_ABORT_WITH_MESSAGE("NewtonSolver::Define(): invalid linear solver type");
    }
    m_linear_solver->define(*m_linear_function);
    m_linear_solver->setVerbose( m_linsol_verbose_int );
    m_linear_solver->setRestartLength( m_linsol_restart_length );
    m_linear_solver->setMaxIters( m_linsol_maxits );

    this->m_is_defined = true;

    // Create diagnostic file and write header
    if (amrex::ParallelDescriptor::IOProcessor()
        && !this->m_diagnostic_file.empty()
        && !amrex::FileExists(this->m_diagnostic_file)) {

        std::filesystem::path const diagnostic_path(this->m_diagnostic_file);
        std::filesystem::path const diagnostic_dir = diagnostic_path.parent_path();
        if (!diagnostic_dir.empty()) {
            std::filesystem::create_directories(diagnostic_dir);
        }

        std::ofstream diagnostic_file{this->m_diagnostic_file, std::ofstream::out | std::ofstream::trunc};
        int c = 0;
        diagnostic_file << "#";
        diagnostic_file << "[" << c++ << "]step()";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]time(s)";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]iters";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]total_iters";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]norm_abs";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]norm_rel";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]gmres_iters";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]gmres_total_iters";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]gmres_last_res";
        diagnostic_file << "\n";
        diagnostic_file.close();
    }
}

template <class Vec, class Ops>
void NewtonSolver<Vec,Ops>::ParseParameters ()
{
    const amrex::ParmParse pp_newton("newton");
    pp_newton.query("verbose",             this->m_verbose);
    pp_newton.query("linear_solver",       m_linear_solver_type);
    pp_newton.query("absolute_tolerance",  m_atol);
    pp_newton.query("relative_tolerance",  m_rtol);
    pp_newton.query("max_iterations",      m_maxits);
    pp_newton.query("require_convergence", m_require_convergence);
    pp_newton.query("jfnk_epsilon",        m_jfnk_epsilon);
    pp_newton.query("line_search",          m_use_line_search);
    pp_newton.query("line_search_max_halvings", m_line_search_max_halvings);
    pp_newton.query("line_search_reuse_residual", m_line_search_reuse);
    pp_newton.query("adaptive_forcing",    m_adaptive_forcing);
    pp_newton.query("forcing_gamma",       m_forcing_gamma);
    pp_newton.query("forcing_xi",          m_forcing_xi);
    pp_newton.query("forcing_zeta_max",    m_forcing_zeta_max);
    pp_newton.query("forcing_zeta_min",    m_forcing_zeta_min);
    pp_newton.query("diagnostic_file",     this->m_diagnostic_file);
    pp_newton.query("diagnostic_interval", this->m_diagnostic_interval);

    const amrex::ParmParse pp_l(amrex::getEnumNameString(m_linear_solver_type));
    pp_l.query("verbose_int",         m_linsol_verbose_int);
    pp_l.query("restart_length",      m_linsol_restart_length);
    pp_l.query("absolute_tolerance",  m_linsol_atol);
    if (pp_l.query("relative_tolerance", m_linsol_rtol)) { m_linsol_rtol_set = true; }
    pp_l.query("max_iterations",      m_linsol_maxits);

    /* backward compatibility */
    const amrex::ParmParse pp_gmres("gmres");
    pp_gmres.query("verbose_int",         m_linsol_verbose_int);
    pp_gmres.query("restart_length",      m_linsol_restart_length);
    pp_gmres.query("absolute_tolerance",  m_linsol_atol);
    if (pp_gmres.query("relative_tolerance", m_linsol_rtol)) { m_linsol_rtol_set = true; }
    pp_gmres.query("max_iterations",      m_linsol_maxits);

    const amrex::ParmParse pp_jac("jacobian");
    pp_jac.query("pc_type", m_pc_type);
}

template <class Vec, class Ops>
void NewtonSolver<Vec,Ops>::Solve (Vec&         a_U,
                             const Vec&         a_b,
                                   amrex::Real  a_time,
                                   amrex::Real  a_dt,
                                   int          a_step) const
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
    this->m_last_converged = false;
    // every exit path below sets the status explicitly; never inherit the previous solve's
    this->m_exit_status = 0;

    amrex::Real norm_abs = 0.;
    amrex::Real norm0 = 1._rt;
    amrex::Real norm_rel = 0.;

    // Eisenstat-Walker adaptive forcing state: per-iteration GMRES relative
    // tolerance zeta_k, plus the previous iteration's zeta and residual norm
    // (Choice 2 with safeguards, Chacon JCP 2025 Eq. 21). With adaptive
    // forcing off, linsol_rtol stays at the fixed m_linsol_rtol.
    amrex::Real linsol_rtol = m_linsol_rtol;
    amrex::Real zeta_prev = m_forcing_zeta_max;
    amrex::Real norm_prev = 1._rt;

    int iter;
    int linear_solver_iters = 0;
    // line-search safeguard state: a direction along which no trial reduces the
    // residual is rejected (the iterate is kept) and the next linear solve is done
    // at the tight (pinned) tolerance; two consecutive rejections end the step
    int ls_fail_count = 0;
    bool force_tight_solve = false;
    // line-search residual reuse: when set, m_F/m_R and the Ops state already hold the
    // residual of the current a_U (evaluated as this iteration's residual by the line
    // search), so the evaluation at the top of the loop is skipped
    bool have_residual = false;
    for (iter = 0; iter < m_maxits;) {

        // Compute residual: F(U) = U - b - R(U)
        if (!have_residual) {
            EvalResidual(m_F, a_U, a_b, a_time, iter);
        }
        have_residual = false;

        // Compute norm of the residual
        norm_abs = m_F.norm2();
        if (iter == 0 && m_ops->LastRHSInvalid()) {
            // the residual of the initial iterate could not be evaluated (particles beyond
            // the deposition guard range already with the step-start field): there is no
            // reference norm to converge against. Report the step as unconverged so that
            // the driver may redo it with a smaller dt (implicit_evolve.dt_halving_max_levels).
            // Later invalid evaluations need no special case: their sentinel norm makes the
            // line search reject the trial.
            this->m_exit_status = (m_require_convergence ? -5 : 0);
            norm_rel = 1._rt;
            amrex::Print() << "Newton: the initial residual is invalid (particles beyond the "
                           << "deposition guard range): exiting the step unconverged\n";
            break;
        }
        if (iter == 0) {
            if (norm_abs > 0.) { norm0 = norm_abs; }
            else { norm0 = 1._rt; }
        }
        norm_rel = norm_abs/norm0;

        // Check for convergence criteria
        if (this->m_verbose || iter == m_maxits) {
            amrex::Print() << "Newton: iteration = " << std::setw(3) << iter <<  ", norm = "
                           << std::scientific << std::setprecision(5) << norm_abs << " (abs.), "
                           << std::scientific << std::setprecision(5) << norm_rel << " (rel.)" << "\n";
        }

        if (norm_abs < m_atol) {
            this->m_exit_status = 2;
            this->m_last_converged = true;
            if (this->m_verbose) {
                amrex::Print() << "Newton: exiting at iteration = " << std::setw(3) << iter
                               << ". Satisfied absolute tolerance " << m_atol << "\n";
            }
            break;
        }

        if (norm_rel < m_rtol) {
            this->m_exit_status = 3;
            this->m_last_converged = true;
            if (this->m_verbose) {
                amrex::Print() << "Newton: exiting at iteration = " << std::setw(3) << iter
                               << ". Satisfied relative tolerance " << m_rtol << "\n";
            }
            break;
        }

        if (norm_abs > 100._rt*norm0) {
            // Return the divergence to the caller instead of aborting here: the
            // step driver may redo the step with a smaller dt
            // (implicit_evolve.dt_halving_max_levels); without that option
            // WarpX::OneStep() aborts on the negative exit status as before.
            this->m_exit_status = -9;
            amrex::Print() << "Newton: exiting at iteration = " << std::setw(3) << iter
                           << ". SOLVER DIVERGED! relative tolerance = " << m_rtol << "\n";
            std::stringstream convergenceMsg;
            convergenceMsg << "Newton: exiting at iteration " << std::setw(3) << iter <<
                              ". SOLVER DIVERGED! absolute norm = " << norm_abs <<
                              " has increased by 100X from that after first iteration.";
            ablastr::warn_manager::WMRecordWarning("NewtonSolver", convergenceMsg.str());
            break;
        }

        // Prepare for the linear solve
        m_ops->PreLinearSolve(iter);
        m_linear_function->setBaseSolution(a_U);
        m_linear_function->setBaseRHS(m_R);
        m_linear_function->updatePreCondMat(a_U);

        // Eisenstat-Walker Choice-2 forcing: set this iteration's GMRES
        // relative tolerance from the observed nonlinear residual decrease.
        if (m_adaptive_forcing) {
            amrex::Real zeta;
            if (iter == 0) {
                // No residual history yet: start from zeta_max, unless the
                // user pinned a fixed linear-solver relative tolerance (which
                // is honored exactly, below the zeta_min floor if so pinned).
                zeta = m_linsol_rtol_set ? m_linsol_rtol : m_forcing_zeta_max;
            } else {
                const amrex::Real eps_newton = m_rtol*norm0 + m_atol;
                const amrex::Real zeta_A = m_forcing_gamma
                    * std::pow(norm_abs/norm_prev, m_forcing_xi);
                const amrex::Real zeta_B = std::min( m_forcing_zeta_max,
                    std::max(zeta_A, m_forcing_gamma*std::pow(zeta_prev, m_forcing_xi)) );
                zeta = std::min( m_forcing_zeta_max,
                    std::max(zeta_B, m_forcing_gamma*eps_newton/norm_abs) );
                zeta = std::max(zeta, m_forcing_zeta_min);
            }
            if (force_tight_solve) {
                // after a rejected direction: the loose forcing (up to zeta_max, and
                // zeta_A > 1 when the residual did not fall) is the usual reason the
                // direction was not a descent direction; solve this one tightly
                zeta = m_linsol_rtol_set ? m_linsol_rtol : m_forcing_zeta_min;
                force_tight_solve = false;
            }
            zeta_prev = zeta;
            norm_prev = norm_abs;
            linsol_rtol = zeta;
        }

        // Solve linear system for Newton step [Jac]*dU = F
        m_dU.zero();
        m_linear_solver->solve( m_dU, m_F, linsol_rtol, m_linsol_atol );
        linear_solver_iters += m_linear_solver->getNumIters();

        // Update solution
        if (m_use_line_search) {
            // Backtracking: try alpha = 1, 1/2, ... and accept the first trial that
            // reduces the residual norm (falling back to the best trial seen); the
            // accepted state is re-evaluated at the top of the next iteration unless reused.
            m_U_ls.Copy(a_U);
            amrex::Real alpha = 1._rt;
            amrex::Real best_alpha = 1._rt;
            amrex::Real best_norm = -1._rt;
            bool last_is_best = false;
            for (int ls = 0; ls <= m_line_search_max_halvings; ++ls) {
                a_U.linComb(1._rt, m_U_ls, -alpha, m_dU);
                // with residual reuse the trials are evaluated as the next iteration's
                // residual (a_iter = iter + 1), so an accepted trial is a full evaluation
                // (the Ops only test a_nl_iter == 0 for the reduced first evaluation)
                EvalResidual(m_F, a_U, a_b, a_time, m_line_search_reuse ? iter + 1 : iter);
                amrex::Real const trial_norm = m_F.norm2();
                last_is_best = false;
                if (best_norm < 0._rt || trial_norm < best_norm) {
                    best_norm = trial_norm;
                    best_alpha = alpha;
                    last_is_best = true;
                }
                if (trial_norm < norm_abs) { break; }
                alpha *= 0.5_rt;
            }
            if (best_norm < norm_abs) {
                a_U.linComb(1._rt, m_U_ls, -best_alpha, m_dU);
                // reuse when the accepted trial was the last one evaluated (the linComb above
                // rebuilds the same a_U); exact only for Ops whose residual has no
                // evaluation-order hysteresis, hence opt-in
                have_residual = m_line_search_reuse && last_is_best;
                ls_fail_count = 0;
                if (this->m_verbose && best_alpha < 1._rt) {
                    amrex::Print() << "Newton: line search accepted alpha = "
                                   << best_alpha << "\n";
                }
            } else {
                // No trial reduced the residual: this is not a descent direction. Moving
                // to the least-bad trial anyway (the previous behaviour) let the iterate
                // drift into states with a residual up to 5 x the initial one, after which
                // the adaptive forcing stayed at zeta_max and the next directions were
                // worse still (FRC formation: residual bouncing 3e3-9e3 for 15 iterations,
                // then GMRES divergence). Keep the iterate, re-evaluate its residual and
                // re-solve with the tight linear tolerance; give up after two in a row.
                a_U.Copy(m_U_ls);
                have_residual = false;
                force_tight_solve = true;
                ++ls_fail_count;
                if (this->m_verbose) {
                    amrex::Print() << "Newton: line search failed (best trial "
                                   << best_norm/norm_abs << " x the residual), direction "
                                   << "rejected" << (ls_fail_count < 2
                                       ? ", re-solving with the tight linear tolerance\n"
                                       : " twice in a row, exiting the step\n");
                }
                if (ls_fail_count >= 2) {
                    // exhausted: report the failure explicitly (a stale success status from
                    // the previous solve must not survive) and put the operator state back
                    // on the kept iterate -- the trial evaluations left particles, currents
                    // and pressure at the last rejected trial
                    EvalResidual(m_F, a_U, a_b, a_time, iter);
                    norm_abs = m_F.norm2();
                    norm_rel = norm_abs/norm0;
                    this->m_exit_status = (m_require_convergence ? -5 : 0);
                    iter++;
                    break;
                }
            }
        } else {
            a_U -= m_dU;
        }

        iter++;
        if (iter >= m_maxits) {
            // the last update was applied but not evaluated: evaluate it so the reported
            // norm (and the caller's halving-limit acceptance test) refer to the returned
            // iterate, and so the operator state matches it
            EvalResidual(m_F, a_U, a_b, a_time, iter);
            norm_abs = m_F.norm2();
            norm_rel = norm_abs/norm0;
            if (this->m_verbose) {
                amrex::Print() << "Newton: exiting at iter = " << std::setw(3) << iter
                               << ". Maximum iteration reached: iter = " << m_maxits
                               << ", relative norm of the returned iterate " << norm_rel << "\n";
            }
            break;
        }

    }

    // Update total iteration count
    m_total_iters += iter;
    this->m_last_iters = iter;
    this->m_last_norm_rel = norm_rel;
    m_total_linsol_iters += linear_solver_iters;

    if (m_rtol > 0. && iter == m_maxits) {
        std::stringstream convergenceMsg;
        convergenceMsg << "Newton solver failed to converge after " << iter <<
                          " iterations. Relative norm is " << norm_rel <<
                          " and the relative tolerance is " << m_rtol <<
                          ". Absolute norm is " << norm_abs <<
                          " and the absolute tolerance is " << m_atol;
        if (this->m_verbose) { amrex::Print() << convergenceMsg.str() << "\n"; }
        ablastr::warn_manager::WMRecordWarning("NewtonSolver", convergenceMsg.str());

        if (m_require_convergence) { this->m_exit_status = -5; }
        else { this->m_exit_status = 0; }
    }

    if (!this->m_diagnostic_file.empty() && amrex::ParallelDescriptor::IOProcessor() &&
        ((a_step+1)%this->m_diagnostic_interval==0 || a_step==0)) {
        std::ofstream diagnostic_file{this->m_diagnostic_file, std::ofstream::out | std::ofstream::app};
        diagnostic_file << std::setprecision(14);
        diagnostic_file << a_step+1;
        diagnostic_file << " ";
        diagnostic_file << a_time + a_dt;
        diagnostic_file << " ";
        diagnostic_file << iter;
        diagnostic_file << " ";
        diagnostic_file << m_total_iters;
        diagnostic_file << " ";
        diagnostic_file << norm_abs;
        diagnostic_file << " ";
        diagnostic_file << norm_rel;
        diagnostic_file << " ";
        diagnostic_file << linear_solver_iters;
        diagnostic_file << " ";
        diagnostic_file << m_total_linsol_iters;
        diagnostic_file << " ";
        diagnostic_file << m_linear_solver->getResidualNorm();
        diagnostic_file << "\n";
        diagnostic_file.close();
    }

}

template <class Vec, class Ops>
void NewtonSolver<Vec,Ops>::EvalResidual ( Vec&         a_F,
                                     const Vec&         a_U,
                                     const Vec&         a_b,
                                           amrex::Real  a_time,
                                           int          a_iter ) const
{
    BL_PROFILE("NewtonSolver::EvalResidual()");
    m_ops->ComputeRHS( m_R, a_U, a_time, a_iter, false );

    // Compute residual: F(U) = U - b - R(U)
    a_F.Copy(a_U);
    a_F -= m_R;
    a_F -= a_b;

}

#endif
