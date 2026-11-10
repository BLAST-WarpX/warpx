/* Copyright 2024 Justin Angus
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef PICARD_SOLVER_H_
#define PICARD_SOLVER_H_

#include "NonlinearSolver.H"

#include <AMReX_ParmParse.H>
#include "Utils/TextMsg.H"

#include <vector>
#include <istream>
#include <filesystem>

/**
 * \brief Picard fixed-point iteration method to solve nonlinear
 *  equation of form: U = b + R(U). U is the solution vector. b
 *  is a constant. R(U) is some nonlinear function of U, which
 *  is computed in the Ops function ComputeRHS().
 */

template<class Vec, class Ops>
class PicardSolver : public NonlinearSolver<Vec,Ops>
{
public:

    PicardSolver() = default;

    ~PicardSolver() override = default;

    // Prohibit Move and Copy operations
    PicardSolver(const PicardSolver&) = delete;
    PicardSolver& operator=(const PicardSolver&) = delete;
    PicardSolver(PicardSolver&&) noexcept = delete;
    PicardSolver& operator=(PicardSolver&&) noexcept = delete;

    void Define (const Vec&  a_U,
                       Ops*  a_ops) override;

    void Solve (Vec&   a_U,
          const Vec&   a_b,
          amrex::Real  a_time,
          amrex::Real  a_dt,
          int          a_step,
          bool verbose_step) const override;

    void PrintParams () const override
    {
        this->m_params.Print("Picard");
    }

private:

    /**
     * \brief Intermediate Vec containers used by the solver.
     */
    mutable Vec m_Usave, m_R;

    /**
     * \brief Pointer to Ops class.
     */
    Ops* m_ops = nullptr;

    /**
     * \brief Total iterations for the diagnostic file
     */
    mutable int m_total_iters = 0;

    void ParseParameters( );

};

template <class Vec, class Ops>
void PicardSolver<Vec,Ops>::Define (const Vec&  a_U,
                                    Ops*        a_ops)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !this->m_is_defined,
        "Picard nonlinear solver object is already defined!");

    ParseParameters();

    m_Usave.Define(a_U);
    m_R.Define(a_U);

    m_ops = a_ops;

    this->m_is_defined = true;

    this->InitDiagnosticFile();

}

template <class Vec, class Ops>
void PicardSolver<Vec,Ops>::ParseParameters ()
{
    this->m_params.Parse("picard");
}

template <class Vec, class Ops>
void PicardSolver<Vec,Ops>::Solve (Vec&         a_U,
                             const Vec&         a_b,
                                   amrex::Real  a_time,
                                   amrex::Real  a_dt,
                                   int          a_step,
                                   bool verbose_step) const
{
    BL_PROFILE("PicardSolver::Solve()");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        this->m_is_defined,
        "PicardSolver::Solve() called on undefined object");
    using namespace amrex::literals;

    //
    // Picard fixed-point iteration method to solve nonlinear
    // equation of form: U = b + R(U)
    //

    amrex::Real norm_abs = 0.;
    amrex::Real norm0 = 1._rt;
    amrex::Real norm_rel = 0.;

    int iter;
    for (iter = 0; iter < this->m_params.maxits;) {

        // Save previous state for norm calculation
        m_Usave.Copy(a_U);

        // Update the solver state (a_U = a_b + m_R)
        m_ops->ComputeRHS( m_R, a_U, a_time, iter, false );
        a_U.Copy(a_b);
        a_U += m_R;

        // Compute the step norm and update iter
        m_Usave -= a_U;
        norm_abs = m_Usave.norm2();
        if (iter == 0) {
            if (norm_abs > 0.) { norm0 = norm_abs; }
            else { norm0 = 1._rt; }
        }
        norm_rel = norm_abs/norm0;
        iter++;

        // Check for convergence criteria
        if ((this->m_params.verbose and verbose_step) || iter == this->m_params.maxits) {
            amrex::Print() << "Picard: iter = " << std::setw(3) << iter <<  ", norm = "
                           << std::scientific << std::setprecision(5) << norm_abs << " (abs.), "
                           << std::scientific << std::setprecision(5) << norm_rel << " (rel.)" << "\n";
        }

        if (norm_abs < this->m_params.atol) {
            this->m_exit_status = 2;
            if (this->m_params.verbose and verbose_step) {
                amrex::Print() << "Picard: exiting at iter = " << std::setw(3) << iter
                               << ". Satisfied absolute tolerance " << this->m_params.atol << "\n";
            }
            break;
        }

        if (norm_rel < this->m_params.rtol) {
            this->m_exit_status = 3;
            if (this->m_params.verbose and verbose_step) {
                amrex::Print() << "Picard: exiting at iter = " << std::setw(3) << iter
                               << ". Satisfied relative tolerance " << this->m_params.rtol << "\n";
            }
            break;
        }

        if (iter >= this->m_params.maxits) {
            if (this->m_params.verbose and verbose_step) {
                amrex::Print() << "Picard: exiting at iter = " << std::setw(3) << iter
                               << ". Maximum iteration reached: iter = " << this->m_params.maxits << "\n";
            }
            break;
        }

    }

    // Update total iteration count
    m_total_iters += iter;

    if (this->m_params.rtol > 0. && iter == this->m_params.maxits) {
        std::stringstream convergenceMsg;
        convergenceMsg << "Picard solver failed to converge after " << iter <<
                          " iterations. Relative norm is " << norm_rel <<
                          " and the relative tolerance is " << this->m_params.rtol <<
                          ". Absolute norm is " << norm_abs <<
                          " and the absolute tolerance is " << this->m_params.atol;
        if (this->m_params.verbose) { amrex::Print() << convergenceMsg.str() << "\n"; }
        ablastr::warn_manager::WMRecordWarning("PicardSolver", convergenceMsg.str());

        if (this->m_params.require_convergence) { this->m_exit_status = -5; }
        else { this->m_exit_status = 0; }
    }

    this->WriteDiagnostics(a_step, a_time + a_dt, iter, m_total_iters, norm_abs, norm_rel);

}

#endif
