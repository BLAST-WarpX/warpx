/* Copyright 2025 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef PETSC_SNES_WRAPPER_H_
#define PETSC_SNES_WRAPPER_H_

#include "NewtonSolverParams.H"
#include "NonlinearSolver.H"
#include "WarpX_PETSc.H"

#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_REAL.H>
#include <AMReX_Config.H>
#include <memory>

#ifdef AMREX_USE_PETSC

/**
 * \brief Wrapper for PETSc's SNES solver
 *
 *    This class is a wrapper for PETSc's SNES nonlinear solver that inheritis from
 *    WarpX's NonlinearSolver base class. See the documentation of PETSc's SNES
 *    solver for more details. This class is templated on a vector class Vec,
 *    and an operator class Ops.
 *
 *    PETScSNES is just a wrapper; the actual implementation is in SNES_impl in
 *    Warpx_PETSc.H/cpp. This is to avoid including PETSc's header files here
 *    and consequent conflicts between PETSc and WarpX variables.
 */

template <class Vec, class Ops>
class PETScSNES : public NonlinearSolver<Vec,Ops>
{
    public:

        using RT = amrex::Real;

        PETScSNES () = default;
        ~PETScSNES () override = default;

        // Prohibit Move and Copy operations
        PETScSNES(const PETScSNES&) = delete;
        PETScSNES& operator=(const PETScSNES&) = delete;
        PETScSNES(PETScSNES&&) noexcept = delete;
        PETScSNES& operator=(PETScSNES&&) noexcept = delete;

        void Define ( const Vec& a_U, Ops* a_op ) override
        {
            this->m_params.Parse("newton");
            this->m_params.ResetInvalidRelativeTolerance("newton");

            // The linear solver is always PETSc's KSP
            m_newton_params.linear_solver_type = LinearSolverType::petsc_ksp;
            m_newton_params.Parse();
            if (m_newton_params.linear_solver_type != LinearSolverType::petsc_ksp) {
                ablastr::warn_manager::WMRecordWarning("PETSc SNES",
                    "newton.linear_solver = " +
                    amrex::getEnumNameString(m_newton_params.linear_solver_type) +
                    " is ignored; petsc_snes uses PETSc KSP as the linear solver.");
                m_newton_params.linear_solver_type = LinearSolverType::petsc_ksp;
            }

            m_solver = std::make_unique<warpx_petsc::SNES_impl>(a_U, a_op, this->m_params,
                                                                m_newton_params);

            this->m_is_defined = m_solver->isDefined();
            this->m_usePC = m_solver->usePC();

            this->InitDiagnosticFile({"gmres_iters", "gmres_total_iters", "gmres_last_res"});
        }

        void Solve (Vec& a_U,
                    const Vec& a_b,
                    RT a_time,
                    RT a_dt,
                    int a_step,
                    bool verbose_step)  const override
        {

            bool const verbose = verbose_step ? this->m_params.verbose : 0;
            m_solver->setVerbose(verbose);
            m_solver->solve(a_U, a_b, a_time, a_dt, a_step);
            this->m_exit_status = m_solver->getStatus();

            this->WriteDiagnostics(a_step, a_time + a_dt,
                                   m_solver->numIters(), m_solver->numTotalIters(),
                                   m_solver->normAbs(), m_solver->normRel(),
                                   m_solver->numLinIters(), m_solver->numTotalLinIters(),
                                   m_solver->normLinResidual());
        }

        void PrintParams () const override
        {
            this->m_params.Print("PETSc SNES");
            m_newton_params.Print("PETSc SNES");
            m_solver->printParams();
        }

        void Verbose (bool a_verbose) override
        {
            this->m_params.verbose = a_verbose;
            m_solver->setVerbose(a_verbose);
        }

        PreconditionerType GetPreconditionerType () const override
        {
            return m_solver->getPreconditionerType();
        }

    private:

        //! Parameters specific to Newton-type solvers.
        NewtonSolverParams m_newton_params;

        std::unique_ptr<warpx_petsc::SNES_impl> m_solver = nullptr;
};

#endif
#endif
