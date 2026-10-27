/* Copyright 2025-2026 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef JACOBI_SOLVER_WRAPPER_H_
#define JACOBI_SOLVER_WRAPPER_H_

#include "LinearSolver.H"
#include "MatrixPrintControl.H"
#include "MatrixSolverOps.H"
#include "WeightedJacobiImpl.H"

#include <AMReX_ParmParse.H>

#include <memory>
#include <string>
#include <type_traits>

/**
 * \brief LinearSolver wrapper for weighted Jacobi.
 *
 *  When use_pcmat=true (default), Jacobi iterates on the assembled PC matrix.
 *  When use_pcmat=false, it iterates on the Jacobian, with the matrix-vector
 *  product provided by JacobianFunctionMF::apply().
 */
template <typename Vec, typename LinOp>
class JacobiSolver : public LinearSolver<Vec, LinOp>
{
public:

    using RT = typename LinOp::RT;

    JacobiSolver () = default;
    ~JacobiSolver () override = default;

    JacobiSolver (const JacobiSolver&) = delete;
    JacobiSolver& operator= (const JacobiSolver&) = delete;
    JacobiSolver (JacobiSolver&&) noexcept = delete;
    JacobiSolver& operator= (JacobiSolver&&) noexcept = delete;

    void define (LinOp& a_linop) override
    {
        m_linop = &a_linop;
        auto* ops = m_linop->getOps();

        const amrex::ParmParse pp("weighted_jacobi");
        pp.query("use_pcmat", m_use_pcmat);

        auto matvec = [linop = m_linop](Vec& y, const Vec& x) {
            linop->apply(y, x);
        };
        using OpsType = std::remove_pointer_t<decltype(ops)>;
        m_matrix_ops = OpsType::MakeJacobianMatrixOps(ops, std::move(matvec), m_use_pcmat);

        m_solver.Define(m_matrix_ops.get(), "weighted_jacobi");
        m_print.Define("weighted_jacobi");
    }

    void update () override
    {
        // The assembled PC matrix follows the PC update schedule
        if (m_use_pcmat && !m_matrix_stale) { return; }
        m_solver.Update();
        m_matrix_stale = false;

        if (m_print.ShouldPrint(m_linop->getOps()->GetCurrentStep())) {
            auto header = m_print.HeaderLines();
            header.emplace_back("use_pcmat: " + std::string(m_use_pcmat ? "true" : "false"));
            header.emplace_back("omega: " + std::to_string(m_solver.getOmega()));
            m_matrix_ops->PrintMatrix(m_print.FileName(), header);
        }
    }

    void invalidatePreconditioner () override { m_matrix_stale = true; }

    void solve (Vec& a_sol, Vec const& a_rhs,
                RT a_tol_rel, RT a_tol_abs, int a_its=-1) override
    {
        m_solver.setTolerances(a_tol_rel, a_tol_abs);
        if (a_its > 0) { m_solver.setMaxIters(a_its); }
        m_solver.Solve(a_sol, a_rhs);
    }

    void setVerbose (int v) override
    {
        m_solver.setVerbose(v > 0);
    }

    void setMaxIters (int niters) override
    {
        m_solver.setMaxIters(niters);
    }

    [[nodiscard]] int getNumIters () const override
    {
        return m_solver.getNumIters();
    }

    [[nodiscard]] int getStatus () const override
    {
        return m_solver.getStatus();
    }

    [[nodiscard]] bool converged () const override
    {
        return (m_solver.getStatus() == 0);
    }

    [[nodiscard]] RT getResidualNorm () const override
    {
        return m_solver.getResidualNorm();
    }

    [[nodiscard]] bool usePCMat () const { return m_use_pcmat; }

private:

    LinOp* m_linop = nullptr;
    bool m_use_pcmat = true;
    bool m_matrix_stale = true;
    std::unique_ptr<MatrixSolverOps<Vec>> m_matrix_ops;
    WeightedJacobiImpl<Vec> m_solver;
    MatrixPrintControl m_print;
};

#endif
