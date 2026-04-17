/* Copyright 2025-2026 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef JACOBI_SOLVER_WRAPPER_H_
#define JACOBI_SOLVER_WRAPPER_H_

#include "LinearSolver.H"
#include "WeightedJacobiImpl.H"
#include "FieldSolver/ImplicitSolvers/MFMatrixSolverOps.H"
#include "FieldSolver/ImplicitSolvers/MultiFabMatrixSolverOps.H"
#include "FieldSolver/ImplicitSolvers/SparseMatrixSolverOps.H"

#include <AMReX_ParmParse.H>

#include <memory>
#include <type_traits>

/**
 * \brief LinearSolver wrapper for weighted Jacobi.
 *
 *  When use_pcmat=false (default), the matrix-vector product is provided
 *  by JacobianFunctionMF::apply(). When use_pcmat=true, the assembled
 *  PC matrix (stencil or CSR) is used instead.
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

        if (m_use_pcmat) {
            // Stencil/CSR ops (same dispatch as JacobiPC)
            using OpsType = std::remove_pointer_t<decltype(ops)>;
            if (ops->GetThetaForPC() > amrex::Real(0.0)) {
                auto sp_ops = std::make_unique<SparseMatrixSolverOps<Vec, OpsType>>();
                sp_ops->Define(ops);
                m_matrix_ops = std::move(sp_ops);
            } else {
                auto mf_ops = std::make_unique<MultiFabMatrixSolverOps<Vec, OpsType>>();
                mf_ops->Define(ops);
                m_matrix_ops = std::move(mf_ops);
            }
        } else {
            // Matrix-free ops wrapping JacobianFunctionMF::apply()
            auto matvec = [linop = m_linop](Vec& y, const Vec& x) {
                linop->apply(y, x);
            };
            auto* bcoefs = ops->GetMassMatricesJacDiag();
            int n_levels = ops->numAMRLevels();
            auto mf_ops = std::make_unique<MFMatrixSolverOps<Vec>>();
            mf_ops->Define(std::move(matvec), bcoefs, n_levels);
            m_matrix_ops = std::move(mf_ops);
        }

        m_solver.Define(m_matrix_ops.get(), "weighted_jacobi");
    }

    void update (const Vec& a_U) override
    {
        if (!m_use_pcmat) {
            m_matrix_ops->SetPCFactor(
                static_cast<RT>(m_linop->getOps()->GetMassMatrixDiagScale()));
        }
        m_solver.Update(a_U);
    }

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

    [[nodiscard]] RT getResidualNorm () const override
    {
        return m_solver.getResidualNorm();
    }

    [[nodiscard]] bool usePCMat () const { return m_use_pcmat; }

private:

    LinOp* m_linop = nullptr;
    bool m_use_pcmat = true;
    std::unique_ptr<MatrixSolverOps<Vec>> m_matrix_ops;
    WeightedJacobiImpl<Vec> m_solver;
};

#endif
