/* Copyright 2025 Debojyoti Ghosh
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

#include <memory>

/**
 * \brief LinearSolver wrapper for weighted Jacobi.
 *
 *  Wraps WeightedJacobiImpl<Vec> and MFMatrixSolverOps<Vec> into
 *  the LinearSolver interface.
 *
 *  The matrix-vector product J*v is provided by JacobianFunctionMF::apply().
 *  The diagonal of the Jacobian comes from the mass matrices (the EM
 *  curl-curl term has no diagonal component): diag(J) = I + M_diag.
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

        // Create a matvec callable that wraps LinOp::apply()
        auto matvec = [linop = m_linop](Vec& y, const Vec& x) {
            linop->apply(y, x);
        };

        // Get mass matrix data from the Ops (ImplicitSolver)
        auto* ops = m_linop->getOps();
        auto* bcoefs = ops->GetMassMatricesCoeff();
        int n_levels = ops->numAMRLevels();

        m_mf_ops = std::make_unique<MFMatrixSolverOps<Vec>>();
        m_mf_ops->Define(std::move(matvec), bcoefs, n_levels);

        m_solver.Define(m_mf_ops.get(), "weighted_jacobi");
    }

    void solve (Vec& a_sol, Vec const& a_rhs,
                RT a_tol_rel, RT a_tol_abs, int a_its=-1) override
    {
        // Update on every solve (allocates work vectors on first call,
        // no-op thereafter for MFMatrixSolverOps)
        m_solver.Update(a_rhs);

        // Override tolerances from Newton
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

private:

    LinOp* m_linop = nullptr;
    std::unique_ptr<MFMatrixSolverOps<Vec>> m_mf_ops;
    WeightedJacobiImpl<Vec> m_solver;
};

#endif
