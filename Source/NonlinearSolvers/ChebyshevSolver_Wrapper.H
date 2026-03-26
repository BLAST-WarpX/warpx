/* Copyright 2026 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef CHEBYSHEV_SOLVER_WRAPPER_H_
#define CHEBYSHEV_SOLVER_WRAPPER_H_

#include "LinearSolver.H"
#include "ChebyshevImpl.H"
#include "FieldSolver/ImplicitSolvers/MFMatrixSolverOps.H"

#include <memory>

/**
 * \brief LinearSolver wrapper for Chebyshev semi-iterative method.
 *
 *  Wraps ChebyshevImpl<Vec> and MFMatrixSolverOps<Vec> into
 *  the LinearSolver interface.
 *
 *  The matrix-vector product J*v is provided by JacobianFunctionMF::apply().
 *  The diagonal of the Jacobian comes from the mass matrices (the EM
 *  curl-curl term has no diagonal component): diag(J) = I + pc_factor * M_diag.
 */
template <typename Vec, typename LinOp>
class ChebyshevSolver : public LinearSolver<Vec, LinOp>
{
public:

    using RT = typename LinOp::RT;

    ChebyshevSolver () = default;
    ~ChebyshevSolver () override = default;

    ChebyshevSolver (const ChebyshevSolver&) = delete;
    ChebyshevSolver& operator= (const ChebyshevSolver&) = delete;
    ChebyshevSolver (ChebyshevSolver&&) noexcept = delete;
    ChebyshevSolver& operator= (ChebyshevSolver&&) noexcept = delete;

    void define (LinOp& a_linop) override
    {
        m_linop = &a_linop;

        // Create a matvec callable that wraps LinOp::apply()
        auto matvec = [linop = m_linop](Vec& y, const Vec& x) {
            linop->apply(y, x);
        };

        // Get diagonal blocks of original (Jacobian) mass matrices
        auto* ops = m_linop->getOps();
        auto* bcoefs = ops->GetMassMatricesJacDiag();
        int n_levels = ops->numAMRLevels();

        m_mf_ops = std::make_unique<MFMatrixSolverOps<Vec>>();
        m_mf_ops->Define(std::move(matvec), bcoefs, n_levels);

        m_solver.Define(m_mf_ops.get(), "chebyshev");
    }

    void solve (Vec& a_sol, Vec const& a_rhs,
                RT a_tol_rel, RT a_tol_abs, int a_its=-1) override
    {
        // Update scaling factor for mass matrix diagonal
        m_mf_ops->SetPCFactor(
            static_cast<RT>(m_linop->getOps()->GetMassMatrixDiagScale()));

        // Update (allocates work vectors, estimates eigenvalues on first call)
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
    ChebyshevImpl<Vec> m_solver;
};

#endif
