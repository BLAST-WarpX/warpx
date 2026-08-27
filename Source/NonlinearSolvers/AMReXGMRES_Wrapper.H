/* Copyright 2025 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef AMREX_GMRES_WRAPPER_H_
#define AMREX_GMRES_WRAPPER_H_

#include "LinearSolver.H"

#include <AMReX_GMRES.H>

#include <memory>

/**
 * \brief Wrapper for AMReX's GMRES solver
 *
 *    This class is a wrapper for AMReX's GMRES algorithm that inherits from
 *    WarpX's LinearSolver base class. See the documentation of AMReX's GMRES
 *    solver for more details. This class is templated on a vector class Vec,
 *    and a linear operator class LinOp.
 *
 *    The LinOp class represents the linear operator A of the linear system
 *    A x = b solved by solve(). It must implement the interface defined in
 *    LinearFunction.H; in particular, an apply( Ax_vec, x_vec ) function
 *    that computes the action A(x) of the operator on a vector x.
 *
 *    The Vec class must have basic math operators, such as Copy, +=, -=,
 *    increment(), linComb(), scale(), etc.. See WarpXSolverVec.H for an example.
 */
template <typename Vec, typename LinOp>
class AMReXGMRES : public LinearSolver<Vec,LinOp>
{
public:

    using RT = typename LinOp::RT; // double or float

    AMReXGMRES () = default;

    ~AMReXGMRES () override = default;

    // Prohibit Move and Copy operations
    AMReXGMRES(const AMReXGMRES&) = delete;
    AMReXGMRES& operator=(const AMReXGMRES&) = delete;
    AMReXGMRES(AMReXGMRES&&) noexcept = delete;
    AMReXGMRES& operator=(AMReXGMRES&&) noexcept = delete;

    //! Defines with a reference to LinOp. It's the user's responsibility to
    //! keep the LinOp object alive for GMRES to be functional. This function
    //! must be called before solve() can be called.
    AMREX_FORCE_INLINE
    void define (LinOp& linop) override
    {
        m_solver = std::make_unique<amrex::GMRES<Vec,LinOp>>();
        m_solver->define(linop);
    }

    /**
     * \brief Solve the linear system A x = b for x, where A is the linear
     *        operator whose action A(x) on a vector x is computed by
     *        LinOp::apply(), and where the right-hand side b is a given
     *        vector that does not depend on x.
     *
     * \param a_sol     unknowns, i.e., x in A x = b.
     * \param a_rhs     RHS, i.e., b in A x = b.
     * \param a_tol_rel relative tolerance.
     * \param a_tol_abs absolute tolerance.
     * \param a_its     optional argument specifying the maximum number of iterations.
     */
    AMREX_FORCE_INLINE
    void solve (Vec& a_sol,
                Vec const& a_rhs,
                RT a_tol_rel,
                RT a_tol_abs,
                int a_its=-1) override
    {
        m_solver->solve(a_sol, a_rhs, a_tol_rel, a_tol_abs, a_its);
    }

    //! Sets verbosity.
    AMREX_FORCE_INLINE
    void setVerbose (int v) override { m_solver->setVerbose(v); }

    //! Sets restart length. The default is 30.
    AMREX_FORCE_INLINE
    void setRestartLength (int rl) override { m_solver->setRestartLength(rl); }

    //! Sets the max number of iterations
    AMREX_FORCE_INLINE
    void setMaxIters (int niters) override { m_solver->setMaxIters(niters); }

    //! Gets the number of iterations.
    AMREX_FORCE_INLINE
    int getNumIters () const override { return m_solver->getNumIters(); }

    //! Gets the solver status.
    AMREX_FORCE_INLINE
    int getStatus () const override { return m_solver->getStatus(); }

    //! Gets the 2-norm of the residual.
    AMREX_FORCE_INLINE
    RT getResidualNorm () const override { return m_solver->getResidualNorm(); }

private:
    std::unique_ptr<amrex::GMRES<Vec,LinOp>> m_solver = nullptr;
};

#endif
