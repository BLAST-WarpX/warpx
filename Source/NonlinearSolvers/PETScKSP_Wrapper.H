/* Copyright 2025 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef PETSC_KSP_WRAPPER_H_
#define PETSC_KSP_WRAPPER_H_

#include "LinearSolver.H"
#include "WarpX_PETSc.H"

#include <AMReX_Config.H>
#include <memory>

#ifdef AMREX_USE_PETSC

/**
 * \brief Wrapper for PETSc's KSP solver
 *
 *    This class is a wrapper for PETSc's KSP linear solver that inheritis from
 *    WarpX's LinearSolver base class. See the documentation of PETSc's KSP
 *    solver for more details. This class is templated on a vector class Vec,
 *    and an operator class Ops.
 *
 *    The Ops class must have the following function:
 *    ComputeRHS( R_vec, U_vec, time, nl_iter, from_jacobian ),
 *    where U_vec and R_vec are of type Vec.
 *
 *    The Vec class must have basic math operators, such as Copy, +=, -=,
 *    increment(), linComb(), scale(), etc.. See WarpXSolverVec.H for an example.
 *
 *    PETScKSP is just a wrapper; the actual implementation is in KSP_impl in
 *    Warpx_PETSc.H/cpp. This is to avoid including PETSc's header files here
 *    and consequent conflicts between PETSc and WarpX variables.
 */

template <typename Vec, typename Ops>
class PETScKSP : public LinearSolver<Vec,Ops>
{
public:

    using RT = typename Ops::RT; // double or float

    PETScKSP () = default;

    virtual ~PETScKSP () = default;

    // Prohibit Move and Copy operations
    PETScKSP(const PETScKSP&) = delete;
    PETScKSP& operator=(const PETScKSP&) = delete;
    PETScKSP(PETScKSP&&) noexcept = delete;
    PETScKSP& operator=(PETScKSP&&) noexcept = delete;

    //! Defines with a reference to Ops. It's the user's responsibility to
    //! keep the Ops object alive for GMRES to be functional. This function
    //! must be called before solve() can be called.
    void define (Ops& a_linop) override
    {
        m_solver = std::make_unique<warpx_petsc::KSP_impl>(a_linop);
    }

    /**
     * \brief Solve the linear system
     *
     * \param a_sol     unknowns, i.e., x in A x = b.
     * \param a_rhs     RHS, i.e., b in A x = b.
     * \param a_tol_rel relative tolerance.
     * \param a_tol_abs absolute tolerance.
     * \patam a_its     optional argument specifying the maximum number of iterations.
     */
    void solve (Vec& a_sol,
                Vec const& a_rhs,
                RT a_tol_rel,
                RT a_tol_abs,
                int a_its=-1) override
    {
        if (!m_solver->isDefined()) {
            m_solver->createObjects(a_sol);
        }
        m_solver->setTolerances(a_tol_rel, a_tol_abs, a_its);
        m_solver->solve(a_sol, a_rhs);
    }

    //! Sets verbosity.
    void setVerbose (int v) override { m_solver->setVerbose(v); }

    //! Sets restart length. The default is 30.
    void setRestartLength (int rl) override { amrex::ignore_unused(rl); }

    //! Sets the max number of iterations
    void setMaxIters (int niters) override { m_solver->setMaxIters(niters); }

    //! Gets the number of iterations.
    [[nodiscard]] int getNumIters () const override { return m_solver->getNumIters(); }

    //! Gets the solver status.
    [[nodiscard]] int getStatus () const override { return m_solver->getStatus(); }

    //! Gets the 2-norm of the residual.
    [[nodiscard]] RT getResidualNorm () const override { return m_solver->getResidualNorm(); }

private:

    std::unique_ptr<warpx_petsc::KSP_impl> m_solver = nullptr;
};

#endif
#endif
