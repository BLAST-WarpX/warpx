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
 *    and an operator class Ops.
 *
 *    The Ops class must have the following function:
 *    ComputeRHS( R_vec, U_vec, time, nl_iter, from_jacobian ),
 *    where U_vec and R_vec are of type Vec.
 *
 *    The Vec class must have basic math operators, such as Copy, +=, -=,
 *    increment(), linComb(), scale(), etc.. See WarpXSolverVec.H for an example.
 */
template <typename Vec, typename Ops>
class AMReXGMRES : public LinearSolver<Vec,Ops>
{
public:

    using RT = typename Ops::RT; // double or float

    AMReXGMRES () = default;

    virtual ~AMReXGMRES () = default;

    // Prohibit Move and Copy operations
    AMReXGMRES(const AMReXGMRES&) = delete;
    AMReXGMRES& operator=(const AMReXGMRES&) = delete;
    AMReXGMRES(AMReXGMRES&&) noexcept = delete;
    AMReXGMRES& operator=(AMReXGMRES&&) noexcept = delete;

    //! Defines with a reference to Ops. It's the user's responsibility to
    //! keep the Ops object alive for GMRES to be functional. This function
    //! must be called before solve() can be called.
    AMREX_FORCE_INLINE
    void define (Ops& linop)
    {
        m_solver = std::make_unique<amrex::GMRES<Vec,Ops>>();
        m_solver->define(linop);
    }

    /**
     * \brief Solve the linear system
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
                int a_its=-1)
    {
        m_solver->solve(a_sol, a_rhs, a_tol_rel, a_tol_abs, a_its);
    }

    //! Sets verbosity.
    AMREX_FORCE_INLINE
    void setVerbose (int v) { m_solver->setVerbose(v); }

    //! Sets restart length. The default is 30.
    AMREX_FORCE_INLINE
    void setRestartLength (int rl) { m_solver->setRestartLength(rl); }

    //! Sets the max number of iterations
    AMREX_FORCE_INLINE
    void setMaxIters (int niters) { m_solver->setMaxIters(niters); }

    //! Gets the number of iterations.
    AMREX_FORCE_INLINE
    int getNumIters () const { return m_solver->getNumIters(); }

    //! Gets the solver status.
    AMREX_FORCE_INLINE
    int getStatus () const { return m_solver->getStatus(); }

    //! Gets the 2-norm of the residual.
    AMREX_FORCE_INLINE
    RT getResidualNorm () const { return m_solver->getResidualNorm(); }

private:
    std::unique_ptr<amrex::GMRES<Vec,Ops>> m_solver = nullptr;
};

#endif
