/* Copyright 2025 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifdef AMREX_USE_PETSC

#ifndef PETSC_KSP_WRAPPER_H_
#define PETSC_KSP_WRAPPER_H_

#include "LinearSolver.H"

#include <memory>

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
    AMREX_FORCE_INLINE
    void define (Ops& a_linop)
    {
        amrex::ignore_unused(a_linop);
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
    AMREX_FORCE_INLINE
    void solve (Vec& a_sol,
                Vec const& a_rhs,
                RT a_tol_rel,
                RT a_tol_abs,
                int a_its=-1)
    {
        amrex::ignore_unused(a_sol);
        amrex::ignore_unused(a_rhs);
        amrex::ignore_unused(a_tol_rel);
        amrex::ignore_unused(a_tol_abs);
        amrex::ignore_unused(a_its);
    }

    //! Sets verbosity.
    AMREX_FORCE_INLINE
    void setVerbose (int v) { amrex::ignore_unused(v); }

    //! Sets restart length. The default is 30.
    AMREX_FORCE_INLINE
    void setRestartLength (int rl) { amrex::ignore_unused(rl); }

    //! Sets the max number of iterations
    AMREX_FORCE_INLINE
    void setMaxIters (int niters) { amrex::ignore_unused(niters); }

    //! Gets the number of iterations.
    AMREX_FORCE_INLINE
    int getNumIters () const { return -1; }

    //! Gets the solver status.
    AMREX_FORCE_INLINE
    int getStatus () const { return -1; }

    //! Gets the 2-norm of the residual.
    AMREX_FORCE_INLINE
    RT getResidualNorm () const { return 0.0; }

private:
};

#endif
#endif
