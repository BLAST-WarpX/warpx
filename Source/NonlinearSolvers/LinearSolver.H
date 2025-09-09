#ifndef WARPX_LINEAR_SOLVER_H_
#define WARPX_LINEAR_SOLVER_H_

/**
 * \brief Top-level class for the linear solver
 *
 *    This class is templated on a vector class Vec, and an operator class Ops.
 *
 *    The Ops class must have the following function:
 *    ComputeRHS( R_vec, U_vec, time, nl_iter, from_jacobian ),
 *    where U_vec and R_vec are of type Vec.
 *
 *    The Vec class must have basic math operators, such as Copy, +=, -=,
 *    increment(), linComb(), scale(), etc.. See WarpXSolverVec.H for an example.
 */

template <typename Vec, typename Ops>
class LinearSolver
{
public:

    using RT = typename Ops::RT; // double or float

    LinearSolver () = default;

    virtual ~LinearSolver() = default;

    // Prohibit Move and Copy operations
    LinearSolver(const LinearSolver&) = delete;
    LinearSolver& operator=(const LinearSolver&) = delete;
    LinearSolver(LinearSolver&&) noexcept = delete;
    LinearSolver& operator=(LinearSolver&&) noexcept = delete;

    //! Defines with a reference to Ops. It's the user's responsibility to
    //! keep the Ops object alive for linear solver to be functional. This function
    //! must be called before solve() can be called.
    virtual void define (Ops& linop) = 0;

    /**
     * \brief Solve the linear system
     *
     * \param a_sol     unknowns, i.e., x in A x = b.
     * \param a_rhs     RHS, i.e., b in A x = b.
     * \param a_tol_rel relative tolerance.
     * \param a_tol_abs absolute tolerance.
     * \param a_its     optional argument specifying the maximum number of iterations.
     */
    virtual void solve (Vec& a_sol,
                        Vec const& a_rhs,
                        RT a_tol_rel,
                        RT a_tol_abs,
                        int a_its=-1) = 0;

    //! Sets verbosity.
    virtual void setVerbose (int v) = 0;

    //! Sets restart length for GMRES-like solvers. Base implementation
    //! provided for solvers that don't need this.
    virtual void setRestartLength (int rl) { amrex::ignore_unused(rl); }

    //! Sets the max number of iterations
    virtual void setMaxIters (int niters) = 0;

    //! Gets the number of iterations.
    virtual int getNumIters () const = 0;

    //! Gets the solver status.
    virtual int getStatus () const = 0;

    //! Gets the 2-norm of the residual.
    virtual RT getResidualNorm () const = 0;

protected:

private:
};

#endif
