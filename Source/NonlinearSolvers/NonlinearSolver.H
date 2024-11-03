#ifndef WARPX_NONLINEAR_SOLVER_H_
#define WARPX_NONLINEAR_SOLVER_H_

#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>

#include <AMReX_BaseFwd.H>

#include <string>
#include <array>
#include <memory>

/**
 * \brief Top-level class for the nonlinear solver
 *
 *    This class is templated on a vector class Vec, and an operator class Ops.
 *
 *    The Ops class must have the following function:
 *    ComputeRHS( R_vec, U_vec, time, dt, nl_iter, from_jacobian ),
 *    where U_vec and R_vec are of type Vec.
 *
 *    The Vec class must have basic math operators, such as Copy, +=, -=,
 *    increment(), linComb(), scale(), etc.. See WarpXSolverVec.H for an example.
 */

template<class Vec, class Ops>
class NonlinearSolver
{
public:

    NonlinearSolver() = default;

    virtual ~NonlinearSolver() = default;

    // Prohibit Move and Copy operations
    NonlinearSolver(const NonlinearSolver&) = delete;
    NonlinearSolver& operator=(const NonlinearSolver&) = delete;
    NonlinearSolver(NonlinearSolver&&) noexcept = delete;
    NonlinearSolver& operator=(NonlinearSolver&&) noexcept = delete;

    /**
     * \brief Read user-provided parameters that control the nonlinear solver.
     * Allocate intermediate data containers needed by the solver. For Newton,
     * setup the linear solver for computing the Newton step.
     */
    virtual void Define ( const Vec&,
                          Ops* ) = 0;

    /**
     * \brief Check if the nonlinear solver has been defined.
     */
    [[nodiscard]] bool IsDefined () const { return m_is_defined; }

    /**
     * \brief Solve the specified nonlinear equation for U.
     *  Picard: U = b + R(U).
     *  Newton: F(U) = U - b - R(U) = 0.
     */
    virtual void Solve ( Vec&,
                   const Vec&,
                         amrex::Real,
                         amrex::Real ) const = 0;

    /**
     * \brief Print parameters used by the nonlinear solver.
     */
    virtual void PrintParams () const = 0;

    /**
     * \brief Return the convergence parameters used by the nonlinear solver.
     */
    virtual void GetSolverParams (amrex::Real&, amrex::Real&, int&) = 0;

    /**
     * \brief Allow caller to dynamically change the verbosity flag. For
     * example, one may want to only print solver information every 100 steps.
     */
    void Verbose ( bool  a_verbose ) { m_verbose = a_verbose; }

protected:

    bool m_is_defined = false;
    mutable bool m_verbose = true;

};

#endif
