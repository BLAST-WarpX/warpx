/* Copyright 2026 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef CHEBYSHEV_IMPL_H_
#define CHEBYSHEV_IMPL_H_

#include "MatrixSolverOps.H"
#include "Utils/TextMsg.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <string>

/**
 * \brief Chebyshev semi-iterative solver.
 *
 *  Solves A x = b using the Chebyshev three-term recurrence:
 *
 *      x_{k+1} = w_{k+1} * (tau * D^{-1} r_k + x_k - x_{k-1}) + x_{k-1}
 *
 *  where r_k = b - A x_k, D = diag(A), tau = 2/(lambda_max + lambda_min),
 *  and w_{k+1} are Chebyshev acceleration parameters derived from the
 *  eigenvalue bounds [lambda_min, lambda_max] of D^{-1} A.
 *
 *  lambda_max is estimated by the power method on D^{-1} A, and lambda_min by
 *  the power method on lambda_max I - D^{-1} A. Both are re-estimated on Update().
 *
 *  All matrix operations are performed through a MatrixSolverOps<T>
 *  implementation which must support Matvec() and InitialGuess() (= D^{-1}).
 */
template <class T>
class ChebyshevImpl
{
public:

    using RT = typename T::value_type;

    ChebyshevImpl () = default;
    ~ChebyshevImpl () = default;

    ChebyshevImpl (const ChebyshevImpl&) = delete;
    ChebyshevImpl& operator= (const ChebyshevImpl&) = delete;
    ChebyshevImpl (ChebyshevImpl&&) noexcept = delete;
    ChebyshevImpl& operator= (ChebyshevImpl&&) noexcept = delete;

    /**
     * \brief Define the solver with a MatrixSolverOps and ParmParse prefix.
     *
     * \param[in] a_ops       pointer to a MatrixSolverOps (caller retains ownership)
     * \param[in] a_pp_prefix ParmParse prefix for reading solver parameters
     */
    void Define (MatrixSolverOps<T>* a_ops, const std::string& a_pp_prefix);

    /** \brief Update matrix ops; a new operator also needs new eigenvalue bounds. */
    void Update (bool a_new_operator = true);

    /** \brief Solve A x = b using Chebyshev iteration.
     *         Estimates the eigenvalue bounds on the first call if needed. */
    void Solve (T& a_x, const T& a_b);

    void printParameters (const std::string& a_name) const;

    [[nodiscard]] bool IsDefined () const { return m_is_defined; }
    [[nodiscard]] bool isVerbose () const { return m_verbose; }
    [[nodiscard]] int getNumIters () const { return m_num_iters; }
    [[nodiscard]] RT getResidualNorm () const { return m_res_norm; }
    [[nodiscard]] int getStatus () const { return m_status; }
    [[nodiscard]] RT getLambdaMin () const { return m_lambda_min; }
    [[nodiscard]] RT getLambdaMax () const { return m_lambda_max; }

    void setVerbose (bool v) { m_verbose = v; }
    void setMaxIters (int n) { m_max_iter = n; }
    void setTolerances (RT rtol, RT atol) { m_rtol = rtol; m_atol = atol; }

private:

    /** \brief Power-method estimate of the largest |eigenvalue| of a_shift I - D^{-1} A,
     *         or of D^{-1} A if a_shift = 0. Uses m_w1 and m_w2. */
    RT PowerMethod (RT a_shift);

    /** \brief Estimate the eigenvalue bounds of D^{-1} A that are not user inputs. */
    void EstimateEigenvalues ();

    bool m_is_defined = false;
    std::string m_name = "chebyshev";

    bool m_verbose = false;
    int m_max_iter = 200;
    RT m_atol = RT(1.0e-16);
    RT m_rtol = RT(1.0e-4);

    RT m_lambda_min = RT(0.0);  /**< lower eigenvalue bound */
    RT m_lambda_max = RT(0.0);  /**< upper eigenvalue bound */
    int m_power_iters = 20;     /**< power method iterations per bound */
    bool m_user_lambda_min = false;
    bool m_user_lambda_max = false;
    bool m_eigenvalues_estimated = false;

    int m_num_iters = 0;
    RT m_res_norm = RT(0);
    int m_status = 0;  /**< 0 = converged, 1 = max iterations reached */

    bool m_work_defined = false;
    T m_x_old;  /**< previous iterate for three-term recurrence */
    T m_w1;     /**< work vector (residual, then saved x) */
    T m_w2;     /**< work vector (preconditioned residual) */

    MatrixSolverOps<T>* m_ops = nullptr;
};

// Implementation

template <class T>
void ChebyshevImpl<T>::Define (MatrixSolverOps<T>* a_ops,
                               const std::string& a_pp_prefix)
{
    m_ops = a_ops;
    m_name = a_pp_prefix;

    const amrex::ParmParse pp(a_pp_prefix);
    pp.query("verbose", m_verbose);
    int verbose_int = 0;
    if (pp.query("verbose_int", verbose_int)) {
        m_verbose = (verbose_int > 0);
    }
    pp.query("max_iter", m_max_iter);
    pp.query("absolute_tolerance", m_atol);
    pp.query("relative_tolerance", m_rtol);
    pp.query("power_iterations", m_power_iters);
    m_user_lambda_min = pp.query("lambda_min", m_lambda_min);
    m_user_lambda_max = pp.query("lambda_max", m_lambda_max);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (!m_user_lambda_min || m_lambda_min > RT(0)) &&
        (!m_user_lambda_max || m_lambda_max > m_lambda_min),
        a_pp_prefix + ": requires 0 < lambda_min < lambda_max");
    m_eigenvalues_estimated = m_user_lambda_min && m_user_lambda_max;

    m_is_defined = true;
}

template <class T>
void ChebyshevImpl<T>::Update (bool a_new_operator)
{
    m_ops->Update();
    if (a_new_operator && !(m_user_lambda_min && m_user_lambda_max)) {
        m_eigenvalues_estimated = false;
    }
}

template <class T>
typename ChebyshevImpl<T>::RT ChebyshevImpl<T>::PowerMethod (RT a_shift)
{
    // Pseudo-random start, so that no eigenvector (e.g. curl-free modes) is missed
    m_w1.setValPseudoRandom();
    m_w1.scale(RT(1.0) / m_w1.norm2());

    RT mu = RT(0);
    for (int iter = 0; iter < m_power_iters; iter++) {
        // w1 = (a_shift I - D^{-1} A) v, with ||v|| = 1
        m_ops->Matvec(m_w2, m_w1);
        if (a_shift > RT(0)) { m_x_old.Copy(m_w1); }
        m_ops->InitialGuess(m_w1, m_w2);
        if (a_shift > RT(0)) { m_w1.linComb(a_shift, m_x_old, RT(-1), m_w1); }
        mu = m_w1.norm2();
        if (mu < RT(1.0e-30)) { break; }
        m_w1.scale(RT(1.0) / mu);
    }
    return mu;
}

template <class T>
void ChebyshevImpl<T>::EstimateEigenvalues ()
{
    using namespace amrex;

    // 10% safety margin: Chebyshev diverges on eigenvalues above lambda_max
    if (!m_user_lambda_max) {
        m_lambda_max = RT(1.1) * PowerMethod(RT(0));
    }
    // The power method underestimates lambda_max - lambda_min, so lambda_min is
    // overestimated; eigenvalues below lambda_min converge, only more slowly.
    if (!m_user_lambda_min) {
        const RT spread = PowerMethod(m_lambda_max);
        m_lambda_min = std::max(m_lambda_max - spread, RT(1.0e-6) * m_lambda_max);
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_lambda_max > m_lambda_min,
        m_name + ": eigenvalue estimate failed, lambda_max <= lambda_min");

    if (m_verbose) {
        Print() << "  " << m_name << ": power method eigenvalue estimate:"
                << " lambda_min = " << m_lambda_min
                << ", lambda_max = " << m_lambda_max << "\n";
    }

    m_eigenvalues_estimated = true;
}

template <class T>
void ChebyshevImpl<T>::printParameters (const std::string& a_name) const
{
    using namespace amrex;
    Print() << a_name << " verbose:              " << (m_verbose?"true":"false") << "\n";
    Print() << a_name << " max iter:             " << m_max_iter << "\n";
    Print() << a_name << " lambda min:           " << m_lambda_min << "\n";
    Print() << a_name << " lambda max:           " << m_lambda_max << "\n";
    Print() << a_name << " power iterations:     " << m_power_iters << "\n";
    Print() << a_name << " absolute tolerance:   " << m_atol << "\n";
    Print() << a_name << " relative tolerance:   " << m_rtol << "\n";
}

template <class T>
void ChebyshevImpl<T>::Solve (T& a_x, const T& a_b)
{
    BL_PROFILE("ChebyshevImpl::Solve()");
    using namespace amrex;

    m_status = 0;
    m_num_iters = 0;
    m_res_norm = RT(0);

    if (!m_ops->IsDefined()) {
        a_x.Copy(a_b);
        return;
    }

    // Initial guess: x = D^{-1} b
    m_ops->InitialGuess(a_x, a_b);

    if (!m_ops->HasOffDiag()) {
        if (m_verbose) {
            const RT bnorm = a_b.norm2();
            const RT res = m_ops->ResidualNorm(a_x, a_b);
            Print() << "  " << m_name << " (diag): residual = " << res
                    << ", " << res / std::max(bnorm, RT(1.e-30))
                    << " (rel.)\n";
        }
        return;
    }

    // Work vectors and eigenvalue bounds are set up on the first solve
    if (!m_work_defined) {
        m_x_old.Define(a_b);
        m_w1.Define(a_b);
        m_w2.Define(a_b);
        m_work_defined = true;
    }
    if (!m_eigenvalues_estimated) {
        EstimateEigenvalues();
    }

    // Chebyshev parameters from eigenvalue bounds of D^{-1} A
    const RT theta = (m_lambda_max + m_lambda_min) / RT(2);
    const RT delta = (m_lambda_max - m_lambda_min) / RT(2);
    const RT sigma = delta / theta;  // spectral radius ratio
    const RT tau   = RT(2) / (m_lambda_max + m_lambda_min);  // = 1/theta

    // Save x_0 as x_{-1} for three-term recurrence
    m_x_old.Copy(a_x);

    // Compute initial residual norm
    const RT bnorm = a_b.norm2();
    const RT res0 = m_ops->ResidualNorm(a_x, a_b);

    if (m_verbose) {
        Print() << "  " << m_name << ": iter = 0, residual = " << res0
                << ", " << res0 / std::max(bnorm, RT(1.e-30))
                << " (rel.)\n";
    }

    RT omega = RT(1);

    // Chebyshev iteration loop
    for (int k = 0; k < m_max_iter; k++) {

        // r = b - A*x (stored in m_w1)
        m_ops->Matvec(m_w1, a_x);
        m_w1.scale(RT(-1));
        m_w1.increment(a_b, RT(1));

        // z = D^{-1} r (stored in m_w2)
        m_ops->InitialGuess(m_w2, m_w1);

        if (k == 0) {
            // First iteration: x_1 = x_0 + tau * z (Richardson step)
            omega = RT(1);
            a_x.increment(m_w2, tau);
        } else {
            // Three-term recurrence:
            // x_{k+1} = omega * (tau * z + x_k - x_{k-1}) + x_{k-1}
            const RT omega_new = (k == 1)
                ? RT(1) / (RT(1) - sigma * sigma / RT(2))
                : RT(1) / (RT(1) - sigma * sigma * omega / RT(4));

            // Save x_k in m_w1 (reuse, we're done with the residual)
            m_w1.Copy(a_x);

            // x = omega_new * x_k + (1 - omega_new) * x_{k-1} + omega_new * tau * z
            a_x.linComb(omega_new, m_w1, RT(1) - omega_new, m_x_old);
            a_x.increment(m_w2, omega_new * tau);

            // x_{k-1} = x_k for next iteration
            m_x_old.Copy(m_w1);
            omega = omega_new;
        }

        // Check convergence
        const RT res = m_ops->ResidualNorm(a_x, a_b);
        m_num_iters = k + 1;
        m_res_norm = res;

        if (m_verbose) {
            Print() << "  " << m_name << ": iter = " << (k+1)
                    << ", residual = " << res
                    << ", " << res / std::max(bnorm, RT(1.e-30))
                    << " (rel.)\n";
        }

        if (res < m_atol || res < m_rtol * res0) {
            if (m_verbose) {
                Print() << "  " << m_name << ": converged at iter "
                        << (k+1) << "\n";
            }
            return;
        }
    }

    m_status = 1;
}

#endif
