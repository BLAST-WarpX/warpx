/* Copyright 2026 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef CHEBYSHEV_IMPL_H_
#define CHEBYSHEV_IMPL_H_

#include "MatrixSolverOps.H"

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
 *  lambda_min is set to 1.0 (from the identity in the Jacobian J = I + ...).
 *  lambda_max is estimated by the power method on D^{-1} A.
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

    /** \brief Update matrix ops and estimate eigenvalues if needed. */
    void Update (const T& a_U);

    /** \brief Solve A x = b using Chebyshev iteration. */
    void Solve (T& a_x, const T& a_b);

    void printParameters (const std::string& a_name) const;

    [[nodiscard]] bool IsDefined () const { return m_is_defined; }
    [[nodiscard]] bool isVerbose () const { return m_verbose; }
    [[nodiscard]] int getNumIters () const { return m_num_iters; }
    [[nodiscard]] RT getResidualNorm () const { return m_res_norm; }
    [[nodiscard]] int getStatus () const { return m_status; }

    void setVerbose (bool v) { m_verbose = v; }
    void setMaxIters (int n) { m_max_iter = n; }
    void setTolerances (RT rtol, RT atol) { m_rtol = rtol; m_atol = atol; }

private:

    /** \brief Estimate the largest eigenvalue of D^{-1} A via power method. */
    void EstimateLargestEigenvalue (const T& a_ref);

    bool m_is_defined = false;
    std::string m_name = "chebyshev";

    bool m_verbose = false;
    int m_max_iter = 200;
    RT m_atol = RT(1.0e-16);
    RT m_rtol = RT(1.0e-4);

    RT m_lambda_min = RT(1.0);  /**< lower eigenvalue bound (from identity) */
    RT m_lambda_max = RT(0.0);  /**< upper eigenvalue bound (estimated) */
    int m_power_iters = 20;     /**< power method iterations for lambda_max */
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

// ============================================================================
//  Implementation
// ============================================================================

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
    pp.query("lambda_min", m_lambda_min);
    if (pp.query("lambda_max", m_lambda_max)) {
        if (m_lambda_max > RT(0)) {
            m_eigenvalues_estimated = true;
        }
    }

    m_is_defined = true;
}

template <class T>
void ChebyshevImpl<T>::Update (const T& a_U)
{
    m_ops->Update(a_U);

    if (!m_work_defined) {
        m_x_old.Define(a_U);
        m_w1.Define(a_U);
        m_w2.Define(a_U);
        m_work_defined = true;
    }

    if (!m_eigenvalues_estimated) {
        EstimateLargestEigenvalue(a_U);
    }
}

template <class T>
void ChebyshevImpl<T>::EstimateLargestEigenvalue (const T& a_ref)
{
    using namespace amrex;

    // Power method: repeatedly compute v = D^{-1} A v, normalize
    // Uses m_w1, m_w2 as work vectors

    // Initialize v for power iteration (use m_w1)
    m_w1.Copy(a_ref);
    RT vnorm = m_w1.norm2();
    if (vnorm < RT(1.0e-30)) {
        // a_ref is zero (e.g. initial E=0); seed with ones instead
        m_w1.setVal(RT(1.0));
        vnorm = m_w1.norm2();
    }
    m_w1.scale(RT(1.0) / vnorm);

    RT lambda_est = RT(0);
    for (int iter = 0; iter < m_power_iters; iter++) {
        // w2 = A * v
        m_ops->Matvec(m_w2, m_w1);
        // w1 = D^{-1} * w2 = D^{-1} A v
        m_ops->InitialGuess(m_w1, m_w2);
        // lambda = ||w1|| (since ||v|| was 1)
        lambda_est = m_w1.norm2();
        if (lambda_est < RT(1.0e-30)) { break; }
        // Normalize
        m_w1.scale(RT(1.0) / lambda_est);
    }

    // Add a small safety margin (10%) to the estimate
    m_lambda_max = RT(1.1) * std::max(lambda_est, m_lambda_min + RT(0.1));

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

    // === Initial guess: x = D^{-1} b ===
    m_ops->InitialGuess(a_x, a_b);

    if (!m_ops->HasOffDiag()) {
        if (m_verbose) {
            RT bnorm = a_b.norm2();
            RT res = m_ops->ResidualNorm(a_x, a_b);
            Print() << "  " << m_name << " (diag): residual = " << res
                    << ", " << res / std::max(bnorm, RT(1.e-30))
                    << " (rel.)\n";
        }
        return;
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

    // === Chebyshev iteration loop ===
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
            RT omega_new = RT(1) / (RT(1) - sigma * sigma * omega / RT(4));

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
        RT res = m_ops->ResidualNorm(a_x, a_b);
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
