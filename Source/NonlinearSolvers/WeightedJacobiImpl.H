/* Copyright 2025-2026 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WEIGHTED_JACOBI_IMPL_H_
#define WEIGHTED_JACOBI_IMPL_H_

#include "MatrixSolverOps.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <string>

/**
 * \brief Weighted (damped) Jacobi iteration solver.
 *
 *  Solves A x = b via the iteration:
 *      x^{k+1} = x^k + omega * D^{-1} (b - A x^k)
 *  where D = diag(A) and omega is the damping parameter.
 *
 *  All matrix operations (matvec, diagonal extraction, sweep) are
 *  performed through a MatrixSolverOps<T> implementation, which may be
 *  stencil-based, CSR-based, or matrix-free.
 *
 *  Adaptive omega (enabled by default): omega starts at its initial
 *  value (1.0 or Gershgorin estimate) and is automatically reduced
 *  (by a factor of 0.9) when the residual increases. When the user
 *  explicitly sets omega, adaptive behavior is disabled.
 *
 *  This class is used by both:
 *  - JacobiPC (as a preconditioner)
 *  - JacobiSolver (as a standalone linear solver)
 */
template <class T>
class WeightedJacobiImpl
{
public:

    using RT = typename T::value_type;

    WeightedJacobiImpl () = default;
    ~WeightedJacobiImpl () = default;

    WeightedJacobiImpl (const WeightedJacobiImpl&) = delete;
    WeightedJacobiImpl& operator= (const WeightedJacobiImpl&) = delete;
    WeightedJacobiImpl (WeightedJacobiImpl&&) noexcept = delete;
    WeightedJacobiImpl& operator= (WeightedJacobiImpl&&) noexcept = delete;

    /**
     * \brief Define the solver with a MatrixSolverOps and ParmParse prefix.
     *
     * \param[in] a_ops       pointer to a MatrixSolverOps (caller retains ownership)
     * \param[in] a_pp_prefix ParmParse prefix for reading solver parameters
     */
    void Define (MatrixSolverOps<T>* a_ops, const std::string& a_pp_prefix);

    /** \brief Update matrix ops and re-estimate omega if adaptive. */
    void Update (const T& a_U);

    /** \brief Solve A x = b using weighted Jacobi iteration. */
    void Solve (T& a_x, const T& a_b);

    void printParameters (const std::string& a_name) const;

    [[nodiscard]] bool IsDefined () const { return m_is_defined; }
    [[nodiscard]] bool isVerbose () const { return m_verbose; }
    [[nodiscard]] RT getOmega () const { return m_omega; }
    [[nodiscard]] int getNumIters () const { return m_num_iters; }
    [[nodiscard]] RT getResidualNorm () const { return m_res_norm; }
    [[nodiscard]] int getStatus () const { return m_status; }

    void setVerbose (bool v) { m_verbose = v; }
    void setMaxIters (int n) { m_max_iter = n; }
    void setTolerances (RT rtol, RT atol) { m_rtol = rtol; m_atol = atol; }

private:

    bool m_is_defined = false;
    std::string m_name = "weighted_jacobi";

    bool m_verbose = false;
    int m_max_iter = 200;
    RT m_omega = RT(1.0);
    bool m_adaptive_omega = true;
    RT m_atol = RT(1.0e-16);
    RT m_rtol = RT(1.0e-4);

    int m_num_iters = 0;
    RT m_res_norm = RT(0);
    int m_status = 0;  /**< 0 = converged/exact, 1 = max iterations reached */

    MatrixSolverOps<T>* m_matrix_ops = nullptr;
};

// ============================================================================
//  Implementation
// ============================================================================

template <class T>
void WeightedJacobiImpl<T>::Define (MatrixSolverOps<T>* a_ops,
                                 const std::string& a_pp_prefix)
{
    m_matrix_ops = a_ops;
    m_name = a_pp_prefix;

    const amrex::ParmParse pp(a_pp_prefix);
    int verbose_int = 0;
    if (pp.query("verbose_int", verbose_int)) {
        m_verbose = (verbose_int > 0);
    }
    pp.query("max_iter", m_max_iter);
    if (pp.query("omega", m_omega)) {
        m_adaptive_omega = false;
    }
    pp.query("adaptive_omega", m_adaptive_omega);
    pp.query("absolute_tolerance", m_atol);
    pp.query("relative_tolerance", m_rtol);

    m_is_defined = true;
}

template <class T>
void WeightedJacobiImpl<T>::Update (const T& a_U)
{
    m_matrix_ops->Update(a_U);

    if (m_adaptive_omega) {
        m_omega = m_matrix_ops->EstimateOmega();
    }
}

template <class T>
void WeightedJacobiImpl<T>::printParameters (const std::string& a_name) const
{
    using namespace amrex;
    Print() << a_name << " verbose:              " << (m_verbose?"true":"false") << "\n";
    Print() << a_name << " max iter:             " << m_max_iter << "\n";
    Print() << a_name << " omega:                " << m_omega << "\n";
    Print() << a_name << " adaptive omega:       " << (m_adaptive_omega?"true":"false") << "\n";
    Print() << a_name << " absolute tolerance:   " << m_atol << "\n";
    Print() << a_name << " relative tolerance:   " << m_rtol << "\n";
}

template <class T>
void WeightedJacobiImpl<T>::Solve (T& a_x, const T& a_b)
{
    BL_PROFILE("WeightedJacobiImpl::Solve()");
    using namespace amrex;

    m_status = 0;
    m_num_iters = 0;
    m_res_norm = RT(0);

    if (!m_matrix_ops->IsDefined()) {
        a_x.Copy(a_b);
        return;
    }

    const int max_omega_reductions = m_adaptive_omega ? 10 : 0;
    RT omega = m_omega;

    for (int adapt = 0; adapt <= max_omega_reductions; adapt++) {

        // === Initial guess: x = D^{-1} b ===
        m_matrix_ops->InitialGuess(a_x, a_b);

        if (!m_matrix_ops->HasOffDiag()) {
            if (m_verbose) {
                RT bnorm = a_b.norm2();
                RT res = m_matrix_ops->ResidualNorm(a_x, a_b);
                Print() << "  " << m_name << " (diag): residual = " << res
                        << ", " << res / std::max(bnorm, RT(1.e-30))
                        << " (rel.)\n";
            }
            return;
        }

        // === Compute initial residual ===
        RT bnorm = a_b.norm2();
        RT res0 = m_matrix_ops->ResidualNorm(a_x, a_b);
        RT res_prev = res0;

        if (m_verbose) {
            Print() << "  " << m_name << ": iter = 0, residual = " << res0
                    << ", " << res0 / std::max(bnorm, RT(1.e-30))
                    << " (rel.), omega = " << omega << "\n";
        }

        // === Jacobi iteration loop ===
        bool diverged = false;

        for (int iter = 0; iter < m_max_iter; iter++) {

            m_matrix_ops->JacobiSweep(a_x, a_b, omega);

            RT res_norm = m_matrix_ops->ResidualNorm(a_x, a_b);

            m_num_iters = iter + 1;
            m_res_norm = res_norm;

            if (m_verbose) {
                Print() << "  " << m_name << ": iter = " << (iter+1)
                        << ", residual = " << res_norm
                        << ", " << res_norm / std::max(bnorm, RT(1.e-30))
                        << " (rel.)\n";
            }

            // --- Convergence / omega adaptation ---
            if (m_adaptive_omega && res_norm > res_prev) {
                omega *= RT(0.9);
                if (m_verbose) {
                    Print() << "  " << m_name << ": residual increased, "
                            << "reducing omega to " << omega << "\n";
                }
                diverged = true;
                break;
            }

            if (res_norm < m_atol || res_norm < m_rtol * res0) {
                if (m_verbose) {
                    Print() << "  " << m_name << ": converged at iter "
                            << (iter+1) << "\n";
                }
                m_omega = omega;
                return;
            }
            res_prev = res_norm;
        }

        if (!diverged) {
            m_status = 1;
            break;
        }
    }

    m_omega = omega;
}

#endif
