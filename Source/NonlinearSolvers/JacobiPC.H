/* Copyright 2024-2025 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef JACOBI_PC_H_
#define JACOBI_PC_H_

#include "Fields.H"
#include "Utils/WarpXConst.H"
#include "Preconditioner.H"

#include "MatrixSolverOps.H"
#include "MultiFabMatrixSolverOps.H"
#include "FieldSolver/ImplicitSolvers/SparseMatrixSolverOps.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include <memory>

/**
 * \brief Point-Jacobi Preconditioner
 *
 *  Approximately solves A x = b via damped Jacobi iterations.
 *
 *  Two operating modes, selected automatically:
 *
 *  1) Stencil path (semi-implicit, GetThetaForPC()==0):
 *     A = I + M where M is the mass matrix stored as MultiFab stencil
 *     coefficients. When diagonal-only (mass_matrices_pc_width=0), the
 *     solve is exact in one step.
 *
 *  2) CSR path (theta-implicit, GetThetaForPC()>0):
 *     A = I + curl(alpha*curl[]) + M assembled as a sparse CSR matrix
 *     by SparseJacobianMatrix. Operations (matvec, sweep, residual)
 *     are performed on flat device vectors in local-extended DOF space.
 *
 *  Adaptive omega (enabled by default): the relaxation parameter omega
 *  starts at 1.0 and is automatically reduced (by a factor of 0.9) when
 *  the Jacobi residual increases. When the user explicitly sets
 *  pc_jacobi.omega, adaptive behavior is disabled.
 */

template <class T, class Ops>
class JacobiPC : public Preconditioner<T,Ops>
{
    public:

        using RT = typename T::value_type;

        JacobiPC () = default;

        ~JacobiPC () override = default;

        JacobiPC(const JacobiPC&) = delete;
        JacobiPC& operator=(const JacobiPC&) = delete;
        JacobiPC(JacobiPC&&) noexcept = delete;
        JacobiPC& operator=(JacobiPC&&) noexcept = delete;

        void Define (const T&, Ops* const) override;

        void Update (const T& a_U) override;

        /**
         * \brief Solve A x = b via damped Jacobi iteration.
         *        Dispatches to stencil or CSR path via MatrixSolverOps.
         */
        void Apply (T&, const T&) override;

        void printParameters() const override;

        [[nodiscard]] inline bool IsDefined () const override { return m_is_defined; }

    protected:

        bool m_is_defined = false;

        bool m_verbose = false;
        int m_max_iter = 200;
        RT m_omega = RT(1.0);
        bool m_adaptive_omega = true;

        RT m_atol = 1.0e-16;
        RT m_rtol = 1.0e-4;

        Ops* m_ops = nullptr;

        int m_num_amr_levels = 0;

        std::unique_ptr<MatrixSolverOps<T>> m_matrix_ops;

        void readParameters();

    private:

};

template <class T, class Ops>
void JacobiPC<T,Ops>::printParameters() const
{
    using namespace amrex;
    auto pc_name = getEnumNameString(PreconditionerType::pc_jacobi);
    Print() << pc_name << " verbose:              " << (m_verbose?"true":"false") << "\n";
    Print() << pc_name << " max iter:             " << m_max_iter << "\n";
    Print() << pc_name << " omega:                " << m_omega << "\n";
    Print() << pc_name << " adaptive omega:       " << (m_adaptive_omega?"true":"false") << "\n";
    Print() << pc_name << " absolute tolerance:   " << m_atol << "\n";
    Print() << pc_name << " relative tolerance:   " << m_rtol << "\n";
    bool use_csr = (m_ops != nullptr && m_ops->GetThetaForPC() > amrex::Real(0.0));
    Print() << pc_name << " use CSR path:         " << (use_csr?"true":"false") << "\n";
}

template <class T, class Ops>
void JacobiPC<T,Ops>::readParameters()
{
    const amrex::ParmParse pp(amrex::getEnumNameString(PreconditionerType::pc_jacobi));
    pp.query("verbose", m_verbose);
    pp.query("max_iter", m_max_iter);
    if (pp.query("omega", m_omega)) {
        m_adaptive_omega = false;
    }
    pp.query("adaptive_omega", m_adaptive_omega);
    pp.query("absolute_tolerance",  m_atol);
    pp.query("relative_tolerance",  m_rtol);
}

template <class T, class Ops>
void JacobiPC<T,Ops>::Define ( const T& a_U,
                               Ops* const a_ops )
{
    BL_PROFILE("JacobiPC::Define()");
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !IsDefined(),
        "JacobiPC::Define() called on defined object" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (a_ops != nullptr),
        "JacobiPC::Define(): a_ops is nullptr" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_U.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "JacobiPC::Define() must be called with Efield_fp type");

    m_ops = a_ops;
    m_num_amr_levels = m_ops->numAMRLevels();

    readParameters();

    // Create the appropriate MatrixSolverOps based on solver type
    if (m_ops->GetThetaForPC() > Real(0.0)) {
        auto sp_ops = std::make_unique<SparseMatrixSolverOps<T,Ops>>();
        sp_ops->Define(m_ops);
        m_matrix_ops = std::move(sp_ops);
    } else {
        auto mf_ops = std::make_unique<MultiFabMatrixSolverOps<T,Ops>>();
        mf_ops->Define(m_ops);
        m_matrix_ops = std::move(mf_ops);
    }

    m_is_defined = true;
}

template <class T, class Ops>
void JacobiPC<T,Ops>::Update (const T& a_U)
{
    BL_PROFILE("JacobiPC::Update()");
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "JacobiPC::Update() called on undefined object" );

    m_matrix_ops->Update(a_U);

    if (m_adaptive_omega) {
        m_omega = m_matrix_ops->EstimateOmega();
    }

    if (m_verbose) {
        bool use_csr = (m_ops->GetThetaForPC() > Real(0.0));
        amrex::Print() << "Updating " << amrex::getEnumNameString(PreconditionerType::pc_jacobi)
                       << ": theta*dt = " << this->m_dt
                       << ", path = " << (use_csr ? "CSR" : "stencil")
                       << ", omega = " << m_omega << "\n";
    }
}

template <class T, class Ops>
void JacobiPC<T,Ops>::Apply (T& a_x, const T& a_b)
{
    BL_PROFILE("JacobiPC::Apply()");
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "JacobiPC::Apply() called on undefined object" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_x.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "JacobiPC::Apply() - a_x must be Efield_fp type");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_b.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "JacobiPC::Apply() - a_b must be Efield_fp type");

    if (!m_matrix_ops->IsDefined()) {
        a_x.Copy(a_b);
        return;
    }

    const int max_omega_reductions = m_adaptive_omega ? 10 : 0;
    RT omega = m_omega;

    for (int adapt = 0; adapt <= max_omega_reductions; adapt++) {

        // === Initial guess ===
        m_matrix_ops->InitialGuess(a_x, a_b);

        if (!m_matrix_ops->HasOffDiag()) {
            if (m_verbose) {
                RT bnorm = a_b.norm2();
                RT res = m_matrix_ops->ResidualNorm(a_x, a_b);
                amrex::Print() << "  pc_jacobi (diag): residual = " << res
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
            amrex::Print() << "  pc_jacobi: iter = 0, residual = " << res0
                           << ", " << res0 / std::max(bnorm, RT(1.e-30))
                           << " (rel.), omega = " << omega << "\n";
        }

        // === Jacobi iteration loop ===
        bool diverged = false;

        for (int iter = 0; iter < m_max_iter; iter++) {

            m_matrix_ops->JacobiSweep(a_x, a_b, omega);

            RT res_norm = m_matrix_ops->ResidualNorm(a_x, a_b);

            if (m_verbose) {
                amrex::Print() << "  pc_jacobi: iter = " << (iter+1)
                               << ", residual = " << res_norm
                               << ", " << res_norm / std::max(bnorm, RT(1.e-30))
                               << " (rel.)\n";
            }

            // --- Convergence / omega adaptation ---
            if (m_adaptive_omega && res_norm > res_prev) {
                omega *= RT(0.9);
                if (m_verbose) {
                    amrex::Print() << "  pc_jacobi: residual increased, "
                                   << "reducing omega to " << omega << "\n";
                }
                diverged = true;
                break;
            }

            if (res_norm < m_atol || res_norm < m_rtol * res0) {
                if (m_verbose) {
                    amrex::Print() << "  pc_jacobi: converged at iter "
                                   << (iter+1) << "\n";
                }
                m_omega = omega;
                return;
            }
            res_prev = res_norm;
        }

        if (!diverged) { break; }
    }

    m_omega = omega;
}

#endif
