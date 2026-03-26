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
#include "WeightedJacobiImpl.H"

#include "MatrixSolverOps.H"
#include "FieldSolver/ImplicitSolvers/MultiFabMatrixSolverOps.H"
#include "FieldSolver/ImplicitSolvers/SparseMatrixSolverOps.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include <memory>

/**
 * \brief Point-Jacobi Preconditioner
 *
 *  Approximately solves A x = b via damped Jacobi iterations,
 *  delegating the iteration logic to WeightedJacobiImpl<T>.
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

        Ops* m_ops = nullptr;

        int m_num_amr_levels = 0;

        std::unique_ptr<MatrixSolverOps<T>> m_matrix_ops;

        WeightedJacobiImpl<T> m_solver;

    private:

};

template <class T, class Ops>
void JacobiPC<T,Ops>::printParameters() const
{
    using namespace amrex;
    auto pc_name = getEnumNameString(PreconditionerType::pc_jacobi);
    m_solver.printParameters(pc_name);
    bool use_csr = (m_ops != nullptr && m_ops->GetThetaForPC() > amrex::Real(0.0));
    Print() << pc_name << " use CSR path:         " << (use_csr?"true":"false") << "\n";
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

    m_solver.Define(m_matrix_ops.get(),
                    getEnumNameString(PreconditionerType::pc_jacobi));

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

    m_solver.Update(a_U);

    if (m_solver.isVerbose()) {
        bool use_csr = (m_ops->GetThetaForPC() > Real(0.0));
        amrex::Print() << "Updating " << amrex::getEnumNameString(PreconditionerType::pc_jacobi)
                       << ": theta*dt = " << this->m_dt
                       << ", path = " << (use_csr ? "CSR" : "stencil")
                       << ", omega = " << m_solver.getOmega() << "\n";
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

    m_solver.Solve(a_x, a_b);
}

#endif
