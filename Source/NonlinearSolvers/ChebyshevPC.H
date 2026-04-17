/* Copyright 2026 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef CHEBYSHEV_PC_H_
#define CHEBYSHEV_PC_H_

#include "Fields.H"
#include "Preconditioner.H"
#include "ChebyshevImpl.H"

#include "MatrixSolverOps.H"
#include "FieldSolver/ImplicitSolvers/MultiFabMatrixSolverOps.H"
#include "FieldSolver/ImplicitSolvers/SparseMatrixSolverOps.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include <memory>

/**
 * \brief Chebyshev polynomial preconditioner.
 *
 *  Approximately solves A x = b by running Chebyshev semi-iterative
 *  steps, delegating the iteration logic to ChebyshevImpl<T>.
 *
 *  Two operating modes, selected automatically (same as JacobiPC):
 *
 *  1) Stencil path (semi-implicit, GetThetaForPC()==0):
 *     A = I + M where M is the mass matrix stored as MultiFab stencil
 *     coefficients.
 *
 *  2) CSR path (theta-implicit, GetThetaForPC()>0):
 *     A = I + curl(alpha*curl[]) + M assembled as a sparse CSR matrix
 *     by SparseJacobianMatrix.
 */
template <class T, class Ops>
class ChebyshevPC : public Preconditioner<T,Ops>
{
public:

    using RT = typename T::value_type;

    ChebyshevPC () = default;

    ~ChebyshevPC () override = default;

    ChebyshevPC (const ChebyshevPC&) = delete;
    ChebyshevPC& operator= (const ChebyshevPC&) = delete;
    ChebyshevPC (ChebyshevPC&&) noexcept = delete;
    ChebyshevPC& operator= (ChebyshevPC&&) noexcept = delete;

    void Define (const T&, Ops* const) override;

    void Update (const T& a_U) override;

    void Apply (T&, const T&) override;

    void printParameters () const override;

    [[nodiscard]] inline bool IsDefined () const override { return m_is_defined; }

protected:

    bool m_is_defined = false;

    Ops* m_ops = nullptr;

    int m_num_amr_levels = 0;

    std::unique_ptr<MatrixSolverOps<T>> m_matrix_ops;

    ChebyshevImpl<T> m_solver;

private:

};

template <class T, class Ops>
void ChebyshevPC<T,Ops>::printParameters () const
{
    auto pc_name = amrex::getEnumNameString(PreconditionerType::pc_chebyshev);
    m_solver.printParameters(pc_name);
    bool use_csr = (m_ops != nullptr && m_ops->GetThetaForPC() > amrex::Real(0.0));
    amrex::Print() << pc_name << " use CSR path:         " << (use_csr?"true":"false") << "\n";
}

template <class T, class Ops>
void ChebyshevPC<T,Ops>::Define (const T& a_U,
                                  Ops* const a_ops)
{
    BL_PROFILE("ChebyshevPC::Define()");
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !IsDefined(),
        "ChebyshevPC::Define() called on defined object");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (a_ops != nullptr),
        "ChebyshevPC::Define(): a_ops is nullptr");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_U.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "ChebyshevPC::Define() must be called with Efield_fp type");

    m_ops = a_ops;
    m_num_amr_levels = m_ops->numAMRLevels();

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
                    getEnumNameString(PreconditionerType::pc_chebyshev));

    m_is_defined = true;
}

template <class T, class Ops>
void ChebyshevPC<T,Ops>::Update (const T& a_U)
{
    BL_PROFILE("ChebyshevPC::Update()");
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "ChebyshevPC::Update() called on undefined object");

    m_solver.Update(a_U);
}

template <class T, class Ops>
void ChebyshevPC<T,Ops>::Apply (T& a_x, const T& a_b)
{
    BL_PROFILE("ChebyshevPC::Apply()");
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "ChebyshevPC::Apply() called on undefined object");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_x.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "ChebyshevPC::Apply() - a_x must be Efield_fp type");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_b.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "ChebyshevPC::Apply() - a_b must be Efield_fp type");

    m_solver.Solve(a_x, a_b);
}

#endif
