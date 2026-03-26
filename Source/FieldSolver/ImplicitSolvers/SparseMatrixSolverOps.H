/* Copyright 2025-2026 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef SPARSE_MATRIX_SOLVER_OPS_H_
#define SPARSE_MATRIX_SOLVER_OPS_H_

#include "FieldSolver/ImplicitSolvers/SparseJacobianMatrix.H"
#include "FieldSolver/ImplicitSolvers/SparseMatrixUtils.H"
#include "NonlinearSolvers/MatrixSolverOps.H"

#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_Vector.H>

/**
 * \brief CSR sparse matrix ops for A = I + curl(alpha*curl[]) + M (theta-implicit).
 */
template <class T, class Ops>
class SparseMatrixSolverOps : public MatrixSolverOps<T>
{
public:

    using RT = typename T::value_type;

    SparseMatrixSolverOps () = default;
    ~SparseMatrixSolverOps () override = default;

    /** \brief Set the Ops pointer and cache geometry. */
    void Define (Ops* a_ops);

    void Update (const T& a_U) override;
    void InitialGuess (T& a_x, const T& a_b) override;
    void JacobiSweep (T& a_x, const T& a_b, RT a_omega) override;
    [[nodiscard]] RT ResidualNorm (const T& a_x, const T& a_b) override;
    void Matvec (T& a_y, const T& a_x) override;
    [[nodiscard]] RT EstimateOmega () const override;
    [[nodiscard]] bool HasOffDiag () const override { return true; }
    [[nodiscard]] bool IsDefined () const override { return m_is_defined; }

private:

    /** \brief Copy b into flat device vector m_b_flat. */
    void linearizeRHS (const T& a_b);
    /** \brief FillBoundary + linearize owned+ghost into m_x_ext. */
    void buildExtended (const T& a_x);

    bool m_is_defined = false;

    Ops* m_ops = nullptr;
    int m_num_amr_levels = 0;

    amrex::Vector<amrex::Geometry> m_geom_vec;

    amrex::Gpu::DeviceVector<RT> m_x_flat;
    amrex::Gpu::DeviceVector<RT> m_y_flat;
    amrex::Gpu::DeviceVector<RT> m_b_flat;
    amrex::Gpu::DeviceVector<RT> m_x_ext;
};

template <class T, class Ops>
void SparseMatrixSolverOps<T,Ops>::Define (Ops* a_ops)
{
    m_ops = a_ops;
    m_num_amr_levels = m_ops->numAMRLevels();
    m_geom_vec.resize(m_num_amr_levels);
    for (int lev = 0; lev < m_num_amr_levels; lev++) {
        m_geom_vec[lev] = m_ops->GetGeometry(lev);
    }
    m_is_defined = true;
}

template <class T, class Ops>
void SparseMatrixSolverOps<T,Ops>::Update (const T& a_U)
{
    m_ops->AssemblePCMatrix(a_U);
    const auto* sp = m_ops->GetSparseJacobian();
    if (!sp->IsRemapped()) {
        auto* sp_nc = const_cast<SparseJacobianMatrix*>(sp);
        sp_nc->RemapColumns(a_U.getDOFsObject().get());
    }
}

template <class T, class Ops>
void SparseMatrixSolverOps<T,Ops>::linearizeRHS (const T& a_b)
{
    SparseMatrixUtils::CopyToLocalDOFVector(a_b, m_b_flat);
}

template <class T, class Ops>
void SparseMatrixSolverOps<T,Ops>::buildExtended (const T& a_x)
{
    const auto* sp = m_ops->GetSparseJacobian();
    SparseMatrixUtils::BuildExtendedDOFVector(a_x, m_x_ext, *sp,
                                               m_geom_vec, m_num_amr_levels);
}

template <class T, class Ops>
void SparseMatrixSolverOps<T,Ops>::InitialGuess (T& a_x, const T& a_b)
{
    linearizeRHS(a_b);
    const auto* sp = m_ops->GetSparseJacobian();
    m_x_flat.resize(sp->nLocalRows());
    sp->MatInitialGuess(m_x_flat.data(), m_b_flat.data());
    SparseMatrixUtils::CopyFromLocalDOFVector(a_x, m_x_flat);
}

template <class T, class Ops>
void SparseMatrixSolverOps<T,Ops>::JacobiSweep (T& a_x, const T& a_b, RT a_omega)
{
    using namespace amrex;
    amrex::ignore_unused(a_b);

    const auto* sp = m_ops->GetSparseJacobian();
    buildExtended(a_x);
    sp->MatJacobiSweep(m_x_flat.data(), m_x_ext.data(),
                       m_b_flat.data(), a_omega);
    SparseMatrixUtils::CopyFromLocalDOFVector(a_x, m_x_flat);
}

template <class T, class Ops>
auto SparseMatrixSolverOps<T,Ops>::ResidualNorm (const T& a_x, const T& a_b) -> RT
{
    using namespace amrex;
    amrex::ignore_unused(a_b);

    const auto* sp = m_ops->GetSparseJacobian();
    buildExtended(a_x);
    RT res_sq = sp->MatResidualNorm(m_x_ext.data(), m_b_flat.data());
    ParallelAllReduce::Sum(res_sq, ParallelContext::CommunicatorSub());
    return std::sqrt(res_sq);
}

template <class T, class Ops>
void SparseMatrixSolverOps<T,Ops>::Matvec (T& a_y, const T& a_x)
{
    const auto* sp = m_ops->GetSparseJacobian();
    buildExtended(a_x);
    m_y_flat.resize(sp->nLocalRows());
    sp->MatVecMult(m_y_flat.data(), m_x_ext.data());
    SparseMatrixUtils::CopyFromLocalDOFVector(a_y, m_y_flat);
}

template <class T, class Ops>
auto SparseMatrixSolverOps<T,Ops>::EstimateOmega () const -> RT
{
    const auto* sp = m_ops->GetSparseJacobian();
    return static_cast<RT>(sp->EstimateOmega());
}

#endif
