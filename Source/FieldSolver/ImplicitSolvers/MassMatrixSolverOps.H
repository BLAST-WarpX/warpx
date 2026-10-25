/* Copyright 2025-2026 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef MASS_MATRIX_SOLVER_OPS_H_
#define MASS_MATRIX_SOLVER_OPS_H_

#include "NonlinearSolvers/MatrixSolverOps.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_Reduce.H>
#include <AMReX_Vector.H>

/**
 * \brief Matrix ops for A = I + M, with M the preconditioner mass matrices.
 *
 *  Storage of M:
 *  - M has only the diagonal blocks M_xx, M_yy and M_zz (no coupling between E components).
 *  - Each block is one MultiFab, on the grid of its E component.
 *  - At cell (i,j,k), the MultiFab components are the entries of row (i,j,k) of the block,
 *    one per neighbor (i+di, j+dj, k+dk).
 *  - Each offset di, dj, dk runs from -w to w, where 2*w+1 = GetMassMatricesPCnComp() in that
 *    direction (w = 0 in the directions above AMREX_SPACEDIM).
 *  - The components are ordered with di varying fastest, then dj, then dk.
 *  - The diagonal entry, offset (0,0,0), is component (nComp-1)/2.
 *
 *  Example: 2D with w = 1 in both directions, 9 components.
 *  ```
 *  component   0   1   2   3   4   5   6   7   8
 *  di         -1   0   1  -1   0   1  -1   0   1
 *  dj         -1  -1  -1   0   0   0   1   1   1
 *
 *  The same components, placed at their neighbors of cell (i,j):
 *
 *            i-1    i    i+1
 *          +-----+-----+-----+
 *  j+1     |  6  |  7  |  8  |
 *          +-----+-----+-----+
 *  j       |  3  |  4  |  5  |
 *          +-----+-----+-----+
 *  j-1     |  0  |  1  |  2  |
 *          +-----+-----+-----+
 *  ```
 *  Component 4 is the diagonal.
 *
 *  Matrix-vector product, with (di_n, dj_n, dk_n) the offset of component n:
 *      (A x)(i,j,k) = x(i,j,k) + sum_n M(i,j,k,n) * x(i+di_n, j+dj_n, k+dk_n)
 */
template <class T, class Ops>
class MassMatrixSolverOps : public MatrixSolverOps<T>
{
public:

    using RT = typename T::value_type;

    MassMatrixSolverOps () = default;
    ~MassMatrixSolverOps () override = default;

    MassMatrixSolverOps (const MassMatrixSolverOps&) = delete;
    MassMatrixSolverOps& operator= (const MassMatrixSolverOps&) = delete;
    MassMatrixSolverOps (MassMatrixSolverOps&&) noexcept = delete;
    MassMatrixSolverOps& operator= (MassMatrixSolverOps&&) noexcept = delete;

    /** \brief Set the Ops pointer and read mass matrix coefficients. */
    void Define (Ops* a_ops);

    void Update () override;
    void InitialGuess (T& a_x, const T& a_b) override;
    void JacobiSweep (T& a_x, const T& a_b, RT a_omega) override;
    [[nodiscard]] RT ResidualNorm (const T& a_x, const T& a_b) override;
    [[nodiscard]] RT EstimateOmega () const override;
    [[nodiscard]] bool HasOffDiag () const override { return m_has_offdiag; }
    [[nodiscard]] bool IsDefined () const override { return m_is_defined; }

    /** \brief Compute ||b - A*x||_2 on owned cells (MPI-reduced). */
    RT computeResidualNormInternal (const T& a_x, const T& a_b) const;

private:

    /** \brief Copy x to ghost-padded MultiFab and call FillBoundary. */
    void copyToGhostAndFill (
        const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>& a_x_mfarrvec);

    bool m_is_defined = false;

    Ops* m_ops = nullptr;
    int m_num_amr_levels = 0;

    const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>* m_bcoefs = nullptr;

    bool m_has_offdiag = false;
    bool m_work_defined = false;

    amrex::Vector<amrex::Array<amrex::MultiFab,3>> m_work;
    amrex::Vector<amrex::Array<amrex::MultiFab,3>> m_x_ghost;

    int m_stencil_width = 0;
};

template <class T, class Ops>
void MassMatrixSolverOps<T,Ops>::Define (Ops* a_ops)
{
    m_ops = a_ops;
    m_num_amr_levels = m_ops->numAMRLevels();
    m_bcoefs = m_ops->GetMassMatricesCoeff();
    m_is_defined = (m_bcoefs != nullptr);
}

template <class T, class Ops>
void MassMatrixSolverOps<T,Ops>::Update ()
{
    using namespace amrex;

    if (m_bcoefs == nullptr || m_work_defined) { return; }

    m_work.resize(m_num_amr_levels);
    m_x_ghost.resize(m_num_amr_levels);

    m_stencil_width = 0;
    for (int n = 0; n < m_num_amr_levels; n++) {
        for (int dim = 0; dim < 3; dim++) {
            if ((*m_bcoefs)[n][dim]->nComp() > 1) {
                m_has_offdiag = true;
                for (int d = 0; d < AMREX_SPACEDIM; d++) {
                    const int nc_d = m_ops->GetMassMatricesPCnComp(dim, d);
                    const int w_d = (nc_d - 1) / 2;
                    m_stencil_width = std::max(m_stencil_width, w_d);
                }
            }
        }
    }

    // The mass matrices have the same grid layout as the solver vector
    const IntVect nghost(m_stencil_width);
    for (int n = 0; n < m_num_amr_levels; n++) {
        for (int dim = 0; dim < 3; dim++) {
            const auto& ba = (*m_bcoefs)[n][dim]->boxArray();
            const auto& dm = (*m_bcoefs)[n][dim]->DistributionMap();
            m_work[n][dim].define(ba, dm, 1, 0);
            if (m_has_offdiag) {
                m_x_ghost[n][dim].define(ba, dm, 1, nghost);
            }
        }
    }
    m_work_defined = true;
}

template <class T, class Ops>
void MassMatrixSolverOps<T,Ops>::InitialGuess (T& a_x, const T& a_b)
{
    using namespace amrex;

    auto& b_mfarrvec = a_b.getArrayVec();
    auto& x_mfarrvec = a_x.getArrayVec();

    for (int n = 0; n < m_num_amr_levels; n++) {
        for (int dim = 0; dim < 3; dim++) {
            const auto& b_mf = *(b_mfarrvec[n][dim]);
            const auto& a_mf = *((*m_bcoefs)[n][dim]);
            auto& x_mf = *(x_mfarrvec[n][dim]);
            const int diag_comp = (a_mf.nComp() - 1) / 2;

            for (MFIter mfi(x_mf,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box bx = mfi.tilebox();
                auto x_arr = x_mf.array(mfi);
                auto b_arr = b_mf.const_array(mfi);
                auto a_arr = a_mf.const_array(mfi);

                ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    x_arr(i,j,k) = b_arr(i,j,k)
                        / (RT(1.0) + a_arr(i,j,k,diag_comp));
                });
            }
        }
    }
}

template <class T, class Ops>
void MassMatrixSolverOps<T,Ops>::JacobiSweep (T& a_x, const T& a_b, RT a_omega)
{
    using namespace amrex;

    auto& b_mfarrvec = a_b.getArrayVec();
    auto& x_mfarrvec = a_x.getArrayVec();
    copyToGhostAndFill(x_mfarrvec);

    for (int n = 0; n < m_num_amr_levels; n++) {
        for (int dim = 0; dim < 3; dim++) {
            const auto& b_mf = *(b_mfarrvec[n][dim]);
            const auto& a_mf = *((*m_bcoefs)[n][dim]);
            auto& x_mf = *(x_mfarrvec[n][dim]);
            auto& w_mf = m_work[n][dim];
            const auto& xg_mf = m_x_ghost[n][dim];

            const int ncomp = a_mf.nComp();
            if (ncomp <= 1) { continue; }

            const int diag_comp = (ncomp - 1) / 2;

            GpuArray<int,3> nc = {1, 1, 1};
            GpuArray<int,3> w = {0, 0, 0};
            for (int d = 0; d < AMREX_SPACEDIM; d++) {
                nc[d] = m_ops->GetMassMatricesPCnComp(dim, d);
                w[d] = (nc[d] - 1) / 2;
            }

            const RT om = a_omega;

            for (MFIter mfi(x_mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box bx = mfi.tilebox();
                auto w_arr = w_mf.array(mfi);
                auto xg_arr = xg_mf.const_array(mfi);
                auto a_arr = a_mf.const_array(mfi);

                ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    RT mx = RT(0.0);
                    int comp = 0;
                    for (int c2 = 0; c2 < nc[2]; c2++) {
                        const int kk = k + c2 - w[2];
                        for (int c1 = 0; c1 < nc[1]; c1++) {
                            const int jj = j + c1 - w[1];
                            for (int c0 = 0; c0 < nc[0]; c0++) {
                                const int ii = i + c0 - w[0];
                                mx += a_arr(i,j,k,comp)
                                    * xg_arr(ii,jj,kk);
                                comp++;
                            }
                        }
                    }
                    w_arr(i,j,k) = xg_arr(i,j,k) + mx;
                });
            }

            for (MFIter mfi(x_mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box bx = mfi.tilebox();
                auto x_arr = x_mf.array(mfi);
                auto b_arr = b_mf.const_array(mfi);
                auto w_arr = w_mf.const_array(mfi);
                auto a_arr = a_mf.const_array(mfi);

                ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    const RT r = b_arr(i,j,k) - w_arr(i,j,k);
                    x_arr(i,j,k) += om * r
                        / (RT(1.0) + a_arr(i,j,k,diag_comp));
                });
            }
        }
    }
}

template <class T, class Ops>
auto MassMatrixSolverOps<T,Ops>::ResidualNorm (const T& a_x, const T& a_b) -> RT
{
    auto& x_mfarrvec = a_x.getArrayVec();
    if (m_has_offdiag) {
        copyToGhostAndFill(x_mfarrvec);
    }
    return computeResidualNormInternal(a_x, a_b);
}

template <class T, class Ops>
auto MassMatrixSolverOps<T,Ops>::EstimateOmega () const -> RT
{
    using namespace amrex;

    if (!m_is_defined || !m_has_offdiag) { return RT(1.0); }

    // Gershgorin: for A = I + M, r_i = sum_{comp != diag} |M(i,comp)| / (1 + M(i,diag))
    RT max_r = RT(0.0);
    for (int n = 0; n < m_num_amr_levels; n++) {
        for (int dim = 0; dim < 3; dim++) {
            const auto& a_mf = *((*m_bcoefs)[n][dim]);
            const int ncomp = a_mf.nComp();
            if (ncomp <= 1) { continue; }
            const int diag_comp = (ncomp - 1) / 2;

            ReduceOps<ReduceOpMax> reduce_op;
            ReduceData<RT> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;

            for (MFIter mfi(a_mf); mfi.isValid(); ++mfi) {
                const Box bx = mfi.validbox();
                auto a_arr = a_mf.const_array(mfi);
                const int nc = ncomp;
                const int dc = diag_comp;

                reduce_op.eval(bx, reduce_data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                    {
                        RT offdiag_sum = RT(0.0);
                        for (int c = 0; c < nc; c++) {
                            if (c != dc) {
                                offdiag_sum += amrex::Math::abs(a_arr(i,j,k,c));
                            }
                        }
                        const RT diag = RT(1.0) + a_arr(i,j,k,dc);
                        return {offdiag_sum / amrex::Math::abs(diag)};
                    });
            }
            const RT dim_max_r = amrex::get<0>(reduce_data.value(reduce_op));
            max_r = std::max(max_r, dim_max_r);
        }
    }
    ParallelAllReduce::Max(max_r, ParallelContext::CommunicatorSub());

    return RT(2.0) / (RT(2.0) + max_r);
}

template <class T, class Ops>
void MassMatrixSolverOps<T,Ops>::copyToGhostAndFill (
    const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>& a_x_mfarrvec)
{
    for (int n = 0; n < m_num_amr_levels; n++) {
        const auto& geom = m_ops->GetGeometry(n);
        for (int dim = 0; dim < 3; dim++) {
            m_x_ghost[n][dim].setVal(RT(0.0));
            amrex::MultiFab::Copy(m_x_ghost[n][dim], *(a_x_mfarrvec[n][dim]),
                                  0, 0, 1, amrex::IntVect::TheZeroVector());
            m_x_ghost[n][dim].FillBoundary(geom.periodicity());
        }
    }
}

template <class T, class Ops>
auto MassMatrixSolverOps<T,Ops>::computeResidualNormInternal (
    const T& a_x, const T& a_b) const -> RT
{
    using namespace amrex;
    auto& b_mfarrvec = a_b.getArrayVec();
    auto& x_mfarrvec = a_x.getArrayVec();

    RT res_sq = RT(0.0);
    for (int n = 0; n < m_num_amr_levels; n++) {
        for (int dim = 0; dim < 3; dim++) {
            const auto& b_mf = *(b_mfarrvec[n][dim]);
            const auto& a_mf = *((*m_bcoefs)[n][dim]);
            const int ncomp = a_mf.nComp();

            GpuArray<int,3> nc = {1, 1, 1};
            GpuArray<int,3> w = {0, 0, 0};
            if (ncomp > 1) {
                for (int d = 0; d < AMREX_SPACEDIM; d++) {
                    nc[d] = m_ops->GetMassMatricesPCnComp(dim, d);
                    w[d] = (nc[d] - 1) / 2;
                }
            }

            ReduceOps<ReduceOpSum> reduce_op;
            ReduceData<RT> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;

            const auto& xg_mf = (ncomp > 1 && m_work_defined)
                ? m_x_ghost[n][dim] : *(x_mfarrvec[n][dim]);

            for (MFIter mfi(b_mf); mfi.isValid(); ++mfi) {
                const Box bx = mfi.validbox();
                auto xg_arr = xg_mf.const_array(mfi);
                auto b_arr = b_mf.const_array(mfi);
                auto a_arr = a_mf.const_array(mfi);

                if (ncomp == 1) {
                    auto x_arr = x_mfarrvec[n][dim]->const_array(mfi);
                    reduce_op.eval(bx, reduce_data,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                        {
                            const RT Ax = x_arr(i,j,k) * (RT(1.0) + a_arr(i,j,k,0));
                            const RT r = b_arr(i,j,k) - Ax;
                            return {r * r};
                        });
                } else {
                    reduce_op.eval(bx, reduce_data,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                        {
                            RT mx = RT(0.0);
                            int comp = 0;
                            for (int c2 = 0; c2 < nc[2]; c2++) {
                                const int kk = k + c2 - w[2];
                                for (int c1 = 0; c1 < nc[1]; c1++) {
                                    const int jj = j + c1 - w[1];
                                    for (int c0 = 0; c0 < nc[0]; c0++) {
                                        const int ii = i + c0 - w[0];
                                        mx += a_arr(i,j,k,comp) * xg_arr(ii,jj,kk);
                                        comp++;
                                    }
                                }
                            }
                            const RT Ax = xg_arr(i,j,k) + mx;
                            const RT r = b_arr(i,j,k) - Ax;
                            return {r * r};
                        });
                }
            }
            RT dim_res_sq = amrex::get<0>(reduce_data.value(reduce_op));
            ParallelAllReduce::Sum(dim_res_sq, ParallelContext::CommunicatorSub());
            res_sq += dim_res_sq;
        }
    }
    return std::sqrt(res_sq);
}

#endif
