/* Copyright 2025 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef MULTIFAB_MATRIX_SOLVER_OPS_H_
#define MULTIFAB_MATRIX_SOLVER_OPS_H_

#include "MatrixSolverOps.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_Reduce.H>
#include <AMReX_Vector.H>

/**
 * \brief MultiFab stencil-based matrix operations for Jacobi iteration.
 *
 *  Used when A = I + M (semi-implicit path, theta == 0).
 *  The mass matrix M is stored as MultiFab stencil coefficients.
 *  When diagonal-only (nComp==1), the solve is exact in one step.
 */
template <class T, class Ops>
class MultiFabMatrixSolverOps : public MatrixSolverOps<T>
{
public:

    using RT = typename T::value_type;

    MultiFabMatrixSolverOps () = default;
    ~MultiFabMatrixSolverOps () override = default;

    void Define (Ops* a_ops);

    void Update (const T& a_U) override;

    void InitialGuess (T& a_x, const T& a_b) override;

    void JacobiSweep (T& a_x, const T& a_b, RT a_omega) override;

    [[nodiscard]] RT ResidualNorm (const T& a_x, const T& a_b) override;

    [[nodiscard]] bool HasOffDiag () const override { return m_has_offdiag; }

    [[nodiscard]] bool IsDefined () const override { return m_is_defined; }

private:

    void copyToGhostAndFill (
        const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>& a_x_mfarrvec);

    RT computeResidualNormInternal (const T& a_x, const T& a_b) const;

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
void MultiFabMatrixSolverOps<T,Ops>::Define (Ops* a_ops)
{
    m_ops = a_ops;
    m_num_amr_levels = m_ops->numAMRLevels();
    m_bcoefs = m_ops->GetMassMatricesCoeff();
    // When m_bcoefs is null (no mass matrices), A = I and x = b is exact.
    // Report as "not defined" so the caller can handle the identity case.
    m_is_defined = (m_bcoefs != nullptr);
}

template <class T, class Ops>
void MultiFabMatrixSolverOps<T,Ops>::Update (const T& a_U)
{
    using namespace amrex;

    if (m_bcoefs == nullptr || m_work_defined) { return; }

    auto& u_mfarrvec = a_U.getArrayVec();
    m_work.resize(m_num_amr_levels);
    m_x_ghost.resize(m_num_amr_levels);

    m_stencil_width = 0;
    for (int n = 0; n < m_num_amr_levels; n++) {
        for (int dim = 0; dim < 3; dim++) {
            if ((*m_bcoefs)[n][dim]->nComp() > 1) {
                m_has_offdiag = true;
                for (int d = 0; d < AMREX_SPACEDIM; d++) {
                    int nc_d = m_ops->GetMassMatricesPCnComp(dim, d);
                    int w_d = (nc_d - 1) / 2;
                    m_stencil_width = std::max(m_stencil_width, w_d);
                }
            }
        }
    }

    const IntVect nghost(m_stencil_width);
    for (int n = 0; n < m_num_amr_levels; n++) {
        for (int dim = 0; dim < 3; dim++) {
            m_work[n][dim].define(
                u_mfarrvec[n][dim]->boxArray(),
                u_mfarrvec[n][dim]->DistributionMap(),
                1, 0);
            if (m_has_offdiag) {
                m_x_ghost[n][dim].define(
                    u_mfarrvec[n][dim]->boxArray(),
                    u_mfarrvec[n][dim]->DistributionMap(),
                    1, nghost);
            }
        }
    }
    m_work_defined = true;
}

template <class T, class Ops>
void MultiFabMatrixSolverOps<T,Ops>::InitialGuess (T& a_x, const T& a_b)
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
                Box bx = mfi.tilebox();
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
void MultiFabMatrixSolverOps<T,Ops>::JacobiSweep (T& a_x, const T& a_b, RT a_omega)
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
                Box bx = mfi.tilebox();
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
                Box bx = mfi.tilebox();
                auto x_arr = x_mf.array(mfi);
                auto b_arr = b_mf.const_array(mfi);
                auto w_arr = w_mf.const_array(mfi);
                auto a_arr = a_mf.const_array(mfi);

                ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    RT r = b_arr(i,j,k) - w_arr(i,j,k);
                    x_arr(i,j,k) += om * r
                        / (RT(1.0) + a_arr(i,j,k,diag_comp));
                });
            }
        }
    }
}

template <class T, class Ops>
auto MultiFabMatrixSolverOps<T,Ops>::ResidualNorm (const T& a_x, const T& a_b) -> RT
{
    auto& x_mfarrvec = a_x.getArrayVec();
    if (m_has_offdiag) {
        copyToGhostAndFill(x_mfarrvec);
    }
    return computeResidualNormInternal(a_x, a_b);
}

template <class T, class Ops>
void MultiFabMatrixSolverOps<T,Ops>::copyToGhostAndFill (
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
auto MultiFabMatrixSolverOps<T,Ops>::computeResidualNormInternal (
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
                Box bx = mfi.validbox();
                auto xg_arr = xg_mf.const_array(mfi);
                auto b_arr = b_mf.const_array(mfi);
                auto a_arr = a_mf.const_array(mfi);

                if (ncomp == 1) {
                    auto x_arr = x_mfarrvec[n][dim]->const_array(mfi);
                    reduce_op.eval(bx, reduce_data,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                        {
                            RT Ax = x_arr(i,j,k) * (RT(1.0) + a_arr(i,j,k,0));
                            RT r = b_arr(i,j,k) - Ax;
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
                            RT Ax = xg_arr(i,j,k) + mx;
                            RT r = b_arr(i,j,k) - Ax;
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
