/* Copyright 2024 Debojyoti Ghosh
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

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Array.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_Reduce.H>
#include <AMReX_Vector.H>
#include <AMReX_MultiFab.H>

/**
 * \brief Point-Jacobi Preconditioner
 *
 *  Approximately solves (I + M) x = b using damped Jacobi iterations,
 *  where M is the scaled mass matrix (plasma response dJ/dE).
 *
 *  When the mass matrix is purely diagonal (mass_matrices_pc_width=0),
 *  the solve is exact in one step: x = b / (1 + M_diag). When the mass
 *  matrix has off-diagonal stencil entries, damped Jacobi iterations are
 *  performed.
 *
 *  Adaptive omega (enabled by default): the relaxation parameter omega
 *  starts at 1.0 and is automatically reduced (by a factor of 0.9) when
 *  the Jacobi residual increases, indicating that the current omega is
 *  too large for the spectral properties of the mass matrix. The
 *  adapted omega is cached for subsequent Apply() calls. This handles
 *  marginally diagonally dominant or non-symmetric mass matrices. When
 *  the user explicitly sets pc_jacobi.omega, adaptive behavior is disabled
 *  (can be re-enabled with pc_jacobi.adaptive_omega = true).
 *
 *  Matrix Elements Layout:
 *  The mass matrix MultiFab stores stencil coefficients as components.
 *  For a field direction dim, the per-spatial-direction component counts
 *  are given by GetMassMatricesPCnComp(dim, d). The total number of
 *  components is the product over spatial dimensions. The diagonal
 *  component is at index (nComp-1)/2. The stencil offset for component c
 *  is decoded from the component index using the per-direction counts.
 *
 *  Note: WarpXSolverVec MultiFabs have zero ghost cells. This class
 *  allocates its own ghost-padded temporaries (m_x_ghost) for stencil
 *  neighbor access. All FillBoundary calls operate on these temporaries.
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

        void Define (const T&, Ops*) override;

        void Update (const T& a_U) override;

        /**
         * \brief Solve (I + M) x = b via damped Jacobi iteration
         */
        void Apply (T&, const T&) override;

        void printParameters() const override;

        [[nodiscard]] inline bool IsDefined () const override { return m_is_defined; }

        /**
         * \brief Compute ||b - (I+M)*x|| for residual monitoring.
         *        Uses the ghost-padded m_x_ghost (which must already be
         *        populated and boundary-filled) for stencil neighbor access.
         *        For diagonal-only mass matrices (ncomp=1), uses the fast
         *        diagonal formula from x directly.
         */
        RT computeResidualNorm (const T& a_x, const T& a_b) const;

    protected:

        using MFArr = amrex::Array<amrex::MultiFab,3>;

        bool m_is_defined = false;

        bool m_verbose = false;
        int m_max_iter = 200;
        RT m_omega = RT(1.0);
        bool m_adaptive_omega = true;

        RT m_atol = 1.0e-16;
        RT m_rtol = 1.0e-4;

        Ops* m_ops = nullptr;

        int m_num_amr_levels = 0;

        const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>* m_bcoefs = nullptr;

        bool m_has_offdiag = false;
        bool m_work_defined = false;

        amrex::Vector<amrex::Array<amrex::MultiFab,3>> m_work;
        amrex::Vector<amrex::Array<amrex::MultiFab,3>> m_x_ghost;

        int m_stencil_width = 0;

        void readParameters();

        void copyToGhostAndFill (
            const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>& a_x_mfarrvec);

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
void JacobiPC<T,Ops>::copyToGhostAndFill (
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
auto JacobiPC<T,Ops>::computeResidualNorm (const T& a_x, const T& a_b) const -> RT
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

            // Use the ghost-padded copy for stencil access when available
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
    m_bcoefs = m_ops->GetMassMatricesCoeff();

    readParameters();

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

    if (m_bcoefs != nullptr && !m_work_defined) {
        auto& u_mfarrvec = a_U.getArrayVec();
        m_work.resize(m_num_amr_levels);
        m_x_ghost.resize(m_num_amr_levels);

        // Determine max stencil width needed across all dims/levels
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

        const amrex::IntVect nghost(m_stencil_width);
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

    if (m_verbose) {
        amrex::Print() << "Updating " << amrex::getEnumNameString(PreconditionerType::pc_jacobi)
                       << ": theta*dt = " << this->m_dt << "\n";
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

    if (m_bcoefs == nullptr) {
        a_x.Copy(a_b);
        return;
    }

    auto& b_mfarrvec = a_b.getArrayVec();
    auto& x_mfarrvec = a_x.getArrayVec();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        ((b_mfarrvec.size() == m_num_amr_levels) && (x_mfarrvec.size() == m_num_amr_levels)),
        "Error in JacobiPC::Apply() - mismatch in number of levels." );

    const int max_omega_reductions = m_adaptive_omega ? 10 : 0;
    RT omega = m_omega;

    for (int adapt = 0; adapt <= max_omega_reductions; adapt++) {

        // Initial guess: x = b / (1 + M_diag)
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

        if (!m_has_offdiag) {
            if (m_verbose) {
                const RT bnorm = a_b.norm2();
                const RT res = computeResidualNorm(a_x, a_b);
                amrex::Print() << "  pc_jacobi (diag): residual = " << res
                               << ", " << res / std::max(bnorm, RT(1.e-30))
                               << " (rel.)\n";
            }
            return;
        }

        copyToGhostAndFill(x_mfarrvec);

        const RT bnorm = a_b.norm2();
        const RT res0 = computeResidualNorm(a_x, a_b);
        RT res_prev = res0;

        if (m_verbose) {
            amrex::Print() << "  pc_jacobi: iter = 0, residual = " << res0
                           << ", " << res0 / std::max(bnorm, RT(1.e-30))
                           << " (rel.), omega = " << omega << "\n";
        }

        bool diverged = false;

        for (int iter = 0; iter < m_max_iter; iter++) {

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

                    amrex::GpuArray<int,3> nc = {1, 1, 1};
                    amrex::GpuArray<int,3> w = {0, 0, 0};
                    for (int d = 0; d < AMREX_SPACEDIM; d++) {
                        nc[d] = m_ops->GetMassMatricesPCnComp(dim, d);
                        w[d] = (nc[d] - 1) / 2;
                    }

                    const RT om = omega;

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

            copyToGhostAndFill(x_mfarrvec);

            const RT res_norm = computeResidualNorm(a_x, a_b);
            if (m_verbose) {
                amrex::Print() << "  pc_jacobi: iter = " << (iter+1)
                               << ", residual = " << res_norm
                               << ", " << res_norm / std::max(bnorm, RT(1.e-30))
                               << " (rel.)\n";
            }

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
