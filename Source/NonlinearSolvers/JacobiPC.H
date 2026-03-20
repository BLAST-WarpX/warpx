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
#include <AMReX_Vector.H>
#include <AMReX_MultiFab.H>

/**
 * \brief Point-Jacobi Preconditioner
 *
 *  Approximately solves (I + M) x = b using damped Jacobi iterations,
 *  where M is the scaled mass matrix (plasma response dJ/dE).
 *
 *  When the mass matrix is purely diagonal (mass_matrices_pc_width=0),
 *  the solve is exact in one step: x = b / (1 + M_diag).
 *
 *  When the mass matrix has off-diagonal stencil entries, damped Jacobi
 *  iterations are performed. This is effective when the operator is
 *  diagonally dominant, which occurs when there is no implicit EM
 *  component (SemiImplicitEM, StrangImplicitSpectralEM) or when the
 *  EM component is weak.
 *
 *  The mass matrix MultiFab stores stencil coefficients as components.
 *  For a field direction dim, the per-spatial-direction component counts
 *  are given by GetMassMatricesPCnComp(dim, d). The total number of
 *  components is the product over spatial dimensions. The diagonal
 *  component is at index (nComp-1)/2. The stencil offset for component c
 *  is decoded from the component index using the per-direction counts.
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
         * \brief Solve (I + M) x = b via damped Jacobi iteration
         */
        void Apply (T&, const T&) override;

        void printParameters() const override;

        [[nodiscard]] inline bool IsDefined () const override { return m_is_defined; }

    protected:

        using MFArr = amrex::Array<amrex::MultiFab,3>;

        bool m_is_defined = false;

        bool m_verbose = true;
        int m_max_iter = 10;
        RT m_omega = RT(2.0/3.0);

        RT m_atol = 1.0e-16;
        RT m_rtol = 1.0e-4;

        Ops* m_ops = nullptr;

        int m_num_amr_levels = 0;

        const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>* m_bcoefs = nullptr;

        /**
         * \brief Whether any mass matrix has off-diagonal stencil entries
         */
        bool m_has_offdiag = false;

        /**
         * \brief Work MultiFabs for storing (I+M)*x during Jacobi iterations.
         *        Indexed as m_work[level][dim].
         */
        amrex::Vector<amrex::Array<amrex::MultiFab,3>> m_work;

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
    Print() << pc_name << " absolute tolerance:   " << m_atol << "\n";
    Print() << pc_name << " relative tolerance:   " << m_rtol << "\n";
}

template <class T, class Ops>
void JacobiPC<T,Ops>::readParameters()
{
    const amrex::ParmParse pp(amrex::getEnumNameString(PreconditionerType::pc_jacobi));
    pp.query("verbose", m_verbose);
    pp.query("max_iter", m_max_iter);
    pp.query("omega", m_omega);
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
    m_bcoefs = m_ops->GetMassMatricesCoeff();

    readParameters();

    if (m_bcoefs != nullptr) {
        auto& u_mfarrvec = a_U.getArrayVec();
        m_work.resize(m_num_amr_levels);
        for (int n = 0; n < m_num_amr_levels; n++) {
            for (int dim = 0; dim < 3; dim++) {
                if ((*m_bcoefs)[n][dim]->nComp() > 1) {
                    m_has_offdiag = true;
                }
                m_work[n][dim].define(
                    u_mfarrvec[n][dim]->boxArray(),
                    u_mfarrvec[n][dim]->DistributionMap(),
                    1, 0);
            }
        }
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

    amrex::ignore_unused(a_U);

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

    // Initial guess: x = b / (1 + M_diag)
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

                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    x_arr(i,j,k) = b_arr(i,j,k) / (RT(1.0) + a_arr(i,j,k,diag_comp));
                });
            }
        }
    }

    if (!m_has_offdiag) { return; }

    // Damped Jacobi iterations for multi-component (off-diagonal) mass matrices.
    // Solves (I + M) x = b where M is stored as a stencil in the mass matrix
    // MultiFab components.
    for (int iter = 0; iter < m_max_iter; iter++) {

        // Fill ghost cells so neighbor values are available for stencil
        for (int n = 0; n < m_num_amr_levels; n++) {
            const auto& geom = m_ops->GetGeometry(n);
            for (int dim = 0; dim < 3; dim++) {
                x_mfarrvec[n][dim]->FillBoundary(geom.periodicity());
            }
        }

        // For each field direction, compute (I+M)*x into work, then update x
        for (int n = 0; n < m_num_amr_levels; n++) {
            const auto& geom = m_ops->GetGeometry(n);
            amrex::ignore_unused(geom);

            for (int dim = 0; dim < 3; dim++) {
                const auto& b_mf = *(b_mfarrvec[n][dim]);
                const auto& a_mf = *((*m_bcoefs)[n][dim]);
                auto& x_mf = *(x_mfarrvec[n][dim]);
                auto& w_mf = m_work[n][dim];

                const int ncomp = a_mf.nComp();
                if (ncomp <= 1) { continue; }

                const int diag_comp = (ncomp - 1) / 2;

                amrex::GpuArray<int,3> nc = {1, 1, 1};
                amrex::GpuArray<int,3> w = {0, 0, 0};
                for (int d = 0; d < AMREX_SPACEDIM; d++) {
                    nc[d] = m_ops->GetMassMatricesPCnComp(dim, d);
                    w[d] = (nc[d] - 1) / 2;
                }

                const RT omega = m_omega;

                // Pass 1: compute (I+M)*x into work MultiFab
                for (MFIter mfi(x_mf,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    Box bx = mfi.tilebox();
                    auto w_arr = w_mf.array(mfi);
                    auto x_arr = x_mf.const_array(mfi);
                    auto a_arr = a_mf.const_array(mfi);

                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
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
                                        * x_arr(ii,jj,kk);
                                    comp++;
                                }
                            }
                        }
                        w_arr(i,j,k) = x_arr(i,j,k) + mx;
                    });
                }

                // Pass 2: x += omega * (b - work) / (1 + M_diag)
                for (MFIter mfi(x_mf,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    Box bx = mfi.tilebox();
                    auto x_arr = x_mf.array(mfi);
                    auto b_arr = b_mf.const_array(mfi);
                    auto w_arr = w_mf.const_array(mfi);
                    auto a_arr = a_mf.const_array(mfi);

                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
                        RT res = b_arr(i,j,k) - w_arr(i,j,k);
                        x_arr(i,j,k) += omega * res
                            / (RT(1.0) + a_arr(i,j,k,diag_comp));
                    });
                }
            }
        }
    }
}

#endif
