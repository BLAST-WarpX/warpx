/* Copyright 2025-2026 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef MF_MATRIX_SOLVER_OPS_H_
#define MF_MATRIX_SOLVER_OPS_H_

#include "NonlinearSolvers/MatrixSolverOps.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <functional>

/**
 * \brief Matrix-free MatrixSolverOps using an external matvec callable.
 *
 *  The diagonal of the Jacobian is diag(J) = I + scale * M_diag where
 *  M_diag is the diagonal component of the mass matrix stencil coefficients
 *  and scale = c^2 * mu0 * theta * dt. The EM curl-curl term has no
 *  diagonal component (it couples only off-diagonal staggered grid cells),
 *  so the diagonal comes entirely from the identity and the mass matrix.
 *
 *  The matrix-vector product (J*v) is provided as a std::function callable,
 *  typically bound to JacobianFunctionMF::apply().
 *
 *  EstimateOmega() returns 1.0 since no spectral information is available
 *  from the matrix-free product.
 */
template <class T>
class MFMatrixSolverOps : public MatrixSolverOps<T>
{
public:

    using RT = typename T::value_type;
    using MatvecFn = std::function<void(T&, const T&)>;

    MFMatrixSolverOps () = default;
    ~MFMatrixSolverOps () override = default;

    MFMatrixSolverOps (const MFMatrixSolverOps&) = delete;
    MFMatrixSolverOps& operator= (const MFMatrixSolverOps&) = delete;
    MFMatrixSolverOps (MFMatrixSolverOps&&) noexcept = delete;
    MFMatrixSolverOps& operator= (MFMatrixSolverOps&&) noexcept = delete;

    /**
     * \brief Define with a matvec callable and mass matrix coefficients.
     *
     * \param[in] a_matvec          callable computing y = J*x
     * \param[in] a_bcoefs          mass matrix stencil MultiFab pointers (may be nullptr)
     * \param[in] a_num_amr_levels  number of AMR levels
     */
    void Define (MatvecFn a_matvec,
                 const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>* a_bcoefs,
                 int a_num_amr_levels);

    void Update (const T& a_U) override;
    void InitialGuess (T& a_x, const T& a_b) override;
    void JacobiSweep (T& a_x, const T& a_b, RT a_omega) override;
    [[nodiscard]] RT ResidualNorm (const T& a_x, const T& a_b) override;

    /** No spectral information available from matrix-free product. */
    [[nodiscard]] RT EstimateOmega () const override { return RT(1.0); }

    /** The curl-curl term always contributes off-diagonal entries. */
    [[nodiscard]] bool HasOffDiag () const override { return true; }

    [[nodiscard]] bool IsDefined () const override { return m_is_defined; }

    /** Set the scaling factor (c^2 * mu0 * theta * dt) for mass matrix diagonal. */
    void SetPCFactor (RT a_pc_factor) { m_pc_factor = a_pc_factor; }

private:

    bool m_is_defined = false;
    bool m_work_defined = false;

    MatvecFn m_matvec;
    const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>* m_bcoefs = nullptr;
    int m_num_amr_levels = 0;
    RT m_pc_factor = RT(1.0);  /**< c^2 * mu0 * theta * dt scaling for diagonal */

    T m_Jx;  /**< workspace for matvec result */
};

// ============================================================================
//  Implementation
// ============================================================================

template <class T>
void MFMatrixSolverOps<T>::Define (
    MatvecFn a_matvec,
    const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>* a_bcoefs,
    int a_num_amr_levels)
{
    m_matvec = std::move(a_matvec);
    m_bcoefs = a_bcoefs;
    m_num_amr_levels = a_num_amr_levels;
    m_is_defined = true;
}

template <class T>
void MFMatrixSolverOps<T>::Update (const T& a_U)
{
    if (!m_work_defined) {
        m_Jx.Define(a_U);
        m_work_defined = true;
    }
}

template <class T>
void MFMatrixSolverOps<T>::InitialGuess (T& a_x, const T& a_b)
{
    using namespace amrex;

    if (m_bcoefs == nullptr) {
        // diag(J) = I, so x = b
        a_x.Copy(a_b);
        return;
    }

    auto& b_mfarrvec = a_b.getArrayVec();
    auto& x_mfarrvec = a_x.getArrayVec();
    const RT pc_factor = m_pc_factor;

    for (int n = 0; n < m_num_amr_levels; n++) {
        for (int dim = 0; dim < 3; dim++) {
            const auto& b_mf = *(b_mfarrvec[n][dim]);
            const auto& a_mf = *((*m_bcoefs)[n][dim]);
            auto& x_mf = *(x_mfarrvec[n][dim]);
            const int diag_comp = (a_mf.nComp() - 1) / 2;

            for (MFIter mfi(x_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                Box bx = mfi.tilebox();
                auto x_arr = x_mf.array(mfi);
                auto b_arr = b_mf.const_array(mfi);
                auto a_arr = a_mf.const_array(mfi);

                ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    x_arr(i,j,k) = b_arr(i,j,k)
                        / (RT(1.0) + pc_factor * a_arr(i,j,k,diag_comp));
                });
            }
        }
    }
}

template <class T>
void MFMatrixSolverOps<T>::JacobiSweep (T& a_x, const T& a_b, RT a_omega)
{
    using namespace amrex;

    // Compute Jx = J * x using the matrix-free product
    m_matvec(m_Jx, a_x);

    auto& x_mfarrvec = a_x.getArrayVec();
    auto& b_mfarrvec = a_b.getArrayVec();
    auto& jx_mfarrvec = m_Jx.getArrayVec();

    const RT om = a_omega;
    const RT pc_factor = m_pc_factor;

    for (int n = 0; n < m_num_amr_levels; n++) {
        for (int dim = 0; dim < 3; dim++) {
            auto& x_mf = *(x_mfarrvec[n][dim]);
            const auto& b_mf = *(b_mfarrvec[n][dim]);
            const auto& jx_mf = *(jx_mfarrvec[n][dim]);

            if (m_bcoefs != nullptr) {
                const auto& a_mf = *((*m_bcoefs)[n][dim]);
                const int diag_comp = (a_mf.nComp() - 1) / 2;

                for (MFIter mfi(x_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    Box bx = mfi.tilebox();
                    auto x_arr = x_mf.array(mfi);
                    auto b_arr = b_mf.const_array(mfi);
                    auto jx_arr = jx_mf.const_array(mfi);
                    auto a_arr = a_mf.const_array(mfi);

                    ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        RT r = b_arr(i,j,k) - jx_arr(i,j,k);
                        x_arr(i,j,k) += om * r
                            / (RT(1.0) + pc_factor * a_arr(i,j,k,diag_comp));
                    });
                }
            } else {
                // No mass matrices: diag(J) = I
                for (MFIter mfi(x_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    Box bx = mfi.tilebox();
                    auto x_arr = x_mf.array(mfi);
                    auto b_arr = b_mf.const_array(mfi);
                    auto jx_arr = jx_mf.const_array(mfi);

                    ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        x_arr(i,j,k) += om * (b_arr(i,j,k) - jx_arr(i,j,k));
                    });
                }
            }
        }
    }
}

template <class T>
auto MFMatrixSolverOps<T>::ResidualNorm (const T& a_x, const T& a_b) -> RT
{
    // Compute Jx = J * x
    m_matvec(m_Jx, a_x);

    // Compute ||b - Jx||_2 by reusing m_Jx as workspace
    m_Jx.scale(RT(-1.0));           // m_Jx = -J*x
    m_Jx.increment(a_b, RT(1.0));   // m_Jx = b - J*x
    return m_Jx.norm2();
}

#endif
