/* Copyright 2025-2026 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef MF_MATRIX_SOLVER_OPS_H_
#define MF_MATRIX_SOLVER_OPS_H_

#include "MatrixMarketWriter.H"
#include "WarpXSolverDOF.H"
#include "NonlinearSolvers/MatrixSolverOps.H"
#include "Utils/TextMsg.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <functional>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

/**
 * \brief Matrix-free MatrixSolverOps using an external matvec callable.
 *
 *  The diagonal of the Jacobian is
 *      diag(J) = I + scale * M_diag + alpha * CC_diag
 *  where M_diag is the diagonal component of the mass matrix stencil
 *  coefficients, scale = c^2 * mu0 * theta * dt, CC_diag is the diagonal of
 *  the curl-curl operator, and alpha = (c * theta * dt)^2. The curl-curl
 *  term is absent when light waves are not treated implicitly.
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
     * \brief Define with a matvec callable and the diagonal coefficients.
     *
     * \param[in] a_matvec          callable computing y = J*x
     * \param[in] a_bcoefs          mass matrix stencil MultiFab pointers (may be nullptr)
     * \param[in] a_curlcurl_diag   curl-curl diagonal MultiFab pointers (may be nullptr)
     * \param[in] a_num_amr_levels  number of AMR levels
     */
    void Define (MatvecFn a_matvec,
                 const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>* a_bcoefs,
                 const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>* a_curlcurl_diag,
                 int a_num_amr_levels);

    void Update () override {}
    void InitialGuess (T& a_x, const T& a_b) override;
    void JacobiSweep (T& a_x, const T& a_b, RT a_omega) override;
    [[nodiscard]] RT ResidualNorm (const T& a_x, const T& a_b) override;
    void Matvec (T& a_y, const T& a_x) override { m_matvec(a_y, a_x); }

    /** Writes the metadata and diagonal range only: the entries are not stored. */
    void PrintMatrix (const std::string& a_filename,
                      const std::vector<std::string>& a_header) override;

    /** No spectral information available from matrix-free product. */
    [[nodiscard]] RT EstimateOmega () const override { return RT(1.0); }
    [[nodiscard]] bool HasOmegaEstimate () const override { return false; }

    /** The curl-curl term always contributes off-diagonal entries. */
    [[nodiscard]] bool HasOffDiag () const override { return true; }

    [[nodiscard]] bool IsDefined () const override { return m_is_defined; }

    void SetPCFactor (RT a_pc_factor) override { m_pc_factor = a_pc_factor; }

    void SetCurlCurlFactor (RT a_curlcurl_factor) override { m_curlcurl_factor = a_curlcurl_factor; }

private:

    /** \brief Define the matvec workspace on first use. */
    void defineWork (const T& a_x)
    {
        if (!m_work_defined) {
            m_Jx.Define(a_x);
            m_work_defined = true;
        }
    }

    bool m_is_defined = false;
    bool m_work_defined = false;

    MatvecFn m_matvec;
    const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>* m_bcoefs = nullptr;
    const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>* m_curlcurl_diag = nullptr;
    int m_num_amr_levels = 0;
    RT m_pc_factor = RT(1.0);  /**< c^2 * mu0 * theta * dt scaling for diagonal */
    RT m_curlcurl_factor = RT(0.0);  /**< (c * theta * dt)^2 scaling for curl-curl diagonal */

    T m_Jx;  /**< workspace for matvec result */
};

// Implementation

template <class T>
void MFMatrixSolverOps<T>::Define (
    MatvecFn a_matvec,
    const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>* a_bcoefs,
    const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>* a_curlcurl_diag,
    int a_num_amr_levels)
{
    m_matvec = std::move(a_matvec);
    m_bcoefs = a_bcoefs;
    m_curlcurl_diag = a_curlcurl_diag;
    m_num_amr_levels = a_num_amr_levels;
    m_is_defined = true;
}

template <class T>
void MFMatrixSolverOps<T>::InitialGuess (T& a_x, const T& a_b)
{
    using namespace amrex;

    const bool has_mm = (m_bcoefs != nullptr);
    const bool has_cc = (m_curlcurl_diag != nullptr);

    if (!has_mm && !has_cc) {
        // diag(J) = I, so x = b
        a_x.Copy(a_b);
        return;
    }

    auto& b_mfarrvec = a_b.getArrayVec();
    auto& x_mfarrvec = a_x.getArrayVec();
    const RT pc_factor = m_pc_factor;
    const RT curlcurl_factor = m_curlcurl_factor;

    for (int n = 0; n < m_num_amr_levels; n++) {
        for (int dim = 0; dim < 3; dim++) {
            const auto& b_mf = *(b_mfarrvec[n][dim]);
            auto& x_mf = *(x_mfarrvec[n][dim]);
            const int diag_comp = has_mm ? ((*m_bcoefs)[n][dim]->nComp() - 1) / 2 : 0;

            for (MFIter mfi(x_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box bx = mfi.tilebox();
                auto x_arr = x_mf.array(mfi);
                auto b_arr = b_mf.const_array(mfi);
                const auto a_arr = has_mm ? (*m_bcoefs)[n][dim]->const_array(mfi)
                                          : Array4<const Real>{};
                const auto c_arr = has_cc ? (*m_curlcurl_diag)[n][dim]->const_array(mfi)
                                          : Array4<const Real>{};

                ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    RT diag = RT(1.0);
                    if (has_mm) { diag += pc_factor * a_arr(i,j,k,diag_comp); }
                    if (has_cc) { diag += curlcurl_factor * c_arr(i,j,k); }
                    x_arr(i,j,k) = b_arr(i,j,k) / diag;
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
    defineWork(a_x);
    m_matvec(m_Jx, a_x);

    auto& x_mfarrvec = a_x.getArrayVec();
    auto& b_mfarrvec = a_b.getArrayVec();
    auto& jx_mfarrvec = m_Jx.getArrayVec();

    const bool has_mm = (m_bcoefs != nullptr);
    const bool has_cc = (m_curlcurl_diag != nullptr);
    const RT om = a_omega;
    const RT pc_factor = m_pc_factor;
    const RT curlcurl_factor = m_curlcurl_factor;

    for (int n = 0; n < m_num_amr_levels; n++) {
        for (int dim = 0; dim < 3; dim++) {
            auto& x_mf = *(x_mfarrvec[n][dim]);
            const auto& b_mf = *(b_mfarrvec[n][dim]);
            const auto& jx_mf = *(jx_mfarrvec[n][dim]);
            const int diag_comp = has_mm ? ((*m_bcoefs)[n][dim]->nComp() - 1) / 2 : 0;

            for (MFIter mfi(x_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box bx = mfi.tilebox();
                auto x_arr = x_mf.array(mfi);
                auto b_arr = b_mf.const_array(mfi);
                auto jx_arr = jx_mf.const_array(mfi);
                const auto a_arr = has_mm ? (*m_bcoefs)[n][dim]->const_array(mfi)
                                          : Array4<const Real>{};
                const auto c_arr = has_cc ? (*m_curlcurl_diag)[n][dim]->const_array(mfi)
                                          : Array4<const Real>{};

                ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    RT diag = RT(1.0);
                    if (has_mm) { diag += pc_factor * a_arr(i,j,k,diag_comp); }
                    if (has_cc) { diag += curlcurl_factor * c_arr(i,j,k); }
                    x_arr(i,j,k) += om * (b_arr(i,j,k) - jx_arr(i,j,k)) / diag;
                });
            }
        }
    }
}

template <class T>
auto MFMatrixSolverOps<T>::ResidualNorm (const T& a_x, const T& a_b) -> RT
{
    // Compute Jx = J * x
    defineWork(a_x);
    m_matvec(m_Jx, a_x);

    // Compute ||b - Jx||_2 by reusing m_Jx as workspace
    m_Jx.scale(RT(-1.0));           // m_Jx = -J*x
    m_Jx.increment(a_b, RT(1.0));   // m_Jx = b - J*x
    return m_Jx.norm2();
}

template <class T>
void MFMatrixSolverOps<T>::PrintMatrix (const std::string& a_filename,
                                       const std::vector<std::string>& a_header)
{
    BL_PROFILE("MFMatrixSolverOps::PrintMatrix()");
    using namespace amrex;

    const auto* dofs = T::getDOFsObject().get();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(dofs != nullptr,
        "MFMatrixSolverOps::PrintMatrix(): DOF object is a nullptr");

    // Range of the diagonal 1 + pc_factor * M_diag + curlcurl_factor * CC_diag
    const bool has_mm = (m_bcoefs != nullptr);
    const bool has_cc = (m_curlcurl_diag != nullptr);
    const RT pc_factor = m_pc_factor;
    const RT curlcurl_factor = m_curlcurl_factor;
    RT dmin = std::numeric_limits<RT>::max();
    RT dmax = std::numeric_limits<RT>::lowest();
    if (!has_mm && !has_cc) {
        dmin = dmax = RT(1.0);
    } else {
        for (int n = 0; n < m_num_amr_levels; n++) {
            for (int dim = 0; dim < 3; dim++) {
                const MultiFab& mf = has_mm ? *(*m_bcoefs)[n][dim] : *(*m_curlcurl_diag)[n][dim];
                const int diag_comp = has_mm ? ((*m_bcoefs)[n][dim]->nComp() - 1) / 2 : 0;
                ReduceOps<ReduceOpMin, ReduceOpMax> reduce_op;
                ReduceData<RT, RT> reduce_data(reduce_op);
                using ReduceTuple = typename decltype(reduce_data)::Type;
                for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
                    const auto a_arr = has_mm ? (*m_bcoefs)[n][dim]->const_array(mfi)
                                              : Array4<const Real>{};
                    const auto c_arr = has_cc ? (*m_curlcurl_diag)[n][dim]->const_array(mfi)
                                              : Array4<const Real>{};
                    reduce_op.eval(mfi.validbox(), reduce_data,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                        {
                            RT diag = RT(1.0);
                            if (has_mm) { diag += pc_factor * a_arr(i,j,k,diag_comp); }
                            if (has_cc) { diag += curlcurl_factor * c_arr(i,j,k); }
                            return {diag, diag};
                        });
                }
                const auto r = reduce_data.value(reduce_op);
                dmin = std::min(dmin, amrex::get<0>(r));
                dmax = std::max(dmax, amrex::get<1>(r));
            }
        }
        ParallelDescriptor::ReduceRealMin(dmin);
        ParallelDescriptor::ReduceRealMax(dmax);
    }

    std::ostringstream ss;
    ss.precision(10);
    std::vector<std::string> header = a_header;
    header.emplace_back("matrix_ops: MFMatrixSolverOps (matrix-free)");
    header.emplace_back("operator: Jacobian, applied by matrix-free products");
    header.emplace_back("diagonal: 1 + pc_factor * M_diag + curlcurl_factor * CC_diag");
    header.emplace_back(std::string("mass_matrix_diagonal: ") + (has_mm ? "yes" : "no"));
    header.emplace_back(std::string("curlcurl_diagonal: ") + (has_cc ? "yes" : "no"));
    ss << pc_factor;
    header.emplace_back("pc_factor: " + ss.str());
    ss.str("");
    ss << curlcurl_factor;
    header.emplace_back("curlcurl_factor: " + ss.str());
    ss.str("");
    ss << dmin;
    header.emplace_back("diagonal_min: " + ss.str());
    ss.str("");
    ss << dmax;
    header.emplace_back("diagonal_max: " + ss.str());

    // The entries are not stored, so the file holds the metadata only
    std::string filename = a_filename;
    const std::string ext = ".mtx";
    if (filename.ends_with(ext)) {
        filename.replace(filename.size() - ext.size(), ext.size(), ".info");
    }
    MatrixMarketWriter::WriteInfo(filename, header, *dofs);
}

#endif
