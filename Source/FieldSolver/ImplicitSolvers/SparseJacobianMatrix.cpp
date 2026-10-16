/* Copyright 2025-2026 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "SparseJacobianMatrix.H"
#include "MatrixMarketWriter.H"
#include "WarpXSolverDOF.H"
#include "Utils/WarpXConst.H"
#include "Utils/TextMsg.H"

#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_GpuBuffer.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

void SparseJacobianMatrix::readParameters ()
{
    const amrex::ParmParse pp("precon_mat");
    pp.query("verbose", m_verbose);

    // Backward compatibility: pc_petsc.verbose was read by the old MatrixPC
    // class (renamed to ExternLibPC).
    const amrex::ParmParse pp_old("pc_petsc");
    bool backward_bool;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_old.query("verbose", backward_bool),
        "pc_petsc.verbose is no longer supported. "
        "Use precon_mat.verbose instead.");
}

void SparseJacobianMatrix::printParameters () const
{
    using namespace amrex;
    Print() << "precon_mat verbose:              "
            << (m_verbose ? "true" : "false") << "\n";
    Print() << "precon_mat include_mass_matrices: "
            << (m_include_mass_matrices ? "true" : "false") << "\n";
}

void SparseJacobianMatrix::Define (int a_ndofs_l, int a_ndofs_g)
{
    BL_PROFILE("SparseJacobianMatrix::Define()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !IsDefined(),
        "SparseJacobianMatrix::Define() called on defined object");

    readParameters();

    m_ndofs_l = a_ndofs_l;
    m_ndofs_g = a_ndofs_g;

    auto n_rows = static_cast<size_t>(m_ndofs_l);
    auto n_cols = static_cast<size_t>(m_pc_mat_nnz) * n_rows;

    m_r_indices_g.resize(n_rows);
    m_num_nz.resize(n_rows);
    m_c_indices_g.resize(n_cols);
    m_a_ij.resize(n_cols);

    m_is_defined = true;
}

void SparseJacobianMatrix::Update (
    const WarpXSolverDOF* a_dofs,
    const amrex::Vector<amrex::Geometry>& a_geom,
    amrex::Real a_theta_dt,
    const amrex::Vector<amrex::Array<const amrex::MultiFab*,3>>& a_bc_masks,
    const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>* a_mass_matrices,
    const amrex::Array<amrex::IntVect,3>& a_mm_ncomp)
{
    BL_PROFILE("SparseJacobianMatrix::Update()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "SparseJacobianMatrix::Update() called on undefined object");

    m_include_mass_matrices = (a_mass_matrices != nullptr);
    m_theta_dt = a_theta_dt;

    const amrex::Real alpha = (a_theta_dt * PhysConst::c) * (a_theta_dt * PhysConst::c);
    if (m_verbose) {
        amrex::Print() << "Updating precon_mat: theta*dt = " << a_theta_dt
                        << ", alpha = " << alpha << "\n";
    }

    while (true) {

        auto nnz_diff = Assemble(a_dofs, a_geom, a_theta_dt,
                                 a_bc_masks, a_mass_matrices, a_mm_ncomp);
        AMREX_ALWAYS_ASSERT(nnz_diff >= 0);
        if (nnz_diff > 0) {
            m_num_realloc++;
            amrex::Print() << "precon_mat: reallocating CSR arrays"
                           << " (nnz_max " << m_pc_mat_nnz
                           << " -> " << m_pc_mat_nnz + nnz_diff << ")\n";
            m_pc_mat_nnz += nnz_diff;
        } else {
            break;
        }
    }

    if (m_num_realloc > 1) {
        std::stringstream warning_message;
        warning_message << "Number of times arrays were reallocated due to new nonzero elements "
                        << "is greater than 1 (" << m_num_realloc << "). This is unexpected.\n";
        ablastr::warn_manager::WMRecordWarning("SparseJacobianMatrix", warning_message.str());
    }

    // The column pattern is rebuilt in the same order on every assembly, so the
    // remapped columns stay valid unless the row stride has changed.
    if (m_is_remapped) {
        if (m_pc_mat_nnz != m_remap_nnz) {
            m_is_remapped = false;
        } else {
            ExtractDiagonal();
        }
    }
}

void SparseJacobianMatrix::ExtractDiagonal ()
{
    BL_PROFILE("SparseJacobianMatrix::ExtractDiagonal()");

    m_diagonal.resize(m_ndofs_l);
    const int nnz_max = m_pc_mat_nnz;
    const auto* num_nz_ptr = m_num_nz.data();
    const auto* r_g_ptr = m_r_indices_g.data();
    const auto* c_g_ptr = m_c_indices_g.data();
    const auto* a_ij_ptr = m_a_ij.data();
    auto* diag_ptr = m_diagonal.data();

    amrex::ParallelFor(m_ndofs_l, [=] AMREX_GPU_DEVICE (int row)
    {
        auto diag = amrex::Real(1.0);
        for (int col = 0; col < num_nz_ptr[row]; col++) {
            const int idx = row * nnz_max + col;
            if (c_g_ptr[idx] == r_g_ptr[row]) {
                diag = a_ij_ptr[idx];
                break;
            }
        }
        diag_ptr[row] = diag;
    });
    amrex::Gpu::streamSynchronize();
}

void SparseJacobianMatrix::GetMatrix (
    amrex::Gpu::DeviceVector<int>& a_r_indices_g,
    amrex::Gpu::DeviceVector<int>& a_num_nz,
    amrex::Gpu::DeviceVector<int>& a_c_indices_g,
    amrex::Gpu::DeviceVector<amrex::Real>& a_a_ij,
    int& a_n,
    int& a_ncols_max) const
{
    a_n = m_ndofs_l;
    a_ncols_max = m_pc_mat_nnz;

    a_r_indices_g.resize(m_r_indices_g.size());
    amrex::Gpu::copyAsync(amrex::Gpu::deviceToDevice,
                          m_r_indices_g.begin(), m_r_indices_g.end(),
                          a_r_indices_g.begin());

    a_num_nz.resize(m_num_nz.size());
    amrex::Gpu::copyAsync(amrex::Gpu::deviceToDevice,
                          m_num_nz.begin(), m_num_nz.end(),
                          a_num_nz.begin());

    a_c_indices_g.resize(m_c_indices_g.size());
    amrex::Gpu::copyAsync(amrex::Gpu::deviceToDevice,
                          m_c_indices_g.begin(), m_c_indices_g.end(),
                          a_c_indices_g.begin());

    a_a_ij.resize(m_a_ij.size());
    amrex::Gpu::copyAsync(amrex::Gpu::deviceToDevice,
                          m_a_ij.begin(), m_a_ij.end(),
                          a_a_ij.begin());

    amrex::Gpu::streamSynchronize();
}

void SparseJacobianMatrix::PrintMatrix (const std::string& a_filename,
                                        const std::vector<std::string>& a_header,
                                        const WarpXSolverDOF& a_dofs) const
{
    BL_PROFILE("SparseJacobianMatrix::PrintMatrix()");
    using namespace amrex;

    const auto n = static_cast<std::size_t>(m_ndofs_l);
    const auto nnz_max = static_cast<std::size_t>(m_pc_mat_nnz);
    std::vector<int> r_h(n), nz_h(n), c_h(n*nnz_max);
    std::vector<Real> a_h(n*nnz_max);
    Gpu::copyAsync(Gpu::deviceToHost, m_r_indices_g.begin(), m_r_indices_g.begin() + n, r_h.begin());
    Gpu::copyAsync(Gpu::deviceToHost, m_num_nz.begin(), m_num_nz.begin() + n, nz_h.begin());
    Gpu::copyAsync(Gpu::deviceToHost, m_c_indices_g.begin(), m_c_indices_g.begin() + n*nnz_max, c_h.begin());
    Gpu::copyAsync(Gpu::deviceToHost, m_a_ij.begin(), m_a_ij.begin() + n*nnz_max, a_h.begin());
    Gpu::streamSynchronize();

    MatrixMarketWriter::Triplets local;
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t k = 0; k < static_cast<std::size_t>(nz_h[row]); ++k) {
            local.add(r_h[row], c_h[row*nnz_max + k], a_h[row*nnz_max + k]);
        }
    }

    const Real alpha = (m_theta_dt * PhysConst::c) * (m_theta_dt * PhysConst::c);
    std::ostringstream ss_theta_dt, ss_alpha;
    ss_theta_dt.precision(10);
    ss_alpha.precision(10);
    ss_theta_dt << "theta_dt: " << m_theta_dt;
    ss_alpha << "alpha: " << alpha;
    std::vector<std::string> header = a_header;
    header.emplace_back("matrix: SparseJacobianMatrix (CSR)");
    header.emplace_back(std::string("operator: A = I")
                        + (m_theta_dt > 0 ? " + curl(alpha curl)" : "")
                        + (m_include_mass_matrices ? " + M, M = PC mass matrices" : ""));
    header.emplace_back(ss_theta_dt.str());
    header.emplace_back(ss_alpha.str());
    header.emplace_back("nnz_max_per_row: " + std::to_string(m_pc_mat_nnz));

    MatrixMarketWriter::Write(a_filename, header, a_dofs, local);
}

int SparseJacobianMatrix::Assemble (
    const WarpXSolverDOF* a_dofs,
    const amrex::Vector<amrex::Geometry>& a_geom,
    amrex::Real a_theta_dt,
    const amrex::Vector<amrex::Array<const amrex::MultiFab*,3>>& a_bc_masks,
    const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>* a_mass_matrices,
    const amrex::Array<amrex::IntVect,3>& a_mm_ncomp)
{
    // Assemble the sparse matrix representation of the preconditioner
    //      A = curl (alpha * curl []) + M
    // where M is the mass matrix. The following data is set in this function:
    // - m_r_indices_g: integer array of size n with the global row indices
    // - m_num_nz:      integer array of size n with the number of non-zero elements
    //                  in each row
    // - m_c_indices_g: integer array of size n*ncmax with the global column indices
    //                  of non-zero elements in each row (row-major)
    // - m_a_ij:        real-type array of size n*ncmax with the matrix element values
    //                  (row-major format)
    // where n is the local number of rows, and ncmax is the maximum number of non-zero
    // elements per row.

    BL_PROFILE("SparseJacobianMatrix::Assemble()");
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "SparseJacobianMatrix::Assemble() called on undefined object" );

    // set the alpha coefficient for the curl-curl op
    const Real thetaDt = a_theta_dt;
    const Real alpha = (thetaDt*PhysConst::c) * (thetaDt*PhysConst::c);

    const auto& dofs_mfarrvec = a_dofs->m_array;

    m_r_indices_g.clear();
    m_num_nz.clear();
    m_c_indices_g.clear();
    m_a_ij.clear();

    auto n_rows = size_t(m_ndofs_l);
    auto n_cols = size_t(m_pc_mat_nnz) * size_t(m_ndofs_l);

    m_r_indices_g.resize(n_rows);
    m_num_nz.resize(n_rows);
    m_c_indices_g.resize(n_cols);
    m_a_ij.resize(n_cols);

    auto* r_indices_g_ptr = m_r_indices_g.data();
    auto* num_nz_ptr = m_num_nz.data();
    auto* c_indices_g_ptr = m_c_indices_g.data();
    auto* a_ij_ptr = m_a_ij.data();

    const auto nnz_max = m_pc_mat_nnz;
    auto nnz_actual = nnz_max;

    const int num_amr_levels = static_cast<int>(a_geom.size());
    for (int lev = 0; lev < num_amr_levels; lev++) {

        auto ncomp = dofs_mfarrvec[lev][0]->nComp();
        AMREX_ALWAYS_ASSERT(ncomp == 2); // local, global

        const auto& geom = a_geom[lev];
        const auto dxi = geom.InvCellSizeArray();

        Gpu::Buffer<int> nnz_actual_d({nnz_max});
        auto* nnz_actual_ptr = nnz_actual_d.data();

        for (int dir = 0; dir < 3; dir++) {

            for (amrex::MFIter mfi(*dofs_mfarrvec[lev][dir]); mfi.isValid(); ++mfi) {

                const amrex::Box bx = mfi.tilebox();
                const amrex::Box full_bx = mfi.fabbox();

                auto dof_arr = dofs_mfarrvec[lev][dir]->const_array(mfi);

                // Set row indices and identity diagonal (unconditional).
                // amrex::For: iterations share the nnz_actual overflow counter
                // (no SIMD pragma, see issue #7097)
                For(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    const int ridx_l = dof_arr(i,j,k,0);
                    if (ridx_l < 0) { return; }

                    int icol = 0;
                    const int ridx_g = dof_arr(i,j,k,1);

                    r_indices_g_ptr[ridx_l] = ridx_g;

                    {
                        const int cidx_g_lhs = dof_arr(i,j,k,1);
                        const amrex::Real val = 1.0_rt;
                        auto flag = SparseJacobianMatrixUtil::insertOrAdd( cidx_g_lhs, val,
                                                                &c_indices_g_ptr[ridx_l*nnz_max],
                                                                &a_ij_ptr[ridx_l*nnz_max],
                                                                nnz_max, icol );
                        if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                    }

                    num_nz_ptr[ridx_l] = icol;
                });

                // Add the curl-curl stencil entries (only when alpha > 0)
                if (thetaDt > 0.0) {

#if defined(WARPX_DIM_RSPHERE)
                    // 1D spherical geometry is electrostatic
                    amrex::ignore_unused(dxi);
#else
                    const amrex::MultiFab* BC_mask_Edir = a_bc_masks[lev][dir];
                    AMREX_ALWAYS_ASSERT(BC_mask_Edir != nullptr);
                    const auto BC_mask_Edir_arr = BC_mask_Edir->const_array(mfi);
#endif

#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                    int tdir = -1;
                    if  (dir == 0) { tdir = 2; }
                    else if (dir == 2) { tdir = 0; }
                    else { tdir = 1; }
                    auto dof_tdir_arr = dofs_mfarrvec[lev][tdir]->const_array(mfi);
#elif defined(WARPX_DIM_3D)
                    const int tdir1 = (dir + 1) % 3;
                    const int tdir2 = (dir + 2) % 3;
                    GpuArray<Array4<const int>, AMREX_SPACEDIM>
                        const dof_arrays {{ AMREX_D_DECL( dofs_mfarrvec[lev][dir]->const_array(mfi),
                                                          dofs_mfarrvec[lev][tdir1]->const_array(mfi),
                                                          dofs_mfarrvec[lev][tdir2]->const_array(mfi) ) }};
#endif

                    // amrex::For: iterations share the nnz_actual overflow counter
                    // (no SIMD pragma, see issue #7097)
                    For(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        const int ridx_l = dof_arr(i,j,k,0);
                        if (ridx_l < 0) { return; }

                        int icol = num_nz_ptr[ridx_l]; //NOLINT (misc-const-correctness)

#if defined(WARPX_DIM_1D_Z)
                        // dir = 0: xhat\cdot[\nabla\times\nabla E] = -[d2/dx2]Ex
                        // dir = 1: yhat\cdot[\nabla\times\nabla E] = -[d2/dx2]Ey
                        // dir = 2: zhat\cdot[\nabla\times\nabla E] = 0
                        if (dir != 2) {
                            {
                                // diagonal component (i) of 2nd derivative operator
                                const int cidx_g_rhs = dof_arr(i,j,k,1);
                                const amrex::Real val = 2.0_rt*alpha * dxi[0]*dxi[0] * BC_mask_Edir_arr(i,j,k,0);
                                const auto flag = SparseJacobianMatrixUtil::insertOrAdd( cidx_g_rhs, val,
                                                                        &c_indices_g_ptr[ridx_l*nnz_max],
                                                                        &a_ij_ptr[ridx_l*nnz_max],
                                                                        nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            {
                                // left component (i-1) of 2nd derivative operator
                                const int cidx_g_rhs = dof_arr(i-1,j,k,1);
                                const amrex::Real val = -alpha * dxi[0]*dxi[0] * BC_mask_Edir_arr(i,j,k,1);
                                const auto flag = SparseJacobianMatrixUtil::insertOrAdd( cidx_g_rhs, val,
                                                                        &c_indices_g_ptr[ridx_l*nnz_max],
                                                                        &a_ij_ptr[ridx_l*nnz_max],
                                                                        nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            {
                                // right component (i+1) of 2nd derivative operator
                                const int cidx_g_rhs = dof_arr(i+1,j,k,1);
                                const amrex::Real val = -alpha * dxi[0]*dxi[0] * BC_mask_Edir_arr(i,j,k,1);
                                const auto flag = SparseJacobianMatrixUtil::insertOrAdd( cidx_g_rhs, val,
                                                                        &c_indices_g_ptr[ridx_l*nnz_max],
                                                                        &a_ij_ptr[ridx_l*nnz_max],
                                                                        nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                        }
#elif defined(WARPX_DIM_RCYLINDER)
                        // dir = 0: rhat\cdot[\nabla\times\nabla E] = 0
                        // dir = 1: that\cdot[\nabla\times\nabla E] = -d/dr[1/r*d/dr(r*Et)]
                        // dir = 2: zhat\cdot[\nabla\times\nabla E] = -1/r*d/dr[r*dEz/dr]
                        if (dir != 0) {
                            const auto i_real = static_cast<amrex::Real>(i);
                            {
                                // diagonal component (i) of 2nd derivative operator
                                int cidx_g_rhs = dof_arr(i,j,k,1);
                                amrex::Real geom_factor = 1.0_rt;
                                if (dir == 1) {
                                    // r_{i} / r_{i-1/2} + r_{i} / r_{i+1/2}
                                    geom_factor = i_real / (i_real - 0.5_rt) + i_real / (i_real + 0.5_rt);
                                }
                                else if (dir == 2) {
                                    geom_factor = 2.0_rt; // (r_{i+1/2} + r_{i-1/2}) / r_{i}
                                }
                                amrex::Real val = geom_factor * alpha * dxi[0]*dxi[0] * BC_mask_Edir_arr(i,j,k,0);
                                auto flag = SparseJacobianMatrixUtil::insertOrAdd( cidx_g_rhs, val,
                                                                        &c_indices_g_ptr[ridx_l*nnz_max],
                                                                        &a_ij_ptr[ridx_l*nnz_max],
                                                                        nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            {
                                // left component (i-1) of 2nd derivative operator
                                int cidx_g_rhs = dof_arr(i-1,j,k,1);
                                amrex::Real geom_factor = 1.0_rt;
                                if (dir == 1) {
                                    geom_factor = (i_real - 1.0_rt) / (i_real - 0.5_rt); // r_{i-1} / r_{i-1/2}
                                }
                                else if (dir == 2 && i != 0) {
                                    geom_factor = 1.0_rt - 0.5_rt / i_real; // r_{i-1/2} / r_{i}
                                }
                                amrex::Real val = -geom_factor * alpha * dxi[0]*dxi[0] * BC_mask_Edir_arr(i,j,k,1);
                                auto flag = SparseJacobianMatrixUtil::insertOrAdd( cidx_g_rhs, val,
                                                                        &c_indices_g_ptr[ridx_l*nnz_max],
                                                                        &a_ij_ptr[ridx_l*nnz_max],
                                                                        nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            {
                                // right component (i+1) of 2nd derivative operator
                                int cidx_g_rhs = dof_arr(i+1,j,k,1);
                                amrex::Real geom_factor = 1.0_rt;
                                if (dir == 1) {
                                    geom_factor = (i_real + 1.0_rt) / (i_real + 0.5_rt); // r_{i+1} / r_{i+1/2}
                                }
                                else if (dir == 2 && i != 0) {
                                    geom_factor = 1.0_rt + 0.5_rt / i_real; // r_{i+1/2} / r_{i}
                                }
                                amrex::Real val = -geom_factor * alpha * dxi[0]*dxi[0] * BC_mask_Edir_arr(i,j,k,1);
                                auto flag = SparseJacobianMatrixUtil::insertOrAdd( cidx_g_rhs, val,
                                                                        &c_indices_g_ptr[ridx_l*nnz_max],
                                                                        &a_ij_ptr[ridx_l*nnz_max],
                                                                        nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                        }
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                        // DIM_XZ:
                        // dir = 0: xhat\cdot[\nabla\times\nabla E] = d/dx[dEz/dz] - [d2/dz2]Ex
                        // dir = 1: yhat\cdot[\nabla\times\nabla E] = -[d2/dx2 + d2/dz2]Ey
                        // dir = 2: zhat\cdot[\nabla\times\nabla E] = d/dz[dEx/dx] - [d2/dx2]Ez
                        // DIM_RZ:
                        // dir = 0: rhat\cdot[\nabla\times\nabla E] = d/dr[dEz/dz] - [d2/dz2]Er
                        // dir = 1: that\cdot[\nabla\times\nabla E] = -[d2/dz2]Et - d/dr[1/r*d/dr(r*Et)]
                        // dir = 2: zhat\cdot[\nabla\times\nabla E] = 1/r*d/dr[r*dEr/dz] - 1/r*d/dr[r*dEz/dr]
#if defined(WARPX_DIM_RZ)
                        const auto i_real = static_cast<amrex::Real>(i);
#endif
                        {
                            // diagonal component (i,j) of second derivative operator
                            const int cidx_g_rhs = dof_arr(i,j,k,1);
                            amrex::Real val = 0.0_rt;
                            if (dir == 0) {
                                val = 2.0_rt * alpha * dxi[1]*dxi[1] * BC_mask_Edir_arr(i,j,k,0);
                            } else if (dir == 2) {
                                val = 2.0_rt * alpha * dxi[0]*dxi[0] * BC_mask_Edir_arr(i,j,k,0);
                            } else if (dir == 1) {
#if defined(WARPX_DIM_RZ)
                                const amrex::Real geom_factor = i_real / (i_real - 0.5_rt) + i_real / (i_real + 0.5_rt);
#else
                                const amrex::Real geom_factor = 2.0_rt;
#endif
                                val = geom_factor * alpha * dxi[0]*dxi[0] * BC_mask_Edir_arr(i,j,k,0)
                                    +      2.0_rt * alpha * dxi[1]*dxi[1] * BC_mask_Edir_arr(i,j,k,2);
                            }
                            auto flag = SparseJacobianMatrixUtil::insertOrAdd( cidx_g_rhs, val,
                                                                    &c_indices_g_ptr[ridx_l*nnz_max],
                                                                    &a_ij_ptr[ridx_l*nnz_max],
                                                                    nnz_max, icol );
                            if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                        }
                        if ((dir == 0) || (dir == 2)) {
                            {
                                // left component (i-1,j) or (i,j-1) of second derivative operator
#if defined(WARPX_DIM_RZ)
                                const amrex::Real geom_factor = (dir == 2 && i != 0 ? 1.0_rt - 0.5_rt / i_real : 1.0_rt);
#else
                                const amrex::Real geom_factor = 1.0_rt;
#endif
                                const int cidx_g_rhs = (dir == 0 ? dof_arr(i,j-1,k,1) : dof_arr(i-1,j,k,1));
                                const amrex::Real val = -geom_factor * alpha * dxi[dir==0?1:0] * dxi[dir==0?1:0] * BC_mask_Edir_arr(i,j,k,1);
                                const auto flag = SparseJacobianMatrixUtil::insertOrAdd( cidx_g_rhs, val,
                                                                              &c_indices_g_ptr[ridx_l*nnz_max],
                                                                              &a_ij_ptr[ridx_l*nnz_max],
                                                                              nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            {
                                // right component (i+1,j) or (i,j+1) of second derivative operator
#if defined(WARPX_DIM_RZ)
                                const amrex::Real geom_factor = (dir == 2 && i != 0 ? 1.0_rt + 0.5_rt / i_real : 1.0_rt);
#else
                                const amrex::Real geom_factor = 1.0_rt;
#endif
                                const int cidx_g_rhs = (dir == 0 ? dof_arr(i,j+1,k,1) : dof_arr(i+1,j,k,1));
                                const amrex::Real val = -geom_factor * alpha * dxi[dir==0?1:0] * dxi[dir==0?1:0] * BC_mask_Edir_arr(i,j,k,1);
                                const auto flag = SparseJacobianMatrixUtil::insertOrAdd( cidx_g_rhs, val,
                                                                              &c_indices_g_ptr[ridx_l*nnz_max],
                                                                              &a_ij_ptr[ridx_l*nnz_max],
                                                                              nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            // The following four blocks are for the mixed derivative terms of the curl curl operator
                            // DIM_XZ:
                            // dir = 0: d/dx(dEz/dz) at Ex(i,j) with Ex centered in x and nodal in z
                            // dir = 2: d/dz(dEx/dx) at Ez(i,j) with Ez centered in z and nodal in x
                            // DIM_RZ:
                            // dir = 0: d/dr(dEz/dz) at Er(i,j) with Er centered in r and nodal in z
                            // dir = 2: 1/r*d/dr[r*dEr/dz] at Ez(i,j) with Ez centered in z and nodal in r
#if defined(WARPX_DIM_RZ)
                            const amrex::Real geom_m = (dir == 2 && i != 0 ? 1.0_rt - 0.5_rt / i_real : 1.0_rt);
                            const amrex::Real geom_p = (dir == 2 && i != 0 ? 1.0_rt + 0.5_rt / i_real : 1.0_rt);
#else
                            const amrex::Real geom_m = 1.0_rt;
                            const amrex::Real geom_p = 1.0_rt;
#endif
                            {
                                const int cidx_g_rhs = (dir == 0 ? dof_tdir_arr(i,j-1,k,1) : dof_tdir_arr(i-1,j,k,1));
                                const amrex::Real val = geom_m * alpha * dxi[0] * dxi[1] * BC_mask_Edir_arr(i,j,k,2);
                                const auto flag = SparseJacobianMatrixUtil::insertOrAdd( cidx_g_rhs, val,
                                                                              &c_indices_g_ptr[ridx_l*nnz_max],
                                                                              &a_ij_ptr[ridx_l*nnz_max],
                                                                              nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            {
                                const int cidx_g_rhs = dof_tdir_arr(i,j,k,1);
                                const amrex::Real val = -geom_p * alpha * dxi[0] * dxi[1] * BC_mask_Edir_arr(i,j,k,2);
                                const auto flag = SparseJacobianMatrixUtil::insertOrAdd( cidx_g_rhs, val,
                                                                              &c_indices_g_ptr[ridx_l*nnz_max],
                                                                              &a_ij_ptr[ridx_l*nnz_max],
                                                                              nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            {
                                const int cidx_g_rhs = (dir == 0 ? dof_tdir_arr(i+1,j-1,k,1) : dof_tdir_arr(i-1,j+1,k,1));
                                const amrex::Real val = -geom_m * alpha * dxi[0] * dxi[1] * BC_mask_Edir_arr(i,j,k,2);
                                const auto flag = SparseJacobianMatrixUtil::insertOrAdd( cidx_g_rhs, val,
                                                                              &c_indices_g_ptr[ridx_l*nnz_max],
                                                                              &a_ij_ptr[ridx_l*nnz_max],
                                                                              nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            {
                                const int cidx_g_rhs = (dir == 0 ? dof_tdir_arr(i+1,j,k,1) : dof_tdir_arr(i,j+1,k,1));
                                const amrex::Real val = geom_p * alpha * dxi[0] * dxi[1] * BC_mask_Edir_arr(i,j,k,2);
                                const auto flag = SparseJacobianMatrixUtil::insertOrAdd( cidx_g_rhs, val,
                                                                              &c_indices_g_ptr[ridx_l*nnz_max],
                                                                              &a_ij_ptr[ridx_l*nnz_max],
                                                                              nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                        } else if (dir==1) {
                            for (int jdir = 0; jdir <= 2; jdir+=2) {
                                {
#if defined(WARPX_DIM_RZ)
                                    const amrex::Real geom_factor = (jdir == 0 ? (i_real - 1.0_rt) / (i_real - 0.5_rt) : 1.0_rt);
#else
                                    const amrex::Real geom_factor = 1.0_rt;
#endif
                                    const int cidx_g_rhs = (jdir == 0 ? dof_arr(i-1,j,k,1) : dof_arr(i,j-1,k,1));
                                    const amrex::Real val = -geom_factor * alpha * dxi[jdir==0?0:1] * dxi[jdir==0?0:1] * BC_mask_Edir_arr(i,j,k,jdir+1);
                                    const auto flag = SparseJacobianMatrixUtil::insertOrAdd( cidx_g_rhs, val,
                                                                                  &c_indices_g_ptr[ridx_l*nnz_max],
                                                                                  &a_ij_ptr[ridx_l*nnz_max],
                                                                                  nnz_max, icol );
                                    if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                                }
                                {
#if defined(WARPX_DIM_RZ)
                                    const amrex::Real geom_factor = (jdir == 0 ? (i_real + 1.0_rt) / (i_real + 0.5_rt) : 1.0_rt);
#else
                                    const amrex::Real geom_factor = 1.0_rt;
#endif
                                    const int cidx_g_rhs = (jdir == 0 ? dof_arr(i+1,j,k,1) : dof_arr(i,j+1,k,1));
                                    const amrex::Real val = -geom_factor * alpha * dxi[jdir==0?0:1] * dxi[jdir==0?0:1] * BC_mask_Edir_arr(i,j,k,jdir+1);
                                    const auto flag = SparseJacobianMatrixUtil::insertOrAdd( cidx_g_rhs, val,
                                                                                  &c_indices_g_ptr[ridx_l*nnz_max],
                                                                                  &a_ij_ptr[ridx_l*nnz_max],
                                                                                  nnz_max, icol );
                                    if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                                }
                            }
                        }
#elif defined(WARPX_DIM_3D)
                        // xhat\cdot[\nabla\times\nabla E] = d/dx[dEy/dy + dEz/dz] - [d2/dy2 + d2/dz2]Ex
                        // yhat\cdot[\nabla\times\nabla E] = d/dy[dEx/dx + dEz/dz] - [d2/dx2 + d2/dz2]Ey
                        // zhat\cdot[\nabla\times\nabla E] = d/dz[dEx/dx + dEy/dy] - [d2/dx2 + d2/dy2]Ez
                        amrex::IntVect dvec(AMREX_D_DECL(dir,tdir1,tdir2));
                        const amrex::IntVect ic(AMREX_D_DECL(i,j,k));
                        {
                            // diagonal component (i,j,k) of second derivative operator
                            const int cidx_g_rhs = dof_arrays[0](ic,1);
                            const amrex::Real val = 2.0_rt * alpha * ( dxi[dvec[1]]*dxi[dvec[1]] * BC_mask_Edir_arr(i,j,k,0)
                                                               + dxi[dvec[2]]*dxi[dvec[2]] * BC_mask_Edir_arr(i,j,k,3) );
                            const auto flag = SparseJacobianMatrixUtil::insertOrAdd( cidx_g_rhs, val,
                                                                    &c_indices_g_ptr[ridx_l*nnz_max],
                                                                    &a_ij_ptr[ridx_l*nnz_max],
                                                                    nnz_max, icol );
                            if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                        }
                        // left and right components (i+/-1,j,k), (i,j+/-1,k), or (i,j,k+/-1) of second derivative operator
                        for (int ctr = -1; ctr <= 1; ctr += 2) {
                            for (int tdir = 1; tdir <= 2; tdir++) {
                                auto iv = ic; iv[dvec[tdir]] += ctr;
                                const int cidx_g_rhs = dof_arrays[0](iv,1);
                                const int comp_shift = (dvec[tdir] == tdir1) ? 0:3;
                                const amrex::Real val = -alpha * dxi[dvec[tdir]]*dxi[dvec[tdir]] * BC_mask_Edir_arr(i,j,k,comp_shift+1);
                                const auto flag = SparseJacobianMatrixUtil::insertOrAdd( cidx_g_rhs, val,
                                                                        &c_indices_g_ptr[ridx_l*nnz_max],
                                                                        &a_ij_ptr[ridx_l*nnz_max],
                                                                        nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                        }
                        // mixed derivative terms of the curl curl operator
                        for (int ctr_dir = -1; ctr_dir <= 0; ctr_dir++) {
                            for (int ctr_tdir = -1; ctr_tdir <= 0; ctr_tdir++) {
                                for (int tdir = 1; tdir <= 2; tdir++) {
                                    auto iv = ic; iv[dvec[0]] += (ctr_dir+1); iv[dvec[tdir]] += ctr_tdir;
                                    const auto sign = std::copysign(1,ctr_dir) * std::copysign(1,ctr_tdir);
                                    const int cidx_g_rhs = dof_arrays[tdir](iv,1);
                                    const int comp_shift = (dvec[tdir] == tdir1) ? 0:3;
                                    const amrex::Real val = amrex::Real(sign) * alpha * dxi[dvec[0]]*dxi[dvec[tdir]] * BC_mask_Edir_arr(i,j,k,comp_shift+2);
                                    const auto flag = SparseJacobianMatrixUtil::insertOrAdd( cidx_g_rhs, val,
                                                                            &c_indices_g_ptr[ridx_l*nnz_max],
                                                                            &a_ij_ptr[ridx_l*nnz_max],
                                                                            nnz_max, icol );
                                    if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                                }
                            }
                        }
#endif
                        num_nz_ptr[ridx_l] = icol;
                    });
                }

                // Add the mass matrix piece
                // See Figure B.9 of JCP 491, 112383 (2023) for an illustrative diagram
                // of the mass matrices (https://doi.org/10.1016/j.jcp.2023.112383).
                //
                // The coupling of Jx(i,j,k) to Ex(i+i0,j+j0,k+k0), where i0 ranges from
                // -MM_width[0] to +MM_width[0], j0 ranges from -MM_width[1] to +MM_width[1], and k0 ranges
                // from -MM_width[2] to +MM_width[2], is stored as components of a_mass_matrices[dir=0].
                // Similarly for Jy/Ey (a_mass_matrices[dir=1]) and Jz/Ez (a_mass_matrices[dir=2]).
                // The mapping to the components is given by:
                // mm_comp = i0 + MM_width[0] + MM_comp[0]*(j0 + MM_width[1]) + MM_comp[0]*MM_comp[1]*(k0 + MM_width[2])
                if (a_mass_matrices != nullptr) {

                    auto sigma_ii_arr = (*a_mass_matrices)[lev][dir]->const_array(mfi);
                    amrex::GpuArray<int,3> MM_ncomp = {1,1,1};
                    amrex::GpuArray<int,3> MM_width = {0,0,0};
                    for (int space_dir=0; space_dir<AMREX_SPACEDIM; space_dir++) {
                        MM_ncomp[space_dir] = a_mm_ncomp[dir][space_dir];
                        MM_width[space_dir] = (MM_ncomp[space_dir] - 1)/2;
                    }

                    // amrex::For: iterations share the nnz_actual overflow counter
                    // (no SIMD pragma, see issue #7097)
                    For(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        const int ridx_l = dof_arr(i,j,k,0);
                        if (ridx_l < 0) { return; }

                        const amrex::IntVect iv_base = IntVect(AMREX_D_DECL(i,j,k));
                        int icol = num_nz_ptr[ridx_l];

                        int mm_comp = 0;
                        for (int comp2 = 0; comp2 < MM_ncomp[2]; comp2++) {
                            [[maybe_unused]] const int kk0 = comp2 - MM_width[2];
                            for (int comp1 = 0; comp1 < MM_ncomp[1]; comp1++) {
                                [[maybe_unused]] const int jj0 = comp1 - MM_width[1];
                                for (int comp0 = 0; comp0 < MM_ncomp[0]; comp0++) {
                                    const int ii0 = comp0 - MM_width[0]; // ..., -2, -1, 0, 1, 2, ...
                                    const amrex::IntVect iv_shift = IntVect(AMREX_D_DECL(ii0, jj0, kk0));
                                    if (full_bx.contains(iv_base + iv_shift)) {
                                        const int cidx_g_rhs = dof_arr(iv_base + iv_shift,1);
                                        const amrex::Real val = sigma_ii_arr(iv_base,mm_comp);
                                        auto flag = SparseJacobianMatrixUtil::insertOrAdd( cidx_g_rhs, val,
                                                                                &c_indices_g_ptr[ridx_l*nnz_max],
                                                                                &a_ij_ptr[ridx_l*nnz_max],
                                                                                nnz_max, icol );
                                        if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                                    }
                                    ++mm_comp;
                                }
                            }
                        }

                        num_nz_ptr[ridx_l] = icol;
                    });
                }
                Gpu::synchronize();
            }
        }

        nnz_actual = std::max(nnz_actual, *(nnz_actual_d.copyToHost()));
    }

    amrex::ParallelDescriptor::ReduceIntMax(&nnz_actual, 1);
    return (nnz_actual - nnz_max);
}

void SparseJacobianMatrix::RemapColumns (const WarpXSolverDOF* a_dofs)
{
    BL_PROFILE("SparseJacobianMatrix::RemapColumns()");
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "SparseJacobianMatrix::RemapColumns() called on undefined object");

    const auto& dofs_mfarrvec = a_dofs->m_array;
    const int num_amr_levels = static_cast<int>(dofs_mfarrvec.size());

    // Step 1: Build global-to-local-extended map on host from the owned + ghost
    // cells of the DOF iMultiFab. Owned DOFs (local id >= 0) keep their local id,
    // in [0, m_ndofs_l). All other DOFs, including valid shared nodes owned by
    // another box, get ghost indices from m_ndofs_l, after all owned DOFs are known.

    std::vector<Gpu::PinnedVector<int>> ldofs_h;
    std::vector<Gpu::PinnedVector<int>> gdofs_h;

    for (int lev = 0; lev < num_amr_levels; lev++) {
        for (int dir = 0; dir < 3; dir++) {
            const auto& dof_fab = *(dofs_mfarrvec[lev][dir]);
            for (MFIter mfi(dof_fab); mfi.isValid(); ++mfi) {
                const Box gbx = mfi.growntilebox();

                // Use a GPU kernel to read DOF data via const_array() — the
                // same access path used by Assemble — then copy to host.
                // Direct dataPtr() device-to-host copies returned stale data
                // on some GPU platforms.
                const auto npts = static_cast<int>(gbx.numPts());
                Gpu::DeviceVector<int> ldof_d(npts), gdof_d(npts);
                auto* ldof_ptr = ldof_d.data();
                auto* gdof_ptr = gdof_d.data();
                auto dof_arr = dof_fab.const_array(mfi);
                const auto gbx_lo = lbound(gbx);
                const auto gbx_len = length(gbx);
                ParallelFor(npts, [=] AMREX_GPU_DEVICE (int idx)
                {
                    int k = idx / (gbx_len.x * gbx_len.y);
                    int j = (idx - k * gbx_len.x * gbx_len.y) / gbx_len.x;
                    int i = idx - k * gbx_len.x * gbx_len.y - j * gbx_len.x;
                    i += gbx_lo.x; j += gbx_lo.y; k += gbx_lo.z;
                    ldof_ptr[idx] = dof_arr(i, j, k, 0);
                    gdof_ptr[idx] = dof_arr(i, j, k, 1);
                });
                Gpu::streamSynchronize();

                ldofs_h.emplace_back(npts);
                gdofs_h.emplace_back(npts);
                Gpu::copyAsync(Gpu::deviceToHost, ldof_d.begin(), ldof_d.end(),
                               ldofs_h.back().begin());
                Gpu::copyAsync(Gpu::deviceToHost, gdof_d.begin(), gdof_d.end(),
                               gdofs_h.back().begin());
                Gpu::streamSynchronize();
            }
        }
    }

    std::unordered_map<int, int> global_to_ext;
    for (std::size_t ibox = 0; ibox < ldofs_h.size(); ++ibox) {
        for (std::size_t idx = 0; idx < ldofs_h[ibox].size(); ++idx) {
            const int ldof = ldofs_h[ibox][idx];
            if (ldof >= 0) { global_to_ext[gdofs_h[ibox][idx]] = ldof; }
        }
    }
    int next_ghost_idx = m_ndofs_l;
    for (std::size_t ibox = 0; ibox < ldofs_h.size(); ++ibox) {
        for (std::size_t idx = 0; idx < ldofs_h[ibox].size(); ++idx) {
            const int gdof = gdofs_h[ibox][idx];
            if (gdof >= 0 && !global_to_ext.contains(gdof)) {
                global_to_ext[gdof] = next_ghost_idx;
                next_ghost_idx++;
            }
        }
    }

    m_ndofs_ext = next_ghost_idx;

    // Step 2: Remap CSR column indices from global to local-extended.
    // Copy global columns to host, remap, copy back to device.
    const auto n_entries = static_cast<int>(m_c_indices_g.size());
    Gpu::PinnedVector<int> c_global_h(n_entries);
    Gpu::copyAsync(Gpu::deviceToHost, m_c_indices_g.begin(), m_c_indices_g.end(),
                   c_global_h.begin());

    Gpu::PinnedVector<int> num_nz_h(m_ndofs_l);
    Gpu::copyAsync(Gpu::deviceToHost, m_num_nz.begin(), m_num_nz.end(),
                   num_nz_h.begin());
    Gpu::streamSynchronize();

    Gpu::PinnedVector<int> c_ext_h(n_entries, -1);
    const int nnz_max = m_pc_mat_nnz;

    for (int row = 0; row < m_ndofs_l; row++) {
        for (int col = 0; col < num_nz_h[row]; col++) {
            const int idx = row * nnz_max + col;
            const int gcol = c_global_h[idx];
            if (gcol < 0) { continue; }
            auto it = global_to_ext.find(gcol);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                it != global_to_ext.end(),
                "SparseJacobianMatrix::RemapColumns(): column global DOF "
                + std::to_string(gcol) + " not found in owned or ghost cells");
            c_ext_h[idx] = it->second;
        }
    }

    m_c_indices_ext.resize(n_entries);
    Gpu::copyAsync(Gpu::hostToDevice, c_ext_h.begin(), c_ext_h.end(),
                   m_c_indices_ext.begin());

    // Step 3: Extract diagonal
    ExtractDiagonal();

    // Step 4: Store the local-extended index of every DOF cell, ghost cells
    // included, for BuildExtendedDOFVector. Owned DOFs are taken only from their
    // owned cell; cells without a DOF, or with an owned DOF, hold -1.
    m_ext_index.resize(num_amr_levels);
    std::size_t ibox = 0;
    for (int lev = 0; lev < num_amr_levels; lev++) {
        for (int dir = 0; dir < 3; dir++) {
            const auto& dof_fab = *(dofs_mfarrvec[lev][dir]);
            m_ext_index[lev][dir] = std::make_unique<iMultiFab>(
                dof_fab.boxArray(), dof_fab.DistributionMap(), 1, dof_fab.nGrowVect());
            for (MFIter mfi(dof_fab); mfi.isValid(); ++mfi, ++ibox) {
                const Box gbx = mfi.growntilebox();
                const auto& ldof_h = ldofs_h[ibox];
                const auto& gdof_h = gdofs_h[ibox];
                const auto npts = static_cast<int>(gbx.numPts());
                Gpu::PinnedVector<int> ext_h(npts);
                for (int idx = 0; idx < npts; ++idx) {
                    const int gdof = gdof_h[idx];
                    if (ldof_h[idx] >= 0) {
                        ext_h[idx] = ldof_h[idx];
                    } else if (gdof >= 0 && global_to_ext.at(gdof) >= m_ndofs_l) {
                        ext_h[idx] = global_to_ext.at(gdof);
                    } else {
                        ext_h[idx] = -1;
                    }
                }
                Gpu::DeviceVector<int> ext_d(npts);
                Gpu::copyAsync(Gpu::hostToDevice, ext_h.begin(), ext_h.end(), ext_d.begin());
                Gpu::streamSynchronize();

                const auto* ext_ptr = ext_d.data();
                auto ext_arr = m_ext_index[lev][dir]->array(mfi);
                const auto gbx_lo = lbound(gbx);
                const auto gbx_len = length(gbx);
                ParallelFor(npts, [=] AMREX_GPU_DEVICE (int idx)
                {
                    int k = idx / (gbx_len.x * gbx_len.y);
                    int j = (idx - k * gbx_len.x * gbx_len.y) / gbx_len.x;
                    int i = idx - k * gbx_len.x * gbx_len.y - j * gbx_len.x;
                    i += gbx_lo.x; j += gbx_lo.y; k += gbx_lo.z;
                    ext_arr(i, j, k) = ext_ptr[idx];
                });
                Gpu::streamSynchronize();
            }
        }
    }

    m_is_remapped = true;
    m_remap_nnz = m_pc_mat_nnz;

    if (m_verbose) {
        Print() << "SparseJacobianMatrix::RemapColumns(): "
                << m_ndofs_l << " local DOFs, "
                << (m_ndofs_ext - m_ndofs_l) << " ghost DOFs, "
                << m_ndofs_ext << " total extended DOFs\n";
    }
}

void SparseJacobianMatrix::MatVecMult (amrex::Real* a_y,
                                        const amrex::Real* a_x) const
{
    BL_PROFILE("SparseJacobianMatrix::MatVecMult()");
    using namespace amrex;

    AMREX_ALWAYS_ASSERT(m_is_remapped);

    const int nrows = m_ndofs_l;
    const int nnz_max = m_pc_mat_nnz;
    const auto* num_nz_ptr = m_num_nz.data();
    const auto* c_ext_ptr = m_c_indices_ext.data();
    const auto* a_ij_ptr = m_a_ij.data();

    ParallelFor(nrows, [=] AMREX_GPU_DEVICE (int row)
    {
        Real dot = Real(0.0);
        const int ncols = num_nz_ptr[row];
        for (int col = 0; col < ncols; col++) {
            const int idx = row * nnz_max + col;
            const int j_ext = c_ext_ptr[idx];
            if (j_ext >= 0) {
                dot += a_ij_ptr[idx] * a_x[j_ext];
            }
        }
        a_y[row] = dot;
    });
    Gpu::streamSynchronize();
}

void SparseJacobianMatrix::MatJacobiSweep (amrex::Real* a_x_local,
                                            const amrex::Real* a_x_ext,
                                            const amrex::Real* a_b,
                                            amrex::Real a_omega) const
{
    BL_PROFILE("SparseJacobianMatrix::MatJacobiSweep()");
    using namespace amrex;

    AMREX_ALWAYS_ASSERT(m_is_remapped);

    const int nrows = m_ndofs_l;
    const int nnz_max = m_pc_mat_nnz;
    const auto* num_nz_ptr = m_num_nz.data();
    const auto* c_ext_ptr = m_c_indices_ext.data();
    const auto* a_ij_ptr = m_a_ij.data();
    const auto* diag_ptr = m_diagonal.data();
    const Real omega = a_omega;

    ParallelFor(nrows, [=] AMREX_GPU_DEVICE (int row)
    {
        // Compute A*x for this row
        Real Ax = Real(0.0);
        const int ncols = num_nz_ptr[row];
        for (int col = 0; col < ncols; col++) {
            const int idx = row * nnz_max + col;
            const int j_ext = c_ext_ptr[idx];
            if (j_ext >= 0) {
                Ax += a_ij_ptr[idx] * a_x_ext[j_ext];
            }
        }
        // Jacobi update: x += omega * (b - Ax) / diag
        a_x_local[row] += omega * (a_b[row] - Ax) / diag_ptr[row];
    });
    Gpu::streamSynchronize();
}

amrex::Real SparseJacobianMatrix::MatResidualNorm (const amrex::Real* a_x_ext,
                                                    const amrex::Real* a_b) const
{
    BL_PROFILE("SparseJacobianMatrix::MatResidualNorm()");
    using namespace amrex;

    AMREX_ALWAYS_ASSERT(m_is_remapped);

    const int nrows = m_ndofs_l;
    const int nnz_max = m_pc_mat_nnz;
    const auto* num_nz_ptr = m_num_nz.data();
    const auto* c_ext_ptr = m_c_indices_ext.data();
    const auto* a_ij_ptr = m_a_ij.data();

    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    reduce_op.eval(nrows, reduce_data,
        [=] AMREX_GPU_DEVICE (int row) -> ReduceTuple
        {
            Real Ax = Real(0.0);
            const int ncols = num_nz_ptr[row];
            for (int col = 0; col < ncols; col++) {
                const int idx = row * nnz_max + col;
                const int j_ext = c_ext_ptr[idx];
                if (j_ext >= 0) {
                    Ax += a_ij_ptr[idx] * a_x_ext[j_ext];
                }
            }
            const Real r = a_b[row] - Ax;
            return {r * r};
        });

    return amrex::get<0>(reduce_data.value(reduce_op));
}

void SparseJacobianMatrix::MatInitialGuess (amrex::Real* a_x,
                                             const amrex::Real* a_b) const
{
    BL_PROFILE("SparseJacobianMatrix::MatInitialGuess()");
    using namespace amrex;

    AMREX_ALWAYS_ASSERT(m_is_remapped);

    const int nrows = m_ndofs_l;
    const auto* diag_ptr = m_diagonal.data();

    ParallelFor(nrows, [=] AMREX_GPU_DEVICE (int row)
    {
        a_x[row] = a_b[row] / diag_ptr[row];
    });
    Gpu::streamSynchronize();
}

amrex::Real SparseJacobianMatrix::EstimateOmega () const
{
    BL_PROFILE("SparseJacobianMatrix::EstimateOmega()");
    using namespace amrex;

    AMREX_ALWAYS_ASSERT(m_is_remapped);

    const int nrows = m_ndofs_l;
    const int nnz = m_pc_mat_nnz;
    const auto* aij_ptr = m_a_ij.data();
    const auto* diag_ptr = m_diagonal.data();
    const auto* num_nz_ptr = m_num_nz.data();

    // Gershgorin: r_i = (sum_j |A_ij| - |A_ii|) / |A_ii|
    ReduceOps<ReduceOpMax> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    reduce_op.eval(nrows, reduce_data,
        [=] AMREX_GPU_DEVICE (int row) -> ReduceTuple
        {
            Real row_sum = Real(0.0);
            const int ncols = num_nz_ptr[row];
            for (int col = 0; col < ncols; col++) {
                row_sum += amrex::Math::abs(aij_ptr[row * nnz + col]);
            }
            const Real abs_diag = amrex::Math::abs(diag_ptr[row]);
            const Real r = (row_sum - abs_diag) / abs_diag;
            return {r};
        });

    Real max_r = amrex::get<0>(reduce_data.value(reduce_op));
    ParallelAllReduce::Max(max_r, ParallelContext::CommunicatorSub());

    return Real(2.0) / (Real(2.0) + max_r);
}
