/* Copyright 2025-2026 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "SparseJacobianMatrix.H"
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
#include <sstream>
#include <unordered_map>

void SparseJacobianMatrix::readParameters ()
{
    const amrex::ParmParse pp("sparse_jacobian_matrix");
    pp.query("verbose", m_verbose);
    pp.query("pc_diagonal_only", m_pc_diag_only);
}

void SparseJacobianMatrix::printParameters () const
{
    using namespace amrex;
    Print() << "sparse_jacobian_matrix verbose:              "
            << (m_verbose ? "true" : "false") << "\n";
    Print() << "sparse_jacobian_matrix pc_diagonal_only:     "
            << (m_pc_diag_only ? "true" : "false") << "\n";
    Print() << "sparse_jacobian_matrix include_mass_matrices: "
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

    const amrex::Real alpha = (a_theta_dt * PhysConst::c) * (a_theta_dt * PhysConst::c);
    if (m_verbose) {
        amrex::Print() << "Updating sparse_jacobian_matrix: theta*dt = " << a_theta_dt
                        << ", alpha = " << alpha << "\n";
    }

    while (true) {

        auto nnz_diff = Assemble(a_dofs, a_geom, a_theta_dt,
                                 a_bc_masks, a_mass_matrices, a_mm_ncomp);
        AMREX_ALWAYS_ASSERT(nnz_diff >= 0);
        if (nnz_diff > 0) {
            m_num_realloc++;
            amrex::Print() << "sparse_jacobian_matrix: reallocating CSR arrays"
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

int SparseJacobianMatrix::Assemble (
    const WarpXSolverDOF* a_dofs,
    const amrex::Vector<amrex::Geometry>& a_geom,
    amrex::Real a_theta_dt,
    const amrex::Vector<amrex::Array<const amrex::MultiFab*,3>>& a_bc_masks,
    const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>* a_mass_matrices,
    const amrex::Array<amrex::IntVect,3>& a_mm_ncomp)
{
    // Assemble the sparse matrix representation of the preconditioner
    //      A = I + curl (alpha * curl []) + M
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
        "SparseJacobianMatrix::Assemble() called on undefined object");

    // Get DOF arrays
    const auto& dofs_mfarrvec = a_dofs->m_array;
    AMREX_ALWAYS_ASSERT(m_ndofs_l == a_dofs->m_nDoFs_l);
    AMREX_ALWAYS_ASSERT(m_ndofs_g == a_dofs->m_nDoFs_g);

    m_r_indices_g.clear();
    m_num_nz.clear();
    m_c_indices_g.clear();
    m_a_ij.clear();

    auto n_rows = static_cast<size_t>(m_ndofs_l);
    auto n_cols = static_cast<size_t>(m_pc_mat_nnz) * n_rows;

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

            for (MFIter mfi(*dofs_mfarrvec[lev][dir]); mfi.isValid(); ++mfi) {

                const Box bx = mfi.tilebox();
                const Box full_bx = mfi.fabbox();

                auto dof_arr = dofs_mfarrvec[lev][dir]->const_array(mfi);

#if defined(WARPX_DIM_RCYLINDER)
                ignore_unused(dxi);
                WARPX_ABORT_WITH_MESSAGE(
                    "SparseJacobianMatrix::Assemble() not yet implemented for WARPX_DIM_RCYLINDER");
#elif defined(WARPX_DIM_RSPHERE)
                ignore_unused(dxi);
                WARPX_ABORT_WITH_MESSAGE(
                    "SparseJacobianMatrix::Assemble() not yet implemented for WARPX_DIM_RSPHERE");
#elif defined(WARPX_DIM_RZ)
                ignore_unused(dxi);
                WARPX_ABORT_WITH_MESSAGE(
                    "SparseJacobianMatrix::Assemble() not yet implemented for WARPX_DIM_RZ");
#endif

                // Set row indices and identity diagonal (unconditional)
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    int ridx_l = dof_arr(i,j,k,0);
                    if (ridx_l < 0) { return; }

                    int icol = 0;
                    int ridx_g = dof_arr(i,j,k,1);

                    r_indices_g_ptr[ridx_l] = ridx_g;

                    {
                        int cidx_g_lhs = dof_arr(i,j,k,1);
                        Real val = 1.0;
                        auto flag = SparseJacobianMatrixUtil::insertOrAdd(
                            cidx_g_lhs, val,
                            &c_indices_g_ptr[ridx_l*nnz_max],
                            &a_ij_ptr[ridx_l*nnz_max],
                            nnz_max, icol);
                        if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                    }

                    num_nz_ptr[ridx_l] = icol;
                });

                // Add the curl-curl stencil entries (only when alpha > 0)
                if (a_theta_dt > 0.0) {

                    const Real alpha = (a_theta_dt * PhysConst::c) * (a_theta_dt * PhysConst::c);
                    const MultiFab* BC_mask_Edir = a_bc_masks[lev][dir];
                    AMREX_ALWAYS_ASSERT(BC_mask_Edir != nullptr);
                    const auto BC_mask_Edir_arr = BC_mask_Edir->const_array(mfi);

#if defined(WARPX_DIM_XZ)
                    int tdir = -1;
                    if      (dir == 0) { tdir = 2; }
                    else if (dir == 2) { tdir = 0; }
                    else               { tdir = 1; }
                    auto dof_tdir_arr = dofs_mfarrvec[lev][tdir]->const_array(mfi);
#elif defined(WARPX_DIM_3D)
                    int tdir1 = (dir + 1) % 3;
                    int tdir2 = (dir + 2) % 3;
                    GpuArray<Array4<const int>, AMREX_SPACEDIM>
                        const dof_arrays {{ AMREX_D_DECL(
                            dofs_mfarrvec[lev][dir]->const_array(mfi),
                            dofs_mfarrvec[lev][tdir1]->const_array(mfi),
                            dofs_mfarrvec[lev][tdir2]->const_array(mfi)) }};
#elif !defined(WARPX_DIM_1D_Z)
                    ignore_unused(alpha, dxi, BC_mask_Edir_arr);
#endif

                    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        int ridx_l = dof_arr(i,j,k,0);
                        if (ridx_l < 0) { return; }

                        int icol = num_nz_ptr[ridx_l];

#if defined(WARPX_DIM_1D_Z)
                        if (dir != 2) {
                            {
                                int cidx_g_rhs = dof_arr(i,j,k,1);
                                Real val = 2.0_rt*alpha * dxi[0]*dxi[0] * BC_mask_Edir_arr(i,j,k,0);
                                auto flag = SparseJacobianMatrixUtil::insertOrAdd(
                                    cidx_g_rhs, val,
                                    &c_indices_g_ptr[ridx_l*nnz_max],
                                    &a_ij_ptr[ridx_l*nnz_max],
                                    nnz_max, icol);
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            {
                                int cidx_g_rhs = dof_arr(i-1,j,k,1);
                                Real val = -alpha * dxi[0]*dxi[0] * BC_mask_Edir_arr(i,j,k,1);
                                auto flag = SparseJacobianMatrixUtil::insertOrAdd(
                                    cidx_g_rhs, val,
                                    &c_indices_g_ptr[ridx_l*nnz_max],
                                    &a_ij_ptr[ridx_l*nnz_max],
                                    nnz_max, icol);
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            {
                                int cidx_g_rhs = dof_arr(i+1,j,k,1);
                                Real val = -alpha * dxi[0]*dxi[0] * BC_mask_Edir_arr(i,j,k,1);
                                auto flag = SparseJacobianMatrixUtil::insertOrAdd(
                                    cidx_g_rhs, val,
                                    &c_indices_g_ptr[ridx_l*nnz_max],
                                    &a_ij_ptr[ridx_l*nnz_max],
                                    nnz_max, icol);
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                        }
#elif defined(WARPX_DIM_XZ)
                        {
                            int cidx_g_rhs = dof_arr(i,j,k,1);
                            Real val = 0.0;
                            if (dir == 0) {
                                val += 2.0_rt*alpha * dxi[1]*dxi[1] * BC_mask_Edir_arr(i,j,k,0);
                            } else if (dir == 2) {
                                val += 2.0_rt*alpha * dxi[0]*dxi[0] * BC_mask_Edir_arr(i,j,k,0);
                            } else if (dir == 1) {
                                val += 2.0_rt*alpha * ( dxi[0]*dxi[0] * BC_mask_Edir_arr(i,j,k,0)
                                                      + dxi[1]*dxi[1] * BC_mask_Edir_arr(i,j,k,2) );
                            }
                            auto flag = SparseJacobianMatrixUtil::insertOrAdd(
                                cidx_g_rhs, val,
                                &c_indices_g_ptr[ridx_l*nnz_max],
                                &a_ij_ptr[ridx_l*nnz_max],
                                nnz_max, icol);
                            if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                        }
                        if ((dir == 0) || (dir == 2)) {
                            {
                                int cidx_g_rhs = (dir == 0 ? dof_arr(i,j-1,k,1) : dof_arr(i-1,j,k,1));
                                Real val = -alpha * dxi[dir==0?1:0] * dxi[dir==0?1:0] * BC_mask_Edir_arr(i,j,k,1);
                                auto flag = SparseJacobianMatrixUtil::insertOrAdd(
                                    cidx_g_rhs, val,
                                    &c_indices_g_ptr[ridx_l*nnz_max],
                                    &a_ij_ptr[ridx_l*nnz_max],
                                    nnz_max, icol);
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            {
                                int cidx_g_rhs = (dir == 0 ? dof_arr(i,j+1,k,1) : dof_arr(i+1,j,k,1));
                                Real val = -alpha * dxi[dir==0?1:0] * dxi[dir==0?1:0] * BC_mask_Edir_arr(i,j,k,1);
                                auto flag = SparseJacobianMatrixUtil::insertOrAdd(
                                    cidx_g_rhs, val,
                                    &c_indices_g_ptr[ridx_l*nnz_max],
                                    &a_ij_ptr[ridx_l*nnz_max],
                                    nnz_max, icol);
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            // The following four blocks are for
                            // dir = 0: d/dx(dEz/dz) at Ex(i,j) with Ex centered in x and nodal in z
                            // dir = 2: d/dz(dEx/dx) at Ez(i,j) with Ez centered in z and nodal in x
                            {
                                int cidx_g_rhs = (dir == 0 ? dof_tdir_arr(i,j-1,k,1) : dof_tdir_arr(i-1,j,k,1));
                                Real val = alpha * dxi[0] * dxi[1] * BC_mask_Edir_arr(i,j,k,2);
                                auto flag = SparseJacobianMatrixUtil::insertOrAdd(
                                    cidx_g_rhs, val,
                                    &c_indices_g_ptr[ridx_l*nnz_max],
                                    &a_ij_ptr[ridx_l*nnz_max],
                                    nnz_max, icol);
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            {
                                int cidx_g_rhs = dof_tdir_arr(i,j,k,1);
                                Real val = -alpha * dxi[0] * dxi[1] * BC_mask_Edir_arr(i,j,k,2);
                                auto flag = SparseJacobianMatrixUtil::insertOrAdd(
                                    cidx_g_rhs, val,
                                    &c_indices_g_ptr[ridx_l*nnz_max],
                                    &a_ij_ptr[ridx_l*nnz_max],
                                    nnz_max, icol);
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            {
                                int cidx_g_rhs = (dir == 0 ? dof_tdir_arr(i+1,j-1,k,1) : dof_tdir_arr(i-1,j+1,k,1));
                                Real val = -alpha * dxi[0] * dxi[1] * BC_mask_Edir_arr(i,j,k,2);
                                auto flag = SparseJacobianMatrixUtil::insertOrAdd(
                                    cidx_g_rhs, val,
                                    &c_indices_g_ptr[ridx_l*nnz_max],
                                    &a_ij_ptr[ridx_l*nnz_max],
                                    nnz_max, icol);
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            {
                                int cidx_g_rhs = (dir == 0 ? dof_tdir_arr(i+1,j,k,1) : dof_tdir_arr(i,j+1,k,1));
                                Real val = alpha * dxi[0] * dxi[1] * BC_mask_Edir_arr(i,j,k,2);
                                auto flag = SparseJacobianMatrixUtil::insertOrAdd(
                                    cidx_g_rhs, val,
                                    &c_indices_g_ptr[ridx_l*nnz_max],
                                    &a_ij_ptr[ridx_l*nnz_max],
                                    nnz_max, icol);
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                        } else if (dir==1) {
                            for (int jdir = 0; jdir <= 2; jdir+=2) {
                                {
                                    int cidx_g_rhs = (jdir == 0 ? dof_arr(i-1,j,k,1) : dof_arr(i,j-1,k,1));
                                    Real val = -alpha * dxi[jdir==0?0:1] * dxi[jdir==0?0:1] * BC_mask_Edir_arr(i,j,k,jdir+1);
                                    auto flag = SparseJacobianMatrixUtil::insertOrAdd(
                                        cidx_g_rhs, val,
                                        &c_indices_g_ptr[ridx_l*nnz_max],
                                        &a_ij_ptr[ridx_l*nnz_max],
                                        nnz_max, icol);
                                    if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                                }
                                {
                                    int cidx_g_rhs = (jdir == 0 ? dof_arr(i+1,j,k,1) : dof_arr(i,j+1,k,1));
                                    Real val = -alpha * dxi[jdir==0?0:1] * dxi[jdir==0?0:1] * BC_mask_Edir_arr(i,j,k,jdir+1);
                                    auto flag = SparseJacobianMatrixUtil::insertOrAdd(
                                        cidx_g_rhs, val,
                                        &c_indices_g_ptr[ridx_l*nnz_max],
                                        &a_ij_ptr[ridx_l*nnz_max],
                                        nnz_max, icol);
                                    if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                                }
                            }
                        }
#elif defined(WARPX_DIM_3D)
                        IntVect dvec(AMREX_D_DECL(dir,tdir1,tdir2));
                        IntVect ic(AMREX_D_DECL(i,j,k));
                        {
                            int cidx_g_rhs = dof_arrays[0](ic,1);
                            Real val = 2.0_rt * alpha * ( dxi[dvec[1]]*dxi[dvec[1]] * BC_mask_Edir_arr(i,j,k,0)
                                                        + dxi[dvec[2]]*dxi[dvec[2]] * BC_mask_Edir_arr(i,j,k,3) );
                            auto flag = SparseJacobianMatrixUtil::insertOrAdd(
                                cidx_g_rhs, val,
                                &c_indices_g_ptr[ridx_l*nnz_max],
                                &a_ij_ptr[ridx_l*nnz_max],
                                nnz_max, icol);
                            if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                        }
                        for (int ctr = -1; ctr <= 1; ctr += 2) {
                            for (int tdir = 1; tdir <= 2; tdir++) {
                                auto iv = ic; iv[dvec[tdir]] += ctr;
                                int cidx_g_rhs = dof_arrays[0](iv,1);
                                const int comp_shift = (dvec[tdir] == tdir1) ? 0 : 3;
                                Real val = -alpha * dxi[dvec[tdir]]*dxi[dvec[tdir]] * BC_mask_Edir_arr(i,j,k,comp_shift+1);
                                auto flag = SparseJacobianMatrixUtil::insertOrAdd(
                                    cidx_g_rhs, val,
                                    &c_indices_g_ptr[ridx_l*nnz_max],
                                    &a_ij_ptr[ridx_l*nnz_max],
                                    nnz_max, icol);
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                        }
                        for (int ctr_dir = -1; ctr_dir <= 0; ctr_dir++) {
                            for (int ctr_tdir = -1; ctr_tdir <= 0; ctr_tdir++) {
                                for (int tdir = 1; tdir <= 2; tdir++) {
                                    auto iv = ic; iv[dvec[0]] += (ctr_dir+1); iv[dvec[tdir]] += ctr_tdir;
                                    auto sign = std::copysign(1,ctr_dir) * std::copysign(1,ctr_tdir);
                                    int cidx_g_rhs = dof_arrays[tdir](iv,1);
                                    const int comp_shift = (dvec[tdir] == tdir1) ? 0 : 3;
                                    Real val = Real(sign) * alpha * dxi[dvec[0]]*dxi[dvec[tdir]] * BC_mask_Edir_arr(i,j,k,comp_shift+2);
                                    auto flag = SparseJacobianMatrixUtil::insertOrAdd(
                                        cidx_g_rhs, val,
                                        &c_indices_g_ptr[ridx_l*nnz_max],
                                        &a_ij_ptr[ridx_l*nnz_max],
                                        nnz_max, icol);
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
                // -MM_width[0] to +MM_width[0], j0 ranges from -MM_width[1] to +MM_width[1],
                // and k0 ranges from -MM_width[2] to +MM_width[2], is stored as components
                // of a_mass_matrices[dir=0]. Similarly for Jy/Ey and Jz/Ez.
                // The mapping to the components is given by:
                // mm_comp = i0+MW[0] + MC[0]*(j0+MW[1]) + (MC[0]*MC[1])*(k0+MW[2])
                if (m_include_mass_matrices && a_mass_matrices != nullptr) {

                    auto sigma_ii_arr = (*a_mass_matrices)[lev][dir]->const_array(mfi);
                    GpuArray<int,3> MM_ncomp = {1,1,1};
                    GpuArray<int,3> MM_width = {0,0,0};
                    for (int space_dir = 0; space_dir < AMREX_SPACEDIM; space_dir++) {
                        MM_ncomp[space_dir] = a_mm_ncomp[dir][space_dir];
                        MM_width[space_dir] = (MM_ncomp[space_dir] - 1) / 2;
                    }

                    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        int ridx_l = dof_arr(i,j,k,0);
                        if (ridx_l < 0) { return; }

                        const IntVect iv_base = IntVect(AMREX_D_DECL(i,j,k));
                        int icol = num_nz_ptr[ridx_l];

                        int mm_comp = 0;
                        for (int comp2 = 0; comp2 < MM_ncomp[2]; comp2++) {
                            [[maybe_unused]] const int kk0 = comp2 - MM_width[2];
                            for (int comp1 = 0; comp1 < MM_ncomp[1]; comp1++) {
                                [[maybe_unused]] const int jj0 = comp1 - MM_width[1];
                                for (int comp0 = 0; comp0 < MM_ncomp[0]; comp0++) {
                                    const int ii0 = comp0 - MM_width[0];
                                    const IntVect iv_shift = IntVect(AMREX_D_DECL(ii0, jj0, kk0));
                                    if (full_bx.contains(iv_base + iv_shift)) {
                                        int cidx_g_rhs = dof_arr(iv_base + iv_shift, 1);
                                        Real val = sigma_ii_arr(iv_base, mm_comp);
                                        auto flag = SparseJacobianMatrixUtil::insertOrAdd(
                                            cidx_g_rhs, val,
                                            &c_indices_g_ptr[ridx_l*nnz_max],
                                            &a_ij_ptr[ridx_l*nnz_max],
                                            nnz_max, icol);
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

    ParallelDescriptor::ReduceIntMax(&nnz_actual, 1);
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

    // Step 1: Build global-to-local-extended map on host by iterating
    // over owned + ghost cells of the DOF iMultiFab.
    // Owned cells: local_ext = local_dof (comp 0), range [0, m_ndofs_l)
    // Ghost cells: assigned sequential indices starting from m_ndofs_l

    std::unordered_map<int, int> global_to_ext;
    int next_ghost_idx = m_ndofs_l;

    for (int lev = 0; lev < num_amr_levels; lev++) {
        for (int dir = 0; dir < 3; dir++) {
            const auto& dof_fab = *(dofs_mfarrvec[lev][dir]);
            for (MFIter mfi(dof_fab); mfi.isValid(); ++mfi) {
                const Box gbx = mfi.growntilebox();
                const Box vbx = mfi.validbox();

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

                Gpu::PinnedVector<int> ldof_h(npts), gdof_h(npts);
                Gpu::copyAsync(Gpu::deviceToHost, ldof_d.begin(), ldof_d.end(),
                               ldof_h.begin());
                Gpu::copyAsync(Gpu::deviceToHost, gdof_d.begin(), gdof_d.end(),
                               gdof_h.begin());
                Gpu::streamSynchronize();

                for (int idx = 0; idx < npts; ++idx) {
                    const int gdof = gdof_h[idx];
                    if (gdof < 0) { continue; }
                    if (global_to_ext.find(gdof) != global_to_ext.end()) {
                        continue;
                    }
                    // Recover (i,j,k) from flat index for the owned-cell check
                    int k = idx / (gbx_len.x * gbx_len.y);
                    int j = (idx - k * gbx_len.x * gbx_len.y) / gbx_len.x;
                    int i = idx - k * gbx_len.x * gbx_len.y - j * gbx_len.x;
                    i += gbx_lo.x; j += gbx_lo.y; k += gbx_lo.z;
                    if (vbx.contains(IntVect(AMREX_D_DECL(i, j, k)))) {
                        global_to_ext[gdof] = ldof_h[idx];
                    } else {
                        global_to_ext[gdof] = next_ghost_idx;
                        next_ghost_idx++;
                    }
                }
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
    m_diagonal.resize(m_ndofs_l);
    Gpu::PinnedVector<Real> diag_h(m_ndofs_l, Real(1.0));

    Gpu::PinnedVector<int> r_indices_h(m_ndofs_l);
    Gpu::copyAsync(Gpu::deviceToHost, m_r_indices_g.begin(), m_r_indices_g.end(),
                   r_indices_h.begin());
    Gpu::PinnedVector<Real> a_ij_h(m_a_ij.size());
    Gpu::copyAsync(Gpu::deviceToHost, m_a_ij.begin(), m_a_ij.end(),
                   a_ij_h.begin());
    Gpu::streamSynchronize();

    for (int row = 0; row < m_ndofs_l; row++) {
        const int row_g = r_indices_h[row];
        for (int col = 0; col < num_nz_h[row]; col++) {
            const int idx = row * nnz_max + col;
            if (c_global_h[idx] == row_g) {
                diag_h[row] = a_ij_h[idx];
                break;
            }
        }
    }

    Gpu::copyAsync(Gpu::hostToDevice, diag_h.begin(), diag_h.end(),
                   m_diagonal.begin());
    Gpu::streamSynchronize();

    // Step 4: Store ghost DOF mapping as parallel device vectors
    // for use by BuildExtendedDOFVector
    const int nghost_dofs = m_ndofs_ext - m_ndofs_l;
    std::vector<int> ghost_gdofs_h;
    std::vector<int> ghost_ext_h;
    ghost_gdofs_h.reserve(nghost_dofs);
    ghost_ext_h.reserve(nghost_dofs);

    for (const auto& kv : global_to_ext) {
        if (kv.second >= m_ndofs_l) {
            ghost_gdofs_h.push_back(kv.first);
            ghost_ext_h.push_back(kv.second);
        }
    }

    m_ghost_global_dofs.resize(ghost_gdofs_h.size());
    m_ghost_ext_indices.resize(ghost_ext_h.size());
    if (!ghost_gdofs_h.empty()) {
        Gpu::copyAsync(Gpu::hostToDevice, ghost_gdofs_h.begin(), ghost_gdofs_h.end(),
                       m_ghost_global_dofs.begin());
        Gpu::copyAsync(Gpu::hostToDevice, ghost_ext_h.begin(), ghost_ext_h.end(),
                       m_ghost_ext_indices.begin());
        Gpu::streamSynchronize();
    }

    m_is_remapped = true;

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
            Real r = a_b[row] - Ax;
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
            Real abs_diag = amrex::Math::abs(diag_ptr[row]);
            Real r = (row_sum - abs_diag) / abs_diag;
            return {r};
        });

    Real max_r = amrex::get<0>(reduce_data.value(reduce_op));
    ParallelAllReduce::Max(max_r, ParallelContext::CommunicatorSub());

    return Real(2.0) / (Real(2.0) + max_r);
}
