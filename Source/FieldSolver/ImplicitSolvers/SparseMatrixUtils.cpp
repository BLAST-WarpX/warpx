/* Copyright 2025-2026 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "SparseMatrixUtils.H"

#include "FieldSolver/ImplicitSolvers/SparseJacobianMatrix.H"
#include "FieldSolver/ImplicitSolvers/WarpXSolverVec.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

void SparseMatrixUtils::BuildExtendedDOFVector (
    const WarpXSolverVec& a_V,
    amrex::Gpu::DeviceVector<amrex::Real>& a_ext,
    const SparseJacobianMatrix& a_sp,
    const amrex::Vector<amrex::Geometry>& a_geom,
    const int a_num_levels)
{
    BL_PROFILE("SparseMatrixUtils::BuildExtendedDOFVector()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_sp.IsRemapped(),
        "SparseMatrixUtils::BuildExtendedDOFVector() called before RemapColumns()");

    const int ndofs_ext = a_sp.nExtended();
    a_ext.resize(ndofs_ext);
    auto* ext_ptr = a_ext.data();
    amrex::ParallelFor(ndofs_ext,
        [=] AMREX_GPU_DEVICE (int i) {
            ext_ptr[i] = amrex::Real(0.0);
        });
    amrex::Gpu::streamSynchronize();

    const auto& data_mfarrvec = a_V.getArrayVec();

    for (int lev = 0; lev < a_num_levels; lev++) {
        for (int dir = 0; dir < 3; dir++) {

            const auto& ext_index = a_sp.extIndex(lev, dir);
            const auto& data_src = *(data_mfarrvec[lev][dir]);

            const amrex::IntVect idx_ng = ext_index.nGrowVect();
            const amrex::IntVect data_ng = data_src.nGrowVect();
            const bool need_tmp = (data_ng != idx_ng);

            amrex::MultiFab data_tmp;
            if (need_tmp) {
                data_tmp.define(data_src.boxArray(),
                                data_src.DistributionMap(),
                                data_src.nComp(), idx_ng);
                data_tmp.setVal(amrex::Real(0.0));
                amrex::MultiFab::Copy(data_tmp, data_src, 0, 0,
                    data_src.nComp(),
                    amrex::IntVect::TheZeroVector());
                data_tmp.FillBoundary(
                    a_geom[lev].periodicity());
            } else {
                data_mfarrvec[lev][dir]->FillBoundary(
                    a_geom[lev].periodicity());
            }
            const auto& data_fab =
                need_tmp ? data_tmp : data_src;

            for (amrex::MFIter mfi(ext_index); mfi.isValid(); ++mfi) {
                const amrex::Box gbx = mfi.growntilebox();
                auto idx_arr = ext_index.const_array(mfi);
                auto data_arr = data_fab.const_array(mfi);

                // Scatter into ext_ptr with data-dependent indices, but
                // iterations stay independent as ParallelFor requires (see
                // issue #7097): owned DOFs are written from their owned cell
                // only, and repeat ghost writes store the same value (ghost
                // cells are FillBoundary'd, shared nodes agree across boxes).
                amrex::ParallelFor(gbx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    const int eidx = idx_arr(i, j, k);
                    if (eidx >= 0) {
                        ext_ptr[eidx] = data_arr(i, j, k);
                    }
                });
            }
        }
    }
    amrex::Gpu::streamSynchronize();
}

void SparseMatrixUtils::CopyToLocalDOFVector (
    const WarpXSolverVec& a_V,
    amrex::Gpu::DeviceVector<amrex::Real>& a_local)
{
    const int ndofs_l = static_cast<int>(a_V.nDOF_local());
    a_local.resize(ndofs_l);
    a_V.copyTo(a_local.data());
}

void SparseMatrixUtils::CopyFromLocalDOFVector (
    WarpXSolverVec& a_V,
    const amrex::Gpu::DeviceVector<amrex::Real>& a_local)
{
    a_V.copyFrom(a_local.data());
}

void SparseMatrixUtils::FillCurlCurlDiag (
    amrex::MultiFab& a_diag,
    const int a_dir,
    const amrex::Geometry& a_geom,
    const amrex::MultiFab* a_bc_mask)
{
    using namespace amrex;

#if defined(WARPX_DIM_RSPHERE)
    // 1D spherical geometry is electrostatic
    ignore_unused(a_dir, a_geom, a_bc_mask);
    a_diag.setVal(0.0_rt);
#else
    AMREX_ALWAYS_ASSERT(a_bc_mask != nullptr);
    const auto dxi = a_geom.InvCellSizeArray();

    for (MFIter mfi(a_diag, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box bx = mfi.tilebox();
        auto diag_arr = a_diag.array(mfi);
        const auto bc_mask_arr = a_bc_mask->const_array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            diag_arr(i,j,k) = SparseJacobianMatrixUtil::curlCurlDiag(
                i, j, k, a_dir, 1.0_rt, dxi, bc_mask_arr);
        });
    }
#endif
}
