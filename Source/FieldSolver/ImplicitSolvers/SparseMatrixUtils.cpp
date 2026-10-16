/* Copyright 2025-2026 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "SparseMatrixUtils.H"

#include "FieldSolver/ImplicitSolvers/SparseJacobianMatrix.H"

#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

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
