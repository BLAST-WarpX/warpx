/* Copyright 2025 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "SparseMatrixUtils.H"

#include "FieldSolver/ImplicitSolvers/SparseJacobianMatrix.H"
#include "FieldSolver/ImplicitSolvers/WarpXSolverVec.H"
#include "FieldSolver/ImplicitSolvers/WarpXSolverDOF.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>

using namespace amrex;

void SparseMatrixUtils::BuildExtendedDOFVector (
    const WarpXSolverVec& a_V,
    Gpu::DeviceVector<Real>& a_ext,
    const SparseJacobianMatrix& a_sp,
    const Vector<Geometry>& a_geom,
    const int a_num_levels)
{
    BL_PROFILE("SparseMatrixUtils::BuildExtendedDOFVector()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_sp.IsRemapped(),
        "SparseMatrixUtils::BuildExtendedDOFVector() called before RemapColumns()");

    const int ndofs_ext = a_sp.nExtended();
    const int ndofs_l = a_sp.nLocalRows();
    a_ext.resize(ndofs_ext);
    auto* ext_ptr = a_ext.data();
    ParallelFor(ndofs_ext, [=] AMREX_GPU_DEVICE (int i) { ext_ptr[i] = Real(0.0); });
    Gpu::streamSynchronize();

    const auto* dofs = a_V.getDOFsObject().get();
    const auto& dofs_mfarrvec = dofs->m_array;
    const auto& data_mfarrvec = a_V.getArrayVec();

    const auto& ghost_gdofs = a_sp.ghostGlobalDOFs();
    const auto& ghost_ext = a_sp.ghostExtIndices();
    const int nghost = a_sp.nGhostDOFs();
    const int* ghost_gdofs_ptr = nghost > 0 ? ghost_gdofs.data() : nullptr;
    const int* ghost_ext_ptr = nghost > 0 ? ghost_ext.data() : nullptr;

    for (int lev = 0; lev < a_num_levels; lev++) {
        for (int dir = 0; dir < 3; dir++) {
            // FillBoundary to populate ghost cells with neighboring rank data
            data_mfarrvec[lev][dir]->FillBoundary(a_geom[lev].periodicity());

            const auto& dof_fab = *(dofs_mfarrvec[lev][dir]);
            const auto& data_fab = *(data_mfarrvec[lev][dir]);

            for (MFIter mfi(dof_fab); mfi.isValid(); ++mfi) {
                const Box gbx = mfi.growntilebox();
                const Box vbx = mfi.validbox();
                auto dof_arr = dof_fab.const_array(mfi);
                auto data_arr = data_fab.const_array(mfi);
                const int nl = ndofs_l;
                const int ng = nghost;

                // Iterate over the full grown box (owned + ghost cells).
                // For owned cells: use local DOF index (comp 0) as ext index.
                // For ghost cells: look up the global DOF in the ghost mapping.
                ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (vbx.contains(IntVect(AMREX_D_DECL(i, j, k)))) {
                        // Owned cell
                        const int ldof = dof_arr(i, j, k, 0);
                        if (ldof >= 0 && ldof < nl) {
                            ext_ptr[ldof] = data_arr(i, j, k);
                        }
                    } else if (ng > 0) {
                        // Ghost cell: find global DOF in ghost mapping
                        const int gdof = dof_arr(i, j, k, 1);
                        if (gdof < 0) { return; }
                        for (int g = 0; g < ng; g++) {
                            if (ghost_gdofs_ptr[g] == gdof) {
                                ext_ptr[ghost_ext_ptr[g]] = data_arr(i, j, k);
                                break;
                            }
                        }
                    }
                });
            }
        }
    }
    Gpu::streamSynchronize();
}

void SparseMatrixUtils::CopyToLocalDOFVector (
    const WarpXSolverVec& a_V,
    Gpu::DeviceVector<Real>& a_local)
{
    const int ndofs_l = static_cast<int>(a_V.nDOF_local());
    a_local.resize(ndofs_l);
    a_V.copyTo(a_local.data());
}

void SparseMatrixUtils::CopyFromLocalDOFVector (
    WarpXSolverVec& a_V,
    const Gpu::DeviceVector<Real>& a_local)
{
    a_V.copyFrom(a_local.data());
}
