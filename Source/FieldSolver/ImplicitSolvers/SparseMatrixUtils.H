/* Copyright 2025 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef SPARSE_MATRIX_UTILS_H_
#define SPARSE_MATRIX_UTILS_H_

#include "FieldSolver/ImplicitSolvers/WarpXSolverVec.H"

#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

class SparseJacobianMatrix;

/**
 * \brief Utility functions for converting between WarpXSolverVec and
 *        flat device vectors used by SparseJacobianMatrix.
 */
namespace SparseMatrixUtils
{
    /**
     * \brief Build extended vector from a WarpXSolverVec.
     *
     * Performs FillBoundary on the data MultiFabs, then linearizes
     * owned + ghost cells into a flat device vector using the DOF iMultiFab.
     * Owned cells use local DOF indices; ghost cells use the ghost mapping
     * stored in the SparseJacobianMatrix after RemapColumns().
     *
     * \param[in]  a_V        the solver vector to linearize
     * \param[out] a_ext      flat device vector (resized to nExtended)
     * \param[in]  a_sp       sparse Jacobian matrix (must have IsRemapped()==true)
     * \param[in]  a_geom     geometry for each AMR level (for FillBoundary periodicity)
     * \param[in]  a_num_levels  number of AMR levels
     */
    void BuildExtendedDOFVector (const WarpXSolverVec& a_V,
                                 amrex::Gpu::DeviceVector<amrex::Real>& a_ext,
                                 const SparseJacobianMatrix& a_sp,
                                 const amrex::Vector<amrex::Geometry>& a_geom,
                                 int a_num_levels);

    /**
     * \brief Linearize owned DOFs from a WarpXSolverVec into a flat device vector.
     *
     * \param[in]  a_V      the solver vector
     * \param[out] a_local  flat device vector (resized to nDOF_local)
     */
    void CopyToLocalDOFVector (const WarpXSolverVec& a_V,
                               amrex::Gpu::DeviceVector<amrex::Real>& a_local);

    /**
     * \brief Copy local DOF values from flat device vector back into a WarpXSolverVec.
     *
     * \param[out] a_V      the solver vector to write to
     * \param[in]  a_local  flat device vector of size nDOF_local
     */
    void CopyFromLocalDOFVector (WarpXSolverVec& a_V,
                                 const amrex::Gpu::DeviceVector<amrex::Real>& a_local);
}

#endif
