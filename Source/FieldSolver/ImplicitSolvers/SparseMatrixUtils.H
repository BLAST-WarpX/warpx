/* Copyright 2025-2026 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef SPARSE_MATRIX_UTILS_H_
#define SPARSE_MATRIX_UTILS_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

/**
 * \brief Utility functions for the matrices of the implicit solvers.
 */
namespace SparseMatrixUtils
{
    /**
     * \brief Fill a MultiFab with the diagonal of the curl-curl operator
     *        (without the (c*theta*dt)^2 factor) for one field component.
     *
     * \param[out] a_diag     MultiFab on the grid of the field component
     * \param[in]  a_dir      field component (0, 1, 2)
     * \param[in]  a_geom     geometry of the AMR level
     * \param[in]  a_bc_mask  curl-curl boundary mask of the field component
     */
    void FillCurlCurlDiag (amrex::MultiFab& a_diag,
                           int a_dir,
                           const amrex::Geometry& a_geom,
                           const amrex::MultiFab* a_bc_mask);
}

#endif
