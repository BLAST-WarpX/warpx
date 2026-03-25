/* Copyright 2025 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef SPARSE_JACOBIAN_MATRIX_H_
#define SPARSE_JACOBIAN_MATRIX_H_

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

struct WarpXSolverDOF;

namespace SparseJacobianMatrixUtil
{
    /**
     * \brief Insert a new (column, value) entry or add to an existing column.
     *
     * Used inside GPU ParallelFor to build sparse matrix rows incrementally.
     * Returns true on success; false if the row is full (need reallocation).
     * Entries with a_cidx < 0 (outside domain) are silently skipped.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool insertOrAdd (const int a_cidx,
                      const amrex::Real a_val,
                      int* const a_cidxs,
                      amrex::Real* const a_aij,
                      const int a_nnz,
                      int& a_ncol)
    {
        if (a_cidx < 0) { return true; /* outside domain */ }
        int loc = -1;
        for (int icol = 0; icol < std::min(a_ncol, a_nnz); icol++) {
            if (a_cidxs[icol] == a_cidx) {
                loc = icol;
                break;
            }
        }
        if (loc < 0) {
            a_ncol++;
            if (a_ncol > a_nnz) { return false; }
            else {
                a_cidxs[a_ncol-1] = a_cidx;
                a_aij[a_ncol-1] = a_val;
            }
        } else {
            a_aij[loc] += a_val;
        }
        return true;
    }
}

/**
 * \brief Sparse matrix representation of the linearized implicit operator
 *
 *  Assembles and stores the sparse matrix:
 *      A = I + curl(alpha * curl[]) + M
 *  where alpha = (theta*dt*c)^2 and M is the scaled mass matrix (plasma response).
 *
 *  The matrix is stored in a row-major fixed-width format on GPU:
 *  - m_r_indices_g: global row indices (size n_local_rows)
 *  - m_num_nz:      non-zero count per row (size n_local_rows)
 *  - m_c_indices_g: column indices (size n_local_rows * nnz_max)
 *  - m_a_ij:        matrix values  (size n_local_rows * nnz_max)
 *
 *  This class is used by ImplicitSolver and accessed by external library
 *  preconditioners (e.g., PETSc) through the ImplicitSolver interface.
 */
class SparseJacobianMatrix
{
public:

    SparseJacobianMatrix () = default;

    ~SparseJacobianMatrix () = default;

    SparseJacobianMatrix (const SparseJacobianMatrix&) = delete;
    SparseJacobianMatrix& operator= (const SparseJacobianMatrix&) = delete;
    SparseJacobianMatrix (SparseJacobianMatrix&&) noexcept = delete;
    SparseJacobianMatrix& operator= (SparseJacobianMatrix&&) noexcept = delete;

    /**
     * \brief Allocate sparse matrix storage and read parameters.
     *
     * \param[in] a_ndofs_l local number of degrees of freedom
     * \param[in] a_ndofs_g global number of degrees of freedom
     */
    void Define (int a_ndofs_l, int a_ndofs_g);

    [[nodiscard]] bool IsDefined () const { return m_is_defined; }

    /**
     * \brief Assemble the sparse matrix, reallocating if needed.
     *
     * Calls Assemble() in a loop, growing nnz_max on insufficient storage.
     */
    void Update (const WarpXSolverDOF* a_dofs,
                 const amrex::Vector<amrex::Geometry>& a_geom,
                 amrex::Real a_theta_dt,
                 const amrex::Vector<amrex::Array<const amrex::MultiFab*,3>>& a_bc_masks,
                 const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>* a_mass_matrices,
                 const amrex::Array<amrex::IntVect,3>& a_mm_ncomp);

    /**
     * \brief Copy sparse matrix data to caller-provided device vectors.
     */
    void GetMatrix (amrex::Gpu::DeviceVector<int>& a_r_indices_g,
                    amrex::Gpu::DeviceVector<int>& a_num_nz,
                    amrex::Gpu::DeviceVector<int>& a_c_indices_g,
                    amrex::Gpu::DeviceVector<amrex::Real>& a_a_ij,
                    int& a_n,
                    int& a_ncols_max) const;

    void printParameters () const;

    /**
     * \brief Core assembly routine. Returns 0 on success, positive int
     *        (actual_nnz - nnz_max) if storage is insufficient.
     *
     * Public because CUDA requires public access for functions containing
     * extended __device__ lambdas (used with amrex::ParallelFor).
     */
    int Assemble (const WarpXSolverDOF* a_dofs,
                  const amrex::Vector<amrex::Geometry>& a_geom,
                  amrex::Real a_theta_dt,
                  const amrex::Vector<amrex::Array<const amrex::MultiFab*,3>>& a_bc_masks,
                  const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>* a_mass_matrices,
                  const amrex::Array<amrex::IntVect,3>& a_mm_ncomp);

private:

    void readParameters ();

    bool m_is_defined = false;
    bool m_verbose = true;
    bool m_pc_diag_only = false;
    bool m_include_mass_matrices = false;

    int m_ndofs_l = 0;
    int m_ndofs_g = 0;
    int m_pc_mat_nnz = 1;
    int m_num_realloc = 0;

    amrex::Gpu::DeviceVector<int> m_r_indices_g;
    amrex::Gpu::DeviceVector<int> m_num_nz;
    amrex::Gpu::DeviceVector<int> m_c_indices_g;
    amrex::Gpu::DeviceVector<amrex::Real> m_a_ij;
};

#endif
