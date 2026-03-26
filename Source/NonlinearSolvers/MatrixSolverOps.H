/* Copyright 2025-2026 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef MATRIX_SOLVER_OPS_H_
#define MATRIX_SOLVER_OPS_H_

#include <AMReX.H>

/**
 * \brief Abstract interface for matrix operations used by iterative solvers.
 *
 *  Implementations: MultiFabMatrixSolverOps (stencil), SparseMatrixSolverOps (CSR),
 *  MFMatrixSolverOps (matrix-free).
 */
template <class T>
class MatrixSolverOps
{
public:

    using RT = typename T::value_type;

    MatrixSolverOps () = default;
    virtual ~MatrixSolverOps () = default;

    MatrixSolverOps (const MatrixSolverOps&) = delete;
    MatrixSolverOps& operator= (const MatrixSolverOps&) = delete;
    MatrixSolverOps (MatrixSolverOps&&) noexcept = delete;
    MatrixSolverOps& operator= (MatrixSolverOps&&) noexcept = delete;

    /** \brief Update internal state for a new linearization point. */
    virtual void Update (const T& a_U) = 0;

    /** x[i] = b[i] / diag(A)[i] */
    virtual void InitialGuess (T& a_x, const T& a_b) = 0;

    /** x += omega * (b - A*x) / diag(A) */
    virtual void JacobiSweep (T& a_x, const T& a_b, RT a_omega) = 0;

    /** ||b - A*x||_2 (MPI-reduced) */
    [[nodiscard]] virtual RT ResidualNorm (const T& a_x, const T& a_b) = 0;

    /** Gershgorin-based estimate: omega = 2 / (2 + max_i r_i) where
     *  r_i = sum_{j!=i} |A_ij| / |A_ii| is the off-diagonal row ratio. */
    [[nodiscard]] virtual RT EstimateOmega () const = 0;

    /** y = A * x (matrix-vector product) */
    virtual void Matvec ([[maybe_unused]] T& a_y, [[maybe_unused]] const T& a_x)
    {
        amrex::Abort("MatrixSolverOps::Matvec() not implemented");
    }

    /** False when A is diagonal (initial guess is exact) */
    [[nodiscard]] virtual bool HasOffDiag () const = 0;

    [[nodiscard]] virtual bool IsDefined () const = 0;
};

#endif
