/* Copyright 2025 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef MATRIX_SOLVER_OPS_H_
#define MATRIX_SOLVER_OPS_H_

/**
 * \brief Abstract interface for matrix operations needed by Jacobi-type
 *        iterative solvers.
 *
 *  Concrete implementations provide either:
 *  - MultiFab stencil operations (MultiFabMatrixSolverOps), or
 *  - CSR sparse matrix operations (SparseMatrixSolverOps).
 *
 *  The Jacobi solver (JacobiPC) holds a pointer to this interface and
 *  drives the iteration loop without knowing which path is active.
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

    /**
     * \brief Update internal state for a new linearization point.
     *
     * Called every Newton iteration. CSR path reassembles the sparse matrix;
     * stencil path allocates work arrays on first call.
     */
    virtual void Update (const T& a_U) = 0;

    /**
     * \brief Compute initial guess: x[i] = b[i] / diag(A)[i]
     */
    virtual void InitialGuess (T& a_x, const T& a_b) = 0;

    /**
     * \brief One damped Jacobi sweep: x += omega * (b - A*x) / diag(A)
     */
    virtual void JacobiSweep (T& a_x, const T& a_b, RT a_omega) = 0;

    /**
     * \brief Compute ||b - A*x||_2 (fully MPI-reduced).
     */
    [[nodiscard]] virtual RT ResidualNorm (const T& a_x, const T& a_b) = 0;

    /**
     * \brief Whether the matrix has off-diagonal entries.
     *
     * When false, the initial guess x = b/diag(A) is exact and the
     * iteration loop can be skipped entirely.
     */
    [[nodiscard]] virtual bool HasOffDiag () const = 0;

    [[nodiscard]] virtual bool IsDefined () const = 0;
};

#endif
