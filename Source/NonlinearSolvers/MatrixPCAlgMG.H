/* Copyright 2026 Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_MATRIX_PC_ALGMG_H_
#define WARPX_MATRIX_PC_ALGMG_H_

#include <AMReX_INT.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <memory>
#include <string>

/**
 * \brief AMReX algebraic multigrid (AlgMG) on the matrix assembled by MatrixPC
 *
 *  Kept out of MatrixPC.H so that the AlgMG headers are only compiled here.
 */
class MatrixPCAlgMG
{
public:

    /**
     * \brief Read the parameters with prefix a_name
     *
     *  a_row_offsets holds the first global row of each process, plus the total.
     */
    MatrixPCAlgMG (const std::string& a_name, const amrex::Vector<amrex::Long>& a_row_offsets);

    ~MatrixPCAlgMG ();

    MatrixPCAlgMG (const MatrixPCAlgMG&) = delete;
    MatrixPCAlgMG& operator= (const MatrixPCAlgMG&) = delete;
    MatrixPCAlgMG (MatrixPCAlgMG&&) noexcept = delete;
    MatrixPCAlgMG& operator= (MatrixPCAlgMG&&) noexcept = delete;

    void printParameters () const;

    /**
     * \brief Build the hierarchy from the local rows
     *
     *  Row i has a_num_nz[i] entries, stored at a_cols and a_vals starting at
     *  i*a_ncols_max. Column indices are global.
     */
    void Setup (int a_nrows, int a_ncols_max, const int* a_num_nz,
                const int* a_cols, const amrex::Real* a_vals);

    //! Device pointer to the right-hand side, to be filled before Apply
    [[nodiscard]] amrex::Real* rhs ();

    //! Device pointer to the solution computed by Apply
    [[nodiscard]] const amrex::Real* sol () const;

    //! Apply the preconditioner: solve A sol = rhs approximately
    void Apply ();

private:

    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif
