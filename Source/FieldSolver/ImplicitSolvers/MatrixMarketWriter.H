/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef MATRIX_MARKET_WRITER_H_
#define MATRIX_MARKET_WRITER_H_

#include <AMReX_Arena.H>
#include <AMReX_BaseFab.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_INT.H>
#include <AMReX_REAL.H>

#include <string>
#include <vector>

struct WarpXSolverDOF;

/**
 * \brief Writing of the preconditioner matrices in MatrixMarket coordinate format.
 *
 *  Rows and columns are global DOF indices (1-based in the file). The header
 *  holds metadata as "% key: value" comment lines. Each rank writes its own
 *  rows, in rank order, so the matrix is never gathered on one rank.
 */
namespace MatrixMarketWriter
{
    /** \brief Matrix entries of the rows owned by this rank (0-based global indices). */
    struct Triplets
    {
        std::vector<amrex::Long> row;
        std::vector<amrex::Long> col;
        std::vector<amrex::Real> val;

        void add (amrex::Long a_row, amrex::Long a_col, amrex::Real a_val)
        {
            row.push_back(a_row);
            col.push_back(a_col);
            val.push_back(a_val);
        }
    };

    /**
     * \brief Write a matrix with its row statistics and DOF map. Collective.
     *
     * \param[in]     a_filename  output file name
     * \param[in]     a_header    metadata lines ("key: value"), written first
     * \param[in]     a_dofs      DOF object (row distribution and DOF map)
     * \param[in,out] a_local     entries of the local rows; sorted on output
     */
    void Write (const std::string& a_filename,
                const std::vector<std::string>& a_header,
                const WarpXSolverDOF& a_dofs,
                Triplets& a_local);

    /**
     * \brief Write the DOF map (global row -> level, field component, i, j, k)
     *        to a_dir/dofmap.txt, once per directory and run. Collective.
     */
    void WriteDOFMap (const std::string& a_dir, const WarpXSolverDOF& a_dofs);

    /** \brief Host copy of a (possibly device) FAB, ghost cells included. */
    template <class T>
    amrex::BaseFab<T> HostCopy (const amrex::BaseFab<T>& a_fab)
    {
        amrex::BaseFab<T> h(a_fab.box(), a_fab.nComp(), amrex::The_Pinned_Arena());
        amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
                              a_fab.dataPtr(), a_fab.dataPtr() + a_fab.size(),
                              h.dataPtr());
        amrex::Gpu::streamSynchronize();
        return h;
    }
}

#endif
