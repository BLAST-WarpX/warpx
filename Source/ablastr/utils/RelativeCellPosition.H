/* Copyright 2020 Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ABLASTR_UTILS_RELATIVE_CELL_POSITION_H_
#define ABLASTR_UTILS_RELATIVE_CELL_POSITION_H_

#include <AMReX_BaseFwd.H>

#include <vector>

namespace ablastr::utils
{
    /** Get the Relative Cell Position of Values in an MultiFab
     *
     * Translate the IndexType of a given MultiFab into a position relative to
     * the lower corner of a cell.
     *
     * @param[in] mf the amrex::MultiFab to get relative cell positions for
     * @return relative position to the lower corner, scaled to cell size [0.0:1.0)
     */
    [[nodiscard]] std::vector< double >
    getRelativeCellPosition (amrex::MultiFab const& mf);
}

#endif // ABLASTR_UTILS_RELATIVE_CELL_POSITION_H_
