/* Copyright 2020 Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "RelativeCellPosition.H"

#include <AMReX_Config.H>
#include <AMReX_IndexType.H>
#include <AMReX_MultiFab.H>

std::vector< double >
utils::getRelativeCellPosition(amrex::MultiFab const& mf)
{
    amrex::IndexType const idx_type = mf.ixType();

    std::vector< double > relative_position(AMREX_SPACEDIM, 0.0);

    // amrex::CellIndex::CELL means: 0.5 from lower corner for that index/direction
    // amrex::CellIndex::NODE means: at corner for that index/direction
    // WarpX::grid_type==GridType::Collocated means: all indices/directions on CellIndex::NODE
    for (int d = 0; d < AMREX_SPACEDIM; d++)
    {
        if (idx_type.cellCentered(d)) {
            relative_position.at(d) = 0.5;
        }
    }

    return relative_position;
}
