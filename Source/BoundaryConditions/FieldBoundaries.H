/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Utils/WarpXAlgorithmSelection.H"

#include <AMReX_Array.H>
#include <AMReX_Config.H>

#include <utility>

namespace warpx::boundary_conditions
{
    /**
     * \brief Reads the field boundary conditions from the input file
     *
     * \returns a pair of amrex::Array<FieldBoundaryType, AMREX_SPACEDIM>
     *          containing the boundary conditions for the lower and the
     *          upper boundary.
     */
    std::pair<
        amrex::Array<FieldBoundaryType, AMREX_SPACEDIM>,
        amrex::Array<FieldBoundaryType, AMREX_SPACEDIM>
    >
    parse_field_boundaries ();
}
