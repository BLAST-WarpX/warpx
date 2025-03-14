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
    std::pair<
        amrex::Array<FieldBoundaryType, AMREX_SPACEDIM>,
        amrex::Array<FieldBoundaryType, AMREX_SPACEDIM>
    >
    parse_field_boundaries ();
}
