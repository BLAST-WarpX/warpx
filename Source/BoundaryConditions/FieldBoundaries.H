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
     * \brief Reads the field boundary conditions from the inputfile.
     * Consistency between lower and upper boundary is checked for each direction:
     * if one is periodic, the other must be periodic as well.
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

    /**
     * \brief returns a bool array of size AMREX_SPACEDIM where a given element is
     * true if the corresponding boundary conditions are periodic and false otherwise.
     * Consistency between lower and upper boundary is checked for each direction:
     * if one is periodic, the other must be periodic as well.
     *
     * \param[in] field_boundary_lo the field boundary condition for the lower boundary
     * \param[in] field_boundary_hi the field boundary condition for the upper boundary
     *
     * \returns a bool array of size AMREX_SPACEDIM indicating if field boundary conditions
     *          are periodic or not.
     */
    amrex::Array<bool, AMREX_SPACEDIM>
    get_periodicity_array (
        const amrex::Array<FieldBoundaryType, AMREX_SPACEDIM>& field_boundary_lo,
        const amrex::Array<FieldBoundaryType, AMREX_SPACEDIM>& field_boundary_hi);
}
