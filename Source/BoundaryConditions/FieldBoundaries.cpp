/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FieldBoundaries.H"

#include <AMReX_ParmParse.H>
#include <AMReX_SPACE.H>

namespace warpx::boundary_conditions
{
    std::pair<
        amrex::Array<FieldBoundaryType, AMREX_SPACEDIM>,
        amrex::Array<FieldBoundaryType, AMREX_SPACEDIM>
    >
    parse_field_boundaries ()
    {
        auto field_boundary_lo =
            amrex::Array<FieldBoundaryType, AMREX_SPACEDIM>{
                AMREX_D_DECL(
                    FieldBoundaryType::Default,
                    FieldBoundaryType::Default,
                    FieldBoundaryType::Default)};
        auto field_boundary_hi = field_boundary_lo;

        const auto pp_boundary = amrex::ParmParse{"boundary"};

        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            pp_boundary.query_enum_sloppy("field_lo",
                field_boundary_lo[idim], "-_", idim);
            pp_boundary.query_enum_sloppy("field_hi",
                field_boundary_hi[idim], "-_", idim);
        }

        return {field_boundary_lo, field_boundary_hi};
    }
}
