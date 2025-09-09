/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FieldBoundaries.H"

#include "Utils/TextMsg.H"

#include <AMReX_ParmParse.H>
#include <AMReX_SPACE.H>

#include <algorithm>

namespace warpx::boundary_conditions
{

    namespace detail
    {
        void check_periodicity_consistency (
            const amrex::Array<FieldBoundaryType, AMREX_SPACEDIM>& field_boundary_lo,
            const amrex::Array<FieldBoundaryType, AMREX_SPACEDIM>& field_boundary_hi)
        {
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
                const bool is_lo_periodic =
                    (field_boundary_lo[idim]  == FieldBoundaryType::Periodic);
                const bool is_hi_periodic =
                    (field_boundary_hi[idim]  == FieldBoundaryType::Periodic);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    (is_lo_periodic == is_hi_periodic),
                    "field boundary must be consistenly periodic in both lo and hi");
            }
        }
    }

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

        detail::check_periodicity_consistency(field_boundary_lo, field_boundary_hi);

        return {field_boundary_lo, field_boundary_hi};
    }

    amrex::Array<bool, AMREX_SPACEDIM>
    get_periodicity_array (
        const amrex::Array<FieldBoundaryType, AMREX_SPACEDIM>& field_boundary_lo,
        const amrex::Array<FieldBoundaryType, AMREX_SPACEDIM>& field_boundary_hi)
    {
        detail::check_periodicity_consistency(field_boundary_lo, field_boundary_hi);

        auto is_field_boundary_periodic = amrex::Array<bool, AMREX_SPACEDIM>{};
        std::transform (
            field_boundary_lo.begin(), field_boundary_lo.end(),
            is_field_boundary_periodic.begin(),
            [](const auto& fb){return (fb == FieldBoundaryType::Periodic);});

        return is_field_boundary_periodic;
    }
}
