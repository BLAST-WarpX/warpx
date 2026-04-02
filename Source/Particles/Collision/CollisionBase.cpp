/* Copyright 2020 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "CollisionBase.H"

#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"

#include <AMReX_ParmParse.H>

CollisionBase::CollisionBase (const std::string& collision_name)
{
    m_collision_name = collision_name;
    BackwardCompatibility();

    // read collision species
    const amrex::ParmParse pp_collision_name(collision_name);
    pp_collision_name.getarr("species", m_species_names);

    // time step control: ndt_supercycle or ndt_subcycle (mutually exclusive)
    int ndt_supercycle = 0;
    int ndt_subcycle   = 0;
    const bool has_supercycle = utils::parser::queryWithParser(
        pp_collision_name, "ndt_supercycle", ndt_supercycle);
    const bool has_subcycle   = utils::parser::queryWithParser(
        pp_collision_name, "ndt_subcycle", ndt_subcycle);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !(has_supercycle && has_subcycle),
        "<collision_name>.ndt_supercycle and <collision_name>.ndt_subcycle "
        "are mutually exclusive. Specify at most one."
    );

    if (has_subcycle) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            ndt_subcycle >= 1,
            "<collision_name>.ndt_subcycle must be >= 1."
        );
        m_ndt                     = ndt_subcycle;
        m_collision_stepping_mode = CollisionSteppingMode::Subcycle;
    } else {
        if (has_supercycle) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                ndt_supercycle >= 1,
                "<collision_name>.ndt_supercycle must be >= 1."
            );
            m_ndt = ndt_supercycle;
        } else {
            m_ndt = 1;
        }
        m_collision_stepping_mode = CollisionSteppingMode::Supercycle;
    }
}

void
CollisionBase::BackwardCompatibility ()
{
    const amrex::ParmParse pp_collision_name(m_collision_name);
    int backward_int;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !pp_collision_name.query("ndt", backward_int),
        "<collision_name>.ndt is no longer a valid option. "
        "Please use <collision_name>.ndt_supercycle (run collision every N PIC steps) "
    );
}
