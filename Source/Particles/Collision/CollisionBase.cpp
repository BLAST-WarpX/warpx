/* Copyright 2020 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "CollisionBase.H"

#include "Utils/Parser/ParserUtils.H"

#include <AMReX_ParmParse.H>

CollisionBase::CollisionBase (const std::string& collision_name)
{

    // read collision species
    const amrex::ParmParse pp_collision_name(collision_name);
    pp_collision_name.getarr("species", m_species_names);

    // number of time steps between collisions
    int ndt = 1;
    utils::parser::queryWithParser(
        pp_collision_name, "ndt", ndt);
    m_ndt = ndt;
}
