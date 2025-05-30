/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "SplitAndScatterFunc.H"

SplitAndScatterFunc::SplitAndScatterFunc (const std::string& collision_name,
                                          MultiParticleContainer const * const mypc):
    m_collision_type{BinaryCollisionUtils::get_collision_type(collision_name, mypc)}
{
    if (m_collision_type == CollisionType::DSMC)
    {
        const amrex::ParmParse pp_collision_name(collision_name);

        // Check if ionization is one of the scattering processes by querying
        // for any specified product species (ionization is the only current
        // DSMC process with products)
        amrex::Vector<std::string> product_species;
        pp_collision_name.queryarr("product_species", product_species);

        const bool ionization_flag = (!product_species.empty());

        if (ionization_flag) {
            m_num_product_species = 4;
            m_num_products_host.push_back(1); // the non-target species
            m_num_products_host.push_back(0); // the target species
            m_num_products_host.push_back(1); // first product species
            m_num_products_host.push_back(1); // second product species

            // get the ionization energy
            pp_collision_name.get("ionization_energy", m_ionization_energy);

        } else {
            m_num_product_species = 2;
            m_num_products_host.push_back(0);
            m_num_products_host.push_back(0);
        }
    }
    else
    {
        WARPX_ABORT_WITH_MESSAGE("Unknown collision type in SplitAndScatterFunc");
    }
}
