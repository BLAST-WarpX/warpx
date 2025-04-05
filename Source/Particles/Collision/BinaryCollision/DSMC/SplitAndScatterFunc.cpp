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

        // if ionization is one of the processes, check if one of the colliding
        // species is also used as a product species
        if (ionization_flag) {
            // grab the colliding species
            amrex::Vector<std::string> colliding_species;
            pp_collision_name.getarr("species", colliding_species);
            // grab the target species (i.e., the species that looses an
            // electron during the collision)
            std::string target_species;
            pp_collision_name.query("ionization_target_species", target_species);
            // find the index of the non-target species (the one that could
            // also be used as a product species)
            int non_target_idx = 0;
            if (colliding_species[0] == target_species) {
                non_target_idx = 1;
            }

            // check if the non-target species is in ``product_species``
            auto it = std::find(product_species.begin(), product_species.end(), colliding_species[non_target_idx]);

            if (it != product_species.end()) {
                m_num_product_species = 3;
                m_num_products_host.push_back(2); // the non-target species
                m_num_products_host.push_back(1); // the target species
                m_num_products_host.push_back(1); // corresponds to whichever ionization product species1 is not (ion or electron)
            } else {
                m_num_product_species = 4;
                m_num_products_host.push_back(1); // the non-target species
                m_num_products_host.push_back(1); // the target species
                m_num_products_host.push_back(1); // first product species
                m_num_products_host.push_back(1); // second product species
            }

            // get the ionization energy
            pp_collision_name.get("ionization_energy", m_ionization_energy);

        } else {
            m_num_product_species = 2;
            m_num_products_host.push_back(1);
            m_num_products_host.push_back(1);
        }
    }
    else
    {
        WARPX_ABORT_WITH_MESSAGE("Unknown collision type in SplitAndScatterFunc");
    }
}
