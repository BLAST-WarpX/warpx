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

        // Check if the scattering processes include reactions that produce macroparticles in new species
        // (i.e. not in the incident species list), i.e. if it contains ionization or charge exchange
        amrex::Vector<std::string> scattering_processes;
        pp_collision_name.queryarr("scattering_processes", scattering_processes);
        const bool reaction_produces_new_species = std::any_of(scattering_processes.begin(), scattering_processes.end(), [](const std::string& process) {
            return process == "ionization" || process == "charge_exchange";
        });

        if (reaction_produces_new_species) {

            // Check that product species have been specified
            amrex::Vector<std::string> product_species;
            pp_collision_name.getarr("product_species", product_species);

            // For ionization:
            if (std::find(scattering_processes.begin(), scattering_processes.end(), "ionization") != scattering_processes.end()) {
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
                    m_num_products_host.push_back(2); // the non-target species: one particle for the product ; one particle for the fact that the incident particle loses energy
                    m_num_products_host.push_back(0); // the target species
                    m_num_products_host.push_back(1); // the other product of the reaction
                } else {
                    m_num_product_species = 4;
                    m_num_products_host.push_back(1); // the non-target species
                    m_num_products_host.push_back(0); // the target species
                    m_num_products_host.push_back(1); // first product species
                    m_num_products_host.push_back(1); // second product species
                }

                // get the ionization energy
                pp_collision_name.get("ionization_energy", m_ionization_energy);
            }

            // For charge exchange:
            if (std::find(scattering_processes.begin(), scattering_processes.end(), "charge_exchange") != scattering_processes.end()) {
                m_num_products_host.push_back(0); // the colliding species are consumed in the reaction
                m_num_products_host.push_back(0); // the colliding species are consumed in the reaction
                m_num_products_host.push_back(1); // first product species
                m_num_products_host.push_back(1); // second product species
            }

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
