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
        // (i.e. not in the incident species list), i.e. if it contains ionization, charge exchange or two-product reaction
        amrex::Vector<std::string> scattering_processes;
        pp_collision_name.queryarr("scattering_processes", scattering_processes);
        const bool reaction_produces_new_species = std::any_of(scattering_processes.begin(), scattering_processes.end(), [](const std::string& process) {
            return process == "ionization" || process == "charge_exchange" || process == "two_product_reaction";
        });

        if (reaction_produces_new_species) {

            // Check that product species have been specified
            amrex::Vector<std::string> product_species;
            pp_collision_name.getarr("product_species", product_species);
            // TODO: check number of product species

            // For ionization:
            if (std::find(scattering_processes.begin(), scattering_processes.end(), "ionization") != scattering_processes.end()) {
                m_num_product_species = 4;
                m_num_products_host.push_back(1); // the non-target species
                m_num_products_host.push_back(0); // the target species
                m_num_products_host.push_back(1); // first product species
                m_num_products_host.push_back(1); // second product species

                // get the reaction energy
                pp_collision_name.get("ionization_energy", m_reaction_energy);
            }

            // For charge exchange or two-product reaction:
            if (std::find(scattering_processes.begin(), scattering_processes.end(), "charge_exchange") != scattering_processes.end() ||
                std::find(scattering_processes.begin(), scattering_processes.end(), "two_product_reaction") != scattering_processes.end()) {
                m_num_product_species = 4;
                m_num_products_host.push_back(0); // the colliding species are consumed in the reaction
                m_num_products_host.push_back(0); // the colliding species are consumed in the reaction
                m_num_products_host.push_back(1); // first product species
                m_num_products_host.push_back(1); // second product species

                // get the reaction energy, assuming zero energy for charge exchange
                pp_collision_name.query("two_product_reaction_energy", m_reaction_energy);
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
