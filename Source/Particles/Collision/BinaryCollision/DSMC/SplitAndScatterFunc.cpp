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
    const amrex::ParmParse pp_collision_name(collision_name);

    // Check if ionization is one of the scattering processes
    amrex::Vector<std::string> scattering_process_names;
    pp_collision_name.queryarr("scattering_processes", scattering_process_names);
    for (const auto& scattering_process : scattering_process_names) {
        if (scattering_process.find("ionization") != std::string::npos) {
           m_ionization_flag = true;
        }
    }

    if (m_collision_type == CollisionType::DSMC)
    {
        if (m_ionization_flag) {
            // Product species include the ion
            // TODO: as an alternative, we could use the runtime attribute `ionization_level` for this species
            m_num_product_species = 3;
            m_num_products_host.push_back(2); // electron species:
            // potentially 2 products per reaction: the scattered incoming electron, and the new electron from ionization
            m_num_products_host.push_back(1);
            m_num_products_host.push_back(1); // corresponds to the ionized species
        } else {
            m_num_product_species = 2;
            m_num_products_host.push_back(1);
            m_num_products_host.push_back(1);
        }

#ifndef AMREX_USE_GPU
        // On CPU, the device vector can be filled immediately
        for (int i = 0; i < m_num_product_species; i++) {
            m_num_products_device.push_back(m_num_products_host[i]);
        }
#endif
    }
    else
    {
        WARPX_ABORT_WITH_MESSAGE("Unknown collision type in SplitAndScatterFunc");
    }

#ifdef AMREX_USE_GPU
     m_num_products_device.resize(m_num_product_species);
     amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_num_products_host.begin(),
                           m_num_products_host.end(),
                           m_num_products_device.begin());
     amrex::Gpu::streamSynchronize();
#endif
}
