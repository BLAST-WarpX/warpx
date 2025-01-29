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

    if (m_collision_type == CollisionType::DSMC)
    {
        // here we can add logic to deal with cases where products are created,
        // for example with impact ionization
        m_num_product_species = 2;
        m_num_products_host.push_back(1);
        m_num_products_host.push_back(1);
#ifndef AMREX_USE_GPU
        // On CPU, the device vector can be filled immediately
        m_num_products_device.push_back(1);
        m_num_products_device.push_back(1);
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
