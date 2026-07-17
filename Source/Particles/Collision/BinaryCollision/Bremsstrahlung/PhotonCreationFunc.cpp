/* Copyright 2024 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "PhotonCreationFunc.H"

#include "Particles/Collision/BinaryCollision/BinaryCollisionUtils.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/TextMsg.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include <string>

PhotonCreationFunc::PhotonCreationFunc (const std::string& collision_name,
                                        MultiParticleContainer const * const mypc):
    m_collision_type{BinaryCollisionUtils::get_collision_type(collision_name, mypc)}
{

    if (m_collision_type == CollisionType::Bremsstrahlung)
    {

        // Only photons created and only one photon per collision
        m_num_product_species = 1;

    } else {
        WARPX_ABORT_WITH_MESSAGE("Unknown collision type in PhotonCreationFunc");
    }

    const amrex::ParmParse pp_collision_name(collision_name);

    bool create_photons = true;
    pp_collision_name.query("create_photons", create_photons);
    m_create_photons = create_photons;

}
