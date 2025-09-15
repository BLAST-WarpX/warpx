/* Copyright 2020 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "CollisionHandler.H"

#include "Particles/Collision/BackgroundMCC/BackgroundMCCCollision.H"
#include "Particles/Collision/BackgroundStopping/BackgroundStopping.H"
#include "Particles/Collision/BinaryCollision/BinaryCollision.H"
#include "Particles/Collision/BinaryCollision/Bremsstrahlung/BremsstrahlungFunc.H"
#include "Particles/Collision/BinaryCollision/Bremsstrahlung/PhotonCreationFunc.H"
#include "Particles/Collision/BinaryCollision/Coulomb/PairWiseCoulombCollisionFunc.H"
#include "Particles/Collision/BinaryCollision/DSMC/DSMCFunc.H"
#include "Particles/Collision/BinaryCollision/DSMC/SplitAndScatterFunc.H"
#include "Particles/Collision/BinaryCollision/NuclearFusion/NuclearFusionFunc.H"
#include "Particles/Collision/BinaryCollision/LinearBreitWheeler/LinearBreitWheelerCollisionFunc.H"
#include "Particles/Collision/BinaryCollision/LinearCompton/LinearComptonCollisionFunc.H"
#include "Particles/Collision/BinaryCollision/ParticleCreationFunc.H"
#include "Utils/TextMsg.H"

#include <AMReX_ParmParse.H>

#include <vector>

CollisionHandler::CollisionHandler(MultiParticleContainer const * const mypc)
{

    // Read in collision input
    const amrex::ParmParse pp_collisions("collisions");
    pp_collisions.queryarr("collision_names", collision_names);

    // Create instances based on the collision type
    auto const ncollisions = collision_names.size();
    collision_types.resize(ncollisions);
    allcollisions.resize(ncollisions);
    for (int i = 0; i < static_cast<int>(ncollisions); ++i) {
        const amrex::ParmParse pp_collision_name(collision_names[i]);

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(WarpX::n_rz_azimuthal_modes==1,
        "RZ mode `warpx.n_rz_azimuthal_modes` must be 1 when using the binary collision module.");

        // For legacy, pairwisecoulomb is the default
        std::string type = "pairwisecoulomb";

        pp_collision_name.query("type", type);
        collision_types[i] = type;

        if (type == "pairwisecoulomb") {
            allcollisions[i] =
               std::make_unique<BinaryCollision<PairWiseCoulombCollisionFunc>>(
                    collision_names[i], mypc
                );
            m_use_global_debye_length |= allcollisions[i]->use_global_debye_length();
        }
        else if (type == "background_mcc") {
            allcollisions[i] = std::make_unique<BackgroundMCCCollision>(collision_names[i]);
        }
        else if (type == "background_stopping") {
            allcollisions[i] = std::make_unique<BackgroundStopping>(collision_names[i]);
        }
        else if (type == "dsmc") {
            allcollisions[i] =
                std::make_unique<BinaryCollision<DSMCFunc, SplitAndScatterFunc>>(
                    collision_names[i], mypc
                );
        }
        else if (type == "nuclearfusion") {
            allcollisions[i] =
               std::make_unique<BinaryCollision<NuclearFusionFunc, ParticleCreationFunc>>(
                    collision_names[i], mypc
                );
        }
        else if (type == "bremsstrahlung") {
            allcollisions[i] =
               std::make_unique<BinaryCollision<BremsstrahlungFunc, PhotonCreationFunc>>(
                    collision_names[i], mypc
                );
        }
        else if (type == "linear_breit_wheeler") {
            allcollisions[i] =
               std::make_unique<BinaryCollision<LinearBreitWheelerCollisionFunc, ParticleCreationFunc>>(
                    collision_names[i], mypc
               );
        }
        else if (type == "linear_compton") {
            allcollisions[i] =
               std::make_unique<BinaryCollision<LinearComptonCollisionFunc, ParticleCreationFunc>>(
                    collision_names[i], mypc
               );
        }
        else{
            WARPX_ABORT_WITH_MESSAGE("Unknown collision type.");
        }

    }

}

/** Perform all collisions
 *
 * @param step Current iteration
 * @param cur_time Current time
 * @param dt Time step
 * @param mypc MultiParticleContainer calling this method
 *
 */
void CollisionHandler::doCollisions ( int step, amrex::Real cur_time, amrex::Real dt, MultiParticleContainer* mypc)
{

    if (m_use_global_debye_length) {
        mypc->GenerateGlobalDebyeLength();
    }

    for (auto& collision : allcollisions) {
        int const ndt = collision->get_ndt();
        if ( step % ndt == 0 ) {
            collision->doCollisions(cur_time, dt*ndt, mypc);
        }
    }

}
