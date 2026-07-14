/* Copyright 2020 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "CollisionHandler.H"

#include "Particles/Collision/BackgroundMCC/BackgroundMCCCollision.H"
#include "Particles/Collision/PulsedDecay/PulsedDecay.H"
#include "Particles/Collision/BackgroundStopping/BackgroundStopping.H"
#include "Particles/Collision/HybridResistiveDrag/HybridResistiveDrag.H"
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
#include "Particles/Collision/InverseBremsstrahlung/InverseBremsstrahlung.H"
#include "Utils/TextMsg.H"

#include "Particles/ParticleCreation/SmartCopy.H"
#ifdef WARPX_QED
#include "Particles/Collision/BinaryCollision/VirtualPhotonCreation.H"
#endif
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
        else if (type == "pulsed_decay") {
            allcollisions[i] = std::make_unique<PulsedDecay>(collision_names[i], mypc);
        }
        else if (type == "background_stopping") {
            allcollisions[i] = std::make_unique<BackgroundStopping>(collision_names[i]);
        }
        else if (type == "hybrid_resistive_drag") {
            allcollisions[i] = std::make_unique<HybridResistiveDrag>(collision_names[i]);
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
        else if (type == "inverse_bremsstrahlung") {
            allcollisions[i] = std::make_unique<InverseBremsstrahlung>(collision_names[i], mypc);
            m_use_global_debye_length = true;
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

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    /* In RZ and RCYLINDER geometry, macroparticles can collide with other macroparticles
     * in the same *cylindrical* cell, or in RSPHERE the same *spherical* shell.
     * Because of this, the colliding macroparticles would not nessecarily be spatially
     * near each other. This would violate the underlying assumptions that particles within the
     * same cylindrical or spherical cell represent a cylindrically- or spherically-symmetric
     * momentum distribution function and are spatially local. Therefore, we temporarily rotate
     * the momentum of the macroparticles to the curvilinear frame, equivalent to the x-axis.
     * (This is only valid if we use only the m=0 azimuthal mode in the simulation;
     * there is a corresponding assert statement at initialization.) */
    mypc->TransformMomentumToCurvilinear(/*forward*/true);
#endif

#ifdef WARPX_QED
    // For QED incoherent processes (e.g. Bethe-Heitler, Landau-Lifschitz), the process is mediated by virtual photons.
    // The virtual photons are newly generated here and participate in the collisions.
    // Here, the virtual photons are regenerated from scratch, i.e. they are overwritten by new ones at each time step.
    if(mypc->nSpecies() > 0) {
        collision::binarycollision::virtualphotons::GenerateVirtualPhotons(mypc);
    }
#endif

    if (m_use_global_debye_length) {
        // This will calculate the temperature, Vbar, and particle number that are needed by
        // the various collision algorithms
        mypc->GenerateGlobalDebyeLength();
    }

    for (auto& collision : allcollisions) {
        const int ndt = collision->get_ndt();
        const auto collision_stepping_mode = collision->get_collision_stepping_mode();

        if (collision_stepping_mode == CollisionSteppingMode::Subcycle) {
            // Subcycle: run ndt times per PIC step, each with dt_collision = dt / ndt
            const amrex::Real dt_sub = dt / ndt;
            for (int i_sub = 0; i_sub < ndt; ++i_sub) {
                const amrex::Real sub_time = cur_time + i_sub * dt_sub;
                collision->doCollisions(sub_time, dt_sub, mypc);
            }
        } else {
            // Supercycle: run once every ndt PIC steps, with dt_collision = dt * ndt
            if ( step % ndt == 0 ) {
                collision->doCollisions(cur_time, dt*ndt, mypc);
            }
        }
    }

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    // Undo the rotation above
    mypc->TransformMomentumToCurvilinear(/*forward*/false);
#endif

}
