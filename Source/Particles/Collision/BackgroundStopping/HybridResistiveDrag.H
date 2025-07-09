/* Copyright 2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_HYBRIDRESISTIVEDRAG_H_
#define WARPX_PARTICLES_COLLISION_HYBRIDRESISTIVEDRAG_H_

#include "Particles/MultiParticleContainer.H"
#include "Particles/Collision/CollisionBase.H"

#include <AMReX_REAL.H>

#include <string>

class HybridResistiveDrag final
    : public CollisionBase
{
public:
    HybridResistiveDrag (const std::string& collision_name);

    ~HybridResistiveDrag () override = default;

    HybridResistiveDrag ( HybridResistiveDrag const &)             = delete;
    HybridResistiveDrag& operator= ( HybridResistiveDrag const & ) = delete;
    HybridResistiveDrag ( HybridResistiveDrag&& )                  = delete;
    HybridResistiveDrag& operator= ( HybridResistiveDrag&& )       = delete;

    /** Perform the stopping calculation
     *
     * @param cur_time Current time
     * @param dt Time step size
     * @param mypc Container of species involved
     *
     */
    void doCollisions (amrex::Real cur_time, amrex::Real dt, MultiParticleContainer* mypc) override;

    /** Perform the stopping calculation within a tile for stopping on electrons
     *
     * @param pti particle iterator
     * @param dt time step size
     * @param t current simulation time
     * @param species_mass mass of the active species
     * @param species_charge charge of the active species
     *
     */
    void doApplyResistiveDrag (WarpXParIter& pti, amrex::Real dt, amrex::Real t,
                               amrex::ParticleReal species_mass, amrex::ParticleReal species_charge);

private:

    amrex::Parser m_resistivity_parser;

    amrex::ParserExecutor<4> m_resistivity_func;
};

#endif // WARPX_PARTICLES_COLLISION_HYBRIDRESISTIVEDRAG_H_
