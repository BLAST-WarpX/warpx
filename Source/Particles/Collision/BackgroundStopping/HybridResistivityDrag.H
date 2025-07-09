/* Copyright 2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_HYBRIDRESISTIVITYDRAG_H_
#define WARPX_PARTICLES_COLLISION_HYBRIDRESISTIVITYDRAG_H_

#include "Particles/MultiParticleContainer.H"
#include "Particles/Collision/CollisionBase.H"

#include <AMReX_REAL.H>

#include <string>

class HybridResistivityDrag final
    : public CollisionBase
{
public:
    HybridResistivityDrag (const std::string& collision_name);

    ~HybridResistivityDrag () override = default;

    HybridResistivityDrag ( HybridResistivityDrag const &)             = delete;
    HybridResistivityDrag& operator= ( HybridResistivityDrag const & ) = delete;
    HybridResistivityDrag ( HybridResistivityDrag&& )                  = delete;
    HybridResistivityDrag& operator= ( HybridResistivityDrag&& )       = delete;

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

    amrex::Parser m_background_resistivity_parser;

    amrex::ParserExecutor<4> m_background_resistivity_func;
};

#endif // WARPX_PARTICLES_COLLISION_HYBRIDRESISTIVITYDRAG_H_
