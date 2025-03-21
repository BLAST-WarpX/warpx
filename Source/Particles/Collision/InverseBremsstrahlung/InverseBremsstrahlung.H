/* Copyright 2025 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_INVERSEBREMSSTRAHLUNG_H_
#define WARPX_PARTICLES_COLLISION_INVERSEBREMSSTRAHLUNG_H_

#include "Particles/MultiParticleContainer.H"
#include "Particles/Collision/CollisionBase.H"
#include "Particles/Collision/ScatteringProcess.H"

#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_GpuContainers.H>

#include <memory>
#include <string>

class InverseBremsstrahlung final
    : public CollisionBase
{

    using ParticleTileType = WarpXParticleContainer::ParticleTileType;
    using ParticleTileDataType = ParticleTileType::ParticleTileDataType;
    using ParticleBins = amrex::DenseBins<ParticleTileDataType>;
    using index_type = ParticleBins::index_type;

public:
    InverseBremsstrahlung (std::string const& collision_name, MultiParticleContainer const* mypc);

    ~InverseBremsstrahlung () override = default;

    InverseBremsstrahlung ( InverseBremsstrahlung const &)             = delete;
    InverseBremsstrahlung& operator= ( InverseBremsstrahlung const & ) = delete;
    InverseBremsstrahlung ( InverseBremsstrahlung&& )                  = delete;
    InverseBremsstrahlung& operator= ( InverseBremsstrahlung&& )       = delete;

    /** Perform the collisions
     *
     * @param cur_time Current time
     * @param dt Time step size
     * @param mypc Container of species involved
     *
     */
    void doCollisions (amrex::Real cur_time, amrex::Real dt, MultiParticleContainer* mypc) override;

    /** Perform inverse Bremsstrahlung
     *
     * @param pti particle iterator
     * @param t current time
     *
     */
    void doInverseBremsstrahlungWithinTile (amrex::Real dt, int const lev, amrex::MFIter const& mfi,
                                            WarpXParticleContainer& species_1,
                                            WarpXParticleContainer& species_2);

private:

};

#endif // WARPX_PARTICLES_COLLISION_INVERSEBREMSSTRAHLUNG_H
