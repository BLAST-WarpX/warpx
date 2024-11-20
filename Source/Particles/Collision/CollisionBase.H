/* Copyright 2020 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_COLLISIONBASE_H_
#define WARPX_PARTICLES_COLLISION_COLLISIONBASE_H_

#include "Particles/MultiParticleContainer_fwd.H"

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <string>

class CollisionBase
{
public:

    CollisionBase (const std::string& collision_name);

    virtual void doCollisions (amrex::Real /*cur_time*/, amrex::Real /*dt*/, MultiParticleContainer* /*mypc*/ ){}

    CollisionBase(CollisionBase const &) = delete;
    CollisionBase(CollisionBase &&) = delete;
    CollisionBase & operator=(CollisionBase const &) = delete;
    CollisionBase & operator=(CollisionBase const &&) = delete;

    virtual ~CollisionBase() = default;

    [[nodiscard]] int get_ndt() const {return m_ndt;}

protected:

    amrex::Vector<std::string> m_species_names;
    int m_ndt;

};

#endif // WARPX_PARTICLES_COLLISION_COLLISIONBASE_H_
