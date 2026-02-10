/* Copyright 2026
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PULSED_DECAY_H
#define WARPX_PULSED_DECAY_H

#include "Particles/MultiParticleContainer.H"
#include "Particles/Collision/CollisionBase.H"

#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_GpuContainers.H>

#include <memory>
#include <string>

/**
 * \brief This class performs the decay of species 1 into species A and B via a user-specified rate.
 *        dnA/dt = +nu(t)*n1(t)= dnB/dt
 *        dn1/dt = -nu(t)*n1(t)
 *        ==> nA(t+dt) - nA(t) = n1(t)*(1 - exp(-nu(t)*t))
 *        For example, this can represent ionization of species 1 from a laser pulse.
 *        The user-specified decay rate nu is a function of space and time.
 *        The velocity of the product particles is set to a random parent species particle plus
 *        a direction-dependent thermal part obtained from a user-specified normal distribution.
 */

class PulsedDecay final
    : public CollisionBase
{
public:
    PulsedDecay (std::string const& collision_name, MultiParticleContainer const * mypc);

    ~PulsedDecay () override = default;

    PulsedDecay ( PulsedDecay const &)             = delete;
    PulsedDecay& operator= ( PulsedDecay const & ) = delete;
    PulsedDecay ( PulsedDecay&& )                  = delete;
    PulsedDecay& operator= ( PulsedDecay&& )       = delete;

    /** Perform the collisions
     *
     * @param cur_time Current time
     * @param dt       Time step size
     * @param mypc     Container of species involved
     *
     */
    void doCollisions (amrex::Real cur_time, amrex::Real dt, MultiParticleContainer* mypc) override;

private:

    amrex::Vector<std::string> m_product_species;

    amrex::ParticleReal m_fixed_product_weight;
    amrex::GpuArray<amrex::ParticleReal,3> m_productA_thermal_speed;
    amrex::GpuArray<amrex::ParticleReal,3> m_productB_thermal_speed;

    amrex::Parser m_decay_rate_parser;
    amrex::ParserExecutor<4> m_decay_rate_func;
};

#endif // WARPX_PULSED_DECAY_H
