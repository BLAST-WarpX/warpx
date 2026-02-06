/* Copyright 2026
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PULSED_IONIZATION_H
#define WARPX_PULSED_IONIZATION_H

#include "Particles/MultiParticleContainer.H"
#include "Particles/Collision/CollisionBase.H"

#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_GpuContainers.H>

#include <memory>
#include <string>

/**
 * \brief This class performs ionization of a species via a user-specified rate.
 *        dne/dt = +nu(t)*n1(t)
 *        dn1/dt = -nu(t)*n1(t)
 *        ==> ne(t+dt) - ne(t) = n1(t)*(1 - exp(-nu(t)*t))
 *        For example, this can represent ionization from a laser pulse.
 *        The user-specified ionization rate nu is a function of space and time.
 *        The ion velocity is obtained from a random parent species particle.
 *        The electron velocity is obtained from a user-specified normal distribution.
 */

class PulsedIonization final
    : public CollisionBase
{
public:
    PulsedIonization (std::string const& collision_name, MultiParticleContainer const * mypc);

    ~PulsedIonization () override = default;

    PulsedIonization ( PulsedIonization const &)             = delete;
    PulsedIonization& operator= ( PulsedIonization const & ) = delete;
    PulsedIonization ( PulsedIonization&& )                  = delete;
    PulsedIonization& operator= ( PulsedIonization&& )       = delete;

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
    amrex::GpuArray<amrex::ParticleReal,3> m_electron_drift_velocity;
    amrex::GpuArray<amrex::ParticleReal,3> m_electron_thermal_speed;

    amrex::Parser m_ionization_rate_parser;
    amrex::ParserExecutor<4> m_ionization_rate_func;
};

#endif // WARPX_PULSED_IONIZATION_H
