/* Copyright 2021 Modern Electron
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_IMPACT_IONIZATION_H_
#define WARPX_PARTICLES_COLLISION_IMPACT_IONIZATION_H_

#include "Particles/Collision/ScatteringProcess.H"

#include "Utils/ParticleUtils.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Random.H>
#include <AMReX_REAL.H>

/** @file
 *
 * This file contains filter and transform functors for impact ionization
 */

/**
 * \brief Filter functor for impact ionization
 */
class ImpactIonizationFilterFunc
{
public:

    /**
    * \brief Constructor of the ImpactIonizationFilterFunc functor.
    *
    * This function sample a random number and compares it to the total
    * collision probability to see if the given particle should be considered
    * for an ionization event. If the particle passes this stage the collision
    * cross-section is calculated given its energy and another random number
    * is used to determine whether it actually collides.
    * Note that the mass and energy quantities have to be stored as doubles to
    * ensure accurate energy calculations (otherwise errors occur with single
    * or mixed precision builds of WarpX).
    *
    * @param[in] mcc_process an ScatteringProcess object associated with the ionization
    * @param[in] mass colliding particle's mass (could also assume electron)
    * @param[in] total_collision_prob total probability for a collision to occur
    * @param[in] nu_max maximum collision frequency
    * @param[in] n_a_func ParserExecutor<4> function to get the background
                 density in m^-3 as a function of space and time
    * @param[in] t the current simulation time
    */
    ImpactIonizationFilterFunc(
        ScatteringProcess const& mcc_process,
        double const mass,
        amrex::ParticleReal const total_collision_prob,
        amrex::ParticleReal const nu_max,
        amrex::ParserExecutor<4> const& n_a_func,
        amrex::Real t
    ) : m_mcc_process(mcc_process.executor()), m_mass(mass),
        m_total_collision_prob(total_collision_prob),
        m_nu_max(nu_max), m_n_a_func(n_a_func), m_t(t) { }

    /**
    * \brief Functor call. This method determines if a given (electron) particle
    * should undergo an ionization collision.
    *
    * @param[in] ptd particle tile data
    * @param[in] i particle index
    * @param[in] engine the random number state and factory
    * @return true if a collision occurs, false otherwise
    */
    template <typename PData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool operator() (
        const PData& ptd, int const i, amrex::RandomEngine const& engine
    ) const noexcept
    {
        using namespace amrex;
        using std::sqrt;

        // determine if this particle should collide
        if (Random(engine) > m_total_collision_prob) { return false; }

        // get references to the particle to get its position
        const auto& p = ptd.getSuperParticle(i);
        ParticleReal x, y, z;
        double E_coll;
        get_particle_position(p, x, y, z);

        // calculate neutral density at particle location
        const ParticleReal n_a = m_n_a_func(x, y, z, m_t);

        // get the particle velocity
        const ParticleReal ux = ptd.m_rdata[PIdx::ux][i];
        const ParticleReal uy = ptd.m_rdata[PIdx::uy][i];
        const ParticleReal uz = ptd.m_rdata[PIdx::uz][i];

        // calculate kinetic energy
        const ParticleReal u_coll2 = ux*ux + uy*uy + uz*uz;
        ParticleUtils::getEnergy(u_coll2, m_mass, E_coll);

        // get collision cross-section
        const ParticleReal sigma_E = m_mcc_process.getCrossSection(static_cast<amrex::ParticleReal>(E_coll));

        // calculate normalized collision frequency
        const ParticleReal nu_i = n_a * sigma_E * sqrt(u_coll2) / m_nu_max;

        // check if this collision should be performed
        return (Random(engine) <= nu_i);
    }

private:
    ScatteringProcess::Executor m_mcc_process;
    double m_mass;
    amrex::ParticleReal m_total_collision_prob = 0;
    amrex::ParticleReal m_nu_max;
    amrex::ParserExecutor<4> m_n_a_func;
    amrex::Real m_t;
};


/**
 * \brief Transform functor for impact ionization
 */
class ImpactIonizationTransformFunc
{
public:

    /**
    * \brief Constructor of the ImpactIonizationTransformFunc functor.
    *
    * The transform is responsible for appropriately decreasing the kinetic
    * energy of the colliding particle and assigning appropriate velocities
    * to the two newly created particles. To this end the energy cost of
    * ionization is passed to the constructor as well as the mass of the
    * colliding species and the standard deviation of the ion velocity
    * (normalized temperature).
    * Note that the mass and energy quantities have to be stored as doubles to
    * ensure accurate energy calculations (otherwise errors occur with single
    * or mixed precision builds of WarpX).
    *
    * @param[in] energy_cost energy cost of ionization
    * @param[in] mass1 mass of the colliding species
    * @param[in] sqrt_kb_m value of sqrt(kB/m), where kB is Boltzmann's constant
                 and m is the background neutral mass
    * @param[in] T_a_func ParserExecutor<4> function to get the background
                 temperature in Kelvin as a function of space and time
    * @param[in] t the current simulation time
    */
    ImpactIonizationTransformFunc(
        amrex::ParticleReal energy_cost, double mass1, amrex::ParticleReal sqrt_kb_m,
        amrex::ParserExecutor<4> const& T_a_func, amrex::Real t
    ) :  m_energy_cost(energy_cost), m_mass1(mass1),
         m_sqrt_kb_m(sqrt_kb_m), m_T_a_func(T_a_func), m_t(t) { }

    /**
    * \brief Functor call. It determines the properties of the generated pair
    * and decreases the kinetic energy of the colliding particle. Inputs are
    * standard from the FilterCopyTransfrom::filterCopyTransformParticles
    * function.
    *
    * @param[in,out] dst1 target species 1 (electrons)
    * @param[in,out] dst2 target species 2 (ions)
    * @param[in] src source species (electrons)
    * @param[in] i_src particle index of the source species
    * @param[in] i_dst1 particle index of target species 1
    * @param[in] i_dst2 particle index of target species 2
    * @param[in] engine random number generator engine
    */
    template <typename DstData, typename SrcData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (DstData& dst1, DstData& dst2, SrcData& src,
        int const i_src, int const i_dst1, int const i_dst2,
        amrex::RandomEngine const& engine) const noexcept
    {
        using namespace amrex;
        using std::sqrt;

        // get references to the particle to get its position
        const auto& p = src.getSuperParticle(i_src);
        ParticleReal x, y, z;
        double E_coll;
        get_particle_position(p, x, y, z);

        // calculate standard deviation in neutral velocity distribution using
        // the local temperature
        const ParticleReal ion_vel_std = m_sqrt_kb_m * std::sqrt(m_T_a_func(x, y, z, m_t));

        // get references to the original particle's velocity
        auto& ux = src.m_rdata[PIdx::ux][i_src];
        auto& uy = src.m_rdata[PIdx::uy][i_src];
        auto& uz = src.m_rdata[PIdx::uz][i_src];

        // get references to the new particles' velocities
        auto& e_ux = dst1.m_rdata[PIdx::ux][i_dst1];
        auto& e_uy = dst1.m_rdata[PIdx::uy][i_dst1];
        auto& e_uz = dst1.m_rdata[PIdx::uz][i_dst1];
        auto& i_ux = dst2.m_rdata[PIdx::ux][i_dst2];
        auto& i_uy = dst2.m_rdata[PIdx::uy][i_dst2];
        auto& i_uz = dst2.m_rdata[PIdx::uz][i_dst2];

        // calculate kinetic energy
        const ParticleReal u_coll2 = ux*ux + uy*uy + uz*uz;
        ParticleUtils::getEnergy(u_coll2, m_mass1, E_coll);

        // each electron gets half the energy (could change this later)
        const auto E_out = static_cast<amrex::ParticleReal>((E_coll - m_energy_cost) / 2.0_prt * PhysConst::q_e);

        // precalculate often used value
        constexpr auto c2 = PhysConst::c * PhysConst::c;
        const auto mc2 = m_mass1*c2;

        const amrex::ParticleReal up = sqrt(E_out * (E_out + 2.0_prt*mc2) / c2) / m_mass1;

        // isotropically scatter electrons
        ParticleUtils::RandomizeVelocity(ux, uy, uz, up, engine);
        ParticleUtils::RandomizeVelocity(e_ux, e_uy, e_uz, up, engine);

        // get velocities for the ion from a Maxwellian distribution
        i_ux = ion_vel_std * RandomNormal(0_prt, 1.0_prt, engine);
        i_uy = ion_vel_std * RandomNormal(0_prt, 1.0_prt, engine);
        i_uz = ion_vel_std * RandomNormal(0_prt, 1.0_prt, engine);
    }

private:
    amrex::ParticleReal m_energy_cost;
    double m_mass1;
    amrex::ParticleReal m_sqrt_kb_m;
    amrex::ParserExecutor<4> m_T_a_func;
    amrex::Real m_t;
};
#endif // WARPX_PARTICLES_COLLISION_IMPACT_IONIZATION_H_
