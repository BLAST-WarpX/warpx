/* Copyright 2016-2020 Maxence Thevenet, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_FILTERFUNCTORS_H
#define WARPX_FILTERFUNCTORS_H

#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Gpu.H>
#include <AMReX_Parser.H>
#include <AMReX_Random.H>

using SuperParticleType = typename WarpXParticleContainer::SuperParticleType;

/**
 *  \brief Used to keep track of what inputs units a filter function should expect.
 *         "WarpX units" means the momentum is "gamma*v" (aka proper velocity)
 *         "SI" means the momentum is mass*gamma*v.
 */
enum struct InputUnits{WarpX, SI};

/**
 * \brief Functor that returns 0 or 1 depending on a random draw per particle
 */
struct RandomFilter
{
    /** constructor
     * \param a_is_active whether the test is active
     * \param a_fraction fraction of particles to select
     */
    RandomFilter(bool a_is_active, amrex::Real a_fraction)
        : m_is_active(a_is_active), m_fraction(a_fraction) {}

    /**
     * \brief draw random number, return 1 if number < m_fraction, 1 otherwise
     * \param p one particle
     * \param engine the random number state and factory
     * \return whether or not the particle is selected
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool operator () (const SuperParticleType& p, const amrex::RandomEngine& engine) const noexcept
    {
        amrex::ignore_unused(p);

        return ( (!m_is_active) || (amrex::Random(engine) < m_fraction) );
    }
private:
    const bool m_is_active; //! select all particles if false
    const amrex::Real m_fraction = 1.0; //! range: [0.0:1.0] where 0 is no & 1 is all particles
};

/**
 * \brief Functor that returns 1 if stride divide particle_id, 0 otherwise
 */
struct UniformFilter
{
    /** constructor
     * \param a_is_active whether the test is active
     * \param a_stride one particle every a_stride is written to file
     */
    UniformFilter(bool a_is_active, int a_stride)
        : m_is_active(a_is_active), m_stride(a_stride) {}

    /**
     * \brief return 1 if stride divide particle_id, 0 otherwise
     * \param p one particle
     * \return whether or not the particle is selected
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool operator () (const SuperParticleType& p, const amrex::RandomEngine&) const noexcept
    {
        return ( (!m_is_active) || ( p.id()%m_stride == 0 ) );
    }
private:
    const bool m_is_active; //! select all particles if false
    const int m_stride = 0; //! selection of every n-th particle
};

/**
 * \brief Functor that returns 0 or 1 depending on a parser selection
 */
struct ParserFilter
{
    /** constructor
     * \param a_is_active whether the test is active
     * \param a_filter_parser parser taking t, x, y, z, ux, uy, and uz, and returning a boolean for selected particle
     * \param a_mass mass of the particle species
     * \param time simulation time on the coarsest level
     */
    ParserFilter(bool a_is_active, amrex::ParserExecutor<7> const& a_filter_parser,
        const amrex::ParticleReal a_mass, const amrex::Real time):
            m_is_active{a_is_active},
            m_function_partparser{a_filter_parser},
            m_mass{a_mass},
            m_t{time}
    {}

    /**
     * \brief return 1 if the particle is selected by the parser
     * \param p one particle
     * \return whether or not the particle is selected
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool operator () (const SuperParticleType& p, const amrex::RandomEngine&) const noexcept
    {
        using namespace amrex::literals;

        if (!m_is_active){
            return true;
        }
        else{
            amrex::ParticleReal x, y, z;
            get_particle_position(p, x, y, z);
            amrex::Real ux = p.rdata(PIdx::ux)/PhysConst::c;
            amrex::Real uy = p.rdata(PIdx::uy)/PhysConst::c;
            amrex::Real uz = p.rdata(PIdx::uz)/PhysConst::c;
            if (m_units == InputUnits::SI)
            {
                ux /= m_mass;
                uy /= m_mass;
                uz /= m_mass;
            }
            // ux, uy, uz are now in beta*gamma

            // This is actually a binary true/false (1/0) check,
            // but the parser returns floating point types
            return (m_function_partparser(m_t,x,y,z,ux,uy,uz) != 0.0_rt);
        }
    }
private:
    /** Whether this diagnostics is activated. Select all particles if false */
    const bool m_is_active;
public:
    /** Parser function with 7 input variables, t,x,y,z,ux,uy,uz */
    amrex::ParserExecutor<7> const m_function_partparser;
    /** Mass of particle species */
    amrex::ParticleReal m_mass;
    /** Store physical time on the coarsest level. */
    amrex::Real m_t;
    /** keep track of momentum units particles will come in with **/
    InputUnits m_units{InputUnits::WarpX};
};



/**
 * \brief Functor that returns 1 if the particle is inside a given axis-aligned region
 *        defined by amrex::RealBox, 0 otherwise.
 */
struct GeometryFilter
{
    GeometryFilter(bool a_is_active, amrex::RealBox a_domain)
        : m_is_active(a_is_active), m_domain(a_domain) {}
    /**
     * \brief return 1 if the particle is within the region described by the RealBox
     * \param p one particle
     * \return whether or not the particle is inside the region defined by m_domain
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool operator () (const SuperParticleType& p, const amrex::RandomEngine&) const noexcept
    {
        if ( !m_is_active ) { return true; }
        return AMREX_D_TERM(    (p.pos(0) >= m_domain.lo(0)) && (p.pos(0) <= m_domain.hi(0) ),
                            &&  (p.pos(1) >= m_domain.lo(1)) && (p.pos(1) <= m_domain.hi(1) ),
                            &&  (p.pos(2) >= m_domain.lo(2)) && (p.pos(2) <= m_domain.hi(2) ));
    }
private:
    /** Whether this diagnostics is activated. Select all particles if false */
    const bool m_is_active;
    /** Physical extent of the axis-aligned region used for particle check */
    const amrex::RealBox m_domain;
};

#endif // WARPX_FILTERFUNCTORS_H
