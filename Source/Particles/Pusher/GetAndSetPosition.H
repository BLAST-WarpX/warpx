/* Copyright 2019 David Grote, Maxence Thevenet, Remi Lehe
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_PUSHER_GETANDSETPOSITION_H_
#define WARPX_PARTICLES_PUSHER_GETANDSETPOSITION_H_

#include "Particles/WarpXParticleContainer.H"
#include "Particles/NamedComponentParticleContainer.H"

#include <AMReX.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <limits>

/** \brief Extract the cartesian position coordinates of the particle
 *         p and store them in the variables `x`, `y`, `z`
 *         This version operates on a SuperParticle
 *
 * \tparam T_PIdx particle index enumerator
 */
template<typename T_PIdx = PIdx>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void get_particle_position (const WarpXParticleContainer::SuperParticleType& p,
                            amrex::ParticleReal& x,
                            amrex::ParticleReal& y,
                            amrex::ParticleReal& z) noexcept
{
    using namespace amrex::literals;

#if defined(WARPX_DIM_RZ)
    amrex::ParticleReal const theta = p.rdata(T_PIdx::theta);
    amrex::ParticleReal const r = p.pos(T_PIdx::x);
    x = r*std::cos(theta);
    y = r*std::sin(theta);
    z = p.pos(PIdx::z);
#elif defined(WARPX_DIM_3D)
    x = p.pos(PIdx::x);
    y = p.pos(PIdx::y);
    z = p.pos(PIdx::z);
#elif defined(WARPX_DIM_XZ)
    x = p.pos(PIdx::x);
    y = 0_prt;
    z = p.pos(PIdx::z);
#else
    x = 0_prt;
    y = 0_prt;
    z = p.pos(PIdx::z);
#endif
}

/** \brief Functor that can be used to extract the positions of the macroparticles
 *         inside a ParallelFor kernel
 *
 * \tparam T_PIdx particle index enumerator
*/
template<typename T_PIdx = PIdx>
struct GetParticlePosition
{
    using RType = amrex::ParticleReal;

    const RType* AMREX_RESTRICT m_x = nullptr;
    const RType* AMREX_RESTRICT m_y = nullptr;
    const RType* AMREX_RESTRICT m_z = nullptr;

#if defined(WARPX_DIM_RZ)
    const RType* m_theta = nullptr;
#endif

    static constexpr RType m_x_default = RType(0.0);
    static constexpr RType m_y_default = RType(0.0);

    GetParticlePosition () = default;

    /** Constructor
     *
     * \tparam ptiType the type of the particle iterator used in the constructor
     *
     * \param a_pti iterator to the tile containing the macroparticles
     * \param a_offset offset to apply to the particle indices
     */
    template <typename ptiType>
    GetParticlePosition (const ptiType& a_pti, long a_offset = 0) noexcept
    {
        const auto& soa = a_pti.GetStructOfArrays();

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
        m_x = soa.GetRealData(PIdx::x).dataPtr() + a_offset;
        m_z = soa.GetRealData(PIdx::z).dataPtr() + a_offset;
#elif defined(WARPX_DIM_3D)
        m_x = soa.GetRealData(PIdx::x).dataPtr() + a_offset;
        m_y = soa.GetRealData(PIdx::y).dataPtr() + a_offset;
        m_z = soa.GetRealData(PIdx::z).dataPtr() + a_offset;
#elif defined(WARPX_DIM_1D_Z)
        m_z = soa.GetRealData(PIdx::z).dataPtr() + a_offset;
#endif
#if defined(WARPX_DIM_RZ)
        m_theta = soa.GetRealData(T_PIdx::theta).dataPtr() + a_offset;
#endif
    }

    /** \brief Extract the cartesian position coordinates of the particle
     *         located at index `i + a_offset` and store them in the variables
     *         `x`, `y`, `z` */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (const long i, RType& x, RType& y, RType& z) const noexcept
    {
#if defined(WARPX_DIM_RZ)
        RType const r = m_x[i];
        x = r*std::cos(m_theta[i]);
        y = r*std::sin(m_theta[i]);
        z = m_z[i];
#elif defined(WARPX_DIM_3D)
        x = m_x[i];
        y = m_y[i];
        z = m_z[i];
#elif defined(WARPX_DIM_XZ)
        x = m_x[i];
        y = m_y_default;
        z = m_z[i];
#elif defined(WARPX_DIM_1D_Z)
        x = m_x_default;
        y = m_y_default;
        z = m_z[i];
#endif
    }

    /** \brief Extract the position coordinates of the particle as stored
     *         located at index `i + a_offset` and store them in the variables
     *         `x`, `y`, `z`
     *         This is only different for RZ since this returns (r, theta, z)
     *         in that case. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void AsStored (const long i, RType& x, RType& y, RType& z) const noexcept
    {
#if defined(WARPX_DIM_RZ)
        x = m_x[i];
        y = m_theta[i];
        z = m_z[i];
#elif defined(WARPX_DIM_3D)
        x = m_x[i];
        y = m_y[i];
        z = m_z[i];
#elif defined(WARPX_DIM_XZ)
        x = m_x[i];
        y = m_y_default;
        z = m_z[i];
#elif defined(WARPX_DIM_1D_Z)
        x = m_x_default;
        y = m_y_default;
        z = m_z[i];
#endif
    }
};

/** \brief Functor that can be used to modify the positions of the macroparticles,
 *         inside a ParallelFor kernel.
 *
 * \tparam T_PIdx particle index enumerator
 * \param a_pti iterator to the tile being modified
 * \param a_offset offset to apply to the particle indices
*/
template<typename T_PIdx = PIdx>
struct SetParticlePosition
{
    using RType = amrex::ParticleReal;

#if defined(WARPX_DIM_3D)
    RType* AMREX_RESTRICT m_x;
    RType* AMREX_RESTRICT m_y;
    RType* AMREX_RESTRICT m_z;
#elif defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
    RType* AMREX_RESTRICT m_x;
    RType* AMREX_RESTRICT m_z;
#elif defined(WARPX_DIM_1D_Z)
    RType* AMREX_RESTRICT m_z;
#endif

#if defined(WARPX_DIM_RZ)
    RType* AMREX_RESTRICT m_theta;
#endif

    template <typename ptiType>
    SetParticlePosition (const ptiType& a_pti, long a_offset = 0) noexcept
    {
        auto& soa = a_pti.GetStructOfArrays();
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
        m_x = soa.GetRealData(PIdx::x).dataPtr() + a_offset;
        m_z = soa.GetRealData(PIdx::z).dataPtr() + a_offset;
#elif defined(WARPX_DIM_3D)
        m_x = soa.GetRealData(PIdx::x).dataPtr() + a_offset;
        m_y = soa.GetRealData(PIdx::y).dataPtr() + a_offset;
        m_z = soa.GetRealData(PIdx::z).dataPtr() + a_offset;
#elif defined(WARPX_DIM_1D_Z)
        m_z = soa.GetRealData(PIdx::z).dataPtr() + a_offset;
#endif
#if defined(WARPX_DIM_RZ)
        m_theta = soa.GetRealData(T_PIdx::theta).dataPtr() + a_offset;
#endif
    }

    /** \brief Set the position of the particle at index `i + a_offset`
     *         to `x`, `y`, `z` */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (const long i, RType x, RType y, RType z) const noexcept
    {
        amrex::ignore_unused(x,y,z);
#if defined(WARPX_DIM_RZ)
        m_theta[i] = std::atan2(y, x);
        m_x[i] = std::sqrt(x*x + y*y);
        m_z[i] = z;
#elif defined(WARPX_DIM_3D)
        m_x[i] = x;
        m_y[i] = y;
        m_z[i] = z;
#elif defined(WARPX_DIM_XZ)
        m_x[i] = x;
        m_z[i] = z;
#elif defined(WARPX_DIM_1D_Z)
        m_z[i] = z;
#endif
    }

    /** \brief Set the position of the particle at index `i + a_offset`
     *         to `x`, `y`, `z`
     *         This is only different for RZ since the input should
     *         be (r, theta, z) in that case. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void AsStored (const long i, RType x, RType y, RType z) const noexcept
    {
        amrex::ignore_unused(x,y,z);
#if defined(WARPX_DIM_RZ)
        m_x[i] = x;
        m_theta[i] = y;
        m_z[i] = z;
#elif defined(WARPX_DIM_3D)
        m_x[i] = x;
        m_y[i] = y;
        m_z[i] = z;
#elif defined(WARPX_DIM_XZ)
        m_x[i] = x;
        m_z[i] = z;
#elif defined(WARPX_DIM_1D_Z)
        m_z[i] = z;
#endif
    }
};

#endif // WARPX_PARTICLES_PUSHER_GETANDSETPOSITION_H_
