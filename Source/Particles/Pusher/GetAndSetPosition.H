/* Copyright 2019 David Grote, Maxence Thevenet, Remi Lehe
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_PUSHER_GETANDSETPOSITION_H_
#define WARPX_PARTICLES_PUSHER_GETANDSETPOSITION_H_

#include "Particles/PIdx.H"
#include "Utils/WarpXDim.H"
#include "Utils/WarpXDimIndexing.H"

#include <AMReX.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <limits>

// Note: in the pure-SoA particle containers, the particle positions are
// stored in the first AMREX_SPACEDIM real components, in grid-axis order.
// The position component of a logical axis is therefore the axis' array
// slot, warpx::IdxMap, independent of the particle index enumerator T_PIdx
// (whose position names may differ, cf. PIdx and FieldProbePIdx).

/** \brief Extract the cartesian position coordinates of the particle
 *         p and store them in the variables `x`, `y`, `z`
 *         This version operates on a SuperParticle
 *
 * \tparam T_PIdx particle index enumerator
 * \tparam D dimensionality of the simulation
 */
template<typename T_PIdx = PIdx, warpx::Dim D = warpx::CompileDim, typename T_SuperParticle>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void get_particle_position (const T_SuperParticle& p,
                            amrex::ParticleReal& x,
                            amrex::ParticleReal& y,
                            amrex::ParticleReal& z) noexcept
{
    using namespace amrex::literals;

    static_assert(D != warpx::Dim::Runtime,
        "get_particle_position requires an explicit dimensionality");

    if constexpr (D == warpx::Dim::RZ) {
        amrex::ParticleReal const theta = p.rdata(T_PIdx::theta);
        amrex::ParticleReal const r = p.pos(warpx::IdxMap<D>::x);
        x = r*std::cos(theta);
        y = r*std::sin(theta);
        z = p.pos(warpx::IdxMap<D>::z);
    } else if constexpr (D == warpx::Dim::RCylinder) {
        amrex::ParticleReal const theta = p.rdata(T_PIdx::theta);
        amrex::ParticleReal const r = p.pos(warpx::IdxMap<D>::x);
        x = r*std::cos(theta);
        y = r*std::sin(theta);
        z = 0_prt;
    } else if constexpr (D == warpx::Dim::RSphere) {
        amrex::ParticleReal const theta = p.rdata(T_PIdx::theta);
        amrex::ParticleReal const phi = p.rdata(T_PIdx::phi);
        amrex::ParticleReal const r = p.pos(warpx::IdxMap<D>::x);
        x = r*std::cos(theta)*std::cos(phi);
        y = r*std::sin(theta)*std::cos(phi);
        z = r*std::sin(phi);
    } else {
        if constexpr (warpx::has_x<D>) {
            x = p.pos(warpx::IdxMap<D>::x);
        } else {
            x = 0_prt;
        }
        if constexpr (warpx::has_y<D>) {
            y = p.pos(warpx::IdxMap<D>::y);
        } else {
            y = 0_prt;
        }
        z = p.pos(warpx::IdxMap<D>::z);
    }
}

/** \brief Functor that can be used to extract the positions of the macroparticles
 *         inside a ParallelFor kernel
 *
 * \tparam T_PIdx particle index enumerator
 * \tparam D dimensionality of the simulation
*/
template<typename T_PIdx = PIdx, warpx::Dim D = warpx::CompileDim>
struct GetParticlePosition
{
    using RType = amrex::ParticleReal;

    const RType* AMREX_RESTRICT m_x = nullptr;
    const RType* AMREX_RESTRICT m_y = nullptr;
    const RType* AMREX_RESTRICT m_z = nullptr;
    //! azimuthal angle; only bound in the radial geometries (RZ, RCYLINDER, RSPHERE)
    const RType* AMREX_RESTRICT m_theta = nullptr;
    //! polar angle; only bound in 1D spherical geometry (RSPHERE)
    const RType* AMREX_RESTRICT m_phi = nullptr;

    static constexpr RType m_x_default = RType(0.0);
    static constexpr RType m_y_default = RType(0.0);
    static constexpr RType m_z_default = RType(0.0);

    GetParticlePosition () = default;

    /** Constructor
     *
     * \tparam ptiType the type of the particle iterator used in the constructor
     *
     * \param a_pti iterator to the tile containing the macroparticles
     * \param a_offset offset to apply to the particle indices
     */
    template <typename ptiType>
    explicit
    GetParticlePosition (const ptiType& a_pti, long a_offset = 0) noexcept
    {
        static_assert(D != warpx::Dim::Runtime,
            "GetParticlePosition requires an explicit dimensionality");

        const auto& soa = a_pti.GetStructOfArrays();

        if constexpr (warpx::has_x<D>) {
            m_x = soa.GetRealData(warpx::IdxMap<D>::x).dataPtr() + a_offset;
        }
        if constexpr (warpx::has_y<D>) {
            m_y = soa.GetRealData(warpx::IdxMap<D>::y).dataPtr() + a_offset;
        }
        if constexpr (warpx::has_z<D>) {
            m_z = soa.GetRealData(warpx::IdxMap<D>::z).dataPtr() + a_offset;
        }
        if constexpr (warpx::is_radial<D>) {
            m_theta = soa.GetRealData(T_PIdx::theta).dataPtr() + a_offset;
        }
        if constexpr (D == warpx::Dim::RSphere) {
            m_phi = soa.GetRealData(T_PIdx::phi).dataPtr() + a_offset;
        }
    }

    /** \brief Extract the cartesian position coordinates of the particle
     *         located at index `i + a_offset` and store them in the variables
     *         `x`, `y`, `z` */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (const long i, RType& x, RType& y, RType& z) const noexcept
    {
        if constexpr (D == warpx::Dim::RZ) {
            RType const r = m_x[i];
            x = r*std::cos(m_theta[i]);
            y = r*std::sin(m_theta[i]);
            z = m_z[i];
        } else if constexpr (D == warpx::Dim::RCylinder) {
            RType const r = m_x[i];
            x = r*std::cos(m_theta[i]);
            y = r*std::sin(m_theta[i]);
            z = m_z_default;
        } else if constexpr (D == warpx::Dim::RSphere) {
            RType const r = m_x[i];
            x = r*std::cos(m_theta[i])*std::cos(m_phi[i]);
            y = r*std::sin(m_theta[i])*std::cos(m_phi[i]);
            z = r*std::sin(m_phi[i]);
        } else {
            if constexpr (warpx::has_x<D>) {
                x = m_x[i];
            } else {
                x = m_x_default;
            }
            if constexpr (warpx::has_y<D>) {
                y = m_y[i];
            } else {
                y = m_y_default;
            }
            z = m_z[i];
        }
    }

    /** \brief Extract the position coordinates of the particle as stored
     *         located at index `i + a_offset` and store them in the variables
     *         `x`, `y`, `z`
     *         This is only different for the radial geometries since this
     *         returns the stored angles in that case. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void AsStored (const long i, RType& x, RType& y, RType& z) const noexcept
    {
        if constexpr (D == warpx::Dim::RZ) {
            x = m_x[i];
            y = m_theta[i];
            z = m_z[i];
        } else if constexpr (D == warpx::Dim::RCylinder) {
            x = m_x[i];
            y = m_theta[i];
            z = m_z_default;
        } else if constexpr (D == warpx::Dim::RSphere) {
            x = m_x[i];
            y = m_theta[i];
            z = m_phi[i];
        } else {
            if constexpr (warpx::has_x<D>) {
                x = m_x[i];
            } else {
                x = m_x_default;
            }
            if constexpr (warpx::has_y<D>) {
                y = m_y[i];
            } else {
                y = m_y_default;
            }
            z = m_z[i];
        }
    }
};

/** \brief Functor that can be used to modify the positions of the macroparticles,
 *         inside a ParallelFor kernel.
 *
 * \tparam T_PIdx particle index enumerator
 * \tparam D dimensionality of the simulation
 * \param a_pti iterator to the tile being modified
 * \param a_offset offset to apply to the particle indices
*/
template<typename T_PIdx = PIdx, warpx::Dim D = warpx::CompileDim>
struct SetParticlePosition
{
    using RType = amrex::ParticleReal;

    RType* AMREX_RESTRICT m_x = nullptr;
    RType* AMREX_RESTRICT m_y = nullptr;
    RType* AMREX_RESTRICT m_z = nullptr;
    //! azimuthal angle; only bound in the radial geometries (RZ, RCYLINDER, RSPHERE)
    RType* AMREX_RESTRICT m_theta = nullptr;
    //! polar angle; only bound in 1D spherical geometry (RSPHERE)
    RType* AMREX_RESTRICT m_phi = nullptr;

    template <typename ptiType>
    explicit
    SetParticlePosition (const ptiType& a_pti, long a_offset = 0) noexcept
    {
        static_assert(D != warpx::Dim::Runtime,
            "SetParticlePosition requires an explicit dimensionality");

        auto& soa = a_pti.GetStructOfArrays();

        if constexpr (warpx::has_x<D>) {
            m_x = soa.GetRealData(warpx::IdxMap<D>::x).dataPtr() + a_offset;
        }
        if constexpr (warpx::has_y<D>) {
            m_y = soa.GetRealData(warpx::IdxMap<D>::y).dataPtr() + a_offset;
        }
        if constexpr (warpx::has_z<D>) {
            m_z = soa.GetRealData(warpx::IdxMap<D>::z).dataPtr() + a_offset;
        }
        if constexpr (warpx::is_radial<D>) {
            m_theta = soa.GetRealData(T_PIdx::theta).dataPtr() + a_offset;
        }
        if constexpr (D == warpx::Dim::RSphere) {
            m_phi = soa.GetRealData(T_PIdx::phi).dataPtr() + a_offset;
        }
    }

    /** \brief Set the position of the particle at index `i + a_offset`
     *         to `x`, `y`, `z` */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (const long i, RType x, RType y, RType z) const noexcept
    {
        amrex::ignore_unused(x,y,z);
        if constexpr (D == warpx::Dim::RZ) {
            m_theta[i] = std::atan2(y, x);
            m_x[i] = std::sqrt(x*x + y*y);
            m_z[i] = z;
        } else if constexpr (D == warpx::Dim::RCylinder) {
            m_theta[i] = std::atan2(y, x);
            m_x[i] = std::sqrt(x*x + y*y);
        } else if constexpr (D == warpx::Dim::RSphere) {
            RType const rxy = std::sqrt(x*x + y*y);
            RType const r = std::sqrt(x*x + y*y + z*z);
            m_x[i] = r;
            m_theta[i] = std::atan2(y, x);
            m_phi[i] = std::atan2(z, rxy);
        } else {
            if constexpr (warpx::has_x<D>) {
                m_x[i] = x;
            }
            if constexpr (warpx::has_y<D>) {
                m_y[i] = y;
            }
            m_z[i] = z;
        }
    }

    /** \brief Set the position of the particle at index `i + a_offset`
     *         to `x`, `y`, `z`
     *         This is only different for the radial geometries since the
     *         input should contain the angles in that case. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void AsStored (const long i, RType x, RType y, RType z) const noexcept
    {
        amrex::ignore_unused(x,y,z);
        if constexpr (D == warpx::Dim::RZ) {
            m_x[i] = x;
            m_theta[i] = y;
            m_z[i] = z;
        } else if constexpr (D == warpx::Dim::RCylinder) {
            m_x[i] = x;
            m_theta[i] = y;
        } else if constexpr (D == warpx::Dim::RSphere) {
            m_x[i] = x;
            m_theta[i] = y;
            m_phi[i] = z;
        } else {
            if constexpr (warpx::has_x<D>) {
                m_x[i] = x;
            }
            if constexpr (warpx::has_y<D>) {
                m_y[i] = y;
            }
            m_z[i] = z;
        }
    }
};

#endif // WARPX_PARTICLES_PUSHER_GETANDSETPOSITION_H_
