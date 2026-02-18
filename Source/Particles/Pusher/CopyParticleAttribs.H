/* Copyright 2020 Andrew Myers
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_PUSHER_COPYPARTICLEATTRIBS_H_
#define WARPX_PARTICLES_PUSHER_COPYPARTICLEATTRIBS_H_

#include "Particles/WarpXParticleContainer.H"

#include <AMReX_REAL.H>

#include <limits>


/** \brief Functor that creates copies of the current particle
 *         positions and momenta for later use. This is needed
 *         by the back-transformed diagnostics.
*/
struct CopyParticleAttribs
{
    GetParticlePosition<PIdx> m_get_position;

    const amrex::ParticleReal* AMREX_RESTRICT uxp = nullptr;
    const amrex::ParticleReal* AMREX_RESTRICT uyp = nullptr;
    const amrex::ParticleReal* AMREX_RESTRICT uzp = nullptr;

    amrex::ParticleReal* AMREX_RESTRICT xpold = nullptr;
    amrex::ParticleReal* AMREX_RESTRICT ypold = nullptr;
    amrex::ParticleReal* AMREX_RESTRICT zpold = nullptr;

    amrex::ParticleReal* AMREX_RESTRICT uxpold = nullptr;
    amrex::ParticleReal* AMREX_RESTRICT uypold = nullptr;
    amrex::ParticleReal* AMREX_RESTRICT uzpold = nullptr;

    CopyParticleAttribs () = default;

    /** \brief Construct a new functor
     *
     * \param a_pti iterator to the tile containing the macroparticles
     * \param a_offset offset to apply when reading / writing particle data
     *        This is needed because when we use field gather buffers we don't
     *        always start at the particle with index 0.
     */
    CopyParticleAttribs (
        const WarpXParticleContainer& /*pc*/,
        WarpXParIter& a_pti,
        long a_offset = 0) noexcept
    {
        const auto& attribs = a_pti.GetAttribs();

        uxp = attribs[PIdx::ux].dataPtr() + a_offset;
        uyp = attribs[PIdx::uy].dataPtr() + a_offset;
        uzp = attribs[PIdx::uz].dataPtr() + a_offset;

#if (AMREX_SPACEDIM >= 2)
        xpold = a_pti.GetAttribs("x_n_btd").dataPtr();
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
        ypold = a_pti.GetAttribs("y_n_btd").dataPtr();
#endif
        zpold = a_pti.GetAttribs("z_n_btd").dataPtr();
        uxpold = a_pti.GetAttribs("ux_n_btd").dataPtr();
        uypold = a_pti.GetAttribs("uy_n_btd").dataPtr();
        uzpold = a_pti.GetAttribs("uz_n_btd").dataPtr();

        m_get_position = GetParticlePosition<PIdx>(a_pti, a_offset);
    }

    /** \brief copy the position and momentum of particle i to the
     *         temporary data holder
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (const long i) const noexcept
    {
        AMREX_ASSERT(uxp != nullptr);
        AMREX_ASSERT(uyp != nullptr);
        AMREX_ASSERT(uzp != nullptr);

#if (AMREX_SPACEDIM >= 2)
        AMREX_ASSERT(xpold != nullptr);
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
        AMREX_ASSERT(ypold != nullptr);
#endif
        AMREX_ASSERT(zpold != nullptr);

        AMREX_ASSERT(uxpold != nullptr);
        AMREX_ASSERT(uypold != nullptr);
        AMREX_ASSERT(uzpold != nullptr);

        amrex::ParticleReal x, y, z;
        m_get_position(i, x, y, z);

#if (AMREX_SPACEDIM >= 2)
        xpold[i] = x;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
        ypold[i] = y;
#endif
        zpold[i] = z;

        uxpold[i] = uxp[i];
        uypold[i] = uyp[i];
        uzpold[i] = uzp[i];
    }
};

#endif // WARPX_PARTICLES_PUSHER_COPYPARTICLEATTRIBS_H_
