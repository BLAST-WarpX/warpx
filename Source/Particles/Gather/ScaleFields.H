#ifndef WARPX_PARTICLES_GATHER_SCALEFIELDS_H_
#define WARPX_PARTICLES_GATHER_SCALEFIELDS_H_

#include "Particles/WarpXParticleContainer.H"

#include <AMReX_REAL.H>

#include <limits>

/** \brief Functor that scales E and B by a factor before pushing the particles.
 *   This is used for rigid injection.
 */
struct ScaleFields
{
    bool m_do_scale;
    amrex::Real m_dt;
    amrex::Real m_z_plane_previous;
    amrex::Real m_vz_ave_boosted;
    amrex::Real m_v_boost;

    ScaleFields(bool do_scale) noexcept
        : m_do_scale(do_scale)
    {}

    ScaleFields (bool do_scale, amrex::Real dt, amrex::Real z_plane_previous,
                 amrex::Real vz_ave_boosted, amrex::Real v_boost) noexcept
    : m_do_scale(do_scale), m_dt(dt), m_z_plane_previous(z_plane_previous),
        m_vz_ave_boosted(vz_ave_boosted), m_v_boost(v_boost)
    {}

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator () (amrex::ParticleReal  /*xp*/,
                      amrex::ParticleReal  /*yp*/,
                      amrex::ParticleReal  zp,
                      amrex::ParticleReal& Exp,
                      amrex::ParticleReal& Eyp,
                      amrex::ParticleReal& Ezp,
                      amrex::ParticleReal& Bxp,
                      amrex::ParticleReal& Byp,
                      amrex::ParticleReal& Bzp) const noexcept
    {
        using namespace amrex::literals;

        if (!m_do_scale) { return; }

        // Scale the fields of particles about to cross the injection plane.
        // This only approximates what should be happening. The particles
        // should by advanced a fraction of a time step instead.
        // Scaling the fields is much easier and may be good enough.

        // The scaling factor corresponds to the fraction of time that
        // the particles spends to the right of the injection plane,
        // between (n-1/2)*dt and (n+1/2)*dt, which is the interval
        // over which the velocity is updated, in the leap-frog velocity push.
        // (Note that here, `zp` is the particle position at time n*dt)
        amrex::Real dtscale = 0.5_rt - (m_z_plane_previous - zp)/(m_vz_ave_boosted + m_v_boost)/m_dt;
        // If the particle stays to the left of the plane during the
        // whole push, simply set the scaling factor to 0, and thus
        // the velocity push leaves the velocity unchanged.
        if (dtscale < 0._rt) {
            dtscale = 0;
        }
        // Scale the fields.
        if (dtscale < 1._rt)
        {
            Exp *= dtscale;
            Eyp *= dtscale;
            Ezp *= dtscale;
            Bxp *= dtscale;
            Byp *= dtscale;
            Bzp *= dtscale;
        }
    }
};

#endif //WARPX_PARTICLES_GATHER_SCALEFIELDS_H_
