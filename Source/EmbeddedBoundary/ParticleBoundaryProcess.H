/* Copyright 2021 Andrew Myers
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLEBOUNDARYPROCESS_H_
#define WARPX_PARTICLEBOUNDARYPROCESS_H_

#include "EmbeddedBoundary/DistanceToEB.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/Pusher/UpdatePosition.H"
#include "Particles/WarpXParticleContainer.H"

#include <ablastr/particles/NodalFieldGather.H>

#include <AMReX_Algorithm.H>
#include <AMReX_Array4.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Particle.H>
#include <AMReX_REAL.H>
#include <AMReX_RealVect.H>
#include <AMReX_Random.H>


namespace ParticleBoundaryProcess {

struct NoOp {
    template <typename PData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (const PData& /*ptd*/, int /*i*/,
                     const SetParticlePosition<PIdx>& /*setPosition*/,
                     amrex::ParticleReal /*xp*/, amrex::ParticleReal /*yp*/, amrex::ParticleReal /*zp*/,
                     amrex::Array4<const amrex::Real> const& /*distance_to_eb*/,
                     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& /*plo*/,
                     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& /*dxi*/,
                     amrex::RandomEngine const& /*engine*/) const noexcept
    {}
};

struct Absorb {
    template <typename PData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (PData& ptd, int i,
                     const SetParticlePosition<PIdx>& /*setPosition*/,
                     amrex::ParticleReal /*xp*/, amrex::ParticleReal /*yp*/, amrex::ParticleReal /*zp*/,
                     amrex::Array4<const amrex::Real> const& /*distance_to_eb*/,
                     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& /*plo*/,
                     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& /*dxi*/,
                     amrex::RandomEngine const& /*engine*/) const noexcept
    {
        amrex::ParticleIDWrapper{ptd.m_idcpu[i]}.make_invalid();
    }
};

/** \brief Specular reflection off the embedded boundary.
 *
 *  For each particle that has crossed into the EB (distance_to_eb < 0), this functor:
 *   - Bisects along the particle trajectory to find the contact point on
 *     the EB surface (where distance_to_eb = 0).
 *   - Computes the surface normal at the exact contact point.
 *   - Performs specular reflection on the momentum: u_new = u - 2*(u . n)*n.
 *   - Advances the particle from the contact point with the reflected
 *     velocity for the remaining fraction of the timestep.
 */
struct Reflect {
    amrex::Real m_dt;
    amrex::ParticleReal m_mass;

    template <typename PData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (PData& ptd, int i,
                     const SetParticlePosition<PIdx>& setPosition,
                     amrex::ParticleReal xp, amrex::ParticleReal yp, amrex::ParticleReal zp,
                     amrex::Array4<const amrex::Real> const& distance_to_eb,
                     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
                     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxi,
                     amrex::RandomEngine const& /*engine*/) const noexcept
    {
        using namespace amrex::literals;

        amrex::ParticleReal const ux = ptd.m_rdata[PIdx::ux][i];
        amrex::ParticleReal const uy = ptd.m_rdata[PIdx::uy][i];
        amrex::ParticleReal const uz = ptd.m_rdata[PIdx::uz][i];

        // Find the point of contat with the EB surface
        // Bisect along trajectory to find the fraction of dt
        // to walk backward from end-of-step to reach distance_to_eb = 0.
        amrex::Real const dt = m_dt;
        amrex::ParticleReal const mass = m_mass;
        amrex::Real const dt_fraction = amrex::bisect(0.0, 1.0,
            [=] (amrex::Real dt_frac) {
                int bi, bj, bk;
                amrex::Real bW[AMREX_SPACEDIM][2];
                amrex::ParticleReal x_temp = xp, y_temp = yp, z_temp = zp;
                UpdatePosition(x_temp, y_temp, z_temp,
                               ux, uy, uz, -dt_frac * dt, mass);
                ablastr::particles::compute_weights<amrex::IndexType::NODE>(
                    x_temp, y_temp, z_temp, plo, dxi, bi, bj, bk, bW);
                return ablastr::particles::interp_field_nodal(
                    bi, bj, bk, bW, distance_to_eb);
            });
        // Move to contact point on the EB surface
        amrex::ParticleReal x_c = xp, y_c = yp, z_c = zp;
        UpdatePosition(x_c, y_c, z_c, ux, uy, uz, -dt_fraction * dt, mass);

        // Compute the surface normal vector at the contact point, in 3D
        // Cartesian coordinates (as is the velocity below).
        auto const n3d = DistanceToEB::interp_normal(x_c, y_c, z_c, plo, dxi, distance_to_eb);
        amrex::ParticleReal const n3d_x = n3d[0];
        amrex::ParticleReal const n3d_y = n3d[1];
        amrex::ParticleReal const n3d_z = n3d[2];

        // Specular reflection: u_refl = u - 2 (u.n) n
        amrex::ParticleReal const u_dot_n = ux*n3d_x + uy*n3d_y + uz*n3d_z;
        amrex::ParticleReal const ux_refl = ux - 2.0_prt * u_dot_n * n3d_x;
        amrex::ParticleReal const uy_refl = uy - 2.0_prt * u_dot_n * n3d_y;
        amrex::ParticleReal const uz_refl = uz - 2.0_prt * u_dot_n * n3d_z;
        // Store reflected velocity
        ptd.m_rdata[PIdx::ux][i] = ux_refl;
        ptd.m_rdata[PIdx::uy][i] = uy_refl;
        ptd.m_rdata[PIdx::uz][i] = uz_refl;

        // Advance from the contact point, with the reflected velocity,
        // for the remaining fraction of the timestep
        amrex::ParticleReal x_new = x_c, y_new = y_c, z_new = z_c;
        UpdatePosition(x_new, y_new, z_new, ux_refl, uy_refl, uz_refl,
                       dt_fraction * dt, mass);
        // SetParticlePosition handles the conversion to the stored
        // (reduced-dimension) coordinates.
        setPosition(i, x_new, y_new, z_new);
    }
};

}

#endif //WARPX_PARTICLEBOUNDARYPROCESS_H_
