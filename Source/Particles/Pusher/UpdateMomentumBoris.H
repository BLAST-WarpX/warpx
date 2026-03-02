/* Copyright 2019 David Grote, Maxence Thevenet, Remi Lehe
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_BORIS_H_
#define WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_BORIS_H_

#include <AMReX_REAL.H>

/** \brief Push the particle's positions over one timestep,
 *    given the value of its momenta `ux`, `uy`, `uz`
 *    Note that UpdateMomentumBoris algorithm can be split into FirstHalf and SecondHalf
 *    momentum updates. FirstHalf and SecondHalf updates are constructed so that
 *    performing FirstHalf followed by SecondHalf is mathematically
 *    equivalent to a single Full update. */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void UpdateMomentumBoris(
    amrex::ParticleReal& ux, amrex::ParticleReal& uy, amrex::ParticleReal& uz,
    const amrex::ParticleReal Ex, const amrex::ParticleReal Ey, const amrex::ParticleReal Ez,
    const amrex::ParticleReal Bx, const amrex::ParticleReal By, const amrex::ParticleReal Bz,
    const amrex::ParticleReal q, const amrex::ParticleReal m, const amrex::Real dt, MomentumPushType momentum_push_type = MomentumPushType::Full)
{
    using namespace amrex::literals;

    const amrex::ParticleReal econst = 0.5_prt*q*dt/m;

    if (momentum_push_type == MomentumPushType::FirstHalf || momentum_push_type == MomentumPushType::Full) {
        // First half-push for E
        ux += econst*Ex;
        uy += econst*Ey;
        uz += econst*Ez;
    }
    // Compute temporary gamma factor
    constexpr amrex::ParticleReal inv_c2 = 1._prt/(PhysConst::c*PhysConst::c);
    const amrex::ParticleReal inv_gamma = 1._prt/std::sqrt(1._prt + (ux*ux + uy*uy + uz*uz)*inv_c2);
    // Magnetic rotation
    // - Compute temporary variables
    const amrex::Real bconst = (momentum_push_type == MomentumPushType::FirstHalf || momentum_push_type == MomentumPushType::SecondHalf) ? 0.5_rt*econst : econst;
    const amrex::ParticleReal tx = bconst*inv_gamma*Bx;
    const amrex::ParticleReal ty = bconst*inv_gamma*By;
    const amrex::ParticleReal tz = bconst*inv_gamma*Bz;
    const amrex::ParticleReal tsqi = 2._prt/(1._prt + tx*tx + ty*ty + tz*tz);
    const amrex::ParticleReal sx = tx*tsqi;
    const amrex::ParticleReal sy = ty*tsqi;
    const amrex::ParticleReal sz = tz*tsqi;
    const amrex::ParticleReal ux_p = ux + uy*tz - uz*ty;
    const amrex::ParticleReal uy_p = uy + uz*tx - ux*tz;
    const amrex::ParticleReal uz_p = uz + ux*ty - uy*tx;
    // - Update momentum
    ux += uy_p*sz - uz_p*sy;
    uy += uz_p*sx - ux_p*sz;
    uz += ux_p*sy - uy_p*sx;
    if (momentum_push_type == MomentumPushType::SecondHalf || momentum_push_type == MomentumPushType::Full) {
    // Second half-push for E
        ux += econst*Ex;
        uy += econst*Ey;
        uz += econst*Ez;
    }
}
#endif // WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_BORIS_H_
