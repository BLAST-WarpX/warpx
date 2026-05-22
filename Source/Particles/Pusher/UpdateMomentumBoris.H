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
    constexpr auto inv_c2 = PhysConst::inv_c2_v< amrex::ParticleReal>;
    const amrex::ParticleReal inv_gamma = 1._prt/std::sqrt(1._prt + (ux*ux + uy*uy + uz*uz)*inv_c2);
    // Magnetic rotation
    // - Compute temporary variables
    amrex::ParticleReal tx = econst*inv_gamma*Bx;
    amrex::ParticleReal ty = econst*inv_gamma*By;
    amrex::ParticleReal tz = econst*inv_gamma*Bz;
    // For half pushes, rescale t so that two half rotations compose exactly into the
    // full Boris rotation. The Boris formula rotates by an angle alpha with
    // tan(alpha/2) = |t|; for two half rotations to compose to alpha_full we need
    // |t_half| = tan(alpha_full/4) = (sqrt(1 + |t_full|^2) - 1) / |t_full|.
    if (momentum_push_type == MomentumPushType::FirstHalf || momentum_push_type == MomentumPushType::SecondHalf) {
        const amrex::ParticleReal tsq = tx*tx + ty*ty + tz*tz;
        const amrex::ParticleReal factor = (tsq > 0._prt) ? (std::sqrt(1._prt + tsq) - 1._prt) / tsq : 0.5_prt;
        tx *= factor;
        ty *= factor;
        tz *= factor;
    }
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
