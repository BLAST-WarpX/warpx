/* Copyright 2019 David Grote, Maxence Thevenet, Remi Lehe
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_BLENDED_H_
#define WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_BLENDED_H_

#include <AMReX_REAL.H>
#include "Utils/WarpXConst.H"

/** \brief Update the particle's momentum over a timestep,
 *    given the value of its momenta `ux`, `uy`, `uz` 
 *    fields Ex, Ey, Ez, Bx, By, Bz, gradients gradBx, gradBy, gradBz 
 *    and pre-computed alpha/mu */

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void UpdateMomentumBlended(
    amrex::ParticleReal& ux, amrex::ParticleReal& uy, amrex::ParticleReal& uz,
    const amrex::ParticleReal Ex, const amrex::ParticleReal Ey, const amrex::ParticleReal Ez,
    const amrex::ParticleReal Bx, const amrex::ParticleReal By, const amrex::ParticleReal Bz,
    const amrex::ParticleReal q, const amrex::ParticleReal m, const amrex::Real dt,
    const amrex::ParticleReal gradBx, const amrex::ParticleReal gradBy, const amrex::ParticleReal gradBz,
    const amrex::ParticleReal alpha_gc, const amrex::ParticleReal mu_gc)
{
    using namespace amrex::literals;

    const amrex::ParticleReal econst = 0.5_prt*q*dt/m;
    
    // First half-push for E
    ux += econst*Ex;
    uy += econst*Ey;
    uz += econst*Ez;

    // Compute temporary gamma factor and |B|
    constexpr amrex::ParticleReal inv_c2 = 1._prt/(PhysConst::c*PhysConst::c);
    const amrex::ParticleReal inv_gamma = 1._prt/std::sqrt(1._prt + (ux*ux + uy*uy + uz*uz)*inv_c2);
    const amrex::ParticleReal Bmag = std::sqrt(Bx*Bx + By*By + Bz*Bz);

    // Limit Bmin to avoid division by zero
    constexpr amrex::ParticleReal Bmin  = 1e-12_prt;     // Tesla
    const amrex::Real invB = 1.0_prt / amrex::max(Bmag, Bmin);

    const amrex::ParticleReal bx = Bx*invB;
    const amrex::ParticleReal by = By*invB;
    const amrex::ParticleReal bz = Bz*invB;

    //compute parallel and perpendicular momenta
    const amrex::ParticleReal u_dot_b = ux*bx + uy*by + uz*bz;
    const amrex::ParticleReal upar_x  = u_dot_b * bx;
    const amrex::ParticleReal upar_y  = u_dot_b * by;
    const amrex::ParticleReal upar_z  = u_dot_b * bz;

    const amrex::ParticleReal uperp_x = ux - upar_x;
    const amrex::ParticleReal uperp_y = uy - upar_y;
    const amrex::ParticleReal uperp_z = uz - upar_z;

    // Magnetic rotation
    // - Compute temporary variables
    const amrex::ParticleReal tx = econst*inv_gamma*Bx;
    const amrex::ParticleReal ty = econst*inv_gamma*By;
    const amrex::ParticleReal tz = econst*inv_gamma*Bz;
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
    // Second half-push for E
    ux += econst*Ex;
    uy += econst*Ey;
    uz += econst*Ez;

    // Mirror force correction to momentum
    const amrex::ParticleReal bdotGradB = bx*gradBx + by*gradBy + bz*gradBz;
    const amrex::ParticleReal coeff = -1._prt*(mu_gc/m) * bdotGradB * dt;

    const amrex::ParticleReal dux = bx*coeff;
    const amrex::ParticleReal duy = by*coeff;
    const amrex::ParticleReal duz = bz*coeff;

    ux += (1._prt - alpha_gc)*dux;
    uy += (1._prt - alpha_gc)*duy;
    uz += (1._prt - alpha_gc)*duz;

}

#endif // WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_BLENDED_H_
