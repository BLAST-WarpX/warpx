/* Copyright 2025 Nathan Cook
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

 // Source/Particles/Pusher/ComputeBlendedAlpha.H
#ifndef WARPX_PARTICLES_PUSHER_COMPUTEBLENDEDALPHA_H_
#define WARPX_PARTICLES_PUSHER_COMPUTEBLENDEDALPHA_H_

#include <AMReX_REAL.H>
#include <cmath>


/** \brief Compute the blending coefficient alpha for the 
 *         Blended particle pusher given the values of
 *         magntic field at the particle location `Bx`, `By`, `Bz` */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void ComputeBlendedAlpha(
    amrex::ParticleReal& alpha_gc,
    const amrex::ParticleReal Bx, const amrex::ParticleReal By, const amrex::ParticleReal Bz,
    const amrex::ParticleReal q, const amrex::ParticleReal m, const amrex::Real dt )
{
    const amrex::ParticleReal B2 = Bx*Bx + By*By + Bz*Bz;
    const amrex::ParticleReal Bmag = std::sqrt(B2);
    const amrex::ParticleReal omega_c = std::abs(q)*Bmag/m;

    alpha_gc = 1._prt/std::sqrt(1._prt + (0.5_prt*omega_c*dt)*(0.5_prt*omega_c*dt));

}

#endif