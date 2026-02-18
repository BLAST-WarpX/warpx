/* Copyright 2025 Nathan Cook
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

 // Source/Particles/Pusher/ComputeBlendedMu.H
#ifndef WARPX_PARTICLES_PUSHER_COMPUTEBLENDEDMU_H_
#define WARPX_PARTICLES_PUSHER_COMPUTEBLENDEDMU_H_

#include <AMReX_REAL.H>
#include <cmath>


/** \brief Compute the magnetic moment mu of a particle for the
 *         Blended particle pusher given particle momenta and the values of
 *         magntic field at the particle location `Bx`, `By`, `Bz` */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void ComputeBlendedMu(
    amrex::ParticleReal& mu_gc,
    const amrex::ParticleReal Bx, const amrex::ParticleReal By, const amrex::ParticleReal Bz,
    const amrex::ParticleReal ux, const amrex::ParticleReal uy, const amrex::ParticleReal uz,
    const amrex::ParticleReal m)
{

    const amrex::ParticleReal B2   = Bx*Bx + By*By + Bz*Bz;
    // Limit B2 to threshold value B2min to avoid division by zero
    constexpr amrex::ParticleReal B2min  = 1e-24_prt;     // Tesla
    const amrex::ParticleReal invB = 1._prt / std::sqrt(amrex::max(B2, B2min));

    const amrex::ParticleReal bx = Bx * invB;
    const amrex::ParticleReal by = By * invB;
    const amrex::ParticleReal bz = Bz * invB;

    const amrex::ParticleReal u_par = ux*bx + uy*by + uz*bz;

    const amrex::ParticleReal upar_x = u_par * bx;
    const amrex::ParticleReal upar_y = u_par * by;
    const amrex::ParticleReal upar_z = u_par * bz;

    const amrex::ParticleReal uperp_x = ux - upar_x;
    const amrex::ParticleReal uperp_y = uy - upar_y;
    const amrex::ParticleReal uperp_z = uz - upar_z;

    const amrex::ParticleReal uperp2 =
        uperp_x*uperp_x + uperp_y*uperp_y + uperp_z*uperp_z;

    mu_gc = 0.5_prt * m * uperp2 * invB;
}

#endif