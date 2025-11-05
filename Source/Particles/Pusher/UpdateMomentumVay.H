/* Copyright 2019 David Grote, Maxence Thevenet, Remi Lehe
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_VAY_H_
#define WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_VAY_H_

#include "Utils/WarpXConst.H"

#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>


/** \brief Push the particle's positions over one timestep,
 *    given the value of its momenta `ux`, `uy`, `uz` */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void UpdateMomentumVay(
    amrex::ParticleReal& ux, amrex::ParticleReal& uy, amrex::ParticleReal& uz,
    const amrex::ParticleReal Ex, const amrex::ParticleReal Ey, const amrex::ParticleReal Ez,
    const amrex::ParticleReal Bx, const amrex::ParticleReal By, const amrex::ParticleReal Bz,
    const amrex::ParticleReal q, const amrex::ParticleReal m, const amrex::Real dt )
{
    using namespace amrex::literals;

    // Constants
    const amrex::ParticleReal econst = q*dt/m;
    const amrex::ParticleReal bconst = 0.5_prt*q*dt/m;
    // Compute initial gamma
    const amrex::ParticleReal inv_gamma = 1._prt/std::sqrt(1._prt + (ux*ux + uy*uy + uz*uz)*PhysConst::inv_c2_v<amrex::ParticleReal>);
    // Get tau
    const amrex::ParticleReal taux = bconst*Bx;
    const amrex::ParticleReal tauy = bconst*By;
    const amrex::ParticleReal tauz = bconst*Bz;
    const amrex::ParticleReal tausq = taux*taux+tauy*tauy+tauz*tauz;
    // Get U', gamma'^2
    const amrex::ParticleReal uxpr = ux + econst*Ex + (uy*tauz-uz*tauy)*inv_gamma;
    const amrex::ParticleReal uypr = uy + econst*Ey + (uz*taux-ux*tauz)*inv_gamma;
    const amrex::ParticleReal uzpr = uz + econst*Ez + (ux*tauy-uy*taux)*inv_gamma;
    const amrex::ParticleReal gprsq = (1._prt + (uxpr*uxpr + uypr*uypr + uzpr*uzpr)*PhysConst::inv_c2_v<amrex::ParticleReal>);
    // Get u*
    const amrex::ParticleReal ust = (uxpr*taux + uypr*tauy + uzpr*tauz)*PhysConst::inv_c_v<amrex::ParticleReal>;
    // Get new gamma
    const amrex::ParticleReal sigma = gprsq-tausq;
    const amrex::ParticleReal gisq = 2._prt/(sigma + std::sqrt(sigma*sigma + 4._prt*(tausq + ust*ust)) );
    // Get t, s
    const amrex::ParticleReal bg = bconst*std::sqrt(gisq);
    const amrex::ParticleReal tx = bg*Bx;
    const amrex::ParticleReal ty = bg*By;
    const amrex::ParticleReal tz = bg*Bz;
    const amrex::ParticleReal s = 1._prt/(1._prt+tausq*gisq);
    // Get t.u'
    const amrex::ParticleReal tu = tx*uxpr + ty*uypr + tz*uzpr;
    // Get new U
    ux = s*(uxpr+tx*tu+uypr*tz-uzpr*ty);
    uy = s*(uypr+ty*tu+uzpr*tx-uxpr*tz);
    uz = s*(uzpr+tz*tu+uxpr*ty-uypr*tx);
}

#endif // WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_VAY_H_
