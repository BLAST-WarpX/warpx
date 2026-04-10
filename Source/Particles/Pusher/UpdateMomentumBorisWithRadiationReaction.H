/* Copyright 2019 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_BORIS_WITHRR_H_
#define WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_BORIS_WITHRR_H_

#include "UpdateMomentumBoris.H"

#include <AMReX_REAL.H>

/**
 * Push the particle's positions over one timestep,
 * given the value of its momenta `ux`, `uy`, `uz`.
 * Includes Radiation Reaction according to
 * https://doi.org/10.1088/1367-2630/12/12/123005
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void UpdateMomentumBorisWithRadiationReaction(
    amrex::ParticleReal& ux, amrex::ParticleReal& uy, amrex::ParticleReal& uz,
    const amrex::ParticleReal Ex, const amrex::ParticleReal Ey, const amrex::ParticleReal Ez,
    const amrex::ParticleReal Bx, const amrex::ParticleReal By, const amrex::ParticleReal Bz,
    const amrex::ParticleReal q, const amrex::ParticleReal m, const amrex::Real dt, MomentumPushType momentum_push_type)

{
    using namespace amrex::literals;

    //RR algorithm needs to store old value of the normalized momentum
    const amrex::ParticleReal ux_old = ux;
    const amrex::ParticleReal uy_old = uy;
    const amrex::ParticleReal uz_old = uz;

    //Useful constant
    constexpr amrex::ParticleReal inv_c2 = 1._prt/(PhysConst::c*PhysConst::c);

    //Call to regular Boris pusher
    UpdateMomentumBoris(
        ux, uy, uz,
        Ex, Ey, Ez,
        Bx, By, Bz,
        q, m, dt, momentum_push_type );

    //Estimation of the normalized momentum at intermediate (integer) time
    const amrex::ParticleReal ux_n = (ux+ux_old)*0.5_prt;
    const amrex::ParticleReal uy_n = (uy+uy_old)*0.5_prt;
    const amrex::ParticleReal uz_n = (uz+uz_old)*0.5_prt;

    // Compute Lorentz factor (and inverse) at intermediate (integer) time
    const amrex::ParticleReal gamma_n = std::sqrt( 1._prt +
        (ux_n*ux_n + uy_n*uy_n + uz_n*uz_n)*inv_c2);
    const amrex::ParticleReal inv_gamma_n = 1.0_prt/gamma_n;

    //Estimation of the velocity at intermediate (integer) time
    const amrex::ParticleReal vx_n = ux_n*inv_gamma_n;
    const amrex::ParticleReal vy_n = uy_n*inv_gamma_n;
    const amrex::ParticleReal vz_n = uz_n*inv_gamma_n;
    const amrex::ParticleReal bx_n = vx_n/PhysConst::c;
    const amrex::ParticleReal by_n = vy_n/PhysConst::c;
    const amrex::ParticleReal bz_n = vz_n/PhysConst::c;

    //Lorentz force over charge
    const amrex::ParticleReal flx_q = (Ex + vy_n*Bz - vz_n*By);
    const amrex::ParticleReal fly_q = (Ey + vz_n*Bx - vx_n*Bz);
    const amrex::ParticleReal flz_q = (Ez + vx_n*By - vy_n*Bx);
    const amrex::ParticleReal fl_q2 = flx_q*flx_q + fly_q*fly_q + flz_q*flz_q;

    //Calculation of auxiliary quantities
    const amrex::ParticleReal bdotE = (bx_n*Ex + by_n*Ey + bz_n*Ez);
    const amrex::ParticleReal bdotE2 = bdotE*bdotE;
    const amrex::ParticleReal coeff = gamma_n*gamma_n*(fl_q2-bdotE2);

    //Radiation reaction constant
    const amrex::ParticleReal q_over_mc = q/(m*PhysConst::c);
    const amrex::ParticleReal RRcoeff = (2.0_prt/3.0_prt)*PhysConst::r_e*q_over_mc*q_over_mc;

    //Compute the components of the RR force
    const amrex::ParticleReal frx =
        RRcoeff*(PhysConst::c*(fly_q*Bz - flz_q*By) + bdotE*Ex - coeff*bx_n);
    const amrex::ParticleReal fry =
        RRcoeff*(PhysConst::c*(flz_q*Bx - flx_q*Bz) + bdotE*Ey - coeff*by_n);
    const amrex::ParticleReal frz =
        RRcoeff*(PhysConst::c*(flx_q*By - fly_q*Bx) + bdotE*Ez - coeff*bz_n);

    //Update momentum using the RR force
    const amrex::ParticleReal dt_multiplier = ((momentum_push_type == MomentumPushType::Full) ? 1.0_prt : 0.5_prt);
    ux += frx*dt_multiplier*dt;
    uy += fry*dt_multiplier*dt;
    uz += frz*dt_multiplier*dt;
}

#endif // WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_BORIS_WITHRR_H_
