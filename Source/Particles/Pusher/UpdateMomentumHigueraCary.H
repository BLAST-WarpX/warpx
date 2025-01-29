/* Copyright 2019 Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_HIGUERACARY_H_
#define WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_HIGUERACARY_H_

#include "Utils/WarpXConst.H"

#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>


/**
 * \brief Push the particle's positions over one timestep,
 * given the value of its momenta `ux`, `uy`, `uz`
 */
template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void UpdateMomentumHigueraCary(
    T& ux, T& uy, T& uz,
    const T Ex, const T Ey, const T Ez,
    const T Bx, const T By, const T Bz,
    const T q, const T m, const amrex::Real dt )
{
    using namespace amrex::literals;

    // Constants
    const T qmt = 0.5_prt*q*dt/m;
    constexpr T invclight = 1._prt/PhysConst::c;
    constexpr T invclightsq = 1._prt/(PhysConst::c*PhysConst::c);
    // Compute u_minus
    const T umx = ux + qmt*Ex;
    const T umy = uy + qmt*Ey;
    const T umz = uz + qmt*Ez;
    // Compute gamma squared of u_minus
    T gamma = 1._prt + (umx*umx + umy*umy + umz*umz)*invclightsq;
    // Compute beta and betam squared
    const T betax = qmt*Bx;
    const T betay = qmt*By;
    const T betaz = qmt*Bz;
    const T betam = betax*betax + betay*betay + betaz*betaz;
    // Compute sigma
    const T sigma = gamma - betam;
    // Get u*
    const T ust = (umx*betax + umy*betay + umz*betaz)*invclight;
    // Get new gamma inverse
    gamma = 1._prt/std::sqrt(0.5_prt*(sigma + std::sqrt(sigma*sigma + 4._prt*(betam + ust*ust)) ));
    // Compute t
    const T tx = gamma*betax;
    const T ty = gamma*betay;
    const T tz = gamma*betaz;
    // Compute s
    const T s = 1._prt/(1._prt+(tx*tx + ty*ty + tz*tz));
    // Compute um dot t
    const T umt = umx*tx + umy*ty + umz*tz;
    // Compute u_plus
    const T upx = s*( umx + umt*tx + umy*tz - umz*ty );
    const T upy = s*( umy + umt*ty + umz*tx - umx*tz );
    const T upz = s*( umz + umt*tz + umx*ty - umy*tx );
    // Get new u
    ux = upx + qmt*Ex + upy*tz - upz*ty;
    uy = upy + qmt*Ey + upz*tx - upx*tz;
    uz = upz + qmt*Ez + upx*ty - upy*tx;
}

#endif // WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_HIGUERACARY_H_
