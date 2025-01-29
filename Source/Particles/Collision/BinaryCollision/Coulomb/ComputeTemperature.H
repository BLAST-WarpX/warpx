/* Copyright 2019-2020 Andrew Myers, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_COMPUTE_TEMPERATURE_H_
#define WARPX_PARTICLES_COLLISION_COMPUTE_TEMPERATURE_H_

#include "Utils/WarpXConst.H"


template <typename T_index, typename T_R>
AMREX_GPU_HOST_DEVICE
T_R ComputeTemperature (
    T_index const Is, T_index const Ie, T_index const * AMREX_RESTRICT I,
    T_R const * AMREX_RESTRICT w,
    T_R const * AMREX_RESTRICT ux, T_R const * AMREX_RESTRICT uy, T_R const * AMREX_RESTRICT uz,
    T_R const m )
{

    T_R constexpr inv_c2 = T_R(1.0) / ( PhysConst::c * PhysConst::c );

    const int N = Ie - Is;
    if ( N == 0 ) { return T_R(0.0); }

    T_R vx = T_R(0.0);    T_R vy = T_R(0.0);
    T_R vz = T_R(0.0);    T_R vs = T_R(0.0);
    T_R gm = T_R(0.0);    T_R us = T_R(0.0);
    T_R wtot = T_R(0.0);

    for (int i = Is; i < static_cast<int>(Ie); ++i)
    {
        us = ( ux[ I[i] ] * ux[ I[i] ] +
               uy[ I[i] ] * uy[ I[i] ] +
               uz[ I[i] ] * uz[ I[i] ] );
        gm = std::sqrt( T_R(1.0) + us*inv_c2 );
        wtot += w[ I[i] ];
        vx += w[ I[i] ] * ux[ I[i] ] / gm;
        vy += w[ I[i] ] * uy[ I[i] ] / gm;
        vz += w[ I[i] ] * uz[ I[i] ] / gm;
        vs += w[ I[i] ] * us / gm / gm;
    }

    vx = vx / wtot;    vy = vy / wtot;
    vz = vz / wtot;    vs = vs / wtot;

    return m/T_R(3.0)*(vs-(vx*vx+vy*vy+vz*vz));
}

#endif // WARPX_PARTICLES_COLLISION_COMPUTE_TEMPERATURE_H_
