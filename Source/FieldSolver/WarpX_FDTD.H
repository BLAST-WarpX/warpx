/* Copyright 2019 Axel Huebl, David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_FDTD_H_
#define WARPX_FDTD_H_

#include <AMReX.H>
#include <AMReX_FArrayBox.H>

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void warpx_computedivb(int i, int j, int k, int dcomp,
                       amrex::Array4<amrex::Real> const& divB,
                       amrex::Array4<amrex::Real const> const& Bx,
                       amrex::Array4<amrex::Real const> const& By,
                       amrex::Array4<amrex::Real const> const& Bz,
                       amrex::Real dxinv,
                       amrex::Real dyinv,
                       amrex::Real dzinv
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
                       , amrex::Real const rmin
#endif
                       )
{

    using namespace amrex;

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    const amrex::Real ru = 1._rt + 0.5_rt/(rmin*dxinv + i + 0.5_rt);
    const amrex::Real rd = 1._rt - 0.5_rt/(rmin*dxinv + i + 0.5_rt);
#endif

#if defined WARPX_DIM_3D
    divB(i,j,k,dcomp) = (Bx(i+1,j  ,k  ) - Bx(i,j,k))*dxinv
               +        (By(i  ,j+1,k  ) - By(i,j,k))*dyinv
               +        (Bz(i  ,j  ,k+1) - Bz(i,j,k))*dzinv;
#elif defined WARPX_DIM_XZ
    divB(i,j,0,dcomp) = (Bx(i+1,j  ,0) - Bx(i,j,0))*dxinv
               +        (Bz(i  ,j+1,0) - Bz(i,j,0))*dzinv;
    amrex::ignore_unused(k, By, dyinv);
#elif defined WARPX_DIM_1D_Z
    divB(i,0,0,dcomp) = (Bz(i+1,0  ,0) - Bz(i,0,0))*dzinv;
    amrex::ignore_unused(j, Bx, dxinv);
    amrex::ignore_unused(k, By, dyinv);
#elif defined(WARPX_DIM_RCYLINDER)
    divB(i,0,0,dcomp) = (ru*Bx(i+1,0  ,0) - rd*Bx(i,0,0))*dxinv;
    amrex::ignore_unused(j, By, dyinv);
    amrex::ignore_unused(k, Bz, dzinv);
#elif defined(WARPX_DIM_RSPHERE)
    divB(i,0,0,dcomp) = (ru*ru*Bx(i+1,0  ,0) - rd*rd*Bx(i,0,0))*dxinv;
    amrex::ignore_unused(j, By, dyinv);
    amrex::ignore_unused(k, Bz, dzinv);
#elif defined WARPX_DIM_RZ
    divB(i,j,0,dcomp) = (ru*Bx(i+1,j,0) - rd*Bx(i,j,0))*dxinv
                       + (Bz(i,j+1,0) - Bz(i,j,0))*dzinv;
    amrex::ignore_unused(k, By, dyinv);
#endif
}

#endif
