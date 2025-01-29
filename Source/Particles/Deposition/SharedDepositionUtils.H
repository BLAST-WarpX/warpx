/* Copyright 2022 Noah Kaplan, Andrew Myers, Phil Miller
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_SHAREDDEPOSITIONUTILS_H_
#define WARPX_SHAREDDEPOSITIONUTILS_H_

#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/ShapeFactors.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#ifdef WARPX_DIM_RZ
#   include "Utils/WarpX_Complex.H"
#endif

#include <AMReX.H>

/*
 * \brief gets the maximum width, height, or length of a tilebox. In number of cells.
 * \param nCells : Number of cells in the direction to be considered
 * \param tilesize : The 1D tilesize in the direction to be considered
 */
AMREX_FORCE_INLINE
int getMaxTboxAlongDim (int nCells, int tilesize){
    int maxTilesize = 0;
    const int nTiles = nCells / tilesize;
    const int remainder = nCells % tilesize;
    maxTilesize = tilesize + int(std::ceil((amrex::Real) remainder / nTiles));
    return maxTilesize;
}

/*
 * \brief atomically add the values from the local deposition buffer back to the global array.
 * \param bx : Box defining the index space of the local buffer
 * \param global : The global array
 * \param local : The local array
 */
#if defined(AMREX_USE_HIP) || defined(AMREX_USE_CUDA)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void addLocalToGlobal (const amrex::Box& bx,
                       const amrex::Array4<amrex::Real>& global,
                       const amrex::Array4<amrex::Real>& local) noexcept
{
    using namespace amrex::literals;

    const auto lo  = amrex::lbound(bx);
    const auto len = amrex::length(bx);
    for (int icell = threadIdx.x; icell < bx.numPts(); icell += blockDim.x)
    {
        int k =  icell / (len.x*len.y);
        int j = (icell - k*(len.x*len.y)) /   len.x;
        int i = (icell - k*(len.x*len.y)) - j*len.x;
        i += lo.x;
        j += lo.y;
        k += lo.z;
        if (amrex::Math::abs(local(i, j, k)) > 0.0_rt) {
            amrex::Gpu::Atomic::AddNoRet( &global(i, j, k), local(i, j, k));
        }
    }
}
#endif

#if defined(AMREX_USE_HIP) || defined(AMREX_USE_CUDA)
template <int depos_order>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void depositComponent (const GetParticlePosition<PIdx>& GetPosition,
                       const amrex::ParticleReal * const wp,
                       const amrex::ParticleReal * const uxp,
                       const amrex::ParticleReal * const uyp,
                       const amrex::ParticleReal * const uzp,
                       const int* ion_lev,
                       amrex::Array4<amrex::Real> const& j_buff,
                       amrex::IntVect const j_type,
                       const amrex::Real relative_time,
                       const amrex::XDim3 dinv,
                       const amrex::XDim3 xyzmin,
                       const amrex::Dim3 lo,
                       const amrex::Real q,
                       const int n_rz_azimuthal_modes,
                       const unsigned int ip,
                       const int zdir, const int NODE, const int CELL, const int dir)
{
    using namespace amrex::literals;

#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(n_rz_azimuthal_modes);
#endif

    // Whether ion_lev is a null pointer (do_ionization=0) or a real pointer
    // (do_ionization=1)
    const bool do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

    const amrex::Real clightsq = 1.0_rt/PhysConst::c/PhysConst::c;

    // --- Get particle quantities
    const amrex::Real gaminv = 1.0_rt/std::sqrt(1.0_rt + uxp[ip]*uxp[ip]*clightsq
                                                + uyp[ip]*uyp[ip]*clightsq
                                                + uzp[ip]*uzp[ip]*clightsq);
    amrex::Real wq  = q*wp[ip];
    if (do_ionization){
        wq *= ion_lev[ip];
    }

    amrex::ParticleReal xp, yp, zp;
    GetPosition(ip, xp, yp, zp);

    const amrex::Real vx = uxp[ip]*gaminv;
    const amrex::Real vy = uyp[ip]*gaminv;
    const amrex::Real vz = uzp[ip]*gaminv;
    // pcurrent is the particle current in the deposited direction
#if defined(WARPX_DIM_RZ)
    // In RZ, wqx is actually wqr, and wqy is wqtheta
    // Convert to cylindrical at the mid point
    const amrex::Real xpmid = xp + relative_time*vx;
    const amrex::Real ypmid = yp + relative_time*vy;
    const amrex::Real rpmid = std::sqrt(xpmid*xpmid + ypmid*ypmid);
    amrex::Real costheta;
    amrex::Real sintheta;
    if (rpmid > 0._rt) {
        costheta = xpmid/rpmid;
        sintheta = ypmid/rpmid;
    } else {
        costheta = 1._rt;
        sintheta = 0._rt;
    }
    const Complex xy0 = Complex{costheta, sintheta};
    const amrex::Real wqx = wq*invvol*(+vx*costheta + vy*sintheta);
    const amrex::Real wqy = wq*invvol*(-vx*sintheta + vy*costheta);
#else
    const amrex::Real wqx = wq*invvol*vx;
    const amrex::Real wqy = wq*invvol*vy;
#endif
    const amrex::Real wqz = wq*invvol*vz;

    amrex::Real pcurrent = 0.0;
    if (dir == 0) {
        pcurrent = wqx;
    } else if (dir == 1) {
        pcurrent = wqy;
    } else if (dir == 2) {
        pcurrent = wqz;
    }

    // --- Compute shape factors
    Compute_shape_factor< depos_order > const compute_shape_factor;
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_3D)

    // x direction
    // Get particle position after 1/2 push back in position
#if defined(WARPX_DIM_RZ)
    // Keep these double to avoid bug in single precision
    const double xmid = (rpmid - xyzmin.x)*dinv.x;
#else
    const double xmid = ((xp - xyzmin.x) + relative_time*vx)*dinv.x;
#endif
    // j_j[xyz] leftmost grid point in x that the particle touches for the centering of each current
    // sx_j[xyz] shape factor along x for the centering of each current
    // There are only two possible centerings, node or cell centered, so at most only two shape factor
    // arrays will be needed.
    // Keep these double to avoid bug in single precision
    double sx_node[depos_order + 1] = {0.};
    double sx_cell[depos_order + 1] = {0.};
    int j_node = 0;
    int j_cell = 0;
    if (j_type[0] == NODE) {
        j_node = compute_shape_factor(sx_node, xmid);
    }
    if (j_type[0] == CELL) {
        j_cell = compute_shape_factor(sx_cell, xmid - 0.5);
    }

    amrex::Real sx_j[depos_order + 1] = {0._rt};
    for (int ix=0; ix<=depos_order; ix++)
    {
        sx_j[ix] = ((j_type[0] == NODE) ? amrex::Real(sx_node[ix]) : amrex::Real(sx_cell[ix]));
    }

    int const j_j = ((j_type[0] == NODE) ? j_node : j_cell);
#endif //AMREX_SPACEDIM >= 2

#if defined(WARPX_DIM_3D)
    // y direction
    // Keep these double to avoid bug in single precision
    const double ymid = ((yp - xyzmin.y) + relative_time*vy)*dinv.y;
    double sy_node[depos_order + 1] = {0.};
    double sy_cell[depos_order + 1] = {0.};
    int k_node = 0;
    int k_cell = 0;
    if (j_type[1] == NODE) {
        k_node = compute_shape_factor(sy_node, ymid);
    }
    if (j_type[1] == CELL) {
        k_cell = compute_shape_factor(sy_cell, ymid - 0.5);
    }
    amrex::Real sy_j[depos_order + 1] = {0._rt};
    for (int iy=0; iy<=depos_order; iy++)
    {
        sy_j[iy] = ((j_type[1] == NODE) ? amrex::Real(sy_node[iy]) : amrex::Real(sy_cell[iy]));
    }
    int const k_j = ((j_type[1] == NODE) ? k_node : k_cell);
#endif

    // z direction
    // Keep these double to avoid bug in single precision
    const double zmid = ((zp - xyzmin.z) + relative_time*vz)*dinv.z;
    double sz_node[depos_order + 1] = {0.};
    double sz_cell[depos_order + 1] = {0.};
    int l_node = 0;
    int l_cell = 0;
    if (j_type[zdir] == NODE) {
        l_node = compute_shape_factor(sz_node, zmid);
    }
    if (j_type[zdir] == CELL) {
        l_cell = compute_shape_factor(sz_cell, zmid - 0.5);
    }
    amrex::Real sz_j[depos_order + 1] = {0._rt};
    for (int iz=0; iz<=depos_order; iz++)
    {
        sz_j[iz] = ((j_type[zdir] == NODE) ? amrex::Real(sz_node[iz]) : amrex::Real(sz_cell[iz]));
    }
    int const l_j = ((j_type[zdir] == NODE) ? l_node : l_cell);

    // Deposit current into j_buff
#if defined(WARPX_DIM_1D_Z)
    for (int iz=0; iz<=depos_order; iz++){
        amrex::Gpu::Atomic::AddNoRet(
                                     &j_buff(lo.x+l_j+iz, 0, 0, 0),
                                     sz_j[iz]*pcurrent);
    }
#endif
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    for (int iz=0; iz<=depos_order; iz++){
        for (int ix=0; ix<=depos_order; ix++){
            amrex::Gpu::Atomic::AddNoRet(
                                         &j_buff(lo.x+j_j+ix, lo.y+l_j+iz, 0, 0),
                                         sx_j[ix]*sz_j[iz]*pcurrent);
#if defined(WARPX_DIM_RZ)
            Complex xy = xy0; // Note that xy is equal to e^{i m theta}
            for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                // The factor 2 on the weighting comes from the normalization of the modes
                amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+j_j+ix, lo.y+l_j+iz, 0, 2*imode-1), 2._rt*sx_j[ix]*sz_j[iz]*wqx*xy.real());
                amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+j_j+ix, lo.y+l_j+iz, 0, 2*imode  ), 2._rt*sx_j[ix]*sz_j[iz]*wqx*xy.imag());
                xy = xy*xy0;
            }
#endif
        }
    }
#elif defined(WARPX_DIM_3D)
    for (int iz=0; iz<=depos_order; iz++){
        for (int iy=0; iy<=depos_order; iy++){
            for (int ix=0; ix<=depos_order; ix++){
                amrex::Gpu::Atomic::AddNoRet(
                                             &j_buff(lo.x+j_j+ix, lo.y+k_j+iy, lo.z+l_j+iz),
                                             sx_j[ix]*sy_j[iy]*sz_j[iz]*pcurrent);
            }
        }
    }
#endif
}
#endif

#endif // WARPX_SHAREDDEPOSITIONUTILS_H_
