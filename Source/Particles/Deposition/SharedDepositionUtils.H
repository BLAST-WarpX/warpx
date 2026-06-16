/* Copyright 2022 Noah Kaplan, Andrew Myers, Phil Miller
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_SHAREDDEPOSITIONUTILS_H_
#define WARPX_SHAREDDEPOSITIONUTILS_H_

#include "Particles/Pusher/GetAndSetPosition.H"
#include "ablastr/parallelization/KernelTimer.H"
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

#if (defined(AMREX_USE_HIP) || defined(AMREX_USE_CUDA)) && \
    !(defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE))
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

    const amrex::Real inv_c2 = PhysConst::inv_c2;

    // --- Get particle quantities
    const amrex::Real gaminv = 1.0_rt/std::sqrt(1.0_rt + uxp[ip]*uxp[ip]*inv_c2
                                                + uyp[ip]*uyp[ip]*inv_c2
                                                + uzp[ip]*uzp[ip]*inv_c2);
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

template <int depos_order>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void depositEsirkepovComponent (const GetParticlePosition<PIdx>& GetPosition,
                       const amrex::ParticleReal * const wp,
                       const amrex::ParticleReal * const uxp,
                       const amrex::ParticleReal * const uyp,
                       const amrex::ParticleReal * const uzp,
                       const int * const ion_lev,
                       amrex::Array4<amrex::Real> const& j_buff,
                       //amrex::IntVect const j_type,
                       const amrex::Real dt,
                       const amrex::Real relative_time,
                       const std::array<amrex::Real,3>& dx,
                       const std::array<amrex::Real,3>& xyzmin,
                       const amrex::Dim3 lo,
                       const amrex::Real q,
                       const int n_rz_azimuthal_modes,
                       const unsigned int ip,
                       //const int zdir, const int NODE, const int CELL,
                       const int dir)
{
    using namespace amrex::literals;

#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(n_rz_azimuthal_modes);
#endif

    // Whether ion_lev is a null pointer (do_ionization=0) or a real pointer
    // (do_ionization=1)
    bool const do_ionization = ion_lev;
#if !defined(WARPX_DIM_1D_Z)
    amrex::Real const dxi = 1.0_rt / dx[0];
#endif
#if !defined(WARPX_DIM_1D_Z)
    amrex::Real const xmin = xyzmin[0];
#endif
#if defined(WARPX_DIM_3D)
    amrex::Real const dyi = 1.0_rt / dx[1];
    amrex::Real const ymin = xyzmin[1];
#endif
    amrex::Real const dzi = 1.0_rt / dx[2];
    amrex::Real const zmin = xyzmin[2];

#if defined(WARPX_DIM_3D)
    amrex::Real const invdtdx = 1.0_rt / (dt*dx[1]*dx[2]);
    amrex::Real const invdtdy = 1.0_rt / (dt*dx[0]*dx[2]);
    amrex::Real const invdtdz = 1.0_rt / (dt*dx[0]*dx[1]);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    amrex::Real const invdtdx = 1.0_rt / (dt*dx[2]);
    amrex::Real const invdtdz = 1.0_rt / (dt*dx[0]);
    amrex::Real const invvol = 1.0_rt / (dx[0]*dx[2]);
#elif defined(WARPX_DIM_1D_Z)
    amrex::Real const invdtdz = 1.0_rt / (dt*dx[0]);
    amrex::Real const invvol = 1.0_rt / (dx[2]);
#endif

#if defined(WARPX_DIM_RZ)
    Complex const I = Complex{0._rt, 1._rt};
#endif

    amrex::Real const clightsq = 1.0_rt / ( PhysConst::c * PhysConst::c );
#if !defined(WARPX_DIM_1D_Z)
    amrex::Real constexpr one_third = 1.0_rt / 3.0_rt;
    amrex::Real constexpr one_sixth = 1.0_rt / 6.0_rt;
#endif

    // --- Get particle quantities
    amrex::Real const gaminv = 1.0_rt/std::sqrt(1.0_rt + uxp[ip]*uxp[ip]*clightsq
                                         + uyp[ip]*uyp[ip]*clightsq
                                         + uzp[ip]*uzp[ip]*clightsq);

    // wqx, wqy wqz are particle current in each direction
    amrex::Real wq = q*wp[ip];
    if (do_ionization){
        wq *= ion_lev[ip];
    }

    amrex::ParticleReal xp, yp, zp;
    GetPosition(ip, xp, yp, zp);

#if !defined(WARPX_DIM_1D_Z)
    amrex::Real const wqx = wq*invdtdx;
#endif
#if defined(WARPX_DIM_3D)
    amrex::Real const wqy = wq*invdtdy;
#endif
    amrex::Real const wqz = wq*invdtdz;

    // computes current and old position in grid units
#if defined(WARPX_DIM_RZ)
    amrex::Real const xp_new = xp + (relative_time + 0.5_rt*dt)*uxp[ip]*gaminv;
    amrex::Real const yp_new = yp + (relative_time + 0.5_rt*dt)*uyp[ip]*gaminv;
    amrex::Real const xp_mid = xp_new - 0.5_rt*dt*uxp[ip]*gaminv;
    amrex::Real const yp_mid = yp_new - 0.5_rt*dt*uyp[ip]*gaminv;
    amrex::Real const xp_old = xp_new - dt*uxp[ip]*gaminv;
    amrex::Real const yp_old = yp_new - dt*uyp[ip]*gaminv;
    amrex::Real const rp_new = std::sqrt(xp_new*xp_new + yp_new*yp_new);
    amrex::Real const rp_mid = std::sqrt(xp_mid*xp_mid + yp_mid*yp_mid);
    amrex::Real const rp_old = std::sqrt(xp_old*xp_old + yp_old*yp_old);
    amrex::Real costheta_new, sintheta_new;
    if (rp_new > 0._rt) {
        costheta_new = xp_new/rp_new;
        sintheta_new = yp_new/rp_new;
    } else {
        costheta_new = 1._rt;
        sintheta_new = 0._rt;
    }
    amrex::Real costheta_mid, sintheta_mid;
    if (rp_mid > 0._rt) {
        costheta_mid = xp_mid/rp_mid;
        sintheta_mid = yp_mid/rp_mid;
    } else {
        costheta_mid = 1._rt;
        sintheta_mid = 0._rt;
    }
    amrex::Real costheta_old, sintheta_old;
    if (rp_old > 0._rt) {
        costheta_old = xp_old/rp_old;
        sintheta_old = yp_old/rp_old;
    } else {
        costheta_old = 1._rt;
        sintheta_old = 0._rt;
    }
    const Complex xy_new0 = Complex{costheta_new, sintheta_new};
    const Complex xy_mid0 = Complex{costheta_mid, sintheta_mid};
    const Complex xy_old0 = Complex{costheta_old, sintheta_old};
    // Keep these double to avoid bug in single precision
    double const x_new = (rp_new - xmin)*dxi;
    double const x_old = (rp_old - xmin)*dxi;
#else
#if !defined(WARPX_DIM_1D_Z)
    // Keep these double to avoid bug in single precision
    double const x_new = (xp - xmin + (relative_time + 0.5_rt*dt)*uxp[ip]*gaminv)*dxi;
    double const x_old = x_new - dt*dxi*uxp[ip]*gaminv;
#endif
#endif
#if defined(WARPX_DIM_3D)
    // Keep these double to avoid bug in single precision
    double const y_new = (yp - ymin + (relative_time + 0.5_rt*dt)*uyp[ip]*gaminv)*dyi;
    double const y_old = y_new - dt*dyi*uyp[ip]*gaminv;
#endif
    // Keep these double to avoid bug in single precision
    double const z_new = (zp - zmin + (relative_time + 0.5_rt*dt)*uzp[ip]*gaminv)*dzi;
    double const z_old = z_new - dt*dzi*uzp[ip]*gaminv;

#if defined(WARPX_DIM_RZ)
    amrex::Real const vy = (-uxp[ip]*sintheta_mid + uyp[ip]*costheta_mid)*gaminv;
#elif defined(WARPX_DIM_XZ)
    amrex::Real const vy = uyp[ip]*gaminv;
#elif defined(WARPX_DIM_1D_Z)
    amrex::Real const vx = uxp[ip]*gaminv;
    amrex::Real const vy = uyp[ip]*gaminv;
#endif

    // Shape factor arrays
    // Note that there are extra values above and below
    // to possibly hold the factor for the old particle
    // which can be at a different grid location.
    // Keep these double to avoid bug in single precision
#if !defined(WARPX_DIM_1D_Z)
    double sx_new[depos_order + 3] = {0.};
    double sx_old[depos_order + 3] = {0.};
#endif
#if defined(WARPX_DIM_3D)
    // Keep these double to avoid bug in single precision
    double sy_new[depos_order + 3] = {0.};
    double sy_old[depos_order + 3] = {0.};
#endif
    // Keep these double to avoid bug in single precision
    double sz_new[depos_order + 3] = {0.};
    double sz_old[depos_order + 3] = {0.};

    // --- Compute shape factors
    // Compute shape factors for position as they are now and at old positions
    // [ijk]_new: leftmost grid point that the particle touches
    Compute_shape_factor< depos_order > compute_shape_factor;
    Compute_shifted_shape_factor< depos_order > compute_shifted_shape_factor;

#if !defined(WARPX_DIM_1D_Z)
    const int i_new = compute_shape_factor(sx_new+1, x_new);
    const int i_old = compute_shifted_shape_factor(sx_old, x_old, i_new);
#endif
#if defined(WARPX_DIM_3D)
    const int j_new = compute_shape_factor(sy_new+1, y_new);
    const int j_old = compute_shifted_shape_factor(sy_old, y_old, j_new);
#endif
    const int k_new = compute_shape_factor(sz_new+1, z_new);
    const int k_old = compute_shifted_shape_factor(sz_old, z_old, k_new);

    // computes min/max positions of current contributions
#if !defined(WARPX_DIM_1D_Z)
    int dil = 1, diu = 1;
    if (i_old < i_new) {dil = 0;}
    if (i_old > i_new) {diu = 0;}
#endif
#if defined(WARPX_DIM_3D)
    int djl = 1, dju = 1;
    if (j_old < j_new) {djl = 0;}
    if (j_old > j_new) {dju = 0;}
#endif
    int dkl = 1, dku = 1;
    if (k_old < k_new) {dkl = 0;}
    if (k_old > k_new) {dku = 0;}

#if defined(WARPX_DIM_3D)

    if (dir == 0) {
    for (int k=dkl; k<=depos_order+2-dku; k++) {
        for (int j=djl; j<=depos_order+2-dju; j++) {
            amrex::Real sdxi = 0._rt;
            for (int i=dil; i<=depos_order+1-diu; i++) {
                sdxi += wqx*(sx_old[i] - sx_new[i])*(
                    one_third*(sy_new[j]*sz_new[k] + sy_old[j]*sz_old[k])
                   +one_sixth*(sy_new[j]*sz_old[k] + sy_old[j]*sz_new[k]));
                amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdxi);
            }
        }
    }
    }
    if (dir == 1) {
    for (int k=dkl; k<=depos_order+2-dku; k++) {
        for (int i=dil; i<=depos_order+2-diu; i++) {
            amrex::Real sdyj = 0._rt;
            for (int j=djl; j<=depos_order+1-dju; j++) {
                sdyj += wqy*(sy_old[j] - sy_new[j])*(
                    one_third*(sx_new[i]*sz_new[k] + sx_old[i]*sz_old[k])
                   +one_sixth*(sx_new[i]*sz_old[k] + sx_old[i]*sz_new[k]));
                amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdyj);
            }
        }
    }
    }
    if (dir == 2) {
    for (int j=djl; j<=depos_order+2-dju; j++) {
        for (int i=dil; i<=depos_order+2-diu; i++) {
            amrex::Real sdzk = 0._rt;
            for (int k=dkl; k<=depos_order+1-dku; k++) {
                sdzk += wqz*(sz_old[k] - sz_new[k])*(
                    one_third*(sx_new[i]*sy_new[j] + sx_old[i]*sy_old[j])
                   +one_sixth*(sx_new[i]*sy_old[j] + sx_old[i]*sy_new[j]));
                amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdzk);
            }
        }
    }
    }

#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)

    if (dir == 0) {
    for (int k=dkl; k<=depos_order+2-dku; k++) {
        amrex::Real sdxi = 0._rt;
        for (int i=dil; i<=depos_order+1-diu; i++) {
            sdxi += wqx*(sx_old[i] - sx_new[i])*0.5_rt*(sz_new[k] + sz_old[k]);
            amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdxi);
#if defined(WARPX_DIM_RZ)
            Complex xy_mid = xy_mid0; // Throughout the following loop, xy_mid takes the value e^{i m theta}
            for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                // The factor 2 comes from the normalization of the modes
                const Complex djr_cmplx = 2._rt *sdxi*xy_mid;
                amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djr_cmplx.real());
                amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djr_cmplx.imag());
                xy_mid = xy_mid*xy_mid0;
            }
#endif
        }
    }
    }
    if (dir == 1) {
    for (int k=dkl; k<=depos_order+2-dku; k++) {
        for (int i=dil; i<=depos_order+2-diu; i++) {
            amrex::Real const sdyj = wq*vy*invvol*(
                one_third*(sx_new[i]*sz_new[k] + sx_old[i]*sz_old[k])
               +one_sixth*(sx_new[i]*sz_old[k] + sx_old[i]*sz_new[k]));
            amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdyj);
#if defined(WARPX_DIM_RZ)
            Complex xy_new = xy_new0;
            Complex xy_mid = xy_mid0;
            Complex xy_old = xy_old0;
            // Throughout the following loop, xy_ takes the value e^{i m theta_}
            for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                // The factor 2 comes from the normalization of the modes
                // The minus sign comes from the different convention with respect to Davidson et al.
                const Complex djt_cmplx = -2._rt * I*(i_new-1 + i + xmin*dxi)*wq*invdtdx/(amrex::Real)imode
                                          *(Complex(sx_new[i]*sz_new[k], 0._rt)*(xy_new - xy_mid)
                                          + Complex(sx_old[i]*sz_old[k], 0._rt)*(xy_mid - xy_old));
                amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djt_cmplx.real());
                amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djt_cmplx.imag());
                xy_new = xy_new*xy_new0;
                xy_mid = xy_mid*xy_mid0;
                xy_old = xy_old*xy_old0;
            }
#endif
        }
    }
    }
    if (dir == 2) {
    for (int i=dil; i<=depos_order+2-diu; i++) {
        amrex::Real sdzk = 0._rt;
        for (int k=dkl; k<=depos_order+1-dku; k++) {
            sdzk += wqz*(sz_old[k] - sz_new[k])*0.5_rt*(sx_new[i] + sx_old[i]);
            amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdzk);
#if defined(WARPX_DIM_RZ)
            Complex xy_mid = xy_mid0; // Throughout the following loop, xy_mid takes the value e^{i m theta}
            for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                // The factor 2 comes from the normalization of the modes
                const Complex djz_cmplx = 2._rt * sdzk * xy_mid;
                amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djz_cmplx.real());
                amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djz_cmplx.imag());
                xy_mid = xy_mid*xy_mid0;
            }
#endif
        }
    }
    }
#elif defined(WARPX_DIM_1D_Z)


    if (dir == 0) {
    for (int k=dkl; k<=depos_order+2-dku; k++) {
        amrex::Real sdxi = 0._rt;
        sdxi += wq*vx*invvol*0.5_rt*(sz_old[k] + sz_new[k]);
        amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+k_new-1+k, 0, 0, 0), sdxi);
    }
    }
    if (dir == 1) {
    for (int k=dkl; k<=depos_order+2-dku; k++) {
        amrex::Real sdyj = 0._rt;
        sdyj += wq*vy*invvol*0.5_rt*(sz_old[k] + sz_new[k]);
        amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+k_new-1+k, 0, 0, 0), sdyj);
    }
    }
    if (dir == 2) {
    for (int k=dkl; k<=depos_order+1-dku; k++) {
        amrex::Real sdzk = 0._rt;
        sdzk += wqz*(sz_old[k] - sz_new[k]);
        amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+k_new-1+k, 0, 0, 0), sdzk);
    }
    }
#endif
}

template <int depos_order>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void depositEsirkepovComponentX (const GetParticlePosition<PIdx>& GetPosition,
                       const amrex::ParticleReal * const wp,
                       const amrex::ParticleReal * const uxp,
                       const amrex::ParticleReal * const uyp,
                       const amrex::ParticleReal * const uzp,
                       const int * const ion_lev,
                       amrex::Array4<amrex::Real> const& j_buff,
                       //amrex::IntVect const j_type,
                       const amrex::Real dt,
                       const amrex::Real relative_time,
                       const std::array<amrex::Real,3>& dx,
                       const std::array<amrex::Real,3>& xyzmin,
                       const amrex::Dim3 lo,
                       const amrex::Real q,
                       const int n_rz_azimuthal_modes,
                       const unsigned int ip)
                       //const int zdir, const int NODE, const int CELL,
                       //const int dir)
{
    using namespace amrex::literals;

#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(n_rz_azimuthal_modes);
#endif

    // Whether ion_lev is a null pointer (do_ionization=0) or a real pointer
    // (do_ionization=1)
    bool const do_ionization = ion_lev;
#if !defined(WARPX_DIM_1D_Z)
    amrex::Real const dxi = 1.0_rt / dx[0];
#endif
#if !defined(WARPX_DIM_1D_Z)
    amrex::Real const xmin = xyzmin[0];
#endif
#if defined(WARPX_DIM_3D)
    amrex::Real const dyi = 1.0_rt / dx[1];
    amrex::Real const ymin = xyzmin[1];
#endif
    amrex::Real const dzi = 1.0_rt / dx[2];
    amrex::Real const zmin = xyzmin[2];

#if defined(WARPX_DIM_3D)
    amrex::Real const invdtdx = 1.0_rt / (dt*dx[1]*dx[2]);
    amrex::Real constexpr one_third = 1.0_rt / 3.0_rt;
    amrex::Real constexpr one_sixth = 1.0_rt / 6.0_rt;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    amrex::Real const invdtdx = 1.0_rt / (dt*dx[2]);
#elif defined(WARPX_DIM_1D_Z)
    amrex::Real const invvol = 1.0_rt / (dx[2]);
#endif

    amrex::Real const clightsq = 1.0_rt / ( PhysConst::c * PhysConst::c );

    // --- Get particle quantities
    amrex::Real const gaminv = 1.0_rt/std::sqrt(1.0_rt + uxp[ip]*uxp[ip]*clightsq
                                         + uyp[ip]*uyp[ip]*clightsq
                                         + uzp[ip]*uzp[ip]*clightsq);

    // wqx, wqy wqz are particle current in each direction
    amrex::Real wq = q*wp[ip];
    if (do_ionization){
        wq *= ion_lev[ip];
    }

    amrex::ParticleReal xp, yp, zp;
    GetPosition(ip, xp, yp, zp);

#if !defined(WARPX_DIM_1D_Z)
    amrex::Real const wqx = wq*invdtdx;
#endif

    // computes current and old position in grid units
#if defined(WARPX_DIM_RZ)
    amrex::Real const xp_new = xp + (relative_time + 0.5_rt*dt)*uxp[ip]*gaminv;
    amrex::Real const yp_new = yp + (relative_time + 0.5_rt*dt)*uyp[ip]*gaminv;
    amrex::Real const xp_mid = xp_new - 0.5_rt*dt*uxp[ip]*gaminv;
    amrex::Real const yp_mid = yp_new - 0.5_rt*dt*uyp[ip]*gaminv;
    amrex::Real const xp_old = xp_new - dt*uxp[ip]*gaminv;
    amrex::Real const yp_old = yp_new - dt*uyp[ip]*gaminv;
    amrex::Real const rp_new = std::sqrt(xp_new*xp_new + yp_new*yp_new);
    amrex::Real const rp_mid = std::sqrt(xp_mid*xp_mid + yp_mid*yp_mid);
    amrex::Real const rp_old = std::sqrt(xp_old*xp_old + yp_old*yp_old);

    amrex::Real costheta_mid, sintheta_mid;
    if (rp_mid > 0._rt) {
        costheta_mid = xp_mid/rp_mid;
        sintheta_mid = yp_mid/rp_mid;
    } else {
        costheta_mid = 1._rt;
        sintheta_mid = 0._rt;
    }

    const Complex xy_mid0 = Complex{costheta_mid, sintheta_mid};
    // Keep these double to avoid bug in single precision
    double const x_new = (rp_new - xmin)*dxi;
    double const x_old = (rp_old - xmin)*dxi;
#else
#if !defined(WARPX_DIM_1D_Z)
    // Keep these double to avoid bug in single precision
    double const x_new = (xp - xmin + (relative_time + 0.5_rt*dt)*uxp[ip]*gaminv)*dxi;
    double const x_old = x_new - dt*dxi*uxp[ip]*gaminv;
#endif
#endif
#if defined(WARPX_DIM_3D)
    // Keep these double to avoid bug in single precision
    double const y_new = (yp - ymin + (relative_time + 0.5_rt*dt)*uyp[ip]*gaminv)*dyi;
    double const y_old = y_new - dt*dyi*uyp[ip]*gaminv;
#endif
    // Keep these double to avoid bug in single precision
    double const z_new = (zp - zmin + (relative_time + 0.5_rt*dt)*uzp[ip]*gaminv)*dzi;
    double const z_old = z_new - dt*dzi*uzp[ip]*gaminv;

#if defined(WARPX_DIM_1D_Z)
    amrex::Real const vx = uxp[ip]*gaminv;
#endif

    // Shape factor arrays
    // Note that there are extra values above and below
    // to possibly hold the factor for the old particle
    // which can be at a different grid location.
    // Keep these double to avoid bug in single precision
#if !defined(WARPX_DIM_1D_Z)
    double sx_new[depos_order + 3] = {0.};
    double sx_old[depos_order + 3] = {0.};
#endif
#if defined(WARPX_DIM_3D)
    // Keep these double to avoid bug in single precision
    double sy_new[depos_order + 3] = {0.};
    double sy_old[depos_order + 3] = {0.};
#endif
    // Keep these double to avoid bug in single precision
    double sz_new[depos_order + 3] = {0.};
    double sz_old[depos_order + 3] = {0.};

    // --- Compute shape factors
    // Compute shape factors for position as they are now and at old positions
    // [ijk]_new: leftmost grid point that the particle touches
    const Compute_shape_factor< depos_order > compute_shape_factor;
    const Compute_shifted_shape_factor< depos_order > compute_shifted_shape_factor;

#if !defined(WARPX_DIM_1D_Z)
    const int i_new = compute_shape_factor(sx_new+1, x_new);
    const int i_old = compute_shifted_shape_factor(sx_old, x_old, i_new);
#endif
#if defined(WARPX_DIM_3D)
    const int j_new = compute_shape_factor(sy_new+1, y_new);
    const int j_old = compute_shifted_shape_factor(sy_old, y_old, j_new);
#endif
    const int k_new = compute_shape_factor(sz_new+1, z_new);
    const int k_old = compute_shifted_shape_factor(sz_old, z_old, k_new);

    // computes min/max positions of current contributions
#if !defined(WARPX_DIM_1D_Z)
    int dil = 1, diu = 1;
    if (i_old < i_new) {dil = 0;}
    if (i_old > i_new) {diu = 0;}
#endif
#if defined(WARPX_DIM_3D)
    int djl = 1, dju = 1;
    if (j_old < j_new) {djl = 0;}
    if (j_old > j_new) {dju = 0;}
#endif
    int dkl = 1, dku = 1;
    if (k_old < k_new) {dkl = 0;}
    if (k_old > k_new) {dku = 0;}

#if defined(WARPX_DIM_3D)

    for (int k=dkl; k<=depos_order+2-dku; k++) {
        for (int j=djl; j<=depos_order+2-dju; j++) {
            amrex::Real sdxi = 0._rt;
            AMREX_UNROLL_LOOP(8)
            for (int i=dil; i<=depos_order+1-diu; i++) {
                sdxi += wqx*(sx_old[i] - sx_new[i])*(
                    one_third*(sy_new[j]*sz_new[k] + sy_old[j]*sz_old[k])
                   +one_sixth*(sy_new[j]*sz_old[k] + sy_old[j]*sz_new[k]));
                amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdxi);
            }
        }
    }

#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)

    for (int k=dkl; k<=depos_order+2-dku; k++) {
        amrex::Real sdxi = 0._rt;
        for (int i=dil; i<=depos_order+1-diu; i++) {
            sdxi += wqx*(sx_old[i] - sx_new[i])*0.5_rt*(sz_new[k] + sz_old[k]);
            amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdxi);
#if defined(WARPX_DIM_RZ)
            Complex xy_mid = xy_mid0; // Throughout the following loop, xy_mid takes the value e^{i m theta}
            for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                // The factor 2 comes from the normalization of the modes
                const Complex djr_cmplx = 2._rt *sdxi*xy_mid;
                amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djr_cmplx.real());
                amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djr_cmplx.imag());
                xy_mid = xy_mid*xy_mid0;
            }
#endif
        }
    }
#elif defined(WARPX_DIM_1D_Z)


    for (int k=dkl; k<=depos_order+2-dku; k++) {
        amrex::Real sdxi = 0._rt;
        sdxi += wq*vx*invvol*0.5_rt*(sz_old[k] + sz_new[k]);
        amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+k_new-1+k, 0, 0, 0), sdxi);
    }
#endif
}

template <int depos_order>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void depositEsirkepovComponentY (const GetParticlePosition<PIdx>& GetPosition,
                       const amrex::ParticleReal * const wp,
                       const amrex::ParticleReal * const uxp,
                       const amrex::ParticleReal * const uyp,
                       const amrex::ParticleReal * const uzp,
                       const int * const ion_lev,
                       amrex::Array4<amrex::Real> const& j_buff,
                       //amrex::IntVect const j_type,
                       const amrex::Real dt,
                       const amrex::Real relative_time,
                       const std::array<amrex::Real,3>& dx,
                       const std::array<amrex::Real,3>& xyzmin,
                       const amrex::Dim3 lo,
                       const amrex::Real q,
                       const int n_rz_azimuthal_modes,
                       const unsigned int ip)
                       //const int zdir, const int NODE, const int CELL,
                       //const int dir)
{
    using namespace amrex::literals;

#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(n_rz_azimuthal_modes);
#endif

    // Whether ion_lev is a null pointer (do_ionization=0) or a real pointer
    // (do_ionization=1)
    bool const do_ionization = ion_lev;
#if !defined(WARPX_DIM_1D_Z)
    amrex::Real const dxi = 1.0_rt / dx[0];
#endif
#if !defined(WARPX_DIM_1D_Z)
    amrex::Real const xmin = xyzmin[0];
#endif
#if defined(WARPX_DIM_3D)
    amrex::Real const dyi = 1.0_rt / dx[1];
    amrex::Real const ymin = xyzmin[1];
#endif
    amrex::Real const dzi = 1.0_rt / dx[2];
    amrex::Real const zmin = xyzmin[2];

#if defined(WARPX_DIM_3D)
    amrex::Real const invdtdy = 1.0_rt / (dt*dx[0]*dx[2]);
#elif defined(WARPX_DIM_XZ)
    amrex::Real const invvol = 1.0_rt / (dx[0]*dx[2]);
#elif defined(WARPX_DIM_RZ)
    amrex::Real const invdtdx = 1.0_rt / (dt*dx[2]);
    amrex::Real const invvol = 1.0_rt / (dx[0]*dx[2]);
#elif defined(WARPX_DIM_1D_Z)
    amrex::Real const invvol = 1.0_rt / (dx[2]);
#endif

#if defined(WARPX_DIM_RZ)
    Complex const I = Complex{0._rt, 1._rt};
#endif

    amrex::Real const clightsq = 1.0_rt / ( PhysConst::c * PhysConst::c );
#if !defined(WARPX_DIM_1D_Z)
    amrex::Real constexpr one_third = 1.0_rt / 3.0_rt;
    amrex::Real constexpr one_sixth = 1.0_rt / 6.0_rt;
#endif

    // --- Get particle quantities
    amrex::Real const gaminv = 1.0_rt/std::sqrt(1.0_rt + uxp[ip]*uxp[ip]*clightsq
                                         + uyp[ip]*uyp[ip]*clightsq
                                         + uzp[ip]*uzp[ip]*clightsq);

    // wqx, wqy wqz are particle current in each direction
    amrex::Real wq = q*wp[ip];
    if (do_ionization){
        wq *= ion_lev[ip];
    }

    amrex::ParticleReal xp, yp, zp;
    GetPosition(ip, xp, yp, zp);

#if defined(WARPX_DIM_3D)
    amrex::Real const wqy = wq*invdtdy;
#endif

    // computes current and old position in grid units
#if defined(WARPX_DIM_RZ)
    amrex::Real const xp_new = xp + (relative_time + 0.5_rt*dt)*uxp[ip]*gaminv;
    amrex::Real const yp_new = yp + (relative_time + 0.5_rt*dt)*uyp[ip]*gaminv;
    amrex::Real const xp_mid = xp_new - 0.5_rt*dt*uxp[ip]*gaminv;
    amrex::Real const yp_mid = yp_new - 0.5_rt*dt*uyp[ip]*gaminv;
    amrex::Real const xp_old = xp_new - dt*uxp[ip]*gaminv;
    amrex::Real const yp_old = yp_new - dt*uyp[ip]*gaminv;
    amrex::Real const rp_new = std::sqrt(xp_new*xp_new + yp_new*yp_new);
    amrex::Real const rp_mid = std::sqrt(xp_mid*xp_mid + yp_mid*yp_mid);
    amrex::Real const rp_old = std::sqrt(xp_old*xp_old + yp_old*yp_old);
    amrex::Real costheta_new, sintheta_new;
    if (rp_new > 0._rt) {
        costheta_new = xp_new/rp_new;
        sintheta_new = yp_new/rp_new;
    } else {
        costheta_new = 1._rt;
        sintheta_new = 0._rt;
    }
    amrex::Real costheta_mid, sintheta_mid;
    if (rp_mid > 0._rt) {
        costheta_mid = xp_mid/rp_mid;
        sintheta_mid = yp_mid/rp_mid;
    } else {
        costheta_mid = 1._rt;
        sintheta_mid = 0._rt;
    }
    amrex::Real costheta_old, sintheta_old;
    if (rp_old > 0._rt) {
        costheta_old = xp_old/rp_old;
        sintheta_old = yp_old/rp_old;
    } else {
        costheta_old = 1._rt;
        sintheta_old = 0._rt;
    }
    const Complex xy_new0 = Complex{costheta_new, sintheta_new};
    const Complex xy_mid0 = Complex{costheta_mid, sintheta_mid};
    const Complex xy_old0 = Complex{costheta_old, sintheta_old};
    // Keep these double to avoid bug in single precision
    double const x_new = (rp_new - xmin)*dxi;
    double const x_old = (rp_old - xmin)*dxi;
#else
#if !defined(WARPX_DIM_1D_Z)
    // Keep these double to avoid bug in single precision
    double const x_new = (xp - xmin + (relative_time + 0.5_rt*dt)*uxp[ip]*gaminv)*dxi;
    double const x_old = x_new - dt*dxi*uxp[ip]*gaminv;
#endif
#endif
#if defined(WARPX_DIM_3D)
    // Keep these double to avoid bug in single precision
    double const y_new = (yp - ymin + (relative_time + 0.5_rt*dt)*uyp[ip]*gaminv)*dyi;
    double const y_old = y_new - dt*dyi*uyp[ip]*gaminv;
#endif
    // Keep these double to avoid bug in single precision
    double const z_new = (zp - zmin + (relative_time + 0.5_rt*dt)*uzp[ip]*gaminv)*dzi;
    double const z_old = z_new - dt*dzi*uzp[ip]*gaminv;

#if defined(WARPX_DIM_RZ)
    amrex::Real const vy = (-uxp[ip]*sintheta_mid + uyp[ip]*costheta_mid)*gaminv;
#elif defined(WARPX_DIM_XZ)
    amrex::Real const vy = uyp[ip]*gaminv;
#elif defined(WARPX_DIM_1D_Z)
    amrex::Real const vy = uyp[ip]*gaminv;
#endif

    // Shape factor arrays
    // Note that there are extra values above and below
    // to possibly hold the factor for the old particle
    // which can be at a different grid location.
    // Keep these double to avoid bug in single precision
#if !defined(WARPX_DIM_1D_Z)
    double sx_new[depos_order + 3] = {0.};
    double sx_old[depos_order + 3] = {0.};
#endif
#if defined(WARPX_DIM_3D)
    // Keep these double to avoid bug in single precision
    double sy_new[depos_order + 3] = {0.};
    double sy_old[depos_order + 3] = {0.};
#endif
    // Keep these double to avoid bug in single precision
    double sz_new[depos_order + 3] = {0.};
    double sz_old[depos_order + 3] = {0.};

    // --- Compute shape factors
    // Compute shape factors for position as they are now and at old positions
    // [ijk]_new: leftmost grid point that the particle touches
    const Compute_shape_factor< depos_order > compute_shape_factor;
    const Compute_shifted_shape_factor< depos_order > compute_shifted_shape_factor;

#if !defined(WARPX_DIM_1D_Z)
    const int i_new = compute_shape_factor(sx_new+1, x_new);
    const int i_old = compute_shifted_shape_factor(sx_old, x_old, i_new);
#endif
#if defined(WARPX_DIM_3D)
    const int j_new = compute_shape_factor(sy_new+1, y_new);
    const int j_old = compute_shifted_shape_factor(sy_old, y_old, j_new);
#endif
    const int k_new = compute_shape_factor(sz_new+1, z_new);
    const int k_old = compute_shifted_shape_factor(sz_old, z_old, k_new);

    // computes min/max positions of current contributions
#if !defined(WARPX_DIM_1D_Z)
    int dil = 1, diu = 1;
    if (i_old < i_new) {dil = 0;}
    if (i_old > i_new) {diu = 0;}
#endif
#if defined(WARPX_DIM_3D)
    int djl = 1, dju = 1;
    if (j_old < j_new) {djl = 0;}
    if (j_old > j_new) {dju = 0;}
#endif
    int dkl = 1, dku = 1;
    if (k_old < k_new) {dkl = 0;}
    if (k_old > k_new) {dku = 0;}

#if defined(WARPX_DIM_3D)

    for (int k=dkl; k<=depos_order+2-dku; k++) {
        for (int i=dil; i<=depos_order+2-diu; i++) {
            amrex::Real sdyj = 0._rt;
            AMREX_UNROLL_LOOP(8)
            for (int j=djl; j<=depos_order+1-dju; j++) {
                sdyj += wqy*(sy_old[j] - sy_new[j])*(
                    one_third*(sx_new[i]*sz_new[k] + sx_old[i]*sz_old[k])
                   +one_sixth*(sx_new[i]*sz_old[k] + sx_old[i]*sz_new[k]));
                amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdyj);
            }
        }
    }

#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)

    for (int k=dkl; k<=depos_order+2-dku; k++) {
        for (int i=dil; i<=depos_order+2-diu; i++) {
            amrex::Real const sdyj = wq*vy*invvol*(
                one_third*(sx_new[i]*sz_new[k] + sx_old[i]*sz_old[k])
               +one_sixth*(sx_new[i]*sz_old[k] + sx_old[i]*sz_new[k]));
            amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdyj);
#if defined(WARPX_DIM_RZ)
            Complex xy_new = xy_new0;
            Complex xy_mid = xy_mid0;
            Complex xy_old = xy_old0;
            // Throughout the following loop, xy_ takes the value e^{i m theta_}
            for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                // The factor 2 comes from the normalization of the modes
                // The minus sign comes from the different convention with respect to Davidson et al.
                const Complex djt_cmplx = -2._rt * I*(i_new-1 + i + xmin*dxi)*wq*invdtdx/(amrex::Real)imode
                                          *(Complex(sx_new[i]*sz_new[k], 0._rt)*(xy_new - xy_mid)
                                          + Complex(sx_old[i]*sz_old[k], 0._rt)*(xy_mid - xy_old));
                amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djt_cmplx.real());
                amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djt_cmplx.imag());
                xy_new = xy_new*xy_new0;
                xy_mid = xy_mid*xy_mid0;
                xy_old = xy_old*xy_old0;
            }
#endif
        }
    }
#elif defined(WARPX_DIM_1D_Z)


    for (int k=dkl; k<=depos_order+2-dku; k++) {
        amrex::Real sdyj = 0._rt;
        sdyj += wq*vy*invvol*0.5_rt*(sz_old[k] + sz_new[k]);
        amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+k_new-1+k, 0, 0, 0), sdyj);
    }
#endif
}

template <int depos_order>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void depositEsirkepovComponentZ (const GetParticlePosition<PIdx>& GetPosition,
                       const amrex::ParticleReal * const wp,
                       const amrex::ParticleReal * const uxp,
                       const amrex::ParticleReal * const uyp,
                       const amrex::ParticleReal * const uzp,
                       const int * const ion_lev,
                       amrex::Array4<amrex::Real> const& j_buff,
                       //amrex::IntVect const j_type,
                       const amrex::Real dt,
                       const amrex::Real relative_time,
                       const std::array<amrex::Real,3>& dx,
                       const std::array<amrex::Real,3>& xyzmin,
                       const amrex::Dim3 lo,
                       const amrex::Real q,
                       const int n_rz_azimuthal_modes,
                       const unsigned int ip)
                       //const int zdir, const int NODE, const int CELL, const int dir)
{
    using namespace amrex::literals;

#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(n_rz_azimuthal_modes);
#endif

    // Whether ion_lev is a null pointer (do_ionization=0) or a real pointer
    // (do_ionization=1)
    bool const do_ionization = ion_lev;
#if !defined(WARPX_DIM_1D_Z)
    amrex::Real const dxi = 1.0_rt / dx[0];
#endif
#if !defined(WARPX_DIM_1D_Z)
    amrex::Real const xmin = xyzmin[0];
#endif
#if defined(WARPX_DIM_3D)
    amrex::Real const dyi = 1.0_rt / dx[1];
    amrex::Real const ymin = xyzmin[1];
    amrex::Real constexpr one_third = 1.0_rt / 3.0_rt;
    amrex::Real constexpr one_sixth = 1.0_rt / 6.0_rt;
#endif
    amrex::Real const dzi = 1.0_rt / dx[2];
    amrex::Real const zmin = xyzmin[2];

#if defined(WARPX_DIM_3D)
    amrex::Real const invdtdz = 1.0_rt / (dt*dx[0]*dx[1]);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    amrex::Real const invdtdz = 1.0_rt / (dt*dx[0]);
#elif defined(WARPX_DIM_1D_Z)
    amrex::Real const invdtdz = 1.0_rt / (dt*dx[0]);
#endif

    amrex::Real const clightsq = 1.0_rt / ( PhysConst::c * PhysConst::c );

    // --- Get particle quantities
    amrex::Real const gaminv = 1.0_rt/std::sqrt(1.0_rt + uxp[ip]*uxp[ip]*clightsq
                                         + uyp[ip]*uyp[ip]*clightsq
                                         + uzp[ip]*uzp[ip]*clightsq);

    // wqx, wqy wqz are particle current in each direction
    amrex::Real wq = q*wp[ip];
    if (do_ionization){
        wq *= ion_lev[ip];
    }

    amrex::ParticleReal xp, yp, zp;
    GetPosition(ip, xp, yp, zp);

    amrex::Real const wqz = wq*invdtdz;

    // computes current and old position in grid units
#if defined(WARPX_DIM_RZ)
    amrex::Real const xp_new = xp + (relative_time + 0.5_rt*dt)*uxp[ip]*gaminv;
    amrex::Real const yp_new = yp + (relative_time + 0.5_rt*dt)*uyp[ip]*gaminv;
    amrex::Real const xp_mid = xp_new - 0.5_rt*dt*uxp[ip]*gaminv;
    amrex::Real const yp_mid = yp_new - 0.5_rt*dt*uyp[ip]*gaminv;
    amrex::Real const xp_old = xp_new - dt*uxp[ip]*gaminv;
    amrex::Real const yp_old = yp_new - dt*uyp[ip]*gaminv;
    amrex::Real const rp_new = std::sqrt(xp_new*xp_new + yp_new*yp_new);
    amrex::Real const rp_mid = std::sqrt(xp_mid*xp_mid + yp_mid*yp_mid);
    amrex::Real const rp_old = std::sqrt(xp_old*xp_old + yp_old*yp_old);

    amrex::Real costheta_mid, sintheta_mid;
    if (rp_mid > 0._rt) {
        costheta_mid = xp_mid/rp_mid;
        sintheta_mid = yp_mid/rp_mid;
    } else {
        costheta_mid = 1._rt;
        sintheta_mid = 0._rt;
    }

    const Complex xy_mid0 = Complex{costheta_mid, sintheta_mid};
    // Keep these double to avoid bug in single precision
    double const x_new = (rp_new - xmin)*dxi;
    double const x_old = (rp_old - xmin)*dxi;
#else
#if !defined(WARPX_DIM_1D_Z)
    // Keep these double to avoid bug in single precision
    double const x_new = (xp - xmin + (relative_time + 0.5_rt*dt)*uxp[ip]*gaminv)*dxi;
    double const x_old = x_new - dt*dxi*uxp[ip]*gaminv;
#endif
#endif
#if defined(WARPX_DIM_3D)
    // Keep these double to avoid bug in single precision
    double const y_new = (yp - ymin + (relative_time + 0.5_rt*dt)*uyp[ip]*gaminv)*dyi;
    double const y_old = y_new - dt*dyi*uyp[ip]*gaminv;
#endif
    // Keep these double to avoid bug in single precision
    double const z_new = (zp - zmin + (relative_time + 0.5_rt*dt)*uzp[ip]*gaminv)*dzi;
    double const z_old = z_new - dt*dzi*uzp[ip]*gaminv;

    // Shape factor arrays
    // Note that there are extra values above and below
    // to possibly hold the factor for the old particle
    // which can be at a different grid location.
    // Keep these double to avoid bug in single precision
#if !defined(WARPX_DIM_1D_Z)
    double sx_new[depos_order + 3] = {0.};
    double sx_old[depos_order + 3] = {0.};
#endif
#if defined(WARPX_DIM_3D)
    // Keep these double to avoid bug in single precision
    double sy_new[depos_order + 3] = {0.};
    double sy_old[depos_order + 3] = {0.};
#endif
    // Keep these double to avoid bug in single precision
    double sz_new[depos_order + 3] = {0.};
    double sz_old[depos_order + 3] = {0.};

    // --- Compute shape factors
    // Compute shape factors for position as they are now and at old positions
    // [ijk]_new: leftmost grid point that the particle touches
    const Compute_shape_factor< depos_order > compute_shape_factor;
    const Compute_shifted_shape_factor< depos_order > compute_shifted_shape_factor;

#if !defined(WARPX_DIM_1D_Z)
    const int i_new = compute_shape_factor(sx_new+1, x_new);
    const int i_old = compute_shifted_shape_factor(sx_old, x_old, i_new);
#endif
#if defined(WARPX_DIM_3D)
    const int j_new = compute_shape_factor(sy_new+1, y_new);
    const int j_old = compute_shifted_shape_factor(sy_old, y_old, j_new);
#endif
    const int k_new = compute_shape_factor(sz_new+1, z_new);
    const int k_old = compute_shifted_shape_factor(sz_old, z_old, k_new);

    // computes min/max positions of current contributions
#if !defined(WARPX_DIM_1D_Z)
    int dil = 1, diu = 1;
    if (i_old < i_new) {dil = 0;}
    if (i_old > i_new) {diu = 0;}
#endif
#if defined(WARPX_DIM_3D)
    int djl = 1, dju = 1;
    if (j_old < j_new) {djl = 0;}
    if (j_old > j_new) {dju = 0;}
#endif
    int dkl = 1, dku = 1;
    if (k_old < k_new) {dkl = 0;}
    if (k_old > k_new) {dku = 0;}

#if defined(WARPX_DIM_3D)

    for (int j=djl; j<=depos_order+2-dju; j++) {
        for (int i=dil; i<=depos_order+2-diu; i++) {
            amrex::Real sdzk = 0._rt;
            AMREX_UNROLL_LOOP(8)
            for (int k=dkl; k<=depos_order+1-dku; k++) {
                sdzk += wqz*(sz_old[k] - sz_new[k])*(
                    one_third*(sx_new[i]*sy_new[j] + sx_old[i]*sy_old[j])
                   +one_sixth*(sx_new[i]*sy_old[j] + sx_old[i]*sy_new[j]));
                amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdzk);
            }
        }
    }

#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)

    for (int i=dil; i<=depos_order+2-diu; i++) {
        amrex::Real sdzk = 0._rt;
        for (int k=dkl; k<=depos_order+1-dku; k++) {
            sdzk += wqz*(sz_old[k] - sz_new[k])*0.5_rt*(sx_new[i] + sx_old[i]);
            amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdzk);
#if defined(WARPX_DIM_RZ)
            Complex xy_mid = xy_mid0; // Throughout the following loop, xy_mid takes the value e^{i m theta}
            for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                // The factor 2 comes from the normalization of the modes
                const Complex djz_cmplx = 2._rt * sdzk * xy_mid;
                amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djz_cmplx.real());
                amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djz_cmplx.imag());
                xy_mid = xy_mid*xy_mid0;
            }
#endif
        }
    }
#elif defined(WARPX_DIM_1D_Z)


    for (int k=dkl; k<=depos_order+1-dku; k++) {
        amrex::Real sdzk = 0._rt;
        sdzk += wqz*(sz_old[k] - sz_new[k]);
        amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+k_new-1+k, 0, 0, 0), sdzk);
    }
#endif
}

#endif // WARPX_SHAREDDEPOSITIONUTILS_H_
