/* Copyright 2019 Axel Huebl, David Grote, Maxence Thevenet
 * Remi Lehe, Weiqun Zhang, Michael Rowan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_CURRENTDEPOSITION_H_
#define WARPX_CURRENTDEPOSITION_H_

#include "Particles/Deposition/SharedDepositionUtils.H"
#include "ablastr/parallelization/KernelTimer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/ShapeFactors.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#ifdef WARPX_DIM_RZ
#   include "Utils/WarpX_Complex.H"
#endif

#include "WarpX.H" // todo: remove include and pass globals as args

#include <AMReX.H>
#include <AMReX_Arena.H>
#include <AMReX_Array4.H>
#include <AMReX_Dim3.H>
#include <AMReX_REAL.H>

/**
 * \brief Kernel for the direct current deposition for thread thread_num
 * \tparam depos_order deposition order
 * \param xp, yp, zp    The particle positions.
 * \param wq            The charge of the macroparticle
 * \param vx,vy,vz      The particle velocities
 * \param jx_arr,jy_arr,jz_arr Array4 of current density, either full array or tile.
 * \param jx_type,jy_type,jz_type The grid types along each direction, either NODE or CELL
 * \param relative_time Time at which to deposit J, relative to the time of the
 *                      current positions of the particles. When different than 0,
 *                      the particle position will be temporarily modified to match
 *                      the time of the deposition.
 * \param dinv          3D cell size inverse
 * \param xyzmin        The lower bounds of the domain
 * \param invvol        The inverse volume of a grid cell
 * \param lo            Index lower bounds of domain.
 * \param n_rz_azimuthal_modes Number of azimuthal modes when using RZ geometry.
 */
template <int depos_order>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void doDepositionShapeNKernel([[maybe_unused]] const amrex::ParticleReal xp,
                              [[maybe_unused]] const amrex::ParticleReal yp,
                              [[maybe_unused]] const amrex::ParticleReal zp,
                              const amrex::ParticleReal wq,
                              const amrex::ParticleReal vx,
                              const amrex::ParticleReal vy,
                              const amrex::ParticleReal vz,
                              amrex::Array4<amrex::Real> const& jx_arr,
                              amrex::Array4<amrex::Real> const& jy_arr,
                              amrex::Array4<amrex::Real> const& jz_arr,
                              amrex::IntVect const& jx_type,
                              amrex::IntVect const& jy_type,
                              amrex::IntVect const& jz_type,
                              const amrex::Real relative_time,
                              const amrex::XDim3 & dinv,
                              const amrex::XDim3 & xyzmin,
                              const amrex::Real invvol,
                              const amrex::Dim3 lo,
                              [[maybe_unused]] const int n_rz_azimuthal_modes)
{
    using namespace amrex::literals;

    constexpr int NODE = amrex::IndexType::NODE;
    constexpr int CELL = amrex::IndexType::CELL;

    // wqx, wqy wqz are particle current in each direction
#if defined(WARPX_DIM_RZ)
    // In RZ, wqx is actually wqr, and wqy is wqtheta
    // Convert to cylindrical at the mid point
    const amrex::Real xpmid = xp + relative_time*vx;
    const amrex::Real ypmid = yp + relative_time*vy;
    const amrex::Real rpmid = std::sqrt(xpmid*xpmid + ypmid*ypmid);
    const amrex::Real costheta = (rpmid > 0._rt ? xpmid/rpmid : 1._rt);
    const amrex::Real sintheta = (rpmid > 0._rt ? ypmid/rpmid : 0._rt);
    const Complex xy0 = Complex{costheta, sintheta};
    const amrex::Real wqx = wq*invvol*(+vx*costheta + vy*sintheta);
    const amrex::Real wqy = wq*invvol*(-vx*sintheta + vy*costheta);
#else
    const amrex::Real wqx = wq*invvol*vx;
    const amrex::Real wqy = wq*invvol*vy;
#endif
    const amrex::Real wqz = wq*invvol*vz;

    // --- Compute shape factors
    Compute_shape_factor< depos_order > const compute_shape_factor;
#if (AMREX_SPACEDIM >= 2)
    // x direction
    // Get particle position after 1/2 push back in position
    // Keep these double to avoid bug in single precision
#if defined(WARPX_DIM_RZ)
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
    if (jx_type[0] == NODE || jy_type[0] == NODE || jz_type[0] == NODE) {
        j_node = compute_shape_factor(sx_node, xmid);
    }
    if (jx_type[0] == CELL || jy_type[0] == CELL || jz_type[0] == CELL) {
        j_cell = compute_shape_factor(sx_cell, xmid - 0.5);
    }

    amrex::Real sx_jx[depos_order + 1] = {0._rt};
    amrex::Real sx_jy[depos_order + 1] = {0._rt};
    amrex::Real sx_jz[depos_order + 1] = {0._rt};
    for (int ix=0; ix<=depos_order; ix++)
    {
        sx_jx[ix] = ((jx_type[0] == NODE) ? amrex::Real(sx_node[ix]) : amrex::Real(sx_cell[ix]));
        sx_jy[ix] = ((jy_type[0] == NODE) ? amrex::Real(sx_node[ix]) : amrex::Real(sx_cell[ix]));
        sx_jz[ix] = ((jz_type[0] == NODE) ? amrex::Real(sx_node[ix]) : amrex::Real(sx_cell[ix]));
    }

    int const j_jx = ((jx_type[0] == NODE) ? j_node : j_cell);
    int const j_jy = ((jy_type[0] == NODE) ? j_node : j_cell);
    int const j_jz = ((jz_type[0] == NODE) ? j_node : j_cell);
#endif //AMREX_SPACEDIM >= 2

#if defined(WARPX_DIM_3D)
    // y direction
    // Keep these double to avoid bug in single precision
    const double ymid = ((yp - xyzmin.y) + relative_time*vy)*dinv.y;
    double sy_node[depos_order + 1] = {0.};
    double sy_cell[depos_order + 1] = {0.};
    int k_node = 0;
    int k_cell = 0;
    if (jx_type[1] == NODE || jy_type[1] == NODE || jz_type[1] == NODE) {
        k_node = compute_shape_factor(sy_node, ymid);
    }
    if (jx_type[1] == CELL || jy_type[1] == CELL || jz_type[1] == CELL) {
        k_cell = compute_shape_factor(sy_cell, ymid - 0.5);
    }
    amrex::Real sy_jx[depos_order + 1] = {0._rt};
    amrex::Real sy_jy[depos_order + 1] = {0._rt};
    amrex::Real sy_jz[depos_order + 1] = {0._rt};
    for (int iy=0; iy<=depos_order; iy++)
    {
        sy_jx[iy] = ((jx_type[1] == NODE) ? amrex::Real(sy_node[iy]) : amrex::Real(sy_cell[iy]));
        sy_jy[iy] = ((jy_type[1] == NODE) ? amrex::Real(sy_node[iy]) : amrex::Real(sy_cell[iy]));
        sy_jz[iy] = ((jz_type[1] == NODE) ? amrex::Real(sy_node[iy]) : amrex::Real(sy_cell[iy]));
    }
    int const k_jx = ((jx_type[1] == NODE) ? k_node : k_cell);
    int const k_jy = ((jy_type[1] == NODE) ? k_node : k_cell);
    int const k_jz = ((jz_type[1] == NODE) ? k_node : k_cell);
#endif

    // z direction
    // Keep these double to avoid bug in single precision
    constexpr int zdir = WARPX_ZINDEX;
    const double zmid = ((zp - xyzmin.z) + relative_time*vz)*dinv.z;
    double sz_node[depos_order + 1] = {0.};
    double sz_cell[depos_order + 1] = {0.};
    int l_node = 0;
    int l_cell = 0;
    if (jx_type[zdir] == NODE || jy_type[zdir] == NODE || jz_type[zdir] == NODE) {
        l_node = compute_shape_factor(sz_node, zmid);
    }
    if (jx_type[zdir] == CELL || jy_type[zdir] == CELL || jz_type[zdir] == CELL) {
        l_cell = compute_shape_factor(sz_cell, zmid - 0.5);
    }
    amrex::Real sz_jx[depos_order + 1] = {0._rt};
    amrex::Real sz_jy[depos_order + 1] = {0._rt};
    amrex::Real sz_jz[depos_order + 1] = {0._rt};
    for (int iz=0; iz<=depos_order; iz++)
    {
        sz_jx[iz] = ((jx_type[zdir] == NODE) ? amrex::Real(sz_node[iz]) : amrex::Real(sz_cell[iz]));
        sz_jy[iz] = ((jy_type[zdir] == NODE) ? amrex::Real(sz_node[iz]) : amrex::Real(sz_cell[iz]));
        sz_jz[iz] = ((jz_type[zdir] == NODE) ? amrex::Real(sz_node[iz]) : amrex::Real(sz_cell[iz]));
    }
    int const l_jx = ((jx_type[zdir] == NODE) ? l_node : l_cell);
    int const l_jy = ((jy_type[zdir] == NODE) ? l_node : l_cell);
    int const l_jz = ((jz_type[zdir] == NODE) ? l_node : l_cell);

    // Deposit current into jx_arr, jy_arr and jz_arr
#if defined(WARPX_DIM_1D_Z)
    for (int iz=0; iz<=depos_order; iz++){
        amrex::Gpu::Atomic::AddNoRet(
            &jx_arr(lo.x+l_jx+iz, 0, 0, 0),
            sz_jx[iz]*wqx);
        amrex::Gpu::Atomic::AddNoRet(
            &jy_arr(lo.x+l_jy+iz, 0, 0, 0),
            sz_jy[iz]*wqy);
        amrex::Gpu::Atomic::AddNoRet(
            &jz_arr(lo.x+l_jz+iz, 0, 0, 0),
            sz_jz[iz]*wqz);
    }
#endif
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    for (int iz=0; iz<=depos_order; iz++){
        for (int ix=0; ix<=depos_order; ix++){
            amrex::Gpu::Atomic::AddNoRet(
                &jx_arr(lo.x+j_jx+ix, lo.y+l_jx+iz, 0, 0),
                sx_jx[ix]*sz_jx[iz]*wqx);
            amrex::Gpu::Atomic::AddNoRet(
                &jy_arr(lo.x+j_jy+ix, lo.y+l_jy+iz, 0, 0),
                sx_jy[ix]*sz_jy[iz]*wqy);
            amrex::Gpu::Atomic::AddNoRet(
                &jz_arr(lo.x+j_jz+ix, lo.y+l_jz+iz, 0, 0),
                sx_jz[ix]*sz_jz[iz]*wqz);
#if defined(WARPX_DIM_RZ)
            Complex xy = xy0; // Note that xy is equal to e^{i m theta}
            for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                // The factor 2 on the weighting comes from the normalization of the modes
                amrex::Gpu::Atomic::AddNoRet( &jx_arr(lo.x+j_jx+ix, lo.y+l_jx+iz, 0, 2*imode-1), 2._rt*sx_jx[ix]*sz_jx[iz]*wqx*xy.real());
                amrex::Gpu::Atomic::AddNoRet( &jx_arr(lo.x+j_jx+ix, lo.y+l_jx+iz, 0, 2*imode  ), 2._rt*sx_jx[ix]*sz_jx[iz]*wqx*xy.imag());
                amrex::Gpu::Atomic::AddNoRet( &jy_arr(lo.x+j_jy+ix, lo.y+l_jy+iz, 0, 2*imode-1), 2._rt*sx_jy[ix]*sz_jy[iz]*wqy*xy.real());
                amrex::Gpu::Atomic::AddNoRet( &jy_arr(lo.x+j_jy+ix, lo.y+l_jy+iz, 0, 2*imode  ), 2._rt*sx_jy[ix]*sz_jy[iz]*wqy*xy.imag());
                amrex::Gpu::Atomic::AddNoRet( &jz_arr(lo.x+j_jz+ix, lo.y+l_jz+iz, 0, 2*imode-1), 2._rt*sx_jz[ix]*sz_jz[iz]*wqz*xy.real());
                amrex::Gpu::Atomic::AddNoRet( &jz_arr(lo.x+j_jz+ix, lo.y+l_jz+iz, 0, 2*imode  ), 2._rt*sx_jz[ix]*sz_jz[iz]*wqz*xy.imag());
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
                    &jx_arr(lo.x+j_jx+ix, lo.y+k_jx+iy, lo.z+l_jx+iz),
                    sx_jx[ix]*sy_jx[iy]*sz_jx[iz]*wqx);
                amrex::Gpu::Atomic::AddNoRet(
                    &jy_arr(lo.x+j_jy+ix, lo.y+k_jy+iy, lo.z+l_jy+iz),
                    sx_jy[ix]*sy_jy[iy]*sz_jy[iz]*wqy);
                amrex::Gpu::Atomic::AddNoRet(
                    &jz_arr(lo.x+j_jz+ix, lo.y+k_jz+iy, lo.z+l_jz+iz),
                    sx_jz[ix]*sy_jz[iy]*sz_jz[iz]*wqz);
            }
        }
    }
#endif
}

/**
 * \brief Current Deposition for thread thread_num
 * \tparam depos_order deposition order
 * \param GetPosition  A functor for returning the particle position.
 * \param wp           Pointer to array of particle weights.
 * \param uxp,uyp,uzp  Pointer to arrays of particle momentum.
 * \param ion_lev      Pointer to array of particle ionization level. This is
                         required to have the charge of each macroparticle
                         since q is a scalar. For non-ionizable species,
                         ion_lev is a null pointer.
 * \param jx_fab,jy_fab,jz_fab FArrayBox of current density, either full array or tile.
 * \param np_to_deposit Number of particles for which current is deposited.
 * \param relative_time Time at which to deposit J, relative to the time of the
 *                      current positions of the particles. When different than 0,
 *                      the particle position will be temporarily modified to match
 *                      the time of the deposition.
 * \param dinv         3D cell size inverse
 * \param xyzmin       Physical lower bounds of domain.
 * \param lo           Index lower bounds of domain.
 * \param q            species charge.
 * \param n_rz_azimuthal_modes Number of azimuthal modes when using RZ geometry.
 */
template <int depos_order>
void doDepositionShapeN (const GetParticlePosition<PIdx>& GetPosition,
                         const amrex::ParticleReal * const wp,
                         const amrex::ParticleReal * const uxp,
                         const amrex::ParticleReal * const uyp,
                         const amrex::ParticleReal * const uzp,
                         const int* ion_lev,
                         amrex::FArrayBox& jx_fab,
                         amrex::FArrayBox& jy_fab,
                         amrex::FArrayBox& jz_fab,
                         long np_to_deposit,
                         amrex::Real relative_time,
                         const amrex::XDim3 & dinv,
                         const amrex::XDim3 & xyzmin,
                         amrex::Dim3 lo,
                         amrex::Real q,
                         [[maybe_unused]]int n_rz_azimuthal_modes)
{
    using namespace amrex::literals;

    // Whether ion_lev is a null pointer (do_ionization=0) or a real pointer
    // (do_ionization=1)
    const bool do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

    const amrex::Real clightsq = 1.0_rt/PhysConst::c/PhysConst::c;

    amrex::Array4<amrex::Real> const& jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const& jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const& jz_arr = jz_fab.array();
    amrex::IntVect const jx_type = jx_fab.box().type();
    amrex::IntVect const jy_type = jy_fab.box().type();
    amrex::IntVect const jz_type = jz_fab.box().type();

    // Loop over particles and deposit into jx_fab, jy_fab and jz_fab
    amrex::ParallelFor(
        np_to_deposit,
        [=] AMREX_GPU_DEVICE (long ip) {
            amrex::ParticleReal xp, yp, zp;
            GetPosition(ip, xp, yp, zp);

            // --- Get particle quantities
            const amrex::Real gaminv = 1.0_rt/std::sqrt(1.0_rt + uxp[ip]*uxp[ip]*clightsq
                                                        + uyp[ip]*uyp[ip]*clightsq
                                                        + uzp[ip]*uzp[ip]*clightsq);
            const amrex::Real vx  = uxp[ip]*gaminv;
            const amrex::Real vy  = uyp[ip]*gaminv;
            const amrex::Real vz  = uzp[ip]*gaminv;

            amrex::Real wq  = q*wp[ip];
            if (do_ionization){
                wq *= ion_lev[ip];
            }

            doDepositionShapeNKernel<depos_order>(xp, yp, zp, wq, vx, vy, vz, jx_arr, jy_arr, jz_arr,
                                                  jx_type, jy_type, jz_type,
                                                  relative_time, dinv, xyzmin,
                                                  invvol, lo, n_rz_azimuthal_modes);

        }
    );
}

/**
 * \brief Direct current deposition for thread thread_num for the implicit scheme
 *        The only difference from doDepositionShapeN is in how the particle gamma
 *        is calculated.
 * \tparam depos_order deposition order
 * \param GetPosition  A functor for returning the particle position.
 * \param wp           Pointer to array of particle weights.
 * \param uxp_n,uyp_n,uzp_n  Pointer to arrays of particle momentum at time n.
 * \param uxp,uyp,uzp  Pointer to arrays of particle momentum at time n+1/2.
 * \param ion_lev      Pointer to array of particle ionization level. This is
                         required to have the charge of each macroparticle
                         since q is a scalar. For non-ionizable species,
                         ion_lev is a null pointer.
 * \param jx_fab,jy_fab,jz_fab FArrayBox of current density, either full array or tile.
 * \param np_to_deposit Number of particles for which current is deposited.
 * \param dinv         3D cell size inverse
 * \param xyzmin       Physical lower bounds of domain.
 * \param lo           Index lower bounds of domain.
 * \param q            species charge.
 * \param n_rz_azimuthal_modes Number of azimuthal modes when using RZ geometry.
 */
template <int depos_order>
void doDepositionShapeNImplicit(const GetParticlePosition<PIdx>& GetPosition,
                                const amrex::ParticleReal * const wp,
                                const amrex::ParticleReal * const uxp_n,
                                const amrex::ParticleReal * const uyp_n,
                                const amrex::ParticleReal * const uzp_n,
                                const amrex::ParticleReal * const uxp,
                                const amrex::ParticleReal * const uyp,
                                const amrex::ParticleReal * const uzp,
                                const int * const ion_lev,
                                amrex::FArrayBox& jx_fab,
                                amrex::FArrayBox& jy_fab,
                                amrex::FArrayBox& jz_fab,
                                const long np_to_deposit,
                                const amrex::XDim3 & dinv,
                                const amrex::XDim3 & xyzmin,
                                const amrex::Dim3 lo,
                                const amrex::Real q,
                                [[maybe_unused]]const int n_rz_azimuthal_modes)
{
    using namespace amrex::literals;

    // Whether ion_lev is a null pointer (do_ionization=0) or a real pointer
    // (do_ionization=1)
    const bool do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

    amrex::Array4<amrex::Real> const& jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const& jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const& jz_arr = jz_fab.array();
    amrex::IntVect const jx_type = jx_fab.box().type();
    amrex::IntVect const jy_type = jy_fab.box().type();
    amrex::IntVect const jz_type = jz_fab.box().type();

    // Loop over particles and deposit into jx_fab, jy_fab and jz_fab
    amrex::ParallelFor(
            np_to_deposit,
            [=] AMREX_GPU_DEVICE (long ip) {
            amrex::ParticleReal xp, yp, zp;
            GetPosition(ip, xp, yp, zp);

            constexpr amrex::ParticleReal inv_c2 = 1._prt/(PhysConst::c*PhysConst::c);

            // Compute inverse Lorentz factor, the average of gamma at time levels n and n+1
            // The uxp,uyp,uzp are the velocities at time level n+1/2
            const amrex::ParticleReal uxp_np1 = 2._prt*uxp[ip] - uxp_n[ip];
            const amrex::ParticleReal uyp_np1 = 2._prt*uyp[ip] - uyp_n[ip];
            const amrex::ParticleReal uzp_np1 = 2._prt*uzp[ip] - uzp_n[ip];
            const amrex::ParticleReal gamma_n = std::sqrt(1._prt + (uxp_n[ip]*uxp_n[ip] + uyp_n[ip]*uyp_n[ip] + uzp_n[ip]*uzp_n[ip])*inv_c2);
            const amrex::ParticleReal gamma_np1 = std::sqrt(1._prt + (uxp_np1*uxp_np1 + uyp_np1*uyp_np1 + uzp_np1*uzp_np1)*inv_c2);
            const amrex::ParticleReal gaminv = 2.0_prt/(gamma_n + gamma_np1);

            const amrex::Real vx  = uxp[ip]*gaminv;
            const amrex::Real vy  = uyp[ip]*gaminv;
            const amrex::Real vz  = uzp[ip]*gaminv;

            amrex::Real wq  = q*wp[ip];
            if (do_ionization){
                wq *= ion_lev[ip];
            }

            const amrex::Real relative_time = 0._rt;
            doDepositionShapeNKernel<depos_order>(xp, yp, zp, wq, vx, vy, vz, jx_arr, jy_arr, jz_arr,
                                                  jx_type, jy_type, jz_type,
                                                  relative_time, dinv, xyzmin,
                                                  invvol, lo, n_rz_azimuthal_modes);

        }
    );
}

/**
 * \brief Current Deposition for thread thread_num using shared memory
 * \tparam depos_order deposition order
 * \param GetPosition  A functor for returning the particle position.
 * \param wp           Pointer to array of particle weights.
 * \param uxp,uyp,uzp  Pointer to arrays of particle momentum.
 * \param ion_lev      Pointer to array of particle ionization level. This is
                         required to have the charge of each macroparticle
                         since q is a scalar. For non-ionizable species,
                         ion_lev is a null pointer.
 * \param jx_fab,jy_fab,jz_fab FArrayBox of current density, either full array or tile.
 * \param np_to_deposit Number of particles for which current is deposited.
 * \param dt           Time step for particle level
 * \param relative_time Time at which to deposit J, relative to the time of the
 *                      current positions of the particles. When different than 0,
 *                      the particle position will be temporarily modified to match
 *                      the time of the deposition.
 * \param dinv         3D cell size inverse
 * \param xyzmin       Physical lower bounds of domain.
 * \param lo           Index lower bounds of domain.
 * \param q            species charge.
 * \param n_rz_azimuthal_modes Number of azimuthal modes when using RZ geometry.
 */
template <int depos_order>
void doDepositionSharedShapeN (const GetParticlePosition<PIdx>& GetPosition,
                               const amrex::ParticleReal * const wp,
                               const amrex::ParticleReal * const uxp,
                               const amrex::ParticleReal * const uyp,
                               const amrex::ParticleReal * const uzp,
                               const int*  ion_lev,
                               amrex::FArrayBox& jx_fab,
                               amrex::FArrayBox& jy_fab,
                               amrex::FArrayBox& jz_fab,
                               long np_to_deposit,
                               const amrex::Real relative_time,
                               const amrex::XDim3 & dinv,
                               const amrex::XDim3 & xyzmin,
                               amrex::Dim3 lo,
                               amrex::Real q,
                               int n_rz_azimuthal_modes,
                               const amrex::DenseBins<WarpXParticleContainer::ParticleTileType::ParticleTileDataType>& a_bins,
                               const amrex::Box& box,
                               const amrex::Geometry& geom,
                               const amrex::IntVect& a_tbox_max_size)
{
    using namespace amrex::literals;

#if defined(AMREX_USE_HIP) || defined(AMREX_USE_CUDA)
    using namespace amrex;

    auto permutation = a_bins.permutationPtr();

    amrex::Array4<amrex::Real> const& jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const& jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const& jz_arr = jz_fab.array();
    amrex::IntVect const jx_type = jx_fab.box().type();
    amrex::IntVect const jy_type = jy_fab.box().type();
    amrex::IntVect const jz_type = jz_fab.box().type();

    constexpr int zdir = WARPX_ZINDEX;
    constexpr int NODE = amrex::IndexType::NODE;
    constexpr int CELL = amrex::IndexType::CELL;

    // Loop over particles and deposit into jx_fab, jy_fab and jz_fab
    const auto dxiarr = geom.InvCellSizeArray();
    const auto plo = geom.ProbLoArray();
    const auto domain = geom.Domain();

    amrex::Box sample_tbox(IntVect(AMREX_D_DECL(0,0,0)), a_tbox_max_size - 1);
    sample_tbox.grow(depos_order);

    amrex::Box sample_tbox_x = convert(sample_tbox, jx_type);
    amrex::Box sample_tbox_y = convert(sample_tbox, jy_type);
    amrex::Box sample_tbox_z = convert(sample_tbox, jz_type);

    const auto npts = amrex::max(sample_tbox_x.numPts(), sample_tbox_y.numPts(), sample_tbox_z.numPts());

    const int nblocks = a_bins.numBins();
    const int threads_per_block = WarpX::shared_mem_current_tpb;
    const auto offsets_ptr = a_bins.offsetsPtr();

    const std::size_t shared_mem_bytes = npts*sizeof(amrex::Real);
    const amrex::IntVect bin_size = WarpX::shared_tilesize;
    const std::size_t max_shared_mem_bytes = amrex::Gpu::Device::sharedMemPerBlock();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(shared_mem_bytes <= max_shared_mem_bytes,
                                     "Tile size too big for GPU shared memory current deposition");

    amrex::ignore_unused(np_to_deposit);
    // Launch one thread-block per bin
    amrex::launch(
            nblocks, threads_per_block, shared_mem_bytes, amrex::Gpu::gpuStream(),
            [=] AMREX_GPU_DEVICE () noexcept {
        const int bin_id = blockIdx.x;
        const unsigned int bin_start = offsets_ptr[bin_id];
        const unsigned int bin_stop = offsets_ptr[bin_id+1];

        if (bin_start == bin_stop) { return; /*this bin has no particles*/ }

        // These boxes define the index space for the shared memory buffers
        amrex::Box buffer_box;
        {
            ParticleReal xp, yp, zp;
            GetPosition(permutation[bin_start], xp, yp, zp);
#if defined(WARPX_DIM_3D)
            IntVect iv = IntVect(int( amrex::Math::floor((xp-plo[0]) * dxiarr[0]) ),
                                 int( amrex::Math::floor((yp-plo[1]) * dxiarr[1]) ),
                                 int( amrex::Math::floor((zp-plo[2]) * dxiarr[2]) ));
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            IntVect iv = IntVect(int( amrex::Math::floor((xp-plo[0]) * dxiarr[0]) ),
                                 int( amrex::Math::floor((zp-plo[1]) * dxiarr[1]) ));
#elif defined(WARPX_DIM_1D_Z)
            IntVect iv = IntVect(int( amrex::Math::floor((zp-plo[0]) * dxiarr[0]) ));
#endif
            iv += domain.smallEnd();
            getTileIndex(iv, box, true, bin_size, buffer_box);
        }

        buffer_box.grow(depos_order);
        Box tbox_x = convert(buffer_box, jx_type);
        Box tbox_y = convert(buffer_box, jy_type);
        Box tbox_z = convert(buffer_box, jz_type);

        Gpu::SharedMemory<amrex::Real> gsm;
        amrex::Real* const shared = gsm.dataPtr();

        amrex::Array4<amrex::Real> const jx_buff(shared,
                amrex::begin(tbox_x), amrex::end(tbox_x), 1);
        amrex::Array4<amrex::Real> const jy_buff(shared,
                amrex::begin(tbox_y), amrex::end(tbox_y), 1);
        amrex::Array4<amrex::Real> const jz_buff(shared,
                amrex::begin(tbox_z), amrex::end(tbox_z), 1);

        // Zero-initialize the temporary array in shared memory
        volatile amrex::Real* vs = shared;
        for (int i = threadIdx.x; i < npts; i += blockDim.x){
            vs[i] = 0.0;
        }
        __syncthreads();
        for (unsigned int ip_orig = bin_start+threadIdx.x; ip_orig<bin_stop; ip_orig += blockDim.x)
        {
            const unsigned int ip = permutation[ip_orig];
            depositComponent<depos_order>(GetPosition, wp, uxp, uyp, uzp, ion_lev, jx_buff, jx_type,
                                          relative_time, dinv, xyzmin, lo, q, n_rz_azimuthal_modes,
                                          ip, zdir, NODE, CELL, 0);
        }

        __syncthreads();
        addLocalToGlobal(tbox_x, jx_arr, jx_buff);
        for (int i = threadIdx.x; i < npts; i += blockDim.x){
            vs[i] = 0.0;
        }

        __syncthreads();
        for (unsigned int ip_orig = bin_start+threadIdx.x; ip_orig<bin_stop; ip_orig += blockDim.x)
        {
            const unsigned int ip = permutation[ip_orig];
            depositComponent<depos_order>(GetPosition, wp, uxp, uyp, uzp, ion_lev, jy_buff, jy_type,
                                          relative_time, dinv, xyzmin, lo, q, n_rz_azimuthal_modes,
                                          ip, zdir, NODE, CELL, 1);
        }

        __syncthreads();
        addLocalToGlobal(tbox_y, jy_arr, jy_buff);
        for (int i = threadIdx.x; i < npts; i += blockDim.x){
            vs[i] = 0.0;
        }

        __syncthreads();
        for (unsigned int ip_orig = bin_start+threadIdx.x; ip_orig<bin_stop; ip_orig += blockDim.x)
        {
            const unsigned int ip = permutation[ip_orig];
            depositComponent<depos_order>(GetPosition, wp, uxp, uyp, uzp, ion_lev, jz_buff, jz_type,
                                          relative_time, dinv, xyzmin, lo, q, n_rz_azimuthal_modes,
                                          ip, zdir, NODE, CELL, 2);
        }

        __syncthreads();
        addLocalToGlobal(tbox_z, jz_arr, jz_buff);
    });
#else // not using hip/cuda
    // Note, you should never reach this part of the code. This funcion cannot be called unless
    // using HIP/CUDA, and those things are checked prior
    //don't use any args
    ignore_unused(GetPosition, wp, uxp, uyp, uzp, ion_lev, jx_fab, jy_fab, jz_fab, np_to_deposit, relative_time, dinv, xyzmin, lo, q, n_rz_azimuthal_modes, a_bins, box, geom, a_tbox_max_size);
    WARPX_ABORT_WITH_MESSAGE("Shared memory only implemented for HIP/CUDA");
#endif
}

/**
 * \brief Esirkepov Current Deposition for thread thread_num
 *
 * \tparam depos_order  deposition order
 * \param GetPosition  A functor for returning the particle position.
 * \param wp           Pointer to array of particle weights.
 * \param uxp,uyp,uzp  Pointer to arrays of particle momentum.
 * \param ion_lev      Pointer to array of particle ionization level. This is
                       required to have the charge of each macroparticle
                       since q is a scalar. For non-ionizable species,
                       ion_lev is a null pointer.
 * \param Jx_arr,Jy_arr,Jz_arr Array4 of current density, either full array or tile.
 * \param np_to_deposit Number of particles for which current is deposited.
 * \param dt           Time step for particle level
 * \param[in] relative_time Time at which to deposit J, relative to the time of the
 *                          current positions of the particles. When different than 0,
 *                          the particle position will be temporarily modified to match
 *                          the time of the deposition.
 * \param dinv         3D cell size inverse
 * \param xyzmin       Physical lower bounds of domain.
 * \param lo           Index lower bounds of domain.
 * \param q            species charge.
 * \param n_rz_azimuthal_modes Number of azimuthal modes when using RZ geometry.
 * \param reduced_particle_shape_mask  Array4 of int, Mask that indicates whether a particle
 * should use its regular shape factor or a reduced, order-1 shape factor instead in a given cell.
 * \param enable_reduced_shape Flag to indicate whether to use the reduced shape factor
 */
template <int depos_order>
void doEsirkepovDepositionShapeN (const GetParticlePosition<PIdx>& GetPosition,
                                  const amrex::ParticleReal * const wp,
                                  const amrex::ParticleReal * const uxp,
                                  const amrex::ParticleReal * const uyp,
                                  const amrex::ParticleReal * const uzp,
                                  const int* ion_lev,
                                  const amrex::Array4<amrex::Real>& Jx_arr,
                                  const amrex::Array4<amrex::Real>& Jy_arr,
                                  const amrex::Array4<amrex::Real>& Jz_arr,
                                  long np_to_deposit,
                                  amrex::Real dt,
                                  amrex::Real relative_time,
                                  const amrex::XDim3 & dinv,
                                  const amrex::XDim3 & xyzmin,
                                  amrex::Dim3 lo,
                                  amrex::Real q,
                                  [[maybe_unused]] int n_rz_azimuthal_modes,
                                  const amrex::Array4<const int>& reduced_particle_shape_mask,
                                  bool enable_reduced_shape
                                  )
{
    using namespace amrex;
    using namespace amrex::literals;

    // Whether ion_lev is a null pointer (do_ionization=0) or a real pointer
    // (do_ionization=1)
    bool const do_ionization = ion_lev;
#if !defined(WARPX_DIM_3D)
    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;
#endif

    amrex::XDim3 const invdtd = amrex::XDim3{(1.0_rt/dt)*dinv.y*dinv.z,
                                             (1.0_rt/dt)*dinv.x*dinv.z,
                                             (1.0_rt/dt)*dinv.x*dinv.y};

    Real constexpr clightsq = 1.0_rt / ( PhysConst::c * PhysConst::c );

#if !defined(WARPX_DIM_1D_Z)
    Real constexpr one_third = 1.0_rt / 3.0_rt;
    Real constexpr one_sixth = 1.0_rt / 6.0_rt;
#endif

    // Loop over particles and deposit into Jx_arr, Jy_arr and Jz_arr

    // (Compile 2 versions of the kernel: with and without reduced shape)
    enum eb_flags : int { has_reduced_shape, no_reduced_shape };
    const int reduce_shape_runtime_flag = (enable_reduced_shape && (depos_order>1))? has_reduced_shape : no_reduced_shape;

    amrex::ParallelFor( TypeList<CompileTimeOptions<has_reduced_shape,no_reduced_shape>>{},
        {reduce_shape_runtime_flag},
        np_to_deposit, [=] AMREX_GPU_DEVICE (long ip, auto reduce_shape_control) {
            // --- Get particle quantities
            Real const gaminv = 1.0_rt/std::sqrt(1.0_rt + uxp[ip]*uxp[ip]*clightsq
                                                 + uyp[ip]*uyp[ip]*clightsq
                                                 + uzp[ip]*uzp[ip]*clightsq);

            Real wq = q*wp[ip];
            if (do_ionization){
                wq *= ion_lev[ip];
            }

            ParticleReal xp, yp, zp;
            GetPosition(ip, xp, yp, zp);

            // computes current and old position in grid units
#if defined(WARPX_DIM_RZ)
            Real const xp_new = xp + (relative_time + 0.5_rt*dt)*uxp[ip]*gaminv;
            Real const yp_new = yp + (relative_time + 0.5_rt*dt)*uyp[ip]*gaminv;
            Real const xp_mid = xp_new - 0.5_rt*dt*uxp[ip]*gaminv;
            Real const yp_mid = yp_new - 0.5_rt*dt*uyp[ip]*gaminv;
            Real const xp_old = xp_new - dt*uxp[ip]*gaminv;
            Real const yp_old = yp_new - dt*uyp[ip]*gaminv;
            Real const rp_new = std::sqrt(xp_new*xp_new + yp_new*yp_new);
            Real const rp_mid = std::sqrt(xp_mid*xp_mid + yp_mid*yp_mid);
            Real const rp_old = std::sqrt(xp_old*xp_old + yp_old*yp_old);
            const amrex::Real costheta_mid = (rp_mid > 0._rt ? xp_mid/rp_mid : 1._rt);
            const amrex::Real sintheta_mid = (rp_mid > 0._rt ? yp_mid/rp_mid : 0._rt);
            const amrex::Real costheta_new = (rp_new > 0._rt ? xp_new/rp_new : 1._rt);
            const amrex::Real sintheta_new = (rp_new > 0._rt ? yp_new/rp_new : 0._rt);
            const amrex::Real costheta_old = (rp_old > 0._rt ? xp_old/rp_old : 1._rt);
            const amrex::Real sintheta_old = (rp_old > 0._rt ? yp_old/rp_old : 0._rt);
            const Complex xy_new0 = Complex{costheta_new, sintheta_new};
            const Complex xy_mid0 = Complex{costheta_mid, sintheta_mid};
            const Complex xy_old0 = Complex{costheta_old, sintheta_old};
            // Keep these double to avoid bug in single precision
            double const x_new = (rp_new - xyzmin.x)*dinv.x;
            double const x_old = (rp_old - xyzmin.x)*dinv.x;
#else
#if !defined(WARPX_DIM_1D_Z)
            // Keep these double to avoid bug in single precision
            double const x_new = (xp - xyzmin.x + (relative_time + 0.5_rt*dt)*uxp[ip]*gaminv)*dinv.x;
            double const x_old = x_new - dt*dinv.x*uxp[ip]*gaminv;
#endif
#endif
#if defined(WARPX_DIM_3D)
            // Keep these double to avoid bug in single precision
            double const y_new = (yp - xyzmin.y + (relative_time + 0.5_rt*dt)*uyp[ip]*gaminv)*dinv.y;
            double const y_old = y_new - dt*dinv.y*uyp[ip]*gaminv;
#endif
            // Keep these double to avoid bug in single precision
            double const z_new = (zp - xyzmin.z + (relative_time + 0.5_rt*dt)*uzp[ip]*gaminv)*dinv.z;
            double const z_old = z_new - dt*dinv.z*uzp[ip]*gaminv;

            // Check whether the particle is close to the EB at the old and new position
            bool reduce_shape_old, reduce_shape_new;
#ifdef AMREX_USE_CUDA
            amrex::ignore_unused(reduced_particle_shape_mask, lo); // Needed to avoid compilation error with nvcc
#endif
            if constexpr (reduce_shape_control == has_reduced_shape) {
#if defined(WARPX_DIM_3D)
                reduce_shape_old = reduced_particle_shape_mask(
                    lo.x + int(amrex::Math::floor(x_old)),
                    lo.y + int(amrex::Math::floor(y_old)),
                    lo.z + int(amrex::Math::floor(z_old)));
                reduce_shape_new = reduced_particle_shape_mask(
                    lo.x + int(amrex::Math::floor(x_new)),
                    lo.y + int(amrex::Math::floor(y_new)),
                    lo.z + int(amrex::Math::floor(z_new)));
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                reduce_shape_old = reduced_particle_shape_mask(
                    lo.x + int(amrex::Math::floor(x_old)),
                    lo.y + int(amrex::Math::floor(z_old)),
                    0);
                reduce_shape_new = reduced_particle_shape_mask(
                    lo.x + int(amrex::Math::floor(x_new)),
                    lo.y + int(amrex::Math::floor(z_new)),
                    0);
#elif defined(WARPX_DIM_1D_Z)
                reduce_shape_old = reduced_particle_shape_mask(
                    lo.x + int(amrex::Math::floor(z_old)),
                    0, 0);
                reduce_shape_new = reduced_particle_shape_mask(
                    lo.x + int(amrex::Math::floor(z_new)),
                    0, 0);
#endif
            } else {
                reduce_shape_old = false;
                reduce_shape_new = false;
            }

#if defined(WARPX_DIM_RZ)
            Real const vy = (-uxp[ip]*sintheta_mid + uyp[ip]*costheta_mid)*gaminv;
#elif defined(WARPX_DIM_XZ)
            Real const vy = uyp[ip]*gaminv;
#elif defined(WARPX_DIM_1D_Z)
            Real const vx = uxp[ip]*gaminv;
            Real const vy = uyp[ip]*gaminv;
#endif

            // --- Compute shape factors
            // Compute shape factors for position as they are now and at old positions
            // [ijk]_new: leftmost grid point that the particle touches
            const Compute_shape_factor< depos_order > compute_shape_factor;
            const Compute_shifted_shape_factor< depos_order > compute_shifted_shape_factor;
            // In cells marked by reduced_particle_shape_mask, we need order 1 deposition
            const Compute_shifted_shape_factor< 1 > compute_shifted_shape_factor_order1;
            amrex::ignore_unused(compute_shifted_shape_factor_order1); // unused for `no_reduced_shape`

            // Shape factor arrays
            // Note that there are extra values above and below
            // to possibly hold the factor for the old particle
            // which can be at a different grid location.
            // Keep these double to avoid bug in single precision
#if !defined(WARPX_DIM_1D_Z)
            double sx_new[depos_order + 3] = {0.};
            double sx_old[depos_order + 3] = {0.};
            const int i_new = compute_shape_factor(sx_new+1, x_new );
            const int i_old = compute_shifted_shape_factor(sx_old, x_old, i_new);
            // If particle is close to the embedded boundary, recompute deposition with order 1 shape
            if constexpr (reduce_shape_control == has_reduced_shape) {
                if (reduce_shape_new) {
                    for (int i=0; i<depos_order+3; i++) {sx_new[i] = 0.;} // Erase previous deposition
                    compute_shifted_shape_factor_order1( sx_new+depos_order/2, x_new, i_new+depos_order/2 ); // Redeposit with order 1
                }
                if (reduce_shape_old) {
                    for (int i=0; i<depos_order+3; i++) {sx_old[i] = 0.;} // Erase previous deposition
                    compute_shifted_shape_factor_order1( sx_old+depos_order/2, x_old, i_new+depos_order/2 ); // Redeposit with order 1
                }
                // Note: depos_order/2 in the above code corresponds to the shift between the index of the lowest point
                // to which the particle can deposit, with shape of order `depos_order` vs with shape of order 1
            }
#endif
#if defined(WARPX_DIM_3D)
            double sy_new[depos_order + 3] = {0.};
            double sy_old[depos_order + 3] = {0.};
            const int j_new = compute_shape_factor(sy_new+1, y_new);
            const int j_old = compute_shifted_shape_factor(sy_old, y_old, j_new);
            // If particle is close to the embedded boundary, recompute deposition with order 1 shape
            if constexpr (reduce_shape_control == has_reduced_shape) {
                if (reduce_shape_new) {
                    for (int j=0; j<depos_order+3; j++) {sy_new[j] = 0.;} // Erase previous deposition
                    compute_shifted_shape_factor_order1( sy_new+depos_order/2, y_new, j_new+depos_order/2 ); // Redeposit with order 1
                }
                if (reduce_shape_old) {
                    for (int j=0; j<depos_order+3; j++) {sy_old[j] = 0.;} // Erase previous deposition
                    compute_shifted_shape_factor_order1( sy_old+depos_order/2, y_old, j_new+depos_order/2 ); // Redeposit with order 1
                }
                // Note: depos_order/2 in the above code corresponds to the shift between the index of the lowest point
                // to which the particle can deposit, with shape of order `depos_order` vs with shape of order 1
            }
#endif
            double sz_new[depos_order + 3] = {0.};
            double sz_old[depos_order + 3] = {0.};
            const int k_new = compute_shape_factor(sz_new+1, z_new );
            const int k_old = compute_shifted_shape_factor(sz_old, z_old, k_new );
            // If particle is close to the embedded boundary, recompute deposition with order 1 shape
            if constexpr (reduce_shape_control == has_reduced_shape) {
                if (reduce_shape_new) {
                    for (int k=0; k<depos_order+3; k++) {sz_new[k] = 0.;} // Erase previous deposition
                    compute_shifted_shape_factor_order1( sz_new+depos_order/2, z_new, k_new+depos_order/2 ); // Redeposit with order 1
                }
                if (reduce_shape_old) {
                    for (int k=0; k<depos_order+3; k++) {sz_old[k] = 0.;} // Erase previous deposition
                    compute_shifted_shape_factor_order1( sz_old+depos_order/2, z_old, k_new+depos_order/2 ); // Redeposit with order 1
                }
                // Note: depos_order/2 in the above code corresponds to the shift between the index of the lowest point
                // to which the particle can deposit, with shape of order `depos_order` vs with shape of order 1
            }

            // computes min/max positions of current contributions
#if !defined(WARPX_DIM_1D_Z)
            int dil = 1, diu = 1;
            if (i_old < i_new) { dil = 0; }
            if (i_old > i_new) { diu = 0; }
#endif
#if defined(WARPX_DIM_3D)
            int djl = 1, dju = 1;
            if (j_old < j_new) { djl = 0; }
            if (j_old > j_new) { dju = 0; }
#endif
            int dkl = 1, dku = 1;
            if (k_old < k_new) { dkl = 0; }
            if (k_old > k_new) { dku = 0; }

#if defined(WARPX_DIM_3D)

            for (int k=dkl; k<=depos_order+2-dku; k++) {
                for (int j=djl; j<=depos_order+2-dju; j++) {
                    amrex::Real sdxi = 0._rt;
                    for (int i=dil; i<=depos_order+1-diu; i++) {
                        sdxi += wq*invdtd.x*(sx_old[i] - sx_new[i])*(
                            one_third*(sy_new[j]*sz_new[k] + sy_old[j]*sz_old[k])
                           +one_sixth*(sy_new[j]*sz_old[k] + sy_old[j]*sz_new[k]));
                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdxi);
                    }
                }
            }
            for (int k=dkl; k<=depos_order+2-dku; k++) {
                for (int i=dil; i<=depos_order+2-diu; i++) {
                    amrex::Real sdyj = 0._rt;
                    for (int j=djl; j<=depos_order+1-dju; j++) {
                        sdyj += wq*invdtd.y*(sy_old[j] - sy_new[j])*(
                            one_third*(sx_new[i]*sz_new[k] + sx_old[i]*sz_old[k])
                           +one_sixth*(sx_new[i]*sz_old[k] + sx_old[i]*sz_new[k]));
                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdyj);
                    }
                }
            }
            for (int j=djl; j<=depos_order+2-dju; j++) {
                for (int i=dil; i<=depos_order+2-diu; i++) {
                    amrex::Real sdzk = 0._rt;
                    for (int k=dkl; k<=depos_order+1-dku; k++) {
                        sdzk += wq*invdtd.z*(sz_old[k] - sz_new[k])*(
                            one_third*(sx_new[i]*sy_new[j] + sx_old[i]*sy_old[j])
                           +one_sixth*(sx_new[i]*sy_old[j] + sx_old[i]*sy_new[j]));
                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdzk);
                    }
                }
            }

#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)

            for (int k=dkl; k<=depos_order+2-dku; k++) {
                amrex::Real sdxi = 0._rt;
                for (int i=dil; i<=depos_order+1-diu; i++) {
                    sdxi += wq*invdtd.x*(sx_old[i] - sx_new[i])*0.5_rt*(sz_new[k] + sz_old[k]);
                    amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdxi);
#if defined(WARPX_DIM_RZ)
                    Complex xy_mid = xy_mid0; // Throughout the following loop, xy_mid takes the value e^{i m theta}
                    for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                        // The factor 2 comes from the normalization of the modes
                        const Complex djr_cmplx = 2._rt *sdxi*xy_mid;
                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djr_cmplx.real());
                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djr_cmplx.imag());
                        xy_mid = xy_mid*xy_mid0;
                    }
#endif
                }
            }
            for (int k=dkl; k<=depos_order+2-dku; k++) {
                for (int i=dil; i<=depos_order+2-diu; i++) {
                    Real const sdyj = wq*vy*invvol*(
                        one_third*(sx_new[i]*sz_new[k] + sx_old[i]*sz_old[k])
                       +one_sixth*(sx_new[i]*sz_old[k] + sx_old[i]*sz_new[k]));
                    amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdyj);
#if defined(WARPX_DIM_RZ)
                    Complex const I = Complex{0._rt, 1._rt};
                    Complex xy_new = xy_new0;
                    Complex xy_mid = xy_mid0;
                    Complex xy_old = xy_old0;
                    // Throughout the following loop, xy_ takes the value e^{i m theta_}
                    for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                        // The factor 2 comes from the normalization of the modes
                        // The minus sign comes from the different convention with respect to Davidson et al.
                        const Complex djt_cmplx = -2._rt * I*(i_new-1 + i + xyzmin.x*dinv.x)*wq*invdtd.x/(amrex::Real)imode
                                                  *(Complex(sx_new[i]*sz_new[k], 0._rt)*(xy_new - xy_mid)
                                                  + Complex(sx_old[i]*sz_old[k], 0._rt)*(xy_mid - xy_old));
                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djt_cmplx.real());
                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djt_cmplx.imag());
                        xy_new = xy_new*xy_new0;
                        xy_mid = xy_mid*xy_mid0;
                        xy_old = xy_old*xy_old0;
                    }
#endif
                }
            }
            for (int i=dil; i<=depos_order+2-diu; i++) {
                Real sdzk = 0._rt;
                for (int k=dkl; k<=depos_order+1-dku; k++) {
                    sdzk += wq*invdtd.z*(sz_old[k] - sz_new[k])*0.5_rt*(sx_new[i] + sx_old[i]);
                    amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdzk);
#if defined(WARPX_DIM_RZ)
                    Complex xy_mid = xy_mid0; // Throughout the following loop, xy_mid takes the value e^{i m theta}
                    for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                        // The factor 2 comes from the normalization of the modes
                        const Complex djz_cmplx = 2._rt * sdzk * xy_mid;
                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djz_cmplx.real());
                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djz_cmplx.imag());
                        xy_mid = xy_mid*xy_mid0;
                    }
#endif
                }
            }
#elif defined(WARPX_DIM_1D_Z)

            for (int k=dkl; k<=depos_order+2-dku; k++) {
                amrex::Real const sdxi = wq*vx*invvol*0.5_rt*(sz_old[k] + sz_new[k]);
                amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+k_new-1+k, 0, 0, 0), sdxi);
            }
            for (int k=dkl; k<=depos_order+2-dku; k++) {
                amrex::Real const sdyj = wq*vy*invvol*0.5_rt*(sz_old[k] + sz_new[k]);
                amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+k_new-1+k, 0, 0, 0), sdyj);
            }
            amrex::Real sdzk = 0._rt;
            for (int k=dkl; k<=depos_order+1-dku; k++) {
                sdzk += wq*invdtd.z*(sz_old[k] - sz_new[k]);
                amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+k_new-1+k, 0, 0, 0), sdzk);
            }
#endif
        }
    );
}

/**
 * \brief Esirkepov Current Deposition for thread thread_num for implicit scheme
 *        The difference from doEsirkepovDepositionShapeN is in how the old and new
 *        particles positions are determined and in how the particle gamma is calculated.
 *
 * \tparam depos_order  deposition order
 * \param xp_n,yp_n,zp_n  Pointer to arrays of particle position at time level n.
 * \param GetPosition  A functor for returning the particle position.
 * \param wp           Pointer to array of particle weights.
 * \param uxp_n,uyp_n,uzp_n  Pointer to arrays of particle momentum at time level n.
 * \param uxp_nph,uyp_nph,uzp_nph  Pointer to arrays of particle momentum at time level n + 1/2.
 * \param ion_lev      Pointer to array of particle ionization level. This is
                       required to have the charge of each macroparticle
                       since q is a scalar. For non-ionizable species,
                       ion_lev is a null pointer.
 * \param Jx_arr,Jy_arr,Jz_arr Array4 of current density, either full array or tile.
 * \param np_to_deposit Number of particles for which current is deposited.
 * \param dt           Time step for particle level
 * \param dinv         3D cell size inverse
 * \param xyzmin       Physical lower bounds of domain.
 * \param lo           Index lower bounds of domain.
 * \param q            species charge.
 * \param n_rz_azimuthal_modes Number of azimuthal modes when using RZ geometry.
 */
template <int depos_order>
void doChargeConservingDepositionShapeNImplicit ([[maybe_unused]]const amrex::ParticleReal * const xp_n,
                                                 [[maybe_unused]]const amrex::ParticleReal * const yp_n,
                                                 [[maybe_unused]]const amrex::ParticleReal * const zp_n,
                                                 const GetParticlePosition<PIdx>& GetPosition,
                                                 const amrex::ParticleReal * const wp,
                                                 [[maybe_unused]]const amrex::ParticleReal * const uxp_n,
                                                 [[maybe_unused]]const amrex::ParticleReal * const uyp_n,
                                                 [[maybe_unused]]const amrex::ParticleReal * const uzp_n,
                                                 [[maybe_unused]]const amrex::ParticleReal * const uxp_nph,
                                                 [[maybe_unused]]const amrex::ParticleReal * const uyp_nph,
                                                 [[maybe_unused]]const amrex::ParticleReal * const uzp_nph,
                                                 const int * const ion_lev,
                                                 const amrex::Array4<amrex::Real>& Jx_arr,
                                                 const amrex::Array4<amrex::Real>& Jy_arr,
                                                 const amrex::Array4<amrex::Real>& Jz_arr,
                                                 const long np_to_deposit,
                                                 const amrex::Real dt,
                                                 const amrex::XDim3 & dinv,
                                                 const amrex::XDim3 & xyzmin,
                                                 const amrex::Dim3 lo,
                                                 const amrex::Real q,
                                                 [[maybe_unused]] const int n_rz_azimuthal_modes)
{
    using namespace amrex;

    // Whether ion_lev is a null pointer (do_ionization=0) or a real pointer
    // (do_ionization=1)
    bool const do_ionization = ion_lev;

#if !defined(WARPX_DIM_3D)
    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;
#endif

    amrex::XDim3 const invdtd = amrex::XDim3{(1.0_rt/dt)*dinv.y*dinv.z,
                                             (1.0_rt/dt)*dinv.x*dinv.z,
                                             (1.0_rt/dt)*dinv.x*dinv.y};

#if !defined(WARPX_DIM_1D_Z)
    Real constexpr one_third = 1.0_rt / 3.0_rt;
    Real constexpr one_sixth = 1.0_rt / 6.0_rt;
#endif

    // Loop over particles and deposit into Jx_arr, Jy_arr and Jz_arr
    amrex::ParallelFor(
        np_to_deposit,
        [=] AMREX_GPU_DEVICE (long const ip) {
#if !defined(WARPX_DIM_3D)
            constexpr amrex::ParticleReal inv_c2 = 1._prt/(PhysConst::c*PhysConst::c);

            // Compute inverse Lorentz factor, the average of gamma at time levels n and n+1
            // The uxp,uyp,uzp are the velocities at time level n+1/2
            const amrex::ParticleReal uxp_np1 = 2._prt*uxp_nph[ip] - uxp_n[ip];
            const amrex::ParticleReal uyp_np1 = 2._prt*uyp_nph[ip] - uyp_n[ip];
            const amrex::ParticleReal uzp_np1 = 2._prt*uzp_nph[ip] - uzp_n[ip];
            const amrex::ParticleReal gamma_n = std::sqrt(1._prt + (uxp_n[ip]*uxp_n[ip] + uyp_n[ip]*uyp_n[ip] + uzp_n[ip]*uzp_n[ip])*inv_c2);
            const amrex::ParticleReal gamma_np1 = std::sqrt(1._prt + (uxp_np1*uxp_np1 + uyp_np1*uyp_np1 + uzp_np1*uzp_np1)*inv_c2);
            const amrex::ParticleReal gaminv = 2.0_prt/(gamma_n + gamma_np1);
#endif

            Real wq = q*wp[ip];
            if (do_ionization){
                wq *= ion_lev[ip];
            }

            ParticleReal xp_nph, yp_nph, zp_nph;
            GetPosition(ip, xp_nph, yp_nph, zp_nph);

#if !defined(WARPX_DIM_1D_Z)
            ParticleReal const xp_np1 = 2._prt*xp_nph - xp_n[ip];
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
            ParticleReal const yp_np1 = 2._prt*yp_nph - yp_n[ip];
#endif
            ParticleReal const zp_np1 = 2._prt*zp_nph - zp_n[ip];

            // computes current and old position in grid units
#if defined(WARPX_DIM_RZ)
            amrex::Real const xp_new = xp_np1;
            amrex::Real const yp_new = yp_np1;
            amrex::Real const xp_mid = xp_nph;
            amrex::Real const yp_mid = yp_nph;
            amrex::Real const xp_old = xp_n[ip];
            amrex::Real const yp_old = yp_n[ip];
            amrex::Real const rp_new = std::sqrt(xp_new*xp_new + yp_new*yp_new);
            amrex::Real const rp_old = std::sqrt(xp_old*xp_old + yp_old*yp_old);
            amrex::Real const rp_mid = (rp_new + rp_old)/2._rt;
            const amrex::Real costheta_mid = (rp_mid > 0._rt ? xp_mid/rp_mid : 1._rt);
            const amrex::Real sintheta_mid = (rp_mid > 0._rt ? yp_mid/rp_mid : 0._rt);
            const amrex::Real costheta_new = (rp_new > 0._rt ? xp_new/rp_new : 1._rt);
            const amrex::Real sintheta_new = (rp_new > 0._rt ? yp_new/rp_new : 0._rt);
            const amrex::Real costheta_old = (rp_old > 0._rt ? xp_old/rp_old : 1._rt);
            const amrex::Real sintheta_old = (rp_old > 0._rt ? yp_old/rp_old : 0._rt);
            const Complex xy_new0 = Complex{costheta_new, sintheta_new};
            const Complex xy_mid0 = Complex{costheta_mid, sintheta_mid};
            const Complex xy_old0 = Complex{costheta_old, sintheta_old};
            // Keep these double to avoid bug in single precision
            double const x_new = (rp_new - xyzmin.x)*dinv.x;
            double const x_old = (rp_old - xyzmin.x)*dinv.x;
#else
#if !defined(WARPX_DIM_1D_Z)
            // Keep these double to avoid bug in single precision
            double const x_new = (xp_np1 - xyzmin.x)*dinv.x;
            double const x_old = (xp_n[ip] - xyzmin.x)*dinv.x;
#endif
#endif
#if defined(WARPX_DIM_3D)
            // Keep these double to avoid bug in single precision
            double const y_new = (yp_np1 - xyzmin.y)*dinv.y;
            double const y_old = (yp_n[ip] - xyzmin.y)*dinv.y;
#endif
            // Keep these double to avoid bug in single precision
            double const z_new = (zp_np1 - xyzmin.z)*dinv.z;
            double const z_old = (zp_n[ip] - xyzmin.z)*dinv.z;

#if defined(WARPX_DIM_RZ)
            amrex::Real const vy = (-uxp_nph[ip]*sintheta_mid + uyp_nph[ip]*costheta_mid)*gaminv;
#elif defined(WARPX_DIM_XZ)
            amrex::Real const vy = uyp_nph[ip]*gaminv;
#elif defined(WARPX_DIM_1D_Z)
            amrex::Real const vx = uxp_nph[ip]*gaminv;
            amrex::Real const vy = uyp_nph[ip]*gaminv;
#endif

            // --- Compute shape factors
            // Compute shape factors for position as they are now and at old positions
            // [ijk]_new: leftmost grid point that the particle touches
            const Compute_shape_factor< depos_order > compute_shape_factor;
            const Compute_shifted_shape_factor< depos_order > compute_shifted_shape_factor;

            // Shape factor arrays
            // Note that there are extra values above and below
            // to possibly hold the factor for the old particle
            // which can be at a different grid location.
            // Keep these double to avoid bug in single precision
#if !defined(WARPX_DIM_1D_Z)
            double sx_new[depos_order + 3] = {0.};
            double sx_old[depos_order + 3] = {0.};
            const int i_new = compute_shape_factor(sx_new+1, x_new);
            const int i_old = compute_shifted_shape_factor(sx_old, x_old, i_new);
#endif
#if defined(WARPX_DIM_3D)
            double sy_new[depos_order + 3] = {0.};
            double sy_old[depos_order + 3] = {0.};
            const int j_new = compute_shape_factor(sy_new+1, y_new);
            const int j_old = compute_shifted_shape_factor(sy_old, y_old, j_new);
#endif
            double sz_new[depos_order + 3] = {0.};
            double sz_old[depos_order + 3] = {0.};
            const int k_new = compute_shape_factor(sz_new+1, z_new);
            const int k_old = compute_shifted_shape_factor(sz_old, z_old, k_new);

            // computes min/max positions of current contributions
#if !defined(WARPX_DIM_1D_Z)
            int dil = 1, diu = 1;
            if (i_old < i_new) { dil = 0; }
            if (i_old > i_new) { diu = 0; }
#endif
#if defined(WARPX_DIM_3D)
            int djl = 1, dju = 1;
            if (j_old < j_new) { djl = 0; }
            if (j_old > j_new) { dju = 0; }
#endif
            int dkl = 1, dku = 1;
            if (k_old < k_new) { dkl = 0; }
            if (k_old > k_new) { dku = 0; }

#if defined(WARPX_DIM_3D)

            for (int k=dkl; k<=depos_order+2-dku; k++) {
                for (int j=djl; j<=depos_order+2-dju; j++) {
                    amrex::Real sdxi = 0._rt;
                    for (int i=dil; i<=depos_order+1-diu; i++) {
                        sdxi += wq*invdtd.x*(sx_old[i] - sx_new[i])*(
                            one_third*(sy_new[j]*sz_new[k] + sy_old[j]*sz_old[k])
                           +one_sixth*(sy_new[j]*sz_old[k] + sy_old[j]*sz_new[k]));
                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdxi);
                    }
                }
            }
            for (int k=dkl; k<=depos_order+2-dku; k++) {
                for (int i=dil; i<=depos_order+2-diu; i++) {
                    amrex::Real sdyj = 0._rt;
                    for (int j=djl; j<=depos_order+1-dju; j++) {
                        sdyj += wq*invdtd.y*(sy_old[j] - sy_new[j])*(
                            one_third*(sx_new[i]*sz_new[k] + sx_old[i]*sz_old[k])
                           +one_sixth*(sx_new[i]*sz_old[k] + sx_old[i]*sz_new[k]));
                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdyj);
                    }
                }
            }
            for (int j=djl; j<=depos_order+2-dju; j++) {
                for (int i=dil; i<=depos_order+2-diu; i++) {
                    amrex::Real sdzk = 0._rt;
                    for (int k=dkl; k<=depos_order+1-dku; k++) {
                        sdzk += wq*invdtd.z*(sz_old[k] - sz_new[k])*(
                            one_third*(sx_new[i]*sy_new[j] + sx_old[i]*sy_old[j])
                           +one_sixth*(sx_new[i]*sy_old[j] + sx_old[i]*sy_new[j]));
                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdzk);
                    }
                }
            }

#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)

            for (int k=dkl; k<=depos_order+2-dku; k++) {
                amrex::Real sdxi = 0._rt;
                for (int i=dil; i<=depos_order+1-diu; i++) {
                    sdxi += wq*invdtd.x*(sx_old[i] - sx_new[i])*0.5_rt*(sz_new[k] + sz_old[k]);
                    amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdxi);
#if defined(WARPX_DIM_RZ)
                    Complex xy_mid = xy_mid0; // Throughout the following loop, xy_mid takes the value e^{i m theta}
                    for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                        // The factor 2 comes from the normalization of the modes
                        const Complex djr_cmplx = 2._rt *sdxi*xy_mid;
                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djr_cmplx.real());
                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djr_cmplx.imag());
                        xy_mid = xy_mid*xy_mid0;
                    }
#endif
                }
            }
            for (int k=dkl; k<=depos_order+2-dku; k++) {
                for (int i=dil; i<=depos_order+2-diu; i++) {
                    Real const sdyj = wq*vy*invvol*(
                        one_third*(sx_new[i]*sz_new[k] + sx_old[i]*sz_old[k])
                       +one_sixth*(sx_new[i]*sz_old[k] + sx_old[i]*sz_new[k]));
                    amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdyj);
#if defined(WARPX_DIM_RZ)
                    Complex const I = Complex{0._rt, 1._rt};
                    Complex xy_new = xy_new0;
                    Complex xy_mid = xy_mid0;
                    Complex xy_old = xy_old0;
                    // Throughout the following loop, xy_ takes the value e^{i m theta_}
                    for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                        // The factor 2 comes from the normalization of the modes
                        // The minus sign comes from the different convention with respect to Davidson et al.
                        const Complex djt_cmplx = -2._rt * I*(i_new-1 + i + xyzmin.x*dinv.x)*wq*invdtd.x/(amrex::Real)imode
                                                  *(Complex(sx_new[i]*sz_new[k], 0._rt)*(xy_new - xy_mid)
                                                  + Complex(sx_old[i]*sz_old[k], 0._rt)*(xy_mid - xy_old));
                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djt_cmplx.real());
                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djt_cmplx.imag());
                        xy_new = xy_new*xy_new0;
                        xy_mid = xy_mid*xy_mid0;
                        xy_old = xy_old*xy_old0;
                    }
#endif
                }
            }
            for (int i=dil; i<=depos_order+2-diu; i++) {
                Real sdzk = 0._rt;
                for (int k=dkl; k<=depos_order+1-dku; k++) {
                    sdzk += wq*invdtd.z*(sz_old[k] - sz_new[k])*0.5_rt*(sx_new[i] + sx_old[i]);
                    amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdzk);
#if defined(WARPX_DIM_RZ)
                    Complex xy_mid = xy_mid0; // Throughout the following loop, xy_mid takes the value e^{i m theta}
                    for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                        // The factor 2 comes from the normalization of the modes
                        const Complex djz_cmplx = 2._rt * sdzk * xy_mid;
                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djz_cmplx.real());
                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djz_cmplx.imag());
                        xy_mid = xy_mid*xy_mid0;
                    }
#endif
                }
            }
#elif defined(WARPX_DIM_1D_Z)

            for (int k=dkl; k<=depos_order+2-dku; k++) {
                amrex::Real const sdxi = wq*vx*invvol*0.5_rt*(sz_old[k] + sz_new[k]);
                amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+k_new-1+k, 0, 0, 0), sdxi);
            }
            for (int k=dkl; k<=depos_order+2-dku; k++) {
                amrex::Real const sdyj = wq*vy*invvol*0.5_rt*(sz_old[k] + sz_new[k]);
                amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+k_new-1+k, 0, 0, 0), sdyj);
            }
            amrex::Real sdzk = 0._rt;
            for (int k=dkl; k<=depos_order+1-dku; k++) {
                sdzk += wq*invdtd.z*(sz_old[k] - sz_new[k]);
                amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+k_new-1+k, 0, 0, 0), sdzk);
            }
#endif
        }
    );
}

/**
 * \brief Villasenor and Buneman Current Deposition for thread thread_num for implicit scheme.
 *        The specifics for the implicit scheme are in how gamma is determined. This is a charge-
 *        conserving deposition. The difference from Esirkepov is that the deposit is done segment
 *        by segment, where the segments are determined by cell crossings. In general, this results
 *        in a tighter stencil. The implementation is valid for an arbitrary number of cell crossings.
 *
 * \param depos_order  deposition order
 * \param xp_n,yp_n,zp_n  Pointer to arrays of particle position at time level n.
 * \param GetPosition  A functor for returning the particle position.
 * \param wp           Pointer to array of particle weights.
 * \param uxp_n,uyp_n,uzp_n  Pointer to arrays of particle momentum at time level n.
 * \param uxp_nph,uyp_nph,uzp_nph  Pointer to arrays of particle momentum at time level n + 1/2.
 * \param ion_lev      Pointer to array of particle ionization level. This is
                       required to have the charge of each macroparticle
                       since q is a scalar. For non-ionizable species,
                       ion_lev is a null pointer.
 * \param Jx_arr,Jy_arr,Jz_arr  Array4 of current density, either full array or tile.
 * \param np_to_deposit         Number of particles for which current is deposited.
 * \param dt                    Time step for particle level
 * \param dinv                  3D cell size inverse
 * \param xyzmin                Physical lower bounds of domain.
 * \param lo                    Index lower bounds of domain.
 * \param q                     species charge.
 * \param n_rz_azimuthal_modes  Number of azimuthal modes when using RZ geometry.
 */
template <int depos_order>
void doVillasenorDepositionShapeNImplicit ([[maybe_unused]]const amrex::ParticleReal * const xp_n,
                                           [[maybe_unused]]const amrex::ParticleReal * const yp_n,
                                           [[maybe_unused]]const amrex::ParticleReal * const zp_n,
                                           const GetParticlePosition<PIdx>& GetPosition,
                                           const amrex::ParticleReal * const wp,
                                           [[maybe_unused]]const amrex::ParticleReal * const uxp_n,
                                           [[maybe_unused]]const amrex::ParticleReal * const uyp_n,
                                           [[maybe_unused]]const amrex::ParticleReal * const uzp_n,
                                           [[maybe_unused]]const amrex::ParticleReal * const uxp_nph,
                                           [[maybe_unused]]const amrex::ParticleReal * const uyp_nph,
                                           [[maybe_unused]]const amrex::ParticleReal * const uzp_nph,
                                           const int * const ion_lev,
                                           const amrex::Array4<amrex::Real>& Jx_arr,
                                           const amrex::Array4<amrex::Real>& Jy_arr,
                                           const amrex::Array4<amrex::Real>& Jz_arr,
                                           const long np_to_deposit,
                                           const amrex::Real dt,
                                           const amrex::XDim3 & dinv,
                                           const amrex::XDim3 & xyzmin,
                                           const amrex::Dim3 lo,
                                           const amrex::Real q,
                                           [[maybe_unused]] const int n_rz_azimuthal_modes)
{
    using namespace amrex;

    // Whether ion_lev is a null pointer (do_ionization=0) or a real pointer
    // (do_ionization=1)
    bool const do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

#if (AMREX_SPACEDIM > 1)
    Real constexpr one_third = 1.0_rt / 3.0_rt;
    Real constexpr one_sixth = 1.0_rt / 6.0_rt;
#endif

    // Loop over particles and deposit into Jx_arr, Jy_arr and Jz_arr
    amrex::ParallelFor(
        np_to_deposit,
        [=] AMREX_GPU_DEVICE (long const ip) {

#if !defined(WARPX_DIM_3D)
            constexpr amrex::ParticleReal inv_c2 = 1._prt/(PhysConst::c*PhysConst::c);

            // Compute inverse Lorentz factor, the average of gamma at time levels n and n+1
            // The uxp,uyp,uzp are the velocities at time level n+1/2
            const amrex::ParticleReal uxp_np1 = 2._prt*uxp_nph[ip] - uxp_n[ip];
            const amrex::ParticleReal uyp_np1 = 2._prt*uyp_nph[ip] - uyp_n[ip];
            const amrex::ParticleReal uzp_np1 = 2._prt*uzp_nph[ip] - uzp_n[ip];
            const amrex::ParticleReal gamma_n = std::sqrt(1._prt + (uxp_n[ip]*uxp_n[ip] + uyp_n[ip]*uyp_n[ip] + uzp_n[ip]*uzp_n[ip])*inv_c2);
            const amrex::ParticleReal gamma_np1 = std::sqrt(1._prt + (uxp_np1*uxp_np1 + uyp_np1*uyp_np1 + uzp_np1*uzp_np1)*inv_c2);
            const amrex::ParticleReal gaminv = 2.0_prt/(gamma_n + gamma_np1);
#endif

            Real wq = q*wp[ip];
            if (do_ionization){
                wq *= ion_lev[ip];
            }

            ParticleReal xp_nph, yp_nph, zp_nph;
            GetPosition(ip, xp_nph, yp_nph, zp_nph);

#if !defined(WARPX_DIM_1D_Z)
            ParticleReal const xp_np1 = 2._prt*xp_nph - xp_n[ip];
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
            ParticleReal const yp_np1 = 2._prt*yp_nph - yp_n[ip];
#endif
            ParticleReal const zp_np1 = 2._prt*zp_nph - zp_n[ip];

            // computes current and old position in grid units
#if defined(WARPX_DIM_RZ)
            amrex::Real const xp_new = xp_np1;
            amrex::Real const yp_new = yp_np1;
            amrex::Real const xp_mid = xp_nph;
            amrex::Real const yp_mid = yp_nph;
            amrex::Real const xp_old = xp_n[ip];
            amrex::Real const yp_old = yp_n[ip];
            amrex::Real const rp_new = std::sqrt(xp_new*xp_new + yp_new*yp_new);
            amrex::Real const rp_old = std::sqrt(xp_old*xp_old + yp_old*yp_old);
            amrex::Real const rp_mid = (rp_new + rp_old)/2._rt;
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
            double const x_new = (rp_new - xyzmin.x)*dinv.x;
            double const x_old = (rp_old - xyzmin.x)*dinv.x;
            amrex::Real const vx = (rp_new - rp_old)/dt;
            amrex::Real const vy = (-uxp_nph[ip]*sintheta_mid + uyp_nph[ip]*costheta_mid)*gaminv;
#elif defined(WARPX_DIM_XZ)
            // Keep these double to avoid bug in single precision
            double const x_new = (xp_np1 - xyzmin.x)*dinv.x;
            double const x_old = (xp_n[ip] - xyzmin.x)*dinv.x;
            amrex::Real const vx = (xp_np1 - xp_n[ip])/dt;
            amrex::Real const vy = uyp_nph[ip]*gaminv;
#elif defined(WARPX_DIM_1D_Z)
            amrex::Real const vx = uxp_nph[ip]*gaminv;
            amrex::Real const vy = uyp_nph[ip]*gaminv;
#elif defined(WARPX_DIM_3D)
            // Keep these double to avoid bug in single precision
            double const x_new = (xp_np1 - xyzmin.x)*dinv.x;
            double const x_old = (xp_n[ip] - xyzmin.x)*dinv.x;
            double const y_new = (yp_np1 - xyzmin.y)*dinv.y;
            double const y_old = (yp_n[ip] - xyzmin.y)*dinv.y;
            amrex::Real const vx = (xp_np1 - xp_n[ip])/dt;
            amrex::Real const vy = (yp_np1 - yp_n[ip])/dt;
#endif

            // Keep these double to avoid bug in single precision
            double const z_new = (zp_np1 - xyzmin.z)*dinv.z;
            double const z_old = (zp_n[ip] - xyzmin.z)*dinv.z;
            amrex::Real const vz = (zp_np1 - zp_n[ip])/dt;

            // Define velocity kernals to deposit
            amrex::Real const wqx = wq*vx*invvol;
            amrex::Real const wqy = wq*vy*invvol;
            amrex::Real const wqz = wq*vz*invvol;

            // 1) Determine the number of segments.
            // 2) Loop over segments and deposit current.

            // cell crossings are defined at cell edges if depos_order is odd
            // cell crossings are defined at cell centers if depos_order is even

            int num_segments = 1;
            double shift = 0.0;
            if ( (depos_order % 2) == 0 ) { shift = 0.5; }

#if defined(WARPX_DIM_3D)

            // compute cell crossings in X-direction
            const auto i_old = static_cast<int>(x_old-shift);
            const auto i_new = static_cast<int>(x_new-shift);
            const int cell_crossings_x = std::abs(i_new-i_old);
            num_segments += cell_crossings_x;

            // compute cell crossings in Y-direction
            const auto j_old = static_cast<int>(y_old-shift);
            const auto j_new = static_cast<int>(y_new-shift);
            const int cell_crossings_y = std::abs(j_new-j_old);
            num_segments += cell_crossings_y;

            // compute cell crossings in Z-direction
            const auto k_old = static_cast<int>(z_old-shift);
            const auto k_new = static_cast<int>(z_new-shift);
            const int cell_crossings_z = std::abs(k_new-k_old);
            num_segments += cell_crossings_z;

            // need to assert that the number of cell crossings in each direction
            // is within the range permitted by the number of guard cells
            // e.g., if (num_segments > 7) ...

            // compute total change in particle position and the initial cell
            // locations in each direction used to find the position at cell crossings.
            const double dxp = x_new - x_old;
            const double dyp = y_new - y_old;
            const double dzp = z_new - z_old;
            const auto dirX_sign = static_cast<double>(dxp < 0. ? -1. : 1.);
            const auto dirY_sign = static_cast<double>(dyp < 0. ? -1. : 1.);
            const auto dirZ_sign = static_cast<double>(dzp < 0. ? -1. : 1.);
            double Xcell = 0., Ycell = 0., Zcell = 0.;
            if (num_segments > 1) {
                Xcell = static_cast<double>(i_old) + shift + 0.5*(1.-dirX_sign);
                Ycell = static_cast<double>(j_old) + shift + 0.5*(1.-dirY_sign);
                Zcell = static_cast<double>(k_old) + shift + 0.5*(1.-dirZ_sign);
            }

            // loop over the number of segments and deposit
            const Compute_shape_factor< depos_order-1 > compute_shape_factor_cell;
            const Compute_shape_factor_pair< depos_order > compute_shape_factors_node;
            double dxp_seg, dyp_seg, dzp_seg;
            double x0_new, y0_new, z0_new;
            double x0_old = x_old;
            double y0_old = y_old;
            double z0_old = z_old;

            for (int ns=0; ns<num_segments; ns++) {

                if (ns == num_segments-1) { // final segment

                    x0_new = x_new;
                    y0_new = y_new;
                    z0_new = z_new;
                    dxp_seg = x0_new - x0_old;
                    dyp_seg = y0_new - y0_old;
                    dzp_seg = z0_new - z0_old;

                }
                else {

                    x0_new = Xcell + dirX_sign;
                    y0_new = Ycell + dirY_sign;
                    z0_new = Zcell + dirZ_sign;
                    dxp_seg = x0_new - x0_old;
                    dyp_seg = y0_new - y0_old;
                    dzp_seg = z0_new - z0_old;

                    if ( (dyp == 0. || std::abs(dxp_seg) < std::abs(dxp/dyp*dyp_seg))
                      && (dzp == 0. || std::abs(dxp_seg) < std::abs(dxp/dzp*dzp_seg)) ) {
                        Xcell = x0_new;
                        dyp_seg = dyp/dxp*dxp_seg;
                        dzp_seg = dzp/dxp*dxp_seg;
                        y0_new = y0_old + dyp_seg;
                        z0_new = z0_old + dzp_seg;
                    }
                    else if (dzp == 0. || std::abs(dyp_seg) < std::abs(dyp/dzp*dzp_seg)) {
                        Ycell = y0_new;
                        dxp_seg = dxp/dyp*dyp_seg;
                        dzp_seg = dzp/dyp*dyp_seg;
                        x0_new = x0_old + dxp_seg;
                        z0_new = z0_old + dzp_seg;
                    }
                    else {
                        Zcell = z0_new;
                        dxp_seg = dxp/dzp*dzp_seg;
                        dyp_seg = dyp/dzp*dzp_seg;
                        x0_new = x0_old + dxp_seg;
                        y0_new = y0_old + dyp_seg;
                    }

                }

                // compute the segment factors (each equal to dt_seg/dt for nonzero dxp, dyp, or dzp)
                const auto seg_factor_x = static_cast<double>(dxp == 0. ? 1. : dxp_seg/dxp);
                const auto seg_factor_y = static_cast<double>(dyp == 0. ? 1. : dyp_seg/dyp);
                const auto seg_factor_z = static_cast<double>(dzp == 0. ? 1. : dzp_seg/dzp);

                // compute cell-based weights using the average segment position
                double sx_cell[depos_order] = {0.};
                double sy_cell[depos_order] = {0.};
                double sz_cell[depos_order] = {0.};
                double const x0_bar = (x0_new + x0_old)/2.0;
                double const y0_bar = (y0_new + y0_old)/2.0;
                double const z0_bar = (z0_new + z0_old)/2.0;
                const int i0_cell = compute_shape_factor_cell( sx_cell, x0_bar-0.5 );
                const int j0_cell = compute_shape_factor_cell( sy_cell, y0_bar-0.5 );
                const int k0_cell = compute_shape_factor_cell( sz_cell, z0_bar-0.5 );

                if constexpr (depos_order >= 3) { // higher-order correction to the cell-based weights
                    const Compute_shape_factor_pair<depos_order-1> compute_shape_factors_cell;
                    double sx_old_cell[depos_order] = {0.};
                    double sx_new_cell[depos_order] = {0.};
                    double sy_old_cell[depos_order] = {0.};
                    double sy_new_cell[depos_order] = {0.};
                    double sz_old_cell[depos_order] = {0.};
                    double sz_new_cell[depos_order] = {0.};
                    const int i0_cell_2 = compute_shape_factors_cell( sx_old_cell, sx_new_cell, x0_old-0.5, x0_new-0.5 );
                    const int j0_cell_2 = compute_shape_factors_cell( sy_old_cell, sy_new_cell, y0_old-0.5, y0_new-0.5 );
                    const int k0_cell_2 = compute_shape_factors_cell( sz_old_cell, sz_new_cell, z0_old-0.5, z0_new-0.5 );
                    ignore_unused(i0_cell_2, j0_cell_2, k0_cell_2);
                    for (int m=0; m<depos_order; m++) {
                        sx_cell[m] = (4.0*sx_cell[m] + sx_old_cell[m] + sx_new_cell[m])/6.0;
                        sy_cell[m] = (4.0*sy_cell[m] + sy_old_cell[m] + sy_new_cell[m])/6.0;
                        sz_cell[m] = (4.0*sz_cell[m] + sz_old_cell[m] + sz_new_cell[m])/6.0;
                    }
                }

                // compute node-based weights using the old and new segment positions
                double sx_old_node[depos_order+1] = {0.};
                double sx_new_node[depos_order+1] = {0.};
                double sy_old_node[depos_order+1] = {0.};
                double sy_new_node[depos_order+1] = {0.};
                double sz_old_node[depos_order+1] = {0.};
                double sz_new_node[depos_order+1] = {0.};
                const int i0_node = compute_shape_factors_node( sx_old_node, sx_new_node, x0_old, x0_new );
                const int j0_node = compute_shape_factors_node( sy_old_node, sy_new_node, y0_old, y0_new );
                const int k0_node = compute_shape_factors_node( sz_old_node, sz_new_node, z0_old, z0_new );

                // deposit Jx for this segment
                amrex::Real this_Jx;
                for (int i=0; i<=depos_order-1; i++) {
                    for (int j=0; j<=depos_order; j++) {
                        for (int k=0; k<=depos_order; k++) {
                            this_Jx = wqx*sx_cell[i]*( sy_old_node[j]*sz_old_node[k]*one_third
                                                     + sy_old_node[j]*sz_new_node[k]*one_sixth
                                                     + sy_new_node[j]*sz_old_node[k]*one_sixth
                                                     + sy_new_node[j]*sz_new_node[k]*one_third )*seg_factor_x;
                            amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i0_cell+i, lo.y+j0_node+j, lo.z+k0_node+k), this_Jx);
                        }
                    }
                }

                // deposit Jy for this segment
                amrex::Real this_Jy;
                for (int i=0; i<=depos_order; i++) {
                    for (int j=0; j<=depos_order-1; j++) {
                        for (int k=0; k<=depos_order; k++) {
                            this_Jy = wqy*sy_cell[j]*( sx_old_node[i]*sz_old_node[k]*one_third
                                                     + sx_old_node[i]*sz_new_node[k]*one_sixth
                                                     + sx_new_node[i]*sz_old_node[k]*one_sixth
                                                     + sx_new_node[i]*sz_new_node[k]*one_third )*seg_factor_y;
                            amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i0_node+i, lo.y+j0_cell+j, lo.z+k0_node+k), this_Jy);
                        }
                    }
                }

                // deposit Jz for this segment
                amrex::Real this_Jz;
                for (int i=0; i<=depos_order; i++) {
                    for (int j=0; j<=depos_order; j++) {
                        for (int k=0; k<=depos_order-1; k++) {
                            this_Jz = wqz*sz_cell[k]*( sx_old_node[i]*sy_old_node[j]*one_third
                                                     + sx_old_node[i]*sy_new_node[j]*one_sixth
                                                     + sx_new_node[i]*sy_old_node[j]*one_sixth
                                                     + sx_new_node[i]*sy_new_node[j]*one_third )*seg_factor_z;
                            amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i0_node+i, lo.y+j0_node+j, lo.z+k0_cell+k), this_Jz);
                        }
                    }
                }

                // update old segment values
                if (ns < num_segments-1) {
                    x0_old = x0_new;
                    y0_old = y0_new;
                    z0_old = z0_new;
                }

            } // end loop over segments

#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)

            // compute cell crossings in X-direction
            const auto i_old = static_cast<int>(x_old-shift);
            const auto i_new = static_cast<int>(x_new-shift);
            const int cell_crossings_x = std::abs(i_new-i_old);
            num_segments += cell_crossings_x;

            // compute cell crossings in Z-direction
            const auto k_old = static_cast<int>(z_old-shift);
            const auto k_new = static_cast<int>(z_new-shift);
            const int cell_crossings_z = std::abs(k_new-k_old);
            num_segments += cell_crossings_z;

            // need to assert that the number of cell crossings in each direction
            // is within the range permitted by the number of guard cells
            // e.g., if (num_segments > 5) ...

            // compute total change in particle position and the initial cell
            // locations in each direction used to find the position at cell crossings.
            const double dxp = x_new - x_old;
            const double dzp = z_new - z_old;
            const auto dirX_sign = static_cast<double>(dxp < 0. ? -1. : 1.);
            const auto dirZ_sign = static_cast<double>(dzp < 0. ? -1. : 1.);
            double Xcell = 0., Zcell = 0.;
            if (num_segments > 1) {
                Xcell = static_cast<double>(i_old) + shift + 0.5*(1.-dirX_sign);
                Zcell = static_cast<double>(k_old) + shift + 0.5*(1.-dirZ_sign);
            }

            // loop over the number of segments and deposit
            const Compute_shape_factor< depos_order-1 > compute_shape_factor_cell;
            const Compute_shape_factor_pair< depos_order > compute_shape_factors_node;
            double dxp_seg, dzp_seg;
            double x0_new, z0_new;
            double x0_old = x_old;
            double z0_old = z_old;

            for (int ns=0; ns<num_segments; ns++) {

                if (ns == num_segments-1) { // final segment

                    x0_new = x_new;
                    z0_new = z_new;
                    dxp_seg = x0_new - x0_old;
                    dzp_seg = z0_new - z0_old;

                }
                else {

                    x0_new = Xcell + dirX_sign;
                    z0_new = Zcell + dirZ_sign;
                    dxp_seg = x0_new - x0_old;
                    dzp_seg = z0_new - z0_old;

                    if (dzp == 0. || std::abs(dxp_seg) < std::abs(dxp/dzp*dzp_seg)) {
                        Xcell = x0_new;
                        dzp_seg = dzp/dxp*dxp_seg;
                        z0_new = z0_old + dzp_seg;
                    }
                    else {
                        Zcell = z0_new;
                        dxp_seg = dxp/dzp*dzp_seg;
                        x0_new = x0_old + dxp_seg;
                    }

                }

                // compute the segment factors (each equal to dt_seg/dt for nonzero dxp, or dzp)
                const auto seg_factor_x = static_cast<double>(dxp == 0. ? 1. : dxp_seg/dxp);
                const auto seg_factor_z = static_cast<double>(dzp == 0. ? 1. : dzp_seg/dzp);

                // compute cell-based weights using the average segment position
                double sx_cell[depos_order] = {0.};
                double sz_cell[depos_order] = {0.};
                double const x0_bar = (x0_new + x0_old)/2.0;
                double const z0_bar = (z0_new + z0_old)/2.0;
                const int i0_cell = compute_shape_factor_cell( sx_cell, x0_bar-0.5 );
                const int k0_cell = compute_shape_factor_cell( sz_cell, z0_bar-0.5 );

                if constexpr (depos_order >= 3) { // higher-order correction to the cell-based weights
                    const Compute_shape_factor_pair<depos_order-1> compute_shape_factors_cell;
                    double sx_old_cell[depos_order] = {0.};
                    double sx_new_cell[depos_order] = {0.};
                    double sz_old_cell[depos_order] = {0.};
                    double sz_new_cell[depos_order] = {0.};
                    const int i0_cell_2 = compute_shape_factors_cell( sx_old_cell, sx_new_cell, x0_old-0.5, x0_new-0.5 );
                    const int k0_cell_2 = compute_shape_factors_cell( sz_old_cell, sz_new_cell, z0_old-0.5, z0_new-0.5 );
                    ignore_unused(i0_cell_2, k0_cell_2);
                    for (int m=0; m<depos_order; m++) {
                        sx_cell[m] = (4.0*sx_cell[m] + sx_old_cell[m] + sx_new_cell[m])/6.0;
                        sz_cell[m] = (4.0*sz_cell[m] + sz_old_cell[m] + sz_new_cell[m])/6.0;
                    }
                }

                // compute node-based weights using the old and new segment positions
                double sx_old_node[depos_order+1] = {0.};
                double sx_new_node[depos_order+1] = {0.};
                double sz_old_node[depos_order+1] = {0.};
                double sz_new_node[depos_order+1] = {0.};
                const int i0_node = compute_shape_factors_node( sx_old_node, sx_new_node, x0_old, x0_new );
                const int k0_node = compute_shape_factors_node( sz_old_node, sz_new_node, z0_old, z0_new );

                // deposit Jx for this segment
                amrex::Real this_Jx;
                for (int i=0; i<=depos_order-1; i++) {
                    for (int k=0; k<=depos_order; k++) {
                        this_Jx = wqx*sx_cell[i]*(sz_old_node[k] + sz_new_node[k])/2.0_rt*seg_factor_x;
                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i0_cell+i, lo.y+k0_node+k, 0, 0), this_Jx);
#if defined(WARPX_DIM_RZ)
                        Complex xy_mid = xy_mid0; // Throughout the following loop, xy_mid takes the value e^{i m theta}
                        for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                            // The factor 2 comes from the normalization of the modes
                            const Complex djr_cmplx = 2._rt*this_Jx*xy_mid;
                            amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i0_cell+i, lo.y+k0_node+k, 0, 2*imode-1), djr_cmplx.real());
                            amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i0_cell+i, lo.y+k0_node+k, 0, 2*imode), djr_cmplx.imag());
                            xy_mid = xy_mid*xy_mid0;
                        }
#endif
                    }
                }

                // deposit out-of-plane Jy for this segment
                const auto seg_factor_y = std::min(seg_factor_x,seg_factor_z);
                amrex::Real this_Jy;
                for (int i=0; i<=depos_order; i++) {
                    for (int k=0; k<=depos_order; k++) {
                        this_Jy = wqy*( sx_old_node[i]*sz_old_node[k]*one_third
                                      + sx_old_node[i]*sz_new_node[k]*one_sixth
                                      + sx_new_node[i]*sz_old_node[k]*one_sixth
                                      + sx_new_node[i]*sz_new_node[k]*one_third )*seg_factor_y;
                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i0_node+i, lo.y+k0_node+k, 0, 0), this_Jy);
#if defined(WARPX_DIM_RZ)
                        Complex xy_mid = xy_mid0;
                        // Throughout the following loop, xy_ takes the value e^{i m theta_}
                        for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                            // The factor 2 comes from the normalization of the modes
                            const Complex djy_cmplx = 2._rt*this_Jy*xy_mid;
                            amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i0_node+i, lo.y+k0_node+k, 0, 2*imode-1), djy_cmplx.real());
                            amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i0_node+i, lo.y+k0_node+k, 0, 2*imode), djy_cmplx.imag());
                            xy_mid = xy_mid*xy_mid0;
                        }
#endif
                    }
                }

                // deposit Jz for this segment
                amrex::Real this_Jz;
                for (int i=0; i<=depos_order; i++) {
                    for (int k=0; k<=depos_order-1; k++) {
                        this_Jz = wqz*sz_cell[k]*(sx_old_node[i] + sx_new_node[i])/2.0_rt*seg_factor_z;
                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i0_node+i, lo.y+k0_cell+k, 0, 0), this_Jz);
#if defined(WARPX_DIM_RZ)
                        Complex xy_mid = xy_mid0; // Throughout the following loop, xy_mid takes the value e^{i m theta}
                        for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                            // The factor 2 comes from the normalization of the modes
                            const Complex djz_cmplx = 2._rt*this_Jz*xy_mid;
                            amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i0_node+i, lo.y+k0_cell+k, 0, 2*imode-1), djz_cmplx.real());
                            amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i0_node+i, lo.y+k0_cell+k, 0, 2*imode), djz_cmplx.imag());
                            xy_mid = xy_mid*xy_mid0;
                        }
#endif
                    }
                }

                // update old segment values
                if (ns < num_segments-1) {
                    x0_old = x0_new;
                    z0_old = z0_new;
                }

            } // end loop over segments

#elif defined(WARPX_DIM_1D_Z)

            // compute cell crossings in Z-direction
            const auto k_old = static_cast<int>(z_old-shift);
            const auto k_new = static_cast<int>(z_new-shift);
            const int cell_crossings_z = std::abs(k_new-k_old);
            num_segments += cell_crossings_z;

            // need to assert that the number of cell crossings in each direction
            // is within the range permitted by the number of guard cells
            // e.g., if (num_segments > 3) ...

            // compute dzp and the initial cell location used to find the cell crossings.
            double const dzp = z_new - z_old;
            const auto dirZ_sign = static_cast<double>(dzp < 0. ? -1. : 1.);
            double Zcell = static_cast<double>(k_old) + shift + 0.5*(1.-dirZ_sign);

            // loop over the number of segments and deposit
            const Compute_shape_factor< depos_order-1 > compute_shape_factor_cell;
            const Compute_shape_factor_pair< depos_order > compute_shape_factors_node;
            double dzp_seg;
            double z0_new;
            double z0_old = z_old;

            for (int ns=0; ns<num_segments; ns++) {

                if (ns == num_segments-1) { // final segment
                    z0_new = z_new;
                    dzp_seg = z0_new - z0_old;
                }
                else {
                    Zcell = Zcell + dirZ_sign;
                    z0_new = Zcell;
                    dzp_seg = z0_new - z0_old;
                }

                // compute the segment factor (equal to dt_seg/dt for nonzero dzp)
                const auto seg_factor = static_cast<double>(dzp == 0. ? 1. : dzp_seg/dzp);

                // compute cell-based weights using the average segment position
                double sz_cell[depos_order] = {0.};
                double const z0_bar = (z0_new + z0_old)/2.0;
                const int k0_cell = compute_shape_factor_cell( sz_cell, z0_bar-0.5 );

                if constexpr (depos_order >= 3) { // higher-order correction to the cell-based weights
                    const Compute_shape_factor_pair<depos_order-1> compute_shape_factors_cell;
                    double sz_old_cell[depos_order] = {0.};
                    double sz_new_cell[depos_order] = {0.};
                    const int k0_cell_2 = compute_shape_factors_cell( sz_old_cell, sz_new_cell, z0_old-0.5, z0_new-0.5 );
                    ignore_unused(k0_cell_2);
                    for (int m=0; m<depos_order; m++) {
                        sz_cell[m] = (4.0*sz_cell[m] + sz_old_cell[m] + sz_new_cell[m])/6.0;
                    }
                }

                // compute node-based weights using the old and new segment positions
                double sz_old_node[depos_order+1] = {0.};
                double sz_new_node[depos_order+1] = {0.};
                const int k0_node = compute_shape_factors_node( sz_old_node, sz_new_node, z0_old, z0_new );

                // deposit out-of-plane Jx and Jy for this segment
                for (int k=0; k<=depos_order; k++) {
                    const amrex::Real weight = 0.5_rt*(sz_old_node[k] + sz_new_node[k])*seg_factor;
                    amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+k0_node+k, 0, 0), wqx*weight);
                    amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+k0_node+k, 0, 0), wqy*weight);
                }

                // deposit Jz for this segment
                for (int k=0; k<=depos_order-1; k++) {
                    const amrex::Real this_Jz = wqz*sz_cell[k]*seg_factor;
                    amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+k0_cell+k, 0, 0), this_Jz);
                }

                // update old segment values
                if (ns < num_segments-1) {
                    z0_old = z0_new;
                }

            }

#endif
        }
    );
}

/**
 * \brief Vay current deposition
 * (<a href="https://doi.org/10.1016/j.jcp.2013.03.010"> Vay et al, 2013</a>)
 * for thread \c thread_num: deposit \c D in real space and store the result in
 * \c Dx_fab, \c Dy_fab, \c Dz_fab
 *
 * \tparam depos_order  deposition order
 * \param[in] GetPosition  Functor that returns the particle position
 * \param[in] wp           Pointer to array of particle weights
 * \param[in] uxp,uyp,uzp  Pointer to arrays of particle momentum along \c x
 * \param[in] ion_lev      Pointer to array of particle ionization level. This is
                           required to have the charge of each macroparticle since \c q
                           is a scalar. For non-ionizable species, \c ion_lev is \c null
 * \param[in,out] Dx_fab,Dy_fab,Dz_fab FArrayBox of Vay current density, either full array or tile
 * \param[in] np_to_deposit Number of particles for which current is deposited
 * \param[in] dt           Time step for particle level
 * \param[in] relative_time Time at which to deposit D, relative to the time of the
 *                          current positions of the particles. When different than 0,
 *                          the particle position will be temporarily modified to match
 *                          the time of the deposition.
 * \param[in] dinv         3D cell size inverse
 * \param[in] xyzmin       3D lower bounds of physical domain
 * \param[in] lo           Dimension-agnostic lower bounds of index domain
 * \param[in] q            Species charge
 * \param[in] n_rz_azimuthal_modes Number of azimuthal modes in RZ geometry
 */
template <int depos_order>
void doVayDepositionShapeN (const GetParticlePosition<PIdx>& GetPosition,
                            const amrex::ParticleReal* const wp,
                            const amrex::ParticleReal* const uxp,
                            const amrex::ParticleReal* const uyp,
                            const amrex::ParticleReal* const uzp,
                            const int* const ion_lev,
                            amrex::FArrayBox& Dx_fab,
                            amrex::FArrayBox& Dy_fab,
                            amrex::FArrayBox& Dz_fab,
                            long np_to_deposit,
                            amrex::Real dt,
                            amrex::Real relative_time,
                            const amrex::XDim3 & dinv,
                            const amrex::XDim3 & xyzmin,
                            amrex::Dim3 lo,
                            amrex::Real q,
                            [[maybe_unused]]int n_rz_azimuthal_modes)
{
    using namespace amrex::literals;

#if defined(WARPX_DIM_RZ)
    amrex::ignore_unused(GetPosition,
        wp, uxp, uyp, uzp, ion_lev, Dx_fab, Dy_fab, Dz_fab,
        np_to_deposit, dt, relative_time, dinv, xyzmin, lo, q);
    WARPX_ABORT_WITH_MESSAGE("Vay deposition not implemented in RZ geometry");
#endif

#if defined(WARPX_DIM_1D_Z)
    amrex::ignore_unused(GetPosition,
        wp, uxp, uyp, uzp, ion_lev, Dx_fab, Dy_fab, Dz_fab,
        np_to_deposit, dt, relative_time, dinv, xyzmin, lo, q);
    WARPX_ABORT_WITH_MESSAGE("Vay deposition not implemented in 1D geometry");
#endif

#if !(defined WARPX_DIM_RZ || defined WARPX_DIM_1D_Z)

    // If ion_lev is a null pointer, then do_ionization=0, else do_ionization=1
    const bool do_ionization = ion_lev;

    // Inverse of time step
    const amrex::Real invdt = 1._rt / dt;

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

    // Allocate temporary arrays
#if defined(WARPX_DIM_3D)
    AMREX_ALWAYS_ASSERT(Dx_fab.box() == Dy_fab.box() && Dx_fab.box() == Dz_fab.box());
    amrex::FArrayBox temp_fab{Dx_fab.box(), 4};
#elif defined(WARPX_DIM_XZ)
    AMREX_ALWAYS_ASSERT(Dx_fab.box() == Dz_fab.box());
    amrex::FArrayBox temp_fab{Dx_fab.box(), 2};
#endif
    temp_fab.setVal<amrex::RunOn::Device>(0._rt);
    amrex::Array4<amrex::Real> const& temp_arr = temp_fab.array();

    // Inverse of light speed squared
    const amrex::Real invcsq = 1._rt / (PhysConst::c * PhysConst::c);

    // Arrays where D will be stored
    amrex::Array4<amrex::Real> const& Dx_arr = Dx_fab.array();
    amrex::Array4<amrex::Real> const& Dy_arr = Dy_fab.array();
    amrex::Array4<amrex::Real> const& Dz_arr = Dz_fab.array();

    // Loop over particles and deposit (Dx,Dy,Dz) into Dx_fab, Dy_fab and Dz_fab
    amrex::ParallelFor(np_to_deposit, [=] AMREX_GPU_DEVICE (long ip)
    {
        // Inverse of Lorentz factor gamma
        const amrex::Real invgam = 1._rt / std::sqrt(1._rt + uxp[ip] * uxp[ip] * invcsq
                                                           + uyp[ip] * uyp[ip] * invcsq
                                                           + uzp[ip] * uzp[ip] * invcsq);
        // Product of particle charges and weights
        amrex::Real wq = q * wp[ip];
        if (do_ionization) { wq *= ion_lev[ip]; }

        // Current particle positions (in physical units)
        amrex::ParticleReal xp, yp, zp;
        GetPosition(ip, xp, yp, zp);

        // Particle velocities
        const amrex::Real vx = uxp[ip] * invgam;
        const amrex::Real vy = uyp[ip] * invgam;
        const amrex::Real vz = uzp[ip] * invgam;

        // Modify the particle position to match the time of the deposition
        xp += relative_time * vx;
        yp += relative_time * vy;
        zp += relative_time * vz;

        // Current and old particle positions in grid units
        // Keep these double to avoid bug in single precision.
        double const x_new = (xp - xyzmin.x + 0.5_rt*dt*vx) * dinv.x;
        double const x_old = (xp - xyzmin.x - 0.5_rt*dt*vx) * dinv.x;
#if defined(WARPX_DIM_3D)
        // Keep these double to avoid bug in single precision.
        double const y_new = (yp - xyzmin.y + 0.5_rt*dt*vy) * dinv.y;
        double const y_old = (yp - xyzmin.y - 0.5_rt*dt*vy) * dinv.y;
#endif
        // Keep these double to avoid bug in single precision.
        double const z_new = (zp - xyzmin.z + 0.5_rt*dt*vz) * dinv.z;
        double const z_old = (zp - xyzmin.z - 0.5_rt*dt*vz) * dinv.z;

        // Shape factor arrays for current and old positions (nodal)
        // Keep these double to avoid bug in single precision.
        double sx_new[depos_order+1] = {0.};
        double sx_old[depos_order+1] = {0.};
#if defined(WARPX_DIM_3D)
        // Keep these double to avoid bug in single precision.
        double sy_new[depos_order+1] = {0.};
        double sy_old[depos_order+1] = {0.};
#endif
        // Keep these double to avoid bug in single precision.
        double sz_new[depos_order+1] = {0.};
        double sz_old[depos_order+1] = {0.};

        // Compute shape factors for current positions

        // i_new leftmost grid point in x that the particle touches
        // sx_new shape factor along x for the centering of each current
        Compute_shape_factor< depos_order > const compute_shape_factor;
        const int i_new = compute_shape_factor(sx_new, x_new);
#if defined(WARPX_DIM_3D)
        // j_new leftmost grid point in y that the particle touches
        // sy_new shape factor along y for the centering of each current
        const int j_new = compute_shape_factor(sy_new, y_new);
#endif
        // k_new leftmost grid point in z that the particle touches
        // sz_new shape factor along z for the centering of each current
        const int k_new = compute_shape_factor(sz_new, z_new);

        // Compute shape factors for old positions

        // i_old leftmost grid point in x that the particle touches
        // sx_old shape factor along x for the centering of each current
        const int i_old = compute_shape_factor(sx_old, x_old);
#if defined(WARPX_DIM_3D)
        // j_old leftmost grid point in y that the particle touches
        // sy_old shape factor along y for the centering of each current
        const int j_old = compute_shape_factor(sy_old, y_old);
#endif
        // k_old leftmost grid point in z that the particle touches
        // sz_old shape factor along z for the centering of each current
        const int k_old = compute_shape_factor(sz_old, z_old);

        // Deposit current into Dx_arr, Dy_arr and Dz_arr
#if defined(WARPX_DIM_XZ)

        const amrex::Real wqy = wq * vy * invvol;
        for (int k=0; k<=depos_order; k++) {
            for (int i=0; i<=depos_order; i++) {

                // Re-casting sx_new and sz_new from double to amrex::Real so that
                // Atomic::Add has consistent types in its argument
                auto const sxn_szn = static_cast<amrex::Real>(sx_new[i] * sz_new[k]);
                auto const sxo_szn = static_cast<amrex::Real>(sx_old[i] * sz_new[k]);
                auto const sxn_szo = static_cast<amrex::Real>(sx_new[i] * sz_old[k]);
                auto const sxo_szo = static_cast<amrex::Real>(sx_old[i] * sz_old[k]);

                if (i_new == i_old && k_new == k_old) {
                    // temp arrays for Dx and Dz
                    amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + k_new + k, 0, 0),
                        wq * invvol * invdt * (sxn_szn - sxo_szo));

                    amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + k_new + k, 0, 1),
                        wq * invvol * invdt * (sxn_szo - sxo_szn));

                    // Dy
                    amrex::Gpu::Atomic::AddNoRet(&Dy_arr(lo.x + i_new + i, lo.y + k_new + k, 0, 0),
                        wqy * 0.25_rt * (sxn_szn + sxn_szo + sxo_szn + sxo_szo));
                } else {
                    // temp arrays for Dx and Dz
                    amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + k_new + k, 0, 0),
                        wq * invvol * invdt * sxn_szn);

                    amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_old + i, lo.y + k_old + k, 0, 0),
                        - wq * invvol * invdt * sxo_szo);

                    amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + k_old + k, 0, 1),
                        wq * invvol * invdt * sxn_szo);

                    amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_old + i, lo.y + k_new + k, 0, 1),
                        - wq * invvol * invdt * sxo_szn);

                    // Dy
                    amrex::Gpu::Atomic::AddNoRet(&Dy_arr(lo.x + i_new + i, lo.y + k_new + k, 0, 0),
                        wqy * 0.25_rt * sxn_szn);

                    amrex::Gpu::Atomic::AddNoRet(&Dy_arr(lo.x + i_new + i, lo.y + k_old + k, 0, 0),
                        wqy * 0.25_rt * sxn_szo);

                    amrex::Gpu::Atomic::AddNoRet(&Dy_arr(lo.x + i_old + i, lo.y + k_new + k, 0, 0),
                        wqy * 0.25_rt * sxo_szn);

                    amrex::Gpu::Atomic::AddNoRet(&Dy_arr(lo.x + i_old + i, lo.y + k_old + k, 0, 0),
                        wqy * 0.25_rt * sxo_szo);
                }

            }
        }

#elif defined(WARPX_DIM_3D)

        for (int k=0; k<=depos_order; k++) {
            for (int j=0; j<=depos_order; j++) {

                auto const syn_szn = static_cast<amrex::Real>(sy_new[j] * sz_new[k]);
                auto const syo_szn = static_cast<amrex::Real>(sy_old[j] * sz_new[k]);
                auto const syn_szo = static_cast<amrex::Real>(sy_new[j] * sz_old[k]);
                auto const syo_szo = static_cast<amrex::Real>(sy_old[j] * sz_old[k]);

                for (int i=0; i<=depos_order; i++) {

                    auto const sxn_syn_szn = static_cast<amrex::Real>(sx_new[i]) * syn_szn;
                    auto const sxo_syn_szn = static_cast<amrex::Real>(sx_old[i]) * syn_szn;
                    auto const sxn_syo_szn = static_cast<amrex::Real>(sx_new[i]) * syo_szn;
                    auto const sxo_syo_szn = static_cast<amrex::Real>(sx_old[i]) * syo_szn;
                    auto const sxn_syn_szo = static_cast<amrex::Real>(sx_new[i]) * syn_szo;
                    auto const sxo_syn_szo = static_cast<amrex::Real>(sx_old[i]) * syn_szo;
                    auto const sxn_syo_szo = static_cast<amrex::Real>(sx_new[i]) * syo_szo;
                    auto const sxo_syo_szo = static_cast<amrex::Real>(sx_old[i]) * syo_szo;

                    if (i_new == i_old && j_new == j_old && k_new == k_old) {
                        // temp arrays for Dx, Dy and Dz
                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_new + j, lo.z + k_new + k, 0),
                            wq * invvol * invdt * (sxn_syn_szn - sxo_syo_szo));

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_new + j, lo.z + k_new + k, 1),
                            wq * invvol * invdt * (sxn_syn_szo - sxo_syo_szn));

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_new + j, lo.z + k_new + k, 2),
                            wq * invvol * invdt * (sxn_syo_szn - sxo_syn_szo));

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_new + j, lo.z + k_new + k, 3),
                            wq * invvol * invdt * (sxo_syn_szn - sxn_syo_szo));
                    } else {
                        // temp arrays for Dx, Dy and Dz
                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_new + j, lo.z + k_new + k, 0),
                            wq * invvol * invdt * sxn_syn_szn);

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_old + i, lo.y + j_old + j, lo.z + k_old + k, 0),
                            - wq * invvol * invdt * sxo_syo_szo);

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_new + j, lo.z + k_old + k, 1),
                            wq * invvol * invdt * sxn_syn_szo);

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_old + i, lo.y + j_old + j, lo.z + k_new + k, 1),
                            - wq * invvol * invdt * sxo_syo_szn);

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_old + j, lo.z + k_new + k, 2),
                            wq * invvol * invdt * sxn_syo_szn);

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_old + i, lo.y + j_new + j, lo.z + k_old + k, 2),
                            - wq * invvol * invdt * sxo_syn_szo);

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_old + i, lo.y + j_new + j, lo.z + k_new + k, 3),
                            wq * invvol * invdt * sxo_syn_szn);

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_old + j, lo.z + k_old + k, 3),
                            - wq * invvol * invdt * sxn_syo_szo);
                    }
                }
            }
        }
#endif
    } );

#if defined(WARPX_DIM_3D)
    amrex::ParallelFor(Dx_fab.box(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        const amrex::Real t_a = temp_arr(i,j,k,0);
        const amrex::Real t_b = temp_arr(i,j,k,1);
        const amrex::Real t_c = temp_arr(i,j,k,2);
        const amrex::Real t_d = temp_arr(i,j,k,3);
        Dx_arr(i,j,k) += (1._rt/6._rt)*(2_rt*t_a       + t_b       + t_c - 2._rt*t_d);
        Dy_arr(i,j,k) += (1._rt/6._rt)*(2_rt*t_a       + t_b - 2._rt*t_c       + t_d);
        Dz_arr(i,j,k) += (1._rt/6._rt)*(2_rt*t_a - 2._rt*t_b       + t_c       + t_d);
    });
#elif defined(WARPX_DIM_XZ)
    amrex::ParallelFor(Dx_fab.box(), [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
    {
        const amrex::Real t_a = temp_arr(i,j,0,0);
        const amrex::Real t_b = temp_arr(i,j,0,1);
        Dx_arr(i,j,0) += (0.5_rt)*(t_a + t_b);
        Dz_arr(i,j,0) += (0.5_rt)*(t_a - t_b);
    });
#endif
    // Synchronize so that temp_fab can be safely deallocated in its destructor
    amrex::Gpu::streamSynchronize();

#endif // #if !(defined WARPX_DIM_RZ || defined WARPX_DIM_1D_Z)
}
#endif // WARPX_CURRENTDEPOSITION_H_
