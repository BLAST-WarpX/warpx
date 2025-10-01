/* Copyright 2019 Axel Huebl, David Grote, Maxence Thevenet
 * Remi Lehe, Weiqun Zhang, Michael Rowan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_MASS_MATRICES_DEPOSITION_H_
#define WARPX_MASS_MATRICES_DEPOSITION_H_

#include "Particles/Deposition/SharedDepositionUtils.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/Pusher/UpdatePosition.H"
#include "Particles/Gather/FieldGather.H"
#include "Particles/ShapeFactors.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#ifdef WARPX_DIM_RZ
#   include "Utils/WarpX_Complex.H"
#endif

#include <AMReX.H>
#include <AMReX_Arena.H>
#include <AMReX_Array4.H>
#include <AMReX_Dim3.H>
#include <AMReX_REAL.H>

/**
 * \brief Set the mass matrices kernels for thread thread_num
 * \param qs              Species charge
 * \param ms              Species mass
 * \param dt              Simulation time step
 * \param rhop            qs*wp*invvol*2.0/(gamma_n + gamma_np1)
 * \param uxp,uyp,uzp     Proper velocity of particle (time-centered)
 * \param Bxp,Byp,Bzp     Magnetic field vector components on particle
 * \param fpxx,fpxy,fpxz  Mass matrices kernels corresponding to dJx
 * \param fpyx,fpyy,fpyz  Mass matrices kernels corresponding to dJy
 * \param fpzx,fpzy,fpzz  Mass matrices kernels corresponding to dJz
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void setMassMatricesKernels ( const amrex::ParticleReal qs,
                              const amrex::ParticleReal ms,
                              const amrex::ParticleReal dt,
                              const amrex::ParticleReal rhop,
                              const amrex::ParticleReal uxp,
                              const amrex::ParticleReal uyp,
                              const amrex::ParticleReal uzp,
                              const amrex::ParticleReal Bxp,
                              const amrex::ParticleReal Byp,
                              const amrex::ParticleReal Bzp,
                              amrex::ParticleReal& fpxx,
                              amrex::ParticleReal& fpxy,
                              amrex::ParticleReal& fpxz,
                              amrex::ParticleReal& fpyx,
                              amrex::ParticleReal& fpyy,
                              amrex::ParticleReal& fpyz,
                              amrex::ParticleReal& fpzx,
                              amrex::ParticleReal& fpzy,
                              amrex::ParticleReal& fpzz )
{
    using namespace amrex::literals;

    constexpr amrex::ParticleReal inv_c2 = 1._prt/(PhysConst::c*PhysConst::c);

    // Convert B on particle to normalized cyclotron units with dt/2.0
    const amrex::ParticleReal gamma_bar = std::sqrt(1._prt + (uxp*uxp + uyp*uyp + uzp*uzp)*inv_c2);
    const amrex::ParticleReal alpha = qs/ms*0.5_prt*dt/gamma_bar;
    const amrex::ParticleReal bxp = alpha*Bxp;
    const amrex::ParticleReal byp = alpha*Byp;
    const amrex::ParticleReal bzp = alpha*Bzp;

    // Compute Mass Matrix kernels (non-relativistic for now)
    amrex::ParticleReal bpsq = bxp*bxp + byp*byp + bzp*bzp;
    amrex::ParticleReal arogp = alpha*rhop/(1.0_prt + bpsq);

    fpxx = arogp*(bxp*bxp + 1.0_rt);
    fpxy = arogp*(bxp*byp + bzp);
    fpxz = arogp*(bxp*bzp - byp);

    fpyx = arogp*(byp*bxp - bzp);
    fpyy = arogp*(byp*byp + 1.0_rt);
    fpyz = arogp*(byp*bzp + bxp);

    fpzx = arogp*(bzp*bxp + byp);
    fpzy = arogp*(bzp*byp - bxp);
    fpzz = arogp*(bzp*bzp + 1.0_rt);

}

/**
 * \brief Kernel for the direct deposition of J and S (mass matrices) for thread thread_num
 * \tparam depos_order deposition order
 * \param xp, yp, zp               The particle positions.
 * \param wqx,wqy,wqz              The particle velocity multiplied by the charge / volume
 * \param fpxx,fpxy,fpxz           Mass matrix kernels for Jx
 * \param fpyx,fpyy,fpyz           Mass matrix kernels for Jy
 * \param fpzx,fpzy,fpzz           Mass matrix kernels for Jz
 * \param jx_arr,jy_arr,jz_arr     Array4 of current density, either full array or tile.
 * \param Sxx_nComp                number of Sxx components
 * \param Syy_nComp                number of Syy components
 * \param Szz_nComp                number of Szz components
 * \param Sxx_arr,Sxy_arr,Sxz_arr  Array4 of mass matrices for Jx, either full array or tile.
 * \param Syx_arr,Syy_arr,Syz_arr  Array4 of mass matrices for Jy, either full array or tile.
 * \param Szx_arr,Szy_arr,Szz_arr  Array4 of mass matrices for Jz, either full array or tile.
 * \param jx_type,jy_type,jz_type  The j (and S) grid types along each direction, either NODE or CELL
 * \param dinv                     3D cell size inverse
 * \param xyzmin                   The lower bounds of the domain
 * \param lo                       Index lower bounds of domain.
 */
template <int depos_order>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void doDirectJandSigmaDepositionKernel ( [[maybe_unused]] const amrex::ParticleReal xp,
                                         [[maybe_unused]] const amrex::ParticleReal yp,
                                         [[maybe_unused]] const amrex::ParticleReal zp,
                                         const amrex::ParticleReal wqx,
                                         const amrex::ParticleReal wqy,
                                         const amrex::ParticleReal wqz,
                                         const amrex::ParticleReal fpxx,
                                         [[maybe_unused]] const amrex::ParticleReal fpxy,
                                         [[maybe_unused]] const amrex::ParticleReal fpxz,
                                         [[maybe_unused]] const amrex::ParticleReal fpyx,
                                         const amrex::ParticleReal fpyy,
                                         [[maybe_unused]] const amrex::ParticleReal fpyz,
                                         [[maybe_unused]] const amrex::ParticleReal fpzx,
                                         [[maybe_unused]] const amrex::ParticleReal fpzy,
                                         const amrex::ParticleReal fpzz,
                                         amrex::Array4<amrex::Real> const& jx_arr,
                                         amrex::Array4<amrex::Real> const& jy_arr,
                                         amrex::Array4<amrex::Real> const& jz_arr,
                                         [[maybe_unused]] int Sxx_nComp,
                                         [[maybe_unused]] int Syy_nComp,
                                         [[maybe_unused]] int Szz_nComp,
                                         amrex::Array4<amrex::Real> const& Sxx_arr,
                                         [[maybe_unused]] amrex::Array4<amrex::Real> const& Sxy_arr,
                                         [[maybe_unused]] amrex::Array4<amrex::Real> const& Sxz_arr,
                                         [[maybe_unused]] amrex::Array4<amrex::Real> const& Syx_arr,
                                         amrex::Array4<amrex::Real> const& Syy_arr,
                                         [[maybe_unused]] amrex::Array4<amrex::Real> const& Syz_arr,
                                         [[maybe_unused]] amrex::Array4<amrex::Real> const& Szx_arr,
                                         [[maybe_unused]] amrex::Array4<amrex::Real> const& Szy_arr,
                                         amrex::Array4<amrex::Real> const& Szz_arr,
                                         const amrex::IntVect& jx_type,
                                         const amrex::IntVect& jy_type,
                                         const amrex::IntVect& jz_type,
                                         const amrex::XDim3& dinv,
                                         const amrex::XDim3& xyzmin,
                                         const amrex::Dim3 lo )
{
    using namespace amrex::literals;

    constexpr int NODE = amrex::IndexType::NODE;
    constexpr int CELL = amrex::IndexType::CELL;

    // MassMatrices index shift parameter
    amrex::IntVect shift = amrex::IntVect::TheZeroVector();

    // --- Compute shape factors
    Compute_shape_factor< depos_order > const compute_shape_factor;
#if !defined(WARPX_DIM_1D_Z)
    // x direction
    // Get particle position after 1/2 push back in position
    // Keep these double to avoid bug in single precision
    const double xmid = (xp - xyzmin.x)*dinv.x;

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

    // Set the index shift parameter
    if (j_node==j_cell) { shift[0] = 1; }

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
#endif //!defined(WARPX_DIM_1D_Z)

#if defined(WARPX_DIM_3D)
    // y direction
    // Keep these double to avoid bug in single precision
    const double ymid = (yp - xyzmin.y)*dinv.y;
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

    // Set the index shift parameter
    if (k_node==k_cell) { shift[1] = 1; }

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

#if !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    // z direction
    // Keep these double to avoid bug in single precision
    constexpr int zdir = WARPX_ZINDEX;
    const double zmid = (zp - xyzmin.z)*dinv.z;
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

    // Set the index shift parameter
    if (l_node==l_cell) { shift[zdir] = 1; }

#endif

    // Compute index offset needed when x and y comps have different location on grid
    amrex::IntVect offset_xy, offset_xz, offset_yz;
    for (int dir=0; dir<AMREX_SPACEDIM; dir++) {
        offset_xy[dir] = (jx_type[dir] + jy_type[dir]) % 2;
        offset_xz[dir] = (jx_type[dir] + jz_type[dir]) % 2;
        offset_yz[dir] = (jy_type[dir] + jz_type[dir]) % 2;
    }

    // Deposit J and mass matrices
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
        for (int aa=0; aa<=depos_order; aa++){
            //  Deposit mass matrices for X-current
            if (Sxx_nComp==1 && aa==iz) {
                amrex::Gpu::Atomic::AddNoRet(
                    &Sxx_arr(lo.x+l_jx+iz, 0, 0, 0),
                    sz_jx[iz]*sz_jx[aa]*fpxx);
            }
            else if (Sxx_nComp>1) {
                int Nc = depos_order + aa - iz;
                amrex::Gpu::Atomic::AddNoRet(
                    &Sxx_arr(lo.x+l_jx+iz, 0, 0, Nc),
                    sz_jx[iz]*sz_jx[aa]*fpxx);
                Nc = depos_order + shift[0]*offset_xy[0] + aa - iz;
                amrex::Gpu::Atomic::AddNoRet(
                    &Sxy_arr(lo.x+l_jx+iz, 0, 0, Nc),
                    sz_jx[iz]*sz_jy[aa]*fpxy);
                Nc = depos_order + shift[0]*offset_xz[0] + aa - iz;
                amrex::Gpu::Atomic::AddNoRet(
                    &Sxz_arr(lo.x+l_jx+iz, 0, 0, Nc),
                    sz_jx[iz]*sz_jz[aa]*fpxz);
            }
            //  Deposit mass matrices for Y-current
            if (Syy_nComp==1 && aa==iz) {
                amrex::Gpu::Atomic::AddNoRet(
                    &Syy_arr(lo.x+l_jy+iz, 0, 0, 0),
                    sz_jy[iz]*sz_jy[aa]*fpyy);
            }
            else if (Syy_nComp>1) {
                int Nc = depos_order + shift[0]*offset_xy[0] + aa - iz;
                amrex::Gpu::Atomic::AddNoRet(
                    &Syx_arr(lo.x+l_jy+iz, 0, 0, Nc),
                    sz_jy[iz]*sz_jx[aa]*fpyx);
                Nc = depos_order + aa - iz;
                amrex::Gpu::Atomic::AddNoRet(
                    &Syy_arr(lo.x+l_jy+iz, 0, 0, Nc),
                    sz_jy[iz]*sz_jy[aa]*fpyy);
                Nc = depos_order + shift[0]*offset_yz[0] + aa - iz;
                amrex::Gpu::Atomic::AddNoRet(
                    &Syz_arr(lo.x+l_jy+iz, 0, 0, Nc),
                    sz_jy[iz]*sz_jz[aa]*fpyz);
            }
            //  Deposit mass matrices for Z-current
            if (Szz_nComp==1 && aa==iz) {
                amrex::Gpu::Atomic::AddNoRet(
                    &Szz_arr(lo.x+l_jz+iz, 0, 0, 0),
                    sz_jz[iz]*sz_jz[aa]*fpzz);
            }
            else if(Szz_nComp>1) {
                int Nc = depos_order + 1 - shift[0]*offset_xz[0] + aa - iz;
                amrex::Gpu::Atomic::AddNoRet(
                    &Szx_arr(lo.x+l_jz+iz, 0, 0, Nc),
                    sz_jz[iz]*sz_jx[aa]*fpzx);
                Nc = depos_order + 1 - shift[0]*offset_yz[0] + aa - iz;
                amrex::Gpu::Atomic::AddNoRet(
                    &Szy_arr(lo.x+l_jz+iz, 0, 0, Nc),
                    sz_jz[iz]*sz_jy[aa]*fpzy);
                Nc = depos_order + aa - iz;
                amrex::Gpu::Atomic::AddNoRet(
                    &Szz_arr(lo.x+l_jz+iz, 0, 0, Nc),
                    sz_jz[iz]*sz_jz[aa]*fpzz);
            }
        }
    }
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    for (int ix=0; ix<=depos_order; ix++){
        amrex::Gpu::Atomic::AddNoRet(
            &jx_arr(lo.x+j_jx+ix, 0, 0, 0),
            sx_jx[ix]*wqx);
        amrex::Gpu::Atomic::AddNoRet(
            &jy_arr(lo.x+j_jy+ix, 0, 0, 0),
            sx_jy[ix]*wqy);
        amrex::Gpu::Atomic::AddNoRet(
            &jz_arr(lo.x+j_jz+ix, 0, 0, 0),
            sx_jz[ix]*wqz);
        //
        amrex::Gpu::Atomic::AddNoRet(
            &Sxx_arr(lo.x+j_jx+ix, 0, 0, 0),
            sx_jx[ix]*sx_jx[ix]*fpxx);
        amrex::Gpu::Atomic::AddNoRet(
            &Syy_arr(lo.x+j_jy+ix, 0, 0, 0),
            sx_jy[ix]*sx_jy[ix]*fpyy);
        amrex::Gpu::Atomic::AddNoRet(
            &Szz_arr(lo.x+j_jz+ix, 0, 0, 0),
            sx_jz[ix]*sx_jz[ix]*fpzz);
    }
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    const int base_offset = 1 + 2*depos_order;
    for (int iz=0; iz<=depos_order; iz++){
        for (int ix=0; ix<=depos_order; ix++){
            const amrex::Real weight_Jx = sx_jx[ix]*sz_jx[iz];
            const amrex::Real weight_Jy = sx_jy[ix]*sz_jy[iz];
            const amrex::Real weight_Jz = sx_jz[ix]*sz_jz[iz];
            amrex::Gpu::Atomic::AddNoRet(
                &jx_arr(lo.x+j_jx+ix, lo.y+l_jx+iz, 0, 0),
                weight_Jx*wqx);
            amrex::Gpu::Atomic::AddNoRet(
                &jy_arr(lo.x+j_jy+ix, lo.y+l_jy+iz, 0, 0),
                weight_Jy*wqy);
            amrex::Gpu::Atomic::AddNoRet(
                &jz_arr(lo.x+j_jz+ix, lo.y+l_jz+iz, 0, 0),
                weight_Jz*wqz);
            for (int bb=0; bb<=depos_order; bb++){
                for (int aa=0; aa<=depos_order; aa++){
                    const amrex::Real weight_Ex = sx_jx[aa]*sz_jx[bb];
                    const amrex::Real weight_Ey = sx_jy[aa]*sz_jy[bb];
                    const amrex::Real weight_Ez = sx_jz[aa]*sz_jz[bb];
                    //  Deposit mass matrices for X-current
                    if (Sxx_nComp==1 && aa==ix && bb==iz) {
                        amrex::Gpu::Atomic::AddNoRet(
                            &Sxx_arr(lo.x+j_jx+ix, lo.y+l_jx+iz, 0, 0),
                            weight_Jx*weight_Ex*fpxx);
                    }
                    else if (Sxx_nComp>1) {
                        int offset = base_offset;
                        int Nc =  depos_order + aa - ix
                               + (depos_order + bb - iz)*offset;
                        amrex::Gpu::Atomic::AddNoRet(
                            &Sxx_arr(lo.x+j_jx+ix, lo.y+l_jx+iz, 0, Nc),
                            weight_Jx*weight_Ex*fpxx);
                        offset = base_offset + offset_xy[0];
                        Nc =  depos_order + 1 - shift[0]*offset_xy[0] + aa - ix
                           + (depos_order + shift[1]*offset_xy[1] + bb - iz)*offset;
                        amrex::Gpu::Atomic::AddNoRet(
                            &Sxy_arr(lo.x+j_jx+ix, lo.y+l_jx+iz, 0, Nc),
                            weight_Jx*weight_Ey*fpxy);
                        offset = base_offset + offset_xz[0];
                        Nc =  depos_order + 1 - shift[0]*offset_xz[0] + aa - ix
                           + (depos_order + shift[1]*offset_xz[1] + bb - iz)*offset;
                        amrex::Gpu::Atomic::AddNoRet(
                            &Sxz_arr(lo.x+j_jx+ix, lo.y+l_jx+iz, 0, Nc),
                            weight_Jx*weight_Ez*fpxz);
                    }
                    //  Deposit mass matrices for Y-current
                    if (Syy_nComp==1 && aa==ix && bb==iz) {
                        amrex::Gpu::Atomic::AddNoRet(
                            &Syy_arr(lo.x+j_jy+ix, lo.y+l_jy+iz, 0, 0),
                            weight_Jy*weight_Ey*fpyy);
                    }
                    else if (Syy_nComp>1) {
                        int offset = base_offset;
                        int Nc =  depos_order + aa - ix
                               + (depos_order + bb - iz)*offset;
                        amrex::Gpu::Atomic::AddNoRet(
                            &Syy_arr(lo.x+j_jy+ix, lo.y+l_jy+iz, 0, Nc),
                            weight_Jy*weight_Ey*fpyy);
                        offset = base_offset + offset_xy[0];
                        Nc =  depos_order + shift[0]*offset_xy[0] + aa - ix
                           + (depos_order + shift[1]*offset_xy[1] + bb - iz)*offset;
                        amrex::Gpu::Atomic::AddNoRet(
                            &Syx_arr(lo.x+j_jy+ix, lo.y+l_jy+iz, 0, Nc),
                            weight_Jy*weight_Ex*fpyx);
                        offset = base_offset + offset_yz[0];
                        Nc =  depos_order + shift[0]*offset_yz[0] + aa - ix
                           + (depos_order + shift[1]*offset_yz[1] + bb - iz)*offset;
                        amrex::Gpu::Atomic::AddNoRet(
                            &Syz_arr(lo.x+j_jy+ix, lo.y+l_jy+iz, 0, Nc),
                            weight_Jy*weight_Ez*fpyz);
                    }
                    //  Deposit mass matrices for Z-current
                    if (Szz_nComp==1 && aa==ix && bb==iz) {
                        amrex::Gpu::Atomic::AddNoRet(
                            &Szz_arr(lo.x+j_jz+ix, lo.y+l_jz+iz, 0, 0),
                            weight_Jz*weight_Ez*fpzz);
                    }
                    else if (Szz_nComp>1) {
                        int offset = base_offset;
                        int Nc =  depos_order + aa - ix
                               + (depos_order + bb - iz)*offset;
                        amrex::Gpu::Atomic::AddNoRet(
                            &Szz_arr(lo.x+j_jz+ix, lo.y+l_jz+iz, 0, Nc),
                            weight_Jz*weight_Ez*fpzz);
                        offset = base_offset + offset_xz[0];
                        Nc =  depos_order + shift[0]*offset_xz[0] + aa - ix
                           + (depos_order + 1 - shift[1]*offset_xz[1] + bb - iz)*offset;
                        amrex::Gpu::Atomic::AddNoRet(
                            &Szx_arr(lo.x+j_jz+ix, lo.y+l_jz+iz, 0, Nc),
                            weight_Jz*weight_Ex*fpzx);
                        offset = base_offset + offset_yz[0];
                        Nc =  depos_order + shift[0]*offset_yz[0] + aa - ix
                           + (depos_order + 1 - shift[1]*offset_yz[1] + bb - iz)*offset;
                        amrex::Gpu::Atomic::AddNoRet(
                            &Szy_arr(lo.x+j_jz+ix, lo.y+l_jz+iz, 0, Nc),
                            weight_Jz*weight_Ey*fpzy);
                    }
                }
            }
        }
    }
#elif defined(WARPX_DIM_3D)
    for (int iz=0; iz<=depos_order; iz++){
        for (int iy=0; iy<=depos_order; iy++){
            for (int ix=0; ix<=depos_order; ix++){
                const amrex::Real weight_Jx = sx_jx[ix]*sy_jx[iy]*sz_jx[iz];
                const amrex::Real weight_Jy = sx_jy[ix]*sy_jy[iy]*sz_jy[iz];
                const amrex::Real weight_Jz = sx_jz[ix]*sy_jz[iy]*sz_jz[iz];
                amrex::Gpu::Atomic::AddNoRet(
                    &jx_arr(lo.x+j_jx+ix, lo.y+k_jx+iy, lo.z+l_jx+iz),
                    weight_Jx*wqx);
                amrex::Gpu::Atomic::AddNoRet(
                    &jy_arr(lo.x+j_jy+ix, lo.y+k_jy+iy, lo.z+l_jy+iz),
                    weight_Jy*wqy);
                amrex::Gpu::Atomic::AddNoRet(
                    &jz_arr(lo.x+j_jz+ix, lo.y+k_jz+iy, lo.z+l_jz+iz),
                    weight_Jz*wqz);
                //
                amrex::Gpu::Atomic::AddNoRet(
                    &Sxx_arr(lo.x+j_jx+ix, lo.y+k_jx+iy, lo.z+l_jx+iz, 0),
                    weight_Jx*weight_Jx*fpxx);
                amrex::Gpu::Atomic::AddNoRet(
                    &Syy_arr(lo.x+j_jy+ix, lo.y+k_jy+iy, lo.z+l_jy+iz, 0),
                    weight_Jy*weight_Jy*fpyy);
                amrex::Gpu::Atomic::AddNoRet(
                    &Szz_arr(lo.x+j_jz+ix, lo.y+k_jz+iy, lo.z+l_jz+iz, 0),
                    weight_Jz*weight_Jz*fpzz);
            }
        }
    }
#endif
}

/**
 * \brief direct deposition of J and mass matrices for thread thread_num
 * \tparam depos_order deposition order
 * \param GetPosition              A functor for returning the particle position.
 * \param wp                       Pointer to array of particle weights.
 * \param uxp_n,uyp_n,uzp_n        Pointer to arrays of particle momentum at time n.
 * \param uxp_nph,uyp_nph,uzp_nph  Pointer to arrays of particle momentum at time n+1/2.
 * \param jx_fab,jy_fab,jz_fab     FArrayBox of current density, either full array or tile.
 * \param Sxx_nComp                number of Sxx components
 * \param Syy_nComp                number of Syy components
 * \param Szz_nComp                number of Szz components
 * \param Sxx_arr,Sxy_arr,Sxz_arr  Array4 of mass matrices for Jx, either full array or tile.
 * \param Syx_arr,Syy_arr,Syz_arr  Array4 of mass matrices for Jy, either full array or tile.
 * \param Szx_arr,Szy_arr,Szz_arr  Array4 of mass matrices for Jz, either full array or tile.
 * \param Bx_arr,By_arr,Bz_arr     Array4 of the magnetic field, either full array or tile.
 * \param Bx_type,By_type,Bz_type  IndexType of the magnetic field
 * \param np_to_deposit            Number of particles for which current is deposited.
 * \param dt                       Time step for particle.
 * \param dinv                     3D cell size inverse.
 * \param xyzmin                   Physical lower bounds of domain.
 * \param lo                       Index lower bounds of domain.
 * \param qs                       Species charge.
 * \param ms                       Species mass.
 */
template <int depos_order>
void doDirectJandSigmaDeposition ( const GetParticlePosition<PIdx>& GetPosition,
                                   const amrex::ParticleReal* wp,
                                   const amrex::ParticleReal* uxp_n,
                                   const amrex::ParticleReal* uyp_n,
                                   const amrex::ParticleReal* uzp_n,
                                   const amrex::ParticleReal* uxp_nph,
                                   const amrex::ParticleReal* uyp_nph,
                                   const amrex::ParticleReal* uzp_nph,
                                   amrex::FArrayBox& jx_fab,
                                   amrex::FArrayBox& jy_fab,
                                   amrex::FArrayBox& jz_fab,
                                   int Sxx_nComp,
                                   int Syy_nComp,
                                   int Szz_nComp,
                                   amrex::Array4<amrex::Real> const& Sxx_arr,
                                   amrex::Array4<amrex::Real> const& Sxy_arr,
                                   amrex::Array4<amrex::Real> const& Sxz_arr,
                                   amrex::Array4<amrex::Real> const& Syx_arr,
                                   amrex::Array4<amrex::Real> const& Syy_arr,
                                   amrex::Array4<amrex::Real> const& Syz_arr,
                                   amrex::Array4<amrex::Real> const& Szx_arr,
                                   amrex::Array4<amrex::Real> const& Szy_arr,
                                   amrex::Array4<amrex::Real> const& Szz_arr,
                                   const amrex::Array4<amrex::Real const>& Bx_arr,
                                   const amrex::Array4<amrex::Real const>& By_arr,
                                   const amrex::Array4<amrex::Real const>& Bz_arr,
                                   const amrex::IndexType Bx_type,
                                   const amrex::IndexType By_type,
                                   const amrex::IndexType Bz_type,
                                   const long np_to_deposit,
                                   const amrex::Real dt,
                                   const amrex::XDim3& dinv,
                                   const amrex::XDim3& xyzmin,
                                   const amrex::Dim3 lo,
                                   const amrex::Real qs,
                                   const amrex::Real ms )
{
    using namespace amrex::literals;

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
            amrex::ParticleReal xp_nph, yp_nph, zp_nph;
            GetPosition(ip, xp_nph, yp_nph, zp_nph);

            // Compute magnetic field on particle
            amrex::ParticleReal Bxp = 0.0;
            amrex::ParticleReal Byp = 0.0;
            amrex::ParticleReal Bzp = 0.0;
            const int depos_order_perp = 1;
            const int depos_order_para = 1;
            const int n_rz_azimuthal_modes = 0;
            doDirectGatherVectorField<depos_order_perp,depos_order_para>(
                                    xp_nph, yp_nph, zp_nph,
                                    Bxp, Byp, Bzp,
                                    Bx_arr, By_arr, Bz_arr,
                                    Bx_type, By_type, Bz_type,
                                    dinv, xyzmin, lo, n_rz_azimuthal_modes );

            // Compute inverse Lorentz factor, the average of gamma at time levels n and n+1
            const amrex::ParticleReal gaminv = GetImplicitGammaInverse(uxp_n[ip], uyp_n[ip], uzp_n[ip],
                                                                       uxp_nph[ip], uyp_nph[ip], uzp_nph[ip]);

            // Compute current density kernels to deposit
            const amrex::Real rhop = qs*wp[ip]*invvol*gaminv;
            amrex::Real wqx  = rhop*uxp_nph[ip];
            amrex::Real wqy  = rhop*uyp_nph[ip];
            amrex::Real wqz  = rhop*uzp_nph[ip];

            // Set the Mass Matrices kernels
            amrex::ParticleReal fpxx, fpxy, fpxz;
            amrex::ParticleReal fpyx, fpyy, fpyz;
            amrex::ParticleReal fpzx, fpzy, fpzz;
            setMassMatricesKernels( qs, ms, dt, rhop,
                                    uxp_nph[ip], uyp_nph[ip], uzp_nph[ip],
                                    Bxp, Byp, Bzp,
                                    fpxx, fpxy, fpxz,
                                    fpyx, fpyy, fpyz,
                                    fpzx, fpzy, fpzz );

            doDirectJandSigmaDepositionKernel<depos_order>( xp_nph, yp_nph, zp_nph,
                                                            wqx, wqy, wqz,
                                                            fpxx, fpxy, fpxz,
                                                            fpyx, fpyy, fpyz,
                                                            fpzx, fpzy, fpzz,
                                                            jx_arr, jy_arr, jz_arr,
                                                            Sxx_nComp, Syy_nComp, Szz_nComp,
                                                            Sxx_arr, Sxy_arr, Sxz_arr,
                                                            Syx_arr, Syy_arr, Syz_arr,
                                                            Szx_arr, Szy_arr, Szz_arr,
                                                            jx_type, jy_type, jz_type,
                                                            dinv, xyzmin, lo );

        }
    );
}

/**
 * \brief Kernel for the Villasenor deposition of J and S (mass matrices) for thread thread_num
 * \tparam depos_order             Deposition order
 * \tparam full_mass_matrices      Whether to deposit the full mass matrices
 *                                 (can only be true for depos_order = 2)
 * \param xp_old,yp_old,zp_old     Old particle positions (nominally at start of step)
 * \param xp_new,yp_new,zp_new     New particle positions (nominally at end of step)
 * \param wq_invvol                Particle charge / volume
 * \param uxp_mid,uyp_mid,uzp_mid  Particle momentum at middle of step
 * \param gaminv                   One over gamma for particle at middle of step
 * \param fpxx,fpxy,fpxz           Mass matrix kernels for Jx
 * \param fpyx,fpyy,fpyz           Mass matrix kernels for Jy
 * \param fpzx,fpzy,fpzz           Mass matrix kernels for Jz
 * \param Jx_arr,Jy_arr,Jz_arr     Array4 of current density, either full array or tile
 * \param max_crossings            Maximum number of particle cell crossings in each dir
 * \param Sxx_arr,Sxy_arr,Sxz_arr  Array4 of mass matrices for Jx, either full array or tile
 * \param Syx_arr,Syy_arr,Syz_arr  Array4 of mass matrices for Jy, either full array or tile
 * \param Szx_arr,Szy_arr,Szz_arr  Array4 of mass matrices for Jz, either full array or tile
 * \param dt                       Time step for particle level
 * \param dinv                     3D cell size inverse
 * \param xyzmin                   Physical lower bounds of domain
 * \param lo                       Index lower bounds of domain
 */
template <int depos_order, bool full_mass_matrices>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void doVillasenorJandSigmaDepositionKernel ( [[maybe_unused]] const amrex::ParticleReal xp_old,
                                             [[maybe_unused]] const amrex::ParticleReal yp_old,
                                             [[maybe_unused]] const amrex::ParticleReal zp_old,
                                             [[maybe_unused]] const amrex::ParticleReal xp_new,
                                             [[maybe_unused]] const amrex::ParticleReal yp_new,
                                             [[maybe_unused]] const amrex::ParticleReal zp_new,
                                             const amrex::ParticleReal wq_invvol,
                                             [[maybe_unused]] const amrex::ParticleReal uxp_mid,
                                             [[maybe_unused]] const amrex::ParticleReal uyp_mid,
                                             [[maybe_unused]] const amrex::ParticleReal uzp_mid,
                                             [[maybe_unused]] const amrex::ParticleReal gaminv,
                                             const amrex::ParticleReal fpxx,
                                             [[maybe_unused]] const amrex::ParticleReal fpxy,
                                             [[maybe_unused]] const amrex::ParticleReal fpxz,
                                             [[maybe_unused]] const amrex::ParticleReal fpyx,
                                             const amrex::ParticleReal fpyy,
                                             [[maybe_unused]] const amrex::ParticleReal fpyz,
                                             [[maybe_unused]] const amrex::ParticleReal fpzx,
                                             [[maybe_unused]] const amrex::ParticleReal fpzy,
                                             const amrex::ParticleReal fpzz,
                                             amrex::Array4<amrex::Real> const& Jx_arr,
                                             amrex::Array4<amrex::Real> const& Jy_arr,
                                             amrex::Array4<amrex::Real> const& Jz_arr,
                                             [[maybe_unused]] int max_crossings,
                                             amrex::Array4<amrex::Real> const& Sxx_arr,
                                             [[maybe_unused]] amrex::Array4<amrex::Real> const& Sxy_arr,
                                             [[maybe_unused]] amrex::Array4<amrex::Real> const& Sxz_arr,
                                             [[maybe_unused]] amrex::Array4<amrex::Real> const& Syx_arr,
                                             amrex::Array4<amrex::Real> const& Syy_arr,
                                             [[maybe_unused]] amrex::Array4<amrex::Real> const& Syz_arr,
                                             [[maybe_unused]] amrex::Array4<amrex::Real> const& Szx_arr,
                                             [[maybe_unused]] amrex::Array4<amrex::Real> const& Szy_arr,
                                             amrex::Array4<amrex::Real> const& Szz_arr,
                                             const amrex::Real dt,
                                             const amrex::XDim3& dinv,
                                             const amrex::XDim3& xyzmin,
                                             const amrex::Dim3 lo )
{

    using namespace amrex::literals;

#if (AMREX_SPACEDIM > 1)
    amrex::Real constexpr one_third = 1.0_rt / 3.0_rt;
    amrex::Real constexpr one_sixth = 1.0_rt / 6.0_rt;
#endif

    // computes current and old position in grid units
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
    amrex::Real const xp_mid = (xp_new + xp_old)*0.5_rt;
    amrex::Real const yp_mid = (yp_new + yp_old)*0.5_rt;
    amrex::Real const rp_new = std::sqrt(xp_new*xp_new + yp_new*yp_new);
    amrex::Real const rp_old = std::sqrt(xp_old*xp_old + yp_old*yp_old);
    amrex::Real const rp_mid = (rp_new + rp_old)/2._rt;
    amrex::Real const costheta_mid = (rp_mid > 0._rt ? xp_mid/rp_mid : 1._rt);
    amrex::Real const sintheta_mid = (rp_mid > 0._rt ? yp_mid/rp_mid : 0._rt);

    // Keep these double to avoid bug in single precision
    double const x_new = (rp_new - xyzmin.x)*dinv.x;
    double const x_old = (rp_old - xyzmin.x)*dinv.x;
    amrex::Real const vx = (rp_new - rp_old)/dt;
    amrex::Real const vy = (-uxp_mid*sintheta_mid + uyp_mid*costheta_mid)*gaminv;
#if defined(WARPX_DIM_RCYLINDER)
    amrex::Real const vz = uzp_mid*gaminv;
#endif
#elif defined(WARPX_DIM_RSPHERE)
    amrex::Real const xp_mid = (xp_new + xp_old)*0.5_rt;
    amrex::Real const yp_mid = (yp_new + yp_old)*0.5_rt;
    amrex::Real const zp_mid = (zp_new + zp_old)*0.5_rt;
    amrex::Real const rpxy_new = std::sqrt(xp_new*xp_new + yp_new*yp_new);
    amrex::Real const rp_new = std::sqrt(xp_new*xp_new + yp_new*yp_new + zp_new*zp_new);
    amrex::Real const rpxy_old = std::sqrt(xp_old*xp_old + yp_old*yp_old);
    amrex::Real const rp_old = std::sqrt(xp_old*xp_old + yp_old*yp_old + zp_old*zp_old);
    amrex::Real const rpxy_mid = (rpxy_new + rpxy_old)*0.5_rt;
    amrex::Real const rp_mid = (rp_new + rp_old)*0.5_rt;
    amrex::Real const costheta_mid = (rpxy_mid > 0._rt ? xp_mid/rpxy_mid : 1._rt);
    amrex::Real const sintheta_mid = (rpxy_mid > 0._rt ? yp_mid/rpxy_mid : 0._rt);
    amrex::Real const cosphi_mid = (rp_mid > 0._rt ? rpxy_mid/rp_mid : 1._rt);
    amrex::Real const sinphi_mid = (rp_mid > 0._rt ? zp_mid/rp_mid : 0._rt);

    // Keep these double to avoid bug in single precision
    double const x_new = (rp_new - xyzmin.x)*dinv.x;
    double const x_old = (rp_old - xyzmin.x)*dinv.x;
    amrex::Real const vx = (rp_new - rp_old)/dt;
    amrex::Real const vy = (-uxp_mid*sintheta_mid + uyp_mid*costheta_mid)*gaminv;
    amrex::Real const vz = (-uxp_mid*costheta_mid*cosphi_mid - uyp_mid*sintheta_mid*cosphi_mid + uzp_mid*sinphi_mid)*gaminv;
#elif defined(WARPX_DIM_XZ)
    // Keep these double to avoid bug in single precision
    double const x_new = (xp_new - xyzmin.x)*dinv.x;
    double const x_old = (xp_old - xyzmin.x)*dinv.x;
    amrex::Real const vx = (xp_new - xp_old)/dt;
    amrex::Real const vy = uyp_mid*gaminv;
#elif defined(WARPX_DIM_1D_Z)
    amrex::Real const vx = uxp_mid*gaminv;
    amrex::Real const vy = uyp_mid*gaminv;
#elif defined(WARPX_DIM_3D)
    // Keep these double to avoid bug in single precision
    double const x_new = (xp_new - xyzmin.x)*dinv.x;
    double const x_old = (xp_old - xyzmin.x)*dinv.x;
    double const y_new = (yp_new - xyzmin.y)*dinv.y;
    double const y_old = (yp_old - xyzmin.y)*dinv.y;
    amrex::Real const vx = (xp_new - xp_old)/dt;
    amrex::Real const vy = (yp_new - yp_old)/dt;
#endif

#if !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    // Keep these double to avoid bug in single precision
    double const z_new = (zp_new - xyzmin.z)*dinv.z;
    double const z_old = (zp_old - xyzmin.z)*dinv.z;
    amrex::Real const vz = (zp_new - zp_old)/dt;
#endif

    // Define velocity kernels to deposit
    amrex::Real const wqx = wq_invvol*vx;
    amrex::Real const wqy = wq_invvol*vy;
    amrex::Real const wqz = wq_invvol*vz;

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

    // Compute total change in particle position and the initial cell
    // locations in each direction used to find the position at cell crossings.
    // Keep these double to avoid bug in single precision
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

        // Compute the segment factors (each equal to dt_seg/dt for nonzero dxp, dyp, or dzp)
        const auto seg_factor_x = static_cast<amrex::Real>(dxp == 0. ? 1._rt : dxp_seg/dxp);
        const auto seg_factor_y = static_cast<amrex::Real>(dyp == 0. ? 1._rt : dyp_seg/dyp);
        const auto seg_factor_z = static_cast<amrex::Real>(dzp == 0. ? 1._rt : dzp_seg/dzp);

        // Compute cell-based weights using the average segment position
        // Keep these double to avoid bug in single precision
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
            amrex::ignore_unused(i0_cell_2, j0_cell_2, k0_cell_2);
            for (int m=0; m<depos_order; m++) {
                sx_cell[m] = (4.0*sx_cell[m] + sx_old_cell[m] + sx_new_cell[m])/6.0;
                sy_cell[m] = (4.0*sy_cell[m] + sy_old_cell[m] + sy_new_cell[m])/6.0;
                sz_cell[m] = (4.0*sz_cell[m] + sz_old_cell[m] + sz_new_cell[m])/6.0;
            }
        }

        // Compute node-based weights using the old and new segment positions
        // Keep these double to avoid bug in single precision
        double sx_old_node[depos_order+1] = {0.};
        double sx_new_node[depos_order+1] = {0.};
        double sy_old_node[depos_order+1] = {0.};
        double sy_new_node[depos_order+1] = {0.};
        double sz_old_node[depos_order+1] = {0.};
        double sz_new_node[depos_order+1] = {0.};
        const int i0_node = compute_shape_factors_node( sx_old_node, sx_new_node, x0_old, x0_new );
        const int j0_node = compute_shape_factors_node( sy_old_node, sy_new_node, y0_old, y0_new );
        const int k0_node = compute_shape_factors_node( sz_old_node, sz_new_node, z0_old, z0_new );

        // deposit Jx and Sxx for this segment
        amrex::Real weight;
        for (int i=0; i<=depos_order-1; i++) {
            for (int j=0; j<=depos_order; j++) {
                for (int k=0; k<=depos_order; k++) {
                    weight = sx_cell[i]*( sy_old_node[j]*sz_old_node[k]*one_third
                                        + sy_old_node[j]*sz_new_node[k]*one_sixth
                                        + sy_new_node[j]*sz_old_node[k]*one_sixth
                                        + sy_new_node[j]*sz_new_node[k]*one_third )*seg_factor_x;
                    amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i0_cell+i, lo.y+j0_node+j, lo.z+k0_node+k), wqx*weight);
                    amrex::Gpu::Atomic::AddNoRet( &Sxx_arr(lo.x+i0_cell+i, lo.y+j0_node+j, lo.z+k0_node+k, 0), fpxx*weight*weight);
                }
            }
        }

        // deposit Jy and Syy or this segment
        for (int i=0; i<=depos_order; i++) {
            for (int j=0; j<=depos_order-1; j++) {
                for (int k=0; k<=depos_order; k++) {
                    weight = sy_cell[j]*( sx_old_node[i]*sz_old_node[k]*one_third
                                        + sx_old_node[i]*sz_new_node[k]*one_sixth
                                        + sx_new_node[i]*sz_old_node[k]*one_sixth
                                        + sx_new_node[i]*sz_new_node[k]*one_third )*seg_factor_y;
                    amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i0_node+i, lo.y+j0_cell+j, lo.z+k0_node+k), wqy*weight);
                    amrex::Gpu::Atomic::AddNoRet( &Syy_arr(lo.x+i0_node+i, lo.y+j0_cell+j, lo.z+k0_node+k, 0), fpyy*weight*weight);
                }
            }
        }

        // deposit Jz and Sz for this segment
        for (int i=0; i<=depos_order; i++) {
            for (int j=0; j<=depos_order; j++) {
                for (int k=0; k<=depos_order-1; k++) {
                    weight = sz_cell[k]*( sx_old_node[i]*sy_old_node[j]*one_third
                                        + sx_old_node[i]*sy_new_node[j]*one_sixth
                                        + sx_new_node[i]*sy_old_node[j]*one_sixth
                                        + sx_new_node[i]*sy_new_node[j]*one_third )*seg_factor_z;
                    amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i0_node+i, lo.y+j0_node+j, lo.z+k0_cell+k), wqz*weight);
                    amrex::Gpu::Atomic::AddNoRet( &Szz_arr(lo.x+i0_node+i, lo.y+j0_node+j, lo.z+k0_cell+k, 0), fpzz*weight*weight);
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

    // Compute total change in particle position and the initial cell
    // locations in each direction used to find the position at cell crossings.
    // Keep these double to avoid bug in single precision
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

    constexpr int num_segments_max = 1 + 4*AMREX_SPACEDIM;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE( num_segments <= num_segments_max,
        "Error: num_segments must be less than or equal to 1 + 4*AMREX_SPACEDIM.");

    // Save the start index and interpolation weights for each segment
    int i0_cell[num_segments_max];
    int i0_node[num_segments_max];
    int k0_cell[num_segments_max];
    int k0_node[num_segments_max];
    amrex::Real weight_cellX_nodeZ[num_segments_max][depos_order][depos_order+1];
    amrex::Real weight_nodeX_cellZ[num_segments_max][depos_order+1][depos_order];
    amrex::Real weight_nodeX_nodeZ[num_segments_max][depos_order+1][depos_order+1];

    const auto i_mid = static_cast<int>(0.5*(x_new+x_old)-shift);
    const auto k_mid = static_cast<int>(0.5*(z_new+z_old)-shift);
    int SegNumX[num_segments_max];
    int SegNumZ[num_segments_max];

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

        // Compute the segment factors (each equal to dt_seg/dt for nonzero dxp, or dzp)
        const auto seg_factor_x = static_cast<amrex::Real>(dxp == 0. ? 1._rt : dxp_seg/dxp);
        const auto seg_factor_z = static_cast<amrex::Real>(dzp == 0. ? 1._rt : dzp_seg/dzp);

        // Compute cell-based weights using the average segment position
        // Keep these double to avoid bug in single precision
        double sx_cell[depos_order] = {0.};
        double sz_cell[depos_order] = {0.};
        double const x0_bar = (x0_new + x0_old)/2.0;
        double const z0_bar = (z0_new + z0_old)/2.0;
        i0_cell[ns] = compute_shape_factor_cell( sx_cell, x0_bar-0.5 );
        k0_cell[ns] = compute_shape_factor_cell( sz_cell, z0_bar-0.5 );

        // Set the segment number for the mass matrix component calc
        if constexpr (full_mass_matrices) {
            const auto i0_mid = static_cast<int>(x0_bar-shift);
            const auto k0_mid = static_cast<int>(z0_bar-shift);
            SegNumX[ns] = 1 + i0_mid - i_mid;
            SegNumZ[ns] = 1 + k0_mid - k_mid;
        }

        if constexpr (depos_order >= 3) { // higher-order correction to the cell-based weights
            const Compute_shape_factor_pair<depos_order-1> compute_shape_factors_cell;
            double sx_old_cell[depos_order] = {0.};
            double sx_new_cell[depos_order] = {0.};
            double sz_old_cell[depos_order] = {0.};
            double sz_new_cell[depos_order] = {0.};
            const int i0_cell_2 = compute_shape_factors_cell( sx_old_cell, sx_new_cell, x0_old-0.5, x0_new-0.5 );
            const int k0_cell_2 = compute_shape_factors_cell( sz_old_cell, sz_new_cell, z0_old-0.5, z0_new-0.5 );
            amrex::ignore_unused(i0_cell_2, k0_cell_2);
            for (int m=0; m<depos_order; m++) {
                sx_cell[m] = (4.0*sx_cell[m] + sx_old_cell[m] + sx_new_cell[m])/6.0;
                sz_cell[m] = (4.0*sz_cell[m] + sz_old_cell[m] + sz_new_cell[m])/6.0;
            }
        }

        // Compute node-based weights using the old and new segment positions
        // Keep these double to avoid bug in single precision
        double sx_old_node[depos_order+1] = {0.};
        double sx_new_node[depos_order+1] = {0.};
        double sz_old_node[depos_order+1] = {0.};
        double sz_new_node[depos_order+1] = {0.};
        i0_node[ns] = compute_shape_factors_node( sx_old_node, sx_new_node, x0_old, x0_new );
        k0_node[ns] = compute_shape_factors_node( sz_old_node, sz_new_node, z0_old, z0_new );

        // deposit Jx and Sx for this segment
        amrex::Real weight;
        for (int i=0; i<=depos_order-1; i++) {
            for (int k=0; k<=depos_order; k++) {
                const int i_J = lo.x + i0_cell[ns] + i;
                const int k_J = lo.y + k0_node[ns] + k;
                weight = sx_cell[i]*(sz_old_node[k] + sz_new_node[k])/2.0_rt*seg_factor_x;
                amrex::Gpu::Atomic::AddNoRet(&Jx_arr(i_J, k_J, 0, 0), wqx*weight);
                if constexpr (full_mass_matrices) { weight_cellX_nodeZ[ns][i][k] = weight; }
                else {
                    amrex::Gpu::Atomic::AddNoRet(&Sxx_arr(i_J, k_J, 0, 0), fpxx*weight*weight);
                }
            }
        }

        // deposit out-of-plane Jy and Sy for this segment
        const auto seg_factor_y = std::min(seg_factor_x,seg_factor_z);
        for (int i=0; i<=depos_order; i++) {
            for (int k=0; k<=depos_order; k++) {
                const int i_J = lo.x + i0_node[ns] + i;
                const int k_J = lo.y + k0_node[ns] + k;
                weight = ( sx_old_node[i]*sz_old_node[k]*one_third
                       +   sx_old_node[i]*sz_new_node[k]*one_sixth
                       +   sx_new_node[i]*sz_old_node[k]*one_sixth
                       +   sx_new_node[i]*sz_new_node[k]*one_third )*seg_factor_y;
                amrex::Gpu::Atomic::AddNoRet(&Jy_arr(i_J, k_J, 0, 0), wqy*weight);
                if constexpr (full_mass_matrices) { weight_nodeX_nodeZ[ns][i][k] = weight; }
                else {
                    amrex::Gpu::Atomic::AddNoRet(&Syy_arr(i_J, k_J, 0, 0), fpyy*weight*weight);
                }
            }
        }

        // deposit Jz and Szz for this segment
        for (int i=0; i<=depos_order; i++) {
            for (int k=0; k<=depos_order-1; k++) {
                const int i_J = lo.x + i0_node[ns] + i;
                const int k_J = lo.y + k0_cell[ns] + k;
                weight = sz_cell[k]*(sx_old_node[i] + sx_new_node[i])/2.0_rt*seg_factor_z;
                amrex::Gpu::Atomic::AddNoRet(&Jz_arr(i_J, k_J, 0, 0), wqz*weight);
                if constexpr (full_mass_matrices) { weight_nodeX_cellZ[ns][i][k] = weight; }
                else {
                    amrex::Gpu::Atomic::AddNoRet(&Szz_arr(i_J, k_J, 0, 0), fpzz*weight*weight);
                }
            }
        }

        // update old segment values
        if (ns < num_segments-1) {
            x0_old = x0_new;
            z0_old = z0_new;
        }

    } // end loop over segments

    if constexpr (full_mass_matrices) {

    // Loop over segments and deposit full mass matrices
    for (int ns=0; ns<num_segments; ns++) {

        // Deposit Sxx, Sxz, and Sxy for this segment
        for (int i=0; i<=depos_order-1; i++) {
            for (int k=0; k<=depos_order; k++) {
                const int i_J = lo.x + i0_cell[ns] + i;
                const int k_J = lo.y + k0_node[ns] + k;
                const amrex::Real weight_J = weight_cellX_nodeZ[ns][i][k];
                for (int ms=0; ms<num_segments; ms++) {
                    const int SegShiftX = max_crossings + SegNumX[ms] - SegNumX[ns];
                    const int SegShiftZ = max_crossings + SegNumZ[ms] - SegNumZ[ns];
                    // Deposit Sxx
                    const int Ncomp_xx0 = 1 + 2*(depos_order-1) + 2*max_crossings;
                    for (int iE=0; iE<=depos_order-1; iE++) {
                        for (int kE=0; kE<=depos_order; kE++) {
                            const amrex::Real weight_E = weight_cellX_nodeZ[ms][iE][kE];
                            const int comp_xx = depos_order-1 - i + iE + SegShiftX
                                  +  Ncomp_xx0*(depos_order - k + kE + SegShiftZ);
                            amrex::Gpu::Atomic::AddNoRet(&Sxx_arr(i_J, k_J, 0, comp_xx), fpxx*weight_J*weight_E);
                        }
                    }
                    // Deposit Sxz
                    const int Ncomp_xz0 = 2*depos_order + 2*max_crossings;
                    for (int iE=0; iE<=depos_order; iE++) {
                        for (int kE=0; kE<=depos_order-1; kE++) {
                            const amrex::Real weight_E = weight_nodeX_cellZ[ms][iE][kE];
                            const int comp_xz = depos_order-1 - i + iE + SegShiftX
                                   + Ncomp_xz0*(depos_order - k + kE + SegShiftZ);
                            amrex::Gpu::Atomic::AddNoRet(&Sxz_arr(i_J, k_J, 0, comp_xz), fpxz*weight_J*weight_E);
                        }
                    }
                    // Deposit Sxy
                    const int Ncomp_xy0 = 2*depos_order + 2*max_crossings;
                    for (int iE=0; iE<=depos_order; iE++) {
                        for (int kE=0; kE<=depos_order; kE++) {
                            const amrex::Real weight_E = weight_nodeX_nodeZ[ms][iE][kE];
                            const int comp_xy = depos_order-1 - i + iE + SegShiftX
                                   + Ncomp_xy0*(depos_order - k + kE + SegShiftZ);
                            amrex::Gpu::Atomic::AddNoRet(&Sxy_arr(i_J, k_J, 0, comp_xy), fpxy*weight_J*weight_E);
                        }
                    }

                }
            }
        }

        // Deposit Szx, Szz, and Szy for this segment
        for (int i=0; i<=depos_order; i++) {
            for (int k=0; k<=depos_order-1; k++) {
                const int i_J = lo.x + i0_node[ns] + i;
                const int k_J = lo.y + k0_cell[ns] + k;
                const amrex::Real weight_J = weight_nodeX_cellZ[ns][i][k];
                for (int ms=0; ms<num_segments; ms++) {
                    const int SegShiftX = max_crossings + SegNumX[ms] - SegNumX[ns];
                    const int SegShiftZ = max_crossings + SegNumZ[ms] - SegNumZ[ns];
                    // Deposit Szx
                    const int Ncomp_zx0 = 2*depos_order + 2*max_crossings;
                    for (int iE=0; iE<=depos_order-1; iE++) {
                        for (int kE=0; kE<=depos_order; kE++) {
                            const amrex::Real weight_E = weight_cellX_nodeZ[ms][iE][kE];
                            const int comp_zx = depos_order - i + iE + SegShiftX
                                  +  Ncomp_zx0*(depos_order-1 - k + kE + SegShiftZ);
                            amrex::Gpu::Atomic::AddNoRet( &Szx_arr(i_J, k_J, 0, comp_zx), fpzx*weight_J*weight_E);
                        }
                    }
                    // Deposit Szz
                    const int Ncomp_zz0 = 1 + 2*depos_order + 2*max_crossings;
                    for (int iE=0; iE<=depos_order; iE++) {
                        for (int kE=0; kE<=depos_order-1; kE++) {
                            const amrex::Real weight_E = weight_nodeX_cellZ[ms][iE][kE];
                            const int comp_zz = depos_order - i + iE + SegShiftX
                                   + Ncomp_zz0*(depos_order-1 - k + kE + SegShiftZ);
                            amrex::Gpu::Atomic::AddNoRet( &Szz_arr(i_J, k_J, 0, comp_zz), fpzz*weight_J*weight_E);
                        }
                    }
                    // Deposit Szy
                    const int Ncomp_zy0 = 1 + 2*depos_order + 2*max_crossings;
                    for (int iE=0; iE<=depos_order; iE++) {
                        for (int kE=0; kE<=depos_order; kE++) {
                            const amrex::Real weight_E = weight_nodeX_nodeZ[ms][iE][kE];
                            const int comp_zy = depos_order - i + iE + SegShiftX
                                   + Ncomp_zy0*(depos_order-1 - k + kE + SegShiftZ);
                            amrex::Gpu::Atomic::AddNoRet( &Szy_arr(i_J, k_J, 0, comp_zy), fpzy*weight_J*weight_E);
                        }
                    }

                }
            }
        }

        // Deposit Syx, Syz, and Syy for this segment
        for (int i=0; i<=depos_order; i++) {
            for (int k=0; k<=depos_order; k++) {
                const int i_J = lo.x + i0_node[ns] + i;
                const int k_J = lo.y + k0_node[ns] + k;
                const amrex::Real weight_J = weight_nodeX_nodeZ[ns][i][k];
                for (int ms=0; ms<num_segments; ms++) {
                    const int SegShiftX = max_crossings + SegNumX[ms] - SegNumX[ns];
                    const int SegShiftZ = max_crossings + SegNumZ[ms] - SegNumZ[ns];
                    // Deposit Syx
                    const int Ncomp_yx0 = 2*depos_order + 2*max_crossings;
                    for (int iE=0; iE<=depos_order-1; iE++) {
                        for (int kE=0; kE<=depos_order; kE++) {
                            const amrex::Real weight_E = weight_cellX_nodeZ[ms][iE][kE];
                            const int comp_yx = depos_order - i + iE + SegShiftX
                                  +  Ncomp_yx0*(depos_order - k + kE + SegShiftZ);
                            amrex::Gpu::Atomic::AddNoRet( &Syx_arr(i_J, k_J, 0, comp_yx), fpyx*weight_J*weight_E);
                        }
                    }
                    // Deposit Syz
                    const int Ncomp_yz0 = 1 + 2*depos_order + 2*max_crossings;
                    for (int iE=0; iE<=depos_order; iE++) {
                        for (int kE=0; kE<=depos_order-1; kE++) {
                            const amrex::Real weight_E = weight_nodeX_cellZ[ms][iE][kE];
                            const int comp_yz = depos_order - i + iE + SegShiftX
                                   + Ncomp_yz0*(depos_order - k + kE + SegShiftZ);
                            amrex::Gpu::Atomic::AddNoRet( &Syz_arr(i_J, k_J, 0, comp_yz), fpyz*weight_J*weight_E);
                        }
                    }
                    // Deposit Syy
                    const int Ncomp_yy0 = 1 + 2*depos_order + 2*max_crossings;
                    for (int iE=0; iE<=depos_order; iE++) {
                        for (int kE=0; kE<=depos_order; kE++) {
                            const amrex::Real weight_E = weight_nodeX_nodeZ[ms][iE][kE];
                            const int comp_yy = depos_order - i + iE + SegShiftX
                                   + Ncomp_yy0*(depos_order - k + kE + SegShiftZ);
                            amrex::Gpu::Atomic::AddNoRet( &Syy_arr(i_J, k_J, 0, comp_yy), fpyy*weight_J*weight_E);
                        }
                    }

                }
            }
        }

     }

     }

#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)

    // compute cell crossings in X-direction
    const auto i_old = static_cast<int>(x_old-shift);
    const auto i_new = static_cast<int>(x_new-shift);
    const int cell_crossings_x = std::abs(i_new-i_old);
    num_segments += cell_crossings_x;

    // Compute dxp and the initial cell location used to find the cell crossings.
    // Keep these double to avoid bug in single precision
    double const dxp = x_new - x_old;
    const auto dirX_sign = static_cast<double>(dxp < 0. ? -1. : 1.);
    double Xcell = static_cast<double>(i_old) + shift + 0.5*(1.-dirX_sign);

    // loop over the number of segments and deposit
    const Compute_shape_factor< depos_order-1 > compute_shape_factor_cell;
    const Compute_shape_factor_pair< depos_order > compute_shape_factors_node;
    double dxp_seg;
    double x0_new;
    double x0_old = x_old;

    for (int ns=0; ns<num_segments; ns++) {

        if (ns == num_segments-1) { // final segment
            x0_new = x_new;
            dxp_seg = x0_new - x0_old;
        }
        else {
            Xcell = Xcell + dirX_sign;
            x0_new = Xcell;
            dxp_seg = x0_new - x0_old;
        }

        // Compute the segment factor (equal to dt_seg/dt for nonzero dxp)
        const auto seg_factor = static_cast<amrex::Real>(dxp == 0. ? 1._rt : dxp_seg/dxp);

        // Compute cell-based weights using the average segment position
        // Keep these double to avoid bug in single precision
        double sx_cell[depos_order] = {0.};
        double const x0_bar = (x0_new + x0_old)/2.0;
        const int i0_cell = compute_shape_factor_cell( sx_cell, x0_bar-0.5 );

        if constexpr (depos_order >= 3) { // higher-order correction to the cell-based weights
            const Compute_shape_factor_pair<depos_order-1> compute_shape_factors_cell;
            double sx_old_cell[depos_order] = {0.};
            double sx_new_cell[depos_order] = {0.};
            const int i0_cell_2 = compute_shape_factors_cell( sx_old_cell, sx_new_cell, x0_old-0.5, x0_new-0.5 );
            amrex::ignore_unused(i0_cell_2);
            for (int m=0; m<depos_order; m++) {
                sx_cell[m] = (4.0*sx_cell[m] + sx_old_cell[m] + sx_new_cell[m])/6.0;
            }
        }

        // Compute node-based weights using the old and new segment positions
        // Keep these double to avoid bug in single precision
        double sx_old_node[depos_order+1] = {0.};
        double sx_new_node[depos_order+1] = {0.};
        const int i0_node = compute_shape_factors_node( sx_old_node, sx_new_node, x0_old, x0_new );

        // deposit out-of-plane Jy, Jz, Syy, and Szz for this segment
        for (int i=0; i<=depos_order; i++) {
            const amrex::Real weight = 0.5_rt*(sx_old_node[i] + sx_new_node[i])*seg_factor;
            amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i0_node+i, 0, 0), wqy*weight);
            amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i0_node+i, 0, 0), wqz*weight);
            //
            amrex::Gpu::Atomic::AddNoRet( &Syy_arr(lo.x+i0_node+i, 0, 0), fpyy*weight*weight);
            amrex::Gpu::Atomic::AddNoRet( &Szz_arr(lo.x+i0_node+i, 0, 0), fpzz*weight*weight);
        }

        // deposit Jx and Sxx for this segment
        for (int i=0; i<=depos_order-1; i++) {
            const amrex::Real weight = sx_cell[i]*seg_factor;
            amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i0_cell+i, 0, 0), wqx*weight);
            //
            amrex::Gpu::Atomic::AddNoRet( &Sxx_arr(lo.x+i0_cell+i, 0, 0), fpxx*weight*weight);
        }

        // update old segment values
        if (ns < num_segments-1) {
            x0_old = x0_new;
        }

    }

#elif defined(WARPX_DIM_1D_Z)

    // compute cell crossings in Z-direction
    const auto k_old = static_cast<int>(z_old-shift);
    const auto k_new = static_cast<int>(z_new-shift);
    const int cell_crossings_z = std::abs(k_new-k_old);
    num_segments += cell_crossings_z;

    // Compute dzp and the initial cell location used to find the cell crossings.
    // Keep these double to avoid bug in single precision
    double const dzp = z_new - z_old;
    const auto dirZ_sign = static_cast<double>(dzp < 0. ? -1. : 1.);
    double Zcell = static_cast<double>(k_old) + shift + 0.5*(1.-dirZ_sign);

    // loop over the number of segments and deposit
    const Compute_shape_factor< depos_order-1 > compute_shape_factor_cell;
    const Compute_shape_factor_pair< depos_order > compute_shape_factors_node;
    double dzp_seg;
    double z0_new;
    double z0_old = z_old;

    constexpr int num_segments_max = 1 + 4*AMREX_SPACEDIM;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE( num_segments <= num_segments_max,
        "Error: num_segments must be less than or equal to 1 + 4*AMREX_SPACEDIM.");

    // Save the start index and interpolation weights for each segment
    int k0_cell[num_segments_max];
    int k0_node[num_segments_max];
    amrex::Real weight_cell[num_segments_max][depos_order];
    amrex::Real weight_node[num_segments_max][depos_order+1];

    const auto k_mid = static_cast<int>(0.5*(z_new+z_old)-shift);
    int SegNum[num_segments_max];

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

        // Compute the segment factor (equal to dt_seg/dt for nonzero dzp)
        const auto seg_factor = static_cast<amrex::Real>(dzp == 0. ? 1._rt : dzp_seg/dzp);

        // Compute cell-based weights using the average segment position
        // Keep these double to avoid bug in single precision
        double sz_cell[depos_order] = {0.};
        double const z0_bar = (z0_new + z0_old)/2.0;
        k0_cell[ns] = compute_shape_factor_cell( sz_cell, z0_bar-0.5 );

        // Set the segment number for the mass matrix component calc
        if constexpr (full_mass_matrices) {
            const auto k0_mid = static_cast<int>(z0_bar-shift);
            SegNum[ns] = 1 + k0_mid - k_mid;
        }

        if constexpr (depos_order >= 3) { // higher-order correction to the cell-based weights
            const Compute_shape_factor_pair<depos_order-1> compute_shape_factors_cell;
            double sz_old_cell[depos_order] = {0.};
            double sz_new_cell[depos_order] = {0.};
            const int k0_cell_2 = compute_shape_factors_cell( sz_old_cell, sz_new_cell, z0_old-0.5, z0_new-0.5 );
            amrex::ignore_unused(k0_cell_2);
            for (int m=0; m<depos_order; m++) {
                sz_cell[m] = (4.0*sz_cell[m] + sz_old_cell[m] + sz_new_cell[m])/6.0;
            }
        }

        // Compute node-based weights using the old and new segment positions
        // Keep these double to avoid bug in single precision
        double sz_old_node[depos_order+1] = {0.};
        double sz_new_node[depos_order+1] = {0.};
        k0_node[ns] = compute_shape_factors_node( sz_old_node, sz_new_node, z0_old, z0_new );

        // deposit out-of-plane Jx, Jy, Sx, and Sy for this segment
        for (int k=0; k<=depos_order; k++) {
            const amrex::Real weight = 0.5_rt*(sz_old_node[k] + sz_new_node[k])*seg_factor;
            const int k_J = lo.x + k0_node[ns] + k;
            amrex::Gpu::Atomic::AddNoRet(&Jx_arr(k_J, 0, 0), wqx*weight);
            amrex::Gpu::Atomic::AddNoRet(&Jy_arr(k_J, 0, 0), wqy*weight);
            if constexpr (full_mass_matrices) { weight_node[ns][k] = weight; }
            else {
                amrex::Gpu::Atomic::AddNoRet(&Sxx_arr(k_J, 0, 0), fpxx*weight*weight);
                amrex::Gpu::Atomic::AddNoRet(&Syy_arr(k_J, 0, 0), fpyy*weight*weight);
            }
        }

        // deposit Jz and Szz for this segment
        for (int k=0; k<=depos_order-1; k++) {
            const amrex::Real weight = sz_cell[k]*seg_factor;
            const int k_J = lo.x + k0_cell[ns] + k;
            amrex::Gpu::Atomic::AddNoRet(&Jz_arr(k_J, 0, 0), wqz*weight);
            if constexpr (full_mass_matrices) { weight_cell[ns][k] = weight; }
            else {
                amrex::Gpu::Atomic::AddNoRet(&Szz_arr(k_J, 0, 0), fpzz*weight*weight);
            }
        }

        // update old segment values
        if (ns < num_segments-1) {
            z0_old = z0_new;
        }

    }

    if constexpr (full_mass_matrices) {

    // Loop over segments and deposit full mass matrices
    for (int ns=0; ns<num_segments; ns++) {

        // Deposit Sxx, Sxy, Sxz, Syx, Syy, and Syz for this segment
        for (int k=0; k<=depos_order; k++) {

            const int k_J = lo.x + k0_node[ns] + k;
            const amrex::Real weight_J = weight_node[ns][k];
            for (int ms=0; ms<num_segments; ms++) {
                const int SegShift = max_crossings + SegNum[ms] - SegNum[ns];
                for (int kE=0; kE<=depos_order; kE++) {
                    const amrex::Real weight_E = weight_node[ms][kE];
                    const int comp_yy = depos_order - k + kE + SegShift;
                    amrex::Gpu::Atomic::AddNoRet(&Sxx_arr(k_J, 0, comp_yy), fpxx*weight_J*weight_E);
                    amrex::Gpu::Atomic::AddNoRet(&Syy_arr(k_J, 0, comp_yy), fpyy*weight_J*weight_E);
                    amrex::Gpu::Atomic::AddNoRet(&Sxy_arr(k_J, 0, comp_yy), fpxy*weight_J*weight_E);
                    amrex::Gpu::Atomic::AddNoRet(&Syx_arr(k_J, 0, comp_yy), fpyx*weight_J*weight_E);
                }
                for (int kE=0; kE<=depos_order-1; kE++) {
                    const amrex::Real weight_E = weight_cell[ms][kE];
                    const int comp_yz = depos_order - k + kE + SegShift;
                    amrex::Gpu::Atomic::AddNoRet(&Sxz_arr(k_J, 0, comp_yz), fpxz*weight_J*weight_E);
                    amrex::Gpu::Atomic::AddNoRet(&Syz_arr(k_J, 0, comp_yz), fpyz*weight_J*weight_E);
                }
            }

        }

        // Deposit Szx, Szy, and Szz for this segment
        for (int k=0; k<=depos_order-1; k++) {

            const int k_J = lo.x + k0_cell[ns] + k;
            const amrex::Real weight_J = weight_cell[ns][k];
            for (int ms=0; ms<num_segments; ms++) {
                const int SegShift = max_crossings + SegNum[ms] - SegNum[ns];
                for (int kE=0; kE<=depos_order-1; kE++) {
                    const amrex::Real weight_E = weight_cell[ms][kE];
                    const int comp_zz = depos_order-1 - k + kE + SegShift;
                    amrex::Gpu::Atomic::AddNoRet(&Szz_arr(k_J, 0, comp_zz), fpzz*weight_J*weight_E);
                }
                for (int kE=0; kE<=depos_order; kE++) {
                    const amrex::Real weight_E = weight_node[ms][kE];
                    const int comp_zy = depos_order-1 - k + kE + SegShift;
                    amrex::Gpu::Atomic::AddNoRet(&Szx_arr(k_J, 0, comp_zy), fpzx*weight_J*weight_E);
                    amrex::Gpu::Atomic::AddNoRet(&Szy_arr(k_J, 0, comp_zy), fpzy*weight_J*weight_E);
                }
            }

        }

    }

    }

#endif
}

/**
 * \brief Villasenor and Buneman deposition of J and mass matrices for thread thread_num
 * \tparam depos_order             Deposition order
 * \tparam full_mass_matrices      Whether to deposit the full mass matrices
 *                                 (can only be true for depos_order = 2)
 * \param xp_n_data,yp_n_data,zp_n_data Pointer to arrays of particle position at time level n.
 * \param GetPosition              A functor for returning the particle position.
 * \param wp                       Pointer to array of particle weights.
 * \param uxp_n,uyp_n,uzp_n        Pointer to arrays of particle momentum at time level n.
 * \param uxp_nph,uyp_nph,uzp_nph  Pointer to arrays of particle momentum at time level n + 1/2.
 * \param Jx_arr,Jy_arr,Jz_arr     Array4 of the current density, either full array or tile.
 * \param max_crossings            Maximum number of particle cell crossings in each dir
 * \param Sxx_arr,Sxy_arr,Sxz_arr  Array4 of mass matrices for Jx, either full array or tile.
 * \param Syx_arr,Syy_arr,Syz_arr  Array4 of mass matrices for Jy, either full array or tile.
 * \param Szx_arr,Szy_arr,Szz_arr  Array4 of mass matrices for Jz, either full array or tile.
 * \param Bx_arr,By_arr,Bz_arr     Array4 of the magnetic field, either full array or tile.
 * \param Bx_type,By_type,Bz_type  IndexType of the magnetic field
 * \param np_to_deposit            Number of particles for which current is deposited.
 * \param dt                       Time step for particle level
 * \param dinv                     3D cell size inverse
 * \param xyzmin                   Physical lower bounds of domain.
 * \param lo                       Index lower bounds of domain.
 * \param qs                       Species charge.
 * \param ms                       Species mass.
 */
template <int depos_order, bool full_mass_matrices>
void doVillasenorJandSigmaDeposition ( [[maybe_unused]] const amrex::ParticleReal* xp_n_data,
                                       [[maybe_unused]] const amrex::ParticleReal* yp_n_data,
                                       [[maybe_unused]] const amrex::ParticleReal* zp_n_data,
                                       const GetParticlePosition<PIdx>& GetPosition,
                                       const amrex::ParticleReal* wp,
                                       const amrex::ParticleReal* uxp_n,
                                       const amrex::ParticleReal* uyp_n,
                                       const amrex::ParticleReal* uzp_n,
                                       const amrex::ParticleReal* uxp_nph,
                                       const amrex::ParticleReal* uyp_nph,
                                       const amrex::ParticleReal* uzp_nph,
                                       amrex::Array4<amrex::Real> const& Jx_arr,
                                       amrex::Array4<amrex::Real> const& Jy_arr,
                                       amrex::Array4<amrex::Real> const& Jz_arr,
                                       const int max_crossings,
                                       amrex::Array4<amrex::Real> const& Sxx_arr,
                                       amrex::Array4<amrex::Real> const& Sxy_arr,
                                       amrex::Array4<amrex::Real> const& Sxz_arr,
                                       amrex::Array4<amrex::Real> const& Syx_arr,
                                       amrex::Array4<amrex::Real> const& Syy_arr,
                                       amrex::Array4<amrex::Real> const& Syz_arr,
                                       amrex::Array4<amrex::Real> const& Szx_arr,
                                       amrex::Array4<amrex::Real> const& Szy_arr,
                                       amrex::Array4<amrex::Real> const& Szz_arr,
                                       const amrex::Array4<amrex::Real const>& Bx_arr,
                                       const amrex::Array4<amrex::Real const>& By_arr,
                                       const amrex::Array4<amrex::Real const>& Bz_arr,
                                       const amrex::IndexType Bx_type,
                                       const amrex::IndexType By_type,
                                       const amrex::IndexType Bz_type,
                                       const long np_to_deposit,
                                       const amrex::Real dt,
                                       const amrex::XDim3& dinv,
                                       const amrex::XDim3& xyzmin,
                                       const amrex::Dim3 lo,
                                       const amrex::Real qs,
                                       const amrex::Real ms )
{
    using namespace amrex::literals;

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

    // Loop over particles and deposit into Jx_arr, Jy_arr and Jz_arr
    amrex::ParallelFor(
        np_to_deposit,
        [=] AMREX_GPU_DEVICE (long const ip) {

            // Skip particles with zero weight.
            // This should only be the case for particles that will be suborbited.
            if (wp[ip] == 0.) { return; }

            amrex::ParticleReal xp_nph, yp_nph, zp_nph;
            GetPosition(ip, xp_nph, yp_nph, zp_nph);

            // Compute magnetic field on particle
            amrex::ParticleReal Bxp = 0.0;
            amrex::ParticleReal Byp = 0.0;
            amrex::ParticleReal Bzp = 0.0;
            const int depos_order_perp = 1;
            const int depos_order_para = 1;
            const int n_rz_azimuthal_modes = 0;
            doDirectGatherVectorField<depos_order_perp,depos_order_para>(
                                    xp_nph, yp_nph, zp_nph,
                                    Bxp, Byp, Bzp,
                                    Bx_arr, By_arr, Bz_arr,
                                    Bx_type, By_type, Bz_type,
                                    dinv, xyzmin, lo, n_rz_azimuthal_modes );

            // Compute inverse Lorentz factor, the average of gamma at time levels n and n+1
            const amrex::ParticleReal gaminv = GetImplicitGammaInverse(uxp_n[ip], uyp_n[ip], uzp_n[ip],
                                                                       uxp_nph[ip], uyp_nph[ip], uzp_nph[ip]);

            // Compute current density kernels to deposit
            const amrex::Real wq_invvol = qs*wp[ip]*invvol;
            const amrex::Real rhop = wq_invvol*gaminv;

            // Set the Mass Matrices kernels
            amrex::ParticleReal fpxx, fpxy, fpxz;
            amrex::ParticleReal fpyx, fpyy, fpyz;
            amrex::ParticleReal fpzx, fpzy, fpzz;
            setMassMatricesKernels( qs, ms, dt, rhop,
                                    uxp_nph[ip], uyp_nph[ip], uzp_nph[ip],
                                    Bxp, Byp, Bzp,
                                    fpxx, fpxy, fpxz,
                                    fpyx, fpyy, fpyz,
                                    fpzx, fpzy, fpzz );

            amrex::ParticleReal const xp_n = (xp_n_data ? xp_n_data[ip] : 0._prt);
            amrex::ParticleReal const yp_n = (yp_n_data ? yp_n_data[ip] : 0._prt);
            amrex::ParticleReal const zp_n = (zp_n_data ? zp_n_data[ip] : 0._prt);

            // Compute position at time n + 1
            amrex::ParticleReal const xp_np1 = 2._prt*xp_nph - xp_n;
            amrex::ParticleReal const yp_np1 = 2._prt*yp_nph - yp_n;
            amrex::ParticleReal const zp_np1 = 2._prt*zp_nph - zp_n;

            doVillasenorJandSigmaDepositionKernel<depos_order,full_mass_matrices>(
                                                                xp_n, yp_n, zp_n,
                                                                xp_np1, yp_np1, zp_np1,
                                                                wq_invvol,
                                                                uxp_nph[ip], uyp_nph[ip], uzp_nph[ip],
                                                                gaminv,
                                                                fpxx, fpxy, fpxz,
                                                                fpyx, fpyy, fpyz,
                                                                fpzx, fpzy, fpzz,
                                                                Jx_arr, Jy_arr, Jz_arr,
                                                                max_crossings,
                                                                Sxx_arr, Sxy_arr, Sxz_arr,
                                                                Syx_arr, Syy_arr, Syz_arr,
                                                                Szx_arr, Szy_arr, Szz_arr,
                                                                dt, dinv, xyzmin, lo );

    });
}

#endif // WARPX_MASSMATRICESDEPOSITION_H_
