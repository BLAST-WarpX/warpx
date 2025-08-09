/* Copyright 2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_TEMPERATUREDEPOSITION_H_
#define WARPX_TEMPERATUREDEPOSITION_H_

#include "Particles/Deposition/VarianceAccumulationBuffer.H"
#include "Particles/Deposition/SharedDepositionUtils.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/ShapeFactors.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#ifdef WARPX_DIM_RZ
#   include "Utils/WarpX_Complex.H"
#endif

#include "WarpX.H" // todo: remove include and pass globals as args

#include "TemperatureDepositionTypes.H"

#include <AMReX.H>
#include <AMReX_Arena.H>
#include <AMReX_Array4.H>
#include <AMReX_Dim3.H>
#include <AMReX_REAL.H>

namespace warpx::particles::deposition {

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void varianceDepositionSubKernel(
    amrex::ParticleReal vx,
    amrex::ParticleReal vy,
    amrex::ParticleReal vz,
    const amrex::IntVectND<3>& ixv,
    const amrex::IntVectND<3>& iyv,
    const amrex::IntVectND<3>& izv,
    amrex::Real wpx_var,
    amrex::Real wpy_var,
    amrex::Real wpz_var,
    const amrex::Array4<int> & nx_arr,
    const amrex::Array4<int> & ny_arr,
    const amrex::Array4<int> & nz_arr,
    const amrex::Array4<amrex::Real> & wx_arr,
    const amrex::Array4<amrex::Real> & wy_arr,
    const amrex::Array4<amrex::Real> & wz_arr,
    const amrex::Array4<amrex::Real> & w2x_arr,
    const amrex::Array4<amrex::Real> & w2y_arr,
    const amrex::Array4<amrex::Real> & w2z_arr,
    const amrex::Array4<amrex::Real> & vxbar_arr,
    const amrex::Array4<amrex::Real> & vybar_arr,
    const amrex::Array4<amrex::Real> & vzbar_arr,
    const TemperatureDepositionType type,
    const TemperatureDepositionPass pass
)
{
    const amrex::Real vxr = static_cast<amrex::Real>(vx);
    const amrex::Real vyr = static_cast<amrex::Real>(vy);
    const amrex::Real vzr = static_cast<amrex::Real>(vz);

    if (type == TemperatureDepositionType::SINGLE_PASS || pass == TemperatureDepositionPass::FIRST)
    {
        // Update sample count
        amrex::Gpu::Atomic::AddNoRet(
            &nx_arr(ixv[0], ixv[1], ixv[2], 0), 1);
        amrex::Gpu::Atomic::AddNoRet(
            &ny_arr(iyv[0], iyv[1], iyv[2], 0), 1);
        amrex::Gpu::Atomic::AddNoRet(
            &nz_arr(izv[0], izv[1], izv[2], 0), 1);

        // Update w_sum
        amrex::Gpu::Atomic::AddNoRet(
            &wx_arr(ixv[0], ixv[1], ixv[2], 0), wpx_var);
        amrex::Gpu::Atomic::AddNoRet(
            &wy_arr(iyv[0], iyv[1], iyv[2], 0), wpy_var);
        amrex::Gpu::Atomic::AddNoRet(
            &wz_arr(izv[0], izv[1], izv[2], 0), wpz_var);

        // Update wv_sum to be normalized to vbar
        amrex::Gpu::Atomic::AddNoRet(
            &vxbar_arr(ixv[0], ixv[1], ixv[2], 0), wpx_var*vxr);
        amrex::Gpu::Atomic::AddNoRet(
            &vybar_arr(iyv[0], iyv[1], iyv[2], 0), wpy_var*vyr);
        amrex::Gpu::Atomic::AddNoRet(
            &vzbar_arr(izv[0], izv[1], izv[2], 0), wpz_var*vzr);
    }

    if (type == TemperatureDepositionType::SINGLE_PASS || pass == TemperatureDepositionPass::SECOND)
    {
        // For a single pass run through, the vbar arrays will be zeroed, so this should work
        // in either case. Will be non-demeaned in 1 pass
        amrex::Real vxb = 0.;
        amrex::Real vyb = 0.;
        amrex::Real vzb = 0.;

        // Normalization of accumulation arrays are not done until after all deposition, so normalize
        // here for double pass algorithm
        if (type == TemperatureDepositionType::DOUBLE_PASS)
        {
            if (nx_arr(ixv[0], ixv[1], ixv[2], 0) > 0) {
                vxb = vxbar_arr(ixv[0], ixv[1], ixv[2], 0)/wx_arr(ixv[0], ixv[1], ixv[2], 0);
            }
            if (ny_arr(iyv[0], iyv[1], iyv[2], 0) > 0) {
                vyb = vybar_arr(iyv[0], iyv[1], iyv[2], 0)/wy_arr(iyv[0], iyv[1], iyv[2], 0);
            }
            if (nz_arr(izv[0], izv[1], izv[2], 0) > 0) {
                vzb = vzbar_arr(izv[0], izv[1], izv[2], 0)/wz_arr(izv[0], izv[1], izv[2], 0);
            }

        }

        const amrex::Real vxd = vxr - vxb;
        const amrex::Real vyd = vyr - vyb;
        const amrex::Real vzd = vzr - vzb;

            // Update wv2_sum
        amrex::Gpu::Atomic::AddNoRet(
            &w2x_arr(ixv[0], ixv[1], ixv[2], 0), wpx_var*vxd*vxd);
        amrex::Gpu::Atomic::AddNoRet(
            &w2y_arr(iyv[0], iyv[1], iyv[2], 0), wpy_var*vyd*vyd);
        amrex::Gpu::Atomic::AddNoRet(
            &w2z_arr(izv[0], izv[1], izv[2], 0), wpz_var*vzd*vzd);
    }
}

/**
 * \brief Kernel for the direct current deposition for thread thread_num
 * \tparam depos_order deposition order
 * \param xp, yp, zp    The particle positions.
 * \param wp            The weight of the macroparticle
 * \param vx,vy,vz      The particle velocities
 * \param nx_arr,ny_arr,nz_arr Array4<int> of velocity variance sample count, either full array or tile.
 * \param wx_arr,wy_arr,wz_arr Array4 of weight sums, either full array or tile.
 * \param w2x_arr,w2y_arr,w2z_arr Array4 of weight**2 sums, either full array or tile.
 * \param vxbar_arr,vybar_arr,vzbar_arr Array4 of current density, either full array or tile.
 * \param varx_type,vary_type,varz_type The grid types along each direction, either NODE or CELL
 * \param type The type of deposition routine, either SINGLE_PASS or DOUBLE_PASS
 * \param pass The pass that is occurring, either FIRST or SECOND
 * \param relative_time Time at which to deposit J, relative to the time of the
 *                      current positions of the particles. When different than 0,
 *                      the particle position will be temporarily modified to match
 *                      the time of the deposition.
 * \param dinv          3D cell size inverse
 * \param xyzmin        The lower bounds of the domain
 * \param lo            Index lower bounds of domain.
 * \param n_rz_azimuthal_modes Number of azimuthal modes when using RZ geometry.
 */
template <int depos_order>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void doVarianceDepositionShapeNKernel(
    [[maybe_unused]] const amrex::ParticleReal xp,
    [[maybe_unused]] const amrex::ParticleReal yp,
    [[maybe_unused]] const amrex::ParticleReal zp,
    const amrex::ParticleReal wp,
    const amrex::ParticleReal vx,
    const amrex::ParticleReal vy,
    const amrex::ParticleReal vz,
    const amrex::Array4<int> & nx_arr,
    const amrex::Array4<int> & ny_arr,
    const amrex::Array4<int> & nz_arr,
    const amrex::Array4<amrex::Real> & wx_arr,
    const amrex::Array4<amrex::Real> & wy_arr,
    const amrex::Array4<amrex::Real> & wz_arr,
    const amrex::Array4<amrex::Real> & w2x_arr,
    const amrex::Array4<amrex::Real> & w2y_arr,
    const amrex::Array4<amrex::Real> & w2z_arr,
    const amrex::Array4<amrex::Real> & vxbar_arr,
    const amrex::Array4<amrex::Real> & vybar_arr,
    const amrex::Array4<amrex::Real> & vzbar_arr,
    const amrex::IntVect & varx_type,
    const amrex::IntVect & vary_type,
    const amrex::IntVect & varz_type,
    const TemperatureDepositionType type,
    const TemperatureDepositionPass pass,
    const amrex::Real relative_time,
    const amrex::XDim3 & dinv,
    const amrex::XDim3 & xyzmin,
    const amrex::Dim3 lo,
    [[maybe_unused]] const int n_rz_azimuthal_modes)
{
    using namespace amrex::literals;

    constexpr int NODE = amrex::IndexType::NODE;
    constexpr int CELL = amrex::IndexType::CELL;

    // --- Compute shape factors
    Compute_shape_factor< depos_order > const compute_shape_factor;
#if !defined(WARPX_DIM_1D_Z)
    // x direction
    // Get particle position at dt=relative time from particle time
    // Keep these double to avoid bug in single precision

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
    // In RZ, wpx is actually wpr, and wpy is wptheta
    // Convert to cylindrical at the mid point
    const amrex::Real xpmid = xp + relative_time*vx;
    const amrex::Real ypmid = yp + relative_time*vy;
    const amrex::Real rpmid = std::sqrt(xpmid*xpmid + ypmid*ypmid);

    const double xmid = (rpmid - xyzmin.x)*dinv.x;
#elif defined(WARPX_DIM_RSPHERE)
    // Convert to spherical at the mid point
    const amrex::Real xpmid = xp + relative_time*vx;
    const amrex::Real ypmid = yp + relative_time*vy;
    const amrex::Real zpmid = zp + relative_time*vz;
    const amrex::Real rpmid = std::sqrt(xpmid*xpmid + ypmid*ypmid + zpmid*zpmid);

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
    if (varx_type[0] == NODE || vary_type[0] == NODE || varz_type[0] == NODE) {
        j_node = compute_shape_factor(sx_node, xmid);
    }
    if (varx_type[0] == CELL || vary_type[0] == CELL || varz_type[0] == CELL) {
        j_cell = compute_shape_factor(sx_cell, xmid - 0.5);
    }

    amrex::Real sx_jx[depos_order + 1] = {0._rt};
    amrex::Real sx_jy[depos_order + 1] = {0._rt};
    amrex::Real sx_jz[depos_order + 1] = {0._rt};
    for (int ix=0; ix<=depos_order; ix++)
    {
        sx_jx[ix] = ((varx_type[0] == NODE) ? amrex::Real(sx_node[ix]) : amrex::Real(sx_cell[ix]));
        sx_jy[ix] = ((vary_type[0] == NODE) ? amrex::Real(sx_node[ix]) : amrex::Real(sx_cell[ix]));
        sx_jz[ix] = ((varz_type[0] == NODE) ? amrex::Real(sx_node[ix]) : amrex::Real(sx_cell[ix]));
    }

    int const j_jx = ((varx_type[0] == NODE) ? j_node : j_cell);
    int const j_jy = ((vary_type[0] == NODE) ? j_node : j_cell);
    int const j_jz = ((varz_type[0] == NODE) ? j_node : j_cell);
#endif //!defined(WARPX_DIM_1D_Z)

#if defined(WARPX_DIM_3D)
    // y direction
    // Keep these double to avoid bug in single precision
    const double ymid = ((yp - xyzmin.y) + relative_time*vy)*dinv.y;
    double sy_node[depos_order + 1] = {0.};
    double sy_cell[depos_order + 1] = {0.};
    int k_node = 0;
    int k_cell = 0;
    if (varx_type[1] == NODE || vary_type[1] == NODE || varz_type[1] == NODE) {
        k_node = compute_shape_factor(sy_node, ymid);
    }
    if (varx_type[1] == CELL || vary_type[1] == CELL || varz_type[1] == CELL) {
        k_cell = compute_shape_factor(sy_cell, ymid - 0.5);
    }
    amrex::Real sy_jx[depos_order + 1] = {0._rt};
    amrex::Real sy_jy[depos_order + 1] = {0._rt};
    amrex::Real sy_jz[depos_order + 1] = {0._rt};
    for (int iy=0; iy<=depos_order; iy++)
    {
        sy_jx[iy] = ((varx_type[1] == NODE) ? amrex::Real(sy_node[iy]) : amrex::Real(sy_cell[iy]));
        sy_jy[iy] = ((vary_type[1] == NODE) ? amrex::Real(sy_node[iy]) : amrex::Real(sy_cell[iy]));
        sy_jz[iy] = ((varz_type[1] == NODE) ? amrex::Real(sy_node[iy]) : amrex::Real(sy_cell[iy]));
    }
    int const k_jx = ((varx_type[1] == NODE) ? k_node : k_cell);
    int const k_jy = ((vary_type[1] == NODE) ? k_node : k_cell);
    int const k_jz = ((varz_type[1] == NODE) ? k_node : k_cell);
#endif

#if !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
    // z direction
    // Keep these double to avoid bug in single precision
    constexpr int zdir = WARPX_ZINDEX;
    const double zmid = ((zp - xyzmin.z) + relative_time*vz)*dinv.z;
    double sz_node[depos_order + 1] = {0.};
    double sz_cell[depos_order + 1] = {0.};
    int l_node = 0;
    int l_cell = 0;
    if (varx_type[zdir] == NODE || vary_type[zdir] == NODE || varz_type[zdir] == NODE) {
        l_node = compute_shape_factor(sz_node, zmid);
    }
    if (varx_type[zdir] == CELL || vary_type[zdir] == CELL || varz_type[zdir] == CELL) {
        l_cell = compute_shape_factor(sz_cell, zmid - 0.5);
    }
    amrex::Real sz_jx[depos_order + 1] = {0._rt};
    amrex::Real sz_jy[depos_order + 1] = {0._rt};
    amrex::Real sz_jz[depos_order + 1] = {0._rt};
    for (int iz=0; iz<=depos_order; iz++)
    {
        sz_jx[iz] = ((varx_type[zdir] == NODE) ? amrex::Real(sz_node[iz]) : amrex::Real(sz_cell[iz]));
        sz_jy[iz] = ((vary_type[zdir] == NODE) ? amrex::Real(sz_node[iz]) : amrex::Real(sz_cell[iz]));
        sz_jz[iz] = ((varz_type[zdir] == NODE) ? amrex::Real(sz_node[iz]) : amrex::Real(sz_cell[iz]));
    }
    int const l_jx = ((varx_type[zdir] == NODE) ? l_node : l_cell);
    int const l_jy = ((vary_type[zdir] == NODE) ? l_node : l_cell);
    int const l_jz = ((varz_type[zdir] == NODE) ? l_node : l_cell);
#endif

#if defined(WARPX_DIM_1D_Z)
    for (int iz=0; iz<=depos_order; iz++){
        const amrex::Real wpx_var = static_cast<amrex::Real>(wp)*sz_jx[iz];
        const amrex::Real wpy_var = static_cast<amrex::Real>(wp)*sz_jy[iz];
        const amrex::Real wpz_var = static_cast<amrex::Real>(wp)*sz_jz[iz];

        amrex::IntVectND<3> ixv{lo.z+l_jx+iz, 0, 0};
        amrex::IntVectND<3> iyv{lo.z+l_jy+iz, 0, 0};
        amrex::IntVectND<3> izv{lo.z+l_jz+iz, 0, 0};

        varianceDepositionSubKernel(
            vx, vy, vz,
            ixv, iyv, izv,
            wpx_var, wpy_var, wpz_var,
            nx_arr, ny_arr, nz_arr,
            wx_arr, wy_arr, wz_arr,
            w2x_arr, w2y_arr, w2z_arr,
            vxbar_arr, vybar_arr, vzbar_arr,
            type, pass
        );
    }
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    for (int ix=0; ix<=depos_order; ix++){
        const amrex::Real wpx_var = wp*sx_jx[ix];
        const amrex::Real wpy_var = wp*sx_jy[ix];
        const amrex::Real wpz_var = wp*sx_jz[ix];

        amrex::IntVectND<3> ixv{lo.x+j_jx+ix, 0, 0};
        amrex::IntVectND<3> iyv{lo.x+j_jy+ix, 0, 0};
        amrex::IntVectND<3> izv{lo.x+j_jz+ix, 0, 0};

        varianceDepositionSubKernel(
            vx, vy, vz,
            ixv, iyv, izv,
            wpx_var, wpy_var, wpz_var,
            nx_arr, ny_arr, nz_arr,
            wx_arr, wy_arr, wz_arr,
            w2x_arr, w2y_arr, w2z_arr,
            vxbar_arr, vybar_arr, vzbar_arr,
            type, pass
        );
    }
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    for (int iz=0; iz<=depos_order; iz++){
        for (int ix=0; ix<=depos_order; ix++){
            const amrex::Real wpx_var = wp*sx_jx[ix]*sz_jx[iz];
            const amrex::Real wpy_var = wp*sx_jy[ix]*sz_jy[iz];
            const amrex::Real wpz_var = wp*sx_jz[ix]*sz_jz[iz];

            amrex::IntVectND<3> ixv{lo.x+j_jx+ix, lo.z+l_jx+iz, 0};
            amrex::IntVectND<3> iyv{lo.x+j_jy+ix, lo.z+l_jy+iz, 0};
            amrex::IntVectND<3> izv{lo.x+j_jz+ix, lo.z+l_jz+iz, 0};

            varianceDepositionSubKernel(
                vx, vy, vz,
                ixv, iyv, izv,
                wpx_var, wpy_var, wpz_var,
                nx_arr, ny_arr, nz_arr,
                wx_arr, wy_arr, wz_arr,
                w2x_arr, w2y_arr, w2z_arr,
                vxbar_arr, vybar_arr, vzbar_arr,
                type, pass
            );
        }
    }
#elif defined(WARPX_DIM_3D)
    for (int iz=0; iz<=depos_order; iz++){
        for (int iy=0; iy<=depos_order; iy++){
            for (int ix=0; ix<=depos_order; ix++){
                const amrex::Real wpx_var = wp*sx_jx[ix]*sy_jx[iy]*sz_jx[iz];
                const amrex::Real wpy_var = wp*sx_jy[ix]*sy_jy[iy]*sz_jy[iz];
                const amrex::Real wpz_var = wp*sx_jz[ix]*sy_jz[iy]*sz_jz[iz];

                amrex::IntVectND<3> ixv{lo.x+j_jx+ix, lo.y+k_jx+iy, lo.z+l_jx+iz};
                amrex::IntVectND<3> iyv{lo.x+j_jy+ix, lo.y+k_jy+iy, lo.z+l_jy+iz};
                amrex::IntVectND<3> izv{lo.x+j_jz+ix, lo.y+k_jz+iy, lo.z+l_jz+iz};

                varianceDepositionSubKernel(
                    vx, vy, vz,
                    ixv, iyv, izv,
                    wpx_var, wpy_var, wpz_var,
                    nx_arr, ny_arr, nz_arr,
                    wx_arr, wy_arr, wz_arr,
                    w2x_arr, w2y_arr, w2z_arr,
                    vxbar_arr, vybar_arr, vzbar_arr,
                    type, pass
                );
            }
        }
    }
#endif
}

/**
 * \brief Temperature Deposition Algorithm from Bell (1979) https://dl.acm.org/doi/pdf/10.1145/359146.359153
 *        This can run either WV1 (Double Pass), or WV4 (Single Pass) Algorithms
 *        It should be noted that becuase of GPU Weak memory ordering in kernels it is
 *        quite difficult to perform online update algorithms (i.e.) WV2 and WV3 because of
 *        difficulties creating critical regions in kernels for all GPU kernel types.
 * \tparam depos_order deposition order
 * \param GetPosition  A functor for returning the particle position.
 * \param wp           Pointer to array of particle weights.
 * \param uxp,uyp,uzp  Pointer to arrays of particle momentum.
 * \param varx_fab,vary_fab,varz_fab FArrayBox of variance, either full array or tile.
 * \param nx_iab,ny_iab,nz_iab IArrayBox of sample count buffer, either full array or tile.
 * \param wx_fab,wy_fab,wz_fab FArrayBox for running weights buffer, either full array or tile.
 * \param w2x_fab,w2y_fab,w2z_fab FArrayBox for weights^2 buffer, either full array or tile.
 * \param vxbar_fab,vybar_fab,vzbar_fab FArrayBox of current density, either full array or tile.
 * \param type The type of deposition routine, either SINGLE_PASS or DOUBLE_PASS
 * \param pass The pass that is occurring, either FIRST or SECOND
 * \param np_to_deposit Number of particles for which current is deposited.
 * \param relative_time Time at which to deposit J, relative to the time of the
 *                      current positions of the particles. When different than 0,
 *                      the particle position will be temporarily modified to match
 *                      the time of the deposition.
 * \param dinv         3D cell size inverse
 * \param xyzmin       Physical lower bounds of domain.
 * \param lo           Index lower bounds of domain.
 * \param n_rz_azimuthal_modes Number of azimuthal modes when using RZ geometry.
 */
template <int depos_order>
void doVarianceDepositionShapeN (const GetParticlePosition<PIdx>& GetPosition,
                         const amrex::ParticleReal * wp,
                         const amrex::ParticleReal * uxp,
                         const amrex::ParticleReal * uyp,
                         const amrex::ParticleReal * uzp,
                         amrex::FArrayBox& varx_fab,
                         amrex::FArrayBox& vary_fab,
                         amrex::FArrayBox& varz_fab,
                         amrex::IArrayBox& nx_iab,
                         amrex::IArrayBox& ny_iab,
                         amrex::IArrayBox& nz_iab,
                         amrex::FArrayBox& wx_fab,
                         amrex::FArrayBox& wy_fab,
                         amrex::FArrayBox& wz_fab,
                         amrex::FArrayBox& w2x_fab,
                         amrex::FArrayBox& w2y_fab,
                         amrex::FArrayBox& w2z_fab,
                         amrex::FArrayBox& vxbar_fab,
                         amrex::FArrayBox& vybar_fab,
                         amrex::FArrayBox& vzbar_fab,
                         const TemperatureDepositionType type,
                         const TemperatureDepositionPass pass,
                         const long np_to_deposit,
                         const amrex::Real relative_time,
                         const amrex::XDim3 & dinv,
                         const amrex::XDim3 & xyzmin,
                         const amrex::Dim3 lo,
                         [[maybe_unused]]const int n_rz_azimuthal_modes)
{
    using namespace amrex::literals;

#if defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        n_rz_azimuthal_modes == 1,
        "Azimuthal Fourier decomposition for temperature deposition is not implemented."
        " Only mode=0 supported in RZ."
    );
#endif

    const amrex::Real clightsq = 1.0_rt/PhysConst::c/PhysConst::c;

    const amrex::Array4<int> & nx_arr = nx_iab.array();
    const amrex::Array4<int> & ny_arr = ny_iab.array();
    const amrex::Array4<int> & nz_arr = nz_iab.array();
    const amrex::Array4<amrex::Real> & wx_arr = wx_fab.array();
    const amrex::Array4<amrex::Real> & wy_arr = wy_fab.array();
    const amrex::Array4<amrex::Real> & wz_arr = wz_fab.array();
    const amrex::Array4<amrex::Real> & w2x_arr = w2x_fab.array();
    const amrex::Array4<amrex::Real> & w2y_arr = w2y_fab.array();
    const amrex::Array4<amrex::Real> & w2z_arr = w2z_fab.array();
    const amrex::Array4<amrex::Real> & vxbar_arr = vxbar_fab.array();
    const amrex::Array4<amrex::Real> & vybar_arr = vybar_fab.array();
    const amrex::Array4<amrex::Real> & vzbar_arr = vzbar_fab.array();
    const amrex::IntVect & varx_type = varx_fab.box().type();
    const amrex::IntVect & vary_type = vary_fab.box().type();
    const amrex::IntVect & varz_type = varz_fab.box().type();

    // Loop over particles and deposit into the deposition buffers
    amrex::ParallelFor(
        np_to_deposit,
        [=] AMREX_GPU_DEVICE (long ip) {
            amrex::ParticleReal xp, yp, zp;
            GetPosition(ip, xp, yp, zp);

            // --- Get particle quantities
            const amrex::ParticleReal gaminv = 1.0_rt/std::sqrt(1.0_rt + uxp[ip]*uxp[ip]*clightsq
                                                        + uyp[ip]*uyp[ip]*clightsq
                                                        + uzp[ip]*uzp[ip]*clightsq);
            const amrex::ParticleReal vx  = uxp[ip]*gaminv;
            const amrex::ParticleReal vy  = uyp[ip]*gaminv;
            const amrex::ParticleReal vz  = uzp[ip]*gaminv;

            doVarianceDepositionShapeNKernel<depos_order>(
                xp, yp, zp, wp[ip], vx, vy, vz,
                nx_arr, ny_arr, nz_arr,
                wx_arr, wy_arr, wz_arr,
                w2x_arr, w2y_arr, w2z_arr,
                vxbar_arr, vybar_arr, vzbar_arr,
                varx_type, vary_type, varz_type,
                type, pass,
                relative_time, dinv, xyzmin,
                lo, n_rz_azimuthal_modes);
        }
    );

    amrex::Gpu::streamSynchronize();
}

} // namespace warpx::particles::deposition
#endif // WARPX_TEMPERATUREDEPOSITION_H_
